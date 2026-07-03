#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import sanitize_name


DEFAULT_DEFERRED_CSV = BUILDER_ROOT / "board_harness" / "results" / "board_measure_deferred_cases_current_impl.csv"
DEFAULT_OUTPUT_CONFIG = BUILDER_ROOT / "configs" / "candidate_kernels_v2_failed44_fixed_vector.yaml"
DEFAULT_OUTPUT_MAP_CSV = BUILDER_ROOT / "results" / "v2_failed44_fixed_vector" / "failed44_primitive_map.csv"
DEFAULT_OUTPUT_MAP_JSON = BUILDER_ROOT / "results" / "v2_failed44_fixed_vector" / "failed44_primitive_map.json"

DW_RE = re.compile(
    r"^dw_conv_k(?P<k>[35])_dw_ir(?P<ir>\d+)_e(?P<expand>\d+)_"
    r"(?P<h>\d+)_c(?P<channels>\d+)_s(?P<stride>\d+)_baseline_"
)
PW_RE = re.compile(r"^pw_conv_(?P<shape>.+?)_baseline_")
PW_HEAD_RE = re.compile(r"^pw_head_(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)$")
PW_IR_RE = re.compile(
    r"^pw_ir(?P<ir>\d+)_(?P<role>expand|proj)_e(?P<expand>\d+)_"
    r"(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)$"
)
MBCONV_RE = re.compile(
    r"^mbconv_e(?P<expand>\d+)_k(?P<k>[35])_mbconv_stage(?P<stage>\d+)_"
    r"(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)_s(?P<stride>\d+)_baseline_"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v2 fixed-vector primitive HLS config for the 44 current-implementation defer cases."
    )
    parser.add_argument("--deferred-csv", default=str(DEFAULT_DEFERRED_CSV))
    parser.add_argument("--output-config", default=str(DEFAULT_OUTPUT_CONFIG))
    parser.add_argument("--output-map-csv", default=str(DEFAULT_OUTPUT_MAP_CSV))
    parser.add_argument("--output-map-json", default=str(DEFAULT_OUTPUT_MAP_JSON))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deferred_rows = _read_deferred_rows(Path(args.deferred_csv))
    config, mapping_rows = build_config(deferred_rows)

    config_path = Path(args.output_config).expanduser().resolve()
    map_csv_path = Path(args.output_map_csv).expanduser().resolve()
    map_json_path = Path(args.output_map_json).expanduser().resolve()

    config_path.parent.mkdir(parents=True, exist_ok=True)
    map_csv_path.parent.mkdir(parents=True, exist_ok=True)
    map_json_path.parent.mkdir(parents=True, exist_ok=True)

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    _write_mapping_csv(map_csv_path, mapping_rows)
    map_json_path.write_text(
        json.dumps(
            {
                "source_deferred_csv": str(Path(args.deferred_csv).expanduser().resolve()),
                "old_deferred_cases": len(deferred_rows),
                "primitive_case_refs": len({row["component_case_name"] for row in mapping_rows}),
                "mapping_rows": mapping_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote v2 failed44 fixed-vector config: {config_path}")
    print(f"Wrote primitive mapping CSV: {map_csv_path}")
    print(f"Wrote primitive mapping JSON: {map_json_path}")
    print(f"Old defer cases: {len(deferred_rows)}")
    print(f"Unique primitive HLS cases: {len({row['component_case_name'] for row in mapping_rows})}")


def build_config(deferred_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    shapes: dict[str, dict[str, Any]] = {}
    operator_shape_refs: dict[str, list[str]] = {
        "dw_conv_k3": [],
        "dw_conv_k5": [],
        "pw_conv": [],
    }
    mapping_rows: list[dict[str, str]] = []

    for row in deferred_rows:
        case_name = row["case_name"]
        if match := DW_RE.match(case_name):
            _add_direct_dw_case(match, case_name, shapes, operator_shape_refs, mapping_rows)
        elif match := PW_RE.match(case_name):
            _add_direct_pw_case(match, case_name, shapes, operator_shape_refs, mapping_rows)
        elif match := MBCONV_RE.match(case_name):
            _add_mbconv_primitive_cases(match, case_name, shapes, operator_shape_refs, mapping_rows)
        else:
            raise ValueError(f"Unsupported defer case name: {case_name}")

    return _build_yaml_config(shapes, operator_shape_refs), mapping_rows


def _read_deferred_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No deferred rows found in {path}")
    return rows


def _add_shape(shapes: dict[str, dict[str, Any]], shape_name: str, payload: dict[str, Any]) -> None:
    existing = shapes.get(shape_name)
    if existing is not None and existing != payload:
        raise ValueError(f"Shape collision for {shape_name}: {existing} != {payload}")
    shapes[shape_name] = payload


def _add_ref(refs: dict[str, list[str]], operator: str, shape_name: str) -> None:
    if shape_name not in refs[operator]:
        refs[operator].append(shape_name)


def _component_case_name(operator: str, shape_name: str, impl_profile: str) -> str:
    return sanitize_name(f"{operator}__{shape_name}__{impl_profile}__main_5ns")


def _add_mapping(
    mapping_rows: list[dict[str, str]],
    *,
    old_case_name: str,
    old_op_family: str,
    component_role: str,
    component_operator: str,
    shape_name: str,
    impl_profile: str,
    notes: str = "",
) -> None:
    mapping_rows.append(
        {
            "old_case_name": old_case_name,
            "old_op_family": old_op_family,
            "component_role": component_role,
            "component_operator": component_operator,
            "component_case_name": _component_case_name(component_operator, shape_name, impl_profile),
            "shape_name": shape_name,
            "impl_profile": impl_profile,
            "clock_profile": "main_5ns",
            "notes": notes,
        }
    )


def _add_direct_dw_case(
    match: re.Match[str],
    old_case_name: str,
    shapes: dict[str, dict[str, Any]],
    refs: dict[str, list[str]],
    mapping_rows: list[dict[str, str]],
) -> None:
    k = int(match.group("k"))
    ir = int(match.group("ir"))
    expand = int(match.group("expand"))
    h = int(match.group("h"))
    channels = int(match.group("channels"))
    stride = int(match.group("stride"))
    operator = f"dw_conv_k{k}"
    impl_profile = "p16_cb16_ii1_dsp1" if k == 3 else "p16_cb16_ii2_dsp1"
    shape_name = f"failed44_dw_ir{ir}_e{expand}_{h}_c{channels}_s{stride}"
    _add_shape(
        shapes,
        shape_name,
        {
            "feature_h": h,
            "feature_w": h,
            "in_channels": channels,
            "stride": stride,
            "expand_ratio": expand,
        },
    )
    _add_ref(refs, operator, shape_name)
    _add_mapping(
        mapping_rows,
        old_case_name=old_case_name,
        old_op_family=operator,
        component_role="direct",
        component_operator=operator,
        shape_name=shape_name,
        impl_profile=impl_profile,
    )


def _add_direct_pw_case(
    match: re.Match[str],
    old_case_name: str,
    shapes: dict[str, dict[str, Any]],
    refs: dict[str, list[str]],
    mapping_rows: list[dict[str, str]],
) -> None:
    shape_token = match.group("shape")
    if head_match := PW_HEAD_RE.match(shape_token):
        h = int(head_match.group("h"))
        cin = int(head_match.group("cin"))
        cout = int(head_match.group("cout"))
        expand = 1
        apply_relu = 1
    elif ir_match := PW_IR_RE.match(shape_token):
        h = int(ir_match.group("h"))
        cin = int(ir_match.group("cin"))
        cout = int(ir_match.group("cout"))
        expand = int(ir_match.group("expand"))
        apply_relu = 0 if ir_match.group("role") == "proj" else 1
    else:
        raise ValueError(f"Unsupported pw_conv shape token: {shape_token}")

    operator = "pw_conv"
    impl_profile = "p16_cb16_ii1_dsp1"
    shape_name = f"failed44_{shape_token}"
    _add_shape(
        shapes,
        shape_name,
        {
            "feature_h": h,
            "feature_w": h,
            "in_channels": cin,
            "out_channels": cout,
            "stride": 1,
            "expand_ratio": expand,
            "parameters": {
                "apply_relu": apply_relu,
            },
        },
    )
    _add_ref(refs, operator, shape_name)
    _add_mapping(
        mapping_rows,
        old_case_name=old_case_name,
        old_op_family="pw_conv",
        component_role="direct",
        component_operator=operator,
        shape_name=shape_name,
        impl_profile=impl_profile,
    )


def _add_mbconv_primitive_cases(
    match: re.Match[str],
    old_case_name: str,
    shapes: dict[str, dict[str, Any]],
    refs: dict[str, list[str]],
    mapping_rows: list[dict[str, str]],
) -> None:
    expand = int(match.group("expand"))
    k = int(match.group("k"))
    stage = int(match.group("stage"))
    h = int(match.group("h"))
    cin = int(match.group("cin"))
    cout = int(match.group("cout"))
    stride = int(match.group("stride"))
    hidden = cin * expand
    out_h = ((h + (2 * (k // 2)) - k) // stride) + 1
    old_op_family = f"mbconv_e{expand}_k{k}"

    expand_shape = f"failed44_mb_s{stage}_e{expand}_expand_{h}_{cin}_{hidden}"
    _add_shape(
        shapes,
        expand_shape,
        {
            "feature_h": h,
            "feature_w": h,
            "in_channels": cin,
            "out_channels": hidden,
            "stride": 1,
            "expand_ratio": expand,
            "parameters": {
                "apply_relu": 1,
            },
        },
    )
    _add_ref(refs, "pw_conv", expand_shape)
    _add_mapping(
        mapping_rows,
        old_case_name=old_case_name,
        old_op_family=old_op_family,
        component_role="pw_expand",
        component_operator="pw_conv",
        shape_name=expand_shape,
        impl_profile="p16_cb16_ii1_dsp1",
        notes="primitive_sum_upper_bound component for disabled fused MBConv",
    )

    dw_operator = f"dw_conv_k{k}"
    dw_impl = "p16_cb16_ii1_dsp1" if k == 3 else "p16_cb16_ii2_dsp1"
    dw_shape = f"failed44_mb_s{stage}_e{expand}_k{k}_dw_{h}_c{hidden}_s{stride}"
    _add_shape(
        shapes,
        dw_shape,
        {
            "feature_h": h,
            "feature_w": h,
            "in_channels": hidden,
            "stride": stride,
            "expand_ratio": expand,
        },
    )
    _add_ref(refs, dw_operator, dw_shape)
    _add_mapping(
        mapping_rows,
        old_case_name=old_case_name,
        old_op_family=old_op_family,
        component_role="dw",
        component_operator=dw_operator,
        shape_name=dw_shape,
        impl_profile=dw_impl,
        notes="primitive_sum_upper_bound component for disabled fused MBConv",
    )

    project_shape = f"failed44_mb_s{stage}_e{expand}_project_{out_h}_{hidden}_{cout}"
    _add_shape(
        shapes,
        project_shape,
        {
            "feature_h": out_h,
            "feature_w": out_h,
            "in_channels": hidden,
            "out_channels": cout,
            "stride": 1,
            "expand_ratio": expand,
            "parameters": {
                "apply_relu": 0,
            },
        },
    )
    _add_ref(refs, "pw_conv", project_shape)
    _add_mapping(
        mapping_rows,
        old_case_name=old_case_name,
        old_op_family=old_op_family,
        component_role="pw_project",
        component_operator="pw_conv",
        shape_name=project_shape,
        impl_profile="p16_cb16_ii1_dsp1",
        notes="primitive_sum_upper_bound component for disabled fused MBConv",
    )


def _build_yaml_config(shapes: dict[str, dict[str, Any]], refs: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "toolchain": {
            "part": "xc7k325t-ffg676-2",
            "solution_name": "solution1",
            "flow_target": "vivado",
            "cpp_standard": "c++14",
            "vitis_hls_bin": "F:/vivado/Vitis_HLS/2023.2/bin/vitis_hls.bat",
            "vivado_bin": "F:/vivado/Vivado/2023.2/bin/vivado.bat",
        },
        "workspace": {
            "project_root": "../results/v2_failed44_fixed_vector/p",
            "report_archive": "../results/v2_failed44_fixed_vector/r",
            "csv_path": "../results/v2_failed44_fixed_vector/hls.csv",
            "manifest_path": "../results/v2_failed44_fixed_vector/hls_manifest.yaml",
            "synthesis_summary_json": "../results/v2_failed44_fixed_vector/hls_synth_summary.json",
            "parse_summary_json": "../results/v2_failed44_fixed_vector/hls_parse_summary.json",
            "downstream_csv_path": "../results/v2_failed44_fixed_vector/vivado_downstream.csv",
            "downstream_manifest_path": "../results/v2_failed44_fixed_vector/vivado_downstream_manifest.yaml",
            "downstream_parse_summary_json": "../results/v2_failed44_fixed_vector/vivado_downstream_parse_summary.json",
            "downstream_summary_json": "../results/v2_failed44_fixed_vector/vivado_downstream_summary.json",
            "formal_lut_status_json": "../results/v2_failed44_fixed_vector/formal_lut_status_v2_failed44_fixed_vector.json",
        },
        "measurement_contract": {
            "contract_id": "packed_stream_v1_fixed_vector_v2_failed44",
            "declaration": (
                "Fixed-vector primitive HLS rerun for the 44 current-implementation defer cases. "
                "Disabled MBConv fused templates are represented by pw_expand + dw + pw_project "
                "primitive components as a conservative upper-bound decomposition."
            ),
            "io_style": "packed_stream",
            "input_axis_rule": "PACK_CH * bitwidth bits",
            "output_axis_rule": "PACK_CH * bitwidth bits",
            "control_interface": "AXI-Lite",
            "parameter_storage": "BRAM",
            "activation_type": "ap_int<8>",
            "weight_type": "ap_int<8>",
            "accumulator_type": "ap_int<32>",
        },
        "defaults": {
            "data_type": "ap_int<8>",
            "accum_type": "ap_int<32>",
            "bitwidth": 8,
            "fixed_vector_params": 1,
            "max_axis_bits": 512,
            "pack_ch": 16,
            "ch_block": 16,
            "tile_order": "channel_major_replay",
            "stream_order": "channel_major_replay",
            "dsp_pack": "1",
            "pipeline_ii": 1,
            "target_ii": 1,
            "input_parallelism": 1,
            "output_parallelism": 1,
            "unroll_factor": 1,
            "array_partition_factor": 1,
        },
        "clock_profiles": {
            "main_5ns": {
                "period_ns": 5.0,
                "target_clock_mhz": 200.0,
            },
        },
        "implementation_profiles": {
            "p16_cb16_ii1_dsp1": {
                "pack_ch": 16,
                "ch_block": 16,
                "pipeline_ii": 1,
                "target_ii": 1,
                "dsp_pack": "1",
            },
            "p16_cb16_ii2_dsp1": {
                "pack_ch": 16,
                "ch_block": 16,
                "pipeline_ii": 2,
                "target_ii": 2,
                "dsp_pack": "1",
            },
        },
        "representative_shapes": dict(sorted(shapes.items())),
        "operators": {
            "dw_conv_k3": {
                "enabled": True,
                "template": "../templates/dw_conv_fixed_vector.cpp.tmpl",
                "top_function": "dwk3",
                "op_type": "dw_conv_k3",
                "shape_refs": sorted(refs["dw_conv_k3"]),
                "implementation_refs": ["p16_cb16_ii1_dsp1"],
                "clock_profile_refs": ["main_5ns"],
                "parameters": {
                    "kernel_size": 3,
                    "padding": 1,
                    "apply_relu": 1,
                },
                "op_spec_defaults": {
                    "op": "dw_conv_k3",
                },
            },
            "dw_conv_k5": {
                "enabled": True,
                "template": "../templates/dw_conv_fixed_vector.cpp.tmpl",
                "top_function": "dwk5",
                "op_type": "dw_conv_k5",
                "shape_refs": sorted(refs["dw_conv_k5"]),
                "implementation_refs": ["p16_cb16_ii2_dsp1"],
                "clock_profile_refs": ["main_5ns"],
                "parameters": {
                    "kernel_size": 5,
                    "padding": 2,
                    "apply_relu": 1,
                },
                "op_spec_defaults": {
                    "op": "dw_conv_k5",
                },
            },
            "pw_conv": {
                "enabled": True,
                "template": "../templates/pw_conv_fixed_vector.cpp.tmpl",
                "top_function": "pwk",
                "op_type": "pw_conv",
                "shape_refs": sorted(refs["pw_conv"]),
                "implementation_refs": ["p16_cb16_ii1_dsp1"],
                "clock_profile_refs": ["main_5ns"],
                "parameters": {
                    "kernel_size": 1,
                    "padding": 0,
                    "apply_relu": 1,
                    "tile_order": "pixel_major",
                    "stream_order": "pixel_major",
                },
                "op_spec_defaults": {
                    "op": "pw_conv",
                    "groups": 1,
                },
            },
        },
    }


def _write_mapping_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "old_case_name",
        "old_op_family",
        "component_role",
        "component_operator",
        "component_case_name",
        "shape_name",
        "impl_profile",
        "clock_profile",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
