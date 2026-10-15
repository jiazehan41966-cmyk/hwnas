#!/usr/bin/env python3
"""Build the full MobileNetA ZCU104 operator-cost measurement matrix.

This script is a generator only: it expands the measurement rows that need
fresh ZCU104 evidence, writes an HLS builder config, and marks the current
formal coverage as not ready. It does not run HLS, route, board validation, or
power measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HLS_SCRIPT_DIR = REPO_ROOT / "hls_lut_builder" / "scripts"
if str(HLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HLS_SCRIPT_DIR))

from common import build_cases, load_config  # noqa: E402


SEARCH_SPACE_VERSION = "mobilenetA_full_dmdc_cfem_v1"
MEASUREMENT_VERSION = "mobilenetA_full_dmdc_cfem_v3_gap1280"
CAMPAIGN_ID = "zcu104_full_campaign_v1"
BASE_CONFIG = REPO_ROOT / "hls_lut_builder" / "configs" / "candidate_kernels.yaml"
OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / CAMPAIGN_ID
    / "operator_cost_lut"
    / MEASUREMENT_VERSION
)
GENERATED_CONFIG = (
    REPO_ROOT
    / "hls_lut_builder"
    / "configs"
    / "candidate_kernels_zcu104_mobilenetA_full_dmdc_cfem_v3_gap1280.yaml"
)

BOARD = {
    "board": "ZCU104",
    "part": "xczu7ev-ffvc1156-2-e",
    "board_part": "xilinx.com:zcu104:part0:1.1",
    "clock_period_ns": 5.0,
    "tool_version": "Vitis/Vivado 2022.2",
}

TARGET_SLOT_SHAPE = "target_slot_32x28x28_to_32x28x28_s1"
TARGET_SLOT = {
    "slot_id": "mobilenetA_stage3_late_32x28x28_s1",
    "stage_label": "stage3",
    "input_shape": [32, 28, 28],
    "output_shape": [32, 28, 28],
    "in_channels": 32,
    "out_channels": 32,
    "feature_h": 28,
    "feature_w": 28,
    "stride": 1,
    "legal_for_dmdc_cfem": True,
}


TARGET_SLOT_OPERATORS = {
    "mbconv_target_slot_k3_e3": {
        "template": "../templates/mbconv.cpp.tmpl",
        "top_function": "zcu104_mobilenetA_target_k3_e3_kernel",
        "op_type": "mbconv",
        "parameters": {"kernel_size": 3, "padding": 1, "expand_ratio": 3},
        "operator_family": "mobile_core",
        "search_role": "target_slot_control",
    },
    "mbconv_target_slot_k5_e3": {
        "template": "../templates/mbconv.cpp.tmpl",
        "top_function": "zcu104_mobilenetA_target_k5_e3_kernel",
        "op_type": "mbconv",
        "parameters": {"kernel_size": 5, "padding": 2, "expand_ratio": 3},
        "operator_family": "mobile_core",
        "search_role": "target_slot_control",
    },
    "dmdc_conv_sonar_target_slot": {
        "enabled": True,
        "template": "../templates/dmdc_conv_sonar.cpp.tmpl",
        "top_function": "zcu104_mobilenetA_target_dmdc_conv_sonar_kernel",
        "op_type": "dmdc_conv_sonar",
        "parameters": {
            "kernel_size": 3,
            "padding": 1,
            "expand_ratio": 1,
            "dilation_rates": "1_3_5",
        },
        "operator_family": "sonar_extension",
        "search_role": "target_slot_new_candidate",
    },
    "cfem_sonar_target_slot": {
        "enabled": True,
        "template": "../templates/cfem_sonar.cpp.tmpl",
        "top_function": "zcu104_mobilenetA_target_cfem_sonar_kernel",
        "op_type": "cfem_sonar",
        "parameters": {
            "kernel_size": 3,
            "padding": 1,
            "expand_ratio": 1,
            "dilation_rates": "2_3_5",
        },
        "operator_family": "sonar_extension",
        "search_role": "target_slot_new_candidate",
    },
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return payload


def build_zcu104_config(base_config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(base_config_path)
    cfg["toolchain"] = {
        "part": BOARD["part"],
        "solution_name": "solution1",
        "flow_target": "vivado",
        "cpp_standard": "c++14",
        "vitis_hls_bin": "/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls",
        "vivado_bin": "/opt/Xilinx/Vivado/2022.2/bin/vivado",
    }
    cfg["workspace"] = {
        "project_root": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/p",
        "report_archive": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/reports",
        "csv_path": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/hls.csv",
        "manifest_path": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/hls_manifest.yaml",
        "synthesis_summary_json": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/hls_synth_summary.json",
        "parse_summary_json": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/hls/hls_parse_summary.json",
        "downstream_csv_path": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/route/vivado_downstream.csv",
        "downstream_manifest_path": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/route/vivado_downstream_manifest.yaml",
        "downstream_parse_summary_json": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/route/vivado_downstream_parse_summary.json",
        "downstream_summary_json": "../../results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v1/route/vivado_downstream_summary.json",
    }
    cfg["workspace"] = {
        key: value.replace("mobilenetA_full_dmdc_cfem_v1", MEASUREMENT_VERSION)
        for key, value in cfg["workspace"].items()
    }
    cfg["representative_shapes"].pop("gap_7_320", None)
    cfg["representative_shapes"]["gap_7_1280"] = {
        "feature_h": 7, "feature_w": 7, "in_channels": 1280,
        "out_channels": 1280, "stride": 1,
    }
    cfg["operators"]["global_avg_pool"]["shape_refs"] = ["gap_7_1280"]
    cfg["measurement_contract"] = {
        "contract_id": "zcu104_mobilenetA_full_dmdc_cfem_int8_v3_gap1280",
        "hardware_lut_version": MEASUREMENT_VERSION,
        "status": "PENDING_MEASUREMENT",
        "search_space_version": SEARCH_SPACE_VERSION,
        "target_board": BOARD["board"],
        "target_part": BOARD["part"],
        "declaration": (
            "Fresh ZCU104 measurement matrix for MobileNet-style A. "
            "Historical AV7K325 or fourstage rows are forbidden as formal evidence."
        ),
        "io_style": "packed_stream",
        "input_axis_rule": "packed input channels * 8 bits",
        "output_axis_rule": "packed output channels * 8 bits",
        "control_interface": "AXI-Lite",
        "parameter_storage": "BRAM",
        "activation_type": "ap_int<8>",
        "weight_type": "ap_int<8>",
        "accumulator_type": "ap_int<32>",
    }
    cfg["defaults"] = {
        **dict(cfg.get("defaults", {})),
        "input_parallelism": 4,
        "output_parallelism": 4,
        "unroll_factor": 1,
        "array_partition_factor": 4,
        "max_axis_bits": 4096,
    }
    cfg["implementation_profiles"] = {
        "balanced_pi4_po4_u1": {
            "input_parallelism": 4,
            "output_parallelism": 4,
            "unroll_factor": 1,
            "array_partition_factor": 4,
            "pipeline_ii": 1,
        }
    }
    cfg["clock_profiles"] = {"main_5ns": {"period_ns": 5.0, "target_clock_mhz": 200.0}}

    for operator in cfg.get("operators", {}).values():
        operator["implementation_refs"] = ["balanced_pi4_po4_u1"]
        operator["clock_profile_refs"] = ["main_5ns"]

    cfg.setdefault("representative_shapes", {})[TARGET_SLOT_SHAPE] = {
        "feature_h": 28,
        "feature_w": 28,
        "in_channels": 32,
        "out_channels": 32,
        "stride": 1,
    }

    for name, op_cfg in TARGET_SLOT_OPERATORS.items():
        cfg["operators"][name] = {
            "enabled": op_cfg.get("enabled", True),
            **({"blocked_reason": op_cfg["blocked_reason"]} if "blocked_reason" in op_cfg else {}),
            "template": op_cfg["template"],
            "top_function": op_cfg["top_function"],
            "op_type": op_cfg["op_type"],
            "shape_refs": [TARGET_SLOT_SHAPE],
            "implementation_refs": ["balanced_pi4_po4_u1"],
            "clock_profile_refs": ["main_5ns"],
            "parameters": op_cfg["parameters"],
            "op_spec_defaults": {
                "op": op_cfg["op_type"],
                "search_space_version": SEARCH_SPACE_VERSION,
                "operator_family": op_cfg["operator_family"],
                "search_role": op_cfg["search_role"],
                "cost_key": f"target_slot__{name}__32x28x28_to_32x28x28__s1",
            },
        }
    return cfg


def build_case_rows(config_path: Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    active_cases = build_cases(config, sort_cases=True)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in active_cases:
        if case.case_name in seen:
            continue
        seen.add(case.case_name)
        template_exists = case.template_path.exists()
        blocked_reason = None if template_exists else "missing_operator_specific_hls_template"
        parity_status = (
            "NEEDS_SEMANTIC_PARITY_AUDIT"
            if case.operator in {"dmdc_conv_sonar_target_slot", "cfem_sonar_target_slot"}
            else "NOT_REQUIRED_FOR_LEGACY_TEMPLATE"
        )
        rows.append(
            {
                "row_key": case.case_name,
                "case_name": case.case_name,
                "operator": case.operator,
                "shape_name": case.shape_name,
                "implementation_name": case.implementation_name,
                "clock_profile_name": case.clock_profile_name,
                "top_function": case.top_function,
                "target_part": case.part,
                "clock_period_ns": case.clock_period_ns,
                "template_path": str(case.template_path),
                "template_exists": template_exists,
                "op_spec": case.op_spec,
                "op_id": case.op_id,
                "formal_status": "PENDING_FRESH_ZCU104_HLS_ROUTE" if template_exists else "BLOCKED_TEMPLATE",
                "blocked_reason": blocked_reason,
                "semantic_parity_status": parity_status,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_key",
        "operator",
        "shape_name",
        "implementation_name",
        "clock_profile_name",
        "top_function",
        "target_part",
        "clock_period_ns",
        "template_exists",
        "formal_status",
        "blocked_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--generated-config", default=str(GENERATED_CONFIG))
    args = parser.parse_args()

    base_config = Path(args.base_config).resolve()
    output_root = Path(args.output_root).resolve()
    generated_config = Path(args.generated_config).resolve()

    zcu104_config = build_zcu104_config(base_config)
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(yaml.safe_dump(zcu104_config, sort_keys=False, allow_unicode=False), encoding="utf-8")

    rows = build_case_rows(generated_config)
    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        counts[row["operator"]] = counts.get(row["operator"], 0) + 1
        status_counts[row["formal_status"]] = status_counts.get(row["formal_status"], 0) + 1

    matrix = {
        "campaign_id": CAMPAIGN_ID,
        "search_space_version": SEARCH_SPACE_VERSION,
        "board": BOARD,
        "base_config": str(base_config),
        "generated_config": str(generated_config),
        "target_slot": TARGET_SLOT,
        "row_count": len(rows),
        "operator_counts": counts,
        "formal_status_counts": status_counts,
        "rows": rows,
        "matrix_sha256": sha256_payload(rows),
    }
    coverage = {
        "campaign_id": CAMPAIGN_ID,
        "search_space_version": SEARCH_SPACE_VERSION,
        "status": "FAIL_NOT_SEARCH_READY",
        "search_ready": False,
        "reason": "Fresh ZCU104 HLS and route evidence has not been collected for this generated matrix.",
        "row_count": len(rows),
        "measured_rows": 0,
        "blocked_template_rows": status_counts.get("BLOCKED_TEMPLATE", 0),
        "pending_fresh_rows": status_counts.get("PENDING_FRESH_ZCU104_HLS_ROUTE", 0),
        "hardware_claim_boundary": {
            "LUT_DSP_BRAM_latency": "NOT_FORMAL_UNTIL_FRESH_ZCU104_REPORTS_EXIST",
            "board_classification": "NOT_CLAIMED_BY_THIS_MATRIX",
            "external_power": "NOT_CLAIMED_BY_THIS_MATRIX",
        },
        "matrix_sha256": matrix["matrix_sha256"],
    }

    write_json(output_root / "measurement_matrix.json", matrix)
    write_json(output_root / "coverage_audit.scaffold.json", coverage)
    write_csv(output_root / "measurement_matrix.csv", rows)
    write_json(
        output_root / "run_commands.scaffold.json",
        {
            "generate_projects_and_run_hls_route": (
                "python hls_lut_builder/scripts/run_synthesis.py "
                f"--config {generated_config} --downstream-check "
                "--timeout-minutes 90 --downstream-timeout-minutes 120"
            ),
            "blocked_sonar_rows": [
                row["row_key"] for row in rows if row["formal_status"] == "BLOCKED_TEMPLATE"
            ],
        },
    )

    print(f"Wrote generated config: {generated_config}")
    print(f"Wrote matrix: {output_root / 'measurement_matrix.json'}")
    print(f"Rows: {len(rows)}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
