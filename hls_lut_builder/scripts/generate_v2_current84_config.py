#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = BUILDER_ROOT / "results"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import sanitize_name


DEFAULT_STATUS_CSV = BUILDER_ROOT / "board_harness" / "results" / "board_measure_status_current_impl.csv"
DEFAULT_DEPLOYABLE_LUT = RESULTS_ROOT / "v2_deployable_lut" / "deployable_lut_v2.json"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "v2_current84"
DEFAULT_OUTPUT_CONFIG = BUILDER_ROOT / "configs" / "candidate_kernels_v2_current84_missing.yaml"

FORMAL_STATUS_SOURCES = OrderedDict(
    [
        (
            "v2_current84_missing",
            RESULTS_ROOT / "v2_current84" / "formal_lut_status_v2_current84_missing.json",
        ),
        (
            "v2_current84_identity_repack_14_64",
            RESULTS_ROOT / "v2_current84" / "formal_lut_status_v2_current84_identity_repack_14_64.json",
        ),
        (
            "v2_failed44_fixed_vector",
            RESULTS_ROOT / "v2_failed44_fixed_vector" / "formal_lut_status_v2_failed44_fixed_vector.json",
        ),
        (
            "v2_pointwise_v21",
            RESULTS_ROOT / "v2_pointwise_v21" / "formal_lut_status_v2_pointwise_v21.json",
        ),
        (
            "v2_k5_low_complexity",
            RESULTS_ROOT / "v2_k5_low_complexity" / "formal_lut_status_v2_k5_low_complexity.json",
        ),
        (
            "v2_depthwise_fallback_blockers",
            RESULTS_ROOT
            / "v2_depthwise_fallback_blockers"
            / "formal_lut_status_v2_depthwise_fallback_blockers.json",
        ),
        (
            "v2_fixed_vector",
            RESULTS_ROOT / "v2_fixed_vector" / "formal_lut_status_v2_fixed_vector.json",
        ),
    ]
)

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
PW_IR_E1_PROJ_RE = re.compile(
    r"^pw_ir(?P<ir>\d+)_proj_(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)$"
)
MBCONV_RE = re.compile(
    r"^mbconv_e(?P<expand>\d+)_k(?P<k>[35])_mbconv_stage(?P<stage>\d+)_"
    r"(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)_s(?P<stride>\d+)_baseline_"
)
STEM_RE = re.compile(
    r"^stem_conv_k3_s2_stem_(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)_s(?P<stride>\d+)_baseline_"
)
SKIP_RE = re.compile(r"^skip_skip_(?P<h>\d+)_(?P<channels>\d+)_baseline_")
GAP_RE = re.compile(r"^global_avg_pool_gap_(?P<h>\d+)_(?P<channels>\d+)_baseline_")
FC_RE = re.compile(r"^fc_layer_fc_(?P<h>\d+)_(?P<cin>\d+)_(?P<cout>\d+)_baseline_")

K3_FALLBACK_SHAPES = {
    (7, 480, 1, 3),
}
K5_P8_SHAPES = {
    (14, 288, 2, 3),
    (14, 576, 2, 6),
    (7, 480, 1, 3),
    (7, 960, 1, 6),
}

CSV_MAP_FIELDS = [
    "current_case_name",
    "current_status",
    "current_detail",
    "old_op_family",
    "decomposition",
    "component_role",
    "component_operator",
    "component_op_type",
    "shape_name",
    "component_case_name",
    "impl_profile",
    "clock_profile",
    "profile_policy",
    "selected_existing_case_name",
    "selected_source",
    "selected_status",
    "selected_profile",
    "coverage_status",
    "deployable_coverage",
    "notes",
]

CSV_MISSING_FIELDS = [
    "component_case_name",
    "component_operator",
    "component_op_type",
    "shape_name",
    "impl_profile",
    "clock_profile",
    "profile_policy",
    "feature_h",
    "feature_w",
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "groups",
    "expand_ratio",
    "first_current_case_name",
    "current_case_count",
    "current_cases",
    "component_roles",
    "missing_reason",
]

CSV_COVERAGE_FIELDS = [
    "current_case_name",
    "current_status",
    "current_detail",
    "old_op_family",
    "decomposition",
    "component_count",
    "has_full_v2_primitive_coverage",
    "has_full_deployable_primitive_coverage",
    "covered_component_count",
    "deployable_component_count",
    "missing_component_count",
    "component_roles",
    "component_cases",
    "selected_existing_cases",
    "selected_sources",
    "coverage_statuses",
    "missing_component_cases",
    "notes",
]

CSV_BLOCKER_FIELDS = [
    "component_case_name",
    "component_operator",
    "component_op_type",
    "shape_name",
    "selected_source",
    "selected_status",
    "selected_existing_case_name",
    "hls_status",
    "downstream_status",
    "blocker_category",
    "blocker_reason",
    "affected_current_case_count",
    "affected_current_cases",
    "flow_tcl",
    "log_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build current84 v2 primitive map, coverage, and missing-only candidate config."
    )
    parser.add_argument("--status-csv", default=str(DEFAULT_STATUS_CSV))
    parser.add_argument("--deployable-lut", default=str(DEFAULT_DEPLOYABLE_LUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-config", default=str(DEFAULT_OUTPUT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_csv = Path(args.status_csv).expanduser().resolve()
    deployable_lut = Path(args.deployable_lut).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_config = Path(args.output_config).expanduser().resolve()

    current_rows = _read_csv(status_csv)
    existing = _load_existing_coverage(deployable_lut, FORMAL_STATUS_SOURCES)
    builder = Current84Builder(existing)

    for row in current_rows:
        builder.add_case(row)

    map_rows = builder.mapping_rows
    coverage_rows = _build_coverage_rows(map_rows)
    missing_rows = _build_missing_rows(builder.primitives)
    missing_config = _build_missing_config(missing_rows, builder.primitives)

    output_dir.mkdir(parents=True, exist_ok=True)
    map_csv = output_dir / "current84_primitive_map.csv"
    map_json = output_dir / "current84_primitive_map.json"
    missing_csv = output_dir / "current84_missing_primitives.csv"
    coverage_csv = output_dir / "current84_combo_coverage.csv"
    blocker_csv = output_dir / "current84_missing_blockers.csv"
    readme = output_dir / "README.md"
    blocker_rows = _build_blocker_rows(output_dir, map_rows)

    _write_csv(map_csv, map_rows, CSV_MAP_FIELDS)
    _write_json(
        map_json,
        {
            "metadata": _metadata(
                status_csv=status_csv,
                deployable_lut=deployable_lut,
                output_config=output_config,
                current_rows=current_rows,
                map_rows=map_rows,
                missing_rows=missing_rows,
                existing=existing,
            ),
            "mapping_rows": map_rows,
        },
    )
    _write_csv(missing_csv, missing_rows, CSV_MISSING_FIELDS)
    _write_csv(coverage_csv, coverage_rows, CSV_COVERAGE_FIELDS)
    _write_csv(blocker_csv, blocker_rows, CSV_BLOCKER_FIELDS)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(_plain_yaml(missing_config), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    readme.write_text(
        _readme_text(
            status_csv=status_csv,
            deployable_lut=deployable_lut,
            output_config=output_config,
            current_rows=current_rows,
            map_rows=map_rows,
            missing_rows=missing_rows,
            coverage_rows=coverage_rows,
            blocker_rows=blocker_rows,
            existing=existing,
        ),
        encoding="utf-8",
    )

    print(f"Wrote primitive map CSV: {map_csv}")
    print(f"Wrote primitive map JSON: {map_json}")
    print(f"Wrote missing primitives CSV: {missing_csv}")
    print(f"Wrote missing-only config: {output_config}")
    print(f"Wrote combo coverage CSV: {coverage_csv}")
    print(f"Wrote blocker summary CSV: {blocker_csv}")
    print(f"Wrote README: {readme}")
    print(f"Current cases: {len(current_rows)}")
    print(f"Primitive map rows: {len(map_rows)}")
    print(f"Unique current84 primitive shapes: {len(builder.primitives)}")
    print(f"Missing primitive shapes: {len(missing_rows)}")
    print(f"Blocker primitive shapes: {len(blocker_rows)}")
    print(f"Coverage counts: {dict(Counter(row['has_full_v2_primitive_coverage'] for row in coverage_rows))}")


class Current84Builder:
    def __init__(self, existing: dict[str, Any]) -> None:
        self.existing = existing
        self.mapping_rows: list[dict[str, str]] = []
        self.primitives: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._primitive_by_key: dict[tuple[Any, ...], str] = {}

    def add_case(self, row: dict[str, str]) -> None:
        case_name = row["case_name"]
        status = row.get("status", "")
        detail = row.get("detail", "")
        components = self._decompose_case(case_name)
        for component in components:
            primitive = self._intern_primitive(component)
            selected = _select_existing(primitive["op_spec"], self.existing)
            coverage_status = "missing_v2_primitive" if selected is None else "covered_by_v2"
            deployable_coverage = "true" if selected and selected["deployable"] else "false"
            self.mapping_rows.append(
                {
                    "current_case_name": case_name,
                    "current_status": status,
                    "current_detail": detail,
                    "old_op_family": component["old_op_family"],
                    "decomposition": component["decomposition"],
                    "component_role": component["component_role"],
                    "component_operator": primitive["component_operator"],
                    "component_op_type": primitive["component_op_type"],
                    "shape_name": primitive["shape_name"],
                    "component_case_name": primitive["component_case_name"],
                    "impl_profile": primitive["impl_profile"],
                    "clock_profile": primitive["clock_profile"],
                    "profile_policy": primitive["profile_policy"],
                    "selected_existing_case_name": "" if selected is None else selected["case_name"],
                    "selected_source": "" if selected is None else selected["source"],
                    "selected_status": "" if selected is None else selected["status"],
                    "selected_profile": "" if selected is None else selected["profile"],
                    "coverage_status": coverage_status,
                    "deployable_coverage": deployable_coverage,
                    "notes": component.get("notes", ""),
                }
            )
            primitive["current_cases"].append(case_name)
            primitive["component_roles"].append(component["component_role"])
            primitive["_existing"] = self.existing

    def _decompose_case(self, case_name: str) -> list[dict[str, Any]]:
        if match := STEM_RE.match(case_name):
            return [self._direct_stem(match)]
        if match := DW_RE.match(case_name):
            return [self._direct_dw(match)]
        if match := PW_RE.match(case_name):
            return [self._direct_pw(match)]
        if match := MBCONV_RE.match(case_name):
            return self._mbconv(match)
        if match := SKIP_RE.match(case_name):
            return [self._direct_skip(match)]
        if match := GAP_RE.match(case_name):
            return [self._direct_gap(match)]
        if match := FC_RE.match(case_name):
            return [self._direct_fc(match)]
        raise ValueError(f"Unsupported current84 case name: {case_name}")

    def _direct_stem(self, match: re.Match[str]) -> dict[str, Any]:
        h = int(match.group("h"))
        cin = int(match.group("cin"))
        cout = int(match.group("cout"))
        stride = int(match.group("stride"))
        shape_name = f"current84_stem_{h}_{cin}_{cout}_s{stride}"
        return self._component(
            old_op_family="stem_conv_k3_s2",
            decomposition="direct",
            role="direct",
            component_operator="stem_conv_k3_s2",
            component_op_type="stem_conv_k3_s2",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": cin,
                "out_channels": cout,
                "stride": stride,
            },
            impl_profile="stream_baseline_pi1_po1_u1",
            profile_policy="special_op_existing_packed_stream_template",
            kernel_size=3,
            groups=1,
            expand_ratio=1,
            notes="non-depthwise endpoint; old latency is not reused",
        )

    def _direct_dw(self, match: re.Match[str]) -> dict[str, Any]:
        k = int(match.group("k"))
        ir = int(match.group("ir"))
        expand = int(match.group("expand"))
        h = int(match.group("h"))
        channels = int(match.group("channels"))
        stride = int(match.group("stride"))
        component_operator, impl_profile, policy = _depthwise_policy(k, h, channels, stride, expand)
        shape_name = f"current84_dw_ir{ir}_e{expand}_{h}_c{channels}_s{stride}"
        return self._component(
            old_op_family=f"dw_conv_k{k}",
            decomposition="direct",
            role="direct",
            component_operator=component_operator,
            component_op_type=f"dw_conv_k{k}",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": channels,
                "stride": stride,
                "expand_ratio": expand,
            },
            impl_profile=impl_profile,
            profile_policy=policy,
            kernel_size=k,
            groups=channels,
            expand_ratio=expand,
        )

    def _direct_pw(self, match: re.Match[str]) -> dict[str, Any]:
        shape_token = match.group("shape")
        h, cin, cout, expand, apply_relu = _parse_pw_shape(shape_token)
        shape_name = f"current84_{shape_token}"
        return self._component(
            old_op_family="pw_conv",
            decomposition="direct",
            role="direct",
            component_operator="pw_conv_v21",
            component_op_type="pw_conv",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": cin,
                "out_channels": cout,
                "stride": 1,
                "expand_ratio": expand,
                "parameters": {"apply_relu": apply_relu},
            },
            impl_profile="p16_pe16_simd4_ii2_dsp1",
            profile_policy="pointwise_v21_p16_pe16_simd4_ii2",
            kernel_size=1,
            groups=1,
            expand_ratio=expand,
        )

    def _mbconv(self, match: re.Match[str]) -> list[dict[str, Any]]:
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
        notes = "MBConv decomposed as pw_expand+dw+pw_project for v2 current84"

        expand_component = self._component(
            old_op_family=old_op_family,
            decomposition="pw_expand+dw+pw_project",
            role="pw_expand",
            component_operator="pw_conv_v21",
            component_op_type="pw_conv",
            shape_name=f"current84_mb_s{stage}_e{expand}_expand_{h}_{cin}_{hidden}",
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": cin,
                "out_channels": hidden,
                "stride": 1,
                "expand_ratio": expand,
                "parameters": {"apply_relu": 1},
            },
            impl_profile="p16_pe16_simd4_ii2_dsp1",
            profile_policy="pointwise_v21_p16_pe16_simd4_ii2",
            kernel_size=1,
            groups=1,
            expand_ratio=expand,
            notes=notes,
        )
        component_operator, impl_profile, policy = _depthwise_policy(k, h, hidden, stride, expand)
        dw_component = self._component(
            old_op_family=old_op_family,
            decomposition="pw_expand+dw+pw_project",
            role="dw",
            component_operator=component_operator,
            component_op_type=f"dw_conv_k{k}",
            shape_name=f"current84_mb_s{stage}_e{expand}_k{k}_dw_{h}_c{hidden}_s{stride}",
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": hidden,
                "stride": stride,
                "expand_ratio": expand,
            },
            impl_profile=impl_profile,
            profile_policy=policy,
            kernel_size=k,
            groups=hidden,
            expand_ratio=expand,
            notes=notes,
        )
        project_component = self._component(
            old_op_family=old_op_family,
            decomposition="pw_expand+dw+pw_project",
            role="pw_project",
            component_operator="pw_conv_v21",
            component_op_type="pw_conv",
            shape_name=f"current84_mb_s{stage}_e{expand}_project_{out_h}_{hidden}_{cout}",
            shape_payload={
                "feature_h": out_h,
                "feature_w": out_h,
                "in_channels": hidden,
                "out_channels": cout,
                "stride": 1,
                "expand_ratio": expand,
                "parameters": {"apply_relu": 0},
            },
            impl_profile="p16_pe16_simd4_ii2_dsp1",
            profile_policy="pointwise_v21_p16_pe16_simd4_ii2",
            kernel_size=1,
            groups=1,
            expand_ratio=expand,
            notes=notes,
        )
        return [expand_component, dw_component, project_component]

    def _direct_skip(self, match: re.Match[str]) -> dict[str, Any]:
        h = int(match.group("h"))
        channels = int(match.group("channels"))
        shape_name = f"current84_identity_repack_{h}_{channels}"
        return self._component(
            old_op_family="skip",
            decomposition="direct",
            role="direct",
            component_operator="identity_repack",
            component_op_type="identity_repack",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": channels,
                "out_channels": channels,
                "stride": 1,
                "expand_ratio": 1,
            },
            impl_profile="p16_cb16_ii1_dsp1",
            profile_policy="identity_repack_v2_p16_cb16_ii1",
            kernel_size=1,
            groups=1,
            expand_ratio=1,
            notes="skip mapped to v2 identity_repack primitive",
        )

    def _direct_gap(self, match: re.Match[str]) -> dict[str, Any]:
        h = int(match.group("h"))
        channels = int(match.group("channels"))
        shape_name = f"current84_gap_{h}_{channels}"
        return self._component(
            old_op_family="global_avg_pool",
            decomposition="direct",
            role="direct",
            component_operator="global_avg_pool",
            component_op_type="global_avg_pool",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": channels,
                "out_channels": channels,
                "stride": 1,
                "padding": 0,
                "out_h": 1,
                "out_w": 1,
            },
            impl_profile="stream_baseline_pi1_po1_u1",
            profile_policy="special_op_existing_packed_stream_template",
            kernel_size=h,
            groups=1,
            expand_ratio=1,
            notes="non-depthwise endpoint; old latency is not reused",
        )

    def _direct_fc(self, match: re.Match[str]) -> dict[str, Any]:
        h = int(match.group("h"))
        cin = int(match.group("cin"))
        cout = int(match.group("cout"))
        shape_name = f"current84_fc_{h}_{cin}_{cout}"
        return self._component(
            old_op_family="fc_layer",
            decomposition="direct",
            role="direct",
            component_operator="fc_layer",
            component_op_type="fc_layer",
            shape_name=shape_name,
            shape_payload={
                "feature_h": h,
                "feature_w": h,
                "in_channels": cin,
                "out_channels": cout,
                "stride": 1,
                "padding": 0,
                "out_h": 1,
                "out_w": 1,
            },
            impl_profile="stream_baseline_pi1_po1_u1",
            profile_policy="special_op_existing_packed_stream_template",
            kernel_size=1,
            groups=1,
            expand_ratio=1,
            notes="non-depthwise endpoint; old latency is not reused",
        )

    def _component(
        self,
        *,
        old_op_family: str,
        decomposition: str,
        role: str,
        component_operator: str,
        component_op_type: str,
        shape_name: str,
        shape_payload: dict[str, Any],
        impl_profile: str,
        profile_policy: str,
        kernel_size: int,
        groups: int,
        expand_ratio: int,
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "old_op_family": old_op_family,
            "decomposition": decomposition,
            "component_role": role,
            "component_operator": component_operator,
            "component_op_type": component_op_type,
            "shape_name": shape_name,
            "shape_payload": shape_payload,
            "impl_profile": impl_profile,
            "clock_profile": "main_5ns",
            "profile_policy": profile_policy,
            "kernel_size": kernel_size,
            "groups": groups,
            "expand_ratio": expand_ratio,
            "notes": notes,
        }

    def _intern_primitive(self, component: dict[str, Any]) -> dict[str, Any]:
        op_spec = _op_spec_for_component(component)
        primitive_key = (
            _shape_key(op_spec),
            component["component_operator"],
            component["impl_profile"],
            component["clock_profile"],
        )
        existing_case = self._primitive_by_key.get(primitive_key)
        if existing_case is not None:
            return self.primitives[existing_case]

        canonical_shape = component["shape_name"]
        component_case = sanitize_name(
            f"{component['component_operator']}__{canonical_shape}__"
            f"{component['impl_profile']}__{component['clock_profile']}"
        )
        primitive = {
            **component,
            "shape_name": canonical_shape,
            "component_case_name": component_case,
            "op_spec": op_spec,
            "current_cases": [],
            "component_roles": [],
        }
        self.primitives[component_case] = primitive
        self._primitive_by_key[primitive_key] = component_case
        return primitive


def _parse_pw_shape(shape_token: str) -> tuple[int, int, int, int, int]:
    if head_match := PW_HEAD_RE.match(shape_token):
        return (
            int(head_match.group("h")),
            int(head_match.group("cin")),
            int(head_match.group("cout")),
            1,
            1,
        )
    if ir_match := PW_IR_RE.match(shape_token):
        return (
            int(ir_match.group("h")),
            int(ir_match.group("cin")),
            int(ir_match.group("cout")),
            int(ir_match.group("expand")),
            0 if ir_match.group("role") == "proj" else 1,
        )
    if ir1_match := PW_IR_E1_PROJ_RE.match(shape_token):
        return (
            int(ir1_match.group("h")),
            int(ir1_match.group("cin")),
            int(ir1_match.group("cout")),
            1,
            0,
        )
    raise ValueError(f"Unsupported pointwise shape token: {shape_token}")


def _depthwise_policy(k: int, h: int, channels: int, stride: int, expand: int) -> tuple[str, str, str]:
    if k == 3 and (h, channels, stride, expand) in K3_FALLBACK_SHAPES:
        return "dw_conv_k3_fb", "p8_cb8_ii1_dsp1", "depthwise_fallback_p8_cb8_ii1"
    if k == 3:
        return "dw_conv_k3", "p16_cb16_ii1_dsp1", "depthwise_v2_p16_cb16_ii1"
    if (h, channels, stride, expand) in K5_P8_SHAPES:
        return "dw_conv_k5_lc", "p8_cb8_ii2_dsp1", "k5_medium_large_p8_cb8_ii2"
    return "dw_conv_k5", "p16_cb16_ii2_dsp1", "depthwise_v2_p16_cb16_ii2"


def _op_spec_for_component(component: dict[str, Any]) -> dict[str, Any]:
    shape = component["shape_payload"]
    h = int(shape["feature_h"])
    w = int(shape.get("feature_w", h))
    in_channels = int(shape["in_channels"])
    out_channels = int(shape.get("out_channels", in_channels))
    op_type = component["component_op_type"]
    kernel_size = int(component["kernel_size"])
    spec: dict[str, Any] = {
        "op": "pw_conv" if op_type == "pw_conv" else op_type,
        "kernel_size": kernel_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "stride": int(shape.get("stride", 1)),
        "groups": int(component["groups"]),
        "expand_ratio": int(component["expand_ratio"]),
        "input_resolution": [h, w],
        "bitwidth": 8,
        "input_parallelism": 1,
        "output_parallelism": 1,
        "unroll_factor": 1,
        "target_clock_mhz": 200.0,
    }

    impl = component["impl_profile"]
    if impl == "p16_pe16_simd4_ii2_dsp1":
        spec.update(
            {
                "input_parallelism": 4,
                "output_parallelism": 16,
                "pack_ch": 16,
                "ch_block": 16,
                "target_ii": 2,
                "tile_order": "pixel_major",
                "stream_order": "pixel_major",
                "dsp_pack": "1",
            }
        )
    elif impl == "p16_cb16_ii1_dsp1":
        spec.update(_fixed_vector_fields(pack_ch=16, ch_block=16, target_ii=1, tile_order=_tile_order(op_type)))
    elif impl == "p16_cb16_ii2_dsp1":
        spec.update(_fixed_vector_fields(pack_ch=16, ch_block=16, target_ii=2, tile_order=_tile_order(op_type)))
    elif impl == "p8_cb8_ii1_dsp1":
        spec.update(_fixed_vector_fields(pack_ch=8, ch_block=8, target_ii=1, tile_order=_tile_order(op_type)))
    elif impl == "p8_cb8_ii2_dsp1":
        spec.update(_fixed_vector_fields(pack_ch=8, ch_block=8, target_ii=2, tile_order=_tile_order(op_type)))
    elif impl == "stream_baseline_pi1_po1_u1":
        pass
    else:
        raise ValueError(f"Unsupported implementation profile: {impl}")

    return spec


def _fixed_vector_fields(*, pack_ch: int, ch_block: int, target_ii: int, tile_order: str) -> dict[str, Any]:
    return {
        "pack_ch": pack_ch,
        "ch_block": ch_block,
        "target_ii": target_ii,
        "tile_order": tile_order,
        "stream_order": tile_order,
        "dsp_pack": "1",
    }


def _tile_order(op_type: str) -> str:
    if op_type in {"identity_repack", "pw_conv"}:
        return "pixel_major"
    return "channel_major_replay"


def _load_existing_coverage(
    deployable_lut_path: Path,
    formal_status_sources: OrderedDict[str, Path],
) -> dict[str, Any]:
    deployable_by_shape: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    formal_by_shape: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    deployable_payload = _read_json(deployable_lut_path)
    for entry in deployable_payload.get("entries", []):
        spec = entry.get("op_spec")
        if not isinstance(spec, dict):
            continue
        deployable_by_shape[_shape_key(spec)].append(
            {
                "case_name": entry.get("source_case_name", ""),
                "source": f"v2_deployable_lut:{entry.get('source_dataset', '')}",
                "status": str(entry.get("status", "measured_implementation")),
                "profile": _profile_from_spec(spec),
                "deployable": bool(entry.get("deployable_at_200mhz")),
                "op_spec": spec,
            }
        )

    loaded_formal_sources: dict[str, str] = {}
    for source_name, path in formal_status_sources.items():
        if not path.exists():
            continue
        loaded_formal_sources[source_name] = str(path)
        payload = _read_json(path)
        for entry in payload.get("entries", []):
            status = str(entry.get("status", "")).strip()
            if not status or status == "missing":
                continue
            spec = entry.get("op_spec")
            if not isinstance(spec, dict):
                continue
            if _to_float(spec.get("target_clock_mhz"), 0.0) < 199.9:
                continue
            formal_by_shape[_shape_key(spec)].append(
                {
                    "case_name": entry.get("case_name", ""),
                    "source": source_name,
                    "status": status,
                    "profile": _profile_from_spec(spec),
                    "deployable": _is_formal_deployable(entry),
                    "op_spec": spec,
                }
            )

    return {
        "deployable_lut_path": str(deployable_lut_path),
        "formal_status_paths": loaded_formal_sources,
        "deployable_by_shape": deployable_by_shape,
        "formal_by_shape": formal_by_shape,
        "deployable_shape_count": len(deployable_by_shape),
        "formal_shape_count": len(formal_by_shape),
        "deployable_entry_count": sum(len(items) for items in deployable_by_shape.values()),
        "formal_entry_count": sum(len(items) for items in formal_by_shape.values()),
    }


def _select_existing(op_spec: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any] | None:
    key = _shape_key(op_spec)
    deployable = existing["deployable_by_shape"].get(key, [])
    if deployable:
        return sorted(deployable, key=_candidate_rank)[0]
    formal = existing["formal_by_shape"].get(key, [])
    if formal:
        return sorted(formal, key=_candidate_rank)[0]
    return None


def _candidate_rank(entry: dict[str, Any]) -> tuple[int, str, str]:
    status = str(entry.get("status", ""))
    deployable_rank = 0 if entry.get("deployable") else 1
    if status == "measured_implementation":
        status_rank = 0
    elif status == "measured_hls_only":
        status_rank = 1
    elif status.startswith("downstream"):
        status_rank = 2
    elif status.startswith("hls"):
        status_rank = 3
    else:
        status_rank = 4
    return deployable_rank, status_rank, str(entry.get("case_name", ""))


def _build_coverage_rows(map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in map_rows:
        grouped.setdefault(row["current_case_name"], []).append(row)

    coverage_rows: list[dict[str, str]] = []
    for case_name, rows in grouped.items():
        missing = [row for row in rows if row["coverage_status"] == "missing_v2_primitive"]
        covered = [row for row in rows if row["coverage_status"] != "missing_v2_primitive"]
        deployable = [row for row in rows if row["deployable_coverage"] == "true"]
        coverage_rows.append(
            {
                "current_case_name": case_name,
                "current_status": rows[0]["current_status"],
                "current_detail": rows[0]["current_detail"],
                "old_op_family": rows[0]["old_op_family"],
                "decomposition": rows[0]["decomposition"],
                "component_count": str(len(rows)),
                "has_full_v2_primitive_coverage": _bool_text(not missing),
                "has_full_deployable_primitive_coverage": _bool_text(len(deployable) == len(rows)),
                "covered_component_count": str(len(covered)),
                "deployable_component_count": str(len(deployable)),
                "missing_component_count": str(len(missing)),
                "component_roles": "|".join(row["component_role"] for row in rows),
                "component_cases": "|".join(row["component_case_name"] for row in rows),
                "selected_existing_cases": "|".join(row["selected_existing_case_name"] for row in rows),
                "selected_sources": "|".join(row["selected_source"] for row in rows),
                "coverage_statuses": "|".join(row["coverage_status"] for row in rows),
                "missing_component_cases": "|".join(row["component_case_name"] for row in missing),
                "notes": "old measured latency not merged; current84 uses v2 primitive coverage only",
            }
        )
    return coverage_rows


def _build_blocker_rows(output_dir: Path, map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    formal_status_path = output_dir / "formal_lut_status_v2_current84_missing.json"
    if not formal_status_path.exists():
        return []

    formal_payload = _read_json(formal_status_path)
    formal_by_case = {
        str(entry.get("case_name", "")): entry
        for entry in formal_payload.get("entries", [])
        if isinstance(entry, dict)
    }
    affected_by_component: dict[str, set[str]] = defaultdict(set)
    first_row_by_component: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in map_rows:
        component_case = row["component_case_name"]
        affected_by_component[component_case].add(row["current_case_name"])
        first_row_by_component.setdefault(component_case, row)

    blocker_rows: list[dict[str, str]] = []
    for component_case, formal_entry in sorted(formal_by_case.items()):
        status = str(formal_entry.get("status", ""))
        deployable = str(formal_entry.get("deployable_at_200mhz", "")).lower() == "true"
        if status == "measured_implementation" and deployable:
            continue
        map_row = first_row_by_component.get(component_case)
        if map_row is None:
            continue
        status_path = output_dir / "missing_p" / component_case / "vivado_downstream_status.json"
        downstream_payload = _read_json(status_path) if status_path.exists() else {}
        log_path = str(downstream_payload.get("log_path", ""))
        blocker_rows.append(
            {
                "component_case_name": component_case,
                "component_operator": map_row["component_operator"],
                "component_op_type": map_row["component_op_type"],
                "shape_name": map_row["shape_name"],
                "selected_source": map_row["selected_source"],
                "selected_status": map_row["selected_status"],
                "selected_existing_case_name": map_row["selected_existing_case_name"],
                "hls_status": str(formal_entry.get("hls_status", "")),
                "downstream_status": str(formal_entry.get("downstream_status", "")),
                "blocker_category": _blocker_category(formal_entry, downstream_payload),
                "blocker_reason": _blocker_reason(downstream_payload),
                "affected_current_case_count": str(len(affected_by_component.get(component_case, set()))),
                "affected_current_cases": "|".join(sorted(affected_by_component.get(component_case, set()))),
                "flow_tcl": _rel(Path(str(downstream_payload.get("flow_tcl", ""))))
                if downstream_payload.get("flow_tcl")
                else "",
                "log_path": _rel(Path(log_path)) if log_path else "",
            }
        )
    return blocker_rows


def _blocker_category(formal_entry: dict[str, Any], downstream_payload: dict[str, Any]) -> str:
    hls_status = str(formal_entry.get("hls_status", ""))
    downstream_status = str(formal_entry.get("downstream_status", ""))
    if hls_status not in {"success", "skipped_existing"}:
        return "hls_failure"
    if downstream_status in {"missing", ""}:
        return "downstream_missing"
    if downstream_status == "timeout":
        return "downstream_timeout"
    if downstream_status not in {"success", "skipped_existing"}:
        has_post_route = bool(downstream_payload.get("post_route_timing")) and bool(
            downstream_payload.get("post_route_utilization")
        )
        if has_post_route:
            return "post_route_timing_or_resource_failure"
        return "downstream_helper_runtime_failure"
    return "not_deployable"


def _blocker_reason(downstream_payload: dict[str, Any]) -> str:
    log_path = downstream_payload.get("log_path")
    if not log_path:
        return str(downstream_payload.get("status", "missing downstream status"))
    path = Path(str(log_path))
    if not path.exists():
        return f"log missing: {log_path}"
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        "couldn't read file",
        "invalid command name \"rt-undefined\"",
        "No Verilog files found",
        "Timing constraints are not met",
        "route_design failed",
        "place_design failed",
        "synth_design failed",
    ]
    for pattern in patterns:
        matches = [line.strip() for line in text.splitlines() if pattern.lower() in line.lower()]
        if matches:
            non_warning = [line for line in matches if not line.startswith("WARNING:")]
            return (non_warning or matches)[0]
    for line in reversed(text.splitlines()):
        if "ERROR:" in line:
            return line.strip()
    return str(downstream_payload.get("status", "failed"))


def _build_missing_rows(primitives: OrderedDict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for primitive in primitives.values():
        selected = _select_existing(primitive["op_spec"], primitive["_existing"])
        if selected is not None:
            continue
        shape = primitive["shape_payload"]
        spec = primitive["op_spec"]
        rows.append(
            {
                "component_case_name": primitive["component_case_name"],
                "component_operator": primitive["component_operator"],
                "component_op_type": primitive["component_op_type"],
                "shape_name": primitive["shape_name"],
                "impl_profile": primitive["impl_profile"],
                "clock_profile": primitive["clock_profile"],
                "profile_policy": primitive["profile_policy"],
                "feature_h": str(shape["feature_h"]),
                "feature_w": str(shape.get("feature_w", shape["feature_h"])),
                "in_channels": str(shape["in_channels"]),
                "out_channels": str(shape.get("out_channels", shape["in_channels"])),
                "kernel_size": str(spec["kernel_size"]),
                "stride": str(spec["stride"]),
                "groups": str(spec["groups"]),
                "expand_ratio": str(spec["expand_ratio"]),
                "first_current_case_name": primitive["current_cases"][0],
                "current_case_count": str(len(set(primitive["current_cases"]))),
                "current_cases": "|".join(sorted(set(primitive["current_cases"]))),
                "component_roles": "|".join(sorted(set(primitive["component_roles"]))),
                "missing_reason": "absent_from_v2_deployable_lut_and_non_missing_formal_status",
            }
        )
    return sorted(rows, key=lambda row: (row["component_operator"], row["component_case_name"]))


def _build_missing_config(
    missing_rows: list[dict[str, str]],
    primitives: OrderedDict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_cases = {row["component_case_name"] for row in missing_rows}
    shapes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    refs: OrderedDict[str, list[str]] = OrderedDict()
    for case_name, primitive in primitives.items():
        if case_name not in missing_cases:
            continue
        shapes[primitive["shape_name"]] = primitive["shape_payload"]
        refs.setdefault(primitive["component_operator"], []).append(primitive["shape_name"])

    operators: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for operator_name, shape_refs in refs.items():
        operators[operator_name] = _operator_entry(operator_name, sorted(shape_refs))

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
            "project_root": "../results/v2_current84/missing_p",
            "report_archive": "../results/v2_current84/missing_r",
            "csv_path": "../results/v2_current84/current84_missing_hls.csv",
            "manifest_path": "../results/v2_current84/current84_missing_hls_manifest.yaml",
            "synthesis_summary_json": "../results/v2_current84/current84_missing_hls_synth_summary.json",
            "parse_summary_json": "../results/v2_current84/current84_missing_hls_parse_summary.json",
            "downstream_csv_path": "../results/v2_current84/current84_missing_vivado_downstream.csv",
            "downstream_manifest_path": "../results/v2_current84/current84_missing_vivado_downstream_manifest.yaml",
            "downstream_parse_summary_json": "../results/v2_current84/current84_missing_vivado_downstream_parse_summary.json",
            "downstream_summary_json": "../results/v2_current84/current84_missing_vivado_downstream_summary.json",
            "formal_lut_status_json": "../results/v2_current84/formal_lut_status_v2_current84_missing.json",
        },
        "measurement_contract": {
            "contract_id": "packed_stream_v1_current84_missing_v2",
            "declaration": (
                "Missing-only current84 primitive batch. Existing v2 deployable entries and non-missing "
                "formal status rows are reused by shape and are intentionally excluded."
            ),
            "io_style": "packed_stream",
            "input_axis_rule": "PACK_CH * bitwidth bits for fixed-vector primitives; existing packed-stream endpoint templates for stem/GAP/FC",
            "output_axis_rule": "PACK_CH * bitwidth bits for fixed-vector primitives; endpoint template native stream width for stem/GAP/FC",
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
            "p8_cb8_ii1_dsp1": {
                "pack_ch": 8,
                "ch_block": 8,
                "pipeline_ii": 1,
                "target_ii": 1,
                "dsp_pack": "1",
            },
            "p8_cb8_ii2_dsp1": {
                "pack_ch": 8,
                "ch_block": 8,
                "pipeline_ii": 2,
                "target_ii": 2,
                "dsp_pack": "1",
            },
            "p16_pe16_simd4_ii2_dsp1": {
                "pack_ch": 16,
                "ch_block": 16,
                "input_parallelism": 4,
                "output_parallelism": 16,
                "array_partition_factor": 16,
                "pipeline_ii": 2,
                "target_ii": 2,
                "dsp_pack": "1",
            },
            "stream_baseline_pi1_po1_u1": {
                "fixed_vector_params": 0,
                "max_axis_bits": 4096,
                "input_parallelism": 1,
                "output_parallelism": 1,
                "unroll_factor": 1,
                "array_partition_factor": 4,
                "pipeline_ii": 1,
                "target_ii": 1,
            },
        },
        "representative_shapes": shapes,
        "operators": operators,
    }


def _operator_entry(operator_name: str, shape_refs: list[str]) -> dict[str, Any]:
    if operator_name == "dw_conv_k3":
        return _dw_operator("dw_conv_k3", "../templates/dw_conv_fixed_vector.cpp.tmpl", "dwk3_cur84", shape_refs, "p16_cb16_ii1_dsp1", 3, 1)
    if operator_name == "dw_conv_k3_fb":
        return _dw_operator("dw_conv_k3", "../templates/dw_conv_fixed_vector.cpp.tmpl", "dwk3fb_cur84", shape_refs, "p8_cb8_ii1_dsp1", 3, 1)
    if operator_name == "dw_conv_k5":
        return _dw_operator("dw_conv_k5", "../templates/dw_conv_fixed_vector.cpp.tmpl", "dwk5_cur84", shape_refs, "p16_cb16_ii2_dsp1", 5, 2)
    if operator_name == "dw_conv_k5_lc":
        return _dw_operator("dw_conv_k5", "../templates/dw_conv_fixed_vector.cpp.tmpl", "dwk5lc_cur84", shape_refs, "p8_cb8_ii2_dsp1", 5, 2)
    if operator_name == "pw_conv_v21":
        return {
            "enabled": True,
            "template": "../templates/pw_conv_fixed_vector_v21.cpp.tmpl",
            "top_function": "pwk_v21_cur84",
            "op_type": "pw_conv",
            "shape_refs": shape_refs,
            "implementation_refs": ["p16_pe16_simd4_ii2_dsp1"],
            "clock_profile_refs": ["main_5ns"],
            "parameters": {
                "kernel_size": 1,
                "padding": 0,
                "tile_order": "pixel_major",
                "stream_order": "pixel_major",
            },
            "op_spec_defaults": {
                "op": "pw_conv",
                "groups": 1,
            },
        }
    if operator_name == "identity_repack":
        return {
            "enabled": True,
            "template": "../templates/identity_repack.cpp.tmpl",
            "top_function": "identity_repack_cur84",
            "op_type": "identity_repack",
            "shape_refs": shape_refs,
            "implementation_refs": ["p16_cb16_ii1_dsp1"],
            "clock_profile_refs": ["main_5ns"],
            "parameters": {
                "kernel_size": 1,
                "padding": 0,
                "tile_order": "pixel_major",
                "stream_order": "pixel_major",
            },
            "op_spec_defaults": {
                "op": "identity_repack",
                "groups": 1,
                "expand_ratio": 1,
            },
        }
    if operator_name == "stem_conv_k3_s2":
        return _stream_operator(
            operator_name,
            "../templates/stem_conv_k3_s2.cpp.tmpl",
            "stem_conv_k3_s2_cur84",
            shape_refs,
            kernel_size=3,
            padding=1,
            extra_parameters={"apply_relu": 1},
            op_spec_defaults={"op": "stem_conv_k3_s2", "groups": 1, "expand_ratio": 1},
        )
    if operator_name == "global_avg_pool":
        return _stream_operator(
            operator_name,
            "../templates/global_avg_pool.cpp.tmpl",
            "global_avg_pool_cur84",
            shape_refs,
            kernel_size=7,
            padding=0,
            op_spec_defaults={"op": "global_avg_pool", "groups": 1, "expand_ratio": 1},
        )
    if operator_name == "fc_layer":
        return _stream_operator(
            operator_name,
            "../templates/fc_layer.cpp.tmpl",
            "fc_layer_cur84",
            shape_refs,
            kernel_size=1,
            padding=0,
            op_spec_defaults={"op": "fc_layer", "groups": 1, "expand_ratio": 1},
        )
    raise ValueError(f"Unsupported operator for current84 missing config: {operator_name}")


def _dw_operator(
    op_type: str,
    template: str,
    top_function: str,
    shape_refs: list[str],
    impl_profile: str,
    kernel_size: int,
    padding: int,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "template": template,
        "top_function": top_function,
        "op_type": op_type,
        "shape_refs": shape_refs,
        "implementation_refs": [impl_profile],
        "clock_profile_refs": ["main_5ns"],
        "parameters": {
            "kernel_size": kernel_size,
            "padding": padding,
            "apply_relu": 1,
        },
        "op_spec_defaults": {
            "op": op_type,
        },
    }


def _stream_operator(
    operator_name: str,
    template: str,
    top_function: str,
    shape_refs: list[str],
    *,
    kernel_size: int,
    padding: int,
    extra_parameters: dict[str, Any] | None = None,
    op_spec_defaults: dict[str, Any],
) -> dict[str, Any]:
    parameters = {
        "kernel_size": kernel_size,
        "padding": padding,
        "fixed_vector_params": 0,
    }
    if extra_parameters:
        parameters.update(extra_parameters)
    return {
        "enabled": True,
        "template": template,
        "top_function": top_function,
        "op_type": operator_name,
        "shape_refs": shape_refs,
        "implementation_refs": ["stream_baseline_pi1_po1_u1"],
        "clock_profile_refs": ["main_5ns"],
        "parameters": parameters,
        "op_spec_defaults": op_spec_defaults,
    }


def _metadata(
    *,
    status_csv: Path,
    deployable_lut: Path,
    output_config: Path,
    current_rows: list[dict[str, str]],
    map_rows: list[dict[str, str]],
    missing_rows: list[dict[str, str]],
    existing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "task": "current84 v2 primitive reconstruction",
        "source_status_csv": str(status_csv),
        "source_deployable_lut": str(deployable_lut),
        "formal_status_sources": existing["formal_status_paths"],
        "output_missing_config": str(output_config),
        "current_case_count": len(current_rows),
        "current_status_counts": dict(Counter(row.get("status", "") for row in current_rows)),
        "primitive_map_rows": len(map_rows),
        "unique_current84_primitives": len({row["component_case_name"] for row in map_rows}),
        "covered_current84_primitives": len(
            {
                row["component_case_name"]
                for row in map_rows
                if row["coverage_status"] != "missing_v2_primitive"
            }
        ),
        "deployable_current84_primitives": len(
            {
                row["component_case_name"]
                for row in map_rows
                if row["deployable_coverage"] == "true"
            }
        ),
        "missing_current84_primitives": len(missing_rows),
        "missing_by_operator": dict(Counter(row["component_operator"] for row in missing_rows)),
        "coverage_policy": (
            "coverage is shape-level: deployable_lut_v2 is preferred, then non-missing formal status at 200 MHz; "
            "formal status='missing' placeholders do not suppress missing-list generation"
        ),
        "latency_policy": "old board measured latency is not merged into v2 artifacts",
    }


def _readme_text(
    *,
    status_csv: Path,
    deployable_lut: Path,
    output_config: Path,
    current_rows: list[dict[str, str]],
    map_rows: list[dict[str, str]],
    missing_rows: list[dict[str, str]],
    coverage_rows: list[dict[str, str]],
    blocker_rows: list[dict[str, str]],
    existing: dict[str, Any],
) -> str:
    status_counts = Counter(row.get("status", "") for row in current_rows)
    component_counts = Counter(row["component_operator"] for row in map_rows)
    missing_counts = Counter(row["component_operator"] for row in missing_rows)
    blocker_counts = Counter(row["component_operator"] for row in blocker_rows)
    full_coverage = Counter(row["has_full_v2_primitive_coverage"] for row in coverage_rows)
    deployable_coverage = Counter(row["has_full_deployable_primitive_coverage"] for row in coverage_rows)
    current84_missing_status_path = Path(existing["formal_status_paths"].get("v2_current84_missing", ""))
    current84_missing_status_counts = _formal_status_counts(current84_missing_status_path)
    current84_identity_status_path = Path(existing["formal_status_paths"].get("v2_current84_identity_repack_14_64", ""))
    current84_identity_status_counts = _formal_status_counts(current84_identity_status_path)
    formal_lut_path = DEFAULT_OUTPUT_DIR / "formal_lut_v2_current84_missing.json"
    formal_lut_entries = _formal_lut_entry_count(formal_lut_path)
    identity_lut_path = DEFAULT_OUTPUT_DIR / "formal_lut_v2_current84_identity_repack_14_64.json"
    identity_lut_entries = _formal_lut_entry_count(identity_lut_path)
    run15_config = BUILDER_ROOT / "configs" / "candidate_kernels_v2_current84_missing_run15_2026_05_17.yaml"
    identity_config = BUILDER_ROOT / "configs" / "candidate_kernels_v2_current84_identity_repack_14_64.yaml"
    lines = [
        "# current84 v2 Primitive Reconstruction",
        "",
        "This directory contains the current implementation 84-combo v2 reconstruction.",
        "The old board-measured 40 rows are used only as case source rows; their latency is not merged into the v2 LUT.",
        "",
        "## Inputs",
        "",
        f"- `{_rel(status_csv)}`",
        f"- `{_rel(deployable_lut)}`",
        "- `hls_lut_builder/results/v2_deployable_lut/combo_deployability_v2.csv`",
        "- `hls_lut_builder/results/v2_deployable_lut/depthwise_blocker_summary_2026_05_16.csv`",
        "- `hls_lut_builder/results/v2_failed44_fixed_vector/failed44_primitive_map.csv`",
        "- `hls_lut_builder/configs/candidate_kernels_v2_failed44_fixed_vector.yaml`",
        "- `hls_lut_builder/configs/candidate_kernels_v2_pointwise_v21.yaml`",
        "- `hls_lut_builder/configs/candidate_kernels_v2_k5_low_complexity.yaml`",
        "- `hls_lut_builder/configs/candidate_kernels_v2_depthwise_fallback_blockers.yaml`",
        "- `hls_lut_builder/scripts/merge_deployable_lut_v2.py`",
        "- `hls_lut_builder/scripts/generate_v2_failed44_config.py`",
        "",
        "Additional formal-status reuse sources:",
    ]
    for source_name, source_path in existing["formal_status_paths"].items():
        lines.append(f"- `{source_name}`: `{_rel(Path(source_path))}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `current84_primitive_map.csv` and `current84_primitive_map.json`",
            "- `current84_missing_primitives.csv`",
            "- `current84_missing_blockers.csv`",
            "- `formal_lut_v2_current84_missing.json`",
            "- `formal_lut_status_v2_current84_missing.json`",
            "- `formal_lut_v2_current84_identity_repack_14_64.json`",
            "- `formal_lut_status_v2_current84_identity_repack_14_64.json`",
            f"- `{_rel(output_config)}`",
            f"- `{_rel(run15_config)}`",
            f"- `{_rel(identity_config)}`",
            "- `current84_combo_coverage.csv`",
            "",
            "## Counts",
            "",
            f"- Current source combos: {len(current_rows)} ({_format_counter(status_counts)})",
            f"- Primitive map rows: {len(map_rows)}",
            f"- Unique current84 primitive shapes: {len({row['component_case_name'] for row in map_rows})}",
            f"- Component rows by operator: {_format_counter(component_counts)}",
            f"- Combos with full v2 primitive coverage: {_format_counter(full_coverage)}",
            f"- Combos with full deployable primitive coverage: {_format_counter(deployable_coverage)}",
            f"- Missing primitive shapes: {len(missing_rows)} ({_format_counter(missing_counts)})",
            f"- Blocker primitive shapes: {len(blocker_rows)} ({_format_counter(blocker_counts)})",
            "",
            "## current84_missing Verification",
            "",
            "- HLS/Vivado was run only for missing-only current84 primitives; existing failed44 v2 LUT entries were not rerun.",
            "- `deployable_lut_v2.json` was not modified, and old board-measured latency was not merged.",
            "- `formal_lut_status_v2_current84_missing.json` is now a reuse source; only deployable measured implementation rows count toward deployable coverage.",
            f"- current84_missing formal status counts: {_format_counter(current84_missing_status_counts)}",
            f"- `formal_lut_v2_current84_missing.json` entries from post-route downstream manifest: {formal_lut_entries}",
            f"- Initial 15-case run config snapshot: `{_rel(run15_config)}`",
            "- `identity_repack_14_64` was remeasured under the current84 contract to remove the old `v2_fixed_vector` source exception.",
            f"- current84 identity formal status counts: {_format_counter(current84_identity_status_counts)}",
            f"- `formal_lut_v2_current84_identity_repack_14_64.json` entries from post-route downstream manifest: {identity_lut_entries}",
            f"- One-case identity retest config: `{_rel(identity_config)}`",
            f"- Full deployable combo coverage is {deployable_coverage.get('true', 0)} / {len(coverage_rows)} after this run.",
            f"- Non-deployable formal status rows are recorded in `current84_missing_blockers.csv`: {len(blocker_rows)} primitive blocker(s).",
            "",
            "## Policy",
            "",
            "- MBConv is always decomposed as `pw_expand + dw + pw_project`.",
            "- Pointwise primitives use pointwise v2.1 `p16_pe16_simd4_ii2`.",
            "- Depthwise k3 normally uses `p16/cb16/II1`; the resolved k3 C480 fallback shape reuses `p8/cb8/II1`.",
            "- Depthwise k5 normally uses `p16/cb16/II2`; medium/large k5 shapes use the resolved `p8/cb8/II2` policy.",
            "- Coverage is shape-level. `deployable_lut_v2.json` is preferred, then non-missing formal status rows at 200 MHz.",
            "- Formal status rows with `status=missing` are placeholders and do not suppress the missing list.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python .\\hls_lut_builder\\scripts\\generate_v2_current84_config.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _formal_status_counts(path: Path) -> Counter[str]:
    if not str(path) or not path.exists() or not path.is_file():
        return Counter()
    payload = _read_json(path)
    return Counter(str(entry.get("status", "")) for entry in payload.get("entries", []) if isinstance(entry, dict))


def _formal_lut_entry_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    payload = _read_json(path)
    entries = payload.get("entries", [])
    return len(entries) if isinstance(entries, list) else 0


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BUILDER_ROOT.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _shape_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    h, w = _resolution(spec)
    return (
        _normalized_op(spec),
        int(spec.get("kernel_size", 0)),
        int(spec.get("in_channels", 0)),
        int(spec.get("out_channels", 0)),
        int(spec.get("stride", 1)),
        int(spec.get("groups", 1)),
        int(spec.get("expand_ratio", 1)),
        h,
        w,
    )


def _normalized_op(spec: dict[str, Any]) -> str:
    op = str(spec.get("op", ""))
    if op == "conv" and int(spec.get("kernel_size", 0)) == 1 and int(spec.get("groups", 1)) == 1:
        return "pw_conv"
    return op


def _resolution(spec: dict[str, Any]) -> tuple[int, int]:
    resolution = spec.get("input_resolution", [0, 0])
    return int(resolution[0]), int(resolution[1])


def _profile_from_spec(spec: dict[str, Any]) -> str:
    if "pack_ch" not in spec:
        return "stream_baseline"
    pack = int(spec.get("pack_ch", 0))
    cb = int(spec.get("ch_block", 0))
    ii = int(spec.get("target_ii", 0))
    if _normalized_op(spec) == "pw_conv":
        return f"p{pack}_pe{int(spec.get('output_parallelism', 0))}_simd{int(spec.get('input_parallelism', 0))}_ii{ii}"
    return f"p{pack}_cb{cb}_ii{ii}"


def _is_formal_deployable(entry: dict[str, Any]) -> bool:
    return (
        str(entry.get("status")) == "measured_implementation"
        and str(entry.get("deployable_at_200mhz")).lower() == "true"
        and str(entry.get("downstream_status", "success")) in {"success", "skipped_existing", ""}
    )


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plain_yaml(value: Any) -> Any:
    if isinstance(value, OrderedDict):
        return {key: _plain_yaml(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_yaml(item) for item in value]
    return value


if __name__ == "__main__":
    main()
