#!/usr/bin/env python3
"""Merge the original MobileNetA evidence with low-resource recovery rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HLS_SCRIPTS = ROOT / "hls_lut_builder" / "scripts"
if str(HLS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HLS_SCRIPTS))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from common import build_cases, find_first_existing_report, load_config  # noqa: E402
from build_zcu104_mobilenetA_full_formal_lut_from_reports import (  # noqa: E402
    READY_HLS,
    READY_ROUTE,
    canonical_sha256,
    git_value,
    parse_hls_xml,
    sha256_file,
    status_ok,
)
from run_vivado_downstream import parse_downstream_metrics  # noqa: E402


BASE_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_zcu104_mobilenetA_full_dmdc_cfem_v1.yaml"
RECOVERY_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_zcu104_mobilenetA_recovery_v2.yaml"
GAP_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_zcu104_gap1280_v3.yaml"
OUTPUT_DIR = ROOT / "artifacts" / "zcu104_full_campaign_v1" / "operator_cost_lut" / "mobilenetA_full_dmdc_cfem_v2_recovered"
GAP1280_OUTPUT_DIR = ROOT / "artifacts" / "zcu104_full_campaign_v1" / "operator_cost_lut" / "mobilenetA_full_dmdc_cfem_v3_gap1280"
SEARCH_SPACE_VERSION = "mobilenetA_full_dmdc_cfem_v1"
HARDWARE_LUT_VERSION = "mobilenetA_full_dmdc_cfem_v3_gap1280"
FORMAL_DATA_KIND = "FRESH_ZCU104_HLS_ROUTE_OPERATOR_COST_WITH_RECOVERY"


def logical_identity(case: Any) -> tuple[Any, ...]:
    spec = case.op_spec
    return (
        str(spec.get("op")),
        case.shape_name,
        int(spec.get("in_channels", 0)),
        int(spec.get("out_channels", 0)),
        int(spec.get("kernel_size", 0)),
        int(spec.get("stride", 0)),
        int(spec.get("expand_ratio", 1)),
        int(case.parameters.get("apply_relu", 0)),
    )


def collect_evidence(case: Any) -> dict[str, Any]:
    report_path = find_first_existing_report(case.report_candidates)
    hls_ok, hls_status, _hls_payload = status_ok(case.project_dir / "synthesis_status.json", READY_HLS)
    route_ok, route_status, route_payload = status_ok(
        case.project_dir / "vivado_downstream_status.json", READY_ROUTE
    )
    if report_path is None:
        hls_ok = False

    route_metrics = route_payload.get("metrics") if isinstance(route_payload.get("metrics"), dict) else {}
    if route_ok:
        reparsed_metrics = parse_downstream_metrics(case)
        if reparsed_metrics is not None:
            route_metrics = reparsed_metrics
    post_route = route_metrics.get("post_route") if isinstance(route_metrics.get("post_route"), dict) else {}
    if not post_route:
        route_ok = False
    if route_ok and (int(post_route.get("lut") or 0) <= 0 or int(post_route.get("ff") or 0) <= 0):
        route_ok = False

    hls_metrics: dict[str, Any] = {}
    if report_path is not None and report_path.suffix.lower() == ".xml":
        try:
            hls_metrics = parse_hls_xml(report_path, case.target_clock_mhz)
        except Exception as exc:
            hls_ok = False
            hls_metrics = {"parse_error": str(exc)}

    return {
        "case_name": case.case_name,
        "implementation_profile": case.implementation_name,
        "op_id": case.op_id,
        "ready": bool(hls_ok and route_ok),
        "hls_status": hls_status,
        "route_status": route_status,
        "hls_report_path": str(report_path) if report_path else None,
        "hls_report_sha256": sha256_file(report_path) if report_path and report_path.is_file() else None,
        "route_report_dir": route_payload.get("report_dir"),
        "hls_metrics": hls_metrics,
        "post_synth_metrics": route_metrics.get("post_synth", {}),
        "post_route_metrics": post_route,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_config_path = Path(args.base_config).resolve()
    recovery_config_path = Path(args.recovery_config).resolve()
    gap_config_path = Path(args.gap_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cases = build_cases(load_config(base_config_path), sort_cases=True)
    recovery_cases = build_cases(load_config(recovery_config_path), sort_cases=True)
    gap_cases = build_cases(load_config(gap_config_path), sort_cases=True)
    recovery_by_identity = {logical_identity(case): case for case in recovery_cases}
    if len(recovery_by_identity) != 13:
        raise ValueError(f"Recovery config must define 13 unique logical rows, got {len(recovery_by_identity)}")
    gap1280_cases = [
        case
        for case in gap_cases
        if case.operator == "global_avg_pool" and case.shape_name == "gap_7_1280"
    ]
    if len(gap1280_cases) != 1:
        raise ValueError(
            f"Gap config must define exactly one global_avg_pool/gap_7_1280 row, got {len(gap1280_cases)}"
        )
    gap1280_case = gap1280_cases[0]

    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    recovery_used = 0
    gap1280_used = 0
    for base_case in base_cases:
        base_evidence = collect_evidence(base_case)
        selected_case = base_case
        selected_evidence = base_evidence
        recovery_case = recovery_by_identity.get(logical_identity(base_case))
        recovery_evidence = collect_evidence(recovery_case) if recovery_case is not None else None
        gap1280_evidence = None
        if base_case.operator == "global_avg_pool" and base_case.shape_name == "gap_7_320":
            gap1280_evidence = collect_evidence(gap1280_case)
            selected_case = gap1280_case
            selected_evidence = gap1280_evidence
            gap1280_used += 1
        elif not base_evidence["ready"] and recovery_case is not None and recovery_evidence and recovery_evidence["ready"]:
            selected_case = recovery_case
            selected_evidence = recovery_evidence
            recovery_used += 1

        post_route = selected_evidence["post_route_metrics"]
        hls_metrics = selected_evidence["hls_metrics"]
        timing_met = (
            post_route.get("setup_wns_ns") is not None
            and float(post_route.get("setup_wns_ns")) >= 0.0
            and float(post_route.get("setup_tns_ns") or 0.0) == 0.0
        )
        row = {
            "row_key": selected_case.case_name,
            "logical_op_id": selected_case.op_id,
            "operator": selected_case.operator,
            "shape_name": selected_case.shape_name,
            "op_spec": selected_case.op_spec,
            "target_board": "ZCU104",
            "target_part": selected_case.part,
            "target_clock_mhz": selected_case.target_clock_mhz,
            "ready": bool(selected_evidence["ready"]),
            "timing_met": timing_met,
            "evidence_source": (
                "gap1280_v3"
                if selected_case is gap1280_case
                else "recovery_v2" if selected_case is not base_case else "base_v1"
            ),
            "evidence_case_name": selected_evidence["case_name"],
            "implementation_profile": selected_evidence["implementation_profile"],
            "hls_status": selected_evidence["hls_status"],
            "route_status": selected_evidence["route_status"],
            "hls_report_path": selected_evidence["hls_report_path"],
            "hls_report_sha256": selected_evidence["hls_report_sha256"],
            "route_report_dir": selected_evidence["route_report_dir"],
            "hls_metrics": hls_metrics,
            "post_synth_metrics": selected_evidence["post_synth_metrics"],
            "post_route_metrics": post_route,
            "base_v1_terminal_status": {
                "hls_status": base_evidence["hls_status"],
                "route_status": base_evidence["route_status"],
            },
            "cost": {
                "latency_cycles_hls": hls_metrics.get("latency_cycles"),
                "latency_ns_hls": hls_metrics.get("latency_ns"),
                "latency_ms_hls": None if hls_metrics.get("latency_ns") is None else float(hls_metrics["latency_ns"]) / 1_000_000.0,
                "lut_post_route": post_route.get("lut"),
                "ff_post_route": post_route.get("ff"),
                "dsp_post_route": post_route.get("dsp"),
                "bram_tile_post_route": post_route.get("block_ram_tile"),
                "power_w_vivado_vectorless": post_route.get("power_w"),
            },
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
        if not row["ready"]:
            blockers.append(
                {
                    "row_key": selected_case.case_name,
                    "base_hls_status": base_evidence["hls_status"],
                    "base_route_status": base_evidence["route_status"],
                    "recovery_case_name": recovery_evidence.get("case_name") if recovery_evidence else None,
                    "recovery_hls_status": recovery_evidence.get("hls_status") if recovery_evidence else None,
                    "recovery_route_status": recovery_evidence.get("route_status") if recovery_evidence else None,
                    "gap1280_case_name": gap1280_evidence.get("case_name") if gap1280_evidence else None,
                    "gap1280_hls_status": gap1280_evidence.get("hls_status") if gap1280_evidence else None,
                    "gap1280_route_status": gap1280_evidence.get("route_status") if gap1280_evidence else None,
                }
            )

    ready_count = sum(1 for row in rows if row["ready"])
    search_ready = len(rows) == 88 and ready_count == 88 and not blockers
    generated_at = datetime.now().isoformat(timespec="seconds")
    status = "PASS_FORMAL_HLS_ROUTE_COVERAGE_88_OF_88" if search_ready else f"BLOCKED_FORMAL_HLS_ROUTE_COVERAGE_{ready_count}_OF_{len(rows)}"
    payload = {
        "artifact_type": "hardware_cost_lut",
        "data_kind": FORMAL_DATA_KIND,
        "search_space_version": SEARCH_SPACE_VERSION,
        "hardware_lut_version": HARDWARE_LUT_VERSION,
        "generated_at": generated_at,
        "search_ready": search_ready,
        "status": status,
        "source": {
            "git_commit": git_value(["git", "rev-parse", "HEAD"]),
            "git_dirty_porcelain": git_value(["git", "status", "--short"]),
            "base_config_path": str(base_config_path),
            "base_config_sha256": sha256_file(base_config_path),
            "recovery_config_path": str(recovery_config_path),
            "recovery_config_sha256": sha256_file(recovery_config_path),
            "gap_config_path": str(gap_config_path),
            "gap_config_sha256": sha256_file(gap_config_path),
        },
        "claim_boundary": {
            "hls_route_cost": "FORMAL_IF_SEARCH_READY_TRUE",
            "board_classification": "NOT_MEASURED_IN_THIS_LUT",
            "external_power": "NOT_MEASURED_IN_THIS_LUT",
            "vivado_power": "VECTORLESS_ESTIMATE_ONLY",
        },
        "coverage": {
            "expected_rows": 88,
            "ready_rows": ready_count,
            "blocked_rows": len(blockers),
            "base_v1_rows_used": len(rows) - recovery_used - gap1280_used,
            "recovery_v2_rows_used": recovery_used,
            "gap1280_v3_rows_used": gap1280_used,
            "timing_met_rows": sum(1 for row in rows if row["timing_met"]),
        },
        "rows": rows,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    coverage = {
        "artifact_type": "operator_cost_lut_coverage_audit",
        "data_kind": FORMAL_DATA_KIND,
        "search_space_version": SEARCH_SPACE_VERSION,
        "hardware_lut_version": HARDWARE_LUT_VERSION,
        "generated_at": generated_at,
        "search_ready": search_ready,
        "status": status,
        "expected_rows": 88,
        "ready_rows": ready_count,
        "base_v1_rows_used": len(rows) - recovery_used - gap1280_used,
        "recovery_v2_rows_used": recovery_used,
        "gap1280_v3_rows_used": gap1280_used,
        "blocked_rows": blockers,
        "operator_cost_payload_sha256": payload["payload_sha256"],
        "claim_boundary": payload["claim_boundary"],
    }
    coverage["payload_sha256"] = canonical_sha256(coverage)

    lut_path = output_dir / "hardware_cost_lut.formal.json"
    coverage_path = output_dir / "coverage_audit.formal.json"
    lut_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "lut_path": str(lut_path),
        "coverage_path": str(coverage_path),
        "status": status,
        "ready_rows": ready_count,
        "recovery_rows_used": recovery_used,
        "gap1280_rows_used": gap1280_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--recovery-config", default=str(RECOVERY_CONFIG))
    parser.add_argument("--gap-config", default=str(GAP_CONFIG))
    parser.add_argument("--output-dir", default=str(GAP1280_OUTPUT_DIR))
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
