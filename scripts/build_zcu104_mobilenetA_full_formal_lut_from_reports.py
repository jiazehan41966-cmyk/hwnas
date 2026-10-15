#!/usr/bin/env python3
"""Build MobileNetA ZCU104 operator-cost LUT from fresh HLS/route reports.

The script is intentionally fail-closed: it writes search_ready=true only when
every configured case has a HLS report and a Vivado downstream route status.
This is a HLS/route cost table, not board runtime or external-power evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
HLS_SCRIPTS = ROOT / "hls_lut_builder" / "scripts"
if str(HLS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HLS_SCRIPTS))

from common import build_cases, find_first_existing_report, load_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "hls_lut_builder"
    / "configs"
    / "candidate_kernels_zcu104_mobilenetA_full_dmdc_cfem_v1.yaml"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "operator_cost_lut"
    / "mobilenetA_full_dmdc_cfem_v1"
)
READY_HLS = {"success", "skipped_existing"}
READY_ROUTE = {"success", "skipped_existing"}
SEARCH_SPACE_VERSION = "mobilenetA_full_dmdc_cfem_v1"
FORMAL_DATA_KIND = "FRESH_ZCU104_HLS_ROUTE_OPERATOR_COST"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def xml_text(root: ET.Element, path: str) -> str | None:
    elem = root.find(path)
    if elem is None or elem.text is None:
        return None
    text = elem.text.strip()
    return text or None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_hls_xml(path: Path, target_clock_mhz: float) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    latency = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    resources = root.find(".//AreaEstimates/Resources")
    timing = root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis")

    def child_text(parent: ET.Element | None, name: str) -> str | None:
        if parent is None:
            return None
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == name and child.text:
                return child.text.strip()
        return None

    cycles = to_int(child_text(latency, "Worst-caseLatency"))
    latency_ns = to_float(child_text(latency, "Worst-caseRealTimeLatency"))
    estimated_clock_ns = to_float(child_text(timing, "EstimatedClockPeriod"))
    if latency_ns is None and cycles is not None:
        clock_ns = estimated_clock_ns or (1000.0 / target_clock_mhz)
        latency_ns = float(cycles) * clock_ns

    return {
        "latency_cycles": cycles,
        "latency_ns": latency_ns,
        "ii": to_int(child_text(latency, "Interval-max")),
        "estimated_clock_period_ns": estimated_clock_ns,
        "lut": to_int(child_text(resources, "LUT")),
        "ff": to_int(child_text(resources, "FF")),
        "dsp": to_int(child_text(resources, "DSP")),
        "bram_18k": to_int(child_text(resources, "BRAM_18K")),
        "uram": to_int(child_text(resources, "URAM")),
    }


def status_ok(path: Path, allowed: set[str]) -> tuple[bool, str, dict[str, Any]]:
    if not path.is_file():
        return False, "missing", {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return False, "unreadable", {}
    status = str(payload.get("status") or "")
    return status in allowed, status, payload


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    cases = build_cases(config, sort_cases=True)
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for case in cases:
        report_path = find_first_existing_report(case.report_candidates)
        hls_ok, hls_status, hls_payload = status_ok(case.project_dir / "synthesis_status.json", READY_HLS)
        route_ok, route_status, route_payload = status_ok(
            case.project_dir / "vivado_downstream_status.json", READY_ROUTE
        )
        if report_path is None:
            hls_ok = False
        route_metrics = route_payload.get("metrics") if isinstance(route_payload.get("metrics"), dict) else {}
        post_route = route_metrics.get("post_route") if isinstance(route_metrics.get("post_route"), dict) else {}
        if not post_route:
            route_ok = False

        hls_metrics: dict[str, Any] = {}
        if report_path is not None and report_path.suffix.lower() == ".xml":
            try:
                hls_metrics = parse_hls_xml(report_path, case.target_clock_mhz)
            except Exception as exc:
                hls_ok = False
                hls_metrics = {"parse_error": str(exc)}

        row_ready = bool(hls_ok and route_ok)
        timing_met = (
            post_route.get("setup_wns_ns") is not None
            and float(post_route.get("setup_wns_ns")) >= 0.0
            and float(post_route.get("setup_tns_ns") or 0.0) == 0.0
        )
        row = {
            "row_key": case.case_name,
            "case_name": case.case_name,
            "op_id": case.op_id,
            "operator": case.operator,
            "shape_name": case.shape_name,
            "implementation_profile": case.implementation_name,
            "clock_profile": case.clock_profile_name,
            "target_board": "ZCU104",
            "target_part": case.part,
            "target_clock_mhz": case.target_clock_mhz,
            "op_spec": case.op_spec,
            "ready": row_ready,
            "timing_met": timing_met,
            "hls_status": hls_status,
            "route_status": route_status,
            "hls_report_path": str(report_path) if report_path else None,
            "hls_report_sha256": sha256_file(report_path) if report_path and report_path.is_file() else None,
            "route_report_dir": route_payload.get("report_dir"),
            "post_route_timing": route_payload.get("post_route_timing"),
            "post_route_utilization": route_payload.get("post_route_utilization"),
            "hls_metrics": hls_metrics,
            "post_synth_metrics": route_metrics.get("post_synth", {}),
            "post_route_metrics": post_route,
            "cost": {
                "latency_cycles_hls": hls_metrics.get("latency_cycles"),
                "latency_ns_hls": hls_metrics.get("latency_ns"),
                "latency_ms_hls": None
                if hls_metrics.get("latency_ns") is None
                else float(hls_metrics["latency_ns"]) / 1_000_000.0,
                "lut_post_route": post_route.get("lut"),
                "ff_post_route": post_route.get("ff"),
                "dsp_post_route": post_route.get("dsp"),
                "bram_tile_post_route": post_route.get("block_ram_tile"),
                "power_w_vivado_vectorless": post_route.get("power_w"),
            },
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
        if not row_ready:
            blockers.append(
                {
                    "case_name": case.case_name,
                    "hls_status": hls_status,
                    "route_status": route_status,
                    "has_hls_report": report_path is not None,
                    "has_post_route_metrics": bool(post_route),
                }
            )

    ready_count = sum(1 for row in rows if row["ready"])
    search_ready = ready_count == len(rows) and not blockers
    payload = {
        "artifact_type": "hardware_cost_lut",
        "data_kind": FORMAL_DATA_KIND,
        "search_space_version": SEARCH_SPACE_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "search_ready": search_ready,
        "status": "PASS_FORMAL_HLS_ROUTE_COVERAGE_88_OF_88"
        if search_ready
        else f"BLOCKED_FORMAL_HLS_ROUTE_COVERAGE_{ready_count}_OF_{len(rows)}",
        "source": {
            "git_commit": git_value(["git", "rev-parse", "HEAD"]),
            "git_dirty_porcelain": git_value(["git", "status", "--short"]),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
        "claim_boundary": {
            "hls_route_cost": "FORMAL_IF_SEARCH_READY_TRUE",
            "board_classification": "NOT_MEASURED_IN_THIS_LUT",
            "external_power": "NOT_MEASURED_IN_THIS_LUT",
            "vivado_power": "VECTORLESS_ESTIMATE_ONLY",
        },
        "coverage": {
            "expected_rows": len(rows),
            "ready_rows": ready_count,
            "blocked_rows": len(blockers),
            "timing_met_rows": sum(1 for row in rows if row["timing_met"]),
            "timing_failed_rows": sum(1 for row in rows if row["ready"] and not row["timing_met"]),
        },
        "rows": rows,
    }
    coverage = {
        "artifact_type": "operator_cost_lut_coverage_audit",
        "data_kind": FORMAL_DATA_KIND,
        "search_space_version": SEARCH_SPACE_VERSION,
        "generated_at": payload["generated_at"],
        "search_ready": search_ready,
        "status": payload["status"],
        "expected_rows": len(rows),
        "ready_rows": ready_count,
        "blocked_rows": blockers,
        "claim_boundary": payload["claim_boundary"],
    }

    lut_path = output_dir / "hardware_cost_lut.formal.json"
    coverage_path = output_dir / "coverage_audit.formal.json"
    lut_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"lut_path": str(lut_path), "coverage_path": str(coverage_path), "status": payload["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
