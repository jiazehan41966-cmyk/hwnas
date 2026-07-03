#!/usr/bin/env python3
"""Timing and HLS-log gate for the isolated Phase0 v4 sonar stage3 k3 pilot.

This audit is local-only and read-only with respect to canonical ledgers. It
does not train, run Vivado, or access COM ports. It validates the HLS status,
HLS log, and downstream Vivado CSV emitted by the isolated operator-level
pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "hls_lut_builder" / "results" / "phase0_v4_sonar_stage3_k3_pilot"
DEFAULT_PROJECT_ROOT = PILOT_ROOT / "p"
DEFAULT_VIVADO_CSV = PILOT_ROOT / "vivado_downstream.csv"
DEFAULT_OUTPUT_JSON = PILOT_ROOT / "sonar_stage3_k3_timing_gate.json"
DEFAULT_OUTPUT_MD = PILOT_ROOT / "sonar_stage3_k3_timing_gate.md"
DEFAULT_EXPECTED_CASES = (
    "denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
    "edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
)
HLS_II_PATTERN = re.compile(r"\b(?:Final\s+)?II\s*=\s*(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase0 v4 sonar stage3 k3 pilot timing and HLS logs")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--vivado-csv", default=str(DEFAULT_VIVADO_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--expected-case", action="append", default=None)
    parser.add_argument("--min-wns-ns", type=float, default=0.0)
    parser.add_argument("--min-fmax-mhz", type=float, default=199.9)
    parser.add_argument(
        "--blocked-ii-threshold",
        type=int,
        default=64,
        help="Block when max observed HLS II is greater than or equal to this value.",
    )
    parser.add_argument(
        "--max-hls-dsp",
        type=int,
        default=128,
        help="Block when HLS estimated DSP is greater than this value.",
    )
    parser.add_argument(
        "--max-post-route-dsp",
        type=int,
        default=128,
        help="Block when post-route DSP is greater than this value.",
    )
    parser.add_argument(
        "--blocked-exit-zero",
        action="store_true",
        help="Write BLOCKED status but return zero. Useful for handoff reports.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"_json_error": True}
    return payload if isinstance(payload, dict) else {"_json_error": True}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find_hls_report(case_dir: Path, status_payload: dict[str, Any]) -> Path | None:
    raw_report = status_payload.get("report_path")
    if raw_report:
        report_path = Path(str(raw_report)).expanduser()
        if report_path.exists():
            return report_path.resolve()
    candidates = sorted((case_dir / "project").glob("**/syn/report/*_csynth.xml"))
    candidates.extend(sorted((case_dir / "project").glob("**/syn/report/*_csynth.rpt")))
    return candidates[0].resolve() if candidates else None


def _parse_hls_resources_xml(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    resources: dict[str, int] = {}
    for tag in ("BRAM_18K", "DSP", "FF", "LUT", "URAM"):
        element = root.find(f".//AreaEstimates/Resources/{tag}")
        if element is not None and element.text is not None:
            parsed = to_int(element.text)
            if parsed is not None:
                resources[tag] = parsed
    return resources


def _parse_hls_resources_rpt(path: Path) -> dict[str, int]:
    text = read_text(path)
    total_match = re.search(
        r"\|Total\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|",
        text,
    )
    if not total_match:
        return {}
    return {
        "BRAM_18K": int(total_match.group(1)),
        "DSP": int(total_match.group(2)),
        "FF": int(total_match.group(3)),
        "LUT": int(total_match.group(4)),
        "URAM": int(total_match.group(5)),
    }


def parse_hls_resources(report_path: Path | None) -> dict[str, int]:
    if report_path is None or not report_path.exists():
        return {}
    if report_path.suffix.lower() == ".xml":
        resources = _parse_hls_resources_xml(report_path)
        if resources:
            return resources
        companion_rpt = report_path.with_suffix(".rpt")
        return _parse_hls_resources_rpt(companion_rpt)
    return _parse_hls_resources_rpt(report_path)


def find_case_dir(project_root: Path, case_name: str) -> Path:
    candidates: list[Path] = []
    for metadata_path in project_root.glob("*/metadata.json"):
        payload = read_json(metadata_path)
        if payload.get("case_name") == case_name:
            candidates.append(metadata_path.parent)
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime).resolve()
    return (project_root / case_name).resolve()


def audit_hls_case(
    *,
    case_name: str,
    project_root: Path,
    blocked_ii_threshold: int,
    max_hls_dsp: int,
) -> dict[str, Any]:
    case_dir = find_case_dir(project_root, case_name)
    status_path = case_dir / "synthesis_status.json"
    log_path = case_dir / "logs" / "vitis_hls.log"
    status_payload = read_json(status_path)
    hls_status = str(status_payload.get("status", "missing")).strip() or "missing"
    hls_report = find_hls_report(case_dir, status_payload)
    hls_resources = parse_hls_resources(hls_report)
    hls_dsp = hls_resources.get("DSP")
    hls_log = read_text(log_path)
    hls_200_885_count = hls_log.count("HLS 200-885")
    observed_iis = [int(match.group(1)) for match in HLS_II_PATTERN.finditer(hls_log)]
    max_observed_ii = max(observed_iis) if observed_iis else None

    blockers: list[str] = []
    if not status_path.exists():
        blockers.append("missing synthesis_status.json")
    elif status_payload.get("_json_error"):
        blockers.append("invalid synthesis_status.json")
    if hls_status not in {"success", "skipped_existing"}:
        blockers.append(f"hls_status={hls_status}")
    if hls_report is None:
        blockers.append("missing HLS csynth report")
    if not log_path.exists():
        blockers.append("missing HLS log")
    if max_observed_ii is not None and max_observed_ii >= blocked_ii_threshold:
        blockers.append(f"max_observed_ii={max_observed_ii} >= {blocked_ii_threshold}")
    if hls_status == "timeout":
        blockers.append("HLS timeout")
    if hls_report is not None:
        if hls_dsp is None:
            blockers.append("HLS estimated DSP missing")
        elif hls_dsp > max_hls_dsp:
            blockers.append(f"HLS estimated DSP={hls_dsp} > {max_hls_dsp}")

    return {
        "case_name": case_name,
        "case_dir": str(case_dir.resolve()),
        "synthesis_status_path": str(status_path.resolve()),
        "hls_status": hls_status,
        "hls_report_path": str(hls_report) if hls_report is not None else None,
        "hls_resources": hls_resources,
        "hls_estimated_DSP": hls_dsp,
        "hls_estimated_LUT": hls_resources.get("LUT"),
        "hls_estimated_BRAM_18K": hls_resources.get("BRAM_18K"),
        "hls_log_path": str(log_path.resolve()),
        "hls_200_885_count": hls_200_885_count,
        "observed_ii_values": observed_iis,
        "max_observed_ii": max_observed_ii,
        "hls_passed": not blockers,
        "hls_blocking_reason": "; ".join(blockers),
    }


def audit_downstream_row(
    *,
    case_name: str,
    row: dict[str, str] | None,
    min_wns_ns: float,
    min_fmax_mhz: float,
    max_post_route_dsp: int,
) -> dict[str, Any]:
    if row is None:
        return {
            "case_name": case_name,
            "downstream_passed": False,
            "downstream_blocking_reason": "missing downstream Vivado CSV row",
        }

    downstream_status = str(row.get("downstream_status", "")).strip()
    wns = to_float(row.get("post_route_setup_WNS_ns"))
    fmax = to_float(row.get("post_route_Fmax_est"))
    post_route_dsp = to_int(row.get("post_route_DSP"))
    blockers: list[str] = []
    if downstream_status not in {"success", "skipped_existing"}:
        blockers.append(f"downstream_status={downstream_status or 'missing'}")
    if wns is None:
        blockers.append("post_route_setup_WNS_ns missing")
    elif wns < min_wns_ns:
        blockers.append(f"post_route_setup_WNS_ns={wns:.3f} < {min_wns_ns:.3f}")
    if fmax is None:
        blockers.append("post_route_Fmax_est missing")
    elif fmax < min_fmax_mhz:
        blockers.append(f"post_route_Fmax_est={fmax:.2f} < {min_fmax_mhz:.2f}")
    if post_route_dsp is None:
        blockers.append("post_route_DSP missing")
    elif post_route_dsp > max_post_route_dsp:
        blockers.append(f"post_route_DSP={post_route_dsp} > {max_post_route_dsp}")

    return {
        "case_name": case_name,
        "downstream_passed": not blockers,
        "downstream_status": downstream_status,
        "post_route_setup_WNS_ns": wns,
        "post_route_Fmax_est": fmax,
        "downstream_blocking_reason": "; ".join(blockers),
        "post_route_LUT": row.get("post_route_LUT"),
        "post_route_DSP": post_route_dsp,
        "post_route_BRAM_TILE": row.get("post_route_BRAM_TILE"),
        "power_w": row.get("power_w"),
        "report_dir": row.get("report_dir"),
    }


def audit_case(
    *,
    case_name: str,
    hls_result: dict[str, Any],
    downstream_result: dict[str, Any],
) -> dict[str, Any]:
    blockers = [
        reason
        for reason in (
            hls_result.get("hls_blocking_reason", ""),
            downstream_result.get("downstream_blocking_reason", ""),
        )
        if reason
    ]
    passed = bool(hls_result.get("hls_passed")) and bool(downstream_result.get("downstream_passed"))
    return {
        **hls_result,
        **downstream_result,
        "case_name": case_name,
        "status": "PASS" if passed else "BLOCKED",
        "passed": passed,
        "blocking_reason": "; ".join(blockers),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase0 v4 Sonar Stage3 k3 Timing/Log Gate",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Conclusion",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed cases: `{payload['summary']['passed_cases']}` / `{payload['summary']['expected_cases']}`",
        f"- HLS passed cases: `{payload['summary']['hls_passed_cases']}` / `{payload['summary']['expected_cases']}`",
        f"- Minimum WNS: `{payload['thresholds']['min_wns_ns']}` ns",
        f"- Minimum Fmax: `{payload['thresholds']['min_fmax_mhz']}` MHz",
        f"- Blocked II threshold: `>= {payload['thresholds']['blocked_ii_threshold']}`",
        f"- Max HLS DSP: `{payload['thresholds']['max_hls_dsp']}`",
        f"- Max post-route DSP: `{payload['thresholds']['max_post_route_dsp']}`",
        "",
        "## Cases",
        "",
        "| case | status | HLS | HLS DSP | max II | HLS 200-885 | downstream | WNS ns | Fmax MHz | post-route DSP | LUT | reason |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["cases"]:
        lines.append(
            "| "
            f"`{row['case_name']}` | {row['status']} | "
            f"{row.get('hls_status', '')} | "
            f"{'' if row.get('hls_estimated_DSP') is None else row['hls_estimated_DSP']} | "
            f"{'' if row.get('max_observed_ii') is None else row['max_observed_ii']} | "
            f"{row.get('hls_200_885_count', 0)} | "
            f"{row.get('downstream_status', '')} | "
            f"{'' if row.get('post_route_setup_WNS_ns') is None else row['post_route_setup_WNS_ns']} | "
            f"{'' if row.get('post_route_Fmax_est') is None else row['post_route_Fmax_est']} | "
            f"{row.get('post_route_DSP', '')} | {row.get('post_route_LUT', '')} | "
            f"{row.get('blocking_reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    vivado_csv = Path(args.vivado_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    expected_cases = tuple(args.expected_case or DEFAULT_EXPECTED_CASES)

    rows = read_csv(vivado_csv)
    rows_by_case = {row.get("case_name", ""): row for row in rows}
    case_results = []
    for case_name in expected_cases:
        hls_result = audit_hls_case(
            case_name=case_name,
            project_root=project_root,
            blocked_ii_threshold=args.blocked_ii_threshold,
            max_hls_dsp=args.max_hls_dsp,
        )
        downstream_result = audit_downstream_row(
            case_name=case_name,
            row=rows_by_case.get(case_name),
            min_wns_ns=args.min_wns_ns,
            min_fmax_mhz=args.min_fmax_mhz,
            max_post_route_dsp=args.max_post_route_dsp,
        )
        case_results.append(
            audit_case(
                case_name=case_name,
                hls_result=hls_result,
                downstream_result=downstream_result,
            )
        )

    passed_cases = sum(1 for row in case_results if row["passed"])
    hls_passed_cases = sum(1 for row in case_results if row["hls_passed"])
    status = "PASS" if passed_cases == len(expected_cases) else "BLOCKED"
    payload = {
        "generated_at": timestamp(),
        "runs_server": False,
        "runs_training": False,
        "runs_vivado": False,
        "runs_com_measurement": False,
        "project_root": str(project_root),
        "vivado_csv": str(vivado_csv),
        "status": status,
        "thresholds": {
            "min_wns_ns": args.min_wns_ns,
            "min_fmax_mhz": args.min_fmax_mhz,
            "blocked_ii_threshold": args.blocked_ii_threshold,
            "max_hls_dsp": args.max_hls_dsp,
            "max_post_route_dsp": args.max_post_route_dsp,
        },
        "summary": {
            "expected_cases": len(expected_cases),
            "parsed_downstream_rows": len(rows),
            "passed_cases": passed_cases,
            "blocked_cases": len(expected_cases) - passed_cases,
            "hls_passed_cases": hls_passed_cases,
            "hls_blocked_cases": len(expected_cases) - hls_passed_cases,
            "cases_with_hls_200_885": sum(1 for row in case_results if int(row.get("hls_200_885_count", 0)) > 0),
        },
        "cases": case_results,
        "next_action": (
            "Generate/use the pilot formal LUT only if this gate is PASS. "
            "Do not create a Phase0 v4 RL300 config from this operator-level timing gate alone."
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Timing/log gate status: {status}")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    return 0 if status == "PASS" or args.blocked_exit_zero else 1


if __name__ == "__main__":
    raise SystemExit(main())
