#!/usr/bin/env python3
"""Vivado implementation recovery sweep for the arch_84 e2e harness.

This script never runs COM/UART measurement.  It only attempts to recover the
complete-network implementation from existing checkpoints and records per-
attempt evidence under sonar_classifier_end_to_end_board.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BUILDER_ROOT = ROOT / "hls_lut_builder"
BOARD_HARNESS_DIR = BUILDER_ROOT / "board_harness"
E2E_DIR = BOARD_HARNESS_DIR / "results" / "sonar_classifier_end_to_end_board"
TARGET_NAME = "sonar_classifier_top8_03_arch_84_e2e"
MEASUREMENT_ROOT = E2E_DIR / "real_measurement" / TARGET_NAME
PROJECT_ROOT = MEASUREMENT_ROOT / "harness_project"
SWEEP_ROOT = MEASUREMENT_ROOT / "implementation_sweep"

sys.path.insert(0, str(SCRIPT_DIR))

from batch_measure_ops import DEFAULT_VIVADO  # noqa: E402


SUMMARY_FIELDS = [
    "attempt_name",
    "status",
    "returncode",
    "source_checkpoint",
    "place_directive",
    "phys_opt_directive",
    "route_directive",
    "bitstream",
    "wns_ns",
    "lut",
    "ff",
    "bram_tile",
    "dsp",
    "global_congestion_level",
    "failure_summary",
    "attempt_dir",
    "stdout_log",
    "timing_report",
    "utilization_report",
    "congestion_report",
    "route_status_report",
]


@dataclass(frozen=True)
class AttemptConfig:
    name: str
    source: str
    place_directive: str = ""
    phys_opt_directive: str = ""
    route_directive: str = "MoreGlobalIterations"


ATTEMPTS: dict[str, AttemptConfig] = {
    "placed_route_more_global": AttemptConfig(
        name="placed_route_more_global",
        source="placed",
        route_directive="MoreGlobalIterations",
    ),
    "placed_route_no_timing": AttemptConfig(
        name="placed_route_no_timing",
        source="placed",
        route_directive="NoTimingRelaxation",
    ),
    "physopt_route_more_global": AttemptConfig(
        name="physopt_route_more_global",
        source="physopt",
        route_directive="MoreGlobalIterations",
    ),
    "physopt_route_higher_delay": AttemptConfig(
        name="physopt_route_higher_delay",
        source="physopt",
        route_directive="HigherDelayCost",
    ),
    "opt_place_altspread_physopt_route_more_global": AttemptConfig(
        name="opt_place_altspread_physopt_route_more_global",
        source="opt",
        place_directive="AltSpreadLogic_high",
        phys_opt_directive="Explore",
        route_directive="MoreGlobalIterations",
    ),
    "opt_place_wlblock_physopt_aggressive_route_no_timing": AttemptConfig(
        name="opt_place_wlblock_physopt_aggressive_route_no_timing",
        source="opt",
        place_directive="WLDrivenBlockPlacement",
        phys_opt_directive="AggressiveExplore",
        route_directive="NoTimingRelaxation",
    ),
    "opt_place_extranet_high_route_more_global": AttemptConfig(
        name="opt_place_extranet_high_route_more_global",
        source="opt",
        place_directive="ExtraNetDelay_high",
        route_directive="MoreGlobalIterations",
    ),
}

DEFAULT_SWEEP = [
    "placed_route_more_global",
    "physopt_route_more_global",
    "opt_place_altspread_physopt_route_more_global",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run e2e Vivado implementation recovery sweep from existing DCPs.")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--attempts", default=",".join(DEFAULT_SWEEP), help="Comma-separated attempt names, or 'all'.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-minutes", type=int, default=90)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--promote-latest", action="store_true", help="Write the latest attempt status to main build_result.json.")
    parser.add_argument("--promote-success", action="store_true", default=True, help="Copy a successful bitstream/reports into the main harness outputs.")
    parser.add_argument("--no-promote-success", action="store_false", dest="promote_success")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def tcl_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def selected_attempts(raw: str) -> list[AttemptConfig]:
    names = list(ATTEMPTS) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    missing = [name for name in names if name not in ATTEMPTS]
    if missing:
        raise ValueError(f"Unsupported attempt(s): {missing}. Available: {sorted(ATTEMPTS)}")
    return [ATTEMPTS[name] for name in names]


def current_impl_dir() -> Path:
    build_tcl = PROJECT_ROOT / "scripts" / "build_bitstream.tcl"
    if not build_tcl.exists():
        raise FileNotFoundError(f"Missing e2e build Tcl: {build_tcl}")
    text = build_tcl.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'^\s*set\s+vivado_project_dir\s+"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse vivado_project_dir from {build_tcl}")
    project_dir = Path(match.group(1))
    project_name = project_dir.name
    return project_dir / f"{project_name}.runs" / "impl_1"


def checkpoints() -> dict[str, Path]:
    impl_dir = current_impl_dir()
    return {
        "opt": impl_dir / "harness_top_opt.dcp",
        "placed": impl_dir / "harness_top_placed.dcp",
        "physopt": impl_dir / "harness_top_physopt.dcp",
    }


def parse_wns(timing_report: Path) -> str:
    if not timing_report.exists():
        return ""
    text = timing_report.read_text(encoding="utf-8", errors="replace")
    number_pattern = re.compile(r"^-?\d+(?:\.\d+)?$")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 8]:
            parts = candidate.split()
            if parts and number_pattern.match(parts[0]):
                return parts[0]
    return ""


def parse_utilization(util_report: Path) -> dict[str, str]:
    values = {"lut": "", "ff": "", "bram_tile": "", "dsp": ""}
    if not util_report.exists():
        return values
    text = util_report.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "lut": r"\|\s*Slice LUTs\*?\s*\|\s*([0-9,]+)",
        "ff": r"\|\s*Slice Registers\s*\|\s*([0-9,]+)",
        "bram_tile": r"\|\s*Block RAM Tile\s*\|\s*([0-9.]+)",
        "dsp": r"\|\s*DSPs\s*\|\s*([0-9,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = match.group(1).replace(",", "")
    return values


def summarize_failure(stdout: str, limit: int = 8) -> str:
    lines = [
        line.strip()
        for line in stdout.splitlines()
        if "ERROR:" in line
        or "CRITICAL WARNING:" in line
        or "WARNING: [Route" in line
        or "Congestion" in line
        or "global congestion level" in line
    ]
    return " | ".join(lines[-limit:])[:1600]


def parse_global_congestion(stdout: str) -> str:
    matches = re.findall(r"global congestion level is\s+([0-9]+)", stdout, flags=re.IGNORECASE)
    return matches[-1] if matches else ""


def attempt_tcl(config: AttemptConfig, source_dcp: Path, attempt_dir: Path, jobs: int) -> str:
    reports_dir = attempt_dir / "reports"
    bitstream_dir = attempt_dir / "bitstream"
    routed_dcp = attempt_dir / "checkpoint_after_route.dcp"
    bitstream = bitstream_dir / f"{TARGET_NAME}.bit"
    place_cmd = ""
    if config.place_directive:
        place_cmd = f"""
puts "INFO: place_design -directive {config.place_directive}"
if {{[catch {{place_design -directive {config.place_directive}}} msg]}} {{
    puts "ERROR: place_design failed: $msg"
    set attempt_ok 0
}} else {{
    safe_report "post_place_congestion" "report_design_analysis -congestion -file {{{tcl_path(reports_dir / 'post_place_congestion.rpt')}}}"
    safe_report "post_place_utilization" "report_utilization -file {{{tcl_path(reports_dir / 'post_place_utilization.rpt')}}}"
}}
""".strip()
    phys_cmd = ""
    if config.phys_opt_directive:
        phys_cmd = f"""
if {{$attempt_ok}} {{
    puts "INFO: phys_opt_design -directive {config.phys_opt_directive}"
    if {{[catch {{phys_opt_design -directive {config.phys_opt_directive}}} msg]}} {{
        puts "ERROR: phys_opt_design failed: $msg"
        set attempt_ok 0
    }} else {{
        safe_report "post_physopt_timing" "report_timing_summary -file {{{tcl_path(reports_dir / 'post_physopt_timing_summary.rpt')}}}"
        safe_report "post_physopt_utilization" "report_utilization -file {{{tcl_path(reports_dir / 'post_physopt_utilization.rpt')}}}"
    }}
}}
""".strip()
    return f"""
set_param general.maxThreads {jobs}
set attempt_ok 1
set route_ok 0
file mkdir {{{tcl_path(reports_dir)}}}
file mkdir {{{tcl_path(bitstream_dir)}}}

proc safe_report {{label command}} {{
    puts "INFO: report $label"
    if {{[catch {{eval $command}} msg]}} {{
        puts "WARN: report $label failed: $msg"
    }}
}}

puts "INFO: opening checkpoint {tcl_path(source_dcp)}"
open_checkpoint {{{tcl_path(source_dcp)}}}

safe_report "initial_timing" "report_timing_summary -file {{{tcl_path(reports_dir / 'initial_timing_summary.rpt')}}}"
safe_report "initial_utilization" "report_utilization -file {{{tcl_path(reports_dir / 'initial_utilization.rpt')}}}"
safe_report "initial_congestion" "report_design_analysis -congestion -file {{{tcl_path(reports_dir / 'initial_congestion.rpt')}}}"

{place_cmd}

{phys_cmd}

if {{$attempt_ok}} {{
    puts "INFO: route_design -directive {config.route_directive}"
    if {{[catch {{route_design -directive {config.route_directive}}} msg]}} {{
        puts "ERROR: route_design failed: $msg"
        set attempt_ok 0
    }} else {{
        set route_ok 1
    }}
}}

safe_report "route_status" "report_route_status -file {{{tcl_path(reports_dir / 'route_status.rpt')}}}"
safe_report "timing_summary" "report_timing_summary -file {{{tcl_path(reports_dir / 'timing_summary.rpt')}}}"
safe_report "utilization" "report_utilization -file {{{tcl_path(reports_dir / 'utilization.rpt')}}}"
safe_report "clock_utilization" "report_clock_utilization -file {{{tcl_path(reports_dir / 'clock_utilization.rpt')}}}"
safe_report "congestion" "report_design_analysis -congestion -file {{{tcl_path(reports_dir / 'congestion.rpt')}}}"
safe_report "drc" "report_drc -file {{{tcl_path(reports_dir / 'drc.rpt')}}}"
if {{[catch {{write_checkpoint -force {{{tcl_path(routed_dcp)}}}}} msg]}} {{
    puts "WARN: write_checkpoint failed: $msg"
}}

if {{$route_ok}} {{
    if {{[catch {{write_bitstream -force {{{tcl_path(bitstream)}}}}} msg]}} {{
        puts "ERROR: write_bitstream failed: $msg"
        set attempt_ok 0
    }}
}}

if {{$attempt_ok && [file exists {{{tcl_path(bitstream)}}}]}} {{
    puts "BITSTREAM_OK: {tcl_path(bitstream)}"
    exit 0
}}
puts "BITSTREAM_FAIL: no usable bitstream generated"
exit 1
""".lstrip()


def run_attempt(
    config: AttemptConfig,
    run_dir: Path,
    vivado: Path,
    timeout_seconds: int | None,
    jobs: int,
    checkpoint_map: dict[str, Path],
) -> dict[str, Any]:
    source_dcp = checkpoint_map[config.source]
    attempt_dir = run_dir / config.name
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = attempt_dir / "vivado_stdout.log"
    tcl = attempt_dir / "run_attempt.tcl"
    build_json = attempt_dir / "build_result.json"
    if not source_dcp.exists():
        payload = {
            "attempt_name": config.name,
            "status": "failed_missing_checkpoint",
            "returncode": "",
            "source_checkpoint": str(source_dcp),
            "place_directive": config.place_directive,
            "phys_opt_directive": config.phys_opt_directive,
            "route_directive": config.route_directive,
            "bitstream": "",
            "failure_summary": f"Missing checkpoint: {source_dcp}",
            "attempt_dir": str(attempt_dir),
            "stdout_log": str(stdout_log),
            "updated_at": timestamp(),
        }
        write_json(build_json, payload)
        return payload

    tcl.write_text(attempt_tcl(config, source_dcp, attempt_dir, jobs), encoding="utf-8")
    command = ["cmd", "/c", str(vivado), "-mode", "batch", "-source", str(tcl)]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
        stdout = result.stdout
        returncode: int | str = result.returncode
        status = "success" if result.returncode == 0 and (attempt_dir / "bitstream" / f"{TARGET_NAME}.bit").exists() else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        returncode = "timeout"
        status = "timeout"
    stdout_log.write_text(stdout, encoding="utf-8", errors="replace")

    reports_dir = attempt_dir / "reports"
    bitstream = attempt_dir / "bitstream" / f"{TARGET_NAME}.bit"
    timing_report = reports_dir / "timing_summary.rpt"
    utilization_report = reports_dir / "utilization.rpt"
    congestion_report = reports_dir / "congestion.rpt"
    route_status_report = reports_dir / "route_status.rpt"
    util = parse_utilization(utilization_report)
    payload = {
        "attempt_name": config.name,
        "status": status,
        "returncode": returncode,
        "source_checkpoint": str(source_dcp),
        "place_directive": config.place_directive,
        "phys_opt_directive": config.phys_opt_directive,
        "route_directive": config.route_directive,
        "bitstream": str(bitstream) if bitstream.exists() else "",
        "wns_ns": parse_wns(timing_report),
        "lut": util["lut"],
        "ff": util["ff"],
        "bram_tile": util["bram_tile"],
        "dsp": util["dsp"],
        "global_congestion_level": parse_global_congestion(stdout),
        "failure_summary": "" if status == "success" else summarize_failure(stdout),
        "attempt_dir": str(attempt_dir),
        "stdout_log": str(stdout_log),
        "timing_report": str(timing_report) if timing_report.exists() else "",
        "utilization_report": str(utilization_report) if utilization_report.exists() else "",
        "congestion_report": str(congestion_report) if congestion_report.exists() else "",
        "route_status_report": str(route_status_report) if route_status_report.exists() else "",
        "build_result_json": str(build_json),
        "updated_at": timestamp(),
    }
    write_json(build_json, payload)
    return payload


def promote_success(attempt: dict[str, Any]) -> None:
    bitstream = Path(str(attempt.get("bitstream") or ""))
    if not bitstream.exists():
        return
    main_bitstream_dir = PROJECT_ROOT / "bitstream"
    main_reports_dir = PROJECT_ROOT / "reports"
    main_bitstream_dir.mkdir(parents=True, exist_ok=True)
    main_reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bitstream, main_bitstream_dir / f"{TARGET_NAME}.bit")
    for report_name in ["timing_summary.rpt", "utilization.rpt", "clock_utilization.rpt"]:
        report = Path(attempt["attempt_dir"]) / "reports" / report_name
        if report.exists():
            shutil.copy2(report, main_reports_dir / report.name)


def write_main_build_result(latest: dict[str, Any], summary_path: Path) -> None:
    bitstream = latest.get("bitstream") or ""
    status = "success" if latest.get("status") == "success" and bitstream else latest.get("status", "failed")
    payload = {
        "target_name": TARGET_NAME,
        "status": status,
        "returncode": latest.get("returncode", ""),
        "bitstream": bitstream or None,
        "stdout_log": latest.get("stdout_log", ""),
        "failure_summary": latest.get("failure_summary", ""),
        "implementation_attempt": latest.get("attempt_name", ""),
        "implementation_sweep_summary": str(summary_path),
        "updated_at": timestamp(),
    }
    write_json(MEASUREMENT_ROOT / "build_result.json", payload)


def main() -> int:
    args = parse_args()
    vivado = Path(args.vivado).expanduser().resolve()
    attempts = selected_attempts(args.attempts)
    checkpoint_map = checkpoints()
    run_dir = SWEEP_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = args.timeout_minutes * 60 if args.timeout_minutes > 0 else None
    results: list[dict[str, Any]] = []
    for config in attempts:
        print(f"[{timestamp()}] running {config.name}", flush=True)
        result = run_attempt(config, run_dir, vivado, timeout_seconds, args.jobs, checkpoint_map)
        results.append(result)
        print(
            f"[{timestamp()}] {config.name}: status={result.get('status')} "
            f"wns={result.get('wns_ns')} congestion={result.get('global_congestion_level')}",
            flush=True,
        )
        if result.get("status") == "success":
            if args.promote_success:
                promote_success(result)
            break

    summary = {
        "target_name": TARGET_NAME,
        "generated_at": timestamp(),
        "run_id": args.run_id,
        "attempt_count": len(results),
        "success_count": sum(1 for row in results if row.get("status") == "success"),
        "attempts": results,
    }
    summary_json = run_dir / "implementation_sweep_summary.json"
    summary_csv = run_dir / "implementation_sweep_summary.csv"
    write_json(summary_json, summary)
    write_csv(summary_csv, results)
    write_json(MEASUREMENT_ROOT / "implementation_sweep_latest.json", summary)
    write_csv(MEASUREMENT_ROOT / "implementation_sweep_latest.csv", results)
    if args.promote_latest and results:
        write_main_build_result(results[-1], summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["success_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
