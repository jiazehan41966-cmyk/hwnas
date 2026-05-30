#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_ROOT = SCRIPT_DIR.parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import (
    build_cases,
    load_config,
    resolve_workspace_path,
    select_pilot_cases,
)
from hwnas_fpga.hardware import (
    parse_vivado_power_text,
    parse_vivado_timing_summary_text,
    parse_vivado_utilization_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream Vivado synthesis/implementation on HLS RTL")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Run only specific operators")
    parser.add_argument("--case", action="append", default=None, help="Run only specific cases")
    parser.add_argument("--pilot", action="store_true", help="Select one representative case per enabled operator")
    parser.add_argument("--vivado", default=None, help="Override Vivado executable path")
    parser.add_argument("--flow-tcl", default=None, help="Optional downstream Vivado Tcl override")
    parser.add_argument(
        "--vivado-work-root",
        default=None,
        help="Optional short clean Vivado work root. RTL is copied here before running Vivado.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Vivado")
    parser.add_argument("--force", action="store_true", help="Re-run even if downstream reports already exist")
    parser.add_argument("--timeout-minutes", type=int, default=60, help="Per-case timeout")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON override")
    return parser.parse_args()


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()


def _run_command_with_timeout(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_minutes: int,
) -> tuple[int, bool]:
    timeout_seconds = max(1, timeout_minutes) * 60
    timed_out = False

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            log_file.write(
                f"\n[timeout] Exceeded {timeout_seconds} seconds. "
                f"Terminating process tree for PID {process.pid}.\n"
            )
            log_file.flush()
            _terminate_process_tree(process)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = -1

    return returncode, timed_out


def parse_downstream_metrics(case: Any) -> dict[str, Any] | None:
    post_synth = _parse_stage_reports(case.vivado_downstream_reports_dir, "post_synth", target_clock_ns=case.clock_period_ns)
    post_route = _parse_stage_reports(case.vivado_downstream_reports_dir, "post_route", target_clock_ns=case.clock_period_ns)
    if post_synth is None or post_route is None:
        return None
    power = _parse_post_route_power(case.vivado_downstream_reports_dir)
    post_route_metrics = {
        **post_route["timing"],
        **post_route["utilization"],
    }
    if power is not None:
        post_route_metrics.update(power["metrics"])
    return {
        "post_synth": {
            **post_synth["timing"],
            **post_synth["utilization"],
        },
        "post_route": post_route_metrics,
    }


def run_downstream_case(
    case: Any,
    *,
    vivado_bin: str,
    flow_tcl_path: str | Path | None = None,
    vivado_work_root: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    timeout_minutes: int = 60,
) -> dict[str, Any]:
    flow_tcl = Path(flow_tcl_path).expanduser().resolve() if flow_tcl_path else (SCRIPT_DIR / "run_vivado_downstream.tcl").resolve()
    rtl_dir = case.project_dir / "project" / case.solution_name / "syn" / "verilog"
    post_route_timing = case.vivado_downstream_reports_dir / "post_route_timing_summary.rpt"
    post_route_util = case.vivado_downstream_reports_dir / "post_route_utilization.rpt"
    status_path = case.vivado_downstream_status_path
    log_path = case.vivado_downstream_dir / "vivado_downstream.log"

    if post_route_timing.exists() and post_route_util.exists() and not force:
        status = {
            "case_name": case.case_name,
            "op_id": case.op_id,
            "status": "skipped_existing",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "report_dir": str(case.vivado_downstream_reports_dir),
            "log_path": str(log_path) if log_path.exists() else None,
            "metrics": parse_downstream_metrics(case),
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    if not rtl_dir.exists() or not any(rtl_dir.glob("*.v")):
        status = {
            "case_name": case.case_name,
            "op_id": case.op_id,
            "status": "missing_hls_rtl",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "rtl_dir": str(rtl_dir),
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    if case.vivado_downstream_dir.exists() and force:
        shutil.rmtree(case.vivado_downstream_dir)

    # Remove stale Vivado runtime cache to avoid reusing broken realtime scripts.
    xil_runtime_dir = case.project_dir / ".Xil"
    if xil_runtime_dir.exists():
        shutil.rmtree(xil_runtime_dir, ignore_errors=True)

    runtime_work_dir: Path | None = None
    runtime_out_dir = case.vivado_downstream_dir
    runtime_rtl_dir = rtl_dir
    runtime_cwd = case.project_dir
    if vivado_work_root:
        runtime_work_dir = _prepare_short_work_dir(
            case=case,
            rtl_dir=rtl_dir,
            vivado_work_root=Path(vivado_work_root),
            force=force,
        )
        runtime_rtl_dir = runtime_work_dir / "rtl"
        runtime_out_dir = runtime_work_dir / "out"
        runtime_cwd = runtime_work_dir

    command = [
        vivado_bin,
        "-mode",
        "batch",
        "-source",
        str(flow_tcl),
        "-tclargs",
        str(runtime_rtl_dir),
        case.top_function,
        case.part,
        f"{case.clock_period_ns:.3f}",
        str(runtime_out_dir),
    ]
    if dry_run:
        return {
            "case_name": case.case_name,
            "op_id": case.op_id,
            "status": "dry_run",
            "command": command,
            "runtime_work_dir": str(runtime_work_dir) if runtime_work_dir else None,
        }

    case.vivado_downstream_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now().isoformat(timespec="seconds")
    returncode, timed_out = _run_command_with_timeout(
        command,
        cwd=runtime_cwd,
        log_path=log_path,
        timeout_minutes=timeout_minutes,
    )
    if runtime_work_dir is not None:
        _sync_runtime_output(runtime_out_dir, case.vivado_downstream_dir)

    post_route_timing = case.vivado_downstream_reports_dir / "post_route_timing_summary.rpt"
    post_route_util = case.vivado_downstream_reports_dir / "post_route_utilization.rpt"
    success = returncode == 0 and post_route_timing.exists() and post_route_util.exists()
    status = {
        "case_name": case.case_name,
        "op_id": case.op_id,
        "status": "success" if success else ("timeout" if timed_out else "failed"),
        "returncode": returncode,
        "started_at": start_time,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rtl_dir": str(rtl_dir),
        "report_dir": str(case.vivado_downstream_reports_dir) if case.vivado_downstream_reports_dir.exists() else None,
        "post_route_timing": str(post_route_timing) if post_route_timing.exists() else None,
        "post_route_utilization": str(post_route_util) if post_route_util.exists() else None,
        "log_path": str(log_path),
        "flow_tcl": str(flow_tcl),
        "runtime_work_dir": str(runtime_work_dir) if runtime_work_dir else None,
        "target_clock_mhz": case.target_clock_mhz,
        "clock_profile_name": case.clock_profile_name,
        "metrics": parse_downstream_metrics(case) if success else None,
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def _prepare_short_work_dir(
    *,
    case: Any,
    rtl_dir: Path,
    vivado_work_root: Path,
    force: bool,
) -> Path:
    work_root = vivado_work_root.expanduser().resolve()
    digest = hashlib.sha1(case.case_name.encode("utf-8")).hexdigest()[:10]
    work_dir = work_root / f"{case.top_function}_{digest}"
    if work_dir.exists():
        _safe_remove_within_root(work_dir, work_root)
    work_rtl_dir = work_dir / "rtl"
    work_rtl_dir.mkdir(parents=True, exist_ok=True)
    for rtl_file in sorted(rtl_dir.glob("*.v")):
        shutil.copy2(rtl_file, work_rtl_dir / rtl_file.name)
    return work_dir


def _sync_runtime_output(runtime_out_dir: Path, final_out_dir: Path) -> None:
    if not runtime_out_dir.exists():
        return
    for child in runtime_out_dir.iterdir():
        dest = final_out_dir / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)


def _safe_remove_within_root(path: Path, root: Path) -> None:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to remove path outside Vivado work root: {resolved_path}")
    shutil.rmtree(resolved_path, ignore_errors=True)


def _parse_stage_reports(report_dir: Path, stage_name: str, *, target_clock_ns: float) -> dict[str, Any] | None:
    timing_report = report_dir / f"{stage_name}_timing_summary.rpt"
    utilization_report = report_dir / f"{stage_name}_utilization.rpt"
    if not timing_report.exists() or not utilization_report.exists():
        return None

    timing = parse_vivado_timing_summary_text(
        timing_report.read_text(encoding="utf-8", errors="ignore"),
        target_clock_ns=target_clock_ns,
    )
    utilization = parse_vivado_utilization_text(
        utilization_report.read_text(encoding="utf-8", errors="ignore"),
    )
    return {
        "timing": timing,
        "utilization": utilization,
        "timing_report": timing_report,
        "utilization_report": utilization_report,
    }


def _parse_post_route_power(report_dir: Path) -> dict[str, Any] | None:
    power_report = report_dir / "post_route_power.rpt"
    if not power_report.exists():
        return None
    power_metrics = parse_vivado_power_text(power_report.read_text(encoding="utf-8", errors="ignore"))
    if power_metrics.get("power_w") is None:
        return None
    return {
        "metrics": power_metrics,
        "power_report": power_report,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cases = build_cases(
        config,
        operator_filter=args.operator,
        case_filter=args.case,
        project_root_override=args.project_root,
        sort_cases=not args.pilot,
    )
    if args.pilot:
        cases = select_pilot_cases(cases)
    if not cases:
        raise SystemExit("No downstream Vivado cases matched the provided filters.")

    toolchain = config.get("toolchain", {})
    workspace = config.get("workspace", {})
    vivado_bin = str(args.vivado or toolchain.get("vivado_bin", "vivado"))
    summary: list[dict[str, Any]] = []

    for case in cases:
        status = run_downstream_case(
            case,
            vivado_bin=vivado_bin,
            flow_tcl_path=args.flow_tcl,
            vivado_work_root=args.vivado_work_root,
            dry_run=args.dry_run,
            force=args.force,
            timeout_minutes=args.timeout_minutes,
        )
        summary.append(status)
        if args.dry_run:
            print(f"[dry-run] {case.case_name}: {' '.join(status.get('command', []))}")
        elif status["status"] == "skipped_existing":
            print(f"[skip] {case.case_name} -> {case.vivado_downstream_reports_dir}")
        elif status["status"] == "missing_hls_rtl":
            print(f"[missing-hls] {case.case_name} -> {status['rtl_dir']}")
        else:
            print(f"[{status['status']}] {case.case_name}")

    summary_path = resolve_workspace_path(
        config,
        args.summary_json or workspace.get("downstream_summary_json", "../results/vivado_downstream_summary.json"),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote downstream Vivado summary: {summary_path}")


if __name__ == "__main__":
    main()
