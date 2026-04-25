#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_cases, find_first_existing_report, load_config, resolve_workspace_path, select_pilot_cases
from run_vivado_downstream import run_downstream_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vitis HLS synthesis for generated operator cases")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Run only specific operators")
    parser.add_argument("--case", action="append", default=None, help="Run only specific cases")
    parser.add_argument("--pilot", action="store_true", help="Run one representative case per enabled operator")
    parser.add_argument("--vitis-hls", default=None, help="Override Vitis HLS executable path")
    parser.add_argument("--downstream-check", action="store_true", help="After HLS, run the Vivado downstream OOC flow for the same case")
    parser.add_argument("--vivado", default=None, help="Override Vivado executable path used by --downstream-check")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running synthesis")
    parser.add_argument("--force", action="store_true", help="Re-run synthesis even if a report already exists")
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Per-case timeout")
    parser.add_argument("--downstream-timeout-minutes", type=int, default=60, help="Per-case timeout for --downstream-check")
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
        raise SystemExit("No HLS cases matched the provided filters.")

    toolchain = config.get("toolchain", {})
    vitis_hls = str(args.vitis_hls or toolchain.get("vitis_hls_bin", "vitis_hls"))
    vivado_bin = str(args.vivado or toolchain.get("vivado_bin", "vivado"))
    summary: list[dict[str, Any]] = []

    for case in cases:
        report_path = find_first_existing_report(case.report_candidates)
        log_dir = case.project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "vitis_hls.log"
        status_path = case.project_dir / "synthesis_status.json"

        if report_path is not None and not args.force:
            status = {
                "case_name": case.case_name,
                "op_id": case.op_id,
                "status": "skipped_existing",
                "report_path": str(report_path),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            if args.downstream_check:
                status["downstream_check"] = run_downstream_case(
                    case,
                    vivado_bin=vivado_bin,
                    dry_run=False,
                    force=args.force,
                    timeout_minutes=args.downstream_timeout_minutes,
                )
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            summary.append(status)
            if args.downstream_check and isinstance(status.get("downstream_check"), dict):
                downstream_status = status["downstream_check"].get("status", "unknown")
                print(f"[skip] {case.case_name} -> {report_path} | downstream={downstream_status}")
            else:
                print(f"[skip] {case.case_name} -> {report_path}")
            continue

        command = [vitis_hls, "-f", "synth.tcl"]
        if args.dry_run:
            print(f"[dry-run] {case.case_name}: {' '.join(command)}")
            status = {"case_name": case.case_name, "op_id": case.op_id, "status": "dry_run"}
            if args.downstream_check:
                status["downstream_check"] = run_downstream_case(
                    case,
                    vivado_bin=vivado_bin,
                    dry_run=True,
                    force=args.force,
                    timeout_minutes=args.downstream_timeout_minutes,
                )
                downstream_status = status["downstream_check"].get("status", "unknown")
                if "command" in status["downstream_check"]:
                    print(f"[dry-run] {case.case_name} downstream: {' '.join(status['downstream_check']['command'])}")
                else:
                    print(f"[dry-run] {case.case_name} downstream: {downstream_status}")
            summary.append(status)
            continue

        print(f"[run] {case.case_name}")
        start_time = datetime.now().isoformat(timespec="seconds")
        returncode, timed_out = _run_command_with_timeout(
            command,
            cwd=case.project_dir,
            log_path=log_path,
            timeout_minutes=args.timeout_minutes,
        )

        report_path = find_first_existing_report(case.report_candidates)
        status = {
            "case_name": case.case_name,
            "op_id": case.op_id,
            "status": "success" if returncode == 0 and report_path else ("timeout" if timed_out else "failed"),
            "returncode": returncode,
            "started_at": start_time,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "report_path": str(report_path) if report_path else None,
            "log_path": str(log_path),
            "target_clock_mhz": case.target_clock_mhz,
            "clock_profile_name": case.clock_profile_name,
        }
        if args.downstream_check and status["status"] in {"success", "skipped_existing"}:
            status["downstream_check"] = run_downstream_case(
                case,
                vivado_bin=vivado_bin,
                dry_run=False,
                force=args.force,
                timeout_minutes=args.downstream_timeout_minutes,
            )
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(status)
        if "downstream_check" in status:
            print(f"[downstream:{status['downstream_check'].get('status', 'unknown')}] {case.case_name}")

    workspace = config.get("workspace", {})
    summary_path = resolve_workspace_path(
        config,
        args.summary_json or workspace.get("synthesis_summary_json", "../results/operator_screening_synthesis_summary.json"),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote synthesis summary: {summary_path}")


if __name__ == "__main__":
    main()
