#!/usr/bin/env python3
"""Run the AV7K325 5 ns route gate after HWC RTL co-simulation passes.

This gate consumes ``fourstage_hwc_parallel_rtl_cosim_summary.json`` and refuses
to run unless every selected candidate has real RTL co-sim PASS evidence.  It
then runs a Vivado out-of-context implementation for the generated HLS RTL and
checks the registered route gates:

* post-route WNS >= 0 ns;
* post-route DSP <= 700.

The generated HLS RTL is not a board harness by itself, so this script does not
claim a deployable bitstream.  Bitstream generation remains a later board-level
harness gate after this route gate passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.hardware import (  # noqa: E402
    parse_vivado_power_text,
    parse_vivado_timing_summary_text,
    parse_vivado_utilization_text,
)
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import evidence, read_json, write_json  # noqa: E402
from run_fourstage_hls_synthesis_gate import TARGET_CLOCK_NS, TARGET_PART  # noqa: E402
from run_fourstage_hwc_parallel_hls_gate import IMPLEMENTATION_STYLE  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
DEFAULT_RTL_COSIM_SUMMARY = ARTIFACT_ROOT / "fourstage_hwc_parallel_rtl_cosim_summary.json"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "fourstage_hwc_parallel_route_summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sonar_fourstage_operator_v2" / "hwcpar32_route"
DEFAULT_VIVADO = Path(r"F:\vivado\Vivado\2023.2\bin\vivado.bat")
ROUTE_DSP_LIMIT = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-cosim-summary", default=str(DEFAULT_RTL_COSIM_SUMMARY))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Candidate role to include; repeatable. Defaults to all candidates in the RTL summary.",
    )
    parser.add_argument("--run-vivado", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    return parser.parse_args()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def tcl_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def route_tcl(*, rtl_dir: Path, out_dir: Path) -> str:
    return dedent(
        f"""
        set rtl_dir "{tcl_path(rtl_dir)}"
        set out_dir "{tcl_path(out_dir)}"
        set report_dir [file join $out_dir reports]
        set checkpoint_dir [file join $out_dir checkpoints]
        file mkdir $out_dir
        file mkdir $report_dir
        file mkdir $checkpoint_dir

        set_param general.maxThreads 1
        catch {{set_param synth.maxThreads 1}}

        puts "=== Four-stage HWC route gate ==="
        puts "RTL dir: $rtl_dir"
        puts "Part: {TARGET_PART}"
        puts "Clock: {TARGET_CLOCK_NS:.3f} ns"
        puts "Output dir: $out_dir"

        set rtl_files [lsort [glob -nocomplain -directory $rtl_dir *.v]]
        if {{[llength $rtl_files] == 0}} {{
            puts "ERROR: No Verilog files found under $rtl_dir"
            exit 2
        }}
        foreach rtl_file $rtl_files {{
            read_verilog $rtl_file
        }}

        if {{[catch {{synth_design -top fourstage_top_hls -part {TARGET_PART} -mode out_of_context}} err]}} {{
            puts "ERROR: synth_design failed: $err"
            exit 3
        }}
        create_clock -period {TARGET_CLOCK_NS:.3f} -name ap_clk [get_ports ap_clk]
        if {{[llength [get_ports -quiet ap_rst_n]] > 0}} {{
            set_false_path -from [get_ports ap_rst_n]
        }}

        write_checkpoint -force [file join $checkpoint_dir post_synth.dcp]
        report_utilization -hierarchical -hierarchical_depth 4 -file [file join $report_dir post_synth_utilization_hier.rpt]
        report_utilization -file [file join $report_dir post_synth_utilization.rpt]
        report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $report_dir post_synth_timing_summary.rpt]

        if {{[catch {{opt_design}} err]}} {{
            puts "ERROR: opt_design failed: $err"
            exit 4
        }}
        if {{[catch {{place_design}} err]}} {{
            puts "ERROR: place_design failed: $err"
            exit 5
        }}
        if {{[catch {{phys_opt_design}} err]}} {{
            puts "WARNING: phys_opt_design failed: $err"
        }}
        if {{[catch {{route_design}} err]}} {{
            puts "ERROR: route_design failed: $err"
            exit 6
        }}

        write_checkpoint -force [file join $checkpoint_dir post_route.dcp]
        report_utilization -hierarchical -hierarchical_depth 4 -file [file join $report_dir post_route_utilization_hier.rpt]
        report_utilization -file [file join $report_dir post_route_utilization.rpt]
        report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $report_dir post_route_timing_summary.rpt]
        report_clock_utilization -file [file join $report_dir post_route_clock_utilization.rpt]
        if {{[catch {{report_power -file [file join $report_dir post_route_power.rpt]}} err]}} {{
            puts "WARNING: report_power failed: $err"
        }}
        puts "=== Four-stage HWC route gate completed successfully ==="
        exit 0
        """
    ).lstrip()


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:  # pragma: no cover - current execution is Windows.
        process.kill()


def run_vivado(
    *,
    vivado: Path,
    work_dir: Path,
    tcl_path_: Path,
    log_path: Path,
    timeout_minutes: int,
) -> tuple[int, Path, bool]:
    command = [
        "cmd",
        "/c",
        str(vivado),
        "-mode",
        "batch",
        "-source",
        str(tcl_path_),
    ]
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    timed_out = False
    try:
        stdout, _stderr = process.communicate(
            timeout=(None if timeout_minutes <= 0 else int(timeout_minutes) * 60)
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
        try:
            stdout, _stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _stderr = process.communicate()
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + ("\nTIMEOUT\n" if timed_out else "\n")
        + "\nSTDOUT\n"
        + (stdout or "")
        + "\n",
        encoding="utf-8",
        errors="replace",
    )
    return (-1 if timed_out else int(process.returncode)), log_path, timed_out


def parse_reports(report_dir: Path) -> dict[str, Any] | None:
    timing_path = report_dir / "post_route_timing_summary.rpt"
    util_path = report_dir / "post_route_utilization.rpt"
    if not timing_path.is_file() or not util_path.is_file():
        return None
    timing = parse_vivado_timing_summary_text(
        timing_path.read_text(encoding="utf-8", errors="ignore"),
        target_clock_ns=TARGET_CLOCK_NS,
    )
    util = parse_vivado_utilization_text(
        util_path.read_text(encoding="utf-8", errors="ignore")
    )
    power_path = report_dir / "post_route_power.rpt"
    power: dict[str, Any] = {}
    if power_path.is_file():
        power = parse_vivado_power_text(
            power_path.read_text(encoding="utf-8", errors="ignore")
        )
    return {
        "timing": timing,
        "utilization": util,
        "power_estimate": power,
        "reports": {
            "post_route_timing": evidence(timing_path),
            "post_route_utilization": evidence(util_path),
            "post_route_power": evidence(power_path) if power_path.is_file() else None,
        },
    }


def load_selected_candidates(payload: dict[str, Any], roles: set[str], limit: int | None) -> tuple[list[dict[str, Any]], int]:
    candidates = list(payload.get("candidates", []))
    formal_candidate_count = int(payload.get("formal_candidate_count") or len(candidates))
    if roles:
        candidates = [row for row in candidates if str(row.get("role")) in roles]
    if limit is not None:
        candidates = candidates[: int(limit)]
    if not candidates:
        raise RuntimeError("no candidates selected for route gate")
    for row in candidates:
        if row.get("status") != "PASS":
            raise RuntimeError(f"{row.get('arch_id')}: RTL co-sim did not PASS")
    return candidates, formal_candidate_count


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    vivado: Path,
    run_vivado_flag: bool,
    force: bool,
    timeout_minutes: int,
) -> dict[str, Any]:
    hls_dir = Path(candidate["hls_project_dir"]).resolve()
    rtl_dir = hls_dir / "project" / "solution1" / "syn" / "verilog"
    if not rtl_dir.is_dir() or not list(rtl_dir.glob("*.v")):
        raise RuntimeError(f"{candidate['arch_id']}: HLS Verilog RTL missing: {rtl_dir}")
    csynth = (candidate.get("source_hls_candidate") or {}).get("csynth_report") or {}
    csynth_path = Path(str(csynth.get("path") or "")).resolve()
    if not csynth_path.is_file():
        raise RuntimeError(f"{candidate['arch_id']}: source csynth report missing")
    if sha256_file(csynth_path) != csynth.get("sha256"):
        raise RuntimeError(f"{candidate['arch_id']}: source csynth SHA mismatch")

    role_dir = output_root / str(candidate["role"])
    if role_dir.exists() and force:
        resolved = role_dir.resolve()
        allowed = output_root.resolve()
        if resolved == allowed or allowed not in resolved.parents:
            raise RuntimeError(f"refusing to remove unsafe route dir: {resolved}")
        shutil.rmtree(role_dir)
    role_dir.mkdir(parents=True, exist_ok=True)
    tcl = role_dir / "run_route.tcl"
    vivado_out = role_dir / "vivado_ooc_route"
    reports = vivado_out / "reports"
    tcl.write_text(route_tcl(rtl_dir=rtl_dir, out_dir=vivado_out), encoding="utf-8")
    generated = {"tcl": evidence(tcl)}

    if not run_vivado_flag:
        return {
            "role": candidate["role"],
            "arch_id": candidate["arch_id"],
            "status": "NOT_RUN",
            "hls_project_dir": str(hls_dir),
            "rtl_dir": str(rtl_dir),
            "route_work_dir": str(role_dir),
            "generated": generated,
            "vitis_rtl_cosim_candidate": {
                "cosim_result": candidate.get("cosim_result"),
                "vitis_log": candidate.get("vitis_log"),
            },
            "vivado_returncode": None,
            "timeout_minutes": int(timeout_minutes),
            "failure_category": None,
            "vivado_log": None,
            "metrics": None,
            "bitstream": {
                "status": "NOT_GENERATED_OOC_ROUTE_ONLY",
                "reason": "Generated HLS RTL is not yet wrapped in a board-level harness.",
            },
        }

    log_path = role_dir / "vivado_ooc_route.log"
    returncode, log_path, timed_out = run_vivado(
        vivado=vivado,
        work_dir=role_dir,
        tcl_path_=tcl,
        log_path=log_path,
        timeout_minutes=timeout_minutes,
    )
    metrics = parse_reports(reports)
    timing = (metrics or {}).get("timing") or {}
    util = (metrics or {}).get("utilization") or {}
    wns = timing.get("setup_wns_ns")
    dsp = util.get("dsp")
    route_success = returncode == 0 and metrics is not None
    wns_pass = wns is not None and float(wns) >= 0.0
    dsp_pass = dsp is not None and int(dsp) <= ROUTE_DSP_LIMIT
    if timed_out:
        status = "TIMEOUT"
        failure_category = "HWC_PARALLEL_ROUTE_TIMEOUT"
    elif not route_success:
        status = "FAIL"
        failure_category = "HWC_PARALLEL_ROUTE_VIVADO_FAIL"
    elif not wns_pass:
        status = "FAIL"
        failure_category = "HWC_PARALLEL_ROUTE_WNS_FAIL"
    elif not dsp_pass:
        status = "FAIL"
        failure_category = "HWC_PARALLEL_ROUTE_DSP_FAIL"
    else:
        status = "PASS"
        failure_category = None
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": status,
        "hls_project_dir": str(hls_dir),
        "rtl_dir": str(rtl_dir),
        "route_work_dir": str(role_dir),
        "generated": generated,
        "vitis_rtl_cosim_candidate": {
            "cosim_result": candidate.get("cosim_result"),
            "vitis_log": candidate.get("vitis_log"),
        },
        "vivado_returncode": int(returncode),
        "timeout_minutes": int(timeout_minutes),
        "failure_category": failure_category,
        "vivado_log": evidence(log_path),
        "acceptance": {
            "target_clock_ns": TARGET_CLOCK_NS,
            "wns_ns": wns,
            "wns_pass": bool(wns_pass),
            "route_dsp": dsp,
            "route_dsp_limit": ROUTE_DSP_LIMIT,
            "route_dsp_pass": bool(dsp_pass),
        },
        "metrics": metrics,
        "bitstream": {
            "status": "NOT_GENERATED_OOC_ROUTE_ONLY",
            "reason": "Generated HLS RTL is not yet wrapped in a board-level harness.",
        },
        "claim_boundary": (
            "Vivado out-of-context route evidence for generated HLS RTL. "
            "This is not a board bitstream, COM5 run, or measured power."
        ),
    }


def main() -> int:
    args = parse_args()
    rtl_summary_path = Path(args.rtl_cosim_summary).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    rtl_summary = read_json(rtl_summary_path)
    if rtl_summary.get("implementation_style") != IMPLEMENTATION_STYLE:
        raise RuntimeError(
            f"unexpected RTL implementation style: {rtl_summary.get('implementation_style')}"
        )
    if rtl_summary.get("status") != "PASS" and not args.role:
        raise RuntimeError("formal HWC RTL co-sim summary must PASS before formal route")
    candidates, formal_candidate_count = load_selected_candidates(
        rtl_summary,
        {str(role) for role in args.role},
        args.candidate_limit,
    )
    if args.run_vivado and not vivado.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_hwc_parallel_route",
            "implementation_style": IMPLEMENTATION_STYLE,
            "blocker": f"Vivado executable not found: {vivado}",
            "rtl_cosim_summary": evidence(rtl_summary_path),
            "candidate_count": len(candidates),
            "formal_candidate_count": formal_candidate_count,
            "candidates": [],
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        write_json(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        run_one_candidate(
            candidate,
            output_root=output_root,
            vivado=vivado,
            run_vivado_flag=bool(args.run_vivado),
            force=bool(args.force),
            timeout_minutes=int(args.timeout_minutes),
        )
        for candidate in candidates
    ]
    formal_scope = (
        args.candidate_limit is None
        and not args.role
        and len(rows) == formal_candidate_count
        and rtl_summary.get("status") == "PASS"
    )
    rows_pass = rows and all(row["status"] == "PASS" for row in rows)
    if not bool(args.run_vivado):
        status = "NOT_RUN"
    elif formal_scope and rows_pass:
        status = "PASS"
    elif rows_pass:
        status = "PARTIAL_PASS_NOT_FORMAL"
    elif any(row["status"] == "TIMEOUT" for row in rows):
        status = "TIMEOUT"
    else:
        status = "FAIL"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate": "full_network_hwc_parallel_route",
        "implementation_style": IMPLEMENTATION_STYLE,
        "route_scope": "vivado_ooc_generated_hls_rtl",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "route_dsp_limit": ROUTE_DSP_LIMIT,
        "rtl_cosim_summary": evidence(rtl_summary_path),
        "vivado": {"path": str(vivado), "exists": vivado.is_file()},
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "candidate_count": len(rows),
        "formal_candidate_count": formal_candidate_count,
        "formal_scope": formal_scope,
        "candidate_limit": args.candidate_limit,
        "selected_roles": [row.get("role") for row in candidates],
        "candidates": rows,
        "downstream_gates": {
            "board_harness_bitstream": (
                "PENDING" if status == "PASS" else "NOT_RUN_ROUTE_NOT_PASSED"
            ),
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "This is Vivado out-of-context route evidence for the generated "
            "HLS RTL. It does not produce a deployable board bitstream; a "
            "board-level harness/BD integration gate is required next."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
