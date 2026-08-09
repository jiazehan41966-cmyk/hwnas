#!/usr/bin/env python3
"""Direct-route a four-stage external-scratch HLS candidate without RTL co-sim.

This is a board bring-up shortcut requested after the RTL co-simulation proved
too slow for the current iteration.  It consumes an HLS summary whose candidate
has C-sim/synthesis PASS evidence and runs Vivado out-of-context route on the
generated HLS Verilog.

Evidence boundary: a PASS here is *not* bit-exact RTL co-simulation parity and
is not a deployable board bitstream.  It only answers whether the generated RTL
can pass the post-route timing/resource gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import evidence, read_json, write_json  # noqa: E402
from run_fourstage_hwc_parallel_route_gate import (  # noqa: E402
    DEFAULT_VIVADO,
    ROUTE_DSP_LIMIT,
    parse_reports,
    route_tcl,
    run_vivado,
)
from run_fourstage_hls_synthesis_gate import TARGET_CLOCK_NS, TARGET_PART  # noqa: E402


IMPLEMENTATION_STYLE = "external_scratch_pixel_tiled_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
DEFAULT_HLS_SUMMARY = (
    ARTIFACT_ROOT
    / "fourstage_external_scratch_stage4_k5_sample1_hls_shortpath_summary.json"
)
DEFAULT_SUMMARY = (
    ARTIFACT_ROOT
    / "fourstage_external_scratch_stage4_k5_sample1_direct_route_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results" / "sonar_fourstage_operator_v2" / "external_scratch_direct_route"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hls-summary", default=str(DEFAULT_HLS_SUMMARY))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--run-vivado", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    parser.add_argument(
        "--ack-skip-rtl-cosim",
        action="store_true",
        help="Required when --run-vivado is enabled; records the evidence downgrade.",
    )
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


def selected_candidates(
    hls_summary: dict[str, Any], roles: set[str], limit: int | None
) -> tuple[list[dict[str, Any]], int]:
    if hls_summary.get("implementation_style") != IMPLEMENTATION_STYLE:
        raise RuntimeError(
            f"unexpected HLS implementation style: {hls_summary.get('implementation_style')}"
        )
    if hls_summary.get("status") not in {"PASS", "PARTIAL_PASS_NOT_FORMAL"}:
        raise RuntimeError(f"HLS summary is not routeable: {hls_summary.get('status')}")
    rows = list(hls_summary.get("candidates", []))
    formal_count = int(hls_summary.get("formal_candidate_count") or len(rows))
    if roles:
        rows = [row for row in rows if str(row.get("role")) in roles]
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("no candidates selected for direct route")
    for row in rows:
        if row.get("status") != "PASS":
            raise RuntimeError(f"{row.get('arch_id')}: HLS candidate did not PASS")
    return rows, formal_count


def safe_remove(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RuntimeError(f"refusing to remove unsafe route dir: {resolved}")
    shutil.rmtree(resolved)


def run_one(
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
    csynth = candidate.get("csynth_report") or {}
    csynth_path = Path(str(csynth.get("path") or "")).resolve()
    if not csynth_path.is_file():
        raise RuntimeError(f"{candidate['arch_id']}: source csynth report missing")
    if sha256_file(csynth_path) != csynth.get("sha256"):
        raise RuntimeError(f"{candidate['arch_id']}: source csynth SHA mismatch")

    role_dir = output_root / str(candidate["role"])
    if role_dir.exists() and force:
        safe_remove(role_dir, output_root)
    role_dir.mkdir(parents=True, exist_ok=True)
    for data_file in rtl_dir.glob("*.dat"):
        shutil.copy2(data_file, role_dir / data_file.name)

    tcl = role_dir / "run_route.tcl"
    vivado_out = role_dir / "vivado_ooc_route"
    reports = vivado_out / "reports"
    tcl.write_text(route_tcl(rtl_dir=rtl_dir, out_dir=vivado_out), encoding="utf-8")
    base = {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "hls_project_dir": str(hls_dir),
        "rtl_dir": str(rtl_dir),
        "route_work_dir": str(role_dir),
        "generated": {
            "tcl": evidence(tcl),
            "copied_dat_count": len(list(role_dir.glob("*.dat"))),
        },
        "source_hls_candidate": {
            "csim_result": candidate.get("csim_result"),
            "vitis_log": candidate.get("vitis_log"),
            "csynth_report": candidate.get("csynth_report"),
            "metrics": candidate.get("metrics"),
        },
        "rtl_cosim": {
            "status": "SKIPPED_FOR_BOARD_BRINGUP",
            "claim_boundary": "未做逐位一致的硬件电路联合仿真证明。",
        },
        "timeout_minutes": int(timeout_minutes),
        "bitstream": {
            "status": "NOT_GENERATED_OOC_ROUTE_ONLY",
            "reason": "当前只布线裸 HLS 顶层，还没有板级封装。",
        },
    }
    if not run_vivado_flag:
        return {
            **base,
            "status": "NOT_RUN",
            "vivado_returncode": None,
            "failure_category": None,
            "vivado_log": None,
            "acceptance": None,
            "metrics": None,
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
        status, failure = "TIMEOUT", "EXTERNAL_SCRATCH_DIRECT_ROUTE_TIMEOUT"
    elif not route_success:
        status, failure = "FAIL", "EXTERNAL_SCRATCH_DIRECT_ROUTE_VIVADO_FAIL"
    elif not wns_pass:
        status, failure = "FAIL", "EXTERNAL_SCRATCH_DIRECT_ROUTE_WNS_FAIL"
    elif not dsp_pass:
        status, failure = "FAIL", "EXTERNAL_SCRATCH_DIRECT_ROUTE_DSP_FAIL"
    else:
        status, failure = "PASS", None
    return {
        **base,
        "status": status,
        "vivado_returncode": int(returncode),
        "failure_category": failure,
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
        "claim_boundary": (
            "跳过联合仿真的脱离板子布线证据；不是硬件逐位一致证明、"
            "不是可上板比特流、不是板级运行或实测功耗。"
        ),
    }


def main() -> int:
    args = parse_args()
    if args.run_vivado and not args.ack_skip_rtl_cosim:
        raise RuntimeError("--ack-skip-rtl-cosim is required for direct route")
    hls_summary_path = Path(args.hls_summary).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    hls_summary = read_json(hls_summary_path)
    candidates, formal_count = selected_candidates(
        hls_summary, {str(role) for role in args.role}, args.candidate_limit
    )
    if args.run_vivado and not vivado.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_external_scratch_direct_route",
            "blocker": f"Vivado executable not found: {vivado}",
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        write_json(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        run_one(
            row,
            output_root=output_root,
            vivado=vivado,
            run_vivado_flag=bool(args.run_vivado),
            force=bool(args.force),
            timeout_minutes=int(args.timeout_minutes),
        )
        for row in candidates
    ]
    rows_pass = rows and all(row["status"] == "PASS" for row in rows)
    hls_formal = hls_summary.get("status") == "PASS" and hls_summary.get("formal_scope")
    formal_scope = (
        args.candidate_limit is None
        and not args.role
        and len(rows) == formal_count
        and hls_formal
    )
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
        "gate": "full_network_external_scratch_direct_route",
        "implementation_style": IMPLEMENTATION_STYLE,
        "route_scope": "vivado_ooc_generated_hls_rtl_without_rtl_cosim",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "route_dsp_limit": ROUTE_DSP_LIMIT,
        "hls_summary": evidence(hls_summary_path),
        "vivado": {"path": str(vivado), "exists": vivado.is_file()},
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "candidate_count": len(rows),
        "formal_candidate_count": formal_count,
        "formal_scope": formal_scope,
        "candidate_limit": args.candidate_limit,
        "selected_roles": [row.get("role") for row in candidates],
        "rtl_cosim": {
            "status": "SKIPPED_FOR_BOARD_BRINGUP",
            "claim_boundary": "未做逐位一致的硬件电路联合仿真证明。",
        },
        "candidates": rows,
        "downstream_gates": {
            "board_harness_bitstream": (
                "PENDING"
                if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"}
                else "NOT_RUN_ROUTE_NOT_PASSED"
            ),
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "这是为推进上板而跳过联合仿真的直接布线证据。它不能证明硬件"
            "输出与软件参考逐位一致，也不能单独代表可上板比特流。"
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
