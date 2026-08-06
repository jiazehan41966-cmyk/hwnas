#!/usr/bin/env python3
"""Run RTL co-simulation for formal external-scratch four-stage HLS projects.

This gate consumes ``fourstage_external_scratch_hls_summary.json``.  It does
not regenerate architectures, retrain checkpoints, alter quantization, or route
the design.  It only opens the already-synthesized Vitis HLS projects and runs
``cosim_design`` against the generated testbench.

Passing this gate is still not place-and-route, bitstream, COM5 board execution,
or measured power.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import (  # noqa: E402
    DEFAULT_VITIS_HLS,
    evidence,
    read_json,
    write_json,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
DEFAULT_HLS_SUMMARY = ARTIFACT_ROOT / "fourstage_external_scratch_hls_summary.json"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "fourstage_external_scratch_rtl_cosim_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hls-summary", default=str(DEFAULT_HLS_SUMMARY))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--run-cosim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-minutes", type=int, default=90)
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


def tcl_script() -> str:
    return dedent(
        """
        open_project project
        open_solution solution1
        cosim_design -trace_level none
        exit
        """
    ).lstrip()


def run_vitis_cosim(
    *, vitis_hls: Path, hls_dir: Path, timeout_minutes: int
) -> tuple[int, Path, bool]:
    log_path = hls_dir / "vitis_hls_external_scratch_rtl_cosim.log"
    command = [str(vitis_hls), "-f", "run_rtl_cosim.tcl"]
    process = subprocess.Popen(
        command,
        cwd=hls_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(
            timeout=(None if timeout_minutes <= 0 else int(timeout_minutes) * 60)
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
                text=True,
            )
        else:  # pragma: no cover - current execution is Windows.
            process.kill()
        stdout, stderr = process.communicate()
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + ("\nTIMEOUT\n" if timed_out else "\n")
        + "\nSTDOUT\n"
        + (stdout or "")
        + "\nSTDERR\n"
        + (stderr or "")
        + "\n",
        encoding="utf-8",
        errors="replace",
    )
    return (-1 if timed_out else int(process.returncode)), log_path, timed_out


def classify_failure(
    *,
    status: str,
    timed_out: bool,
    returncode: int,
    result_payload: dict[str, Any] | None,
    log_path: Path | None,
) -> str | None:
    if status == "PASS":
        return None
    if timed_out:
        return "EXTERNAL_SCRATCH_RTL_COSIM_TIMEOUT"
    if not isinstance(result_payload, dict):
        return "EXTERNAL_SCRATCH_RTL_COSIM_NO_RESULT_JSON"
    if result_payload.get("status") != "PASS" or result_payload.get("mismatch_count") != 0:
        return "EXTERNAL_SCRATCH_RTL_COSIM_OUTPUT_MISMATCH"
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "ERROR:" in log_text:
            return "EXTERNAL_SCRATCH_RTL_COSIM_TOOL_ERROR"
    if returncode != 0:
        return "EXTERNAL_SCRATCH_RTL_COSIM_RETURNCODE_NONZERO"
    return "EXTERNAL_SCRATCH_RTL_COSIM_FAIL"


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    vitis_hls: Path,
    run_cosim: bool,
    timeout_minutes: int,
) -> dict[str, Any]:
    if candidate.get("status") != "PASS":
        raise RuntimeError(f"{candidate.get('arch_id')}: HLS candidate did not PASS")
    hls_dir = Path(candidate["hls_project_dir"]).resolve()
    csynth = candidate.get("csynth_report") or {}
    csynth_path = Path(csynth.get("path", "")).resolve()
    if not hls_dir.is_dir():
        raise RuntimeError(f"{candidate['arch_id']}: HLS project dir missing: {hls_dir}")
    if not csynth_path.is_file():
        raise RuntimeError(f"{candidate['arch_id']}: csynth report missing: {csynth_path}")
    if sha256_file(csynth_path) != csynth.get("sha256"):
        raise RuntimeError(f"{candidate['arch_id']}: csynth SHA mismatch")

    tcl_path = hls_dir / "run_rtl_cosim.tcl"
    result_path = hls_dir / "csim_result.json"
    tcl_path.write_text(tcl_script(), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    generated = {"tcl": evidence(tcl_path)}
    if not run_cosim:
        return {
            "role": candidate["role"],
            "arch_id": candidate["arch_id"],
            "status": "NOT_RUN",
            "hls_project_dir": str(hls_dir),
            "generated": generated,
            "source_hls_candidate": {
                "csynth_report": csynth,
                "metrics": candidate.get("metrics"),
            },
            "vitis_returncode": None,
            "timeout_minutes": int(timeout_minutes),
            "failure_category": None,
            "vitis_log": None,
            "cosim_result": None,
            "claim_boundary": "RTL co-sim TCL generated only; co-sim was not run.",
        }

    returncode, log_path, timed_out = run_vitis_cosim(
        vitis_hls=vitis_hls,
        hls_dir=hls_dir,
        timeout_minutes=timeout_minutes,
    )
    result_payload = read_json(result_path) if result_path.is_file() else None
    mismatch_count = (
        int(result_payload["mismatch_count"])
        if isinstance(result_payload, dict) and "mismatch_count" in result_payload
        else None
    )
    status = (
        "PASS"
        if returncode == 0
        and isinstance(result_payload, dict)
        and result_payload.get("status") == "PASS"
        and mismatch_count == 0
        else "FAIL"
    )
    if timed_out:
        status = "TIMEOUT"
    failure_category = classify_failure(
        status=status,
        timed_out=timed_out,
        returncode=returncode,
        result_payload=result_payload,
        log_path=log_path,
    )
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": status,
        "hls_project_dir": str(hls_dir),
        "generated": generated,
        "source_hls_candidate": {
            "csynth_report": csynth,
            "metrics": candidate.get("metrics"),
        },
        "sample_count": (
            int(result_payload["sample_count"])
            if isinstance(result_payload, dict) and "sample_count" in result_payload
            else None
        ),
        "output_count": (
            int(result_payload["output_count"])
            if isinstance(result_payload, dict) and "output_count" in result_payload
            else None
        ),
        "mismatch_count": mismatch_count,
        "max_abs_mismatch": (
            int(result_payload["max_abs_mismatch"])
            if isinstance(result_payload, dict) and "max_abs_mismatch" in result_payload
            else None
        ),
        "vitis_returncode": int(returncode),
        "timeout_minutes": int(timeout_minutes),
        "failure_category": failure_category,
        "vitis_log": evidence(log_path),
        "cosim_result": evidence(result_path) if result_path.is_file() else None,
        "claim_boundary": (
            "Vitis HLS RTL co-simulation evidence only. This is not "
            "place-and-route, bitstream, COM5 board execution, or measured power."
        ),
    }


def main() -> int:
    args = parse_args()
    hls_summary_path = Path(args.hls_summary).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    hls_summary = read_json(hls_summary_path)
    if hls_summary.get("status") != "PASS" or not hls_summary.get("formal_scope"):
        raise RuntimeError("formal external-scratch HLS gate must PASS before RTL co-sim")
    candidates = list(hls_summary["candidates"])
    formal_candidate_count = len(candidates)
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if args.run_cosim and not vitis_hls.is_file():
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "gate": "full_network_external_scratch_rtl_cosim",
            "blocker": f"Vitis HLS executable not found: {vitis_hls}",
            "hls_summary": evidence(hls_summary_path),
            "candidate_count": len(candidates),
            "formal_candidate_count": formal_candidate_count,
            "candidates": [],
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        write_json(summary_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    rows = [
        run_one_candidate(
            candidate,
            vitis_hls=vitis_hls,
            run_cosim=bool(args.run_cosim),
            timeout_minutes=int(args.timeout_minutes),
        )
        for candidate in candidates
    ]
    formal_scope = args.candidate_limit is None and len(rows) == formal_candidate_count
    rows_pass = rows and all(row["status"] == "PASS" for row in rows)
    if not bool(args.run_cosim):
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
        "gate": "full_network_external_scratch_rtl_cosim",
        "implementation_style": "external_scratch_pixel_tiled_v1",
        "hls_summary": evidence(hls_summary_path),
        "vitis_hls": {"path": str(vitis_hls), "exists": vitis_hls.is_file()},
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "candidate_count": len(rows),
        "formal_candidate_count": formal_candidate_count,
        "formal_scope": formal_scope,
        "candidate_limit": args.candidate_limit,
        "candidates": rows,
        "downstream_gates": {
            "place_and_route_5ns": "PENDING" if status == "PASS" else "NOT_RUN_RTL_COSIM_NOT_PASSED",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "Complete-network RTL co-simulation is separate from AV7K325 "
            "place-and-route, bitstream generation, COM5 board execution, "
            "and external-meter power."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "PARTIAL_PASS_NOT_FORMAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
