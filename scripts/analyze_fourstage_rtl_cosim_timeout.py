#!/usr/bin/env python3
"""Analyze why formal four-stage RTL co-sim timed out.

The RTL co-sim gate may emit useful XSIM progress lines before timeout.  This
script converts those lines into a conservative feasibility estimate so the
next step is evidence-led rather than blindly rerunning a multi-hour simulator
job.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from run_fourstage_csim_gate import evidence, read_json, write_json  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
DEFAULT_RTL_SUMMARY = ARTIFACT_ROOT / "fourstage_external_scratch_rtl_cosim_summary.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "rtl_cosim_timeout_feasibility_audit.json"

PROGRESS_RE = re.compile(
    r"RTL Simulation\s*:\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)\s*"
    r"\[(?P<percent>\d+(?:\.\d+)?)%\]\s*@\s*\"(?P<time_ps>\d+)\""
)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_progress(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.is_file():
        return rows
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "line_number": line_number,
                "completed_transactions": int(match.group("done")),
                "total_transactions": int(match.group("total")),
                "intra_transaction_percent": float(match.group("percent")),
                "simulation_time_ps": int(match.group("time_ps")),
                "line": line.strip(),
            }
        )
    return rows


def analyze_candidate(row: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(row["vitis_log"]["path"]).resolve()
    progress_rows = parse_progress(log_path)
    latest = progress_rows[-1] if progress_rows else None
    timeout_minutes = int(row.get("timeout_minutes") or 0)
    sample_count = int(row.get("sample_count") or 0)
    metrics = ((row.get("source_hls_candidate") or {}).get("metrics") or {})
    cycles = metrics.get("cycles")
    estimate: dict[str, Any] = {
        "role": row["role"],
        "arch_id": row["arch_id"],
        "rtl_status": row["status"],
        "failure_category": row.get("failure_category"),
        "sample_count": sample_count,
        "mismatch_count_from_c_precheck": row.get("mismatch_count"),
        "max_abs_mismatch_from_c_precheck": row.get("max_abs_mismatch"),
        "hls_cycles_per_transaction": cycles,
        "hls_latency_ns_per_transaction": metrics.get("latency_ns"),
        "rtl_log": evidence(log_path),
        "progress_observations": progress_rows,
        "latest_progress": latest,
    }
    if latest and latest["intra_transaction_percent"] > 0 and timeout_minutes > 0:
        fraction = latest["intra_transaction_percent"] / 100.0
        minutes_per_transaction = timeout_minutes / fraction
        minutes_for_recorded_sample_count = minutes_per_transaction * max(sample_count, 1)
        estimate.update(
            {
                "estimated_minutes_per_transaction_from_timeout_window": minutes_per_transaction,
                "estimated_hours_per_transaction": minutes_per_transaction / 60.0,
                "estimated_hours_for_recorded_sample_count": (
                    minutes_for_recorded_sample_count / 60.0
                ),
                "single_sample_rerun_still_multi_hour": minutes_per_transaction > 180.0,
            }
        )
    else:
        estimate.update(
            {
                "estimated_minutes_per_transaction_from_timeout_window": None,
                "estimated_hours_per_transaction": None,
                "estimated_hours_for_recorded_sample_count": None,
                "single_sample_rerun_still_multi_hour": True,
            }
        )
    return estimate


def main() -> int:
    rtl_summary = read_json(DEFAULT_RTL_SUMMARY)
    if rtl_summary.get("status") != "TIMEOUT":
        raise RuntimeError("RTL co-sim summary is not a TIMEOUT gate")
    candidates = [analyze_candidate(row) for row in rtl_summary["candidates"]]
    total_hours_recorded = sum(
        float(row.get("estimated_hours_for_recorded_sample_count") or 0.0)
        for row in candidates
    )
    total_hours_single_sample = sum(
        float(row.get("estimated_hours_per_transaction") or 0.0)
        for row in candidates
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RTL_COSIM_TIMEOUT_FEASIBILITY_FAIL",
        "gate": "rtl_cosim_timeout_feasibility_audit",
        "rtl_cosim_summary": evidence(DEFAULT_RTL_SUMMARY),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "aggregate_estimate": {
            "estimated_hours_for_current_8_sample_formal_cosim": total_hours_recorded,
            "estimated_hours_for_single_sample_all_candidates": total_hours_single_sample,
            "single_sample_rerun_recommended": False,
            "reason": (
                "The observed XSIM progress lines imply that even one full "
                "network transaction per candidate would take multiple hours. "
                "Blindly rerunning the same external-scratch RTL co-sim with "
                "fewer samples would not close the gate efficiently."
            ),
        },
        "downstream_gates": {
            "place_and_route_5ns": "NOT_RUN_RTL_COSIM_NOT_PASSED",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "recommended_next_work": [
            "Reduce per-transaction RTL latency before rerunning formal RTL co-sim.",
            "Replace the layer-serial external-scratch probe with a co-sim-feasible tiled/streaming datapath or a validated staged RTL parity strategy.",
            "Do not start AV7K325 route, bitstream, COM5, or power gates until RTL co-sim returns PASS.",
        ],
        "claim_boundary": (
            "This audit estimates RTL simulator feasibility from already "
            "captured timeout logs. It is not a new RTL co-sim PASS and not "
            "route, bitstream, COM5, or power evidence."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(DEFAULT_OUTPUT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
