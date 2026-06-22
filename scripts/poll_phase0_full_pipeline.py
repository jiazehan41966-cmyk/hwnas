#!/usr/bin/env python3
"""Poll and advance the full Phase0 RL300 evidence pipeline.

The command is intentionally conservative:
- fetches remote evidence when the SSH tunnel is reachable;
- records a structured unreachable state when it is not;
- prepares route-gate PENDING artifacts only after full Pareto exists;
- runs the acceptance audit without launching Vivado or COM measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "remote_phase0_full_rl300_status" / "pipeline_poll_latest.json"
DEFAULT_REMOTE_STATUS_STALE_AFTER_SECONDS = 3600


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_utc_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def compute_remote_status_freshness(
    captured_at_utc: Any,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = DEFAULT_REMOTE_STATUS_STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    parsed = parse_utc_timestamp(captured_at_utc)
    payload: dict[str, Any] = {
        "captured_at_utc": captured_at_utc,
        "stale_after_seconds": stale_after_seconds,
        "age_seconds": None,
        "is_stale": None,
    }
    if parsed is None:
        return payload
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    age_seconds = max(0, int((reference.astimezone(timezone.utc) - parsed).total_seconds()))
    payload["age_seconds"] = age_seconds
    payload["is_stale"] = age_seconds > stale_after_seconds
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll full Phase0 RL300 pipeline status")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--password-env", default="HWNAS_REMOTE_PASSWORD")
    parser.add_argument("--fetch-retries", type=int, default=2)
    parser.add_argument("--fetch-retry-delay", type=float, default=5.0)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def select_route_clean_action_candidate(route_summary: dict[str, Any]) -> dict[str, Any]:
    candidates = route_summary.get("candidates")
    if not isinstance(candidates, list):
        return {}

    route_clean_best = (
        route_summary.get("route_clean_best")
        if isinstance(route_summary.get("route_clean_best"), dict)
        else {}
    )
    best_arch_id = str(route_clean_best.get("arch_id") or "")
    best_selection_index = route_clean_best.get("selection_index")
    pass_rows = [
        row
        for row in candidates
        if isinstance(row, dict)
        and isinstance(row.get("full_route_gate"), dict)
        and row["full_route_gate"].get("gate_status") == "PASS"
    ]
    if not pass_rows:
        return {}

    selected = pass_rows[0]
    if best_arch_id or best_selection_index is not None:
        for row in pass_rows:
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            same_arch = not best_arch_id or str(candidate.get("arch_id") or "") == best_arch_id
            same_index = (
                best_selection_index is None
                or candidate.get("selection_index") == best_selection_index
            )
            if same_arch and same_index:
                selected = row
                break

    measurement_json = str(selected.get("measurement_json") or "")
    final_validation_report_json = str(selected.get("final_validation_report_json") or "")
    measurement_path = Path(measurement_json) if measurement_json else Path()
    report_path = Path(final_validation_report_json) if final_validation_report_json else Path()
    return {
        "candidate": selected.get("candidate"),
        "measurement_json": measurement_json,
        "measurement_json_exists": bool(measurement_json) and measurement_path.exists(),
        "final_validation_report_json": final_validation_report_json,
        "final_validation_report_exists": bool(final_validation_report_json) and report_path.exists(),
        "measurement_command": selected.get("measurement_command"),
        "final_report_command": selected.get("final_report_command"),
        "full_build_result": selected.get("full_build_result"),
        "full_route_gate": selected.get("full_route_gate"),
    }


def extract_status() -> dict[str, Any]:
    remote_status = read_json(REPO_ROOT / "results/remote_phase0_full_rl300_status/remote_status_latest.json")
    fetch_error = read_json(REPO_ROOT / "results/remote_phase0_full_rl300_status/fetch_error_latest.json")
    route_prepare = read_json(
        REPO_ROOT
        / "hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending/prepare_phase0_full_route_gate_status.json"
    )
    audit = read_json(REPO_ROOT / "results/hwnas_gap_acceptance_audit/acceptance_audit.json")
    full_results = REPO_ROOT / "results/remote_phase0_full_rl300_search/results"
    route_root = REPO_ROOT / "hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending"
    route_summary = read_json(route_root / "route_gate_summary.json")
    route_candidates = (
        route_summary.get("candidates")
        if isinstance(route_summary.get("candidates"), list)
        else []
    )
    quick_statuses = [
        str((row.get("quick_route_gate") or {}).get("gate_status") or "")
        for row in route_candidates
        if isinstance(row, dict)
    ]
    full_statuses = [
        str((row.get("full_route_gate") or {}).get("gate_status") or "")
        for row in route_candidates
        if isinstance(row, dict)
    ]
    remote_status_details = {
        "captured_at_utc": remote_status.get("captured_at_utc"),
        "search_state": remote_status.get("search_state"),
        "summary_exists": remote_status.get("summary_exists"),
        "pareto_exists": remote_status.get("pareto_exists"),
        "route_summary_exists": remote_status.get("route_summary_exists"),
    }
    return {
        "remote_status": remote_status_details,
        "remote_status_freshness": compute_remote_status_freshness(
            remote_status_details.get("captured_at_utc")
        ),
        "latest_fetch_error": {
            "status": (fetch_error.get("status") or {}).get("connection_status")
            if isinstance(fetch_error.get("status"), dict)
            else None,
            "error_type": (fetch_error.get("status") or {}).get("error_type")
            if isinstance(fetch_error.get("status"), dict)
            else None,
            "error": (fetch_error.get("status") or {}).get("error")
            if isinstance(fetch_error.get("status"), dict)
            else None,
        },
        "local_full_search_artifacts": {
            "summary": (full_results / "summary.json").exists(),
            "pareto_selection": (full_results / "pareto_selection.json").exists(),
            "pareto_ranked_candidates": (full_results / "pareto_ranked_candidates.csv").exists(),
            "lut_stats": (full_results / "lut_stats.json").exists(),
        },
        "local_route_gate_artifacts": {
            "prepare_status": route_prepare.get("status"),
            "route_gate_plan": (route_root / "route_gate_plan.json").exists(),
            "route_gate_summary": (route_root / "route_gate_summary.json").exists(),
            "final_candidate_manifest": (route_root / "final_candidate_manifest.json").exists(),
            "candidate_count": route_summary.get("candidate_count"),
            "quick_statuses": quick_statuses,
            "full_statuses": full_statuses,
            "full_route_clean_count": route_summary.get("full_route_clean_count"),
            "claimable_real_e2e_latency_count": route_summary.get("claimable_real_e2e_latency_count"),
            "route_clean_action_candidate": select_route_clean_action_candidate(route_summary),
        },
        "acceptance_audit": {
            "status": audit.get("status"),
            "status_counts": audit.get("status_counts"),
            "remaining_gaps": audit.get("remaining_gaps"),
        },
    }


def determine_next_actions(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered next actions without redefining completion criteria."""

    actions: list[dict[str, Any]] = []
    fetch_error = status.get("latest_fetch_error") if isinstance(status.get("latest_fetch_error"), dict) else {}
    remote_freshness = (
        status.get("remote_status_freshness")
        if isinstance(status.get("remote_status_freshness"), dict)
        else {}
    )
    full_artifacts = (
        status.get("local_full_search_artifacts")
        if isinstance(status.get("local_full_search_artifacts"), dict)
        else {}
    )
    route_artifacts = (
        status.get("local_route_gate_artifacts")
        if isinstance(status.get("local_route_gate_artifacts"), dict)
        else {}
    )
    audit = status.get("acceptance_audit") if isinstance(status.get("acceptance_audit"), dict) else {}

    if audit.get("status") == "ready_for_completion":
        return [
            {
                "id": "completion_audit",
                "kind": "verify",
                "priority": 0,
                "message": "Acceptance audit reports ready_for_completion; run final requirement-by-requirement verification before marking the goal complete.",
            }
        ]

    if fetch_error.get("status") == "failed":
        evidence = dict(fetch_error)
        if remote_freshness:
            evidence["remote_status_freshness"] = remote_freshness
        actions.append(
            {
                "id": "restore_remote_access",
                "kind": "external",
                "priority": 10,
                "message": "SSH tunnel/port is currently unreachable; retry after the remote port is available or update host/port credentials.",
                "evidence": evidence,
            }
        )

    required_full = ["summary", "pareto_selection", "pareto_ranked_candidates", "lut_stats"]
    missing_full = [name for name in required_full if not full_artifacts.get(name)]
    if missing_full:
        actions.append(
            {
                "id": "fetch_full_phase0_results",
                "kind": "fetch",
                "priority": 20,
                "message": "Fetch full Phase0 RL300 summary/Pareto/ranked CSV/LUT stats from the remote run.",
                "missing": missing_full,
                "command": "python scripts/poll_phase0_full_pipeline.py",
            }
        )
        return actions

    if not route_artifacts.get("route_gate_summary"):
        actions.append(
            {
                "id": "prepare_route_gate_pending_summary",
                "kind": "local",
                "priority": 30,
                "message": "Generate route-gate plan and PENDING summary for full Phase0 Pareto top-K.",
                "command": "python scripts/prepare_phase0_full_route_gate.py",
            }
        )
        return actions

    quick_statuses = list(route_artifacts.get("quick_statuses") or [])
    full_statuses = list(route_artifacts.get("full_statuses") or [])
    if any(status_value == "PENDING" for status_value in quick_statuses):
        actions.append(
            {
                "id": "execute_vivado_quick_gate",
                "kind": "vivado",
                "priority": 40,
                "message": "Run Vivado quick gate for Pareto top-K candidates and refresh route-gate summary.",
                "command": "python hls_lut_builder/board_harness/scripts/pareto_route_gate.py quick --pareto-selection results/remote_phase0_full_rl300_search/results/pareto_selection.json --output-root hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending --execute",
            }
        )
        return actions

    if any(status_value in {"PENDING", "BORDERLINE"} for status_value in full_statuses):
        actions.append(
            {
                "id": "execute_vivado_full_route_gate",
                "kind": "vivado",
                "priority": 50,
                "message": "Run full route gate for top candidates; final best may only come from full_route_gate PASS rows.",
                "command": "python hls_lut_builder/board_harness/scripts/pareto_route_gate.py full --pareto-selection results/remote_phase0_full_rl300_search/results/pareto_selection.json --output-root hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending --execute",
            }
        )
        return actions

    if int(route_artifacts.get("full_route_clean_count") or 0) <= 0:
        actions.append(
            {
                "id": "no_route_clean_candidate",
                "kind": "review",
                "priority": 60,
                "message": "No full-route PASS candidate exists; inspect failures before selecting any final best.",
            }
        )
        return actions

    route_clean_action = (
        route_artifacts.get("route_clean_action_candidate")
        if isinstance(route_artifacts.get("route_clean_action_candidate"), dict)
        else {}
    )
    measurement_exists = bool(route_clean_action.get("measurement_json_exists"))
    if int(route_artifacts.get("claimable_real_e2e_latency_count") or 0) <= 0:
        if measurement_exists:
            actions.append(
                {
                    "id": "refresh_or_review_com5_measurement",
                    "kind": "verify",
                    "priority": 70,
                    "message": (
                        "A COM5 measurement JSON exists for the route-clean best, but the "
                        "route-gate summary does not yet contain a claimable real e2e latency; "
                        "refresh the summary/report and inspect blockers before re-measuring."
                    ),
                    "command": "python hls_lut_builder/board_harness/scripts/pareto_route_gate.py refresh --pareto-selection results/remote_phase0_full_rl300_search/results/pareto_selection.json --output-root hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending",
                    "evidence": route_clean_action,
                }
            )
            return actions
        actions.append(
            {
                "id": "run_com5_measurement",
                "kind": "measurement",
                "priority": 70,
                "message": "Measure route-clean best on COM5 and refresh the final hardware validation report.",
                "command": route_clean_action.get("measurement_command"),
                "after_measurement_command": "python hls_lut_builder/board_harness/scripts/pareto_route_gate.py refresh --pareto-selection results/remote_phase0_full_rl300_search/results/pareto_selection.json --output-root hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending",
                "final_report_command": route_clean_action.get("final_report_command"),
                "evidence": route_clean_action,
            }
        )
        return actions

    actions.append(
        {
            "id": "refresh_acceptance_audit",
            "kind": "verify",
            "priority": 80,
            "message": "Refresh acceptance audit and verify every objective requirement before marking complete.",
            "command": "python scripts/hwnas_gap_acceptance_audit.py",
        }
    )
    return actions


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    commands: dict[str, Any] = {}

    if not args.skip_fetch:
        commands["fetch"] = run_command(
            [
                args.python,
                "scripts/fetch_remote_phase0_full_rl300.py",
                "--password-env",
                args.password_env,
                "--retries",
                str(args.fetch_retries),
                "--retry-delay",
                str(args.fetch_retry_delay),
                "--allow-unreachable",
            ],
            env=env,
        )

    commands["prepare_route_gate"] = run_command(
        [args.python, "scripts/prepare_phase0_full_route_gate.py"],
        env=env,
    )
    commands["acceptance_audit"] = run_command(
        [args.python, "scripts/hwnas_gap_acceptance_audit.py"],
        env={**env, "PYTHONPATH": "src"},
    )
    status = extract_status()
    payload = {
        "generated_at": timestamp(),
        "runs_vivado": False,
        "runs_com_measurement": False,
        "commands": commands,
        "status": status,
        "next_actions": determine_next_actions(status),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
