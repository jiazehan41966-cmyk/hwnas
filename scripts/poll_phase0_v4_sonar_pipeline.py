#!/usr/bin/env python3
"""Poll the Phase0 v4 sonar Stage3 k3 evidence pipeline.

This script is local-only by default. It does not launch server RL300, does not
run Vivado, and does not access COM ports. It reads the launch-readiness bundle
and local search/route-gate artifacts, then emits the next required manual
action.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "hls_lut_builder" / "results" / "phase0_v4_sonar_stage3_k3_pilot"
DEFAULT_READINESS = (
    PILOT_ROOT
    / "launch_readiness"
    / "phase0_v4_sonar_stage3_k3_launch_readiness.json"
)
DEFAULT_OUTPUT = (
    PILOT_ROOT
    / "pipeline_poll"
    / "phase0_v4_sonar_stage3_k3_pipeline_poll_latest.json"
)
DEFAULT_ROUTE_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_v4_sonar_stage3_k3_pending"
)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def command_arg_after(command: list[Any], flag: str) -> str:
    parts = [str(part) for part in command]
    try:
        index = parts.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(parts):
        return ""
    return parts[index + 1]


def readiness_command(readiness: Mapping[str, Any], name: str) -> list[str]:
    commands = readiness.get("commands") if isinstance(readiness.get("commands"), dict) else {}
    command = commands.get(name)
    if not isinstance(command, list):
        return []
    return [str(part) for part in command]


def expected_paths_from_readiness(readiness: Mapping[str, Any]) -> dict[str, str]:
    launch_command = readiness_command(readiness, "manual_server_rl300_launch")
    route_command = readiness_command(readiness, "prepare_route_gate_after_search")
    output_dir = command_arg_after(launch_command, "--output-dir")
    run_name = command_arg_after(launch_command, "--run-name")
    pareto_selection = command_arg_after(route_command, "--pareto-selection")
    route_root = command_arg_after(route_command, "--output-root")
    if not pareto_selection and output_dir and run_name:
        pareto_selection = str(Path(output_dir) / run_name / "results" / "pareto_selection.json")
    return {
        "output_dir": output_dir,
        "run_name": run_name,
        "run_results": str(Path(output_dir) / run_name / "results") if output_dir and run_name else "",
        "pareto_selection": pareto_selection,
        "route_root": route_root or rel(DEFAULT_ROUTE_ROOT),
    }


def local_search_artifacts(paths: Mapping[str, str]) -> dict[str, Any]:
    run_results = Path(str(paths.get("run_results") or ""))
    if not run_results.is_absolute():
        run_results = REPO_ROOT / run_results
    return {
        "run_results": str(run_results),
        "summary": (run_results / "summary.json").exists(),
        "pareto_selection": (run_results / "pareto_selection.json").exists(),
        "pareto_ranked_candidates": (run_results / "pareto_ranked_candidates.csv").exists(),
        "lut_stats": (run_results / "lut_stats.json").exists(),
        "best_candidate": (run_results / "best_candidate.json").exists(),
        "candidates_jsonl": (run_results / "candidates.jsonl").exists(),
    }


def route_artifacts(paths: Mapping[str, str]) -> dict[str, Any]:
    route_root = Path(str(paths.get("route_root") or rel(DEFAULT_ROUTE_ROOT)))
    if not route_root.is_absolute():
        route_root = REPO_ROOT / route_root
    summary = read_json(route_root / "route_gate_summary.json")
    prepare_status = read_json(route_root / "prepare_phase0_full_route_gate_status.json")
    candidates = summary.get("candidates") if isinstance(summary.get("candidates"), list) else []
    quick_statuses = [
        str((row.get("quick_route_gate") or {}).get("gate_status") or "")
        for row in candidates
        if isinstance(row, dict)
    ]
    full_statuses = [
        str((row.get("full_route_gate") or {}).get("gate_status") or "")
        for row in candidates
        if isinstance(row, dict)
    ]
    return {
        "route_root": str(route_root),
        "prepare_status": prepare_status.get("status"),
        "route_gate_plan": (route_root / "route_gate_plan.json").exists(),
        "route_gate_summary": (route_root / "route_gate_summary.json").exists(),
        "final_candidate_manifest": (route_root / "final_candidate_manifest.json").exists(),
        "candidate_count": summary.get("candidate_count"),
        "quick_statuses": quick_statuses,
        "full_statuses": full_statuses,
        "full_route_clean_count": summary.get("full_route_clean_count"),
        "claimable_real_e2e_latency_count": summary.get("claimable_real_e2e_latency_count"),
        "route_summary": summary,
    }


def determine_next_actions(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    readiness = status.get("readiness") if isinstance(status.get("readiness"), dict) else {}
    commands = status.get("commands") if isinstance(status.get("commands"), dict) else {}
    search = status.get("local_search_artifacts") if isinstance(status.get("local_search_artifacts"), dict) else {}
    route = status.get("local_route_gate_artifacts") if isinstance(status.get("local_route_gate_artifacts"), dict) else {}

    if readiness.get("status") != "READY_FOR_MANUAL_SERVER_LAUNCH":
        return [
            {
                "id": "refresh_or_fix_launch_readiness",
                "kind": "local",
                "priority": 10,
                "message": "Launch-readiness is not READY; refresh or fix local admission evidence before any server launch.",
                "command": commands.get("refresh_readiness"),
            }
        ]

    required_search = ["summary", "pareto_selection", "pareto_ranked_candidates", "lut_stats"]
    missing_search = [name for name in required_search if not search.get(name)]
    if missing_search:
        return [
            {
                "id": "manual_server_rl300_launch_or_fetch_results",
                "kind": "manual_server",
                "priority": 20,
                "message": "Server RL300 results are not present locally; launch manually or fetch completed remote artifacts.",
                "missing": missing_search,
                "command": commands.get("manual_server_rl300_launch"),
                "resume_command": commands.get("manual_server_rl300_resume"),
            }
        ]

    if not route.get("route_gate_summary"):
        return [
            {
                "id": "prepare_route_gate_pending_summary",
                "kind": "local",
                "priority": 30,
                "message": "Search results exist; prepare route-gate PENDING artifacts with both low-DSP catalogs.",
                "command": commands.get("prepare_route_gate_after_search"),
            }
        ]

    quick_statuses = list(route.get("quick_statuses") or [])
    full_statuses = list(route.get("full_statuses") or [])
    if any(value == "PENDING" for value in quick_statuses):
        return [
            {
                "id": "manual_vivado_quick_gate",
                "kind": "vivado",
                "priority": 40,
                "message": "Quick route gate is pending; run only after explicit local Vivado approval.",
                "command": commands.get("route_gate_quick_after_manual_approval"),
            }
        ]
    if any(value in {"PENDING", "BORDERLINE"} for value in full_statuses):
        return [
            {
                "id": "manual_vivado_full_route_gate",
                "kind": "vivado",
                "priority": 50,
                "message": "Full route gate is pending or borderline; final claimability still depends on full-route PASS and actual DSP <= 700.",
                "command": commands.get("route_gate_full_after_quick_pass"),
            }
        ]

    if int(route.get("full_route_clean_count") or 0) <= 0:
        return [
            {
                "id": "no_route_clean_candidate",
                "kind": "review",
                "priority": 60,
                "message": "No full-route PASS candidate exists; do not run COM5 or claim board latency.",
            }
        ]

    if int(route.get("claimable_real_e2e_latency_count") or 0) <= 0:
        return [
            {
                "id": "manual_com5_measurement_after_route_pass",
                "kind": "measurement",
                "priority": 70,
                "message": "A route-clean candidate exists; COM5 measurement is a separate explicit step.",
            }
        ]

    return [
        {
            "id": "refresh_final_report",
            "kind": "verify",
            "priority": 80,
            "message": "Claimable evidence exists; refresh final report and verify metrics separation before making claims.",
            "command": commands.get("route_gate_refresh_after_route_or_measurement"),
        }
    ]


def run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
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
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase0 v4 Sonar Pipeline Poll",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## State",
        "",
        f"- Readiness: `{payload['readiness']['status']}`",
        f"- Search summary exists: `{payload['local_search_artifacts']['summary']}`",
        f"- Pareto selection exists: `{payload['local_search_artifacts']['pareto_selection']}`",
        f"- Route summary exists: `{payload['local_route_gate_artifacts']['route_gate_summary']}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in payload.get("next_actions") or []:
        lines.append(f"- `{action['id']}`: {action['message']}")
        command = action.get("command")
        if command:
            lines.append("")
            lines.append("```powershell")
            lines.append(" ".join(str(part) for part in command))
            lines.append("```")
            lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
        ]
    )
    for guardrail in payload.get("guardrails") or []:
        lines.append(f"- {guardrail}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Phase0 v4 sonar Stage3 k3 pipeline")
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--refresh-readiness",
        action="store_true",
        help="Refresh launch-readiness locally before polling. Does not launch server, Vivado, or COM5.",
    )
    parser.add_argument(
        "--prepare-route-gate",
        action="store_true",
        help="Run the local route-gate prepare command only when search artifacts already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands_run: dict[str, Any] = {}
    readiness_path = Path(args.readiness).expanduser().resolve()
    if args.refresh_readiness:
        commands_run["refresh_readiness"] = run_command(
            [
                args.python,
                "scripts/prepare_phase0_v4_sonar_launch_readiness.py",
                "--rerun-local-audits",
                "--require-ready",
            ]
        )
    readiness = read_json(readiness_path)
    commands = readiness.get("commands") if isinstance(readiness.get("commands"), dict) else {}
    paths = expected_paths_from_readiness(readiness)
    search = local_search_artifacts(paths)
    route = route_artifacts(paths)
    status_for_actions = {
        "readiness": {"status": readiness.get("status")},
        "commands": commands,
        "local_search_artifacts": search,
        "local_route_gate_artifacts": route,
    }
    next_actions = determine_next_actions(status_for_actions)
    if (
        args.prepare_route_gate
        and next_actions
        and next_actions[0].get("id") == "prepare_route_gate_pending_summary"
        and isinstance(next_actions[0].get("command"), list)
    ):
        commands_run["prepare_route_gate"] = run_command([str(part) for part in next_actions[0]["command"]])
        route = route_artifacts(paths)
        status_for_actions["local_route_gate_artifacts"] = route
        next_actions = determine_next_actions(status_for_actions)

    payload = {
        "generated_at": timestamp(),
        "runs_server": False,
        "runs_training": False,
        "runs_vivado": False,
        "runs_com_measurement": False,
        "readiness_path": str(readiness_path),
        "readiness": {
            "status": readiness.get("status"),
            "blocking_items": readiness.get("blocking_items"),
        },
        "expected_paths": paths,
        "local_search_artifacts": search,
        "local_route_gate_artifacts": {key: value for key, value in route.items() if key != "route_summary"},
        "commands_run": commands_run,
        "next_actions": next_actions,
        "guardrails": [
            "This poller does not launch server RL300.",
            "This poller does not run Vivado unless --prepare-route-gate is used, and that mode still only prepares local PENDING artifacts.",
            "COM5 remains blocked until a full-route PASS candidate exists.",
            "Search proxy metrics, retrain metrics, and real board measurements must remain separate.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
