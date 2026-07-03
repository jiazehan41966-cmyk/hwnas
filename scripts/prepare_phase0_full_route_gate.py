#!/usr/bin/env python3
"""Prepare route-gate planning artifacts for the full Phase0 RL300 search.

This script does not run Vivado. It creates the same PENDING route-gate plan and
summary that the remote post-search watcher is expected to create after the
full search writes pareto_selection.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARETO = (
    REPO_ROOT
    / "results"
    / "remote_phase0_full_rl300_search"
    / "results"
    / "pareto_selection.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_full_rl300_pending"
)
ROUTE_GATE_SCRIPT = REPO_ROOT / "hls_lut_builder" / "board_harness" / "scripts" / "pareto_route_gate.py"
DEFAULT_LOWDSP_OVERRIDE_CATALOG = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "configs"
    / "phase0_v3_lowdsp_override_catalog.yaml"
)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full Phase0 route-gate plan artifacts")
    parser.add_argument("--pareto-selection", default=str(DEFAULT_PARETO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--topk", type=int, default=24)
    parser.add_argument("--full-topk", type=int, default=8)
    parser.add_argument("--full-max-actual-dsp", type=int, default=700)
    parser.add_argument(
        "--lowdsp-override-catalog",
        action="append",
        default=None,
        help=(
            "Low-DSP override catalog path. May be repeated, e.g. to layer "
            "the Phase0 v3 catalog with a Phase0 v4 sonar catalog."
        ),
    )
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return non-zero when pareto_selection.json is missing.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    pareto_path = Path(args.pareto_selection).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    status_path = output_root / "prepare_phase0_full_route_gate_status.json"
    lowdsp_catalog_paths = list(
        args.lowdsp_override_catalog or [str(DEFAULT_LOWDSP_OVERRIDE_CATALOG)]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    base_payload: dict[str, Any] = {
        "generated_at": timestamp(),
        "pareto_selection": str(pareto_path),
        "output_root": str(output_root),
        "route_gate_script": str(ROUTE_GATE_SCRIPT),
        "topk": args.topk,
        "full_topk": args.full_topk,
        "full_max_actual_dsp": args.full_max_actual_dsp,
        "lowdsp_override_catalog": lowdsp_catalog_paths,
        "serial_port": args.serial_port,
        "runs_vivado": False,
    }

    if not pareto_path.exists():
        payload = {
            **base_payload,
            "status": "waiting_for_pareto_selection",
            "message": "Full Phase0 RL300 pareto_selection.json is not available yet.",
            "artifacts": {
                "route_gate_plan": str(output_root / "route_gate_plan.json"),
                "route_gate_summary": str(output_root / "route_gate_summary.json"),
                "final_candidate_manifest": str(output_root / "final_candidate_manifest.json"),
            },
        }
        write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2 if args.require_ready else 0

    lowdsp_catalog_args = []
    for catalog_path in lowdsp_catalog_paths:
        lowdsp_catalog_args.extend(["--lowdsp-override-catalog", catalog_path])

    plan_command = [
        args.python,
        str(ROUTE_GATE_SCRIPT),
        "plan",
        "--pareto-selection",
        str(pareto_path),
        "--output-root",
        str(output_root),
        "--topk",
        str(args.topk),
        "--full-topk",
        str(args.full_topk),
        "--full-max-actual-dsp",
        str(args.full_max_actual_dsp),
        *lowdsp_catalog_args,
        "--serial-port",
        args.serial_port,
    ]
    refresh_command = [
        args.python,
        str(ROUTE_GATE_SCRIPT),
        "refresh",
        "--pareto-selection",
        str(pareto_path),
        "--output-root",
        str(output_root),
        "--topk",
        str(args.topk),
        "--full-topk",
        str(args.full_topk),
        "--full-max-actual-dsp",
        str(args.full_max_actual_dsp),
        *lowdsp_catalog_args,
        "--serial-port",
        args.serial_port,
    ]
    plan_result = run_command(plan_command)
    refresh_result = run_command(refresh_command)
    ok = plan_result["returncode"] == 0 and refresh_result["returncode"] == 0
    payload = {
        **base_payload,
        "status": "prepared" if ok else "failed",
        "commands": {
            "plan": plan_result,
            "refresh": refresh_result,
        },
        "artifacts": {
            "route_gate_plan": str(output_root / "route_gate_plan.json"),
            "route_gate_summary": str(output_root / "route_gate_summary.json"),
            "final_candidate_manifest": str(output_root / "final_candidate_manifest.json"),
            "final_route_clean_best": str(output_root / "final_route_clean_best.json"),
            "final_claimable_board_best": str(output_root / "final_claimable_board_best.json"),
        },
    }
    write_json(status_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
