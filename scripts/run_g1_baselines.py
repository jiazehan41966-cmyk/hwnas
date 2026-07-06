#!/usr/bin/env python3
"""Run/resume the three frozen-protocol G1 baselines (45 training tasks)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL135 = (
    REPO_ROOT
    / "hls_lut_builder/board_harness/results/"
    "pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp/"
    "candidates/003_rl_arch_135.candidate.json"
)


def command_specs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    common = [
        sys.executable,
        str(REPO_ROOT / "run_eval_protocol.py"),
        "--data-dir",
        str(Path(args.data_dir)),
        "--output-dir",
        str(Path(args.output_dir)),
        "--folds",
        "0,1,2,3,4",
        "--seeds",
        "42,43,44",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "8",
        "--gradient-accumulation-steps",
        "4",
        "--amp",
        "--save-checkpoints",
        "--resume",
    ]
    return [
        (
            "mobilenet_v2_scratch",
            common
            + [
                "--arch",
                "mobilenet_v2",
                "--selection-provenance",
                "baseline_predeclared",
                "--run-name",
                "g1_mobilenet_v2_scratch",
            ],
        ),
        (
            "mobilenet_v2_grayscale_imagenet",
            common
            + [
                "--arch",
                "mobilenet_v2",
                "--pretrained",
                "--selection-provenance",
                "baseline_predeclared",
                "--run-name",
                "g1_mobilenet_v2_grayscale_imagenet",
            ],
        ),
        (
            "rl_arch_135_legacy_selected",
            common
            + [
                "--candidate-path",
                str(Path(args.rl135_candidate).resolve()),
                "--selection-provenance",
                "legacy_fold0_selected",
                "--run-name",
                "g1_rl_arch_135_legacy_selected",
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--output-dir", default="results/protocol")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--rl135-candidate", default=str(DEFAULT_RL135))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not Path(args.rl135_candidate).exists():
        raise FileNotFoundError(args.rl135_candidate)

    specs = command_specs(args)
    ledger = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "planned_training_tasks": 45,
        "models": [],
        "dry_run": args.dry_run,
    }
    overall = 0
    for name, command in specs:
        print(subprocess.list2cmdline(command), flush=True)
        returncode = None
        if not args.dry_run:
            returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
            if returncode != 0:
                overall = returncode
        ledger["models"].append(
            {
                "name": name,
                "planned_tasks": 15,
                "command": command,
                "returncode": returncode,
            }
        )
        if overall:
            break

    output = Path(args.output_dir) / "g1_baseline_launcher.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
