#!/usr/bin/env python3
"""Run/resume the three frozen-protocol G1 baselines (45 training tasks)."""

from __future__ import annotations

import argparse
import hashlib
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
        "--campaign-id",
        "ccf_ab_nksid_av7k325_v1",
        "--paper-id",
        "project_internal",
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
        "--num-workers",
        str(getattr(args, "num_workers", 0)),
        "--amp",
        "--save-checkpoints",
        "--resume",
    ]
    source_freeze_manifest = getattr(args, "source_freeze_manifest", None)
    if source_freeze_manifest:
        common.extend(
            ["--source-freeze-manifest", str(Path(source_freeze_manifest).resolve())]
        )
    return [
        (
            "mobilenet_v2_scratch",
            common
            + [
                "--arch",
                "mobilenet_v2",
                "--selection-provenance",
                "baseline_predeclared",
                "--method-id",
                "scratch_mobilenet_v2",
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
                "--method-id",
                "imagenet_pretrained_mobilenet_v2",
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
                "--method-id",
                "frozen_nas_champion",
                "--run-name",
                "g1_rl_arch_135_legacy_selected",
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument(
        "--output-dir",
        default="results/protocol/g1_clean_20260711",
        help="Independent clean root; legacy/mixed result directories are never reused.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rl135-candidate", default=str(DEFAULT_RL135))
    parser.add_argument("--source-freeze-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not Path(args.rl135_candidate).exists():
        raise FileNotFoundError(args.rl135_candidate)
    if not args.dry_run and not args.source_freeze_manifest:
        raise RuntimeError(
            "formal G1 launch requires --source-freeze-manifest; create it only after "
            "the source and regression suite are final"
        )

    freeze_manifest = None
    freeze_manifest_sha256 = None
    if args.source_freeze_manifest:
        freeze_manifest = Path(args.source_freeze_manifest).resolve()
        if not freeze_manifest.is_file():
            raise FileNotFoundError(freeze_manifest)
        freeze_manifest_sha256 = hashlib.sha256(freeze_manifest.read_bytes()).hexdigest()

    def verify_source() -> int:
        if freeze_manifest is None:
            return 0
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/freeze_experiment_source.py"),
                "verify",
                "--manifest",
                str(freeze_manifest),
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode

    if verify_source() != 0:
        raise RuntimeError("source freeze verification failed before G1 launch")

    specs = command_specs(args)
    ledger = {
        "schema_version": 2,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "planned_training_tasks": 45,
        "models": [],
        "dry_run": args.dry_run,
        "clean_protocol_root": str(Path(args.output_dir).resolve()),
        "normalization": {"channels": 1, "mean": [0.5], "std": [0.5]},
        "group_split_available": False,
        "group_generalization_claimable": False,
        "legacy_results_merged": False,
        "num_workers": args.num_workers,
        "source_freeze_manifest": str(freeze_manifest) if freeze_manifest else None,
        "source_freeze_manifest_sha256": freeze_manifest_sha256,
    }
    overall = 0
    for name, command in specs:
        if verify_source() != 0:
            overall = 2
            ledger["models"].append(
                {
                    "name": name,
                    "planned_tasks": 15,
                    "command": command,
                    "returncode": overall,
                    "reason": "source_freeze_verification_failed",
                }
            )
            break
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
