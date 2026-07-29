#!/usr/bin/env python3
"""Run the frozen Protocol V2 2x2x2 base factorial, one candidate at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--contract",
        default="configs/evaluation/nksid_frozen_protocol_v2.yaml",
    )
    parser.add_argument(
        "--candidate-dir",
        default="artifacts/sonar_fourstage_operator_v2/base8_candidates",
    )
    parser.add_argument("--source-freeze-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/sonar_fourstage_operator_v2/base8_formal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    manifest_path = Path(args.candidate_dir).resolve() / "base8_manifest.json"
    candidate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("candidate_count") != 8:
        raise ValueError("base8 manifest must contain exactly eight candidates")
    output_dir = Path(args.output_dir).resolve()
    status_path = output_dir / "orchestration_status.json"
    status = {
        "schema_version": 1,
        "experiment": "fourstage_base8_full_factorial",
        "protocol": contract["protocol"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "source_freeze_manifest": str(Path(args.source_freeze_manifest).resolve()),
        "started_at": now(),
        "status": "RUNNING",
        "completed_candidates": [],
        "current_candidate": None,
    }
    write_status(status_path, status)

    dataset = contract["dataset"]
    recipe = contract["training_recipe"]
    for row in candidate_manifest["rows"]:
        arch_id = str(row["arch_id"])
        candidate_path = Path(row["path"]).resolve()
        status["current_candidate"] = arch_id
        status["updated_at"] = now()
        write_status(status_path, status)
        command = [
            sys.executable,
            str(ROOT / "run_eval_protocol.py"),
            "--data-dir",
            str(Path(args.data_dir).resolve()),
            "--candidate-path",
            str(candidate_path),
            "--folds",
            ",".join(str(value) for value in contract["formal_reporting"]["folds"]),
            "--seeds",
            ",".join(str(value) for value in contract["formal_reporting"]["seeds"]),
            "--epochs",
            str(recipe["epochs"]),
            "--batch-size",
            str(recipe["batch_size"]),
            "--gradient-accumulation-steps",
            str(recipe["gradient_accumulation_steps"]),
            "--image-size",
            str(dataset["image_size"]),
            "--geometry-mode",
            str(dataset["geometry_mode"]),
            "--geometry-padding-value",
            str(dataset["geometry_padding_value"]),
            "--augmentation-profile",
            str(dataset["augmentation_profile"]),
            "--lr",
            str(recipe["lr"]),
            "--weight-decay",
            str(recipe["weight_decay"]),
            "--warmup-epochs",
            str(recipe["warmup_epochs"]),
            "--min-lr-ratio",
            str(recipe["min_lr_ratio"]),
            "--label-smoothing",
            str(recipe["label_smoothing"]),
            "--logit-adjust-tau",
            str(recipe["logit_adjust_tau"]),
            "--inner-val-fraction",
            str(dataset["inner_val_fraction"]),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--output-dir",
            str(output_dir),
            "--run-name",
            arch_id,
            "--selection-provenance",
            "new_nested",
            "--experiment-contract",
            str(contract_path),
            "--source-freeze-manifest",
            str(Path(args.source_freeze_manifest).resolve()),
            "--campaign-id",
            "sonar_fourstage_operator_v2",
            "--paper-id",
            "project_internal",
            "--method-id",
            arch_id,
            "--resume",
            "--save-checkpoints",
        ]
        if recipe["amp"]:
            command.append("--amp")
        else:
            command.append("--no-amp")
        if dataset["fixed_scale_factor"] is not None:
            command.extend(
                ["--fixed-scale-factor", str(dataset["fixed_scale_factor"])]
            )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            status.update(
                {
                    "status": "FAILED",
                    "failed_candidate": arch_id,
                    "returncode": completed.returncode,
                    "updated_at": now(),
                }
            )
            write_status(status_path, status)
            return int(completed.returncode)
        summary_path = output_dir / arch_id / "protocol_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary["claimability"]["claimable"]:
            status.update(
                {
                    "status": "FAILED",
                    "failed_candidate": arch_id,
                    "reason": "formal protocol summary is not claimable",
                    "updated_at": now(),
                }
            )
            write_status(status_path, status)
            return 2
        status["completed_candidates"].append(
            {
                "arch_id": arch_id,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path),
                "outer_macro_f1": summary["outer_macro_f1"],
            }
        )
    status.update(
        {
            "status": "COMPLETE",
            "current_candidate": None,
            "completed_at": now(),
        }
    )
    write_status(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
