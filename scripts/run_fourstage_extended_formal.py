#!/usr/bin/env python3
"""Run a gated subset of the 12/16-row Protocol V2 operator enumeration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
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


def claimable_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not payload.get("claimability", {}).get("claimable"):
        return None
    return payload


def build_command(
    *,
    row: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    source_freeze_manifest: Path,
    data_dir: Path,
    output_dir: Path,
    device: str,
    num_workers: int,
) -> list[str]:
    dataset = contract["dataset"]
    recipe = contract["training_recipe"]
    arch_id = str(row["arch_id"])
    command = [
        sys.executable,
        str(ROOT / "run_eval_protocol.py"),
        "--data-dir",
        str(data_dir),
        "--candidate-path",
        str(Path(row["path"]).resolve()),
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
        str(num_workers),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--run-name",
        arch_id,
        "--selection-provenance",
        "new_nested",
        "--experiment-contract",
        str(contract_path),
        "--source-freeze-manifest",
        str(source_freeze_manifest),
        "--campaign-id",
        "sonar_fourstage_operator_v2_extended",
        "--paper-id",
        "project_internal",
        "--method-id",
        arch_id,
        "--resume",
        "--save-checkpoints",
    ]
    command.append("--amp" if recipe["amp"] else "--no-amp")
    if dataset["fixed_scale_factor"] is not None:
        command.extend(
            ["--fixed-scale-factor", str(dataset["fixed_scale_factor"])]
        )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--contract",
        default="configs/evaluation/nksid_frozen_protocol_v2.yaml",
    )
    parser.add_argument(
        "--candidate-manifest",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "extended_candidates/extended_manifest.json"
        ),
    )
    parser.add_argument("--source-freeze-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/sonar_fourstage_operator_v2/extended_formal",
    )
    parser.add_argument(
        "--stage4",
        default="MBConv-k5-e3",
        choices=(
            "Skip",
            "MBConv-k3-e3",
            "Dir-MBConv3-e3",
            "MBConv-k5-e3",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least one")

    contract_path = Path(args.contract).resolve()
    manifest_path = Path(args.candidate_manifest).resolve()
    source_freeze_manifest = Path(args.source_freeze_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_count") not in (12, 16):
        raise ValueError("extended manifest must contain 12 or 16 candidates")
    selected = [
        row
        for row in manifest["rows"]
        if row.get("factors", {}).get("stage4") == args.stage4
    ]
    if len(selected) != 4:
        raise ValueError(
            f"Expected four {args.stage4} candidates, found {len(selected)}"
        )
    verifier = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(source_freeze_manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if verifier.returncode:
        raise RuntimeError(
            f"source-freeze verification failed: {verifier.stdout}"
        )

    status_path = output_dir / f"{args.stage4}_orchestration_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fourstage_extended_formal_subset",
        "stage4_filter": args.stage4,
        "protocol": contract["protocol"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "source_freeze_manifest": str(source_freeze_manifest),
        "source_freeze_manifest_sha256": sha256_file(
            source_freeze_manifest
        ),
        "source_freeze_verification": json.loads(verifier.stdout),
        "data_dir": str(Path(args.data_dir).resolve()),
        "device": args.device,
        "num_workers_per_candidate": args.num_workers,
        "max_parallel": args.max_parallel,
        "started_at": now(),
        "status": "RUNNING",
        "completed_candidates": [],
        "running_candidates": [],
        "pending_candidates": [row["arch_id"] for row in selected],
    }
    write_status(status_path, status)

    pending = list(selected)
    running: dict[str, tuple[subprocess.Popen[Any], Any, Path]] = {}
    while pending or running:
        while pending and len(running) < args.max_parallel:
            row = pending.pop(0)
            arch_id = str(row["arch_id"])
            summary_path = output_dir / arch_id / "protocol_summary.json"
            summary = claimable_summary(summary_path)
            if summary is not None:
                status["completed_candidates"].append(
                    {
                        "arch_id": arch_id,
                        "summary_path": str(summary_path),
                        "summary_sha256": sha256_file(summary_path),
                        "outer_macro_f1": summary["outer_macro_f1"],
                        "resume_state": "SKIPPED_CLAIMABLE_EXISTING",
                    }
                )
                continue
            candidate_dir = output_dir / arch_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            log_path = candidate_dir / "orchestrator_subprocess.log"
            command = build_command(
                row=row,
                contract=contract,
                contract_path=contract_path,
                source_freeze_manifest=source_freeze_manifest,
                data_dir=Path(args.data_dir).resolve(),
                output_dir=output_dir,
                device=args.device,
                num_workers=args.num_workers,
            )
            log_stream = log_path.open("a", encoding="utf-8", errors="replace")
            log_stream.write(
                "\n[launch] " + json.dumps(command, ensure_ascii=False) + "\n"
            )
            log_stream.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            running[arch_id] = (process, log_stream, summary_path)

        status["running_candidates"] = sorted(running)
        status["pending_candidates"] = [
            str(row["arch_id"]) for row in pending
        ]
        status["updated_at"] = now()
        write_status(status_path, status)
        if not running:
            continue
        time.sleep(2)
        for arch_id in list(running):
            process, log_stream, summary_path = running[arch_id]
            returncode = process.poll()
            if returncode is None:
                continue
            log_stream.close()
            del running[arch_id]
            summary = claimable_summary(summary_path)
            if returncode != 0 or summary is None:
                for other_process, other_log, _ in running.values():
                    other_process.terminate()
                    other_log.close()
                status.update(
                    {
                        "status": "FAILED",
                        "failed_candidate": arch_id,
                        "returncode": returncode,
                        "running_candidates": sorted(running),
                        "updated_at": now(),
                    }
                )
                write_status(status_path, status)
                return int(returncode or 2)
            status["completed_candidates"].append(
                {
                    "arch_id": arch_id,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "outer_macro_f1": summary["outer_macro_f1"],
                    "resume_state": "COMPLETED",
                }
            )

    status.update(
        {
            "status": "COMPLETE",
            "running_candidates": [],
            "pending_candidates": [],
            "completed_at": now(),
        }
    )
    write_status(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
