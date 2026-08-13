#!/usr/bin/env python3
"""Run a tiny, test-only search-result -> retrain -> Protocol V2 interface smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import architecture_sha256, enumerate_cfem20  # noqa: E402
from hwnas_fpga.training import load_architecture_from_artifact, retrain_architecture  # noqa: E402
from hwnas_fpga.training.protocol_reporting import canonical_sha256, sha256_file  # noqa: E402
from hwnas_fpga.training.trainer import evaluate_classifier  # noqa: E402
from run_eval_protocol import build_run_model, load_checkpoint_state  # noqa: E402


SEED = 20260813
ARCH_ID = "fourstage_s2_k3_e6_s4_cfem_sonar"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zcu104_search_train_eval_migration_smoke_20260813",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    candidate_row = next(row for row in enumerate_cfem20() if row.arch_id == ARCH_ID)
    architecture_hash = architecture_sha256(candidate_row.architecture)
    candidate_path = output_dir / "search_result.test_only.json"
    candidate_payload = {
        "schema_version": 2,
        "saved_at": "2026-08-13",
        "candidate": {
            "arch_id": candidate_row.arch_id,
            "encoding": candidate_row.architecture.to_dict(),
            "architecture_sha256": architecture_hash,
            "metrics": {
                "macro_f1": None,
                "top1": None,
                "claim_boundary": "test-only architecture handoff; no search performance",
            },
        },
        "search_space_version": "fourstage_cfem20_v1",
        "test_only": True,
        "formal_eligible": False,
    }
    write_json(candidate_path, candidate_payload)
    candidate_file_sha = sha256_file(candidate_path)

    # Tiny deterministic synthetic data. It verifies interfaces only and is
    # deliberately outside every formal dataset/result directory.
    generator = torch.Generator().manual_seed(SEED)
    train_x = torch.randn(8, 1, 32, 32, generator=generator)
    train_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    val_x = torch.randn(8, 1, 32, 32, generator=generator)
    val_y = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=2, shuffle=False)
    data_split = {
        "kind": "TEST_ONLY_SYNTHETIC",
        "seed": SEED,
        "train_samples": 8,
        "val_samples": 8,
        "image_shape": [1, 32, 32],
        "batch_size": 2,
    }
    training_config = {
        "epochs": 1,
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "selection_metric": "macro_f1",
        "device": "cpu",
        "seed": SEED,
    }
    architecture = load_architecture_from_artifact(candidate_path)
    model, training_metrics, history = retrain_architecture(
        architecture=architecture,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=8,
        epochs=1,
        optimizer_name="adamw",
        lr=0.001,
        weight_decay=0.0001,
        device="cpu",
        selection_metric="macro_f1",
    )
    checkpoint_path = output_dir / "final_best_model.test_only.pt"
    checkpoint_payload = {
        "schema_version": 2,
        "source_candidate": candidate_payload["candidate"],
        "source_candidate_file_sha256": candidate_file_sha,
        "architecture": architecture.to_dict(),
        "architecture_sha256": architecture_hash,
        "training_config": training_config,
        "training_config_sha256": canonical_sha256(training_config),
        "data_split_config": data_split,
        "data_split_config_sha256": canonical_sha256(data_split),
        "data_split_sha256": canonical_sha256(data_split),
        "seed": SEED,
        "metrics": training_metrics,
        "history": history,
        "model_state_dict": model.state_dict(),
        "test_only": True,
        "formal_eligible": False,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    # Exercise the actual Protocol V2 consumer functions: candidate rebuild and
    # checkpoint-state loading, followed by the existing classification metric code.
    protocol_args = argparse.Namespace(candidate_path=str(candidate_path))
    protocol_model, protocol_model_meta = build_run_model(protocol_args, num_classes=8)
    protocol_model.load_state_dict(load_checkpoint_state(checkpoint_path))
    protocol_metrics = evaluate_classifier(
        protocol_model,
        val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        num_classes=8,
        topk=5,
    )
    summary_path = output_dir / "protocol_v2_interface_smoke.json"
    summary = {
        "schema_version": 1,
        "status": "PASS_INTERFACE_SMOKE_ONLY",
        "formal_eligible": False,
        "claim_boundary": (
            "Synthetic 8+8 sample, one-epoch CPU smoke. It proves interface and "
            "SHA256 propagation only; it is not NKSID performance or a Protocol V2 result."
        ),
        "command": (
            "python scripts/run_zcu104_search_train_eval_migration_smoke.py "
            f"--output-dir {output_dir}"
        ),
        "seed": SEED,
        "search_space_version": "fourstage_cfem20_v1",
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_file_sha,
            "arch_id": ARCH_ID,
            "architecture_sha256": architecture_hash,
        },
        "training_config": training_config,
        "training_config_sha256": checkpoint_payload["training_config_sha256"],
        "data_split_config": data_split,
        "data_split_config_sha256": checkpoint_payload["data_split_config_sha256"],
        "data_split_sha256": checkpoint_payload["data_split_sha256"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
        },
        "protocol_v2_consumer": {
            "model_meta": protocol_model_meta,
            "metrics": protocol_metrics,
        },
    }
    write_json(summary_path, summary)
    manifest = {
        "status": summary["status"],
        "formal_eligible": False,
        "files": {
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
