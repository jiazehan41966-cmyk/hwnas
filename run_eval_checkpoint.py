#!/usr/bin/env python3
"""Evaluate a searched or retrained checkpoint on the configured validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.deploy.inference import load_checkpoint_model
from hwnas_fpga.runtime import create_data_pipeline, load_config, pick
from hwnas_fpga.training.trainer import evaluate_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final HW-NAS checkpoint metrics on a validation split"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=("dummy", "nksid"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _default_output_path(checkpoint: Path) -> Path:
    run_dir = checkpoint.parent.parent
    return run_dir / "results" / "final_eval_summary.json"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    project_cfg = config.get("project", {})

    device = pick(args.device, None, "cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = pick(args.dataset, dataset_cfg.get("name"), "dummy")
    image_size = pick(args.image_size, dataset_cfg.get("image_size"), 224)
    input_channels = dataset_cfg.get("input_channels", 1)
    batch_size = pick(args.batch_size, training_cfg.get("batch_size"), 32)
    num_workers = pick(args.num_workers, dataset_cfg.get("num_workers"), 0)
    fold = pick(args.fold, dataset_cfg.get("fold"), 0)
    data_dir = pick(args.data_dir, dataset_cfg.get("data_dir"), None)
    use_kfold = bool(dataset_cfg.get("use_kfold", True))
    valid_size = dataset_cfg.get("valid_size")
    split_seed = int(dataset_cfg.get("split_seed", project_cfg.get("seed", 42)))

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(checkpoint)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, architecture, checkpoint_payload, class_names = load_checkpoint_model(
        checkpoint,
        device=device,
    )
    _train_loader, val_loader, class_weights, resolved_num_classes = create_data_pipeline(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=dataset_cfg.get("num_classes"),
        input_channels=input_channels,
        fold=fold,
        num_workers=num_workers,
        device=device,
        use_kfold=use_kfold,
        valid_size=valid_size,
        split_seed=split_seed,
    )
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )
    metrics = evaluate_classifier(
        model,
        val_loader,
        criterion=criterion,
        device=device,
        num_classes=resolved_num_classes,
        topk=5,
    )
    payload = {
        "checkpoint_path": str(checkpoint),
        "config_path": str(Path(args.config).expanduser().resolve()) if args.config else None,
        "dataset": {
            "name": dataset_name,
            "data_dir": data_dir,
            "fold": fold,
            "use_kfold": use_kfold,
            "valid_size": valid_size,
            "split_seed": split_seed,
            "image_size": image_size,
            "batch_size": batch_size,
            "val_samples": len(val_loader.dataset),
            "num_classes": resolved_num_classes,
            "class_names": class_names,
        },
        "architecture": architecture.to_dict(),
        "checkpoint_metrics": checkpoint_payload.get("metrics", {}),
        "metrics": metrics,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final evaluation summary: {output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
