#!/usr/bin/env python3
"""Frozen-protocol evaluation entrypoint.

Trains a model under the frozen NKSID evaluation protocol
(outer 5-fold x multi-seed, inner contiguous-block selection set) and reports
outer-fold metrics with mean +/- std. This is the only entrypoint whose
classification numbers are claimable; fold-0-only results are legacy.

Examples:

  # MobileNetV2 from scratch, all 5 outer folds, 3 seeds
  python run_eval_protocol.py --arch mobilenet_v2 --epochs 150 \
      --folds 0,1,2,3,4 --seeds 42,43,44 --run-name mnv2_scratch

  # Grayscale-adapted ImageNet-pretrained MobileNetV2
  python run_eval_protocol.py --arch mobilenet_v2 --pretrained --epochs 150 \
      --folds 0,1,2,3,4 --seeds 42,43,44 --run-name mnv2_pretrained

  # A searched candidate (ArchitectureSpec artifact)
  python run_eval_protocol.py --candidate-path <best_candidate.json> \
      --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --run-name rl_arch_135
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import NKSID_CLASSES, create_protocol_dataloaders
from hwnas_fpga.models import build_model
from hwnas_fpga.models.backbones import build_backbone
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.recipe import RecipeConfig, train_with_recipe
from hwnas_fpga.training.trainer import evaluate_classifier


def parse_int_list(text: str) -> list[int]:
    return [int(token) for token in str(text).replace(";", ",").split(",") if token.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_model(args: argparse.Namespace, num_classes: int) -> tuple[nn.Module, dict]:
    if args.candidate_path:
        architecture = load_architecture_from_artifact(args.candidate_path)
        model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            head_channels=architecture.head_channels,
        )
        return model, {
            "model_source": "candidate",
            "candidate_path": str(args.candidate_path),
        }

    model, metadata = build_backbone(
        name=args.arch,
        num_classes=num_classes,
        input_channels=1,
        pretrained=args.pretrained,
        strict_pretrained=args.pretrained,
    )
    metadata["model_source"] = "backbone"
    return model, metadata


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def per_class_f1(confusion: list[list[int]]) -> list[float]:
    num_classes = len(confusion)
    scores = []
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes)) - tp
        fn = sum(confusion[class_index]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--arch", default="mobilenet_v2",
                        help="backbone name (mobilenet_v2 / shufflenet_v2 / efficientnet_b0 / simplecnn)")
    parser.add_argument("--pretrained", action="store_true",
                        help="use grayscale-adapted ImageNet weights (backbones only)")
    parser.add_argument("--candidate-path", default=None,
                        help="ArchitectureSpec candidate artifact; overrides --arch")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--logit-adjust-tau", type=float, default=1.0,
                        help="0 disables logit adjustment (plain smoothed CE)")
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/protocol")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    run_name = args.run_name or (
        f"protocol_{args.arch if not args.candidate_path else 'candidate'}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    recipe = RecipeConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        logit_adjust_tau=args.logit_adjust_tau,
        early_stopping_patience=args.early_stopping_patience,
    )

    runs: list[dict] = []
    class_names: list[str] | None = None
    for fold in folds:
        for seed in seeds:
            print(f"\n=== fold {fold} seed {seed} ===")
            set_global_seed(seed)
            bundle = create_protocol_dataloaders(
                args.data_dir,
                fold=fold,
                seed=seed,
                batch_size=args.batch_size,
                image_size=args.image_size,
                inner_val_fraction=args.inner_val_fraction,
                num_workers=args.num_workers,
            )
            num_classes = bundle["num_classes"]
            class_names = bundle["classes"]
            model, model_meta = build_run_model(args, num_classes)

            result = train_with_recipe(
                model,
                train_loader=bundle["train_loader"],
                inner_val_loader=bundle["inner_val_loader"],
                num_classes=num_classes,
                recipe=recipe,
                device=device,
                class_counts=bundle["train_class_counts"].tolist(),
            )

            model.load_state_dict(result.best_state)
            model = model.to(device)
            outer_summary = evaluate_classifier(
                model,
                bundle["outer_val_loader"],
                criterion=nn.CrossEntropyLoss().to(device),
                device=device,
                num_classes=num_classes,
                topk=recipe.topk,
            )

            record = {
                "fold": fold,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "inner_val": {
                    key: result.best_inner_eval.get(key)
                    for key in ("macro_f1", "top1", "weighted_f1", "loss")
                },
                "outer_val": {
                    key: outer_summary[key]
                    for key in ("macro_f1", "top1", "weighted_f1", "top5", "loss")
                },
                "outer_confusion_matrix": outer_summary["confusion_matrix"],
                "outer_per_class_f1": per_class_f1(outer_summary["confusion_matrix"]),
                "split": bundle["split"].to_dict(),
                "model": model_meta,
            }
            runs.append(record)

            run_tag = f"fold{fold}_seed{seed}"
            with (run_dir / f"run_{run_tag}.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            if args.save_checkpoints:
                torch.save(result.best_state, run_dir / f"best_{run_tag}.pt")
            print(
                f"fold {fold} seed {seed}: outer macro_f1={outer_summary['macro_f1']:.4f} "
                f"top1={outer_summary['top1']:.4f} (best epoch {result.best_epoch})"
            )

    class_names = class_names or list(NKSID_CLASSES)
    aggregate = {
        "protocol": "nksid_outer5fold_inner_contiguous_v1",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "device": device,
        "recipe": recipe.to_dict(),
        "model": runs[0]["model"] if runs else None,
        "folds": folds,
        "seeds": seeds,
        "outer_macro_f1": summarize([r["outer_val"]["macro_f1"] for r in runs]),
        "outer_top1": summarize([r["outer_val"]["top1"] for r in runs]),
        "outer_weighted_f1": summarize([r["outer_val"]["weighted_f1"] for r in runs]),
        "per_class_f1_mean": [
            statistics.fmean(r["outer_per_class_f1"][idx] for r in runs)
            for idx in range(len(class_names))
        ] if runs else [],
        "class_names": class_names,
        "runs": [
            {key: r[key] for key in ("fold", "seed", "best_epoch", "inner_val", "outer_val")}
            for r in runs
        ],
    }
    with (run_dir / "protocol_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    lines = [
        f"# Protocol summary: {run_name}",
        "",
        f"- protocol: `nksid_outer5fold_inner_contiguous_v1`",
        f"- folds: {folds}, seeds: {seeds}, epochs: {recipe.epochs}, device: {device}",
        f"- model: {aggregate['model']}",
        "",
        "| metric | mean | std | n |",
        "|---|---:|---:|---:|",
    ]
    for key in ("outer_macro_f1", "outer_top1", "outer_weighted_f1"):
        stats = aggregate[key]
        if stats["mean"] is not None:
            lines.append(f"| {key} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} |")
    lines += ["", "| class | mean outer F1 |", "|---|---:|"]
    for name, value in zip(class_names, aggregate["per_class_f1_mean"]):
        lines.append(f"| {name} | {value:.4f} |")
    (run_dir / "protocol_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSummary written to {run_dir / 'protocol_summary.json'}")
    macro = aggregate["outer_macro_f1"]
    if macro["mean"] is not None:
        print(f"outer macro_f1 = {macro['mean']:.4f} +/- {macro['std']:.4f} (n={macro['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
