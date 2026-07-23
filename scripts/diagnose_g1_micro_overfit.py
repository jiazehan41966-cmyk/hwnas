#!/usr/bin/env python3
"""Micro-overfit gate for the frozen rl_arch_135 architecture.

The test trains fresh copies of the architecture on one fixed, class-balanced
96-sample subset drawn only from fold-0/seed-42 training indices.  The input
view is deterministic and has no random augmentation.  This is a mechanism
diagnostic, not a performance experiment and not claimable model selection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_g1_checkpoint_train_fit import (  # noqa: E402
    build_from_record,
    relative,
    resolve,
    sha256,
    verify_dataset,
)
from hwnas_fpga.data.dataset import create_dataloader, create_protocol_dataloaders  # noqa: E402
from hwnas_fpga.training.recipe import build_warmup_cosine_scheduler  # noqa: E402
from hwnas_fpga.training.trainer import evaluate_classifier  # noqa: E402

RECORD = (
    "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/"
    "run_fold0_seed42.json"
)
VARIANTS = (
    {
        "name": "frozen_regularization_lr1e-3",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "warmup_epochs": 5,
    },
    {
        "name": "plain_ce_lr3e-4",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_epochs": 0,
    },
    {
        "name": "plain_ce_lr1e-3",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_epochs": 0,
    },
    {
        "name": "plain_ce_lr3e-3",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "warmup_epochs": 0,
    },
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_balanced_indices(dataset, train_indices: list[int], per_class: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for index in train_indices:
        label = int(dataset.samples[index][1])
        by_class.setdefault(label, []).append(index)
    rng = random.Random(seed)
    selected = []
    for label in sorted(by_class):
        candidates = list(by_class[label])
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise RuntimeError(
                f"class {label} has only {len(candidates)} training samples; need {per_class}"
            )
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    return selected


def train_variant(
    *,
    variant: dict,
    record: dict,
    record_dir: Path,
    train_loader,
    eval_loader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    seed: int,
) -> dict:
    set_seed(seed)
    model, model_meta = build_from_record(record, record_dir, num_classes)
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=float(variant["label_smoothing"])
    ).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant["lr"]),
        weight_decay=float(variant["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(variant["warmup_epochs"]),
        min_lr_ratio=0.01,
    )

    history = []
    best_top1 = -1.0
    best_macro_f1 = -1.0
    first_epoch_top1_099 = None
    consecutive_perfect = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = train_criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * targets.numel()
            total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total_samples += targets.numel()
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        summary = evaluate_classifier(
            model,
            eval_loader,
            criterion=eval_criterion,
            device=str(device),
            num_classes=num_classes,
            topk=5,
        )
        best_top1 = max(best_top1, float(summary["top1"]))
        best_macro_f1 = max(best_macro_f1, float(summary["macro_f1"]))
        if first_epoch_top1_099 is None and summary["top1"] >= 0.99:
            first_epoch_top1_099 = epoch
        consecutive_perfect = consecutive_perfect + 1 if summary["top1"] >= 0.999999 else 0
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_mode_loss": total_loss / max(1, total_samples),
                "train_mode_top1": total_correct / max(1, total_samples),
                "eval_mode_top1": float(summary["top1"]),
                "eval_mode_macro_f1": float(summary["macro_f1"]),
                "eval_mode_loss": float(summary["loss"]),
            }
        )
        if epoch == 1 or epoch % 25 == 0 or consecutive_perfect == 10:
            print(
                f"  {variant['name']} epoch={epoch:03d} "
                f"eval_top1={summary['top1']:.4f} macro_f1={summary['macro_f1']:.4f}",
                flush=True,
            )
        if consecutive_perfect >= 10:
            break

    return {
        "config": variant,
        "seed": seed,
        "parameter_count": parameter_count,
        "model": model_meta,
        "epochs_executed": len(history),
        "early_stop_rule": "10 consecutive eval-mode epochs with top1 == 1.0",
        "best_eval_top1": best_top1,
        "best_eval_macro_f1": best_macro_f1,
        "first_epoch_eval_top1_ge_0_99": first_epoch_top1_099,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--record", default=RECORD)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_micro_overfit_v1.json"
        ),
    )
    args = parser.parse_args()

    record_path = resolve(args.record)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (int(record["fold"]), int(record["seed"])) != (args.fold, args.seed):
        raise ValueError("record fold/seed does not match requested fold/seed")
    data_dir = resolve(args.data_dir)
    dataset_provenance = verify_dataset(data_dir, [record])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    bundle = create_protocol_dataloaders(
        str(data_dir),
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=224,
        inner_val_fraction=0.15,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    selected_indices = select_balanced_indices(
        bundle["eval_dataset"],
        list(bundle["split"].train_indices),
        args.per_class,
        args.seed,
    )
    selected_labels = [int(bundle["eval_dataset"].samples[index][1]) for index in selected_indices]
    subset = Subset(bundle["eval_dataset"], selected_indices)
    train_loader = create_dataloader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    eval_loader = create_dataloader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed + 1,
    )

    variants = {}
    for variant in VARIANTS:
        print(f"=== {variant['name']} ===", flush=True)
        variants[variant["name"]] = train_variant(
            variant=dict(variant),
            record=record,
            record_dir=record_path.parent,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_classes=bundle["num_classes"],
            device=device,
            epochs=args.epochs,
            seed=args.seed,
        )

    best = max(variant["best_eval_top1"] for variant in variants.values())
    if best >= 0.99:
        decision = {
            "status": "MICRO_SUBSET_MEMORIZATION_PASS__FULL_DATASET_MECHANISM_UNRESOLVED",
            "reason": (
                "rl_arch_135 can memorize the fixed 96-sample clean subset; this rejects a "
                "gross implementation/trainability failure but does not reject a full-dataset "
                "capacity limit or full-recipe optimisation mismatch"
            ),
            "next": (
                "run a single fold/seed full-training ablation that removes augmentation and "
                "regularisation and brackets learning rate before any multi-size capacity sweep"
            ),
        }
    elif best >= 0.95:
        decision = {
            "status": "MICRO_SUBSET_MEMORIZATION_AMBIGUOUS",
            "reason": "best clean-subset top1 is between 0.95 and 0.99",
            "next": "inspect optimisation dynamics and extend only the best LR variant",
        }
    else:
        decision = {
            "status": "MICRO_SUBSET_MEMORIZATION_FAIL",
            "reason": "no tested LR/regularisation variant reached 0.95 on 96 clean samples",
            "next": "treat optimisation or implementation as a blocker before capacity sweep",
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "rl_arch_135 fixed clean-subset micro-overfit and LR/regularisation triage",
        "claimability": "mechanism diagnostic only; not model selection or performance evidence",
        "source_record": relative(record_path),
        "source_record_sha256": sha256(record_path),
        "dataset_provenance": dataset_provenance,
        "split": {
            "fold": args.fold,
            "seed": args.seed,
            "allowed_source": "split.train_indices only",
            "dataset_view": "eval_dataset; deterministic no-augmentation",
            "subset_selection": f"seeded class-balanced sample, {args.per_class} per class",
            "selected_indices": selected_indices,
            "class_counts": {
                str(label): selected_labels.count(label) for label in sorted(set(selected_labels))
            },
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__},
        "variants": variants,
        "decision": decision,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    print(f"written: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
