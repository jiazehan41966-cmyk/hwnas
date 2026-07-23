#!/usr/bin/env python3
"""Single-fold full-training recipe triage for rl_arch_135.

This non-claimable mechanism diagnostic trains fresh models on the complete
fold-0/seed-42 ``split.train_indices`` while never touching outer validation.
It separates three interventions cumulatively:

1. remove random input augmentation but retain the frozen loss/regularisation;
2. additionally remove label smoothing, logit adjustment, and weight decay;
3. additionally raise the learning rate from 1e-3 to 3e-3.

The purpose is to decide whether a capacity sweep is warranted, not to select
or report a deployable model.
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
from torch.utils.data import Subset, TensorDataset

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
from hwnas_fpga.training.recipe import (  # noqa: E402
    RecipeConfig,
    build_train_criterion,
    build_warmup_cosine_scheduler,
)
from hwnas_fpga.training.trainer import evaluate_classifier  # noqa: E402

RECORD = (
    "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/"
    "run_fold0_seed42.json"
)
BASELINE_DIAGNOSTIC = (
    "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
    "g1_checkpoint_clean_train_fit_v1.json"
)
VARIANTS = (
    {
        "name": "clean_input_frozen_loss_lr1e-3",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "logit_adjust_tau": 1.0,
        "warmup_epochs": 5,
    },
    {
        "name": "clean_input_plain_ce_lr1e-3",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "logit_adjust_tau": 0.0,
        "warmup_epochs": 0,
    },
    {
        "name": "clean_input_plain_ce_lr3e-3",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "logit_adjust_tau": 0.0,
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


def cache_dataset(dataset, *, batch_size: int, num_workers: int) -> tuple[TensorDataset, dict]:
    """Materialize a deterministic dataset view once to avoid repeated image decode."""
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        seed=0,
    )
    inputs = []
    targets = []
    for batch_inputs, batch_targets in loader:
        inputs.append(batch_inputs.contiguous())
        targets.append(batch_targets.contiguous())
    input_tensor = torch.cat(inputs, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    cached = TensorDataset(input_tensor, target_tensor)
    return cached, {
        "samples": int(target_tensor.numel()),
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "bytes": int(input_tensor.numel() * input_tensor.element_size()),
        "label_counts": torch.bincount(target_tensor).tolist(),
    }


def load_existing_baseline(path: Path, tag: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload["methods"]["nas_rl_arch_135"][tag]
    return {
        "source": relative(path),
        "source_sha256": sha256(path),
        "tag": tag,
        "best_checkpoint_clean_train_top1": float(run["metrics"]["top1"]),
        "best_checkpoint_clean_train_macro_f1": float(run["metrics"]["macro_f1"]),
        "best_checkpoint_epoch": None,
        "note": (
            "Original augmented/frozen-recipe run, restored at the inner-validation-selected "
            "best checkpoint; clean train metrics are inference-only re-evaluation."
        ),
    }


def train_variant(
    *,
    variant: dict,
    record: dict,
    record_dir: Path,
    train_subset,
    clean_train_loader,
    inner_loader,
    class_counts: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict:
    set_seed(seed)
    # Rebuild the shuffled loader for every variant so all arms see the same
    # deterministic sample order rather than inheriting generator state.
    train_loader = create_dataloader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        seed=seed,
    )
    model, model_meta = build_from_record(record, record_dir, num_classes)
    model.to(device)
    recipe = RecipeConfig(
        epochs=epochs,
        optimizer="adamw",
        lr=float(variant["lr"]),
        weight_decay=float(variant["weight_decay"]),
        warmup_epochs=int(variant["warmup_epochs"]),
        min_lr_ratio=0.01,
        label_smoothing=float(variant["label_smoothing"]),
        logit_adjust_tau=float(variant["logit_adjust_tau"]),
        selection_metric="macro_f1",
        amp=False,
    )
    train_criterion = build_train_criterion(
        recipe, class_counts=class_counts.tolist()
    ).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=recipe.warmup_epochs,
        min_lr_ratio=recipe.min_lr_ratio,
    )

    history = []
    best_clean_top1 = -1.0
    best_clean_macro_f1 = -1.0
    best_clean_joint_min = -1.0
    best_inner_macro_f1 = -1.0
    first_clean_top1_095 = None
    consecutive_fit_checks = 0
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

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_mode_loss": total_loss / max(1, total_samples),
            "train_mode_top1": total_correct / max(1, total_samples),
        }
        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if should_evaluate:
            clean = evaluate_classifier(
                model,
                clean_train_loader,
                criterion=eval_criterion,
                device=str(device),
                num_classes=num_classes,
                topk=5,
            )
            inner = evaluate_classifier(
                model,
                inner_loader,
                criterion=eval_criterion,
                device=str(device),
                num_classes=num_classes,
                topk=5,
            )
            row["clean_train_eval"] = {
                "top1": float(clean["top1"]),
                "macro_f1": float(clean["macro_f1"]),
                "loss": float(clean["loss"]),
            }
            row["inner_eval"] = {
                "top1": float(inner["top1"]),
                "macro_f1": float(inner["macro_f1"]),
                "loss": float(inner["loss"]),
            }
            best_clean_top1 = max(best_clean_top1, float(clean["top1"]))
            best_clean_macro_f1 = max(best_clean_macro_f1, float(clean["macro_f1"]))
            best_clean_joint_min = max(
                best_clean_joint_min,
                min(float(clean["top1"]), float(clean["macro_f1"])),
            )
            best_inner_macro_f1 = max(best_inner_macro_f1, float(inner["macro_f1"]))
            if first_clean_top1_095 is None and clean["top1"] >= 0.95:
                first_clean_top1_095 = epoch
            if clean["top1"] >= 0.99 and clean["macro_f1"] >= 0.99:
                consecutive_fit_checks += 1
            else:
                consecutive_fit_checks = 0
            print(
                f"  {variant['name']} epoch={epoch:03d} "
                f"clean_top1={clean['top1']:.4f} clean_macro_f1={clean['macro_f1']:.4f} "
                f"inner_macro_f1={inner['macro_f1']:.4f}",
                flush=True,
            )
        history.append(row)
        if consecutive_fit_checks >= 3:
            break

    return {
        "config": {**variant, "epochs": epochs, "eval_every": eval_every},
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model": model_meta,
        "epochs_executed": epoch,
        "early_stop_rule": (
            "three consecutive evaluation checkpoints with clean train top1 and macro_f1 >= 0.99"
        ),
        "best_clean_train_top1": best_clean_top1,
        "best_clean_train_macro_f1": best_clean_macro_f1,
        "best_clean_train_joint_min": best_clean_joint_min,
        "best_inner_macro_f1": best_inner_macro_f1,
        "first_epoch_clean_top1_ge_0_95": first_clean_top1_095,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def mechanism_decision(variants: dict, baseline: dict) -> dict:
    frozen = variants["clean_input_frozen_loss_lr1e-3"]
    plain = variants["clean_input_plain_ce_lr1e-3"]
    high_lr = variants["clean_input_plain_ce_lr3e-3"]
    best = max(variants.values(), key=lambda item: item["best_clean_train_joint_min"])
    if best["best_clean_train_joint_min"] >= 0.95:
        if frozen["best_clean_train_joint_min"] >= 0.95:
            mechanism = "RANDOM_AUGMENTATION_SENSITIVITY_CONFIRMED"
        elif plain["best_clean_train_joint_min"] >= 0.95:
            mechanism = "LOSS_OR_REGULARISATION_SENSITIVITY_CONFIRMED"
        elif high_lr["best_clean_train_joint_min"] >= 0.95:
            mechanism = "LEARNING_RATE_SENSITIVITY_CONFIRMED"
        else:
            mechanism = "RECIPE_SENSITIVITY_CONFIRMED"
        return {
            "status": f"{mechanism}__SINGLE_FOLD_DIAGNOSTIC",
            "reason": (
                "at least one modified recipe reaches >=0.95 clean full-training top1 and "
                "macro_f1, versus the original best-checkpoint clean metrics of "
                f"top1={baseline['best_checkpoint_clean_train_top1']:.4f}, "
                f"macro_f1={baseline['best_checkpoint_clean_train_macro_f1']:.4f}"
            ),
            "capacity_sweep_gate": "HOLD",
            "next": (
                "validate the identified recipe factor on all 15 fold/seed runs before changing "
                "architecture capacity"
            ),
        }
    if best["best_clean_train_joint_min"] < 0.90:
        return {
            "status": "FULL_TRAIN_FIT_REMAINS_LIMITED__CAPACITY_OR_ARCHITECTURE_PLAUSIBLE",
            "reason": "no tested clean/plain/LR variant reaches 0.90 on both clean-fit metrics",
            "capacity_sweep_gate": "PROVISIONAL_PASS",
            "next": "run a predeclared multi-size capacity sweep; retain optimisation as a residual confounder",
        }
    return {
        "status": "FULL_TRAIN_RECIPE_TRIAGE_AMBIGUOUS",
        "reason": "best modified recipe lies between the operational 0.90 and 0.95 thresholds",
        "capacity_sweep_gate": "HOLD",
        "next": "repeat the best variant on two more seeds before choosing capacity versus recipe work",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--record", default=RECORD)
    parser.add_argument("--baseline-diagnostic", default=BASELINE_DIAGNOSTIC)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_full_train_recipe_triage_fold0_seed42_v1.json"
        ),
    )
    args = parser.parse_args()

    record_path = resolve(args.record)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (int(record["fold"]), int(record["seed"])) != (args.fold, args.seed):
        raise ValueError("record fold/seed does not match requested fold/seed")
    baseline_path = resolve(args.baseline_diagnostic)
    baseline = load_existing_baseline(
        baseline_path, f"fold{args.fold}_seed{args.seed}"
    )
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
    train_indices = list(bundle["split"].train_indices)
    clean_train_subset = Subset(bundle["eval_dataset"], train_indices)
    print("=== cache deterministic tensor views ===", flush=True)
    cached_train_subset, train_cache_meta = cache_dataset(
        clean_train_subset,
        batch_size=max(args.batch_size, 128),
        num_workers=args.num_workers,
    )
    cached_inner_subset, inner_cache_meta = cache_dataset(
        bundle["inner_val_loader"].dataset,
        batch_size=max(args.batch_size, 128),
        num_workers=args.num_workers,
    )
    clean_train_loader = create_dataloader(
        cached_train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed + 1,
    )
    inner_loader = create_dataloader(
        cached_inner_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        seed=args.seed + 2,
    )

    variants = {}
    for variant in VARIANTS:
        print(f"=== {variant['name']} ===", flush=True)
        variants[variant["name"]] = train_variant(
            variant=dict(variant),
            record=record,
            record_dir=record_path.parent,
            train_subset=cached_train_subset,
            clean_train_loader=clean_train_loader,
            inner_loader=inner_loader,
            class_counts=bundle["train_class_counts"],
            num_classes=bundle["num_classes"],
            device=device,
            epochs=args.epochs,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

    decision = mechanism_decision(variants, baseline)
    payload = {
        "schema_version": 1,
        "diagnostic": "single-fold full-training recipe-factor triage for rl_arch_135",
        "claimability": "mechanism diagnostic only; outer validation never consumed",
        "source_record": relative(record_path),
        "source_record_sha256": sha256(record_path),
        "baseline": baseline,
        "dataset_provenance": dataset_provenance,
        "split": {
            "fold": args.fold,
            "seed": args.seed,
            "train_count": len(train_indices),
            "training_source": "eval_dataset indexed only by split.train_indices",
            "input_augmentation": "none",
            "outer_validation_consumed": False,
        },
        "tensor_cache": {
            "semantics": "materialized deterministic eval_dataset tensors; no augmentation",
            "train": train_cache_meta,
            "inner": inner_cache_meta,
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
