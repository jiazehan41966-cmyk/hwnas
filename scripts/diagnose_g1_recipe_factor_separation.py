#!/usr/bin/env python3
"""Separate which single recipe factor drives the +0.20 inner macro-F1 effect.

The validated bundle contrast changed four factors at once:

    factor            frozen -> plain
    weight_decay      1e-4   -> 0.0
    label_smoothing   0.1    -> 0.0
    logit_adjust_tau  1.0    -> 0.0
    warmup_epochs     5      -> 0

Replicated on fold0 seeds 42/43/44, plain CE gained +0.2014..+0.2148 inner
macro-F1 over the frozen bundle.  Committing that bundle to a formal 15-run
protocol comparison without separation would silently discard
``logit_adjust_tau``, which is the long-tail handling for a 47:1 imbalanced
dataset whose headline metric (macro-F1) is precisely long-tail sensitive.

This script runs leave-one-out arms: each removes exactly one factor from the
frozen recipe, so the factor whose removal recovers most of the gap is the
dominant one.  The two bundle endpoints are read from the existing seed-42
triage artifact rather than retrained.

Only ``split.train_indices`` and inner validation are consumed.  Outer
validation is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from diagnose_g1_full_train_recipe import (  # noqa: E402
    cache_dataset,
    load_existing_baseline,
    set_seed,
    train_variant,
)
from evaluate_g1_checkpoint_train_fit import relative, resolve, verify_dataset  # noqa: E402
from hwnas_fpga.data.dataset import create_dataloader, create_protocol_dataloaders  # noqa: E402

RECORD_TEMPLATE = (
    "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/"
    "run_fold{fold}_seed{seed}.json"
)
BASELINE_DIAGNOSTIC = (
    "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
    "g1_checkpoint_clean_train_fit_v1.json"
)
BUNDLE_TRIAGE = (
    "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
    "g1_full_train_recipe_triage_fold0_seed42_v1.json"
)

FROZEN = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "logit_adjust_tau": 1.0,
    "warmup_epochs": 5,
}
# Each arm removes exactly one factor from the frozen recipe.
LEAVE_ONE_OUT = (
    ("frozen_minus_weight_decay", {"weight_decay": 0.0}),
    ("frozen_minus_label_smoothing", {"label_smoothing": 0.0}),
    ("frozen_minus_logit_adjust", {"logit_adjust_tau": 0.0}),
    ("frozen_minus_warmup", {"warmup_epochs": 0}),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_variants() -> list[dict]:
    variants = []
    for name, override in LEAVE_ONE_OUT:
        variant = dict(FROZEN)
        variant.update(override)
        variant["name"] = name
        variant["removed_factor"] = next(iter(override))
        variants.append(variant)
    return variants


def load_bundle_endpoints(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms = payload["variants"]
    frozen = arms["clean_input_frozen_loss_lr1e-3"]
    plain = arms["clean_input_plain_ce_lr1e-3"]
    return {
        "source": relative(path),
        "source_sha256": sha256(path),
        "frozen_bundle": {
            "best_inner_macro_f1": float(frozen["best_inner_macro_f1"]),
            "best_clean_train_joint_min": float(frozen["best_clean_train_joint_min"]),
        },
        "plain_bundle": {
            "best_inner_macro_f1": float(plain["best_inner_macro_f1"]),
            "best_clean_train_joint_min": float(plain["best_clean_train_joint_min"]),
        },
        "bundle_inner_delta": float(plain["best_inner_macro_f1"])
        - float(frozen["best_inner_macro_f1"]),
    }


def attribution(variants: dict, endpoints: dict) -> dict:
    frozen_inner = endpoints["frozen_bundle"]["best_inner_macro_f1"]
    bundle_delta = endpoints["bundle_inner_delta"]
    rows = []
    for name, result in variants.items():
        delta = float(result["best_inner_macro_f1"]) - frozen_inner
        rows.append(
            {
                "arm": name,
                "removed_factor": result["config"]["removed_factor"],
                "best_inner_macro_f1": float(result["best_inner_macro_f1"]),
                "inner_delta_vs_frozen": delta,
                "share_of_bundle_delta": (delta / bundle_delta) if bundle_delta else None,
                "best_clean_train_joint_min": float(result["best_clean_train_joint_min"]),
            }
        )
    rows.sort(key=lambda r: r["inner_delta_vs_frozen"], reverse=True)
    top = rows[0]
    dominant = top["share_of_bundle_delta"] is not None and top["share_of_bundle_delta"] >= 0.5
    return {
        "status": (
            "SINGLE_FACTOR_DOMINANT" if dominant else "NO_SINGLE_DOMINANT_FACTOR"
        ),
        "bundle_inner_delta": bundle_delta,
        "ranked_arms": rows,
        "dominant_factor": top["removed_factor"] if dominant else None,
        "interpretation": (
            f"removing {top['removed_factor']} alone recovers "
            f"{top['share_of_bundle_delta']:.0%} of the bundle effect"
            if dominant
            else "no single removal recovers >=50% of the bundle effect; the factors interact"
        ),
        "guards": (
            "single fold, single seed, inner validation only; this selects which factor to "
            "carry into a formal protocol comparison, it is not itself a generalisation claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-diagnostic", default=BASELINE_DIAGNOSTIC)
    parser.add_argument("--bundle-triage", default=BUNDLE_TRIAGE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_recipe_factor_separation_fold0_seed42_v1.json"
        ),
    )
    args = parser.parse_args()

    record_path = resolve(RECORD_TEMPLATE.format(fold=args.fold, seed=args.seed))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (int(record["fold"]), int(record["seed"])) != (args.fold, args.seed):
        raise ValueError("record fold/seed does not match requested fold/seed")

    baseline = load_existing_baseline(
        resolve(args.baseline_diagnostic), f"fold{args.fold}_seed{args.seed}"
    )
    endpoints = load_bundle_endpoints(resolve(args.bundle_triage))
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
        clean_train_subset, batch_size=max(args.batch_size, 128), num_workers=args.num_workers
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
    for variant in build_variants():
        print(f"=== {variant['name']} (removed: {variant['removed_factor']}) ===", flush=True)
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

    result = attribution(variants, endpoints)
    payload = {
        "schema_version": 1,
        "diagnostic": "leave-one-out recipe factor separation for rl_arch_135",
        "claimability": "mechanism diagnostic only; outer validation never consumed",
        "motivation": (
            "the validated bundle changed weight_decay, label_smoothing, logit_adjust_tau and "
            "warmup together; logit adjustment is the long-tail handling for a 47:1 imbalanced "
            "dataset reported with macro-F1, so it must not be dropped without evidence"
        ),
        "frozen_reference_recipe": FROZEN,
        "source_record": relative(record_path),
        "source_record_sha256": sha256(record_path),
        "baseline": baseline,
        "bundle_endpoints": endpoints,
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
        "attribution": result,
    }

    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
