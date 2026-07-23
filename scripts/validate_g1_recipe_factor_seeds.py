#!/usr/bin/env python3
"""Validate the G1 loss/regularisation bundle on fold0 seeds 43 and 44.

This is the minimal follow-up to the seed-42 mechanism triage.  For each seed,
it compares exactly two fresh-model arms on deterministic clean training input:

- frozen loss/regularisation: label smoothing 0.1, logit adjustment 1.0,
  weight decay 1e-4, lr 1e-3;
- plain loss: ordinary CE, no weight decay, lr 1e-3.

Only ``split.train_indices`` and inner validation are consumed.  Outer
validation remains untouched until a later, formally frozen 15-run protocol.
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
    VARIANTS,
    cache_dataset,
    load_existing_baseline,
    set_seed,
    train_variant,
)
from evaluate_g1_checkpoint_train_fit import relative, resolve, verify_dataset  # noqa: E402
from hwnas_fpga.data.dataset import create_dataloader, create_protocol_dataloaders  # noqa: E402

RECORD_TEMPLATE = (
    "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/"
    "run_fold0_seed{seed}.json"
)
BASELINE_DIAGNOSTIC = (
    "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
    "g1_checkpoint_clean_train_fit_v1.json"
)
SELECTED_VARIANTS = tuple(
    variant for variant in VARIANTS if variant["name"] != "clean_input_plain_ce_lr3e-3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_decision(variants: dict, baseline: dict) -> dict:
    frozen = variants["clean_input_frozen_loss_lr1e-3"]
    plain = variants["clean_input_plain_ce_lr1e-3"]
    clean_pass = plain["best_clean_train_joint_min"] >= 0.95
    inner_delta = plain["best_inner_macro_f1"] - frozen["best_inner_macro_f1"]
    return {
        "status": (
            "DIRECTIONAL_REPLICATION_PASS" if clean_pass and inner_delta > 0 else
            "DIRECTIONAL_REPLICATION_FAIL"
        ),
        "plain_clean_joint_min": plain["best_clean_train_joint_min"],
        "inner_macro_f1_delta_plain_minus_frozen": inner_delta,
        "original_checkpoint_clean_macro_f1": baseline["best_checkpoint_clean_train_macro_f1"],
        "guards": (
            "clean fit must reach >=0.95 and best inner macro_f1 direction must favor plain CE; "
            "this is a mechanism gate, not a formal outer-fold claim"
        ),
    }


def overall_decision(seed_results: dict) -> dict:
    passes = [result["decision"]["status"] == "DIRECTIONAL_REPLICATION_PASS" for result in seed_results.values()]
    deltas = [result["decision"]["inner_macro_f1_delta_plain_minus_frozen"] for result in seed_results.values()]
    if len(passes) == 2 and all(passes):
        return {
            "status": "RECIPE_FACTOR_REPLICATED_ON_SEEDS_43_44",
            "new_seed_passes": "2/2",
            "inner_macro_f1_deltas": deltas,
            "next_gate": (
                "freeze the plain-CE/no-weight-decay recipe bundle and run a formal paired 15-run "
                "protocol comparison with outer validation consumed once per run"
            ),
            "capacity_sweep": "HOLD",
        }
    return {
        "status": "RECIPE_FACTOR_REPLICATION_NOT_CLOSED",
        "new_seed_passes": f"{sum(passes)}/{len(passes)}",
        "inner_macro_f1_deltas": deltas,
        "next_gate": "do not launch a formal 15-run recipe comparison or capacity sweep",
        "capacity_sweep": "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--seeds", default="43,44")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-diagnostic", default=BASELINE_DIAGNOSTIC)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_recipe_factor_seed43_44_v1.json"
        ),
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if seeds != [43, 44]:
        raise ValueError("This frozen validation is scoped exactly to seeds 43,44")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data_dir = resolve(args.data_dir)
    baseline_path = resolve(args.baseline_diagnostic)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_results = {}
    payload = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "diagnostic": "fold0 seed43/44 minimal loss-regularisation bundle validation",
        "claimability": "mechanism validation only; outer validation never consumed",
        "runtime": {"device": str(device), "torch_version": torch.__version__},
        "source_script": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "selected_variants": [dict(variant) for variant in SELECTED_VARIANTS],
        "seeds": seed_results,
        "overall_decision": None,
    }

    for seed in seeds:
        print(f"##### seed {seed} #####", flush=True)
        record_path = resolve(RECORD_TEMPLATE.format(seed=seed))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (int(record["fold"]), int(record["seed"])) != (0, seed):
            raise ValueError(f"record mismatch: {record_path}")
        dataset_provenance = verify_dataset(data_dir, [record])
        baseline = load_existing_baseline(baseline_path, f"fold0_seed{seed}")
        set_seed(seed)
        bundle = create_protocol_dataloaders(
            str(data_dir),
            fold=0,
            seed=seed,
            batch_size=args.batch_size,
            image_size=224,
            inner_val_fraction=0.15,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_indices = list(bundle["split"].train_indices)
        print("=== cache deterministic tensor views ===", flush=True)
        cached_train, train_cache_meta = cache_dataset(
            Subset(bundle["eval_dataset"], train_indices),
            batch_size=max(args.batch_size, 128),
            num_workers=args.num_workers,
        )
        cached_inner, inner_cache_meta = cache_dataset(
            bundle["inner_val_loader"].dataset,
            batch_size=max(args.batch_size, 128),
            num_workers=args.num_workers,
        )
        clean_train_loader = create_dataloader(
            cached_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            seed=seed + 1,
        )
        inner_loader = create_dataloader(
            cached_inner,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            seed=seed + 2,
        )
        variants = {}
        for variant in SELECTED_VARIANTS:
            print(f"=== seed {seed}: {variant['name']} ===", flush=True)
            variants[variant["name"]] = train_variant(
                variant=dict(variant),
                record=record,
                record_dir=record_path.parent,
                train_subset=cached_train,
                clean_train_loader=clean_train_loader,
                inner_loader=inner_loader,
                class_counts=bundle["train_class_counts"],
                num_classes=bundle["num_classes"],
                device=device,
                epochs=args.epochs,
                eval_every=args.eval_every,
                batch_size=args.batch_size,
                num_workers=0,
                seed=seed,
            )
        seed_result = {
            "source_record": relative(record_path),
            "source_record_sha256": sha256(record_path),
            "dataset_provenance": dataset_provenance,
            "split": {
                "fold": 0,
                "seed": seed,
                "train_count": len(train_indices),
                "training_source": "eval_dataset indexed only by split.train_indices",
                "outer_validation_consumed": False,
            },
            "tensor_cache": {"train": train_cache_meta, "inner": inner_cache_meta},
            "baseline": baseline,
            "variants": variants,
        }
        seed_result["decision"] = seed_decision(variants, baseline)
        seed_results[str(seed)] = seed_result
        payload["seeds"] = seed_results
        payload["status"] = "IN_PROGRESS"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload["overall_decision"] = overall_decision(seed_results)
    payload["status"] = "COMPLETE"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["overall_decision"], indent=2, ensure_ascii=False), flush=True)
    print(f"written: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
