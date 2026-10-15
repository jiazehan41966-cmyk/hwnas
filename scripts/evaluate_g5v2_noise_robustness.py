#!/usr/bin/env python3
"""Evaluate trained G5 v2 classifiers under deterministic synthetic speckle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.data import NKSIDDataset, create_dataloader
from hwnas_fpga.deploy.inference import load_checkpoint_model
from hwnas_fpga.training.robustness import evaluate_sonar_robustness
from hwnas_fpga.training.trainer import evaluate_classifier


DEFAULT_RUNS = {
    "control": PROJECT_ROOT / "results" / "protocol" / "g5v2_mbconv_control_v2",
    "adaptive_denoise_v2": PROJECT_ROOT / "results" / "protocol" / "g5v2_adaptive_denoise",
    "edge_v2": PROJECT_ROOT / "results" / "protocol" / "g5v2_edge_v2",
    "adaptive_denoise_edge_v2": (
        PROJECT_ROOT / "results" / "protocol" / "g5v2_adaptive_denoise_edge_v2"
    ),
}
CHECKPOINT_RE = re.compile(r"best_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--run", action="append", default=[], help="Run in name=path form")
    parser.add_argument("--variances", default="0.01,0.04,0.09")
    parser.add_argument("--noise-seed", type=int, default=314159)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--clean-macro-tolerance", type=float, default=0.02)
    parser.add_argument("--clean-top1-tolerance", type=float, default=0.0025)
    parser.add_argument(
        "--output-dir",
        default="results/g5v2_noise_robustness_20260815",
    )
    return parser.parse_args()


def parse_named_paths(items: list[str]) -> dict[str, Path]:
    if not items:
        return {name: path.resolve() for name, path in DEFAULT_RUNS.items()}
    runs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--run must use name=path")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or name in runs:
            raise ValueError("run names must be non-empty and unique")
        runs[name] = Path(raw_path.strip()).resolve()
    return runs


def parse_variances(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("variances must contain positive finite values")
    return values


def checkpoint_map(root: Path) -> dict[tuple[int, int], Path]:
    mapping: dict[tuple[int, int], Path] = {}
    for path in sorted(root.glob("best_fold*_seed*.pt")):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        key = (int(match.group("fold")), int(match.group("seed")))
        mapping[key] = path.resolve()
    if not mapping:
        raise FileNotFoundError(f"no checkpoints found under: {root}")
    return mapping


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_roots = parse_named_paths(args.run)
    variances = parse_variances(args.variances)
    mappings = {name: checkpoint_map(root) for name, root in run_roots.items()}
    pair_sets = [set(mapping) for mapping in mappings.values()]
    if any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
        raise ValueError("all runs must contain the same fold/seed pairs")
    pairs = sorted(pair_sets[0])
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    conditions = [
        {
            "name": f"speckle_var_{str(variance).replace('.', 'p')}",
            "kind": "speckle",
            "variance": variance,
        }
        for variance in variances
    ]
    config = {
        "enabled": True,
        "protocol": "g5v2_outer_fold_speckle_v1",
        "seed": int(args.noise_seed),
        "normalization_mean": [0.5],
        "normalization_std": [0.5],
        "conditions": conditions,
    }

    per_run: list[dict[str, Any]] = []
    clean_checks: list[dict[str, Any]] = []
    for fold, seed in pairs:
        dataset = NKSIDDataset(
            data_dir=args.data_dir,
            image_size=224,
            is_training=False,
            fold=fold,
            use_kfold=True,
            split="val",
            output_channels=1,
            image_error_policy="raise",
            cache_decoded_images=True,
            cache_geometry_images=True,
        )
        loader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
            seed=seed + fold * 100_000 + 2,
        )
        for method, mapping in mappings.items():
            checkpoint = mapping[(fold, seed)]
            model, _, payload, _ = load_checkpoint_model(checkpoint, device=args.device)
            clean = evaluate_classifier(
                model,
                loader,
                criterion=nn.CrossEntropyLoss().to(args.device),
                device=args.device,
                num_classes=8,
                topk=5,
            )
            expected = payload["metrics"]
            expected_macro_f1 = float(expected["macro_f1"])
            expected_top1 = float(expected["top1"])
            clean_macro_error = abs(float(clean["macro_f1"]) - expected_macro_f1)
            clean_top1_error = abs(float(clean["top1"]) - expected_top1)
            clean_checks.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "macro_f1_error": clean_macro_error,
                    "top1_error": clean_top1_error,
                }
            )
            if (
                clean_macro_error > args.clean_macro_tolerance
                or clean_top1_error > args.clean_top1_tolerance
            ):
                raise ValueError(
                    f"clean checkpoint reproduction failed for {method} fold{fold} seed{seed}: "
                    f"macro error={clean_macro_error}, top1 error={clean_top1_error}"
                )
            per_run.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "condition": "clean",
                    "variance": 0.0,
                    "macro_f1": float(clean["macro_f1"]),
                    "top1": float(clean["top1"]),
                    "checkpoint": str(checkpoint),
                }
            )
            robustness = evaluate_sonar_robustness(
                model,
                loader,
                device=args.device,
                num_classes=8,
                class_weights=None,
                config=config,
                topk=5,
            )
            for condition, result in zip(conditions, robustness["condition_results"]):
                per_run.append(
                    {
                        "method": method,
                        "fold": fold,
                        "seed": seed,
                        "condition": condition["name"],
                        "variance": float(condition["variance"]),
                        "macro_f1": float(result["metrics"]["macro_f1"]),
                        "top1": float(result["metrics"]["top1"]),
                        "checkpoint": str(checkpoint),
                    }
                )
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[(row["method"], row["condition"], float(row["variance"]))].append(row)
    summary: list[dict[str, Any]] = []
    for method in run_roots:
        clean_rows = grouped[(method, "clean", 0.0)]
        clean_macro, _ = _mean_std(float(row["macro_f1"]) for row in clean_rows)
        clean_top1, _ = _mean_std(float(row["top1"]) for row in clean_rows)
        ordered_conditions = [("clean", 0.0)] + [
            (condition["name"], float(condition["variance"])) for condition in conditions
        ]
        for condition_name, variance in ordered_conditions:
            rows = grouped[(method, condition_name, variance)]
            macro_mean, macro_std = _mean_std(float(row["macro_f1"]) for row in rows)
            top1_mean, top1_std = _mean_std(float(row["top1"]) for row in rows)
            summary.append(
                {
                    "method": method,
                    "condition": condition_name,
                    "variance": variance,
                    "run_count": len(rows),
                    "macro_f1_mean": macro_mean,
                    "macro_f1_std": macro_std,
                    "top1_mean": top1_mean,
                    "top1_std": top1_std,
                    "macro_f1_drop_vs_clean": clean_macro - macro_mean,
                    "top1_drop_vs_clean": clean_top1 - top1_mean,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_run.csv",
        per_run,
        ["method", "fold", "seed", "condition", "variance", "macro_f1", "top1", "checkpoint"],
    )
    write_csv(
        output_dir / "summary.csv",
        summary,
        [
            "method",
            "condition",
            "variance",
            "run_count",
            "macro_f1_mean",
            "macro_f1_std",
            "top1_mean",
            "top1_std",
            "macro_f1_drop_vs_clean",
            "top1_drop_vs_clean",
        ],
    )
    payload = {
        "schema_version": 1,
        "protocol": "g5v2_outer_fold_speckle_v1",
        "dataset": "NKSID",
        "data_dir": str(Path(args.data_dir).resolve()),
        "fold_seed_pairs": [{"fold": fold, "seed": seed} for fold, seed in pairs],
        "conditions": conditions,
        "noise_seed": int(args.noise_seed),
        "clean_reproduction_macro_f1_tolerance": float(args.clean_macro_tolerance),
        "clean_reproduction_top1_tolerance": float(args.clean_top1_tolerance),
        "clean_reproduction_max_macro_f1_error": max(row["macro_f1_error"] for row in clean_checks),
        "clean_reproduction_max_top1_error": max(row["top1_error"] for row in clean_checks),
        "summary": summary,
        "claim_boundary": (
            "Full-network classification robustness on NKSID only. Roboflow and Figshare require "
            "separately trained task-compatible classifiers before the same test is valid."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(Path(args.output_dir).resolve()),
                "pairs": len(payload["fold_seed_pairs"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
