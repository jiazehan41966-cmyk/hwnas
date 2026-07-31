#!/usr/bin/env python3
"""Analyze paired Protocol V2 Dir-v1 versus MBConv-k3-e3 accuracy gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import (  # noqa: E402
    canonical_sha256,
    paired_hierarchical_bootstrap,
)
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


STAGE2_BACKGROUNDS = ("k3_e3", "k3_e6", "k5_e3", "k5_e6")


def sign_flip_pvalue(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    observed = abs(float(values_array.mean()))
    extreme = 0
    for mask in range(1 << len(values)):
        signs = np.asarray(
            [
                1.0 if mask & (1 << index) else -1.0
                for index in range(len(values))
            ]
        )
        extreme += (
            abs(float(np.mean(values_array * signs))) >= observed - 1e-15
        )
    return extreme / float(1 << len(values))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(
            running, min(1.0, (len(ordered) - rank) * pvalues[key])
        )
        adjusted[key] = running
    return adjusted


def load_units(path: Path) -> dict[tuple[int, int], dict]:
    summary_path = path / "protocol_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["claimability"]["claimable"]:
        raise ValueError(f"{path.name} is not claimable")
    units = {}
    for record_path in path.glob("run_fold*_seed*.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        units[(int(record["fold"]), int(record["seed"]))] = record
    expected = {(fold, seed) for fold in range(5) for seed in (42, 43, 44)}
    if set(units) != expected:
        raise ValueError(f"{path.name} must contain exactly 15 formal units")
    return units


def recall(matrix: list[list[int]]) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    support = values.sum(axis=1)
    return np.divide(
        np.diag(values),
        support,
        out=np.zeros_like(support),
        where=support > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base8-dir",
        default="results/sonar_fourstage_operator_v2/base8_formal",
    )
    parser.add_argument(
        "--dir-results",
        default="results/sonar_fourstage_operator_v2/extended_formal",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sonar_fourstage_operator_v2/dir_accuracy_gate.json",
    )
    args = parser.parse_args()
    base8 = Path(args.base8_dir).resolve()
    dir_root = Path(args.dir_results).resolve()
    comparisons = {}
    raw_pvalues = {}
    all_deltas: dict[int, list[float]] = defaultdict(list)
    for index, background in enumerate(STAGE2_BACKGROUNDS):
        control_id = f"fourstage_s2_{background}_s4_mbconv_k3_e3"
        dir_id = (
            f"fourstage_s2_{background}_"
            "s4_dir_mbconv3_split11_e3_v1"
        )
        control = load_units(base8 / control_id)
        candidate = load_units(dir_root / dir_id)
        fold_deltas: dict[int, list[float]] = defaultdict(list)
        flat = []
        minority_deltas = []
        pooled_support = np.sum(
            [
                np.asarray(row["outer_confusion_matrix"], dtype=int).sum(axis=1)
                for row in control.values()
            ],
            axis=0,
        )
        minority_indices = np.argsort(pooled_support)[:4].tolist()
        for fold, seed in sorted(control):
            delta = float(candidate[(fold, seed)]["outer_val"]["macro_f1"]) - float(
                control[(fold, seed)]["outer_val"]["macro_f1"]
            )
            fold_deltas[fold].append(delta)
            all_deltas[fold].append(delta)
            flat.append(delta)
            minority_deltas.append(
                float(
                    np.mean(
                        recall(
                            candidate[(fold, seed)]["outer_confusion_matrix"]
                        )[minority_indices]
                        - recall(
                            control[(fold, seed)]["outer_confusion_matrix"]
                        )[minority_indices]
                    )
                )
            )
        bootstrap = paired_hierarchical_bootstrap(
            fold_deltas,
            iterations=20_000,
            seed=20260820 + index,
        )
        pvalue = sign_flip_pvalue(flat)
        raw_pvalues[background] = pvalue
        components = {
            "mean_delta_at_least_0_01": bootstrap["mean_delta"] >= 0.01,
            "ci95_low_above_zero": bootstrap["ci95_low"] > 0.0,
            "positive_fold_means_at_least_4": (
                bootstrap["positive_fold_means"] >= 4
            ),
        }
        comparisons[background] = {
            "control_architecture": control_id,
            "dir_architecture": dir_id,
            "paired_units": 15,
            "macro_f1": bootstrap,
            "positive_units": sum(value > 0 for value in flat),
            "negative_units": sum(value < 0 for value in flat),
            "exact_paired_sign_flip_p": pvalue,
            "minority_class_indices": minority_indices,
            "minority_recall_delta_mean": float(np.mean(minority_deltas)),
            "gate_components": components,
            "passes_dir_vs_k3": all(components.values()),
            "source": {
                "control_summary": {
                    "path": str((base8 / control_id / "protocol_summary.json")),
                    "sha256": sha256_file(
                        base8 / control_id / "protocol_summary.json"
                    ),
                },
                "dir_summary": {
                    "path": str((dir_root / dir_id / "protocol_summary.json")),
                    "sha256": sha256_file(
                        dir_root / dir_id / "protocol_summary.json"
                    ),
                },
            },
        }
    adjusted = holm_adjust(raw_pvalues)
    for background, value in adjusted.items():
        comparisons[background]["holm_adjusted_p_across_backgrounds"] = value

    aggregate = {
        "method": "descriptive_mean_across_4_backgrounds_x_5_folds_x_3_seeds",
        "mean_delta": float(
            np.mean([value for values in all_deltas.values() for value in values])
        ),
        "per_fold_mean_delta": {
            str(fold): float(np.mean(values))
            for fold, values in sorted(all_deltas.items())
        },
        "claim_boundary": (
            "No aggregate CI is used as a gate; preregistered paired 15-run "
            "decisions are evaluated separately within each Stage2 background."
        ),
    }
    any_pass = any(row["passes_dir_vs_k3"] for row in comparisons.values())
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if any_pass else "DIR_ACCURACY_GATE_FAIL",
        "formal_dir_units": 60,
        "formal_control_units_reused": 60,
        "comparisons": comparisons,
        "aggregate_dir_minus_k3": aggregate,
        "preregistered_thresholds": {
            "mean_delta_min": 0.01,
            "bootstrap_ci95_low_gt": 0.0,
            "positive_fold_means_at_least": 4,
            "multiple_comparison_correction": "Holm",
        },
        "downstream_gate": {
            "dir_post_training_int8": (
                "PENDING" if any_pass else "NOT_RUN_ACCURACY_GATE_FAILED"
            ),
            "robustness": (
                "PENDING" if any_pass else "NOT_RUN_ACCURACY_GATE_FAILED"
            ),
            "rtl_cosim": (
                "PENDING" if any_pass else "NOT_RUN_ACCURACY_GATE_FAILED"
            ),
            "full_network_hls_route": (
                "PENDING" if any_pass else "NOT_RUN_ACCURACY_GATE_FAILED"
            ),
            "mechanism_control_training": (
                "PENDING" if any_pass else "NOT_RUN_ACCURACY_GATE_FAILED"
            ),
        },
        "stage4_k5_comparator": "PENDING_HARDWARE",
        "operator_state": (
            "CANDIDATE" if any_pass else "NOT_ADMITTED_ACCURACY_GATE_FAILED"
        ),
        "claim_boundary": (
            "All comparisons are paired by identical fold/seed under frozen "
            "Protocol V2. Dir-v1 failed admission if no Stage2 background "
            "passes all three preregistered K3 thresholds. No robustness or "
            "post-training hardware conclusion is inferred after that gate."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
