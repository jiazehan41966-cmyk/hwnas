#!/usr/bin/env python3
"""Analyze paired Protocol V2 Stage4 MBConv-k5-e3 accuracy gates."""

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
    adjusted: dict[str, float] = {}
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


def compare_background(
    *,
    background: str,
    index: int,
    base8_dir: Path,
    extended_dir: Path,
    threshold: Mapping[str, object],
) -> tuple[dict, float, dict[int, list[float]]]:
    control_id = f"fourstage_s2_{background}_s4_mbconv_k3_e3"
    k5_id = f"fourstage_s2_{background}_s4_mbconv_k5_e3"
    control = load_units(base8_dir / control_id)
    k5 = load_units(extended_dir / k5_id)
    fold_deltas: dict[int, list[float]] = defaultdict(list)
    flat: list[float] = []
    minority_deltas: list[float] = []
    pooled_support = np.sum(
        [
            np.asarray(row["outer_confusion_matrix"], dtype=int).sum(axis=1)
            for row in control.values()
        ],
        axis=0,
    )
    minority_indices = np.argsort(pooled_support)[:4].tolist()
    for fold, seed in sorted(control):
        delta = float(k5[(fold, seed)]["outer_val"]["macro_f1"]) - float(
            control[(fold, seed)]["outer_val"]["macro_f1"]
        )
        fold_deltas[fold].append(delta)
        flat.append(delta)
        minority_deltas.append(
            float(
                np.mean(
                    recall(k5[(fold, seed)]["outer_confusion_matrix"])[
                        minority_indices
                    ]
                    - recall(
                        control[(fold, seed)]["outer_confusion_matrix"]
                    )[minority_indices]
                )
            )
        )
    bootstrap = paired_hierarchical_bootstrap(
        fold_deltas,
        iterations=20_000,
        seed=20260821 + index,
    )
    minority_delta = float(np.mean(minority_deltas))
    pvalue = sign_flip_pvalue(flat)
    components = {
        "mean_delta_at_least_preregistered": (
            bootstrap["mean_delta"]
            >= float(threshold["macro_f1_mean_delta_min"])
        ),
        "ci95_low_above_zero": (
            bootstrap["ci95_low"]
            > float(threshold["paired_hierarchical_bootstrap_ci95_low_gt"])
        ),
        "positive_fold_means_at_least_preregistered": (
            bootstrap["positive_fold_means"]
            >= int(threshold["positive_fold_means_at_least"])
        ),
        "minority_recall_within_boundary": (
            minority_delta >= float(threshold["minority_recall_mean_delta_min"])
        ),
    }
    comparison = {
        "control_architecture": control_id,
        "k5_architecture": k5_id,
        "paired_units": 15,
        "macro_f1": bootstrap,
        "positive_units": sum(value > 0 for value in flat),
        "negative_units": sum(value < 0 for value in flat),
        "exact_paired_sign_flip_p": pvalue,
        "minority_class_indices": minority_indices,
        "minority_recall_delta_mean": minority_delta,
        "gate_components": components,
        "passes_stage4_k5_accuracy_gate": all(components.values()),
        "source": {
            "control_summary": {
                "path": str(base8_dir / control_id / "protocol_summary.json"),
                "sha256": sha256_file(
                    base8_dir / control_id / "protocol_summary.json"
                ),
            },
            "k5_summary": {
                "path": str(
                    extended_dir / k5_id / "protocol_summary.json"
                ),
                "sha256": sha256_file(
                    extended_dir / k5_id / "protocol_summary.json"
                ),
            },
        },
    }
    return comparison, pvalue, fold_deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base8-dir",
        default="results/sonar_fourstage_operator_v2/base8_formal",
    )
    parser.add_argument(
        "--extended-dir",
        default="results/sonar_fourstage_operator_v2/extended_formal",
    )
    parser.add_argument(
        "--preregistration",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_formal_preregistration.json"
        ),
    )
    parser.add_argument(
        "--base8-analysis",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "base8_formal_analysis.json"
        ),
    )
    parser.add_argument(
        "--stage4-k5-hardware-gate",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_exact_shape_hardware_gate.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_formal_analysis.json"
        ),
    )
    args = parser.parse_args()

    base8_dir = Path(args.base8_dir).resolve()
    extended_dir = Path(args.extended_dir).resolve()
    prereg_path = Path(args.preregistration).resolve()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    threshold = prereg["primary_gate_per_stage2_background"]
    comparisons: dict[str, dict] = {}
    raw_pvalues: dict[str, float] = {}
    aggregate_deltas: dict[int, list[float]] = defaultdict(list)
    for index, background in enumerate(STAGE2_BACKGROUNDS):
        comparison, pvalue, fold_deltas = compare_background(
            background=background,
            index=index,
            base8_dir=base8_dir,
            extended_dir=extended_dir,
            threshold=threshold,
        )
        comparisons[background] = comparison
        raw_pvalues[background] = pvalue
        for fold, values in fold_deltas.items():
            aggregate_deltas[fold].extend(values)
    adjusted = holm_adjust(raw_pvalues)
    for background, value in adjusted.items():
        comparisons[background][
            "holm_adjusted_p_across_backgrounds"
        ] = value

    any_stage4_pass = any(
        row["passes_stage4_k5_accuracy_gate"]
        for row in comparisons.values()
    )
    base8_path = Path(args.base8_analysis).resolve()
    base8_analysis = json.loads(base8_path.read_text(encoding="utf-8"))
    stage2_effect = base8_analysis["factorial_effects"]["kernel_K5_minus_K3"]
    stage2_ready = (
        float(stage2_effect["mean_delta"]) >= 0.01
        and float(stage2_effect["ci95_low"]) > 0.0
        and int(stage2_effect["positive_fold_means"]) >= 4
    )
    hardware_gate_path = Path(args.stage4_k5_hardware_gate).resolve()
    hardware_gate = json.loads(hardware_gate_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "GENERAL_OP_SELECTED"
            if stage2_ready or any_stage4_pass
            else "NO_OPERATOR_GAIN"
        ),
        "formal_k5_units": 60,
        "formal_k3_control_units_reused": 60,
        "stage4_k5_state": (
            "READY_ACCURACY_SUPPORTED"
            if any_stage4_pass
            else "NOT_ADMITTED_NO_STAGE4_GAIN"
        ),
        "comparisons": comparisons,
        "aggregate_stage4_k5_minus_k3": {
            "method": (
                "descriptive_mean_across_4_backgrounds_x_5_folds_x_3_seeds"
            ),
            "mean_delta": float(
                np.mean(
                    [
                        value
                        for values in aggregate_deltas.values()
                        for value in values
                    ]
                )
            ),
            "per_fold_mean_delta": {
                str(fold): float(np.mean(values))
                for fold, values in sorted(aggregate_deltas.items())
            },
            "claim_boundary": (
                "No aggregate CI is used as a gate; decisions are paired "
                "within each Stage2 background."
            ),
        },
        "pre_existing_stage2_k5_factor_evidence": {
            **stage2_effect,
            "meets_ready_rule": stage2_ready,
            "source": {
                "path": str(base8_path),
                "sha256": sha256_file(base8_path),
            },
        },
        "preregistration": {
            "path": str(prereg_path),
            "sha256": sha256_file(prereg_path),
        },
        "dir_state": "NOT_ADMITTED_ACCURACY_GATE_FAILED",
        "hardware_state": {
            "stage4_k5_exact_shape_micro_gate": hardware_gate["status"],
            "stage4_k5_exact_shape_micro_gate_source": {
                "path": str(hardware_gate_path),
                "sha256": sha256_file(hardware_gate_path),
            },
            "complete_network_route": "NOT_RUN",
            "bitstream": "NOT_GENERATED",
            "board_measurement": "NOT_RUN",
            "power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "GENERAL_OP_SELECTED selects the mature K5 family from formal "
            "accuracy evidence. Stage2 and Stage4 placement states remain "
            "separate. This is not ADMITTED Dir evidence or a complete-network "
            "AV7K325 route, bitstream, board, or power claim."
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
