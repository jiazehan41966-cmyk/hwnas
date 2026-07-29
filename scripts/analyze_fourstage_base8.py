#!/usr/bin/env python3
"""Analyze the complete Protocol V2 base8 factorial with paired statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import (  # noqa: E402
    canonical_sha256,
    paired_hierarchical_bootstrap,
)
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def sign_flip_pvalue(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    extreme = 0
    total = 1 << array.size
    for mask in range(total):
        signs = np.fromiter(
            (1.0 if mask & (1 << index) else -1.0 for index in range(array.size)),
            dtype=float,
            count=array.size,
        )
        if abs(float(np.mean(array * signs))) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def confusion_recall(matrix: list[list[int]]) -> list[float]:
    values = np.asarray(matrix, dtype=float)
    supports = values.sum(axis=1)
    return [
        float(values[index, index] / supports[index]) if supports[index] else 0.0
        for index in range(values.shape[0])
    ]


def load_runs(root: Path, manifest: Mapping[str, Any]):
    results: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    factors: dict[str, dict[str, str]] = {}
    summary_refs: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        arch_id = str(row["arch_id"])
        run_dir = root / arch_id
        summary_path = run_dir / "protocol_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary["claimability"]["claimable"]:
            raise ValueError(f"{arch_id} is not a complete claimable 15-run protocol")
        records = {}
        for path in sorted(run_dir.glob("run_fold*_seed*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records[(int(record["fold"]), int(record["seed"]))] = record
        expected = {(fold, seed) for fold in range(5) for seed in (42, 43, 44)}
        if set(records) != expected:
            raise ValueError(f"{arch_id} does not contain the exact 15 formal units")
        results[arch_id] = records
        factors[arch_id] = {key: str(value) for key, value in row["factors"].items()}
        summary_refs[arch_id] = {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
            "run_fingerprint": summary["run_fingerprint"],
        }
    return results, factors, summary_refs


def paired_report(
    left: Mapping[tuple[int, int], dict[str, Any]],
    right: Mapping[tuple[int, int], dict[str, Any]],
    *,
    metric: str,
    seed: int,
) -> tuple[dict[str, Any], list[float]]:
    deltas: dict[int, list[float]] = defaultdict(list)
    flat = []
    for fold, unit_seed in sorted(left):
        delta = (
            float(right[(fold, unit_seed)]["outer_val"][metric])
            - float(left[(fold, unit_seed)]["outer_val"][metric])
        )
        deltas[fold].append(delta)
        flat.append(delta)
    report = paired_hierarchical_bootstrap(deltas, iterations=20_000, seed=seed)
    report["direction"] = "right_minus_left"
    report["metric"] = metric
    report["positive_units"] = sum(value > 0 for value in flat)
    report["negative_units"] = sum(value < 0 for value in flat)
    report["zero_units"] = sum(value == 0 for value in flat)
    report["exact_paired_sign_flip_p"] = sign_flip_pvalue(flat)
    return report, flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/sonar_fourstage_operator_v2/base8_formal",
    )
    parser.add_argument(
        "--candidate-manifest",
        default=(
            "artifacts/sonar_fourstage_operator_v2/base8_candidates/"
            "base8_manifest.json"
        ),
    )
    parser.add_argument(
        "--hardware-audit",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "base8_strict_lut_proxy_audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/sonar_fourstage_operator_v2/base8_formal_analysis.json",
    )
    args = parser.parse_args()
    result_root = Path(args.results_dir).resolve()
    manifest_path = Path(args.candidate_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results, factors, summary_refs = load_runs(result_root, manifest)

    architecture_summaries = {}
    for arch_id, records in results.items():
        macro = [float(row["outer_val"]["macro_f1"]) for row in records.values()]
        top1 = [float(row["outer_val"]["top1"]) for row in records.values()]
        recalls = [
            confusion_recall(row["outer_confusion_matrix"])
            for row in records.values()
        ]
        supports = np.sum(
            [
                np.asarray(row["outer_confusion_matrix"], dtype=float).sum(axis=1)
                for row in records.values()
            ],
            axis=0,
        )
        minority_indices = np.argsort(supports)[:4].tolist()
        architecture_summaries[arch_id] = {
            "factors": factors[arch_id],
            "macro_f1": {
                "mean": statistics.fmean(macro),
                "std": statistics.stdev(macro),
                "n": len(macro),
            },
            "top1": {
                "mean": statistics.fmean(top1),
                "std": statistics.stdev(top1),
                "n": len(top1),
            },
            "per_class_recall_mean": np.mean(recalls, axis=0).tolist(),
            "minority_class_indices_by_outer_support": minority_indices,
            "minority_recall_mean": float(
                np.mean(np.asarray(recalls)[:, minority_indices])
            ),
            "fold_mean_macro_f1": {
                str(fold): statistics.fmean(
                    float(records[(fold, seed)]["outer_val"]["macro_f1"])
                    for seed in (42, 43, 44)
                )
                for fold in range(5)
            },
        }

    effect_codes = {
        "kernel_K5_minus_K3": lambda row: 1 if row["kernel"] == "K5" else -1,
        "expansion_e6_minus_e3": lambda row: 1 if row["expansion"] == "e6" else -1,
        "stage4_MBConv_minus_Skip": lambda row: (
            1 if row["stage4"] == "MBConv" else -1
        ),
    }
    main_and_interaction_effects = {}
    base_effect_names = list(effect_codes)
    for order in (1, 2, 3):
        for names in itertools.combinations(base_effect_names, order):
            effect_name = "__x__".join(names)
            deltas: dict[int, list[float]] = defaultdict(list)
            flat = []
            for fold in range(5):
                for seed in (42, 43, 44):
                    positive = []
                    negative = []
                    for arch_id, records in results.items():
                        sign = int(
                            np.prod(
                                [effect_codes[name](factors[arch_id]) for name in names]
                            )
                        )
                        target = positive if sign > 0 else negative
                        target.append(
                            float(records[(fold, seed)]["outer_val"]["macro_f1"])
                        )
                    delta = statistics.fmean(positive) - statistics.fmean(negative)
                    deltas[fold].append(delta)
                    flat.append(delta)
            report = paired_hierarchical_bootstrap(
                deltas, iterations=20_000, seed=20260729 + len(main_and_interaction_effects)
            )
            report["effect_definition"] = "mean(level_product=+1)-mean(level_product=-1)"
            report["exact_paired_sign_flip_p"] = sign_flip_pvalue(flat)
            main_and_interaction_effects[effect_name] = report

    reference_id = "fourstage_s2_k3_e3_s4_mbconv_k3_e3"
    comparisons = {}
    raw_pvalues = {}
    for index, arch_id in enumerate(sorted(results)):
        if arch_id == reference_id:
            continue
        report, _ = paired_report(
            results[reference_id],
            results[arch_id],
            metric="macro_f1",
            seed=20260800 + index,
        )
        comparisons[f"{arch_id}_minus_{reference_id}"] = report
        raw_pvalues[arch_id] = report["exact_paired_sign_flip_p"]
    adjusted = holm_adjust(raw_pvalues)
    for arch_id, value in adjusted.items():
        comparisons[f"{arch_id}_minus_{reference_id}"][
            "holm_adjusted_p_across_7_reference_comparisons"
        ] = value

    hardware_path = Path(args.hardware_audit).resolve()
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FULL_8x5x3",
        "formal_unit_count": 8 * 5 * 3,
        "design": "2x2x2_full_factorial",
        "architecture_summaries": architecture_summaries,
        "factorial_effects": main_and_interaction_effects,
        "paired_reference_comparisons": comparisons,
        "best_mean_macro_f1_architecture": max(
            architecture_summaries,
            key=lambda key: architecture_summaries[key]["macro_f1"]["mean"],
        ),
        "hardware_evidence": {
            "path": str(hardware_path),
            "sha256": sha256_file(hardware_path),
            "evidence_layer": hardware["evidence_layer"],
            "full_network_route_completed": False,
            "power_status": "NOT_MEASURED",
        },
        "source_summaries": summary_refs,
        "candidate_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "claim_boundary": (
            "Accuracy effects are from the complete frozen 8x5-foldx3-seed "
            "Protocol V2 factorial. Strict-LUT cost is a separate proxy layer; "
            "it is not full-network route, COM5, bitstream, or measured power."
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
