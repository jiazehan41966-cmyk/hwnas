#!/usr/bin/env python3
"""Compare G5 sonar ablations against the matched MBConv control.

The G5 gate requires paired, class-stratified bootstrap on outer-validation
per-sample predictions. This script intentionally refuses to reconstruct that
evidence from confusion matrices, because confusion matrices lose the pairing
needed for the test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.hardware.sonar_operator_gate import holm_adjust


VARIANTS = ("denoise", "edge", "denoise_edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-run",
        default="results/protocol/g5_ablation_mbconv_control",
        help="Protocol run directory for mbconv_control.",
    )
    parser.add_argument("--denoise-run", default="results/protocol/g5_ablation_denoise")
    parser.add_argument("--edge-run", default="results/protocol/g5_ablation_edge")
    parser.add_argument(
        "--denoise-edge-run",
        default="results/protocol/g5_ablation_denoise_edge",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--min-meaningful-delta",
        type=float,
        default=0.01,
        help="Minimum paired-run macro-F1 gain counted as an actual gain.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sonar_operator_gate/g5_ablation_bootstrap_comparisons.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary = run_dir / "protocol_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Missing protocol summary: {summary}")
    payload = load_json(summary)
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol summary must be a JSON object: {summary}")
    return payload


def load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("outer_predictions_fold*_seed*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"Missing outer prediction JSONL files in {run_dir}. "
            "Run run_eval_protocol.py after the prediction-recording update."
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("fold", "seed", "sample_index", "target", "prediction"):
                    if key not in row:
                        raise ValueError(f"{path}:{line_number} missing {key!r}")
                rows.append(row)
    return rows


def prediction_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["fold"]), int(row["seed"]), int(row["sample_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate prediction key: {key}")
        indexed[key] = row
    return indexed


def paired_rows(
    control: Sequence[Mapping[str, Any]],
    variant: Sequence[Mapping[str, Any]],
) -> list[dict[str, int]]:
    control_index = prediction_index(control)
    variant_index = prediction_index(variant)
    common = sorted(set(control_index) & set(variant_index))
    if not common:
        raise ValueError("No paired prediction rows share fold/seed/sample_index")

    pairs: list[dict[str, int]] = []
    for fold, seed, sample_index in common:
        left = control_index[(fold, seed, sample_index)]
        right = variant_index[(fold, seed, sample_index)]
        if int(left["target"]) != int(right["target"]):
            raise ValueError(
                "Target mismatch for paired prediction "
                f"fold={fold} seed={seed} sample_index={sample_index}"
            )
        pairs.append(
            {
                "fold": fold,
                "seed": seed,
                "sample_index": sample_index,
                "target": int(left["target"]),
                "control_prediction": int(left["prediction"]),
                "variant_prediction": int(right["prediction"]),
            }
        )
    return pairs


def macro_f1(rows: Sequence[Mapping[str, int]], prediction_key: str, *, num_classes: int) -> float:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for row in rows:
        target = int(row["target"])
        prediction = int(row[prediction_key])
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            confusion[target][prediction] += 1

    scores: list[float] = []
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes)) - tp
        fn = sum(confusion[class_index]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    return statistics.fmean(scores) if scores else 0.0


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_run_delta_std(pairs: Sequence[Mapping[str, int]], *, num_classes: int) -> float:
    grouped: dict[tuple[int, int], list[Mapping[str, int]]] = {}
    for row in pairs:
        grouped.setdefault((int(row["fold"]), int(row["seed"])), []).append(row)
    deltas = [
        macro_f1(rows, "variant_prediction", num_classes=num_classes)
        - macro_f1(rows, "control_prediction", num_classes=num_classes)
        for rows in grouped.values()
    ]
    return statistics.stdev(deltas) if len(deltas) > 1 else 0.0


def stratified_bootstrap(
    pairs: Sequence[Mapping[str, int]],
    *,
    num_classes: int,
    iterations: int,
    seed: int,
    min_meaningful_delta: float = 0.01,
) -> dict[str, Any]:
    by_class: dict[int, list[Mapping[str, int]]] = {}
    for row in pairs:
        by_class.setdefault(int(row["target"]), []).append(row)
    if not by_class:
        raise ValueError("No target classes found in paired rows")

    grouped: dict[tuple[int, int], list[Mapping[str, int]]] = {}
    for row in pairs:
        grouped.setdefault((int(row["fold"]), int(row["seed"])), []).append(row)
    run_metrics = [
        {
            "fold": fold,
            "seed": run_seed,
            "control": macro_f1(rows, "control_prediction", num_classes=num_classes),
            "variant": macro_f1(rows, "variant_prediction", num_classes=num_classes),
        }
        for (fold, run_seed), rows in sorted(grouped.items())
    ]
    point_control = statistics.fmean(row["control"] for row in run_metrics)
    point_variant = statistics.fmean(row["variant"] for row in run_metrics)
    point_delta = point_variant - point_control

    rng = random.Random(seed)
    deltas: list[float] = []
    by_fold: dict[int, list[Mapping[str, float]]] = {}
    for row in run_metrics:
        by_fold.setdefault(int(row["fold"]), []).append(row)
    folds = sorted(by_fold)
    for _ in range(max(1, int(iterations))):
        sampled_runs: list[Mapping[str, float]] = []
        for _fold_draw in range(len(folds)):
            fold = rng.choice(folds)
            rows = by_fold[fold]
            sampled_runs.extend(rng.choice(rows) for _ in range(len(rows)))
        deltas.append(
            statistics.fmean(row["variant"] - row["control"] for row in sampled_runs)
        )

    # Paired sign-flip null at the fold cluster level. Seeds within one outer
    # fold remain clustered instead of being treated as independent images.
    null_rng = random.Random(int(seed) + 1)
    fold_deltas = {
        fold: [float(row["variant"] - row["control"]) for row in rows]
        for fold, rows in by_fold.items()
    }
    null_means: list[float] = []
    for _ in range(max(1, int(iterations))):
        signed = []
        for fold in folds:
            sign = -1.0 if null_rng.random() < 0.5 else 1.0
            signed.extend(sign * value for value in fold_deltas[fold])
        null_means.append(statistics.fmean(signed))
    p_value = (
        sum(abs(value) >= abs(point_delta) for value in null_means) + 1
    ) / (len(null_means) + 1)
    return {
        "method": "paired_hierarchical_bootstrap_fold_sign_flip",
        "metric": "macro_f1",
        "iterations": max(1, int(iterations)),
        "bootstrap_seed": int(seed),
        "paired_prediction_count": len(pairs),
        "paired_run_count": len(run_metrics),
        "fold_count": len(folds),
        "folds": sorted({int(row["fold"]) for row in pairs}),
        "seeds": sorted({int(row["seed"]) for row in pairs}),
        "class_counts": {
            str(label): len(rows) for label, rows in sorted(by_class.items())
        },
        "macro_f1_control": point_control,
        "macro_f1_variant": point_variant,
        "macro_f1_mean_delta": point_delta,
        "paired_run_delta_std": paired_run_delta_std(pairs, num_classes=num_classes),
        "ci95_low": quantile(deltas, 0.025),
        "ci95_high": quantile(deltas, 0.975),
        "p_value": p_value,
        "p_value_method": "fold_cluster_paired_sign_flip",
        "min_meaningful_delta": float(min_meaningful_delta),
        "actual_gain": point_delta >= float(min_meaningful_delta),
    }


def variant_gate_row(summary: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    claimability = summary.get("claimability") or {}
    runs = summary.get("runs") or []
    return {
        "folds": list(summary.get("folds") or []),
        "seeds": list(summary.get("seeds") or []),
        "completed_runs": len(runs),
        "claimable": bool(claimability.get("claimable")),
        "outer_leakage": bool(claimability.get("outer_validation_used_for_selection")),
        "protocol_context_sha256": summary.get("protocol_context_sha256"),
        "group_split_available": bool(summary.get("group_split_available", False)),
        "group_generalization_claimable": bool(
            summary.get("group_generalization_claimable", False)
        ),
        "run_fingerprints": list(summary.get("run_fingerprints") or []),
        "protocol_summary": str((run_dir / "protocol_summary.json").resolve()),
    }


def infer_num_classes(*summaries: Mapping[str, Any], prediction_rows: Sequence[Mapping[str, Any]]) -> int:
    for summary in summaries:
        class_names = summary.get("class_names")
        if isinstance(class_names, list) and class_names:
            return len(class_names)
    labels = [int(row["target"]) for row in prediction_rows]
    return max(labels) + 1 if labels else 0


def compare_all(args: argparse.Namespace) -> dict[str, Any]:
    run_dirs = {
        "mbconv_control": Path(args.control_run).resolve(),
        "denoise": Path(args.denoise_run).resolve(),
        "edge": Path(args.edge_run).resolve(),
        "denoise_edge": Path(args.denoise_edge_run).resolve(),
    }
    summaries = {name: load_summary(path) for name, path in run_dirs.items()}
    contexts = {
        str(summary.get("protocol_context_sha256", ""))
        for summary in summaries.values()
    }
    if len(contexts) != 1 or not next(iter(contexts)):
        raise ValueError(
            "G5 variants must share one protocol_context_sha256; "
            f"observed={sorted(contexts)}"
        )
    for name, summary in summaries.items():
        claimability = summary.get("claimability") or {}
        if claimability.get("claimable") is not True:
            raise ValueError(f"G5 variant {name} is not claimable under the frozen protocol")
        run_keys = {
            (int(row["fold"]), int(row["seed"]))
            for row in summary.get("runs", [])
        }
        if run_keys != {
            (fold, seed) for fold in range(5) for seed in (42, 43, 44)
        }:
            raise ValueError(f"G5 variant {name} does not contain the exact 15 fold/seed pairs")
    predictions = {name: load_prediction_rows(path) for name, path in run_dirs.items()}
    num_classes = infer_num_classes(
        *summaries.values(),
        prediction_rows=predictions["mbconv_control"],
    )

    comparisons: dict[str, Any] = {}
    for offset, variant in enumerate(VARIANTS):
        pairs = paired_rows(predictions["mbconv_control"], predictions[variant])
        comparisons[variant] = stratified_bootstrap(
            pairs,
            num_classes=num_classes,
            iterations=args.iterations,
            seed=int(args.seed) + offset,
            min_meaningful_delta=float(getattr(args, "min_meaningful_delta", 0.01)),
        )

    adjusted = holm_adjust([comparisons[name]["p_value"] for name in VARIANTS])
    for name, adjusted_p in zip(VARIANTS, adjusted):
        comparisons[name]["holm_adjusted_p_value"] = adjusted_p
        comparisons[name]["holm_significant"] = adjusted_p < 0.05
        comparisons[name]["actual_gain"] = (
            comparisons[name]["macro_f1_mean_delta"]
            >= comparisons[name]["min_meaningful_delta"]
        )

    return {
        "schema_version": 2,
        "generated_by": "scripts/compare_sonar_ablation_bootstrap.py",
        "num_classes": num_classes,
        "control_run": str(run_dirs["mbconv_control"]),
        "variant_runs": {name: str(run_dirs[name]) for name in VARIANTS},
        "required_folds": [0, 1, 2, 3, 4],
        "required_seeds": [42, 43, 44],
        "paired_run_count": 15,
        "min_meaningful_delta": float(getattr(args, "min_meaningful_delta", 0.01)),
        "ablation_variants": {
            name: variant_gate_row(summaries[name], run_dirs[name])
            for name in run_dirs
        },
        "comparisons_vs_control": comparisons,
        "boundary": (
            "This file is a manifest fragment for G5. It does not establish "
            "HLS parity, route feasibility, or board evidence."
        ),
    }


def main() -> int:
    args = parse_args()
    payload = compare_all(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
