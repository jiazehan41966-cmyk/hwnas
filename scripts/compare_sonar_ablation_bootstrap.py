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
) -> dict[str, Any]:
    by_class: dict[int, list[Mapping[str, int]]] = {}
    for row in pairs:
        by_class.setdefault(int(row["target"]), []).append(row)
    if not by_class:
        raise ValueError("No target classes found in paired rows")

    point_control = macro_f1(pairs, "control_prediction", num_classes=num_classes)
    point_variant = macro_f1(pairs, "variant_prediction", num_classes=num_classes)
    point_delta = point_variant - point_control

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(max(1, int(iterations))):
        sample: list[Mapping[str, int]] = []
        for rows in by_class.values():
            sample.extend(rng.choice(rows) for _ in range(len(rows)))
        deltas.append(
            macro_f1(sample, "variant_prediction", num_classes=num_classes)
            - macro_f1(sample, "control_prediction", num_classes=num_classes)
        )

    le_zero = (sum(1 for delta in deltas if delta <= 0.0) + 1) / (len(deltas) + 1)
    ge_zero = (sum(1 for delta in deltas if delta >= 0.0) + 1) / (len(deltas) + 1)
    p_value = min(1.0, 2.0 * min(le_zero, ge_zero))
    return {
        "method": "paired_stratified_bootstrap",
        "metric": "macro_f1",
        "iterations": max(1, int(iterations)),
        "bootstrap_seed": int(seed),
        "paired_prediction_count": len(pairs),
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
        )

    adjusted = holm_adjust([comparisons[name]["p_value"] for name in VARIANTS])
    for name, adjusted_p in zip(VARIANTS, adjusted):
        comparisons[name]["holm_adjusted_p_value"] = adjusted_p
        comparisons[name]["holm_significant"] = adjusted_p < 0.05
        comparisons[name]["actual_gain"] = comparisons[name]["macro_f1_mean_delta"] > 0.0

    return {
        "schema_version": 1,
        "generated_by": "scripts/compare_sonar_ablation_bootstrap.py",
        "num_classes": num_classes,
        "control_run": str(run_dirs["mbconv_control"]),
        "variant_runs": {name: str(run_dirs[name]) for name in VARIANTS},
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
