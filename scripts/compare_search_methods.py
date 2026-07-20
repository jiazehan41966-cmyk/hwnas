#!/usr/bin/env python3
"""Compare completed search runs at equal protocol and compute budgets.

This is evidence packaging only.  It never launches training.  Search proxy,
retrain, route/COM5, board accuracy, and measured power remain separate layers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate
from hwnas_fpga.benchmarks.metrics import (
    ObjectiveSpec,
    exact_hypervolume,
    normalize_objective_rows,
)
from hwnas_fpga.search.pareto import compute_pareto_front, is_dominated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Completed run root; repeat for each method.")
    parser.add_argument("--output-dir", default="results/search_method_comparison")
    parser.add_argument(
        "--latency-limit-ms",
        type=float,
        default=None,
        help=(
            "frozen latency constraint used in exact normalized HV; required for the "
            "formal three-objective benchmark"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_run_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "results" / "summary.json").exists():
        return candidate
    if candidate.name == "results" and (candidate / "summary.json").exists():
        return candidate.parent
    raise FileNotFoundError(f"Not a completed run root: {candidate}")


def fairness_payload(config: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    search = dict(config.get("search") or {})
    extra = dict(summary.get("extra") or {})
    method = str(extra.get("search_method") or search.get("method") or "unknown")
    def first_not_none(*values: Any, default: Any = None) -> Any:
        return next((value for value in values if value is not None), default)

    evaluation_budget = int(
        first_not_none(
            extra.get("num_episodes") if method == "rl" else extra.get("num_candidates"),
            search.get("episodes") if method == "rl" else search.get("num_candidates"),
            default=0,
        )
    )
    return {
        "dataset": config.get("dataset"),
        "search_space": config.get("search_space"),
        "constraints": config.get("constraints"),
        "hardware": config.get("hardware"),
        "training_batch_size": (config.get("training") or {}).get("batch_size"),
        "selection_metric": search.get("selection_metric", "macro_f1"),
        "objective_weights": search.get("objective_weights"),
        "robustness": search.get("robustness"),
        "pareto_objectives": (search.get("pareto") or {}).get("objectives"),
        "pareto_directions": (search.get("pareto") or {}).get("directions"),
        "eval_epochs": int(
            first_not_none(extra.get("train_epochs"), search.get("eval_epochs"), default=0)
        ),
        "evaluation_budget": evaluation_budget,
    }


def candidate_signature(candidate: Mapping[str, Any]) -> str:
    return canonical_sha256(candidate.get("encoding") or {})


def summarize_run(run_root: Path) -> dict[str, Any]:
    summary = read_json(run_root / "results" / "summary.json")
    efficiency = read_json(run_root / "results" / "search_efficiency.json")
    job_efficiency_path = run_root / "results" / "job_efficiency.json"
    job_efficiency = (
        read_json(job_efficiency_path) if job_efficiency_path.exists() else None
    )
    pareto = read_json(run_root / "results" / "pareto_selection.json")
    candidates_path = run_root / "results" / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8-sig"))
    if not isinstance(candidates, list):
        raise ValueError(f"Expected candidate list: {candidates_path}")
    config = yaml.safe_load((run_root / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    protocol = fairness_payload(config, summary)
    seed = int((config.get("project") or {}).get("seed", 42))
    best = summary.get("best_candidate") or {}
    metrics = best.get("metrics") or {}
    candidate_metrics = [
        dict(candidate.get("metrics") or {})
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ]
    f_clean_values = [
        float(item["f_clean"]) for item in candidate_metrics if item.get("f_clean") is not None
    ]
    f_robust_values = [
        float(item["f_robust"]) for item in candidate_metrics if item.get("f_robust") is not None
    ]
    representatives_path = run_root / "results" / "pareto_representatives.json"
    representatives = read_json(representatives_path) if representatives_path.exists() else {}
    roles = dict(representatives.get("roles") or {})
    method = str((summary.get("extra") or {}).get("search_method") or efficiency.get("search_method"))
    gpu_hours = float(efficiency.get("gpu_reserved_hours") or 0.0)
    wall_seconds = float(efficiency.get("wall_clock_seconds") or 0.0)
    job_wall_seconds = (
        float(job_efficiency.get("wall_clock_seconds") or 0.0)
        if job_efficiency
        else None
    )
    job_gpu_hours = (
        float(job_efficiency.get("gpu_reserved_hours") or 0.0)
        if job_efficiency
        else None
    )
    candidate_count = int(summary.get("total_evaluated") or 0)
    return {
        "run_root": str(run_root),
        "method": method,
        "seed": seed,
        "run_label": f"{method}:seed={seed}:{run_root.name}",
        "status": summary.get("status"),
        "protocol_fingerprint": canonical_sha256(protocol),
        "evaluation_budget": protocol["evaluation_budget"],
        "eval_epochs": protocol["eval_epochs"],
        "candidate_count": candidate_count,
        "unique_encoding_count": len({candidate_signature(row) for row in candidates}),
        "feasible_count": int(summary.get("feasible") or 0),
        "pareto_front_size": int(pareto.get("pareto_front_size") or 0),
        "reported_hypervolume": pareto.get("hypervolume"),
        "pareto_objectives": list(pareto.get("objectives") or []),
        "best_arch_id": best.get("arch_id"),
        "best_macro_f1": metrics.get("macro_f1"),
        "best_f_clean": max(f_clean_values) if f_clean_values else None,
        "best_f_robust": max(f_robust_values) if f_robust_values else None,
        "accuracy_first_arch_id": (roles.get("accuracy_first") or {}).get("arch_id"),
        "sonar_robust_arch_id": (roles.get("sonar_robust") or {}).get("arch_id"),
        "deployment_balanced_arch_id": (roles.get("deployment_balanced") or {}).get("arch_id"),
        "best_top1": metrics.get("top1"),
        "best_latency_ms": metrics.get("latency_ms"),
        "wall_clock_seconds": wall_seconds,
        "gpu_reserved_hours": gpu_hours,
        "job_efficiency_available": job_efficiency is not None,
        "job_wall_clock_seconds": job_wall_seconds,
        "job_gpu_reserved_hours": job_gpu_hours,
        "job_cuda_used": None if job_efficiency is None else bool(job_efficiency.get("cuda_used")),
        "job_exclusive_gpu_required": (
            None
            if job_efficiency is None
            else bool(job_efficiency.get("exclusive_gpu_required"))
        ),
        "job_segment_count": (
            None if job_efficiency is None else int(job_efficiency.get("segment_count") or 0)
        ),
        "gpu_event_seconds": efficiency.get("gpu_event_seconds"),
        "peak_cuda_memory_bytes": efficiency.get("peak_cuda_memory_bytes"),
        "candidates_per_wall_hour": (
            candidate_count / (wall_seconds / 3600.0) if wall_seconds > 0 else None
        ),
        "candidates_per_gpu_hour": (
            candidate_count / gpu_hours if gpu_hours > 0 else None
        ),
        "candidates_per_job_gpu_hour": (
            candidate_count / job_gpu_hours
            if job_gpu_hours is not None and job_gpu_hours > 0
            else None
        ),
    }


def load_pareto_inputs(
    run_root: Path,
) -> tuple[list[SearchCandidate], list[str], list[str]]:
    pareto = read_json(run_root / "results" / "pareto_selection.json")
    objectives = [str(value) for value in pareto.get("objectives", [])]
    directions = [str(value) for value in pareto.get("directions", [])]
    records = [
        json.loads(line)
        for line in (run_root / "results" / "candidates.jsonl")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    candidates: list[SearchCandidate] = []
    method = read_json(run_root / "results" / "search_efficiency.json").get(
        "search_method", "unknown"
    )
    for record in records:
        if record.get("feasible") is not True:
            continue
        payload = record.get("candidate") or {}
        metrics = CandidateMetrics(**dict(payload.get("metrics") or {}))
        if not objectives or any(getattr(metrics, objective, None) is None for objective in objectives):
            continue
        candidates.append(
            SearchCandidate(
                arch_id=f"{method}:{payload.get('arch_id')}",
                encoding=dict(payload.get("encoding") or {}),
                metrics=metrics,
            )
        )
    return candidates, objectives, directions


def exact_hv_curve_for_candidates(
    candidates: Sequence[SearchCandidate], latency_limit_ms: float
) -> list[dict[str, float | int]]:
    """Return evaluator-order exact HV for [1-f_clean, 1-f_robust, latency/limit]."""

    if not math.isfinite(latency_limit_ms) or latency_limit_ms <= 0:
        raise ValueError("latency_limit_ms must be a finite positive frozen constraint")
    specs = (
        ObjectiveSpec("f_clean", "max", 0.0, 1.0),
        ObjectiveSpec("f_robust", "max", 0.0, 1.0),
        ObjectiveSpec("latency_ms", "min", 0.0, float(latency_limit_ms)),
    )
    rows: list[dict[str, float]] = []
    curve: list[dict[str, float | int]] = []
    for evaluation, candidate in enumerate(candidates, start=1):
        metrics = candidate.metrics
        values = {
            "f_clean": getattr(metrics, "f_clean", None),
            "f_robust": getattr(metrics, "f_robust", None),
            "latency_ms": getattr(metrics, "latency_ms", None),
        }
        if all(value is not None for value in values.values()):
            rows.append({key: float(value) for key, value in values.items()})
        normalized = normalize_objective_rows(rows, specs) if rows else []
        curve.append(
            {
                "evaluation": evaluation,
                "complete_feasible_count": len(rows),
                "exact_normalized_hypervolume": exact_hypervolume(
                    normalized, reference=(1.0, 1.0, 1.0)
                ),
            }
        )
    return curve


def add_exact_hypervolume_metrics(
    rows: list[dict[str, Any]], latency_limit_ms: float | None
) -> dict[str, Any]:
    for row in rows:
        row["exact_normalized_hypervolume"] = None
    if latency_limit_ms is None:
        return {
            "available": False,
            "reason": "latency_limit_ms_not_supplied",
            "definition": "[1-f_clean, 1-f_robust, latency/latency_limit], ref=(1,1,1)",
        }
    curves: dict[str, list[dict[str, float | int]]] = {}
    for row in rows:
        candidates, _objectives, _directions = load_pareto_inputs(Path(row["run_root"]))
        curve = exact_hv_curve_for_candidates(candidates, latency_limit_ms)
        curves[str(row["run_label"])] = curve
        row["exact_normalized_hypervolume"] = (
            curve[-1]["exact_normalized_hypervolume"] if curve else 0.0
        )
    return {
        "available": True,
        "reason": None,
        "latency_limit_ms": float(latency_limit_ms),
        "reference_point": [1.0, 1.0, 1.0],
        "objective_vector": ["1-f_clean", "1-f_robust", "latency/latency_limit"],
        "algorithm": "exact_recursive_union_volume",
        "curves": curves,
    }


def add_joint_pareto_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # Keep the tabular schema stable even when a structural (for example,
    # zero-epoch) smoke run has no complete quality vectors yet.
    for row in rows:
        row["complete_objective_candidate_count"] = 0
        row["joint_pareto_contribution_count"] = 0
        row["joint_pareto_contribution_fraction"] = None

    loaded = [load_pareto_inputs(Path(row["run_root"])) for row in rows]
    objective_sets = {tuple(item[1]) for item in loaded}
    direction_sets = {tuple(item[2]) for item in loaded}
    if len(objective_sets) != 1 or len(direction_sets) != 1:
        return {
            "available": False,
            "reason": "objective_or_direction_mismatch",
            "pairwise_coverage": {},
        }
    objectives = list(next(iter(objective_sets)))
    directions = list(next(iter(direction_sets)))
    if not objectives or any(not item[0] for item in loaded):
        return {
            "available": False,
            "reason": "missing_complete_objective_vectors",
            "objectives": objectives,
            "directions": directions,
            "pairwise_coverage": {},
        }

    combined = [candidate for candidates, _, _ in loaded for candidate in candidates]
    joint_front = compute_pareto_front(combined, objectives, directions)
    joint_ids = {id(candidate) for candidate in joint_front}
    for row, (candidates, _, _) in zip(rows, loaded):
        contribution = sum(id(candidate) in joint_ids for candidate in candidates)
        row["complete_objective_candidate_count"] = len(candidates)
        row["joint_pareto_contribution_count"] = contribution
        row["joint_pareto_contribution_fraction"] = (
            contribution / len(joint_front) if joint_front else None
        )

    coverage: dict[str, dict[str, float]] = {}
    for left_row, (left_candidates, _, _) in zip(rows, loaded):
        left_label = str(left_row["run_label"])
        coverage[left_label] = {}
        for right_row, (right_candidates, _, _) in zip(rows, loaded):
            right_label = str(right_row["run_label"])
            if left_label == right_label:
                continue
            dominated_count = sum(
                any(
                    is_dominated(right, left, objectives, directions)
                    for left in left_candidates
                )
                for right in right_candidates
            )
            coverage[left_label][right_label] = (
                dominated_count / len(right_candidates) if right_candidates else 0.0
            )
    return {
        "available": True,
        "reason": None,
        "objectives": objectives,
        "directions": directions,
        "joint_pareto_size": len(joint_front),
        "pairwise_coverage": coverage,
        "coverage_definition": (
            "C(A,B) is the fraction of B candidates dominated by at least one A candidate."
        ),
    }


def build_method_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return descriptive seed-level summaries without inventing inference."""

    metrics = (
        "exact_normalized_hypervolume",
        "best_f_clean",
        "best_f_robust",
        "best_macro_f1",
        "pareto_front_size",
        "joint_pareto_contribution_count",
        "wall_clock_seconds",
        "gpu_reserved_hours",
        "job_wall_clock_seconds",
        "job_gpu_reserved_hours",
        "candidates_per_wall_hour",
        "candidates_per_gpu_hour",
        "candidates_per_job_gpu_hour",
    )
    aggregates: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method"]) == method]
        aggregate: dict[str, Any] = {
            "method": method,
            "run_count": len(method_rows),
            "seeds": sorted(int(row["seed"]) for row in method_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows if row.get(metric) is not None]
            aggregate[f"{metric}_n"] = len(values)
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else None
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        aggregates.append(aggregate)
    return aggregates


def _t_critical_975(degrees_of_freedom: int) -> float:
    # Two-sided 95% Student-t critical values. The intended benchmark uses
    # three paired seeds, but retaining a compact table keeps the report
    # dependency-free and valid for larger campaigns.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    return table.get(degrees_of_freedom, 1.96)


def _mean_difference_summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_dz": None,
        }
    mean_value = statistics.mean(values)
    if len(values) < 2:
        return {
            "n": len(values),
            "mean": mean_value,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_dz": None,
        }
    std_value = statistics.stdev(values)
    margin = _t_critical_975(len(values) - 1) * std_value / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean_value,
        "std": std_value,
        "ci95_low": mean_value - margin,
        "ci95_high": mean_value + margin,
        "cohens_dz": mean_value / std_value if std_value > 0 else None,
    }


def _exact_two_sided_sign_test(values: Sequence[float]) -> dict[str, Any]:
    non_ties = [float(value) for value in values if float(value) != 0.0]
    positives = sum(value > 0 for value in non_ties)
    negatives = len(non_ties) - positives
    if not non_ties:
        return {"n_non_tie": 0, "positive": 0, "negative": 0, "p_value": 1.0}
    tail = min(positives, negatives)
    probability = sum(math.comb(len(non_ties), k) for k in range(tail + 1)) / (
        2 ** len(non_ties)
    )
    return {
        "n_non_tie": len(non_ties),
        "positive": positives,
        "negative": negatives,
        "p_value": min(1.0, 2.0 * probability),
    }


def _holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, float(value) * (total - index))
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def build_paired_method_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare aging minus RL on matched seeds using paired statistics."""

    methods = {str(row["method"]) for row in rows}
    required = {"aging_evolution", "rl"}
    if not required.issubset(methods):
        return {"available": False, "reason": "requires_aging_evolution_and_rl"}
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["seed"]))
        if key in indexed:
            return {"available": False, "reason": "duplicate_method_seed"}
        indexed[key] = row
    aging_seeds = {seed for method, seed in indexed if method == "aging_evolution"}
    rl_seeds = {seed for method, seed in indexed if method == "rl"}
    if aging_seeds != rl_seeds or not aging_seeds:
        return {
            "available": False,
            "reason": "unmatched_seed_sets",
            "aging_seeds": sorted(aging_seeds),
            "rl_seeds": sorted(rl_seeds),
        }

    metric_specs = {
        "best_f_clean": "higher",
        "best_f_robust": "higher",
        "best_macro_f1": "higher",
        "joint_pareto_contribution_count": "higher",
        "wall_clock_seconds": "lower",
        "gpu_reserved_hours": "lower",
        "job_wall_clock_seconds": "lower",
        "job_gpu_reserved_hours": "lower",
    }
    metrics: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for metric, preferred in metric_specs.items():
        paired_values = []
        for seed in sorted(aging_seeds):
            aging_value = indexed[("aging_evolution", seed)].get(metric)
            rl_value = indexed[("rl", seed)].get(metric)
            if aging_value is None or rl_value is None:
                continue
            paired_values.append(
                {
                    "seed": seed,
                    "aging_evolution": float(aging_value),
                    "rl": float(rl_value),
                    "difference_aging_minus_rl": float(aging_value) - float(rl_value),
                }
            )
        differences = [row["difference_aging_minus_rl"] for row in paired_values]
        if not paired_values:
            continue
        difference_summary = _mean_difference_summary(differences)
        sign_test = _exact_two_sided_sign_test(differences)
        raw_p_values[metric] = float(sign_test["p_value"])
        mean_difference = difference_summary["mean"]
        if mean_difference is None or mean_difference == 0:
            favored_method = "tie_or_unavailable"
        elif (preferred == "higher" and mean_difference > 0) or (
            preferred == "lower" and mean_difference < 0
        ):
            favored_method = "aging_evolution"
        else:
            favored_method = "rl"
        metrics[metric] = {
            "preferred_direction": preferred,
            "paired_values": paired_values,
            "difference_aging_minus_rl": difference_summary,
            "exact_two_sided_sign_test": sign_test,
            "favored_by_mean": favored_method,
        }
    adjusted = _holm_adjust(raw_p_values)
    for metric, adjusted_value in adjusted.items():
        metrics[metric]["holm_adjusted_p_value"] = adjusted_value
    complete_pair_counts = [
        int(metric["difference_aging_minus_rl"]["n"]) for metric in metrics.values()
    ]
    return {
        "available": True,
        "difference_definition": "aging_evolution minus rl",
        "paired_seeds": sorted(aging_seeds),
        "inference_ready": bool(complete_pair_counts and min(complete_pair_counts) >= 3),
        "test": "exact two-sided paired sign test with Holm correction",
        "ci": "two-sided 95% Student-t interval for the paired mean difference",
        "small_sample_boundary": (
            "With only three paired seeds, the exact two-sided sign test cannot attain "
            "p<0.05 and Cohen's dz is unstable; report the interval and raw pairs without "
            "claiming statistical superiority."
        ),
        "metrics": metrics,
        "decision_boundary": (
            "Quality and compute are separate outcomes; no single overall winner is declared "
            "when their directions conflict."
        ),
    }


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    latency_limit_ms: float | None = None,
) -> dict[str, Any]:
    mutable_rows = [dict(row) for row in rows]
    joint_pareto = add_joint_pareto_metrics(mutable_rows)
    exact_hypervolume_report = add_exact_hypervolume_metrics(
        mutable_rows, latency_limit_ms
    )
    rows = mutable_rows
    paired_comparison = build_paired_method_comparison(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = sorted({str(row["protocol_fingerprint"]) for row in rows})
    candidate_counts = {int(row["candidate_count"]) for row in rows}
    methods = sorted({str(row["method"]) for row in rows})
    seed_sets = {
        method: sorted(int(row["seed"]) for row in rows if str(row["method"]) == method)
        for method in methods
    }
    matched_seed_sets = len({tuple(seeds) for seeds in seed_sets.values()}) == 1
    unique_method_seed_pairs = len({(str(row["method"]), int(row["seed"])) for row in rows}) == len(rows)
    all_completed = all(row.get("status") == "completed" for row in rows)
    clean_quality_available = all(
        row.get("best_f_clean") is not None or row.get("best_macro_f1") is not None
        for row in rows
    )
    robustness_required = any(
        "f_robust" in list(row.get("pareto_objectives") or []) for row in rows
    )
    robustness_quality_available = all(row.get("best_f_robust") is not None for row in rows)
    quality_available = bool(
        clean_quality_available
        and (not robustness_required or robustness_quality_available)
    )
    job_efficiency_available = all(
        row.get("job_efficiency_available") is True for row in rows
    )
    gpu_efficiency_ready = bool(
        job_efficiency_available
        and all(row.get("job_cuda_used") is True for row in rows)
        and all(row.get("job_exclusive_gpu_required") is True for row in rows)
    )
    same_protocol = len(fingerprints) == 1
    equal_actual_candidate_count = len(candidate_counts) == 1
    payload = {
        "schema_version": 1,
        "comparison_ready": (
            len(methods) >= 2
            and same_protocol
            and equal_actual_candidate_count
            and matched_seed_sets
            and unique_method_seed_pairs
            and all_completed
            and quality_available
            and bool(joint_pareto.get("available"))
            and bool(paired_comparison.get("available"))
            and gpu_efficiency_ready
        ),
        "structural_smoke_ready": (
            len(methods) >= 2
            and same_protocol
            and equal_actual_candidate_count
            and matched_seed_sets
            and unique_method_seed_pairs
            and all_completed
        ),
        "same_protocol": same_protocol,
        "equal_actual_candidate_count": equal_actual_candidate_count,
        "matched_seed_sets": matched_seed_sets,
        "unique_method_seed_pairs": unique_method_seed_pairs,
        "seed_sets": seed_sets,
        "quality_available": quality_available,
        "clean_quality_available": clean_quality_available,
        "robustness_required": robustness_required,
        "robustness_quality_available": robustness_quality_available,
        "job_efficiency_available": job_efficiency_available,
        "gpu_efficiency_ready": gpu_efficiency_ready,
        "protocol_fingerprints": fingerprints,
        "primary_compute_metric": "job_gpu_reserved_hours",
        "secondary_compute_metrics": [
            "gpu_reserved_hours",
            "gpu_event_seconds",
            "job_wall_clock_seconds",
        ],
        "quality_metrics": ["best_f_clean", "best_f_robust", "pareto_front_size"],
        "joint_pareto": joint_pareto,
        "exact_hypervolume": exact_hypervolume_report,
        "formal_exact_hv_ready": bool(exact_hypervolume_report.get("available")),
        "paired_method_comparison": paired_comparison,
        "inference_ready": bool(paired_comparison.get("inference_ready")),
        "method_aggregates": build_method_aggregates(rows),
        "runs": list(rows),
        "claim_boundary": (
            "Search-proxy quality and search-time efficiency only; this report does not merge "
            "retrain, route/COM5, board accuracy, image quality, or measured power evidence."
        ),
        "hypervolume_boundary": (
            "reported_hypervolume is retained only as legacy evidence. The formal primary metric "
            "is exact_hypervolume and is unavailable unless --latency-limit-ms is explicitly frozen."
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Search Method Comparison",
        "",
        f"- Comparison ready: `{payload['comparison_ready']}`",
        f"- Structural smoke ready: `{payload['structural_smoke_ready']}`",
        f"- Same protocol: `{payload['same_protocol']}`",
        f"- Equal actual candidate count: `{payload['equal_actual_candidate_count']}`",
        f"- Matched seed sets: `{payload['matched_seed_sets']}`",
        f"- Quality metrics available: `{payload['quality_available']}`",
        f"- Full-job GPU efficiency ready: `{payload['gpu_efficiency_ready']}`",
        f"- Paired inference ready: `{payload['inference_ready']}`",
        "- Primary compute metric: `job_gpu_reserved_hours`",
        "- Boundary: search proxy only; no retrain/route/board/power claim.",
        "",
        "| method | seed | candidates | unique | feasible | Pareto | joint contribution | best F_clean | best F_robust | search wall s | search GPU h | full-job GPU h | GPU event s | peak CUDA bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {seed} | {candidate_count} | {unique_encoding_count} | {feasible_count} | "
            "{pareto_front_size} | {joint_pareto_contribution_count} | {best_f_clean} | {best_f_robust} | {wall_clock_seconds:.3f} | "
            "{gpu_reserved_hours:.6f} | {job_gpu_reserved_hours} | {gpu_event_seconds} | {peak_cuda_memory_bytes} |".format(**row)
        )
    if paired_comparison.get("available"):
        lines.extend(
            [
                "",
                "## Paired aging minus RL differences",
                "",
                "| metric | n | mean difference | 95% CI | Cohen dz | Holm p | favored by mean |",
                "|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for metric, result in paired_comparison["metrics"].items():
            difference = result["difference_aging_minus_rl"]
            ci = (
                "n/a"
                if difference["ci95_low"] is None
                else f"[{difference['ci95_low']:.6g}, {difference['ci95_high']:.6g}]"
            )
            effect = (
                "n/a" if difference["cohens_dz"] is None else f"{difference['cohens_dz']:.6g}"
            )
            lines.append(
                f"| {metric} | {difference['n']} | {difference['mean']} | {ci} | "
                f"{effect} | {result['holm_adjusted_p_value']:.6g} | "
                f"{result['favored_by_mean']} |"
            )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    rows = [summarize_run(normalize_run_root(path)) for path in args.run]
    payload = write_outputs(
        rows,
        Path(args.output_dir).expanduser().resolve(),
        latency_limit_ms=args.latency_limit_ms,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
