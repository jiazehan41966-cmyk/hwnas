"""Proxy Reliability Audit (Gate 0) statistics and reporting.

The audit intentionally keeps classification reliability separate from
hardware-estimator reliability.  It never treats search episodes or the
``N(N-1)/2`` architecture pairs as independent experimental repetitions.

Classification input is a long CSV with one row per
``(architecture, proxy, budget, seed, outer_fold, metric)``:

``architecture_id,proxy_name,budget,seed,outer_fold,metric,proxy_value,truth_value``

``truth_value`` is required only on the full-training budget rows and must be
the untouched outer-fold score.  Short-budget rows contain inner-validation
proxy values only.  The analyzer joins short-budget proxies to the matching
full-budget truth by architecture, seed, outer fold, and metric.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


SCHEMA_VERSION = 1
DEFAULT_TRUTH_BUDGET = 150
DEFAULT_TOP_K = (5, 10)
REQUIRED_CLASSIFICATION_COLUMNS = {
    "architecture_id",
    "proxy_name",
    "budget",
    "seed",
    "outer_fold",
    "metric",
    "proxy_value",
    "truth_value",
}
REQUIRED_HARDWARE_COLUMNS = {
    "architecture_id",
    "proxy_feasible",
    "truth_feasible",
    "truth_latency_source",
    "truth_resource_source",
    "truth_feasibility_source",
}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    resolved = float(text)
    if not math.isfinite(resolved):
        raise ValueError(f"expected finite numeric value, got {value!r}")
    return resolved


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "pass", "feasible"}:
        return True
    if normalized in {"0", "false", "no", "n", "fail", "infeasible"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


@dataclass(frozen=True)
class ClassificationObservation:
    architecture_id: str
    proxy_name: str
    budget: int
    seed: int
    outer_fold: int
    metric: str
    proxy_value: Optional[float]
    truth_value: Optional[float]
    proxy_direction: str = "max"
    status: str = "completed"
    recipe_id: str = ""
    run_path: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ClassificationObservation":
        proxy_direction = str(row.get("proxy_direction", "max")).strip().lower()
        if proxy_direction not in {"max", "min"}:
            raise ValueError(
                f"proxy_direction must be max or min, got {proxy_direction!r}"
            )
        return cls(
            architecture_id=str(row["architecture_id"]).strip(),
            proxy_name=str(row["proxy_name"]).strip(),
            budget=int(row["budget"]),
            seed=int(row["seed"]),
            outer_fold=int(row["outer_fold"]),
            metric=str(row["metric"]).strip(),
            proxy_value=_optional_float(row.get("proxy_value")),
            truth_value=_optional_float(row.get("truth_value")),
            proxy_direction=proxy_direction,
            status=str(row.get("status", "completed")).strip().lower(),
            recipe_id=str(row.get("recipe_id", "")).strip(),
            run_path=str(row.get("run_path", "")).strip(),
        )

    @property
    def cell_key(self) -> tuple[str, int, int, str]:
        return (self.architecture_id, self.seed, self.outer_fold, self.metric)

    @property
    def proxy_key(self) -> tuple[str, int, str]:
        return (self.proxy_name, self.budget, self.metric)


def load_classification_observations(
    path: str | Path,
) -> list[ClassificationObservation]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_CLASSIFICATION_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"classification observation CSV is missing columns: {missing}"
            )
        rows = [ClassificationObservation.from_mapping(row) for row in reader]
    if not rows:
        raise ValueError("classification observation CSV contains no rows")
    return rows


def load_hardware_observations(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_HARDWARE_COLUMNS - columns)
        if missing:
            raise ValueError(f"hardware observation CSV is missing columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("hardware observation CSV contains no rows")
    source_columns = (
        "truth_latency_source",
        "truth_resource_source",
        "truth_feasibility_source",
    )
    for row_number, row in enumerate(rows, start=2):
        empty_sources = [
            column
            for column in source_columns
            if not str(row.get(column, "")).strip()
        ]
        if empty_sources:
            raise ValueError(
                f"hardware observation row {row_number} has empty truth "
                f"source fields: {empty_sources}"
            )
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _mean(values: Iterable[float]) -> float:
    resolved = [float(value) for value in values]
    if not resolved:
        raise ValueError("mean requires at least one value")
    return float(statistics.fmean(resolved))


def _mean_square(sum_squares: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        raise ValueError("mean square requires positive degrees of freedom")
    return float(sum_squares) / degrees_freedom


def balanced_two_way_variance(
    truth_rows: Sequence[ClassificationObservation],
    *,
    replicate_axis: str,
    truth_budget: int = DEFAULT_TRUTH_BUDGET,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int = 20260704,
) -> dict[str, Any]:
    """Estimate architecture x seed or architecture x fold variance."""
    if replicate_axis not in {"seed", "outer_fold"}:
        raise ValueError("replicate_axis must be seed or outer_fold")
    completed = [
        row
        for row in truth_rows
        if row.budget == truth_budget
        and row.truth_value is not None
        and row.status == "completed"
    ]
    if not completed:
        return {
            "estimable": False,
            "reason": f"no completed truth rows at budget={truth_budget}",
        }
    metrics = sorted({row.metric for row in completed})
    if len(metrics) != 1:
        return {
            "estimable": False,
            "reason": f"variance decomposition requires one metric, found {metrics}",
        }
    architectures = sorted({row.architecture_id for row in completed})
    replicates = sorted(
        {
            int(row.seed if replicate_axis == "seed" else row.outer_fold)
            for row in completed
        }
    )
    if len(architectures) < 2 or len(replicates) < 2:
        return {
            "estimable": False,
            "reason": "two-way decomposition needs at least 2 architectures and replicates",
        }
    index: dict[tuple[str, int], float] = {}
    duplicates: list[tuple[str, int]] = []
    for row in completed:
        replicate = int(
            row.seed if replicate_axis == "seed" else row.outer_fold
        )
        key = (row.architecture_id, replicate)
        if key in index:
            duplicates.append(key)
        index[key] = float(row.truth_value)
    expected = {
        (architecture, replicate)
        for architecture in architectures
        for replicate in replicates
    }
    missing = sorted(expected - set(index))
    if duplicates or missing:
        return {
            "estimable": False,
            "reason": "truth grid is not a complete architecture x replicate design",
            "duplicate_cell_count": len(duplicates),
            "missing_cell_count": len(missing),
        }
    array = np.asarray(
        [
            [index[(architecture, replicate)] for replicate in replicates]
            for architecture in architectures
        ],
        dtype=float,
    )
    result = _two_way_components_from_array(array)
    result.update(
        {
            "estimable": True,
            "design": f"architecture_x_{replicate_axis}",
            "metric": metrics[0],
            "truth_budget": int(truth_budget),
            "architecture_n": len(architectures),
            "replicate_n": len(replicates),
            "replicate_axis": replicate_axis,
            "replicates": replicates,
        }
    )
    if bootstrap_iterations > 0:
        rng = random.Random(bootstrap_seed)
        bootstrap: dict[str, list[float]] = defaultdict(list)
        for _ in range(int(bootstrap_iterations)):
            draws = [rng.randrange(len(architectures)) for _ in architectures]
            estimate = _two_way_components_from_array(array[draws, :])
            for key in (
                "icc_absolute_single",
                "icc_relative_single",
                "icc_absolute_mean_observed",
                "icc_relative_mean_observed",
            ):
                bootstrap[key].append(float(estimate[key]))
        result["bootstrap"] = {
            key: {
                "ci95_low": _quantile(values, 0.025),
                "ci95_high": _quantile(values, 0.975),
            }
            for key, values in bootstrap.items()
        }
        result["bootstrap"].update(
            {
                "iterations": int(bootstrap_iterations),
                "seed": int(bootstrap_seed),
                "unit": "architecture_cluster",
            }
        )
    return result


def _two_way_components_from_array(array: np.ndarray) -> dict[str, Any]:
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("two-way array must be architecture x replicate")
    n_arch, n_replicate = array.shape
    grand = float(array.mean())
    mean_arch = array.mean(axis=1)
    mean_replicate = array.mean(axis=0)
    ss_arch = n_replicate * float(np.square(mean_arch - grand).sum())
    ss_replicate = n_arch * float(np.square(mean_replicate - grand).sum())
    ss_total = float(np.square(array - grand).sum())
    ss_error = max(0.0, ss_total - ss_arch - ss_replicate)
    ms_arch = _mean_square(ss_arch, n_arch - 1)
    ms_replicate = _mean_square(ss_replicate, n_replicate - 1)
    ms_error = _mean_square(
        ss_error,
        (n_arch - 1) * (n_replicate - 1),
    )
    raw = {
        "architecture": (ms_arch - ms_error) / n_replicate,
        "replicate": (ms_replicate - ms_error) / n_arch,
        "residual": ms_error,
    }
    clipped = {key: max(0.0, float(value)) for key, value in raw.items()}

    def ratio(denominator: float) -> float:
        return (
            clipped["architecture"] / denominator
            if denominator > 0
            else 0.0
        )

    return {
        "raw_variance_components": raw,
        "variance_components": clipped,
        "mean_squares": {
            "architecture": ms_arch,
            "replicate": ms_replicate,
            "residual": ms_error,
        },
        "icc_absolute_single": ratio(sum(clipped.values())),
        "icc_relative_single": ratio(
            clipped["architecture"] + clipped["residual"]
        ),
        "icc_absolute_mean_observed": ratio(
            clipped["architecture"]
            + clipped["replicate"] / n_replicate
            + clipped["residual"] / n_replicate
        ),
        "icc_relative_mean_observed": ratio(
            clipped["architecture"]
            + clipped["residual"] / n_replicate
        ),
        "warnings": [
            f"negative method-of-moments component clipped: {key}={value:.6g}"
            for key, value in raw.items()
            if value < 0
        ],
    }


def balanced_random_effects_variance(
    truth_rows: Sequence[ClassificationObservation],
    *,
    truth_budget: int = DEFAULT_TRUTH_BUDGET,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int = 20260704,
) -> dict[str, Any]:
    """Estimate balanced architecture x seed x outer-fold variance components.

    Method-of-moments expected-mean-square estimates are used for the complete
    three-way crossed grid.  Negative component estimates are retained in
    ``raw_variance_components`` for diagnosis and clipped to zero for ICC
    calculations.
    """

    completed = [
        row
        for row in truth_rows
        if row.budget == truth_budget
        and row.truth_value is not None
        and row.status == "completed"
    ]
    if not completed:
        return {
            "estimable": False,
            "reason": f"no completed truth rows at budget={truth_budget}",
        }

    architectures = sorted({row.architecture_id for row in completed})
    seeds = sorted({row.seed for row in completed})
    folds = sorted({row.outer_fold for row in completed})
    metrics = sorted({row.metric for row in completed})
    if len(metrics) != 1:
        return {
            "estimable": False,
            "reason": f"variance decomposition requires one metric, found {metrics}",
        }
    if min(len(architectures), len(seeds), len(folds)) < 2:
        return {
            "estimable": False,
            "reason": (
                "variance decomposition requires at least 2 architectures, "
                "2 seeds, and 2 outer folds"
            ),
        }

    index: dict[tuple[str, int, int], float] = {}
    duplicate_keys: list[tuple[str, int, int]] = []
    for row in completed:
        key = (row.architecture_id, row.seed, row.outer_fold)
        if key in index:
            duplicate_keys.append(key)
        index[key] = float(row.truth_value)
    expected = {
        (architecture, seed, fold)
        for architecture in architectures
        for seed in seeds
        for fold in folds
    }
    missing = sorted(expected - set(index))
    if duplicate_keys or missing:
        return {
            "estimable": False,
            "reason": "truth grid is not a complete one-observation crossed design",
            "duplicate_cell_count": len(duplicate_keys),
            "missing_cell_count": len(missing),
            "missing_cells_preview": [
                {"architecture_id": a, "seed": s, "outer_fold": f}
                for a, s, f in missing[:20]
            ],
        }

    array = np.asarray(
        [
            [
                [index[(architecture, seed, fold)] for fold in folds]
                for seed in seeds
            ]
            for architecture in architectures
        ],
        dtype=float,
    )
    result = _variance_components_from_array(array)
    result.update(
        {
            "estimable": True,
            "metric": metrics[0],
            "truth_budget": int(truth_budget),
            "architecture_n": len(architectures),
            "seed_n": len(seeds),
            "outer_fold_n": len(folds),
            "architectures": architectures,
            "seeds": seeds,
            "outer_folds": folds,
        }
    )

    if bootstrap_iterations > 0:
        rng = random.Random(bootstrap_seed)
        bootstrap: dict[str, list[float]] = defaultdict(list)
        for _ in range(int(bootstrap_iterations)):
            draws = [rng.randrange(len(architectures)) for _ in architectures]
            sampled = array[draws, :, :]
            estimate = _variance_components_from_array(sampled)
            for key in (
                "icc_absolute_single",
                "icc_relative_single",
                "icc_absolute_mean_seeds_single_fold",
                "icc_relative_mean_seeds_single_fold",
                "icc_absolute_single_seed_mean_folds",
                "icc_relative_single_seed_mean_folds",
                "icc_absolute_mean_observed",
                "icc_relative_mean_observed",
            ):
                bootstrap[key].append(float(estimate[key]))
        result["bootstrap"] = {
            key: {
                "ci95_low": _quantile(values, 0.025),
                "ci95_high": _quantile(values, 0.975),
            }
            for key, values in bootstrap.items()
        }
        result["bootstrap"]["iterations"] = int(bootstrap_iterations)
        result["bootstrap"]["seed"] = int(bootstrap_seed)
        result["bootstrap"]["unit"] = "architecture_cluster"

    return result


def _variance_components_from_array(array: np.ndarray) -> dict[str, Any]:
    if array.ndim != 3:
        raise ValueError("variance component array must be architecture x seed x fold")
    n_arch, n_seed, n_fold = array.shape
    if min(n_arch, n_seed, n_fold) < 2:
        raise ValueError("variance component array needs at least 2 levels per factor")

    grand = float(array.mean())
    mean_a = array.mean(axis=(1, 2))
    mean_s = array.mean(axis=(0, 2))
    mean_f = array.mean(axis=(0, 1))
    mean_as = array.mean(axis=2)
    mean_af = array.mean(axis=1)
    mean_sf = array.mean(axis=0)

    ss_a = n_seed * n_fold * float(np.square(mean_a - grand).sum())
    ss_s = n_arch * n_fold * float(np.square(mean_s - grand).sum())
    ss_f = n_arch * n_seed * float(np.square(mean_f - grand).sum())
    ss_as = n_fold * float(
        np.square(
            mean_as - mean_a[:, None] - mean_s[None, :] + grand
        ).sum()
    )
    ss_af = n_seed * float(
        np.square(
            mean_af - mean_a[:, None] - mean_f[None, :] + grand
        ).sum()
    )
    ss_sf = n_arch * float(
        np.square(
            mean_sf - mean_s[:, None] - mean_f[None, :] + grand
        ).sum()
    )
    ss_total = float(np.square(array - grand).sum())
    ss_error = max(
        0.0,
        ss_total - ss_a - ss_s - ss_f - ss_as - ss_af - ss_sf,
    )

    df_a = n_arch - 1
    df_s = n_seed - 1
    df_f = n_fold - 1
    df_as = df_a * df_s
    df_af = df_a * df_f
    df_sf = df_s * df_f
    df_error = df_a * df_s * df_f

    ms_a = _mean_square(ss_a, df_a)
    ms_s = _mean_square(ss_s, df_s)
    ms_f = _mean_square(ss_f, df_f)
    ms_as = _mean_square(ss_as, df_as)
    ms_af = _mean_square(ss_af, df_af)
    ms_sf = _mean_square(ss_sf, df_sf)
    ms_error = _mean_square(ss_error, df_error)

    raw = {
        "architecture": (
            ms_a
            - ms_error
            - n_fold * ((ms_as - ms_error) / n_fold)
            - n_seed * ((ms_af - ms_error) / n_seed)
        )
        / (n_seed * n_fold),
        "seed": (
            ms_s
            - ms_error
            - n_fold * ((ms_as - ms_error) / n_fold)
            - n_arch * ((ms_sf - ms_error) / n_arch)
        )
        / (n_arch * n_fold),
        "outer_fold": (
            ms_f
            - ms_error
            - n_seed * ((ms_af - ms_error) / n_seed)
            - n_arch * ((ms_sf - ms_error) / n_arch)
        )
        / (n_arch * n_seed),
        "architecture_seed": (ms_as - ms_error) / n_fold,
        "architecture_outer_fold": (ms_af - ms_error) / n_seed,
        "seed_outer_fold": (ms_sf - ms_error) / n_arch,
        "residual": ms_error,
    }
    clipped = {key: max(0.0, float(value)) for key, value in raw.items()}
    warnings = [
        f"negative method-of-moments component clipped: {key}={value:.6g}"
        for key, value in raw.items()
        if value < 0
    ]

    absolute_single_denominator = sum(clipped.values())
    relative_single_denominator = (
        clipped["architecture"]
        + clipped["architecture_seed"]
        + clipped["architecture_outer_fold"]
        + clipped["residual"]
    )
    absolute_mean_denominator = (
        clipped["architecture"]
        + clipped["seed"] / n_seed
        + clipped["outer_fold"] / n_fold
        + clipped["architecture_seed"] / n_seed
        + clipped["architecture_outer_fold"] / n_fold
        + clipped["seed_outer_fold"] / (n_seed * n_fold)
        + clipped["residual"] / (n_seed * n_fold)
    )
    relative_mean_denominator = (
        clipped["architecture"]
        + clipped["architecture_seed"] / n_seed
        + clipped["architecture_outer_fold"] / n_fold
        + clipped["residual"] / (n_seed * n_fold)
    )
    absolute_mean_seeds_single_fold_denominator = (
        clipped["architecture"]
        + clipped["seed"] / n_seed
        + clipped["outer_fold"]
        + clipped["architecture_seed"] / n_seed
        + clipped["architecture_outer_fold"]
        + clipped["seed_outer_fold"] / n_seed
        + clipped["residual"] / n_seed
    )
    relative_mean_seeds_single_fold_denominator = (
        clipped["architecture"]
        + clipped["architecture_seed"] / n_seed
        + clipped["architecture_outer_fold"]
        + clipped["residual"] / n_seed
    )
    absolute_single_seed_mean_folds_denominator = (
        clipped["architecture"]
        + clipped["seed"]
        + clipped["outer_fold"] / n_fold
        + clipped["architecture_seed"]
        + clipped["architecture_outer_fold"] / n_fold
        + clipped["seed_outer_fold"] / n_fold
        + clipped["residual"] / n_fold
    )
    relative_single_seed_mean_folds_denominator = (
        clipped["architecture"]
        + clipped["architecture_seed"]
        + clipped["architecture_outer_fold"] / n_fold
        + clipped["residual"] / n_fold
    )

    def ratio(denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return clipped["architecture"] / denominator

    return {
        "raw_variance_components": raw,
        "variance_components": clipped,
        "mean_squares": {
            "architecture": ms_a,
            "seed": ms_s,
            "outer_fold": ms_f,
            "architecture_seed": ms_as,
            "architecture_outer_fold": ms_af,
            "seed_outer_fold": ms_sf,
            "residual": ms_error,
        },
        "icc_absolute_single": ratio(absolute_single_denominator),
        "icc_relative_single": ratio(relative_single_denominator),
        "icc_absolute_mean_seeds_single_fold": ratio(
            absolute_mean_seeds_single_fold_denominator
        ),
        "icc_relative_mean_seeds_single_fold": ratio(
            relative_mean_seeds_single_fold_denominator
        ),
        "icc_absolute_single_seed_mean_folds": ratio(
            absolute_single_seed_mean_folds_denominator
        ),
        "icc_relative_single_seed_mean_folds": ratio(
            relative_single_seed_mean_folds_denominator
        ),
        "icc_absolute_mean_observed": ratio(absolute_mean_denominator),
        "icc_relative_mean_observed": ratio(relative_mean_denominator),
        "warnings": warnings,
    }


def _rank_metrics(
    proxy_by_architecture: Mapping[str, float],
    truth_by_architecture: Mapping[str, float],
    *,
    top_k: Sequence[int],
    tie_tolerance: float,
) -> dict[str, Any]:
    architectures = sorted(set(proxy_by_architecture) & set(truth_by_architecture))
    if len(architectures) < 2:
        raise ValueError("rank metrics require at least two matched architectures")
    proxy = np.asarray([proxy_by_architecture[key] for key in architectures], dtype=float)
    truth = np.asarray([truth_by_architecture[key] for key in architectures], dtype=float)
    spearman = stats.spearmanr(proxy, truth)
    kendall = stats.kendalltau(proxy, truth, variant="b")

    concordant = 0
    discordant = 0
    tied_or_indeterminate = 0
    for left in range(len(architectures)):
        for right in range(left + 1, len(architectures)):
            truth_delta = truth[left] - truth[right]
            proxy_delta = proxy[left] - proxy[right]
            if abs(truth_delta) <= tie_tolerance or proxy_delta == 0:
                tied_or_indeterminate += 1
            elif (truth_delta > 0) == (proxy_delta > 0):
                concordant += 1
            else:
                discordant += 1
    decided_pairs = concordant + discordant
    pairwise_accuracy = (
        concordant / decided_pairs if decided_pairs else None
    )

    result: dict[str, Any] = {
        "architecture_n": len(architectures),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_pvalue": float(kendall.pvalue),
        "pairwise_accuracy": pairwise_accuracy,
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_tied_or_indeterminate": tied_or_indeterminate,
        "pairwise_decided": decided_pairs,
        "tie_tolerance": float(tie_tolerance),
    }

    proxy_order = sorted(
        architectures,
        key=lambda key: (proxy_by_architecture[key], key),
        reverse=True,
    )
    truth_order = sorted(
        architectures,
        key=lambda key: (truth_by_architecture[key], key),
        reverse=True,
    )
    truth_best = truth_by_architecture[truth_order[0]]
    minimum_truth = min(truth_by_architecture[key] for key in architectures)
    maximum_truth = max(truth_by_architecture[key] for key in architectures)
    truth_span = maximum_truth - minimum_truth
    relevance = {
        key: (
            (truth_by_architecture[key] - minimum_truth) / truth_span
            if truth_span > 0
            else 1.0
        )
        for key in architectures
    }

    for requested_k in top_k:
        k = min(max(1, int(requested_k)), len(architectures))
        proxy_top = proxy_order[:k]
        truth_top = set(truth_order[:k])
        recall = len(set(proxy_top) & truth_top) / k
        selected_best = max(truth_by_architecture[key] for key in proxy_top)
        regret = truth_best - selected_best
        dcg = sum(
            (2.0 ** relevance[key] - 1.0) / math.log2(rank + 2.0)
            for rank, key in enumerate(proxy_top)
        )
        ideal = sum(
            (2.0 ** relevance[key] - 1.0) / math.log2(rank + 2.0)
            for rank, key in enumerate(truth_order[:k])
        )
        result[f"top{k}_recall"] = recall
        result[f"top{k}_random_expected_recall"] = k / len(architectures)
        result[f"ndcg_at{k}"] = dcg / ideal if ideal > 0 else 1.0
        result[f"regret_at{k}"] = regret
        result[f"proxy_top{k}"] = proxy_top
        result[f"truth_top{k}"] = truth_order[:k]
    result["regret_definition"] = (
        "best audited-set truth minus best truth among proxy Top-K"
    )
    return result


def _matched_cells(
    observations: Sequence[ClassificationObservation],
    *,
    proxy_name: str,
    budget: int,
    metric: str,
    truth_budget: int,
) -> list[dict[str, Any]]:
    truth_index: dict[tuple[str, int, int, str], float] = {}
    for row in observations:
        if (
            row.budget == truth_budget
            and row.metric == metric
            and row.truth_value is not None
            and row.status == "completed"
        ):
            if row.cell_key in truth_index:
                if not math.isclose(
                    truth_index[row.cell_key],
                    float(row.truth_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"conflicting truth values for cell {row.cell_key}"
                    )
            truth_index[row.cell_key] = float(row.truth_value)

    matched: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, int, str]] = set()
    for row in observations:
        if (
            row.proxy_name != proxy_name
            or row.budget != budget
            or row.metric != metric
            or row.proxy_value is None
            or row.status != "completed"
        ):
            continue
        unique = (
            row.architecture_id,
            row.proxy_name,
            row.budget,
            row.seed,
            row.outer_fold,
            row.metric,
        )
        if unique in seen:
            raise ValueError(f"duplicate proxy observation: {unique}")
        seen.add(unique)
        if row.cell_key not in truth_index:
            continue
        sign = 1.0 if row.proxy_direction == "max" else -1.0
        matched.append(
            {
                "architecture_id": row.architecture_id,
                "seed": row.seed,
                "outer_fold": row.outer_fold,
                "proxy": sign * float(row.proxy_value),
                "truth": float(truth_index[row.cell_key]),
            }
        )
    return matched


def _aggregate_architecture_means(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    proxy: dict[str, list[float]] = defaultdict(list)
    truth: dict[str, list[float]] = defaultdict(list)
    for row in cells:
        architecture = str(row["architecture_id"])
        proxy[architecture].append(float(row["proxy"]))
        truth[architecture].append(float(row["truth"]))
    return (
        {key: _mean(values) for key, values in proxy.items()},
        {key: _mean(values) for key, values in truth.items()},
    )


def _hierarchical_rank_bootstrap(
    cells: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int],
    tie_tolerance: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_arch_fold: dict[str, dict[int, dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in cells:
        architecture = str(row["architecture_id"])
        fold = int(row["outer_fold"])
        run_seed = int(row["seed"])
        by_arch_fold[architecture][fold][run_seed] = row
    architectures = sorted(by_arch_fold)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(max(1, int(iterations))):
        proxy_means: dict[str, float] = {}
        truth_means: dict[str, float] = {}
        for draw_index in range(len(architectures)):
            architecture = rng.choice(architectures)
            folds = sorted(by_arch_fold[architecture])
            values_proxy: list[float] = []
            values_truth: list[float] = []
            for _fold_draw in range(len(folds)):
                fold = rng.choice(folds)
                seed_rows = by_arch_fold[architecture][fold]
                seeds = sorted(seed_rows)
                for _seed_draw in range(len(seeds)):
                    run_seed = rng.choice(seeds)
                    row = seed_rows[run_seed]
                    values_proxy.append(float(row["proxy"]))
                    values_truth.append(float(row["truth"]))
            pseudo_id = f"{architecture}#draw{draw_index}"
            proxy_means[pseudo_id] = _mean(values_proxy)
            truth_means[pseudo_id] = _mean(values_truth)
        metrics = _rank_metrics(
            proxy_means,
            truth_means,
            top_k=top_k,
            tie_tolerance=tie_tolerance,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
                samples[key].append(float(value))

    return {
        "iterations": max(1, int(iterations)),
        "seed": int(seed),
        "resampling": "architecture_primary_then_outer_fold_then_seed",
        "ci95": {
            key: {
                "low": _quantile(values, 0.025),
                "high": _quantile(values, 0.975),
            }
            for key, values in samples.items()
            if values
        },
    }


def analyze_proxy_reliability(
    observations: Sequence[ClassificationObservation],
    *,
    truth_budget: int = DEFAULT_TRUTH_BUDGET,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    tie_tolerance: float = 0.0,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20260704,
    gate_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Analyze architecture signal and multi-fidelity ranking reliability."""
    completed = [row for row in observations if row.status == "completed"]
    metrics = sorted({row.metric for row in completed})
    architectures = sorted({row.architecture_id for row in completed})
    seeds = sorted({row.seed for row in completed})
    folds = sorted({row.outer_fold for row in completed})
    proxy_groups = sorted({row.proxy_key for row in completed})

    variance: dict[str, Any] = {}
    for metric in metrics:
        metric_rows = [row for row in completed if row.metric == metric]
        metric_seeds = sorted({row.seed for row in metric_rows})
        metric_folds = sorted({row.outer_fold for row in metric_rows})
        if len(metric_folds) == 1 and len(metric_seeds) >= 2:
            variance[metric] = balanced_two_way_variance(
                metric_rows,
                replicate_axis="seed",
                truth_budget=truth_budget,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        elif len(metric_seeds) == 1 and len(metric_folds) >= 2:
            variance[metric] = balanced_two_way_variance(
                metric_rows,
                replicate_axis="outer_fold",
                truth_budget=truth_budget,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        else:
            variance[metric] = balanced_random_effects_variance(
                metric_rows,
                truth_budget=truth_budget,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )

    fidelity: list[dict[str, Any]] = []
    for group_index, (proxy_name, budget, metric) in enumerate(proxy_groups):
        cells = _matched_cells(
            completed,
            proxy_name=proxy_name,
            budget=budget,
            metric=metric,
            truth_budget=truth_budget,
        )
        if not cells:
            fidelity.append(
                {
                    "proxy_name": proxy_name,
                    "budget": budget,
                    "metric": metric,
                    "estimable": False,
                    "reason": "no proxy cells matched full-budget outer truth",
                }
            )
            continue
        proxy_means, truth_means = _aggregate_architecture_means(cells)
        if len(proxy_means) < 2:
            fidelity.append(
                {
                    "proxy_name": proxy_name,
                    "budget": budget,
                    "metric": metric,
                    "estimable": False,
                    "reason": "fewer than two architectures have matched observations",
                }
            )
            continue
        rank = _rank_metrics(
            proxy_means,
            truth_means,
            top_k=top_k,
            tie_tolerance=tie_tolerance,
        )
        bootstrap = _hierarchical_rank_bootstrap(
            cells,
            top_k=top_k,
            tie_tolerance=tie_tolerance,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + group_index,
        )
        coverage_by_architecture: dict[str, int] = defaultdict(int)
        for cell in cells:
            coverage_by_architecture[str(cell["architecture_id"])] += 1
        fidelity.append(
            {
                "proxy_name": proxy_name,
                "budget": budget,
                "metric": metric,
                "estimable": True,
                "matched_cell_n": len(cells),
                "cells_per_architecture": dict(sorted(coverage_by_architecture.items())),
                "architecture_proxy_means": proxy_means,
                "architecture_truth_means": truth_means,
                **rank,
                "bootstrap": bootstrap,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "audit": "proxy_reliability_gate0",
        "truth_budget": int(truth_budget),
        "unit_of_analysis": "architecture",
        "input_summary": {
            "row_n": len(observations),
            "completed_row_n": len(completed),
            "architecture_n": len(architectures),
            "seed_n": len(seeds),
            "outer_fold_n": len(folds),
            "architectures": architectures,
            "seeds": seeds,
            "outer_folds": folds,
            "metrics": metrics,
        },
        "variance_decomposition": variance,
        "multi_fidelity": fidelity,
    }
    result["gate"] = _evaluate_classification_gate(result, gate_config or {})
    return result


def _evaluate_classification_gate(
    analysis: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_architectures = int(config.get("min_architectures", 40))
    minimum_seeds = int(config.get("min_seeds", 5))
    minimum_outer_folds = int(config.get("min_outer_folds", 5))
    target_metric = str(config.get("metric", "macro_f1"))
    minimum_icc = float(config.get("min_icc_relative_mean", 0.60))
    minimum_tau = float(config.get("min_kendall_tau_b", 0.30))
    minimum_pairwise = float(config.get("min_pairwise_accuracy", 0.65))
    maximum_regret = float(config.get("max_regret_at5", 0.02))
    use_bootstrap_bounds = bool(config.get("use_bootstrap_bounds", True))
    minimum_bootstrap_iterations = int(
        config.get("min_bootstrap_iterations", 1)
    )
    required_budgets = sorted(
        {int(value) for value in config.get("required_budgets", ())}
    )
    required_top_key = "regret_at5"

    summary = analysis["input_summary"]
    protocol_reasons: list[str] = []
    if int(summary["architecture_n"]) < minimum_architectures:
        protocol_reasons.append(
            f"architecture_n={summary['architecture_n']} < {minimum_architectures}"
        )
    if int(summary["seed_n"]) < minimum_seeds:
        protocol_reasons.append(f"seed_n={summary['seed_n']} < {minimum_seeds}")
    if int(summary["outer_fold_n"]) < minimum_outer_folds:
        protocol_reasons.append(
            f"outer_fold_n={summary['outer_fold_n']} < {minimum_outer_folds}"
        )

    variance = analysis["variance_decomposition"].get(target_metric, {})
    if not variance.get("estimable"):
        protocol_reasons.append(
            "complete architecture x seed x outer-fold truth grid is missing "
            f"for metric={target_metric}"
        )
    variance_bootstrap_iterations = int(
        variance.get("bootstrap", {}).get("iterations", 0)
    )
    if (
        use_bootstrap_bounds
        and variance_bootstrap_iterations < minimum_bootstrap_iterations
    ):
        protocol_reasons.append(
            "ICC bootstrap iterations "
            f"{variance_bootstrap_iterations} < {minimum_bootstrap_iterations}"
        )
    icc_point = float(variance.get("icc_relative_mean_observed", 0.0))
    icc_ci = (
        variance.get("bootstrap", {})
        .get("icc_relative_mean_observed", {})
    )
    icc_decision_value = (
        float(icc_ci.get("ci95_low", float("-inf")))
        if use_bootstrap_bounds
        else icc_point
    )
    signal_pass = bool(
        variance.get("estimable") and icc_decision_value >= minimum_icc
    )
    usable: list[dict[str, Any]] = []
    for row in analysis["multi_fidelity"]:
        if not row.get("estimable") or row.get("metric") != target_metric:
            continue
        ci = row.get("bootstrap", {}).get("ci95", {})
        tau_decision = (
            float(ci.get("kendall_tau_b", {}).get("low", float("-inf")))
            if use_bootstrap_bounds
            else float(row.get("kendall_tau_b", -1.0))
        )
        pairwise_decision = (
            float(ci.get("pairwise_accuracy", {}).get("low", float("-inf")))
            if use_bootstrap_bounds
            else float(row.get("pairwise_accuracy") or 0.0)
        )
        regret_decision = (
            float(ci.get(required_top_key, {}).get("high", float("inf")))
            if use_bootstrap_bounds
            else float(row.get(required_top_key, float("inf")))
        )
        passed = (
            tau_decision >= minimum_tau
            and pairwise_decision >= minimum_pairwise
            and regret_decision <= maximum_regret
        )
        usable.append(
            {
                "proxy_name": row["proxy_name"],
                "budget": row["budget"],
                "pass": passed,
                "kendall_tau_b": row.get("kendall_tau_b"),
                "pairwise_accuracy": row.get("pairwise_accuracy"),
                required_top_key: row.get(required_top_key),
                "decision_values": {
                    "kendall_tau_b": tau_decision,
                    "pairwise_accuracy": pairwise_decision,
                    required_top_key: regret_decision,
                },
            }
        )

    expected_cells = (
        int(summary["architecture_n"])
        * int(summary["seed_n"])
        * int(summary["outer_fold_n"])
    )
    for budget in required_budgets:
        source_rows = [
            row
            for row in analysis["multi_fidelity"]
            if row.get("estimable")
            and row.get("metric") == target_metric
            and int(row.get("budget", -1)) == budget
        ]
        complete = any(
            int(row.get("matched_cell_n", 0)) == expected_cells
            for row in source_rows
        )
        if not complete:
            protocol_reasons.append(
                f"budget={budget} lacks a complete matched proxy/truth grid "
                f"for metric={target_metric}"
            )
        elif use_bootstrap_bounds and not any(
            int(row.get("bootstrap", {}).get("iterations", 0))
            >= minimum_bootstrap_iterations
            for row in source_rows
        ):
            protocol_reasons.append(
                f"budget={budget} bootstrap iterations are below "
                f"{minimum_bootstrap_iterations}"
            )

    if protocol_reasons:
        status = "not_ready"
    elif not signal_pass:
        status = "fail_no_stable_architecture_signal"
    elif not any(item["pass"] for item in usable):
        status = "fail_no_usable_proxy_budget"
    else:
        status = "pass"
    passing = [item for item in usable if item["pass"]]
    earliest = (
        min(passing, key=lambda item: (int(item["budget"]), str(item["proxy_name"])))
        if passing
        else None
    )
    return {
        "status": status,
        "protocol_complete": not protocol_reasons,
        "protocol_reasons": protocol_reasons,
        "architecture_signal_pass": signal_pass,
        "architecture_signal_decision_value": icc_decision_value,
        "target_metric": target_metric,
        "decision_basis": (
            "bootstrap_ci95_lower_for_icc_tau_pairwise_and_upper_for_regret"
            if use_bootstrap_bounds
            else "point_estimates"
        ),
        "thresholds": {
            "min_architectures": minimum_architectures,
            "min_seeds": minimum_seeds,
            "min_outer_folds": minimum_outer_folds,
            "min_icc_relative_mean": minimum_icc,
            "min_kendall_tau_b": minimum_tau,
            "min_pairwise_accuracy": minimum_pairwise,
            "max_regret_at5": maximum_regret,
            "required_budgets": required_budgets,
            "use_bootstrap_bounds": use_bootstrap_bounds,
            "min_bootstrap_iterations": minimum_bootstrap_iterations,
        },
        "proxy_budget_decisions": usable,
        "earliest_usable_proxy": earliest,
    }


def _numeric_pairs(
    rows: Sequence[Mapping[str, Any]],
    proxy_column: str,
    truth_column: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    architectures: list[str] = []
    proxy: list[float] = []
    truth: list[float] = []
    for row in rows:
        left = _optional_float(row.get(proxy_column))
        right = _optional_float(row.get(truth_column))
        if left is None or right is None:
            continue
        architectures.append(str(row["architecture_id"]))
        proxy.append(left)
        truth.append(right)
    return architectures, np.asarray(proxy), np.asarray(truth)


def _pareto_front(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    suffix: str,
) -> set[str]:
    usable: list[tuple[str, np.ndarray]] = []
    for row in rows:
        if not _as_bool(row[f"{suffix}_feasible"]):
            continue
        values = [_optional_float(row.get(f"{suffix}_{column}")) for column in columns]
        if any(value is None for value in values):
            continue
        usable.append(
            (str(row["architecture_id"]), np.asarray(values, dtype=float))
        )
    front: set[str] = set()
    for index, (architecture, values) in enumerate(usable):
        dominated = False
        for other_index, (_, other) in enumerate(usable):
            if index == other_index:
                continue
            if np.all(other <= values) and np.any(other < values):
                dominated = True
                break
        if not dominated:
            front.add(architecture)
    return front


def analyze_hardware_reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("latency_ms", "dsp", "bram", "lut"),
    pareto_metrics: Sequence[str] = ("latency_ms", "dsp", "bram", "lut"),
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20260704,
) -> dict[str, Any]:
    """Compare search-time hardware estimates with HLS/route/board truth."""
    architecture_ids = [str(row["architecture_id"]) for row in rows]
    if len(set(architecture_ids)) != len(architecture_ids):
        raise ValueError("hardware observations require one row per architecture")

    metric_results: dict[str, Any] = {}
    rng = random.Random(bootstrap_seed)
    for metric in metrics:
        ids, proxy, truth = _numeric_pairs(
            rows,
            f"proxy_{metric}",
            f"truth_{metric}",
        )
        if len(ids) < 2:
            metric_results[metric] = {
                "estimable": False,
                "matched_n": len(ids),
                "reason": "fewer than two matched proxy/truth values",
            }
            continue
        errors = proxy - truth
        spearman = stats.spearmanr(proxy, truth)
        kendall = stats.kendalltau(proxy, truth, variant="b")
        bootstrap_samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(max(1, int(bootstrap_iterations))):
            draw = [rng.randrange(len(ids)) for _ in ids]
            sampled_proxy = proxy[draw]
            sampled_truth = truth[draw]
            sampled_errors = sampled_proxy - sampled_truth
            rho = stats.spearmanr(sampled_proxy, sampled_truth).statistic
            tau = stats.kendalltau(
                sampled_proxy, sampled_truth, variant="b"
            ).statistic
            bootstrap_samples["mae"].append(float(np.abs(sampled_errors).mean()))
            bootstrap_samples["rmse"].append(
                float(np.sqrt(np.square(sampled_errors).mean()))
            )
            if math.isfinite(float(rho)):
                bootstrap_samples["spearman_rho"].append(float(rho))
            if math.isfinite(float(tau)):
                bootstrap_samples["kendall_tau_b"].append(float(tau))
        metric_results[metric] = {
            "estimable": True,
            "matched_n": len(ids),
            "architecture_ids": ids,
            "bias_proxy_minus_truth": float(errors.mean()),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "mape": (
                float((np.abs(errors[truth != 0] / truth[truth != 0])).mean())
                if np.any(truth != 0)
                else None
            ),
            "spearman_rho": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "kendall_tau_b": float(kendall.statistic),
            "kendall_pvalue": float(kendall.pvalue),
            "bootstrap_ci95": {
                key: {
                    "low": _quantile(values, 0.025),
                    "high": _quantile(values, 0.975),
                }
                for key, values in bootstrap_samples.items()
                if values
            },
        }

    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in rows:
        predicted = _as_bool(row["proxy_feasible"])
        observed = _as_bool(row["truth_feasible"])
        if predicted and observed:
            confusion["tp"] += 1
        elif predicted and not observed:
            confusion["fp"] += 1
        elif not predicted and observed:
            confusion["fn"] += 1
        else:
            confusion["tn"] += 1
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    accuracy = (tp + tn) / len(rows) if rows else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )

    proxy_pareto = _pareto_front(rows, pareto_metrics, suffix="proxy")
    truth_pareto = _pareto_front(rows, pareto_metrics, suffix="truth")
    overlap = proxy_pareto & truth_pareto
    pareto_recall = len(overlap) / len(truth_pareto) if truth_pareto else None
    pareto_precision = len(overlap) / len(proxy_pareto) if proxy_pareto else None

    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "hardware_reliability_gate0",
        "unit_of_analysis": "architecture",
        "architecture_n": len(rows),
        "truth_sources": {
            column: sorted(
                {
                    str(row.get(column, "")).strip()
                    for row in rows
                    if str(row.get(column, "")).strip()
                }
            )
            for column in (
                "truth_latency_source",
                "truth_resource_source",
                "truth_feasibility_source",
            )
        },
        "metrics": metric_results,
        "feasibility": {
            **confusion,
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        },
        "pareto": {
            "metrics": list(pareto_metrics),
            "proxy_front": sorted(proxy_pareto),
            "truth_front": sorted(truth_pareto),
            "overlap": sorted(overlap),
            "recall": pareto_recall,
            "precision": pareto_precision,
        },
        "evidence_boundary": (
            "truth columns must identify HLS/route/board sources explicitly; "
            "estimated power is not measured power"
        ),
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _format_ci(
    interval: Optional[Mapping[str, Any]],
    *,
    low_key: str = "low",
    high_key: str = "high",
) -> str:
    if not interval:
        return "NA"
    low = interval.get(low_key)
    high = interval.get(high_key)
    if low is None or high is None:
        return "NA"
    return f"[{_format_number(low)}, {_format_number(high)}]"


def write_proxy_reliability_bundle(
    output_dir: str | Path,
    *,
    classification: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    classification_source: Optional[str] = None,
    hardware_source: Optional[str] = None,
) -> dict[str, str]:
    """Write the strict analysis bundle and real figures when data exist."""
    destination = Path(output_dir)
    figures_dir = destination / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "classification": classification,
        "hardware": hardware,
        "sources": {
            "classification": classification_source,
            "hardware": hardware_source,
        },
    }
    json_path = destination / "audit_summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Proxy Reliability Audit — Gate 0",
        "",
        f"- Gate status: `{classification['gate']['status']}`",
        f"- Unit of analysis: `{classification['unit_of_analysis']}`",
        f"- Architectures: {classification['input_summary']['architecture_n']}",
        f"- Seeds: {classification['input_summary']['seed_n']}",
        f"- Outer folds: {classification['input_summary']['outer_fold_n']}",
        f"- Truth budget: {classification['truth_budget']} epochs",
        "",
        "## Architecture signal",
        "",
        "| metric | estimable | relative ICC single | relative ICC 5-seed / 1-fold | relative ICC 5-seed / 5-fold | full-mean ICC 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, row in classification["variance_decomposition"].items():
        report_lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    _format_number(bool(row.get("estimable"))),
                    _format_number(row.get("icc_relative_single")),
                    _format_number(
                        row.get("icc_relative_mean_seeds_single_fold")
                    ),
                    _format_number(row.get("icc_relative_mean_observed")),
                    _format_ci(
                        row.get("bootstrap", {}).get(
                            "icc_relative_mean_observed"
                        ),
                        low_key="ci95_low",
                        high_key="ci95_high",
                    ),
                ]
            )
            + " |"
        )
    report_lines += [
        "",
        "## Multi-fidelity ranking",
        "",
        "| proxy | budget | metric | n arch | Spearman rho | Kendall tau-b (95% CI) | pairwise acc (95% CI) | Top-5 recall | Top-10 recall | NDCG@5 | NDCG@10 | regret@5 (95% CI) | regret@10 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in classification["multi_fidelity"]:
        if not row.get("estimable"):
            continue
        ci = row.get("bootstrap", {}).get("ci95", {})
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["proxy_name"]),
                    str(row["budget"]),
                    str(row["metric"]),
                    str(row["architecture_n"]),
                    _format_number(row.get("spearman_rho")),
                    (
                        f"{_format_number(row.get('kendall_tau_b'))} "
                        f"{_format_ci(ci.get('kendall_tau_b'))}"
                    ),
                    (
                        f"{_format_number(row.get('pairwise_accuracy'))} "
                        f"{_format_ci(ci.get('pairwise_accuracy'))}"
                    ),
                    _format_number(row.get("top5_recall")),
                    _format_number(row.get("top10_recall")),
                    _format_number(row.get("ndcg_at5")),
                    _format_number(row.get("ndcg_at10")),
                    (
                        f"{_format_number(row.get('regret_at5'))} "
                        f"{_format_ci(ci.get('regret_at5'))}"
                    ),
                    _format_number(row.get("regret_at10")),
                ]
            )
            + " |"
        )
    report_lines += [
        "",
        "## Decision",
        "",
        "The audit is a prerequisite for comparing RL, Random, Proxyless, or a "
        "relative-ranking predictor. `not_ready` means the required evidence "
        "grid is incomplete; it is not a pass and not evidence of equivalence.",
    ]
    if classification["gate"]["protocol_reasons"]:
        report_lines += ["", "Missing formal evidence:"]
        report_lines.extend(
            f"- {reason}" for reason in classification["gate"]["protocol_reasons"]
        )
    if hardware is None:
        report_lines += [
            "",
            "## Hardware reliability",
            "",
            "Not analyzed: no separate hardware truth table was provided.",
        ]
    else:
        report_lines += [
            "",
            "## Hardware reliability",
            "",
            f"- Architectures: {hardware['architecture_n']}",
            f"- Feasibility balanced accuracy: "
            f"{_format_number(hardware['feasibility']['balanced_accuracy'])}",
            f"- Pareto recall: {_format_number(hardware['pareto']['recall'])}",
            f"- Pareto precision: {_format_number(hardware['pareto']['precision'])}",
            "",
            "Classification and hardware evidence remain separate; no scalar "
            "reward is used to hide disagreement between them.",
            "",
            "| metric | n | bias (proxy-truth) | MAE | RMSE | MAPE | Spearman rho | Kendall tau-b |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for metric, row in hardware["metrics"].items():
            if not row.get("estimable"):
                continue
            report_lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric),
                        str(row["matched_n"]),
                        _format_number(row.get("bias_proxy_minus_truth")),
                        _format_number(row.get("mae")),
                        _format_number(row.get("rmse")),
                        _format_number(row.get("mape")),
                        _format_number(row.get("spearman_rho")),
                        _format_number(row.get("kendall_tau_b")),
                    ]
                )
                + " |"
            )
    report_path = destination / "analysis-report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stats_lines = [
        "# Statistical appendix",
        "",
        "## Unit and resampling",
        "",
        "- Primary unit: architecture.",
        "- Ranking intervals: architecture-primary hierarchical bootstrap, then outer fold, then seed.",
        "- Pairwise architecture comparisons are descriptive decisions, not independent replicates.",
        "- ICC: balanced three-way architecture × seed × outer-fold method-of-moments decomposition.",
        "",
        "## Gate configuration",
        "",
        "```json",
        json.dumps(classification["gate"]["thresholds"], indent=2),
        "```",
        "",
        "## Variance decomposition",
        "",
        "```json",
        json.dumps(classification["variance_decomposition"], indent=2),
        "```",
    ]
    stats_path = destination / "stats-appendix.md"
    stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    generated_figures = _write_figures(figures_dir, classification)
    catalog_lines = [
        "# Figure catalog",
        "",
    ]
    if not generated_figures:
        catalog_lines += [
            "No figure was generated because fewer than two estimable proxy budgets were available.",
        ]
    else:
        for figure in generated_figures:
            catalog_lines += [
                f"## {figure['filename']}",
                "",
                f"- Purpose: {figure['purpose']}",
                f"- Data source: {classification_source or 'classification observations'}",
                f"- Reader should notice: {figure['observation']}",
                f"- Decision impact: {figure['decision']}",
                f"- Caveat: {figure['caveat']}",
                "",
            ]
    catalog_path = destination / "figure-catalog.md"
    catalog_path.write_text("\n".join(catalog_lines) + "\n", encoding="utf-8")
    return {
        "analysis_report": str(report_path),
        "stats_appendix": str(stats_path),
        "figure_catalog": str(catalog_path),
        "audit_summary": str(json_path),
    }


def _write_figures(
    figures_dir: Path,
    classification: Mapping[str, Any],
) -> list[dict[str, str]]:
    estimable = [
        row for row in classification["multi_fidelity"] if row.get("estimable")
    ]
    if not estimable:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    generated: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in estimable:
        grouped[(str(row["proxy_name"]), str(row["metric"]))].append(row)

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for (proxy_name, metric), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["budget"]))
        x_values = [int(row["budget"]) for row in ordered]
        y_values = [float(row["kendall_tau_b"]) for row in ordered]
        intervals = [
            row.get("bootstrap", {})
            .get("ci95", {})
            .get("kendall_tau_b", {})
            for row in ordered
        ]
        lower = [
            max(0.0, y - float(interval.get("low", y)))
            for y, interval in zip(y_values, intervals)
        ]
        upper = [
            max(0.0, float(interval.get("high", y)) - y)
            for y, interval in zip(y_values, intervals)
        ]
        axis.errorbar(
            x_values,
            y_values,
            yerr=np.asarray([lower, upper]),
            marker="o",
            capsize=3,
            label=f"{proxy_name}:{metric}",
        )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Proxy training budget (epochs)")
    axis.set_ylabel("Kendall tau-b vs full-training truth")
    axis.set_title("Multi-fidelity rank reliability")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    curve_path = figures_dir / "figure-01-multifidelity-kendall.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)
    generated.append(
        {
            "filename": curve_path.name,
            "purpose": "Locate the earliest budget with stable architecture ordering.",
            "observation": "Whether rank agreement rises above noise as budget increases.",
            "decision": "Only budgets passing the pre-registered gate may drive algorithm comparison.",
            "caveat": "Confidence depends on a complete architecture × seed × outer-fold grid.",
        }
    )

    best_row = max(
        estimable,
        key=lambda row: (
            float(row.get("kendall_tau_b", float("-inf"))),
            -int(row["budget"]),
        ),
    )
    ids = sorted(
        set(best_row["architecture_proxy_means"])
        & set(best_row["architecture_truth_means"])
    )
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    axis.scatter(
        [best_row["architecture_proxy_means"][key] for key in ids],
        [best_row["architecture_truth_means"][key] for key in ids],
        color="#3366cc",
        alpha=0.8,
    )
    axis.set_xlabel(
        f"{best_row['proxy_name']} proxy ({best_row['budget']} epochs)"
    )
    axis.set_ylabel(f"{best_row['metric']} full-training outer-fold truth")
    axis.set_title("Proxy ranking versus truth")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    scatter_path = figures_dir / "figure-02-best-proxy-vs-truth.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    generated.append(
        {
            "filename": scatter_path.name,
            "purpose": "Show architecture-level proxy/truth agreement for the strongest observed budget.",
            "observation": "Outliers and top-region inversions that a single correlation coefficient hides.",
            "decision": "Large top-region inversions argue against using the proxy even when global correlation is positive.",
            "caveat": "The plotted architecture means do not replace seed/fold uncertainty.",
        }
    )
    return generated
