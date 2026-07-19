"""Prespecified paired inference for benchmark fold-seed units."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _paired_arrays(
    left: Sequence[float], right: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1 or left_array.shape != right_array.shape:
        raise ValueError("paired inputs must be equal-length one-dimensional vectors")
    if left_array.size < 2 or not np.all(np.isfinite(left_array - right_array)):
        raise ValueError("paired inference requires at least two finite pairs")
    return left_array, right_array


def paired_stratified_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    strata: Sequence[str | int],
    *,
    iterations: int = 10_000,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Bootstrap the paired mean difference within prespecified strata.

    The unit supplied by the caller remains the fold-seed pair. Resampling is
    performed independently inside each fold stratum and then pooled, which
    preserves the campaign's fold composition.
    """

    left_array, right_array = _paired_arrays(left, right)
    strata_array = np.asarray(strata)
    if strata_array.ndim != 1 or strata_array.size != left_array.size:
        raise ValueError("strata must provide one label per paired unit")
    if iterations < 1_000:
        raise ValueError("formal bootstrap requires at least 1,000 iterations")
    differences = left_array - right_array
    groups = [np.flatnonzero(strata_array == value) for value in np.unique(strata_array)]
    if any(group.size == 0 for group in groups):
        raise ValueError("empty bootstrap stratum")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = [rng.choice(group, size=group.size, replace=True) for group in groups]
        bootstrap[iteration] = float(np.mean(differences[np.concatenate(sampled)]))
    effect = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    return {
        "n": int(differences.size),
        "iterations": int(iterations),
        "seed": int(seed),
        "difference_definition": "left_minus_right",
        "mean_difference": effect,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "cohens_dz": effect / standard_deviation if standard_deviation > 0 else None,
        "strata": [str(value) for value in np.unique(strata_array)],
    }


def paired_permutation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Two-sided paired sign-flip permutation test of the mean difference."""

    left_array, right_array = _paired_arrays(left, right)
    differences = left_array - right_array
    observed = abs(float(np.mean(differences)))
    n = int(differences.size)
    exact = n <= 16
    if exact:
        statistics = [
            abs(float(np.mean(differences * np.asarray(signs, dtype=np.float64))))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        ]
    else:
        if iterations < 1_000:
            raise ValueError("formal permutation test requires at least 1,000 iterations")
        rng = np.random.default_rng(seed)
        statistics = [
            abs(float(np.mean(differences * rng.choice((-1.0, 1.0), size=n))))
            for _ in range(iterations)
        ]
    exceedances = int(np.sum(np.asarray(statistics) >= observed - 1e-15))
    p_value = (
        exceedances / len(statistics)
        if exact
        else (exceedances + 1) / (len(statistics) + 1)
    )
    return {
        "n": n,
        "observed_mean_difference": float(np.mean(differences)),
        "p_value": float(p_value),
        "exact": exact,
        "permutations": len(statistics),
        "seed": None if exact else int(seed),
    }


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error correction."""

    if not raw_p_values:
        return {}
    if any(not 0.0 <= float(value) <= 1.0 for value in raw_p_values.values()):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(raw_p_values.items(), key=lambda item: float(item[1]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running_max
    return adjusted
