"""Pre-registered statistical and deployment gate for MixConv-v2."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence


REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (42, 43, 44)
FORMAL_RUN_COUNT = 15


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validated_deltas(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[float]]:
    if len(rows) != FORMAL_RUN_COUNT:
        raise ValueError(f"formal comparison requires {FORMAL_RUN_COUNT} paired rows")
    by_fold: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        fold = int(row["fold"])
        seed = int(row["seed"])
        key = (fold, seed)
        if fold not in REQUIRED_FOLDS or seed not in REQUIRED_SEEDS or key in seen:
            raise ValueError(f"invalid or duplicate formal pair: {key}")
        seen.add(key)
        by_fold[fold].append(
            float(row["candidate_macro_f1"]) - float(row["control_macro_f1"])
        )
    expected = {
        (fold, seed) for fold in REQUIRED_FOLDS for seed in REQUIRED_SEEDS
    }
    if seen != expected:
        raise ValueError("formal comparison does not cover the complete 5x3 grid")
    return dict(by_fold)


def exact_fold_sign_flip_p_value(fold_deltas: Sequence[float]) -> float:
    """Two-sided exact sign-flip p-value over independent fold means."""
    values = [float(value) for value in fold_deltas]
    if not values:
        raise ValueError("fold_deltas must not be empty")
    observed = abs(mean(values))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(mean(sign * value for sign, value in zip(signs, values)))
        exceed += statistic >= observed - 1e-15
        total += 1
    return exceed / total


def paired_hierarchical_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 10_000,
    bootstrap_seed: int = 20260723,
) -> dict[str, Any]:
    """Fold-stratified paired bootstrap with exact fold sign flipping."""
    if iterations < 1_000:
        raise ValueError("iterations must be at least 1000")
    by_fold = _validated_deltas(rows)
    fold_means = {fold: mean(by_fold[fold]) for fold in REQUIRED_FOLDS}
    all_deltas = [
        value for fold in REQUIRED_FOLDS for value in by_fold[fold]
    ]
    rng = random.Random(int(bootstrap_seed))
    samples: list[float] = []
    for _ in range(int(iterations)):
        sampled_folds = [rng.choice(REQUIRED_FOLDS) for _ in REQUIRED_FOLDS]
        fold_statistics = []
        for fold in sampled_folds:
            values = by_fold[fold]
            fold_statistics.append(
                mean(rng.choice(values) for _ in range(len(values)))
            )
        samples.append(mean(fold_statistics))
    fold_values = [fold_means[fold] for fold in REQUIRED_FOLDS]
    return {
        "method": "paired_hierarchical_bootstrap_fold_sign_flip",
        "paired_run_count": len(all_deltas),
        "folds": list(REQUIRED_FOLDS),
        "seeds": list(REQUIRED_SEEDS),
        "iterations": int(iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "macro_f1_mean_delta": mean(all_deltas),
        "stratified_bootstrap_ci95_lower": _quantile(samples, 0.025),
        "stratified_bootstrap_ci95_upper": _quantile(samples, 0.975),
        "fold_mean_deltas": {
            str(fold): fold_means[fold] for fold in REQUIRED_FOLDS
        },
        "positive_fold_count": sum(value > 0.0 for value in fold_values),
        "two_sided_fold_sign_flip_p_value": exact_fold_sign_flip_p_value(
            fold_values
        ),
        "p_value_role": (
            "descriptive_only_not_an_admission_gate; with five folds the minimum "
            "attainable two-sided exact p-value is 0.0625"
        ),
    }

def _comparison_gates(
    comparison: Mapping[str, Any],
    *,
    min_delta: float,
    noninferiority_margin: float | None = None,
) -> dict[str, bool]:
    complete = (
        int(comparison.get("paired_run_count", 0)) == FORMAL_RUN_COUNT
        and comparison.get("folds") == list(REQUIRED_FOLDS)
        and comparison.get("seeds") == list(REQUIRED_SEEDS)
    )
    lower = float(comparison.get("stratified_bootstrap_ci95_lower", -1.0))
    gates = {
        "complete_5x3_pairing": complete,
        "mean_delta_threshold": float(
            comparison.get("macro_f1_mean_delta", -1.0)
        )
        >= float(min_delta),
        "ci_lower_positive": lower > 0.0,
        "fold_direction_at_least_4_of_5": int(
            comparison.get("positive_fold_count", 0)
        )
        >= 4,
    }
    if noninferiority_margin is not None:
        gates["noninferior_ci"] = lower >= -abs(float(noninferiority_margin))
    return gates


def audit_mixconv_v2_admission(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Classify only complete evidence; missing fields remain explicitly blocked."""
    comparisons = manifest.get("comparisons", {})
    mix_k3 = comparisons.get("mixconv_v2_vs_mbconv_k3", {})
    mix_k5 = comparisons.get("mixconv_v2_vs_mbconv_k5", {})
    k5_k3 = comparisons.get("mbconv_k5_vs_mbconv_k3", {})
    mix_k3_gates = _comparison_gates(mix_k3, min_delta=0.01)
    mix_k5_accuracy = _comparison_gates(mix_k5, min_delta=0.0)
    mix_k5_noninferiority = (
        float(mix_k5.get("stratified_bootstrap_ci95_lower", -1.0)) >= -0.005
    )
    k5_k3_gates = _comparison_gates(k5_k3, min_delta=0.01)

    hardware = manifest.get("hardware", {})
    parity = manifest.get("int8_parity", {})
    robustness = manifest.get("robustness", {})
    provenance = manifest.get("provenance", {})
    mix_hardware_advantage = hardware.get("advantage_vs_mbconv_k5") is True
    mature_control_gate = all(k5_k3_gates.values())
    mix_vs_k5_gate = all(mix_k5_accuracy.values()) or (
        mix_k5_noninferiority and mix_hardware_advantage
    )
    deployment_gates = {
        "robustness_drop_within_0p01": float(
            robustness.get("mean_macro_f1_delta_vs_control", -1.0)
        )
        >= -0.01,
        "real_activation_parity": int(parity.get("real_activation_count", 0)) > 0,
        "boundary_tensor_parity": int(parity.get("boundary_tensor_count", 0)) > 0,
        "random_tensor_parity": int(parity.get("random_tensor_count", 0)) > 0,
        "integer_zero_mismatch": (
            int(parity.get("compared_element_count", 0)) > 0
            and int(parity.get("mismatch_count", -1)) == 0
        ),
        "hls_5ns_synthesis": hardware.get("hls_5ns_synthesis_pass") is True,
        "complete_route": hardware.get("complete_route_pass") is True,
        "resources_within_search_constraints": (
            hardware.get("resources_within_search_constraints") is True
        ),
        "measured_lut_traceable": hardware.get("measured_lut_traceable") is True,
        "formal_run_provenance_complete": (
            provenance.get("complete_15_run_records") is True
            and provenance.get("checkpoint_hashes_complete") is True
            and provenance.get("source_freeze_verified") is True
        ),
    }
    mix_accuracy_gate = all(mix_k3_gates.values()) and mix_vs_k5_gate
    complete = all(
        (
            bool(mix_k3),
            bool(mix_k5),
            bool(k5_k3),
            bool(robustness),
            bool(parity),
            bool(hardware),
            bool(provenance),
        )
    )
    if complete and mix_accuracy_gate and all(deployment_gates.values()):
        status = "ADMITTED"
    elif complete and mature_control_gate:
        status = "GENERAL_OP_SELECTED"
    elif complete:
        status = "NO_OPERATOR_GAIN"
    else:
        status = "NOT_READY"
    return {
        "schema_version": 1,
        "gate": "mixconv_v2_full_admission",
        "status": status,
        "may_enable_mixconv_v2": status == "ADMITTED",
        "mixconv_v2_vs_mbconv_k3_gates": mix_k3_gates,
        "mixconv_v2_vs_mbconv_k5_accuracy_gates": mix_k5_accuracy,
        "mixconv_v2_vs_mbconv_k5_noninferior": mix_k5_noninferiority,
        "mixconv_v2_vs_mbconv_k5_hardware_advantage": mix_hardware_advantage,
        "mbconv_k5_vs_mbconv_k3_gates": k5_k3_gates,
        "deployment_gates": deployment_gates,
        "p_values_are_hard_gates": False,
        "boundary": (
            "ADMITTED requires accuracy, robustness, bit-exact INT8/C-sim, "
            "5 ns HLS, complete route, resource, and provenance gates. A p-value "
            "is reported for context but cannot be the sole five-fold gate."
        ),
    }
