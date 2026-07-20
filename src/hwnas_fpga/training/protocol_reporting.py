"""Reporting and provenance helpers for the frozen evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (42, 43, 44)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_claimability(
    *,
    folds: Sequence[int],
    seeds: Sequence[int],
    completed_pairs: Iterable[tuple[int, int]],
    selection_provenance: str,
    outer_validation_used_for_selection: bool = False,
    provenance_complete: bool = True,
    provenance_fingerprints: Sequence[str] | None = None,
    group_split_available: bool = False,
    protocol_context_sha256: str | None = None,
    provenance_contexts: Sequence[str] | None = None,
    source_freeze_verified: bool = True,
) -> dict[str, Any]:
    normalized_folds = tuple(sorted(set(int(value) for value in folds)))
    normalized_seeds = tuple(sorted(set(int(value) for value in seeds)))
    expected_pairs = {
        (fold, seed) for fold in REQUIRED_FOLDS for seed in REQUIRED_SEEDS
    }
    completed_pair_list = [
        (int(fold), int(seed)) for fold, seed in completed_pairs
    ]
    observed_pairs = set(completed_pair_list)
    duplicate_pairs = len(completed_pair_list) != len(observed_pairs)
    fingerprints = [str(value) for value in (provenance_fingerprints or [])]
    fingerprint_complete = (
        provenance_fingerprints is not None
        and len(fingerprints) == len(completed_pair_list)
        and bool(fingerprints)
        and all(len(value) == 64 for value in fingerprints)
        and len(set(fingerprints)) == 1
    )
    contexts = [str(value) for value in (provenance_contexts or [])]
    context_complete = True
    if protocol_context_sha256 is not None or provenance_contexts is not None:
        context_complete = (
            len(str(protocol_context_sha256 or "")) == 64
            and len(contexts) == len(completed_pair_list)
            and bool(contexts)
            and all(len(value) == 64 for value in contexts)
            and len(set(contexts)) == 1
            and contexts[0] == str(protocol_context_sha256)
        )
    protocol_complete = (
        normalized_folds == REQUIRED_FOLDS
        and normalized_seeds == REQUIRED_SEEDS
        and observed_pairs == expected_pairs
        and not duplicate_pairs
        and not outer_validation_used_for_selection
        and provenance_complete
        and fingerprint_complete
        and context_complete
        and source_freeze_verified
    )
    legacy_fold0_only = normalized_folds == (0,)
    legacy_selected = selection_provenance == "legacy_fold0_selected"
    if protocol_complete and legacy_selected:
        claim_scope = "frozen_legacy_selected_architecture_benchmark"
    elif protocol_complete:
        claim_scope = "frozen_protocol_model_evaluation"
    elif legacy_fold0_only:
        claim_scope = "legacy_fold0_proxy"
    else:
        claim_scope = "development_partial_protocol"
    warnings: list[str] = []
    if legacy_selected:
        warnings.append(
            "Architecture was selected with the historical fold-0 workflow; "
            "this run cannot establish unbiased NAS method generalization."
        )
    if not protocol_complete:
        warnings.append(
            "Formal reporting requires exactly folds 0-4, seeds 42,43,44, "
            "no outer-validation selection, and complete provenance."
        )
    if duplicate_pairs:
        warnings.append("Duplicate fold/seed records were found.")
    if not fingerprint_complete:
        warnings.append(
            "Formal reporting requires one identical 64-character run fingerprint "
            "for every completed fold/seed record."
        )
    if not context_complete:
        warnings.append(
            "Formal reporting requires one identical 64-character protocol context "
            "hash in every completed fold/seed record."
        )
    if not source_freeze_verified:
        warnings.append(
            "Formal reporting requires a verified source-freeze manifest and its "
            "retained snapshot archive."
        )
    return {
        "claimable": protocol_complete,
        "protocol_complete": protocol_complete,
        "claim_scope": claim_scope,
        "legacy": legacy_fold0_only,
        "selection_provenance": selection_provenance,
        "outer_validation_used_for_selection": bool(
            outer_validation_used_for_selection
        ),
        "provenance_complete": bool(provenance_complete),
        "fingerprint_complete": fingerprint_complete,
        "protocol_context_complete": context_complete,
        "source_freeze_verified": bool(source_freeze_verified),
        "run_fingerprint": fingerprints[0] if fingerprint_complete else None,
        "duplicate_pairs": duplicate_pairs,
        "nas_generalization_claimable": protocol_complete and not legacy_selected,
        "group_split_available": bool(group_split_available),
        "group_generalization_claimable": bool(protocol_complete and group_split_available),
        "required_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "expected_run_count": len(expected_pairs),
        "observed_run_count": len(observed_pairs),
        "missing_pairs": [
            {"fold": fold, "seed": seed}
            for fold, seed in sorted(expected_pairs - observed_pairs)
        ],
        "unexpected_pairs": [
            {"fold": fold, "seed": seed}
            for fold, seed in sorted(observed_pairs - expected_pairs)
        ],
        "warnings": warnings,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hierarchical_paired_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    iterations: int = 10_000,
    seed: int = 20260704,
) -> dict[str, Any]:
    """Bootstrap paired ``left - right`` differences by fold then seed."""
    left_index = {
        (int(row["fold"]), int(row["seed"])): float(row[metric])
        for row in left
    }
    right_index = {
        (int(row["fold"]), int(row["seed"])): float(row[metric])
        for row in right
    }
    common = sorted(set(left_index) & set(right_index))
    if not common:
        raise ValueError("paired bootstrap found no common fold/seed pairs")
    by_fold: dict[int, list[float]] = {}
    for fold, run_seed in common:
        by_fold.setdefault(fold, []).append(
            left_index[(fold, run_seed)] - right_index[(fold, run_seed)]
        )
    folds = sorted(by_fold)
    point_differences = [
        difference for fold in folds for difference in by_fold[fold]
    ]
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(max(1, int(iterations))):
        sampled: list[float] = []
        for _fold_draw in range(len(folds)):
            fold = rng.choice(folds)
            seed_differences = by_fold[fold]
            sampled.extend(
                rng.choice(seed_differences)
                for _seed_draw in range(len(seed_differences))
            )
        bootstrap_means.append(statistics.fmean(sampled))
    return {
        "metric": metric,
        "direction": "left_minus_right",
        "paired_n": len(point_differences),
        "fold_n": len(folds),
        "mean_difference": statistics.fmean(point_differences),
        "ci95_low": _quantile(bootstrap_means, 0.025),
        "ci95_high": _quantile(bootstrap_means, 0.975),
        "iterations": max(1, int(iterations)),
        "bootstrap_seed": seed,
    }
