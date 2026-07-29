"""Inner-only selection helpers for the NKSID Protocol V2 gate."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


def dhash64(path: str | Path) -> int:
    """Return a frozen 64-bit horizontal difference hash."""

    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def hamming64(left: int, right: int) -> int:
    return int((int(left) ^ int(right)).bit_count())


class _DisjointSet:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def infer_hash_groups(
    samples: Sequence[tuple[str, int]],
    indices: Sequence[int],
    *,
    hamming_threshold: int = 4,
    precomputed_hashes: Mapping[int, int] | None = None,
) -> dict[int, str]:
    """Infer same-class connected components without claiming real groups."""

    threshold = int(hamming_threshold)
    if threshold < 0 or threshold > 64:
        raise ValueError("hamming_threshold must be in [0, 64]")
    normalized = sorted({int(index) for index in indices})
    hashes = {
        index: (
            int(precomputed_hashes[index])
            if precomputed_hashes is not None and index in precomputed_hashes
            else dhash64(samples[index][0])
        )
        for index in normalized
    }
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in normalized:
        by_class[int(samples[index][1])].append(index)
    assignments: dict[int, str] = {}
    for label, class_indices in sorted(by_class.items()):
        dsu = _DisjointSet(class_indices)
        for position, left in enumerate(class_indices):
            for right in class_indices[position + 1 :]:
                if hamming64(hashes[left], hashes[right]) <= threshold:
                    dsu.union(left, right)
        components: dict[int, list[int]] = defaultdict(list)
        for index in class_indices:
            components[dsu.find(index)].append(index)
        ordered = sorted(
            (sorted(component) for component in components.values()),
            key=lambda component: (component[0], len(component)),
        )
        for group_number, component in enumerate(ordered):
            group_id = f"class{label}_dhash_g{group_number:04d}"
            for index in component:
                assignments[index] = group_id
    if set(assignments) != set(normalized):
        raise AssertionError("hash grouping did not assign every requested sample")
    return assignments


def group_stress_split(
    samples: Sequence[tuple[str, int]],
    outer_train_indices: Sequence[int],
    group_ids: Mapping[int, str],
    *,
    seed: int,
    fraction: float = 0.15,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Build a deterministic class-stratified inner split with whole groups."""

    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    by_class_group: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in sorted({int(value) for value in outer_train_indices}):
        group_id = str(group_ids[index])
        by_class_group[int(samples[index][1])][group_id].append(index)

    inner: list[int] = []
    train: list[int] = []
    class_rows: dict[str, Any] = {}
    for label, groups in sorted(by_class_group.items()):
        ordered_groups = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(
                f"{seed}|{label}|{group_id}".encode("utf-8")
            ).hexdigest(),
        )
        class_count = sum(len(groups[group_id]) for group_id in ordered_groups)
        target = max(1, int(round(class_count * fraction)))
        selected: list[str] = []
        selected_count = 0
        for group_id in ordered_groups:
            groups_left = len(ordered_groups) - len(selected)
            if groups_left <= 1:
                break
            current_distance = abs(selected_count - target)
            candidate_distance = abs(
                selected_count + len(groups[group_id]) - target
            )
            if selected_count == 0 or candidate_distance <= current_distance:
                selected.append(group_id)
                selected_count += len(groups[group_id])
            if selected_count >= target:
                break
        if not selected:
            selected = [ordered_groups[0]]
        selected_set = set(selected)
        class_inner = [
            index
            for group_id in ordered_groups
            if group_id in selected_set
            for index in groups[group_id]
        ]
        class_train = [
            index
            for group_id in ordered_groups
            if group_id not in selected_set
            for index in groups[group_id]
        ]
        if not class_train or not class_inner:
            raise ValueError(
                f"class {label} cannot form non-empty group stress split"
            )
        inner.extend(class_inner)
        train.extend(class_train)
        class_rows[str(label)] = {
            "outer_train_count": class_count,
            "target_inner_count": target,
            "actual_inner_count": len(class_inner),
            "train_count": len(class_train),
            "group_count": len(ordered_groups),
            "inner_group_ids": sorted(selected_set),
        }
    train_tuple = tuple(sorted(train))
    inner_tuple = tuple(sorted(inner))
    if set(train_tuple) & set(inner_tuple):
        raise AssertionError("stress train and inner splits overlap")
    if set(train_tuple) | set(inner_tuple) != {
        int(value) for value in outer_train_indices
    }:
        raise AssertionError("stress split does not cover outer-train exactly")
    train_groups = {group_ids[index] for index in train_tuple}
    inner_groups = {group_ids[index] for index in inner_tuple}
    if train_groups & inner_groups:
        raise AssertionError("inferred hash group crosses train/inner boundary")
    metadata = {
        "method": "same_class_dhash64_connected_components",
        "claim_boundary": (
            "Inferred perceptual-hash groups are a near-duplicate pressure "
            "diagnostic, not real acquisition or mission groups."
        ),
        "seed": int(seed),
        "fraction": fraction,
        "train_count": len(train_tuple),
        "inner_count": len(inner_tuple),
        "classes": class_rows,
    }
    return train_tuple, inner_tuple, metadata


def paired_hierarchical_bootstrap(
    deltas: Mapping[int, Sequence[float]],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired fold/seed deltas while preserving fold clustering."""

    folds = sorted(int(fold) for fold in deltas)
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"expected folds 0..4, got {folds}")
    arrays = {fold: np.asarray(deltas[fold], dtype=float) for fold in folds}
    if any(array.size != 3 for array in arrays.values()):
        raise ValueError("each fold must contain exactly three paired seed deltas")
    observed = float(np.mean(np.concatenate(list(arrays.values()))))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=float)
    fold_array = np.asarray(folds, dtype=int)
    for iteration in range(int(iterations)):
        selected_folds = rng.choice(fold_array, size=len(folds), replace=True)
        selected_values: list[float] = []
        for fold in selected_folds:
            selected_values.extend(
                rng.choice(arrays[int(fold)], size=3, replace=True).tolist()
            )
        draws[iteration] = float(np.mean(selected_values))
    return {
        "method": "paired_hierarchical_bootstrap_fold_then_seed",
        "iterations": int(iterations),
        "seed": int(seed),
        "mean_delta": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "positive_fold_means": sum(
            float(arrays[fold].mean()) > 0.0 for fold in folds
        ),
        "per_fold_mean_delta": {
            str(fold): float(arrays[fold].mean()) for fold in folds
        },
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
