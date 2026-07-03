"""Evidence-first checks for the NKSID sample and split protocol."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any
import re

from .dataset import _resolve_nksid_root, _resolve_sample_path


def _read_samples(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = root / "train_abs.txt"
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Malformed sample row at {source}:{line_number}")
        path = _resolve_sample_path(root, parts[0])
        rows.append(
            {
                "index": len(rows),
                "relative_path": parts[0],
                "path": path,
                "label": int(parts[1]),
            }
        )
    if not rows:
        raise ValueError(f"No samples found in {source}")
    return rows


def _read_folds(path: Path) -> list[list[int]]:
    folds: list[list[int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        indices = [int(value) for value in re.findall(r"-?\d+", line)]
        if indices:
            folds.append(indices)
    if not folds:
        raise ValueError(f"No fold records found in {path}")
    return folds


def _trailing_number(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def audit_nksid_protocol(
    data_dir: str | Path,
    *,
    fold: int = 0,
    neighbor_radius: int = 1,
    hash_files: bool = False,
) -> dict[str, Any]:
    """Audit sample integrity and the exact unit used by the supplied folds."""
    if neighbor_radius < 1:
        raise ValueError(f"neighbor_radius must be >= 1, got {neighbor_radius}")

    root = _resolve_nksid_root(data_dir)
    samples = _read_samples(root)
    train_folds = _read_folds(root / "kfold_train.txt")
    val_folds = _read_folds(root / "kfold_val.txt")
    if len(train_folds) != len(val_folds):
        raise ValueError("kfold_train.txt and kfold_val.txt have different record counts")
    if not 0 <= fold < len(train_folds):
        raise ValueError(f"fold must be in [0, {len(train_folds) - 1}], got {fold}")

    sample_count = len(samples)
    fold_rows: list[dict[str, Any]] = []
    for index, (train_indices, val_indices) in enumerate(zip(train_folds, val_folds)):
        train_set, val_set = set(train_indices), set(val_indices)
        out_of_range = [
            value
            for value in (*train_indices, *val_indices)
            if value < 0 or value >= sample_count
        ]
        fold_rows.append(
            {
                "fold_record": index,
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "train_duplicate_indices": len(train_indices) - len(train_set),
                "val_duplicate_indices": len(val_indices) - len(val_set),
                "train_val_overlap": len(train_set & val_set),
                "union_count": len(train_set | val_set),
                "out_of_range_count": len(out_of_range),
            }
        )

    val_frequency = Counter(value for values in val_folds for value in values)
    mean_val_count = mean(len(values) for values in val_folds)
    inferred_k = round(sample_count / mean_val_count) if mean_val_count else None
    frequencies = Counter(val_frequency.values())
    inferred_repeats = (
        next(iter(frequencies))
        if len(frequencies) == 1 and len(val_frequency) == sample_count
        else None
    )

    selected_train = set(train_folds[fold])
    selected_val = set(val_folds[fold])
    by_label_number: dict[tuple[int, int], int] = {}
    for row in samples:
        number = _trailing_number(row["path"])
        if number is not None:
            by_label_number[(row["label"], number)] = row["index"]

    adjacent_count = 0
    numbered_val_count = 0
    for index in selected_val:
        if not 0 <= index < sample_count:
            continue
        row = samples[index]
        number = _trailing_number(row["path"])
        if number is None:
            continue
        numbered_val_count += 1
        neighbors = (
            by_label_number.get((row["label"], number + offset))
            for offset in range(-neighbor_radius, neighbor_radius + 1)
            if offset
        )
        if any(neighbor in selected_train for neighbor in neighbors):
            adjacent_count += 1

    resolved_paths = [str(row["path"].resolve()) for row in samples]
    path_counts = Counter(resolved_paths)
    missing = [path for path in resolved_paths if not Path(path).is_file()]
    exact_duplicate_groups: int | None = None
    exact_duplicate_extra_copies: int | None = None
    if hash_files and not missing:
        content_counts = Counter(
            sha256(Path(path).read_bytes()).hexdigest() for path in resolved_paths
        )
        duplicates = [count for count in content_counts.values() if count > 1]
        exact_duplicate_groups = len(duplicates)
        exact_duplicate_extra_copies = sum(count - 1 for count in duplicates)

    critical: list[str] = []
    if any(
        row["train_val_overlap"]
        or row["train_duplicate_indices"]
        or row["val_duplicate_indices"]
        or row["out_of_range_count"]
        or row["union_count"] != sample_count
        for row in fold_rows
    ):
        critical.append("fold_index_integrity_failed")
    if missing:
        critical.append("missing_sample_files")

    warnings: list[str] = []
    if inferred_k is not None and len(train_folds) != inferred_k:
        warnings.append("fold_records_are_repeated_splits_not_a_single_kfold_cycle")
    adjacency_fraction = adjacent_count / numbered_val_count if numbered_val_count else None
    if adjacency_fraction is not None and adjacency_fraction >= 0.5:
        warnings.append("image_level_split_has_high_filename_neighbor_leakage_risk")
    if (
        inferred_k is not None
        and min(Counter(row["label"] for row in samples).values()) < inferred_k
    ):
        warnings.append("rarest_class_is_too_small_for_the_inferred_kfold_protocol")

    return {
        "schema_version": 1,
        "dataset_root": str(root.resolve()),
        "sample_count": sample_count,
        "class_counts": dict(sorted(Counter(row["label"] for row in samples).items())),
        "integrity": {
            "missing_file_count": len(missing),
            "duplicate_path_extra_copies": sum(
                count - 1 for count in path_counts.values() if count > 1
            ),
            "exact_duplicate_groups": exact_duplicate_groups,
            "exact_duplicate_extra_copies": exact_duplicate_extra_copies,
            "content_hashing_enabled": hash_files,
        },
        "split_protocol": {
            "record_count": len(train_folds),
            "inferred_k": inferred_k,
            "inferred_repeats": inferred_repeats,
            "validation_frequency_distribution": dict(sorted(frequencies.items())),
            "fold_rows": fold_rows,
        },
        "selected_fold": {
            "fold_record": fold,
            "train_count": len(selected_train),
            "val_count": len(selected_val),
            "neighbor_radius": neighbor_radius,
            "numbered_val_count": numbered_val_count,
            "val_with_train_filename_neighbor": adjacent_count,
            "filename_neighbor_fraction": adjacency_fraction,
            "unit_of_split": "image_index",
        },
        "critical_failures": critical,
        "warnings": warnings,
        "claim_boundary": (
            "Filename adjacency is a leakage-risk diagnostic, not proof that adjacent "
            "files are frames from the same acquisition sequence. Sequence or mission "
            "metadata is required for a definitive group split."
        ),
    }
