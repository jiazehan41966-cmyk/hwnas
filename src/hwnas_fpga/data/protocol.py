"""Frozen NKSID evaluation protocol.

This module implements the project-wide evaluation contract established by the
2026-07-03 first-principles audit:

- Outer evaluation uses the official 5-fold partition (the first repeat,
  records ``p0-k0`` .. ``p0-k4``, of ``kfold_train.txt`` / ``kfold_val.txt``).
  Each outer validation fold is used exactly once, for final reporting.
  It must never participate in architecture screening or best-epoch selection.
- Model/epoch selection uses an inner validation set carved out of the outer
  training indices. To limit the known image-adjacency leakage (95.8% of
  fold-0 validation images have a same-class neighbour-numbered image in
  training), the inner split takes one contiguous filename-number block per
  class instead of a random interleave, so leakage across the inner boundary
  is limited to at most two junction pairs per class.
- Every reported number must aggregate over outer folds (and seeds); single
  fold-0 values are legacy evidence only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, List, Optional, Sequence, Tuple

NUM_OUTER_FOLDS = 5


class ProtocolError(ValueError):
    """Raised when the on-disk split files violate the frozen protocol."""


@dataclass(frozen=True)
class SplitRecord:
    label: Optional[str]
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class OuterFold:
    fold_index: int
    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]


@dataclass
class ProtocolSplit:
    """Fully resolved index sets for one (outer fold, seed) protocol run."""

    fold_index: int
    seed: int
    train_indices: Tuple[int, ...]
    inner_val_indices: Tuple[int, ...]
    outer_val_indices: Tuple[int, ...]
    inner_val_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "seed": self.seed,
            "num_train": len(self.train_indices),
            "num_inner_val": len(self.inner_val_indices),
            "num_outer_val": len(self.outer_val_indices),
            "inner_val_fraction": self.inner_val_fraction,
            "train_indices": list(self.train_indices),
            "inner_val_indices": list(self.inner_val_indices),
            "outer_val_indices": list(self.outer_val_indices),
            "metadata": dict(self.metadata),
        }


def _read_split_records(path: Path) -> List[SplitRecord]:
    if not path.exists():
        raise ProtocolError(f"required split file is missing: {path}")

    records: List[SplitRecord] = []
    pending_label: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                pending_label = line.lstrip("#").strip() or None
                continue
            indices = tuple(int(token) for token in re.findall(r"\d+", line))
            if not indices:
                continue
            records.append(SplitRecord(label=pending_label, indices=indices))
            pending_label = None

    if not records:
        raise ProtocolError(f"no split records found in {path}")
    return records


def load_outer_folds(
    data_dir: str | Path,
    *,
    repeat: int = 0,
    num_samples: Optional[int] = None,
) -> List[OuterFold]:
    """Load one repeat of the official 5-fold partition and verify it.

    Verification is strict: the five validation records must be pairwise
    disjoint, cover every sample exactly once, and each train record must be
    the exact complement of its validation record. Any deviation raises
    :class:`ProtocolError` — the protocol never silently falls back to a
    random split.
    """
    root = Path(data_dir).expanduser()
    nested = root / "NKSID"
    if (nested / "kfold_val.txt").exists():
        root = nested

    train_records = _read_split_records(root / "kfold_train.txt")
    val_records = _read_split_records(root / "kfold_val.txt")
    if len(train_records) != len(val_records):
        raise ProtocolError(
            "kfold_train.txt and kfold_val.txt record counts differ: "
            f"{len(train_records)} vs {len(val_records)}"
        )

    start = repeat * NUM_OUTER_FOLDS
    end = start + NUM_OUTER_FOLDS
    if end > len(val_records):
        raise ProtocolError(
            f"repeat {repeat} needs records [{start}, {end}) but only "
            f"{len(val_records)} records exist"
        )

    selected_train = train_records[start:end]
    selected_val = val_records[start:end]

    for offset, (train_record, val_record) in enumerate(zip(selected_train, selected_val)):
        expected_label_suffix = f"k{offset}"
        for record in (train_record, val_record):
            if record.label is not None and not record.label.endswith(expected_label_suffix):
                raise ProtocolError(
                    f"record label {record.label!r} does not match expected fold "
                    f"suffix {expected_label_suffix!r} for repeat {repeat}"
                )

    all_val: set[int] = set()
    total_val = 0
    for offset, record in enumerate(selected_val):
        record_set = set(record.indices)
        if len(record_set) != len(record.indices):
            raise ProtocolError(f"outer fold {offset} contains duplicate validation indices")
        overlap = all_val & record_set
        if overlap:
            raise ProtocolError(
                f"outer fold {offset} shares {len(overlap)} validation indices with earlier folds"
            )
        all_val |= record_set
        total_val += len(record.indices)

    if num_samples is not None:
        if all_val != set(range(num_samples)):
            raise ProtocolError(
                "outer validation folds do not partition the dataset: "
                f"covered {len(all_val)} of {num_samples} samples"
            )

    universe = all_val
    folds: List[OuterFold] = []
    for offset, (train_record, val_record) in enumerate(zip(selected_train, selected_val)):
        val_set = set(val_record.indices)
        train_set = set(train_record.indices)
        expected_train = universe - val_set
        if train_set != expected_train:
            raise ProtocolError(
                f"outer fold {offset} train record is not the complement of its "
                f"validation record ({len(train_set)} vs expected {len(expected_train)})"
            )
        folds.append(
            OuterFold(
                fold_index=offset,
                train_indices=tuple(sorted(train_set)),
                val_indices=tuple(sorted(val_set)),
            )
        )
    return folds


_FILENAME_NUMBER_PATTERN = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")


def filename_number(path: str) -> Optional[int]:
    """Extract the trailing image number from a sample path (``img_192.jpg`` -> 192)."""
    match = _FILENAME_NUMBER_PATTERN.search(str(path).replace("\\", "/").rsplit("/", 1)[-1])
    if match is None:
        return None
    return int(match.group(1))


def contiguous_inner_split(
    samples: Sequence[Tuple[str, int]],
    outer_train_indices: Sequence[int],
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split outer-train indices into inner-train and inner-val sets.

    Per class, samples are ordered by their filename number and one
    contiguous (wrap-around) block is assigned to inner validation. This keeps
    neighbour-numbered images on the same side of the split except at the two
    block junctions, which bounds adjacency leakage instead of maximising it
    the way a random interleave does.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_class: dict[int, List[int]] = {}
    for index in outer_train_indices:
        _, label = samples[index]
        by_class.setdefault(int(label), []).append(int(index))

    rng = Random(seed)
    inner_train: List[int] = []
    inner_val: List[int] = []

    for label in sorted(by_class):
        indices = by_class[label]
        indices.sort(
            key=lambda i: (
                filename_number(samples[i][0]) is None,
                filename_number(samples[i][0]) or 0,
                samples[i][0],
            )
        )
        count = len(indices)
        if count < 2:
            inner_train.extend(indices)
            continue
        block = max(1, round(count * val_fraction))
        block = min(block, count - 1)
        start = rng.randrange(count)
        selected = {indices[(start + offset) % count] for offset in range(block)}
        inner_val.extend(sorted(selected))
        inner_train.extend(i for i in indices if i not in selected)

    return tuple(sorted(inner_train)), tuple(sorted(inner_val))


def class_counts(
    samples: Sequence[Tuple[str, int]],
    indices: Sequence[int],
    *,
    num_classes: int,
) -> List[int]:
    counts = [0] * int(num_classes)
    for index in indices:
        _, label = samples[index]
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1
    return counts


def build_protocol_split(
    samples: Sequence[Tuple[str, int]],
    data_dir: str | Path,
    *,
    fold_index: int,
    seed: int,
    inner_val_fraction: float = 0.15,
    num_classes: int = 8,
) -> ProtocolSplit:
    """Resolve the full frozen-protocol split for one (fold, seed) run."""
    folds = load_outer_folds(data_dir, num_samples=len(samples))
    if not 0 <= fold_index < len(folds):
        raise ProtocolError(f"fold_index must be in [0, {len(folds)}), got {fold_index}")
    outer = folds[fold_index]

    inner_train, inner_val = contiguous_inner_split(
        samples,
        outer.train_indices,
        val_fraction=inner_val_fraction,
        seed=seed,
    )

    overlap = set(inner_train) & set(outer.val_indices)
    overlap |= set(inner_val) & set(outer.val_indices)
    if overlap:
        raise ProtocolError(
            f"inner split leaked {len(overlap)} indices into the outer validation fold"
        )

    return ProtocolSplit(
        fold_index=fold_index,
        seed=seed,
        train_indices=inner_train,
        inner_val_indices=inner_val,
        outer_val_indices=outer.val_indices,
        inner_val_fraction=inner_val_fraction,
        metadata={
            "protocol": "nksid_outer5fold_inner_contiguous_v1",
            "train_class_counts": class_counts(samples, inner_train, num_classes=num_classes),
            "inner_val_class_counts": class_counts(samples, inner_val, num_classes=num_classes),
            "outer_val_class_counts": class_counts(
                samples, outer.val_indices, num_classes=num_classes
            ),
        },
    )
