#!/usr/bin/env python3
"""Freeze duplicate-safe derived splits for Phase 3C transfer experiments.

The source datasets are never modified. Clean views use hard links and bind a
machine-readable membership manifest plus a sidecar SHA256.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SEED = 42
B1_ID = "P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"


@dataclass(frozen=True)
class Item:
    source_split: str
    image: Path
    label: Path
    sha256: str
    label_sha256: str
    classes: tuple[int, ...]


class UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover(root: Path, split: str) -> list[tuple[str, Path, Path]]:
    image_dir, label_dir = root / split / "images", root / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"missing {split} images/labels under {root}")
    rows = []
    for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        rows.append((split, image, label))
    return rows


def read_classes(label: Path) -> tuple[int, ...]:
    classes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.append(int(line.split()[0]))
    return tuple(classes)


def load_items(rows: list[tuple[str, Path, Path]], workers: int) -> list[Item]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        image_hashes = list(pool.map(lambda row: sha256(row[1]), rows))
        label_hashes = list(pool.map(lambda row: sha256(row[2]), rows))
    return [
        Item(split, image, label, image_hash, label_hash, read_classes(label))
        for (split, image, label), image_hash, label_hash in zip(rows, image_hashes, label_hashes)
    ]


def duplicate_groups(items: list[Item]) -> list[list[Item]]:
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[item.sha256].append(item)
    return [group for group in groups.values() if len(group) > 1]


def grouped_items(items: list[Item], *, consecutive_numeric: bool) -> list[list[Item]]:
    keys = [item.image.name for item in items]
    uf = UnionFind(keys)
    by_hash: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_hash[item.sha256].append(item)
    for members in by_hash.values():
        for item in members[1:]:
            uf.union(members[0].image.name, item.image.name)
    if consecutive_numeric:
        by_number = {int(item.image.stem): item for item in items if item.image.stem.isdigit()}
        for number, item in by_number.items():
            if number + 1 in by_number:
                uf.union(item.image.name, by_number[number + 1].image.name)
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[uf.find(item.image.name)].append(item)
    return [sorted(group, key=lambda item: item.image.name) for group in groups.values()]


def allocate(groups: list[list[Item]], ratios: dict[str, float], seed: int) -> dict[str, list[Item]]:
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to one")
    total_size = sum(len(group) for group in groups)
    total_classes: Counter[int] = Counter(cls for group in groups for item in group for cls in item.classes)
    targets_size = {split: total_size * ratio for split, ratio in ratios.items()}
    targets_class = {split: {cls: count * ratio for cls, count in total_classes.items()} for split, ratio in ratios.items()}
    rng = random.Random(seed)
    decorated = []
    for group in groups:
        counts = Counter(cls for item in group for cls in item.classes)
        rarity = sum(count / max(1, total_classes[cls]) for cls, count in counts.items())
        decorated.append((-rarity, -len(group), rng.random(), group))
    decorated.sort(key=lambda row: row[:3])

    assigned = {split: [] for split in ratios}
    assigned_classes = {split: Counter() for split in ratios}
    for _, _, _, group in decorated:
        group_counts = Counter(cls for item in group for cls in item.classes)
        best_split, best_score = None, None
        for candidate in ratios:
            score = 0.0
            for split in ratios:
                size = len(assigned[split]) + (len(group) if split == candidate else 0)
                score += 4.0 * ((size - targets_size[split]) ** 2) / max(1.0, targets_size[split])
                for cls in total_classes:
                    count = assigned_classes[split][cls] + (group_counts[cls] if split == candidate else 0)
                    score += ((count - targets_class[split][cls]) ** 2) / max(1.0, targets_class[split][cls])
            key = (score, list(ratios).index(candidate))
            if best_score is None or key < best_score:
                best_split, best_score = candidate, key
        assert best_split is not None
        assigned[best_split].extend(group)
        assigned_classes[best_split].update(group_counts)
    return assigned


def materialize(clean_root: Path, assignments: dict[str, list[Item]], class_names: list[str]) -> None:
    if clean_root.exists():
        raise FileExistsError(f"clean root already exists: {clean_root}")
    for split, items in assignments.items():
        image_dir, label_dir = clean_root / split / "images", clean_root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=False)
        label_dir.mkdir(parents=True, exist_ok=False)
        for item in items:
            os.link(item.image, image_dir / item.image.name)
            os.link(item.label, label_dir / item.label.name)
    data = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (clean_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def summarize(assignments: dict[str, list[Item]]) -> dict[str, object]:
    result = {}
    for split, items in assignments.items():
        counts = Counter(cls for item in items for cls in item.classes)
        result[split] = {"images": len(items), "objects": sum(counts.values()), "class_counts": dict(sorted(counts.items()))}
    return result


def assert_duplicate_safe(assignments: dict[str, list[Item]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, items in assignments.items():
        for item in items:
            locations[item.sha256].add(split)
    crossed = {digest: sorted(splits) for digest, splits in locations.items() if len(splits) > 1}
    if crossed:
        raise AssertionError(f"duplicate hashes cross derived splits: {crossed}")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    *,
    dataset_id: str,
    modality: str,
    source_root: Path,
    clean_root: Path,
    manifest_path: Path,
    assignments: dict[str, list[Item]],
    exclusions: list[dict[str, object]],
    policy: dict[str, object],
    identity: dict[str, object],
) -> None:
    members = {}
    for split, items in assignments.items():
        members[split] = [
            {
                "file": item.image.name,
                "source_split": item.source_split,
                "source_image": relative(item.image),
                "source_label": relative(item.label),
                "image_sha256": item.sha256,
                "label_sha256": item.label_sha256,
                "classes": list(item.classes),
            }
            for item in sorted(items, key=lambda row: row.image.name)
        ]
    membership_payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "schema_version": "phase3b-clean-transfer-split-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "modality": modality,
        "identity": identity,
        "source_root": relative(source_root),
        "clean_root": relative(clean_root),
        "raw_data_modified": False,
        "seed": SEED,
        "policy": policy,
        "summary": summarize(assignments),
        "excluded_source_items": exclusions,
        "members": members,
        "membership_sha256": hashlib.sha256(membership_payload).hexdigest(),
        "duplicate_groups_cross_derived_split": 0,
        "test_membership_frozen": True,
        "model_metrics_used": False,
        "architecture_used": False,
    }
    dump(manifest_path, payload)
    digest = sha256(manifest_path)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(f"{digest}  {manifest_path.name}\n", encoding="ascii")


def prepare_uatd(workers: int) -> None:
    source = ROOT / "datasets/UATD/processed_yolo"
    clean = ROOT / "datasets/UATD/clean_phase3b"
    manifest = ROOT / "datasets/UATD/metadata/phase3b_clean_split_manifest.json"
    if clean.exists() or manifest.exists():
        raise FileExistsError("UATD clean split already exists; manifests are immutable")
    train_items = load_items(discover(source, "train"), workers)
    test_items = load_items(discover(source, "test"), workers)
    test_hashes = {item.sha256 for item in test_items}
    excluded_items = [item for item in train_items if item.sha256 in test_hashes]
    development = [item for item in train_items if item.sha256 not in test_hashes]
    dev_groups = grouped_items(development, consecutive_numeric=False)
    development_split = allocate(dev_groups, {"train": 0.8, "valid": 0.2}, SEED)
    assignments = {**development_split, "test": test_items}
    assert_duplicate_safe(assignments)
    names = list(yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    materialize(clean, assignments, names)
    exclusions = [
        {"file": item.image.name, "reason": "exact_duplicate_of_frozen_official_test", "image_sha256": item.sha256}
        for item in excluded_items
    ]
    write_manifest(
        dataset_id="uatd_v3",
        modality="MFLS",
        source_root=source,
        clean_root=clean,
        manifest_path=manifest,
        assignments=assignments,
        exclusions=exclusions,
        policy={
            "official_test_preserved": True,
            "development_duplicate_of_test_action": "exclude development copy",
            "development_split": "fixed-seed multi-label greedy stratification over exact-hash groups",
            "development_train_ratio": 0.8,
            "development_valid_ratio": 0.2,
            "acquisition_group_status": "GROUP_STRUCTURE_UNKNOWN",
        },
        identity={"dataset": "UATD v3", "not_researched": True, "num_classes": 10},
    )


def prepare_sctd(workers: int) -> None:
    source = ROOT / "datasets/SCTD/processed"
    clean = ROOT / "datasets/SCTD/clean_phase3b_v2"
    manifest = ROOT / "datasets/SCTD/metadata/phase3b_clean_split_manifest_v2.json"
    if clean.exists() or manifest.exists():
        raise FileExistsError("SCTD clean split already exists; manifests are immutable")
    items = load_items(discover(source, "train") + discover(source, "valid") + discover(source, "test"), workers)
    if len({item.image.name for item in items}) != len(items):
        raise ValueError("SCTD source views contain duplicate filenames")
    groups = grouped_items(items, consecutive_numeric=False)
    assignments = allocate(groups, {"train": 0.7, "valid": 0.15, "test": 0.15}, SEED)
    assert_duplicate_safe(assignments)
    names = list(yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    materialize(clean, assignments, names)
    write_manifest(
        dataset_id="classic_sctd_1_0",
        modality="SSS",
        source_root=source,
        clean_root=clean,
        manifest_path=manifest,
        assignments=assignments,
        exclusions=[],
        policy={
            "source_official_split": "trainval_only",
            "derived_split": "fixed-seed multi-label greedy allocation over exact-hash groups",
            "ratios": {"train": 0.7, "valid": 0.15, "test": 0.15},
            "acquisition_group_status": "GROUP_STRUCTURE_UNKNOWN",
            "numeric_run_policy": "recorded_as_risk_only; not treated as acquisition identity because it caused missing-class evaluation and is not source metadata",
        },
        identity={"dataset": "classic SCTD / SCTD 1.0", "is_sctd_3_0": False, "num_classes": 3},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uatd", "sctd", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.dataset in ("uatd", "both"):
        prepare_uatd(args.workers)
    if args.dataset in ("sctd", "both"):
        prepare_sctd(args.workers)
    print("Phase3B clean split manifests frozen", flush=True)


if __name__ == "__main__":
    main()
