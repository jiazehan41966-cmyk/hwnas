"""Byte-level cross-split duplicate audit without decoding images or running models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/sonar_datasets_registry.yaml")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    registry_path = Path(args.registry).resolve()
    repo_root = registry_path.parent.parent
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entries = {item["dataset_id"]: item for item in registry["datasets"]}
    if args.dataset_id not in entries:
        raise ValueError(f"Unknown dataset: {args.dataset_id}")
    entry = entries[args.dataset_id]

    split_files: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        value = entry.get(f"{split}_images")
        if value:
            split_files[split] = images((repo_root / value).resolve())
    if len(split_files) < 2:
        raise ValueError("At least two registered splits are required")

    sizes_by_split = {split: defaultdict(list) for split in split_files}
    for split, paths in split_files.items():
        for path in paths:
            sizes_by_split[split][path.stat().st_size].append(path)
    cross_split_sizes = {
        size for split, groups in sizes_by_split.items() for size in groups
        if sum(size in other for other in sizes_by_split.values()) >= 2
    }
    candidates = [
        (split, path) for split, paths in split_files.items() for path in paths
        if path.stat().st_size in cross_split_sizes
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = list(pool.map(lambda item: file_sha256(item[1]), candidates))

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (split, path), digest in zip(candidates, hashes):
        by_hash[digest].append({"split": split, "file": path.name})
    exact = []
    for digest, members in by_hash.items():
        if len({member["split"] for member in members}) > 1:
            exact.append({"sha256": digest, "members": members})
    exact.sort(key=lambda row: row["sha256"])

    stems: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split, paths in split_files.items():
        for path in paths:
            stems[path.stem].append({"split": split, "file": path.name})
    repeated_stems = [
        {"stem": stem, "members": members}
        for stem, members in stems.items()
        if len({member["split"] for member in members}) > 1
    ]

    report = {
        "schema_version": "phase3a-exact-split-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "scope": "registered split image bytes only; no decoding, inference, labels, or metrics",
        "split_counts": {split: len(paths) for split, paths in split_files.items()},
        "method": "size prefilter followed by SHA256 for sizes present in at least two splits",
        "candidate_files_hashed": len(candidates),
        "exact_cross_split_groups": exact,
        "exact_cross_split_group_count": len(exact),
        "repeated_stem_groups": repeated_stems,
        "repeated_stem_group_count": len(repeated_stems),
        "near_duplicate_status": "NOT_TESTED_TO_KEEP_TEST_PIXELS_SEALED",
        "test_evaluated": False,
        "test_pixels_decoded": False,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset_id", "split_counts", "candidate_files_hashed", "exact_cross_split_group_count", "repeated_stem_group_count", "near_duplicate_status")}, indent=2))


if __name__ == "__main__":
    main()
