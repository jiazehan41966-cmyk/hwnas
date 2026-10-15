"""Train-only calibration subset. Valid is excluded from selection; test is forbidden."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .common import ROOT, baseline, dump_json, forbid_split, sha256_file

import baseline as frozen


CALIBRATION_SEED = 42
CALIBRATION_IMAGES = 64
CALIBRATION_BATCH_SIZE = 4


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def build_calibration(output_dir: Path) -> dict[str, Any]:
    forbid_split("train")
    config = baseline()
    data_root = frozen.resolve(config["dataset"]["data_dir"])
    loader = frozen.make_loader(config, data_root, "train", CALIBRATION_SEED, 0, train=False)
    dataset = loader.dataset
    if any("valid" in path.parts or "test" in path.parts for path in dataset.samples):
        raise ValueError("non-train images leaked into the train loader")
    if len(dataset) < CALIBRATION_IMAGES:
        raise ValueError("train split has fewer than 64 calibration images")
    train_listing = []
    for image in dataset.samples:
        if "valid" in image.parts or "test" in image.parts:
            raise ValueError("non-train images leaked into the train loader")
        train_listing.append(_relative(image, dataset.root))
    selected = list(dataset.samples[:CALIBRATION_IMAGES])
    entries = []
    for image in selected:
        label = dataset.labels_dir / f"{image.stem}.txt"
        if "valid" in image.parts or "test" in image.parts:
            raise ValueError("calibration selection is not train-only")
        entries.append(
            {
                "image": _relative(image, dataset.root),
                "image_sha256": sha256_file(image),
                "annotation": _relative(label, dataset.root),
                "annotation_sha256": sha256_file(label),
            }
        )
    payload = {
        "split": "train",
        "valid_used_for_selection": False,
        "test_evaluated": False,
        "calibration_seed": CALIBRATION_SEED,
        "number_of_calibration_images": len(entries),
        "batch_size": CALIBRATION_BATCH_SIZE,
        "augmentation": False,
        "shuffle": False,
        "order": "frozen_loader_sorted_first_64",
        "train_image_count": len(train_listing),
        "entries": entries,
    }
    payload["split_hash"] = hashlib.sha256(
        json.dumps(train_listing, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["dataset_hash"] = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    dump_json(output_dir / "calibration_manifest.json", payload)
    return payload


def calibration_tensor_batches(manifest: dict[str, Any], device: str):
    config = baseline()
    data_root = frozen.resolve(config["dataset"]["data_dir"])
    loader = frozen.make_loader(config, data_root, "train", CALIBRATION_SEED, 0, train=False)
    dataset = loader.dataset
    expected = [entry["image"] for entry in manifest["entries"]]
    actual = [_relative(path, dataset.root) for path in dataset.samples[:CALIBRATION_IMAGES]]
    if actual != expected:
        raise ValueError("calibration image order drifted")
    batch: list[torch.Tensor] = []
    for index in range(CALIBRATION_IMAGES):
        image, _target = dataset[index]
        batch.append(image)
        if len(batch) == CALIBRATION_BATCH_SIZE:
            yield torch.stack(batch, dim=0).to(device)
            batch = []
    if batch:
        yield torch.stack(batch, dim=0).to(device)
