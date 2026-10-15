#!/usr/bin/env python3
"""Record the NKSID classification setup for the ZCU104 RL-HW-NAS campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.dataset import (  # noqa: E402
    FrozenGeometryTransform,
    NKSID_CLASSES,
    NKSIDDataset,
    protocol_normalization,
)
from hwnas_fpga.data.protocol import build_protocol_split, class_counts  # noqa: E402
from hwnas_fpga.runtime import load_config  # noqa: E402


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def dataset_version(samples: list[tuple[str, int]], data_dir: Path) -> dict[str, Any]:
    rows = []
    for index, (path_value, label) in enumerate(samples):
        path = Path(path_value).resolve()
        rows.append(
            {
                "index": index,
                "relative_path": path.relative_to(data_dir).as_posix(),
                "label": int(label),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    split_files = {}
    for name in ("train_abs.txt", "kfold_train.txt", "kfold_val.txt"):
        path = data_dir / name
        split_files[name] = {
            "relative_path": path.relative_to(data_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "samples": rows,
        "split_files": split_files,
        "class_names": list(NKSID_CLASSES),
    }
    return {
        "id": "NKSID_content_sha256:" + canonical_sha256(payload),
        "content_sha256": canonical_sha256(payload),
        "sample_count": len(rows),
        "split_files": split_files,
    }


def named_counts(counts: list[int], class_names: list[str]) -> dict[str, int]:
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def full_distribution(samples: list[tuple[str, int]], class_names: list[str]) -> dict[str, int]:
    counter = Counter(int(label) for _, label in samples)
    return {name: int(counter.get(index, 0)) for index, name in enumerate(class_names)}


def build_record(
    *,
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    fold: int,
    seed: int,
    image_size: int,
    input_channels: int,
    inner_val_fraction: float,
) -> dict[str, Any]:
    view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=image_size,
        is_training=False,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=input_channels,
        image_error_policy="raise",
        augmentation_profile="none",
    )
    samples = list(view.samples)
    class_names = list(view.classes)
    if class_names != list(NKSID_CLASSES):
        raise ValueError(f"NKSID class-name order mismatch: {class_names!r}")

    split = build_protocol_split(
        samples,
        data_dir,
        fold_index=fold,
        seed=seed,
        inner_val_fraction=inner_val_fraction,
        num_classes=len(class_names),
    )
    version = dataset_version(samples, data_dir)
    split_id = (
        f"nksid_outer5fold_inner_contiguous_v1/fold{fold}/seed{seed}/"
        f"inner{inner_val_fraction:g}"
    )
    record = {
        "artifact_type": "zcu104_classification_task_setup_record",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "NKSID image classification",
        "dataset_mixing_policy": {
            "nksid_only": True,
            "excluded_detection_datasets": ["Cylinder2", "Figshare"],
        },
        "dataset_version": version,
        "split_id": split_id,
        "fold": int(fold),
        "seed": int(seed),
        "input_size": [int(input_channels), int(image_size), int(image_size)],
        "num_classes": len(class_names),
        "class_names": class_names,
        "sample_counts": {
            "total": len(samples),
            "train_inner": len(split.train_indices),
            "inner_val": len(split.inner_val_indices),
            "outer_test": len(split.outer_val_indices),
        },
        "class_distribution": {
            "full": full_distribution(samples, class_names),
            "train_inner": named_counts(
                split.metadata["train_class_counts"], class_names
            ),
            "inner_val": named_counts(
                split.metadata["inner_val_class_counts"], class_names
            ),
            "outer_test": named_counts(
                split.metadata["outer_val_class_counts"], class_names
            ),
        },
        "split_protocol": {
            "name": split.metadata["protocol"],
            "outer_validation_exposed_to_search": False,
            "outer_usage": "final_reporting_only",
            "inner_validation_usage": "search_candidate_selection_and_epoch_selection",
            "inner_val_fraction": float(inner_val_fraction),
        },
        "preprocessing": {
            **FrozenGeometryTransform(
                image_size=image_size,
                mode="stretch_224",
                fixed_scale_factor=None,
                padding_value=0,
            ).to_dict(),
            "augmentation_profile": "frozen_strong_for_training_none_for_eval",
            "normalization": protocol_normalization(output_channels=input_channels),
            "output_channels": int(input_channels),
            "image_error_policy": "raise",
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
    }
    record["record_sha256"] = canonical_sha256(record)
    path = output_dir / f"classification_setup_fold{fold}_seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/search/zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-dir",
        default="artifacts/zcu104_full_campaign_v1/classification_setup",
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(str(config_path))
    dataset_cfg = config["dataset"]
    data_dir = resolve(dataset_cfg["data_dir"])
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        build_record(
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            fold=fold,
            seed=seed,
            image_size=int(dataset_cfg.get("image_size", 224)),
            input_channels=int(dataset_cfg.get("input_channels", 1)),
            inner_val_fraction=float(dataset_cfg.get("inner_val_fraction", 0.15)),
        )
        for fold in args.folds
        for seed in args.seeds
    ]
    summary = {
        "artifact_type": "zcu104_classification_task_setup_summary",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "records": [
            {
                "fold": record["fold"],
                "seed": record["seed"],
                "split_id": record["split_id"],
                "path": record["path"],
                "sha256": record["sha256"],
                "sample_counts": record["sample_counts"],
            }
            for record in records
        ],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = output_dir / "classification_setup_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "WROTE_CLASSIFICATION_SETUP_RECORDS",
                "summary": str(summary_path.resolve()),
                "record_count": len(records),
                "folds": args.folds,
                "seeds": args.seeds,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
