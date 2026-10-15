#!/usr/bin/env python3
"""Train external sonar crop classifiers and evaluate synthetic-noise robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.data import get_sonar_transforms
from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec
from hwnas_fpga.training.robustness import evaluate_sonar_robustness
from hwnas_fpga.training.trainer import evaluate_classifier


DEFAULT_ROOTS = {
    "roboflow": PROJECT_ROOT / "data" / "external" / "roboflow_cylider2",
    "figshare": PROJECT_ROOT / "data" / "external" / "figshare_mine_detection_v2",
}
NOISE_CONDITIONS = (
    {"name": "speckle_weak", "kind": "speckle", "variance": 0.01},
    {"name": "speckle_medium", "kind": "speckle", "variance": 0.04},
    {"name": "speckle_strong", "kind": "speckle", "variance": 0.09},
)


@dataclass(frozen=True)
class CropSample:
    image_path: Path
    label: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    group_id: str
    source_dataset: str
    source_split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv_text(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def yolo_to_xyxy(
    row: str,
    *,
    width: int,
    height: int,
) -> tuple[int, tuple[float, float, float, float]]:
    fields = row.split()
    if len(fields) != 5:
        raise ValueError(f"malformed YOLO row: {row!r}")
    cls = int(float(fields[0]))
    cx, cy, bw, bh = (float(value) for value in fields[1:])
    x0 = (cx - bw / 2.0) * width
    y0 = (cy - bh / 2.0) * height
    x1 = (cx + bw / 2.0) * width
    y1 = (cy + bh / 2.0) * height
    return cls, (x0, y0, x1, y1)


def pad_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    px = bw * padding
    py = bh * padding
    return (
        max(0.0, x0 - px),
        max(0.0, y0 - py),
        min(float(width), x1 + px),
        min(float(height), y1 + py),
    )


def load_roboflow_samples(root: Path, *, crop_padding: float) -> tuple[list[CropSample], list[str], dict[str, Any]]:
    manifest_path = root / "dataset_manifest.json"
    index_path = root / "source_index.json"
    if not manifest_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Roboflow manifest/source_index missing under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    class_map = {int(k): str(v) for k, v in index["class_index"].items()}
    samples: list[CropSample] = []
    for record in index["records"]:
        image_path = root / record["relative_image_path"]
        width = int(record["width"])
        height = int(record["height"])
        for box in record["boxes"]:
            label_name = str(box["label"])
            label = {name: idx for idx, name in class_map.items()}[label_name]
            x = float(box["x"])
            y = float(box["y"])
            bw = float(box["width"])
            bh = float(box["height"])
            raw_box = (x - bw / 2.0, y - bh / 2.0, x + bw / 2.0, y + bh / 2.0)
            samples.append(
                CropSample(
                    image_path=image_path,
                    label=label,
                    class_name=label_name,
                    box_xyxy=pad_box(raw_box, width=width, height=height, padding=crop_padding),
                    group_id=str(record["image_id"]),
                    source_dataset="roboflow",
                    source_split=str(record["split"]),
                )
            )
    meta = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_index": str(index_path.resolve()),
        "source_index_sha256": sha256_file(index_path),
        "source_boundary": manifest.get("reconstruction_boundary"),
    }
    return samples, [class_map[index] for index in sorted(class_map)], meta


def _paired_label_path(image_path: Path, year_root: Path) -> Path:
    relative = image_path.relative_to(year_root)
    parts = list(relative.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            parts[index] = "labels"
            break
    return year_root.joinpath(*parts).with_suffix(".txt")


def load_figshare_samples(root: Path, *, crop_padding: float) -> tuple[list[CropSample], list[str], dict[str, Any]]:
    manifest_path = root / "dataset_manifest.json"
    extracted = root / "extracted"
    if not manifest_path.exists() or not extracted.exists():
        raise FileNotFoundError(f"Figshare manifest/extracted tree missing under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = ["MILCO", "NOMBO"]
    samples: list[CropSample] = []
    for year_root in sorted(path for path in extracted.iterdir() if path.is_dir() and path.name.isdigit()):
        for image_path in sorted(year_root.rglob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            label_path = _paired_label_path(image_path, year_root)
            if not label_path.exists():
                continue
            rows = [row.strip() for row in label_path.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
            if not rows:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            for box_index, row in enumerate(rows):
                label, raw_box = yolo_to_xyxy(row, width=width, height=height)
                if label < 0 or label >= len(class_names):
                    raise ValueError(f"unknown Figshare label {label} in {label_path}")
                samples.append(
                    CropSample(
                        image_path=image_path,
                        label=label,
                        class_name=class_names[label],
                        box_xyxy=pad_box(raw_box, width=width, height=height, padding=crop_padding),
                        group_id=f"{year_root.name}/{image_path.relative_to(year_root).as_posix()}",
                        source_dataset="figshare",
                        source_split=year_root.name,
                    )
                )
    meta = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_boundary": "empty label files are excluded from crop-level target classification",
    }
    return samples, class_names, meta


def split_roboflow(samples: Sequence[CropSample]) -> dict[str, list[int]]:
    splits = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(samples):
        name = "val" if sample.source_split == "valid" else sample.source_split
        if name not in splits:
            raise ValueError(f"unexpected Roboflow split: {sample.source_split}")
        splits[name].append(index)
    return splits


def _group_majority_labels(samples: Sequence[CropSample]) -> dict[str, int]:
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for sample in samples:
        grouped[sample.group_id][sample.label] += 1
    return {group: counts.most_common(1)[0][0] for group, counts in grouped.items()}


def split_grouped_stratified(
    samples: Sequence[CropSample],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    group_label = _group_majority_labels(samples)
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for group, label in group_label.items():
        by_label[label].append(group)
    split_by_group: dict[str, str] = {}
    for groups in by_label.values():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        test_n = max(1, round(n * test_fraction))
        val_n = max(1, round(n * val_fraction))
        for group in groups[:test_n]:
            split_by_group[group] = "test"
        for group in groups[test_n : test_n + val_n]:
            split_by_group[group] = "val"
        for group in groups[test_n + val_n :]:
            split_by_group[group] = "train"
    splits = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(samples):
        splits[split_by_group[sample.group_id]].append(index)
    return splits


class ExternalCropDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[CropSample],
        indices: Sequence[int],
        *,
        image_size: int,
        is_training: bool,
    ) -> None:
        self.samples = list(samples)
        self.indices = list(indices)
        self.transform = get_sonar_transforms(
            image_size=image_size,
            is_training=is_training,
            output_channels=1,
            geometry_mode="stretch_224",
            augmentation_profile="frozen_strong" if is_training else "none",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[self.indices[idx]]
        with Image.open(sample.image_path) as image:
            crop = image.convert("L").crop(sample.box_xyxy)
        return self.transform(crop), int(sample.label)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def class_weights(samples: Sequence[CropSample], indices: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.ones(num_classes, dtype=torch.float32)
    for index in indices:
        counts[samples[index].label] += 1
    weights = len(indices) / (num_classes * counts)
    return weights / weights.sum() * num_classes


def architecture_for(method: str, num_classes: int) -> ArchitectureSpec:
    final_blocks = {
        "control": (
            BlockSpec("mbconv", kernel_size=3, expand_ratio=1, stride=1),
            BlockSpec("mbconv", kernel_size=3, expand_ratio=2, stride=1),
        ),
        "adaptive_denoise_v2": (
            BlockSpec("adaptive_denoise", kernel_size=3, expand_ratio=1, stride=1),
            BlockSpec("mbconv", kernel_size=3, expand_ratio=2, stride=1),
        ),
        "edge_v2": (
            BlockSpec("mbconv", kernel_size=3, expand_ratio=1, stride=1),
            BlockSpec("edge_v2", kernel_size=3, expand_ratio=1, stride=1),
        ),
        "adaptive_denoise_edge_v2": (
            BlockSpec("adaptive_denoise", kernel_size=3, expand_ratio=1, stride=1),
            BlockSpec("edge_v2", kernel_size=3, expand_ratio=1, stride=1),
        ),
    }
    if method not in final_blocks:
        raise ValueError(f"unsupported method: {method}")
    return ArchitectureSpec(
        input_channels=1,
        stem_channels=32,
        stem_stride=2,
        post_stem_downsample_stride=1,
        num_classes=num_classes,
        stages=(
            StageSpec(
                channels=16,
                depth=1,
                stride=1,
                blocks=(BlockSpec("conv", kernel_size=1, expand_ratio=1, stride=1),),
            ),
            StageSpec(
                channels=24,
                depth=1,
                stride=2,
                blocks=(BlockSpec("mbconv", kernel_size=3, expand_ratio=6, stride=2),),
            ),
            StageSpec(
                channels=32,
                depth=1,
                stride=2,
                blocks=(BlockSpec("mbconv", kernel_size=3, expand_ratio=3, stride=2),),
            ),
            StageSpec(channels=32, depth=2, stride=1, blocks=final_blocks[method]),
        ),
    )


def train_one(
    *,
    dataset_name: str,
    class_names: Sequence[str],
    samples: Sequence[CropSample],
    splits: dict[str, list[int]],
    method: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    num_classes = len(class_names)
    train_ds = ExternalCropDataset(samples, splits["train"], image_size=args.image_size, is_training=True)
    val_ds = ExternalCropDataset(samples, splits["val"], image_size=args.image_size, is_training=False)
    test_ds = ExternalCropDataset(samples, splits["test"], image_size=args.image_size, is_training=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    architecture = architecture_for(method, num_classes)
    model = build_model(architecture, num_classes=num_classes).to(args.device)
    weights = class_weights(samples, splits["train"], num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_gain = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * inputs.size(0)
            train_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())
            train_total += int(targets.numel())
        scheduler.step()
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            criterion=criterion,
            device=args.device,
            num_classes=num_classes,
            topk=min(5, num_classes),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_total),
                "train_top1": train_correct / max(1, train_total),
                "val_top1": float(val_metrics["top1"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if args.patience > 0 and epochs_without_gain >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    run_dir = output_dir / "checkpoints" / dataset_name / method / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    torch.save(
        {
            "schema_version": 1,
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "architecture": architecture.to_dict(),
            "class_names": list(class_names),
            "model_state_dict": model.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    clean = evaluate_classifier(
        model,
        test_loader,
        criterion=criterion,
        device=args.device,
        num_classes=num_classes,
        topk=min(5, num_classes),
    )
    robustness = evaluate_sonar_robustness(
        model,
        test_loader,
        device=args.device,
        num_classes=num_classes,
        class_weights=weights,
        config={
            "enabled": True,
            "protocol": "external_crop_test_speckle_v1",
            "seed": args.noise_seed,
            "normalization_mean": [0.5],
            "normalization_std": [0.5],
            "conditions": NOISE_CONDITIONS,
        },
        topk=min(5, num_classes),
    )
    rows = [
        {
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "condition": "clean",
            "variance": 0.0,
            "macro_f1": float(clean["macro_f1"]),
            "top1": float(clean["top1"]),
            "weighted_f1": float(clean["weighted_f1"]),
            "loss": float(clean["loss"]),
            "num_samples": int(clean["num_samples"]),
        }
    ]
    for condition, result in zip(NOISE_CONDITIONS, robustness["condition_results"]):
        rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "seed": seed,
                "condition": condition["name"],
                "variance": float(condition["variance"]),
                "macro_f1": float(result["metrics"]["macro_f1"]),
                "top1": float(result["metrics"]["top1"]),
                "weighted_f1": float(result["metrics"]["weighted_f1"]),
                "loss": float(result["metrics"]["loss"]),
                "num_samples": int(result["metrics"]["num_samples"]),
            }
        )
    meta = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "seconds": time.time() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
        "clean_confusion_matrix": clean["confusion_matrix"],
    }
    return meta, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["condition"], float(row["variance"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, method, condition, variance), values in sorted(grouped.items()):
        macro = np.asarray([float(row["macro_f1"]) for row in values], dtype=np.float64)
        top1 = np.asarray([float(row["top1"]) for row in values], dtype=np.float64)
        weighted = np.asarray([float(row["weighted_f1"]) for row in values], dtype=np.float64)
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "condition": condition,
                "variance": variance,
                "run_count": len(values),
                "macro_f1_mean": float(macro.mean()),
                "macro_f1_std": float(macro.std()),
                "top1_mean": float(top1.mean()),
                "top1_std": float(top1.std()),
                "weighted_f1_mean": float(weighted.mean()),
                "weighted_f1_std": float(weighted.std()),
            }
        )
    return output


def sample_inventory(samples: Sequence[CropSample], splits: dict[str, list[int]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "class_counts": dict(Counter(sample.class_name for sample in samples)),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "split_class_counts": {
            split: dict(Counter(samples[index].class_name for index in indices))
            for split, indices in splits.items()
        },
        "group_counts": {
            split: len({samples[index].group_id for index in indices})
            for split, indices in splits.items()
        },
    }


def cap_splits_for_smoke(
    splits: dict[str, list[int]],
    *,
    max_samples: int | None,
) -> dict[str, list[int]]:
    if max_samples is None:
        return splits
    total = sum(len(indices) for indices in splits.values())
    if total <= max_samples:
        return splits
    capped: dict[str, list[int]] = {}
    remaining = int(max_samples)
    for name in ("train", "val", "test"):
        source = list(splits[name])
        if not source:
            capped[name] = []
            continue
        share = max(1, round(max_samples * len(source) / total))
        if name == "test":
            share = remaining
        share = max(1, min(len(source), share, remaining))
        capped[name] = source[:share]
        remaining -= share
    if capped["train"] and capped["val"] and capped["test"]:
        return capped
    raise ValueError("max-samples-per-dataset is too small to keep train/val/test non-empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="roboflow,figshare")
    parser.add_argument("--methods", default="control,adaptive_denoise_v2,edge_v2")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--figshare-split-seed", type=int, default=20260816)
    parser.add_argument("--noise-seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--output-dir", default="results/external_sonar_noise_classifiers_20260816")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv_text(args.datasets)
    methods = parse_csv_text(args.methods)
    seeds = parse_csv_ints(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_payloads: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    for dataset_name in datasets:
        root = DEFAULT_ROOTS[dataset_name]
        if dataset_name == "roboflow":
            samples, class_names, source_meta = load_roboflow_samples(root, crop_padding=args.crop_padding)
            splits = split_roboflow(samples)
            split_policy = "native train/valid/test split"
        elif dataset_name == "figshare":
            samples, class_names, source_meta = load_figshare_samples(root, crop_padding=args.crop_padding)
            splits = split_grouped_stratified(
                samples,
                seed=args.figshare_split_seed,
                val_fraction=0.15,
                test_fraction=0.15,
            )
            split_policy = "grouped stratified crop split by source image"
        else:
            raise ValueError(f"unsupported dataset: {dataset_name}")
        splits = cap_splits_for_smoke(splits, max_samples=args.max_samples_per_dataset)
        dataset_payloads[dataset_name] = {
            "root": str(root.resolve()),
            "class_names": list(class_names),
            "source_meta": source_meta,
            "split_policy": split_policy,
            **sample_inventory(samples, splits),
        }
        for method in methods:
            for seed in seeds:
                meta, rows = train_one(
                    dataset_name=dataset_name,
                    class_names=class_names,
                    samples=samples,
                    splits=splits,
                    method=method,
                    seed=seed,
                    args=args,
                    output_dir=output_dir,
                )
                run_meta.append(meta)
                all_rows.extend(rows)
                write_csv(
                    output_dir / "per_run.csv",
                    all_rows,
                    [
                        "dataset",
                        "method",
                        "seed",
                        "condition",
                        "variance",
                        "macro_f1",
                        "top1",
                        "weighted_f1",
                        "loss",
                        "num_samples",
                    ],
                )
                write_csv(
                    output_dir / "summary.csv",
                    summarize(all_rows),
                    [
                        "dataset",
                        "method",
                        "condition",
                        "variance",
                        "run_count",
                        "macro_f1_mean",
                        "macro_f1_std",
                        "top1_mean",
                        "top1_std",
                        "weighted_f1_mean",
                        "weighted_f1_std",
                    ],
                )
    payload = {
        "schema_version": 1,
        "protocol": "external_crop_classification_speckle_v1",
        "datasets": dataset_payloads,
        "methods": methods,
        "seeds": seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "image_size": args.image_size,
        "crop_padding": args.crop_padding,
        "noise_conditions": NOISE_CONDITIONS,
        "runs": run_meta,
        "summary": summarize(all_rows),
        "claim_boundary": (
            "Crop-level target classification on external detection datasets. "
            "This is full-network recognition robustness, separate from PSNR/SSIM image-quality analysis."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
