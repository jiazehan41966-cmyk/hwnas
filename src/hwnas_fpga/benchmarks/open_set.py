"""Frozen NKSID 5-known/3-unknown protocol for CE+MSP and paper adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import yaml

from hwnas_fpga.benchmarks.metrics import open_set_summary
from hwnas_fpga.data.dataset import NKSIDDataset, create_dataloader
from hwnas_fpga.data.protocol import build_protocol_split


class RemappedKnownDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], label_map: dict[int, int]):
        self.base = base
        self.indices = [int(index) for index in indices]
        self.label_map = dict(label_map)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_label = self.base[self.indices[index]]
        if int(original_label) not in self.label_map:
            raise RuntimeError("unknown-class sample entered a known-only dataset")
        return image, self.label_map[int(original_label)]


@dataclass(frozen=True)
class OpenSetProtocolSpec:
    protocol: str
    fold: int
    class_names: tuple[str, ...]
    known_class_ids: tuple[int, ...]
    unknown_class_ids: tuple[int, ...]
    confidence_quantile: float

    @property
    def model_to_original(self) -> dict[int, int]:
        return {index: class_id for index, class_id in enumerate(self.known_class_ids)}

    @property
    def original_to_model(self) -> dict[int, int]:
        return {class_id: index for index, class_id in self.model_to_original.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "fold": self.fold,
            "class_names": list(self.class_names),
            "known_class_ids": list(self.known_class_ids),
            "unknown_class_ids": list(self.unknown_class_ids),
            "known_class_names": [self.class_names[index] for index in self.known_class_ids],
            "unknown_class_names": [self.class_names[index] for index in self.unknown_class_ids],
            "confidence_quantile": self.confidence_quantile,
            "threshold_source": "inner_validation_known_samples_only",
        }


def load_open_set_spec(
    path: str | Path, *, fold: int, observed_classes: Sequence[str]
) -> OpenSetProtocolSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("open-set protocol schema_version must be 1")
    declared_classes = tuple(str(value) for value in payload.get("class_order", []))
    if declared_classes != tuple(observed_classes):
        raise ValueError(
            "open-set class_order does not match the dataset: "
            f"{declared_classes} != {tuple(observed_classes)}"
        )
    fold_payload = (payload.get("folds") or {}).get(int(fold))
    if fold_payload is None:
        fold_payload = (payload.get("folds") or {}).get(str(int(fold)))
    if not fold_payload:
        raise ValueError(f"open-set protocol has no fold {fold}")
    unknown_names = tuple(str(value) for value in fold_payload.get("unknown", []))
    if len(unknown_names) != 3 or len(set(unknown_names)) != 3:
        raise ValueError(f"fold {fold} must declare exactly three unique unknown classes")
    unknown_ids = tuple(declared_classes.index(name) for name in unknown_names)
    known_ids = tuple(index for index in range(len(declared_classes)) if index not in unknown_ids)
    if len(known_ids) != 5:
        raise ValueError(f"fold {fold} must retain exactly five known classes")
    quantile = float((payload.get("threshold_calibration") or {}).get("confidence_quantile", 0.05))
    if not 0.0 < quantile < 1.0:
        raise ValueError("confidence_quantile must be in (0, 1)")
    return OpenSetProtocolSpec(
        protocol=str(payload["protocol"]),
        fold=int(fold),
        class_names=declared_classes,
        known_class_ids=known_ids,
        unknown_class_ids=unknown_ids,
        confidence_quantile=quantile,
    )


def create_open_set_protocol_dataloaders(
    data_dir: str | Path,
    *,
    protocol_path: str | Path,
    fold: int,
    seed: int,
    batch_size: int,
    image_size: int,
    inner_val_fraction: float,
    num_workers: int,
    pin_memory: bool = True,
) -> dict[str, Any]:
    train_view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=image_size,
        is_training=True,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=1,
        image_error_policy="raise",
    )
    eval_view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=image_size,
        is_training=False,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=1,
        image_error_policy="raise",
    )
    if len(train_view) != len(eval_view):
        raise RuntimeError("open-set train/eval dataset views disagree")
    spec = load_open_set_spec(protocol_path, fold=fold, observed_classes=train_view.classes)
    split = build_protocol_split(
        train_view.samples,
        data_dir,
        fold_index=fold,
        seed=seed,
        inner_val_fraction=inner_val_fraction,
        num_classes=len(train_view.classes),
    )
    known = set(spec.known_class_ids)
    train_indices = [
        index for index in split.train_indices if int(train_view.samples[index][1]) in known
    ]
    inner_indices = [
        index for index in split.inner_val_indices if int(eval_view.samples[index][1]) in known
    ]
    if not train_indices or not inner_indices:
        raise RuntimeError("open-set known-only train/inner split is empty")
    label_map = spec.original_to_model
    train_dataset = RemappedKnownDataset(train_view, train_indices, label_map)
    inner_dataset = RemappedKnownDataset(eval_view, inner_indices, label_map)
    outer_indices = list(split.outer_val_indices)
    outer_dataset = Subset(eval_view, outer_indices)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=int(seed) + int(fold) * 100_000,
    )
    inner_loader = create_dataloader(
        inner_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=int(seed) + int(fold) * 100_000 + 1,
    )
    outer_loader = create_dataloader(
        outer_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=int(seed) + int(fold) * 100_000 + 2,
    )
    counts = np.zeros(len(spec.known_class_ids), dtype=np.float64)
    for index in train_indices:
        counts[label_map[int(train_view.samples[index][1])]] += 1
    return {
        "train_loader": train_loader,
        "inner_val_loader": inner_loader,
        "outer_val_loader": outer_loader,
        "eval_dataset": eval_view,
        "split": split,
        "outer_indices": outer_indices,
        "num_classes": len(spec.known_class_ids),
        "classes": [spec.class_names[index] for index in spec.known_class_ids],
        "train_class_counts": torch.tensor(counts, dtype=torch.float32),
        "open_set_spec": spec,
    }


@torch.no_grad()
def calibrate_msp_threshold(
    model: nn.Module,
    data_loader,
    *,
    device: str,
    confidence_quantile: float,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    confidences = []
    for inputs, _targets in data_loader:
        probabilities = torch.softmax(model(inputs.to(device)), dim=1)
        confidences.extend(float(value) for value in probabilities.max(dim=1).values.cpu())
    if not confidences:
        raise RuntimeError("inner validation produced no confidence values")
    threshold = float(np.quantile(np.asarray(confidences), confidence_quantile, method="lower"))
    return threshold, {
        "source": "inner_validation_known_samples_only",
        "rule": "accept_95_percent_known",
        "confidence_quantile": float(confidence_quantile),
        "threshold": threshold,
        "sample_count": len(confidences),
    }


@torch.no_grad()
def evaluate_open_set_classifier(
    model: nn.Module,
    data_loader,
    *,
    device: str,
    eval_samples: Sequence[tuple[str, int]],
    outer_indices: Sequence[int],
    spec: OpenSetProtocolSpec,
    threshold: float,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    cursor = 0
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []
    model_to_original = spec.model_to_original
    for inputs, batch_targets in data_loader:
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence_values, model_predictions = probabilities.max(dim=1)
        for offset in range(int(batch_targets.size(0))):
            dataset_index = int(outer_indices[cursor + offset])
            sample_path, dataset_label = eval_samples[dataset_index]
            target = int(batch_targets[offset].item())
            if target != int(dataset_label):
                raise RuntimeError("outer open-set label disagrees with dataset metadata")
            model_prediction = int(model_predictions[offset].item())
            original_prediction = int(model_to_original[model_prediction])
            confidence = float(confidence_values[offset].item())
            targets.append(target)
            predictions.append(original_prediction)
            confidences.append(confidence)
            rejected = confidence < threshold
            rows.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "sample_index": dataset_index,
                    "sample_id": str(Path(sample_path).resolve()),
                    "target": target,
                    "prediction": original_prediction,
                    "confidence": confidence,
                    "unknown_score": 1.0 - confidence,
                    "rejected_as_unknown": rejected,
                    "target_is_known": target in spec.known_class_ids,
                    "logits": [float(value) for value in logits[offset].detach().cpu().tolist()],
                }
            )
        cursor += int(batch_targets.size(0))
    if cursor != len(outer_indices):
        raise RuntimeError(f"open-set prediction cursor mismatch: {cursor} != {len(outer_indices)}")
    summary = open_set_summary(
        targets,
        predictions,
        confidences,
        known_class_ids=spec.known_class_ids,
        confidence_threshold=threshold,
    )
    return summary, rows
