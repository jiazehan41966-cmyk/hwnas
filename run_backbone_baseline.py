#!/usr/bin/env python3
"""Backbone baseline experiment entrypoint."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data import create_dummy_dataloaders
from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.hardware import BackboneCostEstimator
from hwnas_fpga.models import BackboneCandidate, build_backbone, default_backbone_candidates
from hwnas_fpga.runtime import (
    build_constraints,
    build_hardware_spec,
    create_data_pipeline,
    load_config,
    pick,
)
from hwnas_fpga.training import create_optimizer


BACKBONE_ARCHITECTURE_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "simplecnn": {
        "stem": None,
        "stages": (
            {
                "stage_name": "conv16",
                "block_type": "conv_bn_relu",
                "depth": 1,
                "channels": 16,
                "stride": 1,
                "downsample_location": "stage_end",
                "downsample_op": "maxpool2d",
                "downsample_stride": 2,
            },
            {
                "stage_name": "conv32",
                "block_type": "conv_bn_relu",
                "depth": 1,
                "channels": 32,
                "stride": 1,
                "downsample_location": "stage_end",
                "downsample_op": "maxpool2d",
                "downsample_stride": 2,
            },
            {
                "stage_name": "conv64",
                "block_type": "conv_bn_relu",
                "depth": 1,
                "channels": 64,
                "stride": 1,
                "downsample_location": None,
                "downsample_op": None,
                "downsample_stride": 1,
            },
        ),
        "head": {
            "pooling": "adaptive_avg_pool_4x4",
            "classifier_hidden_dim": 80,
        },
    },
    "mobilenet_v2": {
        "stem": {
            "out_channels": 32,
            "stride": 2,
            "block_type": "conv_bn_relu6",
        },
        "stages": (
            {"stage_name": "ir16", "block_type": "inverted_residual", "depth": 1, "channels": 16, "stride": 1, "expand": 1, "kernel": 3},
            {"stage_name": "ir24", "block_type": "inverted_residual", "depth": 2, "channels": 24, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "ir32", "block_type": "inverted_residual", "depth": 3, "channels": 32, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "ir64", "block_type": "inverted_residual", "depth": 4, "channels": 64, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "ir96", "block_type": "inverted_residual", "depth": 3, "channels": 96, "stride": 1, "expand": 6, "kernel": 3},
            {"stage_name": "ir160", "block_type": "inverted_residual", "depth": 3, "channels": 160, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "ir320", "block_type": "inverted_residual", "depth": 1, "channels": 320, "stride": 1, "expand": 6, "kernel": 3},
        ),
        "head": {
            "out_channels": 1280,
            "pooling": "adaptive_avg_pool_1x1",
        },
    },
    "fbnet_like": {
        "stem": {
            "out_channels": 16,
            "stride": 2,
            "block_type": "conv_bn_relu",
        },
        "stages": (
            {"stage_name": "fb16", "block_type": "fbnet_bottleneck", "depth": 1, "channels": 16, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
            {"stage_name": "fb24", "block_type": "fbnet_bottleneck", "depth": 4, "channels": 24, "stride": 2, "expand": 3, "kernel": 3, "groups": 2},
            {"stage_name": "fb32", "block_type": "fbnet_bottleneck", "depth": 4, "channels": 32, "stride": 2, "expand": 3, "kernel": 5, "groups": 2},
            {"stage_name": "fb64", "block_type": "fbnet_bottleneck", "depth": 4, "channels": 64, "stride": 2, "expand": 4, "kernel": 5, "groups": 2},
            {"stage_name": "fb112", "block_type": "fbnet_bottleneck", "depth": 4, "channels": 112, "stride": 1, "expand": 4, "kernel": 5, "groups": 2},
            {"stage_name": "fb184", "block_type": "fbnet_bottleneck", "depth": 4, "channels": 184, "stride": 2, "expand": 6, "kernel": 5, "groups": 4},
        ),
        "head": {
            "out_channels": 1280,
            "pooling": "adaptive_avg_pool_1x1",
        },
    },
    "fbnet_a": {
        "stem": {
            "out_channels": 16,
            "stride": 2,
            "block_type": "conv_bn_relu",
        },
        "stages": (
            {"stage_name": "fb16_skip", "block_type": "skip", "depth": 1, "channels": 16, "stride": 1},
            {"stage_name": "fb24", "block_type": "fbnet_reference_ir_mixed", "depth": 4, "channels": 24, "stride": 2, "expand": 3, "kernel": 3},
            {"stage_name": "fb32", "block_type": "fbnet_reference_ir_mixed", "depth": 4, "channels": 32, "stride": 2, "expand": 6, "kernel": 5},
            {"stage_name": "fb112", "block_type": "fbnet_reference_ir_mixed", "depth": 8, "channels": 112, "stride": 2, "expand": 6, "kernel": 5, "groups": 2},
            {"stage_name": "fb352", "block_type": "fbnet_reference_ir_mixed", "depth": 5, "channels": 352, "stride": 2, "expand": 6, "kernel": 5},
        ),
        "head": {
            "out_channels": 1504,
            "pooling": "adaptive_avg_pool_1x1",
        },
    },
    "shufflenet_v2": {
        "stem": {
            "out_channels": 24,
            "stride": 2,
            "block_type": "conv_bn_relu",
            "post_stem_downsample": {
                "op": "maxpool2d",
                "stride": 2,
            },
        },
        "stages": (
            {"stage_name": "shuffle116", "block_type": "shuffle_stage", "depth": 4, "channels": 116, "stride": 2},
            {"stage_name": "shuffle232", "block_type": "shuffle_stage", "depth": 8, "channels": 232, "stride": 2},
            {"stage_name": "shuffle464", "block_type": "shuffle_stage", "depth": 4, "channels": 464, "stride": 2},
        ),
        "head": {
            "out_channels": 1024,
            "pooling": "adaptive_avg_pool_1x1",
        },
    },
    "efficientnet_b0": {
        "stem": {
            "out_channels": 32,
            "stride": 2,
            "block_type": "conv_bn_silu",
        },
        "stages": (
            {"stage_name": "mb16", "block_type": "mbconv", "depth": 1, "channels": 16, "stride": 1, "expand": 1, "kernel": 3},
            {"stage_name": "mb24", "block_type": "mbconv", "depth": 2, "channels": 24, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "mb40", "block_type": "mbconv", "depth": 2, "channels": 40, "stride": 2, "expand": 6, "kernel": 5},
            {"stage_name": "mb80", "block_type": "mbconv", "depth": 3, "channels": 80, "stride": 2, "expand": 6, "kernel": 3},
            {"stage_name": "mb112", "block_type": "mbconv", "depth": 3, "channels": 112, "stride": 1, "expand": 6, "kernel": 5},
            {"stage_name": "mb192", "block_type": "mbconv", "depth": 4, "channels": 192, "stride": 2, "expand": 6, "kernel": 5},
            {"stage_name": "mb320", "block_type": "mbconv", "depth": 1, "channels": 320, "stride": 1, "expand": 6, "kernel": 3},
        ),
        "head": {
            "out_channels": 1280,
            "pooling": "adaptive_avg_pool_1x1",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW-NAS backbone baseline experiment")
    parser.add_argument("--config", type=str, default="configs/backbone_baseline.yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=("dummy", "nksid"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Run only specific arch_id/name; repeatable",
    )
    parser.add_argument("--cpu-latency-runs", type=int, default=None)
    parser.add_argument("--cpu-latency-warmup", type=int, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def div_up(value: int, divisor: int) -> int:
    return int(math.ceil(float(value) / float(max(1, divisor))))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_dataset(dataset: Any) -> Any:
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def extract_class_names(dataset: Any, num_classes: int) -> list[str]:
    base_dataset = unwrap_dataset(dataset)
    class_names = getattr(base_dataset, "classes", None)
    if isinstance(class_names, list) and len(class_names) >= num_classes:
        return [str(name) for name in class_names[:num_classes]]
    return [f"class_{index}" for index in range(num_classes)]


def create_baseline_data_pipeline(
    *,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    batch_size: int,
    image_size: int,
    input_channels: int,
    num_classes: Optional[int],
    fold: int,
    num_workers: int,
    device: str,
) -> tuple[Any, Any, Optional[torch.Tensor], int, list[str]]:
    if dataset_name == "dummy":
        resolved_num_classes = num_classes or int(dataset_cfg.get("num_classes", 8))
        num_train_samples = int(dataset_cfg.get("num_train_samples", 128))
        num_val_samples = int(dataset_cfg.get("num_val_samples", 64))
        train_loader, val_loader = create_dummy_dataloaders(
            batch_size=batch_size,
            image_size=image_size,
            num_classes=resolved_num_classes,
            input_channels=input_channels,
            num_train=num_train_samples,
            num_val=num_val_samples,
        )
        class_names = [f"class_{index}" for index in range(resolved_num_classes)]
        return train_loader, val_loader, None, resolved_num_classes, class_names

    train_loader, val_loader, class_weights, resolved_num_classes = create_data_pipeline(
        dataset_name=dataset_name,
        data_dir=dataset_cfg.get("data_dir"),
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        input_channels=input_channels,
        fold=fold,
        num_workers=num_workers,
        device=device,
        use_kfold=bool(dataset_cfg.get("use_kfold", True)),
        valid_size=dataset_cfg.get("valid_size"),
        split_seed=int(dataset_cfg.get("split_seed", 42)),
    )
    class_names = extract_class_names(train_loader.dataset, resolved_num_classes)
    return train_loader, val_loader, class_weights, resolved_num_classes, class_names


def topk_accuracy_hits(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    if outputs.ndim != 2:
        raise ValueError("Expected classifier logits with shape [batch, num_classes]")
    k = max(1, min(k, outputs.shape[1]))
    topk = outputs.topk(k, dim=1).indices
    return int(topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


def summarize_confusion_matrix(
    confusion: torch.Tensor,
    *,
    class_names: list[str],
) -> dict[str, Any]:
    supports = confusion.sum(dim=1)
    total = int(supports.sum().item())
    per_class: list[dict[str, Any]] = []
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for class_index, class_name in enumerate(class_names):
        tp = float(confusion[class_index, class_index].item())
        fp = float(confusion[:, class_index].sum().item() - tp)
        fn = float(confusion[class_index, :].sum().item() - tp)
        support = int(supports[class_index].item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        macro_f1 += f1
        weighted_f1 += f1 * support
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = macro_f1 / max(1, len(class_names))
    weighted_f1 = weighted_f1 / max(1, total)
    accuracy = float(confusion.diag().sum().item()) / max(1, total)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_backbone(
    model: nn.Module,
    data_loader: Any,
    *,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    class_names: list[str],
    topk: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_topk = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += targets.size(0)
        predictions = outputs.argmax(dim=1)
        total_top1 += int(predictions.eq(targets).sum().item())
        total_topk += topk_accuracy_hits(outputs, targets, topk)

        for target, prediction in zip(targets.view(-1).tolist(), predictions.view(-1).tolist()):
            confusion[int(target), int(prediction)] += 1

    summary = summarize_confusion_matrix(confusion, class_names=class_names)
    summary.update(
        {
            "loss": total_loss / max(1, total_samples),
            "top1": total_top1 / max(1, total_samples),
            "top5": total_topk / max(1, total_samples),
            "num_samples": total_samples,
        }
    )
    return summary


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if epochs <= 1:
        return None
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=min_lr,
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    if warmup_epochs > 0:
        return LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)


def build_architecture_summary(
    *,
    arch_id: str,
    display_name: str,
    backbone_name: str,
    input_channels: int,
    image_size: int,
    num_classes: int,
    pretrained_requested: bool,
    pretrained_loaded: Optional[bool],
) -> dict[str, Any]:
    blueprint = BACKBONE_ARCHITECTURE_BLUEPRINTS.get(backbone_name)
    if blueprint is None:
        return {
            "arch_id": arch_id,
            "display_name": display_name,
            "backbone": backbone_name,
            "input_channels": input_channels,
            "input_resolution": image_size,
            "stage_count": None,
            "stages": [],
            "downsample_positions": [],
            "notes": "No architecture blueprint registered for this backbone.",
            "pretrained_requested": pretrained_requested,
            "pretrained_loaded": pretrained_loaded,
        }

    summary: dict[str, Any] = {
        "arch_id": arch_id,
        "display_name": display_name,
        "backbone": backbone_name,
        "input_channels": input_channels,
        "input_resolution": image_size,
        "pretrained_requested": pretrained_requested,
        "pretrained_loaded": pretrained_loaded,
        "stage_count": len(tuple(blueprint.get("stages", ()))),
        "downsample_positions": [],
        "notes": "Macro-architecture summary used for downstream anchor selection and later search-space design.",
    }

    current_resolution = int(image_size)
    stem_cfg = blueprint.get("stem")
    if stem_cfg is not None:
        stem_stride = int(stem_cfg.get("stride", 1))
        stem_output = div_up(current_resolution, stem_stride)
        stem_summary: dict[str, Any] = {
            "block_type": stem_cfg.get("block_type"),
            "out_channels": int(stem_cfg.get("out_channels", 0)),
            "width": int(stem_cfg.get("out_channels", 0)),
            "stride": stem_stride,
            "input_resolution": current_resolution,
            "output_resolution": stem_output,
        }
        if stem_stride > 1:
            summary["downsample_positions"].append("stem")
        current_resolution = stem_output

        post_stem_cfg = stem_cfg.get("post_stem_downsample")
        if post_stem_cfg is not None:
            post_stride = int(post_stem_cfg.get("stride", 1))
            post_output = div_up(current_resolution, post_stride)
            stem_summary["post_stem_downsample"] = {
                "op": post_stem_cfg.get("op"),
                "stride": post_stride,
                "input_resolution": current_resolution,
                "output_resolution": post_output,
            }
            if post_stride > 1:
                summary["downsample_positions"].append("stem_post_downsample")
            current_resolution = post_output

        summary["stem"] = stem_summary
    else:
        summary["stem"] = None

    stages: list[dict[str, Any]] = []
    for stage_index, stage_cfg in enumerate(blueprint.get("stages", ()), start=1):
        stage_input = current_resolution
        stage_stride = int(stage_cfg.get("stride", 1))
        downsample_location = stage_cfg.get("downsample_location")
        downsample_op = stage_cfg.get("downsample_op")
        output_resolution = stage_input

        if stage_stride > 1:
            output_resolution = div_up(stage_input, stage_stride)
            summary["downsample_positions"].append(f"stage_{stage_index}")
        elif int(stage_cfg.get("downsample_stride", 1)) > 1:
            extra_stride = int(stage_cfg["downsample_stride"])
            output_resolution = div_up(stage_input, extra_stride)
            summary["downsample_positions"].append(f"stage_{stage_index}_{downsample_location or 'downsample'}")
        else:
            extra_stride = 1

        stage_summary: dict[str, Any] = {
            "stage_index": stage_index,
            "stage_name": stage_cfg.get("stage_name", f"stage_{stage_index}"),
            "block_type": stage_cfg.get("block_type"),
            "depth": int(stage_cfg.get("depth", 1)),
            "out_channels": int(stage_cfg.get("channels", 0)),
            "width": int(stage_cfg.get("channels", 0)),
            "stride": stage_stride,
            "input_resolution": stage_input,
            "output_resolution": output_resolution,
            "downsample": output_resolution != stage_input,
            "downsample_location": downsample_location or ("stage_start" if stage_stride > 1 else None),
            "downsample_op": downsample_op,
        }
        if "expand" in stage_cfg:
            stage_summary["expand_ratio"] = int(stage_cfg["expand"])
        if "kernel" in stage_cfg:
            stage_summary["kernel_size"] = int(stage_cfg["kernel"])
        if "groups" in stage_cfg:
            stage_summary["groups"] = int(stage_cfg["groups"])
        if stage_stride <= 1 and extra_stride > 1:
            stage_summary["downsample_stride"] = extra_stride

        stages.append(stage_summary)
        current_resolution = output_resolution

    summary["stages"] = stages
    summary["head"] = {
        **dict(blueprint.get("head", {})),
        "num_classes": num_classes,
        "input_resolution": current_resolution,
    }
    return summary


def build_training_strategy_summary(
    *,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
    batch_size: int,
    image_size: int,
    input_channels: int,
    fold: int,
    class_weights: Optional[torch.Tensor],
    topk: int,
) -> dict[str, Any]:
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    epochs = int(training_cfg.get("epochs", 10))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        scheduler_name = "linear_warmup_plus_cosine_decay"
    elif warmup_epochs > 0:
        scheduler_name = "linear_warmup"
    else:
        scheduler_name = "cosine_annealing"

    augmentation_profile = []
    if dataset_name == "nksid":
        augmentation_profile = [
            "resize",
            "random_horizontal_flip",
            "random_vertical_flip",
            "random_rotation",
            "random_affine",
            "color_jitter",
            "speckle_noise",
            "normalize",
        ]

    return {
        "framework": "supervised single-backbone baseline screening",
        "purpose": "train each backbone independently, rank them by macro_f1 under hardware constraints, and export anchors for later search/selection.",
        "downstream_use": {
            "accuracy_anchor": "strongest feasible backbone by macro_f1/top1",
            "search_anchor": "mobile backbone family used to seed later search",
            "lightweight_anchor": "latency-oriented backbone within a small macro_f1 drop",
        },
        "dataset_protocol": {
            "dataset": dataset_name,
            "batch_size": batch_size,
            "fold": fold,
            "use_kfold": bool(dataset_cfg.get("use_kfold", True)),
            "input_channels": input_channels,
            "input_resolution": image_size,
            "augmentation_profile": augmentation_profile,
        },
        "loss": {
            "name": "cross_entropy",
            "class_balancing": {
                "enabled": class_weights is not None,
                "method": "inverse_frequency_class_weights" if class_weights is not None else None,
            },
        },
        "optimizer": {
            "name": str(training_cfg.get("optimizer", "adamw")),
            "lr": float(training_cfg.get("lr", 0.001)),
            "weight_decay": float(training_cfg.get("weight_decay", 0.0001)),
        },
        "scheduler": {
            "name": scheduler_name,
            "warmup_epochs": warmup_epochs,
            "min_lr": float(training_cfg.get("min_lr", 1.0e-5)),
        },
        "selection": {
            "selection_metric": str(training_cfg.get("selection_metric", "macro_f1")),
            "early_stopping_patience": training_cfg.get("early_stopping_patience"),
            "max_epochs_per_backbone": epochs,
            "checkpoint_policy": "keep best epoch by selection metric",
            "latency_tiebreak_metric": str(
                dict(baseline_cfg.get("ranking", {})).get("latency_tiebreak_metric", "cpu_latency_ms")
            ),
        },
        "anchor_selection": {
            "search_anchor_families": list(
                dict(baseline_cfg.get("pool_selection", {})).get(
                    "search_anchor_families",
                    ["mobilenet_v2", "fbnet_a", "shufflenet_v2"],
                )
            ),
            "prefer_pretrained_for_search_anchor": bool(
                dict(baseline_cfg.get("pool_selection", {})).get(
                    "prefer_pretrained_for_search_anchor",
                    True,
                )
            ),
            "lightweight_metric": str(
                dict(baseline_cfg.get("pool_selection", {})).get(
                    "lightweight_metric",
                    "fpga_latency_ms",
                )
            ),
            "lightweight_max_macro_drop": float(
                dict(baseline_cfg.get("pool_selection", {})).get(
                    "lightweight_max_macro_drop",
                    0.10,
                )
            ),
        },
        "validation": {
            "run_every": "epoch",
            "metrics": ["loss", "top1", f"top{topk}", "macro_f1", "weighted_f1"],
        },
        "execution_order": [
            "estimate hardware cost for the candidate",
            "train candidate backbone",
            "validate every epoch",
            "early-stop if the selection metric stalls",
            "save per-backbone JSON and checkpoint",
            "export run-level best backbone and selected pool",
        ],
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: Any,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += targets.size(0)
        total_correct += int(outputs.argmax(dim=1).eq(targets).sum().item())

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def train_backbone(
    model: nn.Module,
    *,
    train_loader: Any,
    val_loader: Any,
    training_cfg: dict[str, Any],
    class_weights: Optional[torch.Tensor],
    device: str,
    num_classes: int,
    class_names: list[str],
    topk: int,
) -> tuple[nn.Module, dict[str, Any], dict[str, list[float]], dict[str, Any]]:
    optimizer = create_optimizer(
        model,
        optimizer_name=str(training_cfg.get("optimizer", "adamw")),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=int(training_cfg.get("epochs", 10)),
        warmup_epochs=int(training_cfg.get("warmup_epochs", 0)),
        min_lr=float(training_cfg.get("min_lr", 1.0e-5)),
    )
    patience = training_cfg.get("early_stopping_patience")
    patience = None if patience is None else int(patience)

    model = model.to(device)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "lr": [],
    }

    best_state = deepcopy(model.state_dict())
    best_eval: dict[str, Any] | None = None
    best_epoch = 0
    best_score = float("-inf")
    patience_counter = 0
    train_start = perf_counter()

    epochs = int(training_cfg.get("epochs", 10))
    selection_metric = str(training_cfg.get("selection_metric", "macro_f1"))
    if selection_metric != "macro_f1":
        selection_metric = "macro_f1"

    for epoch in range(epochs):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_eval = evaluate_backbone(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            topk=topk,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(float(val_eval["loss"]))
        history["val_top1"].append(float(val_eval["top1"]))
        history["val_top5"].append(float(val_eval["top5"]))
        history["val_macro_f1"].append(float(val_eval["macro_f1"]))
        history["val_weighted_f1"].append(float(val_eval["weighted_f1"]))
        history["lr"].append(current_lr)

        score = float(val_eval["macro_f1"])
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_eval['loss']:.4f}, val_top1={val_eval['top1']:.4f}, "
            f"val_macro_f1={val_eval['macro_f1']:.4f}, lr={current_lr:.6f}"
        )

        if score > best_score:
            best_state = deepcopy(model.state_dict())
            best_eval = deepcopy(val_eval)
            best_epoch = epoch + 1
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_state)
    if best_eval is None:
        best_eval = evaluate_backbone(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            topk=topk,
        )

    elapsed = perf_counter() - train_start
    summary = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history["train_loss"]),
        "train_time_sec": elapsed,
        "selection_metric": selection_metric,
        "best_macro_f1": float(best_eval["macro_f1"]),
        "best_top1": float(best_eval["top1"]),
        "best_top5": float(best_eval["top5"]),
        "best_val_loss": float(best_eval["loss"]),
    }
    return model, summary, history, best_eval


def resolve_candidates(
    baseline_cfg: dict[str, Any],
    requested: Optional[Iterable[str]],
) -> list[BackboneCandidate]:
    raw_candidates = baseline_cfg.get("candidates")
    if raw_candidates:
        candidates = [BackboneCandidate.from_dict(candidate) for candidate in raw_candidates]
    else:
        candidates = list(default_backbone_candidates())

    if not requested:
        return candidates

    filters = {item.strip().lower() for item in requested if item and item.strip()}
    selected = [
        candidate
        for candidate in candidates
        if candidate.arch_id.lower() in filters or candidate.name.lower() in filters
    ]
    if not selected:
        raise ValueError(f"No backbone candidates matched filters: {sorted(filters)}")
    return selected


def _normalize_anchor_metric(metric: Any, *, default: str) -> str:
    normalized = str(metric or default).strip().lower()
    aliases = {
        "fpga": "fpga_latency_ms",
        "fpga_latency": "fpga_latency_ms",
        "fpga_latency_ms": "fpga_latency_ms",
        "latency": "fpga_latency_ms",
        "latency_ms": "fpga_latency_ms",
        "cpu": "cpu_latency_ms",
        "cpu_latency": "cpu_latency_ms",
        "cpu_latency_ms": "cpu_latency_ms",
        "params": "params",
        "macs": "macs",
        "power": "power_w",
        "power_w": "power_w",
    }
    return aliases.get(normalized, normalized)


def _metric_value_from_result(result: dict[str, Any], metric: str) -> float:
    normalized = _normalize_anchor_metric(metric, default="fpga_latency_ms")
    cost_estimate = result["cost_estimate"]
    evaluation = result["evaluation"]
    mapping: dict[str, float] = {
        "fpga_latency_ms": float(cost_estimate["latency_ms"]),
        "cpu_latency_ms": float(cost_estimate["cpu_latency_ms"]),
        "params": float(cost_estimate["params"]),
        "macs": float(cost_estimate["macs"]),
        "power_w": float(cost_estimate["power_w"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "top1": float(evaluation["top1"]),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported anchor metric: {metric}")
    return mapping[normalized]


def compare_results(
    current: dict[str, Any],
    best: Optional[dict[str, Any]],
    *,
    latency_tiebreak_metric: str = "cpu_latency_ms",
) -> bool:
    if best is None:
        return True
    current_feasible = bool(current["cost_estimate"]["feasible"])
    best_feasible = bool(best["cost_estimate"]["feasible"])
    if current_feasible != best_feasible:
        return current_feasible
    current_macro = float(current["evaluation"]["macro_f1"])
    best_macro = float(best["evaluation"]["macro_f1"])
    if not math.isclose(current_macro, best_macro):
        return current_macro > best_macro
    current_top1 = float(current["evaluation"]["top1"])
    best_top1 = float(best["evaluation"]["top1"])
    if not math.isclose(current_top1, best_top1):
        return current_top1 > best_top1
    return _metric_value_from_result(current, latency_tiebreak_metric) < _metric_value_from_result(
        best,
        latency_tiebreak_metric,
    )


def _result_sort_key(
    result: dict[str, Any],
    *,
    latency_tiebreak_metric: str = "cpu_latency_ms",
) -> tuple[float, float, float]:
    evaluation = result["evaluation"]
    return (
        float(evaluation["macro_f1"]),
        float(evaluation["top1"]),
        -_metric_value_from_result(result, latency_tiebreak_metric),
    )


def _lightweight_sort_key(
    result: dict[str, Any],
    *,
    lightweight_metric: str = "fpga_latency_ms",
) -> tuple[float, int, float]:
    cost_estimate = result["cost_estimate"]
    return (
        _metric_value_from_result(result, lightweight_metric),
        int(cost_estimate["params"]),
        -float(result["evaluation"]["macro_f1"]),
    )


def build_pool_candidate_payload(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    build_metadata = result["build_metadata"]
    evaluation = result["evaluation"]
    cost_estimate = result["cost_estimate"]
    return {
        "arch_id": result["arch_id"],
        "display_name": result["display_name"],
        "backbone": candidate["name"],
        "pretrained_requested": candidate["pretrained"],
        "pretrained_loaded": build_metadata["pretrained_loaded"],
        "macro_f1": float(evaluation["macro_f1"]),
        "top1": float(evaluation["top1"]),
        "top5": float(evaluation["top5"]),
        "weighted_f1": float(evaluation["weighted_f1"]),
        "cpu_latency_ms": float(cost_estimate["cpu_latency_ms"]),
        "fpga_latency_ms": float(cost_estimate["latency_ms"]),
        "params": int(cost_estimate["params"]),
        "macs": int(cost_estimate["macs"]),
        "peak_dsp": int(cost_estimate["peak_dsp"]),
        "peak_bram": int(cost_estimate["peak_bram"]),
        "peak_lut": int(cost_estimate["peak_lut"]),
        "power_w": float(cost_estimate["power_w"]),
        "feasible": bool(cost_estimate["feasible"]),
        "violations": list(cost_estimate["violations"]),
    }


def select_backbone_pool(
    results: list[dict[str, Any]],
    *,
    baseline_cfg: dict[str, Any],
    dataset_name: str,
    board_name: str,
) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot select backbone pool from empty results")

    pool_cfg = dict(baseline_cfg.get("pool_selection", {}))
    ranking_cfg = dict(baseline_cfg.get("ranking", {}))
    role_profile_map = {
        "search_anchor": "mobile_anchor",
        "accuracy_anchor": "accuracy_biased",
        "lightweight_anchor": "lightweight_sonar",
        **dict(pool_cfg.get("role_profile_map", {})),
    }
    latency_tiebreak_metric = str(ranking_cfg.get("latency_tiebreak_metric", "cpu_latency_ms"))
    lightweight_metric = str(pool_cfg.get("lightweight_metric", "fpga_latency_ms"))

    accuracy_anchor = max(
        results,
        key=lambda result: _result_sort_key(
            result,
            latency_tiebreak_metric=latency_tiebreak_metric,
        ),
    )

    search_anchor_families = {
        str(name).strip().lower()
        for name in pool_cfg.get("search_anchor_families", ["mobilenet_v2", "fbnet_a", "shufflenet_v2"])
        if str(name).strip()
    }
    search_candidates = [
        result
        for result in results
        if str(result["candidate"]["name"]).lower() in search_anchor_families
    ] or list(results)
    if bool(pool_cfg.get("prefer_pretrained_for_search_anchor", True)):
        pretrained_search_candidates = [
            result for result in search_candidates if bool(result["build_metadata"]["pretrained_loaded"])
        ]
        if pretrained_search_candidates:
            search_candidates = pretrained_search_candidates
    search_anchor = max(
        search_candidates,
        key=lambda result: _result_sort_key(
            result,
            latency_tiebreak_metric=latency_tiebreak_metric,
        ),
    )

    max_macro_drop = float(pool_cfg.get("lightweight_max_macro_drop", 0.10))
    macro_threshold = float(accuracy_anchor["evaluation"]["macro_f1"]) - max_macro_drop
    lightweight_candidates = [
        result
        for result in results
        if float(result["evaluation"]["macro_f1"]) >= macro_threshold
    ] or list(results)
    lightweight_anchor = min(
        lightweight_candidates,
        key=lambda result: _lightweight_sort_key(
            result,
            lightweight_metric=lightweight_metric,
        ),
    )

    roles = {
        "accuracy_anchor": build_pool_candidate_payload(accuracy_anchor),
        "search_anchor": build_pool_candidate_payload(search_anchor),
        "lightweight_anchor": build_pool_candidate_payload(lightweight_anchor),
    }
    recommended_profiles = {
        profile_name: {
            "source_role": role_name,
            "source_arch_id": roles[role_name]["arch_id"],
            "source_backbone": roles[role_name]["backbone"],
        }
        for role_name, profile_name in role_profile_map.items()
        if role_name in roles
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "board": board_name,
        "selection_metric": str(baseline_cfg.get("selection_metric", "macro_f1")),
        "latency_tiebreak_metric": latency_tiebreak_metric,
        "lightweight_metric": lightweight_metric,
        "roles": roles,
        "recommended_profiles": recommended_profiles,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "arch_id",
        "display_name",
        "backbone",
        "pretrained_requested",
        "pretrained_loaded",
        "params",
        "macs",
        "cpu_latency_ms",
        "top1",
        "top5",
        "macro_f1",
        "weighted_f1",
        "peak_dsp",
        "peak_bram",
        "peak_lut",
        "fpga_latency_ms",
        "power_w",
        "feasible",
        "violations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project_cfg = config.get("project", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    baseline_cfg = config.get("baseline", {})
    hardware_cfg = config.get("hardware", {})

    seed = int(pick(args.seed, project_cfg.get("seed"), 42))
    set_seed(seed)

    dataset_name = pick(args.dataset, dataset_cfg.get("name"), "dummy")
    if dataset_name not in {"dummy", "nksid"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    default_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = str(pick(args.device, None, default_device))
    image_size = int(pick(args.image_size, dataset_cfg.get("image_size"), 224))
    input_channels = int(dataset_cfg.get("input_channels", 1))
    batch_size = int(pick(args.batch_size, training_cfg.get("batch_size"), 32))
    num_workers = int(pick(args.num_workers, dataset_cfg.get("num_workers"), 0))
    fold = int(pick(args.fold, dataset_cfg.get("fold"), 0))
    data_dir = pick(args.data_dir, dataset_cfg.get("data_dir"), None)
    num_classes = pick(args.num_classes, dataset_cfg.get("num_classes"), None)
    if data_dir is not None:
        dataset_cfg["data_dir"] = data_dir

    training_cfg = {
        **training_cfg,
        "epochs": int(pick(args.epochs, training_cfg.get("epochs"), 10)),
        "batch_size": batch_size,
        "selection_metric": baseline_cfg.get("selection_metric", "macro_f1"),
    }
    cpu_latency_runs = int(pick(args.cpu_latency_runs, hardware_cfg.get("cpu_latency_runs"), 30))
    cpu_latency_warmup = int(pick(args.cpu_latency_warmup, hardware_cfg.get("cpu_latency_warmup"), 10))
    topk = int(baseline_cfg.get("topk_accuracy", 5))
    strict_pretrained = bool(baseline_cfg.get("strict_pretrained", False))
    dropout = float(training_cfg.get("dropout", 0.2))

    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    if hardware_cfg.get("clock_mhz") is not None:
        hardware_spec = type(hardware_spec)(
            **{
                **hardware_spec.__dict__,
                "clock_mhz": int(hardware_cfg["clock_mhz"]),
            }
        )

    tracker = ExperimentTracker(
        output_root=output_root,
        project_name=str(project_cfg.get("name", "backbone-baseline")),
        script_name="run_backbone_baseline",
        search_method="backbone_baseline",
        dataset_name=str(dataset_name),
        run_name=run_name,
    )
    tracker.save_config(config, vars(args))

    try:
        with tracker.capture_console():
            print(f"Run dir: {tracker.root_dir}")
            print(f"Using device: {device}")
            print(f"Dataset: {dataset_name}")
            print(f"Image size: {image_size}")
            print(f"Input channels: {input_channels}")
            print(f"Board: {hardware_spec.name} @ {hardware_spec.clock_mhz}MHz")

            train_loader, val_loader, class_weights, resolved_num_classes, class_names = create_baseline_data_pipeline(
                dataset_name=str(dataset_name),
                dataset_cfg=dataset_cfg,
                batch_size=batch_size,
                image_size=image_size,
                input_channels=input_channels,
                num_classes=num_classes,
                fold=fold,
                num_workers=num_workers,
                device=device,
            )
            tracker.write_dataset_summary(
                {
                    "dataset": dataset_name,
                    "data_dir": data_dir,
                    "fold": fold,
                    "image_size": image_size,
                    "input_channels": input_channels,
                    "batch_size": batch_size,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "num_classes": resolved_num_classes,
                    "class_names": class_names,
                    "class_weights": None if class_weights is None else class_weights.tolist(),
                }
            )

            estimator = BackboneCostEstimator(
                hardware_spec=hardware_spec,
                constraints=constraints,
                quantization_bits=int(hardware_cfg.get("quantization_bits", 8)),
                cpu_latency_warmup=cpu_latency_warmup,
                cpu_latency_runs=cpu_latency_runs,
            )
            candidates = resolve_candidates(baseline_cfg, args.candidate)
            print(f"Backbone candidates: {[candidate.arch_id for candidate in candidates]}")

            training_strategy = build_training_strategy_summary(
                dataset_name=str(dataset_name),
                dataset_cfg=dataset_cfg,
                training_cfg=training_cfg,
                baseline_cfg=baseline_cfg,
                batch_size=batch_size,
                image_size=image_size,
                input_channels=input_channels,
                fold=fold,
                class_weights=class_weights,
                topk=topk,
            )
            candidate_blueprints = [
                build_architecture_summary(
                    arch_id=candidate.arch_id,
                    display_name=candidate.display_name,
                    backbone_name=str(candidate.name),
                    input_channels=input_channels,
                    image_size=image_size,
                    num_classes=resolved_num_classes,
                    pretrained_requested=bool(candidate.pretrained),
                    pretrained_loaded=None,
                )
                for candidate in candidates
            ]
            write_json(
                tracker.results_dir / "experiment_protocol.json",
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "dataset": dataset_name,
                    "input_resolution": image_size,
                    "input_channels": input_channels,
                    "training_strategy": training_strategy,
                    "candidate_blueprints": candidate_blueprints,
                },
            )

            backbone_results_dir = tracker.results_dir / "backbones"
            backbone_results_dir.mkdir(parents=True, exist_ok=True)
            summary_rows: list[dict[str, Any]] = []
            all_results: list[dict[str, Any]] = []
            selected_pool: dict[str, Any] | None = None
            best_result: dict[str, Any] | None = None
            best_checkpoint_payload: dict[str, Any] | None = None

            for candidate in candidates:
                print(f"\n=== Backbone: {candidate.display_name} ({candidate.arch_id}) ===")
                model, build_metadata = build_backbone(
                    name=candidate.name,
                    num_classes=resolved_num_classes,
                    input_channels=input_channels,
                    pretrained=candidate.pretrained,
                    strict_pretrained=strict_pretrained,
                    dropout=dropout,
                )
                cost_estimate = estimator.estimate(
                    model,
                    input_shape=(1, input_channels, image_size, image_size),
                    cpu_latency_runs=cpu_latency_runs,
                    cpu_latency_warmup=cpu_latency_warmup,
                )
                print(
                    f"Complexity: params={cost_estimate.params:,}, macs={cost_estimate.macs:,}, "
                    f"cpu_latency={cost_estimate.cpu_latency_ms:.2f}ms, "
                    f"fpga_latency={cost_estimate.latency_ms:.2f}ms"
                )
                print(
                    f"Hardware: DSP={cost_estimate.peak_dsp}, BRAM={cost_estimate.peak_bram}, "
                    f"LUT={cost_estimate.peak_lut}, power={cost_estimate.power_w:.2f}W, "
                    f"feasible={cost_estimate.feasible}"
                )

                trained_model, train_summary, history, evaluation = train_backbone(
                    model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    training_cfg=training_cfg,
                    class_weights=class_weights,
                    device=device,
                    num_classes=resolved_num_classes,
                    class_names=class_names,
                    topk=topk,
                )
                architecture_summary = build_architecture_summary(
                    arch_id=candidate.arch_id,
                    display_name=candidate.display_name,
                    backbone_name=str(build_metadata["name"]),
                    input_channels=input_channels,
                    image_size=image_size,
                    num_classes=resolved_num_classes,
                    pretrained_requested=bool(candidate.pretrained),
                    pretrained_loaded=bool(build_metadata.get("pretrained_loaded", False)),
                )

                result = {
                    "arch_id": candidate.arch_id,
                    "display_name": candidate.display_name,
                    "candidate": candidate.to_dict(),
                    "build_metadata": build_metadata,
                    "architecture_summary": architecture_summary,
                    "training_strategy": deepcopy(training_strategy),
                    "train_summary": train_summary,
                    "history": history,
                    "evaluation": evaluation,
                    "cost_estimate": cost_estimate.to_dict(),
                }
                all_results.append(result)
                write_json(backbone_results_dir / f"{candidate.arch_id}.json", result)

                checkpoint_payload = {
                    "candidate": candidate.to_dict(),
                    "build_metadata": build_metadata,
                    "train_summary": train_summary,
                    "evaluation": evaluation,
                    "cost_estimate": cost_estimate.to_dict(),
                    "class_names": class_names,
                    "model_state_dict": trained_model.state_dict(),
                }
                tracker.save_named_checkpoint(f"{candidate.arch_id}.pt", checkpoint_payload)

                summary_rows.append(
                    {
                        "arch_id": candidate.arch_id,
                        "display_name": candidate.display_name,
                        "backbone": candidate.name,
                        "pretrained_requested": candidate.pretrained,
                        "pretrained_loaded": build_metadata["pretrained_loaded"],
                        "params": cost_estimate.params,
                        "macs": cost_estimate.macs,
                        "cpu_latency_ms": f"{cost_estimate.cpu_latency_ms:.4f}",
                        "top1": f"{evaluation['top1']:.4f}",
                        "top5": f"{evaluation['top5']:.4f}",
                        "macro_f1": f"{evaluation['macro_f1']:.4f}",
                        "weighted_f1": f"{evaluation['weighted_f1']:.4f}",
                        "peak_dsp": cost_estimate.peak_dsp,
                        "peak_bram": cost_estimate.peak_bram,
                        "peak_lut": cost_estimate.peak_lut,
                        "fpga_latency_ms": f"{cost_estimate.latency_ms:.4f}",
                        "power_w": f"{cost_estimate.power_w:.4f}",
                        "feasible": cost_estimate.feasible,
                        "violations": "; ".join(cost_estimate.violations),
                    }
                )

                if compare_results(
                    result,
                    best_result,
                    latency_tiebreak_metric=str(
                        dict(baseline_cfg.get("ranking", {})).get(
                            "latency_tiebreak_metric",
                            "cpu_latency_ms",
                        )
                    ),
                ):
                    best_result = result
                    best_checkpoint_payload = checkpoint_payload

            write_json(tracker.results_dir / "backbone_summary.json", summary_rows)
            write_summary_csv(tracker.results_dir / "backbone_summary.csv", summary_rows)
            if bool(baseline_cfg.get("pool_selection", {}).get("export_selected_pool", True)):
                selected_pool = select_backbone_pool(
                    all_results,
                    baseline_cfg=baseline_cfg,
                    dataset_name=str(dataset_name),
                    board_name=str(hardware_spec.name),
                )
                write_json(tracker.results_dir / "selected_backbone_pool.json", selected_pool)
            if best_result is not None:
                write_json(tracker.results_dir / "best_backbone.json", best_result)
                print(
                    f"\nSelected backbone: {best_result['display_name']} "
                    f"(macro_f1={best_result['evaluation']['macro_f1']:.4f}, "
                    f"feasible={best_result['cost_estimate']['feasible']})"
                )
            if best_checkpoint_payload is not None:
                tracker.save_named_checkpoint("best_backbone.pt", best_checkpoint_payload)

            tracker.finalize(
                status="completed",
                candidates=[],
                feasible_candidates=[],
                infeasible_candidates=[],
                best_candidate=None,
                pareto_candidates=None,
                extra={
                    "best_backbone": None if best_result is None else best_result["arch_id"],
                    "backbone_candidates": [candidate.to_dict() for candidate in candidates],
                    "summary_rows": summary_rows,
                    "selected_backbone_pool": selected_pool,
                    "device": device,
                },
            )
    except Exception as exc:
        tracker.mark_failed(str(exc))
        raise


if __name__ == "__main__":
    main()
