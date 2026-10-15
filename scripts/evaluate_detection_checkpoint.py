#!/usr/bin/env python3
"""Evaluate a saved fixed-head detection checkpoint across postprocess settings."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.data import create_yolo_detection_dataloaders
from hwnas_fpga.models import build_detection_model
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config
from hwnas_fpga.training import (
    DenseAnchorDetectionLoss,
    DetectionLossConfig,
    evaluate_detection_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a detection checkpoint without retraining."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pre-nms-topk", type=int, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--thresholds", default="0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.9")
    parser.add_argument("--max-detections", default="1,3,5,10,50")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_checkpoint(args)
    output = _resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    best = payload["best_by_valid_f1"]
    print(
        "best_valid_f1 "
        f"threshold={best['score_threshold']} "
        f"max_detections={best['max_detections']} "
        f"f1={best['valid']['f1']:.4f} "
        f"map={best['valid']['map']:.4f} "
        "(test numbers are reporting-only and not used for selection)",
        flush=True,
    )
    print(f"artifact {output}", flush=True)


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_repo_path(args.config)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    detection_cfg = config.get("detection", {})
    training_cfg = config.get("training", {})
    eval_cfg = training_cfg.get("evaluation", {})
    device = _resolve_device(args.device)

    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else dataset_cfg.get("num_workers", 0))
    loaders = create_yolo_detection_dataloaders(
        _resolve_repo_path(args.data_dir if args.data_dir is not None else dataset_cfg["data_dir"]),
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        seed=int(config.get("project", {}).get("seed", 42)),
    )

    constraints = build_constraints(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_classes=int(dataset_cfg["num_classes"]),
        constraints=constraints,
    )
    model = build_detection_model(
        search_space.baseline_architecture(),
        num_classes=int(dataset_cfg["num_classes"]),
        feature_indices=tuple(detection_cfg["feature_stage_indices"]),
        neck_channels=int(detection_cfg["fixed_neck"]["out_channels"]),
        anchors_per_location=int(detection_cfg["fixed_head"]["anchors_per_location"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    criterion = DenseAnchorDetectionLoss(DetectionLossConfig.from_config(config)).to(device)

    thresholds = _parse_float_csv(args.thresholds)
    max_detection_values = _parse_int_csv(args.max_detections)
    pre_nms_topk = int(args.pre_nms_topk if args.pre_nms_topk is not None else eval_cfg.get("pre_nms_topk", 1000))
    nms_iou_threshold = float(
        args.nms_iou_threshold
        if args.nms_iou_threshold is not None
        else eval_cfg.get("nms_iou_threshold", 0.5)
    )

    rows = []
    for max_detections in max_detection_values:
        for threshold in thresholds:
            valid = evaluate_detection_model(
                model,
                loaders["valid"],
                criterion=criterion,
                device=device,
                score_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                pre_nms_topk=pre_nms_topk,
                max_detections=max_detections,
                max_batches=args.max_eval_batches,
            )
            test = evaluate_detection_model(
                model,
                loaders["test"],
                criterion=criterion,
                device=device,
                score_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                pre_nms_topk=pre_nms_topk,
                max_detections=max_detections,
                max_batches=args.max_eval_batches,
            )
            rows.append(
                {
                    "score_threshold": float(threshold),
                    "max_detections": int(max_detections),
                    "valid": valid,
                    "test": test,
                }
            )
    return {
        "artifact_type": "detection_checkpoint_postprocess_sweep",
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "dataset_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        "class_names": list(getattr(loaders["train"].dataset, "class_names", [])),
        "pre_nms_topk": pre_nms_topk,
        "nms_iou_threshold": nms_iou_threshold,
        "rows": rows,
        "best_by_valid_f1": max(rows, key=lambda row: float(row["valid"].get("f1", 0.0))),
        "test_rows_are_reporting_only": True,
        "selection_split": "valid",
        "reported_test_f1": max(rows, key=lambda row: float(row["test"].get("f1", 0.0))),
        "reported_test_map": max(rows, key=lambda row: float(row["test"].get("map", 0.0))),
    }


def _resolve_device(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return normalized


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
