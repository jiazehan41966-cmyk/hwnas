#!/usr/bin/env python3
"""Overfit a tiny YOLO split to diagnose detection loss/decoder plumbing."""

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

from hwnas_fpga.data import create_yolo_detection_dataloader
from hwnas_fpga.models import build_detection_model
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config
from hwnas_fpga.training import (
    DenseAnchorDetectionLoss,
    DetectionLossConfig,
    evaluate_detection_model,
    train_detection_model,
)
from hwnas_fpga.training.trainer import create_optimizer


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "search"
    / "zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny detection overfit probe.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pre-nms-topk", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_overfit_probe(args)
    print(f"run_dir {run_dir}", flush=True)


def run_overfit_probe(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(str(config_path))
    project_cfg = config.get("project", {})
    dataset_cfg = config.get("dataset", {})
    detection_cfg = config.get("detection", {})
    training_cfg = config.get("training", {})
    eval_cfg = training_cfg.get("evaluation", {})

    run_dir = _resolve_run_dir(args.output_dir, project_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    device = _resolve_device(args.device)
    data_dir = _resolve_repo_path(args.data_dir if args.data_dir is not None else dataset_cfg["data_dir"])
    train_loader = create_yolo_detection_dataloader(
        data_dir,
        split="train",
        batch_size=int(args.batch_size),
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_workers=int(args.num_workers),
        pin_memory=device.startswith("cuda"),
        shuffle=True,
        seed=int(project_cfg.get("seed", 42)),
        max_samples=int(args.max_samples),
    )
    eval_loader = create_yolo_detection_dataloader(
        data_dir,
        split="train",
        batch_size=int(args.batch_size),
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_workers=int(args.num_workers),
        pin_memory=device.startswith("cuda"),
        shuffle=False,
        seed=int(project_cfg.get("seed", 42)),
        max_samples=int(args.max_samples),
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
    criterion = DenseAnchorDetectionLoss(DetectionLossConfig.from_config(config))
    optimizer = create_optimizer(
        model,
        optimizer_name=str(training_cfg.get("optimizer", "adamw")),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )

    run_config: dict[str, Any] = {
        "artifact_type": "detection_overfit_probe_run_config",
        "started_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "device": device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "max_samples": int(args.max_samples),
        "probe_scope": "train_split_subset_evaluated_on_same_subset",
        "loss_config": {
            **criterion.config.__dict__,
            "anchor_sizes": criterion.config.anchor_sizes,
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    history = train_detection_model(
        model,
        train_loader,
        eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(args.epochs),
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        nms_iou_threshold=float(eval_cfg.get("nms_iou_threshold", 0.5)),
        pre_nms_topk=int(args.pre_nms_topk),
        max_detections=int(eval_cfg.get("max_detections", 50)),
        checkpoint_path=checkpoint_path,
        epoch_log_path=run_dir / "epoch_metrics.jsonl",
        verbose=True,
    )
    final_summary = evaluate_detection_model(
        model,
        eval_loader,
        criterion=criterion,
        device=device,
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        nms_iou_threshold=float(eval_cfg.get("nms_iou_threshold", 0.5)),
        pre_nms_topk=int(args.pre_nms_topk),
        max_detections=int(eval_cfg.get("max_detections", 50)),
    )
    status = {
        "artifact_type": "detection_overfit_probe_status",
        "status": "COMPLETE",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_epoch": history.get("best_epoch"),
        "best_eval": history.get("best_eval"),
        "final_train_subset_eval": final_summary,
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def _resolve_device(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return normalized


def _resolve_run_dir(output_dir: str | None, project_cfg: dict[str, Any]) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(project_cfg.get("output_dir", "results/detection_experiment")) / f"overfit_{stamp}"
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


if __name__ == "__main__":
    main()
