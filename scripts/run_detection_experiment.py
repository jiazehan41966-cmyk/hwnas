#!/usr/bin/env python3
"""Train and evaluate the multi-scale detection branch."""

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
    parser = argparse.ArgumentParser(description="Run the fixed-head multi-scale detection experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--pre-nms-topk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_detection_experiment(args)
    print(f"run_dir {run_dir}", flush=True)


def run_detection_experiment(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(str(config_path))
    if str(config.get("task", {}).get("name", "")).strip().lower() != "detection":
        raise ValueError("run_detection_experiment.py requires task.name=detection")

    project_cfg = config.get("project", {})
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    detection_cfg = config.get("detection", {})
    eval_cfg = training_cfg.get("evaluation", {})

    run_dir = _resolve_run_dir(args.output_dir, project_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "config_snapshot.yaml"
    config_snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    device = _resolve_device(args.device)
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 150))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else dataset_cfg.get("num_workers", 0))
    max_samples = _resolve_max_samples(args.max_train_samples, args.max_eval_samples)

    loaders = create_yolo_detection_dataloaders(
        _resolve_repo_path(args.data_dir if args.data_dir is not None else dataset_cfg["data_dir"]),
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        seed=int(project_cfg.get("seed", 42)),
        max_samples_per_split=max_samples,
        train_augment_config=training_cfg.get("detection_augmentation", {}),
        train_class_balanced_sampler=bool(
            training_cfg.get("detection_sampling", {}).get("class_balanced", False)
        ),
    )
    constraints = build_constraints(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_classes=int(dataset_cfg["num_classes"]),
        constraints=constraints,
    )
    architecture = search_space.baseline_architecture()
    model = build_detection_model(
        architecture,
        num_classes=int(dataset_cfg["num_classes"]),
        feature_indices=tuple(detection_cfg["feature_stage_indices"]),
        neck_channels=int(detection_cfg["fixed_neck"]["out_channels"]),
        anchors_per_location=int(detection_cfg["fixed_head"]["anchors_per_location"]),
    )
    loss_config = DetectionLossConfig.from_config(config)
    criterion = DenseAnchorDetectionLoss(loss_config)
    optimizer = create_optimizer(
        model,
        optimizer_name=str(training_cfg.get("optimizer", "adamw")),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )

    run_config = {
        "artifact_type": "detection_experiment_run_config",
        "started_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "config_snapshot": str(config_snapshot),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "dataset_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        "data_dir_override": args.data_dir,
        "class_names": list(getattr(loaders["train"].dataset, "class_names", [])),
        "loss_config": {
            **loss_config.__dict__,
            "anchor_sizes": loss_config.anchor_sizes,
        },
        "hardware": _read_measured_lut_summary(config.get("hardware", {})),
        "claim_boundary": {
            "search_space": "classification-derived backbone/search-space",
            "neck_head": "fixed_not_searched_stage_1",
            "hardware_lut": "operator-level measured fixed Neck/Head supplement",
            "board_runtime_power": "not_measured_by_this_run",
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    epoch_log_path = run_dir / "epoch_metrics.jsonl"
    history = train_detection_model(
        model,
        loaders["train"],
        loaders["valid"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        early_stopping_patience=args.early_stopping_patience,
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        nms_iou_threshold=float(eval_cfg.get("nms_iou_threshold", 0.5)),
        pre_nms_topk=int(args.pre_nms_topk if args.pre_nms_topk is not None else eval_cfg.get("pre_nms_topk", 1000)),
        max_detections=int(eval_cfg.get("max_detections", 100)),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        checkpoint_path=checkpoint_path,
        epoch_log_path=epoch_log_path,
        verbose=True,
    )
    test_summary = evaluate_detection_model(
        model,
        loaders["test"],
        criterion=criterion,
        device=device,
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        nms_iou_threshold=float(eval_cfg.get("nms_iou_threshold", 0.5)),
        pre_nms_topk=int(args.pre_nms_topk if args.pre_nms_topk is not None else eval_cfg.get("pre_nms_topk", 1000)),
        max_detections=int(eval_cfg.get("max_detections", 100)),
        max_batches=args.max_eval_batches,
    )
    threshold_sweep = _run_threshold_sweep(
        model,
        loaders,
        criterion=criterion,
        device=device,
        thresholds=eval_cfg.get("operating_score_thresholds", []),
        nms_iou_threshold=float(eval_cfg.get("nms_iou_threshold", 0.5)),
        pre_nms_topk=int(args.pre_nms_topk if args.pre_nms_topk is not None else eval_cfg.get("pre_nms_topk", 1000)),
        max_detections=int(eval_cfg.get("max_detections", 100)),
        max_eval_batches=args.max_eval_batches,
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "test_eval.json").write_text(
        json.dumps(test_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if threshold_sweep:
        (run_dir / "threshold_sweep.json").write_text(
            json.dumps(threshold_sweep, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    status = {
        "artifact_type": "detection_experiment_status",
        "status": "COMPLETE",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_epoch": history.get("best_epoch"),
        "best_eval": history.get("best_eval"),
        "test_eval": test_summary,
        "threshold_sweep": threshold_sweep,
        "checkpoint": str(checkpoint_path),
    }
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
        path = Path(project_cfg.get("output_dir", "results/detection_experiment")) / f"run_{stamp}"
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_max_samples(train_samples: int | None, eval_samples: int | None) -> int | None:
    if train_samples is None and eval_samples is None:
        return None
    values = [value for value in (train_samples, eval_samples) if value is not None]
    return max(int(value) for value in values)


def _run_threshold_sweep(
    model: torch.nn.Module,
    loaders: dict[str, Any],
    *,
    criterion: DenseAnchorDetectionLoss,
    device: str,
    thresholds: Any,
    nms_iou_threshold: float,
    pre_nms_topk: int,
    max_detections: int,
    max_eval_batches: int | None,
) -> dict[str, Any]:
    values = [float(value) for value in thresholds or []]
    if not values:
        return {}
    valid_rows = [
        {
            **evaluate_detection_model(
                model,
                loaders["valid"],
                criterion=criterion,
                device=device,
                score_threshold=threshold,
                nms_iou_threshold=nms_iou_threshold,
                pre_nms_topk=pre_nms_topk,
                max_detections=max_detections,
                max_batches=max_eval_batches,
            ),
            "score_threshold": threshold,
        }
        for threshold in values
    ]
    best_valid = max(valid_rows, key=lambda row: float(row.get("f1", 0.0)))
    selected_threshold = float(best_valid["score_threshold"])
    test_at_selected = evaluate_detection_model(
        model,
        loaders["test"],
        criterion=criterion,
        device=device,
        score_threshold=selected_threshold,
        nms_iou_threshold=nms_iou_threshold,
        pre_nms_topk=pre_nms_topk,
        max_detections=max_detections,
        max_batches=max_eval_batches,
    )
    test_at_selected["score_threshold"] = selected_threshold
    return {
        "selection": "best_valid_f1",
        "valid": valid_rows,
        "best_valid_f1": best_valid,
        "test_at_best_valid_f1_threshold": test_at_selected,
    }


def _read_measured_lut_summary(hardware_cfg: dict[str, Any]) -> dict[str, Any]:
    table_path = hardware_cfg.get("detection_cost_table_path")
    if not table_path:
        return {"detection_cost_table_path": None, "status": "not_configured"}
    resolved = _resolve_repo_path(table_path)
    if not resolved.is_file():
        return {"detection_cost_table_path": str(resolved), "status": "missing"}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return {
        "detection_cost_table_path": str(resolved),
        "status": payload.get("status"),
        "rows_total": summary.get("rows_total"),
        "rows_measured": summary.get("rows_measured"),
        "payload_sha256": payload.get("payload_sha256"),
    }


if __name__ == "__main__":
    main()
