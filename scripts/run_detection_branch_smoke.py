#!/usr/bin/env python3
"""Run a lightweight detection-branch experiment smoke without full training."""

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
from hwnas_fpga.training import evaluate_detection_predictions


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "search"
    / "zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "detection_multiscale_640" / "detection_branch_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the multi-scale detection branch without a costly training run."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        config_path=Path(args.config),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"status {payload['status']}")
    print(f"output_shapes {payload['model']['prediction_shapes']}")
    print(f"echo_metric_f1 {payload['metrics_plumbing']['echo_metrics'].get('f1')}")
    print(f"artifact {output_path}")


def run_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG,
    max_samples: int = 2,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    config = load_config(str(config_path))
    task_name = str(config.get("task", {}).get("name", "")).strip().lower()
    if task_name != "detection":
        raise ValueError(f"Expected task.name=detection, got: {task_name!r}")

    dataset_cfg = config.get("dataset", {})
    detection_cfg = config.get("detection", {})
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
    resolved_device = torch.device(device)
    model.to(resolved_device)

    data_dir = _resolve_repo_path(dataset_cfg["data_dir"])
    loaders = create_yolo_detection_dataloaders(
        data_dir,
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=resolved_device.type == "cuda",
        seed=int(config.get("project", {}).get("seed", 42)),
        max_samples_per_split=int(max_samples),
    )

    train_images, train_targets = next(iter(loaders["train"]))
    train_images = train_images.to(resolved_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    train_output = model(train_images)
    smoke_loss = sum(pred.float().square().mean() for pred in train_output["predictions"])
    smoke_loss.backward()
    optimizer.step()

    model.eval()
    valid_images, valid_targets = next(iter(loaders["valid"]))
    with torch.no_grad():
        valid_output = model(valid_images.to(resolved_device))

    echo_predictions, echo_targets = _echo_predictions(valid_targets)
    echo_metrics = evaluate_detection_predictions(
        echo_predictions,
        echo_targets,
        num_classes=int(dataset_cfg["num_classes"]),
        iou_thresholds=tuple(float(value) for value in detection_cfg["metrics"]["iou_thresholds"]),
    )
    measured_lut = _read_measured_lut_summary(config.get("hardware", {}))

    return {
        "artifact_type": "detection_branch_smoke",
        "schema_version": 1,
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path.expanduser().resolve()),
        "dataset": {
            "name": dataset_cfg.get("name"),
            "data_dir": str(data_dir),
            "image_size": int(dataset_cfg["image_size"]),
            "input_channels": int(dataset_cfg["input_channels"]),
            "num_classes": int(dataset_cfg["num_classes"]),
            "splits": {
                split: {
                    "num_samples_checked": len(loader.dataset),
                    "class_names": list(getattr(loader.dataset, "class_names", [])),
                }
                for split, loader in loaders.items()
            },
            "first_train_target_count": int(sum(t["labels"].numel() for t in train_targets)),
            "first_valid_target_count": int(sum(t["labels"].numel() for t in valid_targets)),
        },
        "model": {
            "backbone_search_space": "classification-derived MobileNetV2-style backbone/search-space",
            "neck_head": "fixed_not_searched_stage_1",
            "feature_stage_indices": list(detection_cfg["feature_stage_indices"]),
            "backbone_feature_shapes": _shapes(valid_output["backbone_features"]),
            "neck_feature_shapes": _shapes(valid_output["neck_features"]),
            "prediction_shapes": _shapes(valid_output["predictions"]),
        },
        "trainability_smoke": {
            "one_optimizer_step": True,
            "loss_name": "raw_prediction_l2_smoke_loss_not_detection_training_loss",
            "loss": float(smoke_loss.detach().cpu().item()),
        },
        "metrics_plumbing": {
            "mode": "echo_ground_truth_predictions",
            "purpose": "validates mAP/Precision/Recall/F1 interface only",
            "echo_metrics": echo_metrics,
        },
        "hardware": measured_lut,
        "claim_boundary": {
            "detection_accuracy": "not_claimed_from_this_smoke",
            "full_training": "not_run",
            "full_network_hls_route": "not_run",
            "board_runtime_power": "not_run",
        },
    }


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _shapes(tensors: Any) -> list[list[int]]:
    return [list(tensor.shape) for tensor in tensors]


def _echo_predictions(
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    clean_targets: list[dict[str, Any]] = []
    for target in targets:
        boxes = target["boxes"].detach().cpu()
        labels = target["labels"].detach().cpu()
        predictions.append(
            {
                "boxes": boxes,
                "labels": labels,
                "scores": torch.ones(labels.numel(), dtype=torch.float32),
            }
        )
        clean_targets.append({"boxes": boxes, "labels": labels})
    return predictions, clean_targets


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
        "artifact_type": payload.get("artifact_type"),
        "rows_total": summary.get("rows_total"),
        "rows_measured": summary.get("rows_measured"),
        "status_counts": summary.get("status_counts"),
        "payload_sha256": payload.get("payload_sha256"),
    }


if __name__ == "__main__":
    main()
