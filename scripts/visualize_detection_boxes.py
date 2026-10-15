#!/usr/bin/env python3
"""Visualize GT and predicted detection boxes for coordinate sanity checks."""

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

from hwnas_fpga.data import YoloDetectionDataset
from hwnas_fpga.models import build_detection_model
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config
from hwnas_fpga.training import DetectionLossConfig, box_iou_xyxy, decode_detection_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw GT/predicted boxes and summarize localization errors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=500)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--background-iou-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_visualization(args)
    summary_path = Path(args.output_dir).expanduser().resolve() / "box_diagnostics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"images {len(payload['samples'])}")
    print(f"mean_best_iou {payload['summary']['mean_best_iou']:.4f}")
    print(f"mean_center_error_px {payload['summary']['mean_center_error_px']:.4f}")
    print(f"total_duplicate_predictions {payload['summary']['total_duplicate_predictions']}")
    print(f"total_background_false_positives {payload['summary']['total_background_false_positives']}")
    print(f"artifact {summary_path}")


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_repo_path(args.config)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    detection_cfg = config.get("detection", {})
    device = _resolve_device(args.device)
    image_size = int(dataset_cfg["image_size"])
    data_dir = _resolve_repo_path(args.data_dir if args.data_dir is not None else dataset_cfg["data_dir"])

    model = _build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    loss_config = DetectionLossConfig.from_config(config)

    split_names = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    samples = []
    class_names: list[str] = []
    with torch.no_grad():
        for split in split_names:
            dataset = YoloDetectionDataset(
                data_dir,
                split=split,
                image_size=image_size,
                input_channels=int(dataset_cfg["input_channels"]),
            )
            class_names = list(dataset.class_names)
            for sample_index in _select_indices_by_class(dataset, int(args.samples_per_class)):
                image, target = dataset[sample_index]
                output = model(image.unsqueeze(0).to(device))
                prediction = decode_detection_predictions(
                    output,
                    config=loss_config,
                    score_threshold=float(args.score_threshold),
                    nms_iou_threshold=float(args.nms_iou_threshold),
                    pre_nms_topk=int(args.pre_nms_topk),
                    max_detections=int(args.max_detections),
                )[0]
                record = _sample_record(
                    split=split,
                    sample_index=sample_index,
                    image_tensor=image,
                    target=target,
                    prediction=prediction,
                    class_names=class_names,
                    output_dir=output_dir,
                    duplicate_iou_threshold=float(args.duplicate_iou_threshold),
                    background_iou_threshold=float(args.background_iou_threshold),
                )
                samples.append(record)

    return {
        "artifact_type": "detection_box_visual_diagnostics",
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "class_names": class_names,
        "image_size": image_size,
        "postprocess": {
            "score_threshold": float(args.score_threshold),
            "nms_iou_threshold": float(args.nms_iou_threshold),
            "pre_nms_topk": int(args.pre_nms_topk),
            "max_detections": int(args.max_detections),
        },
        "sampling": {
            "splits": split_names,
            "samples_per_class": int(args.samples_per_class),
        },
        "samples": samples,
        "summary": _summarize_samples(samples),
        "claim_boundary": "Visual diagnostics only; do not infer annotation errors without image-level evidence.",
    }


def _build_model(config: dict[str, Any]) -> torch.nn.Module:
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
    return build_detection_model(
        search_space.baseline_architecture(),
        num_classes=int(dataset_cfg["num_classes"]),
        feature_indices=tuple(detection_cfg["feature_stage_indices"]),
        neck_channels=int(detection_cfg["fixed_neck"]["out_channels"]),
        anchors_per_location=int(detection_cfg["fixed_head"]["anchors_per_location"]),
    )


def _select_indices_by_class(dataset: YoloDetectionDataset, samples_per_class: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    class_count = len(dataset.class_names)
    for class_index in range(class_count):
        picked = 0
        for sample_index, labels in enumerate(dataset.sample_labels):
            if class_index not in labels or sample_index in seen:
                continue
            selected.append(sample_index)
            seen.add(sample_index)
            picked += 1
            if picked >= samples_per_class:
                break
    return selected


def _sample_record(
    *,
    split: str,
    sample_index: int,
    image_tensor: torch.Tensor,
    target: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    output_dir: Path,
    duplicate_iou_threshold: float,
    background_iou_threshold: float,
) -> dict[str, Any]:
    gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
    gt_labels = torch.as_tensor(target["labels"], dtype=torch.long).reshape(-1)
    pred_boxes = torch.as_tensor(prediction["boxes"], dtype=torch.float32).reshape(-1, 4)
    pred_labels = torch.as_tensor(prediction["labels"], dtype=torch.long).reshape(-1)
    pred_scores = torch.as_tensor(prediction["scores"], dtype=torch.float32).reshape(-1)

    ious = box_iou_xyxy(pred_boxes, gt_boxes) if pred_boxes.numel() and gt_boxes.numel() else torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))
    matches = []
    matched_pred_indices: set[int] = set()
    duplicate_predictions = 0
    for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        same_class = pred_labels == gt_label
        candidate_indices = torch.where(same_class)[0]
        if candidate_indices.numel() == 0:
            matches.append(_empty_match(gt_index, gt_box, int(gt_label.item()), class_names))
            continue
        candidate_ious = ious[candidate_indices, gt_index]
        best_pos = int(torch.argmax(candidate_ious).item())
        pred_index = int(candidate_indices[best_pos].item())
        matched_pred_indices.add(pred_index)
        duplicate_predictions += int((candidate_ious >= duplicate_iou_threshold).sum().item()) - 1
        matches.append(
            _match_record(
                gt_index=gt_index,
                pred_index=pred_index,
                gt_box=gt_box,
                pred_box=pred_boxes[pred_index],
                label=int(gt_label.item()),
                score=float(pred_scores[pred_index].item()),
                iou=float(ious[pred_index, gt_index].item()),
                class_names=class_names,
            )
        )

    background_false_positives = 0
    for pred_index in range(pred_boxes.shape[0]):
        if gt_boxes.numel() == 0:
            background_false_positives += 1
            continue
        if float(ious[pred_index].max().item()) < background_iou_threshold:
            background_false_positives += 1

    image_path = Path(str(target["path"]))
    output_name = f"{split}_{sample_index:04d}_{image_path.stem}.png"
    output_path = output_dir / output_name
    _draw_overlay(
        image_tensor=image_tensor,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        class_names=class_names,
        output_path=output_path,
    )

    return {
        "split": split,
        "sample_index": int(sample_index),
        "source_path": str(image_path),
        "visualization": str(output_path),
        "gt_count": int(gt_boxes.shape[0]),
        "prediction_count": int(pred_boxes.shape[0]),
        "matched_prediction_count": int(len(matched_pred_indices)),
        "duplicate_predictions": int(max(0, duplicate_predictions)),
        "background_false_positives": int(background_false_positives),
        "gt": [
            {
                "box_xyxy": _box_list(box),
                "label": int(label.item()),
                "class_name": _class_name(int(label.item()), class_names),
            }
            for box, label in zip(gt_boxes, gt_labels)
        ],
        "predictions": [
            {
                "box_xyxy": _box_list(box),
                "label": int(label.item()),
                "class_name": _class_name(int(label.item()), class_names),
                "score": float(score.item()),
                "best_gt_iou": float(ious[index].max().item()) if gt_boxes.numel() else 0.0,
            }
            for index, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores))
        ],
        "matches": matches,
    }


def _match_record(
    *,
    gt_index: int,
    pred_index: int,
    gt_box: torch.Tensor,
    pred_box: torch.Tensor,
    label: int,
    score: float,
    iou: float,
    class_names: list[str],
) -> dict[str, Any]:
    gt_cx, gt_cy, gt_w, gt_h = _cxcywh(gt_box)
    pred_cx, pred_cy, pred_w, pred_h = _cxcywh(pred_box)
    return {
        "gt_index": int(gt_index),
        "pred_index": int(pred_index),
        "label": int(label),
        "class_name": _class_name(label, class_names),
        "score": float(score),
        "iou": float(iou),
        "center_dx_px": float(pred_cx - gt_cx),
        "center_dy_px": float(pred_cy - gt_cy),
        "center_error_px": float(((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5),
        "width_error_px": float(pred_w - gt_w),
        "height_error_px": float(pred_h - gt_h),
        "width_ratio": float(pred_w / max(gt_w, 1e-6)),
        "height_ratio": float(pred_h / max(gt_h, 1e-6)),
    }


def _empty_match(
    gt_index: int,
    gt_box: torch.Tensor,
    label: int,
    class_names: list[str],
) -> dict[str, Any]:
    gt_cx, gt_cy, gt_w, gt_h = _cxcywh(gt_box)
    return {
        "gt_index": int(gt_index),
        "pred_index": None,
        "label": int(label),
        "class_name": _class_name(label, class_names),
        "score": 0.0,
        "iou": 0.0,
        "center_dx_px": None,
        "center_dy_px": None,
        "center_error_px": None,
        "width_error_px": None,
        "height_error_px": None,
        "width_ratio": None,
        "height_ratio": None,
        "gt_center": [float(gt_cx), float(gt_cy)],
        "gt_size": [float(gt_w), float(gt_h)],
    }


def _draw_overlay(
    *,
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        array = (image[0].numpy() * 255.0).astype("uint8")
        canvas = Image.fromarray(array, mode="L").convert("RGB")
    else:
        array = (image.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        canvas = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box, label in zip(gt_boxes, gt_labels):
        _draw_box(
            draw,
            box,
            color=(0, 220, 80),
            label=f"GT {_class_name(int(label.item()), class_names)}",
            font=font,
        )
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        _draw_box(
            draw,
            box,
            color=(240, 55, 55),
            label=f"P {_class_name(int(label.item()), class_names)} {float(score.item()):.2f}",
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _draw_box(draw: Any, box: torch.Tensor, *, color: tuple[int, int, int], label: str, font: Any) -> None:
    x1, y1, x2, y2 = _box_list(box)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text_xy = (x1 + 2, max(0.0, y1 - 12.0))
    draw.text(text_xy, label, fill=color, font=font)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [match for sample in samples for match in sample["matches"] if match["pred_index"] is not None]
    return {
        "num_samples": len(samples),
        "num_gt": int(sum(sample["gt_count"] for sample in samples)),
        "num_predictions": int(sum(sample["prediction_count"] for sample in samples)),
        "mean_best_iou": _mean([match["iou"] for match in matches]),
        "mean_center_error_px": _mean([match["center_error_px"] for match in matches]),
        "mean_abs_width_error_px": _mean([abs(match["width_error_px"]) for match in matches]),
        "mean_abs_height_error_px": _mean([abs(match["height_error_px"]) for match in matches]),
        "total_duplicate_predictions": int(sum(sample["duplicate_predictions"] for sample in samples)),
        "total_background_false_positives": int(sum(sample["background_false_positives"] for sample in samples)),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _box_list(box: torch.Tensor) -> list[float]:
    return [round(float(value), 3) for value in box.detach().cpu().tolist()]


def _cxcywh(box: torch.Tensor) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _class_name(label: int, class_names: list[str]) -> str:
    return class_names[label] if 0 <= label < len(class_names) else f"class_{label}"


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


if __name__ == "__main__":
    main()
