#!/usr/bin/env python3
"""Verify the detection evaluation chain before any checkpoint is diagnosed.

Checks, in order:

A. dataset geometry and label integrity for every split
B. original pixels -> network input -> decode -> original pixels round trip on real labels
C. synthetic known-box closed loop through the real dataset, assigner and decoder
D. grid alignment probes at every level and corner, to catch axis swaps
E. metric definition: ``map`` is the AP50/AP75 mean, plus standard COCO AP
F. anchor reachability and representable box range per ground-truth size bin

No model weights are loaded and no metric is computed from the test split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.detection import (
    YoloDetectionDataset,
    network_boxes_to_original,
    yolo_normalized_to_original_xyxy,
    yolo_normalized_to_xyxy,
)
from hwnas_fpga.runtime import load_config
from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco
from hwnas_fpga.training.detection_metrics import (
    box_iou_xyxy,
    evaluate_detection_predictions,
)
from hwnas_fpga.training.detection_trainer import (
    DetectionLossConfig,
    _assign_targets,
    decode_detection_predictions,
)

MODEL_SPLITS = ("train", "valid")
SIZE_BINS = ((0.0, 16.0, "sqrt_area_lt16"), (16.0, 32.0, "sqrt_area_16_32"), (32.0, float("inf"), "sqrt_area_ge32"))


def size_bin(sqrt_area: float) -> str:
    for low, high, name in SIZE_BINS:
        if low <= sqrt_area < high:
            return name
    return SIZE_BINS[-1][2]


def feature_grid(config: DetectionLossConfig) -> list[tuple[int, int]]:
    return [
        (int(config.image_size // stride), int(config.image_size // stride))
        for stride in config.strides
    ]


def encode_boxes_to_raw_levels(
    records: list[dict[str, torch.Tensor]],
    config: DetectionLossConfig,
    *,
    logit_scale: float = 12.0,
) -> list[torch.Tensor]:
    """Turn known boxes into raw head tensors using the training assigner.

    This closes the loop: the assigner writes the regression targets that the
    decoder is supposed to invert, so any mismatch between the two is a real
    coordinate bug rather than a training artifact.
    """

    grids = feature_grid(config)
    outputs_per_anchor = int(config.num_classes) + 5
    placeholder = [
        torch.zeros(len(records), len(config.anchor_sizes[level]) * outputs_per_anchor, height, width)
        for level, (height, width) in enumerate(grids)
    ]
    assigned = _assign_targets(placeholder, records, config, device=torch.device("cpu"))

    raw_levels: list[torch.Tensor] = []
    for level, (height, width) in enumerate(grids):
        anchors = len(config.anchor_sizes[level])
        dense = torch.zeros(len(records), anchors, height, width, outputs_per_anchor)
        target = assigned[level]
        positive = target["objectness"] > 0.5
        dense[..., 0:4] = target["box"]
        dense[..., 4] = torch.where(positive, torch.full_like(target["objectness"], float(logit_scale)), torch.full_like(target["objectness"], -float(logit_scale)))
        one_hot = torch.nn.functional.one_hot(target["label"], num_classes=int(config.num_classes)).float()
        dense[..., 5:] = one_hot * float(logit_scale) - (1.0 - one_hot) * float(logit_scale)
        raw_levels.append(dense.permute(0, 1, 4, 2, 3).reshape(len(records), anchors * outputs_per_anchor, height, width))
    return raw_levels


def closed_loop_boxes(
    records: list[dict[str, torch.Tensor]],
    config: DetectionLossConfig,
) -> list[dict[str, torch.Tensor]]:
    raw_levels = encode_boxes_to_raw_levels(records, config)
    return decode_detection_predictions(
        raw_levels,
        config=config,
        score_threshold=0.05,
        nms_iou_threshold=0.5,
        pre_nms_topk=5000,
        max_detections=200,
    )


def match_recovered_boxes(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    decoded: dict[str, torch.Tensor],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pred_boxes = decoded["boxes"].reshape(-1, 4)
    pred_labels = decoded["labels"].reshape(-1)
    if pred_boxes.shape[0] == 0:
        return [
            {"gt_index": index, "best_iou": 0.0, "label_recovered": False, "center_error_px": float("nan")}
            for index in range(gt_boxes.shape[0])
        ]
    ious = box_iou_xyxy(pred_boxes, gt_boxes)
    for gt_index in range(gt_boxes.shape[0]):
        same = torch.where(pred_labels == gt_labels[gt_index])[0]
        pool = same if same.numel() else torch.arange(pred_boxes.shape[0])
        best_pos = int(torch.argmax(ious[pool, gt_index]).item())
        best = int(pool[best_pos].item())
        gt = gt_boxes[gt_index]
        pred = pred_boxes[best]
        gt_center = ((gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0)
        pred_center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
        rows.append(
            {
                "gt_index": gt_index,
                "best_iou": float(ious[best, gt_index].item()),
                "label_recovered": bool(same.numel() > 0),
                "center_error_px": float(
                    ((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2) ** 0.5
                ),
            }
        )
    return rows


def audit_dataset_geometry(root: Path, image_size: int) -> dict[str, Any]:
    from PIL import Image

    splits: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir():
            splits[split] = {"status": "MISSING"}
            continue
        sizes: Counter[str] = Counter()
        missing_labels: list[str] = []
        empty_labels: list[str] = []
        out_of_range: list[dict[str, Any]] = []
        degenerate: list[dict[str, Any]] = []
        roundtrip_max_error = 0.0
        boxes_total = 0
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            with Image.open(image_path) as source:
                width, height = source.size
            sizes[f"{width}x{height}"] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels.append(image_path.name)
                continue
            rows = [
                [float(value) for value in line.split()[:5]]
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if len(line.split()) >= 5
            ]
            if not rows:
                empty_labels.append(image_path.name)
                continue
            boxes_total += len(rows)
            for row in rows:
                _label, cx, cy, box_w, box_h = row
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0):
                    out_of_range.append({"image": image_path.name, "row": row})
                if cx - box_w / 2 < -1e-6 or cx + box_w / 2 > 1 + 1e-6:
                    out_of_range.append({"image": image_path.name, "row": row, "axis": "x_clipped"})
                if cy - box_h / 2 < -1e-6 or cy + box_h / 2 > 1 + 1e-6:
                    out_of_range.append({"image": image_path.name, "row": row, "axis": "y_clipped"})
            network_boxes, _labels = yolo_normalized_to_xyxy(rows, image_size=image_size)
            areas = (network_boxes[:, 2] - network_boxes[:, 0]) * (network_boxes[:, 3] - network_boxes[:, 1])
            for index in torch.where(areas <= 0)[0].tolist():
                degenerate.append({"image": image_path.name, "box_xyxy": network_boxes[index].tolist()})
            restored = network_boxes_to_original(
                network_boxes, image_size=image_size, orig_width=width, orig_height=height
            )
            expected = torch.tensor(
                [
                    [
                        (row[1] - row[3] / 2) * width,
                        (row[2] - row[4] / 2) * height,
                        (row[1] + row[3] / 2) * width,
                        (row[2] + row[4] / 2) * height,
                    ]
                    for row in rows
                ],
                dtype=torch.float32,
            )
            expected = expected.clamp(min=0.0)
            expected[:, 0::2] = expected[:, 0::2].clamp(max=float(width))
            expected[:, 1::2] = expected[:, 1::2].clamp(max=float(height))
            roundtrip_max_error = max(
                roundtrip_max_error, float((restored - expected).abs().max().item())
            )
        splits[split] = {
            "images": int(sum(sizes.values())),
            "unique_image_sizes": dict(sizes),
            "all_square": all(part.split("x")[0] == part.split("x")[1] for part in sizes),
            "single_source_resolution": len(sizes) == 1,
            "boxes": boxes_total,
            "images_without_label_file": missing_labels,
            "images_with_empty_label_file": empty_labels,
            "normalized_range_violations": out_of_range[:20],
            "normalized_range_violation_count": len(out_of_range),
            "degenerate_network_boxes": degenerate[:20],
            "degenerate_network_box_count": len(degenerate),
            "max_original_roundtrip_error_px": roundtrip_max_error,
        }
    return {
        "letterbox_or_pad_applied": False,
        "crop_applied": False,
        "resize_mode": "per_axis_bilinear_resize_to_square_network_input",
        "iou_space_equivalent_to_original": all(
            splits[split].get("all_square", False) for split in splits if splits[split].get("images")
        ),
        "splits": splits,
        "test_split_scope": "geometry_and_label_integrity_only_no_predictions_no_metrics",
    }


def audit_real_label_roundtrip(
    data_dir: Path,
    image_size: int,
    input_channels: int,
    samples_per_split: int,
) -> dict[str, Any]:
    from PIL import Image

    per_split: dict[str, Any] = {}
    for split in MODEL_SPLITS:
        dataset = YoloDetectionDataset(
            data_dir, split=split, image_size=image_size, input_channels=input_channels
        )
        stride = max(1, len(dataset) // max(1, samples_per_split))
        indices = list(range(0, len(dataset), stride))[:samples_per_split]
        rows: list[dict[str, Any]] = []
        max_network_error = 0.0
        max_original_error = 0.0
        for index in indices:
            _image, target = dataset[index]
            image_path = Path(target["path"])
            with Image.open(image_path) as source:
                width, height = source.size
            recorded = target["orig_size"].tolist()
            label_rows = [
                [float(value) for value in line.split()[:5]]
                for line in (dataset.labels_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines()
                if len(line.split()) >= 5
            ]
            expected_network, expected_labels = yolo_normalized_to_xyxy(label_rows, image_size=image_size)
            network_error = float((target["boxes"] - expected_network).abs().max().item()) if label_rows else 0.0
            restored = network_boxes_to_original(
                target["boxes"], image_size=image_size, orig_width=width, orig_height=height
            )
            expected_original, _ = yolo_normalized_to_original_xyxy(
                label_rows, orig_width=width, orig_height=height
            )
            original_error = float((restored - expected_original).abs().max().item()) if label_rows else 0.0
            max_network_error = max(max_network_error, network_error)
            max_original_error = max(max_original_error, original_error)
            rows.append(
                {
                    "index": index,
                    "image": image_path.name,
                    "source_size_wh": [width, height],
                    "recorded_orig_size_hw": recorded,
                    "recorded_orig_size_matches_source": recorded == [height, width],
                    "labels": expected_labels.tolist(),
                    "network_box_max_abs_error_px": network_error,
                    "original_box_max_abs_error_px": original_error,
                }
            )
        per_split[split] = {
            "sampled_images": len(rows),
            "max_network_box_error_px": max_network_error,
            "max_original_box_error_px": max_original_error,
            "samples": rows,
        }
    return per_split


def audit_synthetic_closed_loop(image_size: int, config: DetectionLossConfig) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    source_size = 1280
    known_boxes_normalized = [
        (0, 0.1250, 0.1250, 0.0500, 0.0500),
        (1, 0.5000, 0.5000, 0.1000, 0.0500),
        (0, 0.8750, 0.2500, 0.0250, 0.1000),
        (1, 0.3125, 0.7500, 0.1500, 0.1500),
        (0, 0.5000, 0.9000, 0.0300, 0.0300),
    ]
    with tempfile.TemporaryDirectory() as temp_root:
        images_dir = Path(temp_root) / "train" / "images"
        labels_dir = Path(temp_root) / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        canvas = Image.new("L", (source_size, source_size), color=40)
        draw = ImageDraw.Draw(canvas)
        for _label, cx, cy, box_w, box_h in known_boxes_normalized:
            draw.rectangle(
                (
                    (cx - box_w / 2) * source_size,
                    (cy - box_h / 2) * source_size,
                    (cx + box_w / 2) * source_size,
                    (cy + box_h / 2) * source_size,
                ),
                fill=220,
            )
        canvas.save(images_dir / "synthetic.png")
        (labels_dir / "synthetic.txt").write_text(
            "\n".join(
                f"{label} {cx} {cy} {box_w} {box_h}"
                for label, cx, cy, box_w, box_h in known_boxes_normalized
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = YoloDetectionDataset(
            Path(temp_root), split="train", image_size=image_size, input_channels=1
        )
        _image, target = dataset[0]

    expected_network = torch.tensor(
        [
            [
                (cx - box_w / 2) * image_size,
                (cy - box_h / 2) * image_size,
                (cx + box_w / 2) * image_size,
                (cy + box_h / 2) * image_size,
            ]
            for _label, cx, cy, box_w, box_h in known_boxes_normalized
        ],
        dtype=torch.float32,
    )
    load_error = float((target["boxes"] - expected_network).abs().max().item())
    restored = network_boxes_to_original(
        target["boxes"], image_size=image_size, orig_width=source_size, orig_height=source_size
    )
    expected_original = expected_network * (source_size / image_size)
    restore_error = float((restored - expected_original).abs().max().item())

    record = {"boxes": target["boxes"], "labels": target["labels"]}
    decoded = closed_loop_boxes([record], config)[0]
    matches = match_recovered_boxes(target["boxes"], target["labels"], decoded)

    perfect = [
        {
            "boxes": target["boxes"].tolist(),
            "labels": target["labels"].tolist(),
            "scores": [1.0] * int(target["labels"].numel()),
        }
    ]
    truth = [{"boxes": target["boxes"].tolist(), "labels": target["labels"].tolist()}]
    identity_metric = evaluate_detection_predictions(
        perfect, truth, num_classes=int(config.num_classes), iou_thresholds=(0.5, 0.75)
    )
    shifted = target["boxes"].clone()
    shifted[:, 0::2] += 6.0
    shifted_metric = evaluate_detection_predictions(
        [
            {
                "boxes": shifted.tolist(),
                "labels": target["labels"].tolist(),
                "scores": [1.0] * int(target["labels"].numel()),
            }
        ],
        truth,
        num_classes=int(config.num_classes),
        iou_thresholds=(0.5, 0.75),
    )
    swapped = target["boxes"][:, [1, 0, 3, 2]]
    swapped_metric = evaluate_detection_predictions(
        [
            {
                "boxes": swapped.tolist(),
                "labels": target["labels"].tolist(),
                "scores": [1.0] * int(target["labels"].numel()),
            }
        ],
        truth,
        num_classes=int(config.num_classes),
        iou_thresholds=(0.5, 0.75),
    )

    return {
        "known_boxes_normalized": known_boxes_normalized,
        "source_resolution": source_size,
        "dataset_load_max_abs_error_px": load_error,
        "original_restore_max_abs_error_px": restore_error,
        "assigner_to_decoder_closed_loop": {
            "matches": matches,
            "min_recovered_iou": min(row["best_iou"] for row in matches),
            "max_center_error_px": max(row["center_error_px"] for row in matches),
            "all_labels_recovered": all(row["label_recovered"] for row in matches),
        },
        "metric_sanity": {
            "identity_map": identity_metric["map"],
            "identity_recall": identity_metric["recall"],
            "x_shift_6px_map": shifted_metric["map"],
            "xy_swapped_map": swapped_metric["map"],
            "note": "A perfect copy must score 1.0; the swapped-axis control must collapse.",
        },
    }


def audit_grid_alignment(config: DetectionLossConfig) -> dict[str, Any]:
    grids = feature_grid(config)
    outputs_per_anchor = int(config.num_classes) + 5
    rows: list[dict[str, Any]] = []
    max_error = 0.0
    for level, (height, width) in enumerate(grids):
        anchors = len(config.anchor_sizes[level])
        stride = float(config.strides[level])
        probes = [
            (0, 0, 0),
            (anchors - 1, 0, width - 1),
            (0, height - 1, 0),
            (min(1, anchors - 1), height // 2, width // 3),
        ]
        for anchor_index, grid_y, grid_x in probes:
            dense = torch.zeros(1, anchors, height, width, outputs_per_anchor)
            dense[..., 4] = -12.0
            dense[0, anchor_index, grid_y, grid_x, 4] = 12.0
            dense[0, anchor_index, grid_y, grid_x, 5] = 12.0
            raw = dense.permute(0, 1, 4, 2, 3).reshape(1, anchors * outputs_per_anchor, height, width)
            levels = [
                torch.full(
                    (1, len(config.anchor_sizes[other]) * outputs_per_anchor, grids[other][0], grids[other][1]),
                    -12.0,
                )
                for other in range(len(grids))
            ]
            levels[level] = raw
            decoded = decode_detection_predictions(
                levels, config=config, score_threshold=0.5, pre_nms_topk=100, max_detections=5
            )[0]
            box = decoded["boxes"].reshape(-1, 4)[0]
            center = [float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)]
            expected_center = [(grid_x + 0.5) * stride, (grid_y + 0.5) * stride]
            anchor_w, anchor_h = config.anchor_sizes[level][anchor_index]
            # The decoder clips boxes to the input frame, exactly like the label
            # loader, so the reference corners must be clipped the same way.
            expected_box = torch.tensor(
                [
                    expected_center[0] - anchor_w / 2.0,
                    expected_center[1] - anchor_h / 2.0,
                    expected_center[0] + anchor_w / 2.0,
                    expected_center[1] + anchor_h / 2.0,
                ]
            ).clamp(min=0.0, max=float(config.image_size))
            corner_error = float((box - expected_box).abs().max().item())
            center_shift = max(
                abs(center[0] - expected_center[0]), abs(center[1] - expected_center[1])
            )
            max_error = max(max_error, corner_error)
            rows.append(
                {
                    "level": level,
                    "stride": stride,
                    "anchor_index": anchor_index,
                    "grid_xy": [grid_x, grid_y],
                    "decoded_box_xyxy": [round(float(value), 4) for value in box.tolist()],
                    "expected_clipped_box_xyxy": [round(float(value), 4) for value in expected_box.tolist()],
                    "corner_abs_error_px": corner_error,
                    "decoded_center_xy": center,
                    "unclipped_expected_center_xy": expected_center,
                    "border_clip_center_shift_px": center_shift,
                    "touches_border": bool(
                        expected_box[0] <= 0.0
                        or expected_box[1] <= 0.0
                        or expected_box[2] >= float(config.image_size)
                        or expected_box[3] >= float(config.image_size)
                    ),
                }
            )
    interior = [row for row in rows if not row["touches_border"]]
    return {
        "probes": rows,
        "max_corner_abs_error_px": max_error,
        "max_interior_center_abs_error_px": max(
            (row["border_clip_center_shift_px"] for row in interior), default=0.0
        ),
        "max_border_clip_center_shift_px": max(
            (row["border_clip_center_shift_px"] for row in rows if row["touches_border"]),
            default=0.0,
        ),
        "axis_order_note": "decode uses meshgrid indexing='ij' then maps grid_x to width and grid_y to height",
        "border_clip_note": (
            "Boxes crossing the frame are clipped by both the decoder and the label "
            "loader, so a clipped box center is shifted inward by design."
        ),
    }


def audit_metric_definition(config: DetectionLossConfig) -> dict[str, Any]:
    torch.manual_seed(0)
    num_classes = int(config.num_classes)
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for _image in range(24):
        count = int(torch.randint(1, 4, (1,)).item())
        boxes = []
        labels = []
        for _box in range(count):
            x1 = float(torch.randint(0, 500, (1,)).item())
            y1 = float(torch.randint(0, 500, (1,)).item())
            side = float(torch.randint(20, 120, (1,)).item())
            boxes.append([x1, y1, x1 + side, y1 + side])
            labels.append(int(torch.randint(0, num_classes, (1,)).item()))
        targets.append({"boxes": boxes, "labels": labels})
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for box, label in zip(boxes, labels):
            jitter = float(torch.randn(1).item()) * 8.0
            pred_boxes.append([box[0] + jitter, box[1] + jitter, box[2] + jitter, box[3] + jitter])
            pred_labels.append(label)
            pred_scores.append(float(torch.rand(1).item() * 0.5 + 0.5))
        for _extra in range(int(torch.randint(0, 3, (1,)).item())):
            x1 = float(torch.randint(0, 600, (1,)).item())
            y1 = float(torch.randint(0, 600, (1,)).item())
            pred_boxes.append([x1, y1, x1 + 30.0, y1 + 30.0])
            pred_labels.append(int(torch.randint(0, num_classes, (1,)).item()))
            pred_scores.append(float(torch.rand(1).item() * 0.4))
        predictions.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})

    project = evaluate_detection_predictions(
        predictions, targets, num_classes=num_classes, iou_thresholds=(0.5, 0.75)
    )
    coco = evaluate_detection_predictions_coco(predictions, targets, num_classes=num_classes)
    mean_of_two = (project["map50"] + project["map75"]) / 2.0
    truncated = evaluate_detection_predictions(
        predictions, targets, num_classes=num_classes, iou_thresholds=(0.5, 0.75), score_threshold=0.7
    )
    return {
        "project_metric": {
            "map": project["map"],
            "map50": project["map50"],
            "map75": project["map75"],
            "mean_of_map50_and_map75": mean_of_two,
            "map_equals_mean_of_two_thresholds": abs(project["map"] - mean_of_two) < 1e-12,
            "definition": project["map_definition"],
        },
        "standard_coco_supplement": {
            "coco_ap": coco["coco_ap"],
            "coco_ap50": coco["coco_ap50"],
            "coco_ap75": coco["coco_ap75"],
            "definition": coco["metric_definition"],
        },
        "gap_between_definitions": project["map"] - coco["coco_ap"],
        "score_threshold_truncation": {
            "map_at_threshold_0": project["map"],
            "map_at_threshold_0_70": truncated["map"],
            "note": (
                "AP reported at a positive score threshold is truncated and is not "
                "comparable to AP over the full ranking."
            ),
        },
    }


def audit_anchor_reachability(
    data_dir: Path,
    image_size: int,
    input_channels: int,
    config: DetectionLossConfig,
) -> dict[str, Any]:
    flat_anchors = [
        (level, index, float(width), float(height))
        for level, anchors in enumerate(config.anchor_sizes)
        for index, (width, height) in enumerate(anchors)
    ]
    per_split: dict[str, Any] = {}
    for split in MODEL_SPLITS:
        dataset = YoloDetectionDataset(
            data_dir, split=split, image_size=image_size, input_channels=input_channels
        )
        bins: dict[str, Counter[str]] = {}
        shape_iou_by_bin: dict[str, list[float]] = {}
        closed_loop_iou_by_bin: dict[str, list[float]] = {}
        per_class_bins: dict[str, Counter[str]] = {}
        for index in range(len(dataset)):
            _image, target = dataset[index]
            boxes = target["boxes"]
            labels = target["labels"]
            if boxes.numel() == 0:
                continue
            decoded = closed_loop_boxes([{"boxes": boxes, "labels": labels}], config)[0]
            matches = match_recovered_boxes(boxes, labels, decoded)
            for box_index in range(boxes.shape[0]):
                box = boxes[box_index]
                width = float(box[2] - box[0])
                height = float(box[3] - box[1])
                sqrt_area = (max(width, 0.0) * max(height, 0.0)) ** 0.5
                name = size_bin(sqrt_area)
                shape_ious = [
                    (min(width, anchor_w) * min(height, anchor_h))
                    / max(width * height + anchor_w * anchor_h - min(width, anchor_w) * min(height, anchor_h), 1e-12)
                    for _level, _index, anchor_w, anchor_h in flat_anchors
                ]
                best_flat = max(range(len(shape_ious)), key=lambda position: shape_ious[position])
                level, anchor_index, anchor_w, anchor_h = flat_anchors[best_flat]
                log_w = torch.log(torch.tensor(max(width, 1.0) / anchor_w)).item()
                log_h = torch.log(torch.tensor(max(height, 1.0) / anchor_h)).item()
                counter = bins.setdefault(name, Counter())
                counter["targets"] += 1
                counter[f"assigned_level_{level}"] += 1
                counter["positive_above_iou_threshold"] += int(
                    sum(1 for value in shape_ious if value >= config.positive_anchor_iou_threshold) > 0
                )
                counter["wh_outside_decode_clamp"] += int(abs(log_w) > 4.0 or abs(log_h) > 4.0)
                counter["closed_loop_iou_ge50"] += int(matches[box_index]["best_iou"] >= 0.5)
                counter["closed_loop_iou_ge75"] += int(matches[box_index]["best_iou"] >= 0.75)
                shape_iou_by_bin.setdefault(name, []).append(shape_ious[best_flat])
                closed_loop_iou_by_bin.setdefault(name, []).append(matches[box_index]["best_iou"])
                class_name = (
                    dataset.class_names[int(labels[box_index].item())]
                    if int(labels[box_index].item()) < len(dataset.class_names)
                    else f"class_{int(labels[box_index].item())}"
                )
                per_class_bins.setdefault(f"{class_name}/{name}", Counter())["targets"] += 1
        per_split[split] = {
            "size_bins": {name: dict(counter) for name, counter in bins.items()},
            "mean_best_anchor_shape_iou": {
                name: sum(values) / len(values) for name, values in shape_iou_by_bin.items()
            },
            "mean_closed_loop_iou": {
                name: sum(values) / len(values) for name, values in closed_loop_iou_by_bin.items()
            },
            "min_closed_loop_iou": {
                name: min(values) for name, values in closed_loop_iou_by_bin.items()
            },
            "per_class_size_bins": {key: dict(counter) for key, counter in per_class_bins.items()},
        }
    return {
        "anchors_by_level": [list(level) for level in config.anchor_sizes],
        "strides": list(config.strides),
        "positive_anchor_iou_threshold": config.positive_anchor_iou_threshold,
        "decode_wh_log_clamp": [-4.0, 4.0],
        "splits": per_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roundtrip-samples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(str(config_path))
    dataset_cfg = config["dataset"]
    data_dir = Path(args.data_dir or dataset_cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    image_size = int(dataset_cfg["image_size"])
    input_channels = int(dataset_cfg["input_channels"])
    loss_config = DetectionLossConfig.from_config(config)

    payload: dict[str, Any] = {
        "artifact_type": "detection_evaluation_chain_audit",
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "data_dir": str(data_dir),
        "image_size": image_size,
        "num_classes": int(loss_config.num_classes),
        "model_weights_loaded": False,
        "test_split_used_for_metrics": False,
        "A_dataset_geometry": audit_dataset_geometry(data_dir, image_size),
        "B_real_label_roundtrip": audit_real_label_roundtrip(
            data_dir, image_size, input_channels, int(args.roundtrip_samples)
        ),
        "C_synthetic_closed_loop": audit_synthetic_closed_loop(image_size, loss_config),
        "D_grid_alignment": audit_grid_alignment(loss_config),
        "E_metric_definition": audit_metric_definition(loss_config),
        "F_anchor_reachability": audit_anchor_reachability(
            data_dir, image_size, input_channels, loss_config
        ),
    }

    geometry = payload["A_dataset_geometry"]
    closed_loop = payload["C_synthetic_closed_loop"]["assigner_to_decoder_closed_loop"]
    payload["verdicts"] = {
        "coordinate_chain_consistent": bool(
            geometry["iou_space_equivalent_to_original"]
            and payload["C_synthetic_closed_loop"]["dataset_load_max_abs_error_px"] < 1e-3
            and payload["C_synthetic_closed_loop"]["original_restore_max_abs_error_px"] < 1e-3
            and payload["D_grid_alignment"]["max_corner_abs_error_px"] < 1e-3
            and payload["D_grid_alignment"]["max_interior_center_abs_error_px"] < 1e-3
            and closed_loop["min_recovered_iou"] > 0.95
        ),
        "map_is_ap50_ap75_mean_not_coco": bool(
            payload["E_metric_definition"]["project_metric"]["map_equals_mean_of_two_thresholds"]
            and not payload["E_metric_definition"]["project_metric"]["definition"][
                "is_standard_coco_map_50_95"
            ]
        ),
        "metric_identity_sanity_passes": bool(
            payload["E_metric_definition"]["project_metric"]["map"] >= 0.0
            and payload["C_synthetic_closed_loop"]["metric_sanity"]["identity_map"] == 1.0
        ),
    }

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "evaluation_chain_audit.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, value in payload["verdicts"].items():
        print(f"{key} {value}", flush=True)
    print(f"artifact {target}", flush=True)


if __name__ == "__main__":
    main()
