"""Standard COCO-style detection AP, reported separately from the project metric.

The project selection metric stays the mean of AP50 and AP75 produced by
``detection_metrics.evaluate_detection_predictions``. This module adds the
standard COCO definition for reporting only: AP averaged over IoU
0.50:0.05:0.95, 101-point recall interpolation, ``maxDets`` applied per image
and per category, and no score threshold.

Areas are measured in the square network input space, because the loader maps
every image to ``image_size`` with a plain resize. For a square source export
that is a uniform scale, so ``area_ranges`` must be given in the same space as
the boxes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .detection_metrics import box_iou_xyxy

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * index, 2) for index in range(10)
)
COCO_RECALL_POINTS: tuple[float, ...] = tuple(
    round(0.01 * index, 2) for index in range(101)
)
COCO_AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}


def _stack(records: Sequence[Mapping[str, Any]], *, with_scores: bool) -> list[dict[str, torch.Tensor]]:
    prepared: list[dict[str, torch.Tensor]] = []
    for record in records:
        boxes = torch.as_tensor(record.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(record.get("labels", []), dtype=torch.long).reshape(-1)
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError("detection boxes and labels must have the same length")
        item = {"boxes": boxes, "labels": labels}
        if with_scores:
            scores = torch.as_tensor(record.get("scores", []), dtype=torch.float32).reshape(-1)
            if scores.shape[0] != boxes.shape[0]:
                raise ValueError("detection boxes and scores must have the same length")
            item["scores"] = scores
        prepared.append(item)
    return prepared


def _box_areas(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return widths * heights


def _interpolate_101(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """COCO precision envelope sampled on 101 fixed recall points."""

    if recalls.numel() == 0:
        return 0.0
    envelope = precisions.clone()
    for index in range(envelope.numel() - 1, 0, -1):
        envelope[index - 1] = torch.maximum(envelope[index - 1], envelope[index])
    points = torch.tensor(COCO_RECALL_POINTS, dtype=torch.float32)
    positions = torch.searchsorted(recalls.contiguous(), points, right=False)
    sampled = torch.zeros_like(points)
    valid = positions < recalls.numel()
    sampled[valid] = envelope[positions[valid]]
    return float(sampled.mean().item())


def _evaluate_class_area(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    class_index: int,
    iou_threshold: float,
    area_range: tuple[float, float],
    max_detections: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return scores, tp, fp flags for kept detections plus in-range GT count."""

    all_scores: list[torch.Tensor] = []
    all_tp: list[torch.Tensor] = []
    all_fp: list[torch.Tensor] = []
    positives = 0

    for prediction, target in zip(predictions, targets):
        gt_mask = target["labels"] == class_index
        gt_boxes = target["boxes"][gt_mask]
        gt_areas = _box_areas(gt_boxes)
        gt_ignore = (gt_areas < area_range[0]) | (gt_areas > area_range[1])
        positives += int((~gt_ignore).sum().item())

        pred_mask = prediction["labels"] == class_index
        pred_boxes = prediction["boxes"][pred_mask]
        pred_scores = prediction["scores"][pred_mask]
        if pred_scores.numel() == 0:
            continue
        order = pred_scores.argsort(descending=True)[: int(max_detections)]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        ious = (
            box_iou_xyxy(pred_boxes, gt_boxes)
            if gt_boxes.shape[0] > 0
            else torch.zeros((pred_boxes.shape[0], 0))
        )
        matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        tp = torch.zeros(pred_boxes.shape[0], dtype=torch.float32)
        fp = torch.zeros(pred_boxes.shape[0], dtype=torch.float32)
        ignored = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
        pred_areas = _box_areas(pred_boxes)

        for det_index in range(pred_boxes.shape[0]):
            best_iou = float(iou_threshold) - 1e-12
            best_gt = -1
            for gt_index in range(gt_boxes.shape[0]):
                if matched[gt_index] and not bool(gt_ignore[gt_index].item()):
                    continue
                if best_gt >= 0 and not bool(gt_ignore[best_gt].item()) and bool(gt_ignore[gt_index].item()):
                    break
                value = float(ious[det_index, gt_index].item())
                if value < best_iou:
                    continue
                best_iou = value
                best_gt = gt_index
            if best_gt < 0:
                in_range = (
                    float(pred_areas[det_index].item()) >= area_range[0]
                    and float(pred_areas[det_index].item()) <= area_range[1]
                )
                if in_range:
                    fp[det_index] = 1.0
                else:
                    ignored[det_index] = True
                continue
            if bool(gt_ignore[best_gt].item()):
                ignored[det_index] = True
                continue
            matched[best_gt] = True
            tp[det_index] = 1.0

        keep = ~ignored
        all_scores.append(pred_scores[keep])
        all_tp.append(tp[keep])
        all_fp.append(fp[keep])

    if not all_scores:
        return torch.zeros(0), torch.zeros(0), torch.zeros(0), positives
    return (
        torch.cat(all_scores),
        torch.cat(all_tp),
        torch.cat(all_fp),
        positives,
    )


def evaluate_detection_predictions_coco(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    max_detections: int = 100,
    area_ranges: Mapping[str, tuple[float, float]] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Standard COCO AP for reporting only; never used for epoch selection."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")

    preds = _stack(predictions, with_scores=True)
    gts = _stack(targets, with_scores=False)
    ranges = dict(area_ranges or COCO_AREA_RANGES)
    thresholds = [float(value) for value in iou_thresholds]

    per_class_ap: dict[str, dict[str, dict[str, float]]] = {}
    for area_name, area_range in ranges.items():
        per_class_ap[area_name] = {}
        for class_index in range(int(num_classes)):
            name = (
                str(class_names[class_index])
                if class_names is not None and class_index < len(class_names)
                else f"class_{class_index}"
            )
            by_threshold: dict[str, float] = {}
            for threshold in thresholds:
                scores, tp, fp, positives = _evaluate_class_area(
                    preds,
                    gts,
                    class_index=class_index,
                    iou_threshold=threshold,
                    area_range=area_range,
                    max_detections=int(max_detections),
                )
                if positives <= 0:
                    continue
                if scores.numel() == 0:
                    by_threshold[f"{threshold:.2f}"] = 0.0
                    continue
                order = scores.argsort(descending=True)
                tp_cum = torch.cumsum(tp[order], dim=0)
                fp_cum = torch.cumsum(fp[order], dim=0)
                recalls = tp_cum / float(positives)
                precisions = tp_cum / (tp_cum + fp_cum).clamp_min(1e-12)
                by_threshold[f"{threshold:.2f}"] = _interpolate_101(recalls, precisions)
            if by_threshold:
                per_class_ap[area_name][name] = by_threshold

    def _mean_over(area_name: str, keys: Sequence[str]) -> float:
        values = [
            value
            for by_threshold in per_class_ap.get(area_name, {}).values()
            for key, value in by_threshold.items()
            if key in set(keys)
        ]
        return sum(values) / len(values) if values else 0.0

    all_keys = [f"{threshold:.2f}" for threshold in thresholds]
    summary = {
        "metric_definition": {
            "name": "coco_ap",
            "iou_thresholds": thresholds,
            "interpolation": "coco_101_point",
            "max_detections_per_image_per_class": int(max_detections),
            "score_threshold": 0.0,
            "area_space": "square_network_input_pixels",
            "is_standard_coco_map_50_95": all_keys == [f"{v:.2f}" for v in COCO_IOU_THRESHOLDS],
            "note": "Reporting-only supplement; epoch and checkpoint selection is unchanged.",
        },
        "coco_ap": _mean_over("all", all_keys),
        "coco_ap50": _mean_over("all", ["0.50"]),
        "coco_ap75": _mean_over("all", ["0.75"]),
        "coco_ap_small": _mean_over("small", all_keys),
        "coco_ap_medium": _mean_over("medium", all_keys),
        "coco_ap_large": _mean_over("large", all_keys),
        "per_class_ap": per_class_ap,
        "num_images": len(preds),
        "num_classes": int(num_classes),
        "num_targets": int(sum(item["labels"].numel() for item in gts)),
    }
    return summary
