"""Detection metric helpers for the multi-scale detection branch."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def _as_tensor(value: Any, *, columns: int, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.numel() == 0:
        return torch.empty((0, columns), dtype=dtype) if columns > 1 else torch.empty((0,), dtype=dtype)
    if columns == 1:
        return tensor.reshape(-1)
    return tensor.reshape(-1, columns)


def box_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    boxes_a = _as_tensor(boxes_a, columns=4, dtype=torch.float32)
    boxes_b = _as_tensor(boxes_b, columns=4, dtype=torch.float32)
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=torch.float32)

    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0)
        * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0)
    )
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0)
        * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0)
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(1e-12)


def _prepare_records(
    records: Sequence[Mapping[str, Any]],
    *,
    with_scores: bool,
) -> list[dict[str, torch.Tensor]]:
    prepared: list[dict[str, torch.Tensor]] = []
    for record in records:
        boxes = _as_tensor(record.get("boxes", []), columns=4, dtype=torch.float32)
        labels = _as_tensor(record.get("labels", []), columns=1, dtype=torch.long)
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError("detection boxes and labels must have the same length")
        item = {"boxes": boxes, "labels": labels}
        if with_scores:
            scores = _as_tensor(record.get("scores", []), columns=1, dtype=torch.float32)
            if scores.shape[0] != boxes.shape[0]:
                raise ValueError("detection boxes and scores must have the same length")
            item["scores"] = scores
        prepared.append(item)
    return prepared


def _average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    mrec = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    changed = torch.where(mrec[1:] != mrec[:-1])[0]
    if changed.numel() == 0:
        return 0.0
    return float(((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]).sum().item())


def _match_class_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    class_index: int,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    detections: list[tuple[float, int, int]] = []
    target_boxes_by_image: dict[int, torch.Tensor] = {}
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        pred_mask = (
            (prediction["labels"] == class_index)
            & (prediction["scores"] >= float(score_threshold))
        )
        for pred_index, score in zip(
            torch.where(pred_mask)[0].tolist(),
            prediction["scores"][pred_mask].tolist(),
        ):
            detections.append((float(score), image_index, int(pred_index)))
        target_mask = target["labels"] == class_index
        target_boxes_by_image[image_index] = target["boxes"][target_mask]

    target_count = sum(int(boxes.shape[0]) for boxes in target_boxes_by_image.values())
    if not detections:
        return torch.zeros(0), torch.zeros(0), target_count

    detections.sort(key=lambda item: item[0], reverse=True)
    matched_targets = {
        image_index: torch.zeros(boxes.shape[0], dtype=torch.bool)
        for image_index, boxes in target_boxes_by_image.items()
    }
    tp = torch.zeros(len(detections), dtype=torch.float32)
    fp = torch.zeros(len(detections), dtype=torch.float32)

    for det_index, (_score, image_index, pred_index) in enumerate(detections):
        pred_box = predictions[image_index]["boxes"][pred_index].view(1, 4)
        target_boxes = target_boxes_by_image[image_index]
        if target_boxes.numel() == 0:
            fp[det_index] = 1.0
            continue
        ious = box_iou_xyxy(pred_box, target_boxes).view(-1)
        best_iou, best_target = torch.max(ious, dim=0)
        if (
            float(best_iou.item()) >= float(iou_threshold)
            and not bool(matched_targets[image_index][best_target].item())
        ):
            tp[det_index] = 1.0
            matched_targets[image_index][best_target] = True
        else:
            fp[det_index] = 1.0
    return tp, fp, target_count


def evaluate_detection_predictions(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    primary_iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute mAP plus aggregate precision, recall, and F1.

    Records use ``boxes`` in xyxy format, integer ``labels``, and prediction
    ``scores``. ``map`` is the mean AP over classes with at least one target
    and over the requested IoU thresholds. With the project default
    ``iou_thresholds=(0.5, 0.75)`` that is the mean of AP50 and AP75, which is
    not the standard COCO AP@[0.50:0.95]. Per-class AP uses all-points
    interpolation rather than the COCO 101-point rule. Use
    ``hwnas_fpga.training.detection_coco_eval`` for the standard metric.
    """

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")

    preds = _prepare_records(predictions, with_scores=True)
    gts = _prepare_records(targets, with_scores=False)

    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in thresholds}
    aggregate_tp = 0.0
    aggregate_fp = 0.0
    aggregate_targets = 0

    for threshold in thresholds:
        for class_index in range(int(num_classes)):
            tp, fp, target_count = _match_class_predictions(
                preds,
                gts,
                class_index=class_index,
                iou_threshold=threshold,
                score_threshold=score_threshold,
            )
            if target_count <= 0:
                continue
            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recalls = tp_cum / max(1, target_count)
            precisions = tp_cum / (tp_cum + fp_cum).clamp_min(1e-12)
            ap_by_threshold[threshold].append(_average_precision(recalls, precisions))
            if abs(threshold - float(primary_iou_threshold)) < 1e-12:
                aggregate_tp += float(tp.sum().item())
                aggregate_fp += float(fp.sum().item())
                aggregate_targets += int(target_count)

    all_ap = [value for values in ap_by_threshold.values() for value in values]
    precision = aggregate_tp / (aggregate_tp + aggregate_fp) if aggregate_tp + aggregate_fp > 0 else 0.0
    recall = aggregate_tp / aggregate_targets if aggregate_targets > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    map50_values = ap_by_threshold.get(0.5, [])
    map75_values = ap_by_threshold.get(0.75, [])
    return {
        "map": sum(all_ap) / max(1, len(all_ap)),
        "map50": sum(map50_values) / max(1, len(map50_values)),
        "map75": sum(map75_values) / max(1, len(map75_values)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap_by_iou": {
            f"{threshold:.2f}": values
            for threshold, values in ap_by_threshold.items()
        },
        "primary_iou_threshold": float(primary_iou_threshold),
        "score_threshold": float(score_threshold),
        "num_images": len(preds),
        "num_classes": int(num_classes),
        "num_targets": int(sum(item["labels"].numel() for item in gts)),
        "map_definition": {
            "name": "mean_ap_over_requested_iou_thresholds",
            "iou_thresholds": [float(value) for value in thresholds],
            "interpolation": "all_points",
            "is_standard_coco_map_50_95": False,
            "note": (
                "map is the mean AP over the listed IoU thresholds only. Do not "
                "report it as COCO mAP@[0.50:0.95]."
            ),
        },
    }
