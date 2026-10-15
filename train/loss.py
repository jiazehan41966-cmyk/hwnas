"""Legacy dense-anchor loss and decoder, migrated without numerical changes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F



DEFAULT_ANCHOR_SIZES = (
    ((10.16, 11.44), (10.74, 13.48), (12.65, 17.33)),
    ((18.30, 17.03), (15.04, 22.14), (16.15, 25.94)),
    ((18.84, 22.61), (24.40, 19.94), (22.16, 29.42)),
)


@dataclass(frozen=True)
class DetectionLossConfig:
    num_classes: int
    image_size: int = 640
    strides: tuple[int, ...] = (8, 16, 32)
    anchor_sizes: tuple[tuple[tuple[float, float], ...], ...] = DEFAULT_ANCHOR_SIZES
    box_weight: float = 5.0
    object_weight: float = 1.0
    no_object_weight: float = 0.05
    class_weight: float = 1.0
    positive_anchor_iou_threshold: float = 0.65
    ignore_anchor_iou_threshold: float = 0.35
    neighbor_cell_radius: int = 1
    box_iou_weight: float = 1.0
    box_iou_loss_type: str = "iou"
    object_focal_gamma: float = 2.0
    object_focal_alpha: float = 0.25
    objectness_normalization: str = "fixed_weights"
    class_loss_weights: tuple[float, ...] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DetectionLossConfig":
        dataset_cfg = config.get("dataset", {})
        detection_cfg = config.get("detection", {})
        loss_cfg = config.get("training", {}).get("detection_loss", {})
        anchors = detection_cfg.get("anchor_sizes", DEFAULT_ANCHOR_SIZES)
        return cls(
            num_classes=int(dataset_cfg.get("num_classes", loss_cfg.get("num_classes", 1))),
            image_size=int(detection_cfg.get("input_size", dataset_cfg.get("image_size", 640))),
            strides=tuple(int(value) for value in detection_cfg.get("feature_strides", (8, 16, 32))),
            anchor_sizes=_normalize_anchor_sizes(anchors),
            box_weight=float(loss_cfg.get("box_weight", 5.0)),
            object_weight=float(loss_cfg.get("object_weight", 1.0)),
            no_object_weight=float(loss_cfg.get("no_object_weight", 0.05)),
            class_weight=float(loss_cfg.get("class_weight", 1.0)),
            positive_anchor_iou_threshold=float(loss_cfg.get("positive_anchor_iou_threshold", 0.65)),
            ignore_anchor_iou_threshold=float(loss_cfg.get("ignore_anchor_iou_threshold", 0.35)),
            neighbor_cell_radius=int(loss_cfg.get("neighbor_cell_radius", 1)),
            box_iou_weight=float(loss_cfg.get("box_iou_weight", 1.0)),
            box_iou_loss_type=str(loss_cfg.get("box_iou_loss_type", "iou")).strip().lower(),
            object_focal_gamma=float(loss_cfg.get("object_focal_gamma", 2.0)),
            object_focal_alpha=float(loss_cfg.get("object_focal_alpha", 0.25)),
            objectness_normalization=str(loss_cfg.get("objectness_normalization", "fixed_weights")).strip().lower(),
            class_loss_weights=_normalize_class_loss_weights(
                loss_cfg.get("class_loss_weights"),
                num_classes=int(dataset_cfg.get("num_classes", loss_cfg.get("num_classes", 1))),
            ),
        )


class DenseAnchorDetectionLoss(nn.Module):
    """YOLO-style dense anchor loss for the fixed multi-scale detection head."""

    def __init__(self, config: DetectionLossConfig) -> None:
        super().__init__()
        if config.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(config.strides) != len(config.anchor_sizes):
            raise ValueError("strides and anchor_sizes must have the same length")
        self.config = config
        if config.objectness_normalization not in {"fixed_weights", "legacy_dynamic"}:
            raise ValueError("objectness_normalization must be fixed_weights or legacy_dynamic")
        self.outputs_per_anchor = int(config.num_classes) + 5
        if config.class_loss_weights is None:
            self.register_buffer("class_loss_weight_tensor", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "class_loss_weight_tensor",
                torch.tensor(config.class_loss_weights, dtype=torch.float32),
                persistent=False,
            )

    def forward(
        self,
        predictions: Sequence[torch.Tensor] | dict[str, Sequence[torch.Tensor]],
        targets: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_levels = _extract_prediction_levels(predictions)
        if len(pred_levels) != len(self.config.strides):
            raise ValueError(
                f"expected {len(self.config.strides)} prediction levels, got {len(pred_levels)}"
            )
        assigned = _assign_targets(
            pred_levels,
            targets,
            self.config,
            device=pred_levels[0].device,
        )
        losses = []
        box_loss = pred_levels[0].new_tensor(0.0)
        obj_loss = pred_levels[0].new_tensor(0.0)
        cls_loss = pred_levels[0].new_tensor(0.0)
        positive_count = 0
        ignored_count = 0

        for level_index, raw in enumerate(pred_levels):
            pred = _reshape_predictions(
                raw,
                outputs_per_anchor=self.outputs_per_anchor,
                anchors_per_level=len(self.config.anchor_sizes[level_index]),
            )
            target = assigned[level_index]
            obj_target = target["objectness"]
            pos_mask = obj_target > 0.5
            valid_obj_mask = obj_target >= 0.0
            positive_count += int(pos_mask.sum().item())
            ignored_count += int((obj_target < 0.0).sum().item())

            obj_loss = obj_loss + _weighted_objectness_loss(
                pred[..., 4],
                obj_target,
                valid_obj_mask=valid_obj_mask,
                positive_weight=float(self.config.object_weight),
                negative_weight=float(self.config.no_object_weight),
                focal_gamma=float(self.config.object_focal_gamma),
                focal_alpha=float(self.config.object_focal_alpha),
                normalization=self.config.objectness_normalization,
            )

            if pos_mask.any():
                raw_box_loss = F.smooth_l1_loss(
                    pred[..., 0:4][pos_mask],
                    target["box"][pos_mask],
                    reduction="mean",
                )
                decoded_boxes = _decode_boxes_for_loss(
                    pred[..., 0:4],
                    level_index=level_index,
                    config=self.config,
                )
                iou_loss = _pairwise_box_regression_loss_xyxy(
                    decoded_boxes[pos_mask],
                    target["box_xyxy"][pos_mask],
                    loss_type=self.config.box_iou_loss_type,
                ).mean()
                box_loss = box_loss + raw_box_loss + float(self.config.box_iou_weight) * iou_loss
                cls_loss = cls_loss + F.cross_entropy(
                    pred[..., 5:][pos_mask],
                    target["label"][pos_mask],
                    weight=self.class_loss_weight_tensor if self.class_loss_weight_tensor.numel() else None,
                    reduction="mean",
                )

        total = (
            float(self.config.box_weight) * box_loss
            + obj_loss
            + float(self.config.class_weight) * cls_loss
        )
        losses.append(total)
        return losses[0], {
            "loss": float(total.detach().cpu().item()),
            "box_loss": float(box_loss.detach().cpu().item()),
            "object_loss": float(obj_loss.detach().cpu().item()),
            "class_loss": float(cls_loss.detach().cpu().item()),
            "positive_anchors": float(positive_count),
            "ignored_anchors": float(ignored_count),
        }


def decode_detection_predictions(
    predictions: Sequence[torch.Tensor] | dict[str, Sequence[torch.Tensor]],
    *,
    config: DetectionLossConfig,
    score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    pre_nms_topk: int = 1000,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    pred_levels = _extract_prediction_levels(predictions)
    batch_size = int(pred_levels[0].shape[0])
    per_image: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(batch_size)]

    for level_index, raw in enumerate(pred_levels):
        pred = _reshape_predictions(
            raw,
            outputs_per_anchor=config.num_classes + 5,
            anchors_per_level=len(config.anchor_sizes[level_index]),
        )
        batch, anchors, height, width, _dims = pred.shape
        stride = float(config.strides[level_index])
        anchors_tensor = torch.tensor(
            config.anchor_sizes[level_index],
            dtype=pred.dtype,
            device=pred.device,
        ).view(1, anchors, 1, 1, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=pred.dtype, device=pred.device),
            torch.arange(width, dtype=pred.dtype, device=pred.device),
            indexing="ij",
        )
        center_x = ((torch.sigmoid(pred[..., 0]) * 2.0 - 0.5) + grid_x.view(1, 1, height, width)) * stride
        center_y = ((torch.sigmoid(pred[..., 1]) * 2.0 - 0.5) + grid_y.view(1, 1, height, width)) * stride
        box_wh = anchors_tensor * torch.exp(pred[..., 2:4].clamp(min=-4.0, max=4.0))
        boxes = torch.stack(
            (
                center_x - box_wh[..., 0] / 2.0,
                center_y - box_wh[..., 1] / 2.0,
                center_x + box_wh[..., 0] / 2.0,
                center_y + box_wh[..., 1] / 2.0,
            ),
            dim=-1,
        ).clamp(min=0.0, max=float(config.image_size))
        objectness = torch.sigmoid(pred[..., 4])
        class_prob, labels = torch.softmax(pred[..., 5:], dim=-1).max(dim=-1)
        scores = objectness * class_prob

        for image_index in range(batch):
            mask = scores[image_index] >= float(score_threshold)
            if not mask.any():
                continue
            per_image[image_index].append(
                {
                    "boxes": boxes[image_index][mask],
                    "scores": scores[image_index][mask],
                    "labels": labels[image_index][mask].to(torch.long),
                }
            )

    results: list[dict[str, torch.Tensor]] = []
    for image_parts in per_image:
        if not image_parts:
            results.append(_empty_prediction(pred_levels[0].device))
            continue
        boxes = torch.cat([part["boxes"] for part in image_parts], dim=0)
        scores = torch.cat([part["scores"] for part in image_parts], dim=0)
        labels = torch.cat([part["labels"] for part in image_parts], dim=0)
        if int(pre_nms_topk) > 0 and scores.numel() > int(pre_nms_topk):
            topk = scores.topk(int(pre_nms_topk)).indices
            boxes = boxes[topk]
            scores = scores[topk]
            labels = labels[topk]
        keep_indices: list[int] = []
        for label in labels.unique().tolist():
            class_mask = labels == int(label)
            class_indices = torch.where(class_mask)[0]
            keep = _nms_xyxy(
                boxes[class_indices],
                scores[class_indices],
                iou_threshold=float(nms_iou_threshold),
            )
            keep_indices.extend(class_indices[keep].tolist())
        if keep_indices:
            keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
            order = scores[keep_tensor].argsort(descending=True)
            keep_tensor = keep_tensor[order][: int(max_detections)]
            results.append(
                {
                    "boxes": boxes[keep_tensor].detach().cpu(),
                    "scores": scores[keep_tensor].detach().cpu(),
                    "labels": labels[keep_tensor].detach().cpu(),
                }
            )
        else:
            results.append(_empty_prediction(pred_levels[0].device))
    return results






def _normalize_anchor_sizes(value: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    levels = []
    for level in value:
        anchors = []
        for anchor in level:
            if len(anchor) != 2:
                raise ValueError("anchor sizes must be [width, height] pairs")
            anchors.append((float(anchor[0]), float(anchor[1])))
        if not anchors:
            raise ValueError("each detection level must have at least one anchor")
        levels.append(tuple(anchors))
    if not levels:
        raise ValueError("anchor_sizes must not be empty")
    return tuple(levels)


def _normalize_class_loss_weights(value: Any, *, num_classes: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    weights = tuple(float(item) for item in value)
    if len(weights) != int(num_classes):
        raise ValueError("class_loss_weights must match num_classes")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("class_loss_weights must be positive")
    return weights


def _extract_prediction_levels(
    predictions: Sequence[torch.Tensor] | dict[str, Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    if isinstance(predictions, dict):
        predictions = predictions["predictions"]
    return tuple(predictions)


def _reshape_predictions(
    raw: torch.Tensor,
    *,
    outputs_per_anchor: int,
    anchors_per_level: int,
) -> torch.Tensor:
    batch, channels, height, width = raw.shape
    expected_channels = int(outputs_per_anchor) * int(anchors_per_level)
    if channels != expected_channels:
        raise ValueError(
            f"prediction channels {channels} do not match anchors/output layout {expected_channels}"
        )
    return raw.view(
        batch,
        anchors_per_level,
        outputs_per_anchor,
        height,
        width,
    ).permute(0, 1, 3, 4, 2).contiguous()


def _assign_targets(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[dict[str, Any]],
    config: DetectionLossConfig,
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    assigned = []
    for level_index, raw in enumerate(predictions):
        batch, _channels, height, width = raw.shape
        anchors = len(config.anchor_sizes[level_index])
        assigned.append(
            {
                "objectness": torch.zeros((batch, anchors, height, width), dtype=torch.float32, device=device),
                "box": torch.zeros((batch, anchors, height, width, 4), dtype=torch.float32, device=device),
                "box_xyxy": torch.zeros((batch, anchors, height, width, 4), dtype=torch.float32, device=device),
                "label": torch.zeros((batch, anchors, height, width), dtype=torch.long, device=device),
            }
        )

    flat_anchors = []
    for level_index, anchors in enumerate(config.anchor_sizes):
        for anchor_index, (anchor_w, anchor_h) in enumerate(anchors):
            flat_anchors.append((level_index, anchor_index, float(anchor_w), float(anchor_h)))

    for batch_index, target in enumerate(targets):
        boxes = torch.as_tensor(target.get("boxes", []), dtype=torch.float32)
        labels = torch.as_tensor(target.get("labels", []), dtype=torch.long)
        if boxes.numel() == 0:
            continue
        boxes = boxes.reshape(-1, 4)
        labels = labels.reshape(-1)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            anchor_ious = [
                _anchor_iou(width, height, anchor_w, anchor_h)
                for _level, _anchor, anchor_w, anchor_h in flat_anchors
            ]
            best_index = max(range(len(anchor_ious)), key=lambda idx: anchor_ious[idx])
            positive_indices = {
                idx
                for idx, iou in enumerate(anchor_ious)
                if iou >= float(config.positive_anchor_iou_threshold)
            }
            positive_indices.add(best_index)
            ignore_indices = {
                idx
                for idx, iou in enumerate(anchor_ious)
                if iou >= float(config.ignore_anchor_iou_threshold)
            }
            for flat_index in ignore_indices - positive_indices:
                level_index, anchor_index, _anchor_w, _anchor_h = flat_anchors[flat_index]
                stride = float(config.strides[level_index])
                out_h = int(predictions[level_index].shape[-2])
                out_w = int(predictions[level_index].shape[-1])
                for grid_x, grid_y in _assignment_cells(cx, cy, stride, out_w, out_h, config.neighbor_cell_radius):
                    slot = assigned[level_index]["objectness"]
                    if slot[batch_index, anchor_index, grid_y, grid_x] < 0.5:
                        slot[batch_index, anchor_index, grid_y, grid_x] = -1.0

            for flat_index in positive_indices:
                level_index, anchor_index, anchor_w, anchor_h = flat_anchors[flat_index]
                stride = float(config.strides[level_index])
                out_h = int(predictions[level_index].shape[-2])
                out_w = int(predictions[level_index].shape[-1])
                for grid_x, grid_y in _assignment_cells(cx, cy, stride, out_w, out_h, config.neighbor_cell_radius):
                    target_slot = assigned[level_index]
                    target_slot["objectness"][batch_index, anchor_index, grid_y, grid_x] = 1.0
                    target_slot["box"][batch_index, anchor_index, grid_y, grid_x] = torch.tensor(
                        [
                            _safe_logit(((cx / stride - grid_x) + 0.5) / 2.0),
                            _safe_logit(((cy / stride - grid_y) + 0.5) / 2.0),
                            torch.log(torch.tensor(width / anchor_w)).item(),
                            torch.log(torch.tensor(height / anchor_h)).item(),
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    target_slot["box_xyxy"][batch_index, anchor_index, grid_y, grid_x] = torch.tensor(
                        [x1, y1, x2, y2],
                        dtype=torch.float32,
                        device=device,
                    )
                    target_slot["label"][batch_index, anchor_index, grid_y, grid_x] = int(label.item())
    return assigned


def _assignment_cells(
    cx: float,
    cy: float,
    stride: float,
    out_w: int,
    out_h: int,
    radius: int,
) -> list[tuple[int, int]]:
    grid_cx = float(cx) / float(stride)
    grid_cy = float(cy) / float(stride)
    base_x = min(int(out_w) - 1, max(0, int(grid_cx)))
    base_y = min(int(out_h) - 1, max(0, int(grid_cy)))
    cells = {(base_x, base_y)}
    if int(radius) <= 0:
        return sorted(cells)
    frac_x = grid_cx - base_x
    frac_y = grid_cy - base_y
    if frac_x < 0.5 and base_x > 0:
        cells.add((base_x - 1, base_y))
    if frac_x > 0.5 and base_x + 1 < int(out_w):
        cells.add((base_x + 1, base_y))
    if frac_y < 0.5 and base_y > 0:
        cells.add((base_x, base_y - 1))
    if frac_y > 0.5 and base_y + 1 < int(out_h):
        cells.add((base_x, base_y + 1))
    return sorted(cells)


def _decode_boxes_for_loss(
    raw_box: torch.Tensor,
    *,
    level_index: int,
    config: DetectionLossConfig,
) -> torch.Tensor:
    batch, anchors, height, width, _dims = raw_box.shape
    stride = float(config.strides[level_index])
    anchors_tensor = torch.tensor(
        config.anchor_sizes[level_index],
        dtype=raw_box.dtype,
        device=raw_box.device,
    ).view(1, anchors, 1, 1, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=raw_box.dtype, device=raw_box.device),
        torch.arange(width, dtype=raw_box.dtype, device=raw_box.device),
        indexing="ij",
    )
    center_x = ((torch.sigmoid(raw_box[..., 0]) * 2.0 - 0.5) + grid_x.view(1, 1, height, width)) * stride
    center_y = ((torch.sigmoid(raw_box[..., 1]) * 2.0 - 0.5) + grid_y.view(1, 1, height, width)) * stride
    box_wh = anchors_tensor * torch.exp(raw_box[..., 2:4].clamp(min=-4.0, max=4.0))
    return torch.stack(
        (
            center_x - box_wh[..., 0] / 2.0,
            center_y - box_wh[..., 1] / 2.0,
            center_x + box_wh[..., 0] / 2.0,
            center_y + box_wh[..., 1] / 2.0,
        ),
        dim=-1,
    ).clamp(min=0.0, max=float(config.image_size))


def _anchor_iou(width: float, height: float, anchor_w: float, anchor_h: float) -> float:
    inter = min(float(width), float(anchor_w)) * min(float(height), float(anchor_h))
    union = float(width) * float(height) + float(anchor_w) * float(anchor_h) - inter
    return inter / max(union, 1e-12)


def _safe_logit(value: float, eps: float = 1e-4) -> float:
    clipped = min(1.0 - float(eps), max(float(eps), float(value)))
    return float(torch.logit(torch.tensor(clipped)).item())


def _weighted_objectness_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_obj_mask: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
    focal_gamma: float,
    focal_alpha: float,
    normalization: str = "fixed_weights",
) -> torch.Tensor:
    if normalization not in {"fixed_weights", "legacy_dynamic"}:
        raise ValueError("objectness_normalization must be fixed_weights or legacy_dynamic")
    if not valid_obj_mask.any():
        return logits.sum() * 0.0
    valid_logits = logits[valid_obj_mask]
    valid_targets = targets[valid_obj_mask]
    bce = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        reduction="none",
    )
    weights = torch.full_like(valid_targets, float(negative_weight))
    weights[valid_targets > 0.5] = float(positive_weight)
    if focal_gamma > 0.0:
        prob = torch.sigmoid(valid_logits)
        pt = prob * valid_targets + (1.0 - prob) * (1.0 - valid_targets)
        focal = (1.0 - pt).clamp_min(0.0).pow(float(focal_gamma))
        alpha = torch.full_like(valid_targets, 1.0 - float(focal_alpha))
        alpha[valid_targets > 0.5] = float(focal_alpha)
        weights = weights * alpha
    else:
        focal = torch.ones_like(weights)
    # Normalizing by differentiable focal weights couples unrelated anchors:
    # on an all-background batch it can produce negative logit gradients,
    # encouraging false objectness. Keep only target-dependent weights in the
    # denominator. The legacy mode is explicit, for historical reproduction.
    numerator = (bce * weights * focal).sum()
    denominator = (weights * focal).sum() if normalization == "legacy_dynamic" else weights.sum()
    return numerator / denominator.clamp_min(1e-12)


def _nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        current_box = boxes[current].view(1, 4)
        rest = order[1:]
        ious = _box_iou_xyxy(current_box, boxes[rest]).view(-1)
        order = rest[ious <= float(iou_threshold)]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-12)


def _pairwise_box_regression_loss_xyxy(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError("paired box regression loss requires matching pred/target shapes")
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros((0,))

    iou = torch.diag(_box_iou_xyxy(pred_boxes, target_boxes))
    normalized = str(loss_type or "iou").strip().lower()
    if normalized == "iou":
        return 1.0 - iou
    if normalized not in {"diou", "ciou"}:
        raise ValueError(f"unsupported box_iou_loss_type: {loss_type}")

    pred_center = (pred_boxes[:, 0:2] + pred_boxes[:, 2:4]) * 0.5
    target_center = (target_boxes[:, 0:2] + target_boxes[:, 2:4]) * 0.5
    center_distance_sq = (pred_center - target_center).square().sum(dim=1)

    enclosing_lt = torch.minimum(pred_boxes[:, 0:2], target_boxes[:, 0:2])
    enclosing_rb = torch.maximum(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    enclosing_diag_sq = (enclosing_rb - enclosing_lt).clamp_min(0).square().sum(dim=1).clamp_min(1e-12)
    diou = iou - center_distance_sq / enclosing_diag_sq
    if normalized == "diou":
        return 1.0 - diou

    pred_wh = (pred_boxes[:, 2:4] - pred_boxes[:, 0:2]).clamp_min(1e-6)
    target_wh = (target_boxes[:, 2:4] - target_boxes[:, 0:2]).clamp_min(1e-6)
    aspect_term = (4.0 / (torch.pi**2)) * (
        torch.atan(target_wh[:, 0] / target_wh[:, 1])
        - torch.atan(pred_wh[:, 0] / pred_wh[:, 1])
    ).square()
    with torch.no_grad():
        alpha = aspect_term / (1.0 - iou + aspect_term).clamp_min(1e-12)
    ciou = diou - alpha * aspect_term
    return 1.0 - ciou


def _empty_prediction(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device).cpu(),
        "scores": torch.empty((0,), dtype=torch.float32, device=device).cpu(),
        "labels": torch.empty((0,), dtype=torch.long, device=device).cpu(),
    }


def _cpu_targets(targets: Sequence[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.as_tensor(target.get("boxes", []), dtype=torch.float32).reshape(-1, 4).cpu(),
            "labels": torch.as_tensor(target.get("labels", []), dtype=torch.long).reshape(-1).cpu(),
        }
        for target in targets
    ]
