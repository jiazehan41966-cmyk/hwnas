"""Three paper-inspired adaptations, not author-method reproductions.

No labels, assignments, model parameters, or inference computation are changed.
Rank: same-class foreground/background log-score ordering and quality ordering.
PISA: bounded, group-normalized reliable-positive/local-negative reweighting.
FG/BG: box-center foreground and exterior-background feature prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from hwnas_fpga.training.detection_metrics import box_iou_xyxy
from hwnas_fpga.training.detection_trainer import (
    DenseAnchorDetectionLoss, DetectionLossConfig, _assign_targets,
    _decode_boxes_for_loss, _extract_prediction_levels,
    _pairwise_box_regression_loss_xyxy, _reshape_predictions,
)


@dataclass(frozen=True)
class BackgroundOptions:
    variant: str = "baseline"
    rank_weight: float = 0.1
    rank_margin: float = 0.5
    positive_topk: int = 16
    negative_topk: int = 32
    pisa_strength: float = 0.5
    negative_tile_cells: int = 4
    prototype_weight: float = 0.1
    prototype_temperature: float = 0.2
    prototype_samples_per_scale: int = 32


def log_class_scores(pred: torch.Tensor) -> torch.Tensor:
    """Stable log of the unchanged inference score for every class."""
    return F.logsigmoid(pred[..., 4:5]) + F.log_softmax(pred[..., 5:], dim=-1)


def rank_pairs(positive, negative, margin=0.5):
    if positive.numel() == 0 or negative.numel() == 0:
        return (positive.sum() + negative.sum()) * 0.0
    return F.softplus(margin + negative[:, None] - positive[None, :]).mean()


def exterior_cells(target, height, width, stride, device, margin_cells=1.0):
    ys, xs = torch.meshgrid(
        (torch.arange(height, device=device) + 0.5) * stride,
        (torch.arange(width, device=device) + 0.5) * stride, indexing="ij")
    outside = torch.ones((height, width), dtype=torch.bool, device=device)
    for box in target["boxes"].to(device):
        outside &= ~((xs >= box[0] - margin_cells * stride)
                     & (xs <= box[2] + margin_cells * stride)
                     & (ys >= box[1] - margin_cells * stride)
                     & (ys <= box[3] + margin_cells * stride))
    return outside


def quality_positive_weights(quality, groups, strength):
    """Within each GT group, reward reliable boxes without changing total mass."""
    weights = torch.ones_like(quality)
    if quality.numel() == 0:
        return weights
    _, group_ids = torch.unique(groups, dim=0, return_inverse=True)
    for group in group_ids.unique():
        mask = group_ids == group
        values = quality[mask]
        # Average ranks make ties neutral, instead of arbitrary winner selection.
        priority = ((values[:, None] > values[None, :]).float().sum(1)
                    + 0.5 * (values[:, None] == values[None, :]).float().sum(1))
        priority = priority / max(1, values.numel())
        local = 1 + strength * (2 * priority - 1)
        weights[mask] = local / local.mean().clamp_min(1e-8)
    return weights.detach()


def local_negative_weights(scores, eligible, tile_cells, strength):
    """One high-score representative per spatial tile, normalized per image."""
    weights = torch.ones_like(scores)
    batch, anchors, height, width = scores.shape
    ys, xs = torch.meshgrid(torch.arange(height, device=scores.device),
                            torch.arange(width, device=scores.device), indexing="ij")
    tiles_per_row = (width + tile_cells - 1) // tile_cells
    ids = ((ys // tile_cells) * tiles_per_row + xs // tile_cells)
    ids = ids.expand(anchors, -1, -1).reshape(-1)
    for image in range(batch):
        mask = eligible[image].reshape(-1)
        if not mask.any():
            continue
        values = scores[image].detach().reshape(-1)[mask]
        groups = ids[mask]
        maxima = values.new_full((int(ids.max()) + 1,), float("-inf"))
        maxima.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
        representative = values == maxima[groups]
        local = torch.where(representative, 1 + strength, 1 - strength)
        local = local / local.mean().clamp_min(1e-8)
        flat = weights[image].reshape(-1)
        flat[mask] = local
    return weights.detach()


def weighted_objectness(logits, targets, valid, cfg, extra):
    values, labels = logits[valid], targets[valid]
    if not values.numel():
        return logits.sum() * 0.0
    weights = torch.full_like(labels, cfg.no_object_weight)
    weights[labels > 0.5] = cfg.object_weight
    probability = values.sigmoid()
    pt = probability * labels + (1 - probability) * (1 - labels)
    if cfg.object_focal_gamma > 0:
        alpha = torch.full_like(labels, 1 - cfg.object_focal_alpha)
        alpha[labels > 0.5] = cfg.object_focal_alpha
        weights = weights * alpha
        focal = (1 - pt).pow(cfg.object_focal_gamma)
    else:
        focal = torch.ones_like(labels)
    weights = weights * extra[valid]
    # Detached, target/sample-dependent normalizer; never normalize by focal.
    return (F.binary_cross_entropy_with_logits(values, labels, reduction="none")
            * weights * focal).sum() / weights.sum().clamp_min(1e-12)


def prototype_separation(foreground, backgrounds, temperature):
    """Leave-one-out within-group attraction and cross-group separation."""
    groups = [g for g in list(foreground) + list(backgrounds) if g.numel()]
    n_foreground = sum(bool(g.numel()) for g in foreground)
    if not n_foreground or len(groups) <= n_foreground:
        return sum((g.sum() for g in groups), torch.tensor(0.0, device=groups[0].device)) * 0
    groups = [F.normalize(g, dim=-1, eps=1e-6) for g in groups]
    prototypes = [F.normalize(g.mean(0), dim=0, eps=1e-6) for g in groups]
    losses = []
    for index, group in enumerate(groups):
        if len(group) > 1:
            own = F.normalize((group.sum(0)[None] - group) / (len(group) - 1), dim=-1, eps=1e-6)
        else:
            # No attraction term from an example to itself.
            own = group.detach()
        own_similarity = (group * own).sum(-1)
        opposite = prototypes[n_foreground:] if index < n_foreground else prototypes[:n_foreground]
        competitors = group @ torch.stack(opposite).T
        losses.append(F.softplus((competitors.max(-1).values - own_similarity + 0.2)
                                 / temperature).mean())
    return torch.stack(losses).mean()


class BackgroundDetectionLoss(DenseAnchorDetectionLoss):
    def __init__(self, config: DetectionLossConfig, options: BackgroundOptions):
        super().__init__(config)
        if options.variant not in {"baseline", "rank", "pisa", "fg_bg"}:
            raise ValueError("unknown background variant")
        if config.objectness_normalization != "fixed_weights":
            raise ValueError("background experiments require fixed_weights")
        if not 0 <= options.pisa_strength < 1:
            raise ValueError("pisa_strength must be in [0, 1)")
        if options.prototype_temperature <= 0 or options.negative_tile_cells < 1:
            raise ValueError("temperature and tile size must be positive")
        self.options = options
        self.features = None
        self.last_auxiliary = {}

    def capture_features(self, _module, _inputs, output):
        self.features = output

    def forward(self, predictions, targets):
        if self.options.variant == "baseline":
            return super().forward(predictions, targets)
        levels = _extract_prediction_levels(predictions)
        assigned = _assign_targets(levels, targets, self.config, device=levels[0].device)
        if self.options.variant == "pisa" and self.options.pisa_strength > 0:
            return self._pisa(levels, assigned)
        base, parts = super().forward(predictions, targets)
        if self.options.variant == "rank":
            auxiliary = self._ranking(levels, assigned, targets)
            coefficient = self.options.rank_weight
        elif self.options.variant == "fg_bg":
            if self.features is None:
                raise RuntimeError("fg_bg requires the training-only neck feature hook")
            auxiliary = self._prototypes(levels, targets)
            coefficient = self.options.prototype_weight
            self.features = None
        else:
            auxiliary = base * 0
            coefficient = 0.0
        total = base + coefficient * auxiliary
        parts.update(loss=float(total.detach()), background_auxiliary=float(auxiliary.detach()))
        self.last_auxiliary = {"raw": float(auxiliary.detach()), "weight": coefficient}
        return total, parts

    def _ranking(self, levels, assigned, targets):
        rank_losses, sort_losses = [], []
        for level_index, raw in enumerate(levels):
            pred = _reshape_predictions(raw, outputs_per_anchor=self.outputs_per_anchor,
                                       anchors_per_level=len(self.config.anchor_sizes[level_index]))
            target = assigned[level_index]
            boxes = _decode_boxes_for_loss(pred[..., :4], level_index=level_index, config=self.config)
            scores = log_class_scores(pred)
            for image, actual in enumerate(targets):
                positive = target["objectness"][image] > 0.5
                background = target["objectness"][image] == 0
                exterior = exterior_cells(actual, pred.shape[2], pred.shape[3],
                                           self.config.strides[level_index], raw.device)
                background &= exterior[None]
                if len(actual["boxes"]) and background.any():
                    ious = box_iou_xyxy(boxes[image][background].detach(), actual["boxes"].to(raw.device))
                    selected = background.nonzero(as_tuple=True)
                    background[selected] = ious.max(-1).values < 0.1
                for label in range(self.config.num_classes):
                    mask = positive & (target["label"][image] == label)
                    if not mask.any() or not background.any():
                        continue
                    positives = scores[image, ..., label][mask]
                    quality = _pairwise_box_regression_loss_xyxy(
                        boxes[image][mask].detach(), target["box_xyxy"][image][mask], loss_type="iou")
                    indices = (-quality).topk(min(self.options.positive_topk, len(quality))).indices
                    negatives = scores[image, ..., label][background]
                    negatives = negatives.topk(min(self.options.negative_topk, len(negatives))).values
                    rank_losses.append(rank_pairs(positives[indices], negatives, self.options.rank_margin))
                    # Sort only boxes assigned to the same GT, without box-loss gradients.
                    gt = target["box_xyxy"][image][mask]
                    same_gt = (gt[:, None] == gt[None, :]).all(-1)
                    better = same_gt & ((quality[None, :] - quality[:, None]) > 0.1)
                    if better.any():
                        sort_losses.append(F.softplus(0.2 + positives[None, :] - positives[:, None])[better].mean())
        zero = sum(x.sum() * 0 for x in levels)
        rank = torch.stack(rank_losses).mean() if rank_losses else zero
        sort = torch.stack(sort_losses).mean() if sort_losses else zero
        return rank + 0.25 * sort

    def _pisa(self, levels, assigned):
        obj, box, cls = [levels[0].new_zeros(()) for _ in range(3)]
        count, ignored = 0, 0
        for level_index, raw in enumerate(levels):
            pred = _reshape_predictions(raw, outputs_per_anchor=self.outputs_per_anchor,
                                       anchors_per_level=len(self.config.anchor_sizes[level_index]))
            target = assigned[level_index]
            labels = target["objectness"]
            pos, valid, neg = labels > 0.5, labels >= 0, labels == 0
            count += int(pos.sum())
            ignored += int((~valid).sum())
            decoded = _decode_boxes_for_loss(pred[..., :4], level_index=level_index, config=self.config)
            weights = local_negative_weights(log_class_scores(pred).detach().max(-1).values,
                                             neg, self.options.negative_tile_cells, self.options.pisa_strength)
            if pos.any():
                loss_iou = _pairwise_box_regression_loss_xyxy(decoded[pos], target["box_xyxy"][pos],
                                                             loss_type=self.config.box_iou_loss_type)
                quality = 1 - _pairwise_box_regression_loss_xyxy(decoded[pos].detach(),
                                                               target["box_xyxy"][pos], loss_type="iou")
                image_ids = torch.arange(raw.shape[0], device=raw.device).view(-1, 1, 1, 1).expand_as(pos)[pos]
                groups = torch.cat((image_ids[:, None], target["box_xyxy"][pos]), dim=-1)
                positive_weights = quality_positive_weights(quality, groups, self.options.pisa_strength)
                weights[pos] = positive_weights
                regression = F.smooth_l1_loss(pred[..., :4][pos], target["box"][pos], reduction="none").mean(-1)
                box = box + (positive_weights * (regression + self.config.box_iou_weight * loss_iou)).mean()
                class_weights = self.class_loss_weight_tensor if self.class_loss_weight_tensor.numel() else None
                class_losses = F.cross_entropy(pred[..., 5:][pos], target["label"][pos],
                                               weight=class_weights, reduction="none")
                denominator = (positive_weights.sum() if class_weights is None else
                               (positive_weights * class_weights[target["label"][pos]]).sum())
                cls = cls + (positive_weights * class_losses).sum() / denominator.clamp_min(1e-8)
            obj = obj + weighted_objectness(pred[..., 4], labels, valid, self.config, weights)
        total = self.config.box_weight * box + obj + self.config.class_weight * cls
        return total, {"loss": float(total.detach()), "box_loss": float(box.detach()),
                       "object_loss": float(obj.detach()), "class_loss": float(cls.detach()),
                       "positive_anchors": float(count), "ignored_anchors": float(ignored)}

    def _prototypes(self, levels, targets):
        losses = []
        for image, target in enumerate(targets):
            foreground = [[] for _ in range(self.config.num_classes)]
            backgrounds = []
            for level_index, feature in enumerate(self.features):
                _, channels, height, width = feature.shape
                stride = self.config.strides[level_index]
                ys, xs = torch.meshgrid((torch.arange(height, device=feature.device) + 0.5) * stride,
                                        (torch.arange(width, device=feature.device) + 0.5) * stride, indexing="ij")
                vectors = feature[image].permute(1, 2, 0)
                for gt_box, gt_label in zip(target["boxes"].to(feature.device), target["labels"]):
                    center = (gt_box[:2] + gt_box[2:]) / 2
                    half = (gt_box[2:] - gt_box[:2]) * 0.25
                    mask = ((xs >= center[0] - half[0]) & (xs <= center[0] + half[0])
                            & (ys >= center[1] - half[1]) & (ys <= center[1] + half[1]))
                    if not mask.any():
                        # A box center sample is weak box supervision, not a pixel mask.
                        index = ((xs - center[0]).square() + (ys - center[1]).square()).argmin()
                        mask.reshape(-1)[index] = True
                    foreground[int(gt_label)].append(vectors[mask])
                exterior = exterior_cells(target, height, width, stride, feature.device)
                if exterior.any():
                    pred = _reshape_predictions(levels[level_index], outputs_per_anchor=self.outputs_per_anchor,
                                               anchors_per_level=len(self.config.anchor_sizes[level_index]))
                    score = log_class_scores(pred[image]).detach().amax(dim=(0, 3))
                    ids = score[exterior].topk(min(self.options.prototype_samples_per_scale, int(exterior.sum()))).indices
                    backgrounds.append(vectors[exterior][ids])
            fg = [torch.cat(group) for group in foreground if group]
            if fg and backgrounds:
                losses.append(prototype_separation(fg, backgrounds, self.options.prototype_temperature))
        return torch.stack(losses).mean() if losses else sum(f.sum() * 0 for f in self.features)
