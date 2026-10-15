"""FCOS quality-score/DFL adaptation with an untouched AAA control.

QFL and DFL follow equations (2), (5), and (6) in Li et al., NeurIPS 2020:
https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf

This is a bounded FCOS adaptation, not a reproduction of the paper's complete
GFL detector: FCOS assignment, unweighted GIoU/positive normalization, towers,
and postprocessing limits are retained. QFL learns detached predicted-box IoU
at the assigned class. Scores are its sigmoid output, with no centerness branch.
DFL supervises stride-normalized l/t/r/b distributions on bins 0 through 16.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.fcos import FCOS
from torchvision.ops import boxes as box_ops
from torchvision.ops import generalized_box_iou_loss

from tools.sonar_fcos_pilot.model import build_model as build_baseline
from tools.sonar_fcos_pilot.model import model_contract as baseline_contract


VARIANTS = ("AAA", "QFL_DFL")
REG_MAX = 16
QFL_BETA = 2.0
DFL_WEIGHT = 0.25
GIOU_WEIGHT = 1.0
QUALITY_PARAMETERS = {"reg_max": REG_MAX, "qfl_beta": QFL_BETA,
                      "dfl_weight": DFL_WEIGHT, "giou_weight": GIOU_WEIGHT}


def quality_focal_loss(logits, targets, beta=QFL_BETA):
    """Elementwise BCE(logits, y) * abs(sigmoid(logits) - y)**beta."""
    logits, targets = logits.float(), targets.float()
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                * (logits.sigmoid() - targets).abs().pow(beta))


def distribution_focal_loss(logits, targets):
    """Elementwise adjacent-bin CE; the last bin is valid without clipping.

    ``logits`` has one additional final bin dimension relative to ``targets``.
    Out-of-support labels fail rather than silently changing the regression
    target. The inclusive upper endpoint uses only the final bin.
    """
    logits, targets = logits.float(), targets.float()
    maximum = logits.shape[-1] - 1
    if logits.shape[:-1] != targets.shape:
        raise ValueError("DFL logits and targets have incompatible shapes")
    if bool(((targets < 0) | (targets > maximum) | ~torch.isfinite(targets)).any()):
        raise ValueError(f"DFL target outside finite bin support [0, {maximum}]")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        left = targets.floor().long()
        right = (left + 1).clamp(max=maximum)
        right_weight = targets - left
        flat_logits = logits.reshape(-1, maximum + 1)
        left_ce = F.cross_entropy(flat_logits, left.reshape(-1), reduction="none").reshape_as(targets)
        right_ce = F.cross_entropy(flat_logits, right.reshape(-1), reduction="none").reshape_as(targets)
        return left_ce * (1.0 - right_weight) + right_ce * right_weight


def aligned_box_iou(boxes, targets):
    """One IoU per aligned pair, without allocating a quadratic IoU matrix."""
    intersection = (torch.minimum(boxes[:, 2:], targets[:, 2:])
                    - torch.maximum(boxes[:, :2], targets[:, :2])).clamp(min=0).prod(dim=-1)
    area_boxes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=-1)
    area_targets = (targets[:, 2:] - targets[:, :2]).clamp(min=0).prod(dim=-1)
    return intersection / (area_boxes + area_targets - intersection).clamp(min=1e-7)


class DistributionRegressionHead(nn.Module):
    """Reuse the baseline regression tower; replace its two output branches."""

    def __init__(self, baseline_head):
        super().__init__()
        self.conv = baseline_head.conv
        self.reg_max = REG_MAX
        old_projection = baseline_head.bbox_reg
        if old_projection.out_channels != 4:
            raise ValueError("This adaptation requires one FCOS point per location")
        self.bbox_distribution = nn.Conv2d(old_projection.in_channels, 4 * (REG_MAX + 1),
                                           kernel_size=3, padding=1)
        nn.init.normal_(self.bbox_distribution.weight, std=0.01)
        nn.init.zeros_(self.bbox_distribution.bias)
        self.register_buffer("bins", torch.arange(REG_MAX + 1, dtype=torch.float32))

    def forward(self, features):
        all_logits = []
        for feature in features:
            logits = self.bbox_distribution(self.conv(feature))
            batch, _, height, width = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(batch, height * width, 4, REG_MAX + 1)
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=1)
        # Softmax, expectation, and geometry stay fp32 even with fp16 towers.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            distances = (logits.float().softmax(dim=-1) * self.bins.float()).sum(dim=-1)
        return distances, logits.flatten(start_dim=2)


class QualityFCOSHead(nn.Module):
    def __init__(self, baseline_head):
        super().__init__()
        self.classification_head = baseline_head.classification_head
        self.regression_head = DistributionRegressionHead(baseline_head.regression_head)
        self.box_coder = baseline_head.box_coder
        self.last_dfl_target_stats = None

    def forward(self, features):
        distances, distributions = self.regression_head(features)
        return {"cls_logits": self.classification_head(features),
                "bbox_regression": distances, "bbox_distribution": distributions}

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # FCOS.compute_loss remains inherited: it supplies unchanged assignments.
        device_type = head_outputs["cls_logits"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = head_outputs["cls_logits"].float()
            regression = head_outputs["bbox_regression"].float()
            distributions = head_outputs["bbox_distribution"].float().reshape(*regression.shape, REG_MAX + 1)
            labels, boxes = [], []
            for target, matches in zip(targets, matched_idxs):
                if len(target["labels"]):
                    image_labels = target["labels"][matches.clamp(min=0)].clone()
                    image_boxes = target["boxes"][matches.clamp(min=0)].float()
                else:
                    image_labels = matches.new_full(matches.shape, -1)
                    image_boxes = regression.new_zeros((len(matches), 4))
                image_labels[matches < 0] = -1
                labels.append(image_labels)
                boxes.append(image_boxes)
            labels, boxes, anchors = torch.stack(labels), torch.stack(boxes), torch.stack(anchors).float()
            foreground = labels >= 0
            count = int(foreground.sum())
            denominator = max(1, count)
            predictions = self.box_coder.decode(regression, anchors)
            positive_predictions, positive_boxes = predictions[foreground], boxes[foreground]
            quality_targets = torch.zeros_like(logits)
            if count:
                quality = aligned_box_iou(positive_predictions.detach(), positive_boxes).clamp(0, 1)
                quality_targets[foreground, labels[foreground]] = quality
                distance_targets = self.box_coder.encode(anchors[foreground], positive_boxes)
                outside = ((distance_targets < 0) | (distance_targets > REG_MAX)
                           | ~torch.isfinite(distance_targets))
                self.last_dfl_target_stats = {"positive_count": count,
                    "distance_count": int(distance_targets.numel()),
                    "minimum": float(distance_targets.min().detach()),
                    "maximum": float(distance_targets.max().detach()),
                    "out_of_support_count": int(outside.sum()), "clipped_count": 0,
                    "support": [0, REG_MAX]}
                dfl = distribution_focal_loss(distributions[foreground], distance_targets).mean(dim=-1).sum()
                giou = generalized_box_iou_loss(positive_predictions, positive_boxes, reduction="sum")
            else:
                self.last_dfl_target_stats = {"positive_count": 0, "distance_count": 0,
                    "minimum": None, "maximum": None, "out_of_support_count": 0,
                    "clipped_count": 0, "support": [0, REG_MAX]}
                # Preserve the regression graph on background-only minibatches.
                dfl, giou = distributions.sum() * 0.0, regression.sum() * 0.0
            return {"classification": quality_focal_loss(logits, quality_targets).sum() / denominator,
                    "bbox_regression": GIOU_WEIGHT * giou / denominator,
                    "bbox_distribution": DFL_WEIGHT * dfl / denominator}


class QualityFCOS(FCOS):
    """Keep FCOS forward/assignment and change only its inference score."""

    _sonar_quality_score_mode = "quality_sigmoid"

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        detections = []
        for index, image_shape in enumerate(image_shapes):
            image_boxes, image_scores, image_labels = [], [], []
            for logits, regression, level_anchors in zip(head_outputs["cls_logits"],
                                                         head_outputs["bbox_regression"], anchors[index]):
                class_count = logits.shape[-1]
                scores = logits[index].float().sigmoid().flatten()
                candidate_indices = torch.where(scores > self.score_thresh)[0]
                scores, order = scores[candidate_indices].topk(min(self.topk_candidates, candidate_indices.numel()))
                candidate_indices = candidate_indices[order]
                anchor_indices = torch.div(candidate_indices, class_count, rounding_mode="floor")
                boxes = self.box_coder.decode(regression[index][anchor_indices].float(),
                                              level_anchors[anchor_indices].float())
                image_boxes.append(box_ops.clip_boxes_to_image(boxes, image_shape))
                image_scores.append(scores)
                image_labels.append(candidate_indices % class_count)
            boxes, scores, labels = torch.cat(image_boxes), torch.cat(image_scores), torch.cat(image_labels)
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
            detections.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return detections


def build_model(variant, num_classes, seed, pretrained=True, image_size=640, cache_dir=None):
    """AAA returns the original builder's model without any adaptation."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    baseline = build_baseline("AAA", num_classes, seed, pretrained, image_size, cache_dir)
    if variant == "AAA":
        return baseline
    model = QualityFCOS(backbone=baseline.backbone, num_classes=num_classes,
                        anchor_generator=baseline.anchor_generator, head=QualityFCOSHead(baseline.head),
                        min_size=image_size, max_size=image_size,
                        center_sampling_radius=baseline.center_sampling_radius,
                        score_thresh=baseline.score_thresh, nms_thresh=baseline.nms_thresh,
                        detections_per_img=baseline.detections_per_img,
                        topk_candidates=baseline.topk_candidates)
    model.transform = baseline.transform
    model._sonar_pilot_contract = copy.deepcopy(baseline._sonar_pilot_contract)
    return model


def model_contract(model):
    contract = baseline_contract(model)
    quality = isinstance(model, QualityFCOS)
    contract.update({"variant": "QFL_DFL" if quality else "AAA",
                     "score_mode": "sigmoid_classification_iou_joint" if quality else "sqrt_sigmoid_classification_times_centerness",
                     "assignment": "inherited_torchvision_FCOS_unchanged",
                     "center_sampling_radius": model.center_sampling_radius,
                     "topk_candidates_per_level": model.topk_candidates,
                     "nms_threshold": model.nms_thresh})
    if quality:
        contract["quality"] = {**QUALITY_PARAMETERS,
            "classification_targets": "detached_predicted_box_IoU_at_assigned_class; all_other_classes_zero",
            "centerness_branch": False, "loss_arithmetic": "float32",
            "bbox_representation": "four_stride_normalized_distributions_with_17_bins_each",
            "dfl_reduction": "mean_over_four_sides_then_sum_over_positives_divide_positive_count",
            "regression_weighting": "uniform_positive_weights_as_baseline; no_confidence_weighting",
            "out_of_support_policy": "fail_without_clipping",
            "new_projection_initialization": "normal_std_0.01_weight_zero_bias",
            "preserved_initial_tensors": ["backbone", "fpn", "classification_head", "regression_head.conv"],
            "reference": "https://arxiv.org/abs/2006.04388",
            "method_scope": "QFL_and_DFL_adapted_to_existing_FCOS_not_full_GFL_reproduction"}
    return contract
