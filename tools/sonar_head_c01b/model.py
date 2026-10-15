"""Isolated C01b: paired E0 quality head with direct distances and GIoU.

C01 actually uses 16 bins per edge (66 raw channels per level), CIoU and
DFL. This is a combined representation/loss change, not a single-loss DFL
ablation. The frozen C01 adapter and pinned author source remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from tools.detector_elastic_pilot.model import _hash_named_tensors
from tools.sonar_head_factorial.model import (
    DEFAULT_SOURCE_ROOT, FactorialDetector, batch_loss, decode,
    loss_from_output,
)
from tools.sonar_yolov9_baseline.author_adapter import _validate_assigner_environment

SEEDS = (42, 43, 44)
DIRECT_INITIALIZATION_POLICY = "C01B_DIRECT_DISTANCE_FINAL_PREDICTION_CPU_SEED_V1"
DISTANCE_ACTIVATION = "ReLU(ltrb), identical in training and inference"


def _shared_head_tensors(head):
    """Only tensors whose semantic roles and shapes are unchanged from C01."""
    excluded = tuple(f"cv2.{level}.2." for level in range(head.nl)) + ("dfl.",)
    return [(name, tensor) for name, tensor in head.state_dict().items()
            if not name.startswith(excluded)]


def _seed_final(seed: int, level: str) -> None:
    value = f"{DIRECT_INITIALIZATION_POLICY}:{seed}:{level}".encode()
    digest = hashlib.sha256(value).digest()
    torch.random.default_generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


class DirectDistanceHead(nn.Module):
    """Keep all C01 hidden towers and cls weights; predict four distances.

    In particular the regression hidden width remains 64, including the
    grouped second 3x3 layer. Setting reg_max=1 in a fresh DDetect constructor
    would shrink that hidden width and would change more than the output layer.
    """

    def __init__(self, reference_head, seed: int, levels):
        super().__init__()
        if reference_head.reg_max != 16 or reference_head.nc != 2:
            raise ValueError("C01b must be paired with the frozen 16-bin, two-class C01")
        self.nc, self.nl = reference_head.nc, reference_head.nl
        self.reg_max, self.no = 1, 4 + self.nc
        self.inplace = reference_head.inplace
        self.dynamic, self.export, self.shape = False, False, None
        self.stride = reference_head.stride.clone()
        self.anchors, self.strides = torch.empty(0), torch.empty(0)
        self.cv2, self.cv3 = deepcopy(reference_head.cv2), deepcopy(reference_head.cv3)
        for index, (tower, level) in enumerate(zip(self.cv2, levels)):
            old_final = tower[-1]
            if (old_final.in_channels != 64 or old_final.out_channels != 64
                    or old_final.groups != 4 or old_final.kernel_size != (1, 1)):
                raise ValueError(f"Unexpected C01 regression predictor at level {index}")
            _seed_final(seed, level)
            tower[-1] = nn.Conv2d(64, 4, 1, groups=4, bias=True)
            with torch.no_grad():
                # The same author distance-bias rule, now for four direct values.
                tower[-1].bias.fill_(1.0)
        self.dfl = nn.Identity()
        self.distance_activation = nn.ReLU(inplace=False)

    def forward(self, features):
        if len(features) != self.nl:
            raise ValueError("Head/feature level count differs")
        shape = features[0].shape
        raw = [torch.cat((self.distance_activation(self.cv2[index](feature)),
                          self.cv3[index](feature)), dim=1)
               for index, feature in enumerate(features)]
        if self.training:
            return raw
        geometry = importlib.import_module("utils.tal.anchor_generator")
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                value.transpose(0, 1) for value in geometry.make_anchors(raw, self.stride, 0.5))
            self.shape = shape
        box, cls = torch.cat([tensor.view(shape[0], self.no, -1) for tensor in raw], dim=2).split((4, self.nc), dim=1)
        decoded_box = geometry.dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        decoded = torch.cat((decoded_box, cls.sigmoid()), dim=1)
        return decoded if self.export else (decoded, raw)


class DirectGIoUBboxLoss(nn.Module):
    """The author's TAL score weighting/reduction, with only GIoU regression."""

    def __init__(self):
        super().__init__()
        self.reg_max, self.use_dfl = 0, False

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        del anchor_points
        bbox_mask = fg_mask.unsqueeze(-1).expand(-1, -1, 4)
        predicted = torch.masked_select(pred_bboxes, bbox_mask).reshape(-1, 4)
        target = torch.masked_select(target_bboxes, bbox_mask).reshape(-1, 4)
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        bbox_iou = importlib.import_module("utils.metrics").bbox_iou
        giou = bbox_iou(predicted, target, xywh=False, GIoU=True)
        loss = ((1.0 - giou) * weight).sum() / target_scores_sum
        return loss, pred_dist.new_zeros(()), giou


class C01bDetector(FactorialDetector):
    def __init__(self, seed: int, source_root: Path):
        # Construct exactly the C01 same-seed state before replacing any layer.
        super().__init__("E0", "quality_aware", seed, source_root)
        self._c01_initialization = self.initialization_identity()
        self._c01_head_shared_sha256 = _hash_named_tensors(_shared_head_tensors(self.head))
        self.head = DirectDistanceHead(self.head, seed, self.levels)
        self.hyp["dfl"] = 0.0
        self._initialization["head_sha256"] = _hash_named_tensors(self.head.state_dict().items())
        shared = _hash_named_tensors(_shared_head_tensors(self.head))
        if shared != self._c01_head_shared_sha256:
            raise ValueError("C01b changed a paired shared head tensor")
        self._initialization.update(
            direct_prediction_policy=DIRECT_INITIALIZATION_POLICY,
            shared_head_sha256=shared,
            c01_shared_head_sha256=self._c01_head_shared_sha256,
            c01_reference=deepcopy(self._c01_initialization),
            paired_components=["backbone", "neck", "head.cv2 hidden layers", "head.cv3 classification"],
            changed_components=["head.cv2 final grouped 1x1", "distribution expectation to ReLU direct distances",
                                "regression CIoU plus DFL to GIoU only"],
        )

    def metadata(self):
        result = super().metadata()
        result.update(
            adapter="C01b_E0_quality_direct_GIoU",
            c01_reference=dict(reg_max=16, raw_channels_per_level=66, box="CIoU", distribution="DFL"),
            raw_channels_per_level=6,
            regression_prediction=dict(input_channels=64, output_channels=4, kernel=1, groups=4),
            regression_distance_activation=DISTANCE_ACTIVATION,
            factor_scope="direct-distance/ReLU/no-DFL plus CIoU-to-GIoU combined engineering adaptation",
            head_capacity="same C01 regression hidden64, classification hidden48 and four native E0 levels",
            adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        result["quality_package"] = {
            "head": "isolated direct-distance head with C01 shared tower initialization",
            "assignment": {"topk": 10, "alpha": 0.5, "beta": 6.0},
            "classification": "BCEWithLogitsLoss against TAL quality soft targets",
            "box": "GIoU", "distribution": "disabled", "reg_max": 1,
            "distance_activation": DISTANCE_ACTIVATION,
            "gains": {"box": 7.5, "cls": 0.5, "dfl": 0.0},
            "reduction": "native sum(box,cls) * batch_size",
            "no_varifocal_loss": True, "no_auxiliary_branch_or_PGI": True,
        }
        return result


def build_model(seed: int, *, source_root: str | Path | None = None):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("C01b seed must be an integer from 42 through 44")
    with torch.random.fork_rng(devices=[]):
        return C01bDetector(seed, Path(source_root or DEFAULT_SOURCE_ROOT).resolve())


def build_criterion(model: C01bDetector, baseline: Mapping[str, Any]):
    del baseline  # All C01 quality losses use the pinned package's own gains.
    if not isinstance(model, C01bDetector) or model.head.reg_max != 1 or model.head.no != 6:
        raise ValueError("C01b criterion requires the direct-distance E0 adapter")
    _validate_assigner_environment()
    criterion = importlib.import_module("utils.loss_tal").ComputeLoss(model, use_dfl=False)
    criterion.bbox_loss = DirectGIoUBboxLoss().to(next(model.parameters()).device)
    return criterion


def criterion_metadata(model, criterion):
    if criterion.use_dfl or not isinstance(criterion.bbox_loss, DirectGIoUBboxLoss):
        raise ValueError("C01b criterion has drifted from GIoU-only direct regression")
    return model.metadata()["quality_package"]
