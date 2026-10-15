"""Isolated P3 fusion comparison on the original A1 detector structure.

All ordinary A1 backbone, lateral/output convolutions and detection head remain
trainable and retain their parameter paths.  ``equal_add`` preserves the original
FPN expression.  ``weighted`` changes only P3 to L3 + alpha * nearest(M4), with a
single direct nonnegative scalar initialized to exactly one without using RNG.
Call the projection after every optimizer step, or install its optimizer hook.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer

from hwnas_fpga.models.detection import FixedFPNNeck
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, SonarCandidateDetector, build_sonar_candidate


FusionVariant = Literal["equal_add", "weighted"]
A1_ARCHITECTURE_ID = "P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k5"
# Exact A1 kernel contract, independently checked against the original A1 in
# tests.  Keep this module usable with the pre-SonarOp frozen source bundle.
A1_ARCHITECTURE = {
    "output_combination": "P3_P4_P5",
    "kernels": {"stage1": 5, "stage2": 3, "stage4": 3,
                "stage5": 3, "stage6": 5, "stage7": 5},
}


class P3FusionNeck(nn.Module):
    """Reuse an existing A1 FPN, adding no convolution or random initialization."""

    def __init__(self, original: FixedFPNNeck, *, variant: FusionVariant) -> None:
        super().__init__()
        if variant not in ("equal_add", "weighted"):
            raise ValueError("variant must be 'equal_add' or 'weighted'")
        if (tuple(original.in_channels) != (32, 96, 320)
                or original.out_channels != 96
                or len(original.lateral_convs) != 3 or len(original.output_convs) != 3):
            raise ValueError("P3 fusion comparison requires the fixed A1 P3/P4/P5 neck")
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.lateral_convs = original.lateral_convs
        self.output_convs = original.output_convs
        self.variant = variant
        self.levels = ("P3", "P4", "P5")
        self.training = original.training
        # new_ones uses no random numbers and follows the original module dtype
        # and device.  The control buffer is absent from optimizer parameters.
        scalar = self.lateral_convs[0][0].weight.new_ones(())
        if variant == "weighted":
            self.alpha = nn.Parameter(scalar)
        else:
            self.register_buffer("alpha", scalar)

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(features)}")
        lateral = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        fused = list(lateral)
        for index in range(len(fused) - 2, -1, -1):
            deep = F.interpolate(fused[index + 1], size=fused[index].shape[-2:], mode="nearest")
            if self.variant == "weighted" and index == 0:
                fused[index] = fused[index] + self.alpha * deep
            else:
                fused[index] = fused[index] + deep
        return tuple(conv(feature) for conv, feature in zip(self.output_convs, fused))

    @torch.no_grad()
    def project_nonnegative_(self) -> None:
        """Project the direct scalar after a step; never modify ordinary weights."""
        if self.variant == "weighted":
            if not bool(torch.isfinite(self.alpha)):
                raise FloatingPointError("P3 fusion alpha is nonfinite")
            self.alpha.clamp_(min=0)


def build_p3_fusion_candidate(
    variant: FusionVariant = "equal_add", *, num_classes: int = 2, image_size: int = 640,
) -> SonarCandidateDetector:
    """Build the original A1, then reuse its FPN modules for the paired variant.

    Restrict the experiment to the original 640-pixel, two-class contract.  The
    builder intentionally passes no newer adapter kwargs, allowing the module
    to run against the frozen original remote ``build_sonar_candidate`` API.
    Calling with the same initial RNG state yields identical ordinary parameter
    values and the same final RNG state for original A1 and both variants.
    """
    if variant not in ("equal_add", "weighted"):
        raise ValueError("variant must be 'equal_add' or 'weighted'")
    if num_classes != 2 or image_size != 640:
        raise ValueError("P3 fusion comparison is fixed to num_classes=2 and image_size=640")
    model = build_sonar_candidate(SPEC_PATH, deepcopy(A1_ARCHITECTURE))
    if tuple(model.levels) != ("P3", "P4", "P5"):
        raise ValueError("P3 fusion comparison requires A1 P3/P4/P5 output levels")
    model.neck = P3FusionNeck(model.neck, variant=variant)
    model.p3_fusion_variant = variant
    model.p3_fusion_config = {
        "backbone_architecture_id": A1_ARCHITECTURE_ID,
        "variant": variant,
        "weighted_level": "P3",
        "formula": "M3=L3+alpha*nearest(M4)" if variant == "weighted" else "M3=L3+nearest(M4)",
        "alpha_initial": 1.0,
        "alpha_constraint": "project_nonnegative_after_each_optimizer_step",
        "ordinary_parameters_trainable": True,
    }
    return model


def _neck(model: nn.Module) -> P3FusionNeck:
    neck = model if isinstance(model, P3FusionNeck) else getattr(model, "neck", None)
    if not isinstance(neck, P3FusionNeck):
        raise TypeError("expected a P3FusionNeck or a detector built by build_p3_fusion_candidate")
    return neck


@torch.no_grad()
def project_nonnegative_alpha_(model: nn.Module) -> None:
    """Project only alpha; safe as an optimizer post-step hook returning None."""
    _neck(model).project_nonnegative_()


def p3_fusion_parameter_groups(
    model: nn.Module, *, weight_decay: float = 1e-4,
) -> list[dict]:
    """Preserve ordinary AdamW decay and give the single alpha no decay.

    This helper changes neither trainability nor learning rate.  All ordinary
    parameters remain in one group with the original decay; the weighted variant
    adds exactly one scalar group with weight_decay=0.  No parameter is duplicated.
    """
    weight_decay = float(weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    neck = _neck(model)
    alpha = neck.alpha if isinstance(neck.alpha, nn.Parameter) else None
    ordinary = [parameter for parameter in model.parameters()
                if parameter.requires_grad and parameter is not alpha]
    groups = [{"name": "ordinary_model", "params": ordinary, "weight_decay": weight_decay}]
    if alpha is not None and alpha.requires_grad:
        groups.append({"name": "p3_alpha", "params": [alpha], "weight_decay": 0.0})
    return groups


def register_p3_alpha_projection(optimizer: Optimizer, model: nn.Module):
    """Install a removable post-step hook without editing the frozen trainer."""
    _neck(model)

    def after_step(_optimizer, _args, _kwargs) -> None:
        project_nonnegative_alpha_(model)

    return optimizer.register_step_post_hook(after_step)


__all__ = [
    "A1_ARCHITECTURE_ID", "P3FusionNeck", "build_p3_fusion_candidate",
    "project_nonnegative_alpha_", "p3_fusion_parameter_groups", "register_p3_alpha_projection",
]
