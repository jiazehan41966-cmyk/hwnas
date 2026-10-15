"""Stage7-only adapter for the SonarOp V0.2-P0 controlled experiment."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from .sonar_dw_reparam import ReparamDualRFDepthwise


SONAROP_VERSION = "v0.2-p0"
A3_ID = "P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7SonarOp"
A1_ARCHITECTURE = dict(output_combination="P3_P4_P5", kernels={
    "stage1": 5, "stage2": 3, "stage4": 3, "stage5": 3, "stage6": 5, "stage7": 5,
})
A2_ARCHITECTURE = dict(output_combination="P3_P4_P5", kernels={
    "stage1": 5, "stage2": 3, "stage4": 3, "stage5": 3, "stage6": 5, "stage7": 3,
})


class SonarOpCandidateDetector(nn.Module):
    """Thin wrapper preserving the A1 detector API and replacing only stage7 DW."""

    def __init__(self, *, deploy: bool = False) -> None:
        super().__init__()
        self.detector = build_sonar_candidate(SPEC_PATH, deepcopy(A1_ARCHITECTURE))
        block = self.detector.backbone.stages[7][0]
        channels = block.dw_conv.in_channels
        stride = block.dw_conv.stride[0]
        if channels != 480 or stride != 1:
            raise ValueError(f"Unexpected stage7 depthwise contract C={channels}, stride={stride}")
        block.dw_conv = ReparamDualRFDepthwise(channels, stride, deploy=deploy)
        block.dw_bn = nn.Identity()
        block.dw_relu = nn.Identity()
        self.sonarop_version = SONAROP_VERSION
        self.candidate_architecture = dict(
            output_combination="P3_P4_P5",
            kernels={k: v for k, v in A1_ARCHITECTURE["kernels"].items() if k != "stage7"},
            stage7_operator="sonarop_v0.2-p0",
        )

    @property
    def backbone(self):
        return self.detector.backbone

    @property
    def neck(self):
        return self.detector.neck

    @property
    def head(self):
        return self.detector.head

    @property
    def levels(self):
        return self.detector.levels

    @property
    def feature_strides(self):
        return self.detector.feature_strides

    @property
    def anchor_sizes(self):
        return self.detector.anchor_sizes

    def loss_config(self, baseline_config):
        return self.detector.loss_config(baseline_config)

    def forward(self, x: torch.Tensor):
        return self.detector(x)

    def sonarop(self) -> ReparamDualRFDepthwise:
        module = self.detector.backbone.stages[7][0].dw_conv
        if not isinstance(module, ReparamDualRFDepthwise):
            raise TypeError("stage7 no longer contains the SonarOp prototype")
        return module

    def switch_to_deploy(self) -> "SonarOpCandidateDetector":
        self.eval()
        self.sonarop().switch_to_deploy()
        return self


def build_sonarop_candidate(*, deploy: bool = False) -> SonarOpCandidateDetector:
    return SonarOpCandidateDetector(deploy=deploy)


@torch.no_grad()
def switch_candidate_to_deploy(model: SonarOpCandidateDetector) -> SonarOpCandidateDetector:
    return model.switch_to_deploy()


@torch.no_grad()
def fold_standard_stage7_depthwise(model: nn.Module) -> nn.Module:
    """Fold only the standard stage7 DW+BN for A1/A2 deploy-cost comparison."""
    model.eval()
    block = model.backbone.stages[7][0]
    conv, bn = block.dw_conv, block.dw_bn
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("expected an unfused standard stage7 Conv2d+BatchNorm2d")
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    conv_bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
    bias = bn.bias + (conv_bias - bn.running_mean) * scale
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True, device=weight.device, dtype=weight.dtype)
    fused.weight.copy_(weight); fused.bias.copy_(bias)
    block.dw_conv = fused; block.dw_bn = nn.Identity()
    return model
