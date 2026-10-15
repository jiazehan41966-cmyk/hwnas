"""Phase 4A Sonar Operator Probe Pool P0.

This module is intentionally outside frozen X_V0.1.  It only replaces the
spatial/depthwise operator inside one MBConv block of the D all-k3 detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.models.builder import MBConvBlock
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate


D_ARCHITECTURE_ID = "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3"
D_ARCHITECTURE = {
    "output_combination": "P3_P4_P5",
    "kernels": {"stage1": 3, "stage2": 3, "stage4": 3, "stage5": 3, "stage6": 3, "stage7": 3},
}
P0_STAGES = (1, 2, 5, 7)
P0_OPERATORS = ("DW3", "DW5", "DilatedDW3", "DirectionalDW", "DCN", "FrequencyLite")
SCREENING_OPERATORS = tuple(op for op in P0_OPERATORS if op != "DW3")


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    hypothesis: str
    formula: str
    deploy_note: str
    geometry_adaptive: bool = False


OPERATOR_DEFINITIONS: dict[str, OperatorDefinition] = {
    "DW3": OperatorDefinition(
        name="DW3",
        hypothesis="baseline local receptive field control",
        formula="depthwise Conv2d(k=3,p=1,d=1,stride=s)",
        deploy_note="existing MBConv depthwise 3x3 primitive",
    ),
    "DW5": OperatorDefinition(
        name="DW5",
        hypothesis="larger local receptive field",
        formula="depthwise Conv2d(k=5,p=2,d=1,stride=s)",
        deploy_note="existing V0.1 MBConv depthwise 5x5 primitive",
    ),
    "DilatedDW3": OperatorDefinition(
        name="DilatedDW3",
        hypothesis="5x5 effective receptive field with sparse 3x3 weights",
        formula="depthwise Conv2d(k=3,p=2,d=2,stride=s)",
        deploy_note="software probe; hardware support not assumed",
    ),
    "DirectionalDW": OperatorDefinition(
        name="DirectionalDW",
        hypothesis="axis-asymmetric receptive field probe",
        formula="0.5 * (DWConv(1x5,p=(0,2),stride=s) + DWConv(5x1,p=(2,0),stride=s))",
        deploy_note="can be algebraically represented as one sparse 5x5 depthwise kernel after branch fusion",
    ),
    "DCN": OperatorDefinition(
        name="DCN",
        hypothesis="GEOMETRY_ADAPTIVE_PROBE",
        formula="per-channel offset branch Conv3x3 -> depthwise DeformConv2d(k=3,p=1,stride=s,groups=C)",
        deploy_note="software-only probe; no FPGA primitive admitted",
        geometry_adaptive=True,
    ),
    "FrequencyLite": OperatorDefinition(
        name="FrequencyLite",
        hypothesis="minimal explicit low/high-frequency decomposition",
        formula="fixed low=avg3x3(x,stride=s) + learnable DW3(high=x-avg3x3(x),stride=s)",
        deploy_note="software-only probe; not FFT/DWT/DCT and not hardware-admitted",
    ),
}


def parse_p0_architecture_id(value: str) -> tuple[str, int]:
    """Parse IDs such as ``D_P0_DCN_stage5``."""
    parts = str(value).split("_")
    if len(parts) != 4 or parts[0] != "D" or parts[1] != "P0" or not parts[3].startswith("stage"):
        raise ValueError(f"invalid P0 architecture ID: {value!r}")
    operator = parts[2]
    stage = int(parts[3].replace("stage", ""))
    validate_probe(operator, stage)
    return operator, stage


def p0_architecture_id(operator: str, stage: int) -> str:
    validate_probe(operator, stage)
    return f"D_P0_{operator}_stage{int(stage)}"


def validate_probe(operator: str, stage: int) -> None:
    if operator not in P0_OPERATORS:
        raise ValueError(f"unsupported P0 operator {operator!r}; allowed={P0_OPERATORS}")
    if int(stage) not in P0_STAGES:
        raise ValueError(f"unsupported P0 stage {stage!r}; allowed={P0_STAGES}")


class DirectionalDepthwise(nn.Module):
    """Fixed 1x5 and 5x1 depthwise branches fused by arithmetic mean."""

    def __init__(self, channels: int, stride: int | tuple[int, int]) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stride = stride
        self.dw_1x5 = nn.Conv2d(channels, channels, kernel_size=(1, 5), stride=stride, padding=(0, 2), groups=channels, bias=False)
        self.dw_5x1 = nn.Conv2d(channels, channels, kernel_size=(5, 1), stride=stride, padding=(2, 0), groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.dw_1x5(x) + self.dw_5x1(x))


class DeformableDepthwise3x3(nn.Module):
    """Depthwise DCN probe with a per-channel offset branch initialized to zero."""

    def __init__(self, channels: int, stride: int | tuple[int, int]) -> None:
        super().__init__()
        try:
            from torchvision.ops import DeformConv2d
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("DCN probe requires torchvision.ops.DeformConv2d") from exc
        self.channels = int(channels)
        self.stride = stride
        self.offset = nn.Conv2d(channels, 2 * channels * 3 * 3, kernel_size=3, stride=stride, padding=1, groups=channels, bias=True)
        self.deform = DeformConv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deform(x, self.offset(x))


class FrequencyLiteDepthwise(nn.Module):
    """Minimal fixed low-pass plus learnable high-frequency depthwise probe."""

    def __init__(self, channels: int, stride: int | tuple[int, int]) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stride = stride
        self.high_dw = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)
        low_kernel = torch.full((channels, 1, 3, 3), 1.0 / 9.0, dtype=torch.float32)
        self.register_buffer("low_kernel", low_kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = F.conv2d(x, self.low_kernel, bias=None, stride=1, padding=1, groups=self.channels)
        high = x - smooth
        low_out = F.conv2d(x, self.low_kernel, bias=None, stride=self.stride, padding=1, groups=self.channels)
        return low_out + self.high_dw(high)


def _single_stride(value: int | tuple[int, int]) -> int | tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"P0 expects square stride, got {value}")
        return int(value[0])
    return int(value)


def make_spatial_operator(operator: str, reference: nn.Conv2d) -> nn.Module:
    """Construct the P0 replacement for an existing MBConv depthwise conv."""
    channels = int(reference.in_channels)
    if reference.in_channels != reference.out_channels or reference.groups != reference.in_channels:
        raise ValueError("P0 replacement requires a depthwise Conv2d reference")
    stride = _single_stride(reference.stride)
    if operator == "DW3":
        return reference
    if operator == "DW5":
        return nn.Conv2d(channels, channels, kernel_size=5, stride=stride, padding=2, groups=channels, bias=False)
    if operator == "DilatedDW3":
        return nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=2, dilation=2, groups=channels, bias=False)
    if operator == "DirectionalDW":
        return DirectionalDepthwise(channels, stride)
    if operator == "DCN":
        return DeformableDepthwise3x3(channels, stride)
    if operator == "FrequencyLite":
        return FrequencyLiteDepthwise(channels, stride)
    raise ValueError(operator)


def get_stage_block(model: nn.Module, stage: int) -> MBConvBlock:
    stage = int(stage)
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "stages"):
        raise ValueError("P0 model must be a SonarCandidateDetector-like detector")
    try:
        stage_blocks = model.backbone.stages[stage]
        block = stage_blocks[0]
    except Exception as exc:
        raise ValueError(f"cannot access stage{stage} block") from exc
    if not isinstance(block, MBConvBlock):
        raise ValueError(f"stage{stage} is not an MBConvBlock and cannot host P0 operators")
    return block


def apply_p0_operator(model: nn.Module, operator: str, stage: int) -> nn.Module:
    """Mutate ``model`` by replacing exactly one MBConv depthwise spatial operator."""
    validate_probe(operator, stage)
    if operator == "DW3":
        model.p0_probe = {"operator": "DW3", "stage": int(stage), "replacement": "identity_noop"}
        return model
    block = get_stage_block(model, stage)
    block.dw_conv = make_spatial_operator(operator, block.dw_conv)
    model.p0_probe = {"operator": operator, "stage": int(stage), "replacement": "mbconv_dw_conv"}
    return model


def build_d_baseline(*, num_classes: int = 2, image_size: int = 640) -> nn.Module:
    return build_sonar_candidate(SPEC_PATH, D_ARCHITECTURE, num_classes=num_classes, image_size=image_size)


def build_p0_candidate(operator: str, stage: int, *, num_classes: int = 2, image_size: int = 640) -> nn.Module:
    model = build_d_baseline(num_classes=num_classes, image_size=image_size)
    return apply_p0_operator(model, operator, stage)


def operator_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_info_with_p0_ops(model: nn.Module, input_shape: tuple[int, ...] = (1, 1, 640, 640)) -> dict[str, Any]:
    """Conv/Linear/DeformConv2d MAC estimate for P0 diagnostics; not latency."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = 0
    deform_cls = None
    try:
        from torchvision.ops import DeformConv2d
        deform_cls = DeformConv2d
    except Exception:
        deform_cls = None

    def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += int(kernel_ops * out_h * out_w * module.out_channels)
        elif isinstance(module, nn.Linear):
            macs += int(module.in_features * module.out_features)
        elif deform_cls is not None and isinstance(module, deform_cls):
            out_h, out_w = output.shape[2], output.shape[3]
            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            groups = int(module.groups)
            kernel_ops = kh * kw * module.in_channels // groups
            macs += int(kernel_ops * out_h * out_w * module.out_channels)

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) or (deform_cls is not None and isinstance(module, deform_cls)):
            hooks.append(module.register_forward_hook(hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))
    for item in hooks:
        item.remove()
    model.train(was_training)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": float(total_params * 4 / (1024 * 1024)),
        "estimated_macs": int(macs),
        "estimated_FLOPs_2x_MACs": int(2 * macs),
        "input_shape": list(input_shape),
        "scope": "Conv2d/Linear/torchvision_DeformConv2d estimate; MACs are not latency",
    }
