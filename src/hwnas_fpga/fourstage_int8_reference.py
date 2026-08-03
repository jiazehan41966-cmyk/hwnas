"""Activation-calibrated INT8 reference for frozen four-stage candidates.

The implementation is intentionally narrow.  It supports the mature fixed
four-stage deployment candidates used in Protocol V2:

* ConvBlock stem / Stage1
* MBConvBlock Stage2 / Stage3 / mature Stage4 K3/K5
* Stage4 SkipBlock identity
* GAP -> FC8 head

Dir-MBConv3 and other research blocks are rejected here because Dir-v1 failed
the accuracy gate and must not proceed into the full-network deployment chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn

from hwnas_fpga.deploy.fixed_point import (
    avg_pool2d_int_reference,
    conv2d_int_reference,
    linear_int_reference,
    quantize_bias_int32,
    quantize_symmetric_int8,
    requantize_per_output_int8,
    requantize_int8,
    residual_add_int_reference,
    scale_to_multiplier,
)
from hwnas_fpga.fourstage_operator import validate_frozen_fourstage
from hwnas_fpga.models import build_model
from hwnas_fpga.models.builder import ConvBlock, HeadBlock, MBConvBlock, SkipBlock
from hwnas_fpga.search_space import ArchitectureSpec


MIN_SCALE = 1.0 / (127.0 * 1024.0)
CALIBRATION_CONTRACT = "fourstage_per_tensor_symmetric_int8_activation_v1"
SUPPORTED_BLOCKS = (ConvBlock, MBConvBlock, SkipBlock)


@dataclass(frozen=True)
class QuantizedConv:
    weight: torch.Tensor
    bias: torch.Tensor
    weight_scale: float


@dataclass(frozen=True)
class QuantizedLinear:
    weight: torch.Tensor
    bias: torch.Tensor | None
    weight_scale: float


def scale_from_absmax(max_abs: float) -> float:
    if not math.isfinite(float(max_abs)):
        raise ValueError(f"non-finite activation range: {max_abs}")
    return max(float(max_abs), MIN_SCALE) / 127.0


def _update_max_abs(stats: dict[str, float], name: str, tensor: torch.Tensor) -> None:
    value = float(tensor.detach().abs().max().cpu().item()) if tensor.numel() else 0.0
    stats[name] = max(stats.get(name, 0.0), value)


def _hooked_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, ConvBlock):
            modules[name] = module
        elif isinstance(module, MBConvBlock):
            modules[name] = module
            if module.use_expand:
                modules[f"{name}.expand_relu"] = module.expand_relu
            modules[f"{name}.dw_relu"] = module.dw_relu
        elif isinstance(module, SkipBlock):
            modules[name] = module
        elif isinstance(module, HeadBlock):
            if module.conv_head is not None or hasattr(module, "fc1"):
                raise ValueError("four-stage Protocol V2 requires GAP->FC8 only")
            modules[f"{name}.gap"] = module.gap
            modules[f"{name}.fc"] = module.fc
    return modules


def validate_reference_supported_model(model: nn.Module) -> None:
    architecture = getattr(model, "architecture", None)
    if not isinstance(architecture, ArchitectureSpec):
        raise ValueError("model must carry an ArchitectureSpec")
    validate_frozen_fourstage(architecture)
    for stage in model.stages:
        for block in stage:
            if not isinstance(block, SUPPORTED_BLOCKS):
                raise ValueError(
                    "unsupported block for mature four-stage deployment "
                    f"reference: {block.__class__.__name__}"
                )


def collect_activation_stats(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor | int]],
    *,
    device: str = "cpu",
    max_samples: int = 64,
) -> dict[str, Any]:
    """Collect real activation ranges from a training-only calibration loader."""

    validate_reference_supported_model(model)
    model.to(device)
    model.eval()
    stats: dict[str, float] = {}
    samples_seen = 0
    handles = []

    for name, module in _hooked_modules(model).items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: _update_max_abs(
                    stats, key, output if isinstance(output, torch.Tensor) else output[0]
                )
            )
        )

    try:
        with torch.no_grad():
            for inputs, _labels in loader:
                if samples_seen >= max_samples:
                    break
                remaining = max_samples - samples_seen
                if int(inputs.shape[0]) > remaining:
                    inputs = inputs[:remaining]
                inputs = inputs.to(device)
                _update_max_abs(stats, "input", inputs)
                model(inputs)
                samples_seen += int(inputs.shape[0])
    finally:
        for handle in handles:
            handle.remove()

    if samples_seen <= 0:
        raise ValueError("activation calibration saw zero samples")

    scales = {name: scale_from_absmax(value) for name, value in sorted(stats.items())}
    return {
        "schema_version": 1,
        "contract": CALIBRATION_CONTRACT,
        "samples_seen": samples_seen,
        "max_abs": {name: float(value) for name, value in sorted(stats.items())},
        "scales": scales,
        "zero_point": 0,
        "activation_range": [-127, 127],
        "rounding": "torch_round_ties_to_even",
        "claim_boundary": (
            "Training-only activation calibration for the Python INT8 "
            "reference. It is not HLS parity, route, COM5, or power evidence."
        ),
    }


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight.detach().to(torch.float32)
    bias = (
        torch.zeros(conv.out_channels, dtype=torch.float32, device=weight.device)
        if conv.bias is None
        else conv.bias.detach().to(torch.float32)
    )
    bn_weight = bn.weight.detach().to(torch.float32)
    bn_bias = bn.bias.detach().to(torch.float32)
    running_mean = bn.running_mean.detach().to(torch.float32)
    running_var = bn.running_var.detach().to(torch.float32)
    multiplier = bn_weight / torch.sqrt(running_var + float(bn.eps))
    folded_weight = weight * multiplier.reshape(-1, 1, 1, 1)
    folded_bias = bn_bias + (bias - running_mean) * multiplier
    return folded_weight.cpu(), folded_bias.cpu()


def _quantized_conv(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    *,
    input_scale: float,
) -> QuantizedConv:
    folded_weight, folded_bias = _fold_conv_bn(conv, bn)
    max_abs = float(folded_weight.abs().max().item()) if folded_weight.numel() else 0.0
    weight_scale = scale_from_absmax(max_abs)
    weight_q = quantize_symmetric_int8(folded_weight, scale=weight_scale).cpu()
    bias_q = quantize_bias_int32(
        folded_bias,
        input_scale=float(input_scale),
        weight_scale=float(weight_scale),
    ).cpu()
    return QuantizedConv(weight=weight_q, bias=bias_q, weight_scale=float(weight_scale))


def _quantized_linear(linear: nn.Linear, *, input_scale: float) -> QuantizedLinear:
    weight = linear.weight.detach().to(torch.float32).cpu()
    max_abs = float(weight.abs().max().item()) if weight.numel() else 0.0
    weight_scale = scale_from_absmax(max_abs)
    weight_q = quantize_symmetric_int8(weight, scale=weight_scale).cpu()
    bias_q = None
    if linear.bias is not None:
        bias_q = quantize_bias_int32(
            linear.bias.detach().to(torch.float32).cpu(),
            input_scale=float(input_scale),
            weight_scale=float(weight_scale),
        ).cpu()
    return QuantizedLinear(weight=weight_q, bias=bias_q, weight_scale=float(weight_scale))


def _conv_bn_reference(
    inputs: torch.Tensor,
    *,
    input_scale: float,
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    output_scale: float,
    activation: str | None,
) -> torch.Tensor:
    qconv = _quantized_conv(conv, bn, input_scale=input_scale)
    padding = conv.padding[0]
    if tuple(conv.padding) != (padding, padding):
        raise ValueError("only symmetric square padding is supported")
    stride = conv.stride[0]
    if tuple(conv.stride) != (stride, stride):
        raise ValueError("only symmetric square stride is supported")
    acc = conv2d_int_reference(
        inputs.cpu(),
        qconv.weight,
        qconv.bias,
        stride=int(stride),
        padding=int(padding),
        groups=int(conv.groups),
    )
    output = requantize_per_output_int8(
        acc,
        input_scale=float(input_scale),
        weight_scale=float(qconv.weight_scale),
        output_scale=float(output_scale),
        channel_axis=1,
    )
    if activation == "relu":
        output = output.to(torch.int16).clamp(0, 127).to(torch.int8)
    elif activation == "relu6":
        upper = min(127, max(0, int(round(6.0 / float(output_scale)))))
        output = output.to(torch.int16).clamp(0, upper).to(torch.int8)
    elif activation is not None:
        raise ValueError(f"unsupported activation: {activation}")
    return output


def _linear_reference(
    inputs: torch.Tensor,
    *,
    input_scale: float,
    linear: nn.Linear,
    output_scale: float,
) -> torch.Tensor:
    qlinear = _quantized_linear(linear, input_scale=input_scale)
    acc = linear_int_reference(inputs.cpu(), qlinear.weight, qlinear.bias)
    return requantize_per_output_int8(
        acc,
        input_scale=float(input_scale),
        weight_scale=float(qlinear.weight_scale),
        output_scale=float(output_scale),
        channel_axis=1,
    )


def _rescale_int8(inputs: torch.Tensor, *, input_scale: float, output_scale: float) -> torch.Tensor:
    if math.isclose(float(input_scale), float(output_scale), rel_tol=0.0, abs_tol=1e-15):
        return inputs.to(torch.int8)
    numerator, denominator = scale_to_multiplier(float(input_scale) / float(output_scale))
    return requantize_int8(
        inputs.to(torch.int64),
        multiplier_numerator=numerator,
        multiplier_denominator=denominator,
        symmetric=True,
    )


def _conv_block_reference(
    block: ConvBlock,
    inputs: torch.Tensor,
    *,
    input_scale: float,
    output_scale: float,
) -> torch.Tensor:
    return _conv_bn_reference(
        inputs,
        input_scale=input_scale,
        conv=block.conv,
        bn=block.bn,
        output_scale=output_scale,
        activation="relu",
    )


def _mbconv_reference(
    name: str,
    block: MBConvBlock,
    inputs: torch.Tensor,
    *,
    input_scale: float,
    scales: dict[str, float],
) -> tuple[torch.Tensor, float]:
    current = inputs
    current_scale = float(input_scale)
    if block.use_expand:
        expand_scale = float(scales[f"{name}.expand_relu"])
        current = _conv_bn_reference(
            current,
            input_scale=current_scale,
            conv=block.expand_conv,
            bn=block.expand_bn,
            output_scale=expand_scale,
            activation="relu6",
        )
        current_scale = expand_scale

    dw_scale = float(scales[f"{name}.dw_relu"])
    current = _conv_bn_reference(
        current,
        input_scale=current_scale,
        conv=block.dw_conv,
        bn=block.dw_bn,
        output_scale=dw_scale,
        activation="relu6",
    )
    current_scale = dw_scale

    output_scale = float(scales[name])
    projected = _conv_bn_reference(
        current,
        input_scale=current_scale,
        conv=block.project_conv,
        bn=block.project_bn,
        output_scale=output_scale,
        activation=None,
    )
    if block.use_residual:
        identity = _rescale_int8(
            inputs,
            input_scale=float(input_scale),
            output_scale=output_scale,
        )
        projected = residual_add_int_reference(projected, identity, output_bit_width=8)
    return projected, output_scale


def _skip_reference(
    name: str,
    block: SkipBlock,
    inputs: torch.Tensor,
    *,
    input_scale: float,
    scales: dict[str, float],
) -> tuple[torch.Tensor, float]:
    if not block.use_conv:
        return inputs, float(input_scale)
    output_scale = float(scales[name])
    output = _conv_bn_reference(
        inputs,
        input_scale=float(input_scale),
        conv=block.conv,
        bn=block.bn,
        output_scale=output_scale,
        activation=None,
    )
    return output, output_scale


def full_network_int8_reference(
    model: nn.Module,
    inputs_fp32: torch.Tensor,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Run the calibrated integer reference and return INT8 logits."""

    validate_reference_supported_model(model)
    model.eval()
    scales = {str(key): float(value) for key, value in calibration["scales"].items()}
    with torch.no_grad():
        fp32_logits = model(inputs_fp32.cpu()).detach().cpu()

    current_scale = float(scales["input"])
    current = quantize_symmetric_int8(inputs_fp32.cpu(), scale=current_scale)

    stem_scale = float(scales["stem.conv"])
    current = _conv_block_reference(
        model.stem.conv,
        current,
        input_scale=current_scale,
        output_scale=stem_scale,
    )
    current_scale = stem_scale

    for stage_index, stage in enumerate(model.stages):
        for block_index, block in enumerate(stage):
            name = f"stages.{stage_index}.{block_index}"
            if isinstance(block, ConvBlock):
                output_scale = float(scales[name])
                current = _conv_block_reference(
                    block,
                    current,
                    input_scale=current_scale,
                    output_scale=output_scale,
                )
                current_scale = output_scale
            elif isinstance(block, MBConvBlock):
                current, current_scale = _mbconv_reference(
                    name,
                    block,
                    current,
                    input_scale=current_scale,
                    scales=scales,
                )
            elif isinstance(block, SkipBlock):
                current, current_scale = _skip_reference(
                    name,
                    block,
                    current,
                    input_scale=current_scale,
                    scales=scales,
                )
            else:  # pragma: no cover - validate_reference_supported_model catches this.
                raise ValueError(f"unsupported block: {block.__class__.__name__}")

    gap_scale = float(scales["head.gap"])
    kernel_size = (int(current.shape[2]), int(current.shape[3]))
    current = avg_pool2d_int_reference(current, kernel_size=kernel_size)
    current = _rescale_int8(
        current,
        input_scale=current_scale,
        output_scale=gap_scale,
    )
    current_scale = gap_scale
    current = current.flatten(1)
    logits_scale = float(scales["head.fc"])
    logits_int8 = _linear_reference(
        current,
        input_scale=current_scale,
        linear=model.head.fc,
        output_scale=logits_scale,
    )
    logits_dequant = logits_int8.to(torch.float32) * logits_scale
    fp32_argmax = fp32_logits.argmax(dim=1)
    int8_argmax = logits_int8.to(torch.int16).argmax(dim=1)
    return {
        "logits_int8": logits_int8,
        "logits_dequant": logits_dequant,
        "fp32_logits": fp32_logits,
        "fp32_argmax": fp32_argmax,
        "int8_argmax": int8_argmax,
        "argmax_match": (fp32_argmax == int8_argmax),
        "max_abs_logit_error": float((fp32_logits - logits_dequant).abs().max().item()),
    }


def build_reference_model_from_architecture(architecture: ArchitectureSpec, num_classes: int = 8) -> nn.Module:
    validate_frozen_fourstage(architecture)
    model = build_model(architecture=architecture, num_classes=num_classes).eval()
    validate_reference_supported_model(model)
    return model
