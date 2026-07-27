"""Bit-exact INT8 contract for the versioned MixConv-v2 operator.

This module is deliberately operator-specific.  It fixes channel splitting,
weight/bias layout, requantization, ReLU6 placement, and residual ordering in
one place so PyTorch export, the software integer reference, C-sim, and HLS do
not silently implement different graphs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from hwnas_fpga.deploy.fixed_point import (
    conv2d_int_reference,
    quantize_bias_int32,
    quantize_symmetric_int8,
    requantize_per_output_int8,
    residual_add_int_reference,
    scale_to_multiplier,
)
from hwnas_fpga.deploy.reparam import FoldedMixConvV2Block


MIXCONV_V2_INTEGER_SCHEMA = "mixconv_v2_integer_v1"
MIXCONV_V2_WEIGHT_LAYOUT = "dw3_oihw_then_dw5_oihw_then_pw_oihw"
MIXCONV_V2_BIAS_LAYOUT = "dw3_then_dw5_then_pw"


def _quantize_weight_per_output(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    if weight.ndim < 2 or weight.shape[0] <= 0:
        raise ValueError("weight must have a non-empty output-channel dimension")
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales: list[float] = []
    for output_channel in range(int(weight.shape[0])):
        channel = weight[output_channel].detach().to(torch.float32)
        max_abs = float(channel.abs().max().item())
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        quantized[output_channel] = torch.round(channel / scale).clamp(-127, 127).to(
            torch.int8
        )
        scales.append(float(scale))
    return quantized.cpu(), scales


def _multipliers(
    input_scale: float,
    weight_scales: Sequence[float],
    output_scale: float,
) -> tuple[list[int], list[int]]:
    numerators: list[int] = []
    denominators: list[int] = []
    for weight_scale in weight_scales:
        numerator, denominator = scale_to_multiplier(
            float(input_scale) * float(weight_scale) / float(output_scale)
        )
        numerators.append(numerator)
        denominators.append(denominator)
    return numerators, denominators


def build_mixconv_v2_integer_package(
    block: FoldedMixConvV2Block,
    *,
    input_scale: float,
    branch_output_scale: float,
    output_scale: float | None = None,
) -> dict[str, Any]:
    """Quantize one folded block and emit its canonical HLS layout."""
    if block.training:
        raise RuntimeError("mixconv_v2 integer export requires eval mode")
    if input_scale <= 0 or branch_output_scale <= 0:
        raise ValueError("activation scales must be positive")
    resolved_output_scale = float(output_scale or input_scale)
    if resolved_output_scale <= 0:
        raise ValueError("output_scale must be positive")
    if block.use_residual and abs(resolved_output_scale - float(input_scale)) > 1e-12:
        raise ValueError(
            "residual mixconv_v2 requires output_scale == input_scale for exact add"
        )

    dw_weights: list[torch.Tensor] = []
    dw_biases: list[torch.Tensor] = []
    dw_scales: list[float] = []
    for conv in block.dw_convs:
        quantized_weight, weight_scales = _quantize_weight_per_output(conv.weight)
        quantized_bias = quantize_bias_int32(
            conv.bias.detach(),
            input_scale=float(input_scale),
            weight_scale=weight_scales,
        ).cpu()
        dw_weights.append(quantized_weight)
        dw_biases.append(quantized_bias)
        dw_scales.extend(weight_scales)

    pw_weight, pw_scales = _quantize_weight_per_output(block.pw_conv.weight)
    pw_bias = quantize_bias_int32(
        block.pw_conv.bias.detach(),
        input_scale=float(branch_output_scale),
        weight_scale=pw_scales,
    ).cpu()
    dw_num, dw_den = _multipliers(input_scale, dw_scales, branch_output_scale)
    pw_num, pw_den = _multipliers(
        branch_output_scale, pw_scales, resolved_output_scale
    )
    relu6_limit = min(127, max(0, int(round(6.0 / float(branch_output_scale)))))

    weights_flat = torch.cat(
        [
            dw_weights[0].reshape(-1),
            dw_weights[1].reshape(-1),
            pw_weight.reshape(-1),
        ]
    ).to(torch.int8)
    biases_flat = torch.cat([dw_biases[0], dw_biases[1], pw_bias]).to(torch.int32)

    return {
        "schema": MIXCONV_V2_INTEGER_SCHEMA,
        "weight_layout": MIXCONV_V2_WEIGHT_LAYOUT,
        "bias_layout": MIXCONV_V2_BIAS_LAYOUT,
        "group_sizes": list(block.group_sizes),
        "kernel_sizes": list(block.kernel_sizes),
        "stride": int(block.dw_convs[0].stride[0]),
        "in_channels": int(sum(block.group_sizes)),
        "out_channels": int(block.pw_conv.out_channels),
        "use_residual": bool(block.use_residual),
        "input_scale": float(input_scale),
        "branch_output_scale": float(branch_output_scale),
        "output_scale": resolved_output_scale,
        "relu6_limit": relu6_limit,
        "dw_weights": tuple(dw_weights),
        "dw_biases": tuple(dw_biases),
        "pw_weight": pw_weight,
        "pw_bias": pw_bias,
        "weights_flat": weights_flat,
        "biases_flat": biases_flat,
        "dw_weight_scales": dw_scales,
        "pw_weight_scales": pw_scales,
        "dw_multiplier_numerators": dw_num,
        "dw_multiplier_denominators": dw_den,
        "pw_multiplier_numerators": pw_num,
        "pw_multiplier_denominators": pw_den,
        "rounding": "nearest_even",
        "activation_range": [-127, 127],
        "accumulator_width": 32,
    }


def quantize_mixconv_v2_input(
    inputs: torch.Tensor, package: Mapping[str, Any]
) -> torch.Tensor:
    return quantize_symmetric_int8(inputs, scale=float(package["input_scale"]))


def simulate_mixconv_v2_int8(
    inputs: torch.Tensor,
    package: Mapping[str, Any],
) -> torch.Tensor:
    """Execute the canonical operator graph on an already-quantized tensor."""
    if str(package.get("schema")) != MIXCONV_V2_INTEGER_SCHEMA:
        raise ValueError("unsupported mixconv_v2 integer package")
    if inputs.dtype != torch.int8 or inputs.ndim != 4:
        raise ValueError("inputs must be an NCHW torch.int8 tensor")
    group_sizes = [int(value) for value in package["group_sizes"]]
    if int(inputs.shape[1]) != sum(group_sizes):
        raise ValueError("input channel count does not match package")

    branch_outputs: list[torch.Tensor] = []
    channel_offset = 0
    scale_offset = 0
    for branch_index, (channels, kernel_size) in enumerate(
        zip(group_sizes, package["kernel_sizes"])
    ):
        branch_input = inputs[:, channel_offset : channel_offset + channels]
        accumulator = conv2d_int_reference(
            branch_input,
            package["dw_weights"][branch_index].to(inputs.device),
            package["dw_biases"][branch_index].to(inputs.device),
            stride=int(package["stride"]),
            padding=int(kernel_size) // 2,
            groups=channels,
        )
        branch_scales = package["dw_weight_scales"][
            scale_offset : scale_offset + channels
        ]
        output = requantize_per_output_int8(
            accumulator,
            input_scale=float(package["input_scale"]),
            weight_scale=branch_scales,
            output_scale=float(package["branch_output_scale"]),
            channel_axis=1,
        )
        output = output.to(torch.int64).clamp(
            0, int(package["relu6_limit"])
        ).to(torch.int8)
        branch_outputs.append(output)
        channel_offset += channels
        scale_offset += channels

    mixed = torch.cat(branch_outputs, dim=1)
    accumulator = conv2d_int_reference(
        mixed,
        package["pw_weight"].to(inputs.device),
        package["pw_bias"].to(inputs.device),
        stride=1,
        padding=0,
        groups=1,
    )
    output = requantize_per_output_int8(
        accumulator,
        input_scale=float(package["branch_output_scale"]),
        weight_scale=package["pw_weight_scales"],
        output_scale=float(package["output_scale"]),
        channel_axis=1,
    )
    if bool(package["use_residual"]):
        output = residual_add_int_reference(output, inputs)
    return output
