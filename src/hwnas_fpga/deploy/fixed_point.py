"""Bit-exact integer reference primitives for HLS parity tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedPointContract:
    bit_width: int = 8
    accumulator_width: int = 32
    signed: bool = True
    weight_scheme: str = "symmetric_per_tensor"
    activation_scheme: str = "symmetric_per_tensor"
    rounding: str = "nearest_even"
    saturation: str = "signed_clamp"
    bn_folding: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def signed_bounds(bit_width: int) -> tuple[int, int]:
    if bit_width <= 1:
        raise ValueError("bit_width must be greater than 1")
    return -(1 << (bit_width - 1)), (1 << (bit_width - 1)) - 1


def saturate_signed(tensor: torch.Tensor, bit_width: int) -> torch.Tensor:
    lower, upper = signed_bounds(bit_width)
    return tensor.clamp(lower, upper)


def quantize_symmetric_int8(
    tensor: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return saturate_signed(torch.round(tensor / scale), 8).to(torch.int8)


def round_divide_nearest_even(
    numerator: torch.Tensor,
    denominator: int,
) -> torch.Tensor:
    """Integer division with ties-to-even, matching ``torch.round``."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    values = numerator.to(torch.int64)
    sign = torch.where(values < 0, -1, 1)
    absolute = values.abs()
    quotient = torch.div(absolute, denominator, rounding_mode="floor")
    remainder = absolute % denominator
    twice = remainder * 2
    increment = (twice > denominator) | (
        (twice == denominator) & ((quotient & 1) == 1)
    )
    rounded = quotient + increment.to(torch.int64)
    return rounded * sign


def requantize_int8(
    accumulator: torch.Tensor,
    *,
    multiplier_numerator: int,
    multiplier_denominator: int,
) -> torch.Tensor:
    scaled = round_divide_nearest_even(
        accumulator.to(torch.int64) * int(multiplier_numerator),
        int(multiplier_denominator),
    )
    return saturate_signed(scaled, 8).to(torch.int8)


def linear_int_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    accumulator_width: int = 32,
) -> torch.Tensor:
    x = inputs.to(torch.int64)
    w = weights.to(torch.int64)
    output = x @ w.transpose(0, 1)
    if bias is not None:
        output = output + bias.to(torch.int64)
    return saturate_signed(output, accumulator_width).to(torch.int64)


def conv2d_int_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    accumulator_width: int = 32,
) -> torch.Tensor:
    """Exact int64 convolution followed by signed accumulator saturation."""
    if inputs.ndim != 4 or weights.ndim != 4:
        raise ValueError("inputs and weights must be NCHW/OIHW tensors")
    batch, in_channels, height, width = inputs.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weights.shape
    if in_channels % groups or out_channels % groups:
        raise ValueError("channels must be divisible by groups")
    if channels_per_group != in_channels // groups:
        raise ValueError("weight input channels do not match groups")

    patches = F.unfold(
        inputs.to(torch.float32),
        kernel_size=(kernel_h, kernel_w),
        padding=padding,
        stride=stride,
    ).round().to(torch.int64)
    output_per_group = out_channels // groups
    patch_width = channels_per_group * kernel_h * kernel_w
    group_outputs = []
    for group_index in range(groups):
        patch_slice = patches[
            :,
            group_index * patch_width : (group_index + 1) * patch_width,
            :,
        ]
        weight_slice = weights[
            group_index * output_per_group : (group_index + 1) * output_per_group
        ].reshape(output_per_group, patch_width).to(torch.int64)
        batch_outputs = [
            weight_slice @ patch_slice[batch_index]
            for batch_index in range(batch)
        ]
        group_outputs.append(torch.stack(batch_outputs))
    output = torch.cat(group_outputs, dim=1)
    if bias is not None:
        output = output + bias.to(torch.int64).reshape(1, -1, 1)
    out_height = (height + 2 * padding - kernel_h) // stride + 1
    out_width = (width + 2 * padding - kernel_w) // stride + 1
    output = output.reshape(batch, out_channels, out_height, out_width)
    return saturate_signed(output, accumulator_width).to(torch.int64)

