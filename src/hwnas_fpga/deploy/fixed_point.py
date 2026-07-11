"""Bit-exact integer reference primitives for HLS parity tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedPointContract:
    schema_version: int = 2
    bit_width: int = 8
    accumulator_width: int = 32
    signed: bool = True
    weight_scheme: str = "symmetric_int8_range_neg127_pos127"
    activation_scheme: str = "symmetric_int8_range_neg127_pos127"
    weight_range: tuple[int, int] = (-127, 127)
    activation_range: tuple[int, int] = (-127, 127)
    bias_dtype: str = "int32"
    bias_scale: str = "input_scale_times_weight_scale"
    accumulator_dtype: str = "int32"
    zero_point: int = 0
    rounding: str = "nearest_even"
    saturation: str = "signed_clamp"
    requantization: str = "per_output_tensor_scale"
    bn_folding: bool = True
    bias_quantization: str = "int32_at_input_times_weight_scale"
    activation_requantization: str = "per_output_tensor_scale"
    parity_status: str = "not_verified"

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
    # The deployment contract reserves -128 and uses the symmetric range
    # [-127, 127].  Generic ``saturate_signed`` intentionally remains the
    # ordinary signed INT8 clamp for accumulator/output tests.
    return torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)


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


def _round_divide_nearest_even_tensor(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """Elementwise ties-to-even division for integer denominator tensors."""
    values = numerator.to(torch.int64)
    divisor = denominator.to(torch.int64).clamp_min(1)
    sign = torch.where(values < 0, -1, 1)
    absolute = values.abs()
    quotient = torch.div(absolute, divisor, rounding_mode="floor")
    remainder = absolute % divisor
    increment = (remainder * 2 > divisor) | (
        (remainder * 2 == divisor) & ((quotient & 1) == 1)
    )
    return (quotient + increment.to(torch.int64)) * sign


def requantize_int8(
    accumulator: torch.Tensor,
    *,
    multiplier_numerator: int,
    multiplier_denominator: int,
    symmetric: bool = False,
) -> torch.Tensor:
    scaled = round_divide_nearest_even(
        accumulator.to(torch.int64) * int(multiplier_numerator),
        int(multiplier_denominator),
    )
    lower = -127 if symmetric else -128
    return scaled.clamp(lower, 127).to(torch.int8)


def quantize_bias_int32(
    bias: torch.Tensor,
    *,
    input_scale: float,
    weight_scale: float | Sequence[float],
) -> torch.Tensor:
    """Quantize bias using the exact input-scale times weight-scale contract."""
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")
    scales = torch.as_tensor(weight_scale, dtype=torch.float64, device=bias.device)
    if torch.any(scales <= 0):
        raise ValueError("weight_scale must be positive")
    values = bias.detach().to(torch.float64) / (float(input_scale) * scales)
    rounded = torch.round(values)  # torch.round is ties-to-even
    return rounded.clamp(-(1 << 31), (1 << 31) - 1).to(torch.int32)


def scale_to_multiplier(scale: float, *, max_denominator: int = 1 << 30) -> tuple[int, int]:
    """Represent a positive real scale as a deterministic integer ratio."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    ratio = Fraction(str(float(scale))).limit_denominator(max_denominator)
    return int(ratio.numerator), int(ratio.denominator)


def requantize_per_output_int8(
    accumulator: torch.Tensor,
    *,
    input_scale: float,
    weight_scale: float | Sequence[float],
    output_scale: float,
    channel_axis: int = 1,
) -> torch.Tensor:
    """Requantize INT32 accumulators with one deterministic scale per output.

    ``accumulator`` may be NCHW, NLC, or a 2-D linear output.  The output
    channel dimension is broadcast from ``channel_axis``.
    """
    if output_scale <= 0 or input_scale <= 0:
        raise ValueError("input_scale and output_scale must be positive")
    scales = torch.as_tensor(weight_scale, dtype=torch.float64, device=accumulator.device)
    if scales.ndim == 0:
        scales = scales.reshape(1)
    if torch.any(scales <= 0):
        raise ValueError("weight_scale must be positive")
    if scales.numel() == 1 and accumulator.shape[channel_axis] != 1:
        scales = scales.expand(accumulator.shape[channel_axis])
    if accumulator.shape[channel_axis] != scales.numel():
        raise ValueError(
            "weight_scale length must match accumulator output channels: "
            f"{scales.numel()} != {accumulator.shape[channel_axis]}"
        )
    ratio = (float(input_scale) * scales / float(output_scale)).tolist()
    output = torch.empty_like(accumulator, dtype=torch.int8)
    for index, value in enumerate(ratio):
        numerator, denominator = scale_to_multiplier(float(value))
        channel = accumulator.select(channel_axis, index)
        quantized = requantize_int8(
            channel,
            multiplier_numerator=numerator,
            multiplier_denominator=denominator,
            symmetric=True,
        )
        output.select(channel_axis, index).copy_(quantized)
    return output


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

    x = inputs.to(torch.int64)
    if padding:
        x = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
    # ``Tensor.unfold`` keeps the complete operation in integer arithmetic;
    # the previous implementation converted patches to float before rounding.
    patches = (
        x.unfold(2, kernel_h, stride)
        .unfold(3, kernel_w, stride)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(batch, -1, in_channels * kernel_h * kernel_w)
    )
    output_per_group = out_channels // groups
    patch_width = channels_per_group * kernel_h * kernel_w
    group_outputs = []
    for group_index in range(groups):
        patch_slice = patches[
            :,
            :,
            group_index * patch_width : (group_index + 1) * patch_width,
        ]
        weight_slice = weights[
            group_index * output_per_group : (group_index + 1) * output_per_group
        ].reshape(output_per_group, patch_width).to(torch.int64)
        # [N, L, K] @ [K, O] -> [N, L, O]
        group_outputs.append(patch_slice @ weight_slice.transpose(0, 1))
    output = torch.cat(group_outputs, dim=2)
    if bias is not None:
        output = output + bias.to(torch.int64).reshape(1, 1, -1)
    out_height = (height + 2 * padding - kernel_h) // stride + 1
    out_width = (width + 2 * padding - kernel_w) // stride + 1
    output = output.reshape(batch, out_height, out_width, out_channels).permute(0, 3, 1, 2)
    return saturate_signed(output, accumulator_width).to(torch.int64)


def relu_int_reference(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.to(torch.int64).clamp_min(0).clamp(-128, 127).to(torch.int8)


def residual_add_int_reference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    output_bit_width: int = 8,
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"residual shapes differ: {tuple(left.shape)} != {tuple(right.shape)}")
    lower = -((1 << (int(output_bit_width) - 1)) - 1)
    upper = (1 << (int(output_bit_width) - 1)) - 1
    return (left.to(torch.int64) + right.to(torch.int64)).clamp(lower, upper).to(torch.int8)


def max_pool2d_int_reference(
    inputs: torch.Tensor,
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    return F.max_pool2d(
        inputs.to(torch.int8), kernel_size, stride, padding, dilation, ceil_mode
    ).to(torch.int8)


def avg_pool2d_int_reference(
    inputs: torch.Tensor,
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> torch.Tensor:
    """Integer average pooling with nearest-even division and INT8 clamp."""
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (
        (stride, stride)
        if isinstance(stride, int)
        else (stride if stride is not None else (k_h, k_w))
    )
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    x = inputs.to(torch.int64)
    if p_h or p_w:
        x = F.pad(x, (p_w, p_w, p_h, p_h), value=0)
    windows = x.unfold(2, k_h, s_h).unfold(3, k_w, s_w)
    sums = windows.sum(dim=(-1, -2))
    denominator = k_h * k_w
    if not count_include_pad and (p_h or p_w):
        # This branch is uncommon in deployment graphs; calculate the valid
        # count with the same window geometry instead of using float pooling.
        ones = torch.ones_like(inputs, dtype=torch.int64)
        ones = F.pad(ones, (p_w, p_w, p_h, p_h), value=0)
        counts = ones.unfold(2, k_h, s_h).unfold(3, k_w, s_w).sum(dim=(-1, -2))
        result = _round_divide_nearest_even_tensor(sums, counts)
    else:
        result = round_divide_nearest_even(sums, denominator)
    return result.clamp(-127, 127).to(torch.int8)
