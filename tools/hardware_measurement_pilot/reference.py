"""Integer-only Experiment 3 oracle; software evidence, never hardware evidence.

The INT8 arithmetic follows tools.fpga_h2.numeric.NUMERIC_CONTRACT_V1.
INT16 requantization follows tools.modnas_comparison.quantization_mixed.
Convolutions have linear outputs, CHW activations, OIHW weights (C1KK for
depthwise), signed INT32 bias, exact INT64 accumulation, and INT32 wrap.
Add first requantizes and saturates each input into a common signed range,
then sums and requantizes again. Concat requantizes each input separately.
No floating-point arithmetic is used in any numeric operation below.
"""
from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bits(bits):
    bits = _integer(bits, "bits")
    if bits not in (8, 16):
        raise ValueError("only signed INT8 and INT16 are supported")
    return bits


def _integers(values, name):
    result = np.asarray(values)
    if result.dtype.kind != "i" or result.dtype.itemsize > 8:
        raise ValueError(f"{name} must contain signed integers fitting INT64")
    return result.astype(np.int64, copy=False)


def _bounded(values, bits, name):
    values = _integers(values, name)
    if np.any(values < -(1 << (bits - 1))) or np.any(values > (1 << (bits - 1)) - 1):
        raise ValueError(f"{name} outside signed INT{bits} range")
    return values


def _chw(values, bits, name):
    values = _bounded(values, _bits(bits), name)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError(f"{name} must be nonempty CHW")
    return values


def _dtype(bits):
    return np.dtype("i1" if _bits(bits) == 8 else "<i2")


def wrap_int32(values):
    """Modulo 2**32 with a signed result, safe even at INT64 extrema."""
    values = _integers(values, "values")
    low = np.bitwise_and(values, np.int64(0xFFFFFFFF))
    return np.where(low >= (1 << 31), low - (1 << 32), low).astype(np.int64)


def round_shift_nearest_even(values, shift):
    """Divide by 2**shift, round ties to even; supports all signed INT64."""
    values = _integers(values, "values")
    shift = _integer(shift, "shift")
    if not 0 <= shift <= 31:
        raise ValueError("shift must be in [0, 31]")
    if shift == 0:
        return values.copy()
    # Floor quotient/remainder avoids abs(INT64_MIN), which would overflow.
    divisor = 1 << shift
    quotient, remainder = np.divmod(values, divisor)
    half = divisor >> 1
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    return quotient + increment.astype(np.int64)


def requantize(values, multiplier, shift, *, bits=8):
    """Wrapped INT32 * positive multiplier -> nearest-even -> saturation."""
    bits = _bits(bits)
    values = _bounded(values, 32, "requantization input")
    multiplier = _integer(multiplier, "multiplier")
    if not 1 <= multiplier <= (1 << 31) - 1:
        raise ValueError("multiplier must be in [1, 2**31 - 1]")
    # INT32 * multiplier <= (2**31)*(2**31-1), strictly inside INT64.
    rounded = round_shift_nearest_even(values * multiplier, shift)
    return np.clip(rounded, -(1 << (bits - 1)), (1 << (bits - 1)) - 1).astype(_dtype(bits))


def conv_accumulator(x, weight, bias, *, input_bits=8, weight_bits=8,
                     stride=1, padding=0, depthwise=False):
    """Return wrapped INT32 values in an INT64 CHW array for pilot convolutions."""
    x = _chw(x, input_bits, "input")
    weight = _bounded(weight, _bits(weight_bits), "weight")
    stride, padding = _integer(stride, "stride"), _integer(padding, "padding")
    if stride < 1 or padding < 0:
        raise ValueError("stride must be positive and padding nonnegative")
    if not isinstance(depthwise, (bool, np.bool_)):
        raise ValueError("depthwise must be boolean")
    if weight.ndim != 4 or any(size == 0 for size in weight.shape):
        raise ValueError("weight must be nonempty OIHW or C1KK")
    cin, height, width = x.shape
    cout, channels_per_group, kh, kw = weight.shape
    if depthwise:
        if cout != cin or channels_per_group != 1 or kh != kw or kh not in (3, 5):
            raise ValueError("depthwise weight must be C1KK with K=3 or K=5")
    elif channels_per_group != cin or (kh, kw) != (1, 1):
        raise ValueError("pointwise weight must be OI11 with matching input channels")
    bias = np.zeros(cout, dtype=np.int64) if bias is None else _bounded(bias, 32, "bias")
    if bias.shape != (cout,):
        raise ValueError("bias must have shape (output_channels,)")
    oh, ow = (height + 2 * padding - kh) // stride + 1, (width + 2 * padding - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel produces an empty output")
    terms = channels_per_group * kh * kw
    bound = terms * int(np.max(np.abs(x))) * int(np.max(np.abs(weight))) + int(np.max(np.abs(bias)))
    if bound > np.iinfo(np.int64).max:
        raise ValueError("convolution may overflow INT64 before INT32 wrap")
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    accumulator = np.zeros((cout, oh, ow), dtype=np.int64)
    for row in range(kh):
        for col in range(kw):
            sample = padded[:, row:row + oh * stride:stride, col:col + ow * stride:stride]
            if depthwise:
                accumulator += sample * weight[:, 0, row, col, None, None]
            else:
                accumulator += (weight[:, :, row, col] @ sample.reshape(cin, -1)).reshape(cout, oh, ow)
    return wrap_int32(accumulator + bias[:, None, None])


def pointwise_conv(x, weight, bias, *, multiplier, shift, input_bits=8,
                   weight_bits=8, output_bits=8, stride=1, padding=0):
    accumulator = conv_accumulator(x, weight, bias, input_bits=input_bits,
        weight_bits=weight_bits, stride=stride, padding=padding, depthwise=False)
    return requantize(accumulator, multiplier, shift, bits=output_bits)


def depthwise_conv(x, weight, bias, *, multiplier, shift, input_bits=8,
                   weight_bits=8, output_bits=8, stride=1, padding=0):
    accumulator = conv_accumulator(x, weight, bias, input_bits=input_bits,
        weight_bits=weight_bits, stride=stride, padding=padding, depthwise=True)
    return requantize(accumulator, multiplier, shift, bits=output_bits)


def nearest2x(x, *, bits=8):
    """Nearest-neighbor 2x upsampling, with an explicit signed precision."""
    values = _chw(x, bits, "input")
    return np.repeat(np.repeat(values, 2, axis=1), 2, axis=2).astype(_dtype(bits))


def _scale(scale):
    if not isinstance(scale, Sequence) or len(scale) != 2:
        raise ValueError("scale must be (integer multiplier, integer shift)")
    return scale[0], scale[1]


def staged_add(a, b, *, a_scale, b_scale, output_scale, bits=8):
    """Align/saturate each input to common precision, sum, then requantize.

    Scales are (multiplier, shift): a_scale and b_scale map to the chosen
    common scale; output_scale maps from common scale to output scale.
    Caller supplies those ratios, so no floating-point scale is inferred.
    """
    a, b = _chw(a, bits, "input a"), _chw(b, bits, "input b")
    if a.shape != b.shape:
        raise ValueError("add input shapes must match exactly; broadcasting is forbidden")
    aligned_a = requantize(a, *_scale(a_scale), bits=bits)
    aligned_b = requantize(b, *_scale(b_scale), bits=bits)
    total = aligned_a.astype(np.int64) + aligned_b.astype(np.int64)
    return requantize(wrap_int32(total), *_scale(output_scale), bits=bits)


def requant_concat(inputs, scales, *, bits=8):
    """Per-input requantization followed by ordered concatenation on C axis."""
    if not isinstance(inputs, Sequence) or len(inputs) < 2:
        raise ValueError("concat requires at least two input tensors")
    if not isinstance(scales, Sequence) or len(scales) != len(inputs):
        raise ValueError("concat requires one scale pair per input")
    tensors = [_chw(value, bits, f"input {index}") for index, value in enumerate(inputs)]
    if any(value.shape[1:] != tensors[0].shape[1:] for value in tensors[1:]):
        raise ValueError("concat spatial dimensions must match")
    return np.concatenate([requantize(value, *_scale(scale), bits=bits)
                           for value, scale in zip(tensors, scales)], axis=0)


__all__ = ["wrap_int32", "round_shift_nearest_even", "requantize", "conv_accumulator",
           "pointwise_conv", "depthwise_conv", "nearest2x", "staged_add", "requant_concat"]
