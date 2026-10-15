"""Integer oracle and full-shape fixtures for the T6 template coverage work.

The already-tested pilot oracle remains the authority for modulo INT32,
nearest-even INT64 requantization, signed saturation, staged addition, and
nearest 2x resize. This module adds OIHW 3x3 convolution and max pooling.
It supplies synthetic integer inputs and scales, never trained/calibrated
network parameters. Python parity is software evidence only.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tools.hardware_measurement_pilot import reference as legacy


wrap_int32 = legacy.wrap_int32
round_shift_nearest_even = legacy.round_shift_nearest_even
requantize = legacy.requantize
pointwise_conv = legacy.pointwise_conv
depthwise_conv = legacy.depthwise_conv
nearest2x = legacy.nearest2x
staged_add = legacy.staged_add
requant_concat = legacy.requant_concat

fixture_modes = (
    "random_linear", "random_activation", "ties_even",
    "overflow_extremes", "saturate_shift0",
)
ACTIVATION_CODES = {"linear": 0, "relu": 1, "relu6": 2}
SYNTHETIC_RELU6_CAP = 12


def _geometry(height, width, kernel, stride, padding):
    kernel = legacy._integer(kernel, "kernel")
    stride = legacy._integer(stride, "stride")
    padding = legacy._integer(padding, "padding")
    if kernel < 1 or stride < 1 or padding < 0:
        raise ValueError("kernel/stride must be positive and padding nonnegative")
    oh = (height + 2 * padding - kernel) // stride + 1
    ow = (width + 2 * padding - kernel) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel produces an empty output")
    return kernel, stride, padding, oh, ow


def conv_accumulator(x, weight, bias, *, input_bits=8, weight_bits=8,
                     stride=1, padding=0, depthwise=False):
    """Extend the unchanged pilot oracle to standard OI33 convolution.

    Activations are CHW, standard weights are OIHW, depthwise weights are
    C1KK. Products/sums use exact signed INT64, followed by signed INT32
    modulo wrap. Spatial padding is zero, including at stride-2 borders.
    """
    if not isinstance(depthwise, (bool, np.bool_)):
        raise ValueError("depthwise must be boolean")
    w = np.asarray(weight)
    if depthwise or (w.ndim == 4 and w.shape[2:] == (1, 1)):
        return legacy.conv_accumulator(x, weight, bias, input_bits=input_bits,
            weight_bits=weight_bits, stride=stride, padding=padding,
            depthwise=depthwise)
    x = legacy._chw(x, input_bits, "input")
    w = legacy._bounded(weight, legacy._bits(weight_bits), "weight")
    if w.ndim != 4 or any(size == 0 for size in w.shape):
        raise ValueError("weight must be nonempty OIHW")
    cin, height, width = x.shape
    cout, channels, kh, kw = w.shape
    if channels != cin or (kh, kw) != (3, 3):
        raise ValueError("standard convolution supports OI11 or OI33 only")
    _, stride, padding, oh, ow = _geometry(height, width, kh, stride, padding)
    bias = np.zeros(cout, dtype=np.int64) if bias is None else legacy._bounded(bias, 32, "bias")
    if bias.shape != (cout,):
        raise ValueError("bias must have shape (output_channels,)")
    bound = cin * kh * kw * int(np.max(np.abs(x))) * int(np.max(np.abs(w))) + int(np.max(np.abs(bias)))
    if bound > np.iinfo(np.int64).max:
        raise ValueError("convolution may overflow INT64 before INT32 wrap")
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    accumulator = np.zeros((cout, oh, ow), dtype=np.int64)
    for row in range(kh):
        for col in range(kw):
            sample = padded[:, row:row + oh * stride:stride,
                            col:col + ow * stride:stride]
            accumulator += (w[:, :, row, col] @ sample.reshape(cin, -1)).reshape(cout, oh, ow)
    return wrap_int32(accumulator + bias[:, None, None])


def standard_conv3x3(x, weight, bias, *, multiplier, shift, input_bits=8,
                     weight_bits=8, output_bits=8, stride=1, padding=1):
    """Requantized standard 3x3 convolution; activation remains explicit."""
    if np.asarray(weight).ndim != 4 or np.asarray(weight).shape[2:] != (3, 3):
        raise ValueError("standard_conv3x3 requires OI33 weights")
    acc = conv_accumulator(x, weight, bias, input_bits=input_bits,
        weight_bits=weight_bits, stride=stride, padding=padding)
    return requantize(acc, multiplier, shift, bits=output_bits)


def maxpool(x, kernel=2, stride=2, padding=0, *, bits=8):
    """Floor-mode CHW max pooling, with INT minimum as padding.

    INT minimum is the exact negative-infinity sentinel on the supported
    signed integer domain. Zero padding would incorrectly lift negative
    border windows. There is no requantization or activation in max pooling.
    """
    bits = legacy._bits(bits)
    x = legacy._chw(x, bits, "input")
    cin, height, width = x.shape
    kernel, stride, padding, oh, ow = _geometry(height, width, kernel, stride, padding)
    minimum = -(1 << (bits - 1))
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)),
                    mode="constant", constant_values=minimum)
    result = np.full((cin, oh, ow), minimum, dtype=np.int64)
    for row in range(kernel):
        for col in range(kernel):
            sample = padded[:, row:row + oh * stride:stride,
                            col:col + ow * stride:stride]
            np.maximum(result, sample, out=result)
    return result.astype(legacy._dtype(bits))


def operator_name(entry):
    """Normalize existing inventory families and prepared pilot operator names."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get("shape"), Mapping):
        raise ValueError("entry requires a shape mapping")
    name = entry.get("operator") or entry.get("family")
    aliases = {"conv1x1": "pw1x1", "conv3x3": "conv3x3", "DW3": "dw3",
               "DW5": "dw5", "nearest": "nearest2x", "maxpool": "maxpool",
               "pw1x1": "pw1x1", "dw3": "dw3", "dw5": "dw5",
               "nearest2x": "nearest2x", "requant_add": "requant_add",
               "body8_to_head16": "body8_to_head16"}
    if name not in aliases:
        raise ValueError(f"unsupported inventory family/operator: {name!r}")
    return aliases[name]


def _fixture_spec(entry):
    op = operator_name(entry)
    shape = entry["shape"]
    precision = entry.get("precision", {})
    bits = legacy._bits(entry.get("input_bits", precision.get("input_bitwidth")))
    outbits = legacy._bits(entry.get("output_bits", precision.get("output_bitwidth")))
    wb = entry.get("weight_bits", precision.get("weight_bitwidth"))
    conv = op in ("pw1x1", "conv3x3", "dw3", "dw5")
    if conv:
        wb = legacy._bits(wb)
        if wb != 8:
            raise ValueError("T6 active profiles require INT8 convolution weights")
    elif wb is not None:
        raise ValueError("non-convolution precision must have no weight bitwidth")
    else:
        wb = 8  # One-byte dummy weight file for the unchanged fixture container.
    inputs = shape.get("input_shapes")
    if not isinstance(inputs, (list, tuple)) or len(inputs) != (2 if op == "requant_add" else 1):
        raise ValueError("operator input shape count does not match")
    def chw(value, name):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must be a CHW triple")
        value = tuple(legacy._integer(v, name) for v in value)
        if min(value) < 1:
            raise ValueError(f"{name} dimensions must be positive")
        return value
    cin, height, width = chw(inputs[0], "input shape")
    output = chw(shape.get("output_shape"), "output shape")
    cout, oh, ow = output
    activation = shape.get("activation", "linear")
    if activation not in ACTIVATION_CODES:
        raise ValueError(f"unsupported activation {activation!r}")
    if conv or op == "maxpool":
        kernel, stride, padding, computed_oh, computed_ow = _geometry(
            height, width, shape.get("kernel"), shape.get("stride"), shape.get("padding"))
        if (oh, ow) != (computed_oh, computed_ow):
            raise ValueError("declared output spatial shape disagrees with operator geometry")
        if conv:
            required_k = {"pw1x1": 1, "conv3x3": 3, "dw3": 3, "dw5": 5}[op]
            if kernel != required_k:
                raise ValueError("operator kernel disagrees with inventory family")
            groups = legacy._integer(shape.get("groups", cin if op.startswith("dw") else 1), "groups")
            if op.startswith("dw") and (cout != cin or groups != cin):
                raise ValueError("depthwise requires Cout=Cin=groups")
            if not op.startswith("dw") and groups != 1:
                raise ValueError("standard convolution requires groups=1")
        elif cout != cin or outbits != bits or activation != "linear":
            raise ValueError("maxpool preserves channels, precision and linear activation")
    else:
        kernel, stride, padding = 1, 1, 0
        if op == "nearest2x":
            if output != (cin, 2 * height, 2 * width) or bits != outbits:
                raise ValueError("nearest2x must preserve channels/precision and double both spatial sizes")
        elif output != (cin, height, width):
            raise ValueError("elementwise/boundary output must match input shape")
        if op == "requant_add":
            if chw(inputs[1], "input b shape") != (cin, height, width) or bits != outbits:
                raise ValueError("add requires equal input/output shapes and precision")
        if op == "body8_to_head16" and (bits, outbits) != (8, 16):
            raise ValueError("body8_to_head16 requires INT8 input and INT16 output")
        if activation != "linear":
            raise ValueError("non-convolution primitive must have linear activation")
    return dict(op=op, shape=shape, bits=bits, outbits=outbits, wb=wb, conv=conv,
                cin=cin, cout=cout, h=height, w=width, k=kernel,
                stride=stride, padding=padding, output=output, activation=activation)


def make_fixture(entry, mode, seed):
    """Return the unchanged a/b/w/bias/params/expected binary-array container.

    All dimensions come from the exact inventory row; no downscaling occurs.
    Each mode is deterministic from (seed, mode), so PE/SIMD variants can
    share byte-identical fixtures. params[6] always preserves the row's
    actual activation. params[7]=12 is a synthetic ReLU6 integer cap.
    """
    if mode not in fixture_modes:
        raise ValueError(f"unknown fixture mode {mode!r}")
    seed = legacy._integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    spec = _fixture_spec(entry)
    op, bits, outbits, wb = (spec[key] for key in ("op", "bits", "outbits", "wb"))
    cin, cout, h, width, k = (spec[key] for key in ("cin", "cout", "h", "w", "k"))
    dtype, wdtype = legacy._dtype(bits), legacy._dtype(wb)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    wlow, whigh = -(1 << (wb - 1)), (1 << (wb - 1)) - 1
    rng = np.random.default_rng(seed + fixture_modes.index(mode))
    a = rng.integers(low, high + 1, (cin, h, width), dtype=np.int32).astype(dtype)
    b = rng.integers(low, high + 1, a.shape, dtype=np.int32).astype(dtype)
    weight_shape = (cout, 1 if op.startswith("dw") else cin, k, k) if spec["conv"] else (1,)
    weight = rng.integers(wlow, whigh + 1, weight_shape, dtype=np.int32).astype(wdtype)
    bias = rng.integers(-20000, 20001, cout, dtype=np.int32).astype("<i4")
    p = np.array([3, 11 if outbits == 8 else 16, 1, 0, 1, 1,
                  ACTIVATION_CODES[spec["activation"]], SYNTHETIC_RELU6_CAP], dtype="<i4")
    if not spec["conv"]:
        p[:6] = [1, 1, 1, 0, 1, 1]
    if op == "body8_to_head16":
        p[:2] = [16, 0]
    if mode == "ties_even":
        pattern = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=dtype)
        a.flat[:] = np.resize(pattern, a.size)
        b.flat[:] = np.resize(pattern[::-1], b.size)
        weight.fill(0)
        bias[:] = np.resize(pattern, cout)
        p[:6] = [1, 1, 1, 1, 1, 0]
    elif mode == "overflow_extremes":
        a.fill(low)
        b.fill(high)
        weight.fill(wlow)
        bias[:] = np.resize(np.array([2**31 - 1, -2**31], dtype="<i4"), cout)
        p[:6] = [2**31 - 1, 31, 1, 0, 1, 0]
    elif mode == "saturate_shift0":
        a.flat[:] = np.resize(np.array([low, high], dtype=dtype), a.size)
        b.flat[:] = np.resize(np.array([high, low], dtype=dtype), b.size)
        p[:6] = [2**31 - 1, 0, 2**31 - 1, 0, 1, 0]
    if spec["conv"]:
        acc = conv_accumulator(a, weight, bias, input_bits=bits, weight_bits=wb,
            stride=spec["stride"], padding=spec["padding"], depthwise=op.startswith("dw"))
        expected = requantize(acc, int(p[0]), int(p[1]), bits=outbits)
        if p[6]:
            np.maximum(expected, 0, out=expected)
        if p[6] == 2:
            np.minimum(expected, p[7], out=expected)
    elif op == "maxpool":
        expected = maxpool(a, k, spec["stride"], spec["padding"], bits=bits)
    elif op == "nearest2x":
        expected = nearest2x(a, bits=bits)
    elif op == "body8_to_head16":
        expected = requantize(a.astype(np.int32), int(p[0]), int(p[1]), bits=outbits)
    elif op == "requant_add":
        expected = staged_add(a, b, a_scale=tuple(map(int, p[:2])),
            b_scale=tuple(map(int, p[2:4])), output_scale=tuple(map(int, p[4:6])), bits=bits)
    else:
        raise AssertionError("unhandled operator")
    if expected.shape != spec["output"]:
        raise AssertionError("fixture output shape disagrees with exact inventory row")
    return dict(a=a, b=b, w=weight, bias=bias, params=p, expected=expected)


__all__ = ["wrap_int32", "round_shift_nearest_even", "requantize", "conv_accumulator",
           "pointwise_conv", "depthwise_conv", "standard_conv3x3", "maxpool",
           "nearest2x", "staged_add", "requant_concat", "fixture_modes",
           "ACTIVATION_CODES", "SYNTHETIC_RELU6_CAP", "operator_name", "make_fixture"]
