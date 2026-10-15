"""Retained H2 quantization primitives; source ranges are recorded in MIGRATION.json."""
from __future__ import annotations


import math


from copy import deepcopy


from typing import Any, Iterable, Mapping


import torch


import torch.nn as nn


import torch.nn.functional as F


from .h2_contract import (
    EXPECTED_HASH as NUMERIC_CONTRACT_HASH,
    load_numeric_contract_v1,
    require_h2_aligned,
)


QMIN, QMAX = -128, 127


INT16_QMIN, INT16_QMAX = -32768, 32767


EXP_MIN, EXP_MAX = -24, 16


def signed_range(bits: int) -> tuple[int, int]:
    value = int(bits)
    if value == 8:
        return QMIN, QMAX
    if value == 16:
        return INT16_QMIN, INT16_QMAX
    raise ValueError(f"unsupported signed symmetric bitwidth: {bits}")


def signed_dtype(bits: int) -> torch.dtype:
    value = int(bits)
    if value == 8:
        return torch.int8
    if value == 16:
        return torch.int16
    raise ValueError(f"unsupported signed symmetric bitwidth: {bits}")


def exponent_from_maxabs(max_abs: float, *, qmax: int = QMAX) -> int:
    value = float(max_abs)
    if not math.isfinite(value) or value <= 0.0:
        return EXP_MIN
    exponent = int(math.ceil(math.log2(value / float(qmax))))
    return max(EXP_MIN, min(EXP_MAX, exponent))


def power_of_two(exponent: int) -> float:
    if not EXP_MIN <= int(exponent) <= EXP_MAX:
        raise ValueError("scale exponent outside NUMERIC_CONTRACT_V1")
    return float(2.0 ** int(exponent))


def product_power_of_two(exponent: int) -> float:
    """Scale for INT32 bias, equal to input_scale * weight_scale."""
    return float(2.0 ** int(exponent))


def ratio_from_exponents(from_exp: int, to_exp: int) -> tuple[int, int]:
    delta = int(from_exp) - int(to_exp)
    if not -31 <= delta <= 30:
        raise ValueError("scale ratio is not representable in H2 multiplier/shift")
    if delta >= 0:
        return (1 << delta, 0)
    return (1, -delta)


def quantize_h2(tensor: torch.Tensor, exponent: int) -> torch.Tensor:
    scale = power_of_two(exponent)
    quantized = torch.round(tensor.to(torch.float32) / scale)
    return quantized.clamp(QMIN, QMAX)


def quantize_symmetric(tensor: torch.Tensor, exponent: int, *, bits: int) -> torch.Tensor:
    qmin, qmax = signed_range(bits)
    scale = power_of_two(exponent)
    quantized = torch.round(tensor.to(torch.float32) / scale)
    return quantized.clamp(qmin, qmax)


def dequantize_h2(quantized: torch.Tensor, exponent: int) -> torch.Tensor:
    return quantized.to(torch.float32) * power_of_two(exponent)


def quantize_weight(
    weight: torch.Tensor,
    *,
    granularity: str = "per_tensor",
    bits: int = 8,
) -> tuple[torch.Tensor, list[int]]:
    _qmin, qmax = signed_range(bits)
    dtype = signed_dtype(bits)
    if granularity == "per_tensor":
        exponent = exponent_from_maxabs(float(weight.detach().abs().max().item()), qmax=qmax)
        return quantize_symmetric(weight, exponent, bits=bits).to(dtype), [exponent]
    if granularity != "per_channel":
        raise ValueError(f"unsupported weight granularity: {granularity}")
    reduce_dims = tuple(range(1, weight.ndim))
    maxima = weight.detach().abs().amax(dim=reduce_dims)
    exponents = [exponent_from_maxabs(float(value), qmax=qmax) for value in maxima]
    scales = torch.tensor([power_of_two(exp) for exp in exponents], dtype=torch.float32, device=weight.device)
    while scales.ndim < weight.ndim:
        scales = scales.unsqueeze(-1)
    qmin, qmax = signed_range(bits)
    quantized = torch.round(weight.to(torch.float32) / scales).clamp(qmin, qmax).to(dtype)
    return quantized, exponents


def quantize_bias(bias: torch.Tensor, input_exp: int, weight_exps: Iterable[int]) -> torch.Tensor:
    exponents = list(weight_exps)
    if len(exponents) == 1:
        scale = product_power_of_two(int(input_exp) + int(exponents[0]))
        values = bias.detach().to(torch.float64) / scale
        return torch.round(values).clamp(-(1 << 31), (1 << 31) - 1).to(torch.int32)
    scales = torch.tensor(
        [product_power_of_two(int(input_exp) + int(exp)) for exp in exponents],
        dtype=torch.float64,
        device=bias.device,
    )
    values = bias.detach().to(torch.float64) / scales
    return torch.round(values).clamp(-(1 << 31), (1 << 31) - 1).to(torch.int32)


def relu6_bound(output_exp: int, *, bits: int = 8) -> int:
    _qmin, qmax = signed_range(bits)
    return int(max(0, min(qmax, round(6.0 / power_of_two(output_exp)))))


def apply_activation(quantized: torch.Tensor, kind: str, output_exp: int, *, bits: int = 8) -> torch.Tensor:
    if kind == "linear":
        return quantized
    if kind == "relu":
        return quantized.clamp_min(0)
    if kind == "relu6":
        return quantized.clamp(0, relu6_bound(output_exp, bits=bits))
    raise ValueError(f"unsupported activation {kind}")


def _apply_activation_float(values: torch.Tensor, kind: str, output_exp: int, *, bits: int = 8) -> torch.Tensor:
    if kind == "linear":
        return values
    if kind == "relu":
        return values.clamp_min(0)
    if kind == "relu6":
        return values.clamp(0, relu6_bound(output_exp, bits=bits) * power_of_two(output_exp))
    raise ValueError(f"unsupported activation {kind}")


def aligned_add_dequant(
    left: torch.Tensor,
    right: torch.Tensor,
    left_exp: int,
    right_exp: int,
    output_exp: int,
) -> torch.Tensor:
    common = max(int(left_exp), int(right_exp))
    left_q = requant_exponents(quantize_h2(left, left_exp), left_exp, common)
    right_q = requant_exponents(quantize_h2(right, right_exp), right_exp, common)
    total = left_q + right_q
    return dequantize_h2(requant_exponents(total, common, output_exp), output_exp)


def requant_exponents(values: torch.Tensor, from_exp: int, to_exp: int, *, bits: int = 8) -> torch.Tensor:
    multiplier, shift = ratio_from_exponents(from_exp, to_exp)
    scaled = torch.round(values.to(torch.float32) * float(multiplier) / float(1 << shift))
    qmin, qmax = signed_range(bits)
    return scaled.clamp(qmin, qmax)


class H2FakeQuantConv(nn.Module):
    """Conv2d QDQ using H2 scales, nearest-even, and [-128, 127]."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        input_exp: int,
        output_exp: int,
        weight_exps: list[int],
        activation: str,
        weight_granularity: str,
        input_bits: int = 8,
        weight_bits: int = 8,
        output_bits: int = 8,
        quantize_input: bool = True,
        quantize_weight_values: bool = True,
        quantize_output: bool = True,
    ) -> None:
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.register_buffer("fp32_weight", conv.weight.detach().to(torch.float32).clone())
        if conv.bias is None:
            self.register_buffer("fp32_bias", None)
        else:
            self.register_buffer("fp32_bias", conv.bias.detach().to(torch.float32).clone())
        self.input_bits = int(input_bits)
        self.weight_bits = int(weight_bits)
        self.output_bits = int(output_bits)
        quantized_weight, resolved = quantize_weight(
            conv.weight,
            granularity=weight_granularity,
            bits=self.weight_bits,
        )
        if resolved != list(weight_exps):
            raise ValueError("weight exponents drifted after wrapping")
        self.register_buffer("int_weight", quantized_weight)
        if conv.bias is None:
            zeros = torch.zeros(conv.out_channels, dtype=torch.float32, device=conv.weight.device)
            self.register_buffer("int_bias", quantize_bias(zeros, input_exp, weight_exps))
        else:
            self.register_buffer("int_bias", quantize_bias(conv.bias, input_exp, weight_exps))
        self.input_exp = int(input_exp)
        self.output_exp = int(output_exp)
        self.weight_exps = [int(value) for value in weight_exps]
        self.activation = str(activation)
        self.weight_granularity = str(weight_granularity)
        self.quantize_input = bool(quantize_input)
        self.quantize_weight_values = bool(quantize_weight_values)
        self.quantize_output = bool(quantize_output)
        if self.weight_granularity == "per_tensor":
            self.multiplier, self.shift = ratio_from_exponents(
                self.input_exp + self.weight_exps[0], self.output_exp
            )
        else:
            self.multiplier, self.shift = 1, 0

    def extra_repr(self) -> str:
        return (
            f"input_exp={self.input_exp}, output_exp={self.output_exp}, "
            f"activation={self.activation}, granularity={self.weight_granularity}, "
            f"bits=I{self.input_bits}/W{self.weight_bits}/O{self.output_bits}, "
            f"q_input={self.quantize_input}, q_weight={self.quantize_weight_values}, "
            f"q_output={self.quantize_output}"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (self.input_bits, self.weight_bits, self.output_bits) != (8, 8, 8):
            return self._forward_qdq(inputs)
        if not (self.quantize_input and self.quantize_weight_values and self.quantize_output):
            return self._forward_diagnostic(inputs)
        quantized_input = quantize_h2(inputs, self.input_exp)
        if self.weight_granularity == "per_tensor":
            weight = self.int_weight.to(dtype=inputs.dtype)
        else:
            scales = torch.tensor(
                [power_of_two(exp) / power_of_two(self.weight_exps[0]) for exp in self.weight_exps],
                dtype=inputs.dtype,
                device=inputs.device,
            ).view(-1, 1, 1, 1)
            weight = self.int_weight.to(dtype=inputs.dtype) * scales
        bias = self.int_bias.to(dtype=inputs.dtype)
        if self.weight_granularity == "per_channel":
            bias = bias * torch.tensor(
                [power_of_two(self.input_exp + exp) / power_of_two(self.input_exp + self.weight_exps[0]) for exp in self.weight_exps],
                dtype=inputs.dtype,
                device=inputs.device,
            )
        accumulator = F.conv2d(
            quantized_input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.weight_granularity == "per_tensor":
            requantized = requant_exponents(
                accumulator,
                self.input_exp + self.weight_exps[0],
                self.output_exp,
            )
        else:
            requantized = torch.empty_like(accumulator)
            for channel, exponent in enumerate(self.weight_exps):
                requantized[:, channel] = requant_exponents(
                    accumulator[:, channel],
                    self.input_exp + exponent,
                    self.output_exp,
                )
        activated = apply_activation(requantized, self.activation, self.output_exp)
        return dequantize_h2(activated, self.output_exp)

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.weight_granularity == "per_tensor":
            return self.int_weight.to(device=device, dtype=dtype) * power_of_two(self.weight_exps[0])
        scales = torch.tensor(
            [power_of_two(exp) for exp in self.weight_exps],
            dtype=dtype,
            device=device,
        ).view(-1, 1, 1, 1)
        return self.int_weight.to(device=device, dtype=dtype) * scales

    def _dequantized_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.weight_granularity == "per_tensor":
            scale = product_power_of_two(self.input_exp + self.weight_exps[0])
            return self.int_bias.to(device=device, dtype=dtype) * scale
        scales = torch.tensor(
            [product_power_of_two(self.input_exp + exp) for exp in self.weight_exps],
            dtype=dtype,
            device=device,
        )
        return self.int_bias.to(device=device, dtype=dtype) * scales

    def _forward_diagnostic(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quantize_input:
            conv_input = dequantize_h2(quantize_h2(inputs, self.input_exp), self.input_exp)
        else:
            conv_input = inputs
        if self.quantize_weight_values:
            weight = self._dequantized_weight(inputs.dtype, inputs.device)
            bias = self._dequantized_bias(inputs.dtype, inputs.device)
        else:
            weight = self.fp32_weight.to(device=inputs.device, dtype=inputs.dtype)
            bias = None if self.fp32_bias is None else self.fp32_bias.to(device=inputs.device, dtype=inputs.dtype)
        output = F.conv2d(
            conv_input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.quantize_output:
            output = dequantize_h2(quantize_h2(output, self.output_exp), self.output_exp)
        return _apply_activation_float(output, self.activation, self.output_exp)

    def _forward_qdq(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quantize_input:
            conv_input = dequantize_h2(
                quantize_symmetric(inputs, self.input_exp, bits=self.input_bits),
                self.input_exp,
            )
        else:
            conv_input = inputs
        if self.quantize_weight_values:
            weight = self._dequantized_weight(inputs.dtype, inputs.device)
            bias = self._dequantized_bias(inputs.dtype, inputs.device)
        else:
            weight = self.fp32_weight.to(device=inputs.device, dtype=inputs.dtype)
            bias = None if self.fp32_bias is None else self.fp32_bias.to(device=inputs.device, dtype=inputs.dtype)
        output = F.conv2d(
            conv_input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.quantize_output:
            output = dequantize_h2(
                quantize_symmetric(output, self.output_exp, bits=self.output_bits),
                self.output_exp,
            )
        return _apply_activation_float(output, self.activation, self.output_exp, bits=self.output_bits)
