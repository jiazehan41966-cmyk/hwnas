"""H2-aligned PTQ primitives for frozen detector candidates.

This is a software Q(x) pipeline, not QAT and not a new numeric contract.
Official FPGA-facing profiles remain NUMERIC_CONTRACT_V1: INT8 activations and
weights, INT32 accumulation, INT64 requant product, nearest-even, saturate to
[-128, 127], zero-point 0, power-of-two scales.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.deploy.h2_contract import (
    EXPECTED_HASH as NUMERIC_CONTRACT_HASH,
    load_numeric_contract_v1,
    require_h2_aligned,
)
from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace


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


def following_activation(parent: nn.Module, conv_name: str) -> str:
    children = list(parent.named_children())
    names = [name for name, _ in children]
    if conv_name not in names:
        return "linear"
    index = names.index(conv_name) + 1
    while index < len(children) and isinstance(children[index][1], nn.Identity):
        index += 1
    if index >= len(children):
        return "linear"
    module = children[index][1]
    if isinstance(module, nn.ReLU6):
        return "relu6"
    if isinstance(module, nn.ReLU):
        return "relu"
    return "linear"


def conv_activation_map(model: nn.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in parent.named_children():
            path = f"{prefix}{name}"
            if isinstance(child, nn.Conv2d):
                mapping[path] = following_activation(parent, name)
            else:
                visit(child, f"{path}.")

    visit(model, "")
    return mapping


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


def _replace_fused_activations(model: nn.Module) -> None:
    for parent in model.modules():
        children = list(parent.named_children())
        for index, (name, child) in enumerate(children):
            if isinstance(child, (nn.ReLU, nn.ReLU6)):
                previous = index - 1
                while previous >= 0 and isinstance(children[previous][1], nn.Identity):
                    previous -= 1
                if previous >= 0 and isinstance(children[previous][1], H2FakeQuantConv):
                    setattr(parent, name, nn.Identity())


class H2QuantizedFPN(nn.Module):
    def __init__(self, neck: nn.Module, add_exponents: list[dict[str, int]]) -> None:
        super().__init__()
        self.lateral_convs = neck.lateral_convs
        self.output_convs = neck.output_convs
        self.add_exponents = list(add_exponents)

    def forward(self, features: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        fused = list(laterals)
        for index in range(len(fused) - 2, -1, -1):
            upsampled = F.interpolate(fused[index + 1], size=fused[index].shape[-2:], mode="nearest")
            spec = self.add_exponents[index]
            fused[index] = aligned_add_dequant(
                fused[index],
                upsampled,
                spec["left_exp"],
                spec["right_exp"],
                spec["output_exp"],
            )
        return tuple(conv(feature) for conv, feature in zip(self.output_convs, fused))


@torch.no_grad()
def observe_activation_extrema(
    model: nn.Module,
    images: Iterable[torch.Tensor],
    *,
    device: str,
) -> dict[str, dict[str, float]]:
    model = model.to(device).eval()
    stats: dict[str, dict[str, float]] = {}
    hooks = []

    def make_pre(name: str):
        def capture(_module, inputs):
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor):
                stats.setdefault(name, {"input_max_abs": 0.0, "output_max_abs": 0.0})
                stats[name]["input_max_abs"] = max(
                    stats[name]["input_max_abs"], float(tensor.detach().abs().max().item())
                )
        return capture

    def make_post(name: str):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor):
                stats.setdefault(name, {"input_max_abs": 0.0, "output_max_abs": 0.0})
                stats[name]["output_max_abs"] = max(
                    stats[name]["output_max_abs"], float(tensor.detach().abs().max().item())
                )
        return capture

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_pre_hook(make_pre(name)))
            hooks.append(module.register_forward_hook(make_post(name)))
    add_stats: list[dict[str, float]] = []

    original_neck = model.neck
    original_forward = original_neck.forward

    def observing_neck(features):
        laterals = [conv(feature) for conv, feature in zip(original_neck.lateral_convs, features)]
        fused = list(laterals)
        for index in range(len(fused) - 2, -1, -1):
            upsampled = F.interpolate(fused[index + 1], size=fused[index].shape[-2:], mode="nearest")
            add_stats.append(
                {
                    "left_max_abs": float(fused[index].detach().abs().max().item()),
                    "right_max_abs": float(upsampled.detach().abs().max().item()),
                }
            )
            fused[index] = fused[index] + upsampled
            add_stats[-1]["output_max_abs"] = float(fused[index].detach().abs().max().item())
        return tuple(conv(feature) for conv, feature in zip(original_neck.output_convs, fused))

    model.neck.forward = observing_neck  # type: ignore[method-assign]
    for batch in images:
        model(batch.to(device))
    for hook in hooks:
        hook.remove()
    model.neck.forward = original_forward
    return {"convs": stats, "adds": add_stats}


def wrap_h2_ptq(
    model: nn.Module,
    extrema: Mapping[str, Any],
    *,
    weight_granularity: str = "per_tensor",
    activation_granularity: str = "per_tensor",
    scheme: str = "symmetric",
    fp32_module_prefixes: Iterable[str] | None = None,
    fp32_weight_prefixes: Iterable[str] | None = None,
    fp32_activation_prefixes: Iterable[str] | None = None,
    weight_bits_by_prefix: Mapping[str, int] | None = None,
    activation_bits_by_prefix: Mapping[str, int] | None = None,
    quantize_neck_adds: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    if scheme != "symmetric":
        raise ValueError("asymmetric FPGA policy is outside NUMERIC_CONTRACT_V1")
    if activation_granularity != "per_tensor":
        raise ValueError("official H2 activation scale is per-tensor; per-channel FPGA profile is not admitted")
    load_numeric_contract_v1()
    require_h2_aligned(
        {
            "activation_bits": 8,
            "weight_bits": 8,
            "accumulator_bits": 32,
            "signed_range": [QMIN, QMAX],
            "zero_point": 0,
            "rounding": "nearest_even",
            "requant_product_bits": 64,
            "scale": "positive_power_of_two_exponent_int32",
        }
    )
    activations = conv_activation_map(model)
    conv_stats: dict[str, dict[str, float]] = dict(extrema["convs"])
    layer_records: list[dict[str, Any]] = []
    fp32_prefixes = tuple(str(prefix) for prefix in (fp32_module_prefixes or ()))
    fp32_weight_prefixes = tuple(str(prefix) for prefix in (fp32_weight_prefixes or ()))
    fp32_activation_prefixes = tuple(str(prefix) for prefix in (fp32_activation_prefixes or ()))
    weight_bit_prefixes = {str(prefix): int(bits) for prefix, bits in (weight_bits_by_prefix or {}).items()}
    activation_bit_prefixes = {str(prefix): int(bits) for prefix, bits in (activation_bits_by_prefix or {}).items()}
    fp32_bypassed: list[str] = []
    fp32_weight_layers: list[str] = []
    fp32_activation_layers: list[str] = []
    mixed_precision_layers: list[str] = []

    def has_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
        return any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes)

    def bits_for(path: str, mapping: Mapping[str, int], default: int) -> int:
        matches = [(prefix, bits) for prefix, bits in mapping.items() if path == prefix or path.startswith(f"{prefix}.")]
        if not matches:
            return default
        prefix, bits = max(matches, key=lambda item: len(item[0]))
        if bits not in {8, 16}:
            raise ValueError(f"unsupported bitwidth {bits} for {prefix}")
        return int(bits)

    def wrap(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, nn.Conv2d):
                if has_prefix(path, fp32_prefixes):
                    fp32_bypassed.append(path)
                    continue
                fp32_weight = has_prefix(path, fp32_weight_prefixes)
                fp32_activation = has_prefix(path, fp32_activation_prefixes)
                if fp32_weight:
                    fp32_weight_layers.append(path)
                if fp32_activation:
                    fp32_activation_layers.append(path)
                if path not in conv_stats:
                    raise KeyError(f"no calibration observation for conv {path}")
                observed = conv_stats[path]
                activation_bits = bits_for(path, activation_bit_prefixes, 8)
                weight_bits = bits_for(path, weight_bit_prefixes, 8)
                input_bits = activation_bits
                output_bits = activation_bits
                _input_qmin, input_qmax = signed_range(input_bits)
                _output_qmin, output_qmax = signed_range(output_bits)
                input_exp = exponent_from_maxabs(observed["input_max_abs"], qmax=input_qmax)
                output_exp = exponent_from_maxabs(observed["output_max_abs"], qmax=output_qmax)
                _, weight_exps = quantize_weight(child.weight, granularity=weight_granularity, bits=weight_bits)
                if (input_bits, weight_bits, output_bits) != (8, 8, 8):
                    mixed_precision_layers.append(path)
                fake = H2FakeQuantConv(
                    child,
                    input_exp=input_exp,
                    output_exp=output_exp,
                    weight_exps=weight_exps,
                    activation=activations[path],
                    weight_granularity=weight_granularity,
                    input_bits=input_bits,
                    weight_bits=weight_bits,
                    output_bits=output_bits,
                    quantize_input=not fp32_activation,
                    quantize_weight_values=not fp32_weight,
                    quantize_output=not fp32_activation,
                )
                setattr(parent, name, fake)
                if weight_granularity == "per_tensor" and (input_bits, weight_bits, output_bits) == (8, 8, 8):
                    multiplier, shift = ratio_from_exponents(input_exp + weight_exps[0], output_exp)
                else:
                    multiplier, shift = (None, None)
                layer_records.append(
                    {
                        "name": path,
                        "input_exp": input_exp,
                        "weight_exps": weight_exps,
                        "output_exp": output_exp,
                        "activation": activations[path],
                        "input_bits": input_bits,
                        "weight_bits": weight_bits,
                        "output_bits": output_bits,
                        "input_scale": power_of_two(input_exp),
                        "weight_scale": power_of_two(weight_exps[0]),
                        "output_scale": power_of_two(output_exp),
                        "input_saturation": list(signed_range(input_bits)),
                        "weight_saturation": list(signed_range(weight_bits)),
                        "output_saturation": list(signed_range(output_bits)),
                        "multiplier": multiplier,
                        "shift": shift,
                        "groups": int(child.groups),
                        "kernel": list(child.kernel_size),
                        "fp32_weight": fp32_weight,
                        "fp32_activation": fp32_activation,
                    }
                )
            elif not isinstance(child, H2FakeQuantConv):
                wrap(child, f"{path}.")

    wrap(model, "")
    _replace_fused_activations(model)
    add_records = []
    for observed in extrema["adds"]:
        add_records.append(
            {
                "left_exp": exponent_from_maxabs(observed["left_max_abs"]),
                "right_exp": exponent_from_maxabs(observed["right_max_abs"]),
                "output_exp": exponent_from_maxabs(observed["output_max_abs"]),
            }
        )
    if quantize_neck_adds:
        model.neck = H2QuantizedFPN(model.neck, add_records)
    profile = {
        "numeric_contract": "NUMERIC_CONTRACT_V1",
        "numeric_contract_hash": NUMERIC_CONTRACT_HASH,
        "bitwidth": {"activation": 8, "weight": 8, "accumulator": 32, "requant_product": 64},
        "zero_point": 0,
        "rounding": "nearest_even",
        "saturation": [QMIN, QMAX],
        "scheme": scheme,
        "weight_granularity": weight_granularity,
        "activation_granularity": activation_granularity,
        "layers": layer_records,
        "fpn_adds": add_records,
        "fp32_bypass": {
            "module_prefixes": list(fp32_prefixes),
            "conv_layers": sorted(fp32_bypassed),
            "weight_prefixes": list(fp32_weight_prefixes),
            "weight_fp32_layers": sorted(fp32_weight_layers),
            "activation_prefixes": list(fp32_activation_prefixes),
            "activation_fp32_layers": sorted(fp32_activation_layers),
            "neck_adds_quantized": bool(quantize_neck_adds),
            "software_diagnostic_only": (
                bool(fp32_prefixes)
                or bool(fp32_weight_prefixes)
                or bool(fp32_activation_prefixes)
                or not bool(quantize_neck_adds)
            ),
        },
        "mixed_precision": {
            "weight_bits_by_prefix": dict(weight_bit_prefixes),
            "activation_bits_by_prefix": dict(activation_bit_prefixes),
            "conv_layers": sorted(mixed_precision_layers),
            "int16_semantics": {
                "signed": True,
                "symmetric": True,
                "zero_point": 0,
                "saturation": [INT16_QMIN, INT16_QMAX],
                "rounding": "nearest_even",
                "scale_derivation": "deterministic_power_of_two_from_train_calibration_or_weight_maxabs",
            },
            "hardware_cost_status": "NEEDS_MEASUREMENT" if mixed_precision_layers else "H4_COST_BINDING_COMPATIBLE",
        },
        "h2_aligned": not bool(mixed_precision_layers),
        "qat": False,
        "parity_ready": False,
    }
    return model, profile


def prepare_h2_ptq(
    model: nn.Module,
    calibration_images: Iterable[torch.Tensor],
    *,
    device: str,
    weight_granularity: str = "per_tensor",
    activation_granularity: str = "per_tensor",
    scheme: str = "symmetric",
    fp32_module_prefixes: Iterable[str] | None = None,
    fp32_weight_prefixes: Iterable[str] | None = None,
    fp32_activation_prefixes: Iterable[str] | None = None,
    weight_bits_by_prefix: Mapping[str, int] | None = None,
    activation_bits_by_prefix: Mapping[str, int] | None = None,
    quantize_neck_adds: bool = True,
) -> tuple[nn.Module, nn.Module, dict[str, Any], dict[str, Any]]:
    """Return folded FP32 model, H2 fake-quant model, extrema, and FPGA profile."""
    if model.training:
        raise ValueError("PTQ requires eval mode")
    folded = deepcopy(model).to(device).eval()
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(module, nn.BatchNorm2d) for module in folded.modules()):
        raise ValueError("unfolded BatchNorm remains after folding")
    extrema = observe_activation_extrema(folded, calibration_images, device=device)
    quantized = deepcopy(folded)
    quantized, profile = wrap_h2_ptq(
        quantized,
        extrema,
        weight_granularity=weight_granularity,
        activation_granularity=activation_granularity,
        scheme=scheme,
        fp32_module_prefixes=fp32_module_prefixes,
        fp32_weight_prefixes=fp32_weight_prefixes,
        fp32_activation_prefixes=fp32_activation_prefixes,
        weight_bits_by_prefix=weight_bits_by_prefix,
        activation_bits_by_prefix=activation_bits_by_prefix,
        quantize_neck_adds=quantize_neck_adds,
    )
    profile["bn_folding"] = {"enabled": True, "folded_pair_count": len(pairs), "pairs": pairs}
    return folded, quantized, extrema, profile
