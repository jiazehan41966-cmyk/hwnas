"""H2-aligned fake quantization modules for short QAT fine-tuning."""

from __future__ import annotations

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
from hwnas_fpga.deploy.h2_ptq import (
    QMAX,
    QMIN,
    aligned_add_dequant,
    conv_activation_map,
    exponent_from_maxabs,
    fold_batch_norms_inplace,
    observe_activation_extrema,
    power_of_two,
    product_power_of_two,
    quantize_weight,
    relu6_bound,
    ratio_from_exponents,
)


def _ste_dequantize(values: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    return values + (dequantized - values).detach()


def ste_quant_dequant_h2(values: torch.Tensor, exponent: int) -> torch.Tensor:
    scale = power_of_two(int(exponent))
    quantized = torch.round(values.to(torch.float32) / scale).clamp(QMIN, QMAX)
    return _ste_dequantize(values, quantized.to(values.dtype) * scale)


def ste_quant_dequant_bias(values: torch.Tensor | None, exponent: int) -> torch.Tensor | None:
    if values is None:
        return None
    scale = product_power_of_two(int(exponent))
    quantized = torch.round(values.to(torch.float64) / scale).clamp(-(1 << 31), (1 << 31) - 1)
    dequantized = quantized.to(values.dtype) * scale
    return _ste_dequantize(values, dequantized)


def _apply_activation_float(values: torch.Tensor, kind: str, output_exp: int) -> torch.Tensor:
    if kind == "linear":
        return values
    if kind == "relu":
        return values.clamp_min(0)
    if kind == "relu6":
        return values.clamp(0, relu6_bound(output_exp) * power_of_two(output_exp))
    raise ValueError(f"unsupported activation {kind}")


class H2QATConv(nn.Module):
    """Trainable Conv2d with H2 fixed-scale fake quantization."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        input_exp: int,
        output_exp: int,
        weight_exps: list[int],
        activation: str,
        weight_granularity: str,
    ) -> None:
        super().__init__()
        if weight_granularity != "per_tensor":
            raise ValueError("Phase5B admits only per-tensor H2 weights")
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.input_exp = int(input_exp)
        self.output_exp = int(output_exp)
        self.weight_exps = [int(value) for value in weight_exps]
        self.activation = str(activation)
        self.weight_granularity = str(weight_granularity)
        self.multiplier, self.shift = ratio_from_exponents(
            self.input_exp + self.weight_exps[0], self.output_exp
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_input = ste_quant_dequant_h2(inputs, self.input_exp)
        quantized_weight = ste_quant_dequant_h2(self.weight, self.weight_exps[0])
        quantized_bias = ste_quant_dequant_bias(
            self.bias,
            self.input_exp + self.weight_exps[0],
        )
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            quantized_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        output = ste_quant_dequant_h2(output, self.output_exp)
        return _apply_activation_float(output, self.activation, self.output_exp)


class H2QATFPN(nn.Module):
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
            quantized = aligned_add_dequant(
                fused[index],
                upsampled,
                spec["left_exp"],
                spec["right_exp"],
                spec["output_exp"],
            )
            fused[index] = _ste_dequantize(fused[index] + upsampled, quantized)
        return tuple(conv(feature) for conv, feature in zip(self.output_convs, fused))


def _replace_fused_activations(model: nn.Module) -> None:
    for parent in model.modules():
        children = list(parent.named_children())
        for index, (name, child) in enumerate(children):
            if isinstance(child, (nn.ReLU, nn.ReLU6)):
                previous = index - 1
                while previous >= 0 and isinstance(children[previous][1], nn.Identity):
                    previous -= 1
                if previous >= 0 and isinstance(children[previous][1], H2QATConv):
                    setattr(parent, name, nn.Identity())


def wrap_h2_qat(
    model: nn.Module,
    extrema: Mapping[str, Any],
    *,
    weight_granularity: str = "per_tensor",
    activation_granularity: str = "per_tensor",
    scheme: str = "symmetric",
) -> tuple[nn.Module, dict[str, Any]]:
    if scheme != "symmetric":
        raise ValueError("asymmetric FPGA policy is outside NUMERIC_CONTRACT_V1")
    if activation_granularity != "per_tensor":
        raise ValueError("official H2 activation scale is per-tensor")
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

    def wrap(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, nn.Conv2d):
                if path not in conv_stats:
                    raise KeyError(f"no calibration observation for conv {path}")
                observed = conv_stats[path]
                input_exp = exponent_from_maxabs(observed["input_max_abs"])
                output_exp = exponent_from_maxabs(observed["output_max_abs"])
                _, weight_exps = quantize_weight(child.weight, granularity=weight_granularity)
                fake = H2QATConv(
                    child,
                    input_exp=input_exp,
                    output_exp=output_exp,
                    weight_exps=weight_exps,
                    activation=activations[path],
                    weight_granularity=weight_granularity,
                )
                setattr(parent, name, fake)
                multiplier, shift = ratio_from_exponents(input_exp + weight_exps[0], output_exp)
                layer_records.append(
                    {
                        "name": path,
                        "input_exp": input_exp,
                        "weight_exps": weight_exps,
                        "output_exp": output_exp,
                        "activation": activations[path],
                        "input_scale": power_of_two(input_exp),
                        "weight_scale": power_of_two(weight_exps[0]),
                        "output_scale": power_of_two(output_exp),
                        "multiplier": multiplier,
                        "shift": shift,
                        "groups": int(child.groups),
                        "kernel": list(child.kernel_size),
                    }
                )
            elif not isinstance(child, H2QATConv):
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
    model.neck = H2QATFPN(model.neck, add_records)
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
        "h2_aligned": True,
        "qat": True,
        "scale_source": "phase5a_train_calibration_initialized_fixed_h2_exponents",
    }
    return model, profile


def prepare_h2_qat(
    model: nn.Module,
    calibration_images: Iterable[torch.Tensor],
    *,
    device: str,
    weight_granularity: str = "per_tensor",
    activation_granularity: str = "per_tensor",
    scheme: str = "symmetric",
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    if model.training:
        raise ValueError("QAT initialization requires eval mode")
    folded = deepcopy(model).to(device).eval()
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(module, nn.BatchNorm2d) for module in folded.modules()):
        raise ValueError("unfolded BatchNorm remains after folding")
    extrema = observe_activation_extrema(folded, calibration_images, device=device)
    qat_model = deepcopy(folded)
    qat_model, profile = wrap_h2_qat(
        qat_model,
        extrema,
        weight_granularity=weight_granularity,
        activation_granularity=activation_granularity,
        scheme=scheme,
    )
    profile["bn_folding"] = {"enabled": True, "folded_pair_count": len(pairs), "pairs": pairs}
    return qat_model, extrema, profile
