"""Fail-closed software INT8 graph reference for the HLS contract.

The reference deliberately operates on integer tensors after the input
quantization step.  It is not a fake-quantized FP32 model: unsupported graph
operators raise :class:`UnsupportedIntegerOperatorError` so a missing integer
implementation cannot be mistaken for parity evidence.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.fx as fx
import torch.nn as nn

from hwnas_fpga.deploy.fixed_point import (
    avg_pool2d_int_reference,
    conv2d_int_reference,
    linear_int_reference,
    max_pool2d_int_reference,
    quantize_symmetric_int8,
    relu_int_reference,
    requantize_per_output_int8,
    residual_add_int_reference,
)


class UnsupportedIntegerOperatorError(RuntimeError):
    """Raised when a model graph contains an operator without integer semantics."""


def quantization_spec_sha256(spec: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        spec, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IntegerReferenceResult:
    output_int8: torch.Tensor
    output_scale: float
    layer_traces: tuple[dict[str, Any], ...]

    @property
    def output_float(self) -> torch.Tensor:
        return self.output_int8.to(torch.float32) * float(self.output_scale)


def _first_value(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _as_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected a scalar or pair, got {value!r}")
    return int(value[0]), int(value[1])


def _resolve_arg(arg: Any, env: Mapping[fx.Node, Any]) -> Any:
    if isinstance(arg, fx.Node):
        return env[arg]
    if isinstance(arg, tuple):
        return tuple(_resolve_arg(item, env) for item in arg)
    if isinstance(arg, list):
        return [_resolve_arg(item, env) for item in arg]
    return arg


def _layer_index(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("INT8 quantization spec must contain non-empty layers")
    return {str(layer["name"]): layer for layer in layers}


def _package_tensor(
    package: Mapping[str, Any], group: str, key: str, *, device: torch.device | None = None
) -> torch.Tensor:
    tensors = package.get(group) or {}
    if key not in tensors:
        raise ValueError(f"quantization package is missing {group}[{key!r}]")
    tensor = tensors[key]
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _trace(
    *,
    layer: str,
    input_kind: str,
    tensor: torch.Tensor,
    simulator_sha256: str,
    spec_sha256: str,
) -> dict[str, Any]:
    return {
        "layer": str(layer),
        "input_kind": str(input_kind),
        "element_count": int(tensor.numel()),
        "mismatch_count": 0,
        "simulator_sha256": simulator_sha256,
        "quantization_spec_sha256": spec_sha256,
    }


def compare_integer_tensors(
    reference: torch.Tensor,
    simulator: torch.Tensor,
    *,
    layer: str,
    input_kind: str,
    quantization_spec_sha256_value: str,
) -> dict[str, Any]:
    """Create one auditable parity row for a simulator/HLS tensor pair."""
    if reference.shape != simulator.shape:
        mismatch = max(int(reference.numel()), int(simulator.numel()))
        element_count = mismatch
    else:
        element_count = int(reference.numel())
        mismatch = int((reference.to(torch.int64) != simulator.to(torch.int64)).sum().item())
    digest = hashlib.sha256(simulator.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "layer": str(layer),
        "input_kind": str(input_kind),
        "element_count": element_count,
        "mismatch_count": mismatch,
        "simulator_sha256": digest,
        "quantization_spec_sha256": str(quantization_spec_sha256_value),
    }


class _IntegerInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: nn.Module,
        package: Mapping[str, Any],
        spec: Mapping[str, Any],
        *,
        input_tensor: torch.Tensor,
        input_kind: str,
    ) -> None:
        self.root_module = module
        self.package = package
        self.spec = spec
        self.layer_specs = _layer_index(spec)
        self.input_kind = input_kind
        self.input_tensor = input_tensor
        self.spec_sha256 = quantization_spec_sha256(spec)
        self.traces: list[dict[str, Any]] = []
        super().__init__(module)

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        digest = hashlib.sha256(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        self.traces.append(
            _trace(
                layer=name,
                input_kind=self.input_kind,
                tensor=tensor,
                simulator_sha256=digest,
                spec_sha256=self.spec_sha256,
            )
        )

    def _quantized_module(self, name: str, module: nn.Module, value: tuple[torch.Tensor, float]):
        if name not in self.layer_specs:
            raise UnsupportedIntegerOperatorError(
                f"missing integer layer specification for module {name!r}"
            )
        inputs, input_scale = value
        layer = self.layer_specs[name]
        expected_scale = float(layer["input_scale"])
        if abs(float(input_scale) - expected_scale) > 1e-12:
            raise UnsupportedIntegerOperatorError(
                f"scale transition into {name!r} is unspecified: "
                f"{input_scale} != {expected_scale}"
            )
        weight = _package_tensor(
            self.package,
            "weights",
            str(layer["weight_key"]),
            device=inputs.device,
        )
        bias_key = layer.get("bias_key")
        bias = (
            _package_tensor(
                self.package,
                "biases",
                str(bias_key),
                device=inputs.device,
            )
            if bias_key
            else None
        )
        if isinstance(module, nn.Conv2d):
            accumulator = conv2d_int_reference(
                inputs,
                weight,
                bias,
                stride=_as_pair(module.stride)[0],
                padding=_as_pair(module.padding)[0],
                groups=int(module.groups),
            )
            axis = 1
        elif isinstance(module, nn.Linear):
            original_shape = inputs.shape
            flat = inputs.reshape(-1, inputs.shape[-1])
            accumulator = linear_int_reference(flat, weight, bias)
            accumulator = accumulator.reshape(*original_shape[:-1], accumulator.shape[-1])
            axis = accumulator.ndim - 1
        else:
            raise UnsupportedIntegerOperatorError(
                f"module {name!r} ({module.__class__.__name__}) is not integer-supported"
            )
        output = requantize_per_output_int8(
            accumulator,
            input_scale=expected_scale,
            weight_scale=layer["weight_scale"],
            output_scale=float(layer["output_scale"]),
            channel_axis=axis,
        )
        self._record(name, output)
        return output, float(layer["output_scale"])

    def run_node(self, n: fx.Node) -> Any:  # noqa: C901 - explicit fail-closed dispatch
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        if n.op == "placeholder":
            # ``torch.fx.Interpreter`` normally consumes placeholder values
            # through its private argument iterator.  This custom dispatcher
            # deliberately keeps the input explicit so the integer boundary is
            # auditable and independent of FX internals.
            inputs = self.input_tensor
            if not isinstance(inputs, torch.Tensor):
                raise UnsupportedIntegerOperatorError("integer reference expects one tensor input")
            input_scale = float(self.spec.get("input_scale", 1.0 / 127.0))
            return quantize_symmetric_int8(inputs, scale=input_scale), input_scale
        if n.op == "get_attr":
            return getattr(self.module, n.target)
        if n.op == "call_module":
            module = self.fetch_attr(n.target)
            value = _first_value(args[0]) if args else None
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return self._quantized_module(str(n.target), module, value)
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                tensor, scale = value
                result = relu_int_reference(tensor)
                self._record(str(n.target), result)
                return result, scale
            if isinstance(module, nn.MaxPool2d):
                tensor, scale = value
                result = max_pool2d_int_reference(
                    tensor,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode,
                )
                self._record(str(n.target), result)
                return result, scale
            if isinstance(module, nn.AvgPool2d):
                tensor, scale = value
                result = avg_pool2d_int_reference(
                    tensor,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    ceil_mode=module.ceil_mode,
                    count_include_pad=module.count_include_pad,
                )
                self._record(str(n.target), result)
                return result, scale
            if isinstance(module, nn.AdaptiveAvgPool2d):
                tensor, scale = value
                if module.output_size != (1, 1):
                    raise UnsupportedIntegerOperatorError(
                        f"AdaptiveAvgPool2d output_size={module.output_size!r} is unsupported"
                    )
                denominator = int(tensor.shape[-2] * tensor.shape[-1])
                summed = tensor.to(torch.int64).sum(dim=(-1, -2), keepdim=True)
                result = torch.round(summed.to(torch.float64) / denominator).clamp(-127, 127).to(torch.int8)
                self._record(str(n.target), result)
                return result, scale
            if isinstance(module, (nn.Identity, nn.Dropout)):
                return value
            if isinstance(module, nn.Flatten):
                tensor, scale = value
                result = torch.flatten(tensor, module.start_dim, module.end_dim)
                return result, scale
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                raise UnsupportedIntegerOperatorError(
                    f"batch normalization {n.target!r} must be folded before integer export"
                )
            raise UnsupportedIntegerOperatorError(
                f"module {n.target!r} ({module.__class__.__name__}) is unsupported"
            )
        if n.op == "call_function":
            if n.target in (operator.add, torch.add):
                left, left_scale = args[0]
                right, right_scale = args[1]
                if abs(float(left_scale) - float(right_scale)) > 1e-12:
                    raise UnsupportedIntegerOperatorError(
                        "residual add with different scales requires an explicit add scale"
                    )
                result = residual_add_int_reference(left, right)
                self._record("add", result)
                return result, left_scale
            if n.target in (torch.relu, torch.nn.functional.relu):
                tensor, scale = args[0]
                result = relu_int_reference(tensor)
                self._record("relu", result)
                return result, scale
            if n.target is torch.flatten:
                tensor, scale = args[0]
                return torch.flatten(tensor, *args[1:]), scale
            raise UnsupportedIntegerOperatorError(f"function {n.target!r} is unsupported")
        if n.op == "call_method":
            tensor, scale = args[0]
            if n.target in {"flatten", "reshape", "view", "contiguous"}:
                method_args = args[1:]
                return getattr(tensor, n.target)(*method_args), scale
            raise UnsupportedIntegerOperatorError(f"method {n.target!r} is unsupported")
        if n.op == "output":
            value = args[0]
            return _first_value(value)
        raise UnsupportedIntegerOperatorError(f"graph node operation {n.op!r} is unsupported")


@torch.no_grad()
def run_integer_reference(
    model: nn.Module,
    inputs: torch.Tensor,
    package: Mapping[str, Any],
    *,
    input_kind: str = "real_sample",
) -> IntegerReferenceResult:
    """Execute a model through the integer graph and return per-layer traces."""
    spec = package.get("quantization") or package
    if int(spec.get("schema_version", 0)) < 2:
        raise UnsupportedIntegerOperatorError(
            "legacy weight-only quantization packages are not parity-ready"
        )
    model = model.eval()
    graph = fx.symbolic_trace(model)
    interpreter = _IntegerInterpreter(
        graph,
        package,
        spec,
        input_tensor=inputs,
        input_kind=input_kind,
    )
    result = interpreter.run(inputs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise UnsupportedIntegerOperatorError("integer graph did not return (tensor, scale)")
    output, output_scale = result
    if not isinstance(output, torch.Tensor):
        raise UnsupportedIntegerOperatorError("integer graph output is not a tensor")
    return IntegerReferenceResult(
        output_int8=output.to(torch.int8),
        output_scale=float(output_scale),
        layer_traces=tuple(interpreter.traces),
    )


class IntegerReferenceClassifier(nn.Module):
    """``nn.Module`` adapter exposing integer-reference logits."""

    def __init__(
        self,
        model: nn.Module,
        package: Mapping[str, Any],
        *,
        input_kind: str = "real_sample",
    ) -> None:
        super().__init__()
        self.model = model
        self.package = dict(package)
        self.input_kind = input_kind

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = run_integer_reference(
            self.model,
            inputs,
            self.package,
            input_kind=self.input_kind,
        )
        return result.output_float
