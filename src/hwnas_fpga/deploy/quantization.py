"""INT8 weight quantization helpers for deployment packaging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn

from hwnas_fpga.deploy.fixed_point import (
    FixedPointContract,
    quantize_bias_int32,
)
from hwnas_fpga.deploy.inference import load_checkpoint_model


@dataclass(frozen=True)
class QuantizationConfig:
    bit_width: int = 8
    scheme: str = "symmetric"
    quantize_bias: bool = False
    target_modules: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
    input_scale: float = 1.0 / 127.0
    output_scale: float = 1.0 / 127.0
    activation_scales: Optional[dict[str, float]] = None


def quantize_tensor_symmetric(
    tensor: torch.Tensor,
    *,
    bit_width: int = 8,
) -> tuple[torch.Tensor, float]:
    """Quantize a tensor to signed INT format using symmetric scaling."""
    if bit_width != 8:
        raise ValueError("Only INT8 quantization is supported in the current deployment path")
    if tensor.numel() == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0

    tensor_fp32 = tensor.detach().to(torch.float32)
    max_abs = float(tensor_fp32.abs().max().item())
    if max_abs == 0.0:
        return torch.zeros_like(tensor_fp32, dtype=torch.int8), 1.0

    qmax = 127.0
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor_fp32 / scale), -qmax, qmax).to(torch.int8)
    return quantized, float(scale)


def build_quantized_weight_package(
    model: nn.Module,
    *,
    architecture: Optional[dict[str, Any]] = None,
    candidate: Optional[dict[str, Any]] = None,
    class_names: Optional[Iterable[str]] = None,
    config: Optional[QuantizationConfig] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an INT8 deployment package from a PyTorch model."""
    cfg = config or QuantizationConfig()
    if cfg.scheme != "symmetric":
        raise ValueError("Only symmetric quantization is supported")

    quantized_weights: dict[str, torch.Tensor] = {}
    quantized_biases: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    layers: list[dict[str, Any]] = []
    tensor_summaries: list[dict[str, Any]] = []
    total_original_bytes = 0
    total_quantized_bytes = 0

    module_lookup = dict(model.named_modules())
    state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        module_name, _, param_name = name.rpartition(".")
        module = module_lookup.get(module_name)
        should_quantize = (
            module is not None
            and isinstance(module, cfg.target_modules)
            and (param_name == "weight" or (cfg.quantize_bias and param_name == "bias"))
        )
        if not should_quantize:
            continue

        if param_name == "bias":
            # Bias is never stored as INT8. It is an INT32 accumulator-domain
            # tensor whose scale is input_scale * weight_scale.
            weight_name = f"{module_name}.weight"
            weight_scale = scales.get(weight_name)
            if weight_scale is None:
                raise ValueError(f"weight scale missing before bias quantization: {name}")
            quantized_tensor = quantize_bias_int32(
                tensor,
                input_scale=cfg.input_scale,
                weight_scale=weight_scale,
            )
            quantized_biases[name] = quantized_tensor.cpu()
            scales[name] = float(cfg.input_scale * weight_scale)
        else:
            quantized_tensor, scale = quantize_tensor_symmetric(tensor, bit_width=cfg.bit_width)
            quantized_weights[name] = quantized_tensor.cpu()
            scales[name] = scale
        original_bytes = int(tensor.numel() * tensor.element_size())
        quantized_bytes = int(quantized_tensor.numel() * quantized_tensor.element_size())
        total_original_bytes += original_bytes
        total_quantized_bytes += quantized_bytes
        tensor_summaries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "module_type": module.__class__.__name__,
                "scale": scale,
                "dtype": str(quantized_tensor.dtype),
                "original_bytes": original_bytes,
                "quantized_bytes": quantized_bytes,
            }
        )

    summary = {
        "schema_version": 2 if cfg.quantize_bias else 1,
        "bit_width": cfg.bit_width,
        "scheme": cfg.scheme,
        "fixed_point_contract": FixedPointContract().to_dict(),
        "zero_point": 0,
        "input_scale": float(cfg.input_scale),
        "output_scale": float(cfg.output_scale),
        "num_quantized_tensors": len(tensor_summaries),
        "original_weight_bytes": total_original_bytes,
        "quantized_weight_bytes": total_quantized_bytes,
        "compression_ratio": (
            float(total_original_bytes) / float(total_quantized_bytes)
            if total_quantized_bytes > 0
            else 1.0
        ),
        "tensors": tensor_summaries,
        "bias_dtype": "int32" if cfg.quantize_bias else None,
        "activation_requantization": "per_output_tensor_scale" if cfg.quantize_bias else None,
        "parity_ready": False,
        "claim_boundary": (
            "Legacy weight-only package; parity_ready=false."
            if not cfg.quantize_bias
            else "INT8 weights plus INT32 bias metadata are packaged, but software/HLS "
            "layer parity must pass before deployment claims are allowed."
        ),
    }
    if cfg.quantize_bias:
        for module_name, module in module_lookup.items():
            if not isinstance(module, cfg.target_modules):
                continue
            weight_name = f"{module_name}.weight"
            if weight_name not in quantized_weights:
                continue
            weight_scale = scales[weight_name]
            output_scale = float((cfg.activation_scales or {}).get(module_name, cfg.output_scale))
            if output_scale <= 0:
                raise ValueError(f"output scale must be positive for {module_name}")
            layers.append(
                {
                    "name": module_name,
                    "op": module.__class__.__name__,
                    "weight_key": weight_name,
                    "bias_key": f"{module_name}.bias" if module.bias is not None else None,
                    "input_scale": float(cfg.input_scale),
                    "weight_scale": float(weight_scale),
                    "output_scale": output_scale,
                    "stride": list(module.stride) if isinstance(module, nn.Conv2d) else None,
                    "padding": list(module.padding) if isinstance(module, nn.Conv2d) else None,
                    "dilation": list(module.dilation) if isinstance(module, nn.Conv2d) else None,
                    "groups": int(module.groups) if isinstance(module, nn.Conv2d) else None,
                }
            )
        summary["layers"] = layers
        summary["layer_count"] = len(layers)
    package = {
        # Keep the old format label for the legacy default so existing readers
        # remain compatible. A bias-enabled package is explicitly versioned v2.
        "format": "hwnas_fpga_int8_weights_v2" if cfg.quantize_bias else "hwnas_fpga_int8_weights_v1",
        "schema_version": 2 if cfg.quantize_bias else 1,
        "architecture": architecture,
        "candidate": candidate,
        "class_names": list(class_names or []),
        "quantization": summary,
        "weights": quantized_weights,
        "biases": quantized_biases,
        "scales": scales,
        "layers": layers,
    }
    return package, summary


def export_checkpoint_quantized_weights(
    checkpoint_path: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    device: str = "cpu",
    config: Optional[QuantizationConfig] = None,
) -> tuple[Path, dict[str, Any]]:
    """Load a checkpoint, quantize supported weights, and save a deployable package."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    model, architecture, payload, class_names = load_checkpoint_model(checkpoint, device=device)
    package, summary = build_quantized_weight_package(
        model,
        architecture=architecture.to_dict(),
        candidate=payload.get("candidate") or payload.get("source_candidate"),
        class_names=class_names,
        config=config,
    )

    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else checkpoint.parent / "quantized_weights_int8.pt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, target)

    summary_path = target.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target, summary
