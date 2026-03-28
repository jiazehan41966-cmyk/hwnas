"""INT8 weight quantization helpers for deployment packaging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn

from hwnas_fpga.deploy.inference import load_checkpoint_model


@dataclass(frozen=True)
class QuantizationConfig:
    bit_width: int = 8
    scheme: str = "symmetric"
    quantize_bias: bool = False
    target_modules: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)


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
    scales: dict[str, float] = {}
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
        "bit_width": cfg.bit_width,
        "scheme": cfg.scheme,
        "num_quantized_tensors": len(tensor_summaries),
        "original_weight_bytes": total_original_bytes,
        "quantized_weight_bytes": total_quantized_bytes,
        "compression_ratio": (
            float(total_original_bytes) / float(total_quantized_bytes)
            if total_quantized_bytes > 0
            else 1.0
        ),
        "tensors": tensor_summaries,
    }
    package = {
        "format": "hwnas_fpga_int8_weights_v1",
        "architecture": architecture,
        "candidate": candidate,
        "class_names": list(class_names or []),
        "quantization": summary,
        "weights": quantized_weights,
        "scales": scales,
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
