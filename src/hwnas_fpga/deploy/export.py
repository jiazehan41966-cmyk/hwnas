"""Model export utilities for FPGA deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


def _default_dynamic_axes() -> dict[str, dict[int, str]]:
    return {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    opset_version: int = 13,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
) -> Path:
    """Export PyTorch model to ONNX format."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = _default_dynamic_axes()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    if verbose:
        print(f"[Export] ONNX model saved to: {output_path}")

    return output_path


def get_model_info(model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 224, 224)) -> dict[str, Any]:
    """Get model information including parameter count and FLOPs estimate."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 * 1024)

    model.eval()
    macs = 0

    def hook_fn(module, input, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += kernel_ops * out_h * out_w * module.out_channels
        elif isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(torch.randn(*input_shape))

    for hook in hooks:
        hook.remove()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size_mb,
        "estimated_macs": macs,
        "input_shape": input_shape,
    }


def export_model_to_onnx(
    model: nn.Module,
    export_path: str | Path,
    *,
    input_shape: Tuple[int, ...] = (1, 1, 224, 224),
    opset_version: int = 13,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Path, dict[str, Any]]:
    """Export a ready-built model and save sidecar metadata."""
    resolved_dynamic_axes = dynamic_axes or _default_dynamic_axes()
    onnx_path = export_to_onnx(
        model,
        export_path,
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_axes=resolved_dynamic_axes,
        verbose=verbose,
    )
    payload = {
        "onnx_path": str(onnx_path),
        "input_shape": list(input_shape),
        "opset_version": int(opset_version),
        "dynamic_axes": resolved_dynamic_axes,
        "model_info": get_model_info(model, input_shape=input_shape),
    }
    if metadata:
        payload.update(metadata)
    onnx_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return onnx_path, payload


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    *,
    export_path: str | Path,
    device: str = "cpu",
    image_size_override: Optional[int] = None,
    input_channels_override: Optional[int] = None,
    opset_version: int = 13,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Load a searched/retrained checkpoint and export it to ONNX."""
    from hwnas_fpga.deploy.inference import load_checkpoint_model, resolve_inference_settings

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    model, architecture, payload, class_names = load_checkpoint_model(checkpoint, device=device)
    image_size, input_channels = resolve_inference_settings(
        checkpoint,
        image_size_override=image_size_override,
        input_channels_override=input_channels_override,
    )
    return export_model_to_onnx(
        model,
        export_path,
        input_shape=(1, input_channels, image_size, image_size),
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        metadata={
            "checkpoint_path": str(checkpoint),
            "device": device,
            "class_names": class_names,
            "architecture": architecture.to_dict(),
            "checkpoint_metrics": payload.get("metrics", {}),
            "export_source": "checkpoint",
        },
    )
