"""Matching-scheme QAT fallback for the INT8 deployment gate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_ste(tensor: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device)
    safe_scale = scale_tensor.clamp_min(torch.finfo(tensor.dtype).eps)
    quantized = torch.clamp(torch.round(tensor / safe_scale), -127.0, 127.0)
    recovered = quantized * safe_scale
    return tensor + (recovered - tensor).detach()


class QATQuantizedOp(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise TypeError("QATQuantizedOp supports Conv2d and Linear")
        self.module = module
        self.register_buffer("act_max_abs", torch.zeros(()))
        self.register_buffer("act_scale", torch.zeros(()))
        self.calibrating = True

    def freeze_calibration(self) -> None:
        self.calibrating = False
        maximum = float(self.act_max_abs.item())
        self.act_scale.fill_(maximum / 127.0 if maximum > 0 else 1.0)

    def quantized_weight(self) -> torch.Tensor:
        maximum = self.module.weight.detach().abs().max()
        scale = (maximum / 127.0).clamp_min(
            torch.finfo(self.module.weight.dtype).eps
        )
        return fake_quant_ste(self.module.weight, scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                torch.maximum(
                    self.act_max_abs,
                    inputs.detach().abs().max(),
                    out=self.act_max_abs,
                )
            quantized_inputs = inputs
        else:
            quantized_inputs = fake_quant_ste(inputs, self.act_scale)
        weight = self.quantized_weight()
        if isinstance(self.module, nn.Conv2d):
            return F.conv2d(
                quantized_inputs,
                weight,
                self.module.bias,
                self.module.stride,
                self.module.padding,
                self.module.dilation,
                self.module.groups,
            )
        return F.linear(quantized_inputs, weight, self.module.bias)


def _wrap(parent: nn.Module, wrapped: list[QATQuantizedOp]) -> None:
    for name, child in list(parent.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            replacement = QATQuantizedOp(child)
            setattr(parent, name, replacement)
            wrapped.append(replacement)
        elif not isinstance(child, QATQuantizedOp):
            _wrap(child, wrapped)


@torch.no_grad()
def prepare_qat(
    model: nn.Module,
    calibration_loader,
    *,
    device: Optional[str] = None,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    wrapped: list[QATQuantizedOp] = []
    _wrap(model, wrapped)
    if not wrapped:
        raise ValueError("model contains no QAT-compatible modules")
    batches = 0
    for inputs, _ in calibration_loader:
        model(inputs.to(device))
        batches += 1
    if batches == 0:
        raise ValueError("QAT calibration loader yielded no batches")
    for module in wrapped:
        module.freeze_calibration()
    return {
        "scheme": "symmetric_per_tensor_ste",
        "num_quantized_ops": len(wrapped),
        "num_calibration_batches": batches,
        "activation_scales": [float(module.act_scale.item()) for module in wrapped],
    }


def finalize_qat(model: nn.Module) -> nn.Module:
    """Return a plain model with learned weights snapped to the INT8 grid."""
    finalized = deepcopy(model).cpu()

    def unwrap(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, QATQuantizedOp):
                module = child.module
                with torch.no_grad():
                    maximum = module.weight.detach().abs().max()
                    scale = (maximum / 127.0).clamp_min(
                        torch.finfo(module.weight.dtype).eps
                    )
                    quantized = torch.clamp(
                        torch.round(module.weight / scale),
                        -127,
                        127,
                    )
                    module.weight.copy_(quantized * scale)
                setattr(parent, name, module)
            else:
                unwrap(child)

    unwrap(finalized)
    return finalized

