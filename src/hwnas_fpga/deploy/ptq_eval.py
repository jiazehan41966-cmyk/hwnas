"""Post-training INT8 quantization simulation for accuracy evaluation.

Closes audit gate #5 ("calibrated INT8 validation accuracy before full
validation-set board inference") on the software side:

- weights: per-tensor symmetric INT8 quantize-dequantize, the same scheme as
  the deployment packaging in :mod:`hwnas_fpga.deploy.quantization`;
- activations: per-tensor symmetric INT8 with max-abs scales calibrated on
  batches supplied by the caller (use training-side data, never the outer
  validation fold).

Claim boundary: this simulates PTQ arithmetic in floating point. It matches
the exported weight quantization exactly, but it is not bit-exact against the
HLS fixed-point pipeline; the board chain must still verify numeric parity.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from hwnas_fpga.deploy.quantization import quantize_tensor_symmetric


def quantize_dequantize(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return tensor
    return torch.clamp(torch.round(tensor / scale), -127.0, 127.0) * scale


class FakeQuantizedOp(nn.Module):
    """Wraps a Conv2d/Linear with weight fake-quant and calibrated input fake-quant."""

    def __init__(self, module: nn.Module, *, bit_width: int = 8) -> None:
        super().__init__()
        if bit_width != 8:
            raise ValueError("Only INT8 PTQ simulation is supported")
        self.module = module
        self.register_buffer("act_max_abs", torch.zeros(()))
        self.register_buffer("act_scale", torch.zeros(()))
        self.calibrating = True

        with torch.no_grad():
            quantized, scale = quantize_tensor_symmetric(module.weight.data)
            module.weight.data.copy_(quantized.to(torch.float32) * scale)
        self.weight_scale = float(scale)

    def freeze_calibration(self) -> None:
        self.calibrating = False
        max_abs = float(self.act_max_abs.item())
        self.act_scale.fill_(max_abs / 127.0 if max_abs > 0 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                batch_max = x.detach().abs().max()
                torch.maximum(self.act_max_abs, batch_max, out=self.act_max_abs)
        else:
            x = quantize_dequantize(x, float(self.act_scale.item()))
        return self.module(x)


def _wrap_quantizable_modules(model: nn.Module, *, bit_width: int) -> list[FakeQuantizedOp]:
    wrapped: list[FakeQuantizedOp] = []

    def wrap(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                fake_quant = FakeQuantizedOp(child, bit_width=bit_width)
                setattr(parent, name, fake_quant)
                wrapped.append(fake_quant)
            elif not isinstance(child, FakeQuantizedOp):
                wrap(child)

    wrap(model)
    return wrapped


@torch.no_grad()
def apply_ptq(
    model: nn.Module,
    calibration_loader,
    *,
    num_calibration_batches: int = 16,
    bit_width: int = 8,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Convert ``model`` in place to a PTQ-simulated INT8 model.

    Calibration batches must come from training-side data. Returns metadata
    with per-op activation scales for the deployment record.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    wrapped = _wrap_quantizable_modules(model, bit_width=bit_width)
    if not wrapped:
        raise ValueError("model contains no quantizable Conv2d/Linear modules")

    batches_used = 0
    for inputs, _ in calibration_loader:
        if batches_used >= num_calibration_batches:
            break
        model(inputs.to(device))
        batches_used += 1
    if batches_used == 0:
        raise ValueError("calibration loader yielded no batches")

    for op in wrapped:
        op.freeze_calibration()

    return {
        "bit_width": bit_width,
        "scheme": "symmetric_per_tensor",
        "num_quantized_ops": len(wrapped),
        "num_calibration_batches": batches_used,
        "activation_scales": [float(op.act_scale.item()) for op in wrapped],
        "weight_scales": [op.weight_scale for op in wrapped],
        "claim_boundary": (
            "PTQ float simulation; matches exported weight quantization, "
            "not bit-exact vs the HLS fixed-point pipeline"
        ),
    }
