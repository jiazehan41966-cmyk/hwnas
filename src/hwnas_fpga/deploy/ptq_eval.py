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

import random
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from hwnas_fpga.deploy.quantization import quantize_tensor_symmetric
from hwnas_fpga.deploy.fixed_point import FixedPointContract


def fold_batch_norms_inplace(model: nn.Module) -> list[dict[str, str]]:
    """Fold adjacent Conv2d/BatchNorm2d pairs while preserving their paths."""

    model.eval()
    folded: list[dict[str, str]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        children = list(parent.named_children())
        for index in range(len(children) - 1):
            conv_name, conv = children[index]
            bn_name, bn = children[index + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                setattr(parent, conv_name, fused)
                setattr(parent, bn_name, nn.Identity())
                folded.append(
                    {
                        "conv": f"{prefix}{conv_name}",
                        "bn": f"{prefix}{bn_name}",
                    }
                )
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Identity):
                visit(child, f"{prefix}{name}.")

    visit(model, "")
    return folded


def restore_identity_bn_export_model(
    finalized_folded_model: nn.Module,
    original_template: nn.Module,
    folded_pairs: Sequence[Mapping[str, str]],
) -> nn.Module:
    """Encode fused Conv+BN parameters in the original checkpoint topology.

    The restored BN is an exact identity except that its beta stores the fused
    convolution bias. Folding it again in the HLS exporter recovers the same
    effective weight and bias.
    """

    source_modules = dict(finalized_folded_model.named_modules())
    target_modules = dict(original_template.named_modules())
    paired_conv = {str(item["conv"]): str(item["bn"]) for item in folded_pairs}
    with torch.no_grad():
        for name, target in target_modules.items():
            source = source_modules.get(name)
            if not isinstance(target, (nn.Conv2d, nn.Linear)):
                continue
            if not isinstance(source, target.__class__):
                raise ValueError(f"missing finalized module for {name}")
            target.weight.copy_(source.weight)
            if name in paired_conv:
                bn = target_modules.get(paired_conv[name])
                if not isinstance(bn, nn.BatchNorm2d):
                    raise ValueError(f"missing original BatchNorm for {name}")
                if target.bias is not None:
                    target.bias.zero_()
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)
                bn.weight.fill_((1.0 + float(bn.eps)) ** 0.5)
                if source.bias is None:
                    bn.bias.zero_()
                else:
                    bn.bias.copy_(source.bias)
                bn.num_batches_tracked.zero_()
            elif target.bias is not None and source.bias is not None:
                target.bias.copy_(source.bias)
    return original_template


def quantize_dequantize(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return tensor
    return torch.clamp(torch.round(tensor / scale), -127.0, 127.0) * scale


def stratified_calibration_indices(
    labels: Sequence[int],
    candidate_indices: Sequence[int],
    *,
    max_samples: int = 512,
    seed: int = 42,
) -> list[int]:
    """Select a deterministic, approximately proportional training-only subset."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in candidate_indices:
        by_class[int(labels[int(index)])].append(int(index))
    if not by_class:
        raise ValueError("candidate_indices is empty")
    rng = random.Random(seed)
    for values in by_class.values():
        rng.shuffle(values)
    target = min(int(max_samples), sum(len(values) for values in by_class.values()))
    selected: list[int] = []
    # First preserve rare-class coverage, then fill proportionally by cycling
    # through classes according to remaining population.
    for label in sorted(by_class):
        if by_class[label] and len(selected) < target:
            selected.append(by_class[label].pop())
    while len(selected) < target:
        available = [
            (label, len(values)) for label, values in by_class.items() if values
        ]
        if not available:
            break
        total = sum(count for _, count in available)
        draw = rng.randrange(total)
        cumulative = 0
        chosen = available[-1][0]
        for label, count in available:
            cumulative += count
            if draw < cumulative:
                chosen = label
                break
        selected.append(by_class[chosen].pop())
    return sorted(selected)


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
    folded_pairs = fold_batch_norms_inplace(model)
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
        "fixed_point_contract": FixedPointContract().to_dict(),
        "bn_folding": {
            "enabled": True,
            "folded_pair_count": len(folded_pairs),
            "pairs": folded_pairs,
        },
        "num_quantized_ops": len(wrapped),
        "num_calibration_batches": batches_used,
        "calibration_source": "training_side_only",
        "activation_scales": [float(op.act_scale.item()) for op in wrapped],
        "weight_scales": [op.weight_scale for op in wrapped],
        "claim_boundary": (
            "PTQ float simulation; matches exported weight quantization, "
            "not bit-exact vs the HLS fixed-point pipeline"
        ),
    }
