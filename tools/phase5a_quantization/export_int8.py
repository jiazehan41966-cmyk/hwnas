"""Integer-graph and ONNX export for Q(x). Does not change detector topology."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import json
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "tools/fpga_v0_1"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv, H2QuantizedFPN
try:
    from hwnas_fpga.deploy.h2_qat import H2QATConv, H2QATFPN
except Exception:  # pragma: no cover - older remote snapshots may lack QAT helpers.
    H2QATConv = None
    H2QATFPN = None

H2_CONV_TYPES = (H2FakeQuantConv,) + (() if H2QATConv is None else (H2QATConv,))
H2_FPN_TYPES = (H2QuantizedFPN,) + (() if H2QATFPN is None else (H2QATFPN,))

H2_FPGA_OPS = {
    "Conv": "conv",
    "Relu": "relu",
    "Clip": "relu6_or_sat",
    "Add": "aligned_add",
    "Resize": "nearest_upsample",
    "Upsample": "nearest_upsample",
    "Identity": "identity_skip",
    "Constant": "constant",
}
ONNX_SOFTWARE_ONLY = {
    "Round",
    "Mul",
    "Div",
    "Cast",
    "Sub",
    "Pad",
    "Unsqueeze",
    "Squeeze",
    "Shape",
    "Gather",
    "Concat",
    "Reshape",
    "Transpose",
    "QuantizeLinear",
    "DequantizeLinear",
}


class IntegerGraphError(RuntimeError):
    pass


def classify_conv(module: H2FakeQuantConv) -> str:
    weight = module.int_weight if hasattr(module, "int_weight") else module.weight
    if module.groups == weight.shape[0] and weight.shape[1] == 1:
        return "dw_conv"
    if tuple(weight.shape[-2:]) == (1, 1):
        return "pw_conv"
    return "conv"


def operator_coverage(model: nn.Module) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    unsupported: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, H2_CONV_TYPES):
            counts[classify_conv(module)] += 1
            counts[module.activation] += 1
        elif isinstance(module, H2_FPN_TYPES):
            counts["aligned_add"] += len(module.add_exponents)
            counts["nearest_upsample"] += len(module.add_exponents)
        elif isinstance(module, nn.Conv2d):
            unsupported.append(f"unquantized_conv:{name}")
        elif isinstance(module, nn.BatchNorm2d):
            unsupported.append(f"unfolded_bn:{name}")
        elif type(module).__name__ == "SkipBlock":
            counts["identity_skip"] += 1
    h2_ops = {"conv", "dw_conv", "pw_conv", "relu", "relu6", "linear", "nearest_upsample", "aligned_add", "identity_skip"}
    covered = sorted(op for op in counts if op in h2_ops)
    return {
        "operator_counts": dict(counts),
        "h2_covered_operators": covered,
        "unsupported_operators": unsupported,
        "all_operators_h2_covered": not unsupported,
    }


def export_fp32_onnx(model: nn.Module, path: Path) -> dict[str, Any]:
    from export_contract import PredictionOutputs

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    wrapper = PredictionOutputs(model).eval()
    example = torch.zeros(1, 1, 640, 640)
    levels = list(model.levels)
    torch.onnx.export(
        wrapper,
        example,
        str(path),
        input_names=["image"],
        output_names=levels,
        opset_version=13,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return inspect_onnx(path, kind="fp32")


def export_int8_onnx(model: nn.Module, path: Path) -> dict[str, Any]:
    from export_contract import PredictionOutputs

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    wrapper = PredictionOutputs(model).eval()
    example = torch.zeros(1, 1, 640, 640)
    levels = list(model.levels)
    torch.onnx.export(
        wrapper,
        example,
        str(path),
        input_names=["image"],
        output_names=levels,
        opset_version=13,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return inspect_onnx(path, kind="int8_qdq")


def inspect_onnx(path: Path, *, kind: str) -> dict[str, Any]:
    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    ops = [node.op_type for node in graph.graph.node]
    counts = dict(Counter(ops))
    mapped = []
    unmapped = []
    for op in sorted(set(ops)):
        if op in H2_FPGA_OPS:
            mapped.append({"onnx": op, "h2": H2_FPGA_OPS[op]})
        elif op in ONNX_SOFTWARE_ONLY:
            unmapped.append({"onnx": op, "status": "software_export_only_not_hls_opcode"})
        else:
            unmapped.append({"onnx": op, "status": "unsupported"})
    return {
        "kind": kind,
        "path": str(path),
        "onnx_op_counts": counts,
        "mapped_to_h2": mapped,
        "unsupported_or_software_only": unmapped,
        "checker": "onnx.checker.check_model PASS",
    }


def write_quantization_profile(
    path: Path,
    *,
    network: str,
    architecture_id: str,
    seed: int,
    profile: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "network": network,
        "architecture_id": architecture_id,
        "seed": int(seed),
        "bitwidth": profile["bitwidth"],
        "activation_scale": {
            layer["name"]: layer["output_scale"] for layer in profile["layers"]
        },
        "weight_scale": {
            layer["name"]: layer["weight_scale"] for layer in profile["layers"]
        },
        "activation_exponents": {
            layer["name"]: layer["output_exp"] for layer in profile["layers"]
        },
        "weight_exponents": {
            layer["name"]: layer["weight_exps"] for layer in profile["layers"]
        },
        "requant": [
            {
                "name": layer["name"],
                "multiplier": layer["multiplier"],
                "shift": layer["shift"],
            }
            for layer in profile["layers"]
        ],
        "fpn_adds": profile["fpn_adds"],
        "operator_coverage": coverage,
        "h2_aligned": profile["h2_aligned"],
        "qat": False,
        "parity_ready": False,
        "purpose": "input_to_M_x_q",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload
