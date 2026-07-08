#!/usr/bin/env python3
"""Export folded sonar weights for the G5 deployment evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.deploy.inference import load_checkpoint_model
from hwnas_fpga.deploy.quantization import (
    QuantizationConfig,
    build_quantized_weight_package,
)
from hwnas_fpga.deploy.reparam import (
    FoldedDenoiseBlock,
    FoldedEdgeBlock,
    fold_sonar_blocks,
)


QUANTIZATION_CONTRACT = "per_tensor_symmetric_int8_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Protocol checkpoint .pt file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <checkpoint-dir>/folded_sonar_export.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combined_sha256(records: Iterable[dict[str, str]]) -> str:
    normalized = sorted(
        (str(record["name"]), str(record["sha256"])) for record in records
    )
    return canonical_sha256(normalized)


def conv_spec(name: str, module: nn.Conv2d) -> dict[str, Any]:
    return {
        "name": name,
        "module_type": module.__class__.__name__,
        "in_channels": int(module.in_channels),
        "out_channels": int(module.out_channels),
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": list(module.padding),
        "groups": int(module.groups),
        "bias": module.bias is not None,
        "weight_shape": list(module.weight.shape),
    }


def folded_block_specs(model: nn.Module) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, FoldedDenoiseBlock):
            specs.append(
                {
                    "name": name,
                    "module_type": "FoldedDenoiseBlock",
                    "deployment_form": "depthwise_conv_relu_pointwise_conv",
                    "use_residual": bool(module.use_residual),
                    "convs": [
                        conv_spec(f"{name}.dw_conv", module.dw_conv),
                        conv_spec(f"{name}.pw_conv", module.pw_conv),
                    ],
                }
            )
        elif isinstance(module, FoldedEdgeBlock):
            specs.append(
                {
                    "name": name,
                    "module_type": "FoldedEdgeBlock",
                    "deployment_form": "dense_conv_relu",
                    "use_residual": bool(module.use_residual),
                    "convs": [conv_spec(f"{name}.conv", module.conv)],
                }
            )
    return specs


def all_conv_specs(model: nn.Module) -> list[dict[str, Any]]:
    return [
        conv_spec(name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


def export_folded_sonar_weights(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str,
) -> dict[str, Any]:
    model, architecture, payload, class_names = load_checkpoint_model(checkpoint, device=device)
    model.eval()
    replaced = fold_sonar_blocks(model)
    if replaced <= 0:
        raise ValueError(f"No DenoiseBlock/EdgeAwareBlock modules were folded in {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    folded_state_path = output_dir / "folded_model_state_dict.pt"
    torch.save(
        {
            "schema_version": 1,
            "source_checkpoint": str(checkpoint.resolve()),
            "architecture": architecture.to_dict(),
            "model_state_dict": model.state_dict(),
            "folded_sonar_block_count": replaced,
        },
        folded_state_path,
    )

    q_package, q_summary = build_quantized_weight_package(
        model,
        architecture=architecture.to_dict(),
        candidate=payload.get("candidate") or payload.get("source_candidate"),
        class_names=class_names,
        config=QuantizationConfig(bit_width=8, scheme="symmetric", quantize_bias=False),
    )
    quantized_path = output_dir / "folded_int8_weight_package.pt"
    torch.save(q_package, quantized_path)

    quantization_spec = {
        "schema_version": 1,
        "quantization_contract": QUANTIZATION_CONTRACT,
        "scope": "folded_weights_only",
        "weight_scheme": "per_tensor_symmetric",
        "bit_width": 8,
        "quantize_bias": False,
        "folded_sonar_block_count": replaced,
        "quantization_summary": q_summary,
        "folded_blocks": folded_block_specs(model),
        "conv_layers": all_conv_specs(model),
    }
    spec_path = output_dir / "quantization_spec.json"
    spec_path.write_text(
        json.dumps(quantization_spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    file_records = [
        {"name": folded_state_path.name, "sha256": sha256_file(folded_state_path)},
        {"name": quantized_path.name, "sha256": sha256_file(quantized_path)},
        {"name": spec_path.name, "sha256": sha256_file(spec_path)},
    ]
    export_sha256 = combined_sha256(file_records)
    spec_sha256 = sha256_file(spec_path)

    manifest_fragment = {
        "schema_version": 1,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "output_dir": str(output_dir.resolve()),
        "manifest_path": str((output_dir / "folded_export_manifest.json").resolve()),
        "folded_sonar_block_count": replaced,
        "quantization_contract": QUANTIZATION_CONTRACT,
        "software_spec_sha256": spec_sha256,
        "hls_spec_sha256": spec_sha256,
        "weight_export_complete": True,
        "weight_export_sha256": export_sha256,
        "files": file_records,
        "boundary": (
            "This export proves folded PyTorch packaging and shared INT8 spec "
            "identity only. HLS parity and route feasibility require separate records."
        ),
    }
    manifest_path = output_dir / "folded_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_fragment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_fragment


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parent / "folded_sonar_export"
    )
    manifest = export_folded_sonar_weights(checkpoint, output_dir, device=args.device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": manifest["output_dir"],
                "folded_sonar_block_count": manifest["folded_sonar_block_count"],
                "weight_export_sha256": manifest["weight_export_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
