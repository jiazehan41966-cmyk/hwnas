#!/usr/bin/env python3
"""Generate the four G5 sonar-ablation candidate JSONs and verify matched capacity.

Backbone: sonar_ablation_backbone_v1 = rl_arch_135 encoding with stage-3 depth
changed from 1 to 2, forming slots A/B. Variants differ only in the slot ops:

    mbconv_control: [mbconv k3 e1, mbconv k3 e4]
    denoise:        [denoise k3,   mbconv k3 e4]
    edge:           [mbconv k3 e1, edge k3]
    denoise_edge:   [denoise k3,   edge k3]

Matching is verified on the folded (deployment) form via
hwnas_fpga.deploy.reparam, which is the semantics the G5 gate compares.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from hwnas_fpga.deploy.reparam import fold_sonar_blocks
from hwnas_fpga.models.builder import build_model
from hwnas_fpga.search_space import ArchitectureSpec

STAGE3_RESOLUTION = 28
MATCH_TOLERANCE = 0.05

BASE_ENCODING = {
    "input_channels": 1,
    "stem_channels": 32,
    "stem_stride": 2,
    "post_stem_downsample_stride": 1,
    "head_conv_channels": None,
    "head_channels": None,
    "num_classes": 8,
    "stages": [
        {
            "channels": 16,
            "depth": 1,
            "stride": 1,
            "blocks": [{"op": "conv", "kernel_size": 1, "expand_ratio": 1, "stride": 1}],
        },
        {
            "channels": 24,
            "depth": 1,
            "stride": 2,
            "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 6, "stride": 2}],
        },
        {
            "channels": 32,
            "depth": 1,
            "stride": 2,
            "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2}],
        },
        # stage-3: two ablation slots, filled per variant below
        {"channels": 32, "depth": 2, "stride": 1, "blocks": []},
    ],
}

SLOT_MBCONV_E1 = {"op": "mbconv", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
SLOT_MBCONV_E4 = {"op": "mbconv", "kernel_size": 3, "expand_ratio": 4, "stride": 1}
SLOT_DENOISE = {"op": "denoise", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
SLOT_EDGE = {"op": "edge", "kernel_size": 3, "expand_ratio": 1, "stride": 1}

VARIANTS = {
    "mbconv_control": [SLOT_MBCONV_E1, SLOT_MBCONV_E4],
    "denoise": [SLOT_DENOISE, SLOT_MBCONV_E4],
    "edge": [SLOT_MBCONV_E1, SLOT_EDGE],
    "denoise_edge": [SLOT_DENOISE, SLOT_EDGE],
}


def build_variant_encoding(slot_blocks: list[dict]) -> dict:
    encoding = copy.deepcopy(BASE_ENCODING)
    encoding["stages"][3]["blocks"] = copy.deepcopy(slot_blocks)
    return encoding


def folded_param_count(encoding: dict) -> tuple[int, int]:
    """Return (train_params, folded_deploy_params) for the full model."""
    architecture = ArchitectureSpec.from_dict(encoding)
    model = build_model(architecture, num_classes=int(encoding["num_classes"]))
    model.eval()
    train_params = sum(p.numel() for p in model.parameters())
    fold_sonar_blocks(model)
    # 部署形态: 卷积权重 + bias（其余 BN 由部署管线折叠，这里统一只计卷积）
    deploy_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            deploy_params += module.weight.numel()
            if module.bias is not None:
                deploy_params += module.bias.numel()
        elif isinstance(module, torch.nn.Linear):
            deploy_params += module.weight.numel() + (
                module.bias.numel() if module.bias is not None else 0
            )
    return train_params, deploy_params


def stage3_macs(encoding: dict) -> int:
    """Approximate stage-3 MACs from conv weights at 28x28 (matching doc numbers)."""
    architecture = ArchitectureSpec.from_dict(encoding)
    model = build_model(architecture, num_classes=int(encoding["num_classes"]))
    model.eval()
    fold_sonar_blocks(model)
    stage3 = model.stages[3]
    total = 0
    for module in stage3.modules():
        if isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            total += (
                module.out_channels
                * (module.in_channels // module.groups)
                * k
                * STAGE3_RESOLUTION
                * STAGE3_RESOLUTION
            )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="configs/ablation/sonar_g5_v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    control_deploy = None
    control_macs = None
    for name, slots in VARIANTS.items():
        encoding = build_variant_encoding(slots)
        train_params, deploy_params = folded_param_count(encoding)
        macs = stage3_macs(encoding)
        if name == "mbconv_control":
            control_deploy = deploy_params
            control_macs = macs
        payload = {
            "candidate": {
                "arch_id": f"sonar_g5_{name}",
                "backbone": "sonar_ablation_backbone_v1",
                "ablation_variant": name,
                "encoding": encoding,
            },
            "provenance": {
                "plan": "docs/SONAR_OPERATOR_G5_EXPERIMENT_PLAN.md",
                "base_arch": "rl_arch_135",
                "stage3_slots": slots,
            },
        }
        path = output_dir / f"{name}.candidate.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report[name] = {
            "train_params": train_params,
            "deploy_params": deploy_params,
            "stage3_macs": macs,
            "path": str(path),
        }

    assert control_deploy is not None and control_macs is not None
    ok = True
    for name, row in report.items():
        param_err = abs(row["deploy_params"] - control_deploy) / control_deploy
        macs_err = abs(row["stage3_macs"] - control_macs) / control_macs
        row["deploy_param_error_vs_control"] = round(param_err, 4)
        row["stage3_macs_error_vs_control"] = round(macs_err, 4)
        if param_err > MATCH_TOLERANCE or macs_err > MATCH_TOLERANCE:
            ok = False

    report_path = output_dir / "matching_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "tolerance": MATCH_TOLERANCE,
                "note": "deploy_params/stage3_macs computed on the folded (deployment) form",
                "variants": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if ok else "FAIL", "report": str(report_path)}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
