#!/usr/bin/env python3
"""Generate the preregistered MBConv-k3/k5/MixConv-v2 screen candidates."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hwnas_fpga.models import build_model  # noqa: E402
from hwnas_fpga.search_space import ArchitectureSpec  # noqa: E402


CONTROL = {"op": "mbconv", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
VARIANTS = {
    "mbconv_k3_control": CONTROL,
    "mbconv_k5_control": {
        "op": "mbconv",
        "kernel_size": 5,
        "expand_ratio": 1,
        "stride": 1,
    },
    "mixconv_v2": {
        "op": "mixconv_v2",
        "kernel_size": 5,
        "expand_ratio": 1,
        "stride": 1,
    },
}


def _base_encoding() -> dict:
    path = REPO / "configs/ablation/sonar_g5_v2/mbconv_control_v2.candidate.json"
    return json.loads(path.read_text(encoding="utf-8"))["candidate"]["encoding"]


def _measure(encoding: dict) -> dict[str, int]:
    model = build_model(
        ArchitectureSpec.from_dict(encoding),
        num_classes=int(encoding["num_classes"]),
    ).eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    stage_params = sum(parameter.numel() for parameter in model.stages[3][0].parameters())
    stage_macs = 0
    for module in model.stages[3][0].modules():
        if isinstance(module, torch.nn.Conv2d):
            stage_macs += (
                28
                * 28
                * module.out_channels
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
    with torch.no_grad():
        output = model(torch.zeros(1, int(encoding["input_channels"]), 224, 224))
    if tuple(output.shape) != (1, int(encoding["num_classes"])):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    return {
        "total_trainable_params": int(total_params),
        "target_slot_trainable_params": int(stage_params),
        "target_slot_macs": int(stage_macs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="configs/ablation/sonar_mixconv_v2"
    )
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = _base_encoding()

    report: dict[str, dict] = {}
    for name, target_slot in VARIANTS.items():
        encoding = copy.deepcopy(base)
        encoding["stages"][3]["blocks"][0] = copy.deepcopy(target_slot)
        metrics = _measure(encoding)
        payload = {
            "candidate": {
                "arch_id": f"sonar_mixconv_v2_{name}",
                "backbone": "rl_arch_135_fixed_screen_backbone",
                "ablation_variant": name,
                "encoding": encoding,
            },
            "provenance": {
                "base_arch": "rl_arch_135 (legacy_fold0_selected)",
                "comparison_role": name,
                "target": "stages[3].blocks[0] (32->32, stride=1, resolution=28)",
                "target_slot": target_slot,
                "outer_validation_used_for_candidate_generation": False,
            },
        }
        path = output / f"{name}.candidate.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report[name] = {**metrics, "path": str(path)}

    control = report["mbconv_k3_control"]
    for row in report.values():
        row["total_param_delta_fraction_vs_k3"] = (
            row["total_trainable_params"] - control["total_trainable_params"]
        ) / control["total_trainable_params"]
        row["target_slot_mac_delta_fraction_vs_k3"] = (
            row["target_slot_macs"] - control["target_slot_macs"]
        ) / control["target_slot_macs"]
        row["within_total_param_5pct"] = (
            row["total_param_delta_fraction_vs_k3"] <= 0.05
        )

    report_payload = {
        "schema_version": 1,
        "screen": "mbconv_k3_vs_mbconv_k5_vs_mixconv_v2",
        "primary_metric": "inner_macro_f1",
        "selection_thresholds": {
            "mixconv_v2_delta_vs_k3": 0.01,
            "mixconv_v2_delta_vs_k5": 0.005,
            "mixconv_v2_noninferiority_vs_k5": -0.005,
            "positive_seed_count_minimum": 2,
            "total_parameter_increase_maximum": 0.05,
        },
        "variants": report,
    }
    report_path = output / "matching_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
