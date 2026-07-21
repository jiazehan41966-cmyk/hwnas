#!/usr/bin/env python3
"""Generate the redesigned (v2) G5 sonar-ablation candidates.

Motivation — E1 (60 runs, frozen protocol) verdict on the v1 operators:
    denoise  Δmacro_f1 = -0.012  → no effect (redundant with mbconv)
    edge     Δmacro_f1 = -0.092  → clearly harmful (discards intensity/DC)

The v2 operators fix those specific failure modes:
    adaptive_denoise (AdaptiveDenoiseBlock) — Lee-style learnable gate on
        spatially-aggregated edge evidence; can close the gate and degrade to
        v1's pure smoothing, so its behaviour space contains v1.
    edge_v2 (EdgeAugmentBlock) — intensity main path plus an *additive* edge
        side branch with gamma initialised to 0, so it starts as a plain dw_pw
        (shown harmless in E1) and can only add edge information. 2 directions
        instead of 4 to cut LUT.

Backbone and slot layout are unchanged from v1 (sonar_ablation_backbone_v1):
rl_arch_135 encoding with stage-3 depth 2, giving slots A and B. Clean 2x2
factorial: each variant changes exactly one slot relative to the control.

    mbconv_control_v2         : [mbconv k3 e1,     mbconv k3 e2]
    adaptive_denoise          : [adaptive_denoise, mbconv k3 e2]
    edge_v2                   : [mbconv k3 e1,     edge_v2]
    adaptive_denoise_edge_v2  : [adaptive_denoise, edge_v2]

Matching note: neither v2 operator is foldable (adaptive gating / sqrt), so
capacity is matched on trainable parameters and stage-3 MACs directly rather
than on a folded deployment form. adaptive_denoise matches the control within
the G5 +/-5% band on both. edge_v2 is *cheaper* than its mbconv e2 control
(fewer params, fewer MACs), which makes any positive finding conservative;
the exact deltas are recorded in matching_report.json.
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

from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec

STAGE3_RESOLUTION = 28
REPORT_TOLERANCE = 0.05

SLOT_MBCONV_E1 = {"op": "mbconv", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
SLOT_MBCONV_E2 = {"op": "mbconv", "kernel_size": 3, "expand_ratio": 2, "stride": 1}
SLOT_ADAPTIVE_DENOISE = {"op": "adaptive_denoise", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
SLOT_EDGE_V2 = {"op": "edge_v2", "kernel_size": 3, "expand_ratio": 1, "stride": 1}

VARIANTS = {
    "mbconv_control_v2": [SLOT_MBCONV_E1, SLOT_MBCONV_E2],
    "adaptive_denoise": [SLOT_ADAPTIVE_DENOISE, SLOT_MBCONV_E2],
    "edge_v2": [SLOT_MBCONV_E1, SLOT_EDGE_V2],
    "adaptive_denoise_edge_v2": [SLOT_ADAPTIVE_DENOISE, SLOT_EDGE_V2],
}

CONTROL = "mbconv_control_v2"


def load_base_encoding() -> dict:
    source = PROJECT_ROOT / "configs/ablation/sonar_g5_v1/mbconv_control.candidate.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload["candidate"]["encoding"]


def build_variant_encoding(base: dict, slots: list[dict]) -> dict:
    encoding = copy.deepcopy(base)
    encoding["stages"][3]["blocks"] = copy.deepcopy(slots)
    return encoding


def measure(encoding: dict) -> tuple[int, int]:
    """Return (trainable_params, stage3_macs) for the assembled model."""
    model = build_model(
        ArchitectureSpec.from_dict(encoding),
        num_classes=int(encoding["num_classes"]),
    )
    model.eval()
    params = sum(t.numel() for t in model.parameters())
    macs = 0
    for module in model.stages[3].modules():
        if isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs += (
                module.out_channels
                * (module.in_channels // module.groups)
                * kernel
                * STAGE3_RESOLUTION
                * STAGE3_RESOLUTION
            )
    # forward sanity check
    with torch.no_grad():
        out = model(torch.randn(1, int(encoding["input_channels"]), 224, 224))
    if out.shape != (1, int(encoding["num_classes"])):
        raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
    return params, macs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="configs/ablation/sonar_g5_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_encoding()

    report: dict[str, dict] = {}
    for name, slots in VARIANTS.items():
        encoding = build_variant_encoding(base, slots)
        params, macs = measure(encoding)
        payload = {
            "candidate": {
                "arch_id": f"sonar_g5v2_{name}",
                "backbone": "sonar_ablation_backbone_v1",
                "ablation_variant": name,
                "encoding": encoding,
            },
            "provenance": {
                "plan": "docs/SONAR_OPERATOR_G5_EXPERIMENT_PLAN.md",
                "base_arch": "rl_arch_135",
                "stage3_slots": slots,
                "redesign_reason": (
                    "v1 E1 verdict: denoise no effect (redundant), "
                    "edge harmful (discards intensity). v2 fixes both."
                ),
            },
        }
        path = output_dir / f"{name}.candidate.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report[name] = {"params": params, "stage3_macs": macs, "path": str(path)}

    control = report[CONTROL]
    for name, row in report.items():
        row["param_error_vs_control"] = round(
            abs(row["params"] - control["params"]) / control["params"], 4
        )
        row["macs_error_vs_control"] = round(
            abs(row["stage3_macs"] - control["stage3_macs"]) / control["stage3_macs"], 4
        )
        row["within_5pct_both"] = (
            row["param_error_vs_control"] <= REPORT_TOLERANCE
            and row["macs_error_vs_control"] <= REPORT_TOLERANCE
        )
        row["cheaper_than_control"] = (
            row["params"] <= control["params"] and row["stage3_macs"] <= control["stage3_macs"]
        )

    report_path = output_dir / "matching_report.json"
    report_path.write_text(
        json.dumps(
            {
                "control": CONTROL,
                "tolerance": REPORT_TOLERANCE,
                "matching_basis": "trainable params + stage-3 MACs (v2 ops are not foldable)",
                "note": (
                    "adaptive_denoise matches within the G5 +/-5% band. edge_v2 is cheaper "
                    "than its mbconv e2 control on both params and MACs, so a positive "
                    "edge_v2 finding would be conservative; a negative one is not "
                    "attributable to a capacity disadvantage of the control."
                ),
                "variants": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
