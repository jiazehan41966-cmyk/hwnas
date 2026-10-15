#!/usr/bin/env python3
"""Build a ZCU104 operator-cost scaffold for the MobileNetA sonar target slot.

The artifact produced by this script is intentionally not search-ready.  It
freezes the MobileNet-style A target-slot search-space version and creates the
empty operator-cost rows that must later be filled by fresh ZCU104 evidence.
It does not run HLS, route, bitstream generation, board validation, or power
measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any


SEARCH_SPACE_VERSION = "mobilenetA_dmdc_cfem_slot_v1"
CAMPAIGN_ID = "zcu104_full_campaign_v1"
BOARD = {
    "board": "ZCU104",
    "part": "xczu7ev-ffvc1156-2-e",
    "board_part": "xilinx.com:zcu104:part0:1.1",
    "clock_period_ns": 5.0,
    "tool_version": "Vitis/Vivado 2022.2",
}

TARGET_SLOT = {
    "slot_id": "mobilenetA_stage3_late_block",
    "stage_index_zero_based": 2,
    "block_index_zero_based": 1,
    "stage_label": "stage3",
    "in_channels": 32,
    "out_channels": 32,
    "input_resolution": 28,
    "output_resolution": 28,
    "stride": 1,
    "reason": "Only this fixed MobileNet-style A slot matches the frozen DMDC/CFEM 32->32 stride-1 contract.",
}

MOBILE_A_MACRO = {
    "input": {"channels": 1, "height": 224, "width": 224},
    "stem": {
        "op": "Conv3x3",
        "in_channels": 1,
        "out_channels": 32,
        "stride": 2,
        "output_shape": [32, 112, 112],
    },
    "stage_count": 7,
    "stage_channels": [16, 24, 32, 64, 96, 160, 320],
    "stage_strides": [1, 2, 2, 2, 1, 2, 1],
    "stage_depths": [1, 2, 2, 2, 2, 2, 1],
    "fixed_policy": (
        "For this direction and LUT freeze, non-target blocks remain fixed "
        "or MBConv-only; DMDC/CFEM are admitted only at the target slot."
    ),
    "target_slot": TARGET_SLOT,
    "head": {"op": "GAP->FC8", "num_classes": 8},
}

OP_DEFINITIONS = [
    {
        "candidate_id": "A0_B1_A1_B1",
        "candidate_key": "skip",
        "op_name": "Skip",
        "kernel": None,
        "expand_ratio": 1,
        "structure": "identity mapping for 32-channel stride-1 same-shape slot",
    },
    {
        "candidate_id": "A0_B2_A1_B2",
        "candidate_key": "mbconv_k3_e3",
        "op_name": "MBConv-K3-E3",
        "kernel": 3,
        "expand_ratio": 3,
        "structure": "1x1 expand -> 3x3 depthwise -> 1x1 project with residual",
    },
    {
        "candidate_id": "A0_B3_A1_B3",
        "candidate_key": "mbconv_k5_e3",
        "op_name": "MBConv-K5-E3",
        "kernel": 5,
        "expand_ratio": 3,
        "structure": "1x1 expand -> 5x5 depthwise -> 1x1 project with residual",
    },
    {
        "candidate_id": "A1_B4",
        "candidate_key": "dmdc_conv_sonar",
        "op_name": "DMDC_Conv",
        "kernel": 3,
        "expand_ratio": None,
        "structure": "parallel dilated multiscale sonar-context branches with final 1x1 compression",
        "frozen_shape_contract": "32->32, stride=1 only",
    },
    {
        "candidate_id": "A1_B5",
        "candidate_key": "cfem_sonar",
        "op_name": "CFEM",
        "kernel": 3,
        "expand_ratio": None,
        "structure": "multiscale dilated context extraction with attention-based feature fusion",
        "frozen_shape_contract": "32->32, stride=1 only",
    },
]

SPACE_VARIANTS = {
    "A0_mobile_no_cfem": ["skip", "mbconv_k3_e3", "mbconv_k5_e3"],
    "A1_mobile_dmdc_cfem": [
        "skip",
        "mbconv_k3_e3",
        "mbconv_k5_e3",
        "dmdc_conv_sonar",
        "cfem_sonar",
    ],
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def add_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_sha256(unsigned)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    add_payload_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def op_for_arch(candidate_key: str) -> dict[str, Any]:
    if candidate_key == "skip":
        return {"op": "skip", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
    if candidate_key == "mbconv_k3_e3":
        return {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 1}
    if candidate_key == "mbconv_k5_e3":
        return {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3, "stride": 1}
    if candidate_key == "dmdc_conv_sonar":
        return {"op": "dmdc_conv_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
    if candidate_key == "cfem_sonar":
        return {"op": "cfem_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1}
    raise KeyError(candidate_key)


def fixed_block(stride: int) -> dict[str, Any]:
    return {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": int(stride)}


def mobilenetA_architecture(candidate_key: str) -> dict[str, Any]:
    stages = []
    channels = MOBILE_A_MACRO["stage_channels"]
    strides = MOBILE_A_MACRO["stage_strides"]
    depths = MOBILE_A_MACRO["stage_depths"]
    for stage_index, (channels_i, stride_i, depth_i) in enumerate(
        zip(channels, strides, depths)
    ):
        blocks = []
        for block_index in range(depth_i):
            stride = stride_i if block_index == 0 else 1
            if stage_index == TARGET_SLOT["stage_index_zero_based"] and block_index == TARGET_SLOT["block_index_zero_based"]:
                blocks.append(op_for_arch(candidate_key))
            else:
                blocks.append(fixed_block(stride))
        stages.append(
            {
                "channels": int(channels_i),
                "depth": int(depth_i),
                "stride": int(stride_i),
                "blocks": blocks,
            }
        )
    return {
        "input_channels": 1,
        "stem_channels": 32,
        "stem_stride": 2,
        "post_stem_downsample_stride": 1,
        "head_conv_channels": 1280,
        "head_channels": None,
        "num_classes": 8,
        "stages": stages,
    }


def make_candidate_rows() -> list[dict[str, Any]]:
    rows = []
    for op_def in OP_DEFINITIONS:
        key = op_def["candidate_key"]
        architecture = mobilenetA_architecture(key)
        rows.append(
            {
                "candidate_key": key,
                "candidate_id": op_def["candidate_id"],
                "search_space_membership": [
                    name for name, keys in SPACE_VARIANTS.items() if key in keys
                ],
                "target_slot": TARGET_SLOT,
                "architecture": architecture,
                "architecture_sha256": canonical_sha256(architecture),
                "operator_lut_row_key": lut_row_key(key),
            }
        )
    return rows


def lut_row_key(candidate_key: str) -> str:
    return (
        "target_slot__"
        f"{candidate_key}__32x28x28_to_32x28x28__s1"
    )


def make_lut_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    op_by_key = {row["candidate_key"]: row for row in OP_DEFINITIONS}
    rows = []
    for candidate in candidate_rows:
        key = candidate["candidate_key"]
        op_def = op_by_key[key]
        rows.append(
            {
                "row_key": candidate["operator_lut_row_key"],
                "candidate_key": key,
                "candidate_id": op_def["candidate_id"],
                "search_space_version": SEARCH_SPACE_VERSION,
                "campaign_id": CAMPAIGN_ID,
                "board": BOARD,
                "input_shape": [32, 28, 28],
                "output_shape": [32, 28, 28],
                "stride": 1,
                "kernel": op_def.get("kernel"),
                "expand_ratio": op_def.get("expand_ratio"),
                "metrics": {
                    "LUT": None,
                    "DSP": None,
                    "BRAM_18K": None,
                    "FF": None,
                    "latency_cycles": None,
                    "latency_ns": None,
                    "power_w": None,
                },
                "measurement_status": "PENDING_FRESH_ZCU104_MEASUREMENT",
                "formal_evidence_status": "NOT_FORMAL_IN_THIS_SCAFFOLD",
                "allowed_evidence_sources": [
                    "fresh Vitis HLS 2022.2 csynth for this row",
                    "fresh Vivado route/timing reports for this row when route evidence is claimed",
                    "fresh ZCU104 board measurements only when board evidence is separately claimed",
                ],
                "forbidden_shortcuts": [
                    "reuse fourstage_cfem20_v1 row as formal MobileNetA evidence",
                    "reuse historical AV7K325 evidence as ZCU104 evidence",
                    "fill LUT/DSP/BRAM/latency/power from analytical proxy",
                ],
            }
        )
    return rows


def run_smoke(src_root: Path | None, candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if src_root is None:
        return {"status": "SKIPPED", "reason": "--src-root was not provided"}
    src_path = src_root / "src" if (src_root / "src").exists() else src_root
    sys.path.insert(0, str(src_path))
    try:
        import torch

        space_mod = importlib.import_module("hwnas_fpga.search_space")
        models_mod = importlib.import_module("hwnas_fpga.models")
        ArchitectureSpec = getattr(space_mod, "ArchitectureSpec")
        build_model = getattr(models_mod, "build_model")
    except Exception as exc:  # pragma: no cover - diagnostic artifact only
        return {"status": "SKIPPED", "reason": f"import failed: {type(exc).__name__}: {exc}"}

    smoke_rows = []
    for row in candidate_rows:
        key = row["candidate_key"]
        try:
            arch = ArchitectureSpec.from_dict(row["architecture"])
            model = build_model(arch, num_classes=8)
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 1, 224, 224))
            smoke_rows.append(
                {
                    "candidate_key": key,
                    "status": "PASS",
                    "output_shape": list(output.shape),
                    "params": int(sum(p.numel() for p in model.parameters())),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic artifact only
            smoke_rows.append(
                {
                    "candidate_key": key,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in smoke_rows) else "FAIL",
        "rows": smoke_rows,
    }


def build_payloads(output_dir: Path, src_root: Path | None) -> dict[str, Path]:
    candidate_rows = make_candidate_rows()
    lut_rows = make_lut_rows(candidate_rows)
    smoke = run_smoke(src_root, candidate_rows)

    search_space = {
        "artifact_type": "search_space_definition",
        "search_space_version": SEARCH_SPACE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "direction_decision": {
            "selected_organization": "MobileNet-style A",
            "basis": (
                "The seed-42 20-candidate pilot showed a stronger positive "
                "trend for A1-A0 than B1-B0; this artifact freezes the A "
                "target-slot design for the next LUT step."
            ),
            "not_a_claim": [
                "not a final multi-seed result",
                "not a formal ZCU104 LUT",
                "not a board deployment result",
            ],
        },
        "macroarchitecture": MOBILE_A_MACRO,
        "space_variants": SPACE_VARIANTS,
        "operator_definitions": OP_DEFINITIONS,
        "constraints": [
            "stage count, stage channels, stage strides, and stage depths are fixed",
            "DMDC/CFEM appear only at the 32->32, 28x28, stride-1 target slot",
            "A1 differs from A0 only by adding dmdc_conv_sonar and cfem_sonar at the same target slot",
        ],
    }
    candidate_manifest = {
        "artifact_type": "candidate_manifest",
        "search_space_version": SEARCH_SPACE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "candidate_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
    }
    lut_scaffold = {
        "artifact_type": "operator_cost_lut_scaffold",
        "data_kind": "SCAFFOLD_ONLY_NOT_FORMAL",
        "search_space_version": SEARCH_SPACE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "target_board": BOARD,
        "search_ready": False,
        "row_count": len(lut_rows),
        "measured_row_count": 0,
        "rows": lut_rows,
    }
    coverage = {
        "artifact_type": "operator_cost_lut_coverage_audit",
        "search_space_version": SEARCH_SPACE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "target_board": BOARD,
        "status": "PENDING_FRESH_ZCU104_MEASUREMENT",
        "search_ready": False,
        "required_rows": [row["row_key"] for row in lut_rows],
        "measured_rows": [],
        "coverage": {"measured": 0, "required": len(lut_rows), "ratio": 0.0},
        "candidate_mapping_coverage": {
            "unique_architecture_candidates": len(candidate_rows),
            "a0_candidate_keys": SPACE_VARIANTS["A0_mobile_no_cfem"],
            "a1_candidate_keys": SPACE_VARIANTS["A1_mobile_dmdc_cfem"],
        },
        "smoke_build": smoke,
        "claim_boundary": {
            "LUT_DSP_BRAM_latency_power": "NOT_FORMAL_IN_THIS_SCAFFOLD",
            "board_classification": "NOT_RUN",
            "board_power": "NOT_RUN",
            "old_artifacts": "NOT_OVERWRITTEN_AND_NOT_IMPORTED_AS_MEASUREMENT",
        },
    }
    manifest = {
        "artifact_type": "build_manifest",
        "search_space_version": SEARCH_SPACE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "target_board": BOARD,
        "output_dir": str(output_dir),
        "files": {},
        "entrypoint": "scripts/build_zcu104_mobilenetA_slot_operator_cost_scaffold.py",
        "status": (
            "SCAFFOLD_READY_MEASUREMENT_PENDING"
            if smoke["status"] in {"PASS", "SKIPPED"}
            else "SCAFFOLD_WRITTEN_SMOKE_FAILED"
        ),
        "next_gate": (
            "Run fresh ZCU104 HLS/route measurements for the five required "
            "target-slot rows before enabling formal hardware-cost search."
        ),
    }

    paths = {
        "search_space_definition": output_dir / "search_space_definition.json",
        "candidate_manifest": output_dir / "candidate_manifest.json",
        "hardware_cost_lut_scaffold": output_dir / "operator_cost_lut" / "hardware_cost_lut.scaffold.json",
        "coverage_audit": output_dir / "operator_cost_lut" / "coverage_audit.json",
        "build_manifest": output_dir / "operator_cost_lut" / "build_manifest.json",
    }
    write_json(paths["search_space_definition"], search_space)
    write_json(paths["candidate_manifest"], candidate_manifest)
    write_json(paths["hardware_cost_lut_scaffold"], lut_scaffold)
    write_json(paths["coverage_audit"], coverage)

    manifest["files"] = {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in paths.items()
        if key != "build_manifest"
    }
    write_json(paths["build_manifest"], manifest)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for this MobileNetA ZCU104 LUT scaffold.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Optional repository or src root used for model construction smoke tests.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty output directory: {output_dir}")
    paths = build_payloads(output_dir, args.src_root.resolve() if args.src_root else None)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
