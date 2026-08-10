#!/usr/bin/env python3
"""Freeze the current fixed four-stage sonar search space with DMDC_Conv.

This emits architecture-definition artifacts only.  It does not run training,
hardware-cost measurement, HLS, RTL, route, bitstream, board tests, or power
measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import (  # noqa: E402
    DMDC16_STAGE4_CHOICES,
    DMDC16_STAGE4_LABELS,
    EXCLUDED_FROM_CURRENT_STAGE4_SPACE,
    PENDING_STAGE4_OPERATORS,
    STAGE2_CHOICES,
    architecture_sha256,
    build_fourstage_architecture,
    enumerate_base8,
    enumerate_dmdc16,
    enumerate_extended,
    enumerate_raw20,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import DMDCConvSonarBlock, build_model  # noqa: E402
from hwnas_fpga.training.protocol_reporting import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "frozen_search_space_v4_dmdc16"
)

STAGE2_IDS = {
    "k3_e3": "A1",
    "k3_e6": "A2",
    "k5_e3": "A3",
    "k5_e6": "A4",
}

STAGE4_IDS = {
    "skip": "B1",
    "mbconv_k3_e3": "B2",
    "mbconv_k5_e3": "B3",
    "dmdc_conv_sonar": "B4",
}


def add_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_sha256(unsigned)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    add_payload_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def fixed_macroarchitecture() -> dict[str, Any]:
    return {
        "input": {"channels": 1, "height": 224, "width": 224},
        "stem": {
            "op": "Conv3x3",
            "in_channels": 1,
            "out_channels": 32,
            "stride": 2,
            "output_shape": [32, 112, 112],
        },
        "stage1": {
            "op": "Conv1x1",
            "in_channels": 32,
            "out_channels": 16,
            "stride": 1,
            "depth": 1,
            "output_shape": [16, 112, 112],
            "searchable": False,
        },
        "stage2": {
            "op_family": "MBConv",
            "in_channels": 16,
            "out_channels": 24,
            "stride": 2,
            "depth": 1,
            "output_shape": [24, 56, 56],
            "searchable": True,
        },
        "stage3": {
            "op": "MBConv-k3-e3",
            "in_channels": 24,
            "out_channels": 32,
            "stride": 2,
            "depth": 1,
            "output_shape": [32, 28, 28],
            "searchable": False,
        },
        "stage4": {
            "in_channels": 32,
            "out_channels": 32,
            "stride": 1,
            "depth": 1,
            "output_shape": [32, 28, 28],
            "searchable": True,
        },
        "head": {"op": "GAP->FC8", "conv_head": None, "num_classes": 8},
    }


def stage2_definitions() -> list[dict[str, Any]]:
    rows = []
    for name, kernel, expansion in STAGE2_CHOICES:
        rows.append(
            {
                "candidate_id": STAGE2_IDS[name],
                "candidate_key": name,
                "op_name": "MBConv",
                "structure": "1x1 expand -> depthwise spatial convolution -> 1x1 project",
                "kernel": kernel,
                "expand_ratio": expansion,
                "stride": 2,
                "input_shape": [16, 112, 112],
                "output_shape": [24, 56, 56],
                "residual_condition": "disabled because stride=2 and channels change from 16 to 24",
            }
        )
    return rows


def stage4_definitions() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "B1",
            "candidate_key": "skip",
            "op_name": "Skip",
            "structure": "identity mapping",
            "kernel": None,
            "expand_ratio": 1,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "identity is valid only when stride=1 and channels match",
            "implementation_status": "implemented_existing",
        },
        {
            "candidate_id": "B2",
            "candidate_key": "mbconv_k3_e3",
            "op_name": "MBConv-K3-E3",
            "structure": "1x1 expand -> depthwise 3x3 -> 1x1 project -> residual add",
            "kernel": 3,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "implementation_status": "implemented_existing",
        },
        {
            "candidate_id": "B3",
            "candidate_key": "mbconv_k5_e3",
            "op_name": "MBConv-K5-E3",
            "structure": "1x1 expand -> depthwise 5x5 -> 1x1 project -> residual add",
            "kernel": 5,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "implementation_status": "implemented_existing",
        },
        {
            "candidate_id": "B4",
            "candidate_key": "dmdc_conv_sonar",
            "op_name": "DMDC_Conv",
            "operator_name": "dmdc_conv_sonar",
            "structure": (
                "parallel k={1,3,5} DMDC branches; each branch has a 3x3 "
                "dilated-convolution path and an adaptive-pooling regional "
                "context path, branch fusion, 1x1 branch projection, channel "
                "concat, and final 1x1 channel compression"
            ),
            "dilation_rates": [1, 3, 5],
            "kernel": 3,
            "expand_ratio": 1,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "not added; no residual connection is shown inside the paper DMDC_Conv module",
            "source_type": "literature_operator",
            "source_task": "side_scan_sonar_segmentation",
            "sonar_specific": True,
            "source_paper": (
                "Fused Adaptive Receptive Field Mechanism and Dynamic "
                "Multiscale Dilated Convolution for Side-Scan Sonar Image "
                "Segmentation"
            ),
            "source_doi": "10.1109/TGRS.2022.3201248",
            "core_mechanism": "dynamic_multiscale_dilated_convolution",
            "implementation_status": "implemented",
            "adaptation_note": (
                "sealed as a 32->32 stride-1 Stage4 block; regional context "
                "features are resized back to 28x28 for the paper fusion "
                "operation without removing the dynamic/context branch"
            ),
        },
    ]


def pending_operator_definitions() -> list[dict[str, Any]]:
    return [
        {
            "operator_name": "msconv_sonar",
            "active_space_status": "pending_adaptation_review",
            "source_type": "literature_operator",
            "source_task": "sonar_object_detection",
            "sonar_specific": True,
            "source_paper": (
                "MLFFNet: Multilevel Feature Fusion Network for Object "
                "Detection in Sonar Images"
            ),
            "source_doi": "10.1109/TGRS.2022.3214748",
            "core_mechanism": "multiscale_group_convolution_with_csp",
            "pending_reason": PENDING_STAGE4_OPERATORS["msconv_sonar"]["reason"],
        }
    ]


def excluded_operator_definitions() -> list[dict[str, Any]]:
    return [
        {
            "operator_name": "fused_mbconv_e3",
            "active_space_status": EXCLUDED_FROM_CURRENT_STAGE4_SPACE[
                "fused_mbconv_e3"
            ],
            "reason": "removed from the current active Stage4 space; historical implementation retained",
        },
        {
            "operator_name": "ghost_bottleneck",
            "active_space_status": EXCLUDED_FROM_CURRENT_STAGE4_SPACE[
                "ghost_bottleneck"
            ],
            "reason": "removed from the current active Stage4 space; historical implementation retained",
        },
    ]


def candidate_manifest() -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(enumerate_dmdc16(), start=1):
        stage2_key = f"k{row.kernel}_e{row.expansion}"
        stage4_key = row.arch_id.split("_s4_", 1)[1]
        rows.append(
            {
                "candidate_index": index,
                "architecture_id": row.arch_id,
                "architecture_sha256": architecture_sha256(row.architecture),
                "stage2": {
                    "candidate_id": STAGE2_IDS[stage2_key],
                    "candidate_key": stage2_key,
                    "kernel": row.kernel,
                    "expand_ratio": row.expansion,
                    "op_name": "MBConv",
                },
                "stage4": {
                    "candidate_id": STAGE4_IDS[stage4_key],
                    "candidate_key": stage4_key,
                    "op_name": DMDC16_STAGE4_LABELS[stage4_key],
                },
                "architecture": row.architecture.to_dict(),
                "claim_boundary": (
                    "Search-space definition only; no training, hardware-cost "
                    "measurement, HLS, route, board, bitstream, or power evidence."
                ),
            }
        )
    return rows


def historical_manifest_checks() -> dict[str, Any]:
    base8_path = ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "base8_candidates" / "base8_manifest.json"
    extended_path = ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "extended_candidates" / "extended_manifest.json"
    raw20_path = ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "frozen_search_space_v3_raw20" / "candidate_manifest_20.json"
    return {
        "base8_manifest_path": rel(base8_path),
        "base8_manifest_exists": base8_path.exists(),
        "extended_manifest_path": rel(extended_path),
        "extended_manifest_exists": extended_path.exists(),
        "raw20_historical_manifest_path": rel(raw20_path),
        "raw20_historical_manifest_exists": raw20_path.exists(),
        "base8_runtime_count": len(enumerate_base8()),
        "extended12_runtime_count": len(enumerate_extended(include_stage4_k5=False)),
        "extended16_runtime_count": len(enumerate_extended(include_stage4_k5=True)),
        "raw20_runtime_count": len(enumerate_raw20()),
    }


def builder_checks() -> dict[str, Any]:
    architecture = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="dmdc_conv_sonar",
    )
    model = build_model(architecture, num_classes=8).eval()
    block = model.stages[3][0]
    return {
        "dmdc_builder_recognized": isinstance(block, DMDCConvSonarBlock),
        "dmdc_dilation_rates": list(getattr(block, "dilation_rates", ())),
        "dmdc_implementation_status": getattr(
            block, "implementation_status", None
        ),
    }


def source_file_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src" / "hwnas_fpga" / "fourstage_operator.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "builder.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "__init__.py",
        ROOT / "scripts" / "freeze_fourstage_search_space_dmdc16.py",
    ]
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def make_payloads() -> dict[str, dict[str, Any]]:
    stage2 = stage2_definitions()
    stage4 = stage4_definitions()
    rows = candidate_manifest()
    arch_ids = [row["architecture_id"] for row in rows]
    fixed_macro_errors = []
    for row in enumerate_dmdc16():
        try:
            validate_frozen_fourstage(row.architecture)
        except ValueError as exc:
            fixed_macro_errors.append(f"{row.arch_id}: {exc}")
    historical = historical_manifest_checks()
    builder = builder_checks()
    excluded_keys = set(EXCLUDED_FROM_CURRENT_STAGE4_SPACE)
    active_stage4_keys = {row["candidate_key"] for row in stage4}
    audit_checks = {
        "stage2_candidate_count_eq_4": len(stage2) == 4,
        "stage4_candidate_count_eq_4": len(stage4) == 4,
        "total_candidate_count_eq_16": len(rows) == 16,
        "architecture_id_unique": len(arch_ids) == len(set(arch_ids)),
        "fixed_macroarchitecture_unchanged": not fixed_macro_errors,
        "dmdc_builder_recognized": builder["dmdc_builder_recognized"],
        "dmdc_dilation_rates_eq_1_3_5": builder["dmdc_dilation_rates"]
        == [1, 3, 5],
        "msconv_pending_not_in_active_space": "msconv_sonar"
        not in active_stage4_keys,
        "fused_and_ghost_excluded_from_current_space": excluded_keys.isdisjoint(
            active_stage4_keys
        ),
        "historical_configs_compatible": (
            historical["base8_manifest_exists"]
            and historical["extended_manifest_exists"]
            and historical["raw20_historical_manifest_exists"]
            and historical["base8_runtime_count"] == 8
            and historical["extended12_runtime_count"] == 12
            and historical["extended16_runtime_count"] == 16
            and historical["raw20_runtime_count"] == 20
        ),
    }
    definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_dmdc16_v1",
        "status": "FROZEN_SEARCH_SPACE_DEFINITION",
        "description": (
            "Fixed MobileNetV2-style 4-Stage macroarchitecture; Stage2 keeps "
            "mature MBConv choices, while current active Stage4 choices are "
            "Skip, MBConv-K3-E3, MBConv-K5-E3, and the literature-derived "
            "DMDC_Conv sonar operator."
        ),
        "fixed_macroarchitecture": fixed_macroarchitecture(),
        "stage2_candidates": stage2,
        "stage4_candidates": stage4,
        "pending_stage4_operators": pending_operator_definitions(),
        "excluded_from_current_space": excluded_operator_definitions(),
        "stage2_candidate_count": len(stage2),
        "stage4_candidate_count": len(stage4),
        "candidate_count": len(rows),
        "enumeration_method": "deterministic_cartesian_product",
        "forbidden_search_methods": ["RL", "Aging", "Random", "Evolutionary sampling"],
        "source_file_sha256": source_file_hashes(),
    }
    operator_definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_dmdc16_v1",
        "status": "FROZEN_OPERATOR_DEFINITIONS",
        "stage2_operator_definitions": stage2,
        "active_stage4_operator_definitions": stage4,
        "pending_stage4_operator_definitions": pending_operator_definitions(),
        "excluded_stage4_operator_definitions": excluded_operator_definitions(),
    }
    manifest = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_dmdc16_v1",
        "status": "FROZEN_CANDIDATE_MANIFEST",
        "candidate_count": len(rows),
        "rows": rows,
    }
    audit = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_dmdc16_v1",
        "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "checks": audit_checks,
        "fixed_macroarchitecture_errors": fixed_macro_errors,
        "historical_manifest_checks": historical,
        "builder_checks": builder,
        "candidate_count": len(rows),
        "active_stage2_candidates": [row["candidate_key"] for row in stage2],
        "active_stage4_candidates": [row["candidate_key"] for row in stage4],
        "pending_stage4_operators": ["msconv_sonar"],
        "active_architecture_ids": arch_ids,
        "claim_boundary": {
            "training": "NOT_RUN",
            "formal_evaluation": "NOT_RUN",
            "hardware_cost_table": "NOT_MEASURED",
            "hls": "NOT_RUN",
            "rtl": "NOT_RUN",
            "route": "NOT_RUN",
            "bitstream": "NOT_GENERATED",
            "board": "NOT_RUN",
            "power": "NOT_MEASURED",
        },
    }
    return {
        "search_space_definition.json": definition,
        "operator_definition.json": operator_definition,
        "candidate_manifest_16.json": manifest,
        "search_space_audit.json": audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = make_payloads()
    for filename, payload in payloads.items():
        write_json(args.output_dir / filename, payload)
    audit = payloads["search_space_audit.json"]
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_dir": rel(args.output_dir),
                "candidate_count": audit["candidate_count"],
                "checks": audit["checks"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
