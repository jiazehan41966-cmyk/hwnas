#!/usr/bin/env python3
"""Freeze the fixed four-stage RAW20 sonar HW-NAS search space.

This script deliberately emits architecture definitions only.  It does not run
training, hardware-cost measurement, HLS, RTL co-simulation, route, bitstream
generation, board tests, or power measurement.
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
    RAW20_STAGE4_CHOICES,
    RAW20_STAGE4_LABELS,
    STAGE2_CHOICES,
    architecture_sha256,
    build_fourstage_architecture,
    enumerate_base8,
    enumerate_extended,
    enumerate_raw20,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import (  # noqa: E402
    FusedMBConvBlock,
    GhostBottleneckBlock,
    build_model,
)
from hwnas_fpga.training.protocol_reporting import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "frozen_search_space_v3_raw20"
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
    "fused_mbconv_e3": "B4",
    "ghost_bottleneck": "B5",
}

EXCLUDED_OPERATORS = (
    "dir_mbconv3_split11_e3_v1",
    "edge",
    "denoise",
    "fused_edge",
    "fused_denoise",
    "mixed_dw_k3_k5",
    "dual_dw3",
    "dilated_dw",
    "rep_k5",
    "ghost_k5",
    "attention",
    "dynamic_branch",
)


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
        "frozen_dimensions": [
            "input_resolution",
            "stem",
            "stage_count",
            "stage_channels",
            "stage_strides",
            "stage_depths",
            "stage1",
            "stage3",
            "head",
        ],
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
                "search_role": "Stage2 transition block",
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
            "search_role": "Stage4 normal block",
        },
        {
            "candidate_id": "B2",
            "candidate_key": "mbconv_k3_e3",
            "op_name": "MBConv-k3-e3",
            "structure": "1x1 expand -> depthwise 3x3 -> 1x1 project -> residual add",
            "kernel": 3,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "search_role": "Stage4 normal block",
        },
        {
            "candidate_id": "B3",
            "candidate_key": "mbconv_k5_e3",
            "op_name": "MBConv-k5-e3",
            "structure": "1x1 expand -> depthwise 5x5 -> 1x1 project -> residual add",
            "kernel": 5,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "search_role": "Stage4 normal block",
        },
        {
            "candidate_id": "B4",
            "candidate_key": "fused_mbconv_e3",
            "op_name": "Fused-MBConv-e3",
            "structure": "fused 3x3 spatial expansion convolution -> 1x1 project -> residual add",
            "kernel": 3,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "search_role": "Stage4 normal block",
            "new_search_dimensions": [],
        },
        {
            "candidate_id": "B5",
            "candidate_key": "ghost_bottleneck",
            "op_name": "Ghost Bottleneck",
            "structure": "Ghost expansion module -> depthwise 3x3 -> Ghost projection module -> residual add",
            "kernel": 3,
            "expand_ratio": 3,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "residual_condition": "enabled because stride=1 and channels are 32->32",
            "search_role": "Stage4 normal block",
            "new_search_dimensions": [],
        },
    ]


def candidate_manifest() -> list[dict[str, Any]]:
    manifest = []
    for index, row in enumerate(enumerate_raw20(), start=1):
        stage2_key = f"k{row.kernel}_e{row.expansion}"
        stage4_key = row.arch_id.split("_s4_", 1)[1]
        manifest.append(
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
                    "op_name": RAW20_STAGE4_LABELS[stage4_key],
                },
                "architecture": row.architecture.to_dict(),
                "claim_boundary": (
                    "Search-space definition only; no training, hardware-cost "
                    "measurement, HLS, route, board, bitstream, or power evidence."
                ),
            }
        )
    return manifest


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def historical_manifest_checks() -> dict[str, Any]:
    base8_path = ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "base8_candidates" / "base8_manifest.json"
    extended_path = ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "extended_candidates" / "extended_manifest.json"
    base8_manifest = load_json_if_exists(base8_path)
    extended_manifest = load_json_if_exists(extended_path)
    base8_runtime_count = len(enumerate_base8())
    extended12_runtime_count = len(enumerate_extended(include_stage4_k5=False))
    extended16_runtime_count = len(enumerate_extended(include_stage4_k5=True))
    return {
        "base8_manifest_path": rel(base8_path),
        "base8_manifest_exists": base8_manifest is not None,
        "base8_manifest_candidate_count": (
            base8_manifest.get("candidate_count") if base8_manifest else None
        ),
        "base8_runtime_count": base8_runtime_count,
        "extended_manifest_path": rel(extended_path),
        "extended_manifest_exists": extended_manifest is not None,
        "extended_manifest_candidate_count": (
            extended_manifest.get("candidate_count") if extended_manifest else None
        ),
        "extended12_runtime_count": extended12_runtime_count,
        "extended16_runtime_count": extended16_runtime_count,
    }


def builder_checks() -> dict[str, Any]:
    fused_arch = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="fused_mbconv_e3",
    )
    ghost_arch = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="ghost_bottleneck",
    )
    fused_model = build_model(fused_arch, num_classes=8)
    ghost_model = build_model(ghost_arch, num_classes=8)
    return {
        "fused_mbconv_builder_recognized": isinstance(
            fused_model.stages[3][0], FusedMBConvBlock
        ),
        "ghost_bottleneck_builder_recognized": isinstance(
            ghost_model.stages[3][0], GhostBottleneckBlock
        ),
    }


def source_file_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src" / "hwnas_fpga" / "fourstage_operator.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "builder.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "__init__.py",
        ROOT / "scripts" / "freeze_fourstage_search_space_raw20.py",
    ]
    return {rel(path): sha256_file(path) for path in paths if path.exists()}


def make_payloads() -> dict[str, dict[str, Any]]:
    stage2 = stage2_definitions()
    stage4 = stage4_definitions()
    manifest_rows = candidate_manifest()
    arch_ids = [row["architecture_id"] for row in manifest_rows]
    rows = enumerate_raw20()
    fixed_macro_ok = True
    fixed_macro_errors: list[str] = []
    for row in rows:
        try:
            validate_frozen_fourstage(row.architecture)
        except ValueError as exc:
            fixed_macro_ok = False
            fixed_macro_errors.append(f"{row.arch_id}: {exc}")

    historical = historical_manifest_checks()
    historical_compatible = (
        historical["base8_manifest_exists"] is True
        and historical["base8_manifest_candidate_count"] == 8
        and historical["base8_runtime_count"] == 8
        and historical["extended_manifest_exists"] is True
        and historical["extended_manifest_candidate_count"] == 16
        and historical["extended12_runtime_count"] == 12
        and historical["extended16_runtime_count"] == 16
    )
    builder = builder_checks()
    dir_not_active = all("dir" not in arch_id.lower() for arch_id in arch_ids) and all(
        "dir" not in str(candidate).lower() for candidate in stage4
    )

    definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_raw20_v1",
        "status": "FROZEN_SEARCH_SPACE_DEFINITION",
        "description": (
            "Fixed MobileNetV2-style 4-Stage macroarchitecture; Stage2 keeps "
            "mature MBConv choices, Stage4 introduces Skip, MBConv, "
            "Fused-MBConv, and Ghost Bottleneck as block-level operator "
            "candidates."
        ),
        "fixed_macroarchitecture": fixed_macroarchitecture(),
        "stage2_candidates": stage2,
        "stage4_candidates": stage4,
        "stage2_candidate_count": len(stage2),
        "stage4_candidate_count": len(stage4),
        "candidate_count": len(manifest_rows),
        "enumeration_method": "deterministic_cartesian_product",
        "forbidden_search_methods": [
            "RL",
            "Aging",
            "Random",
            "Evolutionary sampling",
        ],
        "excluded_from_active_space": list(EXCLUDED_OPERATORS),
        "historical_space_policy": {
            "historical_16_space": "preserved, not overwritten",
            "historical_dir_status": "eliminated from active RAW20 space",
            "new_raw_space": "Stage2 4 x Stage4 5 = 20",
        },
        "source_file_sha256": source_file_hashes(),
    }

    operator_definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_raw20_v1",
        "status": "FROZEN_OPERATOR_DEFINITIONS",
        "stage2_operator_definitions": stage2,
        "stage4_operator_definitions": stage4,
        "excluded_operator_definitions": [
            {
                "operator": name,
                "active_space_status": "EXCLUDED",
                "reason": (
                    "historical eliminated operator"
                    if name == "dir_mbconv3_split11_e3_v1"
                    else "outside this RAW20 search-space freeze"
                ),
            }
            for name in EXCLUDED_OPERATORS
        ],
    }

    manifest = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_raw20_v1",
        "status": "FROZEN_CANDIDATE_MANIFEST",
        "candidate_count": len(manifest_rows),
        "rows": manifest_rows,
    }

    audit_checks = {
        "stage2_candidate_count_eq_4": len(stage2) == 4,
        "stage4_candidate_count_eq_5": len(stage4) == 5,
        "total_candidate_count_eq_20": len(manifest_rows) == 20,
        "architecture_id_unique": len(arch_ids) == len(set(arch_ids)),
        "fixed_macroarchitecture_unchanged": fixed_macro_ok,
        "historical_configs_compatible": historical_compatible,
        "dir_not_in_active_space": dir_not_active,
        "fused_mbconv_builder_recognized": builder[
            "fused_mbconv_builder_recognized"
        ],
        "ghost_bottleneck_builder_recognized": builder[
            "ghost_bottleneck_builder_recognized"
        ],
    }
    audit = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_raw20_v1",
        "status": (
            "PASS" if all(audit_checks.values()) else "FAIL"
        ),
        "checks": audit_checks,
        "fixed_macroarchitecture_errors": fixed_macro_errors,
        "historical_manifest_checks": historical,
        "builder_checks": builder,
        "candidate_count": len(manifest_rows),
        "active_stage2_candidates": [row["candidate_key"] for row in stage2],
        "active_stage4_candidates": [row["candidate_key"] for row in stage4],
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
        "candidate_manifest_20.json": manifest,
        "search_space_audit.json": audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the frozen RAW20 JSON artifacts.",
    )
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
