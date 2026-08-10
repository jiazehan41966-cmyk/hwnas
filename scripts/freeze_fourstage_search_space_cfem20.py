#!/usr/bin/env python3
"""Freeze the current fixed four-stage sonar search space with CFEM.

This script writes architecture-definition artifacts only.  It does not run
training, NAS search, hardware-cost measurement, HLS, RTL, route, bitstream,
board validation, or power measurement.
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
    CFEM20_STAGE4_CHOICES,
    CFEM20_STAGE4_LABELS,
    EXCLUDED_FROM_CURRENT_STAGE4_SPACE,
    PENDING_STAGE4_OPERATORS,
    STAGE2_CHOICES,
    architecture_sha256,
    build_fourstage_architecture,
    enumerate_base8,
    enumerate_cfem20,
    enumerate_dmdc16,
    enumerate_extended,
    enumerate_raw20,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import CFEMSonarBlock, build_model  # noqa: E402
from hwnas_fpga.training.protocol_reporting import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "frozen_search_space_v5_cfem20"
)
SOURCE_PDF = Path(r"C:/Users/Lenovo/Downloads/fmars-12-1539210.pdf")

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
    "cfem_sonar": "B5",
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
            "implementation_status": "implemented_existing",
        },
        {
            "candidate_id": "B4",
            "candidate_key": "dmdc_conv_sonar",
            "op_name": "DMDC_Conv",
            "operator_name": "dmdc_conv_sonar",
            "structure": (
                "parallel k={1,3,5} DMDC branches with a dilated-convolution "
                "path and an adaptive-pooling regional-context path per branch"
            ),
            "dilation_rates": [1, 3, 5],
            "kernel": 3,
            "expand_ratio": 1,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "implementation_status": "implemented",
        },
        {
            "candidate_id": "B5",
            "candidate_key": "cfem_sonar",
            "op_name": "CFEM",
            "operator_name": "cfem_sonar",
            "operator_full_name": "Context Feature Extraction Module",
            "structure": (
                "input feature -> three 1x1+dilated-3x3 branches with "
                "dilation rates 2/3/5 -> concat F -> CAM produces "
                "alpha/beta/gamma -> weighted feature aggregation -> 1x1 "
                "context projection Y; projected input X' and Y are fused by "
                "CBAM-derived dynamic weights, then channel/spatial attention "
                "is applied to form the output feature"
            ),
            "dilation_rates": [2, 3, 5],
            "kernel": 3,
            "expand_ratio": 1,
            "stride": 1,
            "input_shape": [32, 28, 28],
            "output_shape": [32, 28, 28],
            "source_type": "literature_operator",
            "source_task": "forward_looking_sonar_object_detection",
            "sonar_specific": True,
            "candidate_role": "sonar_context_multiscale_candidate",
            "source_paper": (
                "YOLO-SONAR: A lightweight object detection network for "
                "forward-looking sonar images"
            ),
            "source_doi": "10.3389/fmars.2025.1539210",
            "source_pdf": str(SOURCE_PDF).replace("\\", "/"),
            "core_mechanism": (
                "multiscale_dilated_context_extraction_with_attention_fusion"
            ),
            "implementation_status": "implemented",
            "adaptation_note": (
                "sealed as a 32->32 stride-1 Stage4 block without removing "
                "multiscale dilation, alpha/beta/gamma CAM weighting, context "
                "aggregation, or CBAM-style attention feature fusion"
            ),
        },
    ]


def pending_operator_definitions() -> list[dict[str, Any]]:
    msconv = PENDING_STAGE4_OPERATORS["msconv_sonar"]
    return [
        {
            "operator_name": "msconv_sonar",
            "active_space_status": msconv["implementation_status"],
            "active_searchable": msconv["active_searchable"],
            "source_type": "literature_operator",
            "source_task": "sonar_object_detection",
            "sonar_specific": True,
            "source_paper": (
                "MLFFNet: Multilevel Feature Fusion Network for Object "
                "Detection in Sonar Images"
            ),
            "source_doi": "10.1109/TGRS.2022.3214748",
            "core_mechanism": "multiscale_group_convolution_with_csp",
            "reason": msconv["reason"],
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
    for index, row in enumerate(enumerate_cfem20(), start=1):
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
                    "op_name": CFEM20_STAGE4_LABELS[stage4_key],
                },
                "architecture": row.architecture.to_dict(),
                "claim_boundary": (
                    "RAW candidate definition only; no accuracy, hardware "
                    "feasibility, deployability, LUT, route, board, or power "
                    "claim is made here."
                ),
            }
        )
    return rows


def historical_manifest_checks() -> dict[str, Any]:
    root = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
    paths = {
        "base8": root / "base8_candidates" / "base8_manifest.json",
        "extended16": root / "extended_candidates" / "extended_manifest.json",
        "raw20_v3": root / "frozen_search_space_v3_raw20" / "candidate_manifest_20.json",
        "dmdc16_v4": root / "frozen_search_space_v4_dmdc16" / "candidate_manifest_16.json",
    }
    return {
        **{f"{key}_manifest_path": rel(path) for key, path in paths.items()},
        **{f"{key}_manifest_exists": path.exists() for key, path in paths.items()},
        "base8_runtime_count": len(enumerate_base8()),
        "extended12_runtime_count": len(enumerate_extended(include_stage4_k5=False)),
        "extended16_runtime_count": len(enumerate_extended(include_stage4_k5=True)),
        "raw20_runtime_count": len(enumerate_raw20()),
        "dmdc16_runtime_count": len(enumerate_dmdc16()),
    }


def builder_checks() -> dict[str, Any]:
    architecture = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="cfem_sonar",
    )
    model = build_model(architecture, num_classes=8).eval()
    block = model.stages[3][0]
    return {
        "cfem_builder_recognized": isinstance(block, CFEMSonarBlock),
        "cfem_dilation_rates": list(getattr(block, "dilation_rates", ())),
        "cfem_implementation_status": getattr(block, "implementation_status", None),
    }


def source_file_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src" / "hwnas_fpga" / "fourstage_operator.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "builder.py",
        ROOT / "src" / "hwnas_fpga" / "models" / "__init__.py",
        ROOT / "scripts" / "freeze_fourstage_search_space_cfem20.py",
    ]
    result = {rel(path): sha256_file(path) for path in paths if path.exists()}
    if SOURCE_PDF.exists():
        result[str(SOURCE_PDF).replace("\\", "/")] = sha256_file(SOURCE_PDF)
    return result


def make_payloads() -> dict[str, dict[str, Any]]:
    stage2 = stage2_definitions()
    stage4 = stage4_definitions()
    rows = candidate_manifest()
    arch_ids = [row["architecture_id"] for row in rows]
    fixed_macro_errors = []
    for row in enumerate_cfem20():
        try:
            validate_frozen_fourstage(row.architecture)
        except ValueError as exc:
            fixed_macro_errors.append(f"{row.arch_id}: {exc}")
    historical = historical_manifest_checks()
    builder = builder_checks()
    active_stage4_keys = {row["candidate_key"] for row in stage4}
    audit_checks = {
        "stage2_candidate_count_eq_4": len(stage2) == 4,
        "stage4_candidate_count_eq_5": len(stage4) == 5,
        "raw_network_count_eq_20": len(rows) == 20,
        "architecture_ids_unique": len(arch_ids) == len(set(arch_ids)),
        "fixed_macroarchitecture_unchanged": not fixed_macro_errors,
        "cfem_builder_recognized": builder["cfem_builder_recognized"],
        "cfem_dilation_rates_eq_2_3_5": builder["cfem_dilation_rates"]
        == [2, 3, 5],
        "msconv_pending_not_in_active_space": "msconv_sonar"
        not in active_stage4_keys,
        "msconv_pending_original_spec": PENDING_STAGE4_OPERATORS[
            "msconv_sonar"
        ]["implementation_status"]
        == "pending_original_spec",
        "fused_and_ghost_excluded_from_current_space": {
            "fused_mbconv_e3",
            "ghost_bottleneck",
        }.isdisjoint(active_stage4_keys),
        "historical_configs_compatible": (
            historical["base8_manifest_exists"]
            and historical["extended16_manifest_exists"]
            and historical["raw20_v3_manifest_exists"]
            and historical["dmdc16_v4_manifest_exists"]
            and historical["base8_runtime_count"] == 8
            and historical["extended12_runtime_count"] == 12
            and historical["extended16_runtime_count"] == 16
            and historical["raw20_runtime_count"] == 20
            and historical["dmdc16_runtime_count"] == 16
        ),
    }
    definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_cfem20_v1",
        "status": "FROZEN_SEARCH_SPACE_DEFINITION",
        "description": (
            "Fixed MobileNetV2-style 4-Stage macroarchitecture. Stage2 keeps "
            "the four mature MBConv choices. Current active Stage4 choices "
            "are Skip, MBConv-K3-E3, MBConv-K5-E3, DMDC_Conv, and the "
            "literature-derived CFEM sonar context candidate."
        ),
        "fixed_macroarchitecture": fixed_macroarchitecture(),
        "stage2_candidates": stage2,
        "stage4_candidates": stage4,
        "pending_stage4_operators": pending_operator_definitions(),
        "excluded_from_current_space": excluded_operator_definitions(),
        "stage2_candidate_count": len(stage2),
        "stage4_candidate_count": len(stage4),
        "raw_network_count": len(rows),
        "enumeration_method": "deterministic_cartesian_product",
        "forbidden_methods": [
            "training",
            "NAS search",
            "HLS",
            "RTL",
            "Vivado route",
            "bitstream",
            "power measurement",
            "ZCU104 hardware test",
            "hardware LUT measurement",
        ],
        "source_file_sha256": source_file_hashes(),
    }
    operator_definition = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_cfem20_v1",
        "status": "FROZEN_OPERATOR_DEFINITIONS",
        "stage2_operator_definitions": stage2,
        "active_stage4_operator_definitions": stage4,
        "pending_stage4_operator_definitions": pending_operator_definitions(),
        "excluded_stage4_operator_definitions": excluded_operator_definitions(),
    }
    manifest = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_cfem20_v1",
        "status": "FROZEN_CANDIDATE_MANIFEST",
        "raw_network_count": len(rows),
        "rows": rows,
    }
    audit = {
        "schema_version": "1.0",
        "search_space_version": "fourstage_cfem20_v1",
        "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "checks": audit_checks,
        "fixed_macroarchitecture_errors": fixed_macro_errors,
        "historical_manifest_checks": historical,
        "builder_checks": builder,
        "raw_network_count": len(rows),
        "active_stage2_candidates": [row["candidate_key"] for row in stage2],
        "active_stage4_candidates": [row["candidate_key"] for row in stage4],
        "pending_stage4_operators": ["msconv_sonar"],
        "active_architecture_ids": arch_ids,
        "claim_boundary": {
            "training": "NOT_RUN",
            "nas_search": "NOT_RUN",
            "formal_evaluation": "NOT_RUN",
            "hardware_lut": "NOT_MEASURED",
            "hls": "NOT_RUN",
            "rtl": "NOT_RUN",
            "route": "NOT_RUN",
            "bitstream": "NOT_GENERATED",
            "zcu104_board": "NOT_RUN",
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
                "raw_network_count": audit["raw_network_count"],
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
