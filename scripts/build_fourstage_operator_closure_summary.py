#!/usr/bin/env python3
"""Build the final evidence ledger for the four-stage operator closure."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def evidence(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def maybe_evidence(relative: str) -> dict[str, Any] | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    return evidence(relative)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_source_freeze(relative_manifest: str) -> dict[str, Any]:
    manifest_path = ROOT / relative_manifest
    if not manifest_path.is_file():
        return {"status": "NOT_GENERATED", "manifest": str(manifest_path)}
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"status": "PARSE_FAIL", "stdout": completed.stdout}
    payload["returncode"] = completed.returncode
    return payload


def source_freeze_snapshot() -> dict[str, Any]:
    experiment_manifest = (
        "artifacts/sonar_fourstage_operator_v2/"
        "closure_source_freeze/source_freeze_manifest.json"
    )
    slim_pr_manifest = (
        "artifacts/sonar_fourstage_operator_v2/"
        "slim_pr_source_freeze/source_freeze_manifest.json"
    )
    archive_index = (
        "artifacts/sonar_fourstage_operator_v2/"
        "source_snapshot_archive_index.json"
    )
    return {
        "experiment_closure_source_freeze": {
            "manifest": maybe_evidence(experiment_manifest),
            "current_verification": (
                "NOT_RECHECKED_AFTER_SLIM_PR_HOUSEKEEPING"
            ),
            "reason": (
                "This manifest binds the source state used for formal "
                "operator closure before PR housekeeping. Current PR code is "
                "verified by slim_pr_current_source_freeze."
            ),
        },
        "slim_pr_current_source_freeze": {
            "verification": verify_source_freeze(slim_pr_manifest),
            "manifest": maybe_evidence(slim_pr_manifest),
        },
        "source_snapshot_archive_index": maybe_evidence(archive_index),
    }


def count_run_units(relative_dir: str) -> int:
    return len(list((ROOT / relative_dir).glob("*/run_fold*_seed*.json")))


def protocol_snapshot() -> dict[str, Any]:
    binding = load_json(
        "artifacts/sonar_fourstage_operator_v2/protocol_v2_freeze_binding.json"
    )
    bundle = load_json(
        "results/sonar_fourstage_operator_v2/protocol_selection/"
        "bundle_decision.json"
    )
    geometry = load_json(
        "results/sonar_fourstage_operator_v2/protocol_selection/"
        "geometry_decision.json"
    )
    return {
        "status": "FROZEN",
        "protocol_file": evidence(
            "configs/evaluation/nksid_frozen_protocol_v2.yaml"
        ),
        "freeze_binding": {
            "status": binding["status"],
            "source_manifest_status": binding["source_freeze"][
                "verification"
            ]["status"],
            "path": evidence(
                "artifacts/sonar_fourstage_operator_v2/"
                "protocol_v2_freeze_binding.json"
            ),
        },
        "selection": {
            "bundle": bundle["selected_bundle"],
            "geometry": geometry["selected_geometry"],
            "outer_accessed": (
                bool(bundle["outer_validation_accessed"])
                or bool(geometry["outer_validation_accessed"])
            ),
            "bundle_decision_path": evidence(
                "results/sonar_fourstage_operator_v2/protocol_selection/"
                "bundle_decision.json"
            ),
            "geometry_decision_path": evidence(
                "results/sonar_fourstage_operator_v2/protocol_selection/"
                "geometry_decision.json"
            ),
        },
    }


def base8_snapshot() -> dict[str, Any]:
    analysis = load_json(
        "artifacts/sonar_fourstage_operator_v2/base8_formal_analysis.json"
    )
    lut = load_json(
        "artifacts/sonar_fourstage_operator_v2/base8_strict_lut_proxy_audit.json"
    )
    kernel = analysis["factorial_effects"]["kernel_K5_minus_K3"]
    stage4 = analysis["factorial_effects"]["stage4_MBConv_minus_Skip"]
    return {
        "status": analysis["status"],
        "formal_unit_count": analysis["formal_unit_count"],
        "observed_run_json_count": count_run_units(
            "results/sonar_fourstage_operator_v2/base8_formal"
        ),
        "design": analysis["design"],
        "stage2_k5_minus_k3": {
            "mean_delta": kernel["mean_delta"],
            "ci95_low": kernel["ci95_low"],
            "ci95_high": kernel["ci95_high"],
            "positive_fold_means": kernel["positive_fold_means"],
            "exact_paired_sign_flip_p": kernel[
                "exact_paired_sign_flip_p"
            ],
        },
        "stage4_mbconv_minus_skip": {
            "mean_delta": stage4["mean_delta"],
            "ci95_low": stage4["ci95_low"],
            "ci95_high": stage4["ci95_high"],
            "positive_fold_means": stage4["positive_fold_means"],
            "exact_paired_sign_flip_p": stage4[
                "exact_paired_sign_flip_p"
            ],
        },
        "strict_lut_proxy": {
            "status": lut["status"],
            "candidate_count": lut["candidate_count"],
            "all_candidates_strict_lut_covered": lut[
                "all_candidates_strict_lut_covered"
            ],
            "full_network_route_completed": lut[
                "full_network_route_completed"
            ],
            "operator_micro_harness_route_completed": lut[
                "operator_micro_harness_route_completed"
            ],
        },
        "analysis_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/base8_formal_analysis.json"
        ),
        "strict_lut_proxy_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/"
            "base8_strict_lut_proxy_audit.json"
        ),
    }


def direction_snapshot() -> dict[str, Any]:
    direction = load_json(
        "artifacts/sonar_fourstage_operator_v2/direction_gate_summary.json"
    )
    return {
        "status": direction["status"],
        "outer_accessed": direction["outer_validation_accessed"],
        "summary_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/direction_gate_summary.json"
        ),
    }


def dir_snapshot() -> dict[str, Any]:
    accuracy = load_json(
        "artifacts/sonar_fourstage_operator_v2/dir_accuracy_gate.json"
    )
    return {
        "operator_state": accuracy["operator_state"],
        "accuracy_status": accuracy["status"],
        "formal_dir_units": accuracy["formal_dir_units"],
        "formal_control_units_reused": accuracy[
            "formal_control_units_reused"
        ],
        "downstream_gate": accuracy["downstream_gate"],
        "accuracy_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/dir_accuracy_gate.json"
        ),
        "software_pretrain_gate": evidence(
            "artifacts/sonar_fourstage_operator_v2/"
            "dir_pretrain_software_gate.json"
        ),
        "int8_hls_pretrain_gate": evidence(
            "artifacts/sonar_fourstage_operator_v2/"
            "dir_int8_hls_pretrain_gate.json"
        ),
    }


def stage4_k5_snapshot() -> dict[str, Any]:
    analysis = load_json(
        "artifacts/sonar_fourstage_operator_v2/stage4_k5_formal_analysis.json"
    )
    hardware = load_json(
        "artifacts/sonar_fourstage_operator_v2/"
        "stage4_k5_exact_shape_hardware_gate.json"
    )
    passed = [
        key
        for key, row in analysis["comparisons"].items()
        if row["passes_stage4_k5_accuracy_gate"]
    ]
    return {
        "status": analysis["status"],
        "stage4_k5_state": analysis["stage4_k5_state"],
        "formal_k5_units": analysis["formal_k5_units"],
        "formal_k3_control_units_reused": analysis[
            "formal_k3_control_units_reused"
        ],
        "observed_extended_run_json_count": count_run_units(
            "results/sonar_fourstage_operator_v2/extended_formal"
        ),
        "passed_stage2_backgrounds": passed,
        "aggregate_stage4_k5_minus_k3": analysis[
            "aggregate_stage4_k5_minus_k3"
        ],
        "micro_harness_gate": {
            "status": hardware["status"],
            "shape": hardware["shape"],
            "hls_synthesis": hardware["hls_synthesis"],
            "micro_harness_route": hardware["micro_harness_route"],
            "hls": hardware["hls"],
            "micro_harness": hardware["micro_harness"],
            "av7k325_ffg900_micro_route_attempt": hardware[
                "av7k325_ffg900_micro_route_attempt"
            ],
            "power": hardware["power"],
        },
        "formal_analysis_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_formal_analysis.json"
        ),
        "hardware_gate_path": evidence(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_exact_shape_hardware_gate.json"
        ),
    }


def main() -> int:
    protocol = protocol_snapshot()
    base8 = base8_snapshot()
    direction = direction_snapshot()
    dir_v1 = dir_snapshot()
    stage4_k5 = stage4_k5_snapshot()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_status": "GENERAL_OP_SELECTED",
        "final_status_reason": (
            "Dir-v1 reached the direction basis gate but failed the "
            "preregistered accuracy admission gate. Mature MBConv-k5 has "
            "formal Protocol V2 accuracy support in Stage2, and Stage4 K5 "
            "also passes accuracy on 3 of 4 Stage2 backgrounds after the "
            "exact-shape micro-harness hardware gate opened it."
        ),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output(
                "status", "--porcelain=v1"
            ).splitlines(),
        },
        "source_freezes": source_freeze_snapshot(),
        "macro_architecture_frozen": {
            "input": "1x224x224",
            "stem": "Conv3x3 1->32 stride=2",
            "stage_channels": [16, 24, 32, 32],
            "stage_strides": [1, 2, 2, 1],
            "stage_depths": [1, 1, 1, 1],
            "head": "GAP->FC8",
            "not_searched": [
                "width",
                "depth",
                "stage_count",
                "stride",
                "stem",
                "head",
                "input_resolution",
                "activation",
            ],
        },
        "protocol_v2": protocol,
        "base8_factorial": base8,
        "direction_gate": direction,
        "dir_v1": dir_v1,
        "stage4_k5": stage4_k5,
        "deployment_candidate_selection": (
            maybe_evidence(
                "artifacts/sonar_fourstage_operator_v2/"
                "fourstage_deployment_candidate_selection.json"
            )
        ),
        "checkpoint_export_gate": (
            maybe_evidence(
                "artifacts/sonar_fourstage_operator_v2/"
                "fourstage_checkpoint_export_summary.json"
            )
        ),
        "int8_reference_gate": (
            maybe_evidence(
                "artifacts/sonar_fourstage_operator_v2/"
                "fourstage_int8_reference_summary.json"
            )
        ),
        "csim_zero_mismatch_gate": (
            maybe_evidence(
                "artifacts/sonar_fourstage_operator_v2/"
                "fourstage_csim_zero_mismatch_summary.json"
            )
        ),
        "operator_states": {
            "MBConv-k5-e3@Stage2@ProtocolV2@16to24_s2": (
                "READY_FORMAL_ACCURACY_STRICT_LUT_PROXY_FULL_NETWORK_CSIM_BOUNDARY"
            ),
            "MBConv-k5-e3@Stage4@ProtocolV2@28x28_32to32_s1": (
                "READY_ACCURACY_SUPPORTED_MICRO_HARNESS_ROUTE_FULL_NETWORK_CSIM_BOUNDARY"
            ),
            "Edge_v1@ProtocolV1": "NOT_ADMITTED",
            "Edge@ProtocolV2": "NOT_RETESTED",
            "old_MixConv": "PAUSED",
            "MixConv-v2": "NOT_TESTED",
            "dir_mbconv3_split11_e3_v1@Stage4@ProtocolV2@28x28_32to32_s1": (
                "NOT_ADMITTED_ACCURACY_GATE_FAILED"
            ),
        },
        "evidence_boundaries": {
            "eval10": "NOT_USED_AS_NEW_FORMAL_CONCLUSION",
            "strict_lut_proxy": (
                "Kept separate from operator micro-harness route and "
                "complete-network route."
            ),
            "operator_micro_harness_route": (
                "Stage4 K5 exact-shape route evidence only; not a "
                "complete-network route."
            ),
            "complete_network_csim": (
                "Four frozen deployment representatives passed Vitis HLS "
                "C-sim integer zero-mismatch against the Python INT8 "
                "reference. This is not RTL co-sim, HLS synthesis, "
                "full-network route, bitstream, board, or power evidence."
            ),
            "complete_network_route": "NOT_RUN",
            "rtl_cosim": "NOT_RUN",
            "bitstream": "NOT_GENERATED",
            "com5_board_run": "NOT_RUN",
            "power": "NOT_MEASURED",
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = (
        ROOT
        / "artifacts/sonar_fourstage_operator_v2/"
        "fourstage_operator_closure_summary.json"
    )
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
