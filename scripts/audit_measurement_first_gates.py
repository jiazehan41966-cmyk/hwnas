#!/usr/bin/env python3
"""Build one honest G0-G5 status ledger from traceable repository artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.research_gates import stage3_gate_status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def run_patch_integrity(summary_path: Path) -> bool:
    """Verify the manifest-bound tracked patch instead of trusting its summary."""

    manifest_path = summary_path.parent / "run_manifest.json"
    try:
        manifest = read_json(manifest_path) or {}
        tracked_patch = (
            (manifest.get("code_provenance") or {}).get("tracked_patch") or {}
        )
        expected_sha256 = str(tracked_patch.get("sha256") or "")
        raw_path = str(tracked_patch.get("path") or "")
        if (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
            or not raw_path
        ):
            return False
        patch_path = Path(raw_path)
        if not patch_path.is_absolute():
            patch_path = manifest_path.parent / patch_path
        return patch_path.is_file() and sha256_file(patch_path) == expected_sha256
    except (OSError, TypeError, ValueError):
        return False


def g1_status(root: Path) -> dict[str, Any]:
    legacy_clean_root = root / "results/protocol/g1_clean_20260711"
    repaired_clean_root = root / "results/protocol/g1_clean_20260718"
    paths = {
        "scratch": repaired_clean_root
        / "g1_mobilenet_v2_scratch_v2/protocol_summary.json",
        "pretrained": legacy_clean_root
        / "g1_mobilenet_v2_grayscale_imagenet/protocol_summary.json",
        "rl_arch_135": legacy_clean_root
        / "g1_rl_arch_135_legacy_selected/protocol_summary.json",
    }
    summaries = {name: read_json(path) for name, path in paths.items()}
    gates = {
        f"{name}_claimable": bool(
            payload and (payload.get("claimability") or {}).get("claimable") is True
        )
        for name, payload in summaries.items()
    }
    pretrained = summaries["pretrained"] or {}
    gates["pretrained_loaded"] = (
        (pretrained.get("model") or {}).get("pretrained_loaded") is True
    )
    completed = sum(len((payload or {}).get("runs", [])) for payload in summaries.values())
    gates["completed_45_of_45"] = completed == 45
    for name, payload in summaries.items():
        prefix = f"{name}_"
        gates[prefix + "tracked_patch_integrity"] = run_patch_integrity(paths[name])
        gates[prefix + "uniform_run_fingerprint"] = bool(
            payload
            and len(payload.get("run_fingerprints", [])) == 1
            and len(str(payload.get("run_fingerprints", [""])[0])) == 64
        )
        gates[prefix + "uniform_protocol_context"] = bool(
            payload and len(str(payload.get("protocol_context_sha256", ""))) == 64
        )
        gates[prefix + "grayscale_normalization"] = bool(
            payload
            and (payload.get("normalization") or {}).get("mean") == [0.5]
            and (payload.get("normalization") or {}).get("std") == [0.5]
        )
    passed = all(gates.values())
    return {
        "status": "PASS" if passed else "PENDING",
        "pass": passed,
        "completed_tasks": completed,
        "required_tasks": 45,
        "gates": gates,
        "evidence": {name: str(path) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g4-int8",
        default="results/g4_rl_arch_193_fold1_seed42/int8_ptq_summary.json",
    )
    parser.add_argument(
        "--g4-parity",
        default="results/g4_rl_arch_193_fold1_seed42/int8_hls_parity_summary.json",
    )
    parser.add_argument(
        "--g4-board",
        default="results/g4_rl_arch_193_fold1_seed42/board_validation_summary.json",
    )
    parser.add_argument(
        "--power",
        default="results/power_measurement/power_campaign_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/measurement_first_rebuild",
    )
    args = parser.parse_args()

    protocol_config = REPO_ROOT / "configs/evaluation/nksid_frozen_protocol_v1.yaml"
    entrypoint = REPO_ROOT / "run_eval_protocol.py"
    g0_pass = protocol_config.exists() and entrypoint.exists()
    g0 = {
        "status": "PASS" if g0_pass else "FAIL",
        "pass": g0_pass,
        "protocol_config": str(protocol_config),
        "protocol_config_sha256": sha256_file(protocol_config) if protocol_config.exists() else None,
        "entrypoint": str(entrypoint),
        "entrypoint_sha256": sha256_file(entrypoint) if entrypoint.exists() else None,
        "claim_rule": (
            "Only exact folds 0-4, seeds 42/43/44, no outer selection, "
            "and complete checkpoint/split/code provenance are claimable."
        ),
    }
    g1 = g1_status(REPO_ROOT)

    calibration_path = (
        REPO_ROOT / "artifacts/hw_surrogate_calibration_v2/calibration_v2.json"
    )
    calibration = read_json(calibration_path)
    g2_gates = {
        "declared_pass": bool(calibration and calibration.get("g2_pass") is True),
        "independent_full_network_probes": bool(
            calibration
            and int(
                calibration.get(
                    "independent_full_network_probe_count",
                    calibration.get("probe_count", 0),
                )
            ) >= 4
        ),
        "semantic_safe_full_network_samples": bool(
            calibration
            and int(
                calibration.get(
                    "semantic_safe_full_network_sample_count",
                    calibration.get("semantic_safe_sample_count", 0),
                )
            ) >= 8
        ),
        "leave_one_architecture_out": bool(
            calibration and calibration.get("leave_one_architecture_out") is True
        ),
        "shortlist_coverage_100pct": bool(
            calibration
            and float(calibration.get("shortlist_coverage", 0.0)) >= 1.0
        ),
        "p90_p95_quality": bool(
            calibration and calibration.get("p90_p95_quality_gate") is True
        ),
    }
    g2_pass = all(g2_gates.values())
    g2 = {
        "status": "PASS" if g2_pass else "PENDING",
        "pass": g2_pass,
        "gates": g2_gates,
        "evidence": str(calibration_path),
        "blockers": (calibration or {}).get(
            "g2_blockers", ["calibration_v2.json is missing"]
        ),
    }
    g3 = stage3_gate_status(REPO_ROOT)

    int8_path = REPO_ROOT / args.g4_int8
    parity_path = REPO_ROOT / args.g4_parity
    board_path = REPO_ROOT / args.g4_board
    int8 = read_json(int8_path)
    parity = read_json(parity_path)
    board = read_json(board_path)
    g4_gates = {
        "ptq_or_qat_accuracy": bool(
            int8 and (int8.get("ptq_gate") or {}).get("pass") is True
        ),
        "hls_bit_exact_parity": bool(
            parity and parity.get("overall_pass") is True
        ),
        "full_outer_validation_board": bool(
            board and board.get("claimable") is True
        ),
        "zero_board_numeric_mismatch": bool(
            board and int(board.get("numeric_mismatch_count", -1)) == 0
        ),
        "no_missing_board_samples": bool(
            board and not board.get("missing_sample_ids", [None])
        ),
    }
    g4_pass = all(g4_gates.values())
    g4 = {
        "status": "PASS" if g4_pass else "PENDING",
        "pass": g4_pass,
        "target": "rl_arch_193/fold1/seed42",
        "gates": g4_gates,
        "evidence": {
            "int8": str(int8_path),
            "parity": str(parity_path),
            "board": str(board_path),
        },
    }

    power_path = REPO_ROOT / args.power
    power = read_json(power_path)
    power_gates = {
        "declared_pass": bool(power and power.get("overall_pass") is True),
        "three_candidates": bool(
            power
            and int(power.get("candidate_count", len(power.get("candidates", [])))) >= 3
        ),
        "same_instrument_protocol": bool(
            power and power.get("same_instrument_protocol") is True
        ),
        "raw_csv_hashes": bool(power and power.get("raw_csv_hashes_complete") is True),
    }
    power_pass = all(power_gates.values())
    power_status = {
        "status": "MEASURED" if power_pass else "NOT_MEASURED",
        "pass": power_pass,
        "gates": power_gates,
        "pareto_eligible": False,
        "evidence": str(power_path),
        "reentry_rule": (
            "At least three candidates must pass with the same instrument and protocol."
        ),
    }

    sonar_path = REPO_ROOT / "artifacts/sonar_operator_gate/sonar_operator_gate.json"
    sonar = read_json(sonar_path)
    g5_pass = bool(sonar and sonar.get("overall_pass") is True)
    g5 = {
        "status": "PASS" if g5_pass else "PAUSED",
        "pass": g5_pass,
        "evidence": str(sonar_path),
        "operators": (sonar or {}).get("operators", {}),
    }

    payload = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "overall_status": (
            "COMPLETE" if all(item["pass"] for item in (g0, g1, g2, g4, g5))
            else "IN_PROGRESS"
        ),
        "gates": {
            "G0_protocol": g0,
            "G1_accuracy_baselines": g1,
            "G2_hardware_measurement": g2,
            "G3_search": g3,
            "G4_int8_board": g4,
            "power": power_status,
            "G5_sonar_ablation": g5,
        },
        "claim_boundary": (
            "Implemented software gates are not experimental results. Missing "
            "training, csynth, route, COM5, or meter evidence remains PENDING."
        ),
    }
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "status.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Measurement-first rebuild status",
        "",
        f"- overall: `{payload['overall_status']}`",
        f"- boundary: {payload['claim_boundary']}",
        "",
        "| gate | status |",
        "|---|---|",
    ]
    for name, status in payload["gates"].items():
        lines.append(f"| {name} | {status['status']} |")
    lines += ["", "## Current blockers", ""]
    for blocker in g2["blockers"]:
        lines.append(f"- G2: {blocker}")
    for name, passed in g4_gates.items():
        if not passed:
            lines.append(f"- G4: {name}")
    if not power_pass:
        lines.append("- power: external meter CSV acceptance has not passed")
    if not g5_pass:
        lines.append("- G5: denoise/edge remain paused")
    (output_dir / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_json), "status": payload["overall_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
