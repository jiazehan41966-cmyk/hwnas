#!/usr/bin/env python3
"""Build portable, SHA-bound INT8 vectors for ZCU104 HLS on Linux."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from run_fourstage_csim_gate import (  # noqa: E402
    read_json,
    rebuild_inputs_from_summary_if_needed,
    write_json,
)


DEFAULT_INT8_SUMMARY = (
    ROOT / "artifacts" / "sonar_fourstage_operator_v2" / "fourstage_int8_reference_summary.json"
)
DEFAULT_CAMPAIGN_CONFIG = ROOT / "configs" / "experiments" / "zcu104_full_campaign_v1.yaml"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "zcu104_full_campaign_v1" / "inputs" / "hls_reference"
)


def verified_copy(record: dict[str, Any], destination: Path, label: str) -> dict[str, Any]:
    source = Path(str(record["path"])).expanduser().resolve()
    expected = str(record["sha256"])
    if not source.is_file():
        raise FileNotFoundError(f"{label} missing: {source}")
    if sha256_file(source) != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise RuntimeError(f"copied {label} SHA256 mismatch: {destination}")
    return {
        "path": destination.relative_to(ROOT).as_posix(),
        "sha256": expected,
        "size_bytes": destination.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8-summary", default=str(DEFAULT_INT8_SUMMARY))
    parser.add_argument("--campaign-config", default=str(DEFAULT_CAMPAIGN_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    int8_summary_path = Path(args.int8_summary).expanduser().resolve()
    config_path = Path(args.campaign_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    int8_summary = read_json(int8_summary_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bindings = {
        str(row["arch_id"]): row
        for row in config.get("candidates", [])
        if isinstance(row, dict)
    }
    rows = []
    for candidate in int8_summary.get("candidates", []):
        arch_id = str(candidate["arch_id"])
        role = str(candidate["role"])
        binding = bindings.get(arch_id)
        if not binding:
            raise RuntimeError(f"campaign binding missing for {arch_id}")
        candidate_root = output_root / role
        calibration_record = verified_copy(
            candidate["activation_calibration"],
            candidate_root / "activation_calibration.json",
            f"{arch_id} calibration",
        )
        reference_record = verified_copy(
            candidate["python_int8_reference"],
            candidate_root / "python_int8_reference.json",
            f"{arch_id} reference",
        )
        checkpoint_path = ROOT / str(binding["checkpoint"]["path"])
        if sha256_file(checkpoint_path) != str(binding["checkpoint"]["sha256"]):
            raise RuntimeError(f"campaign checkpoint SHA256 mismatch for {arch_id}")
        calibration = read_json(ROOT / calibration_record["path"])
        reference = read_json(ROOT / reference_record["path"])
        inputs = rebuild_inputs_from_summary_if_needed(
            reference,
            calibration,
            candidate_row=candidate,
            int8_summary=int8_summary,
        )
        expected = [
            [int(value) for value in row["logits_int8"]]
            for row in reference["reference_records"]
        ]
        inputs_array = np.asarray(inputs, dtype=np.int8)
        expected_array = np.asarray(expected, dtype=np.int8)
        inputs_path = candidate_root / "inputs.npy"
        expected_path = candidate_root / "expected.npy"
        np.save(inputs_path, inputs_array, allow_pickle=False)
        np.save(expected_path, expected_array, allow_pickle=False)
        vectors_record = {
            "inputs": {
                "path": inputs_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(inputs_path),
                "size_bytes": inputs_path.stat().st_size,
                "shape": list(inputs_array.shape),
                "dtype": str(inputs_array.dtype),
            },
            "expected": {
                "path": expected_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(expected_path),
                "size_bytes": expected_path.stat().st_size,
                "shape": list(expected_array.shape),
                "dtype": str(expected_array.dtype),
            },
        }
        manifest = {
            "schema_version": 1,
            "campaign_id": "zcu104_full_campaign_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "arch_id": arch_id,
            "architecture_sha256": str(binding["architecture_sha256"]),
            "checkpoint": binding["checkpoint"],
            "activation_calibration": calibration_record,
            "python_int8_reference": reference_record,
            "test_vectors": vectors_record,
            "reference_sample_count": len(inputs),
            "outer_validation_accessed": False,
            "claim_boundary": (
                "Portable test vectors reproduce the frozen INT8 C-sim inputs only. "
                "They are not HLS, RTL co-sim, route, board, or power evidence."
            ),
        }
        manifest["payload_sha256"] = canonical_sha256(manifest)
        manifest_path = candidate_root / "manifest.json"
        write_json(manifest_path, manifest)
        rows.append(
            {
                "role": role,
                "arch_id": arch_id,
                "manifest": {
                    "path": manifest_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(manifest_path),
                    "size_bytes": manifest_path.stat().st_size,
                },
            }
        )
    summary = {
        "schema_version": 1,
        "campaign_id": "zcu104_full_campaign_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if len(rows) == 4 else "FAIL",
        "candidate_count": len(rows),
        "int8_summary": {
            "path": int8_summary_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(int8_summary_path),
        },
        "candidates": rows,
        "claim_boundary": "This proves portable HLS input traceability only.",
    }
    summary["payload_sha256"] = canonical_sha256(summary)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
