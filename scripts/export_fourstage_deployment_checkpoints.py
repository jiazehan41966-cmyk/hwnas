#!/usr/bin/env python3
"""Export real checkpoints for the frozen four-stage deployment queue.

The export is deliberately conservative:

* candidate architectures are read from the frozen deployment selection JSON;
* checkpoint choice is fixed by fold/seed, not by outer-validation accuracy;
* generated weight packages live under results/;
* the tracked artifact is only a summary/hash index;
* activation calibration, integer reference parity, HLS, route, bitstream,
  COM5, and power remain separate downstream gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.deploy.quantization import (  # noqa: E402
    QuantizationConfig,
    export_checkpoint_quantized_weights,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


DEFAULT_SELECTION = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_deployment_candidate_selection.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sonar_fourstage_operator_v2"
    / "full_network_deployment_closure"
)
DEFAULT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_checkpoint_export_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def checkpoint_for_candidate(candidate: dict[str, Any], fold: int, seed: int) -> Path:
    expected = f"best_fold{fold}_seed{seed}.pt"
    for record in candidate.get("checkpoint_index") or []:
        path = Path(str(record.get("path") or ""))
        if path.name == expected:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint index path is missing: {path}")
            actual_sha = sha256_file(path)
            expected_sha = str(record.get("sha256") or "")
            if actual_sha != expected_sha:
                raise RuntimeError(
                    "Checkpoint SHA256 mismatch for "
                    f"{candidate['arch_id']} {expected}: "
                    f"{actual_sha} != {expected_sha}"
                )
            return path
    raise FileNotFoundError(f"{candidate['arch_id']} has no checkpoint {expected}")


def export_one(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    fold: int,
    seed: int,
    device: str,
    force: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint_for_candidate(candidate, fold, seed)
    export_dir = (
        output_root
        / f"{safe_name(candidate['role'])}__{safe_name(candidate['arch_id'])}"
        / f"fold{fold}_seed{seed}"
        / "checkpoint_export"
    )
    manifest_path = export_dir / "fourstage_checkpoint_export_manifest.json"
    quantized_path = export_dir / "quantized_weights_int8.pt"
    if manifest_path.is_file() and quantized_path.is_file() and not force:
        existing = read_json(manifest_path)
        return {**existing, "status": "SKIPPED_EXISTING"}

    export_dir.mkdir(parents=True, exist_ok=True)
    quantized_path, quant_summary = export_checkpoint_quantized_weights(
        checkpoint,
        output_path=quantized_path,
        device=device,
        config=QuantizationConfig(
            bit_width=8,
            scheme="symmetric",
            quantize_bias=True,
            input_scale=1.0 / 127.0,
            output_scale=1.0 / 127.0,
        ),
    )
    quant_summary_path = quantized_path.with_suffix(".json")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "checkpoint_selection": {
            "fold": fold,
            "seed": seed,
            "rule": (
                "fixed fold/seed hardware-proof checkpoint; not selected by "
                "outer-validation accuracy"
            ),
        },
        "source_checkpoint": evidence(checkpoint),
        "candidate_json": candidate["candidate_json"],
        "protocol_summary": candidate["protocol_summary"],
        "output_dir": str(export_dir.resolve()),
        "quantized_weight_package": evidence(quantized_path),
        "quantization_summary": evidence(quant_summary_path),
        "quantization_contract": {
            "status": "WEIGHT_EXPORT_ONLY",
            "schema_version": quant_summary.get("schema_version"),
            "num_quantized_tensors": quant_summary.get("num_quantized_tensors"),
            "layer_count": quant_summary.get("layer_count"),
            "parity_ready": bool(quant_summary.get("parity_ready")),
            "activation_calibration": "PENDING_REAL_ACTIVATION_CALIBRATION",
            "integer_reference": "PENDING_FULL_NETWORK_INT8_REFERENCE",
        },
        "claim_boundary": (
            "This passes the real-checkpoint export gate only. It does not "
            "establish calibrated INT8 parity, C-sim, HLS, route, bitstream, "
            "COM5 latency, or measured power."
        ),
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    selection_path = Path(args.selection).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    selection = read_json(selection_path)
    rows = [
        export_one(
            candidate,
            output_root=output_root,
            fold=args.fold,
            seed=args.seed,
            device=args.device,
            force=args.force,
        )
        for candidate in selection["selected_candidates"]
    ]
    all_pass = all(row.get("status") in {"PASS", "SKIPPED_EXISTING"} for row in rows)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_pass else "FAIL",
        "gate": "real_checkpoint_export",
        "selection": evidence(selection_path),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "checkpoint_selection_rule": {
            "fold": int(args.fold),
            "seed": int(args.seed),
            "rationale": (
                "Deterministic hardware-proof representative. The choice does "
                "not use outer-validation accuracy and does not alter the "
                "Protocol V2 accuracy conclusions."
            ),
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "downstream_gates": {
            "int8_activation_calibration": "PENDING_REAL_ACTIVATION_CALIBRATION",
            "python_full_network_int8_reference": "PENDING",
            "full_network_c_sim": "PENDING",
            "pytorch_int8_vs_csim_zero_mismatch": "PENDING",
            "hls_synthesis": "PENDING",
            "place_and_route_5ns": "PENDING",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "known_blockers_before_next_gate": [
            {
                "gate": "int8_activation_calibration",
                "status": "PENDING_IMPLEMENTATION_OR_ADAPTER",
                "reason": (
                    "Existing export path creates a real-checkpoint INT8 "
                    "weight package, but does not yet compute real activation "
                    "calibration scales or a full-network integer reference "
                    "for four-stage candidates."
                ),
            },
            {
                "gate": "full_network_hls_route",
                "status": "PENDING_IMPLEMENTATION_OR_ADAPTER",
                "reason": (
                    "Existing board harness scripts are bound to fixed "
                    "sonar_classifier/current84 flows and cannot be reused as "
                    "four-stage K5 full-network route evidence without a "
                    "candidate-specific exporter/harness adapter."
                ),
            },
        ],
        "claim_boundary": (
            "Real checkpoint export is complete for the frozen four-candidate "
            "queue. No calibrated INT8, C-sim, HLS, route, bitstream, COM5, or "
            "power claim is made."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
