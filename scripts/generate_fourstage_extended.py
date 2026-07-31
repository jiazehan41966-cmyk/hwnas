#!/usr/bin/env python3
"""Materialize the gated fixed-macro 12/16 candidate enumeration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import (  # noqa: E402
    candidate_payload,
    enumerate_extended,
    parameter_and_mac_count,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction-summary",
        default="artifacts/sonar_fourstage_operator_v2/direction_gate_summary.json",
    )
    parser.add_argument(
        "--stage4-k5-hardware-summary",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "stage4_k5_exact_shape_hardware_gate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sonar_fourstage_operator_v2/extended_candidates",
    )
    args = parser.parse_args()
    direction_path = Path(args.direction_summary).resolve()
    direction = json.loads(direction_path.read_text(encoding="utf-8"))
    if direction.get("status") != "DIRECTIONAL_BASIS_PASS":
        raise ValueError("Dir candidates require DIRECTIONAL_BASIS_PASS")

    k5_path = Path(args.stage4_k5_hardware_summary).resolve()
    k5_gate = (
        json.loads(k5_path.read_text(encoding="utf-8"))
        if k5_path.is_file()
        else {}
    )
    include_k5 = bool(
        k5_gate.get("status") == "PASS"
        and k5_gate.get("shape") == {
            "input_resolution": 28,
            "in_channels": 32,
            "out_channels": 32,
            "stride": 1,
            "kernel_size": 5,
            "expand_ratio": 3,
        }
        and k5_gate.get("hls_synthesis") == "PASS"
        and k5_gate.get("micro_harness_route") == "PASS"
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in enumerate_extended(include_stage4_k5=include_k5):
        payload = candidate_payload(row)
        payload["software_cost"] = parameter_and_mac_count(row.architecture)
        payload["status"] = "READY_FOR_PROTOCOL_V2_FORMAL_EVALUATION"
        payload["payload_sha256"] = canonical_sha256(payload)
        path = output_dir / f"{row.arch_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "arch_id": row.arch_id,
                "path": str(path),
                "factors": row.factors(),
                "architecture_sha256": payload["candidate"][
                    "architecture_sha256"
                ],
                "parameter_count": payload["software_cost"]["parameter_count"],
                "macs": payload["software_cost"]["macs"],
            }
        )
    manifest = {
        "schema_version": 1,
        "design": "fixed_stage2_x_gated_stage4_enumeration",
        "candidate_count": len(rows),
        "stage4_k5_opened": include_k5,
        "stage4_k5_status": "READY" if include_k5 else "PENDING_HARDWARE",
        "direction_gate": {
            "path": str(direction_path),
            "payload_sha256": direction.get("payload_sha256"),
            "status": direction["status"],
        },
        "stage4_k5_hardware_gate": (
            {"path": str(k5_path), "status": k5_gate.get("status")}
            if k5_path.is_file()
            else {"path": str(k5_path), "status": "MISSING"}
        ),
        "rows": rows,
        "claim_boundary": (
            (
                "This is a 16-row enumeration after exact-shape Stage4 K5 "
                "HLS synthesis and operator micro-harness route passed. "
            )
            if include_k5
            else (
                "This is a 12-row enumeration until exact-shape Stage4 K5 "
                "HLS synthesis and operator micro-harness route both pass. "
            )
        )
        + (
            "Enumeration alone contains no new accuracy, complete-network "
            "route, board, or external-instrument power evidence."
        ),
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    target = output_dir / "extended_manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
