#!/usr/bin/env python3
"""Materialize the preregistered 2x2x2 four-stage candidate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import (  # noqa: E402
    candidate_payload,
    enumerate_base8,
    parameter_and_mac_count,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/sonar_fourstage_operator_v2/base8_candidates",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in enumerate_base8():
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
                "path": str(path.resolve()),
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
        "design": "2x2x2_full_factorial",
        "factors": {
            "kernel": ["K3", "K5"],
            "expansion": ["e3", "e6"],
            "stage4": ["MBConv", "Skip"],
        },
        "candidate_count": len(rows),
        "encoding": "ArchitectureSpec_StageSpec_BlockSpec",
        "macroarchitecture_frozen": True,
        "rows": rows,
        "claim_boundary": (
            "Parameter and MAC counts are exact software graph counts. They are "
            "not LUT, HLS, route, board, latency, or power measurements."
        ),
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    target = output_dir / "base8_manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
