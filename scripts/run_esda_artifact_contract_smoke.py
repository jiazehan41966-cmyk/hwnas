#!/usr/bin/env python3
"""Audit the pinned ESDA checkout as a non-claimable class-C workflow reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.esda import build_esda_evidence_chain_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="reference/ESDA")
    parser.add_argument(
        "--output",
        default=(
            "results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/"
            "esda_author_artifact_contract/evidence_chain.json"
        ),
    )
    args = parser.parse_args()
    payload = build_esda_evidence_chain_manifest(PROJECT_ROOT / args.checkout)
    payload["status"] = (
        "PASS_AUTHOR_ARTIFACT_CONTRACT_ONLY"
        if payload["contract_valid"]
        else "FAIL_AUTHOR_ARTIFACT_CONTRACT"
    )
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0 if payload["contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
