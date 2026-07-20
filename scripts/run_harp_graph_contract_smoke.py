#!/usr/bin/env python3
"""Validate one author HARP source/GEXF pair without legacy model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.harp import build_harp_program_graph_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="reference/HARP/dse_database/generated_graphs/machsuite/aes/aes.c",
    )
    parser.add_argument(
        "--graph",
        default=(
            "reference/HARP/dse_database/generated_graphs/machsuite/processed/"
            "extended-pseudo-block-connected-hierarchy/aes_processed_result.gexf"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/"
            "harp_author_graph_contract/graph_contract.json"
        ),
    )
    args = parser.parse_args()
    payload = build_harp_program_graph_manifest(
        sample_id="author_machsuite_aes",
        hls_source_path=PROJECT_ROOT / args.source,
        gexf_path=PROJECT_ROOT / args.graph,
        llvm_version="13 (author-declared repository graph provenance)",
        source_kind="author_database",
    )
    payload["status"] = (
        "PASS_AUTHOR_GRAPH_CONTRACT_ONLY"
        if payload["contract_valid"]
        else "FAIL_AUTHOR_GRAPH_CONTRACT"
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
