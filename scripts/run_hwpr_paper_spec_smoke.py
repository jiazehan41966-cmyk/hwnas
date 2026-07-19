#!/usr/bin/env python3
"""Run a non-claimable HW-PR-NAS paper-spec loss smoke on real candidate logs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.hwpr import (
    audit_hwpr_author_runtime,
    candidate_from_record,
    fit_paper_spec_surrogate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--objectives", nargs="+", default=["f_clean", "f_robust", "latency_ms"]
    )
    parser.add_argument(
        "--directions", nargs="+", default=["max", "max", "min"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.candidates_jsonl).resolve()
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    candidates = [
        candidate_from_record(record)
        for record in records
        if record.get("feasible") is True
    ][: int(args.limit)]
    if len(candidates) != int(args.limit):
        raise RuntimeError(
            f"need exactly {args.limit} feasible evaluator records, found {len(candidates)}"
        )
    result = fit_paper_spec_surrogate(
        candidates,
        objectives=args.objectives,
        directions=args.directions,
        epochs=args.epochs,
        seed=args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "paper_id": "hw_pr_nas_2023",
        "method_id": "hwpr_paper_spec_listmle_local_tabular_smoke",
        "status": "PASS_NONCLAIMABLE_PAPER_SPEC_SMOKE",
        "claimability_status": "NOT_CLAIMABLE",
        "author_runtime": audit_hwpr_author_runtime(PROJECT_ROOT / "reference/HW-PR-NAS"),
        "candidate_source": str(source),
        "evaluator_record_count": len(candidates),
        "objectives": list(args.objectives),
        "directions": list(args.directions),
        "seed": args.seed,
        "epochs": args.epochs,
        "ranks": result.ranks,
        "scores": result.scores,
        "ordered_arch_ids": result.ordered_arch_ids,
        "final_loss": result.final_loss,
        "feature_schema": dict(result.feature_schema),
        "implementation_boundary": (
            "Implements equations 7-8 listwise Pareto ranking behavior with a local "
            "three-layer MLP and local tabular encoding. It is not the missing author "
            "feature+GCN+LSTM implementation and cannot populate T5."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
