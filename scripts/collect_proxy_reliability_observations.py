#!/usr/bin/env python3
"""Validate Gate 0 work records and build the long classification CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.analysis.proxy_collection import (  # noqa: E402
    collect_observations,
    write_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-matrix", required=True)
    parser.add_argument("--observations-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="collect only a named v2 stage; repeat for multiple stages",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = Path(args.run_matrix).expanduser().resolve()
    observations = (
        Path(args.observations_dir).expanduser().resolve()
        if args.observations_dir
        else matrix.parent / "observations"
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            matrix.parent / f"collected_{observations.name}"
            if args.observations_dir
            else matrix.parent / "collected"
        )
    )
    rows, summary = collect_observations(
        run_matrix=matrix,
        observations_dir=observations,
        stages=args.stage,
    )
    paths = write_collection(output, rows=rows, summary=summary)
    console_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"missing_work_ids", "invalid_records"}
    }
    console_summary["missing_work_ids_preview"] = summary["missing_work_ids"][:10]
    console_summary["invalid_records_preview"] = summary["invalid_records"][:10]
    print(json.dumps({**console_summary, **paths}, ensure_ascii=False, indent=2))
    required_ready = (
        summary["ready_for_selected_scope"]
        if args.stage
        else summary["ready_for_formal_analysis"]
    )
    if args.require_complete and not required_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
