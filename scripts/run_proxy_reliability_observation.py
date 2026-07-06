#!/usr/bin/env python3
"""Run one frozen Gate 0 architecture/fold/seed/budget work unit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.analysis.proxy_execution import (  # noqa: E402
    execute_prefix_work_unit,
    execute_work_unit,
    load_work_unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-matrix", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--work-id")
    selector.add_argument("--work-index", type=int)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-budget",
        type=int,
        default=None,
        help="v2 benchmark only: stop at a registered budget and do not emit formal evidence",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = Path(args.run_matrix).expanduser().resolve()
    unit = load_work_unit(
        matrix,
        work_id=args.work_id,
        work_index=args.work_index,
    )
    is_prefix = str(unit.get("work_type")) == "prefix_train"
    default_output_name = (
        "benchmarks" if args.max_budget is not None else "observations"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else matrix.parent / default_output_name
    )
    if is_prefix:
        output = execute_prefix_work_unit(
            unit,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            data_dir_override=args.data_dir,
            device_override=args.device,
            num_workers_override=args.num_workers,
            max_budget_override=args.max_budget,
            force=args.force,
        )
    else:
        if args.max_budget is not None:
            raise ValueError("--max-budget is supported only for v2 prefix units")
        output = execute_work_unit(
            unit,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            data_dir_override=args.data_dir,
            device_override=args.device,
            num_workers_override=args.num_workers,
            force=args.force,
        )
    print(
        json.dumps(
            {
                "work_id": unit["work_id"],
                "observation": str(output),
                "status": (
                    "benchmark_completed"
                    if args.max_budget is not None
                    else "completed"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
