#!/usr/bin/env python3
"""Run a resumable shard of Gate 0 v2 prefix trajectories on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.analysis.proxy_collection import load_run_matrix  # noqa: E402
from hwnas_fpga.analysis.proxy_execution import (  # noqa: E402
    execute_prefix_work_unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-matrix", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="stage to run; defaults to phase_a_signal_discovery",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for formal Gate 0 training; use the single-unit "
            "runner with --max-budget for a CPU benchmark"
        )

    matrix = Path(args.run_matrix).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else matrix.parent / "observations"
    )
    stages = set(args.stage or ["phase_a_signal_discovery"])
    eligible = [
        unit
        for unit in load_run_matrix(matrix)
        if str(unit.get("stage")) in stages
    ]
    shard = [
        unit
        for index, unit in enumerate(eligible)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_units is not None:
        shard = shard[: max(0, int(args.max_units))]
    if not shard:
        raise ValueError(
            f"no units selected for stages={sorted(stages)}, "
            f"shard={args.shard_index}/{args.num_shards}"
        )

    status_path = (
        output_dir
        / "worker_status"
        / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    started = time.time()
    completed: list[str] = []
    failed: list[dict[str, str]] = []
    for unit in shard:
        try:
            output = execute_prefix_work_unit(
                unit,
                repo_root=REPO_ROOT,
                output_dir=output_dir,
                data_dir_override=args.data_dir,
                device_override=args.device,
                num_workers_override=args.num_workers,
                force=args.force,
            )
            completed.append(str(unit["work_id"]))
            print(f"completed {unit['work_id']} -> {output}", flush=True)
        except Exception as exc:  # noqa: BLE001 - persist scheduler failure context
            failed.append(
                {
                    "work_id": str(unit["work_id"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"failed {unit['work_id']}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.stop_on_error:
                break
        finally:
            _write_status(
                status_path,
                {
                    "schema_version": 1,
                    "run_matrix": str(matrix),
                    "stages": sorted(stages),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "selected_units": len(shard),
                    "completed_units": len(completed),
                    "failed_units": len(failed),
                    "completed_work_ids": completed,
                    "failures": failed,
                    "elapsed_seconds": time.time() - started,
                    "finished": len(completed) + len(failed) == len(shard),
                },
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
