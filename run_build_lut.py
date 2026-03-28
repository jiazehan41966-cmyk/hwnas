#!/usr/bin/env python3
"""Build an FPGA LUT table from HLS profiling reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.hardware import save_lut_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FPGA LUT table from HLS profiling manifest")
    parser.add_argument("--manifest", required=True, help="YAML/JSON manifest describing profiled operators")
    parser.add_argument("--output", required=True, help="Output .pkl LUT table path")
    parser.add_argument("--clock-mhz", type=int, default=None, help="Default clock used when reports only contain cycles")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write the build summary as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = save_lut_from_manifest(
        args.manifest,
        args.output,
        default_clock_mhz=args.clock_mhz,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Built LUT table: {summary['output_path']}")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Entries: {summary['entries_built']} / {summary['entries_requested']}")
    print(f"Duplicates: {summary['table_stats']['duplicate_specs']}")


if __name__ == "__main__":
    main()
