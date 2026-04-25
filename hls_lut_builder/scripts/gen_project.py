#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_cases, load_config, materialize_case_project, select_pilot_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vitis HLS projects for operator screening")
    parser.add_argument("--config", required=True, help="Path to builder config YAML")
    parser.add_argument("--project-root", default=None, help="Optional override for generated project root")
    parser.add_argument("--operator", action="append", default=None, help="Generate only specific operators")
    parser.add_argument("--case", action="append", default=None, help="Generate only specific cases")
    parser.add_argument("--pilot", action="store_true", help="Generate one representative case per enabled operator")
    parser.add_argument("--force", action="store_true", help="Delete and recreate an existing case directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cases = build_cases(
        config,
        operator_filter=args.operator,
        case_filter=args.case,
        project_root_override=args.project_root,
        sort_cases=not args.pilot,
    )
    if args.pilot:
        cases = select_pilot_cases(cases)
    if not cases:
        raise SystemExit("No HLS cases matched the provided filters.")

    generated = []
    for case in cases:
        materialize_case_project(case, force=args.force)
        generated.append(case.case_name)

    print(f"Generated {len(generated)} HLS case directories.")
    for case_name in generated:
        print(case_name)


if __name__ == "__main__":
    main()
