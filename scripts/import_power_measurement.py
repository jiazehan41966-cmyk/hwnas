#!/usr/bin/env python3
"""Audit external power-meter CSVs under the measurement-first contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.hardware.power_measurement import load_and_audit_power_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = load_and_audit_power_manifest(args.manifest)
    output = (
        Path(args.output)
        if args.output
        else Path(args.manifest).with_name("power_measurement_summary.json")
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "pass": result["overall_pass"]}))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
