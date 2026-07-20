#!/usr/bin/env python3
"""Audit at least three same-protocol external power-meter manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.hardware.power_measurement import audit_power_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="candidate power_measurement_manifest.json files")
    parser.add_argument("--output", default="results/power_measurement/power_campaign_summary.json")
    args = parser.parse_args()
    result = audit_power_campaign(args.manifests)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "overall_pass": result["overall_pass"]}))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
