#!/usr/bin/env python3
"""Audit G5 denoise/edge admission evidence and emit sonar_operator_gate.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.hardware.sonar_operator_gate import (
    audit_sonar_operator_manifest_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/sonar_operator_gate/sonar_operator_gate.json",
    )
    args = parser.parse_args()
    result = audit_sonar_operator_manifest_path(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": result["status"]}))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
