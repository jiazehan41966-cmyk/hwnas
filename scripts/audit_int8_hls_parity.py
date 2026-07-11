#!/usr/bin/env python3
"""Audit per-layer integer simulator/HLS records before board validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.deploy.parity_gate import audit_parity_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument(
        "--quantization-contract",
        default="per_tensor_symmetric_int8_v2",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit_parity_jsonl(
        args.records,
        quantization_contract=args.quantization_contract,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.records).with_name("int8_hls_parity_summary.json")
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": result["status"]}))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
