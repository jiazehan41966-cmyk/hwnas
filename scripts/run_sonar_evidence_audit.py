#!/usr/bin/env python3
"""Run a config-driven sonar detection dataset evidence audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.data.evidence_audit import load_config, run_evidence_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-only sonar dataset evidence audit.")
    parser.add_argument("--config", required=True, type=Path, help="YAML audit config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = run_evidence_audit(config)
    print("evidence audit complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
