#!/usr/bin/env python3
"""Compatibility wrapper for the maintained visualization CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve().parent / "scripts" / "visualize_results.py"
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
