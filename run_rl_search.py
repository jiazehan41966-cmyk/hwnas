#!/usr/bin/env python3
"""Compatibility wrapper for RL-based HW-NAS search."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "--search-method" not in sys.argv:
    sys.argv.extend(["--search-method", "rl"])

from run_search import main


if __name__ == "__main__":
    main()
