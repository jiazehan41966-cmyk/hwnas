#!/usr/bin/env python3
"""Deprecated end-to-end wrapper kept for backward compatibility."""

from __future__ import annotations

import sys


def main() -> int:
    message = (
        "run_full_nas.py has been retired.\n"
        "Use the modular flow instead:\n"
        "  1. python run_search.py --config <search-config>\n"
        "  2. python run_retrain.py --run-dir <search-run-dir>\n"
        "  3. python run_export.py --checkpoint <checkpoint-path>\n"
        "The previous monolithic implementation was moved to:\n"
        "  scripts/legacy/run_full_nas_legacy.py\n"
    )
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
