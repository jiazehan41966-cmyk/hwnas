#!/usr/bin/env python3
"""Deprecated retraining wrapper kept for backward compatibility."""

from __future__ import annotations

import sys


def main() -> int:
    message = (
        "retrain_best.py has been retired.\n"
        "Use run_retrain.py instead:\n"
        "  python run_retrain.py --run-dir <search-run-dir>\n"
        "or\n"
        "  python run_retrain.py --candidate-path <best_candidate.json>\n"
        "The previous implementation was moved to:\n"
        "  scripts/legacy/retrain_best_legacy.py\n"
    )
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
