#!/usr/bin/env python3
"""Launch official ProxylessNAS (reference clone) for NKSID strict-LUT search."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.search.official_proxyless_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
