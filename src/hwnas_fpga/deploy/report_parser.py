"""Deployment-facing report parser helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hwnas_fpga.hardware import parse_hls_report, parse_hls_report_text


def parse_backend_report(report_path: str | Path) -> dict[str, Any]:
    return parse_hls_report(report_path)


def parse_backend_report_text(report_text: str) -> dict[str, Any]:
    return parse_hls_report_text(report_text)
