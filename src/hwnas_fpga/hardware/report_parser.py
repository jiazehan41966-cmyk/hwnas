"""Minimal Vivado/Vitis HLS report parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from hwnas_fpga.hardware.lookup_table import LutEntry, OpSpec


def _search_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _search_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_hls_report_text(report_text: str) -> dict[str, Any]:
    """Parse a subset of HLS report fields used for LUT construction."""
    cycles = (
        _search_int(r"Latency.*?max\s*=\s*([0-9,]+)", report_text)
        or _search_int(r"Latency\s*\(cycles\).*?([0-9,]+)\s*$", report_text)
        or _search_int(r"\bLatency\b.*?([0-9,]+)", report_text)
        or 0
    )
    latency_ns = _search_float(r"Latency\s*\(ns\).*?([0-9]+(?:\.[0-9]+)?)", report_text)
    bram = (
        _search_int(r"\bBRAM(?:_18K)?\b\s*\|?\s*([0-9,]+)", report_text)
        or _search_int(r"\bBRAM(?:_18K)?\b[^0-9]*([0-9,]+)", report_text)
        or 0
    )
    dsp = (
        _search_int(r"\bDSP(?:48E)?\b\s*\|?\s*([0-9,]+)", report_text)
        or _search_int(r"\bDSP(?:48E)?\b[^0-9]*([0-9,]+)", report_text)
        or 0
    )
    lut = _search_int(r"\bLUT\b\s*\|?\s*([0-9,]+)", report_text) or 0
    ff = _search_int(r"\bFF\b\s*\|?\s*([0-9,]+)", report_text) or 0
    power_w = _search_float(r"\bPower\b[^0-9]*([0-9]+(?:\.[0-9]+)?)", report_text) or 0.0
    return {
        "cycles": cycles,
        "latency_ns": latency_ns,
        "bram": bram,
        "dsp": dsp,
        "lut": lut,
        "ff": ff,
        "power_w": power_w,
    }


def parse_hls_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    return parse_hls_report_text(path.read_text(encoding="utf-8", errors="ignore"))


def lut_entry_from_report(
    *,
    op_spec: OpSpec,
    report_metrics: dict[str, Any],
    clock_mhz: int,
) -> LutEntry:
    cycles = int(report_metrics.get("cycles") or 0)
    if cycles > 0:
        latency_ms = cycles / (clock_mhz * 1_000)
    elif report_metrics.get("latency_ns") is not None:
        latency_ms = float(report_metrics["latency_ns"]) / 1_000_000.0
    else:
        latency_ms = 0.0

    power_w = float(report_metrics.get("power_w") or 0.0)
    energy_mj = power_w * latency_ms
    return LutEntry(
        op_spec=op_spec,
        latency_ms=latency_ms,
        cycles=cycles,
        dsp=int(report_metrics.get("dsp") or 0),
        bram=int(report_metrics.get("bram") or 0),
        lut=int(report_metrics.get("lut") or 0),
        power_w=power_w,
        energy_mj=energy_mj,
    )
