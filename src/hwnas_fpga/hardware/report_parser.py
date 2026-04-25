"""Minimal Vivado/Vitis HLS report parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

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


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_int_like(value: str) -> Optional[int]:
    match = re.search(r"[-+]?[0-9][0-9,]*", value)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def _parse_float_like(value: str) -> Optional[float]:
    match = re.search(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", value)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _parse_time_like_ns(value: str) -> Optional[float]:
    match = re.search(
        r"(?P<number>[-+]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>ns|us|µs|ms|s)?",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    number = float(match.group("number").replace(",", ""))
    unit = (match.group("unit") or "ns").lower()
    scale = {
        "ns": 1.0,
        "us": 1_000.0,
        "µs": 1_000.0,
        "ms": 1_000_000.0,
        "s": 1_000_000_000.0,
    }.get(unit)
    if scale is None:
        return None
    return number * scale


def _find_xml_child(element: ET.Element | None, name: str) -> Optional[ET.Element]:
    if element is None:
        return None
    target = _normalize_token(name)
    for child in element:
        if _normalize_token(_strip_namespace(child.tag)) == target:
            return child
    return None


def _find_xml_path(element: ET.Element | None, *path: str) -> Optional[ET.Element]:
    current = element
    for name in path:
        current = _find_xml_child(current, name)
        if current is None:
            return None
    return current


def _xml_text(element: ET.Element | None, *path: str) -> Optional[str]:
    target = _find_xml_path(element, *path)
    if target is None:
        return None
    text = (target.text or "").strip()
    return text or None


def _collect_xml_scalars(root: ET.Element) -> dict[str, list[str]]:
    scalars: dict[str, list[str]] = {}
    for element in root.iter():
        key = _normalize_token(_strip_namespace(element.tag))
        text = (element.text or "").strip()
        if not key or not text:
            continue
        scalars.setdefault(key, []).append(text)
    return scalars


def _pick_int(scalars: dict[str, list[str]], *keys: str) -> Optional[int]:
    for key in keys:
        values = scalars.get(_normalize_token(key), ())
        for value in reversed(values):
            parsed = _parse_int_like(value)
            if parsed is not None:
                return parsed
    return None


def _pick_float(scalars: dict[str, list[str]], *keys: str) -> Optional[float]:
    for key in keys:
        values = scalars.get(_normalize_token(key), ())
        for value in reversed(values):
            parsed = _parse_float_like(value)
            if parsed is not None:
                return parsed
    return None


def _pick_time_ns(scalars: dict[str, list[str]], *keys: str) -> Optional[float]:
    for key in keys:
        values = scalars.get(_normalize_token(key), ())
        for value in reversed(values):
            parsed = _parse_time_like_ns(value)
            if parsed is not None:
                return parsed
    return None


def _fmax_from_clock_period(clock_period_ns: Optional[float]) -> Optional[float]:
    if clock_period_ns is None or clock_period_ns <= 0:
        return None
    return 1_000.0 / float(clock_period_ns)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_pipeline_metrics_from_log_text(log_text: str) -> dict[str, Any]:
    matches = re.findall(
        r"Pipelining result\s*:\s*Target II =\s*([0-9,]+),\s*Final II =\s*([0-9,]+),\s*Depth =\s*([0-9,]+),\s*loop '([^']+)'",
        log_text,
        flags=re.IGNORECASE,
    )

    ii_values: list[int] = []
    depth_values: list[int] = []
    for _target, final_ii, depth, _loop_name in matches:
        ii_values.append(int(final_ii.replace(",", "")))
        depth_values.append(int(depth.replace(",", "")))

    estimated_fmax = _search_float(r"\*{4}\s+Estimated\s+Fmax:\s*([0-9]+(?:\.[0-9]+)?)\s*MHz", log_text)
    return {
        "ii": max(ii_values) if ii_values else None,
        "depth": max(depth_values) if depth_values else None,
        "fmax_est_mhz": estimated_fmax,
    }


def _load_log_metrics_for_report(report_path: Path) -> dict[str, Any]:
    try:
        case_dir = report_path.resolve().parents[4]
    except IndexError:
        return {}

    log_path = case_dir / "logs" / "vitis_hls.log"
    if not log_path.exists():
        return {}

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    return _parse_pipeline_metrics_from_log_text(log_text)


def parse_hls_report_xml_text(report_text: str) -> dict[str, Any]:
    """Parse a Vivado/Vitis HLS XML report."""
    root = ET.fromstring(report_text)
    scalars = _collect_xml_scalars(root)
    latency_summary = _find_xml_path(root, "PerformanceEstimates", "SummaryOfOverallLatency")
    timing_summary = _find_xml_path(root, "PerformanceEstimates", "SummaryOfTimingAnalysis")
    resource_summary = _find_xml_path(root, "AreaEstimates", "Resources")
    pipeline_type = (_xml_text(root, "PerformanceEstimates", "PipelineType") or "").strip().lower()

    cycles = (
        _coalesce(
            _parse_int_like(_xml_text(latency_summary, "Worst-caseLatency") or ""),
            _parse_int_like(_xml_text(latency_summary, "Average-caseLatency") or ""),
            _parse_int_like(_xml_text(latency_summary, "Best-caseLatency") or ""),
            _pick_int(scalars, "Worst-caseLatency", "Average-caseLatency", "Best-caseLatency", "LatencyMax", "Latency"),
        )
        or 0
    )
    latency_ns = (
        _coalesce(
            _parse_time_like_ns(_xml_text(latency_summary, "Worst-caseRealTimeLatency") or ""),
            _parse_time_like_ns(_xml_text(latency_summary, "Average-caseRealTimeLatency") or ""),
            _parse_time_like_ns(_xml_text(latency_summary, "Best-caseRealTimeLatency") or ""),
            _pick_time_ns(scalars, "Worst-caseRealTimeLatency", "Average-caseRealTimeLatency", "Best-caseRealTimeLatency", "LatencyNs"),
        )
    )
    clock_period_ns = (
        _coalesce(
            _parse_float_like(_xml_text(timing_summary, "EstimatedClockPeriod") or ""),
            _parse_float_like(_xml_text(root, "UserAssignments", "TargetClockPeriod") or ""),
            _pick_float(scalars, "EstimatedClockPeriod", "TargetClockPeriod", "ClockPeriod"),
        )
    )
    if latency_ns is None and cycles > 0 and clock_period_ns:
        latency_ns = cycles * clock_period_ns

    ii = _coalesce(
        _parse_int_like(_xml_text(latency_summary, "Interval-max") or ""),
        _parse_int_like(_xml_text(latency_summary, "IntervalMax") or ""),
        _parse_int_like(_xml_text(latency_summary, "Interval-min") or ""),
        _pick_int(scalars, "IntervalMax", "IntervalMin", "Interval"),
        0,
    )
    if pipeline_type == "no" and cycles > 0 and ii >= cycles:
        ii = 0

    depth = _coalesce(
        _parse_int_like(_xml_text(latency_summary, "PipelineDepth") or ""),
        _parse_int_like(_xml_text(latency_summary, "Depth") or ""),
        _pick_int(scalars, "PipelineDepth", "Depth"),
        0,
    )
    bram_18k = _coalesce(
        _parse_int_like(_xml_text(resource_summary, "BRAM_18K") or ""),
        _parse_int_like(_xml_text(resource_summary, "BRAM18K") or ""),
        _pick_int(scalars, "BRAM_18K", "BRAM18K", "BRAM", "BlockRAMTile"),
        0,
    )
    bram_36k = _coalesce(
        _parse_int_like(_xml_text(resource_summary, "BRAM_36K") or ""),
        _parse_int_like(_xml_text(resource_summary, "BRAM36K") or ""),
        _pick_int(scalars, "BRAM_36K", "BRAM36K"),
        0,
    )
    dsp = _coalesce(
        _parse_int_like(_xml_text(resource_summary, "DSP") or ""),
        _parse_int_like(_xml_text(resource_summary, "DSP48E") or ""),
        _pick_int(scalars, "DSP", "DSP48E", "DSP48E1", "DSP48E2"),
        0,
    )
    lut = _coalesce(
        _parse_int_like(_xml_text(resource_summary, "LUT") or ""),
        _pick_int(scalars, "LUT", "CLBLUT", "SliceLUTs"),
        0,
    )
    ff = _coalesce(
        _parse_int_like(_xml_text(resource_summary, "FF") or ""),
        _pick_int(scalars, "FF", "Register", "Registers"),
        0,
    )
    power_w = _pick_float(scalars, "AveragePower", "TotalOnChipPower", "Power") or 0.0
    fmax_est_mhz = _fmax_from_clock_period(clock_period_ns)
    return {
        "cycles": cycles,
        "latency_ns": latency_ns,
        "ii": ii,
        "depth": depth,
        "bram": bram_18k,
        "bram_18k": bram_18k,
        "bram_36k": bram_36k,
        "dsp": dsp,
        "lut": lut,
        "ff": ff,
        "power_w": power_w,
        "estimated_clock_period_ns": clock_period_ns,
        "fmax_est_mhz": fmax_est_mhz,
        "report_format": "xml",
    }


def parse_hls_report_text(report_text: str) -> dict[str, Any]:
    """Parse a subset of HLS report fields used for LUT construction."""
    if report_text.lstrip().startswith("<"):
        return parse_hls_report_xml_text(report_text)

    cycles = (
        _search_int(r"Latency.*?max\s*=\s*([0-9,]+)", report_text)
        or _search_int(r"Latency\s*\(cycles\).*?([0-9,]+)\s*$", report_text)
        or _search_int(r"Latency\s*\(cycles\)\s*:\s*([0-9,]+)", report_text)
        or _search_int(r"\bLatency\b.*?([0-9,]+)", report_text)
        or 0
    )
    latency_ns = _search_float(r"Latency\s*\(ns\).*?([0-9]+(?:\.[0-9]+)?)", report_text)
    ii = (
        _search_int(r"\bInitiation\s+Interval\b[^0-9]*([0-9,]+)", report_text)
        or _search_int(r"\bInterval\b[^0-9]*([0-9,]+)", report_text)
        or _search_int(r"\bII\b[^0-9]*([0-9,]+)", report_text)
        or 0
    )
    depth = (
        _search_int(r"\bPipeline\s+Depth\b[^0-9]*([0-9,]+)", report_text)
        or _search_int(r"\bDepth\b[^0-9]*([0-9,]+)", report_text)
        or 0
    )
    bram_18k = (
        _search_int(r"\bBRAM_18K\b\s*\|?\s*([0-9,]+)", report_text)
        or _search_int(r"\bBRAM_18K\b[^0-9]*([0-9,]+)", report_text)
        or _search_int(r"\bBRAM\b\s*\|?\s*([0-9,]+)", report_text)
        or _search_int(r"\bBRAM\b[^0-9]*([0-9,]+)", report_text)
        or 0
    )
    bram_36k = (
        _search_int(r"\bBRAM_36K\b\s*\|?\s*([0-9,]+)", report_text)
        or _search_int(r"\bBRAM_36K\b[^0-9]*([0-9,]+)", report_text)
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
    estimated_clock_period_ns = (
        _search_float(r"Estimated\s+Clock\s+Period[^0-9]*([0-9]+(?:\.[0-9]+)?)", report_text)
        or _search_float(r"Clock\s+Period[^0-9]*([0-9]+(?:\.[0-9]+)?)", report_text)
    )
    if latency_ns is None and cycles > 0 and estimated_clock_period_ns:
        latency_ns = cycles * estimated_clock_period_ns
    fmax_est_mhz = _fmax_from_clock_period(estimated_clock_period_ns)
    return {
        "cycles": cycles,
        "latency_ns": latency_ns,
        "ii": ii,
        "depth": depth,
        "bram": bram_18k,
        "bram_18k": bram_18k,
        "bram_36k": bram_36k,
        "dsp": dsp,
        "lut": lut,
        "ff": ff,
        "power_w": power_w,
        "estimated_clock_period_ns": estimated_clock_period_ns,
        "fmax_est_mhz": fmax_est_mhz,
        "report_format": "text",
    }


def parse_hls_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    report_text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".xml":
        metrics = parse_hls_report_xml_text(report_text)
    else:
        metrics = parse_hls_report_text(report_text)

    log_metrics = _load_log_metrics_for_report(path)
    if log_metrics:
        if (metrics.get("ii") or 0) <= 0 and log_metrics.get("ii") is not None:
            metrics["ii"] = int(log_metrics["ii"])
        if (metrics.get("depth") or 0) <= 0 and log_metrics.get("depth") is not None:
            metrics["depth"] = int(log_metrics["depth"])
        if metrics.get("fmax_est_mhz") is None and log_metrics.get("fmax_est_mhz") is not None:
            metrics["fmax_est_mhz"] = float(log_metrics["fmax_est_mhz"])
    return metrics


def approximate_fmax_from_wns(*, target_clock_ns: float, wns_ns: Optional[float]) -> Optional[float]:
    if wns_ns is None or target_clock_ns <= 0:
        return None
    effective_period = float(target_clock_ns) - float(wns_ns)
    if effective_period <= 0:
        return None
    return 1_000.0 / effective_period


def parse_vivado_utilization_text(report_text: str) -> dict[str, Any]:
    lut = _search_float(r"\|\s*Slice LUTs\*?\s*\|\s*([0-9.]+)\s*\|", report_text)
    ff = _search_float(r"\|\s*Slice Registers\s*\|\s*([0-9.]+)\s*\|", report_text)
    block_ram_tile = _search_float(r"\|\s*Block RAM Tile\s*\|\s*([0-9.]+)\s*\|", report_text)
    ramb36 = _search_float(r"\|\s*RAMB36/FIFO\*?\s*\|\s*([0-9.]+)\s*\|", report_text)
    ramb18 = _search_float(r"\|\s*RAMB18\s*\|\s*([0-9.]+)\s*\|", report_text)
    dsp = _search_float(r"\|\s*DSPs\s*\|\s*([0-9.]+)\s*\|", report_text)
    return {
        "lut": int(lut) if lut is not None else 0,
        "ff": int(ff) if ff is not None else 0,
        "block_ram_tile": block_ram_tile if block_ram_tile is not None else 0.0,
        "ramb36": int(ramb36) if ramb36 is not None else 0,
        "ramb18": int(ramb18) if ramb18 is not None else 0,
        "dsp": int(dsp) if dsp is not None else 0,
        "report_format": "text",
    }


def parse_vivado_timing_summary_text(report_text: str, *, target_clock_ns: Optional[float] = None) -> dict[str, Any]:
    endpoint_pattern = r"Failing Endpoint(?:s)?"
    setup_wns = _search_float(
        rf"Setup\s*:\s*\d+\s+{endpoint_pattern}\s*,\s*Worst Slack\s+([-0-9.]+)ns",
        report_text,
    )
    setup_tns = _search_float(
        rf"Setup\s*:\s*\d+\s+{endpoint_pattern}\s*,\s*Worst Slack\s+[-0-9.]+ns,\s+Total Violation\s+([-0-9.]+)ns",
        report_text,
    )
    hold_whs = _search_float(
        rf"Hold\s*:\s*\d+\s+{endpoint_pattern}\s*,\s*Worst Slack\s+([-0-9.]+)ns",
        report_text,
    )
    hold_ths = _search_float(
        rf"Hold\s*:\s*\d+\s+{endpoint_pattern}\s*,\s*Worst Slack\s+[-0-9.]+ns,\s+Total Violation\s+([-0-9.]+)ns",
        report_text,
    )
    requirement = _search_float(r"Requirement:\s*([-0-9.]+)ns", report_text)
    data_path_delay = _search_float(r"Data Path Delay:\s*([-0-9.]+)ns", report_text)
    logic_delay = _search_float(r"logic\s+([-0-9.]+)ns", report_text)
    route_delay = _search_float(r"route\s+([-0-9.]+)ns", report_text)
    clock_period_ns = requirement if requirement is not None else target_clock_ns
    approx_fmax = approximate_fmax_from_wns(target_clock_ns=clock_period_ns or 0.0, wns_ns=setup_wns)
    return {
        "setup_wns_ns": setup_wns,
        "setup_tns_ns": setup_tns,
        "hold_whs_ns": hold_whs,
        "hold_ths_ns": hold_ths,
        "target_clock_ns": clock_period_ns,
        "data_path_delay_ns": data_path_delay,
        "logic_delay_ns": logic_delay,
        "route_delay_ns": route_delay,
        "fmax_est_mhz": approx_fmax,
        "report_format": "text",
    }


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
