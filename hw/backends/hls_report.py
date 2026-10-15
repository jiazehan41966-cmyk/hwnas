"""Pure Vitis/Vivado HLS XML parsing; callers must supply the actual scope."""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class HLSReport:
    evidence_level: str
    metrics: dict
    device: str | None
    clock_mhz: float | None
    tool_version: str | None
    timing_status: str


def _one(root, query):
    values = root.findall(query)
    if len(values) > 1:
        raise ValueError(f"Ambiguous HLS XML field: {query}")
    return values[0].text.strip() if values and values[0].text else None


def _number(value, name, *, integer=False, positive=False):
    if value is None or value.lower() in {"undef", "undefined", "?", "n/a", "-"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HLS {name}") from exc
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ValueError(f"Invalid HLS {name}")
    if integer and not number.is_integer():
        raise ValueError(f"HLS {name} must be an integer")
    return int(number) if integer else number


def parse_hls_xml(xml, *, scope):
    """Parse bytes/text without I/O. A scope declaration never proves board execution.

    Latency is worst-case cycles at the *target* clock, in microseconds. Used
    resources come only from AreaEstimates/Resources, never AvailableResources.
    Missing/undefined fields remain absent. Estimated clock above target fails
    timing, and missing timing identity stays UNKNOWN.
    """
    if scope not in {"operator_hls", "full_hls"}:
        raise ValueError("HLS XML can only describe operator_hls or full_hls")
    if not isinstance(xml, (str, bytes)):
        raise TypeError("HLS XML must be text or bytes")
    raw = xml.encode("utf-8") if isinstance(xml, str) else xml
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("DTD/entity declarations are not permitted in HLS evidence")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("Malformed HLS XML") from exc
    # Namespace removal allows ordinary vendor XML without accepting fuzzy tags.
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    for query, expected in ((".//UserAssignments/unit", "ns"),
                            (".//SummaryOfTimingAnalysis/unit", "ns"),
                            (".//SummaryOfOverallLatency/unit", "clock cycles")):
        unit = _one(root, query)
        if unit is not None and unit != expected:
            raise ValueError(f"Unsupported HLS unit for {query}: {unit}")
    device = _one(root, ".//UserAssignments/Part")
    tool_version = _one(root, ".//ReportVersion/Version")
    target = _number(_one(root, ".//UserAssignments/TargetClockPeriod"), "target period", positive=True)
    estimated = _number(_one(root, ".//SummaryOfTimingAnalysis/EstimatedClockPeriod"), "estimated period", positive=True)
    cycles = _number(_one(root, ".//SummaryOfOverallLatency/Worst-caseLatency"), "cycles", integer=True, positive=True)
    realtime = _one(root, ".//SummaryOfOverallLatency/Worst-caseRealTimeLatency")
    realtime_us = None
    if realtime is not None and realtime.lower() not in {"undef", "undefined", "?", "n/a", "-"}:
        match = re.fullmatch(r"([0-9.eE+\-]+)\s+(ns|us|ms|s)", realtime)
        if match is None:
            raise ValueError("HLS real-time latency needs an explicit ns/us/ms/s unit")
        realtime_us = _number(match[1], "real-time latency", positive=True) * {"ns": .001, "us": 1, "ms": 1000, "s": 1_000_000}[match[2]]
    metrics = {}
    if cycles is not None and target is not None:
        metrics["latency_us"] = cycles * target / 1000
    elif realtime_us is not None:
        metrics["latency_us"] = realtime_us
    if "latency_us" in metrics and not math.isfinite(metrics["latency_us"]):
        raise ValueError("Derived HLS latency is not finite")
    for metric, tag in (("lut", "LUT"), ("dsp", "DSP"), ("bram18", "BRAM_18K")):
        value = _number(_one(root, f".//AreaEstimates/Resources/{tag}"), tag, integer=True)
        if value is not None:
            metrics[metric] = value
    timing = "UNKNOWN" if target is None or estimated is None else ("PASS" if estimated <= target else "FAIL")
    return HLSReport(scope, metrics, device, 1000 / target if target else None, tool_version, timing)


def parse_hls_report(path, *, scope):
    """Read a local XML report; never run the tool that produced it."""
    return parse_hls_xml(Path(path).read_bytes(), scope=scope)
