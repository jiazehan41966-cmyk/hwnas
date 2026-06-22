"""Final hardware validation report helpers.

These helpers keep NAS estimator metrics separate from real board measurements.
Real e2e latency is claimable only when the full-network implementation is
route-clean and the COM5 measurement frame reports status_code=0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .route_gate import classify_route_gate_result


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _frame_value(measurement_payload: Mapping[str, Any], key: str) -> Any:
    frame = measurement_payload.get("frame")
    if isinstance(frame, Mapping) and key in frame:
        return frame.get(key)
    return measurement_payload.get(key)


def normalize_candidate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else payload
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _paths_match(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return str(left) == str(right)


def build_final_hardware_validation_report(
    *,
    candidate_payload: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    measurement_payload: Mapping[str, Any],
    measurement_json_path: str = "",
    expected_serial_port: str = "COM5",
) -> dict[str, Any]:
    """Build a strict final-report payload for route-clean board validation."""
    candidate = normalize_candidate_payload(candidate_payload)
    metrics = _metrics(candidate)
    route_gate = classify_route_gate_result(route_payload)
    measurement_present = bool(measurement_payload)
    status_code = _parse_int(_frame_value(measurement_payload, "status_code")) if measurement_present else None
    e2e_cycles = _parse_int(_frame_value(measurement_payload, "cycles")) if measurement_present else None
    e2e_latency_ms = _parse_float(measurement_payload.get("latency_ms")) if measurement_present else None
    serial_port = str(measurement_payload.get("serial_port") or "") if measurement_present else ""
    route_bitstream = route_payload.get("bitstream")
    measurement_bitstream = measurement_payload.get("bitstream") if measurement_present else ""
    bitstream_matches = _paths_match(route_bitstream, measurement_bitstream)

    blockers: list[str] = []
    if route_gate.get("gate_status") != "PASS":
        blockers.append("full_route_not_clean")
    if not measurement_present:
        blockers.append("measurement_json_missing")
    if measurement_present and status_code != 0:
        blockers.append(f"measurement_status_code_{status_code}")
    if measurement_present and expected_serial_port and serial_port.upper() != expected_serial_port.upper():
        blockers.append(f"measurement_serial_port_not_{expected_serial_port}")
    if measurement_present and not bitstream_matches:
        blockers.append("measurement_bitstream_does_not_match_route_bitstream")
    if measurement_present and e2e_latency_ms is None:
        blockers.append("measurement_latency_ms_missing")
    if measurement_present and e2e_cycles is None:
        blockers.append("measurement_cycles_missing")

    claimable = not blockers
    return {
        "status": "valid_board_latency" if claimable else "blocked",
        "claimable_real_e2e_latency": claimable,
        "blockers": blockers,
        "candidate": {
            "arch_id": candidate.get("arch_id"),
            "encoding_signature": candidate.get("encoding_signature"),
        },
        "nas_lut_estimate": {
            "latency_ms": metrics.get("latency_ms"),
            "lut": metrics.get("lut"),
            "dsp": metrics.get("dsp"),
            "bram": metrics.get("bram"),
            "power_w": metrics.get("power_w"),
            "latency_scope": "NAS strict LUT aggregate estimate only; not board e2e latency.",
        },
        "route_gate": route_gate,
        "bitstream": {
            "route_bitstream": route_bitstream or "",
            "measurement_bitstream": measurement_bitstream or "",
            "bitstream_sha256": route_payload.get("bitstream_sha256") or "",
            "bitstream_matches_measurement": bitstream_matches,
        },
        "real_board_e2e": {
            "latency_source": (
                "COM5 measurement JSON from full-network bitstream"
                if claimable
                else "not claimable"
            ),
            "measurement_json_path": measurement_json_path,
            "serial_port": serial_port,
            "status_code": status_code,
            "e2e_cycles": e2e_cycles,
            "e2e_latency_ms": e2e_latency_ms if claimable else None,
            "raw_latency_ms": e2e_latency_ms,
            "clock_freq_hz": measurement_payload.get("clock_freq_hz") if measurement_present else None,
        },
        "guardrails": [
            "Do not report nas_lut_estimate.latency_ms as real board e2e latency.",
            "Real e2e latency requires route_gate.gate_status == PASS.",
            "Real e2e latency requires COM5 measurement status_code == 0.",
            "Measurement bitstream must match the route-clean bitstream.",
        ],
    }
