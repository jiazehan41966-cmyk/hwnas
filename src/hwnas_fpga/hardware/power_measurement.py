"""Instrument-agnostic external board-input power evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_power_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    samples = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        has_power = "power_w" in fields
        has_vi = {"voltage_v", "current_a"} <= fields
        if "timestamp_s" not in fields or not (has_power or has_vi):
            raise ValueError(
                f"{csv_path} requires timestamp_s and power_w or voltage_v,current_a"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp_s"])
                power = (
                    float(row["power_w"])
                    if has_power and str(row.get("power_w", "")).strip()
                    else float(row["voltage_v"]) * float(row["current_a"])
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_number}: {exc}") from exc
            samples.append((timestamp, power))
    if len(samples) < 2:
        raise ValueError(f"{csv_path} needs at least two samples")
    if any(current[0] <= previous[0] for previous, current in zip(samples, samples[1:])):
        raise ValueError(f"{csv_path} timestamps must be strictly increasing")
    energy_j = sum(
        (current_t - previous_t) * (previous_p + current_p) / 2.0
        for (previous_t, previous_p), (current_t, current_p)
        in zip(samples, samples[1:])
    )
    duration_s = samples[-1][0] - samples[0][0]
    return {
        "path": str(csv_path),
        "sha256": sha256_file(csv_path),
        "sample_count": len(samples),
        "duration_s": duration_s,
        "energy_j": energy_j,
        "time_weighted_mean_w": energy_j / duration_s,
    }


def resolve_capture_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def audit_power_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    source_path = Path(manifest_path).resolve()
    idle_values = manifest.get("idle_csvs", [])
    active_values = manifest.get("active_captures", [])
    idle = [
        load_power_csv(resolve_capture_path(source_path, str(value)))
        for value in idle_values
    ]
    active = []
    for entry in active_values:
        capture = load_power_csv(
            resolve_capture_path(source_path, str(entry["csv"]))
        )
        capture["inference_count"] = int(entry["inference_count"])
        active.append(capture)
    gates = {
        "rail_scope": manifest.get("rail_scope") == "board_input_total",
        "measurement_source": manifest.get("measurement_source")
        == "external_power_meter_csv",
        "instrument_model": bool((manifest.get("instrument") or {}).get("model")),
        "instrument_sample_rate": float(
            (manifest.get("instrument") or {}).get("sample_rate_hz", 0)
        )
        > 0,
        "bitstream_sha256": len(str(manifest.get("bitstream_sha256", ""))) == 64,
        "contains_programming_false": manifest.get("contains_programming") is False,
        "contains_uart_upload_false": manifest.get("contains_uart_upload") is False,
        "idle_repetitions": len(idle) >= 3,
        "active_repetitions": len(active) >= 3,
        "idle_duration": bool(idle)
        and all(capture["duration_s"] >= 60.0 for capture in idle),
        "active_duration": bool(active)
        and all(capture["duration_s"] >= 60.0 for capture in active),
        "active_inference_count": bool(active)
        and all(capture["inference_count"] >= 1000 for capture in active),
    }
    idle_power = (
        statistics.fmean(capture["time_weighted_mean_w"] for capture in idle)
        if idle
        else None
    )
    active_power = (
        statistics.fmean(capture["time_weighted_mean_w"] for capture in active)
        if active
        else None
    )
    dynamic_power = (
        active_power - idle_power
        if idle_power is not None and active_power is not None
        else None
    )
    total_active_energy = sum(capture["energy_j"] for capture in active)
    active_duration = sum(capture["duration_s"] for capture in active)
    inference_count = sum(capture["inference_count"] for capture in active)
    dynamic_energy = (
        total_active_energy - idle_power * active_duration
        if idle_power is not None
        else None
    )
    gates["dynamic_power_nonnegative"] = (
        dynamic_power is not None and dynamic_power >= 0
    )
    pass_status = all(gates.values())
    return {
        "schema_version": 1,
        "power_measurement_status": "PASS" if pass_status else "FAIL",
        "overall_pass": pass_status,
        "gates": gates,
        "instrument": manifest.get("instrument"),
        "rail_scope": manifest.get("rail_scope"),
        "bitstream_sha256": manifest.get("bitstream_sha256"),
        "idle_captures": idle,
        "active_captures": active,
        "idle_power_mean_w": idle_power,
        "active_power_mean_w": active_power,
        "dynamic_power_mean_w": dynamic_power,
        "total_energy_mj_per_inference": (
            1000.0 * total_active_energy / inference_count
            if inference_count
            else None
        ),
        "dynamic_energy_mj_per_inference": (
            1000.0 * dynamic_energy / inference_count
            if dynamic_energy is not None and inference_count
            else None
        ),
        "active_inference_count": inference_count,
        "claim_boundary": (
            "Measured board-input total power for this bitstream/protocol only; "
            "power is not a search objective until at least three candidates "
            "pass under the same instrument contract."
        ),
    }


def load_and_audit_power_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("power manifest must be a JSON object")
    return audit_power_manifest(payload, manifest_path=manifest_path)

