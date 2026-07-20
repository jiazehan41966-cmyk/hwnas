"""Instrument-agnostic external board-input power evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
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
        receipt_value = entry.get("run_repeat_receipt")
        receipt = None
        if receipt_value:
            receipt_path = resolve_capture_path(source_path, str(receipt_value))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            if not isinstance(receipt, dict):
                raise ValueError(f"{receipt_path} must contain a JSON object")
            capture["run_repeat_receipt"] = {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "payload": receipt,
            }
        active.append(capture)
    bitstream_sha256 = str(manifest.get("bitstream_sha256", ""))
    gates = {
        "rail_scope": manifest.get("rail_scope") == "board_input_total",
        "measurement_source": manifest.get("measurement_source")
        == "external_power_meter_csv",
        "instrument_model": bool((manifest.get("instrument") or {}).get("model")),
        "instrument_sample_rate": float(
            (manifest.get("instrument") or {}).get("sample_rate_hz", 0)
        )
        > 0,
        "instrument_calibration": bool(
            str((manifest.get("instrument") or {}).get("calibration", "")).strip()
        )
        and str((manifest.get("instrument") or {}).get("calibration", "")).upper()
        != "TODO",
        "bitstream_sha256": bool(re.fullmatch(r"[0-9a-fA-F]{64}", bitstream_sha256)),
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
        "active_run_repeat_receipts": bool(active)
        and all(capture.get("run_repeat_receipt") for capture in active),
    }
    gates["receipt_count_match"] = bool(active) and all(
        capture.get("run_repeat_receipt")
        and int(capture["run_repeat_receipt"]["payload"].get("repeat_count", -1))
        == capture["inference_count"]
        for capture in active
    )
    gates["receipt_bitstream_match"] = bool(active) and all(
        capture.get("run_repeat_receipt")
        and str(
            capture["run_repeat_receipt"]["payload"].get("bitstream_sha256", "")
        ).lower()
        == bitstream_sha256.lower()
        for capture in active
    )
    gates["receipt_active_duration"] = bool(active) and all(
        capture.get("run_repeat_receipt")
        and float(
            capture["run_repeat_receipt"]["payload"].get(
                "host_active_elapsed_s", 0
            )
        )
        >= 60.0
        for capture in active
    )
    gates["receipt_excludes_upload_and_programming"] = bool(active) and all(
        capture.get("run_repeat_receipt")
        and capture["run_repeat_receipt"]["payload"].get("contains_programming")
        is False
        and capture["run_repeat_receipt"]["payload"].get("contains_uart_upload")
        is False
        for capture in active
    )
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
        "candidate_id": manifest.get("candidate_id"),
        "measurement_protocol_fingerprint": manifest.get(
            "measurement_protocol_fingerprint"
        ),
        "raw_csv_hashes_complete": bool(
            idle and active and all(capture.get("sha256") for capture in idle + active)
        ),
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


def audit_power_campaign(manifest_paths: list[str | Path]) -> dict[str, Any]:
    """Audit the multi-candidate power evidence required for Pareto claims."""
    if not manifest_paths:
        return {
            "schema_version": 2,
            "overall_pass": False,
            "candidate_count": 0,
            "same_instrument_protocol": False,
            "raw_csv_hashes_complete": False,
            "boundary": "At least three independent candidate manifests are required.",
        }
    audits = []
    manifests = []
    for path in manifest_paths:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"{source} must contain a JSON object")
        manifests.append(payload)
        audits.append(audit_power_manifest(payload, manifest_path=source))
    candidate_ids = [
        str(payload.get("candidate_id") or Path(path).stem)
        for payload, path in zip(manifests, manifest_paths)
    ]
    instrument_keys = {
        json.dumps(payload.get("instrument") or {}, sort_keys=True)
        for payload in manifests
    }
    protocol_keys = {
        str(payload.get("measurement_protocol_fingerprint", ""))
        for payload in manifests
    }
    gates = {
        "three_candidates": len(set(candidate_ids)) == 3,
        "same_instrument_protocol": len(instrument_keys) == 1
        and len(protocol_keys) == 1
        and "" not in protocol_keys,
        "all_candidate_manifests_pass": all(audit["overall_pass"] for audit in audits),
        "raw_csv_hashes_complete": all(
            audit.get("raw_csv_hashes_complete") is True for audit in audits
        ),
    }
    return {
        "schema_version": 2,
        "overall_pass": all(gates.values()),
        "candidate_count": len(set(candidate_ids)),
        "candidate_ids": candidate_ids,
        "same_instrument_protocol": gates["same_instrument_protocol"],
        "raw_csv_hashes_complete": gates["raw_csv_hashes_complete"],
        "gates": gates,
        "manifests": [str(Path(path).resolve()) for path in manifest_paths],
        "audits": audits,
        "pareto_eligible": all(gates.values()),
        "boundary": (
            "Power/energy may enter a Pareto claim only after three candidates "
            "pass under one instrument and one measurement protocol."
        ),
    }


def load_and_audit_power_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("power manifest must be a JSON object")
    return audit_power_manifest(payload, manifest_path=manifest_path)
