#!/usr/bin/env python3
"""Import board-level power CSVs and compute retrained reinjection energy.

This script intentionally consumes only external power-meter CSV files plus the
existing retrained-board reinjection reports. It does not run Vivado, program the
board, or trigger COM5 measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_RAIL_SCOPE = "board_input_total"
EXPECTED_MEASUREMENT_SOURCE = "csv_import"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(manifest_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gate(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"gate": name, "pass": bool(passed), "observed": observed, "expected": expected}


def normalize_measurements(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    measurements = manifest.get("measurements")
    if isinstance(measurements, list):
        defaults = {
            key: manifest.get(key)
            for key in [
                "measurement_source",
                "rail_scope",
                "instrument",
                "contains_programming",
                "same_com5_input",
            ]
            if key in manifest
        }
        rows: list[dict[str, Any]] = []
        for entry in measurements:
            if isinstance(entry, dict):
                merged = dict(defaults)
                merged.update(entry)
                rows.append(merged)
        return rows
    return [manifest]


def load_power_csv(path: Path) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    errors: list[str] = []
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha256": "",
            "samples": samples,
            "sample_count": 0,
            "duration_s": 0.0,
            "energy_j": 0.0,
            "time_weighted_mean_w": None,
            "sample_mean_w": None,
            "timestamp_monotonic": False,
            "errors": [f"missing CSV: {path}"],
        }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_power = "power_w" in fieldnames
        has_voltage_current = "voltage_v" in fieldnames and "current_a" in fieldnames
        if "timestamp_s" not in fieldnames:
            errors.append("CSV missing timestamp_s column")
        if not has_power and not has_voltage_current:
            errors.append("CSV must contain power_w or voltage_v,current_a columns")
        if errors:
            rows = list(reader)
        else:
            for row_idx, row in enumerate(reader, start=2):
                try:
                    ts = float(str(row.get("timestamp_s", "")).strip())
                    if has_power and str(row.get("power_w", "")).strip() != "":
                        power = float(str(row.get("power_w", "")).strip())
                    else:
                        voltage = float(str(row.get("voltage_v", "")).strip())
                        current = float(str(row.get("current_a", "")).strip())
                        power = voltage * current
                    samples.append({"timestamp_s": ts, "power_w": power})
                except ValueError as exc:
                    errors.append(f"row {row_idx}: {exc}")

    timestamps = [sample["timestamp_s"] for sample in samples]
    powers = [sample["power_w"] for sample in samples]
    timestamp_monotonic = all(timestamps[idx] > timestamps[idx - 1] for idx in range(1, len(timestamps)))
    if len(samples) < 2:
        errors.append("CSV needs at least two samples")
    if samples and not timestamp_monotonic:
        errors.append("timestamp_s must be strictly increasing")

    energy_j = 0.0
    if len(samples) >= 2 and timestamp_monotonic:
        for prev, cur in zip(samples, samples[1:]):
            dt = cur["timestamp_s"] - prev["timestamp_s"]
            energy_j += dt * (prev["power_w"] + cur["power_w"]) / 2.0
    duration_s = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 and timestamp_monotonic else 0.0
    time_weighted_mean = energy_j / duration_s if duration_s > 0 else None
    sample_mean = statistics.fmean(powers) if powers else None
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "sample_count": len(samples),
        "duration_s": duration_s,
        "energy_j": energy_j,
        "time_weighted_mean_w": time_weighted_mean,
        "sample_mean_w": sample_mean,
        "timestamp_monotonic": timestamp_monotonic,
        "errors": errors,
    }


def row_by_arch(rows: list[dict[str, Any]], arch_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arch_id") == arch_id:
            return row
    return {}


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def compute_measurement(
    *,
    entry: dict[str, Any],
    manifest_path: Path,
    result_root: Path,
    comparison_row: dict[str, Any],
) -> dict[str, Any]:
    idle_path = resolve_data_path(manifest_path, entry.get("idle_csv", ""))
    active_paths = [resolve_data_path(manifest_path, value) for value in entry.get("active_csvs", [])]
    idle = load_power_csv(idle_path)
    active_series = [load_power_csv(path) for path in active_paths]

    active_inference_count = as_int(entry.get("active_inference_count"))
    idle_mean = idle["time_weighted_mean_w"]
    active_energy_j = sum(float(series["energy_j"]) for series in active_series)
    active_duration_s = sum(float(series["duration_s"]) for series in active_series)
    active_sample_count = sum(int(series["sample_count"]) for series in active_series)
    active_power_mean = active_energy_j / active_duration_s if active_duration_s > 0 else None
    active_capture_means = [
        float(series["time_weighted_mean_w"])
        for series in active_series
        if series.get("time_weighted_mean_w") is not None
    ]
    active_power_std = statistics.pstdev(active_capture_means) if len(active_capture_means) > 1 else 0.0
    dynamic_power_mean = (
        active_power_mean - idle_mean
        if active_power_mean is not None and idle_mean is not None
        else None
    )
    dynamic_energy_j = (
        active_energy_j - float(idle_mean) * active_duration_s
        if idle_mean is not None and active_duration_s > 0
        else None
    )
    total_energy_mj = (
        1000.0 * active_energy_j / active_inference_count
        if active_inference_count and active_inference_count > 0
        else None
    )
    dynamic_energy_mj = (
        1000.0 * dynamic_energy_j / active_inference_count
        if dynamic_energy_j is not None and active_inference_count and active_inference_count > 0
        else None
    )
    latency_ms = as_float(comparison_row.get("real_e2e_latency_ms"))
    latency_dynamic_energy_mj = (
        dynamic_power_mean * latency_ms
        if dynamic_power_mean is not None and latency_ms is not None
        else None
    )
    limited_sampling = active_duration_s < 30.0 or (active_inference_count is not None and active_inference_count < 30)

    idle_errors = list(idle.get("errors", []))
    active_errors = [error for series in active_series for error in series.get("errors", [])]
    active_hashes = [{"path": series["path"], "sha256": series["sha256"]} for series in active_series]
    gates = [
        gate("csv_files_exist", bool(idle.get("exists")) and bool(active_series) and all(series.get("exists") for series in active_series), [idle.get("path"), *[series.get("path") for series in active_series]], "idle and at least one active CSV exist"),
        gate("csv_sha256_recorded", bool(idle.get("sha256")) and all(series.get("sha256") for series in active_series), {"idle": idle.get("sha256"), "active": active_hashes}, "non-empty SHA256 for every CSV"),
        gate("rail_scope", entry.get("rail_scope") == EXPECTED_RAIL_SCOPE, entry.get("rail_scope"), EXPECTED_RAIL_SCOPE),
        gate("measurement_source", entry.get("measurement_source") == EXPECTED_MEASUREMENT_SOURCE, entry.get("measurement_source"), EXPECTED_MEASUREMENT_SOURCE),
        gate("bitstream_sha256_matches_report", entry.get("bitstream_sha256") == comparison_row.get("bitstream_sha256"), entry.get("bitstream_sha256"), comparison_row.get("bitstream_sha256")),
        gate("idle_timestamp_monotonic", idle.get("timestamp_monotonic") is True, idle.get("timestamp_monotonic"), "true"),
        gate("active_timestamp_monotonic", all(series.get("timestamp_monotonic") is True for series in active_series), [series.get("timestamp_monotonic") for series in active_series], "true for every active CSV"),
        gate("idle_sample_count", int(idle.get("sample_count", 0)) > 1, idle.get("sample_count"), "> 1"),
        gate("active_sample_count", active_sample_count > 1, active_sample_count, "> 1"),
        gate("contains_programming_false", entry.get("contains_programming") is False, entry.get("contains_programming"), "false"),
        gate("same_com5_input_true", entry.get("same_com5_input") is True, entry.get("same_com5_input"), "true"),
        gate("active_inference_count", active_inference_count is not None and active_inference_count > 0, entry.get("active_inference_count"), "> 0"),
        gate("dynamic_power_nonnegative", dynamic_power_mean is not None and dynamic_power_mean >= 0.0, dynamic_power_mean, ">= 0 W"),
        gate("comparison_latency_present", latency_ms is not None and latency_ms > 0.0, comparison_row.get("real_e2e_latency_ms"), "> 0 ms"),
    ]
    pass_status = all(item["pass"] for item in gates)
    power_status = "PASS" if pass_status else "FAIL"
    return {
        "arch_id": entry.get("arch_id", ""),
        "run_name": entry.get("run_name", ""),
        "positioning": comparison_row.get("positioning", ""),
        "rail_scope": entry.get("rail_scope", ""),
        "measurement_source": entry.get("measurement_source", ""),
        "instrument": entry.get("instrument", {}),
        "bitstream_sha256": entry.get("bitstream_sha256", ""),
        "retrain_macro_f1": comparison_row.get("retrain_macro_f1", ""),
        "retrain_top1": comparison_row.get("retrain_top1", ""),
        "real_e2e_latency_ms": comparison_row.get("real_e2e_latency_ms", ""),
        "idle_csv": relative_or_absolute(idle_path, result_root),
        "idle_csv_sha256": idle.get("sha256", ""),
        "active_csvs": [relative_or_absolute(path, result_root) for path in active_paths],
        "active_csv_sha256": active_hashes,
        "idle_sample_count": idle.get("sample_count", 0),
        "active_sample_count": active_sample_count,
        "idle_duration_s": idle.get("duration_s", 0.0),
        "active_duration_s": active_duration_s,
        "active_inference_count": active_inference_count,
        "idle_power_mean_w": idle_mean,
        "active_power_mean_w": active_power_mean,
        "active_power_mean_std_w": active_power_std,
        "dynamic_power_mean_w": dynamic_power_mean,
        "total_energy_mj": total_energy_mj,
        "dynamic_energy_mj": dynamic_energy_mj,
        "latency_dynamic_energy_mj": latency_dynamic_energy_mj,
        "limited_sampling": limited_sampling,
        "idle_errors": idle_errors,
        "active_errors": active_errors,
        "power_measurement_status": power_status,
        "gates": gates,
    }


def summary_fields() -> list[str]:
    return [
        "arch_id",
        "positioning",
        "run_name",
        "rail_scope",
        "bitstream_sha256",
        "retrain_macro_f1",
        "retrain_top1",
        "real_e2e_latency_ms",
        "idle_power_mean_w",
        "active_power_mean_w",
        "active_power_mean_std_w",
        "dynamic_power_mean_w",
        "total_energy_mj",
        "dynamic_energy_mj",
        "latency_dynamic_energy_mj",
        "active_inference_count",
        "active_duration_s",
        "limited_sampling",
        "power_measurement_status",
    ]


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = summary_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    lines = [
        "# Phase0 v3 Retrained-Weight Board Power/Energy Summary",
        "",
        f"- generated_at: `{timestamp()}`",
        "- boundary: CSV-imported board-input total power only; Vivado power estimates are not treated as measured power.",
        f"- json: `{json_path}`",
        f"- csv: `{csv_path}`",
        "",
        "| Arch | Positioning | idle W | active W | dynamic W | dynamic energy mJ | total energy mJ | latency approx mJ | active count | limited | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{arch}` | {positioning} | {idle} | {active} | {dynamic} | {dyn_e} | {total_e} | {lat_e} | {count} | {limited} | {status} |".format(
                arch=row.get("arch_id", ""),
                positioning=row.get("positioning", ""),
                idle=fmt(row.get("idle_power_mean_w")),
                active=fmt(row.get("active_power_mean_w")),
                dynamic=fmt(row.get("dynamic_power_mean_w")),
                dyn_e=fmt(row.get("dynamic_energy_mj")),
                total_e=fmt(row.get("total_energy_mj")),
                lat_e=fmt(row.get("latency_dynamic_energy_mj")),
                count=row.get("active_inference_count", ""),
                limited=row.get("limited_sampling", ""),
                status=row.get("power_measurement_status", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v3 Retrained-Weight Board Power/Energy Acceptance Audit",
        "",
        f"- generated_at: `{audit['generated_at']}`",
        f"- overall_pass: `{audit['overall_pass']}`",
        "- boundary: audit validates CSV-imported board-input total power evidence only.",
        "",
        "| Arch | Overall | Failed gates | dynamic W | dynamic energy mJ | limited |",
        "|---|---|---|---:|---:|---|",
    ]
    for run in audit["runs"]:
        failed = [item["gate"] for item in run["gates"] if not item["pass"]]
        summary = run["summary"]
        lines.append(
            "| `{arch}` | {overall} | {failed} | {dynamic_w} | {dynamic_e} | {limited} |".format(
                arch=run["arch_id"],
                overall="PASS" if run["pass"] else "FAIL",
                failed=", ".join(failed) if failed else "-",
                dynamic_w=fmt(summary.get("dynamic_power_mean_w")),
                dynamic_e=fmt(summary.get("dynamic_energy_mj")),
                limited=summary.get("limited_sampling", ""),
            )
        )
    lines.extend(["", "## Gate Details", ""])
    for run in audit["runs"]:
        lines.extend([f"### {run['arch_id']}", ""])
        for item in run["gates"]:
            status = "PASS" if item["pass"] else "FAIL"
            lines.append(f"- `{status}` `{item['gate']}`: observed `{item['observed']}`, expected `{item['expected']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def import_power_energy(result_root: Path, manifest_path: Path) -> dict[str, Any]:
    result_root = result_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Missing or empty power manifest: {manifest_path}")
    comparison_path = result_root / "reports" / "retrained_board_reinject_comparison.json"
    comparison = read_json(comparison_path)
    comparison_rows = comparison.get("rows", []) if isinstance(comparison.get("rows"), list) else []
    measurements = normalize_measurements(manifest)
    rows: list[dict[str, Any]] = []
    for entry in measurements:
        arch_id = str(entry.get("arch_id", ""))
        comparison_row = row_by_arch(comparison_rows, arch_id)
        rows.append(
            compute_measurement(
                entry=entry,
                manifest_path=manifest_path,
                result_root=result_root,
                comparison_row=comparison_row,
            )
        )

    root_gates = [
        gate("comparison_report_exists", comparison_path.exists(), str(comparison_path), "exists"),
        gate("power_manifest_exists", manifest_path.exists(), str(manifest_path), "exists"),
        gate("measurements_present", len(rows) > 0, len(rows), "> 0"),
    ]
    audit = {
        "generated_at": timestamp(),
        "result_root": str(result_root),
        "manifest": str(manifest_path),
        "overall_pass": all(item["pass"] for item in root_gates) and all(row["power_measurement_status"] == "PASS" for row in rows),
        "root_gates": root_gates,
        "runs": [
            {
                "arch_id": row["arch_id"],
                "run_name": row["run_name"],
                "pass": row["power_measurement_status"] == "PASS",
                "gates": row["gates"],
                "summary": {
                    key: row.get(key)
                    for key in [
                        "idle_power_mean_w",
                        "active_power_mean_w",
                        "dynamic_power_mean_w",
                        "total_energy_mj",
                        "dynamic_energy_mj",
                        "latency_dynamic_energy_mj",
                        "limited_sampling",
                    ]
                },
            }
            for row in rows
        ],
    }
    output_root = result_root / "power_measurements"
    reports = output_root / "reports"
    summary_json = reports / "retrained_board_power_energy_summary.json"
    summary_csv = reports / "retrained_board_power_energy_summary.csv"
    summary_md = reports / "retrained_board_power_energy_summary.md"
    audit_json = reports / "retrained_board_power_energy_acceptance_audit.json"
    audit_md = reports / "retrained_board_power_energy_acceptance_audit.md"
    write_json(
        summary_json,
        {
            "generated_at": timestamp(),
            "result_root": str(result_root),
            "manifest": str(manifest_path),
            "measurement_source": manifest.get("measurement_source", ""),
            "rail_scope": manifest.get("rail_scope", ""),
            "rows": rows,
        },
    )
    write_csv_report(summary_csv, rows)
    write_summary_markdown(summary_md, rows, summary_json, summary_csv)
    write_json(audit_json, audit)
    write_audit_markdown(audit_md, audit)
    return {
        "overall_pass": audit["overall_pass"],
        "rows": len(rows),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "summary_markdown": str(summary_md),
        "audit_json": str(audit_json),
        "audit_markdown": str(audit_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import board power CSVs and compute energy.")
    parser.add_argument(
        "--result-root",
        default="hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608",
    )
    parser.add_argument(
        "--manifest",
        default="hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608/power_measurements/power_measurement_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = import_power_energy(Path(args.result_root), Path(args.manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
