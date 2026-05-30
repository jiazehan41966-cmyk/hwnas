#!/usr/bin/env python3
"""current84 primitive60 AV7K325 board validation ledger and isolated retests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import serial.tools.list_ports
except ImportError:  # pragma: no cover - board host dependency
    serial = None  # type: ignore[assignment]
else:
    serial = serial  # type: ignore[name-defined]

from batch_measure_ops import (
    BOARD_HARNESS_DIR,
    DEFAULT_BOARD_CONFIG,
    DEFAULT_VITIS_HLS,
    DEFAULT_VIVADO,
    GENERATE_HARNESS_SCRIPT,
    MEASURE_SCRIPT,
    RESULTS_DIR,
    ROOT,
    build_bitstream,
    ensure_csynth,
    run_command,
)


CURRENT84_DEPLOYABLE_DIR = RESULTS_DIR / "v2_current84_deployable_lut"
TRACE_CSV = CURRENT84_DEPLOYABLE_DIR / "current84_deployable_source_trace.csv"
DEPLOYABLE_LUT_JSON = CURRENT84_DEPLOYABLE_DIR / "deployable_lut_v2_current84.json"

BOARD_RESULTS_DIR = BOARD_HARNESS_DIR / "results"
FULL44_RUNLIST_CSV = BOARD_RESULTS_DIR / "v2_board_validation_full44_runlist.csv"
FULL44_STATUS_CSV = BOARD_RESULTS_DIR / "v2_board_validation_full44_status.csv"
FULL44_MEASUREMENTS_CSV = BOARD_RESULTS_DIR / "v2_board_validation_full44_measurements.csv"
FULL44_TIMING_FAILURES_CSV = BOARD_RESULTS_DIR / "v2_board_validation_full44_timing_failures.csv"

MISSING15_DIR = BOARD_RESULTS_DIR / "v2_current84_missing15_board"
MISSING15_STATUS_CSV = MISSING15_DIR / "v2_current84_missing15_board_status.csv"
MISSING15_TIMING_FAILURES_CSV = MISSING15_DIR / "v2_current84_missing15_board_timing_failures.csv"

CURRENT84_DIR = RESULTS_DIR / "v2_current84"
IDENTITY_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_v2_current84_identity_repack_14_64.yaml"
IDENTITY_STATUS_JSON = CURRENT84_DIR / "formal_lut_status_v2_current84_identity_repack_14_64.json"
IDENTITY_HLS_CSV = CURRENT84_DIR / "current84_identity_repack_14_64_hls.csv"
IDENTITY_DOWNSTREAM_CSV = CURRENT84_DIR / "current84_identity_repack_14_64_vivado_downstream.csv"
IDENTITY_CASE_ROOT = CURRENT84_DIR / "identity_repack_14_64_p"
IDENTITY_CASE_NAME = "identity_repack_current84_identity_repack_14_64_p16_cb16_ii1_dsp1_main_5ns"

OUT_DIR = BOARD_RESULTS_DIR / "v2_current84_primitive60_board"
RUNLIST_CSV = OUT_DIR / "v2_current84_primitive60_board_runlist.csv"
STATUS_CSV = OUT_DIR / "v2_current84_primitive60_board_status.csv"
TIMING_FAILURES_CSV = OUT_DIR / "v2_current84_primitive60_board_timing_failures.csv"
PENDING_CSV = OUT_DIR / "v2_current84_primitive60_board_pending.csv"
SUMMARY_JSON = OUT_DIR / "v2_current84_primitive60_board_summary.json"
README_MD = OUT_DIR / "README.md"

OVERLAY_CSV = OUT_DIR / "board_latency_overlay_current84_primitive60.csv"
OVERLAY_JSON = OUT_DIR / "board_latency_overlay_current84_primitive60.json"
OVERLAY_STRICT_CLEAN_CSV = OUT_DIR / "board_latency_overlay_current84_primitive60_strict_clean.csv"
OVERLAY_BLOCKERS_CSV = OUT_DIR / "board_latency_overlay_current84_primitive60_blockers.csv"

IDENTITY_OUT_DIR = OUT_DIR / "identity_repack_14_64"
IDENTITY_RUNLIST_CSV = IDENTITY_OUT_DIR / "identity_repack_14_64_board_runlist.csv"
IDENTITY_STATUS_CSV = IDENTITY_OUT_DIR / "identity_repack_14_64_board_status.csv"
IDENTITY_MEASUREMENTS_CSV = IDENTITY_OUT_DIR / "identity_repack_14_64_board_measurements.csv"
IDENTITY_MEASUREMENTS_DIR = IDENTITY_OUT_DIR / "measurements"
IDENTITY_HARNESS_PROJECTS_DIR = IDENTITY_OUT_DIR / "harness_projects"
IDENTITY_SUMMARY_JSON = IDENTITY_OUT_DIR / "identity_repack_14_64_board_summary.json"
IDENTITY_STDOUT_LOG = IDENTITY_OUT_DIR / "identity_repack_14_64_run.out.log"

RECOVERY_DIR = OUT_DIR / "timing_recovery"
RECOVERY_SUMMARY_JSON = RECOVERY_DIR / "timing_recovery_summary.json"
RECOVERY_STATUS_CSV = RECOVERY_DIR / "timing_recovery_status.csv"

MEASURED_STATUS_VALUES = {
    "board_measured_timing_clean",
    "board_measured_uart_ok_timing_fail",
}

RUNLIST_FIELDS = [
    "priority",
    "component_case_name",
    "component_operator",
    "component_op_type",
    "shape_name",
    "selected_source",
    "output_source_dataset",
    "source_case_name",
    "selected_existing_case_name",
    "formal_cycles",
    "formal_latency_ms",
    "current_case_count",
    "current_cases",
    "component_roles",
    "notes",
]

STATUS_FIELDS = RUNLIST_FIELDS + [
    "status_timestamp",
    "board_measurement_source",
    "status",
    "board_execution_status",
    "failure_category",
    "failure_reason",
    "uart_status_code",
    "uart_status_label",
    "board_cycles",
    "board_latency_ms",
    "board_word_count",
    "board_checksum",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "measurement_csv_path",
    "bitstream_path",
    "harness_project_dir",
    "timing_report_path",
    "utilization_report_path",
    "board_harness_lut",
    "board_harness_ff",
    "board_harness_bram_tile",
    "board_harness_dsp",
    "source_status_csv",
    "source_summary_json",
    "source_stdout_log",
    "source_stderr_log",
]

TIMING_FAILURE_FIELDS = [
    "priority",
    "component_case_name",
    "source_case_name",
    "output_source_dataset",
    "board_measurement_source",
    "status",
    "board_execution_status",
    "failure_category",
    "failure_reason",
    "uart_status_code",
    "wns_ns",
    "harness_project_dir",
    "measurement_json_path",
]

OVERLAY_FIELDS = [
    "current84_component_case_name",
    "source_dataset",
    "source_case_name",
    "board_measurement_source",
    "uart_status_code",
    "board_cycles",
    "board_latency_ms",
    "formal_cycles",
    "formal_latency_ms",
    "delta_cycles",
    "delta_percent",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "timing_report_path",
    "notes",
]

IDENTITY_RUNLIST_FIELDS = [
    "priority",
    "case_name",
    "op_id",
    "op_type",
    "case_dir",
    "board_config",
    "board_target_part",
    "metric_source_target_part",
    "hls_status",
    "downstream_status",
    "expected_hls_latency_ms",
    "expected_hls_cycles",
    "expected_post_route_wns_ns",
    "expected_post_route_fmax_mhz",
    "expected_lut",
    "expected_ff",
    "expected_bram_tile",
    "expected_dsp",
    "expected_power_w",
    "contract_id",
    "source_status_json",
    "source_hls_csv",
    "source_downstream_csv",
    "source_config",
]

IDENTITY_STATUS_FIELDS = IDENTITY_RUNLIST_FIELDS + [
    "status_timestamp",
    "case_dir_exists",
    "synth_tcl_exists",
    "csynth_report_available",
    "component_xml_exists",
    "export_rtl_exists",
    "harness_project_dir",
    "harness_manifest_exists",
    "bitstream_exists",
    "measurement_json",
    "measurement_csv",
    "csynth_status",
    "export_status",
    "harness_status",
    "bitstream_status",
    "measurement_status",
    "timing_status",
    "support_status",
    "board_execution_status",
    "failure_category",
    "failure_reason",
    "board_status_code",
    "board_status_label",
    "board_cycles",
    "board_latency_ms",
    "board_word_count",
    "board_checksum",
    "board_timing_wns_ns",
    "board_timing_met",
    "board_timing_clean",
    "board_harness_lut",
    "board_harness_ff",
    "board_harness_bram_tile",
    "board_harness_dsp",
    "summary_json",
    "stdout_log",
]

RECOVERY_FIELDS = [
    "started_at",
    "completed_at",
    "component_case_name",
    "source_case_name",
    "source_dataset",
    "attempt_name",
    "strategy",
    "place_directive",
    "phys_opt_enabled",
    "phys_opt_directive",
    "route_directive",
    "seed",
    "build_status",
    "build_returncode",
    "measurement_status",
    "uart_status_code",
    "board_cycles",
    "board_latency_ms",
    "wns_ns",
    "timing_clean",
    "attempt_dir",
    "bitstream_path",
    "measurement_json_path",
    "timing_report_path",
    "utilization_report_path",
    "stdout_log",
    "stderr_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="current84 primitive60 AV7K325 board validation closure")
    parser.add_argument("mode", choices=("init", "refresh", "run-identity", "recover", "all"))
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--csynth-timeout-minutes", type=int, default=60)
    parser.add_argument("--harness-timeout-minutes", type=int, default=10)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=90)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20)
    parser.add_argument("--recovery-limit", type=int, default=None)
    parser.add_argument("--recovery-case", action="append", default=None)
    parser.add_argument("--recovery-attempts", type=int, default=1)
    parser.add_argument("--skip-recovery-measure", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def parse_wns(timing_report: Path) -> str:
    if not timing_report.exists():
        return ""
    lines = timing_report.read_text(encoding="utf-8", errors="replace").splitlines()
    number_pattern = re.compile(r"^-?\d+(?:\.\d+)?$")
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 8]:
            parts = candidate.split()
            if parts and number_pattern.match(parts[0]):
                return parts[0]
    return ""


def parse_utilization(util_report: Path) -> dict[str, str]:
    values = {
        "board_harness_lut": "",
        "board_harness_ff": "",
        "board_harness_bram_tile": "",
        "board_harness_dsp": "",
    }
    if not util_report.exists():
        return values
    text = util_report.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "board_harness_lut": r"\|\s*Slice LUTs\s*\|\s*([0-9,]+)",
        "board_harness_ff": r"\|\s*Slice Registers\s*\|\s*([0-9,]+)",
        "board_harness_bram_tile": r"\|\s*Block RAM Tile\s*\|\s*([0-9.]+)",
        "board_harness_dsp": r"\|\s*DSPs\s*\|\s*([0-9,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = match.group(1).replace(",", "")
    return values


def has_csynth_artifacts(case_dir: Path) -> bool:
    report_dir = case_dir / "project" / "solution1" / "syn" / "report"
    return report_dir.exists() and (
        any(report_dir.glob("*_csynth.xml"))
        or any(report_dir.glob("*_csynth.rpt"))
        or (report_dir / "csynth.xml").exists()
        or (report_dir / "csynth.rpt").exists()
    )


def component_xml(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"


def ip_rtl_dir(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "impl" / "ip" / "hdl" / "verilog"


def syn_rtl_dir(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "syn" / "verilog"


def has_harness_rtl(case_dir: Path) -> bool:
    return ip_rtl_dir(case_dir).exists() or syn_rtl_dir(case_dir).exists()


def load_measurement_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    return payload if isinstance(payload, dict) else {}


def frame_value(measurement: dict[str, Any], key: str) -> Any:
    frame = measurement.get("frame")
    if isinstance(frame, dict) and key in frame:
        return frame.get(key)
    return measurement.get(key, "")


def assert_input_files() -> dict[str, Any]:
    inputs = [
        TRACE_CSV,
        DEPLOYABLE_LUT_JSON,
        FULL44_RUNLIST_CSV,
        FULL44_STATUS_CSV,
        FULL44_MEASUREMENTS_CSV,
        FULL44_TIMING_FAILURES_CSV,
        MISSING15_STATUS_CSV,
        MISSING15_TIMING_FAILURES_CSV,
        IDENTITY_CONFIG,
        IDENTITY_STATUS_JSON,
        IDENTITY_HLS_CSV,
        IDENTITY_DOWNSTREAM_CSV,
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required input files are missing: {missing}")

    trace_rows = read_csv(TRACE_CSV)
    deployable_payload = load_json(DEPLOYABLE_LUT_JSON)
    deployable_entries = deployable_payload.get("entries", [])
    full44_rows = read_csv(FULL44_STATUS_CSV)
    missing15_rows = read_csv(MISSING15_STATUS_CSV)
    identity_entries = load_json(IDENTITY_STATUS_JSON).get("entries", [])
    identity_hls_rows = read_csv(IDENTITY_HLS_CSV)
    identity_downstream_rows = read_csv(IDENTITY_DOWNSTREAM_CSV)

    if len(trace_rows) != 60 or len({row["component_case_name"] for row in trace_rows}) != 60:
        raise AssertionError(f"source trace must contain 60 unique component rows: {TRACE_CSV}")
    if not isinstance(deployable_entries, list) or len(deployable_entries) != 60:
        raise AssertionError(f"deployable LUT must contain 60 entries: {DEPLOYABLE_LUT_JSON}")
    if len(read_csv(FULL44_RUNLIST_CSV)) != 44 or len(full44_rows) != 44:
        raise AssertionError("full44 runlist/status must each contain 44 rows")
    if len(missing15_rows) != 15:
        raise AssertionError("current84 missing15 status must contain 15 rows")
    if len(read_csv(FULL44_TIMING_FAILURES_CSV)) != 13:
        raise AssertionError("full44 timing failures must contain 13 rows")
    if len(read_csv(MISSING15_TIMING_FAILURES_CSV)) != 1:
        raise AssertionError("current84 missing15 timing failures must contain 1 row")
    if len(identity_entries) != 1 or len(identity_hls_rows) != 1 or len(identity_downstream_rows) != 1:
        raise AssertionError("identity_repack_14_64 formal/HLS/downstream inputs must each contain one entry")
    if any("identity_repack_14_64" in row.get("case_name", "") for row in full44_rows):
        raise AssertionError("identity_repack_14_64 must not be present in full44 status")

    return {
        "trace_rows": len(trace_rows),
        "deployable_entries": len(deployable_entries),
        "full44_status_rows": len(full44_rows),
        "missing15_status_rows": len(missing15_rows),
        "identity_formal_entries": len(identity_entries),
        "input_hashes": {str(path): sha256_file(path) for path in inputs},
    }


def build_primitive60_runlist() -> list[dict[str, Any]]:
    rows = []
    for priority, trace in enumerate(read_csv(TRACE_CSV), start=1):
        rows.append(
            {
                "priority": priority,
                "component_case_name": trace.get("component_case_name", ""),
                "component_operator": trace.get("component_operator", ""),
                "component_op_type": trace.get("component_op_type", ""),
                "shape_name": trace.get("shape_name", ""),
                "selected_source": trace.get("selected_source", ""),
                "output_source_dataset": trace.get("output_source_dataset", ""),
                "source_case_name": trace.get("source_case_name", ""),
                "selected_existing_case_name": trace.get("selected_existing_case_name", ""),
                "formal_cycles": trace.get("cycles", ""),
                "formal_latency_ms": trace.get("latency_ms", ""),
                "current_case_count": trace.get("current_case_count", ""),
                "current_cases": trace.get("current_cases", ""),
                "component_roles": trace.get("component_roles", ""),
                "notes": trace.get("notes", ""),
            }
        )
    return rows


def status_path_candidates(harness_project: str, case_name: str) -> dict[str, str]:
    project = Path(harness_project) if harness_project else Path()
    return {
        "bitstream_path": str(project / "bitstream" / f"{case_name}.bit") if harness_project else "",
        "timing_report_path": str(project / "reports" / "timing_summary.rpt") if harness_project else "",
        "utilization_report_path": str(project / "reports" / "utilization.rpt") if harness_project else "",
    }


def normalize_measured_status(evidence: dict[str, Any]) -> tuple[str, str, str, bool]:
    status_code = parse_int(evidence.get("board_status_code"))
    wns = parse_float(evidence.get("board_timing_wns_ns"))
    if status_code == 0 and wns is not None and wns >= 0.0:
        return "board_measured_timing_clean", "success", "", True
    if status_code == 0 and wns is not None and wns < 0.0:
        return (
            "board_measured_uart_ok_timing_fail",
            "success_uart_timing_violation",
            f"AV7K325 harness WNS={evidence.get('board_timing_wns_ns')}",
            False,
        )
    return "missing_board_evidence", "missing_board_evidence", "missing strict UART/timing evidence", False


def primitive_status_from_evidence(
    run_row: dict[str, Any],
    *,
    evidence: dict[str, str] | None,
    source_label: str,
    source_status_csv: Path | None,
) -> dict[str, Any]:
    row = dict(run_row)
    row["status_timestamp"] = timestamp()
    row["board_measurement_source"] = source_label
    row["source_status_csv"] = str(source_status_csv) if source_status_csv else ""

    if evidence is None:
        row.update(
            {
                "status": "pending_board_measurement" if source_label == "pending_identity_repack_14_64_board_measurement" else "missing_board_evidence",
                "board_execution_status": "pending_board_measurement" if source_label == "pending_identity_repack_14_64_board_measurement" else "missing_board_evidence",
                "failure_category": "pending_board_measurement" if source_label == "pending_identity_repack_14_64_board_measurement" else "missing_board_evidence",
                "failure_reason": source_label,
                "timing_clean": "False",
            }
        )
        return row

    status, board_execution, failure_reason, timing_clean = normalize_measured_status(evidence)
    harness_project = evidence.get("harness_project_dir", "")
    source_case_name = run_row["source_case_name"]
    paths = status_path_candidates(harness_project, source_case_name)
    util = parse_utilization(Path(paths["utilization_report_path"])) if paths["utilization_report_path"] else {}
    row.update(
        {
            "status": status,
            "board_execution_status": board_execution,
            "failure_category": "negative_wns" if status == "board_measured_uart_ok_timing_fail" else "",
            "failure_reason": failure_reason,
            "uart_status_code": evidence.get("board_status_code", ""),
            "uart_status_label": evidence.get("board_status_label", ""),
            "board_cycles": evidence.get("board_cycles", ""),
            "board_latency_ms": evidence.get("board_latency_ms", ""),
            "board_word_count": evidence.get("board_word_count", ""),
            "board_checksum": evidence.get("board_checksum", ""),
            "wns_ns": evidence.get("board_timing_wns_ns", ""),
            "timing_clean": str(timing_clean),
            "measurement_json_path": evidence.get("measurement_json", ""),
            "measurement_csv_path": evidence.get("measurement_csv", ""),
            "harness_project_dir": harness_project,
            "source_summary_json": evidence.get("summary_json", ""),
            "source_stdout_log": evidence.get("stdout_log", ""),
            "source_stderr_log": evidence.get("stderr_log", ""),
        }
    )
    row.update(paths)
    row.update(util)
    return row


def identity_status_lookup() -> dict[str, dict[str, str]]:
    rows = read_csv(IDENTITY_STATUS_CSV)
    return {row["case_name"]: row for row in rows if row.get("case_name")}


def recovery_clean_lookup() -> dict[str, dict[str, str]]:
    clean: dict[str, dict[str, str]] = {}
    for row in read_csv(RECOVERY_STATUS_CSV):
        if not parse_bool(row.get("timing_clean")):
            continue
        component = row.get("component_case_name", "")
        if not component:
            continue
        existing = clean.get(component)
        if existing is None or str(row.get("completed_at", "")) > str(existing.get("completed_at", "")):
            clean[component] = row
    return clean


def apply_recovery_evidence(status_row: dict[str, Any], recovery: dict[str, str]) -> dict[str, Any]:
    row = dict(status_row)
    measurement_json = Path(recovery.get("measurement_json_path", ""))
    measurement = load_measurement_payload(measurement_json)
    wns = recovery.get("wns_ns", "")
    bitstream_path = recovery.get("bitstream_path", "")
    timing_report_path = recovery.get("timing_report_path", "")
    utilization_report_path = recovery.get("utilization_report_path", "")
    util = parse_utilization(Path(utilization_report_path)) if utilization_report_path else {}
    row.update(
        {
            "status_timestamp": timestamp(),
            "board_measurement_source": "measured_in_timing_recovery",
            "status": "board_measured_timing_clean",
            "board_execution_status": "success_recovered_timing_clean",
            "failure_category": "",
            "failure_reason": "",
            "uart_status_code": recovery.get("uart_status_code", frame_value(measurement, "status_code")),
            "uart_status_label": measurement.get("status_label", "ok" if recovery.get("uart_status_code") == "0" else ""),
            "board_cycles": recovery.get("board_cycles", frame_value(measurement, "cycles")),
            "board_latency_ms": recovery.get("board_latency_ms", measurement.get("latency_ms", "")),
            "board_word_count": frame_value(measurement, "word_count"),
            "board_checksum": frame_value(measurement, "checksum"),
            "wns_ns": wns,
            "timing_clean": "True",
            "measurement_json_path": recovery.get("measurement_json_path", ""),
            "measurement_csv_path": str(Path(recovery.get("attempt_dir", "")) / "measurement.csv") if recovery.get("attempt_dir") else "",
            "bitstream_path": bitstream_path,
            "harness_project_dir": recovery.get("attempt_dir", ""),
            "timing_report_path": timing_report_path,
            "utilization_report_path": utilization_report_path,
            "source_status_csv": str(RECOVERY_STATUS_CSV),
            "source_summary_json": str(RECOVERY_SUMMARY_JSON) if RECOVERY_SUMMARY_JSON.exists() else "",
            "source_stdout_log": recovery.get("stdout_log", ""),
            "source_stderr_log": recovery.get("stderr_log", ""),
        }
    )
    row.update(util)
    return row


def build_primitive60_status(runlist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full44 = {row["case_name"]: row for row in read_csv(FULL44_STATUS_CSV)}
    missing15 = {row["case_name"]: row for row in read_csv(MISSING15_STATUS_CSV)}
    identity = identity_status_lookup()
    recovery_clean = recovery_clean_lookup()

    status_rows: list[dict[str, Any]] = []
    for row in runlist:
        source_dataset = row["output_source_dataset"]
        source_case = row["source_case_name"]
        if source_dataset == "v2_current84_missing":
            status_rows.append(
                primitive_status_from_evidence(
                    row,
                    evidence=missing15.get(source_case),
                    source_label="reused_from_current84_missing15_board",
                    source_status_csv=MISSING15_STATUS_CSV,
                )
            )
        elif source_dataset == "v2_current84_identity_repack_14_64":
            evidence = identity.get(source_case)
            if evidence and evidence.get("support_status") in MEASURED_STATUS_VALUES:
                status_rows.append(
                    primitive_status_from_evidence(
                        row,
                        evidence=evidence,
                        source_label="measured_in_primitive60_identity_repack_14_64",
                        source_status_csv=IDENTITY_STATUS_CSV,
                    )
                )
            else:
                status_rows.append(
                    primitive_status_from_evidence(
                        row,
                        evidence=None,
                        source_label="pending_identity_repack_14_64_board_measurement",
                        source_status_csv=IDENTITY_STATUS_CSV if IDENTITY_STATUS_CSV.exists() else None,
                    )
                )
        else:
            status_rows.append(
                primitive_status_from_evidence(
                    row,
                    evidence=full44.get(source_case),
                    source_label="reused_from_full44_board",
                    source_status_csv=FULL44_STATUS_CSV,
                )
            )
        if row["component_case_name"] in recovery_clean:
            status_rows[-1] = apply_recovery_evidence(status_rows[-1], recovery_clean[row["component_case_name"]])
    return status_rows


def summary_counts(status_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_entries": len(status_rows),
        "uart_ok": sum(1 for row in status_rows if parse_int(row.get("uart_status_code")) == 0),
        "timing_clean": sum(1 for row in status_rows if parse_bool(row.get("timing_clean"))),
        "timing_violation": sum(1 for row in status_rows if row.get("status") == "board_measured_uart_ok_timing_fail"),
        "pending": sum(1 for row in status_rows if row.get("status") == "pending_board_measurement"),
        "missing_board_evidence": sum(1 for row in status_rows if row.get("status") == "missing_board_evidence"),
        "reused_from_full44_board": sum(1 for row in status_rows if row.get("board_measurement_source") == "reused_from_full44_board"),
        "reused_from_current84_missing15_board": sum(1 for row in status_rows if row.get("board_measurement_source") == "reused_from_current84_missing15_board"),
        "measured_in_primitive60_identity_repack_14_64": sum(1 for row in status_rows if row.get("board_measurement_source") == "measured_in_primitive60_identity_repack_14_64"),
    }


def overlay_rows(status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in status_rows:
        formal_cycles = parse_int(row.get("formal_cycles"))
        board_cycles = parse_int(row.get("board_cycles"))
        delta_cycles = ""
        delta_percent = ""
        if formal_cycles is not None and board_cycles is not None:
            delta_cycles = board_cycles - formal_cycles
            if formal_cycles:
                delta_percent = f"{(delta_cycles / formal_cycles) * 100.0:.6f}"
        rows.append(
            {
                "current84_component_case_name": row.get("component_case_name", ""),
                "source_dataset": row.get("output_source_dataset", ""),
                "source_case_name": row.get("source_case_name", ""),
                "board_measurement_source": row.get("board_measurement_source", ""),
                "uart_status_code": row.get("uart_status_code", ""),
                "board_cycles": row.get("board_cycles", ""),
                "board_latency_ms": row.get("board_latency_ms", ""),
                "formal_cycles": row.get("formal_cycles", ""),
                "formal_latency_ms": row.get("formal_latency_ms", ""),
                "delta_cycles": delta_cycles,
                "delta_percent": delta_percent,
                "wns_ns": row.get("wns_ns", ""),
                "timing_clean": row.get("timing_clean", "False"),
                "measurement_json_path": row.get("measurement_json_path", ""),
                "bitstream_path": row.get("bitstream_path", ""),
                "timing_report_path": row.get("timing_report_path", ""),
                "notes": "; ".join(
                    part
                    for part in [
                        row.get("status", ""),
                        row.get("board_measurement_source", ""),
                        row.get("failure_reason", ""),
                        row.get("notes", ""),
                    ]
                    if part
                ),
            }
        )
    return rows


def latest_successful_recovery_by_component(rows: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    recovery_rows = rows if rows is not None else (read_csv(RECOVERY_STATUS_CSV) if RECOVERY_STATUS_CSV.exists() else [])
    latest: dict[str, dict[str, Any]] = {}
    for row in recovery_rows:
        if row.get("build_status") != "success" or row.get("measurement_status") != "success":
            continue
        component = row.get("component_case_name", "")
        if not component:
            continue
        latest[component] = row
    return latest


def write_readme(status_rows: list[dict[str, Any]]) -> None:
    counts = summary_counts(status_rows)
    blockers = [row for row in status_rows if not parse_bool(row.get("timing_clean"))]
    recovery_rows = read_csv(RECOVERY_STATUS_CSV) if RECOVERY_STATUS_CSV.exists() else []
    latest_recovery = latest_successful_recovery_by_component(recovery_rows)
    fresh_recovery_rows = [row for row in recovery_rows if row.get("attempt_name") == "fresh_default_shortpath"]
    fresh_recovery_success = [
        row
        for row in fresh_recovery_rows
        if row.get("build_status") == "success" and row.get("measurement_status") == "success"
    ]
    recovery_clean = [row for row in recovery_rows if parse_bool(row.get("timing_clean"))]
    lines = [
        "# current84 primitive60 AV7K325 Board Validation",
        "",
        f"Generated at: {timestamp()}",
        "",
        "This directory is current84 primitive-level board evidence only.",
        "",
        "Scope boundaries:",
        "- This is not MBConv fused latency.",
        "- This is not full-combo board latency.",
        "- This is not sonar classifier end-to-end latency.",
        "- Strict board latency may only use rows where `timing_clean=true`.",
        "",
        "Strict timing-clean rule:",
        "- `timing_clean=true` means UART `status_code=0` and AV7K325 harness WNS >= 0.",
        "- UART-ok rows with WNS < 0 are `board_measured_uart_ok_timing_fail`.",
        "",
        "Counts:",
        f"- total entries: {counts['total_entries']}",
        f"- UART status_code=0: {counts['uart_ok']}",
        f"- timing clean: {counts['timing_clean']}",
        f"- timing violations: {counts['timing_violation']}",
        f"- pending: {counts['pending']}",
        f"- missing board evidence: {counts['missing_board_evidence']}",
        "",
        "Outputs:",
        f"- runlist: `{RUNLIST_CSV}`",
        f"- status: `{STATUS_CSV}`",
        f"- timing failures: `{TIMING_FAILURES_CSV}`",
        f"- pending: `{PENDING_CSV}`",
        f"- summary: `{SUMMARY_JSON}`",
        f"- overlay: `{OVERLAY_CSV}`",
        f"- strict clean overlay: `{OVERLAY_STRICT_CLEAN_CSV}`",
        f"- blocker overlay: `{OVERLAY_BLOCKERS_CSV}`",
        "",
        "Reusable evidence sources:",
        "- full44 rows are marked `reused_from_full44_board`.",
        "- current84 missing15 rows are marked `reused_from_current84_missing15_board`.",
        "- identity_repack_14_64 is measured only under `identity_repack_14_64/`.",
        "",
        "Timing recovery:",
    ]
    if recovery_rows:
        lines.extend(
            [
                f"- attempts CSV: `{RECOVERY_STATUS_CSV}`",
                f"- summary JSON: `{RECOVERY_SUMMARY_JSON}`",
                f"- fresh bitstream rerun rows: {len(fresh_recovery_rows)}",
                f"- fresh bitstream rerun build+measurement success: {len(fresh_recovery_success)}",
                f"- recovered timing-clean attempts: {len(recovery_clean)}",
            ]
        )
    else:
        lines.append("- not run")
    lines.extend(
        [
            "",
            "Current blockers:",
        ]
    )
    if blockers:
        for row in blockers:
            recovery = latest_recovery.get(row.get("component_case_name", ""))
            if recovery:
                lines.append(
                    f"- `{row.get('component_case_name')}`: {row.get('status')} "
                    f"base WNS={row.get('wns_ns')} latest recovery `{recovery.get('attempt_name')}` "
                    f"WNS={recovery.get('wns_ns')}"
                )
            else:
                lines.append(f"- `{row.get('component_case_name')}`: {row.get('status')} WNS={row.get('wns_ns')}")
    else:
        lines.append("- none")
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primitive60_outputs(mode: str) -> list[dict[str, Any]]:
    preflight = assert_input_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runlist = build_primitive60_runlist()
    write_csv(RUNLIST_CSV, runlist, RUNLIST_FIELDS)
    status_rows = build_primitive60_status(runlist)
    write_csv(STATUS_CSV, status_rows, STATUS_FIELDS)
    write_csv(
        TIMING_FAILURES_CSV,
        [row for row in status_rows if row.get("status") == "board_measured_uart_ok_timing_fail"],
        TIMING_FAILURE_FIELDS,
    )
    write_csv(
        PENDING_CSV,
        [row for row in status_rows if row.get("status") in {"pending_board_measurement", "missing_board_evidence"}],
        STATUS_FIELDS,
    )
    overlay = overlay_rows(status_rows)
    write_csv(OVERLAY_CSV, overlay, OVERLAY_FIELDS)
    OVERLAY_JSON.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OVERLAY_STRICT_CLEAN_CSV, [row for row in overlay if parse_bool(row.get("timing_clean"))], OVERLAY_FIELDS)
    write_csv(OVERLAY_BLOCKERS_CSV, [row for row in overlay if not parse_bool(row.get("timing_clean"))], OVERLAY_FIELDS)
    payload = {
        "generated_at": timestamp(),
        "mode": mode,
        "output_dir": str(OUT_DIR),
        "inputs": {
            "current84_deployable_source_trace_csv": str(TRACE_CSV),
            "deployable_lut_v2_current84_json": str(DEPLOYABLE_LUT_JSON),
            "full44_runlist_csv": str(FULL44_RUNLIST_CSV),
            "full44_status_csv": str(FULL44_STATUS_CSV),
            "full44_measurements_csv": str(FULL44_MEASUREMENTS_CSV),
            "full44_timing_failures_csv": str(FULL44_TIMING_FAILURES_CSV),
            "missing15_status_csv": str(MISSING15_STATUS_CSV),
            "missing15_timing_failures_csv": str(MISSING15_TIMING_FAILURES_CSV),
            "identity_config": str(IDENTITY_CONFIG),
            "identity_formal_status_json": str(IDENTITY_STATUS_JSON),
            "identity_hls_csv": str(IDENTITY_HLS_CSV),
            "identity_vivado_downstream_csv": str(IDENTITY_DOWNSTREAM_CSV),
        },
        "preflight": preflight,
        "counts": summary_counts(status_rows),
        "status_counts": count_by(status_rows, "status"),
        "measurement_source_counts": count_by(status_rows, "board_measurement_source"),
        "timing_recovery": {
            "attempts_csv": str(RECOVERY_STATUS_CSV),
            "summary_json": str(RECOVERY_SUMMARY_JSON),
            "clean_recovery_rows": len(recovery_clean_lookup()),
        },
        "timing_failures": [
            {
                "component_case_name": row.get("component_case_name", ""),
                "source_case_name": row.get("source_case_name", ""),
                "source_dataset": row.get("output_source_dataset", ""),
                "wns_ns": row.get("wns_ns", ""),
            }
            for row in status_rows
            if row.get("status") == "board_measured_uart_ok_timing_fail"
        ],
        "outputs": {
            "runlist_csv": str(RUNLIST_CSV),
            "status_csv": str(STATUS_CSV),
            "timing_failures_csv": str(TIMING_FAILURES_CSV),
            "pending_csv": str(PENDING_CSV),
            "overlay_csv": str(OVERLAY_CSV),
            "overlay_json": str(OVERLAY_JSON),
            "overlay_strict_clean_csv": str(OVERLAY_STRICT_CLEAN_CSV),
            "overlay_blockers_csv": str(OVERLAY_BLOCKERS_CSV),
            "readme": str(README_MD),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(status_rows)
    return status_rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_identity_runlist(board_config: Path) -> list[dict[str, Any]]:
    board_cfg = load_yaml(board_config)
    status_payload = load_json(IDENTITY_STATUS_JSON)
    contract_id = status_payload.get("metadata", {}).get("measurement_contract", {}).get("contract_id", "")
    hls = {row["case_name"]: row for row in read_csv(IDENTITY_HLS_CSV)}
    downstream = {row["case_name"]: row for row in read_csv(IDENTITY_DOWNSTREAM_CSV)}
    rows = []
    for priority, entry in enumerate(status_payload["entries"], start=1):
        case_name = entry["case_name"]
        hls_row = hls.get(case_name, {})
        downstream_row = downstream.get(case_name, {})
        rows.append(
            {
                "priority": priority,
                "case_name": case_name,
                "op_id": entry.get("op_id", ""),
                "op_type": entry.get("op_type", ""),
                "case_dir": str(IDENTITY_CASE_ROOT / case_name),
                "board_config": str(board_config),
                "board_target_part": board_cfg.get("target_part", ""),
                "metric_source_target_part": downstream_row.get("target_part", ""),
                "hls_status": hls_row.get("synthesis_status", entry.get("hls_status", "")),
                "downstream_status": downstream_row.get("downstream_status", entry.get("downstream_status", "")),
                "expected_hls_latency_ms": hls_row.get("latency_ms", ""),
                "expected_hls_cycles": hls_row.get("latency_cycles", ""),
                "expected_post_route_wns_ns": downstream_row.get("post_route_setup_WNS_ns", ""),
                "expected_post_route_fmax_mhz": downstream_row.get("post_route_Fmax_est", ""),
                "expected_lut": downstream_row.get("post_route_LUT", ""),
                "expected_ff": downstream_row.get("post_route_FF", ""),
                "expected_bram_tile": downstream_row.get("post_route_BRAM_TILE", ""),
                "expected_dsp": downstream_row.get("post_route_DSP", ""),
                "expected_power_w": downstream_row.get("power_w", hls_row.get("power_w", "")),
                "contract_id": contract_id,
                "source_status_json": str(IDENTITY_STATUS_JSON),
                "source_hls_csv": str(IDENTITY_HLS_CSV),
                "source_downstream_csv": str(IDENTITY_DOWNSTREAM_CSV),
                "source_config": str(IDENTITY_CONFIG),
            }
        )
    return rows


def identity_status_from_run_row(row: dict[str, Any]) -> dict[str, Any]:
    status = dict(row)
    case_name = str(row["case_name"])
    case_dir = Path(str(row["case_dir"]))
    harness_project = IDENTITY_HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    manifest = harness_project / "harness_manifest.json"
    bitstream = harness_project / "bitstream" / f"{case_name}.bit"
    timing_report = harness_project / "reports" / "timing_summary.rpt"
    util_report = harness_project / "reports" / "utilization.rpt"
    measurement_json = IDENTITY_MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    measurement = load_measurement_payload(measurement_json)

    status_code = frame_value(measurement, "status_code")
    status_label = measurement.get("status_label", "")
    cycles = frame_value(measurement, "cycles")
    checksum = frame_value(measurement, "checksum")
    word_count = frame_value(measurement, "word_count")
    latency_ms = measurement.get("latency_ms", "")
    measurement_status = measurement.get("status", "success" if isinstance(measurement.get("frame"), dict) else "")

    wns = parse_wns(timing_report)
    timing_met = ""
    if wns != "":
        timing_met = str(float(wns) >= 0.0)
    uart_ok = parse_int(status_code) == 0
    timing_clean = bool(uart_ok and timing_met == "True")

    support_status = "not_started"
    board_execution_status = "not_started"
    failure_category = ""
    failure_reason = ""
    if timing_clean:
        support_status = "board_measured_timing_clean"
        board_execution_status = "success"
    elif isinstance(measurement.get("frame"), dict):
        if not uart_ok:
            support_status = "missing_board_evidence"
            board_execution_status = "failed_uart_status"
            failure_category = "uart_status_nonzero"
            failure_reason = f"UART status_code={status_code}"
        elif timing_met == "False":
            support_status = "board_measured_uart_ok_timing_fail"
            board_execution_status = "success_uart_timing_violation"
            failure_category = "negative_wns"
            failure_reason = f"AV7K325 harness WNS={wns}"
        else:
            support_status = "missing_board_evidence"
            board_execution_status = "success_timing_unknown"
            failure_category = "timing_report_missing"
            failure_reason = "UART status_code=0 but timing WNS was unavailable"
    elif measurement:
        support_status = "missing_board_evidence"
        board_execution_status = str(measurement.get("status", "measurement_failed"))
        failure_category = str(measurement.get("failure_category", "measurement_failed"))
        failure_reason = str(measurement.get("failure_reason", measurement.get("error", "")))
    elif not case_dir.exists():
        support_status = "missing_board_evidence"
        failure_category = "case_dir_missing"
        failure_reason = str(case_dir)
    elif not (case_dir / "synth.tcl").exists():
        support_status = "missing_board_evidence"
        failure_category = "synth_tcl_missing"
        failure_reason = str(case_dir / "synth.tcl")
    elif not has_csynth_artifacts(case_dir):
        support_status = "needs_csynth"
    elif not has_harness_rtl(case_dir):
        support_status = "needs_hls_export"
    elif not manifest.exists():
        support_status = "ready_for_harness_generation"
    elif not bitstream.exists():
        support_status = "harness_generated_needs_bitstream"
    else:
        support_status = "ready_for_board_measurement"

    status.update(
        {
            "status_timestamp": timestamp(),
            "case_dir_exists": str(case_dir.exists()),
            "synth_tcl_exists": str((case_dir / "synth.tcl").exists()),
            "csynth_report_available": str(has_csynth_artifacts(case_dir)),
            "component_xml_exists": str(component_xml(case_dir).exists()),
            "export_rtl_exists": str(has_harness_rtl(case_dir)),
            "harness_project_dir": str(harness_project),
            "harness_manifest_exists": str(manifest.exists()),
            "bitstream_exists": str(bitstream.exists()),
            "measurement_json": str(measurement_json) if measurement_json.exists() else "",
            "measurement_csv": str(IDENTITY_MEASUREMENTS_CSV) if IDENTITY_MEASUREMENTS_CSV.exists() else "",
            "csynth_status": "available" if has_csynth_artifacts(case_dir) else "",
            "export_status": "available" if has_harness_rtl(case_dir) else "",
            "harness_status": "available" if manifest.exists() else "",
            "bitstream_status": "available" if bitstream.exists() else "",
            "measurement_status": measurement_status,
            "timing_status": "met" if timing_met == "True" else "failed" if timing_met == "False" else "",
            "support_status": support_status,
            "board_execution_status": board_execution_status,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "board_status_code": status_code,
            "board_status_label": status_label,
            "board_cycles": cycles,
            "board_latency_ms": latency_ms,
            "board_word_count": word_count,
            "board_checksum": checksum,
            "board_timing_wns_ns": wns,
            "board_timing_met": timing_met,
            "board_timing_clean": str(timing_clean),
            "summary_json": str(IDENTITY_SUMMARY_JSON) if IDENTITY_SUMMARY_JSON.exists() else "",
            "stdout_log": str(IDENTITY_STDOUT_LOG) if IDENTITY_STDOUT_LOG.exists() else "",
        }
    )
    status.update(parse_utilization(util_report))
    return status


def identity_summary_counts(status_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_cases": len(status_rows),
        "uart_ok": sum(1 for row in status_rows if parse_int(row.get("board_status_code")) == 0),
        "timing_clean": sum(1 for row in status_rows if parse_bool(row.get("board_timing_clean"))),
        "timing_violation": sum(1 for row in status_rows if row.get("support_status") == "board_measured_uart_ok_timing_fail"),
        "pending": sum(1 for row in status_rows if row.get("support_status") not in MEASURED_STATUS_VALUES),
    }


def refresh_identity_outputs(board_config: Path, mode: str) -> list[dict[str, Any]]:
    IDENTITY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    IDENTITY_MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    IDENTITY_HARNESS_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    runlist = build_identity_runlist(board_config)
    write_csv(IDENTITY_RUNLIST_CSV, runlist, IDENTITY_RUNLIST_FIELDS)
    status_rows = [identity_status_from_run_row(row) for row in runlist]
    write_csv(IDENTITY_STATUS_CSV, status_rows, IDENTITY_STATUS_FIELDS)
    payload = {
        "generated_at": timestamp(),
        "mode": mode,
        "output_dir": str(IDENTITY_OUT_DIR),
        "counts": identity_summary_counts(status_rows),
        "runlist_csv": str(IDENTITY_RUNLIST_CSV),
        "status_csv": str(IDENTITY_STATUS_CSV),
        "measurements_csv": str(IDENTITY_MEASUREMENTS_CSV),
        "measurements_dir": str(IDENTITY_MEASUREMENTS_DIR),
        "harness_projects_dir": str(IDENTITY_HARNESS_PROJECTS_DIR),
        "cases": [
            {
                "case_name": row.get("case_name", ""),
                "support_status": row.get("support_status", ""),
                "board_status_code": row.get("board_status_code", ""),
                "board_timing_wns_ns": row.get("board_timing_wns_ns", ""),
                "board_timing_clean": row.get("board_timing_clean", ""),
            }
            for row in status_rows
        ],
    }
    IDENTITY_SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status_rows


def append_identity_log(message: str) -> None:
    IDENTITY_STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with IDENTITY_STDOUT_LOG.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def assert_serial_port(serial_port: str) -> None:
    if serial is None:
        raise RuntimeError("pyserial is required for COM board measurement")
    ports = {info.device for info in serial.tools.list_ports.comports()}
    if serial_port not in ports:
        raise RuntimeError(f"Serial port {serial_port} is not available. Detected ports: {sorted(ports)}")


def generate_identity_harness(case_dir: Path, board_config: Path, timeout_seconds: int | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(GENERATE_HARNESS_SCRIPT),
        "--case-dir",
        str(case_dir),
        "--board-config",
        str(board_config),
        "--output-dir",
        str(IDENTITY_HARNESS_PROJECTS_DIR),
        "--force",
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    manifest_path = IDENTITY_HARNESS_PROJECTS_DIR / f"harness_{case_dir.name}" / "harness_manifest.json"
    if timeout_payload is not None:
        timeout_payload["manifest"] = str(manifest_path) if manifest_path.exists() else None
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and manifest_path.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "manifest": str(manifest_path) if manifest_path.exists() else None,
    }


def write_identity_failure(case_name: str, category: str, reason: str, payload: dict[str, Any] | None = None) -> None:
    path = IDENTITY_MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    compact = dict(payload or {})
    stdout = compact.pop("stdout", None)
    if stdout:
        compact["stdout_log"] = str(IDENTITY_STDOUT_LOG)
        compact["stdout_bytes"] = len(str(stdout))
    data = {
        "case_name": case_name,
        "status": f"failed_{category}",
        "failure_category": category,
        "failure_reason": reason,
        "updated_at": timestamp(),
        "stage_payload": compact,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def augment_identity_measurement(row: dict[str, Any]) -> None:
    case_name = str(row["case_name"])
    path = IDENTITY_MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    payload = load_measurement_payload(path)
    if not payload:
        return
    harness_project = IDENTITY_HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    timing_report = harness_project / "reports" / "timing_summary.rpt"
    wns = parse_wns(timing_report)
    timing_met = bool(wns != "" and float(wns) >= 0.0)
    uart_ok = parse_int(frame_value(payload, "status_code")) == 0
    payload.update(
        {
            "case_name": case_name,
            "op_id": row.get("op_id", ""),
            "op_type": row.get("op_type", ""),
            "harness_project_dir": str(harness_project),
            "timing_report": str(timing_report),
            "board_timing_wns_ns": wns,
            "board_timing_met": timing_met if wns != "" else "",
            "board_timing_clean": bool(uart_ok and timing_met),
            "measurement_csv": str(IDENTITY_MEASUREMENTS_CSV),
            "updated_at": timestamp(),
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def measure_identity_case(row: dict[str, Any], serial_port: str, vivado: Path, runs: int, timeout_seconds: int | None) -> dict[str, Any]:
    case_name = str(row["case_name"])
    harness_project = IDENTITY_HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    json_out = IDENTITY_MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    command = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--project-dir",
        str(harness_project),
        "--serial-port",
        serial_port,
        "--vivado",
        str(vivado),
        "--csv",
        str(IDENTITY_MEASUREMENTS_CSV),
        "--runs",
        str(runs),
        "--json-out",
        str(json_out),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        write_identity_failure(case_name, "measurement_timeout", "UART measurement command timed out", timeout_payload)
        return timeout_payload
    status = {
        "status": "success" if result.returncode == 0 and json_out.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "measurement_json": str(json_out) if json_out.exists() else None,
    }
    if status["status"] != "success" and not json_out.exists():
        write_identity_failure(case_name, "measurement_failed", "UART measurement command failed", status)
    if json_out.exists():
        augment_identity_measurement(row)
    return status


def run_identity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.runs != 1:
        raise ValueError("identity board validation requires --runs 1")
    assert_input_files()
    assert_serial_port(args.serial_port)
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    refresh_identity_outputs(board_config, mode="pre-run")
    rows = read_csv(IDENTITY_RUNLIST_CSV)
    if len(rows) != 1 or rows[0].get("case_name") != IDENTITY_CASE_NAME:
        raise AssertionError(f"unexpected identity runlist: {IDENTITY_RUNLIST_CSV}")
    row = rows[0]
    case_name = row["case_name"]
    case_dir = Path(row["case_dir"])
    append_identity_log(f"[{timestamp()}] START {case_name}")

    csynth_status = ensure_csynth(
        case_dir,
        vitis_hls,
        force=False,
        timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
    )
    append_identity_log(f"[{timestamp()}] {case_name} csynth {csynth_status.get('status')}")
    if csynth_status.get("stdout"):
        append_identity_log(str(csynth_status["stdout"]))
    if csynth_status.get("status") in {"failed", "timeout"}:
        write_identity_failure(case_name, "csynth_failed", "Vitis HLS csynth failed or timed out", csynth_status)
        return refresh_identity_outputs(board_config, mode="run-identity")

    harness_status = generate_identity_harness(
        case_dir,
        board_config,
        timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
    )
    append_identity_log(f"[{timestamp()}] {case_name} harness {harness_status.get('status')}")
    if harness_status.get("stdout"):
        append_identity_log(str(harness_status["stdout"]))
    if harness_status.get("status") in {"failed", "timeout"}:
        write_identity_failure(case_name, "harness_failed", "Harness generation failed or timed out", harness_status)
        return refresh_identity_outputs(board_config, mode="run-identity")

    harness_project = IDENTITY_HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    bitstream_status = build_bitstream(
        harness_project,
        vivado,
        force=True,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    append_identity_log(f"[{timestamp()}] {case_name} bitstream {bitstream_status.get('status')}")
    if bitstream_status.get("stdout"):
        append_identity_log(str(bitstream_status["stdout"]))
    if bitstream_status.get("status") in {"failed", "timeout"}:
        write_identity_failure(case_name, "bitstream_failed", "Vivado bitstream failed or timed out", bitstream_status)
        return refresh_identity_outputs(board_config, mode="run-identity")

    measurement_status = measure_identity_case(
        row,
        serial_port=args.serial_port,
        vivado=vivado,
        runs=args.runs,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
    )
    append_identity_log(f"[{timestamp()}] {case_name} measurement {measurement_status.get('status')}")
    if measurement_status.get("stdout"):
        append_identity_log(str(measurement_status["stdout"]))
    status_rows = refresh_identity_outputs(board_config, mode="run-identity")
    write_primitive60_outputs(mode="post-identity")
    return status_rows


def recovery_attempt_configs(max_attempts: int) -> list[dict[str, Any]]:
    configs = [
        {
            "attempt_name": "fresh_default_shortpath",
            "strategy": "fresh_bitstream_rerun",
            "place_directive": "Default",
            "phys_opt_enabled": "false",
            "phys_opt_directive": "",
            "route_directive": "Default",
            "seed": "not_supported_by_vivado_2023_2_run_object",
        },
        {
            "attempt_name": "phys_opt_explore_route_explore",
            "strategy": "phys_opt_route_strategy",
            "place_directive": "Explore",
            "phys_opt_enabled": "true",
            "phys_opt_directive": "Explore",
            "route_directive": "Explore",
            "seed": "not_supported_by_vivado_2023_2_run_object",
        },
        {
            "attempt_name": "wl_driven_phys_opt",
            "strategy": "place_route_strategy",
            "place_directive": "WLDrivenBlockPlacement",
            "phys_opt_enabled": "true",
            "phys_opt_directive": "AggressiveExplore",
            "route_directive": "NoTimingRelaxation",
            "seed": "not_supported_by_vivado_2023_2_run_object",
        },
    ]
    return configs[: max(1, min(max_attempts, len(configs)))]


def tcl_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def write_recovery_tcl(status_row: dict[str, Any], attempt_dir: Path, cfg: dict[str, Any], board_config: Path) -> Path:
    case_name = status_row["source_case_name"]
    harness_project = Path(status_row["harness_project_dir"])
    board_cfg = load_yaml(board_config)
    build_tag = hashlib.sha1((case_name + cfg["attempt_name"]).encode()).hexdigest()[:12]
    build_dir = BOARD_HARNESS_DIR / "_vivado_builds" / f"rec_{build_tag}"
    bitstream_dir = attempt_dir / "bitstream"
    reports_dir = attempt_dir / "reports"
    stdout_log = attempt_dir / "vivado_stdout.log"
    tcl = f"""set project_name "rec_{hashlib.sha1((case_name + cfg['attempt_name']).encode()).hexdigest()[:12]}"
set target_part "{board_cfg.get('target_part', 'xc7k325t-ffg900-2')}"
set vivado_project_dir "{tcl_path(build_dir)}"
set rtl_dir "{tcl_path(harness_project / 'rtl')}"
set export_rtl_dir "{tcl_path(harness_project / 'export_rtl')}"
set constraint_file "{tcl_path(harness_project / 'constraints' / 'harness.xdc')}"
set bitstream_output_dir "{tcl_path(bitstream_dir)}"
set reports_output_dir "{tcl_path(reports_dir)}"
set artifact_basename "{case_name}"
set top_name "harness_top"

create_project $project_name $vivado_project_dir -part $target_part -force
set rtl_files [concat [glob -nocomplain -directory $rtl_dir *.v] [glob -nocomplain -directory $export_rtl_dir *.v]]
if {{[llength $rtl_files] == 0}} {{ error "No RTL files found for recovery build." }}
foreach rtl_file $rtl_files {{ add_files -norecurse $rtl_file }}
if {{[file exists $constraint_file]}} {{ read_xdc $constraint_file }}
set_property top $top_name [current_fileset]
update_compile_order -fileset sources_1
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {cfg['place_directive']} [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {cfg['route_directive']} [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED {cfg['phys_opt_enabled']} [get_runs impl_1]
"""
    if cfg.get("phys_opt_directive"):
        tcl += f"set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE {cfg['phys_opt_directive']} [get_runs impl_1]\n"
    tcl += f"""
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {{[get_property PROGRESS [get_runs synth_1]] != "100%"}} {{ error "Synthesis failed" }}
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1
if {{[get_property PROGRESS [get_runs impl_1]] != "100%"}} {{ puts "INFO: impl_1 progress = [get_property PROGRESS [get_runs impl_1]]" }}
file mkdir $bitstream_output_dir
file mkdir $reports_output_dir
open_run impl_1
report_timing_summary -file ${{reports_output_dir}}/timing_summary.rpt
report_utilization -file ${{reports_output_dir}}/utilization.rpt
report_clock_utilization -file ${{reports_output_dir}}/clock_utilization.rpt
write_bitstream -force ${{bitstream_output_dir}}/${{artifact_basename}}.bit
puts "Recovery bitstream generated: ${{bitstream_output_dir}}/${{artifact_basename}}.bit"
exit
"""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    path = attempt_dir / "build_recovery_bitstream.tcl"
    path.write_text(tcl, encoding="utf-8")
    stdout_log.write_text("", encoding="utf-8")
    return path


def create_recovery_manifest(status_row: dict[str, Any], attempt_dir: Path) -> Path:
    source_manifest = Path(status_row["harness_project_dir"]) / "harness_manifest.json"
    target_manifest = attempt_dir / "harness_manifest.json"
    if source_manifest.exists():
        payload = load_json(source_manifest)
        payload["recovery_attempt"] = {
            "source_harness_project_dir": status_row["harness_project_dir"],
            "source_status_csv": str(STATUS_CSV),
            "attempt_dir": str(attempt_dir),
            "created_at": timestamp(),
        }
        target_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target_manifest


def measure_recovery_attempt(
    status_row: dict[str, Any],
    attempt_dir: Path,
    serial_port: str,
    vivado: Path,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    case_name = status_row["source_case_name"]
    json_out = attempt_dir / "measurement.json"
    bitstream = attempt_dir / "bitstream" / f"{case_name}.bit"
    manifest = create_recovery_manifest(status_row, attempt_dir)
    command = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--manifest",
        str(manifest),
        "--bitstream",
        str(bitstream),
        "--serial-port",
        serial_port,
        "--vivado",
        str(vivado),
        "--csv",
        str(attempt_dir / "measurement.csv"),
        "--runs",
        "1",
        "--json-out",
        str(json_out),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        payload = {"status": "timeout", "failure_category": "measurement_timeout", "updated_at": timestamp()}
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return timeout_payload
    if json_out.exists():
        payload = load_measurement_payload(json_out)
        wns = parse_wns(attempt_dir / "reports" / "timing_summary.rpt")
        timing_met = bool(wns != "" and float(wns) >= 0.0)
        uart_ok = parse_int(frame_value(payload, "status_code")) == 0
        payload.update(
            {
                "case_name": case_name,
                "source_component_case_name": status_row.get("component_case_name", ""),
                "timing_report": str(attempt_dir / "reports" / "timing_summary.rpt"),
                "board_timing_wns_ns": wns,
                "board_timing_met": timing_met if wns != "" else "",
                "board_timing_clean": bool(uart_ok and timing_met),
                "updated_at": timestamp(),
            }
        )
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "success" if result.returncode == 0 and json_out.exists() else "failed", "returncode": result.returncode, "stdout": result.stdout}


def run_recovery(args: argparse.Namespace) -> list[dict[str, Any]]:
    status_rows = write_primitive60_outputs(mode="pre-recovery")
    failures = [row for row in status_rows if row.get("status") == "board_measured_uart_ok_timing_fail"]
    failures.sort(key=lambda row: parse_float(row.get("wns_ns")) if parse_float(row.get("wns_ns")) is not None else -999.0, reverse=True)
    if args.recovery_case:
        requested = set(args.recovery_case)
        failures = [row for row in failures if row.get("component_case_name") in requested or row.get("source_case_name") in requested]
    if args.recovery_limit is not None:
        failures = failures[: args.recovery_limit]
    board_config = Path(args.board_config).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    assert_serial_port(args.serial_port)
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    all_attempt_rows = read_csv(RECOVERY_STATUS_CSV)

    for row in failures:
        component_name = row["component_case_name"]
        source_case = row["source_case_name"]
        case_dir = RECOVERY_DIR / component_name
        for cfg in recovery_attempt_configs(args.recovery_attempts):
            attempt_dir = case_dir / cfg["attempt_name"]
            if (attempt_dir / "measurement.json").exists() and not args.skip_recovery_measure:
                continue
            started = timestamp()
            tcl = write_recovery_tcl(row, attempt_dir, cfg, board_config)
            stdout_log = attempt_dir / "vivado_stdout.log"
            stderr_log = attempt_dir / "vivado_stderr.log"
            command = ["cmd", "/c", str(vivado), "-mode", "batch", "-source", str(tcl)]
            result, timeout_payload = run_command(
                command,
                cwd=ROOT,
                timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
            )
            if timeout_payload is not None:
                stdout_log.write_text(str(timeout_payload.get("stdout", "")), encoding="utf-8", errors="replace")
                stderr_log.write_text(str(timeout_payload.get("stderr", "")), encoding="utf-8", errors="replace")
                build_status = "timeout"
                returncode = ""
            else:
                stdout_log.write_text(result.stdout or "", encoding="utf-8", errors="replace")
                stderr_log.write_text("", encoding="utf-8", errors="replace")
                bitstream = attempt_dir / "bitstream" / f"{source_case}.bit"
                build_status = "success" if result.returncode == 0 and bitstream.exists() else "failed"
                returncode = result.returncode
            measurement_status = ""
            if build_status == "success" and not args.skip_recovery_measure:
                measurement = measure_recovery_attempt(
                    row,
                    attempt_dir,
                    serial_port=args.serial_port,
                    vivado=vivado,
                    timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
                )
                measurement_status = str(measurement.get("status", ""))
            measurement_payload = load_measurement_payload(attempt_dir / "measurement.json")
            wns = parse_wns(attempt_dir / "reports" / "timing_summary.rpt")
            timing_clean = bool(parse_int(frame_value(measurement_payload, "status_code")) == 0 and parse_float(wns) is not None and parse_float(wns) >= 0.0)
            attempt_row = {
                "started_at": started,
                "completed_at": timestamp(),
                "component_case_name": component_name,
                "source_case_name": source_case,
                "source_dataset": row.get("output_source_dataset", ""),
                "attempt_name": cfg["attempt_name"],
                "strategy": cfg["strategy"],
                "place_directive": cfg["place_directive"],
                "phys_opt_enabled": cfg["phys_opt_enabled"],
                "phys_opt_directive": cfg.get("phys_opt_directive", ""),
                "route_directive": cfg["route_directive"],
                "seed": cfg["seed"],
                "build_status": build_status,
                "build_returncode": returncode,
                "measurement_status": measurement_status,
                "uart_status_code": frame_value(measurement_payload, "status_code"),
                "board_cycles": frame_value(measurement_payload, "cycles"),
                "board_latency_ms": measurement_payload.get("latency_ms", ""),
                "wns_ns": wns,
                "timing_clean": str(timing_clean),
                "attempt_dir": str(attempt_dir),
                "bitstream_path": str(attempt_dir / "bitstream" / f"{source_case}.bit"),
                "measurement_json_path": str(attempt_dir / "measurement.json") if (attempt_dir / "measurement.json").exists() else "",
                "timing_report_path": str(attempt_dir / "reports" / "timing_summary.rpt"),
                "utilization_report_path": str(attempt_dir / "reports" / "utilization.rpt"),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
            all_attempt_rows.append(attempt_row)
            write_csv(RECOVERY_STATUS_CSV, all_attempt_rows, RECOVERY_FIELDS)
            if timing_clean:
                break

    post_recovery_rows = write_primitive60_outputs(mode="post-recovery")
    latest_recovery = latest_successful_recovery_by_component(all_attempt_rows)
    remaining_blockers = []
    for row in post_recovery_rows:
        if row.get("status") != "board_measured_uart_ok_timing_fail":
            continue
        recovery = latest_recovery.get(row.get("component_case_name", ""))
        blocker = {
            "component_case_name": row.get("component_case_name", ""),
            "source_case_name": row.get("source_case_name", ""),
            "base_wns_ns": row.get("wns_ns", ""),
        }
        if recovery:
            blocker.update(
                {
                    "latest_recovery_attempt": recovery.get("attempt_name", ""),
                    "latest_recovery_wns_ns": recovery.get("wns_ns", ""),
                    "latest_recovery_uart_status_code": recovery.get("uart_status_code", ""),
                    "latest_recovery_measurement_json_path": recovery.get("measurement_json_path", ""),
                    "latest_recovery_timing_report_path": recovery.get("timing_report_path", ""),
                }
            )
        remaining_blockers.append(blocker)

    payload = {
        "generated_at": timestamp(),
        "recovery_dir": str(RECOVERY_DIR),
        "ordered_policy": "WNS sorted from near zero to worse",
        "attempts_csv": str(RECOVERY_STATUS_CSV),
        "total_attempt_rows": len(all_attempt_rows),
        "timing_clean_attempts": sum(1 for row in all_attempt_rows if parse_bool(row.get("timing_clean"))),
        "remaining_primitive60_blockers": remaining_blockers,
    }
    RECOVERY_SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return all_attempt_rows


def main() -> int:
    args = parse_args()
    if args.mode == "init":
        status_rows = write_primitive60_outputs(mode="init")
    elif args.mode == "refresh":
        refresh_identity_outputs(Path(args.board_config).expanduser().resolve(), mode="refresh")
        status_rows = write_primitive60_outputs(mode="refresh")
    elif args.mode == "run-identity":
        run_identity(args)
        status_rows = write_primitive60_outputs(mode="post-identity")
    elif args.mode == "recover":
        run_recovery(args)
        status_rows = write_primitive60_outputs(mode="post-recovery")
    else:
        write_primitive60_outputs(mode="init")
        run_identity(args)
        run_recovery(args)
        status_rows = write_primitive60_outputs(mode="all")

    counts = summary_counts(status_rows)
    print(f"status_rows={len(status_rows)} status_csv={STATUS_CSV}")
    print(f"summary_json={SUMMARY_JSON}")
    print(f"counts={json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
