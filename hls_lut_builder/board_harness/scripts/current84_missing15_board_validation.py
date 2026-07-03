#!/usr/bin/env python3
"""Dedicated AV7K325 board validation for the current84 missing15 primitive set."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    ensure_export as batch_ensure_export,
    ensure_csynth,
    run_command,
)


V2_CURRENT84_DIR = RESULTS_DIR / "v2_current84"
DEFAULT_STATUS_JSON = V2_CURRENT84_DIR / "formal_lut_status_v2_current84_missing.json"
DEFAULT_HLS_CSV = V2_CURRENT84_DIR / "current84_missing_hls.csv"
DEFAULT_DOWNSTREAM_CSV = V2_CURRENT84_DIR / "current84_missing_vivado_downstream.csv"
DEFAULT_SOURCE_CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_v2_current84_missing_run15_2026_05_17.yaml"
DEFAULT_MISSING_PRIMITIVES_CSV = V2_CURRENT84_DIR / "current84_missing_primitives.csv"
DEFAULT_BLOCKERS_CSV = V2_CURRENT84_DIR / "current84_missing_blockers.csv"
DEFAULT_CASE_ROOT = V2_CURRENT84_DIR / "missing_p"

RESULTS_SUBDIR = BOARD_HARNESS_DIR / "results" / "v2_current84_missing15_board"
RUNLIST_CSV = RESULTS_SUBDIR / "v2_current84_missing15_board_runlist.csv"
STATUS_CSV = RESULTS_SUBDIR / "v2_current84_missing15_board_status.csv"
MEASUREMENTS_CSV = RESULTS_SUBDIR / "v2_current84_missing15_board_measurements.csv"
SUMMARY_JSON = RESULTS_SUBDIR / "v2_current84_missing15_board_batch_summary.json"
TIMING_FAILURES_CSV = RESULTS_SUBDIR / "v2_current84_missing15_board_timing_failures.csv"
README_MD = RESULTS_SUBDIR / "README.md"
MEASUREMENTS_DIR = RESULTS_SUBDIR / "measurements"
HARNESS_PROJECTS_DIR = RESULTS_SUBDIR / "harness_projects"
STDOUT_LOG = RESULTS_SUBDIR / "v2_current84_missing15_board_run.out.log"

EXPECTED_CASES = 15

RUNLIST_FIELDS = [
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

STATUS_FIELDS = RUNLIST_FIELDS + [
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

TIMING_FAILURE_FIELDS = [
    "priority",
    "case_name",
    "op_type",
    "support_status",
    "board_execution_status",
    "failure_category",
    "failure_reason",
    "board_status_code",
    "board_timing_wns_ns",
    "harness_project_dir",
    "measurement_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current84 missing15 AV7K325 board validation")
    parser.add_argument("mode", choices=("init", "refresh", "run"), help="Operation mode")
    parser.add_argument("--status-json", default=str(DEFAULT_STATUS_JSON))
    parser.add_argument("--hls-csv", default=str(DEFAULT_HLS_CSV))
    parser.add_argument("--downstream-csv", default=str(DEFAULT_DOWNSTREAM_CSV))
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG))
    parser.add_argument("--missing-primitives-csv", default=str(DEFAULT_MISSING_PRIMITIVES_CSV))
    parser.add_argument("--blockers-csv", default=str(DEFAULT_BLOCKERS_CSV))
    parser.add_argument("--case-root", default=str(DEFAULT_CASE_ROOT))
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--case", action="append", default=None, help="Optional case_name filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of cases")
    parser.add_argument("--no-wait-conflicts", action="store_true", help="Do not wait for old full44 board jobs")
    parser.add_argument("--csynth-timeout-minutes", type=int, default=60)
    parser.add_argument("--export-timeout-minutes", type=int, default=30)
    parser.add_argument("--harness-timeout-minutes", type=int, default=10)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=90)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def has_csynth_artifacts(case_dir: Path) -> bool:
    report_dir = case_dir / "project" / "solution1" / "syn" / "report"
    if not report_dir.exists():
        return False
    return any(report_dir.glob("*_csynth.xml")) or any(report_dir.glob("*_csynth.rpt"))


def component_xml(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"


def export_rtl_dir(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "impl" / "ip" / "hdl" / "verilog"


def syn_rtl_dir(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "syn" / "verilog"


def source_module_name(case_dir: Path) -> str:
    src_dir = case_dir / "src"
    candidates = sorted(src_dir.glob("*.cpp"))
    return candidates[0].stem if candidates else case_dir.name


def has_harness_rtl(case_dir: Path) -> bool:
    if export_rtl_dir(case_dir).exists() and any(export_rtl_dir(case_dir).glob("*.v")):
        return True
    module_name = source_module_name(case_dir)
    return (syn_rtl_dir(case_dir) / f"{module_name}.v").exists()


def choose_free_subst_drive() -> str | None:
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        drive = f"{letter}:"
        if not Path(f"{drive}\\").exists():
            return drive
    return None


def run_short_path_export(
    case_dir: Path,
    short_case_dir: Path,
    vitis_hls: Path,
    timeout_seconds: int | None,
    retry_label: str,
) -> dict[str, Any]:
    script_path = short_case_dir / "export_ip_shortpath.tcl"
    script_path.write_text(
        "\n".join(
            [
                "open_project project",
                "open_solution solution1",
                "config_export -format ip_catalog",
                "config_export -rtl verilog",
                "export_design -rtl verilog -format ip_catalog",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if vitis_hls.suffix.lower() in {".bat", ".cmd"}:
        command = ["cmd", "/c", str(vitis_hls), "-f", str(script_path)]
    else:
        command = [str(vitis_hls), "-f", str(script_path)]
    result, timeout_payload = run_command(command, cwd=short_case_dir, timeout_seconds=timeout_seconds)
    log_path = case_dir / "export_ip_shortpath.log"
    export_zip = case_dir / "project" / "solution1" / "impl" / "export.zip"
    if timeout_payload is not None:
        log_path.write_text(str(timeout_payload.get("stdout", "")), encoding="utf-8")
        timeout_payload.update(
            {
                "component_xml": str(component_xml(case_dir)) if component_xml(case_dir).exists() else None,
                "export_zip": str(export_zip) if export_zip.exists() else None,
                "short_path_retry": retry_label,
                "short_path_log": str(log_path),
            }
        )
        return timeout_payload
    log_path.write_text(result.stdout, encoding="utf-8")
    status = {
        "status": "success" if result.returncode == 0 and export_zip.exists() and component_xml(case_dir).exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "component_xml": str(component_xml(case_dir)) if component_xml(case_dir).exists() else None,
        "export_zip": str(export_zip) if export_zip.exists() else None,
        "short_path_retry": retry_label,
        "short_path_log": str(log_path),
    }
    (case_dir / "export_ip_shortpath_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status


def ensure_export(case_dir: Path, vitis_hls: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    status = batch_ensure_export(case_dir, vitis_hls, force=force, timeout_seconds=timeout_seconds)
    if status.get("status") not in {"failed", "timeout"} or component_xml(case_dir).exists():
        return status
    if os.name != "nt":
        return status

    drive = choose_free_subst_drive()
    if drive is None:
        status["short_path_retry"] = "no_free_subst_drive"
        return status

    subst_root = Path(f"{drive}\\")
    setup = subprocess.run(
        ["cmd", "/c", "subst", drive, str(case_dir)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if setup.returncode != 0:
        status["short_path_retry"] = f"subst_failed: {setup.stdout}"
        return status
    try:
        return run_short_path_export(
            case_dir,
            subst_root,
            vitis_hls,
            timeout_seconds,
            retry_label=f"{drive} -> {case_dir}",
        )
    finally:
        subprocess.run(
            ["cmd", "/c", "subst", drive, "/D"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )


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


def parse_wns(timing_report: Path) -> str:
    if not timing_report.exists():
        return ""
    lines = timing_report.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 8]:
            parts = candidate.split()
            if not parts:
                continue
            try:
                float(parts[0])
            except ValueError:
                continue
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
    for line in util_report.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        if parts[1] == "Slice LUTs":
            values["board_harness_lut"] = parts[2]
        elif parts[1] == "Slice Registers":
            values["board_harness_ff"] = parts[2]
        elif parts[1] == "Block RAM Tile":
            values["board_harness_bram_tile"] = parts[2]
        elif parts[1] == "DSPs":
            values["board_harness_dsp"] = parts[2]
    return values


def assert_header_only(path: Path, label: str) -> None:
    rows = read_csv(path)
    if rows:
        raise AssertionError(f"{label} must be header-only, found {len(rows)} rows: {path}")


def assert_serial_port(serial_port: str) -> None:
    if serial is None:
        raise RuntimeError("pyserial is required for COM port validation and board measurement")
    ports = list(serial.tools.list_ports.comports())
    visible = {info.device for info in ports}
    if serial_port not in visible:
        available = ", ".join(sorted(visible)) or "none"
        raise RuntimeError(f"Serial port {serial_port} is not visible; available ports: {available}")


def preflight(args: argparse.Namespace, *, require_serial_port: bool) -> dict[str, Any]:
    status_path = Path(args.status_json).expanduser().resolve()
    hls_csv = Path(args.hls_csv).expanduser().resolve()
    downstream_csv = Path(args.downstream_csv).expanduser().resolve()
    missing_primitives_csv = Path(args.missing_primitives_csv).expanduser().resolve()
    blockers_csv = Path(args.blockers_csv).expanduser().resolve()

    status_payload = load_json(status_path)
    entries = status_payload.get("entries") or []
    if len(entries) != EXPECTED_CASES:
        raise AssertionError(f"Expected {EXPECTED_CASES} formal status entries, found {len(entries)}")
    bad_status = [entry.get("case_name") for entry in entries if entry.get("status") != "measured_implementation"]
    if bad_status:
        raise AssertionError(f"Formal status entries are not all measured_implementation: {bad_status}")

    hls_rows = read_csv(hls_csv)
    if len(hls_rows) != EXPECTED_CASES:
        raise AssertionError(f"Expected {EXPECTED_CASES} HLS rows, found {len(hls_rows)}")
    bad_hls = [row.get("case_name") for row in hls_rows if row.get("synthesis_status") != "success"]
    if bad_hls:
        raise AssertionError(f"HLS rows are not all success: {bad_hls}")

    downstream_rows = read_csv(downstream_csv)
    if len(downstream_rows) != EXPECTED_CASES:
        raise AssertionError(f"Expected {EXPECTED_CASES} downstream rows, found {len(downstream_rows)}")
    bad_downstream = [row.get("case_name") for row in downstream_rows if row.get("downstream_status") != "success"]
    if bad_downstream:
        raise AssertionError(f"Downstream rows are not all success: {bad_downstream}")

    assert_header_only(missing_primitives_csv, "current84_missing_primitives.csv")
    assert_header_only(blockers_csv, "current84_missing_blockers.csv")
    if require_serial_port:
        assert_serial_port(args.serial_port)

    return {
        "status_entries": len(entries),
        "hls_rows": len(hls_rows),
        "downstream_rows": len(downstream_rows),
        "serial_port_checked": require_serial_port,
    }


def build_runlist(args: argparse.Namespace) -> list[dict[str, Any]]:
    status_path = Path(args.status_json).expanduser().resolve()
    hls_csv = Path(args.hls_csv).expanduser().resolve()
    downstream_csv = Path(args.downstream_csv).expanduser().resolve()
    source_config = Path(args.source_config).expanduser().resolve()
    case_root = Path(args.case_root).expanduser().resolve()
    board_config = Path(args.board_config).expanduser().resolve()
    board_cfg = load_yaml(board_config)

    status_payload = load_json(status_path)
    contract_id = (
        status_payload.get("metadata", {})
        .get("measurement_contract", {})
        .get("contract_id", "")
    )
    hls_by_case = {row["case_name"]: row for row in read_csv(hls_csv)}
    downstream_by_case = {row["case_name"]: row for row in read_csv(downstream_csv)}

    rows: list[dict[str, Any]] = []
    for priority, entry in enumerate(status_payload["entries"], start=1):
        case_name = entry["case_name"]
        hls = hls_by_case.get(case_name, {})
        downstream = downstream_by_case.get(case_name, {})
        rows.append(
            {
                "priority": priority,
                "case_name": case_name,
                "op_id": entry.get("op_id", ""),
                "op_type": entry.get("op_type", ""),
                "case_dir": str(case_root / case_name),
                "board_config": str(board_config),
                "board_target_part": board_cfg.get("target_part", ""),
                "metric_source_target_part": downstream.get("target_part", ""),
                "hls_status": hls.get("synthesis_status", entry.get("hls_status", "")),
                "downstream_status": downstream.get("downstream_status", entry.get("downstream_status", "")),
                "expected_hls_latency_ms": hls.get("latency_ms", ""),
                "expected_hls_cycles": hls.get("latency_cycles", ""),
                "expected_post_route_wns_ns": downstream.get("post_route_setup_WNS_ns", ""),
                "expected_post_route_fmax_mhz": downstream.get("post_route_Fmax_est", ""),
                "expected_lut": downstream.get("post_route_LUT", ""),
                "expected_ff": downstream.get("post_route_FF", ""),
                "expected_bram_tile": downstream.get("post_route_BRAM_TILE", ""),
                "expected_dsp": downstream.get("post_route_DSP", ""),
                "expected_power_w": downstream.get("power_w", hls.get("power_w", "")),
                "contract_id": contract_id,
                "source_status_json": str(status_path),
                "source_hls_csv": str(hls_csv),
                "source_downstream_csv": str(downstream_csv),
                "source_config": str(source_config),
            }
        )
    return rows


def load_measurement_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "error": f"Could not parse {path}"}
    return payload if isinstance(payload, dict) else {}


def status_from_run_row(row: dict[str, Any]) -> dict[str, Any]:
    status = dict(row)
    case_name = str(row["case_name"])
    case_dir = Path(str(row["case_dir"]))
    harness_project = HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    manifest = harness_project / "harness_manifest.json"
    bitstream = harness_project / "bitstream" / f"{case_name}.bit"
    timing_report = harness_project / "reports" / "timing_summary.rpt"
    util_report = harness_project / "reports" / "utilization.rpt"
    measurement_json = MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    measurement = load_measurement_payload(measurement_json)
    frame = measurement.get("frame") if isinstance(measurement.get("frame"), dict) else {}

    status_code = frame.get("status_code", measurement.get("status_code", ""))
    status_label = measurement.get("status_label", "")
    cycles = frame.get("cycles", measurement.get("cycles", ""))
    checksum = frame.get("checksum", measurement.get("checksum", ""))
    word_count = frame.get("word_count", measurement.get("word_count", ""))
    latency_ms = measurement.get("latency_ms", "")
    measurement_status = measurement.get("status", "success" if frame else "")

    wns = parse_wns(timing_report)
    timing_met = ""
    if wns != "":
        timing_met = str(float(wns) >= 0.0)
    uart_ok = parse_int(status_code) == 0
    timing_clean = bool(uart_ok and timing_met == "True")

    failure_category = ""
    failure_reason = ""
    support_status = "not_started"
    board_execution_status = "not_started"
    if timing_clean:
        support_status = "board_measured_timing_clean"
        board_execution_status = "success"
    elif frame:
        if not uart_ok:
            support_status = "board_measured_uart_status_fail"
            board_execution_status = "failed_uart_status"
            failure_category = "uart_status_nonzero"
            failure_reason = f"UART status_code={status_code}"
        elif timing_met == "False":
            support_status = "board_measured_uart_ok_timing_fail"
            board_execution_status = "success_uart_timing_violation"
            failure_category = "negative_wns"
            failure_reason = f"AV7K325 harness WNS={wns}"
        else:
            support_status = "board_measured_timing_unknown"
            board_execution_status = "success_timing_unknown"
            failure_category = "timing_report_missing"
            failure_reason = "UART status_code=0 but timing WNS was unavailable"
    elif measurement:
        support_status = str(measurement.get("status", "failed_measurement"))
        board_execution_status = support_status
        failure_category = str(measurement.get("failure_category", "measurement_failed"))
        failure_reason = str(measurement.get("failure_reason", measurement.get("error", "")))
    elif not case_dir.exists():
        support_status = "unsupported"
        failure_category = "case_dir_missing"
        failure_reason = str(case_dir)
    elif not (case_dir / "synth.tcl").exists():
        support_status = "unsupported"
        failure_category = "synth_tcl_missing"
        failure_reason = str(case_dir / "synth.tcl")
    elif not has_csynth_artifacts(case_dir):
        support_status = "needs_csynth"
    elif not component_xml(case_dir).exists() and not has_harness_rtl(case_dir):
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
            "measurement_csv": str(MEASUREMENTS_CSV) if MEASUREMENTS_CSV.exists() else "",
            "csynth_status": "available" if has_csynth_artifacts(case_dir) else "",
            "export_status": (
                "available"
                if component_xml(case_dir).exists() and export_rtl_dir(case_dir).exists()
                else "syn_rtl_fallback"
                if has_harness_rtl(case_dir)
                else ""
            ),
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
            "summary_json": str(SUMMARY_JSON) if SUMMARY_JSON.exists() else "",
            "stdout_log": str(STDOUT_LOG) if STDOUT_LOG.exists() else "",
        }
    )
    status.update(parse_utilization(util_report))
    return status


def refresh_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not RUNLIST_CSV.exists():
        init_outputs(args)
    rows = read_csv(RUNLIST_CSV)
    status_rows = [status_from_run_row(row) for row in rows]
    write_csv(STATUS_CSV, status_rows, STATUS_FIELDS)
    write_timing_failures(status_rows)
    write_summary(status_rows, mode="refresh")
    write_readme(status_rows)
    return status_rows


def write_timing_failures(status_rows: list[dict[str, Any]]) -> None:
    failures = []
    for row in status_rows:
        if row.get("board_timing_clean") == "True":
            continue
        if row.get("support_status") in {"not_started", "needs_csynth", "needs_hls_export", "ready_for_harness_generation", "harness_generated_needs_bitstream", "ready_for_board_measurement"}:
            continue
        failures.append(row)
    write_csv(TIMING_FAILURES_CSV, failures, TIMING_FAILURE_FIELDS)


def summary_counts(status_rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(status_rows)
    timing_clean = sum(1 for row in status_rows if row.get("board_timing_clean") == "True")
    uart_ok = sum(1 for row in status_rows if parse_int(row.get("board_status_code")) == 0)
    negative_wns = sum(1 for row in status_rows if row.get("failure_category") == "negative_wns")
    uart_fail = sum(1 for row in status_rows if row.get("failure_category") == "uart_status_nonzero")
    failed = sum(1 for row in status_rows if row.get("support_status", "").startswith("failed") or row.get("failure_category"))
    return {
        "total_cases": total,
        "timing_clean_pass": timing_clean,
        "not_timing_clean": total - timing_clean,
        "uart_status_code_0": uart_ok,
        "negative_wns": negative_wns,
        "uart_failures": uart_fail,
        "failures_recorded": failed,
    }


def write_summary(status_rows: list[dict[str, Any]], *, mode: str, preflight_payload: dict[str, Any] | None = None) -> None:
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": timestamp(),
        "mode": mode,
        "output_dir": str(RESULTS_SUBDIR),
        "inputs": {
            "status_json": str(DEFAULT_STATUS_JSON),
            "hls_csv": str(DEFAULT_HLS_CSV),
            "downstream_csv": str(DEFAULT_DOWNSTREAM_CSV),
            "source_config": str(DEFAULT_SOURCE_CONFIG),
            "missing_primitives_csv": str(DEFAULT_MISSING_PRIMITIVES_CSV),
            "blockers_csv": str(DEFAULT_BLOCKERS_CSV),
        },
        "outputs": {
            "runlist_csv": str(RUNLIST_CSV),
            "status_csv": str(STATUS_CSV),
            "measurements_csv": str(MEASUREMENTS_CSV),
            "measurements_dir": str(MEASUREMENTS_DIR),
            "summary_json": str(SUMMARY_JSON),
            "timing_failures_csv": str(TIMING_FAILURES_CSV),
            "readme": str(README_MD),
        },
        "execution_commands": [
            r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py init",
            r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py run --serial-port COM5 --runs 1",
            r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py refresh",
        ],
        "preflight": preflight_payload or {},
        "counts": summary_counts(status_rows),
        "cases": [
            {
                "priority": row.get("priority", ""),
                "case_name": row.get("case_name", ""),
                "op_type": row.get("op_type", ""),
                "support_status": row.get("support_status", ""),
                "board_status_code": row.get("board_status_code", ""),
                "board_timing_wns_ns": row.get("board_timing_wns_ns", ""),
                "board_timing_clean": row.get("board_timing_clean", ""),
                "failure_category": row.get("failure_category", ""),
            }
            for row in status_rows
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(status_rows: list[dict[str, Any]]) -> None:
    counts = summary_counts(status_rows)
    lines = [
        "# current84_missing15 AV7K325 Board Validation",
        "",
        f"Generated at: {timestamp()}",
        "",
        "## Inputs",
        "",
        f"- formal status: `{DEFAULT_STATUS_JSON}`",
        f"- HLS CSV: `{DEFAULT_HLS_CSV}`",
        f"- Vivado downstream CSV: `{DEFAULT_DOWNSTREAM_CSV}`",
        f"- source config: `{DEFAULT_SOURCE_CONFIG}`",
        f"- board config: `{DEFAULT_BOARD_CONFIG}`",
        "",
        "## Execution Commands",
        "",
        "```powershell",
        r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py init",
        r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py run --serial-port COM5 --runs 1",
        r"python hls_lut_builder\board_harness\scripts\current84_missing15_board_validation.py refresh",
        "```",
        "",
        "## Results",
        "",
        f"- total cases: {counts['total_cases']}",
        f"- timing-clean pass: {counts['timing_clean_pass']}",
        f"- not timing-clean: {counts['not_timing_clean']}",
        f"- UART status_code=0: {counts['uart_status_code_0']}",
        f"- negative WNS: {counts['negative_wns']}",
        f"- UART failures: {counts['uart_failures']}",
        "",
        "Timing-clean requires UART `status_code=0` and AV7K325 harness post-route `WNS>=0`.",
        "",
        "## Outputs",
        "",
        f"- runlist: `{RUNLIST_CSV}`",
        f"- status: `{STATUS_CSV}`",
        f"- per-case measurements: `{MEASUREMENTS_DIR}`",
        f"- batch summary: `{SUMMARY_JSON}`",
        f"- timing failures: `{TIMING_FAILURES_CSV}`",
        f"- measurement append CSV: `{MEASUREMENTS_CSV}`",
    ]
    README_MD.parent.mkdir(parents=True, exist_ok=True)
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    preflight_payload = preflight(args, require_serial_port=False)
    RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    runlist = build_runlist(args)
    write_csv(RUNLIST_CSV, runlist, RUNLIST_FIELDS)
    status_rows = [status_from_run_row(row) for row in runlist]
    write_csv(STATUS_CSV, status_rows, STATUS_FIELDS)
    write_timing_failures(status_rows)
    write_summary(status_rows, mode="init", preflight_payload=preflight_payload)
    write_readme(status_rows)
    return status_rows


def append_log(message: str) -> None:
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with STDOUT_LOG.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    stdout = compact.pop("stdout", None)
    if stdout:
        compact["stdout_log"] = str(STDOUT_LOG)
        compact["stdout_bytes"] = len(str(stdout))
    return compact


def log_stage(case_name: str, stage: str, payload: dict[str, Any]) -> None:
    append_log(f"[{timestamp()}] {case_name} {stage}")
    stdout = payload.get("stdout")
    if stdout:
        append_log(str(stdout))


def generate_harness_fresh(case_dir: Path, board_config: Path, timeout_seconds: int | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(GENERATE_HARNESS_SCRIPT),
        "--case-dir",
        str(case_dir),
        "--board-config",
        str(board_config),
        "--output-dir",
        str(HARNESS_PROJECTS_DIR),
        "--force",
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    manifest_path = HARNESS_PROJECTS_DIR / f"harness_{case_dir.name}" / "harness_manifest.json"
    if timeout_payload is not None:
        timeout_payload["manifest"] = str(manifest_path) if manifest_path.exists() else None
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and manifest_path.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "manifest": str(manifest_path) if manifest_path.exists() else None,
    }


def write_failure_measurement(case_name: str, category: str, reason: str, stage_payload: dict[str, Any] | None = None) -> None:
    path = MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": case_name,
        "status": f"failed_{category}",
        "failure_category": category,
        "failure_reason": reason,
        "updated_at": timestamp(),
        "stage_payload": compact_payload(stage_payload or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def augment_measurement_json(row: dict[str, Any]) -> None:
    case_name = str(row["case_name"])
    path = MEASUREMENTS_DIR / f"{case_name}.measurement.json"
    payload = load_measurement_payload(path)
    if not payload:
        return
    harness_project = HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    timing_report = harness_project / "reports" / "timing_summary.rpt"
    wns = parse_wns(timing_report)
    timing_met = bool(wns != "" and float(wns) >= 0.0)
    frame = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
    uart_ok = parse_int(frame.get("status_code")) == 0
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
            "measurement_csv": str(MEASUREMENTS_CSV),
            "updated_at": timestamp(),
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def measure_case_fresh(
    row: dict[str, Any],
    *,
    serial_port: str,
    vivado: Path,
    runs: int,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    case_name = str(row["case_name"])
    harness_project = HARNESS_PROJECTS_DIR / f"harness_{case_name}"
    json_out = MEASUREMENTS_DIR / f"{case_name}.measurement.json"
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
        str(MEASUREMENTS_CSV),
        "--runs",
        str(runs),
        "--json-out",
        str(json_out),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        write_failure_measurement(case_name, "measurement_timeout", "UART measurement command timed out", timeout_payload)
        return timeout_payload
    status = {
        "status": "success" if result.returncode == 0 and json_out.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "measurement_json": str(json_out) if json_out.exists() else None,
    }
    if status["status"] != "success" and not json_out.exists():
        write_failure_measurement(case_name, "measurement_failed", "UART measurement command failed", status)
    if json_out.exists():
        augment_measurement_json(row)
    return status


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(rows)
    if args.case:
        requested = set(args.case)
        selected = [row for row in selected if row.get("case_name") in requested]
        present = {row.get("case_name") for row in selected}
        missing = sorted(requested.difference(present))
        if missing:
            raise ValueError(f"Requested cases are not in runlist: {missing}")
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def query_conflicting_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$p = Get-CimInstance Win32_Process | Where-Object { "
            "($_.Name -like 'python*' -and $_.CommandLine -like '*v2_board_validation.py run*') -or "
            "($_.Name -like 'vivado*' -and $_.CommandLine -like '*board_harness*projects*harness_*') -or "
            "($_.Name -like 'vivado*' -and $_.CommandLine -like '*task_worker.tcl*') -or "
            "($_.Name -like 'vivado*' -and $_.CommandLine -like '*harness_top.vds*') -or "
            "($_.Name -like 'vivado*' -and $_.CommandLine -like '*harness_top.vdi*') "
            "} | Select-Object ProcessId,Name,CommandLine; "
            "if ($p) { $p | ConvertTo-Json -Compress }"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    text = result.stdout.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        append_log(f"[warn] Unable to parse process query output: {text}")
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def wait_for_conflicts() -> None:
    clear_checks = 0
    while True:
        conflicts = query_conflicting_processes()
        if not conflicts:
            clear_checks += 1
            if clear_checks >= 4:
                append_log(f"[{timestamp()}] No existing full44/Vivado board process detected after confirmation window")
                return
            append_log(f"[{timestamp()}] No existing full44/Vivado board process detected; confirming idle window")
            time.sleep(60)
            continue
        clear_checks = 0
        append_log(f"[{timestamp()}] Waiting for {len(conflicts)} existing full44/Vivado board process(es) to exit")
        for process in conflicts:
            cmd = str(process.get("CommandLine", ""))
            append_log(f"  pid={process.get('ProcessId')} name={process.get('Name')} cmd={cmd[:240]}")
        time.sleep(60)


def mark_stage_failure(case_name: str, stage: str, payload: dict[str, Any]) -> None:
    reason = str(payload.get("status", "failed"))
    if payload.get("returncode") not in (None, ""):
        reason += f" returncode={payload.get('returncode')}"
    write_failure_measurement(case_name, stage, reason, payload)


def run_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    preflight_payload = preflight(args, require_serial_port=True)
    if not args.no_wait_conflicts:
        wait_for_conflicts()
    if not RUNLIST_CSV.exists():
        init_outputs(args)

    rows = selected_rows(read_csv(RUNLIST_CSV), args)
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    append_log(f"[{timestamp()}] current84 missing15 run start selected_count={len(rows)} serial_port={args.serial_port}")

    for row in rows:
        case_name = str(row["case_name"])
        case_dir = Path(str(row["case_dir"]))
        append_log(f"[{timestamp()}] START {case_name}")

        csynth_status = ensure_csynth(
            case_dir,
            vitis_hls,
            force=False,
            timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
        )
        log_stage(case_name, "csynth", csynth_status)
        if csynth_status.get("status") in {"failed", "timeout"}:
            mark_stage_failure(case_name, "csynth_timeout" if csynth_status.get("status") == "timeout" else "csynth_failed", csynth_status)
            refresh_outputs(args)
            continue

        export_status = ensure_export(
            case_dir,
            vitis_hls,
            force=False,
            timeout_seconds=args.export_timeout_minutes * 60 if args.export_timeout_minutes > 0 else None,
        )
        log_stage(case_name, "export", export_status)
        if export_status.get("status") in {"failed", "timeout"}:
            if not has_harness_rtl(case_dir):
                mark_stage_failure(case_name, "export_timeout" if export_status.get("status") == "timeout" else "export_failed", export_status)
                refresh_outputs(args)
                continue
            append_log(f"[{timestamp()}] {case_name} export failed; continuing with HLS syn/verilog RTL fallback")

        harness_status = generate_harness_fresh(
            case_dir,
            board_config,
            timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
        )
        log_stage(case_name, "harness", harness_status)
        if harness_status.get("status") in {"failed", "timeout"}:
            mark_stage_failure(case_name, "harness_timeout" if harness_status.get("status") == "timeout" else "harness_failed", harness_status)
            refresh_outputs(args)
            continue

        harness_project = HARNESS_PROJECTS_DIR / f"harness_{case_name}"
        bitstream_status = build_bitstream(
            harness_project,
            vivado,
            force=True,
            timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
        )
        log_stage(case_name, "bitstream", bitstream_status)
        if bitstream_status.get("status") in {"failed", "timeout"}:
            mark_stage_failure(case_name, "bitstream_timeout" if bitstream_status.get("status") == "timeout" else "bitstream_failed", bitstream_status)
            refresh_outputs(args)
            continue

        measurement_status = measure_case_fresh(
            row,
            serial_port=args.serial_port,
            vivado=vivado,
            runs=args.runs,
            timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        )
        log_stage(case_name, "measurement", measurement_status)
        if measurement_status.get("status") != "success":
            mark_stage_failure(case_name, "measurement_failed", measurement_status)
        refresh_outputs(args)
        append_log(f"[{timestamp()}] END {case_name} measurement_status={measurement_status.get('status')}")

    status_rows = refresh_outputs(args)
    write_summary(status_rows, mode="run", preflight_payload=preflight_payload)
    write_readme(status_rows)
    return status_rows


def main() -> int:
    args = parse_args()
    if args.runs != 1:
        raise ValueError("This validation requires fresh one-shot COM5 measurement with --runs 1")
    if args.mode == "init":
        rows = init_outputs(args)
    elif args.mode == "refresh":
        rows = refresh_outputs(args)
    else:
        rows = run_cases(args)
    print(f"status_rows={len(rows)} status_csv={STATUS_CSV}")
    print(f"summary_json={SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
