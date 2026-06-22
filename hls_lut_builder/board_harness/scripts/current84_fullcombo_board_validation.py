#!/usr/bin/env python3
"""current84 full-combo AV7K325 board validation.

This script consumes the existing fullcombo84 dry-run plan, runs real board
measurements for direct single-op combos and stitched MBConv combos, and writes
combo-level evidence without modifying primitive LUTs or legacy board CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
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

import generate_harness as gh

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CURRENT84_DEPLOYABLE_DIR = RESULTS_DIR / "v2_current84_deployable_lut"
TRACE_CSV = CURRENT84_DEPLOYABLE_DIR / "current84_deployable_source_trace.csv"
DEPLOYABLE_LUT_JSON = CURRENT84_DEPLOYABLE_DIR / "deployable_lut_v2_current84.json"
PRIMITIVE_OVERLAY_CSV = (
    BOARD_HARNESS_DIR
    / "results"
    / "v2_current84_primitive60_board"
    / "board_latency_overlay_current84_primitive60.csv"
)

OUT_DIR = BOARD_HARNESS_DIR / "results" / "v2_current84_fullcombo_board"
DRY_RUNLIST_CSV = OUT_DIR / "v2_current84_fullcombo84_runlist.csv"
DRY_STATUS_CSV = OUT_DIR / "v2_current84_fullcombo84_status.csv"
PILOTS_CSV = OUT_DIR / "v2_current84_fullcombo84_pilots.csv"

DIRECT_PILOT_DIR = OUT_DIR / "direct_pilot"
MBCONV_PILOT_DIR = OUT_DIR / "mbconv_stitched_pilot"
MBCONV_DEBUG_DIR = OUT_DIR / "mbconv_stitched_debug"
MBCONV_RECOVERY_DIR = OUT_DIR / "mbconv_stitched_recovery"
FULL_MEASUREMENT_DIR = OUT_DIR / "fullcombo84_measurement"
CASE_RESULTS_DIR = OUT_DIR / "case_results"

BOARD_STATUS_CSV = OUT_DIR / "v2_current84_fullcombo84_board_status.csv"
BOARD_MEASUREMENTS_CSV = OUT_DIR / "v2_current84_fullcombo84_board_measurements.csv"
TIMING_FAILURES_CSV = OUT_DIR / "v2_current84_fullcombo84_timing_failures.csv"
BLOCKERS_CSV = OUT_DIR / "v2_current84_fullcombo84_blockers.csv"
SUMMARY_JSON = OUT_DIR / "v2_current84_fullcombo84_summary.json"
LATENCY_OVERLAY_CSV = OUT_DIR / "v2_current84_fullcombo84_latency_overlay.csv"
LATENCY_OVERLAY_JSON = OUT_DIR / "v2_current84_fullcombo84_latency_overlay.json"
README_MD = OUT_DIR / "README.md"

MBCONV_DECOMPOSITION = "pw_expand+dw+pw_project"
MBCONV_TOPOLOGY = "pw_expand->fifo_repack->dw->fifo_repack->pw_project"
MBCONV_DEBUG_MODES = ("full_stitched", "pw_expand_only", "pw_expand_to_dw", "pw_project_only")
MBCONV_DEBUG_DEFAULT_FIFO_DEPTHS = (512, 1024, 2048, 4096)
MBCONV_DEBUG_STATUS_BITS = {
    "counter_running": 0,
    "counter_done": 1,
    "sink_done": 2,
    "all_axil_done": 3,
    "any_axil_error": 4,
    "source_done": 5,
    "watchdog_expired": 6,
    "result_sent": 7,
}

MEASURED_STATUS = {
    "board_measured_timing_clean",
    "board_measured_uart_ok_timing_fail",
}
PILOT_BLOCKER_STATUS = {
    "harness_build_fail",
    "bitstream_build_fail",
    "board_runtime_fail",
}

RUNLIST_FIELDS = [
    "priority",
    "combo_case_name",
    "combo_family",
    "decomposition",
    "harness_kind",
    "execution_scope",
    "pilot_group",
    "component_count",
    "component_roles",
    "component_case_names",
    "selected_existing_case_names",
    "selected_sources",
    "component_operators",
    "component_op_types",
    "component_shape_names",
    "current_status",
    "current_detail",
    "primitive_coverage_status",
    "deployable_primitive_coverage",
    "prediction_baseline_status",
    "prediction_baseline_latency_ms",
    "prediction_baseline_cycles",
    "prediction_baseline_source",
    "board_combo_latency_ms",
    "board_combo_cycles",
    "notes",
]

BOARD_STATUS_FIELDS = RUNLIST_FIELDS + [
    "status_timestamp",
    "combo_measurement_source",
    "support_status",
    "measurement_status",
    "board_execution_status",
    "failure_category",
    "failure_reason",
    "uart_status_code",
    "uart_status_label",
    "board_combo_cycles",
    "board_combo_latency_ms",
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
    "batch_group",
    "stdout_log",
    "stderr_log",
]

MEASUREMENT_FIELDS = [
    "timestamp",
    "combo_case_name",
    "decomposition",
    "harness_kind",
    "pilot_group",
    "batch_group",
    "status",
    "uart_status_code",
    "uart_status_label",
    "board_combo_cycles",
    "board_combo_latency_ms",
    "board_word_count",
    "board_checksum",
    "wns_ns",
    "timing_clean",
    "prediction_baseline_cycles",
    "prediction_baseline_latency_ms",
    "delta_cycles",
    "delta_percent",
    "measurement_json_path",
    "bitstream_path",
    "harness_project_dir",
    "timing_report_path",
    "utilization_report_path",
]

OVERLAY_FIELDS = [
    "combo_case_name",
    "decomposition",
    "harness_kind",
    "component_roles",
    "component_case_names",
    "selected_existing_case_names",
    "board_measurement_source",
    "uart_status_code",
    "board_combo_cycles",
    "board_combo_latency_ms",
    "prediction_baseline_cycles",
    "prediction_baseline_latency_ms",
    "delta_cycles",
    "delta_percent",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "timing_report_path",
    "notes",
]

BLOCKER_FIELDS = [
    "combo_case_name",
    "decomposition",
    "harness_kind",
    "pilot_group",
    "status",
    "failure_category",
    "failure_reason",
    "uart_status_code",
    "wns_ns",
    "measurement_json_path",
    "harness_project_dir",
    "timing_report_path",
    "stdout_log",
    "stderr_log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="current84 fullcombo84 AV7K325 board validation")
    parser.add_argument(
        "mode",
        choices=(
            "init",
            "refresh",
            "sanity-check",
            "remeasure-mbconv-pilots",
            "debug-mbconv",
            "recover-mbconv-pilots",
            "run-mbconv-pilots",
            "run-pilots",
            "run-all",
            "all",
        ),
    )
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-device-pattern", default="*xc7k325t*")
    parser.add_argument("--hw-target-index", type=int, default=0)
    parser.add_argument("--program-retries", type=int, default=2)
    parser.add_argument("--uart-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--post-program-delay-seconds", type=float, default=0.25)
    parser.add_argument("--csynth-timeout-minutes", type=int, default=60)
    parser.add_argument("--harness-timeout-minutes", type=int, default=10)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=120)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20)
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug-mode", action="append", choices=MBCONV_DEBUG_MODES, default=None)
    parser.add_argument("--debug-fifo-depths", default=",".join(str(value) for value in MBCONV_DEBUG_DEFAULT_FIFO_DEPTHS))
    parser.add_argument("--debug-watchdog-cycles", type=int, default=50_000_000)
    parser.add_argument("--debug-limit", type=int, default=0, help="Limit debug case count; 0 means no limit")
    parser.add_argument(
        "--recovery-variants",
        default="default,performance_explore,post_route_phys_opt",
        help="Comma-separated timing recovery variants for recover-mbconv-pilots.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-measure", action="store_true")
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
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    return payload if isinstance(payload, dict) else {}


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


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise ValueError(f"Expected positive integer in list, got {parsed}")
        items.append(parsed)
    if not items:
        raise ValueError("Expected at least one positive integer")
    return items


def frame_value(measurement: dict[str, Any], key: str) -> Any:
    frame = measurement.get("frame")
    if isinstance(frame, dict) and key in frame:
        return frame.get(key)
    return measurement.get(key, "")


def parse_wns(timing_report: Path) -> str:
    if not timing_report.exists():
        return ""
    text = timing_report.read_text(encoding="utf-8", errors="replace")
    number_pattern = re.compile(r"^-?\d+(?:\.\d+)?$")
    lines = text.splitlines()
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


def decode_debug_status_bits(value: Any) -> dict[str, bool]:
    parsed = parse_int(value)
    if parsed is None:
        return {name: False for name in MBCONV_DEBUG_STATUS_BITS}
    return {name: bool(parsed & (1 << bit)) for name, bit in MBCONV_DEBUG_STATUS_BITS.items()}


def assert_serial_port(serial_port: str) -> None:
    if serial is None:
        return
    ports = {info.device.upper(): info.description for info in serial.tools.list_ports.comports()}
    if serial_port.upper() not in ports:
        raise RuntimeError(f"Serial port {serial_port} not found. Available ports: {ports}")


def assert_inputs() -> dict[str, Any]:
    required = [
        DRY_RUNLIST_CSV,
        DRY_STATUS_CSV,
        PILOTS_CSV,
        TRACE_CSV,
        DEPLOYABLE_LUT_JSON,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fullcombo input files: {missing}")
    runlist = read_csv(DRY_RUNLIST_CSV)
    status = read_csv(DRY_STATUS_CSV)
    pilots = read_csv(PILOTS_CSV)
    trace = read_csv(TRACE_CSV)
    deployable = load_json(DEPLOYABLE_LUT_JSON)
    if len(runlist) != 84 or len(status) != 84:
        raise AssertionError("fullcombo dry-run runlist/status must each contain 84 rows")
    if sum(1 for row in runlist if row.get("decomposition") == "direct") != 60:
        raise AssertionError("fullcombo runlist direct count must be 60")
    if sum(1 for row in runlist if row.get("decomposition") == MBCONV_DECOMPOSITION) != 24:
        raise AssertionError("fullcombo runlist MBConv count must be 24")
    if len(pilots) != 9:
        raise AssertionError("pilot file must contain 9 rows")
    if sum(1 for row in pilots if row.get("pilot_group") == "direct_pilot") != 5:
        raise AssertionError("direct pilot count must be 5")
    if sum(1 for row in pilots if row.get("pilot_group") == "mbconv_stage2_pilot") != 4:
        raise AssertionError("MBConv pilot count must be 4")
    if len(trace) != 60:
        raise AssertionError("primitive source trace must contain 60 rows")
    if len(deployable.get("entries", [])) != 60:
        raise AssertionError("current84 deployable LUT must contain 60 entries")
    inputs = list(required)
    if PRIMITIVE_OVERLAY_CSV.exists():
        inputs.append(PRIMITIVE_OVERLAY_CSV)
    return {
        "runlist_rows": len(runlist),
        "status_rows": len(status),
        "pilot_rows": len(pilots),
        "trace_rows": len(trace),
        "deployable_entries": len(deployable.get("entries", [])),
        "primitive_overlay_present": PRIMITIVE_OVERLAY_CSV.exists(),
        "input_hashes": {str(path): sha256_file(path) for path in inputs},
    }


def split_pipe(value: str) -> list[str]:
    return [part for part in str(value).split("|") if part]


def case_search_roots() -> list[Path]:
    return [
        RESULTS_DIR / "v2_arch84_e2e_extension" / "p",
        RESULTS_DIR / "v2_current84" / "missing_p",
        RESULTS_DIR / "v2_current84" / "identity_repack_14_64_p",
        RESULTS_DIR / "v2_pointwise_v21" / "p",
        RESULTS_DIR / "v2_fixed_vector" / "p",
        RESULTS_DIR / "v2_failed44_fixed_vector" / "p",
        RESULTS_DIR / "v2_k5_low_complexity" / "p",
        RESULTS_DIR / "v2_depthwise_fallback_blockers" / "p",
        RESULTS_DIR / "p",
        RESULTS_DIR / "p_ext",
    ]


def resolve_case_dir(case_name: str) -> Path:
    for root in case_search_roots():
        candidate = root / case_name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find case directory for {case_name}")


def component_xml(case_dir: Path) -> Path:
    return case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"


def export_rtl_dir(case_dir: Path) -> Path:
    ip_rtl = case_dir / "project" / "solution1" / "impl" / "ip" / "hdl" / "verilog"
    syn_rtl = case_dir / "project" / "solution1" / "syn" / "verilog"
    return ip_rtl if ip_rtl.exists() else syn_rtl


def first_cpp(case_dir: Path) -> Path:
    candidates = sorted((case_dir / "src").glob("*.cpp"))
    if not candidates:
        raise FileNotFoundError(f"No C++ source in {case_dir / 'src'}")
    return candidates[0]


def top_module_name(case_dir: Path) -> str:
    cxml = component_xml(case_dir)
    if cxml.exists():
        return str(gh.parse_component(cxml)["component_name"])
    return first_cpp(case_dir).stem


def top_verilog_path(case_dir: Path, module_name: str) -> Path:
    path = export_rtl_dir(case_dir) / f"{module_name}.v"
    if path.exists():
        return path
    matches = sorted(export_rtl_dir(case_dir).glob("*.v"))
    if not matches:
        raise FileNotFoundError(f"No RTL files in {export_rtl_dir(case_dir)}")
    return matches[0]


def classify_measurement(measurement: dict[str, Any], wns: str, default_fail_status: str = "board_runtime_fail") -> tuple[str, str, str, str, bool]:
    status_code = parse_int(frame_value(measurement, "status_code"))
    wns_value = parse_float(wns)
    if status_code == 0 and wns_value is not None and wns_value >= 0.0:
        return "board_measured_timing_clean", "success", "", "", True
    if status_code == 0 and wns_value is not None and wns_value < 0.0:
        return (
            "board_measured_uart_ok_timing_fail",
            "success_uart_timing_violation",
            "timing_violation",
            f"AV7K325 harness WNS={wns}",
            False,
        )
    return (
        default_fail_status,
        "failed",
        "board_runtime_fail",
        f"missing strict UART/timing evidence status_code={status_code} WNS={wns}",
        False,
    )


def result_path(combo_case_name: str) -> Path:
    return CASE_RESULTS_DIR / f"{combo_case_name}.result.json"


def write_result(combo_case_name: str, payload: dict[str, Any]) -> None:
    path = result_path(combo_case_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not CASE_RESULTS_DIR.exists():
        return results
    for path in CASE_RESULTS_DIR.glob("*.result.json"):
        payload = load_json(path)
        combo = str(payload.get("combo_case_name") or path.stem.replace(".result", ""))
        results[combo] = payload
    return results


def direct_output_dir(row: dict[str, str], full: bool) -> Path:
    if full:
        return FULL_MEASUREMENT_DIR / "direct" / row["combo_case_name"]
    return DIRECT_PILOT_DIR / row["combo_case_name"]


def mbconv_output_dir(row: dict[str, str], full: bool) -> Path:
    if full:
        return FULL_MEASUREMENT_DIR / "mbconv_stitched" / row["combo_case_name"]
    return MBCONV_PILOT_DIR / row["combo_case_name"]


def run_generate_harness(case_dir: Path, board_config: Path, output_dir: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    manifest = output_dir / f"harness_{case_dir.name}" / "harness_manifest.json"
    if manifest.exists() and not force:
        return {"status": "skipped_existing", "manifest": str(manifest)}
    command = [
        sys.executable,
        str(GENERATE_HARNESS_SCRIPT),
        "--case-dir",
        str(case_dir),
        "--board-config",
        str(board_config),
        "--output-dir",
        str(output_dir),
    ]
    if force:
        command.append("--force")
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        timeout_payload["manifest"] = str(manifest) if manifest.exists() else ""
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and manifest.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "manifest": str(manifest) if manifest.exists() else "",
    }


def measure_harness(
    *,
    manifest: Path,
    bitstream: Path,
    csv_path: Path,
    json_out: Path,
    serial_port: str,
    vivado: Path,
    hw_server_url: str,
    hw_device_pattern: str,
    hw_target_index: int,
    program_retries: int,
    runs: int,
    uart_timeout_seconds: float,
    post_program_delay_seconds: float,
    timeout_seconds: int | None,
    skip_measure: bool,
) -> dict[str, Any]:
    if skip_measure:
        return {"status": "skipped"}
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
        "--hw-server-url",
        hw_server_url,
        "--hw-device-pattern",
        hw_device_pattern,
        "--hw-target-index",
        str(hw_target_index),
        "--program-retries",
        str(program_retries),
        "--csv",
        str(csv_path),
        "--runs",
        str(runs),
        "--timeout-seconds",
        str(uart_timeout_seconds),
        "--post-program-delay-seconds",
        str(post_program_delay_seconds),
        "--json-out",
        str(json_out),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and json_out.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
    }


def run_direct_combo(row: dict[str, str], args: argparse.Namespace, full: bool) -> dict[str, Any]:
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    combo = row["combo_case_name"]
    output_dir = direct_output_dir(row, full)
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_projects_dir = output_dir / "harness_projects"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    component_case = split_pipe(row["selected_existing_case_names"])[0] or split_pipe(row["component_case_names"])[0]
    case_dir = resolve_case_dir(component_case)
    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row.get("decomposition", ""),
        "harness_kind": "single_op_harness",
        "pilot_group": row.get("pilot_group", ""),
        "batch_group": "full_direct" if full else "direct_pilot",
        "component_case_name": component_case,
        "case_dir": str(case_dir),
        "started_at": timestamp(),
    }

    csynth = ensure_csynth(
        case_dir,
        vitis_hls,
        force=False,
        timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
    )
    (logs_dir / "csynth.out.log").write_text(str(csynth.get("stdout", "")), encoding="utf-8", errors="replace")
    result_payload["csynth_status"] = csynth.get("status", "")
    if csynth.get("status") in {"failed", "timeout"}:
        result_payload.update(status="harness_build_fail", failure_category="csynth_failed", failure_reason=str(csynth.get("stdout", ""))[:1000])
        write_result(combo, result_payload)
        return result_payload

    harness = run_generate_harness(
        case_dir,
        board_config,
        harness_projects_dir,
        force=True,
        timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
    )
    (logs_dir / "harness.out.log").write_text(str(harness.get("stdout", "")), encoding="utf-8", errors="replace")
    result_payload["harness_status"] = harness.get("status", "")
    if harness.get("status") not in {"success", "skipped_existing"}:
        result_payload.update(status="harness_build_fail", failure_category="harness_build_fail", failure_reason=str(harness.get("stdout", ""))[:1000])
        write_result(combo, result_payload)
        return result_payload

    harness_project = harness_projects_dir / f"harness_{component_case}"
    bitstream = build_bitstream(
        harness_project,
        vivado,
        force=True,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
    result_payload["bitstream_status"] = bitstream.get("status", "")
    bitstream_path = harness_project / "bitstream" / f"{component_case}.bit"
    if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
        result_payload.update(
            status="bitstream_build_fail",
            failure_category="bitstream_build_fail",
            failure_reason=str(bitstream.get("stdout", ""))[:1000],
            harness_project_dir=str(harness_project),
        )
        write_result(combo, result_payload)
        return result_payload

    json_out = measurements_dir / f"{combo}.measurement.json"
    measurement_csv = output_dir / "direct_measurements_raw.csv"
    measurement = measure_harness(
        manifest=harness_project / "harness_manifest.json",
        bitstream=bitstream_path,
        csv_path=measurement_csv,
        json_out=json_out,
        serial_port=args.serial_port,
        vivado=vivado,
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
    measurement_payload = load_json(json_out)
    wns = parse_wns(harness_project / "reports" / "timing_summary.rpt")
    status, board_execution, failure_category, failure_reason, timing_clean = classify_measurement(measurement_payload, wns)
    measurement_payload.update(
        {
            "combo_case_name": combo,
            "combo_decomposition": "direct",
            "full_combo_latency_claim": "direct_fullcombo_board_latency",
            "board_timing_wns_ns": wns,
            "board_timing_clean": timing_clean,
            "updated_at": timestamp(),
        }
    )
    json_out.write_text(json.dumps(measurement_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    util = parse_utilization(harness_project / "reports" / "utilization.rpt")
    result_payload.update(
        {
            "completed_at": timestamp(),
            "status": status,
            "board_execution_status": board_execution,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "measurement_status": measurement.get("status", ""),
            "uart_status_code": frame_value(measurement_payload, "status_code"),
            "uart_status_label": measurement_payload.get("status_label", ""),
            "board_combo_cycles": frame_value(measurement_payload, "cycles"),
            "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
            "board_word_count": frame_value(measurement_payload, "word_count"),
            "board_checksum": frame_value(measurement_payload, "checksum"),
            "wns_ns": wns,
            "timing_clean": timing_clean,
            "measurement_json_path": str(json_out),
            "measurement_csv_path": str(measurement_csv),
            "bitstream_path": str(bitstream_path),
            "harness_project_dir": str(harness_project),
            "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
            "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
            "stdout_log": str(logs_dir / "measurement.out.log"),
            "stderr_log": "",
        }
    )
    result_payload.update(util)
    write_result(combo, result_payload)
    return result_payload


def module_names_in_rtl(rtl_files: list[Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for path in rtl_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"^\s*(?:\(\*.*?\*\)\s*)*module\s+([A-Za-z_][A-Za-z0-9_$]*)", text, re.MULTILINE):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def rename_and_copy_rtl(src_dir: Path, dst_dir: Path, role: str, top_module: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    rtl_files = sorted(src_dir.glob("*.v"))
    if not rtl_files:
        raise FileNotFoundError(f"No RTL files found in {src_dir}")
    module_names = module_names_in_rtl(rtl_files)
    mapping = {name: f"{role}_{name}" for name in module_names}
    replacements = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
    for path in rtl_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for old, new in replacements:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        (dst_dir / f"{role}_{path.name}").write_text(text, encoding="utf-8")
    return mapping.get(top_module, f"{role}_{top_module}")


def suffix_signal(kind: str, suffix: str, role: str) -> str:
    suffix_map = {
        "Addr_A": "addra",
        "EN_A": "ena",
        "WEN_A": "wena",
        "Din_A": "dina",
        "Dout_A": "douta",
        "Clk_A": "clka",
        "Rst_A": "rsta",
        "Addr_B": "addrb",
        "EN_B": "enb",
        "WEN_B": "wenb",
        "Din_B": "dinb",
        "Dout_B": "doutb",
        "Clk_B": "clkb",
        "Rst_B": "rstb",
    }
    return f"{role}_{kind}_{suffix_map[suffix]}"


def clamp_bram_port_a_reset(param_logic: str) -> str:
    return re.sub(r"\.rsta\([A-Za-z0-9_]+_rsta\)", ".rsta(1'b0)", param_logic)


def clamp_bram_all_resets(param_logic: str) -> str:
    text = re.sub(r"\.rsta\([A-Za-z0-9_]+_rsta\)", ".rsta(1'b0)", param_logic)
    return re.sub(r"\.rstb\([A-Za-z0-9_]+_rstb\)", ".rstb(1'b0)", text)


def build_param_logic_for_spec(spec: dict[str, Any], data_dir: Path, param_reset_style: str = "kernel") -> str:
    role = spec["role"]
    constants = spec["constants"]
    case_meta = spec["case_meta"]
    blocks: list[str] = []
    for kind in ("weights", "biases"):
        iface = spec["param_interfaces"][kind]
        if iface["mode"] == "absent":
            continue
        if iface["mode"] == "scalar":
            blocks.append(gh.build_scalar_param_logic(f"{role}_{kind}", int(iface["data_width"])))
            continue
        if iface["mode"] == "bram":
            depth = int(case_meta[f"{kind}_depth"])
            mem_file = f"{role}_{kind}.mem"
            gh.generate_param_mem(data_dir / mem_file, depth, int(iface["data_width"]), 3 if kind == "weights" else 11)
            blocks.append(
                gh.build_bram_param_logic(
                    f"{role}_{kind}",
                    int(iface["data_width"]),
                    int(iface["addr_width"]),
                    depth,
                    f"../data/{mem_file}",
                    bool(iface["has_port_b"]),
                )
            )
            continue
        if iface["mode"] == "partitioned_bram":
            bank_depth = gh.partitioned_bank_depth(kind, constants, iface["banks"], int(case_meta[f"{kind}_depth"]))
            for bank_idx, bank in enumerate(iface["banks"]):
                bank_kind = f"{role}_{bank['prefix']}"
                mem_file = f"{bank_kind}.mem"
                gh.generate_param_mem(data_dir / mem_file, bank_depth, int(bank["data_width"]), (3 if kind == "weights" else 11) + bank_idx)
                blocks.append(
                    gh.build_bram_param_logic(
                        bank_kind,
                        int(bank["data_width"]),
                        int(bank["addr_width"]),
                        bank_depth,
                        f"../data/{mem_file}",
                        bool(bank["has_port_b"]),
                    )
                )
    text = "\n\n".join(blocks)
    if param_reset_style == "const_zero":
        return clamp_bram_port_a_reset(text)
    if param_reset_style == "const_zero_all":
        return clamp_bram_all_resets(text)
    if param_reset_style != "kernel":
        raise ValueError(f"Unsupported parameter reset style: {param_reset_style}")
    return text


def build_kernel_instance_for_spec(spec: dict[str, Any], input_prefix: str, output_prefix: str) -> str:
    role = spec["role"]
    ports = spec["top_ports"]
    lines = [f"{spec['renamed_module']} u_{role}_kernel ("]

    def connect(port: str, signal: str) -> None:
        if port in ports:
            lines.append(f"    .{port}({signal}),")

    connect("ap_clk", "clk_main")
    connect("ap_rst_n", "rst_n")
    connect("interrupt", f"{role}_interrupt")

    axil = {
        "s_axi_control_AWVALID": f"{role}_s_axi_control_AWVALID",
        "s_axi_control_AWREADY": f"{role}_s_axi_control_AWREADY",
        "s_axi_control_AWADDR": f"{role}_s_axi_control_AWADDR",
        "s_axi_control_WVALID": f"{role}_s_axi_control_WVALID",
        "s_axi_control_WREADY": f"{role}_s_axi_control_WREADY",
        "s_axi_control_WDATA": f"{role}_s_axi_control_WDATA",
        "s_axi_control_WSTRB": f"{role}_s_axi_control_WSTRB",
        "s_axi_control_ARVALID": f"{role}_s_axi_control_ARVALID",
        "s_axi_control_ARREADY": f"{role}_s_axi_control_ARREADY",
        "s_axi_control_ARADDR": f"{role}_s_axi_control_ARADDR",
        "s_axi_control_RVALID": f"{role}_s_axi_control_RVALID",
        "s_axi_control_RREADY": f"{role}_s_axi_control_RREADY",
        "s_axi_control_RDATA": f"{role}_s_axi_control_RDATA",
        "s_axi_control_RRESP": f"{role}_s_axi_control_RRESP",
        "s_axi_control_BVALID": f"{role}_s_axi_control_BVALID",
        "s_axi_control_BREADY": f"{role}_s_axi_control_BREADY",
        "s_axi_control_BRESP": f"{role}_s_axi_control_BRESP",
    }
    for port, signal in axil.items():
        connect(port, signal)

    axis = {
        "input_stream_TDATA": f"{input_prefix}_tdata",
        "input_stream_TKEEP": f"{input_prefix}_tkeep",
        "input_stream_TSTRB": f"{input_prefix}_tstrb",
        "input_stream_TLAST": f"{input_prefix}_tlast",
        "input_stream_TVALID": f"{input_prefix}_tvalid",
        "input_stream_TREADY": f"{input_prefix}_tready",
        "output_stream_TDATA": f"{output_prefix}_tdata",
        "output_stream_TKEEP": f"{output_prefix}_tkeep",
        "output_stream_TSTRB": f"{output_prefix}_tstrb",
        "output_stream_TLAST": f"{output_prefix}_tlast",
        "output_stream_TVALID": f"{output_prefix}_tvalid",
        "output_stream_TREADY": f"{output_prefix}_tready",
    }
    for port, signal in axis.items():
        connect(port, signal)

    for kind in ("weights", "biases"):
        iface = spec["param_interfaces"][kind]
        if iface["mode"] == "absent":
            continue
        if iface["mode"] == "scalar":
            connect(kind, f"{role}_{kind}_scalar")
            continue
        prefixes = [kind] if iface["mode"] == "bram" else [bank["prefix"] for bank in iface["banks"]]
        for prefix in prefixes:
            for suffix in (
                "Addr_A",
                "EN_A",
                "WEN_A",
                "Din_A",
                "Dout_A",
                "Clk_A",
                "Rst_A",
                "Addr_B",
                "EN_B",
                "WEN_B",
                "Din_B",
                "Dout_B",
                "Clk_B",
                "Rst_B",
            ):
                connect(f"{prefix}_{suffix}", suffix_signal(prefix, suffix, role))

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append(");")
    return "\n".join(lines)


def axis_wire_decl(prefix: str, data_width: int, keep_width: int) -> str:
    return "\n".join(
        [
            f"wire [{data_width - 1}:0] {prefix}_tdata;",
            f"wire [{keep_width - 1}:0] {prefix}_tkeep;",
            f"wire [{keep_width - 1}:0] {prefix}_tstrb;",
            f"wire {prefix}_tlast;",
            f"wire {prefix}_tvalid;",
            f"wire {prefix}_tready;",
        ]
    )


def axil_wires_and_ctrl(spec: dict[str, Any]) -> str:
    role = spec["role"]
    addr_width = int(spec["top_ports"]["s_axi_control_AWADDR"])
    data_width = int(spec["top_ports"]["s_axi_control_WDATA"])
    return f"""
wire {role}_axil_done;
wire {role}_axil_busy;
wire {role}_axil_error;
wire {role}_s_axi_control_AWVALID;
wire {role}_s_axi_control_AWREADY;
wire [{addr_width - 1}:0] {role}_s_axi_control_AWADDR;
wire {role}_s_axi_control_WVALID;
wire {role}_s_axi_control_WREADY;
wire [{data_width - 1}:0] {role}_s_axi_control_WDATA;
wire [{(data_width // 8) - 1}:0] {role}_s_axi_control_WSTRB;
wire {role}_s_axi_control_ARVALID;
wire {role}_s_axi_control_ARREADY;
wire [{addr_width - 1}:0] {role}_s_axi_control_ARADDR;
wire {role}_s_axi_control_RVALID;
wire {role}_s_axi_control_RREADY;
wire [{data_width - 1}:0] {role}_s_axi_control_RDATA;
wire [1:0] {role}_s_axi_control_RRESP;
wire {role}_s_axi_control_BVALID;
wire {role}_s_axi_control_BREADY;
wire [1:0] {role}_s_axi_control_BRESP;
wire {role}_interrupt;

axi_lite_kernel_ctrl #(
    .ADDR_WIDTH({addr_width}),
    .DATA_WIDTH({data_width})
) u_{role}_axi_lite_kernel_ctrl (
    .clk(clk_main),
    .rst_n(rst_n),
    .start(start_pulse),
    .done({role}_axil_done),
    .busy({role}_axil_busy),
    .error({role}_axil_error),
    .m_axi_awvalid({role}_s_axi_control_AWVALID),
    .m_axi_awready({role}_s_axi_control_AWREADY),
    .m_axi_awaddr({role}_s_axi_control_AWADDR),
    .m_axi_wvalid({role}_s_axi_control_WVALID),
    .m_axi_wready({role}_s_axi_control_WREADY),
    .m_axi_wdata({role}_s_axi_control_WDATA),
    .m_axi_wstrb({role}_s_axi_control_WSTRB),
    .m_axi_bvalid({role}_s_axi_control_BVALID),
    .m_axi_bready({role}_s_axi_control_BREADY),
    .m_axi_bresp({role}_s_axi_control_BRESP),
    .m_axi_arvalid({role}_s_axi_control_ARVALID),
    .m_axi_arready({role}_s_axi_control_ARREADY),
    .m_axi_araddr({role}_s_axi_control_ARADDR),
    .m_axi_rvalid({role}_s_axi_control_RVALID),
    .m_axi_rready({role}_s_axi_control_RREADY),
    .m_axi_rdata({role}_s_axi_control_RDATA),
    .m_axi_rresp({role}_s_axi_control_RRESP)
);
""".strip()


def axis_fifo_module_text(fifo_style: str = "lut") -> str:
    if fifo_style == "lut":
        return axis_lut_fifo_module_text()
    if fifo_style == "lut_registered":
        return axis_registered_lut_fifo_module_text()
    if fifo_style == "bram":
        return axis_bram_fifo_module_text()
    raise ValueError(f"Unsupported FIFO style: {fifo_style}")


def axis_width_adapter_module_text() -> str:
    return r"""`timescale 1ns / 1ps

module axis_width_adapter #(
    parameter integer S_DATA_WIDTH = 128,
    parameter integer M_DATA_WIDTH = 128,
    parameter integer S_KEEP_WIDTH = S_DATA_WIDTH / 8,
    parameter integer M_KEEP_WIDTH = M_DATA_WIDTH / 8
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [S_DATA_WIDTH-1:0]   s_tdata,
    input  wire [S_KEEP_WIDTH-1:0]   s_tkeep,
    input  wire [S_KEEP_WIDTH-1:0]   s_tstrb,
    input  wire                      s_tlast,
    input  wire                      s_tvalid,
    output wire                      s_tready,
    output wire [M_DATA_WIDTH-1:0]   m_tdata,
    output wire [M_KEEP_WIDTH-1:0]   m_tkeep,
    output wire [M_KEEP_WIDTH-1:0]   m_tstrb,
    output wire                      m_tlast,
    output wire                      m_tvalid,
    input  wire                      m_tready
);

generate
if (S_DATA_WIDTH == M_DATA_WIDTH) begin : g_same_width
    assign s_tready = m_tready;
    assign m_tdata = s_tdata;
    assign m_tkeep = s_tkeep;
    assign m_tstrb = s_tstrb;
    assign m_tlast = s_tlast;
    assign m_tvalid = s_tvalid;
end else if (S_DATA_WIDTH > M_DATA_WIDTH) begin : g_downsize
    localparam integer RATIO = S_DATA_WIDTH / M_DATA_WIDTH;
    localparam integer RATIO_BITS = (RATIO <= 2) ? 1 : $clog2(RATIO);

    reg [S_DATA_WIDTH-1:0] data_reg = {S_DATA_WIDTH{1'b0}};
    reg [S_KEEP_WIDTH-1:0] keep_reg = {S_KEEP_WIDTH{1'b0}};
    reg [S_KEEP_WIDTH-1:0] strb_reg = {S_KEEP_WIDTH{1'b0}};
    reg last_reg = 1'b0;
    reg valid_reg = 1'b0;
    reg [RATIO_BITS-1:0] idx_reg = {RATIO_BITS{1'b0}};

    assign s_tready = !valid_reg;
    assign m_tvalid = valid_reg;
    assign m_tdata = data_reg[idx_reg * M_DATA_WIDTH +: M_DATA_WIDTH];
    assign m_tkeep = keep_reg[idx_reg * M_KEEP_WIDTH +: M_KEEP_WIDTH];
    assign m_tstrb = strb_reg[idx_reg * M_KEEP_WIDTH +: M_KEEP_WIDTH];
    assign m_tlast = last_reg && (idx_reg == RATIO - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_reg <= {S_DATA_WIDTH{1'b0}};
            keep_reg <= {S_KEEP_WIDTH{1'b0}};
            strb_reg <= {S_KEEP_WIDTH{1'b0}};
            last_reg <= 1'b0;
            valid_reg <= 1'b0;
            idx_reg <= {RATIO_BITS{1'b0}};
        end else begin
            if (!valid_reg && s_tvalid) begin
                data_reg <= s_tdata;
                keep_reg <= s_tkeep;
                strb_reg <= s_tstrb;
                last_reg <= s_tlast;
                valid_reg <= 1'b1;
                idx_reg <= {RATIO_BITS{1'b0}};
            end else if (valid_reg && m_tready) begin
                if (idx_reg == RATIO - 1) begin
                    valid_reg <= 1'b0;
                    idx_reg <= {RATIO_BITS{1'b0}};
                end else begin
                    idx_reg <= idx_reg + {{(RATIO_BITS-1){1'b0}}, 1'b1};
                end
            end
        end
    end
end else begin : g_upsize
    localparam integer RATIO = M_DATA_WIDTH / S_DATA_WIDTH;
    localparam integer RATIO_BITS = (RATIO <= 2) ? 1 : $clog2(RATIO);

    reg [M_DATA_WIDTH-1:0] acc_data = {M_DATA_WIDTH{1'b0}};
    reg [M_KEEP_WIDTH-1:0] acc_keep = {M_KEEP_WIDTH{1'b0}};
    reg [M_KEEP_WIDTH-1:0] acc_strb = {M_KEEP_WIDTH{1'b0}};
    reg [RATIO_BITS-1:0] idx_reg = {RATIO_BITS{1'b0}};
    reg [M_DATA_WIDTH-1:0] out_data = {M_DATA_WIDTH{1'b0}};
    reg [M_KEEP_WIDTH-1:0] out_keep = {M_KEEP_WIDTH{1'b0}};
    reg [M_KEEP_WIDTH-1:0] out_strb = {M_KEEP_WIDTH{1'b0}};
    reg out_last = 1'b0;
    reg out_valid = 1'b0;

    wire [M_DATA_WIDTH-1:0] shifted_data = {{(M_DATA_WIDTH-S_DATA_WIDTH){1'b0}}, s_tdata} << (idx_reg * S_DATA_WIDTH);
    wire [M_KEEP_WIDTH-1:0] shifted_keep = {{(M_KEEP_WIDTH-S_KEEP_WIDTH){1'b0}}, s_tkeep} << (idx_reg * S_KEEP_WIDTH);
    wire [M_KEEP_WIDTH-1:0] shifted_strb = {{(M_KEEP_WIDTH-S_KEEP_WIDTH){1'b0}}, s_tstrb} << (idx_reg * S_KEEP_WIDTH);
    wire [M_DATA_WIDTH-1:0] next_data = acc_data | shifted_data;
    wire [M_KEEP_WIDTH-1:0] next_keep = acc_keep | shifted_keep;
    wire [M_KEEP_WIDTH-1:0] next_strb = acc_strb | shifted_strb;
    wire complete_word = (idx_reg == RATIO - 1) || s_tlast;

    assign s_tready = !out_valid;
    assign m_tvalid = out_valid;
    assign m_tdata = out_data;
    assign m_tkeep = out_keep;
    assign m_tstrb = out_strb;
    assign m_tlast = out_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_data <= {M_DATA_WIDTH{1'b0}};
            acc_keep <= {M_KEEP_WIDTH{1'b0}};
            acc_strb <= {M_KEEP_WIDTH{1'b0}};
            idx_reg <= {RATIO_BITS{1'b0}};
            out_data <= {M_DATA_WIDTH{1'b0}};
            out_keep <= {M_KEEP_WIDTH{1'b0}};
            out_strb <= {M_KEEP_WIDTH{1'b0}};
            out_last <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            if (out_valid) begin
                if (m_tready) begin
                    out_valid <= 1'b0;
                end
            end else if (s_tvalid) begin
                if (complete_word) begin
                    out_data <= next_data;
                    out_keep <= next_keep;
                    out_strb <= next_strb;
                    out_last <= s_tlast;
                    out_valid <= 1'b1;
                    acc_data <= {M_DATA_WIDTH{1'b0}};
                    acc_keep <= {M_KEEP_WIDTH{1'b0}};
                    acc_strb <= {M_KEEP_WIDTH{1'b0}};
                    idx_reg <= {RATIO_BITS{1'b0}};
                end else begin
                    acc_data <= next_data;
                    acc_keep <= next_keep;
                    acc_strb <= next_strb;
                    idx_reg <= idx_reg + {{(RATIO_BITS-1){1'b0}}, 1'b1};
                end
            end
        end
    end
end
endgenerate

endmodule
"""


def axis_bram_fifo_module_text() -> str:
    return r"""`timescale 1ns / 1ps

module axis_simple_fifo #(
    parameter integer DATA_WIDTH = 128,
    parameter integer KEEP_WIDTH = DATA_WIDTH / 8,
    parameter integer DEPTH = 512
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [DATA_WIDTH-1:0] s_tdata,
    input  wire [KEEP_WIDTH-1:0] s_tkeep,
    input  wire [KEEP_WIDTH-1:0] s_tstrb,
    input  wire                  s_tlast,
    input  wire                  s_tvalid,
    output wire                  s_tready,
    output wire [DATA_WIDTH-1:0] m_tdata,
    output wire [KEEP_WIDTH-1:0] m_tkeep,
    output wire [KEEP_WIDTH-1:0] m_tstrb,
    output wire                  m_tlast,
    output wire                  m_tvalid,
    input  wire                  m_tready
);

localparam integer ADDR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH);
localparam integer WORD_WIDTH = DATA_WIDTH + (2 * KEEP_WIDTH) + 1;

(* ram_style = "block" *) reg [WORD_WIDTH-1:0] mem [0:DEPTH-1];
reg [ADDR_WIDTH-1:0] wr_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH-1:0] rd_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH:0] count = {(ADDR_WIDTH+1){1'b0}};
reg [WORD_WIDTH-1:0] out_word = {WORD_WIDTH{1'b0}};
reg out_valid = 1'b0;

assign s_tready = (count < DEPTH);
assign m_tvalid = out_valid;
assign m_tlast = out_word[0];
assign m_tstrb = out_word[KEEP_WIDTH:1];
assign m_tkeep = out_word[(2*KEEP_WIDTH):KEEP_WIDTH+1];
assign m_tdata = out_word[WORD_WIDTH-1:(2*KEEP_WIDTH)+1];

wire [WORD_WIDTH-1:0] s_word = {s_tdata, s_tkeep, s_tstrb, s_tlast};

wire push = s_tvalid && s_tready;
wire pop = out_valid && m_tready;
wire can_load_output = !out_valid || pop;
wire load_output = can_load_output && (count != 0);

function [ADDR_WIDTH-1:0] next_ptr;
    input [ADDR_WIDTH-1:0] ptr;
    begin
        if (ptr == DEPTH - 1) begin
            next_ptr = {ADDR_WIDTH{1'b0}};
        end else begin
            next_ptr = ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
        end
    end
endfunction

always @(posedge clk) begin
    if (!rst_n) begin
        wr_ptr <= {ADDR_WIDTH{1'b0}};
        rd_ptr <= {ADDR_WIDTH{1'b0}};
        count <= {(ADDR_WIDTH+1){1'b0}};
        out_word <= {WORD_WIDTH{1'b0}};
        out_valid <= 1'b0;
    end else begin
        if (push) begin
            mem[wr_ptr] <= s_word;
            wr_ptr <= next_ptr(wr_ptr);
        end
        if (load_output) begin
            out_word <= mem[rd_ptr];
            rd_ptr <= next_ptr(rd_ptr);
            out_valid <= 1'b1;
        end else if (pop) begin
            out_valid <= 1'b0;
        end
        case ({push, load_output})
            2'b10: count <= count + {{ADDR_WIDTH{1'b0}}, 1'b1};
            2'b01: count <= count - {{ADDR_WIDTH{1'b0}}, 1'b1};
            default: count <= count;
        endcase
    end
end

endmodule
"""


def axis_lut_fifo_module_text() -> str:
    return r"""`timescale 1ns / 1ps

module axis_simple_fifo #(
    parameter integer DATA_WIDTH = 128,
    parameter integer KEEP_WIDTH = DATA_WIDTH / 8,
    parameter integer DEPTH = 512
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [DATA_WIDTH-1:0] s_tdata,
    input  wire [KEEP_WIDTH-1:0] s_tkeep,
    input  wire [KEEP_WIDTH-1:0] s_tstrb,
    input  wire                  s_tlast,
    input  wire                  s_tvalid,
    output wire                  s_tready,
    output wire [DATA_WIDTH-1:0] m_tdata,
    output wire [KEEP_WIDTH-1:0] m_tkeep,
    output wire [KEEP_WIDTH-1:0] m_tstrb,
    output wire                  m_tlast,
    output wire                  m_tvalid,
    input  wire                  m_tready
);

localparam integer ADDR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH);

reg [DATA_WIDTH-1:0] data_mem [0:DEPTH-1];
reg [KEEP_WIDTH-1:0] keep_mem [0:DEPTH-1];
reg [KEEP_WIDTH-1:0] strb_mem [0:DEPTH-1];
reg last_mem [0:DEPTH-1];
reg [ADDR_WIDTH-1:0] wr_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH-1:0] rd_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH:0] count = {(ADDR_WIDTH+1){1'b0}};

assign s_tready = (count < DEPTH);
assign m_tvalid = (count != 0);
assign m_tdata = data_mem[rd_ptr];
assign m_tkeep = keep_mem[rd_ptr];
assign m_tstrb = strb_mem[rd_ptr];
assign m_tlast = last_mem[rd_ptr];

wire push = s_tvalid && s_tready;
wire pop = m_tvalid && m_tready;

always @(posedge clk) begin
    if (!rst_n) begin
        wr_ptr <= {ADDR_WIDTH{1'b0}};
        rd_ptr <= {ADDR_WIDTH{1'b0}};
        count <= {(ADDR_WIDTH+1){1'b0}};
    end else begin
        if (push) begin
            data_mem[wr_ptr] <= s_tdata;
            keep_mem[wr_ptr] <= s_tkeep;
            strb_mem[wr_ptr] <= s_tstrb;
            last_mem[wr_ptr] <= s_tlast;
            wr_ptr <= wr_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
        end
        if (pop) begin
            rd_ptr <= rd_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
        end
        case ({push, pop})
            2'b10: count <= count + {{ADDR_WIDTH{1'b0}}, 1'b1};
            2'b01: count <= count - {{ADDR_WIDTH{1'b0}}, 1'b1};
            default: count <= count;
        endcase
    end
end

endmodule
"""


def axis_registered_lut_fifo_module_text() -> str:
    return r"""`timescale 1ns / 1ps

module axis_simple_fifo #(
    parameter integer DATA_WIDTH = 128,
    parameter integer KEEP_WIDTH = DATA_WIDTH / 8,
    parameter integer DEPTH = 512
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [DATA_WIDTH-1:0] s_tdata,
    input  wire [KEEP_WIDTH-1:0] s_tkeep,
    input  wire [KEEP_WIDTH-1:0] s_tstrb,
    input  wire                  s_tlast,
    input  wire                  s_tvalid,
    output wire                  s_tready,
    output wire [DATA_WIDTH-1:0] m_tdata,
    output wire [KEEP_WIDTH-1:0] m_tkeep,
    output wire [KEEP_WIDTH-1:0] m_tstrb,
    output wire                  m_tlast,
    output wire                  m_tvalid,
    input  wire                  m_tready
);

localparam integer ADDR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH);

reg [DATA_WIDTH-1:0] data_mem [0:DEPTH-1];
reg [KEEP_WIDTH-1:0] keep_mem [0:DEPTH-1];
reg [KEEP_WIDTH-1:0] strb_mem [0:DEPTH-1];
reg last_mem [0:DEPTH-1];
reg [ADDR_WIDTH-1:0] wr_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH-1:0] rd_ptr = {ADDR_WIDTH{1'b0}};
reg [ADDR_WIDTH:0] count = {(ADDR_WIDTH+1){1'b0}};
reg [DATA_WIDTH-1:0] out_tdata = {DATA_WIDTH{1'b0}};
reg [KEEP_WIDTH-1:0] out_tkeep = {KEEP_WIDTH{1'b0}};
reg [KEEP_WIDTH-1:0] out_tstrb = {KEEP_WIDTH{1'b0}};
reg out_tlast = 1'b0;
reg out_valid = 1'b0;

assign s_tready = (count < DEPTH);
assign m_tvalid = out_valid;
assign m_tdata = out_tdata;
assign m_tkeep = out_tkeep;
assign m_tstrb = out_tstrb;
assign m_tlast = out_tlast;

wire push = s_tvalid && s_tready;
wire pop = out_valid && m_tready;
wire can_load_output = !out_valid || pop;
wire load_output = can_load_output && (count != 0);

always @(posedge clk) begin
    if (!rst_n) begin
        wr_ptr <= {ADDR_WIDTH{1'b0}};
        rd_ptr <= {ADDR_WIDTH{1'b0}};
        count <= {(ADDR_WIDTH+1){1'b0}};
        out_tdata <= {DATA_WIDTH{1'b0}};
        out_tkeep <= {KEEP_WIDTH{1'b0}};
        out_tstrb <= {KEEP_WIDTH{1'b0}};
        out_tlast <= 1'b0;
        out_valid <= 1'b0;
    end else begin
        if (push) begin
            data_mem[wr_ptr] <= s_tdata;
            keep_mem[wr_ptr] <= s_tkeep;
            strb_mem[wr_ptr] <= s_tstrb;
            last_mem[wr_ptr] <= s_tlast;
            wr_ptr <= wr_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
        end
        if (load_output) begin
            out_tdata <= data_mem[rd_ptr];
            out_tkeep <= keep_mem[rd_ptr];
            out_tstrb <= strb_mem[rd_ptr];
            out_tlast <= last_mem[rd_ptr];
            rd_ptr <= rd_ptr + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
            out_valid <= 1'b1;
        end else if (pop) begin
            out_valid <= 1'b0;
        end
        case ({push, load_output})
            2'b10: count <= count + {{ADDR_WIDTH{1'b0}}, 1'b1};
            2'b01: count <= count - {{ADDR_WIDTH{1'b0}}, 1'b1};
            default: count <= count;
        endcase
    end
end

endmodule
"""


def resetless_simple_dual_port_bram_module_text() -> str:
    return r"""module simple_dual_port_bram #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ADDR_WIDTH = 8,
    parameter integer DEPTH = 256,
    parameter INIT_FILE = ""
) (
    input  wire                         clka,
    input  wire                         rsta,
    input  wire                         ena,
    input  wire [(DATA_WIDTH/8)-1:0]    wea,
    input  wire [ADDR_WIDTH-1:0]        addra,
    input  wire [DATA_WIDTH-1:0]        dina,
    output reg  [DATA_WIDTH-1:0]        douta,
    input  wire                         clkb,
    input  wire                         rstb,
    input  wire                         enb,
    input  wire [(DATA_WIDTH/8)-1:0]    web,
    input  wire [ADDR_WIDTH-1:0]        addrb,
    input  wire [DATA_WIDTH-1:0]        dinb,
    output reg  [DATA_WIDTH-1:0]        doutb
);

reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
integer i;
integer byte_idx;

initial begin
    douta = {DATA_WIDTH{1'b0}};
    doutb = {DATA_WIDTH{1'b0}};
    for (i = 0; i < DEPTH; i = i + 1) begin
        mem[i] = {DATA_WIDTH{1'b0}};
    end
    if (INIT_FILE != "") begin
        $readmemh(INIT_FILE, mem);
    end
end

always @(posedge clka) begin
    if (ena) begin
        if (addra < DEPTH) begin
            for (byte_idx = 0; byte_idx < (DATA_WIDTH/8); byte_idx = byte_idx + 1) begin
                if (wea[byte_idx]) begin
                    mem[addra][byte_idx*8 +: 8] <= dina[byte_idx*8 +: 8];
                end
            end
            douta <= mem[addra];
        end else begin
            douta <= {DATA_WIDTH{1'b0}};
        end
    end
end

always @(posedge clkb) begin
    if (enb) begin
        if (addrb < DEPTH) begin
            for (byte_idx = 0; byte_idx < (DATA_WIDTH/8); byte_idx = byte_idx + 1) begin
                if (web[byte_idx]) begin
                    mem[addrb][byte_idx*8 +: 8] <= dinb[byte_idx*8 +: 8];
                end
            end
            doutb <= mem[addrb];
        end else begin
            doutb <= {DATA_WIDTH{1'b0}};
        end
    end
end

endmodule
"""


def build_component_spec(role: str, case_name: str, case_dir: Path, rtl_dst: Path) -> dict[str, Any]:
    module = top_module_name(case_dir)
    renamed_module = rename_and_copy_rtl(export_rtl_dir(case_dir), rtl_dst, role, module)
    cpp_text = first_cpp(case_dir).read_text(encoding="utf-8", errors="replace")
    constants = gh.parse_constants(cpp_text)
    case_meta = gh.compute_case_meta(case_name, constants)
    top_ports = gh.parse_top_ports(top_verilog_path(case_dir, module).read_text(encoding="utf-8", errors="replace"))
    param_interfaces = {
        "weights": gh.detect_param_interface(top_ports, "weights"),
        "biases": gh.detect_param_interface(top_ports, "biases"),
    }
    return {
        "role": role,
        "case_name": case_name,
        "case_dir": str(case_dir),
        "module_name": module,
        "renamed_module": renamed_module,
        "constants": constants,
        "case_meta": case_meta,
        "top_ports": top_ports,
        "param_interfaces": param_interfaces,
    }


def write_constraints(project_root: Path, board_config: Path) -> None:
    board_cfg = load_yaml(board_config)
    clock_cfg = board_cfg["clock"]
    led_cfg = board_cfg.get("status_led", {})
    led_constraints = []
    for idx, pin in enumerate(led_cfg.get("package_pins", [])):
        line = gh.make_constraint_line(f"status_led[{idx}]", pin, led_cfg.get("iostandard"))
        if line:
            led_constraints.append(line)
    if "package_pin_p" in clock_cfg:
        clock_constraints = [
            gh.make_constraint_line(clock_cfg["port_name_p"], clock_cfg.get("package_pin_p"), clock_cfg.get("iostandard")),
            gh.make_constraint_line(clock_cfg["port_name_n"], clock_cfg.get("package_pin_n"), clock_cfg.get("iostandard")),
        ]
        clock_create_port = clock_cfg["port_name_p"]
    else:
        clock_constraints = [
            gh.make_constraint_line(clock_cfg["port_name"], clock_cfg.get("package_pin"), clock_cfg.get("iostandard"))
        ]
        clock_create_port = clock_cfg["port_name"]
    xdc_text = gh.render_template(
        gh.TEMPLATES_DIR / "harness_constraints.xdc.tmpl",
        {
            "CLOCK_PERIOD_NS": str(gh.constraint_clock_period_ns(clock_cfg)),
            "CLOCK_CREATE_PORT": str(clock_create_port),
            "CLOCK_PIN_CONSTRAINTS": "\n".join([line for line in clock_constraints if line]),
            "RESET_PIN_CONSTRAINT": gh.make_constraint_line(
                board_cfg["reset"]["port_name"],
                board_cfg["reset"].get("package_pin"),
                board_cfg["reset"].get("iostandard"),
            ),
            "UART_RX_PIN_CONSTRAINT": gh.make_constraint_line(
                board_cfg["uart"].get("rx_port_name", "uart_rx"),
                board_cfg["uart"].get("rx_package_pin"),
                board_cfg["uart"].get("iostandard"),
            ),
            "UART_TX_PIN_CONSTRAINT": gh.make_constraint_line(
                board_cfg["uart"]["tx_port_name"],
                board_cfg["uart"].get("package_pin"),
                board_cfg["uart"].get("iostandard"),
            ),
            "LED_PIN_CONSTRAINTS": "\n".join(led_constraints),
        },
    )
    (project_root / "constraints").mkdir(parents=True, exist_ok=True)
    (project_root / "constraints" / "harness.xdc").write_text(xdc_text, encoding="utf-8")


def write_stitched_top(
    *,
    project_root: Path,
    combo_case_name: str,
    specs: list[dict[str, Any]],
    board_config: Path,
    fifo_depth: int,
    fifo_style: str = "lut",
    param_reset_style: str = "kernel",
) -> None:
    board_cfg = load_yaml(board_config)
    defaults_cfg = load_yaml(BOARD_HARNESS_DIR / "configs" / "harness_defaults.yaml")
    data_dir = project_root / "data"
    rtl_dir = project_root / "rtl"
    data_dir.mkdir(parents=True, exist_ok=True)
    rtl_dir.mkdir(parents=True, exist_ok=True)
    gh.copy_tree_files(BOARD_HARNESS_DIR / "modules", rtl_dir)
    (rtl_dir / "axis_simple_fifo.v").write_text(axis_fifo_module_text(fifo_style), encoding="utf-8")
    (rtl_dir / "axis_width_adapter.v").write_text(axis_width_adapter_module_text(), encoding="utf-8")
    if param_reset_style == "const_zero_all":
        (rtl_dir / "simple_dual_port_bram.v").write_text(
            resetless_simple_dual_port_bram_module_text(),
            encoding="utf-8",
        )

    expand, dw, project = specs
    input_width = int(expand["top_ports"]["input_stream_TDATA"])
    input_keep = int(expand["top_ports"].get("input_stream_TKEEP", max(1, input_width // 8)))
    mid1_width = int(expand["top_ports"]["output_stream_TDATA"])
    mid1_keep = int(expand["top_ports"].get("output_stream_TKEEP", max(1, mid1_width // 8)))
    dw_input_width = int(dw["top_ports"]["input_stream_TDATA"])
    dw_input_keep = int(dw["top_ports"].get("input_stream_TKEEP", max(1, dw_input_width // 8)))
    dw_output_width = int(dw["top_ports"]["output_stream_TDATA"])
    mid2_width = dw_output_width
    mid2_keep = int(dw["top_ports"].get("output_stream_TKEEP", max(1, mid2_width // 8)))
    project_input_width = int(project["top_ports"]["input_stream_TDATA"])
    project_input_keep = int(project["top_ports"].get("input_stream_TKEEP", max(1, project_input_width // 8)))
    output_width = int(project["top_ports"]["output_stream_TDATA"])
    output_keep = int(project["top_ports"].get("output_stream_TKEEP", max(1, output_width // 8)))
    if max(mid1_width, dw_input_width) % min(mid1_width, dw_input_width) != 0:
        raise ValueError(
            f"expand->dw AXIS widths must be integer multiples: expand_out={mid1_width}, dw_in={dw_input_width}"
        )
    if max(mid2_width, project_input_width) % min(mid2_width, project_input_width) != 0:
        raise ValueError(
            f"dw->project AXIS widths must be integer multiples: dw_out={mid2_width}, project_in={project_input_width}"
        )

    input_word_count = int(expand["case_meta"]["input_word_count"])
    output_word_count = int(project["case_meta"]["output_word_count"])
    input_mem_paths = gh.generate_stream_mem_files(data_dir, "input_stream", input_word_count, input_width)
    input_word_mem = gh.generate_stream_word_mem_file(data_dir, "input_stream_words", input_word_count, input_width)
    input_mem_mapping = {f"INPUT_MEM_FILE_{lane_idx:02d}": '""' for lane_idx in range(32)}
    for lane_idx, mem_path in enumerate(input_mem_paths):
        input_mem_mapping[f"INPUT_MEM_FILE_{lane_idx:02d}"] = f'"../data/{mem_path.name}"'

    param_logic = "\n\n".join(build_param_logic_for_spec(spec, data_dir, param_reset_style) for spec in specs)
    axil_blocks = "\n\n".join(axil_wires_and_ctrl(spec) for spec in specs)
    kernel_instances = "\n\n".join(
        [
            build_kernel_instance_for_spec(expand, "input", "expand_out"),
            build_kernel_instance_for_spec(dw, "fifo1_out", "dw_out"),
            build_kernel_instance_for_spec(project, "fifo2_out", "output"),
        ]
    )
    status_led_width = int(board_cfg.get("status_led", {}).get("width", 2))
    top = f"""`timescale 1ns / 1ps

module harness_top (
    input  wire sys_clk_p,
    input  wire sys_clk_n,
    input  wire sys_rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire [{status_led_width - 1}:0] status_led
);

localparam integer CLK_FREQ_HZ = {int(board_cfg["clock"]["freq_hz"])};
localparam integer UART_BAUD = {int(board_cfg.get("uart", {}).get("baud", defaults_cfg["uart_baud"]))};
localparam integer BOOT_DELAY_CYCLES = {int(defaults_cfg["boot_delay_cycles"])};
localparam integer STATUS_LED_WIDTH = {status_led_width};
localparam integer INPUT_TDATA_WIDTH = {input_width};
localparam integer INPUT_KEEP_WIDTH = {input_keep};
localparam integer OUTPUT_TDATA_WIDTH = {output_width};
localparam integer OUTPUT_KEEP_WIDTH = {output_keep};
localparam integer INPUT_WORD_COUNT = {input_word_count};
localparam integer OUTPUT_WORD_COUNT = {output_word_count};
localparam integer FIFO_DEPTH = {fifo_depth};

{gh.build_clocking_logic(board_cfg)}

wire rst_n;
reset_sync u_reset_sync (.clk(clk_main), .rst_n_async(sys_rst_n & clock_ready), .rst_n_sync(rst_n));

reg [31:0] boot_counter = 32'd0;
reg boot_started = 1'b0;
wire start_pulse = (boot_counter == BOOT_DELAY_CYCLES - 1) && !boot_started;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        boot_counter <= 32'd0;
        boot_started <= 1'b0;
    end else if (!boot_started) begin
        if (boot_counter == BOOT_DELAY_CYCLES - 1) begin
            boot_started <= 1'b1;
        end else begin
            boot_counter <= boot_counter + 32'd1;
        end
    end
end

{axis_wire_decl("input", input_width, input_keep)}
wire source_done;
axis_rom_source #(
    .DATA_WIDTH(INPUT_TDATA_WIDTH),
    .KEEP_WIDTH(INPUT_KEEP_WIDTH),
    .WORD_COUNT(INPUT_WORD_COUNT),
    .WORD_MEM_FILE("../data/{input_word_mem.name}"),
    .MEM_FILE_00({input_mem_mapping["INPUT_MEM_FILE_00"]}),
    .MEM_FILE_01({input_mem_mapping["INPUT_MEM_FILE_01"]}),
    .MEM_FILE_02({input_mem_mapping["INPUT_MEM_FILE_02"]}),
    .MEM_FILE_03({input_mem_mapping["INPUT_MEM_FILE_03"]}),
    .MEM_FILE_04({input_mem_mapping["INPUT_MEM_FILE_04"]}),
    .MEM_FILE_05({input_mem_mapping["INPUT_MEM_FILE_05"]}),
    .MEM_FILE_06({input_mem_mapping["INPUT_MEM_FILE_06"]}),
    .MEM_FILE_07({input_mem_mapping["INPUT_MEM_FILE_07"]}),
    .MEM_FILE_08({input_mem_mapping["INPUT_MEM_FILE_08"]}),
    .MEM_FILE_09({input_mem_mapping["INPUT_MEM_FILE_09"]}),
    .MEM_FILE_10({input_mem_mapping["INPUT_MEM_FILE_10"]}),
    .MEM_FILE_11({input_mem_mapping["INPUT_MEM_FILE_11"]}),
    .MEM_FILE_12({input_mem_mapping["INPUT_MEM_FILE_12"]}),
    .MEM_FILE_13({input_mem_mapping["INPUT_MEM_FILE_13"]}),
    .MEM_FILE_14({input_mem_mapping["INPUT_MEM_FILE_14"]}),
    .MEM_FILE_15({input_mem_mapping["INPUT_MEM_FILE_15"]}),
    .MEM_FILE_16({input_mem_mapping["INPUT_MEM_FILE_16"]}),
    .MEM_FILE_17({input_mem_mapping["INPUT_MEM_FILE_17"]}),
    .MEM_FILE_18({input_mem_mapping["INPUT_MEM_FILE_18"]}),
    .MEM_FILE_19({input_mem_mapping["INPUT_MEM_FILE_19"]}),
    .MEM_FILE_20({input_mem_mapping["INPUT_MEM_FILE_20"]}),
    .MEM_FILE_21({input_mem_mapping["INPUT_MEM_FILE_21"]}),
    .MEM_FILE_22({input_mem_mapping["INPUT_MEM_FILE_22"]}),
    .MEM_FILE_23({input_mem_mapping["INPUT_MEM_FILE_23"]}),
    .MEM_FILE_24({input_mem_mapping["INPUT_MEM_FILE_24"]}),
    .MEM_FILE_25({input_mem_mapping["INPUT_MEM_FILE_25"]}),
    .MEM_FILE_26({input_mem_mapping["INPUT_MEM_FILE_26"]}),
    .MEM_FILE_27({input_mem_mapping["INPUT_MEM_FILE_27"]}),
    .MEM_FILE_28({input_mem_mapping["INPUT_MEM_FILE_28"]}),
    .MEM_FILE_29({input_mem_mapping["INPUT_MEM_FILE_29"]}),
    .MEM_FILE_30({input_mem_mapping["INPUT_MEM_FILE_30"]}),
    .MEM_FILE_31({input_mem_mapping["INPUT_MEM_FILE_31"]})
) u_axis_rom_source (
    .clk(clk_main), .rst_n(rst_n), .start(start_pulse),
    .tdata(input_tdata), .tkeep(input_tkeep), .tstrb(input_tstrb), .tlast(input_tlast),
    .tvalid(input_tvalid), .tready(input_tready), .done(source_done)
);

{axis_wire_decl("expand_out", mid1_width, mid1_keep)}
{axis_wire_decl("repack1_out", dw_input_width, dw_input_keep)}
{axis_wire_decl("fifo1_out", dw_input_width, dw_input_keep)}
{axis_wire_decl("dw_out", mid2_width, mid2_keep)}
{axis_wire_decl("repack2_out", project_input_width, project_input_keep)}
{axis_wire_decl("fifo2_out", project_input_width, project_input_keep)}
{axis_wire_decl("output", output_width, output_keep)}

axis_width_adapter #(
    .S_DATA_WIDTH({mid1_width}),
    .M_DATA_WIDTH({dw_input_width}),
    .S_KEEP_WIDTH({mid1_keep}),
    .M_KEEP_WIDTH({dw_input_keep})
) u_repack_expand_to_dw (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(expand_out_tdata), .s_tkeep(expand_out_tkeep), .s_tstrb(expand_out_tstrb), .s_tlast(expand_out_tlast),
    .s_tvalid(expand_out_tvalid), .s_tready(expand_out_tready),
    .m_tdata(repack1_out_tdata), .m_tkeep(repack1_out_tkeep), .m_tstrb(repack1_out_tstrb), .m_tlast(repack1_out_tlast),
    .m_tvalid(repack1_out_tvalid), .m_tready(repack1_out_tready)
);

axis_simple_fifo #(.DATA_WIDTH({dw_input_width}), .KEEP_WIDTH({dw_input_keep}), .DEPTH(FIFO_DEPTH)) u_fifo_expand_to_dw (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(repack1_out_tdata), .s_tkeep(repack1_out_tkeep), .s_tstrb(repack1_out_tstrb), .s_tlast(repack1_out_tlast),
    .s_tvalid(repack1_out_tvalid), .s_tready(repack1_out_tready),
    .m_tdata(fifo1_out_tdata), .m_tkeep(fifo1_out_tkeep), .m_tstrb(fifo1_out_tstrb), .m_tlast(fifo1_out_tlast),
    .m_tvalid(fifo1_out_tvalid), .m_tready(fifo1_out_tready)
);

axis_width_adapter #(
    .S_DATA_WIDTH({mid2_width}),
    .M_DATA_WIDTH({project_input_width}),
    .S_KEEP_WIDTH({mid2_keep}),
    .M_KEEP_WIDTH({project_input_keep})
) u_repack_dw_to_project (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(dw_out_tdata), .s_tkeep(dw_out_tkeep), .s_tstrb(dw_out_tstrb), .s_tlast(dw_out_tlast),
    .s_tvalid(dw_out_tvalid), .s_tready(dw_out_tready),
    .m_tdata(repack2_out_tdata), .m_tkeep(repack2_out_tkeep), .m_tstrb(repack2_out_tstrb), .m_tlast(repack2_out_tlast),
    .m_tvalid(repack2_out_tvalid), .m_tready(repack2_out_tready)
);

axis_simple_fifo #(.DATA_WIDTH({project_input_width}), .KEEP_WIDTH({project_input_keep}), .DEPTH(FIFO_DEPTH)) u_fifo_dw_to_project (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(repack2_out_tdata), .s_tkeep(repack2_out_tkeep), .s_tstrb(repack2_out_tstrb), .s_tlast(repack2_out_tlast),
    .s_tvalid(repack2_out_tvalid), .s_tready(repack2_out_tready),
    .m_tdata(fifo2_out_tdata), .m_tkeep(fifo2_out_tkeep), .m_tstrb(fifo2_out_tstrb), .m_tlast(fifo2_out_tlast),
    .m_tvalid(fifo2_out_tvalid), .m_tready(fifo2_out_tready)
);

wire sink_done;
wire [31:0] output_checksum;
wire [31:0] output_word_count;
axis_checksum_sink #(
    .DATA_WIDTH(OUTPUT_TDATA_WIDTH),
    .KEEP_WIDTH(OUTPUT_KEEP_WIDTH),
    .EXPECTED_WORD_COUNT(OUTPUT_WORD_COUNT)
) u_axis_checksum_sink (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse),
    .tdata(output_tdata), .tkeep(output_tkeep), .tstrb(output_tstrb), .tlast(output_tlast),
    .tvalid(output_tvalid), .tready(output_tready), .done(sink_done),
    .checksum(output_checksum), .word_count(output_word_count)
);

{axil_blocks}

wire [31:0] measured_cycles;
wire counter_running;
wire counter_done;
cycle_counter u_cycle_counter (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse), .start(start_pulse), .stop(sink_done),
    .cycles(measured_cycles), .running(counter_running), .done(counter_done)
);

wire all_axil_done = pw_expand_axil_done && dw_axil_done && pw_project_axil_done;
wire any_axil_error = pw_expand_axil_error || dw_axil_error || pw_project_axil_error;
reg seen_axil_done = 1'b0;
reg seen_sink_done = 1'b0;
reg result_sent = 1'b0;
reg result_start = 1'b0;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        seen_axil_done <= 1'b0;
        seen_sink_done <= 1'b0;
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else if (start_pulse) begin
        seen_axil_done <= 1'b0;
        seen_sink_done <= 1'b0;
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else begin
        if (all_axil_done) seen_axil_done <= 1'b1;
        if (sink_done) seen_sink_done <= 1'b1;
        if (result_tx_done) result_sent <= 1'b1;
        result_start <= !result_sent && !result_tx_busy && ((seen_sink_done || sink_done) || any_axil_error);
    end
end

wire result_tx_busy;
wire result_tx_done;
wire [7:0] result_status_code = any_axil_error ? 8'd2 : (sink_done ? 8'd0 : 8'd1);
result_uart_tx #(.CLK_FREQ_HZ(CLK_FREQ_HZ), .BAUD(UART_BAUD)) u_result_uart_tx (
    .clk(clk_main), .rst_n(rst_n), .start(result_start),
    .status_code(result_status_code), .cycles(measured_cycles), .checksum(output_checksum),
    .word_count(output_word_count), .tx(uart_tx), .busy(result_tx_busy), .done(result_tx_done)
);

{param_logic}

{kernel_instances}

wire [5:0] status_bits;
assign status_bits[0] = counter_running;
assign status_bits[1] = sink_done;
assign status_bits[2] = all_axil_done;
assign status_bits[3] = any_axil_error;
assign status_bits[4] = source_done;
assign status_bits[5] = result_sent;

genvar status_idx;
generate
    for (status_idx = 0; status_idx < STATUS_LED_WIDTH; status_idx = status_idx + 1) begin : g_status_led
        assign status_led[status_idx] = status_bits[status_idx];
    end
endgenerate

endmodule
"""
    (rtl_dir / "harness_top.v").write_text(top, encoding="utf-8")


def debug_specs_for_mode(specs: list[dict[str, Any]], debug_mode: str) -> list[dict[str, Any]]:
    expand, dw, project = specs
    if debug_mode == "full_stitched":
        return [expand, dw, project]
    if debug_mode == "pw_expand_only":
        return [expand]
    if debug_mode == "pw_expand_to_dw":
        return [expand, dw]
    if debug_mode == "pw_project_only":
        return [project]
    raise ValueError(f"Unsupported MBConv debug mode: {debug_mode}")


def debug_chain_text(selected: list[dict[str, Any]], debug_mode: str, fifo_depth: int) -> tuple[str, str]:
    if debug_mode == "full_stitched":
        expand, dw, project = selected
        mid1_width = int(expand["top_ports"]["output_stream_TDATA"])
        mid1_keep = int(expand["top_ports"].get("output_stream_TKEEP", max(1, mid1_width // 8)))
        mid2_width = int(dw["top_ports"]["output_stream_TDATA"])
        mid2_keep = int(dw["top_ports"].get("output_stream_TKEEP", max(1, mid2_width // 8)))
        if mid1_width != int(dw["top_ports"]["input_stream_TDATA"]) or mid2_width != int(project["top_ports"]["input_stream_TDATA"]):
            raise ValueError("full_stitched debug requires equal adjacent AXIS widths")
        decl = "\n".join(
            [
                axis_wire_decl("expand_out", mid1_width, mid1_keep),
                axis_wire_decl("fifo1_out", mid1_width, mid1_keep),
                axis_wire_decl("dw_out", mid2_width, mid2_keep),
                axis_wire_decl("fifo2_out", mid2_width, mid2_keep),
            ]
        )
        body = "\n\n".join(
            [
                decl,
                f"""axis_simple_fifo #(.DATA_WIDTH({mid1_width}), .KEEP_WIDTH({mid1_keep}), .DEPTH({fifo_depth})) u_fifo_expand_to_dw (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(expand_out_tdata), .s_tkeep(expand_out_tkeep), .s_tstrb(expand_out_tstrb), .s_tlast(expand_out_tlast),
    .s_tvalid(expand_out_tvalid), .s_tready(expand_out_tready),
    .m_tdata(fifo1_out_tdata), .m_tkeep(fifo1_out_tkeep), .m_tstrb(fifo1_out_tstrb), .m_tlast(fifo1_out_tlast),
    .m_tvalid(fifo1_out_tvalid), .m_tready(fifo1_out_tready)
);""",
                f"""axis_simple_fifo #(.DATA_WIDTH({mid2_width}), .KEEP_WIDTH({mid2_keep}), .DEPTH({fifo_depth})) u_fifo_dw_to_project (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(dw_out_tdata), .s_tkeep(dw_out_tkeep), .s_tstrb(dw_out_tstrb), .s_tlast(dw_out_tlast),
    .s_tvalid(dw_out_tvalid), .s_tready(dw_out_tready),
    .m_tdata(fifo2_out_tdata), .m_tkeep(fifo2_out_tkeep), .m_tstrb(fifo2_out_tstrb), .m_tlast(fifo2_out_tlast),
    .m_tvalid(fifo2_out_tvalid), .m_tready(fifo2_out_tready)
);""",
            ]
        )
        instances = "\n\n".join(
            [
                build_kernel_instance_for_spec(expand, "input", "expand_out"),
                build_kernel_instance_for_spec(dw, "fifo1_out", "dw_out"),
                build_kernel_instance_for_spec(project, "fifo2_out", "output"),
            ]
        )
        return body, instances

    if debug_mode == "pw_expand_only":
        return "", build_kernel_instance_for_spec(selected[0], "input", "output")

    if debug_mode == "pw_expand_to_dw":
        expand, dw = selected
        mid_width = int(expand["top_ports"]["output_stream_TDATA"])
        mid_keep = int(expand["top_ports"].get("output_stream_TKEEP", max(1, mid_width // 8)))
        if mid_width != int(dw["top_ports"]["input_stream_TDATA"]):
            raise ValueError("pw_expand_to_dw debug requires equal adjacent AXIS widths")
        body = "\n\n".join(
            [
                axis_wire_decl("expand_out", mid_width, mid_keep),
                axis_wire_decl("fifo1_out", mid_width, mid_keep),
                f"""axis_simple_fifo #(.DATA_WIDTH({mid_width}), .KEEP_WIDTH({mid_keep}), .DEPTH({fifo_depth})) u_fifo_expand_to_dw (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata(expand_out_tdata), .s_tkeep(expand_out_tkeep), .s_tstrb(expand_out_tstrb), .s_tlast(expand_out_tlast),
    .s_tvalid(expand_out_tvalid), .s_tready(expand_out_tready),
    .m_tdata(fifo1_out_tdata), .m_tkeep(fifo1_out_tkeep), .m_tstrb(fifo1_out_tstrb), .m_tlast(fifo1_out_tlast),
    .m_tvalid(fifo1_out_tvalid), .m_tready(fifo1_out_tready)
);""",
            ]
        )
        instances = "\n\n".join(
            [
                build_kernel_instance_for_spec(expand, "input", "expand_out"),
                build_kernel_instance_for_spec(dw, "fifo1_out", "output"),
            ]
        )
        return body, instances

    if debug_mode == "pw_project_only":
        return "", build_kernel_instance_for_spec(selected[0], "input", "output")

    raise ValueError(f"Unsupported MBConv debug mode: {debug_mode}")


def write_debug_stitched_top(
    *,
    project_root: Path,
    combo_case_name: str,
    specs: list[dict[str, Any]],
    board_config: Path,
    debug_mode: str,
    fifo_depth: int,
    watchdog_cycles: int,
    param_reset_style: str = "kernel",
) -> None:
    board_cfg = load_yaml(board_config)
    defaults_cfg = load_yaml(BOARD_HARNESS_DIR / "configs" / "harness_defaults.yaml")
    data_dir = project_root / "data"
    rtl_dir = project_root / "rtl"
    data_dir.mkdir(parents=True, exist_ok=True)
    rtl_dir.mkdir(parents=True, exist_ok=True)
    gh.copy_tree_files(BOARD_HARNESS_DIR / "modules", rtl_dir)
    (rtl_dir / "axis_simple_fifo.v").write_text(axis_fifo_module_text(), encoding="utf-8")

    selected = debug_specs_for_mode(specs, debug_mode)
    input_spec = selected[0]
    output_spec = selected[-1]
    input_width = int(input_spec["top_ports"]["input_stream_TDATA"])
    input_keep = int(input_spec["top_ports"].get("input_stream_TKEEP", max(1, input_width // 8)))
    output_width = int(output_spec["top_ports"]["output_stream_TDATA"])
    output_keep = int(output_spec["top_ports"].get("output_stream_TKEEP", max(1, output_width // 8)))
    input_word_count = int(input_spec["case_meta"]["input_word_count"])
    output_word_count = int(output_spec["case_meta"]["output_word_count"])

    input_mem_paths = gh.generate_stream_mem_files(data_dir, "input_stream", input_word_count, input_width)
    input_word_mem = gh.generate_stream_word_mem_file(data_dir, "input_stream_words", input_word_count, input_width)
    input_mem_mapping = {f"INPUT_MEM_FILE_{lane_idx:02d}": '""' for lane_idx in range(32)}
    for lane_idx, mem_path in enumerate(input_mem_paths):
        input_mem_mapping[f"INPUT_MEM_FILE_{lane_idx:02d}"] = f'"../data/{mem_path.name}"'

    chain_body, kernel_instances = debug_chain_text(selected, debug_mode, fifo_depth)
    param_logic = "\n\n".join(build_param_logic_for_spec(spec, data_dir, param_reset_style) for spec in selected)
    axil_blocks = "\n\n".join(axil_wires_and_ctrl(spec) for spec in selected)
    axil_done_expr = " && ".join(f"{spec['role']}_axil_done" for spec in selected)
    axil_error_expr = " || ".join(f"{spec['role']}_axil_error" for spec in selected)
    status_led_width = int(board_cfg.get("status_led", {}).get("width", 2))
    top = f"""`timescale 1ns / 1ps

module harness_top (
    input  wire sys_clk_p,
    input  wire sys_clk_n,
    input  wire sys_rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire [{status_led_width - 1}:0] status_led
);

localparam integer CLK_FREQ_HZ = {int(board_cfg["clock"]["freq_hz"])};
localparam integer UART_BAUD = {int(board_cfg.get("uart", {}).get("baud", defaults_cfg["uart_baud"]))};
localparam integer BOOT_DELAY_CYCLES = {int(defaults_cfg["boot_delay_cycles"])};
localparam integer STATUS_LED_WIDTH = {status_led_width};
localparam integer INPUT_TDATA_WIDTH = {input_width};
localparam integer INPUT_KEEP_WIDTH = {input_keep};
localparam integer OUTPUT_TDATA_WIDTH = {output_width};
localparam integer OUTPUT_KEEP_WIDTH = {output_keep};
localparam integer INPUT_WORD_COUNT = {input_word_count};
localparam integer OUTPUT_WORD_COUNT = {output_word_count};
localparam integer WATCHDOG_CYCLES = {watchdog_cycles};

{gh.build_clocking_logic(board_cfg)}

wire rst_n;
reset_sync u_reset_sync (.clk(clk_main), .rst_n_async(sys_rst_n & clock_ready), .rst_n_sync(rst_n));

reg [31:0] boot_counter = 32'd0;
reg boot_started = 1'b0;
wire start_pulse = (boot_counter == BOOT_DELAY_CYCLES - 1) && !boot_started;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        boot_counter <= 32'd0;
        boot_started <= 1'b0;
    end else if (!boot_started) begin
        if (boot_counter == BOOT_DELAY_CYCLES - 1) begin
            boot_started <= 1'b1;
        end else begin
            boot_counter <= boot_counter + 32'd1;
        end
    end
end

{axis_wire_decl("input", input_width, input_keep)}
wire source_done;
axis_rom_source #(
    .DATA_WIDTH(INPUT_TDATA_WIDTH),
    .KEEP_WIDTH(INPUT_KEEP_WIDTH),
    .WORD_COUNT(INPUT_WORD_COUNT),
    .WORD_MEM_FILE("../data/{input_word_mem.name}"),
    .MEM_FILE_00({input_mem_mapping["INPUT_MEM_FILE_00"]}),
    .MEM_FILE_01({input_mem_mapping["INPUT_MEM_FILE_01"]}),
    .MEM_FILE_02({input_mem_mapping["INPUT_MEM_FILE_02"]}),
    .MEM_FILE_03({input_mem_mapping["INPUT_MEM_FILE_03"]}),
    .MEM_FILE_04({input_mem_mapping["INPUT_MEM_FILE_04"]}),
    .MEM_FILE_05({input_mem_mapping["INPUT_MEM_FILE_05"]}),
    .MEM_FILE_06({input_mem_mapping["INPUT_MEM_FILE_06"]}),
    .MEM_FILE_07({input_mem_mapping["INPUT_MEM_FILE_07"]}),
    .MEM_FILE_08({input_mem_mapping["INPUT_MEM_FILE_08"]}),
    .MEM_FILE_09({input_mem_mapping["INPUT_MEM_FILE_09"]}),
    .MEM_FILE_10({input_mem_mapping["INPUT_MEM_FILE_10"]}),
    .MEM_FILE_11({input_mem_mapping["INPUT_MEM_FILE_11"]}),
    .MEM_FILE_12({input_mem_mapping["INPUT_MEM_FILE_12"]}),
    .MEM_FILE_13({input_mem_mapping["INPUT_MEM_FILE_13"]}),
    .MEM_FILE_14({input_mem_mapping["INPUT_MEM_FILE_14"]}),
    .MEM_FILE_15({input_mem_mapping["INPUT_MEM_FILE_15"]}),
    .MEM_FILE_16({input_mem_mapping["INPUT_MEM_FILE_16"]}),
    .MEM_FILE_17({input_mem_mapping["INPUT_MEM_FILE_17"]}),
    .MEM_FILE_18({input_mem_mapping["INPUT_MEM_FILE_18"]}),
    .MEM_FILE_19({input_mem_mapping["INPUT_MEM_FILE_19"]}),
    .MEM_FILE_20({input_mem_mapping["INPUT_MEM_FILE_20"]}),
    .MEM_FILE_21({input_mem_mapping["INPUT_MEM_FILE_21"]}),
    .MEM_FILE_22({input_mem_mapping["INPUT_MEM_FILE_22"]}),
    .MEM_FILE_23({input_mem_mapping["INPUT_MEM_FILE_23"]}),
    .MEM_FILE_24({input_mem_mapping["INPUT_MEM_FILE_24"]}),
    .MEM_FILE_25({input_mem_mapping["INPUT_MEM_FILE_25"]}),
    .MEM_FILE_26({input_mem_mapping["INPUT_MEM_FILE_26"]}),
    .MEM_FILE_27({input_mem_mapping["INPUT_MEM_FILE_27"]}),
    .MEM_FILE_28({input_mem_mapping["INPUT_MEM_FILE_28"]}),
    .MEM_FILE_29({input_mem_mapping["INPUT_MEM_FILE_29"]}),
    .MEM_FILE_30({input_mem_mapping["INPUT_MEM_FILE_30"]}),
    .MEM_FILE_31({input_mem_mapping["INPUT_MEM_FILE_31"]})
) u_axis_rom_source (
    .clk(clk_main), .rst_n(rst_n), .start(start_pulse),
    .tdata(input_tdata), .tkeep(input_tkeep), .tstrb(input_tstrb), .tlast(input_tlast),
    .tvalid(input_tvalid), .tready(input_tready), .done(source_done)
);

{axis_wire_decl("output", output_width, output_keep)}
{chain_body}

wire sink_done;
wire [31:0] output_checksum;
wire [31:0] output_word_count;
axis_checksum_sink #(
    .DATA_WIDTH(OUTPUT_TDATA_WIDTH),
    .KEEP_WIDTH(OUTPUT_KEEP_WIDTH),
    .EXPECTED_WORD_COUNT(OUTPUT_WORD_COUNT)
) u_axis_checksum_sink (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse),
    .tdata(output_tdata), .tkeep(output_tkeep), .tstrb(output_tstrb), .tlast(output_tlast),
    .tvalid(output_tvalid), .tready(output_tready), .done(sink_done),
    .checksum(output_checksum), .word_count(output_word_count)
);

{axil_blocks}

wire [31:0] measured_cycles;
wire counter_running;
wire counter_done;
wire all_axil_done = {axil_done_expr};
wire any_axil_error = {axil_error_expr};
wire strict_debug_done = sink_done && all_axil_done;
cycle_counter u_cycle_counter (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse), .start(start_pulse), .stop(strict_debug_done),
    .cycles(measured_cycles), .running(counter_running), .done(counter_done)
);

reg [31:0] watchdog_counter = 32'd0;
reg watchdog_expired = 1'b0;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        watchdog_counter <= 32'd0;
        watchdog_expired <= 1'b0;
    end else if (start_pulse) begin
        watchdog_counter <= 32'd0;
        watchdog_expired <= 1'b0;
    end else if (!strict_debug_done && !watchdog_expired && boot_started) begin
        if (watchdog_counter >= WATCHDOG_CYCLES) begin
            watchdog_expired <= 1'b1;
        end else begin
            watchdog_counter <= watchdog_counter + 32'd1;
        end
    end
end

reg result_sent = 1'b0;
reg result_start = 1'b0;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else if (start_pulse) begin
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else begin
        if (result_tx_done) result_sent <= 1'b1;
        result_start <= !result_sent && !result_tx_busy && (strict_debug_done || watchdog_expired || any_axil_error);
    end
end

wire result_tx_busy;
wire result_tx_done;
wire [7:0] debug_status_byte = {{result_sent, watchdog_expired, source_done, any_axil_error, all_axil_done, sink_done, counter_done, counter_running}};
wire [31:0] debug_status_word = {{output_checksum[23:0], debug_status_byte}};
wire [7:0] result_status_code = any_axil_error ? 8'd2 : (strict_debug_done ? 8'd0 : (watchdog_expired ? 8'd3 : 8'd1));
wire [31:0] result_cycles = strict_debug_done ? measured_cycles : watchdog_counter;
result_uart_tx #(.CLK_FREQ_HZ(CLK_FREQ_HZ), .BAUD(UART_BAUD)) u_result_uart_tx (
    .clk(clk_main), .rst_n(rst_n), .start(result_start),
    .status_code(result_status_code), .cycles(result_cycles), .checksum(debug_status_word),
    .word_count(output_word_count), .tx(uart_tx), .busy(result_tx_busy), .done(result_tx_done)
);

{param_logic}

{kernel_instances}

wire [7:0] status_bits;
assign status_bits[0] = counter_running;
assign status_bits[1] = sink_done;
assign status_bits[2] = all_axil_done;
assign status_bits[3] = any_axil_error;
assign status_bits[4] = source_done;
assign status_bits[5] = watchdog_expired;
assign status_bits[6] = result_sent;
assign status_bits[7] = counter_done;

genvar status_idx;
generate
    for (status_idx = 0; status_idx < STATUS_LED_WIDTH; status_idx = status_idx + 1) begin : g_status_led
        assign status_led[status_idx] = status_bits[status_idx];
    end
endgenerate

endmodule
"""
    (rtl_dir / "harness_top.v").write_text(top, encoding="utf-8")


def timing_recovery_tcl(variant: str) -> tuple[str, str]:
    if variant == "resetless_param_bram":
        return "", ""
    if variant.startswith("resetless_param_bram_"):
        return timing_recovery_tcl(variant[len("resetless_param_bram_") :])
    if variant == "const_param_all_reset":
        return "", ""
    if variant.startswith("const_param_all_reset_"):
        return timing_recovery_tcl(variant[len("const_param_all_reset_") :])
    if variant == "const_param_reset":
        return "", ""
    if variant.startswith("const_param_reset_"):
        return timing_recovery_tcl(variant[len("const_param_reset_") :])
    if variant == "bram_fifo":
        return "", ""
    if variant.startswith("bram_fifo_"):
        return timing_recovery_tcl(variant[len("bram_fifo_") :])
    if variant == "reg_fifo":
        return "", ""
    if variant.startswith("reg_fifo_"):
        return timing_recovery_tcl(variant[len("reg_fifo_") :])
    if variant == "default":
        return "", ""
    if variant == "performance_explore":
        return (
            "\n".join(
                [
                    "if {[catch {set_property strategy {Performance_Explore} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "performance_extra_timing":
        return (
            "\n".join(
                [
                    "if {[catch {set_property strategy {Performance_ExtraTimingOpt} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "route_higher_delay_cost":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {HigherDelayCost} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "route_no_timing_relaxation":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {NoTimingRelaxation} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "place_extranet_high_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {ExtraNetDelay_high} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "place_extranet_medium_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {ExtraNetDelay_medium} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "impl_phys_opt":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {Explore} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE {AggressiveExplore} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "place_extra_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {ExtraTimingOpt} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "place_altspread_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {AltSpreadLogic_high} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "place_wlblock_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {WLDrivenBlockPlacement} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "synth_retiming_route_more_global":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {MoreGlobalIterations} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "",
        )
    if variant == "post_route_phys_opt":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {Explore} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {Explore} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "\n".join(
                [
                    "if {[catch {phys_opt_design -directive AggressiveExplore} msg]} {puts \"WARN: post-route phys_opt_design failed: $msg\"}",
                    "if {[catch {route_design -directive MoreGlobalIterations} msg]} {puts \"WARN: post-phys_opt route_design failed: $msg\"}",
                ]
            ),
        )
    if variant == "route_no_timing_post_route_phys_opt_explore":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {NoTimingRelaxation} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "\n".join(
                [
                    "if {[catch {phys_opt_design -directive Explore} msg]} {puts \"WARN: post-route phys_opt_design failed: $msg\"}",
                    "if {[catch {route_design -directive NoTimingRelaxation} msg]} {puts \"WARN: post-phys_opt route_design failed: $msg\"}",
                ]
            ),
        )
    if variant == "route_no_timing_post_route_phys_opt_critical_cell":
        return (
            "\n".join(
                [
                    "if {[catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {NoTimingRelaxation} [get_runs impl_1]} msg]} {puts \"WARN: $msg\"}",
                ]
            ),
            "\n".join(
                [
                    "if {[catch {phys_opt_design -directive CriticalCellOpt} msg]} {puts \"WARN: post-route phys_opt_design failed: $msg\"}",
                    "if {[catch {route_design -directive NoTimingRelaxation} msg]} {puts \"WARN: post-phys_opt route_design failed: $msg\"}",
                ]
            ),
        )
    raise ValueError(f"Unsupported timing recovery variant: {variant}")


def fifo_style_for_recovery_variant(variant: str) -> str:
    if variant == "reg_fifo" or variant.startswith("reg_fifo_"):
        return "lut_registered"
    if variant == "bram_fifo" or variant.startswith("bram_fifo_"):
        return "bram"
    return "lut"


def param_reset_style_for_recovery_variant(variant: str) -> str:
    if (
        variant == "resetless_param_bram"
        or variant.startswith("resetless_param_bram_")
        or "_resetless_param_bram" in variant
    ):
        return "const_zero_all"
    if (
        variant == "const_param_all_reset"
        or variant.startswith("const_param_all_reset_")
        or "_const_param_all_reset" in variant
    ):
        return "const_zero_all"
    if variant == "const_param_reset" or variant.startswith("const_param_reset_"):
        return "const_zero"
    return "kernel"


def write_build_tcl(project_root: Path, combo_case_name: str, board_config: Path, recovery_variant: str = "default") -> None:
    board_cfg = load_yaml(board_config)
    build_key = f"{combo_case_name}|{project_root.resolve()}"
    build_tag = "fc_" + hashlib.sha1(build_key.encode("utf-8")).hexdigest()[:12]
    vivado_project_dir = BOARD_HARNESS_DIR / "_vivado_builds" / build_tag
    pre_impl_tcl, post_route_tcl = timing_recovery_tcl(recovery_variant)
    drc_relax = ""
    if board_cfg.get("allow_unconstrained_io", False):
        drc_relax = "\n".join(
            [
                "set_property SEVERITY {Warning} [get_drc_checks UCIO-1]",
                "set_property SEVERITY {Warning} [get_drc_checks NSTD-1]",
            ]
        )
    text = gh.render_template(
        gh.TEMPLATES_DIR / "build_bitstream.tcl.tmpl",
        {
            "PROJECT_NAME": build_tag,
            "TARGET_PART": str(board_cfg["target_part"]),
            "VIVADO_PROJECT_DIR": str(vivado_project_dir).replace("\\", "/"),
            "RTL_DIR": str(project_root / "rtl").replace("\\", "/"),
            "EXPORT_RTL_DIR": str(project_root / "export_rtl").replace("\\", "/"),
            "CONSTRAINT_FILE": str(project_root / "constraints" / "harness.xdc").replace("\\", "/"),
            "BITSTREAM_OUTPUT_DIR": str(project_root / "bitstream").replace("\\", "/"),
            "REPORTS_OUTPUT_DIR": str(project_root / "reports").replace("\\", "/"),
            "ARTIFACT_BASENAME": combo_case_name,
            "OPTIONAL_DRC_RELAX": drc_relax,
        },
    )
    if pre_impl_tcl:
        text = text.replace("launch_runs synth_1 -jobs 8", f"{pre_impl_tcl}\n\nlaunch_runs synth_1 -jobs 8")
    if post_route_tcl:
        text = text.replace("open_run impl_1\n", f"open_run impl_1\n{post_route_tcl}\n")
    (project_root / "scripts").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts" / "build_bitstream.tcl").write_text(text, encoding="utf-8")


def generate_stitched_harness(
    row: dict[str, str],
    board_config: Path,
    project_root: Path,
    fifo_depth: int = 512,
    recovery_variant: str = "default",
    fifo_style: str = "lut",
    param_reset_style: str = "kernel",
) -> dict[str, Any]:
    if project_root.exists():
        shutil.rmtree(project_root)
    (project_root / "export_rtl").mkdir(parents=True, exist_ok=True)
    roles = split_pipe(row["component_roles"])
    cases = split_pipe(row["selected_existing_case_names"])
    if roles != ["pw_expand", "dw", "pw_project"] or len(cases) != 3:
        raise ValueError(f"Unexpected MBConv roles/cases for {row['combo_case_name']}: {roles} {cases}")
    specs: list[dict[str, Any]] = []
    for role, case_name in zip(roles, cases):
        case_dir = resolve_case_dir(case_name)
        specs.append(build_component_spec(role, case_name, case_dir, project_root / "export_rtl"))
    write_stitched_top(
        project_root=project_root,
        combo_case_name=row["combo_case_name"],
        specs=specs,
        board_config=board_config,
        fifo_depth=fifo_depth,
        fifo_style=fifo_style,
        param_reset_style=param_reset_style,
    )
    write_constraints(project_root, board_config)
    write_build_tcl(project_root, row["combo_case_name"], board_config, recovery_variant=recovery_variant)
    manifest = {
        "case_name": row["combo_case_name"],
        "module_name": "stitched_mbconv_harness",
        "board_config": str(board_config),
        "generated": {
            "project_root": str(project_root),
            "rtl_dir": str(project_root / "rtl"),
            "export_rtl_dir": str(project_root / "export_rtl"),
            "constraints": str(project_root / "constraints" / "harness.xdc"),
            "build_tcl": str(project_root / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(project_root / "bitstream" / f"{row['combo_case_name']}.bit"),
        },
        "combo": {
            "combo_case_name": row["combo_case_name"],
            "topology": MBCONV_TOPOLOGY,
            "component_roles": roles,
            "component_cases": cases,
            "fifo_depth": fifo_depth,
            "fifo_style": fifo_style,
            "param_reset_style": param_reset_style,
            "timing_recovery_variant": recovery_variant,
            "input_word_count": specs[0]["case_meta"]["input_word_count"],
            "output_word_count": specs[-1]["case_meta"]["output_word_count"],
        },
        "components": specs,
    }
    (project_root / "harness_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def generate_debug_stitched_harness(
    row: dict[str, str],
    board_config: Path,
    project_root: Path,
    *,
    debug_mode: str,
    fifo_depth: int,
    watchdog_cycles: int,
) -> dict[str, Any]:
    if project_root.exists():
        shutil.rmtree(project_root)
    (project_root / "export_rtl").mkdir(parents=True, exist_ok=True)
    roles = split_pipe(row["component_roles"])
    cases = split_pipe(row["selected_existing_case_names"])
    if roles != ["pw_expand", "dw", "pw_project"] or len(cases) != 3:
        raise ValueError(f"Unexpected MBConv roles/cases for {row['combo_case_name']}: {roles} {cases}")
    specs: list[dict[str, Any]] = []
    for role, case_name in zip(roles, cases):
        case_dir = resolve_case_dir(case_name)
        specs.append(build_component_spec(role, case_name, case_dir, project_root / "export_rtl"))
    selected = debug_specs_for_mode(specs, debug_mode)
    write_debug_stitched_top(
        project_root=project_root,
        combo_case_name=row["combo_case_name"],
        specs=specs,
        board_config=board_config,
        debug_mode=debug_mode,
        fifo_depth=fifo_depth,
        watchdog_cycles=watchdog_cycles,
    )
    write_constraints(project_root, board_config)
    artifact = f"{row['combo_case_name']}__debug_{debug_mode}_fifo{fifo_depth}"
    write_build_tcl(project_root, artifact, board_config)
    manifest = {
        "case_name": artifact,
        "module_name": "stitched_mbconv_debug_harness",
        "board_config": str(board_config),
        "debug_only": True,
        "debug_mode": debug_mode,
        "watchdog_cycles": watchdog_cycles,
        "generated": {
            "project_root": str(project_root),
            "rtl_dir": str(project_root / "rtl"),
            "export_rtl_dir": str(project_root / "export_rtl"),
            "constraints": str(project_root / "constraints" / "harness.xdc"),
            "build_tcl": str(project_root / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(project_root / "bitstream" / f"{artifact}.bit"),
        },
        "combo": {
            "combo_case_name": row["combo_case_name"],
            "topology": MBCONV_TOPOLOGY,
            "component_roles": roles,
            "component_cases": cases,
            "debug_selected_roles": [spec["role"] for spec in selected],
            "debug_selected_cases": [spec["case_name"] for spec in selected],
            "fifo_depth": fifo_depth,
            "input_word_count": selected[0]["case_meta"]["input_word_count"],
            "output_word_count": selected[-1]["case_meta"]["output_word_count"],
        },
        "components": specs,
    }
    (project_root / "harness_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_mbconv_combo(row: dict[str, str], args: argparse.Namespace, full: bool) -> dict[str, Any]:
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    combo = row["combo_case_name"]
    output_dir = mbconv_output_dir(row, full)
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_projects_dir = output_dir / "harness_projects"
    harness_project = harness_projects_dir / f"harness_{combo}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row.get("decomposition", ""),
        "harness_kind": "stitched_mbconv_harness",
        "pilot_group": row.get("pilot_group", ""),
        "batch_group": "full_mbconv_stitched" if full else "mbconv_stitched_pilot",
        "component_case_names": row.get("selected_existing_case_names", ""),
        "started_at": timestamp(),
    }
    try:
        for case_name in split_pipe(row["selected_existing_case_names"]):
            case_dir = resolve_case_dir(case_name)
            csynth = ensure_csynth(
                case_dir,
                vitis_hls,
                force=False,
                timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
            )
            (logs_dir / f"{case_name}.csynth.out.log").write_text(str(csynth.get("stdout", "")), encoding="utf-8", errors="replace")
            if csynth.get("status") in {"failed", "timeout"}:
                raise RuntimeError(f"csynth failed for {case_name}: {str(csynth.get('stdout', ''))[:800]}")
        manifest_payload = generate_stitched_harness(row, board_config, harness_project)
        (logs_dir / "harness_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        result_payload.update(
            status="harness_build_fail",
            board_execution_status="failed",
            failure_category="harness_build_fail",
            failure_reason=str(exc),
            harness_project_dir=str(harness_project),
        )
        write_result(combo, result_payload)
        return result_payload

    bitstream = build_bitstream(
        harness_project,
        vivado,
        force=True,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
    bitstream_path = harness_project / "bitstream" / f"{combo}.bit"
    if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
        result_payload.update(
            status="bitstream_build_fail",
            board_execution_status="failed",
            failure_category="bitstream_build_fail",
            failure_reason=str(bitstream.get("stdout", ""))[:1000],
            harness_project_dir=str(harness_project),
            stdout_log=str(logs_dir / "vivado_bitstream.out.log"),
        )
        write_result(combo, result_payload)
        return result_payload

    json_out = measurements_dir / f"{combo}.measurement.json"
    measurement_csv = output_dir / "mbconv_stitched_measurements_raw.csv"
    measurement = measure_harness(
        manifest=harness_project / "harness_manifest.json",
        bitstream=bitstream_path,
        csv_path=measurement_csv,
        json_out=json_out,
        serial_port=args.serial_port,
        vivado=vivado,
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
    measurement_payload = load_json(json_out)
    wns = parse_wns(harness_project / "reports" / "timing_summary.rpt")
    status, board_execution, failure_category, failure_reason, timing_clean = classify_measurement(measurement_payload, wns)
    measurement_payload.update(
        {
            "combo_case_name": combo,
            "combo_decomposition": MBCONV_DECOMPOSITION,
            "full_combo_latency_claim": "stitched_mbconv_board_latency",
            "stitched_topology": MBCONV_TOPOLOGY,
            "board_timing_wns_ns": wns,
            "board_timing_clean": timing_clean,
            "updated_at": timestamp(),
        }
    )
    json_out.write_text(json.dumps(measurement_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    util = parse_utilization(harness_project / "reports" / "utilization.rpt")
    result_payload.update(
        {
            "completed_at": timestamp(),
            "status": status,
            "board_execution_status": board_execution,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "measurement_status": measurement.get("status", ""),
            "uart_status_code": frame_value(measurement_payload, "status_code"),
            "uart_status_label": measurement_payload.get("status_label", ""),
            "board_combo_cycles": frame_value(measurement_payload, "cycles"),
            "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
            "board_word_count": frame_value(measurement_payload, "word_count"),
            "board_checksum": frame_value(measurement_payload, "checksum"),
            "wns_ns": wns,
            "timing_clean": timing_clean,
            "measurement_json_path": str(json_out),
            "measurement_csv_path": str(measurement_csv),
            "bitstream_path": str(bitstream_path),
            "harness_project_dir": str(harness_project),
            "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
            "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
            "stdout_log": str(logs_dir / "measurement.out.log"),
            "stderr_log": "",
        }
    )
    result_payload.update(util)
    write_result(combo, result_payload)
    return result_payload


def recovery_result_path(combo_case_name: str, variant: str) -> Path:
    return MBCONV_RECOVERY_DIR / combo_case_name / variant / "recovery_result.json"


def parse_recovery_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        variants = ["default", "performance_explore", "post_route_phys_opt"]
    for variant in variants:
        timing_recovery_tcl(variant)
    return variants


def mbconv_recovery_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = selected_rows(mbconv_pilot_rows(), args)
    current_results = load_results()

    def priority(row: dict[str, str]) -> tuple[float, str]:
        result = current_results.get(row["combo_case_name"], {})
        wns = parse_float(result.get("wns_ns"))
        if wns is not None and wns < 0:
            return (abs(wns), row["combo_case_name"])
        return (9999.0, row["combo_case_name"])

    return sorted(rows, key=priority)


def run_mbconv_recovery_case(row: dict[str, str], args: argparse.Namespace, *, variant: str) -> dict[str, Any]:
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    combo = row["combo_case_name"]
    fifo_style = fifo_style_for_recovery_variant(variant)
    param_reset_style = param_reset_style_for_recovery_variant(variant)
    output_dir = MBCONV_RECOVERY_DIR / combo / variant
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_project = output_dir / "harness_project"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row.get("decomposition", ""),
        "harness_kind": "stitched_mbconv_harness",
        "pilot_group": row.get("pilot_group", ""),
        "batch_group": f"mbconv_stitched_pilot_recovery_{variant}",
        "component_case_names": row.get("selected_existing_case_names", ""),
        "timing_recovery_variant": variant,
        "fifo_style": fifo_style,
        "param_reset_style": param_reset_style,
        "started_at": timestamp(),
        "strict_latency_usable": False,
    }
    try:
        for case_name in split_pipe(row["selected_existing_case_names"]):
            case_dir = resolve_case_dir(case_name)
            csynth = ensure_csynth(
                case_dir,
                vitis_hls,
                force=False,
                timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
            )
            (logs_dir / f"{case_name}.csynth.out.log").write_text(str(csynth.get("stdout", "")), encoding="utf-8", errors="replace")
            if csynth.get("status") in {"failed", "timeout"}:
                raise RuntimeError(f"csynth failed for {case_name}: {str(csynth.get('stdout', ''))[:800]}")
        manifest_payload = generate_stitched_harness(
            row,
            board_config,
            harness_project,
            recovery_variant=variant,
            fifo_style=fifo_style,
            param_reset_style=param_reset_style,
        )
        (logs_dir / "harness_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        result_payload.update(
            status="harness_build_fail",
            board_execution_status="failed",
            failure_category="harness_build_fail",
            failure_reason=str(exc),
            harness_project_dir=str(harness_project),
        )
        recovery_result_path(combo, variant).parent.mkdir(parents=True, exist_ok=True)
        recovery_result_path(combo, variant).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result_payload

    bitstream = build_bitstream(
        harness_project,
        vivado,
        force=True,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
    bitstream_path = harness_project / "bitstream" / f"{combo}.bit"
    if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
        result_payload.update(
            status="bitstream_build_fail",
            board_execution_status="failed",
            failure_category="bitstream_build_fail",
            failure_reason=str(bitstream.get("stdout", ""))[:1000],
            harness_project_dir=str(harness_project),
            stdout_log=str(logs_dir / "vivado_bitstream.out.log"),
        )
        recovery_result_path(combo, variant).parent.mkdir(parents=True, exist_ok=True)
        recovery_result_path(combo, variant).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result_payload

    json_out = measurements_dir / f"{combo}.recovery_{variant}.measurement.json"
    measurement_csv = output_dir / "mbconv_stitched_recovery_measurements_raw.csv"
    measurement = measure_harness(
        manifest=harness_project / "harness_manifest.json",
        bitstream=bitstream_path,
        csv_path=measurement_csv,
        json_out=json_out,
        serial_port=args.serial_port,
        vivado=vivado,
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
    measurement_payload = load_json(json_out)
    wns = parse_wns(harness_project / "reports" / "timing_summary.rpt")
    status, board_execution, failure_category, failure_reason, timing_clean = classify_measurement(measurement_payload, wns)
    measurement_payload.update(
        {
            "combo_case_name": combo,
            "combo_decomposition": MBCONV_DECOMPOSITION,
            "full_combo_latency_claim": "stitched_mbconv_board_latency",
            "stitched_topology": MBCONV_TOPOLOGY,
            "timing_recovery_variant": variant,
            "fifo_style": fifo_style,
            "param_reset_style": param_reset_style,
            "board_timing_wns_ns": wns,
            "board_timing_clean": timing_clean,
            "updated_at": timestamp(),
        }
    )
    json_out.write_text(json.dumps(measurement_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_payload.update(
        {
            "completed_at": timestamp(),
            "status": status,
            "board_execution_status": board_execution,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "measurement_status": measurement.get("status", ""),
            "uart_status_code": frame_value(measurement_payload, "status_code"),
            "uart_status_label": measurement_payload.get("status_label", ""),
            "board_combo_cycles": frame_value(measurement_payload, "cycles"),
            "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
            "board_word_count": frame_value(measurement_payload, "word_count"),
            "board_checksum": frame_value(measurement_payload, "checksum"),
            "wns_ns": wns,
            "timing_clean": timing_clean,
            "measurement_json_path": str(json_out),
            "measurement_csv_path": str(measurement_csv),
            "bitstream_path": str(bitstream_path),
            "harness_project_dir": str(harness_project),
            "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
            "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
            "stdout_log": str(logs_dir / "measurement.out.log"),
            "stderr_log": "",
            "strict_latency_usable": timing_clean,
        }
    )
    result_payload.update(parse_utilization(harness_project / "reports" / "utilization.rpt"))
    recovery_result_path(combo, variant).parent.mkdir(parents=True, exist_ok=True)
    recovery_result_path(combo, variant).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    existing = load_json(result_path(combo))
    existing_wns = parse_float(existing.get("wns_ns"))
    new_wns = parse_float(result_payload.get("wns_ns"))
    improves_timing_fail = (
        result_payload.get("status") == "board_measured_uart_ok_timing_fail"
        and existing.get("status") != "board_measured_timing_clean"
        and new_wns is not None
        and (existing_wns is None or new_wns > existing_wns)
    )
    if timing_clean or improves_timing_fail:
        write_result(combo, result_payload)
    return result_payload


def run_mbconv_recovery(args: argparse.Namespace) -> dict[str, Any]:
    assert_serial_port(args.serial_port)
    variants = parse_recovery_variants(args.recovery_variants)
    rows = mbconv_recovery_rows(args)
    outputs: list[dict[str, Any]] = []
    for row in rows:
        existing = load_results().get(row["combo_case_name"], {})
        if existing.get("status") == "board_measured_timing_clean" and not args.force:
            outputs.append(existing)
            continue
        for variant in variants:
            result = run_mbconv_recovery_case(row, args, variant=variant)
            outputs.append(result)
            build_outputs("incremental-mbconv-recovery")
            if result.get("status") == "board_measured_timing_clean" and parse_bool(result.get("timing_clean")):
                break
    summary = {
        "generated_at": timestamp(),
        "output_dir": str(MBCONV_RECOVERY_DIR),
        "variants": variants,
        "result_count": len(outputs),
        "counts": dict(Counter(str(result.get("status", "")) for result in outputs)),
        "results": outputs,
    }
    MBCONV_RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    (MBCONV_RECOVERY_DIR / "mbconv_timing_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_outputs("post-mbconv-recovery")
    return summary


def debug_result_path(combo_case_name: str, debug_mode: str, fifo_depth: int) -> Path:
    return MBCONV_DEBUG_DIR / combo_case_name / f"{debug_mode}_fifo{fifo_depth}" / "debug_result.json"


def enrich_debug_measurement(
    measurement_payload: dict[str, Any],
    *,
    row: dict[str, str],
    debug_mode: str,
    fifo_depth: int,
    expected_word_count: Any,
    wns: str,
) -> dict[str, Any]:
    checksum = frame_value(measurement_payload, "checksum")
    observed_word_count = frame_value(measurement_payload, "word_count")
    expected_int = parse_int(expected_word_count)
    observed_int = parse_int(observed_word_count)
    measurement_payload.update(
        {
            "debug_only": True,
            "strict_latency_usable": False,
            "full_combo_latency_claim": "none_debug_only",
            "combo_case_name": row["combo_case_name"],
            "combo_decomposition": MBCONV_DECOMPOSITION,
            "debug_mode": debug_mode,
            "fifo_depth": fifo_depth,
            "expected_word_count": expected_word_count,
            "observed_word_count": observed_word_count,
            "word_count_match": bool(expected_int is not None and observed_int is not None and expected_int == observed_int),
            "debug_status_bits": decode_debug_status_bits(checksum),
            "board_timing_wns_ns": wns,
            "updated_at": timestamp(),
        }
    )
    return measurement_payload


def run_mbconv_debug_case(row: dict[str, str], args: argparse.Namespace, *, debug_mode: str, fifo_depth: int) -> dict[str, Any]:
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    combo = row["combo_case_name"]
    output_dir = MBCONV_DEBUG_DIR / combo / f"{debug_mode}_fifo{fifo_depth}"
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_project = output_dir / "harness_project"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row.get("decomposition", ""),
        "debug_only": True,
        "debug_mode": debug_mode,
        "fifo_depth": fifo_depth,
        "started_at": timestamp(),
        "strict_latency_usable": False,
    }
    try:
        for case_name in split_pipe(row["selected_existing_case_names"]):
            case_dir = resolve_case_dir(case_name)
            csynth = ensure_csynth(
                case_dir,
                vitis_hls,
                force=False,
                timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
            )
            (logs_dir / f"{case_name}.csynth.out.log").write_text(str(csynth.get("stdout", "")), encoding="utf-8", errors="replace")
            if csynth.get("status") in {"failed", "timeout"}:
                raise RuntimeError(f"csynth failed for {case_name}: {str(csynth.get('stdout', ''))[:800]}")
        manifest_payload = generate_debug_stitched_harness(
            row,
            board_config,
            harness_project,
            debug_mode=debug_mode,
            fifo_depth=fifo_depth,
            watchdog_cycles=args.debug_watchdog_cycles,
        )
        (logs_dir / "harness_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        result_payload.update(status="debug_harness_build_fail", failure_category="debug_harness_build_fail", failure_reason=str(exc))
        debug_result_path(combo, debug_mode, fifo_depth).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result_payload

    bitstream = build_bitstream(
        harness_project,
        vivado,
        force=args.force,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
    bitstream_path = harness_project / "bitstream" / f"{manifest_payload['case_name']}.bit"
    if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
        result_payload.update(
            status="debug_bitstream_build_fail",
            failure_category="debug_bitstream_build_fail",
            failure_reason=str(bitstream.get("stdout", ""))[:1000],
            harness_project_dir=str(harness_project),
            stdout_log=str(logs_dir / "vivado_bitstream.out.log"),
        )
        debug_result_path(combo, debug_mode, fifo_depth).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result_payload

    json_out = measurements_dir / f"{manifest_payload['case_name']}.measurement.json"
    measurement_csv = output_dir / "debug_measurements_raw.csv"
    measurement = measure_harness(
        manifest=harness_project / "harness_manifest.json",
        bitstream=bitstream_path,
        csv_path=measurement_csv,
        json_out=json_out,
        serial_port=args.serial_port,
        vivado=vivado,
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=max(float(args.uart_timeout_seconds), 60.0),
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
    measurement_payload = load_json(json_out)
    wns = parse_wns(harness_project / "reports" / "timing_summary.rpt")
    expected_word_count = manifest_payload["combo"]["output_word_count"]
    measurement_payload = enrich_debug_measurement(
        measurement_payload,
        row=row,
        debug_mode=debug_mode,
        fifo_depth=fifo_depth,
        expected_word_count=expected_word_count,
        wns=wns,
    )
    json_out.write_text(json.dumps(measurement_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status_code = parse_int(frame_value(measurement_payload, "status_code"))
    result_payload.update(
        {
            "completed_at": timestamp(),
            "status": "debug_frame_received" if status_code is not None else "debug_no_uart_frame",
            "measurement_status": measurement.get("status", ""),
            "uart_status_code": frame_value(measurement_payload, "status_code"),
            "uart_status_label": measurement_payload.get("status_label", ""),
            "debug_status_bits": measurement_payload.get("debug_status_bits", {}),
            "observed_word_count": measurement_payload.get("observed_word_count", ""),
            "expected_word_count": expected_word_count,
            "word_count_match": measurement_payload.get("word_count_match", False),
            "wns_ns": wns,
            "measurement_json_path": str(json_out),
            "measurement_csv_path": str(measurement_csv),
            "bitstream_path": str(bitstream_path),
            "harness_project_dir": str(harness_project),
            "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
            "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
            "stdout_log": str(logs_dir / "measurement.out.log"),
            "stderr_log": "",
        }
    )
    result_payload.update(parse_utilization(harness_project / "reports" / "utilization.rpt"))
    debug_result_path(combo, debug_mode, fifo_depth).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_payload


def delta_fields(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    pred_cycles = parse_int(row.get("prediction_baseline_cycles"))
    board_cycles = parse_int(result.get("board_combo_cycles"))
    delta_cycles: Any = ""
    delta_percent: Any = ""
    if pred_cycles is not None and board_cycles is not None:
        delta_cycles = board_cycles - pred_cycles
        if pred_cycles:
            delta_percent = f"{(delta_cycles / pred_cycles) * 100.0:.6f}"
    return {"delta_cycles": delta_cycles, "delta_percent": delta_percent}


def strict_pilot_gate_blocked(results: dict[str, dict[str, Any]]) -> bool:
    for row in pilot_rows():
        result = results.get(row["combo_case_name"])
        if not result:
            return True
        if result.get("status") != "board_measured_timing_clean":
            return True
        if not parse_bool(result.get("timing_clean")):
            return True
    return False


def support_status_for(
    row: dict[str, str],
    result: dict[str, Any] | None,
    pilot_gate_blocked: bool,
    pilot_names: set[str],
) -> str:
    if result:
        return str(result.get("status", "pending"))
    if row.get("combo_case_name") in pilot_names:
        return "pending"
    if pilot_gate_blocked:
        return "blocked"
    return "pending"


def ensure_failure_measurement_json(result: dict[str, Any]) -> None:
    measurement_path = str(result.get("measurement_json_path", ""))
    if not measurement_path:
        return
    path = Path(measurement_path)
    status = str(result.get("status", ""))
    if status in MEASURED_STATUS:
        return
    if path.exists():
        existing = load_json(path)
        if existing.get("failure_category") or existing.get("frame"):
            return
    stdout_log = Path(str(result.get("stdout_log", "")))
    stdout_excerpt = ""
    if stdout_log.exists():
        stdout_excerpt = stdout_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    payload = {
        "status": "failed",
        "combo_case_name": result.get("combo_case_name", ""),
        "failure_category": result.get("failure_category", ""),
        "failure_reason": result.get("failure_reason", ""),
        "stdout_log": str(stdout_log) if stdout_log else "",
        "stdout_excerpt": stdout_excerpt,
        "bitstream": result.get("bitstream_path", ""),
        "updated_at": timestamp(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_outputs(mode: str) -> dict[str, Any]:
    preflight = assert_inputs()
    runlist = read_csv(DRY_RUNLIST_CSV)
    results = load_results()
    pilot_names = {row["combo_case_name"] for row in pilot_rows()}
    pilot_gate_blocked = strict_pilot_gate_blocked(results)
    status_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []

    for row in runlist:
        combo = row["combo_case_name"]
        result = results.get(combo)
        if result:
            ensure_failure_measurement_json(result)
        status = support_status_for(row, result, pilot_gate_blocked, pilot_names)
        out = {field: row.get(field, "") for field in RUNLIST_FIELDS}
        out.update(
            {
                "status_timestamp": timestamp(),
                "combo_measurement_source": result.get("batch_group", "") if result else "",
                "support_status": status,
                "measurement_status": result.get("measurement_status", "") if result else ("blocked" if status == "blocked" else "not_run"),
                "board_execution_status": result.get("board_execution_status", "") if result else ("blocked_by_pilot_gate" if status == "blocked" else "not_started"),
                "failure_category": result.get("failure_category", "") if result else ("strict_pilot_gate_blocked" if status == "blocked" else ""),
                "failure_reason": result.get("failure_reason", "") if result else ("all direct/MBConv pilots must be timing-clean before full run" if status == "blocked" else ""),
                "uart_status_code": result.get("uart_status_code", "") if result else "",
                "uart_status_label": result.get("uart_status_label", "") if result else "",
                "board_combo_cycles": result.get("board_combo_cycles", "") if result else "",
                "board_combo_latency_ms": result.get("board_combo_latency_ms", "") if result else "",
                "board_word_count": result.get("board_word_count", "") if result else "",
                "board_checksum": result.get("board_checksum", "") if result else "",
                "wns_ns": result.get("wns_ns", "") if result else "",
                "timing_clean": str(result.get("timing_clean", False)) if result else "False",
                "measurement_json_path": result.get("measurement_json_path", "") if result else "",
                "measurement_csv_path": result.get("measurement_csv_path", "") if result else "",
                "bitstream_path": result.get("bitstream_path", "") if result else "",
                "harness_project_dir": result.get("harness_project_dir", "") if result else "",
                "timing_report_path": result.get("timing_report_path", "") if result else "",
                "utilization_report_path": result.get("utilization_report_path", "") if result else "",
                "board_harness_lut": result.get("board_harness_lut", "") if result else "",
                "board_harness_ff": result.get("board_harness_ff", "") if result else "",
                "board_harness_bram_tile": result.get("board_harness_bram_tile", "") if result else "",
                "board_harness_dsp": result.get("board_harness_dsp", "") if result else "",
                "batch_group": result.get("batch_group", "") if result else "",
                "stdout_log": result.get("stdout_log", "") if result else "",
                "stderr_log": result.get("stderr_log", "") if result else "",
            }
        )
        status_rows.append(out)
        result_for_delta = result or {}
        deltas = delta_fields(row, result_for_delta)
        overlay = {
            "combo_case_name": combo,
            "decomposition": row.get("decomposition", ""),
            "harness_kind": row.get("harness_kind", ""),
            "component_roles": row.get("component_roles", ""),
            "component_case_names": row.get("component_case_names", ""),
            "selected_existing_case_names": row.get("selected_existing_case_names", ""),
            "board_measurement_source": result.get("batch_group", "") if result else "",
            "uart_status_code": result.get("uart_status_code", "") if result else "",
            "board_combo_cycles": result.get("board_combo_cycles", "") if result else "",
            "board_combo_latency_ms": result.get("board_combo_latency_ms", "") if result else "",
            "prediction_baseline_cycles": row.get("prediction_baseline_cycles", ""),
            "prediction_baseline_latency_ms": row.get("prediction_baseline_latency_ms", ""),
            "delta_cycles": deltas["delta_cycles"],
            "delta_percent": deltas["delta_percent"],
            "wns_ns": result.get("wns_ns", "") if result else "",
            "timing_clean": str(result.get("timing_clean", False)) if result else "False",
            "measurement_json_path": result.get("measurement_json_path", "") if result else "",
            "bitstream_path": result.get("bitstream_path", "") if result else "",
            "timing_report_path": result.get("timing_report_path", "") if result else "",
            "notes": (
                "primitive prediction baseline only; measured combo latency is from COM5"
                if result and result.get("status") in MEASURED_STATUS
                else "no measured combo latency"
            ),
        }
        overlay_rows.append(overlay)
        if result and result.get("status") in MEASURED_STATUS:
            measurement_rows.append(
                {
                    "timestamp": result.get("completed_at", ""),
                    "combo_case_name": combo,
                    "decomposition": row.get("decomposition", ""),
                    "harness_kind": row.get("harness_kind", ""),
                    "pilot_group": row.get("pilot_group", ""),
                    "batch_group": result.get("batch_group", ""),
                    "status": result.get("status", ""),
                    "uart_status_code": result.get("uart_status_code", ""),
                    "uart_status_label": result.get("uart_status_label", ""),
                    "board_combo_cycles": result.get("board_combo_cycles", ""),
                    "board_combo_latency_ms": result.get("board_combo_latency_ms", ""),
                    "board_word_count": result.get("board_word_count", ""),
                    "board_checksum": result.get("board_checksum", ""),
                    "wns_ns": result.get("wns_ns", ""),
                    "timing_clean": str(result.get("timing_clean", False)),
                    "prediction_baseline_cycles": row.get("prediction_baseline_cycles", ""),
                    "prediction_baseline_latency_ms": row.get("prediction_baseline_latency_ms", ""),
                    "delta_cycles": deltas["delta_cycles"],
                    "delta_percent": deltas["delta_percent"],
                    "measurement_json_path": result.get("measurement_json_path", ""),
                    "bitstream_path": result.get("bitstream_path", ""),
                    "harness_project_dir": result.get("harness_project_dir", ""),
                    "timing_report_path": result.get("timing_report_path", ""),
                    "utilization_report_path": result.get("utilization_report_path", ""),
                }
            )
        if result and result.get("status") == "board_measured_uart_ok_timing_fail":
            timing_rows.append(out)
        if status not in {"pending", "board_measured_timing_clean"}:
            if status == "blocked" or result:
                blocker_rows.append(
                    {
                        "combo_case_name": combo,
                        "decomposition": row.get("decomposition", ""),
                        "harness_kind": row.get("harness_kind", ""),
                        "pilot_group": row.get("pilot_group", ""),
                        "status": status,
                        "failure_category": out.get("failure_category", ""),
                        "failure_reason": out.get("failure_reason", ""),
                        "uart_status_code": out.get("uart_status_code", ""),
                        "wns_ns": out.get("wns_ns", ""),
                        "measurement_json_path": out.get("measurement_json_path", ""),
                        "harness_project_dir": out.get("harness_project_dir", ""),
                        "timing_report_path": out.get("timing_report_path", ""),
                        "stdout_log": out.get("stdout_log", ""),
                        "stderr_log": out.get("stderr_log", ""),
                    }
                )

    write_csv(BOARD_STATUS_CSV, status_rows, BOARD_STATUS_FIELDS)
    write_csv(BOARD_MEASUREMENTS_CSV, measurement_rows, MEASUREMENT_FIELDS)
    write_csv(TIMING_FAILURES_CSV, timing_rows, BOARD_STATUS_FIELDS)
    write_csv(BLOCKERS_CSV, blocker_rows, BLOCKER_FIELDS)
    write_csv(LATENCY_OVERLAY_CSV, overlay_rows, OVERLAY_FIELDS)
    LATENCY_OVERLAY_JSON.write_text(json.dumps(overlay_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = summary_counts(status_rows)
    payload = {
        "generated_at": timestamp(),
        "mode": mode,
        "output_dir": str(OUT_DIR),
        "preflight": preflight,
        "counts": counts,
        "pilot_gate_blocked": pilot_gate_blocked,
        "outputs": {
            "board_status_csv": str(BOARD_STATUS_CSV),
            "board_measurements_csv": str(BOARD_MEASUREMENTS_CSV),
            "timing_failures_csv": str(TIMING_FAILURES_CSV),
            "blockers_csv": str(BLOCKERS_CSV),
            "summary_json": str(SUMMARY_JSON),
            "latency_overlay_csv": str(LATENCY_OVERLAY_CSV),
            "latency_overlay_json": str(LATENCY_OVERLAY_JSON),
            "readme": str(README_MD),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(status_rows, counts, pilot_gate_blocked)
    return payload


def summary_counts(status_rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct = [row for row in status_rows if row.get("decomposition") == "direct"]
    mbconv = [row for row in status_rows if row.get("decomposition") == MBCONV_DECOMPOSITION]

    def count(rows: list[dict[str, Any]], status: str) -> int:
        return sum(1 for row in rows if row.get("support_status") == status)

    def uart_ok(rows: list[dict[str, Any]]) -> int:
        return sum(1 for row in rows if parse_int(row.get("uart_status_code")) == 0)

    def clean(rows: list[dict[str, Any]]) -> int:
        return sum(1 for row in rows if parse_bool(row.get("timing_clean")))

    return {
        "total_entries": len(status_rows),
        "direct_total": len(direct),
        "mbconv_total": len(mbconv),
        "direct_uart_ok": uart_ok(direct),
        "direct_timing_clean": clean(direct),
        "direct_timing_violation": count(direct, "board_measured_uart_ok_timing_fail"),
        "direct_runtime_fail": count(direct, "board_runtime_fail"),
        "direct_harness_build_fail": count(direct, "harness_build_fail"),
        "direct_bitstream_build_fail": count(direct, "bitstream_build_fail"),
        "direct_pending": count(direct, "pending"),
        "direct_blocked": count(direct, "blocked"),
        "mbconv_uart_ok": uart_ok(mbconv),
        "mbconv_timing_clean": clean(mbconv),
        "mbconv_timing_violation": count(mbconv, "board_measured_uart_ok_timing_fail"),
        "mbconv_runtime_fail": count(mbconv, "board_runtime_fail"),
        "mbconv_harness_build_fail": count(mbconv, "harness_build_fail"),
        "mbconv_bitstream_build_fail": count(mbconv, "bitstream_build_fail"),
        "mbconv_pending": count(mbconv, "pending"),
        "mbconv_blocked": count(mbconv, "blocked"),
        "board_measured_timing_clean": count(status_rows, "board_measured_timing_clean"),
        "board_measured_uart_ok_timing_fail": count(status_rows, "board_measured_uart_ok_timing_fail"),
        "pending": count(status_rows, "pending"),
        "blocked": count(status_rows, "blocked"),
    }


def write_readme(status_rows: list[dict[str, Any]], counts: dict[str, Any], pilot_gate_blocked: bool) -> None:
    lines = [
        "# current84 Full-Combo AV7K325 Board Validation",
        "",
        f"Generated at: {timestamp()}",
        "",
        "Scope:",
        "- Direct combo latency is direct full-combo board latency only when it comes from COM5 measurement.",
        "- MBConv combo latency is claimed only after stitched harness measurement.",
        "- Primitive sum latency is prediction baseline only.",
        "- These measurements are not complete sonar classifier end-to-end latency.",
        "",
        "Strict timing-clean rule:",
        "- `timing_clean=true` means UART `status_code=0` and AV7K325 harness WNS >= 0.",
        "- UART-ok rows with WNS < 0 are `board_measured_uart_ok_timing_fail`.",
        "- The fullcombo84 all-run gate opens only when all 9 pilots are timing-clean.",
        "- Debug or relaxed-clock evidence under `mbconv_stitched_debug/` is diagnostic only and is not strict board latency.",
        "",
        "Counts:",
        f"- total combos: {counts['total_entries']}",
        f"- direct combos: {counts['direct_total']}",
        f"- MBConv stitched combos: {counts['mbconv_total']}",
        f"- direct UART ok: {counts['direct_uart_ok']}",
        f"- direct timing clean: {counts['direct_timing_clean']}",
        f"- MBConv UART ok: {counts['mbconv_uart_ok']}",
        f"- MBConv timing clean: {counts['mbconv_timing_clean']}",
        f"- timing violations: {counts['board_measured_uart_ok_timing_fail']}",
        f"- pending: {counts['pending']}",
        f"- blocked: {counts['blocked']}",
        f"- pilot gate blocked: {pilot_gate_blocked}",
        "",
        "Outputs:",
        f"- board status: `{BOARD_STATUS_CSV}`",
        f"- board measurements: `{BOARD_MEASUREMENTS_CSV}`",
        f"- timing failures: `{TIMING_FAILURES_CSV}`",
        f"- blockers: `{BLOCKERS_CSV}`",
        f"- summary: `{SUMMARY_JSON}`",
        f"- latency overlay CSV: `{LATENCY_OVERLAY_CSV}`",
        f"- latency overlay JSON: `{LATENCY_OVERLAY_JSON}`",
        "",
        "Current non-clean measured or blocked rows:",
    ]
    non_clean = [
        row
        for row in status_rows
        if row.get("support_status") not in {"pending", "board_measured_timing_clean"}
    ]
    if non_clean:
        for row in non_clean:
            lines.append(
                f"- `{row.get('combo_case_name')}`: {row.get('support_status')} "
                f"UART={row.get('uart_status_code')} WNS={row.get('wns_ns')} {row.get('failure_reason')}"
            )
    else:
        lines.append("- none")
    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pilot_rows() -> list[dict[str, str]]:
    return read_csv(PILOTS_CSV)


def mbconv_pilot_rows() -> list[dict[str, str]]:
    rows = [row for row in pilot_rows() if row.get("decomposition") == MBCONV_DECOMPOSITION]
    return rows


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.combo:
        return rows
    requested = set(args.combo)
    return [row for row in rows if row["combo_case_name"] in requested]


def run_rows(rows: list[dict[str, str]], args: argparse.Namespace, *, full: bool) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        if args.combo and row["combo_case_name"] not in set(args.combo):
            continue
        existing = load_json(result_path(row["combo_case_name"]))
        if existing and existing.get("status") in MEASURED_STATUS and not args.force:
            results.append(existing)
            continue
        if row.get("decomposition") == "direct":
            result = run_direct_combo(row, args, full=full)
        elif row.get("decomposition") == MBCONV_DECOMPOSITION:
            result = run_mbconv_combo(row, args, full=full)
        else:
            result = {
                "combo_case_name": row["combo_case_name"],
                "status": "blocked",
                "failure_category": "unsupported_decomposition",
                "failure_reason": row.get("decomposition", ""),
            }
            write_result(row["combo_case_name"], result)
        results.append(result)
        build_outputs("incremental")
    return results


def run_mbconv_pilots(args: argparse.Namespace) -> list[dict[str, Any]]:
    assert_serial_port(args.serial_port)
    return run_rows(selected_rows(mbconv_pilot_rows(), args), args, full=False)


def run_debug_sequence_for_row(row: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    depths = parse_int_list(args.debug_fifo_depths)
    if args.debug_mode:
        sequence = [(mode, depth) for mode in args.debug_mode for depth in depths]
    else:
        first_depth = depths[0]
        sequence = [
            ("full_stitched", first_depth),
            ("pw_expand_only", first_depth),
            ("pw_expand_to_dw", first_depth),
            ("pw_project_only", first_depth),
        ]
        sequence.extend(("full_stitched", depth) for depth in depths[1:])
    results: list[dict[str, Any]] = []
    for debug_mode, fifo_depth in sequence:
        results.append(run_mbconv_debug_case(row, args, debug_mode=debug_mode, fifo_depth=fifo_depth))
    return results


def run_debug_mbconv(args: argparse.Namespace) -> dict[str, Any]:
    assert_serial_port(args.serial_port)
    rows = selected_rows(mbconv_pilot_rows(), args)
    if args.debug_limit > 0:
        rows = rows[: args.debug_limit]
    results: list[dict[str, Any]] = []
    for row in rows:
        results.extend(run_debug_sequence_for_row(row, args))
    summary = {
        "generated_at": timestamp(),
        "debug_only": True,
        "strict_latency_usable": False,
        "output_dir": str(MBCONV_DEBUG_DIR),
        "rows": len(rows),
        "result_count": len(results),
        "counts": dict(Counter(str(result.get("status", "")) for result in results)),
        "results": results,
    }
    MBCONV_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (MBCONV_DEBUG_DIR / "mbconv_stitched_debug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def remeasure_existing_mbconv_pilots(args: argparse.Namespace) -> dict[str, Any]:
    assert_serial_port(args.serial_port)
    vivado = Path(args.vivado).expanduser().resolve()
    rows = selected_rows(mbconv_pilot_rows(), args)
    results_by_combo = load_results()
    outputs: list[dict[str, Any]] = []
    for row in rows:
        combo = row["combo_case_name"]
        existing = results_by_combo.get(combo, {})
        harness_project = Path(str(existing.get("harness_project_dir", "")))
        bitstream_path = Path(str(existing.get("bitstream_path", "")))
        if not harness_project.exists() or not bitstream_path.exists():
            outputs.append(
                {
                    "combo_case_name": combo,
                    "status": "remeasure_missing_existing_artifacts",
                    "harness_project_dir": str(harness_project),
                    "bitstream_path": str(bitstream_path),
                }
            )
            continue
        output_dir = MBCONV_DEBUG_DIR / combo / "remeasure_existing_bitstream"
        logs_dir = output_dir / "logs"
        measurements_dir = output_dir / "measurements"
        logs_dir.mkdir(parents=True, exist_ok=True)
        measurements_dir.mkdir(parents=True, exist_ok=True)
        json_out = measurements_dir / f"{combo}.remeasure.measurement.json"
        measurement_csv = output_dir / "remeasure_existing_raw.csv"
        measurement = measure_harness(
            manifest=harness_project / "harness_manifest.json",
            bitstream=bitstream_path,
            csv_path=measurement_csv,
            json_out=json_out,
            serial_port=args.serial_port,
            vivado=vivado,
            hw_server_url=args.hw_server_url,
            hw_device_pattern=args.hw_device_pattern,
            hw_target_index=args.hw_target_index,
            program_retries=args.program_retries,
            runs=args.runs,
            uart_timeout_seconds=max(float(args.uart_timeout_seconds), 60.0),
            post_program_delay_seconds=args.post_program_delay_seconds,
            timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
            skip_measure=args.skip_measure,
        )
        (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
        payload = load_json(json_out)
        wns = parse_wns(Path(str(existing.get("timing_report_path", ""))))
        payload.update(
            {
                "debug_only": True,
                "strict_latency_usable": False,
                "full_combo_latency_claim": "none_remeasure_existing_bitstream",
                "combo_case_name": combo,
                "combo_decomposition": MBCONV_DECOMPOSITION,
                "remeasure_existing_bitstream": True,
                "board_timing_wns_ns": wns,
                "updated_at": timestamp(),
            }
        )
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append(
            {
                "combo_case_name": combo,
                "status": "remeasure_frame_received" if parse_int(frame_value(payload, "status_code")) is not None else "remeasure_no_uart_frame",
                "measurement_status": measurement.get("status", ""),
                "uart_status_code": frame_value(payload, "status_code"),
                "uart_status_label": payload.get("status_label", ""),
                "observed_word_count": frame_value(payload, "word_count"),
                "wns_ns": wns,
                "measurement_json_path": str(json_out),
                "measurement_csv_path": str(measurement_csv),
                "bitstream_path": str(bitstream_path),
                "harness_project_dir": str(harness_project),
                "stdout_log": str(logs_dir / "measurement.out.log"),
            }
        )
    summary = {
        "generated_at": timestamp(),
        "debug_only": True,
        "strict_latency_usable": False,
        "output_dir": str(MBCONV_DEBUG_DIR),
        "result_count": len(outputs),
        "counts": dict(Counter(str(result.get("status", "")) for result in outputs)),
        "results": outputs,
    }
    MBCONV_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (MBCONV_DEBUG_DIR / "mbconv_existing_bitstream_remeasure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def direct_clean_program_only_candidate() -> dict[str, Any]:
    for row in read_csv(BOARD_MEASUREMENTS_CSV):
        if row.get("decomposition") == "direct" and parse_bool(row.get("timing_clean")):
            return row
    for result in load_results().values():
        if result.get("decomposition") == "direct" and result.get("status") == "board_measured_timing_clean":
            return result
    raise RuntimeError("No direct clean bitstream candidate found for program-only sanity check")


def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = OUT_DIR / "hardware_sanity"
    output_dir.mkdir(parents=True, exist_ok=True)
    vivado = Path(args.vivado).expanduser().resolve()
    list_hw_json = output_dir / "list_hw_targets.json"
    list_hw_cmd = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--list-hw-targets",
        "--vivado",
        str(vivado),
        "--hw-server-url",
        args.hw_server_url,
        "--hw-device-pattern",
        args.hw_device_pattern,
        "--hw-target-index",
        str(args.hw_target_index),
        "--json-out",
        str(list_hw_json),
    ]
    list_hw, _ = run_command(list_hw_cmd, cwd=ROOT, timeout_seconds=600)
    (output_dir / "list_hw_targets.stdout.log").write_text(str(list_hw.stdout or ""), encoding="utf-8", errors="replace")
    (output_dir / "list_hw_targets.stderr.log").write_text(str(list_hw.stderr or ""), encoding="utf-8", errors="replace")

    list_ports_cmd = [sys.executable, str(MEASURE_SCRIPT), "--list-ports"]
    list_ports, _ = run_command(list_ports_cmd, cwd=ROOT, timeout_seconds=60)
    (output_dir / "list_ports.stdout.log").write_text(str(list_ports.stdout or ""), encoding="utf-8", errors="replace")
    (output_dir / "list_ports.stderr.log").write_text(str(list_ports.stderr or ""), encoding="utf-8", errors="replace")

    candidate = direct_clean_program_only_candidate()
    program_only_json = output_dir / "program_only_direct_clean.json"
    program_cmd = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--manifest",
        str(Path(str(candidate["harness_project_dir"])) / "harness_manifest.json"),
        "--bitstream",
        str(candidate["bitstream_path"]),
        "--vivado",
        str(vivado),
        "--hw-server-url",
        args.hw_server_url,
        "--hw-device-pattern",
        args.hw_device_pattern,
        "--hw-target-index",
        str(args.hw_target_index),
        "--program-retries",
        str(args.program_retries),
        "--program-only",
        "--json-out",
        str(program_only_json),
    ]
    program_only, _ = run_command(program_cmd, cwd=ROOT, timeout_seconds=900)
    (output_dir / "program_only_direct_clean.stdout.log").write_text(str(program_only.stdout or ""), encoding="utf-8", errors="replace")
    (output_dir / "program_only_direct_clean.stderr.log").write_text(str(program_only.stderr or ""), encoding="utf-8", errors="replace")
    summary = {
        "generated_at": timestamp(),
        "output_dir": str(output_dir),
        "list_hw_targets_returncode": list_hw.returncode,
        "list_ports_returncode": list_ports.returncode,
        "program_only_returncode": program_only.returncode,
        "list_hw_targets_json": str(list_hw_json),
        "program_only_json": str(program_only_json),
        "serial_port_requested": args.serial_port,
        "list_ports_stdout": str(list_ports.stdout or ""),
        "program_only_candidate": candidate.get("combo_case_name", ""),
        "status": "ok" if list_hw.returncode == 0 and list_ports.returncode == 0 and program_only.returncode == 0 else "failed",
    }
    (output_dir / "hardware_sanity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def has_pilot_blocker() -> bool:
    return strict_pilot_gate_blocked(load_results())


def run_pilots(args: argparse.Namespace) -> list[dict[str, Any]]:
    assert_serial_port(args.serial_port)
    rows = pilot_rows()
    direct = [row for row in rows if row.get("decomposition") == "direct"]
    mbconv = [row for row in rows if row.get("decomposition") == MBCONV_DECOMPOSITION]
    results = run_rows(direct, args, full=False)
    build_outputs("post-direct-pilot")
    if any(result.get("status") in PILOT_BLOCKER_STATUS for result in results):
        return results
    results.extend(run_rows(mbconv, args, full=False))
    build_outputs("post-mbconv-pilot")
    return results


def rows_for_full_run() -> list[dict[str, str]]:
    runlist = read_csv(DRY_RUNLIST_CSV)
    pilot_names = {row["combo_case_name"] for row in pilot_rows()}
    return [row for row in runlist if row["combo_case_name"] not in pilot_names]


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    if has_pilot_blocker():
        build_outputs("blocked-before-full")
        return []
    rows = rows_for_full_run()
    if args.combo:
        requested = set(args.combo)
        rows = [row for row in rows if row["combo_case_name"] in requested]
    results: list[dict[str, Any]] = []
    for batch_idx in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[batch_idx : batch_idx + max(1, args.batch_size)]
        results.extend(run_rows(batch, args, full=True))
        batch_summary = {
            "generated_at": timestamp(),
            "batch_index": batch_idx // max(1, args.batch_size),
            "batch_size": len(batch),
            "combo_case_names": [row["combo_case_name"] for row in batch],
            "results": [result.get("status", "") for result in results[-len(batch) :]],
        }
        batch_dir = FULL_MEASUREMENT_DIR / "batch_summaries"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / f"batch_{batch_idx // max(1, args.batch_size):03d}.json").write_text(
            json.dumps(batch_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if has_pilot_blocker():
            break
    build_outputs("post-full")
    return results


def main() -> int:
    args = parse_args()
    assert_inputs()
    if args.runs != 1:
        raise ValueError("fullcombo board validation expects --runs 1 for one-shot harnesses")
    if args.mode == "init":
        payload = build_outputs("init")
    elif args.mode == "refresh":
        payload = build_outputs("refresh")
    elif args.mode == "sanity-check":
        payload = run_sanity_check(args)
    elif args.mode == "remeasure-mbconv-pilots":
        payload = remeasure_existing_mbconv_pilots(args)
    elif args.mode == "debug-mbconv":
        payload = run_debug_mbconv(args)
    elif args.mode == "recover-mbconv-pilots":
        payload = run_mbconv_recovery(args)
    elif args.mode == "run-mbconv-pilots":
        run_mbconv_pilots(args)
        payload = build_outputs("run-mbconv-pilots")
    elif args.mode == "run-pilots":
        run_pilots(args)
        payload = build_outputs("run-pilots")
    elif args.mode == "run-all":
        run_all(args)
        payload = build_outputs("run-all")
    else:
        run_pilots(args)
        if not has_pilot_blocker():
            run_all(args)
        payload = build_outputs("all")
    if payload.get("output_dir") == str(OUT_DIR):
        print(f"board_status_csv={BOARD_STATUS_CSV}")
        print(f"summary_json={SUMMARY_JSON}")
        print(f"counts={json.dumps(payload['counts'], sort_keys=True)}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
