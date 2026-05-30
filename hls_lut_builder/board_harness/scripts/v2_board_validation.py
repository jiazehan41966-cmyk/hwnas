#!/usr/bin/env python3
"""Full v2 primitive board-validation runlist, status, and batch runner."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from batch_measure_ops import (
    BOARD_HARNESS_DIR,
    DEFAULT_BOARD_CONFIG,
    DEFAULT_VITIS_HLS,
    DEFAULT_VIVADO,
    RESULTS_DIR,
    ROOT,
    build_bitstream,
    ensure_csynth,
    ensure_export,
    generate_harness,
    measure_case,
)


DEPLOYABLE_LUT = RESULTS_DIR / "v2_deployable_lut" / "deployable_lut_v2.json"
COMBO_CSV = RESULTS_DIR / "v2_deployable_lut" / "combo_deployability_v2.csv"
BOARD_RESULTS_DIR = BOARD_HARNESS_DIR / "results"
FULL44_RUNLIST = BOARD_RESULTS_DIR / "v2_board_validation_full44_runlist.csv"
FULL44_STATUS = BOARD_RESULTS_DIR / "v2_board_validation_full44_status.csv"
FULL44_MEASUREMENTS = BOARD_RESULTS_DIR / "v2_board_validation_full44_measurements.csv"
FULL44_TIMING_FAILURES = BOARD_RESULTS_DIR / "v2_board_validation_full44_timing_failures.csv"
FULL44_COMBO_PLAN = BOARD_RESULTS_DIR / "v2_board_validation_combo_plan.md"
LEGACY_V2_MEASUREMENTS = BOARD_RESULTS_DIR / "v2_board_validation_measurements.csv"

SOURCE_DIRS = {
    "failed44_fixed_vector": RESULTS_DIR / "v2_failed44_fixed_vector" / "p",
    "k5_low_complexity": RESULTS_DIR / "v2_k5_low_complexity" / "p",
    "depthwise_fallback_blockers": RESULTS_DIR / "v2_depthwise_fallback_blockers" / "p",
    "pointwise_v21": RESULTS_DIR / "v2_pointwise_v21" / "p",
}

SEED_REASONS = {
    "dw_conv_k3_fb_failed44_fb_k3_7_c480_s1_p8_cb8_ii1_dsp1_main_5ns": "depthwise fallback k3 C480 p8/cb8 II1 closes failed44 direct blocker",
    "dw_conv_k5_fb_failed44_fb_k5_14_c288_s2_p8_cb8_ii2_dsp1_main_5ns": "depthwise fallback k5 C288 p8/cb8 II2 closes failed44 direct blocker",
    "dw_conv_k5_lc_k5_lc_ir320_e3_7_c480_s1_p8_cb8_ii2_dsp1_main_5ns": "k5 low-complexity boundary C480 p8/cb8 II2",
    "dw_conv_k5_lc_k5_lc_ir160_e6_14_c576_s2_p8_cb8_ii2_dsp1_main_5ns": "k5 low-complexity boundary C576 p8/cb8 II2",
    "dw_conv_k5_lc_k5_lc_ir320_e6_7_c960_s1_p8_cb8_ii2_dsp1_main_5ns": "k5 low-complexity boundary C960 p8/cb8 II2",
    "pw_conv_v21_failed44_pw_head_7_320_1280_p16_pe16_simd4_ii2_dsp1_main_5ns": "pointwise v2.1 representative head max-Cout",
    "pw_conv_v21_failed44_mb_s7_e6_project_7_960_320_p16_pe16_simd4_ii2_dsp1_main_5ns": "pointwise v2.1 representative max-Cin project",
    "pw_conv_v21_failed44_mb_s3_e6_expand_56_24_144_p16_pe16_simd4_ii2_dsp1_main_5ns": "pointwise v2.1 representative 56x56 large-spatial expand",
    "dw_conv_k3_failed44_dw_ir320_e6_7_c960_s1_p16_cb16_ii1_dsp1_main_5ns": "direct depthwise k3 small-spatial representative",
    "dw_conv_k3_failed44_dw_ir64_e6_28_c192_s2_p16_cb16_ii1_dsp1_main_5ns": "direct depthwise k3 mid-spatial representative",
    "dw_conv_k3_failed44_dw_ir24_e6_112_c96_s2_p16_cb16_ii1_dsp1_main_5ns": "direct depthwise k3 large-spatial representative",
    "dw_conv_k5_failed44_mb_s5_e6_k5_dw_14_c384_s1_p16_cb16_ii2_dsp1_main_5ns": "direct depthwise k5 small-spatial representative",
    "dw_conv_k5_failed44_dw_ir64_e6_28_c192_s2_p16_cb16_ii2_dsp1_main_5ns": "direct depthwise k5 mid-spatial representative",
}

SEED_ORDER = {case_name: idx for idx, case_name in enumerate(SEED_REASONS, start=1)}

RUNLIST_FIELDS = [
    "case_name",
    "source_dataset",
    "case_dir",
    "op_type",
    "priority",
    "reason",
    "expected_post_route_fmax",
    "expected_power_w",
    "expected_latency_ms",
    "expected_cycles",
]

STATUS_EXTRA_FIELDS = [
    "status_timestamp",
    "board_config",
    "board_target_part",
    "metric_source_target_part",
    "case_dir_exists",
    "synth_tcl_exists",
    "csynth_report_available",
    "component_xml_exists",
    "export_rtl_exists",
    "harness_project_dir",
    "harness_manifest_exists",
    "bitstream_exists",
    "csynth_status",
    "export_status",
    "harness_status",
    "bitstream_status",
    "measurement_status",
    "timing_status",
    "support_status",
    "board_execution_status",
    "unsupported_reason",
    "summary_json",
    "stdout_log",
    "stderr_log",
    "measurement_csv",
    "measurement_json",
    "board_status_code",
    "board_status_label",
    "board_cycles",
    "board_latency_ms",
    "board_word_count",
    "board_checksum",
    "board_timing_wns_ns",
    "board_timing_met",
    "board_harness_lut",
    "board_harness_ff",
    "board_harness_bram_tile",
    "board_harness_dsp",
]

STATUS_FIELDS = RUNLIST_FIELDS + STATUS_EXTRA_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run full v2 board validation")
    parser.add_argument("mode", choices=("init", "refresh", "run"), help="Operation mode")
    parser.add_argument("--deployable-lut", default=str(DEPLOYABLE_LUT), help="v2 deployable LUT JSON")
    parser.add_argument("--combo-csv", default=str(COMBO_CSV), help="v2 combo deployability CSV")
    parser.add_argument("--runlist-csv", default=str(FULL44_RUNLIST), help="Full44 runlist CSV")
    parser.add_argument("--status-csv", default=str(FULL44_STATUS), help="Full44 status CSV")
    parser.add_argument("--measurements-csv", default=str(FULL44_MEASUREMENTS), help="Full44 board measurement CSV")
    parser.add_argument("--timing-failures-csv", default=str(FULL44_TIMING_FAILURES), help="Timing failures CSV")
    parser.add_argument("--combo-plan-md", default=str(FULL44_COMBO_PLAN), help="Combo planning markdown")
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG), help="AV7K325 board config YAML")
    parser.add_argument("--serial-port", default="COM5", help="UART serial port")
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS), help="Vitis HLS launcher")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO), help="Vivado launcher")
    parser.add_argument("--case", action="append", default=None, help="Restrict run/refresh to explicit case_name")
    parser.add_argument("--priority-min", type=int, default=None, help="Minimum priority to run")
    parser.add_argument("--priority-max", type=int, default=None, help="Maximum priority to run")
    parser.add_argument("--limit", type=int, default=None, help="Maximum selected cases to run")
    parser.add_argument("--force", action="store_true", help="Regenerate/rebuild/remeasure selected cases")
    parser.add_argument("--skip-measure", action="store_true", help="Run through bitstream build but skip UART measurement")
    parser.add_argument("--runs", type=int, default=1, help="UART measurement repeats")
    parser.add_argument("--csynth-timeout-minutes", type=int, default=60)
    parser.add_argument("--export-timeout-minutes", type=int, default=30)
    parser.add_argument("--harness-timeout-minutes", type=int, default=10)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=90)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def op_type_for(entry: dict[str, Any]) -> str:
    case_name = entry["source_case_name"]
    if case_name.startswith("pw_conv"):
        return "pw_conv"
    op = str(entry.get("op_spec", {}).get("op", ""))
    return "pw_conv" if op == "conv" else op


def residual_rank(entry: dict[str, Any]) -> tuple[int, float, str]:
    op_type = op_type_for(entry)
    case_name = entry["source_case_name"]
    risk_group = 0 if op_type == "dw_conv_k5" else 1 if op_type == "dw_conv_k3" else 2
    latency = float(entry.get("latency_ms") or 0.0)
    return (risk_group, -latency, case_name)


def reason_for(entry: dict[str, Any]) -> str:
    case_name = entry["source_case_name"]
    if case_name in SEED_REASONS:
        return SEED_REASONS[case_name]
    op_type = op_type_for(entry)
    if op_type == "dw_conv_k5":
        return "full44 residual k5 depthwise case; high timing-risk group"
    if op_type == "dw_conv_k3":
        return "full44 residual k3 depthwise case"
    if op_type == "pw_conv":
        return "full44 residual pointwise v2.1 primitive case"
    return "full44 residual v2 primitive case"


def ordered_entries(deployable_lut: Path) -> list[dict[str, Any]]:
    entries = load_json(deployable_lut)["entries"]
    seeded = sorted(
        [entry for entry in entries if entry["source_case_name"] in SEED_ORDER],
        key=lambda entry: SEED_ORDER[entry["source_case_name"]],
    )
    residual = sorted(
        [entry for entry in entries if entry["source_case_name"] not in SEED_ORDER],
        key=residual_rank,
    )
    return seeded + residual


def runlist_rows(deployable_lut: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for priority, entry in enumerate(ordered_entries(deployable_lut), start=1):
        source_dataset = entry["source_dataset"]
        case_name = entry["source_case_name"]
        case_dir = SOURCE_DIRS[source_dataset] / case_name
        impl = entry.get("implementation_metrics") or {}
        rows.append(
            {
                "case_name": case_name,
                "source_dataset": source_dataset,
                "case_dir": str(case_dir),
                "op_type": op_type_for(entry),
                "priority": priority,
                "reason": reason_for(entry),
                "expected_post_route_fmax": impl.get("post_route_Fmax_est", ""),
                "expected_power_w": impl.get("power_w", ""),
                "expected_latency_ms": entry.get("latency_ms", ""),
                "expected_cycles": entry.get("cycles", ""),
                "metric_source_target_part": impl.get("target_part", ""),
            }
        )
    return rows


def has_csynth_artifacts(case_dir: Path) -> bool:
    report_dir = case_dir / "project" / "solution1" / "syn" / "report"
    return report_dir.exists() and any(report_dir.glob("*_csynth.*"))


def parse_wns(timing_report: Path) -> str:
    if not timing_report.exists():
        return ""
    lines = timing_report.read_text(encoding="utf-8", errors="replace").splitlines()
    number_pattern = re.compile(r"^-?\d+(?:\.\d+)?$")
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 6]:
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
    for line in util_report.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        name, used = parts[1], parts[2]
        if name == "Slice LUTs":
            values["board_harness_lut"] = used
        elif name == "Slice Registers":
            values["board_harness_ff"] = used
        elif name == "Block RAM Tile":
            values["board_harness_bram_tile"] = used
        elif name == "DSPs":
            values["board_harness_dsp"] = used
    return values


def latest_measurements(path: Path) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        case_name = row.get("case_name")
        if case_name:
            latest[case_name] = row
    return latest


def seed_measurements(measurements_csv: Path) -> None:
    if measurements_csv.exists() or not LEGACY_V2_MEASUREMENTS.exists():
        return
    measurements_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_V2_MEASUREMENTS, measurements_csv)


def parse_run_time(value: Any, fallback: float = 0.0) -> float:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return fallback


def latest_case_run_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(results_dir.glob("v2_board_validation_full44_batch_*.summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        file_time = summary_path.stat().st_mtime
        batch_time = parse_run_time(summary.get("timestamp"), file_time)
        for case in summary.get("cases", []):
            case_name = case.get("case_name")
            if not case_name:
                continue
            event_time = parse_run_time(case.get("finished_at"), parse_run_time(case.get("started_at"), batch_time))
            record = {
                "case_name": case_name,
                "case_status": case.get("status", ""),
                "summary_json": str(summary_path),
                "stdout_log": summary.get("stdout_log", ""),
                "stderr_log": summary.get("stderr_log", ""),
                "event_time": event_time,
            }
            if event_time >= float(latest.get(case_name, {}).get("event_time", 0.0)):
                latest[case_name] = record
    return latest


def apply_failed_run_status(status_row: dict[str, Any], run_summary: dict[str, Any]) -> None:
    run_status = str(run_summary.get("case_status") or "")
    if not run_status.startswith("failed_"):
        return
    if status_row.get("board_status_code") == "0" and status_row.get("board_cycles"):
        return

    failure_map = {
        "failed_csynth": ("csynth_status", "vitis_hls_csynth_failed"),
        "failed_csynth_timeout": ("csynth_status", "vitis_hls_csynth_timeout"),
        "failed_export": ("export_status", "vitis_hls_export_failed"),
        "failed_export_timeout": ("export_status", "vitis_hls_export_timeout"),
        "failed_harness": ("harness_status", "board_harness_generation_failed"),
        "failed_harness_timeout": ("harness_status", "board_harness_generation_timeout"),
        "failed_bitstream": ("bitstream_status", "vivado_bitstream_failed"),
        "failed_bitstream_timeout": ("bitstream_status", "vivado_bitstream_timeout"),
        "failed_measurement": ("measurement_status", "board_measurement_failed"),
    }
    stage_field, reason = failure_map.get(run_status, ("board_execution_status", run_status))

    if run_status.startswith("failed_bitstream") and status_row.get("bitstream_exists") == "True":
        return
    if run_status.startswith("failed_harness") and status_row.get("harness_manifest_exists") == "True":
        return
    if run_status.startswith("failed_export") and status_row.get("component_xml_exists") == "True":
        return
    if run_status.startswith("failed_csynth") and status_row.get("csynth_report_available") == "True":
        return

    status_row["support_status"] = run_status
    status_row["board_execution_status"] = run_status
    status_row["unsupported_reason"] = reason
    status_row[stage_field] = "timeout" if run_status.endswith("_timeout") else "failed"
    if run_status.startswith("failed_bitstream"):
        status_row["timing_status"] = "not_available"
        status_row["board_timing_met"] = ""


def status_from_run_row(
    run_row: dict[str, Any],
    *,
    board_config: Path,
    measurements: dict[str, dict[str, str]],
    measurements_csv: Path,
) -> dict[str, Any]:
    row = dict(run_row)
    case_name = str(row["case_name"])
    case_dir = Path(str(row["case_dir"]))
    harness_project = BOARD_HARNESS_DIR / "projects" / f"harness_{case_name}"
    component_xml = case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"
    export_rtl = case_dir / "project" / "solution1" / "impl" / "ip" / "hdl" / "verilog"
    manifest = harness_project / "harness_manifest.json"
    bitstream = harness_project / "bitstream" / f"{case_name}.bit"
    timing_report = harness_project / "reports" / "timing_summary.rpt"
    util_report = harness_project / "reports" / "utilization.rpt"
    measurement = measurements.get(case_name, {})
    wns = parse_wns(timing_report)
    timing_met = ""
    if wns:
        timing_met = str(float(wns) >= 0.0)

    support_status = "not_started"
    board_execution_status = "not_started"
    unsupported_reason = ""
    if measurement.get("status_code") == "0" and measurement.get("cycles"):
        if timing_met == "True":
            support_status = "board_measured_timing_clean"
            board_execution_status = "success"
        elif timing_met == "False":
            support_status = "board_measured_uart_ok_timing_fail"
            board_execution_status = "success_uart_timing_violation"
            unsupported_reason = "board_harness_post_route_timing_violation"
        else:
            support_status = "board_measured_timing_unknown"
            board_execution_status = "success_timing_unknown"
            unsupported_reason = "timing_report_missing"
    elif not case_dir.exists():
        support_status = "unsupported"
        unsupported_reason = "case_dir_missing"
    elif not (case_dir / "synth.tcl").exists():
        support_status = "unsupported"
        unsupported_reason = "synth_tcl_missing"
    elif not has_csynth_artifacts(case_dir):
        support_status = "needs_csynth"
        unsupported_reason = "csynth_report_missing"
    elif not component_xml.exists() or not export_rtl.exists():
        support_status = "needs_hls_export"
        unsupported_reason = "component_xml_or_exported_rtl_missing"
    elif not manifest.exists():
        support_status = "ready_for_harness_generation"
    elif not bitstream.exists():
        support_status = "harness_generated_needs_bitstream"
    else:
        support_status = "ready_for_board_measurement"

    row.update(
        {
            "status_timestamp": datetime.now().isoformat(timespec="seconds"),
            "board_config": str(board_config),
            "board_target_part": "xc7k325t-ffg900-2",
            "case_dir_exists": str(case_dir.exists()),
            "synth_tcl_exists": str((case_dir / "synth.tcl").exists()),
            "csynth_report_available": str(has_csynth_artifacts(case_dir)),
            "component_xml_exists": str(component_xml.exists()),
            "export_rtl_exists": str(export_rtl.exists()),
            "harness_project_dir": str(harness_project),
            "harness_manifest_exists": str(manifest.exists()),
            "bitstream_exists": str(bitstream.exists()),
            "csynth_status": "available" if has_csynth_artifacts(case_dir) else "",
            "export_status": "available" if component_xml.exists() and export_rtl.exists() else "",
            "harness_status": "available" if manifest.exists() else "",
            "bitstream_status": "available" if bitstream.exists() else "",
            "measurement_status": "success" if measurement else "",
            "timing_status": "met" if timing_met == "True" else "failed" if timing_met == "False" else "",
            "support_status": support_status,
            "board_execution_status": board_execution_status,
            "unsupported_reason": unsupported_reason,
            "summary_json": "",
            "stdout_log": "",
            "stderr_log": "",
            "measurement_csv": str(measurements_csv) if measurement else "",
            "measurement_json": str(harness_project / "latest_measurement.json") if measurement else "",
            "board_status_code": measurement.get("status_code", ""),
            "board_status_label": measurement.get("status_label", ""),
            "board_cycles": measurement.get("cycles", ""),
            "board_latency_ms": measurement.get("latency_ms", ""),
            "board_word_count": measurement.get("word_count", ""),
            "board_checksum": measurement.get("checksum", ""),
            "board_timing_wns_ns": wns,
            "board_timing_met": timing_met,
        }
    )
    row.update(parse_utilization(util_report))
    return row


def write_timing_failures(status_rows: list[dict[str, Any]], path: Path) -> None:
    failures = [
        row
        for row in status_rows
        if row.get("board_timing_met") == "False"
        or row.get("support_status") in {"failed_bitstream", "failed_bitstream_timeout"}
    ]
    fields = [
        "priority",
        "case_name",
        "source_dataset",
        "op_type",
        "support_status",
        "board_execution_status",
        "board_timing_wns_ns",
        "unsupported_reason",
        "harness_project_dir",
    ]
    write_csv(path, failures, fields)


def write_combo_plan(combo_csv: Path, output_md: Path) -> None:
    combo_rows = read_csv(combo_csv)
    direct = sum(1 for row in combo_rows if row.get("decomposition") == "direct")
    decomposed = sum(1 for row in combo_rows if row.get("decomposition") == "pw_expand+dw+pw_project")
    lines = [
        "# v2 Board Validation Combo Plan",
        "",
        "This file is planning-only. It does not claim fused MBConv board measurement.",
        "",
        f"- combo rows: {len(combo_rows)}",
        f"- direct rows: {direct}",
        f"- decomposed MBConv rows: {decomposed}",
        "",
        "Primitive validation gate:",
        "- Complete UART measurement for all 44 v2 primitive/direct entries.",
        "- Count timing-clean only when AV7K325 harness post-route WNS is non-negative.",
        "",
        "Future combo harness scope:",
        "- Stitch `pw_expand + dw + pw_project` with explicit inter-stage stream/buffer control.",
        "- Reuse primitive manifests for interface widths, expected word counts, and timing records.",
        "- Keep fused/stitched combo measurements separate from primitive board CSVs.",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    deployable_lut = Path(args.deployable_lut).expanduser().resolve()
    board_config = Path(args.board_config).expanduser().resolve()
    runlist_csv = Path(args.runlist_csv).expanduser().resolve()
    status_csv = Path(args.status_csv).expanduser().resolve()
    measurements_csv = Path(args.measurements_csv).expanduser().resolve()
    timing_failures_csv = Path(args.timing_failures_csv).expanduser().resolve()
    combo_plan_md = Path(args.combo_plan_md).expanduser().resolve()

    seed_measurements(measurements_csv)
    rows = runlist_rows(deployable_lut)
    write_csv(runlist_csv, rows, RUNLIST_FIELDS)
    measurements = latest_measurements(measurements_csv)
    status_rows = [
        status_from_run_row(row, board_config=board_config, measurements=measurements, measurements_csv=measurements_csv)
        for row in rows
    ]
    write_csv(status_csv, status_rows, STATUS_FIELDS)
    write_timing_failures(status_rows, timing_failures_csv)
    write_combo_plan(Path(args.combo_csv).expanduser().resolve(), combo_plan_md)
    return status_rows


def refresh_outputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runlist_csv = Path(args.runlist_csv).expanduser().resolve()
    if not runlist_csv.exists():
        init_outputs(args)
    board_config = Path(args.board_config).expanduser().resolve()
    measurements_csv = Path(args.measurements_csv).expanduser().resolve()
    status_csv = Path(args.status_csv).expanduser().resolve()
    timing_failures_csv = Path(args.timing_failures_csv).expanduser().resolve()
    rows = read_csv(runlist_csv)
    existing_by_case = {row["case_name"]: row for row in read_csv(status_csv) if row.get("case_name")}
    run_summaries = latest_case_run_summaries(BOARD_RESULTS_DIR)
    measurements = latest_measurements(measurements_csv)
    status_rows = []
    for row in rows:
        status_row = status_from_run_row(
            row,
            board_config=board_config,
            measurements=measurements,
            measurements_csv=measurements_csv,
        )
        existing = existing_by_case.get(status_row["case_name"], {})
        for field in ("summary_json", "stdout_log", "stderr_log"):
            if existing.get(field):
                status_row[field] = existing[field]
        run_summary = run_summaries.get(status_row["case_name"], {})
        for field in ("summary_json", "stdout_log", "stderr_log"):
            if run_summary.get(field):
                status_row[field] = run_summary[field]
        apply_failed_run_status(status_row, run_summary)
        if existing.get("measurement_json") and not status_row.get("measurement_json"):
            status_row["measurement_json"] = existing["measurement_json"]
        status_rows.append(status_row)
    write_csv(status_csv, status_rows, STATUS_FIELDS)
    write_timing_failures(status_rows, timing_failures_csv)
    return status_rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.case:
        requested = set(args.case)
        selected = [row for row in selected if row["case_name"] in requested]
    if args.priority_min is not None:
        selected = [row for row in selected if int(row["priority"]) >= args.priority_min]
    if args.priority_max is not None:
        selected = [row for row in selected if int(row["priority"]) <= args.priority_max]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not args.force:
        selected = [
            row
            for row in selected
            if row.get("support_status") not in {"board_measured_timing_clean"}
        ]
    return selected


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def log_stage(log_path: Path, case_name: str, stage: str, payload: dict[str, Any]) -> None:
    append_log(log_path, f"[{datetime.now().isoformat(timespec='seconds')}] {case_name} {stage}")
    stdout = payload.get("stdout")
    if stdout:
        append_log(log_path, str(stdout))


def compact_stage_payload(payload: dict[str, Any], log_path: Path) -> dict[str, Any]:
    compact = dict(payload)
    stdout = compact.pop("stdout", None)
    if stdout:
        compact["stdout_log"] = str(log_path)
        compact["stdout_bytes"] = len(str(stdout))
    return compact


def run_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    status_rows = refresh_outputs(args)
    selected = select_rows(status_rows, args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = BOARD_RESULTS_DIR / f"v2_board_validation_full44_batch_{timestamp}.summary.json"
    stdout_log = BOARD_RESULTS_DIR / f"v2_board_validation_full44_batch_{timestamp}.out.log"
    stderr_log = BOARD_RESULTS_DIR / f"v2_board_validation_full44_batch_{timestamp}.err.log"
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    measurements_csv = Path(args.measurements_csv).expanduser().resolve()

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "selected_count": len(selected),
        "serial_port": args.serial_port,
        "board_config": str(board_config),
        "measurements_csv": str(measurements_csv),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "cases": [],
    }

    for row in selected:
        case_name = row["case_name"]
        case_dir = Path(row["case_dir"])
        harness_project_dir = BOARD_HARNESS_DIR / "projects" / f"harness_{case_name}"
        case_status: dict[str, Any] = {
            "case_name": case_name,
            "priority": row["priority"],
            "case_dir": str(case_dir),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        csynth_status = ensure_csynth(
            case_dir,
            vitis_hls,
            force=args.force,
            timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
        )
        log_stage(stdout_log, case_name, "csynth", csynth_status)
        case_status["csynth"] = compact_stage_payload(csynth_status, stdout_log)
        if csynth_status["status"] in {"failed", "timeout"}:
            case_status["status"] = f"failed_csynth{'_timeout' if csynth_status['status'] == 'timeout' else ''}"
            summary["cases"].append(case_status)
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        export_status = ensure_export(
            case_dir,
            vitis_hls,
            force=args.force,
            timeout_seconds=args.export_timeout_minutes * 60 if args.export_timeout_minutes > 0 else None,
        )
        log_stage(stdout_log, case_name, "export", export_status)
        case_status["export"] = compact_stage_payload(export_status, stdout_log)
        if export_status["status"] in {"failed", "timeout"}:
            case_status["status"] = f"failed_export{'_timeout' if export_status['status'] == 'timeout' else ''}"
            summary["cases"].append(case_status)
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        harness_status = generate_harness(
            case_dir,
            board_config,
            force=args.force,
            timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
        )
        log_stage(stdout_log, case_name, "harness", harness_status)
        case_status["harness"] = compact_stage_payload(harness_status, stdout_log)
        if harness_status["status"] in {"failed", "timeout"}:
            case_status["status"] = f"failed_harness{'_timeout' if harness_status['status'] == 'timeout' else ''}"
            summary["cases"].append(case_status)
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        bitstream_status = build_bitstream(
            harness_project_dir,
            vivado,
            force=args.force,
            timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
        )
        log_stage(stdout_log, case_name, "bitstream", bitstream_status)
        case_status["bitstream"] = compact_stage_payload(bitstream_status, stdout_log)
        if bitstream_status["status"] in {"failed", "timeout"}:
            case_status["status"] = f"failed_bitstream{'_timeout' if bitstream_status['status'] == 'timeout' else ''}"
            summary["cases"].append(case_status)
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        if args.skip_measure:
            measurement_status = {"status": "skipped", "reason": "--skip-measure"}
        else:
            measurement_status = measure_case(
                harness_project_dir,
                serial_port=args.serial_port,
                vivado=vivado,
                results_csv=measurements_csv,
                runs=args.runs,
                timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
            )
        case_status["measurement"] = measurement_status
        log_stage(stdout_log, case_name, "measurement", measurement_status)
        case_status["measurement"] = compact_stage_payload(measurement_status, stdout_log)
        case_status["status"] = "success" if measurement_status.get("status") in {"success", "skipped"} else "failed_measurement"
        case_status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        summary["cases"].append(case_status)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    refreshed = refresh_outputs(args)
    by_name = {row["case_name"]: row for row in refreshed}
    for case_status in summary["cases"]:
        row = by_name.get(case_status["case_name"])
        if not row:
            continue
        row["summary_json"] = str(summary_json)
        row["stdout_log"] = str(stdout_log)
        row["stderr_log"] = str(stderr_log)
    write_csv(Path(args.status_csv).expanduser().resolve(), refreshed, STATUS_FIELDS)
    print(
        json.dumps(
            {
                "timestamp": timestamp,
                "selected_count": len(selected),
                "summary_json": str(summary_json),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "case_statuses": [
                    {"case_name": item["case_name"], "status": item.get("status")}
                    for item in summary["cases"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return refreshed


def main() -> int:
    args = parse_args()
    if args.mode == "init":
        rows = init_outputs(args)
    elif args.mode == "refresh":
        rows = refresh_outputs(args)
    else:
        rows = run_cases(args)
    print(f"status_rows={len(rows)} status_csv={Path(args.status_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
