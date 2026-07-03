#!/usr/bin/env python3
"""Run isolated current84/fullcombo board validation at 150/100 MHz.

This wrapper reuses the proven current84 fullcombo harness builders, but patches
their output directories so relaxed-clock implementation and UART evidence never
overwrite the original 200 MHz ledger.

The 150/100 MHz board configs generate a real MMCM user clock inside the
harness. They are not XDC-only relaxed timing runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import current84_fullcombo_board_validation as fc  # noqa: E402


BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
BASE_RESULTS_DIR = BOARD_HARNESS_DIR / "results" / "v2_current84_fullcombo_board"
BASE_TIMING_FAILURES_CSV = BASE_RESULTS_DIR / "v2_current84_fullcombo84_timing_failures.csv"
MULTICLOCK_ROOT = BOARD_HARNESS_DIR / "results" / "v2_current84_fullcombo_board_multiclock"

PROFILE_CONFIG = {
    "150m": {
        "clock_mhz": 150.0,
        "period_ns": 1000.0 / 150.0,
        "board_config": BOARD_HARNESS_DIR / "configs" / "board_av7k325_150m.yaml",
    },
    "100m": {
        "clock_mhz": 100.0,
        "period_ns": 10.0,
        "board_config": BOARD_HARNESS_DIR / "configs" / "board_av7k325_100m.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "implement", "measure", "measure-existing", "refresh"))
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIG), required=True)
    parser.add_argument(
        "--selection",
        choices=("timing-fail-30", "all84"),
        default="timing-fail-30",
        help="Default validates the 30 original 200 MHz timing-fail rows first.",
    )
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--vitis-hls", default=str(fc.DEFAULT_VITIS_HLS))
    parser.add_argument("--vivado", default=str(fc.DEFAULT_VIVADO))
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def configure_profile_outputs(profile: str) -> Path:
    profile_dir = MULTICLOCK_ROOT / profile
    fc.OUT_DIR = profile_dir
    fc.DIRECT_PILOT_DIR = profile_dir / "direct_pilot"
    fc.MBCONV_PILOT_DIR = profile_dir / "mbconv_stitched_pilot"
    fc.MBCONV_DEBUG_DIR = profile_dir / "mbconv_stitched_debug"
    fc.MBCONV_RECOVERY_DIR = profile_dir / "mbconv_stitched_recovery"
    fc.FULL_MEASUREMENT_DIR = profile_dir / "fullcombo84_measurement"
    fc.CASE_RESULTS_DIR = profile_dir / "case_results"
    fc.BOARD_STATUS_CSV = profile_dir / f"v2_current84_fullcombo84_{profile}_board_status.csv"
    fc.BOARD_MEASUREMENTS_CSV = profile_dir / f"v2_current84_fullcombo84_{profile}_board_measurements.csv"
    fc.TIMING_FAILURES_CSV = profile_dir / f"v2_current84_fullcombo84_{profile}_timing_failures.csv"
    fc.BLOCKERS_CSV = profile_dir / f"v2_current84_fullcombo84_{profile}_blockers.csv"
    fc.SUMMARY_JSON = profile_dir / f"v2_current84_fullcombo84_{profile}_summary.json"
    fc.LATENCY_OVERLAY_CSV = profile_dir / f"v2_current84_fullcombo84_{profile}_latency_overlay.csv"
    fc.LATENCY_OVERLAY_JSON = profile_dir / f"v2_current84_fullcombo84_{profile}_latency_overlay.json"
    fc.README_MD = profile_dir / "README.md"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def selected_combo_names(selection: str) -> set[str]:
    if selection == "all84":
        return {row["combo_case_name"] for row in fc.read_csv(fc.DRY_RUNLIST_CSV)}
    return {row["combo_case_name"] for row in read_csv(BASE_TIMING_FAILURES_CSV)}


def selected_runlist_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    names = selected_combo_names(args.selection)
    if args.combo:
        names &= set(args.combo)
    rows = [row for row in fc.read_csv(fc.DRY_RUNLIST_CSV) if row["combo_case_name"] in names]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def make_fc_args(args: argparse.Namespace, *, skip_measure: bool) -> argparse.Namespace:
    return argparse.Namespace(
        board_config=str(PROFILE_CONFIG[args.profile]["board_config"]),
        serial_port=args.serial_port,
        runs=args.runs,
        vitis_hls=args.vitis_hls,
        vivado=args.vivado,
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        csynth_timeout_minutes=args.csynth_timeout_minutes,
        harness_timeout_minutes=args.harness_timeout_minutes,
        bitstream_timeout_minutes=args.bitstream_timeout_minutes,
        measurement_timeout_minutes=args.measurement_timeout_minutes,
        combo=[row["combo_case_name"] for row in selected_runlist_rows(args)],
        batch_size=args.batch_size,
        force=args.force,
        skip_measure=skip_measure,
    )


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _path_or_none(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


def _path_text(path: Path | None) -> str:
    return str(path) if path is not None else ""


def row_output_dir(row: dict[str, str]) -> Path:
    if row.get("decomposition") == "direct":
        return fc.direct_output_dir(row, full=True)
    if row.get("decomposition") == fc.MBCONV_DECOMPOSITION:
        return fc.mbconv_output_dir(row, full=True)
    return fc.FULL_MEASUREMENT_DIR / "unsupported" / row["combo_case_name"]


def expected_harness_project(row: dict[str, str], output_dir: Path) -> Path:
    if row.get("decomposition") == "direct":
        component_case = fc.split_pipe(row["selected_existing_case_names"])[0] or fc.split_pipe(row["component_case_names"])[0]
        return output_dir / "harness_projects" / f"harness_{component_case}"
    return output_dir / "harness_projects" / f"harness_{row['combo_case_name']}"


def measurement_csv_path(row: dict[str, str], output_dir: Path) -> Path:
    if row.get("decomposition") == "direct":
        return output_dir / "direct_measurements_raw.csv"
    return output_dir / "mbconv_stitched_measurements_raw.csv"


def claim_for_row(row: dict[str, str]) -> str:
    if row.get("decomposition") == "direct":
        return "direct_fullcombo_board_latency"
    return "stitched_mbconv_board_latency"


def artifacts_for_row(row: dict[str, str], result: dict[str, Any] | None = None) -> dict[str, Path | None]:
    result = result or {}
    output_dir = row_output_dir(row)
    harness_project = _path_or_none(result.get("harness_project_dir"))
    if harness_project is None or not harness_project.exists():
        candidate = expected_harness_project(row, output_dir)
        harness_project = candidate if candidate.exists() else None
    if harness_project is None:
        candidates = sorted((output_dir / "harness_projects").glob("harness_*"))
        harness_project = candidates[-1] if candidates else None

    def result_or_search(result_key: str, search_pattern: str) -> Path | None:
        path = _path_or_none(result.get(result_key))
        if path is not None and path.is_file():
            return path
        search_root = harness_project if harness_project is not None else output_dir
        matches = sorted(search_root.glob(search_pattern), key=lambda item: item.stat().st_mtime)
        return matches[-1] if matches else None

    manifest = result_or_search("manifest_path", "harness_manifest.json")
    if manifest is None and harness_project is not None:
        candidate = harness_project / "harness_manifest.json"
        manifest = candidate if candidate.is_file() else None
    bitstream = result_or_search("bitstream_path", "bitstream/*.bit")
    timing = result_or_search("timing_report_path", "reports/timing_summary.rpt")
    utilization = result_or_search("utilization_report_path", "reports/utilization.rpt")
    measurement = _path_or_none(result.get("measurement_json_path"))
    if measurement is None or not measurement.is_file():
        candidate = output_dir / "measurements" / f"{row['combo_case_name']}.measurement.json"
        measurement = candidate if candidate.is_file() else None
    stdout_log = _path_or_none(result.get("stdout_log"))
    if stdout_log is None or not stdout_log.is_file():
        candidate = output_dir / "logs" / "measurement.out.log"
        stdout_log = candidate if candidate.is_file() else None
    vivado_log = output_dir / "logs" / "vivado_bitstream.out.log"
    return {
        "output_dir": output_dir,
        "harness_project": harness_project,
        "manifest": manifest,
        "bitstream": bitstream,
        "timing": timing,
        "utilization": utilization,
        "measurement": measurement,
        "measurement_csv": measurement_csv_path(row, output_dir),
        "stdout_log": stdout_log,
        "vivado_log": vivado_log if vivado_log.is_file() else None,
    }


def collect_profile_summary(profile: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    results = fc.load_results()
    records: list[dict[str, Any]] = []
    rows_by_combo = {row["combo_case_name"]: row for row in rows}
    for combo, row in sorted(rows_by_combo.items()):
        result = results.get(combo, {})
        artifacts = artifacts_for_row(row, result)
        timing_path = artifacts["timing"]
        wns = _float(result.get("wns_ns"))
        if wns is None and timing_path is not None:
            wns = _float(fc.parse_wns(timing_path))
        status_code = _int(result.get("uart_status_code"))
        bitstream_path = artifacts["bitstream"]
        measurement_path = artifacts["measurement"]
        cycles = _int(result.get("board_combo_cycles"))
        latency_ms = _float(result.get("board_combo_latency_ms"))
        status = str(result.get("status", "missing"))
        effective_bitstream_status = str(result.get("bitstream_status", ""))
        if bitstream_path is not None and bitstream_path.is_file() and effective_bitstream_status not in {"success", "skipped_existing"}:
            effective_bitstream_status = "artifact_present_after_timeout"
        records.append(
            {
                "combo_case_name": combo,
                "decomposition": result.get("decomposition", ""),
                "status": status,
                "bitstream_status": result.get("bitstream_status", ""),
                "effective_bitstream_status": effective_bitstream_status,
                "bitstream_exists": bool(bitstream_path is not None and bitstream_path.is_file()),
                "uart_status_code": status_code,
                "cycles": cycles,
                "latency_ms": latency_ms,
                "wns_ns": wns,
                "timing_clean": bool(wns is not None and wns >= 0.0),
                "strict_board_usable": bool(
                    result.get("status") == "board_measured_timing_clean"
                    and status_code == 0
                    and wns is not None
                    and wns >= 0.0
                    and measurement_path is not None
                    and measurement_path.is_file()
                ),
                "measurement_json_path": _path_text(measurement_path),
                "bitstream_path": _path_text(bitstream_path),
                "timing_report_path": _path_text(timing_path),
                "stdout_log": _path_text(artifacts["stdout_log"]),
                "vivado_log": _path_text(artifacts["vivado_log"]),
            }
        )
    counts = Counter(record["status"] for record in records)
    summary = {
        "generated_at": timestamp(),
        "profile": profile,
        "clock_mhz": PROFILE_CONFIG[profile]["clock_mhz"],
        "period_ns": PROFILE_CONFIG[profile]["period_ns"],
        "board_config": str(PROFILE_CONFIG[profile]["board_config"]),
        "output_dir": str(fc.OUT_DIR),
        "selection_count": len(records),
        "counts": dict(counts),
        "bitstream_exists": sum(1 for record in records if record["bitstream_exists"]),
        "implementation_timing_clean": sum(1 for record in records if record["timing_clean"]),
        "uart_status_code_0": sum(1 for record in records if record["uart_status_code"] == 0),
        "strict_board_usable": sum(1 for record in records if record["strict_board_usable"]),
        "records": records,
    }
    out = Path(fc.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"multiclock_{profile}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = out / f"multiclock_{profile}_records.csv"
    fieldnames = [
        "combo_case_name",
        "decomposition",
        "status",
        "bitstream_status",
        "effective_bitstream_status",
        "bitstream_exists",
        "uart_status_code",
        "cycles",
        "latency_ms",
        "wns_ns",
        "timing_clean",
        "strict_board_usable",
        "measurement_json_path",
        "bitstream_path",
        "timing_report_path",
        "stdout_log",
        "vivado_log",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
    return summary


def run_rows(rows: list[dict[str, str]], args: argparse.Namespace, *, skip_measure: bool) -> None:
    fc_args = make_fc_args(args, skip_measure=skip_measure)
    for batch_idx in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[batch_idx : batch_idx + max(1, args.batch_size)]
        if skip_measure and not args.force:
            current_results = fc.load_results()
            batch = [
                row
                for row in batch
                if not (
                    (artifacts := artifacts_for_row(row, current_results.get(row["combo_case_name"], {})))
                    and artifacts["bitstream"] is not None
                    and artifacts["timing"] is not None
                )
            ]
        if not batch:
            collect_profile_summary(args.profile, rows)
            continue
        fc.run_rows(batch, fc_args, full=True)
        collect_profile_summary(args.profile, rows)


def measure_existing_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    fc_args = make_fc_args(args, skip_measure=False)
    fc.assert_serial_port(args.serial_port)
    results = fc.load_results()
    for row in rows:
        combo = row["combo_case_name"]
        output_dir = row_output_dir(row)
        logs_dir = output_dir / "logs"
        measurements_dir = output_dir / "measurements"
        logs_dir.mkdir(parents=True, exist_ok=True)
        measurements_dir.mkdir(parents=True, exist_ok=True)
        result_payload = dict(results.get(combo, {}))
        artifacts = artifacts_for_row(row, result_payload)
        if (
            result_payload.get("status") == "board_measured_timing_clean"
            and artifacts["measurement"] is not None
            and not args.force
        ):
            continue
        manifest = artifacts["manifest"]
        bitstream = artifacts["bitstream"]
        timing = artifacts["timing"]
        if manifest is None or bitstream is None or timing is None:
            result_payload.update(
                {
                    "combo_case_name": combo,
                    "decomposition": row.get("decomposition", ""),
                    "status": "measurement_blocked_missing_artifact",
                    "board_execution_status": "blocked",
                    "failure_category": "missing_existing_artifact",
                    "failure_reason": "Existing manifest, bitstream, or timing report was not found.",
                    "updated_at": timestamp(),
                }
            )
            fc.write_result(combo, result_payload)
            continue
        json_out = measurements_dir / f"{combo}.measurement.json"
        measurement_csv = artifacts["measurement_csv"]
        measurement = fc.measure_harness(
            manifest=manifest,
            bitstream=bitstream,
            csv_path=measurement_csv,
            json_out=json_out,
            serial_port=args.serial_port,
            vivado=Path(fc_args.vivado).expanduser().resolve(),
            hw_server_url=args.hw_server_url,
            hw_device_pattern=args.hw_device_pattern,
            hw_target_index=args.hw_target_index,
            program_retries=args.program_retries,
            runs=args.runs,
            uart_timeout_seconds=args.uart_timeout_seconds,
            post_program_delay_seconds=args.post_program_delay_seconds,
            timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
            skip_measure=False,
        )
        (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
        measurement_payload = fc.load_json(json_out)
        wns = fc.parse_wns(timing)
        status, board_execution, failure_category, failure_reason, timing_clean = fc.classify_measurement(
            measurement_payload,
            wns,
        )
        measurement_payload.update(
            {
                "combo_case_name": combo,
                "combo_decomposition": row.get("decomposition", ""),
                "full_combo_latency_claim": claim_for_row(row),
                "board_timing_wns_ns": wns,
                "board_timing_clean": timing_clean,
                "clock_profile": args.profile,
                "updated_at": timestamp(),
            }
        )
        if row.get("decomposition") == fc.MBCONV_DECOMPOSITION:
            measurement_payload["stitched_topology"] = fc.MBCONV_TOPOLOGY
        json_out.write_text(json.dumps(measurement_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        util = fc.parse_utilization(artifacts["utilization"]) if artifacts["utilization"] is not None else {}
        result_payload.update(
            {
                "combo_case_name": combo,
                "decomposition": row.get("decomposition", ""),
                "completed_at": timestamp(),
                "status": status,
                "board_execution_status": board_execution,
                "failure_category": failure_category,
                "failure_reason": failure_reason,
                "measurement_status": measurement.get("status", ""),
                "uart_status_code": fc.frame_value(measurement_payload, "status_code"),
                "uart_status_label": measurement_payload.get("status_label", ""),
                "board_combo_cycles": fc.frame_value(measurement_payload, "cycles"),
                "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
                "board_word_count": fc.frame_value(measurement_payload, "word_count"),
                "board_checksum": fc.frame_value(measurement_payload, "checksum"),
                "wns_ns": wns,
                "timing_clean": timing_clean,
                "measurement_json_path": str(json_out),
                "measurement_csv_path": str(measurement_csv),
                "bitstream_path": str(bitstream),
                "harness_project_dir": str(artifacts["harness_project"] or bitstream.parent.parent),
                "timing_report_path": str(timing),
                "utilization_report_path": str(artifacts["utilization"] or ""),
                "stdout_log": str(logs_dir / "measurement.out.log"),
                "stderr_log": "",
                "clock_profile": args.profile,
            }
        )
        result_payload.update(util)
        fc.write_result(combo, result_payload)
        collect_profile_summary(args.profile, rows)


def main() -> int:
    args = parse_args()
    profile_dir = configure_profile_outputs(args.profile)
    fc.assert_inputs()
    rows = selected_runlist_rows(args)

    if args.mode == "plan":
        summary = collect_profile_summary(args.profile, rows)
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "output_dir": str(profile_dir),
                    "selection": args.selection,
                    "selected": [row["combo_case_name"] for row in rows],
                    "summary": {
                        key: summary[key]
                        for key in (
                            "selection_count",
                            "bitstream_exists",
                            "implementation_timing_clean",
                            "uart_status_code_0",
                            "strict_board_usable",
                        )
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.mode == "implement":
        run_rows(rows, args, skip_measure=True)
    elif args.mode == "measure":
        fc.assert_serial_port(args.serial_port)
        run_rows(rows, args, skip_measure=False)
    elif args.mode == "measure-existing":
        measure_existing_rows(rows, args)
    elif args.mode == "refresh":
        fc.build_outputs("multiclock-refresh")

    summary = collect_profile_summary(args.profile, rows)
    print(
        f"profile={args.profile} selected={summary['selection_count']} "
        f"bitstreams={summary['bitstream_exists']} "
        f"timing_clean={summary['implementation_timing_clean']} "
        f"uart_ok={summary['uart_status_code_0']} "
        f"strict_board_usable={summary['strict_board_usable']} "
        f"summary={profile_dir / f'multiclock_{args.profile}_summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
