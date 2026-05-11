#!/usr/bin/env python3
"""Batch operator board measurement for exported HLS pilot cases."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
RESULTS_DIR = ROOT / "hls_lut_builder" / "results"
DEFAULT_CASES_CSV = RESULTS_DIR / "stageB_core_pilot_vivado.csv"
DEFAULT_BOARD_CONFIG = BOARD_HARNESS_DIR / "configs" / "board_av7k325.yaml"
DEFAULT_RESULTS_CSV = BOARD_HARNESS_DIR / "results" / "board_measured_lut.csv"
DEFAULT_SUMMARY_JSON = BOARD_HARNESS_DIR / "results" / "batch_measure_summary.json"
DEFAULT_VITIS_HLS = Path(r"F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat")
DEFAULT_VIVADO = Path(r"F:\vivado\Vivado\2023.2\bin\vivado.bat")
EXPORT_SCRIPT = ROOT / "hls_lut_builder" / "scripts" / "export_hls_ip.py"
GENERATE_HARNESS_SCRIPT = BOARD_HARNESS_DIR / "scripts" / "generate_harness.py"
MEASURE_SCRIPT = BOARD_HARNESS_DIR / "scripts" / "measure_op.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch board-measure exported HLS operator cases")
    parser.add_argument("--cases-csv", default=str(DEFAULT_CASES_CSV), help="CSV with case_name column")
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG), help="Board YAML config")
    parser.add_argument("--serial-port", required=True, help="UART serial port, e.g. COM5")
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS), help="Path to vitis_hls launcher")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO), help="Path to vivado launcher")
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV), help="Board measurement CSV")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON), help="Batch summary JSON")
    parser.add_argument("--runs", type=int, default=1, help="Repeated UART measurements per case")
    parser.add_argument("--case", action="append", default=None, help="Optional explicit case_name filter")
    parser.add_argument("--include-extensions", action="store_true", help="Allow p_ext cases if present in CSV/filter")
    parser.add_argument("--force", action="store_true", help="Re-export, re-generate harness, rebuild bitstream and re-measure")
    parser.add_argument("--csynth-timeout-minutes", type=int, default=60, help="Timeout for Vitis HLS csynth stage")
    parser.add_argument("--export-timeout-minutes", type=int, default=30, help="Timeout for HLS IP export stage")
    parser.add_argument("--harness-timeout-minutes", type=int, default=10, help="Timeout for harness generation stage")
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=90, help="Timeout for Vivado bitstream build stage")
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20, help="Timeout for board measurement stage")
    return parser.parse_args()


def load_case_names(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        case_names = [row["case_name"] for row in reader if row.get("case_name")]
    if not case_names:
        raise ValueError(f"No case_name entries found in {csv_path}")
    return case_names


def load_measured_cases(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    measured: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                if int(row.get("status_code", "1")) == 0 and int(row.get("cycles", "0")) > 0:
                    case_name = row.get("case_name")
                    if case_name:
                        measured.add(case_name)
            except ValueError:
                continue
    return measured


def resolve_case_dir(case_name: str, include_extensions: bool) -> Path:
    candidates = [RESULTS_DIR / "p" / case_name]
    if include_extensions:
        candidates.append(RESULTS_DIR / "p_ext" / case_name)
    else:
        candidates.extend([RESULTS_DIR / "p_ext" / case_name, RESULTS_DIR / "p" / case_name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find case directory for {case_name}")


def _coerce_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int | None = None,
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, Any] | None]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
        return result, None
    except subprocess.TimeoutExpired as exc:
        return None, {
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "stdout": _coerce_output(exc.stdout),
            "stderr": _coerce_output(exc.stderr),
        }


def write_summary(summary_json: Path, summary: list[dict[str, Any]]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _has_csynth_artifacts(case_dir: Path) -> bool:
    report_dir = case_dir / "project" / "solution1" / "syn" / "report"
    if not report_dir.exists():
        return False
    if any(report_dir.glob("*_csynth.xml")):
        return True
    if any(report_dir.glob("*_csynth.rpt")):
        return True
    if (report_dir / "csynth.xml").exists():
        return True
    if (report_dir / "csynth.rpt").exists():
        return True
    return False


def ensure_csynth(case_dir: Path, vitis_hls: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    synth_tcl = case_dir / "synth.tcl"
    if not synth_tcl.exists():
        return {
            "status": "failed",
            "returncode": None,
            "stdout": f"synth.tcl not found: {synth_tcl}",
        }
    if _has_csynth_artifacts(case_dir) and not force:
        return {"status": "skipped_existing", "synth_tcl": str(synth_tcl)}
    command = ["cmd", "/c", str(vitis_hls), "-f", str(synth_tcl)]
    result, timeout_payload = run_command(command, cwd=case_dir, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        timeout_payload["synth_tcl"] = str(synth_tcl)
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and _has_csynth_artifacts(case_dir) else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "synth_tcl": str(synth_tcl),
    }


def ensure_export(case_dir: Path, vitis_hls: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    component_xml = case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"
    if component_xml.exists() and not force:
        return {"status": "skipped_existing", "component_xml": str(component_xml)}
    command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--case-dir",
        str(case_dir),
        "--vitis-hls",
        str(vitis_hls),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        timeout_payload["component_xml"] = str(component_xml) if component_xml.exists() else None
        return timeout_payload
    status = {
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "component_xml": str(component_xml) if component_xml.exists() else None,
    }
    return status


def generate_harness(case_dir: Path, board_config: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(GENERATE_HARNESS_SCRIPT),
        "--case-dir",
        str(case_dir),
        "--board-config",
        str(board_config),
        "--output-dir",
        str(BOARD_HARNESS_DIR / "projects"),
    ]
    if force:
        command.append("--force")
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    manifest_path = BOARD_HARNESS_DIR / "projects" / f"harness_{case_dir.name}" / "harness_manifest.json"
    if timeout_payload is not None:
        timeout_payload["manifest"] = str(manifest_path) if manifest_path.exists() else None
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and manifest_path.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "manifest": str(manifest_path) if manifest_path.exists() else None,
    }


def build_bitstream(harness_project_dir: Path, vivado: Path, force: bool, timeout_seconds: int | None) -> dict[str, Any]:
    manifest = json.loads((harness_project_dir / "harness_manifest.json").read_text(encoding="utf-8"))
    case_name = manifest["case_name"]
    bitstream = harness_project_dir / "bitstream" / f"{case_name}.bit"
    if bitstream.exists() and not force:
        return {"status": "skipped_existing", "bitstream": str(bitstream)}
    tcl_path = harness_project_dir / "scripts" / "build_bitstream.tcl"
    command = ["cmd", "/c", str(vivado), "-mode", "batch", "-source", str(tcl_path)]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        timeout_payload["bitstream"] = str(bitstream) if bitstream.exists() else None
        return timeout_payload
    return {
        "status": "success" if result.returncode == 0 and bitstream.exists() else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "bitstream": str(bitstream) if bitstream.exists() else None,
    }


def measure_case(
    harness_project_dir: Path,
    serial_port: str,
    vivado: Path,
    results_csv: Path,
    runs: int,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    json_out = harness_project_dir / "latest_measurement.json"
    command = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--project-dir",
        str(harness_project_dir),
        "--serial-port",
        serial_port,
        "--vivado",
        str(vivado),
        "--csv",
        str(results_csv),
        "--runs",
        str(runs),
        "--json-out",
        str(json_out),
    ]
    result, timeout_payload = run_command(command, cwd=ROOT, timeout_seconds=timeout_seconds)
    if timeout_payload is not None:
        timeout_payload["measurement"] = None
        return timeout_payload
    payload: dict[str, Any] | None = None
    if json_out.exists():
        try:
            payload = json.loads(json_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    return {
        "status": "success" if result.returncode == 0 and payload is not None else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "measurement": payload,
    }


def main() -> int:
    args = parse_args()
    cases_csv = Path(args.cases_csv).expanduser().resolve()
    board_config = Path(args.board_config).expanduser().resolve()
    vitis_hls = Path(args.vitis_hls).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    results_csv = Path(args.results_csv).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()

    case_names = load_case_names(cases_csv)
    if args.case:
        requested = set(args.case)
        case_names = [case_name for case_name in case_names if case_name in requested]
        missing = requested.difference(case_names)
        if missing:
            print(f"[warn] Requested cases not present in CSV: {sorted(missing)}")
    measured_cases = load_measured_cases(results_csv)

    summary: list[dict[str, Any]] = []
    for case_name in case_names:
        case_dir = resolve_case_dir(case_name, include_extensions=args.include_extensions)
        harness_project_dir = BOARD_HARNESS_DIR / "projects" / f"harness_{case_name}"
        case_status: dict[str, Any] = {
            "case_name": case_name,
            "case_dir": str(case_dir),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        if case_name in measured_cases and not args.force:
            case_status["status"] = "skipped_measured"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[skip] {case_name} already has a valid board measurement")
            continue

        csynth_status = ensure_csynth(
            case_dir,
            vitis_hls,
            force=args.force,
            timeout_seconds=args.csynth_timeout_minutes * 60 if args.csynth_timeout_minutes > 0 else None,
        )
        case_status["csynth"] = csynth_status
        if csynth_status["status"] == "failed":
            case_status["status"] = "failed_csynth"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} csynth")
            continue
        if csynth_status["status"] == "timeout":
            case_status["status"] = "failed_csynth_timeout"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} csynth timeout")
            continue

        export_status = ensure_export(
            case_dir,
            vitis_hls,
            force=args.force,
            timeout_seconds=args.export_timeout_minutes * 60 if args.export_timeout_minutes > 0 else None,
        )
        case_status["export"] = export_status
        if export_status["status"] == "failed":
            case_status["status"] = "failed_export"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} export")
            continue
        if export_status["status"] == "timeout":
            case_status["status"] = "failed_export_timeout"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} export timeout")
            continue

        harness_status = generate_harness(
            case_dir,
            board_config,
            force=args.force,
            timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
        )
        case_status["harness"] = harness_status
        if harness_status["status"] == "failed":
            case_status["status"] = "failed_harness"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} harness generation")
            continue
        if harness_status["status"] == "timeout":
            case_status["status"] = "failed_harness_timeout"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} harness generation timeout")
            continue

        bitstream_status = build_bitstream(
            harness_project_dir,
            vivado,
            force=args.force,
            timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
        )
        case_status["bitstream"] = bitstream_status
        if bitstream_status["status"] == "failed":
            case_status["status"] = "failed_bitstream"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} bitstream build")
            continue
        if bitstream_status["status"] == "timeout":
            case_status["status"] = "failed_bitstream_timeout"
            summary.append(case_status)
            write_summary(summary_json, summary)
            print(f"[fail] {case_name} bitstream build timeout")
            continue

        measurement_status = measure_case(
            harness_project_dir,
            serial_port=args.serial_port,
            vivado=vivado,
            results_csv=results_csv,
            runs=args.runs,
            timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        )
        case_status["measurement"] = measurement_status
        if measurement_status["status"] == "timeout":
            case_status["status"] = "failed_measurement_timeout"
        else:
            case_status["status"] = "success" if measurement_status["status"] == "success" else "failed_measurement"
        summary.append(case_status)
        write_summary(summary_json, summary)
        print(f"[{case_status['status']}] {case_name}")

    write_summary(summary_json, summary)
    print(f"Wrote batch summary: {summary_json}")
    failed = [item for item in summary if item.get("status", "").startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
