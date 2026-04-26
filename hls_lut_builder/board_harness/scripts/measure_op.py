#!/usr/bin/env python3
"""Program a single-operator board harness bitstream and collect one UART result frame."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import serial
    import serial.tools.list_ports
except ImportError as exc:  # pragma: no cover - environment dependent
    serial = None  # type: ignore[assignment]
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[3]
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
DEFAULT_RESULTS_CSV = BOARD_HARNESS_DIR / "results" / "board_measured_lut.csv"
DEFAULT_VIVADO = Path(r"F:\vivado\Vivado\2023.2\bin\vivado.bat")
FRAME_MAGIC = bytes((0xA5, 0x5A))
FRAME_SIZE = 15


@dataclass
class MeasurementFrame:
    status_code: int
    cycles: int
    checksum: int
    word_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a single operator board harness over UART")
    parser.add_argument("--project-dir", help="Harness project directory containing harness_manifest.json")
    parser.add_argument("--manifest", help="Explicit harness_manifest.json path")
    parser.add_argument("--board-config", help="Override board config YAML path")
    parser.add_argument("--bitstream", help="Override bitstream path")
    parser.add_argument("--serial-port", help="Serial port such as COM3")
    parser.add_argument("--baud", type=int, help="Override UART baud")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated measurements")
    parser.add_argument("--timeout-seconds", type=float, default=15.0, help="UART frame timeout")
    parser.add_argument(
        "--post-program-delay-seconds",
        type=float,
        default=0.25,
        help="Delay after programming before waiting for UART bytes",
    )
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO), help="Vivado batch executable path")
    parser.add_argument("--skip-program", action="store_true", help="Do not reprogram the FPGA before reading")
    parser.add_argument("--csv", default=str(DEFAULT_RESULTS_CSV), help="Append measurement rows to CSV")
    parser.add_argument("--json-out", help="Optional JSON file for the latest run summary")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
    elif args.project_dir:
        manifest_path = Path(args.project_dir).expanduser().resolve() / "harness_manifest.json"
    else:
        raise ValueError("Either --project-dir or --manifest is required unless --list-ports is used.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Harness manifest not found: {manifest_path}")
    return manifest_path


def resolve_bitstream(project_root: Path, case_name: str, override: str | None) -> Path:
    if override:
        bitstream = Path(override).expanduser().resolve()
    else:
        bitstream = project_root / "bitstream" / f"{case_name}.bit"
    if not bitstream.exists():
        raise FileNotFoundError(f"Bitstream not found: {bitstream}")
    return bitstream


def status_label(status_code: int) -> str:
    return {
        0: "ok",
        1: "sink_incomplete",
        2: "control_error",
    }.get(status_code, f"unknown_{status_code}")


def parse_frame(frame: bytes) -> MeasurementFrame:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"Expected {FRAME_SIZE} bytes, got {len(frame)}")
    if frame[:2] != FRAME_MAGIC:
        raise ValueError(f"Bad frame header: {frame[:2].hex()}")
    status_code = frame[2]
    cycles = struct.unpack("<I", frame[3:7])[0]
    checksum = struct.unpack("<I", frame[7:11])[0]
    word_count = struct.unpack("<I", frame[11:15])[0]
    return MeasurementFrame(status_code=status_code, cycles=cycles, checksum=checksum, word_count=word_count)


def wait_for_frame(port: "serial.Serial", timeout_seconds: float) -> MeasurementFrame:
    deadline = time.monotonic() + timeout_seconds
    prefix = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(1)
        if not chunk:
            continue
        prefix += chunk
        if len(prefix) > 2:
            prefix = prefix[-2:]
        if bytes(prefix) == FRAME_MAGIC:
            remainder = bytearray()
            while len(remainder) < FRAME_SIZE - 2 and time.monotonic() < deadline:
                needed = FRAME_SIZE - 2 - len(remainder)
                block = port.read(needed)
                if not block:
                    continue
                remainder.extend(block)
            if len(remainder) != FRAME_SIZE - 2:
                raise TimeoutError("Timed out while reading the remainder of the UART result frame.")
            return parse_frame(FRAME_MAGIC + remainder)
    raise TimeoutError("Timed out waiting for UART result frame header.")


def require_serial() -> None:
    if serial is None:
        raise RuntimeError(
            "pyserial is required for board measurement but is not installed. "
            "Install it with `python -m pip install pyserial`."
        ) from SERIAL_IMPORT_ERROR


def list_ports() -> int:
    require_serial()
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports detected.")
        return 0
    for info in ports:
        print(f"{info.device}\t{info.description}")
    return 0


def generate_program_tcl(bitstream: Path) -> str:
    bit_path = str(bitstream).replace("\\", "/")
    return f"""
open_hw_manager
connect_hw_server
open_hw_target
set devices [get_hw_devices *xc7k325t*]
if {{[llength $devices] == 0}} {{
    set devices [get_hw_devices]
}}
if {{[llength $devices] == 0}} {{
    error "No hardware device detected."
}}
current_hw_device [lindex $devices 0]
refresh_hw_device [current_hw_device]
set_property PROGRAM.FILE {{{bit_path}}} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
close_hw_manager
exit
""".strip()


def program_fpga(vivado_exe: Path, bitstream: Path) -> None:
    if not vivado_exe.exists():
        raise FileNotFoundError(f"Vivado executable not found: {vivado_exe}")
    with tempfile.TemporaryDirectory(prefix="codex_program_fpga_") as tmp_dir:
        tcl_path = Path(tmp_dir) / "program_fpga.tcl"
        tcl_path.write_text(generate_program_tcl(bitstream), encoding="utf-8")
        result = subprocess.run(
            [str(vivado_exe), "-mode", "batch", "-source", str(tcl_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Vivado programming failed:\n{result.stdout}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["case_name"]), str(row["module_name"]), str(row["serial_port"]))


def _is_valid_row(row: dict[str, Any]) -> bool:
    try:
        return int(row["status_code"]) == 0 and int(row["cycles"]) > 0
    except (KeyError, TypeError, ValueError):
        return False


def append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(csv_path)
    existing_rows: list[dict[str, Any]] = []
    fieldnames = list(rows[0].keys())

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            existing_rows = list(reader)

    for row in rows:
        if _is_valid_row(row):
            key = _row_key(row)
            existing_rows = [existing for existing in existing_rows if _row_key(existing) != key]
        existing_rows.append({key: str(value) for key, value in row.items()})

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.list_ports:
        return list_ports()

    require_serial()
    manifest_path = resolve_manifest(args)
    manifest = load_json(manifest_path)
    project_root = Path(manifest["generated"]["project_root"]).resolve()
    board_config_path = Path(args.board_config).expanduser().resolve() if args.board_config else Path(manifest["board_config"]).resolve()
    board_cfg = load_yaml(board_config_path)
    bitstream_path = resolve_bitstream(project_root, manifest["case_name"], args.bitstream)
    uart_baud = args.baud or int(board_cfg["uart"]["baud"])
    serial_port_name = args.serial_port
    if not serial_port_name:
        raise ValueError("A serial port is required. Pass --serial-port COMx.")

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.skip_program and args.runs > 1:
        raise ValueError("--skip-program cannot be combined with --runs > 1 for the one-shot harness.")

    vivado_exe = Path(args.vivado).expanduser().resolve()
    clock_freq_hz = int(board_cfg["clock"]["freq_hz"])
    rows: list[dict[str, Any]] = []
    latest_summary: dict[str, Any] | None = None

    with serial.Serial(serial_port_name, uart_baud, timeout=0.2) as uart:
        uart.reset_input_buffer()
        uart.reset_output_buffer()

        for run_idx in range(args.runs):
            if not args.skip_program:
                uart.reset_input_buffer()
                program_fpga(vivado_exe, bitstream_path)
                time.sleep(args.post_program_delay_seconds)

            frame = wait_for_frame(uart, timeout_seconds=args.timeout_seconds)
            latency_ns = frame.cycles * (1_000_000_000.0 / clock_freq_hz)
            timestamp = datetime.now(timezone.utc).astimezone().isoformat()
            row = {
                "timestamp": timestamp,
                "case_name": manifest["case_name"],
                "module_name": manifest["module_name"],
                "serial_port": serial_port_name,
                "run_idx": run_idx,
                "status_code": frame.status_code,
                "status_label": status_label(frame.status_code),
                "cycles": frame.cycles,
                "latency_ns": round(latency_ns, 3),
                "latency_ms": round(latency_ns / 1_000_000.0, 6),
                "checksum": frame.checksum,
                "word_count": frame.word_count,
                "clock_freq_hz": clock_freq_hz,
                "bitstream": str(bitstream_path),
                "board_config": str(board_config_path),
                "manifest": str(manifest_path),
            }
            rows.append(row)
            latest_summary = {
                "project_root": str(project_root),
                "bitstream": str(bitstream_path),
                "frame": asdict(frame),
                "status_label": status_label(frame.status_code),
                "latency_ns": latency_ns,
                "latency_ms": latency_ns / 1_000_000.0,
                "clock_freq_hz": clock_freq_hz,
                "serial_port": serial_port_name,
                "run_idx": run_idx,
            }
            print(json.dumps(latest_summary, ensure_ascii=False, indent=2))

    append_rows(Path(args.csv).expanduser().resolve(), rows)
    if args.json_out and latest_summary is not None:
        json_path = Path(args.json_out).expanduser().resolve()
        ensure_parent(json_path)
        json_path.write_text(json.dumps(latest_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
