#!/usr/bin/env python3
"""Program a board harness bitstream and collect one UART result frame."""

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


@dataclass
class VivadoHardwareRun:
    returncode: int
    stdout: str
    hw_server_url: str
    hw_device_pattern: str
    hw_target_index: int
    hw_targets: list[str]
    hw_devices: list[dict[str, str]]
    selected_hw_target: str | None = None
    selected_hw_device: str | None = None
    bitstream: str | None = None
    attempt: int = 1


class VivadoProgrammingError(RuntimeError):
    def __init__(self, message: str, run: VivadoHardwareRun | None = None) -> None:
        super().__init__(message)
        self.run = run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a board harness over UART")
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
    parser.add_argument(
        "--hw-server-url",
        default="TCP:localhost:3121",
        help="Vivado hardware server URL used by connect_hw_server",
    )
    parser.add_argument(
        "--hw-device-pattern",
        default="*xc7k325t*",
        help="Pattern passed to get_hw_devices before falling back to all devices",
    )
    parser.add_argument("--hw-target-index", type=int, default=0, help="Index into get_hw_targets output")
    parser.add_argument(
        "--program-retries",
        type=int,
        default=2,
        help="Number of FPGA programming retries after the first failed attempt",
    )
    parser.add_argument(
        "--list-hw-targets",
        action="store_true",
        help="List Vivado hardware targets/devices as JSON and exit",
    )
    parser.add_argument(
        "--program-only",
        action="store_true",
        help="Program the bitstream and exit without opening UART or appending CSV rows",
    )
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
        raise ValueError(
            "Either --project-dir or --manifest is required unless --list-ports or --list-hw-targets is used."
        )
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
        3: "watchdog_timeout",
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


def _tcl_word(value: str | Path) -> str:
    return str(value).replace("\\", "/").replace("}", "\\}")


def _parse_vivado_hardware_stdout(
    stdout: str,
    *,
    hw_server_url: str,
    hw_device_pattern: str,
    hw_target_index: int,
    bitstream: Path | None = None,
    returncode: int = 0,
    attempt: int = 1,
) -> VivadoHardwareRun:
    targets: list[str] = []
    devices: list[dict[str, str]] = []
    selected_target: str | None = None
    selected_device: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("CODEX_HW_TARGET\t"):
            targets.append(line.split("\t", 1)[1])
        elif line.startswith("CODEX_HW_SELECTED_TARGET\t"):
            selected_target = line.split("\t", 1)[1]
        elif line.startswith("CODEX_HW_DEVICE\t"):
            parts = line.split("\t")
            device: dict[str, str] = {"name": parts[1] if len(parts) > 1 else ""}
            if len(parts) > 2:
                device["part"] = parts[2]
            devices.append(device)
        elif line.startswith("CODEX_HW_SELECTED_DEVICE\t"):
            selected_device = line.split("\t", 1)[1]
    return VivadoHardwareRun(
        returncode=returncode,
        stdout=stdout,
        hw_server_url=hw_server_url,
        hw_device_pattern=hw_device_pattern,
        hw_target_index=hw_target_index,
        hw_targets=targets,
        hw_devices=devices,
        selected_hw_target=selected_target,
        selected_hw_device=selected_device,
        bitstream=str(bitstream) if bitstream else None,
        attempt=attempt,
    )


def generate_list_hardware_tcl(hw_server_url: str, hw_target_index: int) -> str:
    return f"""
open_hw_manager
connect_hw_server -url {{{_tcl_word(hw_server_url)}}}
set targets [get_hw_targets *]
foreach target $targets {{
    puts "CODEX_HW_TARGET\t$target"
}}
if {{[llength $targets] > 0}} {{
    if {{{hw_target_index} >= [llength $targets]}} {{
        error "Requested hw target index {hw_target_index}, but only [llength $targets] target(s) are visible."
    }}
    current_hw_target [lindex $targets {hw_target_index}]
    puts "CODEX_HW_SELECTED_TARGET\t[current_hw_target]"
    open_hw_target [current_hw_target]
    set devices [get_hw_devices *]
    foreach device $devices {{
        set part ""
        catch {{set part [get_property PART $device]}}
        puts "CODEX_HW_DEVICE\t$device\t$part"
    }}
}}
close_hw_manager
exit
""".strip()


def generate_program_tcl(bitstream: Path, hw_server_url: str, hw_device_pattern: str, hw_target_index: int) -> str:
    bit_path = _tcl_word(bitstream)
    return f"""
open_hw_manager
connect_hw_server -url {{{_tcl_word(hw_server_url)}}}
set targets [get_hw_targets *]
foreach target $targets {{
    puts "CODEX_HW_TARGET\t$target"
}}
if {{[llength $targets] == 0}} {{
    error "No hardware targets detected from get_hw_targets."
}}
if {{{hw_target_index} >= [llength $targets]}} {{
    error "Requested hw target index {hw_target_index}, but only [llength $targets] target(s) are visible."
}}
current_hw_target [lindex $targets {hw_target_index}]
puts "CODEX_HW_SELECTED_TARGET\t[current_hw_target]"
open_hw_target [current_hw_target]
set devices [get_hw_devices {{{_tcl_word(hw_device_pattern)}}}]
if {{[llength $devices] == 0}} {{
    set devices [get_hw_devices]
}}
foreach device $devices {{
    set part ""
    catch {{set part [get_property PART $device]}}
    puts "CODEX_HW_DEVICE\t$device\t$part"
}}
if {{[llength $devices] == 0}} {{
    error "No hardware device detected."
}}
current_hw_device [lindex $devices 0]
puts "CODEX_HW_SELECTED_DEVICE\t[current_hw_device]"
refresh_hw_device [current_hw_device]
set_property PROGRAM.FILE {{{bit_path}}} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
close_hw_manager
exit
""".strip()


def _run_vivado_tcl(
    vivado_exe: Path,
    tcl_text: str,
    *,
    hw_server_url: str,
    hw_device_pattern: str,
    hw_target_index: int,
    bitstream: Path | None = None,
    attempt: int = 1,
) -> VivadoHardwareRun:
    if not vivado_exe.exists():
        raise FileNotFoundError(f"Vivado executable not found: {vivado_exe}")
    with tempfile.TemporaryDirectory(prefix="codex_program_fpga_") as tmp_dir:
        tcl_path = Path(tmp_dir) / "vivado_hw.tcl"
        tcl_path.write_text(tcl_text, encoding="utf-8")
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
    return _parse_vivado_hardware_stdout(
        result.stdout,
        hw_server_url=hw_server_url,
        hw_device_pattern=hw_device_pattern,
        hw_target_index=hw_target_index,
        bitstream=bitstream,
        returncode=result.returncode,
        attempt=attempt,
    )


def list_vivado_hardware(vivado_exe: Path, hw_server_url: str, hw_target_index: int) -> VivadoHardwareRun:
    return _run_vivado_tcl(
        vivado_exe,
        generate_list_hardware_tcl(hw_server_url, hw_target_index),
        hw_server_url=hw_server_url,
        hw_device_pattern="*",
        hw_target_index=hw_target_index,
    )


def program_fpga(
    vivado_exe: Path,
    bitstream: Path,
    *,
    hw_server_url: str,
    hw_device_pattern: str,
    hw_target_index: int,
    program_retries: int,
) -> VivadoHardwareRun:
    attempts = max(0, program_retries) + 1
    last_run: VivadoHardwareRun | None = None
    for attempt in range(1, attempts + 1):
        run = _run_vivado_tcl(
            vivado_exe,
            generate_program_tcl(bitstream, hw_server_url, hw_device_pattern, hw_target_index),
            hw_server_url=hw_server_url,
            hw_device_pattern=hw_device_pattern,
            hw_target_index=hw_target_index,
            bitstream=bitstream,
            attempt=attempt,
        )
        last_run = run
        if run.returncode == 0:
            return run
        if attempt < attempts:
            time.sleep(1.0)
    raise VivadoProgrammingError(f"Vivado programming failed after {attempts} attempt(s).", last_run) from None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_summary(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def failure_payload(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    run = exc.run if isinstance(exc, VivadoProgrammingError) else None
    payload: dict[str, Any] = {
        "status": "failed",
        "failure_category": "vivado_programming_failed" if run is not None else "measurement_runtime_failed",
        "failure_reason": str(exc),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    if run is not None:
        payload["vivado_hardware"] = asdict(run)
        payload["vivado_stdout"] = run.stdout
        payload["hw_targets"] = run.hw_targets
        payload["hw_devices"] = run.hw_devices
    if getattr(args, "manifest", None):
        payload["manifest"] = str(Path(args.manifest).expanduser())
    if getattr(args, "project_dir", None):
        payload["project_dir"] = str(Path(args.project_dir).expanduser())
    if getattr(args, "bitstream", None):
        payload["bitstream"] = str(Path(args.bitstream).expanduser())
    return payload


def _run_main(args: argparse.Namespace) -> int:
    if args.list_ports:
        return list_ports()

    vivado_exe = Path(args.vivado).expanduser().resolve()
    if args.list_hw_targets:
        run = list_vivado_hardware(vivado_exe, args.hw_server_url, args.hw_target_index)
        payload = {"status": "ok" if run.returncode == 0 else "failed", "vivado_hardware": asdict(run)}
        if args.json_out:
            write_json_summary(Path(args.json_out).expanduser().resolve(), payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0 if run.returncode == 0 else 1

    manifest_path = resolve_manifest(args)
    manifest = load_json(manifest_path)
    project_root = Path(manifest["generated"]["project_root"]).resolve()
    board_config_path = Path(args.board_config).expanduser().resolve() if args.board_config else Path(manifest["board_config"]).resolve()
    board_cfg = load_yaml(board_config_path)
    bitstream_path = resolve_bitstream(project_root, manifest["case_name"], args.bitstream)

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.program_retries < 0:
        raise ValueError("--program-retries must be >= 0")
    if args.skip_program and args.runs > 1:
        raise ValueError("--skip-program cannot be combined with --runs > 1 for the one-shot harness.")

    if args.program_only:
        run = program_fpga(
            vivado_exe,
            bitstream_path,
            hw_server_url=args.hw_server_url,
            hw_device_pattern=args.hw_device_pattern,
            hw_target_index=args.hw_target_index,
            program_retries=args.program_retries,
        )
        payload = {
            "status": "programmed",
            "program_only": True,
            "project_root": str(project_root),
            "bitstream": str(bitstream_path),
            "vivado_hardware": asdict(run),
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        }
        if args.json_out:
            write_json_summary(Path(args.json_out).expanduser().resolve(), payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    require_serial()
    uart_baud = args.baud or int(board_cfg["uart"]["baud"])
    serial_port_name = args.serial_port
    if not serial_port_name:
        raise ValueError("A serial port is required. Pass --serial-port COMx.")

    clock_freq_hz = int(board_cfg["clock"]["freq_hz"])
    rows: list[dict[str, Any]] = []
    latest_summary: dict[str, Any] | None = None

    with serial.Serial(serial_port_name, uart_baud, timeout=0.2) as uart:
        uart.reset_input_buffer()
        uart.reset_output_buffer()

        for run_idx in range(args.runs):
            programming_run: VivadoHardwareRun | None = None
            if not args.skip_program:
                uart.reset_input_buffer()
                programming_run = program_fpga(
                    vivado_exe,
                    bitstream_path,
                    hw_server_url=args.hw_server_url,
                    hw_device_pattern=args.hw_device_pattern,
                    hw_target_index=args.hw_target_index,
                    program_retries=args.program_retries,
                )
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
                "vivado_hardware": asdict(programming_run) if programming_run is not None else None,
            }
            print(json.dumps(latest_summary, ensure_ascii=True, indent=2))

    append_rows(Path(args.csv).expanduser().resolve(), rows)
    if args.json_out and latest_summary is not None:
        write_json_summary(Path(args.json_out).expanduser().resolve(), latest_summary)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return _run_main(args)
    except Exception as exc:
        if getattr(args, "json_out", None):
            write_json_summary(Path(args.json_out).expanduser().resolve(), failure_payload(args, exc))
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
