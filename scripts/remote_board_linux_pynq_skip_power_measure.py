#!/usr/bin/env python3
"""Run ZCU104 Skip sampling via PYNQ buffers and board INA226 hwmon."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import posixpath
import re
import sys
import time
from pathlib import Path

import paramiko


SERVER_HOST = "10.70.23.121"
SERVER_PORT = 7889
SERVER_USER = "jiajia"
BOARD_HOSTS = ("192.168.1.100", "192.168.31.217")
BOARD_USER = "xilinx"

BASE = "/home/jiajia/experiments/zcu104_operator_lut_measurement_v1/stage4_skip_standard_wrapper_v1"
BIT = (
    BASE
    + "/repo/results/zcu104_operator_lut_measurement_v1/stage4_skip_standard_wrapper_v1"
    + "/platform/bitstream/zcu104_standard_wrapper.bit"
)
HWH = (
    BASE
    + "/repo/results/zcu104_operator_lut_measurement_v1/stage4_skip_standard_wrapper_v1"
    + "/platform/platform/zcu104_standard_wrapper.hwh"
)

_SECRET_CACHE: dict[str, str] = {}


BOARD_SCRIPT = r'''
import argparse, hashlib, json, os, statistics, time, traceback
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pynq import MMIO, Overlay, allocate


NBYTES = 25088
DMA_RESET = 0x4
DMA_RUN = 0x1
DMA_IDLE = 0x2
DMA_ERR = 0x770
AP_START = 0x1
AP_DONE = 0x2


def progress(label):
    print(f"PROGRESS {label}", flush=True)


def sha256_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def read_int(path):
    return int(Path(path).read_text().strip())


def find_ina226_hwmon():
    for name_file in sorted(Path("/sys/class/hwmon").glob("hwmon*/name")):
        try:
            if name_file.read_text().strip() == "ina226":
                return str(name_file.parent)
        except OSError:
            pass
    raise FileNotFoundError("ina226 hwmon device not found")


def sample_power(hwmon, phase, iteration=None):
    h = Path(hwmon)
    out = {
        "phase": phase,
        "timestamp_ns": time.monotonic_ns(),
        "voltage_v": read_int(h / "in2_input") / 1000.0,
        "current_a": read_int(h / "curr1_input") / 1000.0,
        "power_w": read_int(h / "power1_input") / 1_000_000.0,
    }
    if iteration is not None:
        out["iteration"] = iteration
    return out


def pack_int8_words(values):
    raw = np.ascontiguousarray(values, dtype=np.int8).view(np.uint8).tobytes()
    padded = raw + b"\x00" * ((-len(raw)) % 4)
    return np.frombuffer(padded, dtype="<u4")


def write_and_verify(mmio, words, label, method):
    expected = np.ascontiguousarray(words, dtype="<u4")
    if expected.size > mmio.array.size:
        raise RuntimeError(
            f"{label} payload exceeds MMIO window: required={expected.size} available={mmio.array.size}"
        )
    if method == "array":
        np.copyto(mmio.array[: expected.size], expected)
        observed = np.asarray(mmio.array[: expected.size], dtype="<u4").copy()
    else:
        observed = np.empty(expected.shape, dtype="<u4")
        for i, value in enumerate(expected):
            offset = i * 4
            mmio.write(offset, int(value))
        for i in range(expected.size):
            observed[i] = mmio.read(i * 4)
    mismatches = np.flatnonzero(observed != expected)
    if mismatches.size:
        idx = int(mismatches[0])
        raise RuntimeError(
            f"{label} readback mismatch at word {idx}: "
            f"observed=0x{int(observed[idx]):08x} expected=0x{int(expected[idx]):08x}"
        )


def parse_hwh_memory_map(bit_path):
    hwh_path = Path(bit_path).with_suffix(".hwh")
    root = ET.parse(hwh_path).getroot()
    result = {}
    ambiguous = set()
    for element in root.iter("MEMRANGE"):
        instance = element.attrib.get("INSTANCE")
        base = element.attrib.get("BASEVALUE")
        high = element.attrib.get("HIGHVALUE")
        if not instance or base is None or high is None:
            continue
        record = {
            "phys_addr": int(base, 0),
            "addr_range": int(high, 0) - int(base, 0) + 1,
            "type": element.attrib.get("MEMTYPE", ""),
        }
        if instance in ambiguous:
            continue
        current = result.get(instance)
        if current is not None and current != record:
            result.pop(instance, None)
            ambiguous.add(instance)
        else:
            result[instance] = record
    return result


def ip_info(overlay, hwh_map, name):
    if name in overlay.ip_dict:
        return overlay.ip_dict[name]
    if name in hwh_map:
        return hwh_map[name]
    raise KeyError(name)


def wait_set(mmio, reg, mask, timeout_s, label, check_dma_error=False):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = mmio.read(reg)
        if check_dma_error and value & DMA_ERR:
            raise RuntimeError(f"{label} error reg=0x{reg:x} value=0x{value:08x}")
        if value & mask:
            return value
    raise TimeoutError(f"{label} timeout value=0x{mmio.read(reg):08x}")


def wait_clear(mmio, reg, mask, timeout_s, label):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = mmio.read(reg)
        if not (value & mask):
            return value
    raise TimeoutError(f"{label} timeout value=0x{mmio.read(reg):08x}")


def dma_reset(dma):
    dma.write(0x00, DMA_RESET)
    wait_clear(dma, 0x00, DMA_RESET, 1.0, "mm2s reset")
    dma.write(0x30, DMA_RESET)
    wait_clear(dma, 0x30, DMA_RESET, 1.0, "s2mm reset")
    dma.write(0x00, DMA_RUN)
    dma.write(0x30, DMA_RUN)


def run_once(control, dma, in_phys, out_phys, input_bytes, output_bytes, control_mode):
    dma.write(0x48, out_phys & 0xFFFFFFFF)
    dma.write(0x4C, (out_phys >> 32) & 0xFFFFFFFF)
    dma.write(0x18, in_phys & 0xFFFFFFFF)
    dma.write(0x1C, (in_phys >> 32) & 0xFFFFFFFF)
    dma.write(0x58, output_bytes)
    if control_mode == "wrapper":
        control.write(0x00, 0x6)
    t0 = time.monotonic_ns()
    control.write(0x00, AP_START)
    dma.write(0x28, input_bytes)
    wait_set(dma, 0x04, DMA_IDLE, 1.0, "mm2s", check_dma_error=True)
    wait_set(dma, 0x34, DMA_IDLE, 1.0, "s2mm", check_dma_error=True)
    if control_mode == "wrapper":
        wait_set(control, 0x04, 0x4, 1.0, "shell")
    else:
        wait_set(control, 0x00, AP_DONE, 30.0, "hls_ap_done")
    return time.monotonic_ns() - t0


def reg_snapshot(control, dma, control_mode):
    if control_mode != "wrapper":
        return {
            "control": control.read(0x00),
            "dma_mm2s_cr": dma.read(0x00),
            "dma_mm2s_sr": dma.read(0x04),
            "dma_s2mm_cr": dma.read(0x30),
            "dma_s2mm_sr": dma.read(0x34),
        }
    labels = {
        "shell_status": 0x04,
        "exec_lo": 0x08,
        "exec_hi": 0x0C,
        "core_lo": 0x10,
        "core_hi": 0x14,
        "expected_in": 0x18,
        "expected_out": 0x1C,
        "input_seen": 0x20,
        "output_seen": 0x24,
        "err": 0x28,
        "shell_id": 0x2C,
        "version": 0x30,
        "op_id": 0x34,
        "axis_bits": 0x38,
        "clock_hz": 0x3C,
    }
    dma_labels = {
        "dma_mm2s_cr": 0x00,
        "dma_mm2s_sr": 0x04,
        "dma_s2mm_cr": 0x30,
        "dma_s2mm_sr": 0x34,
    }
    out = {name: control.read(off) for name, off in labels.items()}
    out.update({name: dma.read(off) for name, off in dma_labels.items()})
    return out


def summarize(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bit", default="/home/xilinx/zcu104_standard_wrapper.bit")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--idle-samples", type=int, default=25)
    ap.add_argument("--idle-interval-s", type=float, default=0.05)
    ap.add_argument("--operator", default="operator")
    ap.add_argument("--control-mode", choices=("wrapper", "hls_ap_ctrl"), default="wrapper")
    ap.add_argument("--param-write-method", choices=("array", "word"), default="array")
    ap.add_argument("--vectors")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress("args_parsed")
    hwmon = find_ina226_hwmon()
    progress("hwmon_found")
    power_samples = []
    latencies_ns = []
    result = {
        "schema_version": 2,
        "measurement_route": "board_linux_pynq_xrt_buffer_mmio_dma_plus_hwmon_ina226",
        "operator": args.operator,
        "started_unix_s": time.time(),
        "parameters": vars(args),
        "hwmon": hwmon,
        "claim_boundary": (
            "Exploratory full-route board sampling: safe PYNQ/XRT buffers, MMIO DMA, "
            "and internal INA226 hwmon. Treat as operator LUT measurement evidence, "
            "not external power-meter evidence."
        ),
    }

    try:
        progress("parse_hwh_start")
        hwh_map = parse_hwh_memory_map(args.bit)
        progress("parse_hwh_done")
        progress("overlay_download_start")
        overlay = Overlay(args.bit, download=True)
        progress("overlay_download_done")
        control_name = "operator_benchmark_wrapper_0" if args.control_mode == "wrapper" else "fourstage_0"
        wrap_info = ip_info(overlay, hwh_map, control_name)
        dma_info = ip_info(overlay, hwh_map, "axi_dma_0")
        progress("mmio_bind_start")
        wrap = MMIO(wrap_info["phys_addr"], wrap_info["addr_range"])
        dma = MMIO(dma_info["phys_addr"], dma_info["addr_range"])
        progress("mmio_bind_done")
        if args.control_mode == "hls_ap_ctrl":
            if not args.vectors:
                raise RuntimeError("--vectors is required for hls_ap_ctrl")
            progress("load_vectors_start")
            with np.load(args.vectors, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                input_array = np.ascontiguousarray(archive["input_hwc"], dtype=np.int8)
                expected_array = np.ascontiguousarray(archive["expected_hwc"], dtype=np.int8)
                weights = np.ascontiguousarray(archive["weights"], dtype=np.int8)
                biases = np.ascontiguousarray(archive["biases"], dtype=np.int32)
            progress("load_vectors_done")
            weight_words = pack_int8_words(weights)
            bias_words = np.ascontiguousarray(biases, dtype="<i4").view("<u4")
            for name in ("weights_a_axi_bram_ctrl", "weights_b_axi_bram_ctrl"):
                if name in overlay.ip_dict or name in hwh_map:
                    progress(f"write_{name}_start")
                    info = ip_info(overlay, hwh_map, name)
                    write_and_verify(
                        MMIO(info["phys_addr"], info["addr_range"]),
                        weight_words,
                        name,
                        args.param_write_method,
                    )
                    progress(f"write_{name}_done")
            for name in ("biases_a_axi_bram_ctrl", "biases_b_axi_bram_ctrl"):
                if name in overlay.ip_dict or name in hwh_map:
                    progress(f"write_{name}_start")
                    info = ip_info(overlay, hwh_map, name)
                    write_and_verify(
                        MMIO(info["phys_addr"], info["addr_range"]),
                        bias_words,
                        name,
                        args.param_write_method,
                    )
                    progress(f"write_{name}_done")
            in_buf = allocate(shape=input_array.shape, dtype=np.int8)
            out_buf = allocate(shape=expected_array.shape, dtype=np.int8)
            input_bytes = int(input_array.nbytes)
            output_bytes = int(expected_array.nbytes)
        else:
            metadata = {}
            input_array = np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(NBYTES)), dtype=np.uint8)
            expected_array = input_array
            in_buf = allocate(shape=(NBYTES,), dtype=np.uint8)
            out_buf = allocate(shape=(NBYTES,), dtype=np.uint8)
            input_bytes = NBYTES
            output_bytes = NBYTES
        try:
            progress("buffers_prepare_start")
            in_buf[:] = input_array
            out_buf[:] = 0
            in_buf.flush()
            out_buf.flush()
            progress("buffers_prepare_done")

            progress("dma_reset_start")
            dma_reset(dma)
            progress("dma_reset_done")
            before = reg_snapshot(wrap, dma, args.control_mode)
            for _ in range(args.idle_samples):
                power_samples.append(sample_power(hwmon, "idle_pre"))
                time.sleep(args.idle_interval_s)

            progress("warmup_start")
            for _ in range(args.warmup):
                in_buf.flush()
                run_once(
                    wrap,
                    dma,
                    int(in_buf.device_address),
                    int(out_buf.device_address),
                    input_bytes,
                    output_bytes,
                    args.control_mode,
                )
                out_buf.invalidate()
            progress("warmup_done")
            warm_output = np.asarray(out_buf).copy()
            warm_mismatch = int(np.count_nonzero(warm_output != expected_array))

            progress("iterations_start")
            for i in range(args.iterations):
                in_buf.flush()
                latency = run_once(
                    wrap,
                    dma,
                    int(in_buf.device_address),
                    int(out_buf.device_address),
                    input_bytes,
                    output_bytes,
                    args.control_mode,
                )
                out_buf.invalidate()
                latencies_ns.append(latency)
                if args.sample_every > 0 and i % args.sample_every == 0:
                    power_samples.append(sample_power(hwmon, "active", i))
            progress("iterations_done")

            for _ in range(args.idle_samples):
                power_samples.append(sample_power(hwmon, "idle_post"))
                time.sleep(args.idle_interval_s)

            after = reg_snapshot(wrap, dma, args.control_mode)
            output_array = np.asarray(out_buf).copy()
            final_mismatch = int(np.count_nonzero(output_array != expected_array))
            output = output_array.tobytes()
            expected = expected_array.tobytes()
            active_power = [s["power_w"] for s in power_samples if s["phase"] == "active"]
            idle_pre = [s["power_w"] for s in power_samples if s["phase"] == "idle_pre"]
            idle_post = [s["power_w"] for s in power_samples if s["phase"] == "idle_post"]
            cycles = after.get("core_lo", 0) + (after.get("core_hi", 0) << 32)
            clock_hz = before.get("clock_hz") or after.get("clock_hz")
            result.update(
                {
                    "completed_unix_s": time.time(),
                    "overlay_ip_dict_keys": sorted(overlay.ip_dict.keys()),
                    "hwh_memory_map_keys": sorted(hwh_map.keys()),
                    "wrap_phys_addr": wrap_info["phys_addr"],
                    "dma_phys_addr": dma_info["phys_addr"],
                    "input_device_address": int(in_buf.device_address),
                    "output_device_address": int(out_buf.device_address),
                    "vector_metadata": metadata,
                    "input_sha256": hashlib.sha256(input_array.tobytes()).hexdigest(),
                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                    "output_sha256": hashlib.sha256(output).hexdigest(),
                    "output_matches_expected": output == expected,
                    "warmup_mismatch_count": warm_mismatch,
                    "mismatch_count": final_mismatch,
                    "before_registers": before,
                    "after_registers": after,
                    "latency_ns_summary": summarize(latencies_ns),
                    "latency_us_summary": summarize([v / 1000.0 for v in latencies_ns]),
                    "last_core_latency_cycles": cycles,
                    "last_core_latency_us": cycles / clock_hz * 1e6 if clock_hz else None,
                    "power_w_summary": {
                        "idle_pre": summarize(idle_pre),
                        "active": summarize(active_power),
                        "idle_post": summarize(idle_post),
                        "active_minus_idle_pre_mean_w": (
                            statistics.fmean(active_power) - statistics.fmean(idle_pre)
                            if active_power and idle_pre
                            else None
                        ),
                    },
                    "power_samples": power_samples,
                }
            )
        finally:
            try:
                in_buf.freebuffer()
            except Exception:
                pass
            try:
                out_buf.freebuffer()
            except Exception:
                pass
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        compact = dict(result)
        if "power_samples" in compact:
            compact["power_samples"] = f"{len(result['power_samples'])} samples stored in summary.json"
        print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
'''


def connect_server() -> paramiko.SSHClient:
    password = read_secret("ZCU104_SERVER_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER, password=password, timeout=15)
    return client


def connect_board(server: paramiko.SSHClient) -> tuple[str, paramiko.SSHClient]:
    password = read_secret("ZCU104_BOARD_PASSWORD")
    last_error: Exception | None = None
    for host in BOARD_HOSTS:
        try:
            chan = server.get_transport().open_channel("direct-tcpip", (host, 22), ("127.0.0.1", 0), timeout=5)
            board = paramiko.SSHClient()
            board.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            board.connect(
                host,
                username=BOARD_USER,
                password=password,
                sock=chan,
                timeout=10,
                banner_timeout=10,
                auth_timeout=10,
            )
            return host, board
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"could not connect to board: {last_error!r}")


def read_secret(name: str) -> str:
    if name not in _SECRET_CACHE:
        value = os.environ.get(name)
        if not value:
            if sys.stdin.isatty():
                value = getpass.getpass(f"{name}: ")
            else:
                print(f"{name}: ", end="", flush=True)
                value = sys.stdin.readline().rstrip("\n")
        if not value:
            raise RuntimeError(f"{name} is required")
        _SECRET_CACHE[name] = value
    return _SECRET_CACHE[name]


def remote_sha256(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as handle:
        while True:
            data = handle.read(262144)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def copy_server_to_board(
    server: paramiko.SSHClient,
    board: paramiko.SSHClient,
    src: str,
    dst: str,
) -> dict[str, object]:
    stdin, stdout, stderr = server.exec_command(f"sha256sum {src}", timeout=20)
    src_rc = stdout.channel.recv_exit_status()
    if src_rc != 0:
        raise RuntimeError(stderr.read().decode(errors="replace"))
    src_sha = stdout.read().decode(errors="replace").split()[0]
    stdin, stdout, stderr = board.exec_command(f"sha256sum {dst} 2>/dev/null || true", timeout=20)
    dst_text = stdout.read().decode(errors="replace").strip()
    if dst_text and dst_text.split()[0] == src_sha:
        sftp_s = server.open_sftp()
        try:
            size = sftp_s.stat(src).st_size
        finally:
            sftp_s.close()
        return {"src": src, "dst": dst, "bytes": size, "sha256": src_sha, "copied": False}

    sftp_s = server.open_sftp()
    sftp_b = board.open_sftp()
    try:
        size = sftp_s.stat(src).st_size
        digest = hashlib.sha256()
        tmp = dst + ".tmp"
        with sftp_s.open(src, "rb") as read_handle, sftp_b.open(tmp, "wb") as write_handle:
            while True:
                data = read_handle.read(262144)
                if not data:
                    break
                write_handle.write(data)
                digest.update(data)
        stdin, stdout, stderr = board.exec_command(f"mv -f {tmp} {dst}", timeout=20)
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace")
        if rc != 0:
            raise RuntimeError(f"board mv failed rc={rc}: {err}")
        return {"src": src, "dst": dst, "bytes": size, "sha256": digest.hexdigest(), "copied": True}
    finally:
        sftp_s.close()
        sftp_b.close()


def read_remote_file(sftp: paramiko.SFTPClient, path: str) -> bytes:
    with sftp.open(path, "rb") as handle:
        return handle.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--operator", default="stage4_skip")
    parser.add_argument("--server-bit", default=BIT)
    parser.add_argument("--server-hwh", default=HWH)
    parser.add_argument("--server-vectors")
    parser.add_argument("--control-mode", choices=("wrapper", "hls_ap_ctrl"), default="wrapper")
    parser.add_argument("--param-write-method", choices=("array", "word"), default="array")
    parser.add_argument("--command-timeout-s", type=int, default=240)
    args = parser.parse_args()
    board_password = read_secret("ZCU104_BOARD_PASSWORD")

    safe_operator = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.operator).strip("_") or "operator"
    run_id = time.strftime(f"pynq_{safe_operator}_full_sampling_%Y%m%d_%H%M%S")
    server_root = args.server_bit.split("/repo/", 1)[0] if "/repo/" in args.server_bit else BASE
    server_out = f"{server_root}/board_evidence/{run_id}"
    board_out = f"/home/xilinx/zcu104_measurements/{run_id}"
    board_script = f"/home/xilinx/{run_id}.py"

    server = connect_server()
    try:
        board_host, board = connect_board(server)
        try:
            board_bit = f"/home/xilinx/{safe_operator}.bit"
            board_hwh = f"/home/xilinx/{safe_operator}.hwh"
            copies = [
                copy_server_to_board(server, board, args.server_bit, board_bit),
                copy_server_to_board(server, board, args.server_hwh, board_hwh),
            ]
            board_vectors = ""
            if args.server_vectors:
                board_vectors = f"/home/xilinx/{safe_operator}_vectors.npz"
                copies.append(copy_server_to_board(server, board, args.server_vectors, board_vectors))
            sftp_b = board.open_sftp()
            try:
                with sftp_b.open(board_script, "w") as handle:
                    handle.write(BOARD_SCRIPT)
            finally:
                sftp_b.close()

            command = (
                f"mkdir -p {board_out}; "
                "sudo -S -p '' "
                "bash -lc 'source /etc/profile; "
                f"timeout {args.command_timeout_s} python {board_script} --out-dir {board_out} --bit {board_bit} "
                f"--iterations {args.iterations} --warmup {args.warmup} "
                f"--sample-every {args.sample_every} --operator {safe_operator} "
                f"--control-mode {args.control_mode} --param-write-method {args.param_write_method}"
                + (f" --vectors {board_vectors}" if board_vectors else "")
                + "'"
            )
            stdin, stdout, stderr = board.exec_command(command, timeout=args.command_timeout_s + 30)
            stdin.write(board_password + "\n")
            stdin.flush()
            stdout_text = stdout.read().decode(errors="replace")
            stderr_text = stderr.read().decode(errors="replace")
            rc = stdout.channel.recv_exit_status()

            stdin, stdout, stderr = server.exec_command(f"mkdir -p {server_out}", timeout=20)
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(stderr.read().decode(errors="replace"))

            sftp_s = server.open_sftp()
            sftp_b = board.open_sftp()
            try:
                artifacts = {
                    "board_stdout.log": stdout_text.encode(),
                    "board_stderr.log": stderr_text.encode(),
                    "orchestrator_metadata.json": json.dumps(
                        {
                            "server_out": server_out,
                            "board_out": board_out,
                            "board_host": board_host,
                            "board_script": board_script,
                            "board_bit": board_bit,
                            "board_hwh": board_hwh,
                            "copies": copies,
                            "board_returncode": rc,
                            "local_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        },
                        indent=2,
                        sort_keys=True,
                    ).encode(),
                }
                for name, data in artifacts.items():
                    with sftp_s.open(posixpath.join(server_out, name), "wb") as handle:
                        handle.write(data)
                try:
                    summary = read_remote_file(sftp_b, posixpath.join(board_out, "summary.json"))
                except FileNotFoundError:
                    summary = b"{}"
                with sftp_s.open(posixpath.join(server_out, "summary.json"), "wb") as handle:
                    handle.write(summary)
            finally:
                sftp_s.close()
                sftp_b.close()

            print(json.dumps({"server_out": server_out, "board_out": board_out, "rc": rc}, indent=2))
            print(stdout_text)
            if stderr_text:
                print("STDERR_BEGIN")
                print(stderr_text)
                print("STDERR_END")
            if rc != 0:
                raise SystemExit(rc)
        finally:
            board.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
