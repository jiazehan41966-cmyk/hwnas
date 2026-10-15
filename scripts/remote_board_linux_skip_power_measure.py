#!/usr/bin/env python3
"""Run ZCU104 Skip via board Linux /dev/mem and capture INA226 power."""

from __future__ import annotations

import json
import os
import sys
import textwrap

import paramiko


SERVER_HOST = "10.70.23.121"
SERVER_PORT = 7889
SERVER_USER = "jiajia"
BOARD_HOST = "192.168.1.100"
BOARD_USER = "xilinx"

BASE = "/home/jiajia/experiments/zcu104_operator_lut_measurement_v1/stage4_skip_standard_wrapper_v1"

_SECRET_CACHE: dict[str, str] = {}


def read_secret(name: str) -> str:
    if name not in _SECRET_CACHE:
        value = os.environ.get(name)
        if not value:
            print(f"{name}: ", end="", flush=True)
            value = sys.stdin.readline().rstrip("\n")
        if not value:
            raise RuntimeError(f"{name} is required")
        _SECRET_CACHE[name] = value
    return _SECRET_CACHE[name]


BOARD_SCRIPT = r'''
import argparse, hashlib, json, mmap, os, statistics, struct, time
from pathlib import Path

WRAP = 0xA0000000
DMA = 0xA0010000
INADDR = 0x70000000
OUTADDR = 0x70010000
WINDOW = 0x20000
NBYTES = 25088
NWORDS = 784
DMA_RESET = 0x4
DMA_RUN = 0x1
DMA_IDLE = 0x2
DMA_ERR = 0x770


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_int(path):
    return int(Path(path).read_text().strip())


def sample_power(hwmon):
    h = Path(hwmon)
    return {
        "voltage_v": read_int(h / "in2_input") / 1000.0,
        "current_a": read_int(h / "curr1_input") / 1000.0,
        "power_w": read_int(h / "power1_input") / 1_000_000.0,
        "timestamp_ns": time.monotonic_ns(),
    }


def map_phys(fd, addr, size):
    page = addr & ~(mmap.PAGESIZE - 1)
    off = addr - page
    mm = mmap.mmap(fd, size + off, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page)
    return mm, off


def r32(mm, off, reg):
    return struct.unpack_from("<I", mm, off + reg)[0]


def w32(mm, off, reg, val):
    struct.pack_into("<I", mm, off + reg, val & 0xFFFFFFFF)


def wait_set(mm, off, reg, mask, timeout_s, label):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = r32(mm, off, reg)
        if value & DMA_ERR:
            raise RuntimeError(f"{label} error reg=0x{reg:x} value=0x{value:08x}")
        if value & mask:
            return value
    raise TimeoutError(f"{label} timeout value=0x{r32(mm, off, reg):08x}")


def wait_clear(mm, off, reg, mask, timeout_s, label):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = r32(mm, off, reg)
        if not (value & mask):
            return value
    raise TimeoutError(f"{label} timeout value=0x{r32(mm, off, reg):08x}")


def run_once(wrap_mm, wrap_off, dma_mm, dma_off):
    w32(dma_mm, dma_off, 0x58, NBYTES)   # S2MM length
    w32(wrap_mm, wrap_off, 0x00, 0x6)    # clear done/errors
    w32(wrap_mm, wrap_off, 0x00, 0x1)    # start shell
    t0 = time.monotonic_ns()
    w32(dma_mm, dma_off, 0x28, NBYTES)   # MM2S length
    wait_set(dma_mm, dma_off, 0x04, DMA_IDLE, 1.0, "mm2s")
    wait_set(dma_mm, dma_off, 0x34, DMA_IDLE, 1.0, "s2mm")
    wait_set(wrap_mm, wrap_off, 0x04, 0x4, 1.0, "shell")
    t1 = time.monotonic_ns()
    return t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--hwmon", default="/sys/class/hwmon/hwmon2")
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument(
        "--allow-fixed-ddr-scratch",
        action="store_true",
        help="Use fixed physical DDR addresses; unsafe unless the range is known to be unused.",
    )
    args = ap.parse_args()
    if not args.allow_fixed_ddr_scratch:
        raise SystemExit("refusing fixed DDR scratch without --allow-fixed-ddr-scratch")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = bytes((i * 37 + 11) & 0xFF for i in range(NBYTES))
    zero = bytes(NBYTES)
    power_samples = []
    latencies_ns = []

    with open("/dev/mem", "r+b", buffering=0) as mem:
        fd = mem.fileno()
        wrap, wo = map_phys(fd, WRAP, 0x10000)
        dma, do = map_phys(fd, DMA, 0x10000)
        ddr, ddo = map_phys(fd, INADDR, WINDOW)
        backup = None
        try:
            backup = bytes(ddr[ddo : ddo + WINDOW])
            (out / "ddr_backup.bin").write_bytes(backup)
            before = {
                "shell_status": r32(wrap, wo, 0x04),
                "shell_id": r32(wrap, wo, 0x2C),
                "version": r32(wrap, wo, 0x30),
                "op_id": r32(wrap, wo, 0x34),
                "axis_bits": r32(wrap, wo, 0x38),
                "clock_hz": r32(wrap, wo, 0x3C),
                "expected_in": r32(wrap, wo, 0x18),
                "expected_out": r32(wrap, wo, 0x1C),
            }
            ddr[ddo : ddo + NBYTES] = payload
            ddr[ddo + 0x10000 : ddo + 0x10000 + NBYTES] = zero
            w32(dma, do, 0x00, DMA_RESET)
            wait_clear(dma, do, 0x00, DMA_RESET, 1.0, "mm2s reset")
            w32(dma, do, 0x30, DMA_RESET)
            wait_clear(dma, do, 0x30, DMA_RESET, 1.0, "s2mm reset")
            w32(dma, do, 0x00, DMA_RUN)
            w32(dma, do, 0x30, DMA_RUN)
            w32(dma, do, 0x48, OUTADDR)
            w32(dma, do, 0x4C, 0)
            w32(dma, do, 0x18, INADDR)
            w32(dma, do, 0x1C, 0)

            power_samples.append({"phase": "pre", **sample_power(args.hwmon)})
            for i in range(args.iterations):
                latencies_ns.append(run_once(wrap, wo, dma, do))
                if i % args.sample_every == 0:
                    power_samples.append({"phase": "active", "iteration": i, **sample_power(args.hwmon)})
            power_samples.append({"phase": "post", **sample_power(args.hwmon)})

            output = bytes(ddr[ddo + 0x10000 : ddo + 0x10000 + NBYTES])
            after = {
                "shell_status": r32(wrap, wo, 0x04),
                "exec_lo": r32(wrap, wo, 0x08),
                "exec_hi": r32(wrap, wo, 0x0C),
                "core_lo": r32(wrap, wo, 0x10),
                "core_hi": r32(wrap, wo, 0x14),
                "input_seen": r32(wrap, wo, 0x20),
                "output_seen": r32(wrap, wo, 0x24),
                "err": r32(wrap, wo, 0x28),
                "dma_mm2s_sr": r32(dma, do, 0x04),
                "dma_s2mm_sr": r32(dma, do, 0x34),
            }
            ddr[ddo : ddo + WINDOW] = backup
        finally:
            if backup is not None:
                ddr[ddo : ddo + WINDOW] = backup
            wrap.close()
            dma.close()
            ddr.close()

    active_powers = [s["power_w"] for s in power_samples if s.get("phase") == "active"]
    result = {
        "schema_version": 1,
        "measurement_route": "board_linux_devmem_dma_plus_hwmon_ina226",
        "operator": "stage4_skip",
        "iterations": args.iterations,
        "hwmon": args.hwmon,
        "before_registers": before,
        "after_registers": after,
        "input_sha256": sha256(payload),
        "output_sha256": sha256(output),
        "output_matches_input": output == payload,
        "mismatch_count": sum(a != b for a, b in zip(output, payload)) + abs(len(output) - len(payload)),
        "latency_ns_summary": {
            "count": len(latencies_ns),
            "mean": statistics.fmean(latencies_ns),
            "median": statistics.median(latencies_ns),
            "min": min(latencies_ns),
            "max": max(latencies_ns),
        },
        "last_core_latency_cycles": after["core_lo"] + (after["core_hi"] << 32),
        "last_core_latency_us": (after["core_lo"] + (after["core_hi"] << 32)) / before["clock_hz"] * 1e6,
        "power_w_summary": {
            "sample_count": len(active_powers),
            "mean": statistics.fmean(active_powers) if active_powers else None,
            "median": statistics.median(active_powers) if active_powers else None,
            "min": min(active_powers) if active_powers else None,
            "max": max(active_powers) if active_powers else None,
        },
        "power_samples": power_samples,
        "claim_boundary": "Exploratory board Linux measurement. Fixed DDR scratch was backed up/restored; not yet a formal LUT row without repeated idle/active protocol and ambient record.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    server = paramiko.SSHClient()
    server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    server.connect(
        SERVER_HOST,
        port=SERVER_PORT,
        username=SERVER_USER,
        password=read_secret("ZCU104_SERVER_PASSWORD"),
        timeout=15,
    )
    try:
        # Program the bitstream first.
        program_cmd = f'''
BASE={BASE}
BIT=$BASE/repo/results/zcu104_operator_lut_measurement_v1/stage4_skip_standard_wrapper_v1/platform/bitstream/zcu104_standard_wrapper.bit
OUT=$BASE/board_evidence/board_linux_power_reprogram_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
cat > "$OUT/program.tcl" <<EOF
connect -host 127.0.0.1 -port 3121
targets -set -filter {{name =~ "PL"}}
fpga -file {{$BIT}}
puts "PROGRAM_DONE"
EOF
timeout 45 /opt/Xilinx/Vitis/2022.2/bin/xsdb "$OUT/program.tcl" > "$OUT/xsdb.stdout.log" 2> "$OUT/xsdb.stderr.log" || true
echo "$OUT"
'''
        stdin, stdout, stderr = server.exec_command(program_cmd, get_pty=True, timeout=80)
        program_out = stdout.read().decode(errors="replace").strip().splitlines()
        print("PROGRAM_OUT")
        print("\n".join(program_out))
        err = stderr.read().decode(errors="replace")
        if err:
            print("PROGRAM_ERR", err)

        chan = server.get_transport().open_channel("direct-tcpip", (BOARD_HOST, 22), ("127.0.0.1", 0), timeout=5)
        board = paramiko.SSHClient()
        board.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        board_password = read_secret("ZCU104_BOARD_PASSWORD")
        board.connect(BOARD_HOST, username=BOARD_USER, password=board_password, sock=chan, timeout=10)
        try:
            sftp = board.open_sftp()
            with sftp.open("/tmp/board_linux_skip_power_measure.py", "w") as handle:
                handle.write(BOARD_SCRIPT)
            sftp.close()
            out_dir = f"/home/xilinx/jtag_operator_measure_{int(__import__('time').time())}"
            cmd = (
                f"mkdir -p {out_dir}; "
                "sudo -S -p '' python3 /tmp/board_linux_skip_power_measure.py "
                f"--out-dir {out_dir} --iterations 1000 --sample-every 10; "
                f"echo BOARD_OUT={out_dir}"
            )
            stdin, stdout, stderr = board.exec_command(cmd, get_pty=False, timeout=120)
            stdin.write(board_password + "\n")
            stdin.flush()
            print("BOARD_MEASURE_OUT")
            print(stdout.read().decode(errors="replace"))
            err = stderr.read().decode(errors="replace")
            if err:
                print("BOARD_MEASURE_ERR", err)
        finally:
            board.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
