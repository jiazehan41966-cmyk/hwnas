#!/usr/bin/env python3
"""Run one ZCU104 Skip DMA transaction through XSDB/JTAG on the remote server."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import paramiko


HOST = "10.70.23.121"
PORT = 7889
USER = "jiajia"

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

INPUT_BYTES = 25_088
INPUT_WORDS = INPUT_BYTES // 4
DDR_BACKUP_WORDS = 0x20000 // 4


def make_tcl(out_dir: str) -> str:
    return f"""
proc rd {{addr}} {{
    return [lindex [mrd -force -value $addr] 0]
}}
proc wr {{addr value}} {{
    mwr -force $addr $value
}}
proc logreg {{label addr}} {{
    if {{[catch {{set v [rd $addr]}} e]}} {{
        puts "REG_ERR $label 0x[format %08X $addr] $e"
    }} else {{
        puts "REG $label 0x[format %08X $addr] 0x[format %08X $v]"
    }}
}}
proc wait_mask_set {{label addr mask limit}} {{
    for {{set i 0}} {{$i < $limit}} {{incr i}} {{
        set v [rd $addr]
        if {{([expr {{$v & $mask}}]) != 0}} {{
            puts "WAIT_SET_OK $label iter=$i value=0x[format %08X $v]"
            return 0
        }}
        after 1
    }}
    puts "WAIT_SET_TIMEOUT $label value=0x[format %08X [rd $addr]] mask=0x[format %08X $mask]"
    return 1
}}
proc wait_mask_clear {{label addr mask limit}} {{
    for {{set i 0}} {{$i < $limit}} {{incr i}} {{
        set v [rd $addr]
        if {{([expr {{$v & $mask}}]) == 0}} {{
            puts "WAIT_CLEAR_OK $label iter=$i value=0x[format %08X $v]"
            return 0
        }}
        after 1
    }}
    puts "WAIT_CLEAR_TIMEOUT $label value=0x[format %08X [rd $addr]] mask=0x[format %08X $mask]"
    return 1
}}
set WRAP 0xA0000000
set DMA 0xA0010000
set INADDR 0x70000000
set OUTADDR 0x70010000
set NBYTES {INPUT_BYTES}
set NWORDS {INPUT_WORDS}
connect -host 127.0.0.1 -port 3121
puts "TARGETS_BEGIN"
puts [targets]
puts "PROGRAM_BEGIN"
targets -set -filter {{name =~ "PL"}}
fpga -file {{{BIT}}}
after 2000
puts "PROGRAM_DONE"
targets -set -filter {{name =~ "PSU"}}
memmap -addr 0xA0000000 -size 0x20000 -flags 3
memmap -addr 0x70000000 -size 0x20000 -flags 3
puts "MEMMAP_READY"
puts "BACKUP_BEGIN"
mrd -force -bin -file {{{out_dir}/ddr_backup.bin}} $INADDR {DDR_BACKUP_WORDS}
puts "BACKUP_DONE"
puts "WRITE_BUFFERS_BEGIN"
mwr -force -bin -file {{{out_dir}/input.bin}} $INADDR $NWORDS
mwr -force -bin -file {{{out_dir}/zero.bin}} $OUTADDR $NWORDS
puts "WRITE_BUFFERS_DONE"
foreach {{label off}} {{shell_status 0x04 exec_lo 0x08 exec_hi 0x0c core_lo 0x10 core_hi 0x14 expected_in 0x18 expected_out 0x1c input_seen 0x20 output_seen 0x24 err 0x28 shell_id 0x2c version 0x30 op_id 0x34 axis_bits 0x38 clock_hz 0x3c}} {{
    logreg "PRE_$label" [expr {{$WRAP + $off}}]
}}
foreach {{label off}} {{mm2s_cr 0x00 mm2s_sr 0x04 s2mm_cr 0x30 s2mm_sr 0x34}} {{
    logreg "PRE_DMA_$label" [expr {{$DMA + $off}}]
}}
puts "DMA_RESET_BEGIN"
wr [expr {{$DMA + 0x00}}] 0x4
wait_mask_clear MM2S_RESET [expr {{$DMA + 0x00}}] 0x4 1000
wr [expr {{$DMA + 0x30}}] 0x4
wait_mask_clear S2MM_RESET [expr {{$DMA + 0x30}}] 0x4 1000
wr [expr {{$DMA + 0x00}}] 0x1
wr [expr {{$DMA + 0x30}}] 0x1
puts "DMA_RESET_DONE"
wr [expr {{$DMA + 0x48}}] $OUTADDR
wr [expr {{$DMA + 0x4c}}] 0x0
wr [expr {{$DMA + 0x18}}] $INADDR
wr [expr {{$DMA + 0x1c}}] 0x0
puts "RUN_BEGIN"
wr [expr {{$DMA + 0x58}}] $NBYTES
wr [expr {{$WRAP + 0x00}}] 0x6
wr [expr {{$WRAP + 0x00}}] 0x1
wr [expr {{$DMA + 0x28}}] $NBYTES
set fail 0
set fail [expr {{$fail | [wait_mask_set MM2S_IDLE [expr {{$DMA + 0x04}}] 0x2 5000]}}]
set fail [expr {{$fail | [wait_mask_set S2MM_IDLE [expr {{$DMA + 0x34}}] 0x2 5000]}}]
set fail [expr {{$fail | [wait_mask_set SHELL_DONE [expr {{$WRAP + 0x04}}] 0x4 5000]}}]
puts "RUN_WAIT_FAIL $fail"
foreach {{label off}} {{shell_status 0x04 exec_lo 0x08 exec_hi 0x0c core_lo 0x10 core_hi 0x14 expected_in 0x18 expected_out 0x1c input_seen 0x20 output_seen 0x24 err 0x28 shell_id 0x2c version 0x30 op_id 0x34 axis_bits 0x38 clock_hz 0x3c}} {{
    logreg "POST_$label" [expr {{$WRAP + $off}}]
}}
foreach {{label off}} {{mm2s_cr 0x00 mm2s_sr 0x04 s2mm_cr 0x30 s2mm_sr 0x34}} {{
    logreg "POST_DMA_$label" [expr {{$DMA + $off}}]
}}
puts "READ_OUTPUT_BEGIN"
mrd -force -bin -file {{{out_dir}/output.bin}} $OUTADDR $NWORDS
puts "READ_OUTPUT_DONE"
puts "RESTORE_BEGIN"
mwr -force -bin -file {{{out_dir}/ddr_backup.bin}} $INADDR {DDR_BACKUP_WORDS}
puts "RESTORE_DONE"
exit
"""


def summarize(out_dir: str, stdout_text: str, rc: int, sftp: paramiko.SFTPClient) -> dict[str, object]:
    regs: dict[str, dict[str, object]] = {}
    for line in stdout_text.splitlines():
        if line.startswith("REG "):
            _, label, addr, value = line.split()[:4]
            regs[label] = {"addr": addr, "value": value, "int": int(value, 16)}

    def read_remote_bytes(name: str) -> bytes | None:
        try:
            with sftp.open(f"{out_dir}/{name}", "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    result: dict[str, object] = {
        "out_dir": out_dir,
        "returncode": rc,
        "stdout_sha256": hashlib.sha256(stdout_text.encode()).hexdigest(),
        "registers": regs,
        "run_wait_fail_present": "RUN_WAIT_FAIL 0" in stdout_text,
    }
    input_data = read_remote_bytes("input.bin")
    output_data = read_remote_bytes("output.bin")
    for name, data in (("input", input_data), ("output", output_data), ("ddr_backup", read_remote_bytes("ddr_backup.bin"))):
        if data is not None:
            result[f"{name}_bytes"] = len(data)
            result[f"{name}_sha256"] = hashlib.sha256(data).hexdigest()
    if input_data is not None and output_data is not None:
        result["output_matches_input"] = input_data == output_data
        result["mismatch_count"] = sum(a != b for a, b in zip(input_data, output_data)) + abs(
            len(input_data) - len(output_data)
        )
    clock = regs.get("POST_clock_hz", {}).get("int") or regs.get("PRE_clock_hz", {}).get("int")
    core_lo = regs.get("POST_core_lo", {}).get("int")
    core_hi = regs.get("POST_core_hi", {}).get("int", 0)
    if isinstance(clock, int) and isinstance(core_lo, int) and isinstance(core_hi, int):
        cycles = (core_hi << 32) | core_lo
        result["core_latency_cycles"] = cycles
        result["core_latency_us"] = cycles / clock * 1e6
    return result


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=read_secret("ZCU104_SERVER_PASSWORD"), timeout=15)
    try:
        stdin, stdout, stderr = client.exec_command(
            f"mkdir -p {BASE}/board_evidence && mktemp -d {BASE}/board_evidence/jtag_dma_skip_run_XXXXXXXX",
            get_pty=True,
            timeout=30,
        )
        out_dir = stdout.read().decode().strip()
        err = stderr.read().decode(errors="replace")
        if err:
            print(err)
        if not out_dir:
            raise RuntimeError("remote mktemp did not return an output directory")

        payload = bytes((i * 37 + 11) & 0xFF for i in range(INPUT_BYTES))
        zero = bytes(INPUT_BYTES)

        sftp = client.open_sftp()
        try:
            with sftp.open(f"{out_dir}/input.bin", "wb") as handle:
                handle.write(payload)
            with sftp.open(f"{out_dir}/zero.bin", "wb") as handle:
                handle.write(zero)
            with sftp.open(f"{out_dir}/run_jtag_dma_skip.tcl", "w") as handle:
                handle.write(make_tcl(out_dir))
        finally:
            sftp.close()

        command = (
            f"sha256sum {BIT} {HWH} {out_dir}/input.bin {out_dir}/zero.bin > {out_dir}/sha256sums.txt; "
            f"timeout 300 /opt/Xilinx/Vitis/2022.2/bin/xsdb {out_dir}/run_jtag_dma_skip.tcl "
            f"> {out_dir}/xsdb.stdout.log 2> {out_dir}/xsdb.stderr.log; "
            f"rc=$?; echo $rc > {out_dir}/returncode.txt; "
            f"echo RETURN_CODE=$rc; tail -n 140 {out_dir}/xsdb.stdout.log; "
            f"echo STDERR_BEGIN; cat {out_dir}/xsdb.stderr.log"
        )
        stdin, stdout, stderr = client.exec_command(command, get_pty=True, timeout=420)
        stdout_text = stdout.read().decode(errors="replace")
        stderr_text = stderr.read().decode(errors="replace")
        print(stdout_text)
        if stderr_text:
            print("REMOTE_STDERR:", stderr_text)

        sftp = client.open_sftp()
        try:
            with sftp.open(f"{out_dir}/xsdb.stdout.log", "r") as handle:
                xsdb_stdout = handle.read().decode(errors="replace")
            with sftp.open(f"{out_dir}/returncode.txt", "r") as handle:
                rc = int(handle.read().decode().strip())
            result = summarize(out_dir, xsdb_stdout, rc, sftp)
            summary = json.dumps(result, indent=2, sort_keys=True)
            with sftp.open(f"{out_dir}/summary.json", "w") as handle:
                handle.write(summary)
        finally:
            sftp.close()
        print("SUMMARY_JSON_BEGIN")
        print(summary)
    finally:
        client.close()


if __name__ == "__main__":
    main()
