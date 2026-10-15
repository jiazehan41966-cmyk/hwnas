#!/usr/bin/env python3
"""Probe ZCU104 U183 INA226 through PS I2C1 using XSDB/JTAG."""

from __future__ import annotations

import json
import os
import sys

import paramiko


HOST = "10.70.23.121"
PORT = 7889
USER = "jiajia"
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


def make_tcl(out_dir: str) -> str:
    return f"""
proc rd {{addr}} {{ return [lindex [mrd -force -value $addr] 0] }}
proc wr {{addr value}} {{ mwr -force $addr $value }}
proc logreg {{label addr}} {{
    if {{[catch {{set v [rd $addr]}} e]}} {{
        puts "REG_ERR $label 0x[format %08X $addr] $e"
    }} else {{
        puts "REG $label 0x[format %08X $addr] 0x[format %08X $v]"
    }}
}}
proc wait_comp {{label base limit}} {{
    for {{set i 0}} {{$i < $limit}} {{incr i}} {{
        set isr [rd [expr {{$base + 0x10}}]]
        if {{($isr & 0x204) != 0}} {{
            puts "I2C_ERR $label iter=$i isr=0x[format %08X $isr]"
            return 1
        }}
        if {{($isr & 0x001) != 0}} {{
            puts "I2C_COMP $label iter=$i isr=0x[format %08X $isr]"
            return 0
        }}
        after 1
    }}
    puts "I2C_TIMEOUT $label isr=0x[format %08X [rd [expr {{$base + 0x10}}]]]"
    return 1
}}
proc i2c_write1 {{base addr byte label}} {{
    wr [expr {{$base + 0x28}}] 0x2ff
    wr [expr {{$base + 0x10}}] 0x2ff
    set cr [rd [expr {{$base + 0x00}}]]
    set cr [expr {{($cr & 0x0000ff00) | 0x4e}}]
    wr [expr {{$base + 0x00}}] $cr
    wr [expr {{$base + 0x0c}}] $byte
    wr [expr {{$base + 0x08}}] $addr
    return [wait_comp $label $base 2000]
}}
proc i2c_read_reg16 {{base addr reg label}} {{
    wr [expr {{$base + 0x28}}] 0x2ff
    wr [expr {{$base + 0x10}}] 0x2ff
    set cr [rd [expr {{$base + 0x00}}]]
    set cr [expr {{($cr & 0x0000ff00) | 0x5e}}]
    wr [expr {{$base + 0x00}}] $cr
    wr [expr {{$base + 0x0c}}] $reg
    wr [expr {{$base + 0x08}}] $addr
    if {{[wait_comp "${{label}}_ptr" $base 2000]}} {{
        return -1
    }}
    wr [expr {{$base + 0x10}}] 0x2ff
    set cr [expr {{($cr & 0x0000ff00) | 0x5f}}]
    wr [expr {{$base + 0x00}}] $cr
    wr [expr {{$base + 0x14}}] 2
    wr [expr {{$base + 0x08}}] $addr
    set bytes {{}}
    for {{set i 0}} {{$i < 2000}} {{incr i}} {{
        set sr [rd [expr {{$base + 0x04}}]]
        set isr [rd [expr {{$base + 0x10}}]]
        if {{($isr & 0x204) != 0}} {{
            puts "I2C_ERR $label read iter=$i isr=0x[format %08X $isr] sr=0x[format %08X $sr]"
            break
        }}
        if {{($sr & 0x20) != 0}} {{
            lappend bytes [expr {{[rd [expr {{$base + 0x0c}}]] & 0xff}}]
            if {{[llength $bytes] == 2}} {{
                break
            }}
        }}
        after 1
    }}
    wr [expr {{$base + 0x00}}] [expr {{[rd [expr {{$base + 0x00}}]] & ~0x10}}]
    if {{[llength $bytes] != 2}} {{
        puts "I2C_READ_SHORT $label count=[llength $bytes]"
        return -1
    }}
    set value [expr {{([lindex $bytes 0] << 8) | [lindex $bytes 1]}}]
    puts "I2C_READ16 $label reg=0x[format %02X $reg] value=0x[format %04X $value] bytes=$bytes"
    return $value
}}
set I2C1 0xFF030000
connect -host 127.0.0.1 -port 3121
targets -set -filter {{name =~ "PSU"}}
memmap -addr 0xFF030000 -size 0x1000 -flags 3
puts "I2C_PROBE_BEGIN"
foreach {{label off}} {{cr 0x00 sr 0x04 addr 0x08 data 0x0c isr 0x10 xfer 0x14 timeout 0x1c imr 0x20}} {{
    logreg "PRE_$label" [expr {{$I2C1 + $off}}]
}}
set orig_cr [rd [expr {{$I2C1 + 0x00}}]]
set mux_rc [i2c_write1 $I2C1 0x74 0x08 mux_ch3]
set config [i2c_read_reg16 $I2C1 0x40 0x00 ina226_config]
set busv [i2c_read_reg16 $I2C1 0x40 0x02 ina226_bus_voltage]
set power [i2c_read_reg16 $I2C1 0x40 0x03 ina226_power]
set curr [i2c_read_reg16 $I2C1 0x40 0x04 ina226_current]
set manuf [i2c_read_reg16 $I2C1 0x40 0xfe ina226_manufacturer]
wr [expr {{$I2C1 + 0x00}}] $orig_cr
foreach {{label off}} {{cr 0x00 sr 0x04 isr 0x10 xfer 0x14 imr 0x20}} {{
    logreg "POST_$label" [expr {{$I2C1 + $off}}]
}}
puts "I2C_RESULT mux_rc=$mux_rc config=$config busv=$busv power=$power curr=$curr manuf=$manuf"
puts "I2C_PROBE_END"
exit
"""


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=read_secret("ZCU104_SERVER_PASSWORD"), timeout=15)
    try:
        stdin, stdout, stderr = client.exec_command(
            f"mkdir -p {BASE}/board_evidence && mktemp -d {BASE}/board_evidence/jtag_i2c_ina226_probe_XXXXXXXX",
            get_pty=True,
            timeout=30,
        )
        out_dir = stdout.read().decode().strip()
        err = stderr.read().decode(errors="replace")
        if err:
            print(err)
        sftp = client.open_sftp()
        try:
            with sftp.open(f"{out_dir}/probe_ina226.tcl", "w") as handle:
                handle.write(make_tcl(out_dir))
        finally:
            sftp.close()
        cmd = (
            f"timeout 45 /opt/Xilinx/Vitis/2022.2/bin/xsdb {out_dir}/probe_ina226.tcl "
            f"> {out_dir}/xsdb.stdout.log 2> {out_dir}/xsdb.stderr.log; "
            f"rc=$?; echo $rc > {out_dir}/returncode.txt; echo OUT={out_dir}; echo RETURN_CODE=$rc; "
            f"cat {out_dir}/xsdb.stdout.log; echo STDERR_BEGIN; cat {out_dir}/xsdb.stderr.log"
        )
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=80)
        stdout_text = stdout.read().decode(errors="replace")
        stderr_text = stderr.read().decode(errors="replace")
        print(stdout_text)
        if stderr_text:
            print("REMOTE_STDERR:", stderr_text)
        result = {"out_dir": out_dir}
        for line in stdout_text.splitlines():
            if line.startswith("I2C_READ16 ") or line.startswith("I2C_RESULT "):
                result.setdefault("events", []).append(line)
        print("SUMMARY_JSON_BEGIN")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
