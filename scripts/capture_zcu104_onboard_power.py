#!/usr/bin/env python3
"""Capture ZCU104 VCC12 board-input power from the onboard INA226 hwmon node."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any


EXPECTED_SHUNT_UOHM = 5_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_number(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    return float(path.read_text(encoding="utf-8").strip())


def monotonic_raw() -> float:
    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return time.clock_gettime(clock_id)


def read_sample(hwmon: Path) -> dict[str, float]:
    voltage_v = read_number(hwmon / "in1_input") / 1_000.0
    current_a = read_number(hwmon / "curr1_input") / 1_000.0
    power_path = hwmon / "power1_input"
    power_w = (
        read_number(power_path) / 1_000_000.0
        if power_path.is_file()
        else voltage_v * current_a
    )
    if not 9.0 <= voltage_v <= 15.0:
        raise RuntimeError(f"VCC12 INA226 bus voltage is implausible: {voltage_v} V")
    if current_a < 0.0 or power_w < 0.0:
        raise RuntimeError("VCC12 INA226 returned negative current or power")
    return {
        "voltage_v": voltage_v,
        "current_a": current_a,
        "power_w": power_w,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hwmon-dir",
        required=True,
        help="INA226 hwmon directory, for example /sys/class/hwmon/hwmon3.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hwmon = Path(args.hwmon_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_output).expanduser().resolve()
        if args.metadata_output
        else output_csv.with_suffix(".capture.json")
    )
    if args.duration_s < 60.0:
        raise ValueError("formal capture duration must be at least 60 seconds")
    if args.sample_rate_hz <= 0.0:
        raise ValueError("sample rate must be positive")

    name = (hwmon / "name").read_text(encoding="utf-8").strip().lower()
    if name not in {"ina226", "ina2xx"}:
        raise RuntimeError(f"unexpected hwmon device {name!r}; expected INA226/ina2xx")
    shunt_path = hwmon / "shunt_resistor"
    shunt_uohm = int(read_number(shunt_path))
    if shunt_uohm != EXPECTED_SHUNT_UOHM:
        raise RuntimeError(
            f"VCC12 shunt mismatch: {shunt_uohm} uOhm; expected {EXPECTED_SHUNT_UOHM}"
        )
    update_interval_ms = (
        read_number(hwmon / "update_interval")
        if (hwmon / "update_interval").is_file()
        else None
    )
    sensor_configuration = {
        "board": "ZCU104",
        "sensor": "INA226",
        "reference_designator": "U183",
        "rail": "VCC12",
        "driver": name,
        "hwmon_path": str(hwmon),
        "shunt_resistor_uohm": shunt_uohm,
        "update_interval_ms": update_interval_ms,
        "sample_rate_hz": float(args.sample_rate_hz),
        "clock": "CLOCK_MONOTONIC_RAW",
        "i2c_topology": {
            "controller": "PS_I2C1_MIO16_17",
            "mux_model": "TCA9548A",
            "mux_address": "0x74",
            "mux_channel": 3,
            "sensor_address": "0x40",
        },
    }
    configuration_sha256 = canonical_sha256(sensor_configuration)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    start = monotonic_raw()
    next_due = start
    sample_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_s",
                "voltage_v",
                "current_a",
                "power_w",
                "sensor",
            ],
        )
        writer.writeheader()
        while True:
            now = monotonic_raw()
            if now < next_due:
                time.sleep(min(next_due - now, 0.05))
                continue
            sample = read_sample(hwmon)
            elapsed = now - start
            writer.writerow(
                {
                    "timestamp_s": f"{elapsed:.9f}",
                    "voltage_v": f"{sample['voltage_v']:.9f}",
                    "current_a": f"{sample['current_a']:.9f}",
                    "power_w": f"{sample['power_w']:.9f}",
                    "sensor": "ZCU104_U183_VCC12_INA226",
                }
            )
            sample_count += 1
            if elapsed >= float(args.duration_s):
                break
            next_due += 1.0 / float(args.sample_rate_hz)

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_source": "zcu104_vcc12_ina226_i2c",
        "rail_scope": "board_input_total_12v",
        "sensor_configuration": sensor_configuration,
        "sensor_configuration_sha256": configuration_sha256,
        "capture": {
            "path": str(output_csv),
            "sha256": sha256_file(output_csv),
            "sample_count": sample_count,
            "duration_s": float(args.duration_s),
        },
        "claim_boundary": (
            "Read-only VCC12 board-input capture. This file alone is not a formal "
            "candidate power result until the bitstream/runtime/repetition manifest passes."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
