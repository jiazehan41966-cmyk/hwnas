#!/usr/bin/env python3
"""Cache one INT8 input, then issue RUN_REPEAT for external power capture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hwnas_fpga.deploy.board_protocol import (
    BoardRequest,
    CMD_LOAD_RUN,
    CMD_RUN_REPEAT,
    encode_request,
    read_response,
    repeat_payload,
)
from hls_lut_builder.board_harness.scripts.run_dynamic_validation import (
    connect_with_fallback,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--parity-summary", required=True)
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--baud", default="921600,460800")
    parser.add_argument("--repeat-count", type=int, default=3000)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--pre-active-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="results/power_measurement/run_repeat_receipt.json",
    )
    args = parser.parse_args()
    if args.repeat_count < 1000:
        raise ValueError("power protocol requires at least 1000 inferences")

    payload_path = Path(args.payload).resolve()
    bitstream_path = Path(args.bitstream).resolve()
    parity_path = Path(args.parity_summary).resolve()
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("overall_pass") is not True:
        raise RuntimeError("RUN_REPEAT is blocked until HLS parity is PASS")
    payload = payload_path.read_bytes()

    bauds = [int(value) for value in args.baud.split(",") if value.strip()]
    port, selected_baud = connect_with_fallback(
        args.serial_port,
        bauds,
        args.timeout_seconds,
    )
    try:
        # Upload finishes before the active interval and is excluded from power.
        port.write(
            encode_request(
                BoardRequest(CMD_LOAD_RUN, args.sample_id, payload)
            )
        )
        port.flush()
        loaded = read_response(port)
        if loaded.status != 0:
            raise RuntimeError(f"LOAD_RUN failed with status {loaded.status}")
        time.sleep(max(0.0, args.pre_active_delay_seconds))

        request = BoardRequest(
            CMD_RUN_REPEAT,
            args.sample_id,
            repeat_payload(args.repeat_count),
        )
        active_started_utc = datetime.now(timezone.utc).isoformat()
        active_started = time.monotonic()
        port.write(encode_request(request))
        port.flush()
        repeated = read_response(port)
        active_elapsed_s = time.monotonic() - active_started
        active_finished_utc = datetime.now(timezone.utc).isoformat()
    finally:
        port.close()

    if repeated.status != 0:
        raise RuntimeError(f"RUN_REPEAT failed with status {repeated.status}")
    if repeated.repeat_count != args.repeat_count:
        raise RuntimeError(
            f"board repeated {repeated.repeat_count}, expected {args.repeat_count}"
        )
    result = {
        "schema_version": 1,
        "protocol": "dynamic_validation_uart_v1",
        "sample_id": args.sample_id,
        "repeat_count": args.repeat_count,
        "board_reported_final_inference_cycles": repeated.cycles,
        "response_crc32": repeated.frame_crc32,
        "host_active_elapsed_s": active_elapsed_s,
        "active_started_utc": active_started_utc,
        "active_finished_utc": active_finished_utc,
        "baud": selected_baud,
        "payload": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
        "bitstream": str(bitstream_path),
        "bitstream_sha256": sha256_file(bitstream_path),
        "parity_summary": str(parity_path),
        "parity_summary_sha256": sha256_file(parity_path),
        "contains_programming": False,
        "contains_uart_upload": False,
        "measurement_instruction": (
            "Align the external-meter active CSV to active_started_utc/"
            "active_finished_utc. LOAD_RUN ended before this interval."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if active_elapsed_s >= 60.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
