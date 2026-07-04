#!/usr/bin/env python3
"""Run a pre-quantized validation manifest through the dynamic UART harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import serial
except ImportError:  # pragma: no cover - hardware dependency
    serial = None

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.deploy.board_protocol import (
    BoardRequest,
    CMD_LOAD_RUN,
    CMD_PING,
    encode_request,
    read_response,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def per_class_f1(confusion: list[list[int]]) -> list[float]:
    scores = []
    for index in range(len(confusion)):
        tp = confusion[index][index]
        fp = sum(row[index] for row in confusion) - tp
        fn = sum(confusion[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return scores


def summarize_records(
    records: list[dict[str, Any]],
    *,
    expected_sample_ids: set[int],
    num_classes: int = 8,
) -> dict[str, Any]:
    by_id = {int(row["sample_id"]): row for row in records}
    duplicates = len(records) - len(by_id)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    numeric_mismatches = 0
    protocol_failures = 0
    cycles = []
    for row in by_id.values():
        label = int(row["label"])
        prediction = int(row["argmax"])
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            confusion[label][prediction] += 1
        numeric_mismatches += int(row.get("numeric_match") is not True)
        protocol_failures += int(int(row.get("status", -1)) != 0)
        cycles.append(int(row["cycles"]))
    completed = set(by_id)
    missing = sorted(expected_sample_ids - completed)
    extra = sorted(completed - expected_sample_ids)
    f1 = per_class_f1(confusion)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(num_classes))
    return {
        "expected_count": len(expected_sample_ids),
        "completed_count": len(completed & expected_sample_ids),
        "missing_sample_ids": missing,
        "extra_sample_ids": extra,
        "duplicate_record_count": duplicates,
        "protocol_failure_count": protocol_failures,
        "numeric_mismatch_count": numeric_mismatches,
        "top1": correct / total if total else None,
        "macro_f1": statistics.fmean(f1) if f1 else None,
        "per_class_f1": f1,
        "confusion_matrix": confusion,
        "cycles": {
            "mean": statistics.fmean(cycles) if cycles else None,
            "min": min(cycles) if cycles else None,
            "max": max(cycles) if cycles else None,
        },
        "claimable": (
            not missing
            and not extra
            and duplicates == 0
            and protocol_failures == 0
            and numeric_mismatches == 0
            and bool(expected_sample_ids)
        ),
    }


def connect_with_fallback(port_name: str, bauds: list[int], timeout: float):
    if serial is None:
        raise RuntimeError("pyserial is required for dynamic board validation")
    errors = []
    for baud in bauds:
        try:
            port = serial.Serial(port_name, baudrate=baud, timeout=timeout)
            port.reset_input_buffer()
            port.reset_output_buffer()
            ping = encode_request(BoardRequest(CMD_PING, 0, b""))
            port.write(ping)
            port.flush()
            response = read_response(port)
            if response.status == 0 and response.command == CMD_PING:
                return port, baud
            port.close()
            errors.append(f"baud {baud}: ping status={response.status}")
        except Exception as exc:  # pragma: no cover - hardware dependency
            errors.append(f"baud {baud}: {exc}")
    raise RuntimeError("UART negotiation failed: " + "; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL sample manifest")
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--baud", default="921600,460800")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--expected-payload-bytes", type=int, default=224 * 224)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output-dir", default="results/board_dynamic_validation")
    parser.add_argument(
        "--require-expected-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    samples = read_jsonl(manifest_path)
    if not samples:
        raise ValueError("validation manifest is empty")
    expected_ids = {int(row["sample_id"]) for row in samples}
    if len(expected_ids) != len(samples):
        raise ValueError("validation manifest contains duplicate sample_id")
    bitstream_path = Path(args.bitstream).resolve()
    if not bitstream_path.exists():
        raise FileNotFoundError(bitstream_path)
    bitstream_hash = sha256_file(bitstream_path)

    for row in samples:
        payload_path = Path(row["payload_path"]).expanduser().resolve()
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        if payload_path.stat().st_size != args.expected_payload_bytes:
            raise ValueError(
                f"{payload_path} has {payload_path.stat().st_size} bytes; "
                f"expected {args.expected_payload_bytes}"
            )
        if args.require_expected_logits and len(row.get("expected_logits", [])) != 8:
            raise ValueError(f"sample {row['sample_id']} lacks 8 expected logits")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "board_validation.jsonl"
    existing = read_jsonl(output_path)
    completed_ids = {int(row["sample_id"]) for row in existing}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_count": len(samples),
                    "already_completed": len(completed_ids),
                    "bitstream_sha256": bitstream_hash,
                    "dry_run": True,
                },
                indent=2,
            )
        )
        return 0

    bauds = [int(value) for value in args.baud.split(",") if value.strip()]
    port, selected_baud = connect_with_fallback(
        args.serial_port,
        bauds,
        args.timeout_seconds,
    )
    records = list(existing)
    try:
        with output_path.open("a", encoding="utf-8") as output_handle:
            for sample in samples:
                sample_id = int(sample["sample_id"])
                if sample_id in completed_ids:
                    continue
                payload = Path(sample["payload_path"]).resolve().read_bytes()
                last_error = None
                response = None
                for attempt in range(1, args.retries + 1):
                    try:
                        port.reset_input_buffer()
                        request = encode_request(
                            BoardRequest(CMD_LOAD_RUN, sample_id, payload)
                        )
                        started = time.monotonic()
                        port.write(request)
                        port.flush()
                        response = read_response(port)
                        host_elapsed_ms = (time.monotonic() - started) * 1000.0
                        if response.sample_id != sample_id:
                            raise ValueError(
                                f"response sample {response.sample_id} != {sample_id}"
                            )
                        if response.command != CMD_LOAD_RUN:
                            raise ValueError("response command mismatch")
                        break
                    except Exception as exc:  # pragma: no cover - hardware dependency
                        last_error = str(exc)
                        response = None
                if response is None:
                    raise RuntimeError(
                        f"sample {sample_id} failed after {args.retries} retries: {last_error}"
                    )
                expected_logits = tuple(int(value) for value in sample.get("expected_logits", []))
                numeric_match = (
                    tuple(response.logits) == expected_logits
                    if expected_logits
                    else None
                )
                record = {
                    "sample_id": sample_id,
                    "path": sample.get("path"),
                    "label": int(sample["label"]),
                    "status": response.status,
                    "cycles": response.cycles,
                    "logits": list(response.logits),
                    "argmax": response.argmax,
                    "checksum": response.checksum,
                    "expected_logits": list(expected_logits),
                    "numeric_match": numeric_match,
                    "retries_used": attempt,
                    "host_elapsed_ms": host_elapsed_ms,
                    "baud": selected_baud,
                    "bitstream_sha256": bitstream_hash,
                }
                output_handle.write(json.dumps(record) + "\n")
                output_handle.flush()
                records.append(record)
    finally:
        port.close()

    summary = summarize_records(records, expected_sample_ids=expected_ids)
    summary.update(
        {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "bitstream": str(bitstream_path),
            "bitstream_sha256": bitstream_hash,
            "baud": selected_baud,
            "claim_boundary": (
                "Single frozen outer fold/seed deployment-chain evidence; "
                "not 5-fold NAS-method generalization."
            ),
        }
    )
    (output_dir / "board_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["claimable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
