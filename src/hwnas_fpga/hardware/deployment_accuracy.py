"""Deployment accuracy-retention audit for FP32, INT8, and board results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _nested_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = [payload]
    for key in ("metrics", "best_eval", "summary", "classification", "board"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            result.append(value)
    return result


def _first_number(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for mapping in _nested_mappings(payload):
        for key in keys:
            value = mapping.get(key)
            if value is None or value == "":
                continue
            return float(value)
    return None


def _first_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _first_number(payload, keys)
    return None if value is None else int(value)


def _metrics(payload: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "macro_f1": _first_number(payload, ("macro_f1", "Macro-F1", "f1_macro")),
        "top1": _first_number(payload, ("top1", "accuracy", "top1_accuracy")),
    }


def _board_quality(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = _first_int(payload, ("expected_samples", "sample_count", "num_samples"))
    observed = _first_int(payload, ("observed_samples", "processed_samples", "board_samples"))
    coverage = _first_number(payload, ("sample_coverage", "coverage"))
    if coverage is None and expected and observed is not None:
        coverage = observed / expected
    return {
        "expected_samples": expected,
        "observed_samples": observed,
        "sample_coverage": coverage,
        "crc_error_count": _first_int(payload, ("crc_error_count", "crc_errors")) or 0,
        "status_error_count": _first_int(payload, ("status_error_count", "status_errors")) or 0,
        "argmax_mismatch_count": _first_int(
            payload,
            ("argmax_mismatch_count", "board_argmax_mismatch_count"),
        )
        or 0,
    }


def _prediction_jsonl_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    line_count = sum(1 for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "line_count": line_count,
    }


def audit_accuracy_retention(
    *,
    fp32_summary: str | Path,
    int8_summary: str | Path,
    board_summary: str | Path,
    board_predictions_jsonl: str | Path,
    macro_f1_drop_threshold: float = 0.02,
    top1_drop_threshold: float = 0.02,
) -> dict[str, Any]:
    """Audit whether deployment preserves classification accuracy."""

    fp32_payload = _read_json(fp32_summary)
    int8_payload = _read_json(int8_summary)
    board_payload = _read_json(board_summary)
    prediction_record = _prediction_jsonl_record(board_predictions_jsonl)
    fp32 = _metrics(fp32_payload)
    int8 = _metrics(int8_payload)
    board = _metrics(board_payload)
    board_quality = _board_quality(board_payload)

    delta_int8_macro = (
        int8["macro_f1"] - fp32["macro_f1"]
        if int8["macro_f1"] is not None and fp32["macro_f1"] is not None
        else None
    )
    delta_board_macro = (
        board["macro_f1"] - fp32["macro_f1"]
        if board["macro_f1"] is not None and fp32["macro_f1"] is not None
        else None
    )
    delta_int8_top1 = (
        int8["top1"] - fp32["top1"]
        if int8["top1"] is not None and fp32["top1"] is not None
        else None
    )
    delta_board_top1 = (
        board["top1"] - fp32["top1"]
        if board["top1"] is not None and fp32["top1"] is not None
        else None
    )

    coverage = board_quality["sample_coverage"]
    observed = board_quality["observed_samples"]
    gates = {
        "fp32_metrics_present": fp32["macro_f1"] is not None and fp32["top1"] is not None,
        "int8_metrics_present": int8["macro_f1"] is not None and int8["top1"] is not None,
        "board_metrics_present": board["macro_f1"] is not None and board["top1"] is not None,
        "int8_macro_f1_retained": (
            delta_int8_macro is not None and delta_int8_macro >= -float(macro_f1_drop_threshold)
        ),
        "board_macro_f1_retained": (
            delta_board_macro is not None and delta_board_macro >= -float(macro_f1_drop_threshold)
        ),
        "int8_top1_retained": (
            delta_int8_top1 is not None and delta_int8_top1 >= -float(top1_drop_threshold)
        ),
        "board_top1_retained": (
            delta_board_top1 is not None and delta_board_top1 >= -float(top1_drop_threshold)
        ),
        "board_sample_coverage_100pct": coverage is not None and coverage >= 1.0,
        "board_crc_error_count_zero": board_quality["crc_error_count"] == 0,
        "board_status_error_count_zero": board_quality["status_error_count"] == 0,
        "board_argmax_mismatch_count_zero": board_quality["argmax_mismatch_count"] == 0,
        "board_per_sample_predictions_saved": (
            prediction_record["line_count"] > 0
            and (observed is None or prediction_record["line_count"] == observed)
        ),
    }
    return {
        "schema_version": 1,
        "overall_pass": all(gates.values()),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "thresholds": {
            "macro_f1_drop_threshold": float(macro_f1_drop_threshold),
            "top1_drop_threshold": float(top1_drop_threshold),
        },
        "fp32": fp32,
        "int8": int8,
        "board": board,
        "deltas": {
            "delta_int8_macro_f1": delta_int8_macro,
            "delta_board_macro_f1": delta_board_macro,
            "delta_int8_top1": delta_int8_top1,
            "delta_board_top1": delta_board_top1,
        },
        "board_quality": board_quality,
        "board_predictions_jsonl": prediction_record,
        "gates": gates,
        "claim_boundary": (
            "This audit checks accuracy retention and board prediction consistency only. "
            "It does not establish HLS route, latency, GOPS, or measured power."
        ),
    }
