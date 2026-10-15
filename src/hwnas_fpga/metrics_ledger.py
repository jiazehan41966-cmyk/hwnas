"""Explicit, append-only research metrics without missing-value fabrication.

This module does not calculate metrics or change any training/evaluation code.
Consumers must provide the measured value and its source. Missing values remain
PENDING; task-inapplicable quantities are NOT_APPLICABLE, never zero. Detection
F1/confusion matrices require an explicit matching protocol and are never derived
from AP. All fractions use [0, 1], drops are FP32 minus INT8 (signed fractions),
and utilization uses percent. MACs/FLOPs are complexity, never hardware latency.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "sonar-metrics-ledger-v1"
STATUSES = frozenset({"MEASURED", "PENDING", "NOT_APPLICABLE"})


def _metric(unit: str, category: str, kind: str = "nonnegative", **extra: Any) -> dict:
    return {"unit": unit, "category": category, "kind": kind, **extra}


METRIC_SCHEMA: dict[str, dict[str, Any]] = {
    **{key: _metric("fraction", "task", "fraction") for key in (
        "precision", "recall", "macro_f1", "top1_accuracy", "map50", "map75",
        "map50_95", "tail_class_f1", "small_target_f1", "small_target_recall")},
    "per_class_f1": _metric("fraction_by_class", "task", "class_fractions"),
    "confusion_matrix": _metric("count", "task", "confusion"),
    "noise_robustness": _metric("metric_by_noise_condition", "robustness", "noise"),
    "delta_macro_f1": _metric("fraction", "robustness", "signed_fraction"),
    "psnr": _metric("dB", "image_quality", "finite"),
    "ssim": _metric("score", "image_quality", "signed_fraction"),
    "search_time_seconds": _metric("s", "search"),
    "gpu_hours": _metric("GPU*h", "search"),
    "candidate_evaluation_count": _metric("count", "search", "count"),
    "sampling_efficiency": _metric("declared_by_protocol", "search"),
    "hardware_feasible_rate": _metric("fraction", "search", "fraction"),
    "best_so_far_performance": _metric("fraction", "search", "fraction"),
    "hypervolume": _metric("objective_space_volume", "pareto"),
    "non_dominated_count": _metric("count", "pareto", "count"),
    "pareto_front": _metric("objective_vectors", "pareto", "front"),
    "params": _metric("count", "complexity", "count"),
    "macs": _metric("MAC/image", "complexity"),
    "flops": _metric("FLOP/image", "complexity"),
    **{f"{precision}_{metric}": _metric("fraction", "quantization", "fraction")
       for precision in ("fp32", "int8")
       for metric in ("macro_f1", "top1_accuracy", "map50", "map75", "map50_95", "precision", "recall")},
    **{f"int8_{metric}_drop": _metric("fraction", "quantization", "signed_fraction")
       for metric in ("macro_f1", "top1_accuracy", "map50", "map75", "map50_95", "precision", "recall")},
    **{f"cost_{resource}_{statistic}": _metric(
        "ms" if resource == "latency" and statistic == "mae" else
        "count" if statistic == "mae" else "fraction" if statistic == "mape" else "correlation",
        "cost_model", "signed_fraction" if statistic == "spearman" else "nonnegative")
       for resource in ("latency", "lut", "dsp", "bram")
       for statistic in ("mae", "mape", "spearman")},
    "board_latency_ms": _metric("ms/image", "board"),
    "throughput_fps": _metric("image/s", "board"),
    "actual_clock_mhz": _metric("MHz", "board"),
    **{f"{resource}_{quantity}": _metric(
        "count" if quantity == "used" else "%", "fpga_resources",
        "nonnegative" if quantity == "used" else "percent")
       for resource in ("lut", "ff", "dsp", "bram") for quantity in ("used", "utilization_pct")},
    "wns_ns": _metric("ns", "timing", "finite"),
    "tns_ns": _metric("ns", "timing", "finite"),
    "timing_met": _metric("boolean", "timing", "boolean"),
    "power_w": _metric("W", "board"),
    "energy_per_inference_mj": _metric("mJ/image", "board"),
}


def _is_detection_metric(key: str) -> bool:
    return any(word in key for word in ("map50", "map75", "small_target"))


def _automatic_na(key: str, task_type: str, dataset: str, identity: Mapping) -> str | None:
    if task_type == "classification" and _is_detection_metric(key):
        return "Bounding-box detection metric is not defined for this classification task."
    if task_type == "detection" and "top1_accuracy" in key:
        return "Image-level Top-1 accuracy is not a bounding-box detection metric."
    if key == "tail_class_f1" and "nksid" not in dataset.lower():
        return "NKSID-specific tail-class metric; no tail-class task is declared for this dataset."
    if key in {"psnr", "ssim"} and identity.get("clean_reference_available") is False:
        return "No clean reference; reference-based image quality is not applicable."
    return None


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ValueError("Record must be JSON-compatible with no NaN or Inf anywhere") from exc


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_value(key: str, value: Any, kind: str) -> None:
    valid = False
    if kind in {"nonnegative", "fraction", "signed_fraction", "finite", "percent", "count"}:
        valid = _numeric(value)
        if valid and kind in {"nonnegative", "count"}:
            valid = value >= 0
        if valid and kind == "count":
            valid = isinstance(value, int)
        if valid and kind == "fraction":
            valid = 0 <= value <= 1
        if valid and kind == "signed_fraction":
            valid = -1 <= value <= 1
        if valid and kind == "percent":
            # Utilization >100% is useful evidence of an infeasible design.
            valid = value >= 0
    elif kind == "boolean":
        valid = isinstance(value, bool)
    elif kind == "class_fractions":
        valid = isinstance(value, dict) and bool(value) and all(
            _nonempty(k) and _numeric(v) and 0 <= v <= 1 for k, v in value.items())
    elif kind == "confusion":
        valid = isinstance(value, list) and bool(value) and all(
            isinstance(row, list) and len(row) == len(value) and all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in row)
            for row in value)
    elif kind == "front":
        valid = isinstance(value, list) and all(isinstance(row, dict) and row for row in value)
    elif kind == "noise":
        valid = isinstance(value, dict) and bool(value) and all(
            _nonempty(condition) and _numeric(score) for condition, score in value.items())
    if not valid:
        raise ValueError(f"{key}: invalid value for {kind}; missing data must not be MEASURED")


def _require(provenance: Mapping, key: str, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = provenance.get(field)
        if value is None or value == "" or value == [] or value == {}:
            raise ValueError(f"{key}: provenance.{field} is required")


def _validate_provenance(key: str, provenance: Mapping, task_type: str) -> None:
    if not isinstance(provenance, Mapping) or not _nonempty(provenance.get("source")):
        raise ValueError(f"{key}: MEASURED requires provenance.source")
    if key in {"psnr", "ssim"}:
        if provenance.get("clean_reference") is not True or provenance.get("synthetic_noise") is not True:
            raise ValueError(f"{key}: requires clean_reference=true and synthetic_noise=true")
        _require(provenance, key, ("reference_manifest_sha256", "noise_protocol", "data_range"))
    if task_type == "detection" and any(word in key for word in ("macro_f1", "per_class_f1", "confusion_matrix")):
        _require(provenance, key, ("matching_protocol",))
    if "small_target" in key:
        _require(provenance, key, ("size_definition", "matching_protocol"))
    if key == "tail_class_f1":
        _require(provenance, key, ("tail_classes", "tail_definition"))
    if key in {"noise_robustness", "delta_macro_f1"}:
        _require(provenance, key, ("noise_protocol", "reference_condition", "metric"))
    if key == "sampling_efficiency":
        _require(provenance, key, ("definition", "denominator"))
    if key == "hardware_feasible_rate":
        _require(provenance, key, ("feasibility_level", "denominator", "constraint_hash"))
    if key in {"sampling_efficiency", "hardware_feasible_rate"}:
        denominator = provenance["denominator"]
        if not _numeric(denominator) or denominator <= 0:
            raise ValueError(f"{key}: provenance.denominator must be positive")
    if key == "best_so_far_performance":
        _require(provenance, key, ("metric",))
    if key in {"hypervolume", "pareto_front", "non_dominated_count"}:
        _require(provenance, key, ("objectives", "directions", "candidate_population"))
    if key == "hypervolume":
        _require(provenance, key, ("reference_point", "normalization"))
    if key.startswith("cost_"):
        _require(provenance, key, ("measured_network_manifest", "prediction_manifest", "sample_count"))
        if not isinstance(provenance["sample_count"], int) or isinstance(provenance["sample_count"], bool) or provenance["sample_count"] <= 0:
            raise ValueError(f"{key}: sample_count must be a positive integer")
    if key.startswith("int8_") or key.startswith("fp32_"):
        _require(provenance, key, ("checkpoint_sha256", "evaluation_protocol_hash", "split_manifest_sha256"))
    if key.startswith("int8_"):
        _require(provenance, key, ("numeric_contract", "calibration_manifest_sha256"))
    if key.endswith("_drop"):
        if provenance.get("definition") != "fp32_minus_int8":
            raise ValueError(f"{key}: drop definition must be fp32_minus_int8")
    if key in {"board_latency_ms", "throughput_fps", "actual_clock_mhz", "power_w", "energy_per_inference_mj"}:
        if provenance.get("evidence_layer") != "board_measurement":
            raise ValueError(f"{key}: requires actual board_measurement, not HLS/estimate/MACs")
        _require(provenance, key, ("target_device", "artifact_sha256", "measurement_method"))
    if key in {"board_latency_ms", "throughput_fps"}:
        _require(provenance, key, ("precision", "batch_size", "input_shape", "warmup", "repetitions"))
    if key in {"power_w", "energy_per_inference_mj"}:
        _require(provenance, key, ("power_scope", "idle_subtracted"))
    if METRIC_SCHEMA[key]["category"] in {"fpga_resources", "timing"}:
        _require(provenance, key, ("evidence_layer", "target_device", "artifact_sha256", "tool_version"))
        if provenance["evidence_layer"] not in {"hls", "synthesis", "place_and_route"}:
            raise ValueError(f"{key}: distinguish HLS/synthesis/place_and_route resource or timing evidence")
    if key in {"wns_ns", "tns_ns", "timing_met"} and provenance["evidence_layer"] != "place_and_route":
        raise ValueError(f"{key}: routed timing requires place_and_route evidence")


def build_metric_record(*, task_type: str, dataset: str, network: str, seed: int,
                        stage: str, identity: Mapping[str, Any],
                        measured: Mapping[str, Any],
                        not_applicable: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a complete event. ``measured`` entries contain value/provenance.

    Identity must include a stable run_id and should bind checkpoint, dataset,
    protocol, epoch and event kind. Supply identity.event_id to name an event;
    otherwise its ID is SHA256 of the identity and task coordinates. Replaying
    the same event is idempotent; changing its values is rejected at append.
    """
    if task_type not in {"detection", "classification"}:
        raise ValueError("task_type must be detection or classification")
    if not isinstance(identity, Mapping) or not _nonempty(identity.get("run_id")):
        raise ValueError("identity.run_id is required")
    if not all(_nonempty(v) for v in (dataset, network, stage)):
        raise ValueError("dataset, network and stage must be nonempty strings")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(measured, Mapping):
        raise ValueError("measured must be a metric-name mapping")
    not_applicable = dict(not_applicable or {})
    unknown = (set(measured) | set(not_applicable)) - set(METRIC_SCHEMA)
    if unknown:
        raise ValueError(f"Unknown metric keys: {sorted(unknown)}")
    if set(measured) & set(not_applicable):
        raise ValueError("A metric cannot be both MEASURED and NOT_APPLICABLE")
    coordinates = dict(task_type=task_type, dataset=dataset, network=network,
                       seed=seed, stage=stage, identity=deepcopy(dict(identity)))
    event_id = identity.get("event_id") or hashlib.sha256(_json_bytes(coordinates)).hexdigest()
    record = {"schema_version": SCHEMA_VERSION, "event_id": event_id,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              **coordinates, "metrics": {}}
    for key, definition in METRIC_SCHEMA.items():
        reason = _automatic_na(key, task_type, dataset, identity)
        cell = {"status": "PENDING", "value": None, "unit": definition["unit"],
                "category": definition["category"], "provenance": {},
                "reason": "Measurement not supplied for this event."}
        if key in measured:
            if reason:
                raise ValueError(f"{key}: {reason}")
            if not isinstance(measured[key], Mapping) or "value" not in measured[key]:
                raise ValueError(f"{key}: supply value and provenance, not a bare scalar")
            if "unit" in measured[key] and measured[key]["unit"] != definition["unit"]:
                raise ValueError(f"{key}: supplied unit differs from schema")
            extra = set(measured[key]) - {"value", "provenance", "unit"}
            if extra:
                raise ValueError(f"{key}: unrecognized measurement fields: {sorted(extra)}")
            cell.update(status="MEASURED", value=deepcopy(measured[key]["value"]),
                        provenance=deepcopy(measured[key].get("provenance", {})), reason=None)
        elif key in not_applicable or reason:
            cell.update(status="NOT_APPLICABLE", reason=not_applicable.get(key, reason))
        record["metrics"][key] = cell
    validate_metric_record(record)
    return record


def validate_metric_record(record: Mapping[str, Any]) -> None:
    """Reject missing metric keys, ambiguous zeros, invalid units or provenance."""
    _json_bytes(record)
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported metrics schema_version")
    if record.get("task_type") not in {"detection", "classification"}:
        raise ValueError("Invalid task_type")
    for key in ("event_id", "dataset", "network", "stage", "recorded_at_utc"):
        if not _nonempty(record.get(key)):
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(record.get("seed"), int) or isinstance(record["seed"], bool) or record["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or not _nonempty(identity.get("run_id")):
        raise ValueError("identity.run_id is required")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_SCHEMA):
        raise ValueError("Every schema metric must be present exactly once")
    for key, definition in METRIC_SCHEMA.items():
        cell = metrics[key]
        if not isinstance(cell, Mapping) or cell.get("status") not in STATUSES:
            raise ValueError(f"{key}: invalid status")
        if cell.get("unit") != definition["unit"] or cell.get("category") != definition["category"]:
            raise ValueError(f"{key}: units/category differ from schema")
        if cell["status"] == "MEASURED":
            if _automatic_na(key, record["task_type"], record["dataset"], identity):
                raise ValueError(f"{key}: not applicable to this task")
            _validate_value(key, cell.get("value"), definition["kind"])
            _validate_provenance(key, cell.get("provenance"), record["task_type"])
        else:
            if cell.get("value") is not None:
                raise ValueError(f"{key}: unmeasured values must be null, never zero")
            if not _nonempty(cell.get("reason")):
                raise ValueError(f"{key}: unmeasured metric requires an explicit reason")


def require_measured_metrics(record: Mapping[str, Any], metric_names: Iterable[str]) -> None:
    """Fail a caller-defined phase gate when a required measurement is missing.

    This is deliberately stricter than schema validation: a complete schema may
    describe pending work, while completed evaluation/hardware stages must name
    the metrics that actually need to have been measured.
    """
    validate_metric_record(record)
    names = list(metric_names)
    unknown = set(names) - set(METRIC_SCHEMA)
    if unknown:
        raise ValueError(f"Unknown required metric keys: {sorted(unknown)}")
    missing = {name: record["metrics"][name]["status"] for name in names
               if record["metrics"][name]["status"] != "MEASURED"}
    if missing:
        raise ValueError(f"Required measurements are incomplete: {missing}")


@contextmanager
def _ledger_lock(path: Path, timeout: float = 10.0):
    """OS lock releases after process death; lock file need not be deleted."""
    handle = path.open("a+b")
    # Locking beyond EOF is supported by Windows byte-range locks. Do not
    # initialize byte zero before locking: a simultaneous writer may already
    # own that byte and Windows would reject the initialization flush.
    start = time.monotonic()
    locked = False
    try:
        while not locked:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() - start >= timeout:
                    raise TimeoutError(f"Metrics ledger lock timed out: {path}")
                time.sleep(0.05)
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _same_event(left: Mapping, right: Mapping) -> bool:
    return {k: v for k, v in left.items() if k != "recorded_at_utc"} == {
        k: v for k, v in right.items() if k != "recorded_at_utc"}


def append_metric_record(path: str | Path, record: Mapping[str, Any]) -> bool:
    """Durably append an event; return False for an identical already-saved event.

    Conflicting event IDs and corrupt complete lines fail closed. A crash-truncated
    final line is preserved in a SHA-named recovery file before truncating only
    that incomplete suffix. Concurrent writers are serialized by an OS file lock.
    """
    validate_metric_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a+b") as handle:
            handle.seek(0)
            payload = handle.read()
            offset = 0
            seen: dict[str, dict] = {}
            for line in payload.splitlines(keepends=True):
                try:
                    existing = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if line.endswith(b"\n") or offset + len(line) != len(payload):
                        raise ValueError(f"Corrupt complete ledger line at byte {offset}") from exc
                    digest = hashlib.sha256(line).hexdigest()
                    recovery = path.with_name(path.name + f".recovery.{digest}.bin")
                    if not recovery.exists():
                        with recovery.open("xb") as recovered:
                            recovered.write(line)
                            recovered.flush()
                            os.fsync(recovered.fileno())
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                    payload = payload[:offset]
                    break
                validate_metric_record(existing)
                if existing["event_id"] in seen:
                    raise ValueError(f"Duplicate event_id in existing ledger: {existing['event_id']}")
                seen[existing["event_id"]] = existing
                offset += len(line)
            if record["event_id"] in seen:
                if _same_event(seen[record["event_id"]], record):
                    return False
                raise ValueError(f"Conflicting values for existing event_id: {record['event_id']}")
            handle.seek(0, os.SEEK_END)
            if payload and not payload.endswith(b"\n"):
                handle.write(b"\n")
            handle.write(_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def read_metric_records(path: str | Path) -> list[dict[str, Any]]:
    """Read validated events. A partial/corrupt write is never treated as a result."""
    path = Path(path)
    if not path.exists():
        return []
    with _ledger_lock(path.with_suffix(path.suffix + ".lock")):
        records, seen = [], set()
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid metrics ledger JSON on line {number}") from exc
            validate_metric_record(record)
            if record["event_id"] in seen:
                raise ValueError(f"Duplicate event_id on line {number}")
            seen.add(record["event_id"])
            records.append(record)
    return records
