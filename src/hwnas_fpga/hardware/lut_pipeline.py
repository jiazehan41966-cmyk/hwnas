"""Build LUT tables from Vivado/Vitis HLS profiling manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import yaml

from hwnas_fpga.hardware.lookup_table import LutEntry, LutTable, OpSpec
from hwnas_fpga.hardware.report_parser import parse_hls_report


_SPEC_KEYS = {
    "op",
    "kernel_size",
    "in_channels",
    "out_channels",
    "stride",
    "groups",
    "expand_ratio",
    "input_resolution",
    "bitwidth",
    "input_parallelism",
    "output_parallelism",
    "unroll_factor",
    "target_clock_mhz",
}

_METRIC_KEYS = {
    "latency_ms",
    "latency_ns",
    "cycles",
    "ii",
    "depth",
    "dsp",
    "bram",
    "bram_18k",
    "bram_36k",
    "lut",
    "ff",
    "estimated_clock_period_ns",
    "fmax_est_mhz",
    "power_w",
    "energy_mj",
    "bram_tile",
}


def load_lut_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON manifest describing profiled operator reports."""
    manifest_path = Path(path).expanduser().resolve()
    suffix = manifest_path.suffix.lower()
    text = manifest_path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise ValueError(f"LUT manifest must contain a mapping: {manifest_path}")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"LUT manifest must define a non-empty 'entries' list: {manifest_path}")

    payload["_manifest_path"] = str(manifest_path)
    return payload


def build_lut_from_manifest(
    path: str | Path,
    *,
    default_clock_mhz: Optional[int] = None,
) -> tuple[LutTable, dict[str, Any]]:
    """Build a LUT table from a profiling manifest."""
    manifest = load_lut_manifest(path)
    manifest_path = Path(manifest["_manifest_path"])
    manifest_clock = manifest.get("clock_mhz", default_clock_mhz)
    table = LutTable()
    processed_entries: list[dict[str, Any]] = []

    for index, raw_entry in enumerate(manifest["entries"]):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping")

        lut_entry = _build_lut_entry(
            raw_entry,
            manifest_dir=manifest_path.parent,
            manifest_clock_mhz=manifest_clock,
        )
        table.add(lut_entry)
        processed_entries.append(
            {
                "index": index,
                "op": lut_entry.op_spec.op,
                "report": str(_resolve_report_path(raw_entry, manifest_path.parent)) if _get_report_path(raw_entry) else None,
                "latency_ms": lut_entry.latency_ms,
                "dsp": lut_entry.dsp,
                "bram": lut_entry.bram,
                "lut": lut_entry.lut,
            }
        )

    summary = {
        "manifest_path": str(manifest_path),
        "clock_mhz": manifest_clock,
        "entries_requested": len(manifest["entries"]),
        "entries_built": len(table),
        "table_stats": table.get_stats(),
        "entries": processed_entries,
    }
    return table, summary


def save_lut_from_manifest(
    path: str | Path,
    output_path: str | Path,
    *,
    default_clock_mhz: Optional[int] = None,
) -> dict[str, Any]:
    """Build and save a LUT table from a profiling manifest."""
    table, summary = build_lut_from_manifest(path, default_clock_mhz=default_clock_mhz)
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.save(str(output_file))
    summary["output_path"] = str(output_file)
    return summary


def _build_lut_entry(
    raw_entry: dict[str, Any],
    *,
    manifest_dir: Path,
    manifest_clock_mhz: Optional[int],
) -> LutEntry:
    op_spec = _parse_op_spec(raw_entry)
    metrics = _collect_metrics(raw_entry, manifest_dir)
    clock_mhz = raw_entry.get("clock_mhz", manifest_clock_mhz)

    latency_ms = metrics.get("latency_ms")
    if latency_ms is None:
        cycles = int(metrics.get("cycles") or 0)
        latency_ns = metrics.get("latency_ns")
        if cycles > 0:
            if clock_mhz is None:
                raise ValueError(
                    f"Clock must be provided for LUT entry {op_spec} when only cycles are available"
                )
            latency_ms = cycles / (float(clock_mhz) * 1_000.0)
        elif latency_ns is not None:
            latency_ms = float(latency_ns) / 1_000_000.0
        else:
            latency_ms = 0.0

    raw_power_w = metrics.get("power_w")
    power_w = None if raw_power_w in (None, "") else float(raw_power_w)
    energy_mj = metrics.get("energy_mj")
    if energy_mj in (None, "") and power_w is not None:
        energy_mj = power_w * float(latency_ms)
    elif energy_mj in (None, ""):
        energy_mj = None

    return LutEntry(
        op_spec=op_spec,
        latency_ms=float(latency_ms),
        cycles=int(metrics.get("cycles") or 0),
        dsp=int(metrics.get("dsp") or 0),
        bram=int(metrics.get("bram") or 0),
        lut=int(metrics.get("lut") or 0),
        power_w=power_w,
        energy_mj=None if energy_mj is None else float(energy_mj),
    )


def _parse_op_spec(raw_entry: dict[str, Any]) -> OpSpec:
    spec_payload = raw_entry.get("op_spec")
    if spec_payload is None:
        spec_payload = {key: raw_entry[key] for key in _SPEC_KEYS if key in raw_entry}
    if not isinstance(spec_payload, dict):
        raise ValueError("Manifest entry 'op_spec' must be a mapping")
    if "op" not in spec_payload:
        raise ValueError("Manifest entry must define operator fields or an 'op_spec' mapping")
    return OpSpec.from_dict(spec_payload)


def _collect_metrics(raw_entry: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    report_path = _resolve_report_path(raw_entry, manifest_dir)
    if report_path is not None:
        metrics.update(parse_hls_report(report_path))

    hls_estimate = raw_entry.get("hls_estimate", {})
    if hls_estimate:
        if not isinstance(hls_estimate, dict):
            raise ValueError("Manifest entry 'hls_estimate' must be a mapping")
        hls_metrics = hls_estimate.get("metrics", {})
        if hls_metrics and not isinstance(hls_metrics, dict):
            raise ValueError("Manifest entry 'hls_estimate.metrics' must be a mapping")
        metrics.update(hls_metrics)

    raw_metrics = raw_entry.get("metrics", {})
    if raw_metrics:
        if not isinstance(raw_metrics, dict):
            raise ValueError("Manifest entry 'metrics' must be a mapping")
        metrics.update(raw_metrics)

    vivado_actual = raw_entry.get("vivado_actual", {})
    if vivado_actual:
        if not isinstance(vivado_actual, dict):
            raise ValueError("Manifest entry 'vivado_actual' must be a mapping")
        metrics.update(_extract_vivado_actual_metrics(vivado_actual))

    for key in _METRIC_KEYS:
        if key in raw_entry:
            metrics[key] = raw_entry[key]

    return metrics


def _get_report_path(raw_entry: dict[str, Any]) -> Optional[str]:
    for key in ("report", "report_path", "path"):
        value = raw_entry.get(key)
        if value:
            return str(value)
    return None


def _resolve_report_path(raw_entry: dict[str, Any], manifest_dir: Path) -> Optional[Path]:
    raw_path = _get_report_path(raw_entry)
    if raw_path is None:
        return None

    report_path = Path(raw_path).expanduser()
    if not report_path.is_absolute():
        report_path = (manifest_dir / report_path).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Profiling report not found: {report_path}")
    return report_path


def _extract_vivado_actual_metrics(vivado_actual: dict[str, Any]) -> dict[str, Any]:
    payload = vivado_actual.get("metrics", {})
    if not isinstance(payload, dict) or not payload:
        return {}

    preferred_stage = None
    for stage_name in ("post_route", "post_synth"):
        stage_payload = payload.get(stage_name)
        if isinstance(stage_payload, dict) and stage_payload:
            preferred_stage = stage_payload
            break

    if preferred_stage is None:
        return {}

    metrics: dict[str, Any] = {}
    if preferred_stage.get("dsp") is not None:
        metrics["dsp"] = int(preferred_stage["dsp"])
    if preferred_stage.get("lut") is not None:
        metrics["lut"] = int(preferred_stage["lut"])
    if preferred_stage.get("ff") is not None:
        metrics["ff"] = int(preferred_stage["ff"])
    if preferred_stage.get("fmax_est_mhz") is not None:
        metrics["fmax_est_mhz"] = float(preferred_stage["fmax_est_mhz"])
        if metrics["fmax_est_mhz"] > 0:
            metrics["estimated_clock_period_ns"] = 1_000.0 / metrics["fmax_est_mhz"]
    if preferred_stage.get("power_w") is not None:
        metrics["power_w"] = float(preferred_stage["power_w"])

    block_ram_tile = preferred_stage.get("block_ram_tile")
    if block_ram_tile is not None:
        metrics["bram_tile"] = float(block_ram_tile)
        metrics["bram"] = int(math.ceil(float(block_ram_tile)))
    if preferred_stage.get("ramb18") is not None:
        metrics["bram_18k"] = int(preferred_stage["ramb18"])
    if preferred_stage.get("ramb36") is not None:
        metrics["bram_36k"] = int(preferred_stage["ramb36"])

    return metrics
