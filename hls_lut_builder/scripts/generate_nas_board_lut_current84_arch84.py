#!/usr/bin/env python
"""Generate the strict NAS board LUT for current84/arch84 evidence.

This script creates a derived LUT directory only. It does not modify the raw
deployable LUT ledgers, board measurement ledger, or any COM5/e2e measurement
artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_ROOT = REPO_ROOT / "hls_lut_builder"
BOARD_RESULTS_ROOT = BUILDER_ROOT / "board_harness" / "results"

CURRENT84_DIR = BOARD_RESULTS_ROOT / "v2_current84_fullcombo_board"
ARCH84_DIR = BOARD_RESULTS_ROOT / "v2_arch84_e2e_extension_board"
E2E_DIR = BOARD_RESULTS_ROOT / "sonar_classifier_end_to_end_board"
DEFAULT_OUTPUT_DIR = BUILDER_ROOT / "results" / "nas_board_lut_strict_current84_arch84"

PROTECTED_RAW_INPUTS = (
    BUILDER_ROOT / "results" / "v2_deployable_lut" / "deployable_lut_v2.json",
    BUILDER_ROOT / "results" / "v2_current84_deployable_lut" / "deployable_lut_v2_current84.json",
    BOARD_RESULTS_ROOT / "board_measured_lut.csv",
)

ROW_FIELDS = (
    "case_name",
    "op_type",
    "lookup_op",
    "decomposition",
    "layer_role",
    "component_role",
    "latency_cycles",
    "latency_ms",
    "LUT",
    "FF",
    "DSP",
    "BRAM",
    "BRAM_entry_ceil",
    "WNS",
    "UART status_code",
    "timing_clean",
    "strict_usable",
    "status",
    "source_namespace",
    "source_json",
    "power_w",
    "notes",
    "op_spec_json",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clean_case_name(data: dict[str, Any], path: Path) -> str:
    return str(data.get("combo_case_name") or path.name.removesuffix(".result.json"))


def _source_path(path: Path) -> str:
    return str(path.resolve())


def _spec(
    *,
    op: str,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    expand_ratio: int = 1,
    resolution: int = 1,
) -> dict[str, Any]:
    return {
        "op": op,
        "kernel_size": int(kernel_size),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "stride": int(stride),
        "groups": int(groups),
        "expand_ratio": int(expand_ratio),
        "input_resolution": [int(resolution), int(resolution)],
        "bitwidth": 8,
        "input_parallelism": 1,
        "output_parallelism": 1,
        "unroll_factor": 1,
        "target_clock_mhz": None,
    }


def _parse_case_to_op(case_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse board case naming into the NAS-side OpSpec schema."""

    match = re.match(
        r"^mbconv_e(?P<e>\d+)_k(?P<k>\d+)_(?P<tag>mbconv|arch84)_stage"
        r"(?P<stage>\d+)_(?P<res>\d+)_(?P<cin>\d+)_(?P<cout>\d+)_s(?P<stride>\d+)",
        case_name,
    )
    if match:
        expand_ratio = int(match.group("e"))
        kernel_size = int(match.group("k"))
        op = f"mbconv_e{expand_ratio}_k{kernel_size}"
        stage = f"stage{match.group('stage')}"
        if match.group("tag") == "arch84":
            stage = f"arch84_{stage}"
        return (
            _spec(
                op=op,
                kernel_size=kernel_size,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                stride=int(match.group("stride")),
                expand_ratio=expand_ratio,
                resolution=int(match.group("res")),
            ),
            {
                "op_type": op,
                "lookup_op": op,
                "layer_role": stage,
                "component_role": "stitched_mbconv",
            },
        )

    match = re.match(
        r"^stem_conv_k(?P<k>\d+)_s\d+_stem_(?P<res>\d+)_(?P<cin>\d+)_"
        r"(?P<cout>\d+)_s(?P<stride>\d+)",
        case_name,
    )
    if match:
        return (
            _spec(
                op="conv",
                kernel_size=int(match.group("k")),
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                stride=int(match.group("stride")),
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "conv",
                "lookup_op": "conv",
                "layer_role": "stem",
                "component_role": "stem_conv",
            },
        )

    match = re.match(
        r"^pw_conv_pw_head_(?P<res>\d+)_(?P<cin>\d+)_(?P<cout>\d+)",
        case_name,
    )
    if match:
        return (
            _spec(
                op="conv",
                kernel_size=1,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "conv",
                "lookup_op": "conv",
                "layer_role": "head",
                "component_role": "head_pw_conv",
            },
        )

    match = re.match(
        r"^pw_conv_pw_ir(?P<ir>\d+)_(?P<role>expand|proj)_e(?P<e>\d+)_"
        r"(?P<res>\d+)_(?P<cin>\d+)_(?P<cout>\d+)",
        case_name,
    )
    if match:
        role = "pw_expand" if match.group("role") == "expand" else "pw_project"
        return (
            _spec(
                op="conv",
                kernel_size=1,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                expand_ratio=int(match.group("e")),
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "conv",
                "lookup_op": "conv",
                "layer_role": f"ir{match.group('ir')}",
                "component_role": role,
            },
        )

    match = re.match(
        r"^pw_conv_pw_ir(?P<ir>\d+)_proj_(?P<res>\d+)_(?P<cin>\d+)_(?P<cout>\d+)",
        case_name,
    )
    if match:
        return (
            _spec(
                op="conv",
                kernel_size=1,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "conv",
                "lookup_op": "conv",
                "layer_role": f"ir{match.group('ir')}",
                "component_role": "pw_project",
            },
        )

    match = re.match(
        r"^dw_conv_k(?P<k>\d+)_dw_ir(?P<ir>\d+)_e(?P<e>\d+)_(?P<res>\d+)_"
        r"c(?P<c>\d+)_s(?P<stride>\d+)",
        case_name,
    )
    if match:
        channels = int(match.group("c"))
        op = f"dw_conv_k{match.group('k')}"
        return (
            _spec(
                op=op,
                kernel_size=int(match.group("k")),
                in_channels=channels,
                out_channels=channels,
                stride=int(match.group("stride")),
                groups=channels,
                expand_ratio=int(match.group("e")),
                resolution=int(match.group("res")),
            ),
            {
                "op_type": op,
                "lookup_op": op,
                "layer_role": f"ir{match.group('ir')}",
                "component_role": "depthwise_conv",
            },
        )

    match = re.match(r"^skip_skip_(?P<res>\d+)_(?P<c>\d+)", case_name)
    if match:
        channels = int(match.group("c"))
        return (
            _spec(
                op="skip",
                kernel_size=1,
                in_channels=channels,
                out_channels=channels,
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "skip",
                "lookup_op": "skip",
                "layer_role": "skip",
                "component_role": "identity_skip",
            },
        )

    match = re.match(r"^global_avg_pool_gap_(?P<res>\d+)_(?P<c>\d+)", case_name)
    if match:
        channels = int(match.group("c"))
        return (
            _spec(
                op="global_avg_pool",
                kernel_size=1,
                in_channels=channels,
                out_channels=channels,
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "global_avg_pool",
                "lookup_op": "global_avg_pool",
                "layer_role": "head",
                "component_role": "gap",
            },
        )

    match = re.match(r"^gap_(?P<res>\d+)_(?P<c>\d+)_arch84", case_name)
    if match:
        channels = int(match.group("c"))
        return (
            _spec(
                op="global_avg_pool",
                kernel_size=1,
                in_channels=channels,
                out_channels=channels,
                resolution=int(match.group("res")),
            ),
            {
                "op_type": "global_avg_pool",
                "lookup_op": "global_avg_pool",
                "layer_role": "arch84_head",
                "component_role": "gap",
            },
        )

    match = re.match(r"^fc_layer_fc_1_(?P<cin>\d+)_(?P<cout>\d+)", case_name)
    if match:
        return (
            _spec(
                op="fc_layer",
                kernel_size=1,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                resolution=1,
            ),
            {
                "op_type": "fc_layer",
                "lookup_op": "fc_layer",
                "layer_role": "head",
                "component_role": "fc",
            },
        )

    match = re.match(r"^fc_1_(?P<cin>\d+)_(?P<cout>\d+)_arch84", case_name)
    if match:
        return (
            _spec(
                op="fc_layer",
                kernel_size=1,
                in_channels=int(match.group("cin")),
                out_channels=int(match.group("cout")),
                resolution=1,
            ),
            {
                "op_type": "fc_layer",
                "lookup_op": "fc_layer",
                "layer_role": "arch84_head",
                "component_role": "fc",
            },
        )

    return (
        _spec(
            op="unparsed_board_case",
            kernel_size=1,
            in_channels=1,
            out_channels=1,
            resolution=1,
        ),
        {
            "op_type": "unparsed_board_case",
            "lookup_op": "unparsed_board_case",
            "layer_role": "unknown",
            "component_role": "unknown",
        },
    )


def _row_from_result(path: Path, *, namespace: str) -> dict[str, Any]:
    data = _read_json(path)
    case_name = _clean_case_name(data, path)
    op_spec, parsed = _parse_case_to_op(case_name)

    uart_status_code = _as_int(data.get("uart_status_code"))
    wns = _as_float(data.get("wns_ns"))
    reported_timing_clean = _as_bool(data.get("timing_clean"))
    timing_clean = bool(uart_status_code == 0 and wns is not None and wns >= 0.0)
    strict_usable = timing_clean

    if strict_usable:
        status = "measured"
        notes = "strict board LUT cost: UART status_code=0 and WNS>=0"
    elif uart_status_code == 0 and wns is not None:
        status = "timing_fail_reference"
        notes = "UART ok but WNS<0; retained only as reference, excluded from strict NAS cost"
    else:
        status = "missing_unusable"
        notes = "not strict usable under UART status_code=0 and WNS>=0"

    if reported_timing_clean != timing_clean:
        notes = f"{notes}; reported_timing_clean={reported_timing_clean}"
    if parsed["op_type"] == "unparsed_board_case":
        notes = f"{notes}; case_name parse fallback used"

    bram = _as_float(data.get("board_harness_bram_tile"))
    bram_entry_ceil = None if bram is None else int(math.ceil(bram))
    op_spec_json = json.dumps(op_spec, sort_keys=True, separators=(",", ":"))

    return {
        "case_name": case_name,
        "op_type": parsed["op_type"],
        "lookup_op": parsed["lookup_op"],
        "decomposition": str(data.get("decomposition", "")),
        "layer_role": parsed["layer_role"],
        "component_role": parsed["component_role"],
        "latency_cycles": _as_int(data.get("board_combo_cycles")),
        "latency_ms": _as_float(data.get("board_combo_latency_ms")),
        "LUT": _as_int(data.get("board_harness_lut")),
        "FF": _as_int(data.get("board_harness_ff")),
        "DSP": _as_int(data.get("board_harness_dsp")),
        "BRAM": bram,
        "BRAM_entry_ceil": bram_entry_ceil,
        "WNS": wns,
        "UART status_code": uart_status_code,
        "timing_clean": timing_clean,
        "strict_usable": strict_usable,
        "status": status,
        "source_namespace": namespace,
        "source_json": _source_path(path),
        "power_w": None,
        "notes": notes,
        "op_spec": op_spec,
        "op_spec_json": op_spec_json,
    }


def _e2e_boundary_row(e2e_summary_path: Path, blockers_path: Path, recovery_path: Path) -> dict[str, Any]:
    summary = _read_json(e2e_summary_path) if e2e_summary_path.exists() else {}
    case_name = str(summary.get("target_name") or "sonar_classifier_top8_03_arch_84_e2e")
    build_result_path = (
        e2e_summary_path.parent
        / "real_measurement"
        / case_name
        / "build_result.json"
    )
    promotion_manifest_path = (
        e2e_summary_path.parent
        / "real_measurement"
        / case_name
        / "official_promotion_manifest.json"
    )
    build_result = _read_json(build_result_path) if build_result_path.exists() else {}
    op_spec = _spec(
        op="e2e_network",
        kernel_size=1,
        in_channels=1,
        out_channels=int(summary.get("classifier_class_count") or 8),
        resolution=224,
    )
    op_spec_json = json.dumps(op_spec, sort_keys=True, separators=(",", ":"))
    uart_status_code = _as_int(summary.get("uart_status_code"))
    wns = _as_float(summary.get("wns_ns"))
    cycles = _as_int(summary.get("e2e_cycles"))
    latency_ms = _as_float(summary.get("e2e_latency_ms"))
    strict_usable = bool(
        summary.get("strict_latency_usable") is True
        and uart_status_code == 0
        and wns is not None
        and wns >= 0.0
        and cycles is not None
        and latency_ms is not None
    )
    source_namespace = "sonar_classifier_e2e_official" if strict_usable else "sonar_classifier_e2e_boundary"
    status = "measured" if strict_usable else "missing_unusable"
    notes = (
        "real COM5 complete-network e2e latency; official implementation "
        f"promoted_from_variant={summary.get('promoted_from_variant', '')}"
        if strict_usable
        else (
            "no strict real COM5 full-network e2e latency; canonical e2e is not "
            "eligible for NAS cost until UART status_code=0 and WNS>=0"
        )
    )
    bram = _as_float(build_result.get("bram_tile") or summary.get("resources", {}).get("bram_tile"))
    bram_entry_ceil = None if bram is None else int(math.ceil(bram))
    return {
        "case_name": case_name,
        "op_type": "e2e_network",
        "lookup_op": "e2e_network",
        "decomposition": "complete_network_harness",
        "layer_role": "e2e",
        "component_role": "canonical_e2e",
        "latency_cycles": cycles if strict_usable else None,
        "latency_ms": latency_ms if strict_usable else None,
        "LUT": _as_int(build_result.get("lut") or summary.get("resources", {}).get("lut")) if strict_usable else None,
        "FF": _as_int(build_result.get("ff") or summary.get("resources", {}).get("ff")) if strict_usable else None,
        "DSP": _as_int(build_result.get("dsp") or summary.get("resources", {}).get("dsp")) if strict_usable else None,
        "BRAM": bram if strict_usable else None,
        "BRAM_entry_ceil": bram_entry_ceil if strict_usable else None,
        "WNS": wns,
        "UART status_code": uart_status_code,
        "timing_clean": strict_usable,
        "strict_usable": strict_usable,
        "status": status,
        "source_namespace": source_namespace,
        "source_json": _source_path(e2e_summary_path),
        "power_w": None,
        "notes": notes,
        "op_spec": op_spec,
        "op_spec_json": op_spec_json,
        "evidence_paths": {
            "summary_json": _source_path(e2e_summary_path),
            "blockers_csv": _source_path(blockers_path),
            "recovery_md": _source_path(recovery_path),
            "build_result_json": _source_path(build_result_path),
            "promotion_manifest_json": _source_path(promotion_manifest_path),
            "measurement_json": str(summary.get("measurement_json_path", "")),
        },
    }


def _lut_entry_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "op_spec": row["op_spec"],
        "latency_ms": row["latency_ms"],
        "cycles": row["latency_cycles"],
        "dsp": row["DSP"],
        "bram": row["BRAM_entry_ceil"],
        "lut": row["LUT"],
        "power_w": None,
        "energy_mj": None,
        "ff": row["FF"],
        "bram_board_tile": row["BRAM"],
        "case_name": row["case_name"],
        "source_namespace": row["source_namespace"],
        "source_json": row["source_json"],
    }


def _status_entry_from_row(row: dict[str, Any]) -> dict[str, Any]:
    defer_reason = None
    root_cause_bucket = None
    if row["status"] == "timing_fail_reference":
        defer_reason = "timing_fail_wns_negative"
        root_cause_bucket = "timing_violation"
    elif row["status"] == "missing_unusable":
        defer_reason = "no_real_com5_e2e_latency"
        root_cause_bucket = "e2e_latency_unavailable"

    return {
        "case_name": row["case_name"],
        "status": row["status"],
        "op_type": row["op_type"],
        "lookup_op": row["lookup_op"],
        "defer_reason": defer_reason,
        "root_cause_bucket": root_cause_bucket,
        "op_spec": row["op_spec"],
        "board_cycles": row["latency_cycles"],
        "board_latency_ms": row["latency_ms"],
        "wns_ns": row["WNS"],
        "uart_status_code": row["UART status_code"],
        "timing_clean": row["timing_clean"],
        "strict_usable": row["strict_usable"],
        "source_namespace": row["source_namespace"],
        "source_json": row["source_json"],
        "notes": row["notes"],
    }


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in ROW_FIELDS})


def _write_table_json(path: Path, *, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {
        "metadata": metadata,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    e2e_measured = summary["by_source_status"].get("sonar_classifier_e2e_official", {}).get("measured", 0) == 1
    e2e_text = (
        "A real complete-network e2e latency row is exported because the official "
        "harness has COM5 status_code=0 and WNS>=0."
        if e2e_measured
        else (
            "No full-network e2e latency is exported here. Complete-network e2e "
            "latency may only come from a complete-network harness real COM5 measurement."
        )
    )
    text = f"""# NAS Board LUT Strict Current84/Arch84

Generated at: {summary["generated_at"]}

This is a derived lookup table for NAS search/evaluation. It does not modify
the raw deployable LUT ledgers or the raw board measurement ledger.

## Strict Definition

An entry is strict usable only when both conditions hold:

- UART status_code = 0
- WNS >= 0 ns

Timing-fail rows with UART status_code = 0 are retained only in
`timing_fail_reference.*`. They are not present in `nas_board_lut_strict.json`
and must not be used as strict NAS hardware cost.

## Counts

- strict usable rows: {summary["strict_usable_count"]}
- timing-fail reference rows: {summary["timing_fail_reference_count"]}
- missing/unusable rows: {summary["missing_unusable_count"]}
- strict current84 rows: {summary["by_source_status"].get("current84_fullcombo84", {}).get("measured", 0)}
- strict arch84 extension rows: {summary["by_source_status"].get("arch84_e2e_extension", {}).get("measured", 0)}
- strict official e2e rows: {summary["by_source_status"].get("sonar_classifier_e2e_official", {}).get("measured", 0)}

## Outputs

- `strict_usable.csv/json`: strict UART-ok and timing-clean operator/combo measurements.
- `timing_fail_reference.csv/json`: UART-ok but timing-fail reference rows.
- `missing_unusable.csv/json`: unavailable rows, including canonical e2e latency.
- `nas_board_lut_strict.json`: `LutTable.load_json` compatible strict entries only.
- `nas_board_lut_status.json`: strict status table for NAS exact lookup.
- `summary.json`: generation provenance and counts.

## E2E Boundary

{e2e_text}

Operator rows remain operator-level board costs. The official e2e row, when
present, is a separate complete-network measurement and must not be confused
with primitive or combo latency sums.

## Resource Semantics

`BRAM` in the CSV/row tables preserves the board harness tile value, including
fractional values. `nas_board_lut_strict.json` stores `LutEntry.bram` as
ceil(`BRAM`) so the existing cost estimator receives an integer BRAM block
count. `power_w` is intentionally null because no strict measured power value is
available in these board result files.
"""
    path.write_text(text, encoding="utf-8")


def generate(
    *,
    current84_dir: Path,
    arch84_dir: Path,
    e2e_dir: Path,
    output_dir: Path,
    allow_count_drift: bool = False,
) -> dict[str, Any]:
    current_case_dir = current84_dir / "case_results"
    arch_case_dir = arch84_dir / "case_results"
    e2e_summary_path = e2e_dir / "sonar_classifier_e2e_summary.json"
    e2e_blockers_path = e2e_dir / "sonar_classifier_e2e_blockers.csv"
    e2e_recovery_path = (
        e2e_dir
        / "real_measurement"
        / "sonar_classifier_top8_03_arch_84_e2e"
        / "implementation_recovery_20260523.md"
    )

    current_paths = sorted(current_case_dir.glob("*.result.json"))
    arch_paths = sorted(arch_case_dir.glob("*.result.json"))
    if not current_paths:
        raise FileNotFoundError(f"No current84 case results found: {current_case_dir}")
    if not arch_paths:
        raise FileNotFoundError(f"No arch84 case results found: {arch_case_dir}")
    if not e2e_summary_path.is_file():
        raise FileNotFoundError(f"Missing e2e summary: {e2e_summary_path}")

    current_rows = [
        _row_from_result(path, namespace="current84_fullcombo84")
        for path in current_paths
    ]
    arch_rows = [
        _row_from_result(path, namespace="arch84_e2e_extension")
        for path in arch_paths
    ]
    e2e_boundary_row = _e2e_boundary_row(e2e_summary_path, e2e_blockers_path, e2e_recovery_path)
    e2e_measured_rows = [e2e_boundary_row] if e2e_boundary_row["status"] == "measured" else []
    missing_rows = [] if e2e_boundary_row["status"] == "measured" else [e2e_boundary_row]
    measured_rows = current_rows + arch_rows + e2e_measured_rows

    strict_rows = [row for row in measured_rows if row["status"] == "measured"]
    timing_fail_rows = [
        row for row in measured_rows if row["status"] == "timing_fail_reference"
    ]
    unusable_rows = [
        row for row in measured_rows if row["status"] == "missing_unusable"
    ] + missing_rows

    current_status_counts = Counter(row["status"] for row in current_rows)
    arch_status_counts = Counter(row["status"] for row in arch_rows)
    e2e_status_counts = Counter(row["status"] for row in [e2e_boundary_row])
    expected = {
        "current84_measured": 54,
        "current84_timing_fail_reference": 30,
        "arch84_measured": 3,
        "e2e_measured": 1 if e2e_boundary_row["status"] == "measured" else 0,
        "e2e_missing_unusable": 0 if e2e_boundary_row["status"] == "measured" else 1,
    }
    observed = {
        "current84_measured": current_status_counts.get("measured", 0),
        "current84_timing_fail_reference": current_status_counts.get(
            "timing_fail_reference", 0
        ),
        "arch84_measured": arch_status_counts.get("measured", 0),
        "e2e_measured": e2e_status_counts.get("measured", 0),
        "e2e_missing_unusable": e2e_status_counts.get("missing_unusable", 0),
    }
    if observed != expected and not allow_count_drift:
        raise RuntimeError(
            "Strict board LUT count mismatch: "
            f"expected {expected}, observed {observed}. "
            "Pass --allow-count-drift only after reviewing source evidence."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().replace(microsecond=0).isoformat()
    status_counts = Counter(row["status"] for row in strict_rows + timing_fail_rows + unusable_rows)
    by_source_status = {
        "current84_fullcombo84": dict(current_status_counts),
        "arch84_e2e_extension": dict(arch_status_counts),
        e2e_boundary_row["source_namespace"]: dict(e2e_status_counts),
    }
    raw_input_hashes = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in PROTECTED_RAW_INPUTS
    }

    common_metadata = {
        "generated_at": generated_at,
        "generator": str(Path(__file__).resolve()),
        "strict_definition": "UART status_code=0 and WNS>=0",
        "latency_scope": (
            "operator_lut_plus_real_complete_network_e2e_measurement"
            if e2e_boundary_row["status"] == "measured"
            else "operator_lut_aggregate_estimate_only"
        ),
        "e2e_latency_policy": (
            "complete-network e2e row is included only when the official harness has real COM5 status_code=0 and WNS>=0"
        ),
        "power_policy": "power_w is null; no estimated power is promoted as measured power",
        "bram_policy": "row BRAM preserves board harness tile float; LutEntry.bram uses ceil(BRAM)",
        "source_files": {
            "inventory": str(
                (BOARD_RESULTS_ROOT / "current84_arch84_board_file_inventory.md").resolve()
            ),
            "current84_summary": str(
                (current84_dir / "v2_current84_fullcombo84_summary.json").resolve()
            ),
            "current84_status": str(
                (current84_dir / "v2_current84_fullcombo84_board_status.csv").resolve()
            ),
            "arch84_summary": str(
                (arch84_dir / "arch84_e2e_extension_summary.json").resolve()
            ),
            "arch84_status": str(
                (arch84_dir / "arch84_e2e_extension_combo_status.csv").resolve()
            ),
            "e2e_summary": str(e2e_summary_path.resolve()),
            "e2e_blockers": str(e2e_blockers_path.resolve()),
            "e2e_recovery": str(e2e_recovery_path.resolve()),
        },
        "protected_raw_input_hashes": raw_input_hashes,
    }
    summary = {
        **common_metadata,
        "current84_case_result_count": len(current_rows),
        "arch84_case_result_count": len(arch_rows),
        "strict_usable_count": len(strict_rows),
        "timing_fail_reference_count": len(timing_fail_rows),
        "missing_unusable_count": len(unusable_rows),
        "status_counts": dict(status_counts),
        "by_source_status": by_source_status,
        "expected_counts": expected,
        "observed_counts": observed,
        "unparsed_case_count": sum(
            1
            for row in strict_rows + timing_fail_rows + unusable_rows
            if row["op_type"] == "unparsed_board_case"
        ),
    }

    _write_csv(output_dir / "strict_usable.csv", strict_rows)
    _write_csv(output_dir / "timing_fail_reference.csv", timing_fail_rows)
    _write_csv(output_dir / "missing_unusable.csv", unusable_rows)
    _write_table_json(
        output_dir / "strict_usable.json",
        metadata=common_metadata,
        rows=strict_rows,
    )
    _write_table_json(
        output_dir / "timing_fail_reference.json",
        metadata=common_metadata,
        rows=timing_fail_rows,
    )
    _write_table_json(
        output_dir / "missing_unusable.json",
        metadata=common_metadata,
        rows=unusable_rows,
    )

    lut_payload = {
        "metadata": {
            **common_metadata,
            "measured_entries": len(strict_rows),
            "strict_usable_count": len(strict_rows),
            "excluded_timing_fail_reference_count": len(timing_fail_rows),
            "excluded_missing_unusable_count": len(unusable_rows),
            "status_authoritative": True,
        },
        "entries": [_lut_entry_from_row(row) for row in strict_rows],
        "duplicates": {},
    }
    (output_dir / "nas_board_lut_strict.json").write_text(
        json.dumps(lut_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status_payload = {
        "metadata": {
            **common_metadata,
            "status_counts": dict(status_counts),
            "status_authoritative": True,
        },
        "entries": [
            _status_entry_from_row(row)
            for row in strict_rows + timing_fail_rows + unusable_rows
        ],
    }
    (output_dir / "nas_board_lut_status.json").write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_readme(output_dir / "README.md", summary)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current84-dir", type=Path, default=CURRENT84_DIR)
    parser.add_argument("--arch84-dir", type=Path, default=ARCH84_DIR)
    parser.add_argument("--e2e-dir", type=Path, default=E2E_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-count-drift",
        action="store_true",
        help="Allow count changes after manual source evidence review.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = generate(
        current84_dir=args.current84_dir,
        arch84_dir=args.arch84_dir,
        e2e_dir=args.e2e_dir,
        output_dir=args.output_dir,
        allow_count_drift=args.allow_count_drift,
    )
    print(
        "Generated strict NAS board LUT at "
        f"{args.output_dir.resolve()} "
        f"(strict={summary['strict_usable_count']}, "
        f"timing_fail_reference={summary['timing_fail_reference_count']}, "
        f"missing_unusable={summary['missing_unusable_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
