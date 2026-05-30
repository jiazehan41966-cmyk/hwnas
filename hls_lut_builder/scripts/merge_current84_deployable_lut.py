#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = BUILDER_ROOT / "results"


DEFAULT_CURRENT84_DIR = RESULTS_ROOT / "v2_current84"
DEFAULT_DEPLOYABLE_LUT = RESULTS_ROOT / "v2_deployable_lut" / "deployable_lut_v2.json"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "v2_current84_deployable_lut"


CSV_TRACE_FIELDS = [
    "component_case_name",
    "component_operator",
    "component_op_type",
    "shape_name",
    "selected_source",
    "selected_existing_case_name",
    "strict_two_source_status",
    "closed60_status",
    "output_source_dataset",
    "source_case_name",
    "status",
    "deployable_at_200mhz",
    "latency_ms",
    "cycles",
    "post_route_LUT",
    "post_route_FF",
    "post_route_DSP",
    "post_route_BRAM_18K_equiv",
    "post_route_Fmax_est",
    "power_w",
    "current_case_count",
    "current_cases",
    "component_roles",
    "notes",
]


CSV_GAP_FIELDS = [
    "component_case_name",
    "component_operator",
    "component_op_type",
    "shape_name",
    "selected_source",
    "selected_existing_case_name",
    "gap_reason",
    "current_case_count",
    "current_cases",
    "component_roles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current84 deployable primitive LUT artifacts.")
    parser.add_argument("--current84-dir", default=str(DEFAULT_CURRENT84_DIR))
    parser.add_argument("--deployable-lut", default=str(DEFAULT_DEPLOYABLE_LUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current84_dir = Path(args.current84_dir).expanduser().resolve()
    deployable_lut_path = Path(args.deployable_lut).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    map_rows = _read_csv(current84_dir / "current84_primitive_map.csv")
    primitive_rows = _unique_primitives(map_rows)
    usage = _primitive_usage(map_rows)

    deployable_payload = _read_json(deployable_lut_path)
    deployable_source = _load_existing_deployable(deployable_payload)
    current84_missing_source = _load_formal_source(
        name="v2_current84_missing",
        status_path=current84_dir / "formal_lut_status_v2_current84_missing.json",
        hls_csv_path=current84_dir / "current84_missing_hls.csv",
        vivado_csv_path=current84_dir / "current84_missing_vivado_downstream.csv",
    )
    current84_identity_source = _load_formal_source(
        name="v2_current84_identity_repack_14_64",
        status_path=current84_dir / "formal_lut_status_v2_current84_identity_repack_14_64.json",
        hls_csv_path=current84_dir / "current84_identity_repack_14_64_hls.csv",
        vivado_csv_path=current84_dir / "current84_identity_repack_14_64_vivado_downstream.csv",
    )
    current84_formal_sources = {
        "v2_current84_missing": current84_missing_source,
        "v2_current84_identity_repack_14_64": current84_identity_source,
    }

    strict_entries: list[dict[str, Any]] = []
    closed_entries: list[dict[str, Any]] = []
    trace_rows: list[dict[str, str]] = []
    strict_gaps: list[dict[str, str]] = []
    closed_gaps: list[dict[str, str]] = []

    seen_strict: set[str] = set()
    seen_closed: set[str] = set()
    for primitive in primitive_rows:
        strict_entry = _resolve_strict_entry(
            primitive=primitive,
            usage=usage,
            deployable_source=deployable_source,
            current84_formal_sources=current84_formal_sources,
        )
        closed_entry = strict_entry
        strict_status = "covered"

        if strict_entry is None:
            strict_status = "gap"
            strict_gaps.append(_gap_row(primitive, usage, "absent_from_deployable_lut_v2_and_current84_formal_sources"))

        if closed_entry is None:
            closed_gaps.append(_gap_row(primitive, usage, "absent_from_allowed_current84_deployable_sources"))
            trace_rows.append(_trace_row(primitive, usage, None, strict_status, "gap"))
            continue

        closed_entry = _with_current84_metadata(closed_entry, primitive, usage)
        closed_status = "covered"

        if strict_entry is not None:
            strict_entry = _with_current84_metadata(strict_entry, primitive, usage)
            if strict_entry["current84_component_case_name"] not in seen_strict:
                strict_entries.append(strict_entry)
                seen_strict.add(strict_entry["current84_component_case_name"])

        if closed_entry["current84_component_case_name"] not in seen_closed:
            closed_entries.append(closed_entry)
            seen_closed.add(closed_entry["current84_component_case_name"])
        trace_rows.append(_trace_row(primitive, usage, closed_entry, strict_status, closed_status))

    output_dir.mkdir(parents=True, exist_ok=True)
    strict_path = output_dir / "deployable_lut_v2_current84_strict2src.json"
    closed_path = output_dir / "deployable_lut_v2_current84.json"
    trace_path = output_dir / "current84_deployable_source_trace.csv"
    strict_gap_path = output_dir / "current84_deployable_strict2src_gaps.csv"
    closed_gap_path = output_dir / "current84_deployable_closed60_gaps.csv"
    summary_path = output_dir / "merge_summary_current84.json"
    readme_path = output_dir / "README.md"

    metadata_base = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "task": "current84 deployable primitive LUT merge",
        "latency_policy": "HLS/post-route primitive latency only; old board measured latency is not merged.",
        "entry_contract": "v2 post-route primitive implementation with source trace",
        "source_current84_primitive_map": str(current84_dir / "current84_primitive_map.csv"),
        "source_deployable_lut_v2": str(deployable_lut_path),
        "source_current84_missing_status": str(current84_dir / "formal_lut_status_v2_current84_missing.json"),
        "source_current84_missing_hls_csv": str(current84_dir / "current84_missing_hls.csv"),
        "source_current84_missing_vivado_csv": str(current84_dir / "current84_missing_vivado_downstream.csv"),
        "source_current84_identity_status": str(current84_dir / "formal_lut_status_v2_current84_identity_repack_14_64.json"),
        "source_current84_identity_hls_csv": str(current84_dir / "current84_identity_repack_14_64_hls.csv"),
        "source_current84_identity_vivado_csv": str(current84_dir / "current84_identity_repack_14_64_vivado_downstream.csv"),
        "unique_current84_primitives": len(primitive_rows),
    }

    _write_json(
        strict_path,
        {
            "metadata": {
                **metadata_base,
                "variant": "strict_two_source",
                "allowed_sources": [
                    "deployable_lut_v2.json",
                    "current84 formal measurements: current84_missing15 plus identity_repack_14_64 retest",
                ],
                "entry_count": len(strict_entries),
                "gap_count": len(strict_gaps),
                "source_counts": dict(Counter(entry["source_dataset"] for entry in strict_entries)),
            },
            "entries": strict_entries,
        },
    )
    _write_json(
        closed_path,
        {
            "metadata": {
                **metadata_base,
                "variant": "closed60_strict_current84_sources",
                "allowed_sources": [
                    "deployable_lut_v2.json",
                    "current84 formal measurements: current84_missing15 plus identity_repack_14_64 retest",
                ],
                "entry_count": len(closed_entries),
                "gap_count": len(closed_gaps),
                "source_counts": dict(Counter(entry["source_dataset"] for entry in closed_entries)),
            },
            "entries": closed_entries,
        },
    )
    _write_csv(trace_path, trace_rows, CSV_TRACE_FIELDS)
    _write_csv(strict_gap_path, strict_gaps, CSV_GAP_FIELDS)
    _write_csv(closed_gap_path, closed_gaps, CSV_GAP_FIELDS)

    summary = {
        **metadata_base,
        "outputs": {
            "strict_two_source_lut": str(strict_path),
            "closed60_lut": str(closed_path),
            "source_trace": str(trace_path),
            "strict_two_source_gaps": str(strict_gap_path),
            "closed60_gaps": str(closed_gap_path),
        },
        "strict_entry_count": len(strict_entries),
        "strict_gap_count": len(strict_gaps),
        "closed_entry_count": len(closed_entries),
        "closed_gap_count": len(closed_gaps),
        "strict_source_counts": dict(Counter(entry["source_dataset"] for entry in strict_entries)),
        "closed_source_counts": dict(Counter(entry["source_dataset"] for entry in closed_entries)),
        "trace_source_counts": dict(Counter(row["output_source_dataset"] for row in trace_rows)),
    }
    _write_json(summary_path, summary)
    readme_path.write_text(_readme(summary, strict_gaps, closed_gaps), encoding="utf-8")

    print(f"Wrote strict two-source LUT: {strict_path}")
    print(f"Wrote closed current84 LUT: {closed_path}")
    print(f"Wrote source trace: {trace_path}")
    print(f"Wrote strict gaps: {strict_gap_path}")
    print(f"Wrote closed gaps: {closed_gap_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote README: {readme_path}")
    print(f"Strict two-source entries: {len(strict_entries)} / {len(primitive_rows)}")
    print(f"Closed current84 entries: {len(closed_entries)} / {len(primitive_rows)}")


def _resolve_strict_entry(
    *,
    primitive: dict[str, str],
    usage: dict[str, dict[str, Any]],
    deployable_source: dict[str, Any],
    current84_formal_sources: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    selected_source = primitive["selected_source"]
    selected_case = primitive["selected_existing_case_name"]
    if selected_source.startswith("v2_deployable_lut:"):
        entry = deployable_source["by_source_case"].get(selected_case)
        return dict(entry) if entry is not None else None
    if selected_source in current84_formal_sources:
        return _entry_from_formal_source(
            primitive=primitive,
            usage=usage,
            source=current84_formal_sources[selected_source],
            case_name=selected_case,
            source_dataset=selected_source,
            source_note="current84_formal_measured_implementation",
        )
    return None


def _entry_from_formal_source(
    *,
    primitive: dict[str, str],
    usage: dict[str, dict[str, Any]],
    source: dict[str, Any],
    case_name: str,
    source_dataset: str,
    source_note: str,
) -> dict[str, Any] | None:
    status = source["status_by_case"].get(case_name)
    hls = source["hls_by_case"].get(case_name)
    vivado = source["vivado_by_case"].get(case_name)
    if status is None or hls is None or vivado is None:
        return None
    if not _is_deployable_status(status):
        return None
    if not _has_required_post_route_metrics(vivado):
        return None
    latency_ms = _to_float(hls.get("latency_ms"))
    cycles = _to_int(hls.get("latency_cycles"))
    power_w = _to_float(vivado.get("power_w"))
    return {
        "op_spec": status["op_spec"],
        "latency_ms": latency_ms,
        "cycles": cycles,
        "dsp": _to_int(vivado.get("post_route_DSP")),
        "bram": _bram18_total(vivado),
        "lut": _to_int(vivado.get("post_route_LUT")),
        "power_w": power_w,
        "energy_mj": None if latency_ms is None or power_w is None else latency_ms * power_w,
        "source_case_name": case_name,
        "source_dataset": source_dataset,
        "status": status["status"],
        "metrics_source": status.get("metrics_source", "vivado_actual.post_route"),
        "deployable_at_200mhz": True,
        "implementation_metrics": _implementation_metrics(vivado),
        "source_note": source_note,
    }


def _with_current84_metadata(
    entry: dict[str, Any],
    primitive: dict[str, str],
    usage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(entry)
    _normalize_top_level_metrics(merged)
    component_case = primitive["component_case_name"]
    merged["current84_component_case_name"] = component_case
    merged["current84_component_operator"] = primitive["component_operator"]
    merged["current84_component_op_type"] = primitive["component_op_type"]
    merged["current84_shape_name"] = primitive["shape_name"]
    merged["current84_selected_source"] = primitive["selected_source"]
    merged["current84_selected_existing_case_name"] = primitive["selected_existing_case_name"]
    merged["current84_current_cases"] = sorted(usage[component_case]["current_cases"])
    merged["current84_component_roles"] = sorted(usage[component_case]["component_roles"])
    return merged


def _normalize_top_level_metrics(entry: dict[str, Any]) -> None:
    metrics = entry.get("implementation_metrics")
    if not isinstance(metrics, dict):
        return
    if entry.get("power_w") in (None, "") and metrics.get("power_w") not in (None, ""):
        entry["power_w"] = metrics.get("power_w")
    latency_ms = _to_float(entry.get("latency_ms"))
    power_w = _to_float(entry.get("power_w"))
    if entry.get("energy_mj") in (None, "") and latency_ms is not None and power_w is not None:
        entry["energy_mj"] = latency_ms * power_w


def _trace_row(
    primitive: dict[str, str],
    usage: dict[str, dict[str, Any]],
    entry: dict[str, Any] | None,
    strict_status: str,
    closed_status: str,
) -> dict[str, str]:
    component_case = primitive["component_case_name"]
    source_dataset = "" if entry is None else str(entry.get("source_dataset", ""))
    metrics = {} if entry is None else entry.get("implementation_metrics", {})
    return {
        "component_case_name": component_case,
        "component_operator": primitive["component_operator"],
        "component_op_type": primitive["component_op_type"],
        "shape_name": primitive["shape_name"],
        "selected_source": primitive["selected_source"],
        "selected_existing_case_name": primitive["selected_existing_case_name"],
        "strict_two_source_status": strict_status,
        "closed60_status": closed_status,
        "output_source_dataset": source_dataset,
        "source_case_name": "" if entry is None else str(entry.get("source_case_name", "")),
        "status": "" if entry is None else str(entry.get("status", "")),
        "deployable_at_200mhz": "" if entry is None else str(entry.get("deployable_at_200mhz", "")),
        "latency_ms": "" if entry is None else str(entry.get("latency_ms", "")),
        "cycles": "" if entry is None else str(entry.get("cycles", "")),
        "post_route_LUT": str(metrics.get("post_route_LUT", "")),
        "post_route_FF": str(metrics.get("post_route_FF", "")),
        "post_route_DSP": str(metrics.get("post_route_DSP", "")),
        "post_route_BRAM_18K_equiv": str(metrics.get("post_route_BRAM_18K_equiv", "")),
        "post_route_Fmax_est": str(metrics.get("post_route_Fmax_est", "")),
        "power_w": "" if entry is None else str(entry.get("power_w", "")),
        "current_case_count": str(len(usage[component_case]["current_cases"])),
        "current_cases": "|".join(sorted(usage[component_case]["current_cases"])),
        "component_roles": "|".join(sorted(usage[component_case]["component_roles"])),
        "notes": "" if entry is None else str(entry.get("source_note", "")),
    }


def _gap_row(
    primitive: dict[str, str],
    usage: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, str]:
    component_case = primitive["component_case_name"]
    return {
        "component_case_name": component_case,
        "component_operator": primitive["component_operator"],
        "component_op_type": primitive["component_op_type"],
        "shape_name": primitive["shape_name"],
        "selected_source": primitive["selected_source"],
        "selected_existing_case_name": primitive["selected_existing_case_name"],
        "gap_reason": reason,
        "current_case_count": str(len(usage[component_case]["current_cases"])),
        "current_cases": "|".join(sorted(usage[component_case]["current_cases"])),
        "component_roles": "|".join(sorted(usage[component_case]["component_roles"])),
    }


def _load_existing_deployable(payload: dict[str, Any]) -> dict[str, Any]:
    entries = payload.get("entries", [])
    return {
        "by_source_case": {entry.get("source_case_name", ""): entry for entry in entries},
        "by_spec": {_spec_key(entry["op_spec"]): entry for entry in entries if isinstance(entry.get("op_spec"), dict)},
    }


def _load_formal_source(
    *,
    name: str,
    status_path: Path,
    hls_csv_path: Path,
    vivado_csv_path: Path,
) -> dict[str, Any]:
    status_payload = _read_json(status_path)
    status_entries = status_payload.get("entries", [])
    hls_rows = _read_csv(hls_csv_path) if hls_csv_path.exists() else []
    vivado_rows = _read_csv(vivado_csv_path) if vivado_csv_path.exists() else []
    return {
        "name": name,
        "status_by_case": {entry.get("case_name", ""): entry for entry in status_entries},
        "hls_by_case": {row.get("case_name", ""): row for row in hls_rows},
        "vivado_by_case": {row.get("case_name", ""): row for row in vivado_rows},
    }


def _unique_primitives(map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_component: dict[str, dict[str, str]] = {}
    for row in map_rows:
        by_component.setdefault(row["component_case_name"], row)
    return sorted(by_component.values(), key=lambda row: (row["component_operator"], row["component_case_name"]))


def _primitive_usage(map_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    usage: dict[str, dict[str, Any]] = defaultdict(lambda: {"current_cases": set(), "component_roles": set()})
    for row in map_rows:
        item = usage[row["component_case_name"]]
        item["current_cases"].add(row["current_case_name"])
        item["component_roles"].add(row["component_role"])
    return usage


def _spec_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    h, w = _resolution(spec)
    return (
        _normalized_op(spec),
        int(spec.get("kernel_size", 0)),
        int(spec.get("in_channels", 0)),
        int(spec.get("out_channels", 0)),
        int(spec.get("stride", 1)),
        int(spec.get("groups", 1)),
        int(spec.get("expand_ratio", 1)),
        h,
        w,
        int(spec.get("bitwidth", 8)),
        int(spec.get("input_parallelism", 1)),
        int(spec.get("output_parallelism", 1)),
        int(spec.get("unroll_factor", 1)),
        _to_float(spec.get("target_clock_mhz")),
        _to_int(spec.get("pack_ch")),
        _to_int(spec.get("ch_block")),
        _to_int(spec.get("target_ii")),
        spec.get("tile_order"),
        spec.get("stream_order"),
        str(spec.get("dsp_pack")) if spec.get("dsp_pack") is not None else None,
    )


def _normalized_op(spec: dict[str, Any]) -> str:
    op = str(spec.get("op", ""))
    if op == "conv" and int(spec.get("kernel_size", 0)) == 1 and int(spec.get("groups", 1)) == 1:
        return "pw_conv"
    return op


def _resolution(spec: dict[str, Any]) -> tuple[int, int]:
    resolution = spec.get("input_resolution", [0, 0])
    return int(resolution[0]), int(resolution[1])


def _is_deployable_status(entry: dict[str, Any]) -> bool:
    return (
        entry.get("status") == "measured_implementation"
        and str(entry.get("deployable_at_200mhz")).lower() == "true"
        and str(entry.get("downstream_status", "success")) in {"success", "skipped_existing", ""}
    )


def _has_required_post_route_metrics(row: dict[str, Any]) -> bool:
    required = [
        "post_route_LUT",
        "post_route_FF",
        "post_route_DSP",
        "post_route_RAMB18",
        "post_route_RAMB36",
        "post_route_Fmax_est",
        "power_w",
    ]
    return bool(row) and all(str(row.get(key, "")).strip() != "" for key in required)


def _implementation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_part": row.get("target_part"),
        "target_clock_mhz": _to_float(row.get("target_clock_mhz")),
        "post_route_LUT": _to_int(row.get("post_route_LUT")),
        "post_route_FF": _to_int(row.get("post_route_FF")),
        "post_route_DSP": _to_int(row.get("post_route_DSP")),
        "post_route_RAMB18": _to_int(row.get("post_route_RAMB18")),
        "post_route_RAMB36": _to_int(row.get("post_route_RAMB36")),
        "post_route_BRAM_18K_equiv": _bram18_total(row),
        "post_route_Fmax_est": _to_float(row.get("post_route_Fmax_est")),
        "power_w": _to_float(row.get("power_w")),
        "power_source": row.get("power_source"),
        "downstream_status": row.get("downstream_status"),
        "report_dir": row.get("report_dir"),
    }


def _bram18_total(row: dict[str, Any]) -> int | None:
    ramb18 = _to_int(row.get("post_route_RAMB18"))
    ramb36 = _to_int(row.get("post_route_RAMB36"))
    if ramb18 is None and ramb36 is None:
        return None
    return (ramb18 or 0) + 2 * (ramb36 or 0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(BUILDER_ROOT.parent.resolve())).replace("\\", "/")
    except ValueError:
        return path


def _readme(summary: dict[str, Any], strict_gaps: list[dict[str, str]], closed_gaps: list[dict[str, str]]) -> str:
    lines = [
        "# current84 Deployable Primitive LUT",
        "",
        "This directory contains current84 deployable primitive LUT artifacts.",
        "The old 40 board-measured latencies are not merged into these files.",
        "",
        "## Outputs",
        "",
        "- `deployable_lut_v2_current84.json`: closed current84 primitive LUT.",
        "- `deployable_lut_v2_current84_strict2src.json`: strict two-source variant.",
        "- `current84_deployable_source_trace.csv`: one row per current84 unique primitive.",
        "- `current84_deployable_strict2src_gaps.csv`: gaps under the strict two-source policy.",
        "- `current84_deployable_closed60_gaps.csv`: gaps in the closed LUT variant.",
        "- `merge_summary_current84.json`: machine-readable summary.",
        "",
        "## Source Policy",
        "",
        "Strict two-source policy uses only:",
        "- `hls_lut_builder/results/v2_deployable_lut/deployable_lut_v2.json`",
        "- current84 formal measurements: `formal_lut_status_v2_current84_missing.json` plus `formal_lut_status_v2_current84_identity_repack_14_64.json`, with their HLS/Vivado CSV metrics",
        "",
        "`identity_repack_14_64` was remeasured under the current84 contract, so the closed 60-entry variant no longer uses the `v2_fixed_vector` exception.",
        "",
        "## Counts",
        "",
        f"- Unique current84 primitives: {summary['unique_current84_primitives']}",
        f"- Strict two-source entries: {summary['strict_entry_count']}",
        f"- Strict two-source gaps: {summary['strict_gap_count']}",
        f"- Closed LUT entries: {summary['closed_entry_count']}",
        f"- Closed LUT gaps: {summary['closed_gap_count']}",
        f"- Strict source counts: {summary['strict_source_counts']}",
        f"- Closed source counts: {summary['closed_source_counts']}",
        "",
        "## Strict Gaps",
        "",
    ]
    if strict_gaps:
        for row in strict_gaps:
            lines.append(
                f"- `{row['component_case_name']}` from `{row['selected_source']}`: {row['gap_reason']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Closed Gaps", ""])
    if closed_gaps:
        for row in closed_gaps:
            lines.append(
                f"- `{row['component_case_name']}` from `{row['selected_source']}`: {row['gap_reason']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python .\\hls_lut_builder\\scripts\\merge_current84_deployable_lut.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
