#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = BUILDER_ROOT / "results"

DEFAULT_FAILED44_DIR = RESULTS_ROOT / "v2_failed44_fixed_vector"
DEFAULT_K5_DIR = RESULTS_ROOT / "v2_k5_low_complexity"
DEFAULT_DW_FALLBACK_DIR = RESULTS_ROOT / "v2_depthwise_fallback_blockers"
DEFAULT_PW_DIR = RESULTS_ROOT / "v2_pointwise_v21"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "v2_deployable_lut"

BRAM18_BITS = 18_432


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge v2 measured implementations and failed44 combo status into deployable LUT artifacts."
    )
    parser.add_argument("--failed44-dir", default=str(DEFAULT_FAILED44_DIR))
    parser.add_argument("--k5-dir", default=str(DEFAULT_K5_DIR))
    parser.add_argument("--depthwise-fallback-dir", default=str(DEFAULT_DW_FALLBACK_DIR))
    parser.add_argument("--pointwise-dir", default=str(DEFAULT_PW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed44_dir = Path(args.failed44_dir).expanduser().resolve()
    k5_dir = Path(args.k5_dir).expanduser().resolve()
    depthwise_fallback_dir = Path(args.depthwise_fallback_dir).expanduser().resolve()
    pointwise_dir = Path(args.pointwise_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    sources = {
        "failed44_fixed_vector": _load_source(
            name="failed44_fixed_vector",
            status_path=failed44_dir / "formal_lut_status_v2_failed44_fixed_vector.json",
            lut_path=failed44_dir / "formal_lut_v2_failed44_fixed_vector.json",
            hls_csv_path=failed44_dir / "hls.csv",
            vivado_csv_path=failed44_dir / "vivado_downstream.csv",
        ),
        "k5_low_complexity": _load_source(
            name="k5_low_complexity",
            status_path=k5_dir / "formal_lut_status_v2_k5_low_complexity.json",
            lut_path=k5_dir / "formal_lut_v2_k5_low_complexity.json",
            hls_csv_path=k5_dir / "hls.csv",
            vivado_csv_path=k5_dir / "vivado_downstream.csv",
        ),
        "depthwise_fallback_blockers": _load_optional_source(
            name="depthwise_fallback_blockers",
            status_path=depthwise_fallback_dir / "formal_lut_status_v2_depthwise_fallback_blockers.json",
            lut_path=depthwise_fallback_dir / "formal_lut_v2_depthwise_fallback_blockers.json",
            hls_csv_path=depthwise_fallback_dir / "hls.csv",
            vivado_csv_path=depthwise_fallback_dir / "vivado_downstream.csv",
        ),
        "pointwise_v21": _load_source(
            name="pointwise_v21",
            status_path=pointwise_dir / "formal_lut_status_v2_pointwise_v21.json",
            lut_path=pointwise_dir / "formal_lut_v2_pointwise_v21.json",
            hls_csv_path=pointwise_dir / "hls.csv",
            vivado_csv_path=pointwise_dir / "vivado_downstream.csv",
        ),
    }

    deployable_entries = _collect_deployable_entries(sources)
    combo_rows = _build_combo_rows(
        mapping_rows=_read_csv(failed44_dir / "failed44_primitive_map.csv"),
        sources=sources,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    deployable_path = output_dir / "deployable_lut_v2.json"
    combo_path = output_dir / "combo_deployability_v2.csv"
    summary_path = output_dir / "merge_summary_v2.json"

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "merge_policy": "deployable_lut_v2_merge_policy",
        "entry_contract": "post_route_only",
        "source_dirs": {
            "failed44_fixed_vector": str(failed44_dir),
            "k5_low_complexity": str(k5_dir),
            "depthwise_fallback_blockers": str(depthwise_fallback_dir),
            "pointwise_v21": str(pointwise_dir),
        },
        "admission_rules": [
            "HLS synthesis success or skipped_existing with parsed report",
            "observed II <= target II",
            "Vivado downstream success",
            "post-route LUT/FF/DSP/BRAM/Fmax and power present",
            "deployable_at_200mhz is true",
        ],
    }
    deployable_payload = {
        "metadata": {
            **metadata,
            "entry_count": len(deployable_entries),
            "source_counts": dict(Counter(entry["source_dataset"] for entry in deployable_entries)),
        },
        "entries": deployable_entries,
    }
    deployable_path.write_text(json.dumps(deployable_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(combo_path, combo_rows)

    summary = {
        **metadata,
        "outputs": {
            "deployable_lut_v2": str(deployable_path),
            "combo_deployability_v2": str(combo_path),
        },
        "deployable_entry_count": len(deployable_entries),
        "deployable_source_counts": dict(Counter(entry["source_dataset"] for entry in deployable_entries)),
        "combo_status_counts": dict(Counter(row["combo_deployability_status"] for row in combo_rows)),
        "combo_reason_counts": dict(_reason_counter(combo_rows)),
        "pointwise_v21_ii2_hls_rows": sum(
            1
            for row in sources["pointwise_v21"]["hls_by_case"].values()
            if _to_int(row.get("target_ii")) == 2 and _to_int(row.get("II")) == 2
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote deployable LUT: {deployable_path}")
    print(f"Wrote combo deployability: {combo_path}")
    print(f"Wrote merge summary: {summary_path}")
    print(f"Deployable entries: {len(deployable_entries)}")
    print(f"Combo status counts: {summary['combo_status_counts']}")


def _load_source(
    *,
    name: str,
    status_path: Path,
    lut_path: Path,
    hls_csv_path: Path,
    vivado_csv_path: Path,
) -> dict[str, Any]:
    status_payload = _read_json(status_path)
    lut_payload = _read_json(lut_path)
    status_entries = status_payload.get("entries", [])
    lut_entries = lut_payload.get("entries", [])
    hls_rows = _read_csv(hls_csv_path) if hls_csv_path.exists() else []
    vivado_rows = _read_csv(vivado_csv_path) if vivado_csv_path.exists() else []

    return {
        "name": name,
        "status_path": status_path,
        "lut_path": lut_path,
        "hls_csv_path": hls_csv_path,
        "vivado_csv_path": vivado_csv_path,
        "status_by_case": {entry["case_name"]: entry for entry in status_entries},
        "status_by_shape": _index_status_by_shape(status_entries),
        "lut_by_spec": {_spec_key(entry["op_spec"]): entry for entry in lut_entries},
        "hls_by_case": {row["case_name"]: row for row in hls_rows},
        "vivado_by_case": {row["case_name"]: row for row in vivado_rows},
    }


def _load_optional_source(
    *,
    name: str,
    status_path: Path,
    lut_path: Path,
    hls_csv_path: Path,
    vivado_csv_path: Path,
) -> dict[str, Any]:
    if not status_path.exists() or not lut_path.exists():
        return {
            "name": name,
            "status_path": status_path,
            "lut_path": lut_path,
            "hls_csv_path": hls_csv_path,
            "vivado_csv_path": vivado_csv_path,
            "status_by_case": {},
            "status_by_shape": {},
            "lut_by_spec": {},
            "hls_by_case": {
                row["case_name"]: row for row in _read_csv(hls_csv_path)
            }
            if hls_csv_path.exists()
            else {},
            "vivado_by_case": {
                row["case_name"]: row for row in _read_csv(vivado_csv_path)
            }
            if vivado_csv_path.exists()
            else {},
        }
    return _load_source(
        name=name,
        status_path=status_path,
        lut_path=lut_path,
        hls_csv_path=hls_csv_path,
        vivado_csv_path=vivado_csv_path,
    )


def _collect_deployable_entries(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_specs: set[tuple[Any, ...]] = set()
    for source_name in ("failed44_fixed_vector", "k5_low_complexity", "depthwise_fallback_blockers", "pointwise_v21"):
        source = sources[source_name]
        for status_entry in source["status_by_case"].values():
            if not _is_deployable_status(status_entry):
                continue
            vivado_row = source["vivado_by_case"].get(status_entry["case_name"], {})
            if not _has_required_post_route_metrics(vivado_row):
                continue
            spec_key = _spec_key(status_entry["op_spec"])
            if spec_key in seen_specs:
                continue
            lut_entry = source["lut_by_spec"].get(spec_key)
            if lut_entry is None:
                continue
            merged = dict(lut_entry)
            merged["source_case_name"] = status_entry["case_name"]
            merged["source_dataset"] = source_name
            merged["status"] = status_entry["status"]
            merged["metrics_source"] = status_entry["metrics_source"]
            merged["deployable_at_200mhz"] = True
            merged["implementation_metrics"] = _implementation_metrics(vivado_row)
            entries.append(merged)
            seen_specs.add(spec_key)
    return entries


def _build_combo_rows(mapping_rows: list[dict[str, str]], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mapping_rows:
        grouped[row["old_case_name"]].append(row)

    combo_rows: list[dict[str, Any]] = []
    for old_case_name, rows in sorted(grouped.items()):
        selected_components = [_select_component(row, sources) for row in sorted(rows, key=_component_sort_key)]
        is_mbconv = rows[0]["old_op_family"].startswith("mbconv")
        buffer_info = _mbconv_buffer_info(selected_components) if is_mbconv else {}
        deployable_components = [component for component in selected_components if component["deployable"]]
        all_components_deployable = len(deployable_components) == len(selected_components)
        reasons = _combo_reasons(selected_components)

        if all_components_deployable:
            combo_status = "deployable_measured_with_buffers" if is_mbconv else "deployable_measured"
            non_deployable_reason = ""
        else:
            combo_status = "not_deployable"
            non_deployable_reason = "|".join(reasons)

        metric_rows = [component["metrics"] for component in selected_components]
        combo_rows.append(
            {
                "old_case_name": old_case_name,
                "old_op_family": rows[0]["old_op_family"],
                "decomposition": "pw_expand+dw+pw_project" if is_mbconv else "direct",
                "component_count": len(selected_components),
                "component_roles": "|".join(component["role"] for component in selected_components),
                "original_component_cases": "|".join(component["original_case_name"] for component in selected_components),
                "selected_component_cases": "|".join(component["selected_case_name"] for component in selected_components),
                "selected_sources": "|".join(component["source_dataset"] for component in selected_components),
                "selected_statuses": "|".join(component["status"] for component in selected_components),
                "selected_profile_notes": "|".join(component["selection_note"] for component in selected_components),
                "combo_deployability_status": combo_status,
                "non_deployable_reason": non_deployable_reason,
                "sum_latency_ms": _sum_float(metric_rows, "latency_ms"),
                "sum_LUT": _sum_int(metric_rows, "LUT"),
                "sum_FF": _sum_int(metric_rows, "FF"),
                "sum_DSP": _sum_int(metric_rows, "DSP"),
                "sum_BRAM_18K": _sum_int(metric_rows, "BRAM_18K"),
                "max_observed_II": _max_int(metric_rows, "II"),
                "inter_stage_buffer_BRAM_18K": buffer_info.get("inter_stage_buffer_BRAM_18K", ""),
                "expand_output_buffer_BRAM_18K": buffer_info.get("expand_output_buffer_BRAM_18K", ""),
                "depthwise_output_buffer_BRAM_18K": buffer_info.get("depthwise_output_buffer_BRAM_18K", ""),
                "notes": "conservative primitive resource sum; inter-stage buffers included"
                if is_mbconv
                else "",
            }
        )
    return combo_rows


def _select_component(row: dict[str, str], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    original_case_name = row["component_case_name"]
    original_status = sources["failed44_fixed_vector"]["status_by_case"].get(original_case_name)
    if original_status is None:
        return _missing_component(row, "failed44_component_missing")

    original_spec = original_status["op_spec"]
    normalized_op = _normalized_op(original_spec)
    selected_source = sources["failed44_fixed_vector"]
    selected_status = original_status
    selection_note = "failed44_original_profile"

    if normalized_op == "pw_conv":
        candidate = _best_pointwise_v21(original_spec, sources["pointwise_v21"])
        if candidate is not None:
            selected_source = sources["pointwise_v21"]
            selected_status = candidate
            selection_note = "pointwise_v21_p16_pe16_simd4_ii2"
    elif normalized_op == "dw_conv_k5":
        candidate = _deployable_k5_low_complexity(original_spec, sources["k5_low_complexity"])
        if candidate is not None:
            selected_source = sources["k5_low_complexity"]
            selected_status = candidate
            selection_note = "k5_low_complexity_p8_cb8_ii2"

    if (
        normalized_op.startswith("dw_conv")
        and not _component_is_deployable(selected_status, selected_source)
    ):
        candidate = _best_depthwise_fallback(original_spec, sources["depthwise_fallback_blockers"])
        if candidate is not None:
            selected_source = sources["depthwise_fallback_blockers"]
            selected_status = candidate
            selection_note = "depthwise_fallback_p8_cb8"

    if (
        normalized_op != "pw_conv"
        and not _component_is_deployable(selected_status, selected_source)
    ):
        candidate = _deployable_same_shape(original_spec, sources["failed44_fixed_vector"])
        if candidate is not None:
            selected_source = sources["failed44_fixed_vector"]
            selected_status = candidate
            selection_note = "failed44_same_shape_measured"

    metrics = _component_metrics(selected_status, selected_source)
    return {
        "role": row["component_role"],
        "original_case_name": original_case_name,
        "selected_case_name": selected_status["case_name"],
        "source_dataset": selected_source["name"],
        "status": selected_status["status"],
        "deployable": _component_is_deployable(selected_status, selected_source),
        "selection_note": selection_note,
        "op_spec": selected_status["op_spec"],
        "metrics": metrics,
    }


def _best_pointwise_v21(original_spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        entry
        for entry in source["status_by_shape"].get(_shape_key(original_spec), [])
        if _to_int(entry["op_spec"].get("target_ii")) == 2
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: _pointwise_candidate_rank(entry, source))[0]


def _pointwise_candidate_rank(entry: dict[str, Any], source: dict[str, Any]) -> tuple[int, float, str]:
    vivado_row = source["vivado_by_case"].get(entry["case_name"], {})
    if _is_deployable_status(entry) and _has_required_post_route_metrics(vivado_row):
        deployable_rank = 0
    elif entry.get("status") == "measured_hls_only":
        deployable_rank = 1
    else:
        deployable_rank = 2
    latency = _to_float(source["hls_by_case"].get(entry["case_name"], {}).get("latency_ms"))
    return deployable_rank, latency if latency is not None else float("inf"), entry["case_name"]


def _deployable_k5_low_complexity(original_spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    candidates = source["status_by_shape"].get(_shape_key(original_spec), [])
    deployable = [
        entry
        for entry in candidates
        if _is_deployable_status(entry)
        and _has_required_post_route_metrics(source["vivado_by_case"].get(entry["case_name"], {}))
    ]
    if not deployable:
        return None
    return sorted(deployable, key=lambda entry: entry["case_name"])[0]


def _best_depthwise_fallback(original_spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    candidates = source["status_by_shape"].get(_shape_key(original_spec), [])
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: _fallback_candidate_rank(entry, source))[0]


def _fallback_candidate_rank(entry: dict[str, Any], source: dict[str, Any]) -> tuple[int, str]:
    vivado_row = source["vivado_by_case"].get(entry["case_name"], {})
    if _is_deployable_status(entry) and _has_required_post_route_metrics(vivado_row):
        return 0, entry["case_name"]
    if entry.get("status") in {"downstream_failed", "downstream_timeout"} or str(entry.get("status", "")).startswith("downstream_"):
        return 1, entry["case_name"]
    if entry.get("status") == "measured_hls_only":
        return 2, entry["case_name"]
    return 3, entry["case_name"]


def _deployable_same_shape(original_spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    candidates = source["status_by_shape"].get(_shape_key(original_spec), [])
    deployable = [
        entry
        for entry in candidates
        if _is_deployable_status(entry)
        and _has_required_post_route_metrics(source["vivado_by_case"].get(entry["case_name"], {}))
    ]
    if not deployable:
        return None
    return sorted(deployable, key=lambda entry: entry["case_name"])[0]


def _component_is_deployable(status_entry: dict[str, Any], source: dict[str, Any]) -> bool:
    return _is_deployable_status(status_entry) and _has_required_post_route_metrics(
        source["vivado_by_case"].get(status_entry["case_name"], {})
    )


def _component_metrics(status_entry: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    case_name = status_entry["case_name"]
    vivado_row = source["vivado_by_case"].get(case_name)
    if vivado_row and _has_required_post_route_metrics(vivado_row):
        return {
            "latency_ms": _lut_metric(source, status_entry, "latency_ms"),
            "LUT": _to_int(vivado_row.get("post_route_LUT")),
            "FF": _to_int(vivado_row.get("post_route_FF")),
            "DSP": _to_int(vivado_row.get("post_route_DSP")),
            "BRAM_18K": _bram18_total(vivado_row),
            "II": _hls_ii(source, case_name),
        }

    hls_row = source["hls_by_case"].get(case_name, {})
    return {
        "latency_ms": _to_float(hls_row.get("latency_ms")),
        "LUT": _to_int(hls_row.get("LUT")),
        "FF": _to_int(hls_row.get("FF")),
        "DSP": _to_int(hls_row.get("DSP")),
        "BRAM_18K": _to_int(hls_row.get("BRAM_18K")),
        "II": _to_int(hls_row.get("II")),
    }


def _lut_metric(source: dict[str, Any], status_entry: dict[str, Any], key: str) -> Any:
    lut_entry = source["lut_by_spec"].get(_spec_key(status_entry["op_spec"]), {})
    return lut_entry.get(key)


def _hls_ii(source: dict[str, Any], case_name: str) -> int | None:
    return _to_int(source["hls_by_case"].get(case_name, {}).get("II"))


def _mbconv_buffer_info(components: list[dict[str, Any]]) -> dict[str, int]:
    by_role = {component["role"]: component for component in components}
    expand_spec = by_role.get("pw_expand", {}).get("op_spec")
    project_spec = by_role.get("pw_project", {}).get("op_spec")
    if not isinstance(expand_spec, dict) or not isinstance(project_spec, dict):
        return {}

    expand_h, expand_w = _resolution(expand_spec)
    project_h, project_w = _resolution(project_spec)
    bitwidth = int(expand_spec.get("bitwidth", 8))
    hidden_channels = int(expand_spec["out_channels"])

    expand_bits = expand_h * expand_w * hidden_channels * bitwidth
    depthwise_bits = project_h * project_w * hidden_channels * bitwidth
    expand_bram = math.ceil(expand_bits / BRAM18_BITS)
    depthwise_bram = math.ceil(depthwise_bits / BRAM18_BITS)
    return {
        "expand_output_buffer_BRAM_18K": expand_bram,
        "depthwise_output_buffer_BRAM_18K": depthwise_bram,
        "inter_stage_buffer_BRAM_18K": expand_bram + depthwise_bram,
    }


def _combo_reasons(components: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for component in components:
        if component["deployable"]:
            continue
        status = component["status"]
        metrics = component["metrics"]
        if (
            component["selection_note"] == "depthwise_fallback_p8_cb8"
            and (status in {"downstream_failed", "downstream_timeout"} or status.startswith("downstream_"))
        ):
            reasons.append("backend_sensitive_depthwise_profile")
            continue
        if status == "measured_hls_only" and _target_ii_met(metrics):
            reasons.append("hls_only_downstream_missing")
        elif status == "measured_hls_only":
            reasons.append("target_ii_missed")
        elif status in {"downstream_failed", "downstream_timeout"} or status.startswith("downstream_"):
            reasons.append("downstream_failed")
        elif status in {"missing", "hls_missing"}:
            reasons.append("missing")
        elif status.startswith("hls_"):
            reasons.append(status)
        else:
            reasons.append(status)
    return sorted(set(reasons))


def _target_ii_met(metrics: dict[str, Any]) -> bool:
    observed = _to_int(metrics.get("II"))
    return observed is not None


def _missing_component(row: dict[str, str], reason: str) -> dict[str, Any]:
    return {
        "role": row["component_role"],
        "original_case_name": row["component_case_name"],
        "selected_case_name": "",
        "source_dataset": "",
        "status": reason,
        "deployable": False,
        "selection_note": reason,
        "op_spec": {},
        "metrics": {},
    }


def _index_status_by_shape(entries: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        spec = entry.get("op_spec")
        if isinstance(spec, dict):
            index[_shape_key(spec)].append(entry)
    return index


def _shape_key(spec: dict[str, Any]) -> tuple[Any, ...]:
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
    )


def _spec_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        *_shape_key(spec),
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


def _component_sort_key(row: dict[str, str]) -> int:
    order = {"direct": 0, "pw_expand": 0, "dw": 1, "pw_project": 2}
    return order.get(row["component_role"], 99)


def _is_deployable_status(entry: dict[str, Any]) -> bool:
    return (
        entry.get("status") == "measured_implementation"
        and str(entry.get("deployable_at_200mhz")).lower() == "true"
        and entry.get("downstream_status") in {"success", "skipped_existing"}
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


def _implementation_metrics(row: dict[str, str]) -> dict[str, Any]:
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


def _sum_float(rows: list[dict[str, Any]], key: str) -> str:
    values = [_to_float(row.get(key)) for row in rows]
    if not values or any(value is None for value in values):
        return ""
    return f"{sum(value for value in values if value is not None):.9g}"


def _sum_int(rows: list[dict[str, Any]], key: str) -> str:
    values = [_to_int(row.get(key)) for row in rows]
    if not values or any(value is None for value in values):
        return ""
    return str(sum(value for value in values if value is not None))


def _max_int(rows: list[dict[str, Any]], key: str) -> str:
    values = [_to_int(row.get(key)) for row in rows]
    if not values or any(value is None for value in values):
        return ""
    return str(max(value for value in values if value is not None))


def _reason_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        raw = row.get("non_deployable_reason") or ""
        if not raw:
            counter["deployable"] += 1
            continue
        for reason in str(raw).split("|"):
            if reason:
                counter[reason] += 1
    return counter


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
