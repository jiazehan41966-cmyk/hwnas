#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent

DEFAULT_RESULT_DIR = BUILDER_ROOT / "results" / "v2_failed44_fixed_vector"
DEFAULT_HLS_CSV = DEFAULT_RESULT_DIR / "hls.csv"
DEFAULT_MAPPING_CSV = DEFAULT_RESULT_DIR / "failed44_primitive_map.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULT_DIR / "failed44_combo_hls_status.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_RESULT_DIR / "failed44_combo_hls_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v2 failed44 primitive HLS results by old defer case.")
    parser.add_argument("--hls-csv", default=str(DEFAULT_HLS_CSV))
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hls_rows = _read_csv(Path(args.hls_csv))
    mapping_rows = _read_csv(Path(args.mapping_csv))
    by_component = {row["case_name"]: row for row in hls_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mapping_rows:
        grouped[row["old_case_name"]].append(row)

    summary_rows = [
        _summarize_old_case(old_case_name, rows, by_component)
        for old_case_name, rows in sorted(grouped.items())
    ]
    status_counts = Counter(row["combo_hls_status"] for row in summary_rows)
    target_ii_status_counts = Counter(row["target_ii_status"] for row in summary_rows)
    deployable_lut_status_counts = Counter(row["deployable_lut_status"] for row in summary_rows)
    nas_lut_policy_counts = Counter(row["nas_lut_policy"] for row in summary_rows)
    primitive_status_counts = Counter(row.get("synthesis_status", "missing") for row in hls_rows)

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, summary_rows)
    output_json.write_text(
        json.dumps(
            {
                "old_combo_count": len(summary_rows),
                "primitive_case_count": len(by_component),
                "combo_status_counts": dict(status_counts),
                "target_ii_status_counts": dict(target_ii_status_counts),
                "deployable_lut_status_counts": dict(deployable_lut_status_counts),
                "nas_lut_policy_counts": dict(nas_lut_policy_counts),
                "primitive_status_counts": dict(primitive_status_counts),
                "hls_csv": str(Path(args.hls_csv).expanduser().resolve()),
                "mapping_csv": str(Path(args.mapping_csv).expanduser().resolve()),
                "entries": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote failed44 combo HLS status CSV: {output_csv}")
    print(f"Wrote failed44 combo HLS status JSON: {output_json}")
    print(f"Combo status counts: {dict(status_counts)}")
    print(f"Target II status counts: {dict(target_ii_status_counts)}")
    print(f"Deployable LUT status counts: {dict(deployable_lut_status_counts)}")
    print(f"NAS LUT policy counts: {dict(nas_lut_policy_counts)}")
    print(f"Primitive status counts: {dict(primitive_status_counts)}")


def _summarize_old_case(
    old_case_name: str,
    rows: list[dict[str, str]],
    by_component: dict[str, dict[str, str]],
) -> dict[str, Any]:
    unique_components: dict[str, dict[str, str]] = {}
    for row in rows:
        unique_components[row["component_case_name"]] = row

    component_names = sorted(unique_components)
    component_metrics = [by_component.get(name) for name in component_names]
    missing_components = [name for name, metric in zip(component_names, component_metrics) if metric is None]
    failed_components = [
        name
        for name, metric in zip(component_names, component_metrics)
        if metric is not None and metric.get("synthesis_status") != "success"
    ]
    target_ii_missed_components = [
        name
        for name, metric in zip(component_names, component_metrics)
        if metric is not None and _exceeds_target_ii(metric)
    ]

    if missing_components:
        status = "missing_component_hls"
    elif failed_components:
        status = "failed_component_hls"
    else:
        status = "primitive_hls_success"
    deployable_lut_status, nas_lut_policy, non_deployable_reason = _deployable_lut_policy(
        combo_hls_status=status,
        target_ii_missed=bool(target_ii_missed_components),
    )

    component_roles = [
        unique_components[name]["component_role"]
        for name in component_names
    ]
    old_family = rows[0]["old_op_family"] if rows else ""
    is_decomposed = old_family.startswith("mbconv")

    return {
        "old_case_name": old_case_name,
        "old_op_family": old_family,
        "combo_hls_status": status,
        "decomposition": "pw_expand+dw+pw_project" if is_decomposed else "direct",
        "component_count": len(component_names),
        "component_roles": "|".join(component_roles),
        "component_cases": "|".join(component_names),
        "missing_components": "|".join(missing_components),
        "failed_components": "|".join(failed_components),
        "target_ii_status": "target_ii_missed" if target_ii_missed_components else "target_ii_met",
        "target_ii_missed_components": "|".join(target_ii_missed_components),
        "deployable_lut_status": deployable_lut_status,
        "nas_lut_policy": nas_lut_policy,
        "non_deployable_reason": non_deployable_reason,
        "sum_latency_ms": _sum_numeric(component_metrics, "latency_ms"),
        "sum_LUT": _sum_int(component_metrics, "LUT"),
        "sum_FF": _sum_int(component_metrics, "FF"),
        "sum_DSP": _sum_int(component_metrics, "DSP"),
        "sum_BRAM_18K": _sum_int(component_metrics, "BRAM_18K"),
        "max_observed_II": _max_int(component_metrics, "II"),
        "notes": (
            "conservative primitive sum; fused MBConv remains disabled"
            if is_decomposed
            else ""
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sum_numeric(rows: list[dict[str, str] | None], field: str) -> str:
    values = [_to_float(row.get(field)) for row in rows if row is not None]
    if any(value is None for value in values) or not values:
        return ""
    return f"{sum(value for value in values if value is not None):.9g}"


def _sum_int(rows: list[dict[str, str] | None], field: str) -> str:
    values = [_to_int(row.get(field)) for row in rows if row is not None]
    if any(value is None for value in values) or not values:
        return ""
    return str(sum(value for value in values if value is not None))


def _max_int(rows: list[dict[str, str] | None], field: str) -> str:
    values = [_to_int(row.get(field)) for row in rows if row is not None]
    if any(value is None for value in values) or not values:
        return ""
    return str(max(value for value in values if value is not None))


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _exceeds_target_ii(row: dict[str, str]) -> bool:
    observed = _to_int(row.get("II"))
    target = _to_int(row.get("target_ii"))
    return observed is not None and target is not None and observed > target


def _deployable_lut_policy(
    *,
    combo_hls_status: str,
    target_ii_missed: bool,
) -> tuple[str, str, str]:
    if combo_hls_status != "primitive_hls_success":
        return (
            "not_deployable_hls_incomplete",
            "exclude_from_deployable_lut",
            "primitive HLS did not complete for every required component",
        )
    if target_ii_missed:
        return (
            "hls_success_target_ii_missed",
            "exclude_from_deployable_lut",
            "one or more pointwise components observed II greater than target II",
        )
    return (
        "hls_success_downstream_pending",
        "eligible_for_downstream_validation",
        "HLS-only result; post-route metrics are required before NAS deployable LUT use",
    )


if __name__ == "__main__":
    main()
