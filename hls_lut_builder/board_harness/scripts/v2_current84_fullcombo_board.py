#!/usr/bin/env python3
"""Dry-run current84 full-combo board measurement plan artifacts.

This script is intentionally planning-only. It generates runlist/status/README
artifacts for combo-level board measurement without invoking HLS, Vivado, or a
UART board run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "hls_lut_builder" / "results"
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"

CURRENT84_DIR = RESULTS_DIR / "v2_current84"
DEFAULT_COMBO_COVERAGE = CURRENT84_DIR / "current84_combo_coverage.csv"
DEFAULT_PRIMITIVE_MAP = CURRENT84_DIR / "current84_primitive_map.csv"
DEFAULT_CURRENT84_FORMAL_LUT = CURRENT84_DIR / "formal_lut_v2_current84_missing.json"
DEFAULT_DEPLOYABLE_LUT = RESULTS_DIR / "v2_deployable_lut" / "deployable_lut_v2.json"
DEFAULT_CURRENT84_HLS = CURRENT84_DIR / "current84_missing_hls.csv"
DEFAULT_CURRENT84_VIVADO = CURRENT84_DIR / "current84_missing_vivado_downstream.csv"
DEFAULT_OUTPUT_DIR = BOARD_HARNESS_DIR / "results" / "v2_current84_fullcombo_board"

RUNLIST_CSV_NAME = "v2_current84_fullcombo84_runlist.csv"
STATUS_CSV_NAME = "v2_current84_fullcombo84_status.csv"
PILOTS_CSV_NAME = "v2_current84_fullcombo84_pilots.csv"
README_NAME = "README.md"

MBCONV_DECOMPOSITION = "pw_expand+dw+pw_project"
MBCONV_TOPOLOGY = "pw_expand->fifo_repack->dw->fifo_repack->pw_project"

ROLE_ORDER = {
    "direct": 0,
    "pw_expand": 1,
    "dw": 2,
    "pw_project": 3,
}

DIRECT_PILOT_ORDER = [
    "pw_conv_pw_ir16_proj_112_32_16_baseline_pi1_po1_u1_main_5ns",
    "dw_conv_k3_dw_ir16_e1_112_c32_s1_baseline_pi1_po1_u1_main_5ns",
    "skip_skip_56_24_baseline_pi1_po1_u1_main_5ns",
    "global_avg_pool_gap_7_320_baseline_pi1_po1_u1_main_5ns",
    "fc_layer_fc_1_1280_8_baseline_pi1_po1_u1_main_5ns",
]

MBCONV_STAGE2_PILOT_ORDER = [
    "mbconv_e3_k3_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
    "mbconv_e3_k5_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
    "mbconv_e6_k3_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
    "mbconv_e6_k5_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
]

DIRECT_PILOTS = set(DIRECT_PILOT_ORDER)
MBCONV_STAGE2_PILOTS = set(MBCONV_STAGE2_PILOT_ORDER)

RUNLIST_FIELDS = [
    "priority",
    "combo_case_name",
    "combo_family",
    "decomposition",
    "harness_kind",
    "execution_scope",
    "pilot_group",
    "component_count",
    "component_roles",
    "component_case_names",
    "selected_existing_case_names",
    "selected_sources",
    "component_operators",
    "component_op_types",
    "component_shape_names",
    "current_status",
    "current_detail",
    "primitive_coverage_status",
    "deployable_primitive_coverage",
    "prediction_baseline_status",
    "prediction_baseline_latency_ms",
    "prediction_baseline_cycles",
    "prediction_baseline_source",
    "board_combo_latency_ms",
    "board_combo_cycles",
    "notes",
]

STATUS_EXTRA_FIELDS = [
    "status_timestamp",
    "support_status",
    "measurement_status",
    "board_execution_status",
    "stitched_topology",
    "direct_harness_mapping",
    "hls_invoked",
    "vivado_invoked",
    "board_invoked",
    "full_combo_latency_claim",
]

STATUS_FIELDS = RUNLIST_FIELDS + STATUS_EXTRA_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate planning-only current84 full-combo board dry-run artifacts."
    )
    parser.add_argument("mode", choices=("init", "validate"), help="Generate or validate dry-run artifacts.")
    parser.add_argument("--combo-coverage", default=str(DEFAULT_COMBO_COVERAGE))
    parser.add_argument("--primitive-map", default=str(DEFAULT_PRIMITIVE_MAP))
    parser.add_argument("--current84-formal-lut", default=str(DEFAULT_CURRENT84_FORMAL_LUT))
    parser.add_argument("--deployable-lut", default=str(DEFAULT_DEPLOYABLE_LUT))
    parser.add_argument("--current84-hls-csv", default=str(DEFAULT_CURRENT84_HLS))
    parser.add_argument("--current84-vivado-csv", default=str(DEFAULT_CURRENT84_VIVADO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _format_float(value: float) -> str:
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".")


def _valid_metric(latency_ms: Any, cycles: Any) -> tuple[float, int] | None:
    latency = _float_or_none(latency_ms)
    cycle_count = _int_or_none(cycles)
    if latency is None or cycle_count is None:
        return None
    return latency, cycle_count


def add_metric(
    metrics: dict[str, dict[str, Any]],
    case_name: str,
    latency_ms: Any,
    cycles: Any,
    source: str,
) -> None:
    if not case_name:
        return
    parsed = _valid_metric(latency_ms, cycles)
    if parsed is None:
        return
    latency, cycle_count = parsed
    # Prefer current84-specific HLS rows over generic deployable LUT rows.
    if case_name not in metrics or source.startswith("v2_current84_missing_hls"):
        metrics[case_name] = {
            "latency_ms": latency,
            "cycles": cycle_count,
            "source": source,
        }


def load_metric_index(
    *,
    current84_hls_csv: Path,
    current84_formal_lut: Path,
    deployable_lut: Path,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}

    deployable = load_json(deployable_lut)
    for entry in deployable.get("entries", []):
        source_case = str(entry.get("source_case_name") or entry.get("case_name") or "")
        source_dataset = str(entry.get("source_dataset") or "unknown")
        add_metric(
            metrics,
            source_case,
            entry.get("latency_ms"),
            entry.get("cycles"),
            f"deployable_lut_v2:{source_dataset}",
        )

    current84_formal = load_json(current84_formal_lut)
    for entry in current84_formal.get("entries", []):
        case_name = str(entry.get("source_case_name") or entry.get("case_name") or "")
        add_metric(
            metrics,
            case_name,
            entry.get("latency_ms"),
            entry.get("cycles"),
            "formal_lut_v2_current84_missing",
        )

    if current84_hls_csv.exists():
        for row in read_csv(current84_hls_csv):
            add_metric(
                metrics,
                row.get("case_name", ""),
                row.get("latency_ms", ""),
                row.get("latency_cycles", ""),
                "v2_current84_missing_hls",
            )

    return metrics


def group_primitives(primitive_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in primitive_rows:
        grouped[row["current_case_name"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: ROLE_ORDER.get(row.get("component_role", ""), 99))
    return grouped


def pilot_group_for(combo_case_name: str) -> str:
    if combo_case_name in DIRECT_PILOTS:
        return "direct_pilot"
    if combo_case_name in MBCONV_STAGE2_PILOTS:
        return "mbconv_stage2_pilot"
    return ""


def harness_kind_for(decomposition: str) -> str:
    if decomposition == "direct":
        return "single_op_harness"
    if decomposition == MBCONV_DECOMPOSITION:
        return "stitched_mbconv_harness"
    return "unsupported_planning_shape"


def prediction_for(
    components: list[dict[str, str]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, str]:
    latency_sum = 0.0
    cycle_sum = 0
    sources: list[str] = []
    missing: list[str] = []

    for component in components:
        selected_case = component.get("selected_existing_case_name", "")
        component_case = component.get("component_case_name", "")
        metric = metrics.get(selected_case) or metrics.get(component_case)
        if metric is None:
            missing.append(selected_case or component_case)
            continue
        latency_sum += float(metric["latency_ms"])
        cycle_sum += int(metric["cycles"])
        sources.append(str(metric["source"]))

    if missing:
        found_sources = sorted(set(sources))
        missing_sources = [f"missing:{case_name}" for case_name in missing]
        return {
            "prediction_baseline_status": "prediction_incomplete",
            "prediction_baseline_latency_ms": "",
            "prediction_baseline_cycles": "",
            "prediction_baseline_source": "|".join(found_sources + missing_sources),
        }

    return {
        "prediction_baseline_status": "prediction_complete",
        "prediction_baseline_latency_ms": _format_float(latency_sum),
        "prediction_baseline_cycles": str(cycle_sum),
        "prediction_baseline_source": "|".join(sorted(set(sources))),
    }


def build_runlist_rows(
    combo_rows: list[dict[str, str]],
    primitives_by_combo: dict[str, list[dict[str, str]]],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for priority, combo in enumerate(combo_rows, start=1):
        combo_case_name = combo["current_case_name"]
        components = primitives_by_combo.get(combo_case_name, [])
        prediction = prediction_for(components, metrics)
        component_roles = "|".join(row.get("component_role", "") for row in components)
        selected_existing = "|".join(row.get("selected_existing_case_name", "") for row in components)
        selected_sources = "|".join(row.get("selected_source", "") for row in components)
        component_operators = "|".join(row.get("component_operator", "") for row in components)
        component_op_types = "|".join(row.get("component_op_type", "") for row in components)
        component_shape_names = "|".join(row.get("shape_name", "") for row in components)
        component_case_names = "|".join(row.get("component_case_name", "") for row in components)
        decomposition = combo["decomposition"]
        notes = (
            "direct combo maps to existing single-op board harness dry-run row"
            if decomposition == "direct"
            else "MBConv combo requires stitched harness with explicit FIFO/repack boundaries"
        )
        row: dict[str, Any] = {
            "priority": priority,
            "combo_case_name": combo_case_name,
            "combo_family": combo["old_op_family"],
            "decomposition": decomposition,
            "harness_kind": harness_kind_for(decomposition),
            "execution_scope": "dry_run_plan_only",
            "pilot_group": pilot_group_for(combo_case_name),
            "component_count": combo["component_count"],
            "component_roles": component_roles,
            "component_case_names": component_case_names,
            "selected_existing_case_names": selected_existing,
            "selected_sources": selected_sources,
            "component_operators": component_operators,
            "component_op_types": component_op_types,
            "component_shape_names": component_shape_names,
            "current_status": combo["current_status"],
            "current_detail": combo["current_detail"],
            "primitive_coverage_status": combo["has_full_v2_primitive_coverage"],
            "deployable_primitive_coverage": combo["has_full_deployable_primitive_coverage"],
            "board_combo_latency_ms": "",
            "board_combo_cycles": "",
            "notes": notes,
        }
        row.update(prediction)
        rows.append(row)
    return rows


def build_status_rows(runlist_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for run_row in runlist_rows:
        row = dict(run_row)
        is_direct = row["decomposition"] == "direct"
        row.update(
            {
                "status_timestamp": timestamp,
                "support_status": "planned_single_op_harness" if is_direct else "planned_stitched_harness",
                "measurement_status": "not_run",
                "board_execution_status": "not_started",
                "stitched_topology": "" if is_direct else MBCONV_TOPOLOGY,
                "direct_harness_mapping": "single_op_harness" if is_direct else "",
                "hls_invoked": "false",
                "vivado_invoked": "false",
                "board_invoked": "false",
                "full_combo_latency_claim": "none",
            }
        )
        rows.append(row)
    return rows


def build_pilot_rows(runlist_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_name = {row["combo_case_name"]: row for row in runlist_rows}
    ordered_names = DIRECT_PILOT_ORDER + MBCONV_STAGE2_PILOT_ORDER
    return [rows_by_name[name] for name in ordered_names if name in rows_by_name]


def format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def write_readme(
    *,
    output_dir: Path,
    runlist_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    combo_coverage: Path,
    primitive_map: Path,
    current84_formal_lut: Path,
    deployable_lut: Path,
    current84_hls_csv: Path,
    current84_vivado_csv: Path,
) -> None:
    decomposition_counts = Counter(row["decomposition"] for row in runlist_rows)
    pilot_counts = Counter(row["pilot_group"] for row in runlist_rows if row["pilot_group"])
    prediction_counts = Counter(row["prediction_baseline_status"] for row in runlist_rows)
    rows_by_name = {row["combo_case_name"]: row for row in runlist_rows}
    direct_pilots = [name for name in DIRECT_PILOT_ORDER if name in rows_by_name]
    mbconv_pilots = [name for name in MBCONV_STAGE2_PILOT_ORDER if name in rows_by_name]

    lines = [
        "# current84 Full-Combo Board Measurement Dry Run",
        "",
        "This directory contains planning-only artifacts for current84 combo-level board measurement.",
        "No HLS run, Vivado build, bitstream generation, UART transaction, or board execution is performed by this dry run.",
        "",
        "Primitive coverage is complete for current84, but full combo latency has not been measured on the board.",
        "Primitive sum latency is a prediction baseline only; it is not real full-combo board latency.",
        "",
        "## Inputs",
        "",
        f"- `{rel(combo_coverage)}`",
        f"- `{rel(primitive_map)}`",
        f"- `{rel(current84_formal_lut)}`",
        f"- `{rel(deployable_lut)}`",
        f"- `{rel(current84_hls_csv)}`",
        f"- `{rel(current84_vivado_csv)}`",
        "",
        "## Outputs",
        "",
        f"- `{RUNLIST_CSV_NAME}`: one dry-run row per current84 combo.",
        f"- `{STATUS_CSV_NAME}`: planning status for each combo, with all measurements marked `not_run`.",
        f"- `{PILOTS_CSV_NAME}`: selected direct and MBConv pilot rows.",
        "",
        "## Counts",
        "",
        f"- runlist rows: {len(runlist_rows)}",
        f"- status rows: {len(status_rows)}",
        f"- direct combos: {decomposition_counts.get('direct', 0)}",
        f"- MBConv stitched combos: {decomposition_counts.get(MBCONV_DECOMPOSITION, 0)}",
        f"- direct pilots: {pilot_counts.get('direct_pilot', 0)}",
        f"- MBConv stage2 pilots: {pilot_counts.get('mbconv_stage2_pilot', 0)}",
        f"- prediction baseline status: {format_counter(prediction_counts)}",
        "",
        "## Harness Policy",
        "",
        "- Direct combos can map to the existing single-op harness flow.",
        f"- MBConv combos are planned only as `{MBCONV_TOPOLOGY}`.",
        "- Fused MBConv board measurement is out of scope until a fused or stitched harness is implemented and actually measured.",
        "- `board_combo_latency_ms` and `board_combo_cycles` are intentionally blank in both CSV outputs.",
        "",
        "## Pilot Rows",
        "",
        "Direct pilot rows:",
    ]
    lines.extend(f"- `{case_name}`" for case_name in direct_pilots)
    lines.extend(["", "MBConv stage2 pilot rows:"])
    lines.extend(f"- `{case_name}`" for case_name in mbconv_pilots)
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python .\\hls_lut_builder\\board_harness\\scripts\\v2_current84_fullcombo_board.py init",
            "python .\\hls_lut_builder\\board_harness\\scripts\\v2_current84_fullcombo_board.py validate",
            "```",
            "",
        ]
    )
    (output_dir / README_NAME).write_text("\n".join(lines), encoding="utf-8")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    combo_coverage = Path(args.combo_coverage).expanduser().resolve()
    primitive_map = Path(args.primitive_map).expanduser().resolve()
    current84_formal_lut = Path(args.current84_formal_lut).expanduser().resolve()
    deployable_lut = Path(args.deployable_lut).expanduser().resolve()
    current84_hls_csv = Path(args.current84_hls_csv).expanduser().resolve()
    current84_vivado_csv = Path(args.current84_vivado_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    combo_rows = read_csv(combo_coverage)
    primitive_rows = read_csv(primitive_map)
    primitives_by_combo = group_primitives(primitive_rows)
    metrics = load_metric_index(
        current84_hls_csv=current84_hls_csv,
        current84_formal_lut=current84_formal_lut,
        deployable_lut=deployable_lut,
    )
    runlist_rows = build_runlist_rows(combo_rows, primitives_by_combo, metrics)
    status_rows = build_status_rows(runlist_rows)
    pilot_rows = build_pilot_rows(runlist_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / RUNLIST_CSV_NAME, runlist_rows, RUNLIST_FIELDS)
    write_csv(output_dir / STATUS_CSV_NAME, status_rows, STATUS_FIELDS)
    write_csv(output_dir / PILOTS_CSV_NAME, pilot_rows, RUNLIST_FIELDS)
    write_readme(
        output_dir=output_dir,
        runlist_rows=runlist_rows,
        status_rows=status_rows,
        combo_coverage=combo_coverage,
        primitive_map=primitive_map,
        current84_formal_lut=current84_formal_lut,
        deployable_lut=deployable_lut,
        current84_hls_csv=current84_hls_csv,
        current84_vivado_csv=current84_vivado_csv,
    )
    return {
        "output_dir": str(output_dir),
        "runlist_rows": len(runlist_rows),
        "status_rows": len(status_rows),
        "pilot_rows": len(pilot_rows),
        "direct_rows": sum(1 for row in runlist_rows if row["decomposition"] == "direct"),
        "mbconv_rows": sum(1 for row in runlist_rows if row["decomposition"] == MBCONV_DECOMPOSITION),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    runlist_path = output_dir / RUNLIST_CSV_NAME
    status_path = output_dir / STATUS_CSV_NAME
    readme_path = output_dir / README_NAME
    runlist_rows = read_csv(runlist_path)
    status_rows = read_csv(status_path)
    readme = readme_path.read_text(encoding="utf-8")

    failures: list[str] = []
    if len(runlist_rows) != 84:
        failures.append(f"runlist row count is {len(runlist_rows)}, expected 84")
    if len(status_rows) != 84:
        failures.append(f"status row count is {len(status_rows)}, expected 84")
    if sum(1 for row in runlist_rows if row["decomposition"] == "direct") != 60:
        failures.append("direct runlist count is not 60")
    if sum(1 for row in runlist_rows if row["decomposition"] == MBCONV_DECOMPOSITION) != 24:
        failures.append("MBConv runlist count is not 24")
    if sum(1 for row in runlist_rows if row["pilot_group"] == "direct_pilot") != 5:
        failures.append("direct pilot count is not 5")
    if sum(1 for row in runlist_rows if row["pilot_group"] == "mbconv_stage2_pilot") != 4:
        failures.append("MBConv pilot count is not 4")

    mbconv_roles = {
        row["component_roles"]
        for row in runlist_rows
        if row["decomposition"] == MBCONV_DECOMPOSITION
    }
    if mbconv_roles != {"pw_expand|dw|pw_project"}:
        failures.append(f"unexpected MBConv component roles: {sorted(mbconv_roles)}")

    for row in runlist_rows + status_rows:
        if row.get("board_combo_latency_ms") or row.get("board_combo_cycles"):
            failures.append(f"combo board latency fields are not blank for {row.get('combo_case_name')}")
            break

    forbidden = ("board_measured", "fused_measured")
    for row in status_rows:
        row_text = " ".join(str(value) for value in row.values())
        if any(token in row_text for token in forbidden):
            failures.append(f"forbidden measured wording in {row.get('combo_case_name')}")
            break

    required_readme = "Primitive sum latency is a prediction baseline only; it is not real full-combo board latency."
    if required_readme not in readme:
        failures.append("README is missing primitive-sum latency warning")

    if failures:
        raise SystemExit("\n".join(failures))

    return {
        "runlist_rows": len(runlist_rows),
        "status_rows": len(status_rows),
        "direct_rows": 60,
        "mbconv_rows": 24,
        "direct_pilots": 5,
        "mbconv_stage2_pilots": 4,
        "validation": "passed",
    }


def main() -> int:
    args = parse_args()
    result = generate(args) if args.mode == "init" else validate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
