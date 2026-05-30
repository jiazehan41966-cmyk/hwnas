#!/usr/bin/env python3
"""Plan DSP/congestion recovery for the arch_84 sonar e2e harness.

This script is analysis-only.  It reads existing complete-network build
evidence, implementation variants, and harness manifests, then writes a
separate pressure-reduction plan under the e2e implementation recovery tree.
It never runs COM/UART measurement and never updates canonical latency ledgers.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = ROOT / "hls_lut_builder" / "results"
BOARD_RESULTS = ROOT / "hls_lut_builder" / "board_harness" / "results"
E2E_DIR = BOARD_RESULTS / "sonar_classifier_end_to_end_board"
TARGET_NAME = "sonar_classifier_top8_03_arch_84_e2e"
MEASUREMENT_ROOT = E2E_DIR / "real_measurement" / TARGET_NAME
CANONICAL_PROJECT = MEASUREMENT_ROOT / "harness_project"
VARIANTS_ROOT = MEASUREMENT_ROOT / "implementation_variants"
SWEEP_ROOT = MEASUREMENT_ROOT / "implementation_sweep"
PLAN_ROOT = MEASUREMENT_ROOT / "implementation_recovery" / "dsp_pressure_plan_20260523"

SUMMARY_JSON = E2E_DIR / "sonar_classifier_e2e_summary.json"
BLOCKERS_CSV = E2E_DIR / "sonar_classifier_e2e_blockers.csv"
HARNESS_MANIFEST = CANONICAL_PROJECT / "harness_manifest.json"
HIER_UTIL_REPORT = (
    VARIANTS_ROOT
    / "fifo64_fanout_only"
    / "postmortem_reports"
    / "utilization_hierarchical_after_route_error.rpt"
)
VARIANT_SUMMARY = VARIANTS_ROOT / "implementation_variants_summary.json"

CASE_SEARCH_ROOTS = [
    RESULTS_DIR / "v2_arch84_e2e_extension" / "p",
    RESULTS_DIR / "v2_current84" / "missing_p",
    RESULTS_DIR / "v2_current84" / "identity_repack_14_64_p",
    RESULTS_DIR / "v2_pointwise_v21" / "p",
    RESULTS_DIR / "v2_fixed_vector" / "p",
    RESULTS_DIR / "v2_failed44_fixed_vector" / "p",
    RESULTS_DIR / "v2_k5_low_complexity" / "p",
    RESULTS_DIR / "v2_depthwise_fallback_blockers" / "p",
]

HLS_CSVS = [
    RESULTS_DIR / "v2_pointwise_v21" / "hls.csv",
    RESULTS_DIR / "v2_fixed_vector" / "hls.csv",
    RESULTS_DIR / "v2_failed44_fixed_vector" / "hls.csv",
    RESULTS_DIR / "v2_k5_low_complexity" / "hls.csv",
    RESULTS_DIR / "v2_depthwise_fallback_blockers" / "hls.csv",
    RESULTS_DIR / "v2_current84" / "missing_hls.csv",
]

CSV_FIELDS_RESOURCE = [
    "role",
    "layer_index",
    "layer_role",
    "case_name",
    "input_word_count",
    "output_word_count",
    "input_tdata_width",
    "output_tdata_width",
    "kernel_lut",
    "kernel_ff",
    "kernel_ramb36",
    "kernel_ramb18",
    "kernel_dsp",
    "memory_lut",
    "memory_ff",
    "memory_ramb36",
    "memory_ramb18",
]

CSV_FIELDS_CANDIDATE = [
    "role",
    "case_name",
    "shape_token",
    "candidate_case_name",
    "candidate_dir",
    "candidate_hls_dsp",
    "candidate_hls_latency_cycles",
    "candidate_hls_ii",
    "candidate_hls_status",
    "candidate_relation",
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_int(value: str) -> int:
    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    return int(float(text))


def split_util_line(line: str) -> list[str]:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return parts if len(parts) >= 10 else []


def parse_hierarchical_resources(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    kernels: dict[str, dict[str, Any]] = {}
    memories: dict[str, dict[str, Any]] = {}
    top: dict[str, Any] = {}
    if not path.exists():
        return kernels, memories, top

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = split_util_line(line)
        if not parts:
            continue
        instance = parts[0]
        if instance == "harness_top":
            top = {
                "lut": parse_int(parts[2]),
                "ff": parse_int(parts[6]),
                "ramb36": parse_int(parts[7]),
                "ramb18": parse_int(parts[8]),
                "dsp": parse_int(parts[9]),
            }
            continue
        if not instance.startswith("u_l"):
            continue
        if instance.endswith("_kernel"):
            role = instance[2 : -len("_kernel")]
            kernels[role] = {
                "role": role,
                "instance": instance,
                "module": parts[1],
                "kernel_lut": parse_int(parts[2]),
                "kernel_ff": parse_int(parts[6]),
                "kernel_ramb36": parse_int(parts[7]),
                "kernel_ramb18": parse_int(parts[8]),
                "kernel_dsp": parse_int(parts[9]),
            }
            continue
        if "_mem" in instance:
            role = memory_role(instance[2:])
            row = memories.setdefault(
                role,
                {
                    "role": role,
                    "memory_lut": 0,
                    "memory_ff": 0,
                    "memory_ramb36": 0,
                    "memory_ramb18": 0,
                    "memory_instances": 0,
                },
            )
            row["memory_lut"] += parse_int(parts[2])
            row["memory_ff"] += parse_int(parts[6])
            row["memory_ramb36"] += parse_int(parts[7])
            row["memory_ramb18"] += parse_int(parts[8])
            row["memory_instances"] += 1
    return kernels, memories, top


def memory_role(instance: str) -> str:
    for marker in ("_weights", "_biases"):
        if marker in instance:
            return instance.split(marker, 1)[0]
    return instance.rsplit("_", 1)[0]


def hls_lookup() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for csv_path in HLS_CSVS:
        for row in read_csv(csv_path):
            case_name = row.get("case_name", "")
            if case_name:
                rows[case_name] = row

    arch_summary = read_json(RESULTS_DIR / "v2_arch84_e2e_extension" / "arch84_e2e_extension_hls_synth_summary.json")
    if isinstance(arch_summary, list):
        for entry in arch_summary:
            if not isinstance(entry, dict):
                continue
            case_name = str(entry.get("case_name", ""))
            metrics = (
                ((entry.get("downstream_check") or {}).get("metrics") or {}).get("post_route")
                or ((entry.get("downstream_check") or {}).get("metrics") or {}).get("post_synth")
                or {}
            )
            if case_name and metrics:
                rows[case_name] = {
                    "case_name": case_name,
                    "DSP": str(metrics.get("dsp", "")),
                    "latency_cycles": "",
                    "II": "",
                    "synthesis_status": str(entry.get("status", "")),
                }
    return rows


def shape_token(case_name: str) -> str:
    patterns = [
        r"(stem_\d+_\d+_\d+_s\d+)",
        r"(pw_ir\d+_(?:expand|proj)(?:_e\d+)?_\d+_\d+_\d+)",
        r"(failed44_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_(?:expand|project)_\d+_\d+_\d+)",
        r"(dw_ir\d+_e\d+_\d+_c\d+_s\d+)",
        r"(a84_(?:pwe|dw|pwp|gap|fc))",
        r"(gap_\d+_\d+)",
        r"(fc_1_\d+_\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, case_name)
        if match:
            return match.group(1)
    return case_name


def find_candidate_dirs(token: str) -> list[Path]:
    candidates: list[Path] = []
    if not token:
        return candidates
    for root in CASE_SEARCH_ROOTS:
        if not root.exists():
            continue
        candidates.extend(path for path in root.iterdir() if path.is_dir() and token in path.name)
    return sorted(candidates, key=lambda path: path.name)


def candidate_rows(components: list[dict[str, Any]], kernels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = hls_lookup()
    rows: list[dict[str, Any]] = []
    for component in components:
        role = str(component.get("role", ""))
        case_name = str(component.get("case_name", ""))
        token = shape_token(case_name)
        current_dsp = int(kernels.get(role, {}).get("kernel_dsp") or 0)
        for candidate_dir in find_candidate_dirs(token):
            candidate = candidate_dir.name
            hls = lookup.get(candidate, {})
            dsp_text = hls.get("DSP", "")
            candidate_dsp = parse_int(dsp_text) if str(dsp_text).strip() else 0
            if candidate == case_name:
                relation = "current"
            elif candidate_dsp and current_dsp and candidate_dsp < current_dsp:
                relation = "lower_dsp_existing_candidate"
            else:
                relation = "same_shape_existing_candidate"
            rows.append(
                {
                    "role": role,
                    "case_name": case_name,
                    "shape_token": token,
                    "candidate_case_name": candidate,
                    "candidate_dir": str(candidate_dir),
                    "candidate_hls_dsp": dsp_text,
                    "candidate_hls_latency_cycles": hls.get("latency_cycles", ""),
                    "candidate_hls_ii": hls.get("II", ""),
                    "candidate_hls_status": hls.get("synthesis_status", ""),
                    "candidate_relation": relation,
                }
            )
    return rows


def component_resource_rows(
    components: list[dict[str, Any]],
    kernels: dict[str, dict[str, Any]],
    memories: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in components:
        role = str(component.get("role", ""))
        row = {
            "role": role,
            "layer_index": component.get("layer_index", ""),
            "layer_role": component.get("layer_role", ""),
            "case_name": component.get("case_name", ""),
            "input_word_count": component.get("input_word_count", ""),
            "output_word_count": component.get("output_word_count", ""),
            "input_tdata_width": component.get("input_tdata_width", ""),
            "output_tdata_width": component.get("output_tdata_width", ""),
        }
        row.update(kernels.get(role, {}))
        row.update(memories.get(role, {}))
        rows.append(row)
    return rows


def summarize_sweeps() -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(SWEEP_ROOT.glob("*/implementation_sweep_summary.csv")):
        for row in read_csv(path):
            row["sweep_summary_csv"] = str(path)
            rows.append(row)
    wns_values = [float(row["wns_ns"]) for row in rows if str(row.get("wns_ns", "")).strip()]
    congestion_levels = sorted({row.get("global_congestion_level", "") for row in rows if row.get("global_congestion_level", "")})
    return {
        "attempt_count": len(rows),
        "success_count": sum(1 for row in rows if row.get("status") == "success"),
        "best_wns_ns": max(wns_values) if wns_values else "",
        "global_congestion_levels": congestion_levels,
        "latest_summary_csv": str(max(SWEEP_ROOT.glob("*/implementation_sweep_summary.csv"), key=lambda p: p.stat().st_mtime))
        if list(SWEEP_ROOT.glob("*/implementation_sweep_summary.csv"))
        else "",
    }


def summarize_variants() -> dict[str, Any]:
    payload = read_json(VARIANT_SUMMARY)
    variants = payload.get("variants", []) if isinstance(payload, dict) else []
    wns_values = [float(row["wns_ns"]) for row in variants if str(row.get("wns_ns", "")).strip()]
    return {
        "variant_count": len(variants),
        "success_count": sum(1 for row in variants if row.get("status") == "success"),
        "best_wns_ns": max(wns_values) if wns_values else "",
        "summary_json": str(VARIANT_SUMMARY) if VARIANT_SUMMARY.exists() else "",
    }


def proposed_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_name": "arch84_e2e_lowdsp_resynth_v1",
            "type": "new_implementation_variant",
            "topology": "preserve arch_84 layer topology and tensor shapes",
            "main_change": "resynthesize DSP-heavy components with lower PE/ch_block/SIMD parallelism in a separate namespace",
            "targets": [
                "stem l1 direct kernel",
                "l3/l4/l5 depthwise kernels",
                "l5 pointwise expand/project kernels",
            ],
            "why": "strategy, FIFO, and fanout-only variants leave DSP at 840/840 and route congestion at level 6",
            "required_gate": "new primitive/formal/build evidence for the lowered components before any e2e measurement",
            "ledger_rule": "do not merge into current84/fullcombo/arch84 canonical ledgers unless explicitly promoted as a new variant",
        },
        {
            "variant_name": "arch84_e2e_stage_serial_shared_engine_v1",
            "type": "new_microarchitecture_variant",
            "topology": "preserve arch_84 graph, execute layers through shared compute engines and intermediate activation buffers",
            "main_change": "reduce instantiated DSP engines instead of only delaying layer starts",
            "why": "sequential start control alone would still instantiate all kernels and keep placement pressure unchanged",
            "required_gate": "define intermediate storage, scheduler, golden tensor checks, and full-network COM5 protocol",
            "ledger_rule": "valid e2e latency only from the exact full-network bitstream measurement",
        },
        {
            "variant_name": "arch84_e2e_dsp_aware_floorplan_v2",
            "type": "fallback_implementation_variant",
            "topology": "preserve current full-parallel harness",
            "main_change": "capacity-aware soft placement hints only after computing layer DSP demand",
            "why": "coarse floorplan failed because a pblock requested 256 DSP with only 240 available",
            "required_gate": "pblock regions must exceed per-layer DSP demand and avoid splitting DSP/carry macro collections",
            "ledger_rule": "diagnostic only unless it generates a real bitstream and is measured as full-network e2e",
        },
    ]


def write_readme(plan: dict[str, Any]) -> None:
    readme = f"""# DSP Pressure Plan - {TARGET_NAME}

Generated: {plan['generated_at']}

This directory is planning evidence only.  It does not contain a bitstream and
does not authorize COM5 measurement.

## Current Evidence

- Canonical status: `{plan['canonical_status'].get('status', '')}`
- Bitstream status: `{plan['canonical_status'].get('bitstream_status', '')}`
- Strategy sweep attempts: `{plan['strategy_sweep'].get('attempt_count', 0)}`, successes: `{plan['strategy_sweep'].get('success_count', 0)}`
- Implementation variants: `{plan['implementation_variants'].get('variant_count', 0)}`, successes: `{plan['implementation_variants'].get('success_count', 0)}`
- Top-level DSP from evidence report: `{plan['resource_pressure'].get('top', {}).get('dsp', '')}/840`
- Top-level BRAM tiles from evidence report: `{plan['resource_pressure'].get('top_bram_tile_equiv', '')}`

## Decision

The full-parallel harness remains unroutable because implementation-only
changes keep DSP occupancy at 840/840 and route congestion at level 6.  The
next recovery should be a new isolated e2e implementation variant that lowers
instantiated compute pressure, not another promotion to canonical latency.

## Files

- `dsp_pressure_plan.json`
- `component_resource_table.csv`
- `same_shape_candidate_scan.csv`
"""
    (PLAN_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    manifest = read_json(HARNESS_MANIFEST)
    components = manifest.get("components", []) if isinstance(manifest, dict) else []
    kernels, memories, top = parse_hierarchical_resources(HIER_UTIL_REPORT)
    resource_rows = component_resource_rows(components, kernels, memories)
    candidates = candidate_rows(components, kernels)
    lower_dsp_candidates = [row for row in candidates if row.get("candidate_relation") == "lower_dsp_existing_candidate"]
    top_bram_tile = top.get("ramb36", 0) + top.get("ramb18", 0) / 2 if top else ""

    plan = {
        "generated_at": timestamp(),
        "target_name": TARGET_NAME,
        "scope": "analysis_only_no_com5_no_latency_claim",
        "canonical_status": read_json(SUMMARY_JSON),
        "blockers": read_csv(BLOCKERS_CSV),
        "evidence_inputs": {
            "harness_manifest": str(HARNESS_MANIFEST),
            "hierarchical_utilization_report": str(HIER_UTIL_REPORT),
            "variant_summary": str(VARIANT_SUMMARY),
            "summary_json": str(SUMMARY_JSON),
            "blockers_csv": str(BLOCKERS_CSV),
        },
        "strategy_sweep": summarize_sweeps(),
        "implementation_variants": summarize_variants(),
        "resource_pressure": {
            "top": top,
            "top_bram_tile_equiv": top_bram_tile,
            "kernel_dsp_sum": sum(int(row.get("kernel_dsp") or 0) for row in resource_rows),
            "heavy_kernel_roles": [
                row
                for row in resource_rows
                if int(row.get("kernel_dsp") or 0) >= 64
            ],
        },
        "candidate_scan": {
            "candidate_count": len(candidates),
            "lower_dsp_existing_candidate_count": len(lower_dsp_candidates),
            "note": (
                "Existing same-shape RTL candidates do not provide a clear lower-DSP replacement for the "
                "full-network pressure point. New lowered-parallelism synthesis is required."
            ),
        },
        "bram_output_register_assessment": {
            "simple_dual_port_bram_outputs_are_registered": True,
            "extra_output_register_risk": (
                "Adding another wrapper register would change parameter-memory read latency unless the "
                "kernel schedule and golden tensor behavior are revalidated."
            ),
        },
        "decision": {
            "can_enter_com5_measurement": False,
            "reason": "no complete-network e2e bitstream exists",
            "strategy_only_recovery_exhausted": True,
            "next_required_action": "create an isolated lowered-DSP or shared-engine e2e implementation variant",
        },
        "proposed_variants": proposed_variants(),
        "outputs": {
            "plan_json": str(PLAN_ROOT / "dsp_pressure_plan.json"),
            "resource_csv": str(PLAN_ROOT / "component_resource_table.csv"),
            "candidate_csv": str(PLAN_ROOT / "same_shape_candidate_scan.csv"),
            "readme": str(PLAN_ROOT / "README.md"),
        },
    }

    write_csv(PLAN_ROOT / "component_resource_table.csv", resource_rows, CSV_FIELDS_RESOURCE)
    write_csv(PLAN_ROOT / "same_shape_candidate_scan.csv", candidates, CSV_FIELDS_CANDIDATE)
    write_json(PLAN_ROOT / "dsp_pressure_plan.json", plan)
    write_readme(plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
