"""Read-only evidence audit and nonnegative partial latency bound for new72.

The bound certifies failure of the declared SERIAL operator-CSYNTH proxy budget.
It is neither a full network latency estimate nor a physical board lower bound.
Missing primitives remain unknown; none is assigned zero or a proxy substitute.
No HLS, training, board work, or old architecture enumeration is launched.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_elastic.hardware import BLOCKED, build_hardware_gap_report

BUDGET_SOURCE = ROOT / "configs/search/sonar_joint_v2.yaml"


def certify_partial_latency(known: Sequence[Mapping[str, Any]], *, budget_us: float) -> dict[str, Any]:
    if type(budget_us) not in (int, float) or not math.isfinite(budget_us) or budget_us <= 0:
        raise ValueError("Latency budget must be positive finite microseconds")
    values = []
    for item in known:
        value = item["measurement"]["latency_us"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Known measured latency must be finite and nonnegative")
        if item.get("status") != "EXACT_PRIMITIVE_HLS_EVIDENCE":
            raise ValueError("Lower bound can use only source-verified exact measured primitives")
        values.append(Decimal(str(value)))
    lower = sum(values, Decimal(0))
    failed = lower > Decimal(str(budget_us))
    return dict(known_latency_lower_bound_us=float(lower),
                known_latency_lower_bound_decimal_us=str(lower), budget_us=budget_us,
                lower_bound_over_budget_factor=float(lower / Decimal(str(budget_us))),
                constraint_status="PROVEN_SERIAL_PROXY_LATENCY_INFEASIBLE" if failed else "UNDECIDED_MISSING_COST",
                full_latency_us=None, unknown_latency_us=None,
                missing_cost_assigned_zero=False)


def screen_current_space(*, num_classes: int = 2) -> dict[str, Any]:
    config = yaml.safe_load(BUDGET_SOURCE.read_text(encoding="utf-8"))
    hardware = config["hardware_budget"]
    if hardware["scope"] != "operator_peak_and_serial_compute" or hardware["latency_unit"] != "us":
        raise ValueError("The prior budget's scope/units do not match the declared serial cost interface")
    budget_us = hardware["limits"]["Latency"]
    # Fresh admission rechecks primitive shape, bitwidth, source hashes, report
    # metrics, platform, numerical witnesses and timing eligibility.
    gap = build_hardware_gap_report(num_classes=num_classes)
    candidates = []
    known_by_graph = {}
    for candidate in gap["candidates"]:
        known = candidate["exact_measurements"]
        bound = certify_partial_latency(known, budget_us=budget_us)
        evidence_key = hashlib.sha256(json.dumps(known, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        known_by_graph.setdefault(evidence_key, known)
        candidates.append(dict(architecture=candidate["architecture"],
                               architecture_sha256=candidate["architecture_sha256"],
                               cost_graph_sha256=candidate["cost_graph_sha256"],
                               known_evidence_set_sha256=evidence_key,
                               known_primitive_nodes=candidate["exact_primitive_nodes"],
                               missing_primitive_nodes=candidate["missing_primitive_nodes"],
                               **bound, hardware_search_status=BLOCKED,
                               full_LUT=None, full_DSP=None, full_BRAM18=None))
    reference = gap["candidates"][0]["exact_measurements"]
    ranked = sorted(reference, key=lambda item: item["measurement"]["latency_us"], reverse=True)
    blockers = [item["node_id"] for item in ranked if item["measurement"]["latency_us"] > budget_us]
    failures = sum(candidate["constraint_status"] == "PROVEN_SERIAL_PROXY_LATENCY_INFEASIBLE" for candidate in candidates)
    return dict(schema_version=1, audit="source_verified_nonnegative_partial_serial_latency_bound",
                candidate_count=len(candidates), proven_infeasible_candidates=failures,
                hardware_search_status=BLOCKED, complete_new_hardware_labels=gap["complete_hardware_labels"],
                independent_missing_measurement_tasks=len(gap["missing_measurement_tasks"]),
                budget=dict(source=str(BUDGET_SOURCE), source_sha256=hashlib.sha256(BUDGET_SOURCE.read_bytes()).hexdigest(),
                            source_field="hardware_budget.limits.Latency", value_us=budget_us,
                            scope=hardware["scope"], inherited_budget_is_not_new_comparison_preregistration=True),
                mapping_policy=gap["mapping_policy"], precision=gap["precision"],
                source_library=gap["source_library"], live_lookup_accounting=gap["lookup_accounting"],
                known_evidence_sets=known_by_graph, candidates=candidates,
                bottlenecks=[dict(node_id=item["node_id"], **item["measurement"]) for item in ranked],
                single_primitive_budget_blockers=blockers,
                lower_bound_scope="DECLARED_SERIAL_SUM_OF_OPERATOR_CSYNTH_AT_5NS_ONLY",
                proof="All missing operator latencies are nonnegative in this metric. The sum of the "
                      "source-verified known subset already exceeds the serial budget; adding missing "
                      "terms cannot make any of these candidates satisfy that same budget.",
                board_feasibility="NOT_EVALUATED_NOT_INFERRED_FROM_SERIAL_PROXY",
                new_parallel_mapping_feasibility="UNKNOWN_REQUIRES_NEW_SOURCE_BOUND_MEASUREMENTS",
                next_action="HOLD_41_NECK_HLS_TASKS_FOR_INFEASIBLE_CURRENT_SERIAL_BUDGET" if failures == len(candidates)
                            else "COMPLETE_MISSING_COST_EVIDENCE",
                alternative_gate="Freeze a justified implementation/latency metric and budget; first test "
                                 "source-bound faster mappings for the fixed-backbone bottlenecks. Merely "
                                 "changing the budget to fit these values is not demonstrated hardware efficiency.",
                no_hls_or_board_execution=True, no_test_images_accessed=True, no_training=True,
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path to preserve the earlier hardware audit")
    report = screen_current_space(num_classes=args.num_classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(candidate_count=report["candidate_count"],
                          proven_infeasible_candidates=report["proven_infeasible_candidates"],
                          budget_us=report["budget"]["value_us"],
                          lower_bound_us=report["candidates"][0]["known_latency_lower_bound_us"],
                          next_action=report["next_action"]), indent=2))


if __name__ == "__main__":
    main()
