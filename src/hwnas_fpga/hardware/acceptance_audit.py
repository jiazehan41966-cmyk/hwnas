"""Acceptance-audit helpers for the HW-NAS FPGA route-clean gap.

The audit is deliberately evidence-only: it reads existing JSON/YAML artifacts,
classifies each requirement, and never launches Vivado or board measurement.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from hwnas_fpga.search_space import ArchitectureSpec


PASS = "PASS"
WEAK = "WEAK"
MISSING = "MISSING"
FAIL = "FAIL"


@dataclass(frozen=True)
class GapAcceptanceAuditInputs:
    physical_config_path: str = ""
    legacy_candidate_source_path: str = ""
    strict_lut_stats_path: str = ""
    pareto_selection_path: str = ""
    pareto_proof_selection_path: str = ""
    pareto_ranked_csv_path: str = ""
    fresh_search_pareto_selection_path: str = ""
    fresh_search_ranked_csv_path: str = ""
    fresh_search_summary_path: str = ""
    fresh_search_status_path: str = ""
    fresh_fetch_error_path: str = ""
    fresh_route_gate_summary_path: str = ""
    route_gate_summary_path: str = ""
    final_candidate_manifest_path: str = ""
    final_validation_report_path: str = ""
    measurement_json_path: str = ""
    expected_serial_port: str = "COM5"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def read_yaml(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _existing_path(path: str | Path) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve())


def _artifact(path: str | Path, *, role: str = "") -> dict[str, Any]:
    return {
        "path": _existing_path(path),
        "exists": bool(path) and Path(path).exists(),
        "role": role,
    }


def _criterion(
    criterion_id: str,
    status: str,
    requirement: str,
    *,
    evidence: Optional[Iterable[Mapping[str, Any]]] = None,
    details: Optional[Mapping[str, Any]] = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "status": status,
        "requirement": requirement,
        "message": message,
        "evidence": [dict(item) for item in (evidence or [])],
        "details": dict(details or {}),
    }


def _status_counts(criteria: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {PASS: 0, WEAK: 0, MISSING: 0, FAIL: 0}
    for item in criteria:
        status = str(item.get("status") or "")
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def _first_selected_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected_topk")
    if not isinstance(selected, list):
        return {}
    for item in selected:
        if isinstance(item, Mapping):
            return dict(item)
    return {}


def _load_candidate_encoding(source_payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    candidate = _first_selected_candidate(source_payload)
    if not candidate:
        candidate = dict(source_payload.get("candidate") or {}) if isinstance(source_payload.get("candidate"), Mapping) else {}
    if not candidate and "encoding" in source_payload:
        candidate = dict(source_payload)
    encoding = candidate.get("encoding")
    if not isinstance(encoding, Mapping):
        return "", {}
    return str(candidate.get("arch_id") or ""), dict(encoding)


def _early_expand_violations(
    *,
    config_payload: Mapping[str, Any],
    encoding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from hwnas_fpga.runtime import build_constraints, build_search_space

    constraints = build_constraints(dict(config_payload))
    dataset_cfg = config_payload.get("dataset") if isinstance(config_payload.get("dataset"), Mapping) else {}
    search_space = build_search_space(
        dict(config_payload),
        image_size=int(dataset_cfg.get("image_size") or 224),
        input_channels=int(dataset_cfg.get("input_channels") or 1),
        num_classes=(
            None if dataset_cfg.get("num_classes") is None else int(dataset_cfg.get("num_classes"))
        ),
        constraints=constraints,
    )
    architecture = ArchitectureSpec.from_dict(encoding)
    physical = constraints.physical if isinstance(constraints.physical, dict) else {}
    rules: list[tuple[int, int]] = []
    for raw_rule in physical.get("early_expand_limits", ()) or ():
        if not isinstance(raw_rule, Mapping):
            continue
        min_resolution = raw_rule.get("min_input_resolution", raw_rule.get("min_resolution"))
        max_expand = raw_rule.get("max_expand_ratio")
        if min_resolution is None or max_expand is None:
            continue
        rules.append((int(min_resolution), int(max_expand)))
    rules.sort(reverse=True)

    violations: list[dict[str, Any]] = []
    for block in search_space.resolve_blocks(architecture):
        if block.op not in {"mbconv", "fused_mbconv"}:
            continue
        for min_resolution, max_expand in rules:
            if int(block.input_resolution) < min_resolution:
                continue
            if int(block.expand_ratio) > max_expand:
                violations.append(
                    {
                        "stage_index": block.stage_index,
                        "block_index": block.block_index,
                        "op": block.op,
                        "input_resolution": block.input_resolution,
                        "expand_ratio": block.expand_ratio,
                        "max_expand_ratio": max_expand,
                        "min_input_resolution": min_resolution,
                    }
                )
            break
    return violations


def _audit_phase0_legacy_infeasible(
    inputs: GapAcceptanceAuditInputs,
    config_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    requirement = "Old RL best with 112x112 expand=6 is marked infeasible by physical hard constraints."
    if not config_payload or not candidate_payload:
        return _criterion(
            "phase0_legacy_expand6_infeasible",
            MISSING,
            requirement,
            evidence=[
                _artifact(inputs.physical_config_path, role="physical Phase0 config"),
                _artifact(inputs.legacy_candidate_source_path, role="legacy RL candidate source"),
            ],
            message="Missing physical config or legacy candidate encoding.",
        )

    arch_id, encoding = _load_candidate_encoding(candidate_payload)
    if not encoding:
        return _criterion(
            "phase0_legacy_expand6_infeasible",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.legacy_candidate_source_path, role="legacy RL candidate source")],
            message="Legacy candidate source does not contain an encoding.",
        )

    violations = _early_expand_violations(config_payload=config_payload, encoding=encoding)
    has_target_violation = any(
        int(item.get("input_resolution") or 0) >= 112
        and int(item.get("expand_ratio") or 0) >= 6
        for item in violations
    )
    return _criterion(
        "phase0_legacy_expand6_infeasible",
        PASS if has_target_violation else FAIL,
        requirement,
        evidence=[
            _artifact(inputs.physical_config_path, role="physical Phase0 config"),
            _artifact(inputs.legacy_candidate_source_path, role="legacy RL candidate source"),
        ],
        details={"arch_id": arch_id, "early_expand_violations": violations},
        message=(
            "Physical early-expand rule rejects the legacy high-resolution expand candidate."
            if has_target_violation
            else "No 112x112 expand=6 physical hard-constraint violation was found."
        ),
    )


def _audit_physical_proxy_not_reward(
    inputs: GapAcceptanceAuditInputs,
    config_payload: Mapping[str, Any],
) -> dict[str, Any]:
    from hwnas_fpga.runtime import build_constraints

    requirement = "Physical proxy is outside the main scalar reward and represented by hard constraints/Pareto flags."
    if not config_payload:
        return _criterion(
            "physical_proxy_out_of_main_reward",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.physical_config_path, role="physical Phase0 config")],
            message="Missing physical config.",
        )

    search_cfg = config_payload.get("search") if isinstance(config_payload.get("search"), Mapping) else {}
    weights = search_cfg.get("objective_weights") if isinstance(search_cfg.get("objective_weights"), Mapping) else {}
    physical_weights = {
        key: float(weights.get(key) or 0.0)
        for key in (
            "physical_risk",
            "early_expand_pressure",
            "interconnect_pressure",
            "fanout_pressure",
            "memory_pressure",
        )
    }
    constraints = build_constraints(dict(config_payload))
    physical = constraints.physical if isinstance(constraints.physical, dict) else {}
    hard_constraints = bool(physical.get("enabled", True)) and (
        bool(physical.get("early_expand_limits"))
        or any(key.startswith("max_") for key in physical)
    )
    pareto_flags = any(
        bool(physical.get(key))
        for key in (
            "pareto_physical_risk",
            "pareto_early_expand_pressure",
            "pareto_interconnect_pressure",
        )
    )
    reward_clean = all(value == 0.0 for value in physical_weights.values())
    status = PASS if reward_clean and hard_constraints and pareto_flags else FAIL
    return _criterion(
        "physical_proxy_out_of_main_reward",
        status,
        requirement,
        evidence=[_artifact(inputs.physical_config_path, role="physical Phase0 config")],
        details={
            "physical_objective_weights": physical_weights,
            "has_physical_hard_constraints": hard_constraints,
            "has_physical_pareto_flags": pareto_flags,
        },
        message=(
            "Physical proxy weights are zero in objective_weights and physical hard/Pareto controls are present."
            if status == PASS
            else "Physical proxy is still weighted in objective_weights or missing hard/Pareto controls."
        ),
    )


def _audit_strict_lut(inputs: GapAcceptanceAuditInputs, payload: Mapping[str, Any]) -> dict[str, Any]:
    requirement = "Strict LUT remains true_misses=0 and deferred_hits=0."
    if not payload:
        return _criterion(
            "strict_lut_no_misses_or_deferred_hits",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.strict_lut_stats_path, role="strict LUT stats")],
            message="Missing strict LUT stats JSON.",
        )
    true_misses = int(payload.get("true_misses") or 0)
    deferred_hits = int(payload.get("deferred_hits") or 0)
    strict_formal_lut = bool(payload.get("strict_formal_lut"))
    passed = true_misses == 0 and deferred_hits == 0 and strict_formal_lut
    return _criterion(
        "strict_lut_no_misses_or_deferred_hits",
        PASS if passed else FAIL,
        requirement,
        evidence=[_artifact(inputs.strict_lut_stats_path, role="strict LUT stats")],
        details={
            "true_misses": true_misses,
            "deferred_hits": deferred_hits,
            "strict_formal_lut": strict_formal_lut,
            "hits": payload.get("hits"),
            "total": payload.get("total"),
        },
    )


def _csv_header(path: str | Path) -> list[str]:
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _pareto_has_physical(payload: Mapping[str, Any]) -> bool:
    has_objective = _pareto_has_physical_objectives(payload)
    ranked = payload.get("ranked_candidates")
    has_ranked_values = False
    if isinstance(ranked, list):
        has_ranked_values = any(
            isinstance(item, Mapping)
            and isinstance(item.get("objective_values"), Mapping)
            and "physical_risk" in item["objective_values"]
            for item in ranked
        )
    return has_objective and has_ranked_values


def _pareto_has_physical_objectives(payload: Mapping[str, Any]) -> bool:
    objectives = payload.get("objectives")
    physical_objectives = payload.get("physical_objectives")
    metric_columns = payload.get("metric_columns")
    has_objective = isinstance(objectives, list) and "physical_risk" in objectives
    has_physical_objective = isinstance(physical_objectives, list) and "physical_risk" in physical_objectives
    has_metric = isinstance(metric_columns, list) and "physical_risk" in metric_columns
    return has_objective and has_physical_objective and has_metric


def _is_full_phase0_search_summary(payload: Mapping[str, Any]) -> bool:
    extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
    search_summary = (
        extra.get("search_summary") if isinstance(extra.get("search_summary"), Mapping) else {}
    )
    total_evaluated = int(payload.get("total_evaluated") or search_summary.get("total_evaluated") or 0)
    num_episodes = int(extra.get("num_episodes") or 0)
    return (
        payload.get("status") == "completed"
        and str(extra.get("search_method") or "") == "rl"
        and total_evaluated >= 300
        and num_episodes >= 300
        and int(extra.get("train_epochs") or 0) >= 10
        and int(payload.get("feasible") or search_summary.get("feasible") or 0) > 0
    )


def _search_scope_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
    search_summary = (
        extra.get("search_summary") if isinstance(extra.get("search_summary"), Mapping) else {}
    )
    return {
        "status": payload.get("status"),
        "search_method": extra.get("search_method"),
        "total_evaluated": payload.get("total_evaluated", search_summary.get("total_evaluated")),
        "feasible": payload.get("feasible", search_summary.get("feasible")),
        "infeasible": payload.get("infeasible", search_summary.get("infeasible")),
        "num_candidates": extra.get("num_candidates"),
        "num_episodes": extra.get("num_episodes"),
        "train_epochs": extra.get("train_epochs"),
        "device": extra.get("device"),
    }


def _fresh_search_status_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    search_state = payload.get("search_state") if isinstance(payload.get("search_state"), Mapping) else {}
    return {
        "captured_at_utc": payload.get("captured_at_utc"),
        "summary_exists": payload.get("summary_exists"),
        "pareto_exists": payload.get("pareto_exists"),
        "ranked_csv_exists": payload.get("ranked_csv_exists"),
        "lut_stats_exists": payload.get("lut_stats_exists"),
        "route_plan_exists": payload.get("route_plan_exists"),
        "route_summary_exists": payload.get("route_summary_exists"),
        "search_state": {
            key: search_state.get(key)
            for key in [
                "total_evaluated",
                "feasible",
                "infeasible",
                "best_score",
                "best_candidate_id",
                "completed",
            ]
        },
    }


def _fresh_fetch_error_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    status = payload.get("status") if isinstance(payload.get("status"), Mapping) else {}
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "connection_status": status.get("connection_status"),
        "error_type": status.get("error_type"),
        "error": status.get("error"),
        "latest_reliable_status_exists": status.get("latest_reliable_status_exists"),
    }


def _audit_pareto(
    inputs: GapAcceptanceAuditInputs,
    main_payload: Mapping[str, Any],
    proof_payload: Mapping[str, Any],
    fresh_payload: Mapping[str, Any],
    fresh_summary: Mapping[str, Any],
    fresh_status: Mapping[str, Any],
    fresh_fetch_error: Mapping[str, Any],
    fresh_route_gate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    requirement = "Pareto output contains physical-risk-aware ranking."
    main_has_physical = _pareto_has_physical(main_payload)
    main_has_physical_objectives = _pareto_has_physical_objectives(main_payload)
    proof_has_physical = _pareto_has_physical(proof_payload)
    fresh_has_physical = _pareto_has_physical(fresh_payload)
    header = _csv_header(inputs.pareto_ranked_csv_path)
    fresh_header = _csv_header(inputs.fresh_search_ranked_csv_path)
    csv_has_physical = "physical_risk" in header
    fresh_csv_has_physical = "physical_risk" in fresh_header
    reanalysis = main_payload.get("reanalysis") if isinstance(main_payload.get("reanalysis"), Mapping) else {}
    feasible_count = reanalysis.get("feasible_candidate_count")
    fresh_selected_count = fresh_payload.get("selected_count")
    full_fresh_search = _is_full_phase0_search_summary(fresh_summary)
    fresh_status_details = _fresh_search_status_details(fresh_status)
    fresh_fetch_error_details = _fresh_fetch_error_details(fresh_fetch_error)
    fresh_search_state = (
        fresh_status.get("search_state") if isinstance(fresh_status.get("search_state"), Mapping) else {}
    )
    fresh_running_total = int(fresh_search_state.get("total_evaluated") or 0)
    fresh_route_candidates = (
        fresh_route_gate_summary.get("candidates")
        if isinstance(fresh_route_gate_summary.get("candidates"), list)
        else []
    )
    fresh_quick_statuses = _gate_statuses(fresh_route_gate_summary, "quick_route_gate")
    fresh_full_statuses = _gate_statuses(fresh_route_gate_summary, "full_route_gate")
    if (
        fresh_has_physical
        and fresh_csv_has_physical
        and int(fresh_selected_count or 0) > 0
        and full_fresh_search
    ):
        status = PASS
        message = "Full Phase0 RL300 search produced physical-risk-aware Pareto-ranked candidates."
    elif (
        main_has_physical
        and (csv_has_physical or not inputs.pareto_ranked_csv_path)
        and (not fresh_summary or full_fresh_search)
    ):
        status = PASS
        message = "Main Pareto selection artifact includes physical_risk objectives, metrics, and ranked candidates."
    elif fresh_running_total > 0 and not full_fresh_search:
        status = WEAK
        message = (
            "Full Phase0 RL300 search is running or partially fetched, but final summary/Pareto "
            "and matching route-gate evidence are not complete yet."
        )
    elif fresh_has_physical and fresh_csv_has_physical and int(fresh_selected_count or 0) > 0:
        status = WEAK
        message = (
            "Fresh physical-constrained smoke search produced feasible physical-risk Pareto candidates, "
            "but the required full NKSID RL300 search and matching route gate are still outstanding."
        )
    elif main_has_physical_objectives and feasible_count == 0:
        status = WEAK
        message = (
            "Physical-risk Pareto reanalysis was applied to the existing candidate pool, "
            "but no candidate survived the physical hard constraints; a fresh Phase0 search is required."
        )
    elif proof_has_physical:
        status = WEAK
        message = "Physical-risk Pareto behavior is proven only by a separate proof/minirepro artifact, not the main search artifact."
    elif main_payload or proof_payload:
        status = FAIL
        message = "Pareto artifact exists but does not include physical_risk ranking evidence."
    else:
        status = MISSING
        message = "Missing Pareto selection artifacts."
    return _criterion(
        "pareto_physical_risk_ranking",
        status,
        requirement,
        evidence=[
            _artifact(inputs.pareto_selection_path, role="main Pareto selection"),
            _artifact(inputs.pareto_ranked_csv_path, role="main Pareto ranked CSV"),
            _artifact(inputs.fresh_search_pareto_selection_path, role="fresh physical search Pareto selection"),
            _artifact(inputs.fresh_search_ranked_csv_path, role="fresh physical search ranked CSV"),
            _artifact(inputs.fresh_search_summary_path, role="fresh physical search summary"),
            _artifact(inputs.fresh_search_status_path, role="fresh physical search status"),
            _artifact(inputs.fresh_fetch_error_path, role="fresh physical search fetch error"),
            _artifact(inputs.fresh_route_gate_summary_path, role="fresh physical search route-gate summary"),
            _artifact(inputs.pareto_proof_selection_path, role="physical-risk Pareto proof"),
        ],
        details={
            "main_has_physical_risk": main_has_physical,
            "main_has_physical_objectives": main_has_physical_objectives,
            "proof_has_physical_risk": proof_has_physical,
            "fresh_has_physical_risk": fresh_has_physical,
            "main_ranked_csv_has_physical_risk": csv_has_physical,
            "fresh_ranked_csv_has_physical_risk": fresh_csv_has_physical,
            "fresh_selected_count": fresh_selected_count,
            "fresh_search_is_full_phase0": full_fresh_search,
            "fresh_search_scope": _search_scope_details(fresh_summary),
            "fresh_search_status": fresh_status_details,
            "fresh_fetch_error": fresh_fetch_error_details,
            "fresh_route_gate_candidate_count": len(fresh_route_candidates),
            "fresh_route_gate_quick_statuses": fresh_quick_statuses,
            "fresh_route_gate_full_statuses": fresh_full_statuses,
            "main_objectives": main_payload.get("objectives"),
            "main_physical_objectives": main_payload.get("physical_objectives"),
            "main_reanalysis": reanalysis,
        },
        message=message,
    )


def _gate_statuses(summary: Mapping[str, Any], gate_key: str) -> list[str]:
    statuses: list[str] = []
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        return statuses
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        gate = row.get(gate_key)
        if isinstance(gate, Mapping):
            statuses.append(str(gate.get("gate_status") or ""))
    return statuses


def _gate_has_structured_route_fields(gate: Any) -> bool:
    if not isinstance(gate, Mapping):
        return False
    utilization = gate.get("utilization")
    fanout = gate.get("fanout")
    report_paths = gate.get("report_paths")
    return (
        "gate_status" in gate
        and "route_success" in gate
        and "timing_clean" in gate
        and "wns_ns" in gate
        and "tns_ns" in gate
        and "global_congestion_level" in gate
        and "congestion_acceptable" in gate
        and isinstance(utilization, Mapping)
        and all(key in utilization for key in ("lut", "ff", "bram", "dsp"))
        and isinstance(fanout, Mapping)
        and all(key in fanout for key in ("max_fanout", "high_fanout_net_count", "hints", "report"))
        and isinstance(report_paths, Mapping)
    )


def _audit_vivado_gate(inputs: GapAcceptanceAuditInputs, summary: Mapping[str, Any]) -> dict[str, Any]:
    requirement = "Vivado quick/full route gates emit structured JSON with WNS, congestion, fanout, utilization, and status."
    if not summary:
        return _criterion(
            "vivado_gate_structured_json",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.route_gate_summary_path, role="route gate summary")],
            message="Missing route gate summary JSON.",
        )
    full_statuses = _gate_statuses(summary, "full_route_gate")
    quick_statuses = _gate_statuses(summary, "quick_route_gate")
    candidates = summary.get("candidates") if isinstance(summary.get("candidates"), list) else []
    structured = bool(candidates) and all(
        isinstance(row, Mapping)
        and _gate_has_structured_route_fields(row.get("quick_route_gate"))
        and _gate_has_structured_route_fields(row.get("full_route_gate"))
        for row in candidates
    )
    quick_executed = any(status in {"PASS", "BORDERLINE", "FAIL"} for status in quick_statuses)
    full_executed = any(status in {"PASS", "BORDERLINE", "FAIL"} for status in full_statuses)
    status = PASS if structured and quick_executed and full_executed else WEAK if structured else FAIL
    return _criterion(
        "vivado_gate_structured_json",
        status,
        requirement,
        evidence=[_artifact(inputs.route_gate_summary_path, role="route gate summary")],
        details={
            "candidate_count": summary.get("candidate_count"),
            "quick_statuses": quick_statuses,
            "full_statuses": full_statuses,
            "quick_gate_executed": quick_executed,
            "full_gate_executed": full_executed,
        },
        message=(
            "Structured quick/full route gate JSON exists, but one or both gates are still pending."
            if status == WEAK
            else "Structured quick/full route gate JSON is present."
            if status == PASS
            else "Route gate JSON is missing required WNS, congestion, fanout, utilization, or status fields."
        ),
    )


def _audit_final_route_clean(inputs: GapAcceptanceAuditInputs, manifest: Mapping[str, Any]) -> dict[str, Any]:
    requirement = "Final candidate must be selected only from full-route clean candidates."
    if not manifest:
        return _criterion(
            "final_candidate_full_route_clean",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.final_candidate_manifest_path, role="final candidate manifest")],
            message="Missing final candidate manifest.",
        )
    route_gate = manifest.get("route_gate") if isinstance(manifest.get("route_gate"), Mapping) else {}
    selection_stage = str(manifest.get("selection_stage") or "")
    passed = (
        route_gate.get("gate_status") == "PASS"
        and route_gate.get("route_success") is True
        and route_gate.get("timing_clean") is True
        and float(route_gate.get("wns_ns") or 0.0) >= 0.0
        and route_gate.get("congestion_acceptable") is True
    )
    return _criterion(
        "final_candidate_full_route_clean",
        PASS if passed else FAIL,
        requirement,
        evidence=[_artifact(inputs.final_candidate_manifest_path, role="final candidate manifest")],
        details={
            "selection_stage": selection_stage,
            "route_gate": route_gate,
            "candidate": manifest.get("candidate"),
        },
    )


def _measurement_status_from(payload: Mapping[str, Any]) -> Optional[int]:
    frame = payload.get("frame")
    value = frame.get("status_code") if isinstance(frame, Mapping) else payload.get("status_code")
    if value is None or value == "":
        return None
    return int(float(value))


def _measurement_cycles_from(payload: Mapping[str, Any]) -> Optional[int]:
    frame = payload.get("frame")
    value = frame.get("cycles") if isinstance(frame, Mapping) else payload.get("cycles")
    if value is None or value == "":
        return None
    return int(float(value))


def _audit_com5(inputs: GapAcceptanceAuditInputs, validation: Mapping[str, Any], measurement: Mapping[str, Any]) -> dict[str, Any]:
    requirement = "COM5 measurement JSON exists and has status_code=0 with e2e cycles/latency."
    measurement_from_report = validation.get("real_board_e2e") if isinstance(validation.get("real_board_e2e"), Mapping) else {}
    serial_port = str(measurement.get("serial_port") or measurement_from_report.get("serial_port") or "")
    status_code = _measurement_status_from(measurement) if measurement else measurement_from_report.get("status_code")
    cycles = _measurement_cycles_from(measurement) if measurement else measurement_from_report.get("e2e_cycles")
    latency = measurement.get("latency_ms") if measurement else measurement_from_report.get("e2e_latency_ms")
    passed = (
        bool(measurement or measurement_from_report)
        and str(serial_port).upper() == inputs.expected_serial_port.upper()
        and status_code == 0
        and cycles is not None
        and latency is not None
    )
    return _criterion(
        "com5_measurement_status_zero",
        PASS if passed else MISSING if not measurement and not measurement_from_report else FAIL,
        requirement,
        evidence=[
            _artifact(inputs.measurement_json_path, role="COM5 measurement JSON"),
            _artifact(inputs.final_validation_report_path, role="final validation report"),
        ],
        details={
            "serial_port": serial_port,
            "status_code": status_code,
            "e2e_cycles": cycles,
            "e2e_latency_ms": latency,
        },
    )


def _audit_latency_distinction(inputs: GapAcceptanceAuditInputs, validation: Mapping[str, Any]) -> dict[str, Any]:
    requirement = "Report clearly distinguishes NAS LUT estimate latency from real board end-to-end latency."
    if not validation:
        return _criterion(
            "latency_scope_distinction",
            MISSING,
            requirement,
            evidence=[_artifact(inputs.final_validation_report_path, role="final validation report")],
            message="Missing final validation report.",
        )
    nas = validation.get("nas_lut_estimate") if isinstance(validation.get("nas_lut_estimate"), Mapping) else {}
    real = validation.get("real_board_e2e") if isinstance(validation.get("real_board_e2e"), Mapping) else {}
    latency_scope = str(nas.get("latency_scope") or "")
    real_source = str(real.get("latency_source") or "")
    passed = (
        "NAS strict LUT" in latency_scope
        and "board e2e" in latency_scope
        and "COM5" in real_source
        and nas.get("latency_ms") is not None
        and real.get("e2e_latency_ms") is not None
    )
    return _criterion(
        "latency_scope_distinction",
        PASS if passed else FAIL,
        requirement,
        evidence=[_artifact(inputs.final_validation_report_path, role="final validation report")],
        details={
            "nas_lut_estimate_latency_ms": nas.get("latency_ms"),
            "nas_latency_scope": latency_scope,
            "real_board_e2e_latency_ms": real.get("e2e_latency_ms"),
            "real_latency_source": real_source,
        },
    )


def _manifest_artifact_path(manifest: Mapping[str, Any], key: str) -> str:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), Mapping) else {}
    value = artifacts.get(key)
    return str(value) if value else ""


def build_gap_acceptance_audit(inputs: GapAcceptanceAuditInputs) -> dict[str, Any]:
    config_payload = read_yaml(inputs.physical_config_path)
    legacy_candidate_payload = read_json(inputs.legacy_candidate_source_path)
    strict_lut_payload = read_json(inputs.strict_lut_stats_path)
    pareto_payload = read_json(inputs.pareto_selection_path)
    pareto_proof_payload = read_json(inputs.pareto_proof_selection_path)
    fresh_pareto_payload = read_json(inputs.fresh_search_pareto_selection_path)
    fresh_summary_payload = read_json(inputs.fresh_search_summary_path)
    fresh_status_payload = read_json(inputs.fresh_search_status_path)
    fresh_fetch_error_payload = read_json(inputs.fresh_fetch_error_path)
    fresh_route_gate_summary = read_json(inputs.fresh_route_gate_summary_path)
    route_gate_summary = read_json(inputs.route_gate_summary_path)
    final_manifest = read_json(inputs.final_candidate_manifest_path)
    effective_inputs = inputs
    if final_manifest:
        derived_final_validation = (
            inputs.final_validation_report_path
            or _manifest_artifact_path(final_manifest, "final_validation_report_json")
        )
        derived_measurement = inputs.measurement_json_path or _manifest_artifact_path(
            final_manifest,
            "measurement_json",
        )
        if (
            derived_final_validation != inputs.final_validation_report_path
            or derived_measurement != inputs.measurement_json_path
        ):
            effective_inputs = replace(
                inputs,
                final_validation_report_path=derived_final_validation,
                measurement_json_path=derived_measurement,
            )
    final_validation = read_json(effective_inputs.final_validation_report_path)
    measurement = read_json(effective_inputs.measurement_json_path)

    criteria = [
        _audit_phase0_legacy_infeasible(effective_inputs, config_payload, legacy_candidate_payload),
        _audit_physical_proxy_not_reward(effective_inputs, config_payload),
        _audit_strict_lut(effective_inputs, strict_lut_payload),
        _audit_pareto(
            effective_inputs,
            pareto_payload,
            pareto_proof_payload,
            fresh_pareto_payload,
            fresh_summary_payload,
            fresh_status_payload,
            fresh_fetch_error_payload,
            fresh_route_gate_summary,
        ),
        _audit_vivado_gate(effective_inputs, route_gate_summary),
        _audit_final_route_clean(effective_inputs, final_manifest),
        _audit_com5(effective_inputs, final_validation, measurement),
        _audit_latency_distinction(effective_inputs, final_validation),
    ]
    counts = _status_counts(criteria)
    remaining_gaps = [
        {
            "id": item["id"],
            "status": item["status"],
            "message": item.get("message") or item["requirement"],
        }
        for item in criteria
        if item["status"] != PASS
    ]
    return {
        "generated_at": timestamp(),
        "status": "ready_for_completion" if not remaining_gaps else "in_progress",
        "status_counts": counts,
        "expected_serial_port": effective_inputs.expected_serial_port,
        "inputs": {key: value for key, value in effective_inputs.__dict__.items()},
        "acceptance_criteria": criteria,
        "remaining_gaps": remaining_gaps,
        "guardrails": [
            "PASS means the supplied artifacts prove the criterion for the checked scope.",
            "WEAK means implementation/proof evidence exists but final pipeline evidence is incomplete.",
            "This audit does not run search, Vivado, or COM measurement.",
        ],
    }
