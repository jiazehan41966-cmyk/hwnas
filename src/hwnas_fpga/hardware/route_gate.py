"""Route-gate planning and result classification helpers.

The helpers here are intentionally pure and hardware-free. Vivado/board scripts
can use them to create cacheable JSON artifacts, while tests can validate the
gate policy without launching implementation tools.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Optional


def encoding_signature(encoding: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(encoding), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fanout_evidence(build_payload: Mapping[str, Any]) -> dict[str, Any]:
    hints = build_payload.get("high_fanout_or_congestion_hints")
    hint_rows = list(hints) if isinstance(hints, list) else []
    max_fanout = _parse_int(
        build_payload.get("max_fanout")
        or build_payload.get("maximum_fanout")
        or build_payload.get("max_net_fanout")
    )
    high_fanout_net_count = _parse_int(build_payload.get("high_fanout_net_count"))
    if high_fanout_net_count is None:
        high_fanout_net_count = sum(
            1
            for item in hint_rows
            if "fanout" in str(item).lower()
        )
    return {
        "max_fanout": max_fanout,
        "high_fanout_net_count": high_fanout_net_count,
        "hints": hint_rows,
        "report": build_payload.get("fanout_report") or "",
    }


def _utilization_payload(build_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lut": build_payload.get("lut"),
        "ff": build_payload.get("ff"),
        "bram": build_payload.get("bram", build_payload.get("bram_tile")),
        "dsp": build_payload.get("dsp"),
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    selection_index = _parse_float(candidate.get("selection_index"))
    rank = _parse_float(candidate.get("rank"))
    return (
        selection_index if selection_index is not None else float("inf"),
        rank if rank is not None else float("inf"),
        str(candidate.get("arch_id") or ""),
    )


def pareto_topk_candidates(
    pareto_payload: Mapping[str, Any],
    *,
    topk: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return selected Pareto candidates, backfilled from ranked rows when needed."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = None if topk is None else max(0, int(topk))

    def add_candidate(item: Any) -> None:
        if limit is not None and len(candidates) >= limit:
            return
        if not isinstance(item, Mapping):
            return
        encoding = item.get("encoding")
        if not isinstance(encoding, Mapping):
            return
        candidate = dict(item)
        candidate["encoding_signature"] = str(
            item.get("encoding_signature") or encoding_signature(encoding)
        )
        key = candidate["encoding_signature"] or str(candidate.get("arch_id") or "")
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    raw_selected = pareto_payload.get("selected_topk")
    if isinstance(raw_selected, list):
        for item in sorted(
            (row for row in raw_selected if isinstance(row, Mapping)),
            key=_candidate_sort_key,
        ):
            add_candidate(item)

    raw_ranked = pareto_payload.get("ranked_candidates")
    if isinstance(raw_ranked, list):
        for item in sorted(
            (row for row in raw_ranked if isinstance(row, Mapping)),
            key=_candidate_sort_key,
        ):
            add_candidate(item)
    return candidates


def dedupe_candidates_by_signature(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate route-gate candidates by architecture encoding signature."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        candidate = dict(item)
        encoding = candidate.get("encoding")
        signature = str(candidate.get("encoding_signature") or "")
        if not signature and isinstance(encoding, Mapping):
            signature = encoding_signature(encoding)
            candidate["encoding_signature"] = signature
        key = signature or str(candidate.get("arch_id") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def classify_route_gate_result(
    build_payload: Mapping[str, Any],
    *,
    pass_wns_ns: float = 0.0,
    borderline_wns_ns: float = -0.25,
    max_congestion_level: int = 5,
    max_actual_dsp: Optional[int] = None,
) -> dict[str, Any]:
    """Classify one Vivado build payload as PASS, BORDERLINE, FAIL, or PENDING."""
    status = str(build_payload.get("status") or "")
    if not build_payload:
        actual_dsp_acceptable = None if max_actual_dsp is None else False
        return {
            "gate_status": "PENDING",
            "route_success": False,
            "timing_clean": False,
            "wns_ns": None,
            "tns_ns": None,
            "global_congestion_level": None,
            "congestion_acceptable": False,
            "actual_dsp": None,
            "max_actual_dsp": max_actual_dsp,
            "actual_dsp_acceptable": actual_dsp_acceptable,
            "utilization": _utilization_payload({}),
            "fanout": _fanout_evidence({}),
            "report_paths": {
                "timing": None,
                "routed_timing": None,
                "utilization": None,
                "fanout": None,
                "stdout_log": None,
                "stderr_log": None,
            },
            "high_fanout_or_congestion_hints": [],
            "failure_reason": "missing_build_result",
        }

    wns = _parse_float(build_payload.get("wns_ns"))
    tns = _parse_float(build_payload.get("tns_ns"))
    congestion = _parse_int(build_payload.get("global_congestion_level"))
    actual_dsp = _parse_int(build_payload.get("dsp"))
    route_success = status in {"success", "skipped_existing"} or bool(
        build_payload.get("route_success")
    )
    timing_clean = bool(build_payload.get("timing_clean")) or (
        wns is not None and wns >= pass_wns_ns
    )
    congestion_acceptable = congestion is None or congestion <= max_congestion_level
    actual_dsp_acceptable = (
        True
        if max_actual_dsp is None
        else actual_dsp is not None and actual_dsp <= max_actual_dsp
    )

    if route_success and timing_clean and congestion_acceptable and actual_dsp_acceptable:
        gate_status = "PASS"
        failure_reason = ""
    elif (
        route_success
        and wns is not None
        and wns >= borderline_wns_ns
        and congestion_acceptable
        and actual_dsp_acceptable
    ):
        gate_status = "BORDERLINE"
        failure_reason = "wns_borderline"
    else:
        gate_status = "FAIL"
        if not route_success:
            failure_reason = str(build_payload.get("failure_summary") or "route_failed")
        elif not congestion_acceptable:
            failure_reason = f"congestion_level_{congestion}_exceeds_{max_congestion_level}"
        elif not actual_dsp_acceptable:
            failure_reason = f"actual_dsp_{actual_dsp}_exceeds_{max_actual_dsp}"
        else:
            failure_reason = f"wns_{wns}_below_{borderline_wns_ns}"

    evidence = build_payload.get("evidence") if isinstance(build_payload.get("evidence"), Mapping) else {}
    return {
        "gate_status": gate_status,
        "route_success": route_success,
        "timing_clean": timing_clean,
        "wns_ns": wns,
        "tns_ns": tns,
        "global_congestion_level": congestion,
        "congestion_acceptable": congestion_acceptable,
        "actual_dsp": actual_dsp,
        "max_actual_dsp": max_actual_dsp,
        "actual_dsp_acceptable": actual_dsp_acceptable,
        "utilization": _utilization_payload(build_payload),
        "fanout": _fanout_evidence(build_payload),
        "report_paths": {
            "timing": (
                evidence.get("final_timing_summary_report")
                or evidence.get("routed_timing_summary_report")
                or evidence.get("timing_report")
                or evidence.get("timing_summary_report")
            ),
            "routed_timing": evidence.get("routed_timing_summary_report"),
            "utilization": evidence.get("placed_utilization_report") or evidence.get("synth_utilization_report"),
            "fanout": evidence.get("fanout_report") or evidence.get("high_fanout_report"),
            "stdout_log": build_payload.get("stdout_log"),
            "stderr_log": build_payload.get("stderr_log"),
        },
        "high_fanout_or_congestion_hints": build_payload.get("high_fanout_or_congestion_hints", []),
        "failure_reason": failure_reason,
    }


def select_route_clean_best(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    max_actual_dsp: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Pick the best candidate only from full-route PASS rows."""
    clean_rows = []
    for row in candidate_rows:
        full_gate = row.get("full_route_gate") or {}
        if full_gate.get("gate_status") != "PASS":
            continue
        if full_gate.get("actual_dsp_acceptable") is False:
            continue
        if max_actual_dsp is not None:
            actual_dsp = _parse_int(
                full_gate.get("actual_dsp")
                or (full_gate.get("utilization") or {}).get("dsp")
            )
            if actual_dsp is None or actual_dsp > max_actual_dsp:
                continue
        clean_rows.append(dict(row))
    if not clean_rows:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, str]:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), Mapping) else {}
        selection_index = _parse_float(candidate.get("selection_index"))
        rank = _parse_float(candidate.get("rank"))
        macro_f1 = _parse_float(metrics.get("macro_f1"))
        top1 = _parse_float(metrics.get("top1"))
        latency_ms = _parse_float(metrics.get("latency_ms"))
        physical_risk = _parse_float(metrics.get("physical_risk"))
        return (
            -(macro_f1 if macro_f1 is not None else float("-inf")),
            -(top1 if top1 is not None else float("-inf")),
            rank if rank is not None else float("inf"),
            latency_ms if latency_ms is not None else float("inf"),
            physical_risk if physical_risk is not None else float("inf"),
            selection_index if selection_index is not None else float("inf"),
            str(candidate.get("arch_id") or ""),
        )

    return sorted(clean_rows, key=key)[0]
