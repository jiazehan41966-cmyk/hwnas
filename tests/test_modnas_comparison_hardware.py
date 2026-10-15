"""Partial evidence cannot become zero-filled costs or a board fit assertion."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_comparison.hardware_screen import certify_partial_latency, screen_current_space


def _known(*costs):
    return [dict(status="EXACT_PRIMITIVE_HLS_EVIDENCE", measurement=dict(latency_us=cost)) for cost in costs]


def test_partial_bound_certifies_only_failure_and_never_fills_unknown_terms():
    failed = certify_partial_latency(_known(90., 40.), budget_us=120.)
    assert failed["known_latency_lower_bound_us"] == 130.
    assert failed["constraint_status"] == "PROVEN_SERIAL_PROXY_LATENCY_INFEASIBLE"
    assert failed["full_latency_us"] is None and failed["unknown_latency_us"] is None
    assert failed["missing_cost_assigned_zero"] is False
    incomplete = certify_partial_latency(_known(50.), budget_us=120.)
    assert incomplete["constraint_status"] == "UNDECIDED_MISSING_COST"
    assert certify_partial_latency([], budget_us=120.)["constraint_status"] == "UNDECIDED_MISSING_COST"
    assert certify_partial_latency(_known(120.), budget_us=120.)["constraint_status"] == "UNDECIDED_MISSING_COST"


@pytest.mark.parametrize("cost", [-1., float("nan"), float("inf"), "50"])
def test_invalid_or_negative_measured_cost_cannot_enter_nonnegative_proof(cost):
    with pytest.raises(ValueError):
        certify_partial_latency(_known(cost), budget_us=120.)


def test_unverified_proxy_row_is_rejected():
    rows = _known(500.)
    rows[0]["status"] = "PREDICTED"
    with pytest.raises(ValueError):
        certify_partial_latency(rows, budget_us=120.)


def test_live_verified_fixed_B1_alone_blocks_all72_under_current_serial_budget():
    report = screen_current_space()
    assert report["candidate_count"] == report["proven_infeasible_candidates"] == 72
    assert len(report["known_evidence_sets"]) == 1
    assert report["complete_new_hardware_labels"] == 0
    assert report["independent_missing_measurement_tasks"] == 41
    assert report["single_primitive_budget_blockers"][0] == "backbone.stem.conv.conv"
    assert report["candidates"][0]["known_latency_lower_bound_decimal_us"] == "2717069.970"
    assert report["candidates"][0]["known_latency_lower_bound_us"] > 120000
    assert all(candidate["full_latency_us"] is None for candidate in report["candidates"])
    assert report["board_feasibility"] == "NOT_EVALUATED_NOT_INFERRED_FROM_SERIAL_PROXY"
    assert report["no_hls_or_board_execution"] and report["no_test_images_accessed"]
    assert report["next_action"] == "HOLD_41_NECK_HLS_TASKS_FOR_INFEASIBLE_CURRENT_SERIAL_BUDGET"
