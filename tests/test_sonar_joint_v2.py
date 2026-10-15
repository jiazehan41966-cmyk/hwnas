"""Joint promotion decision regressions; all scores/costs below are synthetic fixtures."""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sonar_joint_v2_policy", ROOT / "tools/sonar_joint_v2/policy.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

BUDGET = dict(
    scope="operator_peak_and_serial_compute",
    limits=dict(Latency=100, LUT=100, DSP=100, BRAM=100),
)
PROMOTION = dict(fraction=.15, accuracy_epsilon=.001, resource_fraction=.90)


def cost(aid, q="W8A8", mapping="reference_low", pressure=.2, **updates):
    row = dict(
        architecture_id=aid, profile=q, mapping_policy=mapping,
        status="LOOKED_UP", measurement_valid=True, timing_eligible=True,
        Latency=100*pressure, LUT=100*pressure, DSP=100*pressure, BRAM=100*pressure,
    )
    row.update(updates)
    return row


def evaluation(aid, fp32=.5, profiles=None, budget=40):
    return dict(
        architecture_id=aid, seed=42, budget=budget,
        fp32={"mAP50_95": fp32},
        profiles={q: dict(mAP50_95=score, profile_sha256=f"fixture-{aid}-{q}-{budget}")
                  for q, score in (profiles or {"W8A8": .49}).items()},
        checkpoint_sha256=f"fixture-checkpoint-{aid}-{budget}",
        calibration_sha256="fixture-train-calibration", split="valid", test_evaluated=False,
    )


def points(evaluations, costs):
    admitted = m.filter_costs(costs, BUDGET)["rows"]
    return m.joint_points(evaluations, admitted, BUDGET)


def test_quantized_validation_reverses_float_only_promotion():
    ev = [evaluation("float_winner", fp32=.9, profiles={"W8A8": .4}),
          evaluation("quant_winner", fp32=.8, profiles={"W8A8": .7})]
    joint = points(ev, [cost("float_winner"), cost("quant_winner")])
    decision = m.promote(joint, 1, PROMOTION)
    assert decision["selected"] == ["quant_winner"]
    assert decision["ranked"][0]["quantized_mAP50_95"] == pytest.approx(.7)


def test_accuracy_resource_tradeoff_survives_while_dominated_candidate_drops():
    ev = [evaluation("accurate", profiles={"W8A8": .8}),
          evaluation("efficient", profiles={"W8A8": .7}),
          evaluation("dominated", profiles={"W8A8": .6})]
    costs = [cost("accurate", pressure=.8), cost("efficient", pressure=.2),
             cost("dominated", pressure=.5)]
    decision = m.promote(points(ev, costs), 2, PROMOTION)
    assert set(decision["selected"]) == {"accurate", "efficient"}
    front = {r["architecture_id"]: r["front_rank"] for r in decision["ranked"]}
    assert front["dominated"] > front["accurate"] == front["efficient"]


@pytest.mark.parametrize("update", [
    {"status": "NEEDS_MEASUREMENT"},
    {"measurement_valid": False},
    {"timing_eligible": False},
    {"Latency": float("nan")},
    {"LUT": float("inf")},
    {"DSP": -1},
    {"BRAM": None},
])
def test_unknown_invalid_or_timing_failed_cost_cannot_enter_selection(update):
    report = m.filter_costs([cost("good"), cost("invalid", **update)], BUDGET)
    assert [r["architecture_id"] for r in report["rows"]] == ["good"]
    assert len(report["rejected"]) == 1
    assert report["rejected"][0]["architecture_id"] == "invalid"


def test_budget_filters_exact_profile_and_mapping_without_architecture_substitution():
    rows = [cost("a", mapping="reference_low"),
            cost("a", mapping="reference_high", Latency=101),
            cost("a", q="HEAD_W16A16", LUT=101),
            cost("b", q="HEAD_W8A16", pressure=.4)]
    filtered = m.filter_costs(rows, BUDGET)
    keys = {(r["architecture_id"], r["profile"], r["mapping_policy"])
            for r in filtered["rows"]}
    assert keys == {("a", "W8A8", "reference_low"),
                    ("b", "HEAD_W8A16", "reference_low")}
    ev = [evaluation("a", profiles={"W8A8": .4, "HEAD_W16A16": .9}),
          evaluation("b", profiles={"HEAD_W8A16": .6})]
    joint = m.joint_points(ev, filtered["rows"], BUDGET)
    assert {(p["architecture_id"], p["profile"], p["mapping_policy"]) for p in joint} == keys
    assert max(p["Accuracy"] for p in joint) == pytest.approx(.6)


def test_boundary_preservation_and_ties_are_reproducible_under_input_permutation():
    ev = [evaluation("best", profiles={"W8A8": .8}),
          evaluation("boundary", profiles={"W8A8": .7995}),
          evaluation("near_tie", profiles={"W8A8": .7994}),
          evaluation("cheap", profiles={"W8A8": .7})]
    costs = [cost("best", pressure=.95), cost("boundary", pressure=.94),
             cost("near_tie", pressure=.93), cost("cheap", pressure=.1)]
    joint = points(ev, costs)
    before = copy.deepcopy(joint)
    first = m.promote(joint, 3, PROMOTION)
    second = m.promote(list(reversed(joint)), 3, PROMOTION)
    assert first["selected"] == second["selected"]
    assert first["ranked"] == second["ranked"]
    assert first["boundary"] == second["boundary"]
    assert first["boundary"]
    assert len(first["selected"]) == len(set(first["selected"])) == 3
    assert joint == before


def test_missing_hardware_budget_allows_plan_but_cannot_authorize_run():
    cfg = yaml.safe_load((ROOT / "configs/search/sonar_joint_v2.yaml").read_text(encoding="utf-8"))
    cfg["hardware_budget"]["limits"] = dict.fromkeys(m.METRICS)
    m.validate_config(cfg, require_budget=False)
    with pytest.raises(ValueError, match="budget|预算|limit"):
        m.validate_config(cfg, require_budget=True)
    report = m.filter_costs([cost("otherwise_valid")], cfg["hardware_budget"])
    assert not report["rows"]
