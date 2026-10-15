"""Software checks only; fixtures are explicitly synthetic, never measurements."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest
import torch
from tools.sonar_joint_v2 import policy
from tools.sonar_light_rl.controller import JointController
from tools.sonar_light_rl.protocol import (cost_status, default_protocol,
    validate_protocol)
from tools.sonar_light_rl import run


def test_masked_joint_probability_and_known_reward():
    cfg = default_protocol()
    controller = JointController(cfg)
    mask = [False] * 1296
    mask[0], mask[19] = True, True
    distribution = controller.distribution(mask)
    assert float(distribution.probs.sum()) == pytest.approx(1.)
    assert torch.count_nonzero(distribution.probs) == 2
    for _ in range(10):
        index, logprob, _ = controller.sample(mask)
        assert index in (0, 19)
        assert float(logprob) == pytest.approx(float(distribution.probs[index].log()))
    cost = {k: v / 2 for k, v in cfg["hardware_budget"]["limits"].items()}
    assert controller.compute_reward(cost, .8, feasible=True) == pytest.approx(.55)
    assert controller.compute_reward(cost, feasible=False) == -1.
    assert controller.reward.energy_weight == 0


def test_joint_identity_and_unknown_semantics():
    cfg = default_protocol()
    req = run.requirements()[0]
    cost = dict(req, status="NEEDS_MEASUREMENT", measurement_valid=False,
                timing_eligible=False, Latency=999999999., LUT=1., DSP=1., BRAM=1.)
    assert cost_status(cost, req, cfg) == "NEEDS_MEASUREMENT"
    cost.update(status="LOOKED_UP", measurement_valid=True, timing_eligible=True)
    assert cost_status(cost, req, cfg) == "OVER_BUDGET"
    cost["profile"] = "BAD_ALIAS"
    with pytest.raises(ValueError, match="bind"):
        cost_status(cost, req, cfg)
    pool = run.requirements()
    assert len(pool) == 1296
    assert len({r["architecture_id"] for r in pool}) == 72
    assert {r["profile"] for r in pool} == set(policy.PROFILES)
    assert {r["mapping_policy"] for r in pool} == set(policy.POLICIES)


def test_unknown_cost_does_not_train_or_penalize(tmp_path):
    cfg = default_protocol()
    evidence = run.ReplayEvidence(dict(schema_version="sonar_light_rl_replay_v1"))
    report = run.run_replay(cfg, evidence, tmp_path)
    assert report["status"] == "BLOCKED_NO_EVALUATED_FEASIBLE_POINTS"
    assert report["controller_updates"] == 0
    assert not report["training_started"]
    events = json.loads((tmp_path / "events.json").read_text())["events"]
    assert all(e["reward"] is None and not e["controller_updated"] for e in events)
    assert len(json.loads((tmp_path / "measurement_queue.json").read_text())["tasks"]) == 8


def test_known_overbudget_gets_feedback_without_evaluation(tmp_path):
    cfg = default_protocol()
    costs = [dict(req, status="LOOKED_UP", measurement_valid=True, timing_eligible=True,
        **{k: v * 2 for k, v in cfg["hardware_budget"]["limits"].items()}) for req in run.requirements()]
    evidence = run.ReplayEvidence(dict(schema_version="sonar_light_rl_replay_v1", costs=costs))
    result = run.run_replay(cfg, evidence, tmp_path)
    assert result["controller_updates"] == 8
    assert result["pareto_by_budget"] == {}
    assert all(e["reward"] == -1 for e in json.loads((tmp_path / "events.json").read_text())["events"])


def test_resume_matches_uninterrupted_and_separates_fidelity(tmp_path):
    cfg = default_protocol()
    evidence = run.synthetic_evidence(cfg)
    full = run.run_replay(cfg, evidence, tmp_path / "full")
    first = run.run_replay(cfg, evidence, tmp_path / "resume", max_attempts=5)
    assert first["status"] == "INTERRUPTED_DIAGNOSTIC"
    resumed = run.run_replay(cfg, evidence, tmp_path / "resume", resume=True)
    assert full == resumed
    assert (tmp_path / "full/events.json").read_bytes() == (tmp_path / "resume/events.json").read_bytes()
    assert full["status"] == "COMPLETE_DIAGNOSTIC_NOT_RESEARCH"
    assert full["completed_stages"] == [40, 60, 80]
    for budget, points in full["pareto_by_budget"].items():
        assert all(p["budget"] == int(budget) and not p["formal_eligible"] for p in points)
    assert all(r["status"] == "PENDING_THREE_SEED_CONFIRMATION" and not r["launch_hardware"]
               for r in full["pending_handoffs"])


def test_resume_rejects_evidence_protocol_and_source_drift(tmp_path, monkeypatch):
    cfg = default_protocol()
    evidence = run.synthetic_evidence(cfg)
    run.run_replay(cfg, evidence, tmp_path, max_attempts=1)
    changed = deepcopy(cfg)
    changed["reward"]["lut_weight"] += .01
    with pytest.raises(ValueError, match="drift"):
        run.run_replay(changed, evidence, tmp_path, resume=True)
    modified = deepcopy(evidence.payload)
    modified["costs"][0]["Latency"] += 1
    with pytest.raises(ValueError, match="drift"):
        run.run_replay(cfg, run.ReplayEvidence(modified), tmp_path, resume=True)
    monkeypatch.setattr(run, "source_binding", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="drift"):
        run.run_replay(cfg, evidence, tmp_path, resume=True)


def test_stage_candidate_not_repeated_and_no_false_claim(tmp_path):
    cfg = default_protocol()
    report = run.run_replay(cfg, run.synthetic_evidence(cfg), tmp_path)
    events = json.loads((tmp_path / "events.json").read_text())["events"]
    keys = [(e["budget"], e["candidate"]["architecture_id"], e["candidate"]["profile"],
             e["candidate"]["mapping_policy"]) for e in events]
    assert len(keys) == len(set(keys))
    assert all(e["evidence_kind"] == "synthetic_diagnostic" for e in events)
    assert not report["formal_eligible"] and report["confirmation"] == "NOT_EXECUTED"
    with pytest.raises(FileExistsError):
        run.run_replay(cfg, run.synthetic_evidence(cfg), tmp_path)


def test_formal_run_preserves_existing_blockers(tmp_path, monkeypatch):
    calls = []
    def fake_preflight(cfg):
        calls.append("preflight")
        return dict(blockers=["COMMON_GPU_ACCOUNTING_LAUNCH_DRIVER_NOT_IMPLEMENTED"])
    monkeypatch.setattr(run.base, "preflight", fake_preflight)
    monkeypatch.setattr(run.base, "train_candidate", lambda *a, **k: pytest.fail("Training is forbidden"))
    assert run.main(["run", "--output", str(tmp_path)]) == 2
    report = json.loads((tmp_path / "plan.json").read_text())
    assert calls == ["preflight"]
    assert "COMMON_GPU_ACCOUNTING_LAUNCH_DRIVER_NOT_IMPLEMENTED" in report["blockers"]
    assert not report["training_started"]


@pytest.mark.parametrize("field,value", [("gpu_h_cap", 0), ("gpu_model", ""), ("baseline_momentum", 1.)])
def test_invalid_explicit_protocol_values_rejected(field, value):
    cfg = default_protocol()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_protocol(cfg)


def test_replay_rejects_test_split():
    with pytest.raises(ValueError, match="validation"):
        run.ReplayEvidence(dict(schema_version="sonar_light_rl_replay_v1", evaluations=[
            dict(architecture_id="x", seed=42, budget=40, split="test", test_evaluated=True)]))
