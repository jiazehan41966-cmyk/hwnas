"""V2 orchestration regressions; workers return synthetic fixture scores only."""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sonar_joint_v2_runner_integration", ROOT / "tools/sonar_joint_v2/run.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def config():
    return yaml.safe_load((ROOT / "configs/search/sonar_joint_v2.yaml").read_text(encoding="utf-8"))


def costs(cfg, aids, pressures=None, mappings=("reference_low",)):
    pressures = pressures or {aid: .2 for aid in aids}
    return [dict(architecture_id=aid, profile=q, mapping_policy=mapping,
                 status="LOOKED_UP", measurement_valid=True, timing_eligible=True,
                 graph_hash=f"fixture-graph-{aid}", mapping_hash=f"fixture-mapping-{mapping}",
                 quant_contract_hash="fixture-numeric-contract",
                 **{metric: ceiling*pressures[aid]
                    for metric, ceiling in cfg["hardware_budget"]["limits"].items()})
            for aid in aids for q in cfg["quant_profiles"] for mapping in mappings]


def evaluation(aid, seed=42, budget=80, fp32=.5, uniform=.499, mixed=.5):
    return dict(architecture_id=aid, seed=seed, budget=budget,
                checkpoint=f"training/{aid}/seed_{seed}/best_{budget}.pt",
                checkpoint_sha256=f"fixture-checkpoint-{aid}-{seed}-{budget}",
                calibration_sha256="fixture-train-calibration", split="valid", test_evaluated=False,
                fp32={"mAP50_95": fp32},
                profiles={q: dict(mAP50_95=uniform if q == "W8A8" else mixed,
                                  profile_sha256=f"fixture-{aid}-{seed}-{budget}-{q}")
                          for q in m.policy.PROFILES})


def test_each_training_budget_quantizes_before_promotion_and_confirms_paired_seeds(tmp_path, monkeypatch):
    cfg = config()
    # Small population exercises the real 40/60/80 execution loop without training.
    cfg.update(populations=[4, 3, 2], shortlist=1)
    aids = ["fixture_float_winner", "fixture_quant_winner", "fixture_efficient", "fixture_overbudget"]
    rows = costs(cfg, aids, {aids[0]: .8, aids[1]: .7, aids[2]: .1, aids[3]: 2})
    out = tmp_path / "synthetic_campaign"
    m.save(out / "freeze.json", dict(protocol=cfg, synthetic_fixture=True))
    m.save(out / "cost_snapshot.json", dict(rows=rows))
    monkeypatch.setattr(m, "verify", lambda config_path, target: {"protocol": cfg})
    calls = []

    def fixture_task(target, mode, aid, seed, budget, device):
        assert target == out and device == "cpu"
        calls.append((mode, aid, seed, budget))
        if aid == "fixture_quant_winner":
            result = evaluation(aid, seed, budget, fp32=.8, uniform=.799, mixed=.8)
        elif aid == "fixture_efficient":
            result = evaluation(aid, seed, budget, fp32=.5, uniform=.499, mixed=.5)
        else:
            result = evaluation(aid, seed, budget, fp32=.9, uniform=.4, mixed=.401)
        return result if mode == "quant" else dict(
            architecture_id=aid, seed=seed, budget=budget, mAP50_95=result["fp32"]["mAP50_95"],
            checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"], test_evaluated=False)

    monkeypatch.setattr(m, "task", fixture_task)
    m.execute(ROOT / "configs/search/sonar_joint_v2.yaml", out, "cpu")
    assert all(aid != "fixture_overbudget" for _, aid, _, _ in calls)
    assert len(calls) % 2 == 0
    for train, quant in zip(calls[::2], calls[1::2]):
        assert train[0] == "train" and quant == ("quant", *train[1:])
    for budget in (40, 60, 80):
        stage = m.read(out / f"stage_{budget}.json")
        assert stage["evaluations"]
        assert all(r["budget"] == budget and r["seed"] == 42 for r in stage["evaluations"])
        assert all(p["budget"] == budget for p in stage["promotion"]["points"])
        assert stage["promotion"]["deployment"] == "NOT_CLAIMED"
    shortlist = m.read(out / "shortlist.json")["architectures"]
    assert shortlist == ["fixture_quant_winner"]
    assert {seed for mode, aid, seed, budget in calls
            if mode == "quant" and aid == shortlist[0] and budget == 80} == {42, 43, 44}
    report = m.read(out / "joint_pareto.json")
    assert report["points"] and all(p["n_seeds"] == 3 for p in report["points"])
    assert report["deployment"] == "NOT_CLAIMED" and report["test_evaluated"] is False
    queue = m.read(out / "deployment_queue.json")
    assert queue["points"] and queue["launch_hardware"] is False
    assert queue["deployment"] == "NOT_CLAIMED"
    assert m.read(out / "state.json")["status"] == "COMPLETE_SCREENING"


def test_empty_hardware_screen_refuses_freeze_before_data_read_or_worker_launch(tmp_path, monkeypatch):
    cfg = config()
    out = tmp_path / "rejected_freeze"
    monkeypatch.setattr(m, "load_config", lambda path, require_budget=False: cfg)
    monkeypatch.setattr(m, "preflight", lambda _: dict(
        status="NO_BUDGET_FEASIBLE_TRIPLES", eligible_architectures=[], rows=[]))
    monkeypatch.setattr(m.legacy, "data_binding", lambda _: pytest.fail("Rejected freeze must not inspect data"))
    monkeypatch.setattr(m, "task", lambda *a, **k: pytest.fail("Rejected freeze must not launch a worker"))
    with pytest.raises(ValueError, match="No hardware-screened"):
        m.freeze(ROOT / "configs/search/sonar_joint_v2.yaml", out)
    assert not out.exists()


def test_quantized_rejection_audit_stops_selection_before_confirmation(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(populations=[3, 2, 1], shortlist=1)
    aids = [f"P3_P4_P5_sfixture_{name}" for name in ("a", "b", "c")]
    out = tmp_path / "synthetic_audit_campaign"
    m.save(out / "cost_snapshot.json", dict(rows=costs(cfg, aids)))
    monkeypatch.setattr(m, "verify", lambda config_path, target: {"protocol": cfg})
    calls = []

    def fixture_task(target, mode, aid, seed, budget, device):
        calls.append((mode, aid, seed, budget))
        value = dict(zip(aids, (.8, .7, .6)))[aid]
        if aid == aids[1] and budget == 80:
            value = .81
        # All floating scores are identical; the violation comes from PTQ.
        return evaluation(aid, seed, budget, fp32=.8, uniform=value, mixed=value)

    monkeypatch.setattr(m, "task", fixture_task)
    with pytest.raises(ValueError, match="Quantized rejection audit"):
        m.execute(ROOT / "configs/search/sonar_joint_v2.yaml", out, "cpu")
    audit = m.read(out / "rejection_audit.json")
    assert audit["violations"] == [aids[1]]
    assert audit["quantized_top5_threshold"] == pytest.approx(.8)
    assert all(seed == 42 for _, _, seed, _ in calls)
    assert not (out / "joint_pareto.json").exists()
    state = m.read(out / "state.json")
    assert state["status"] == "FAILED_RESUMABLE" and state["deployment"] == "NOT_CLAIMED"


def test_confirmation_admits_new_architecture_without_old_accuracy_allowlist():
    cfg = config()
    aid = "new_fixture_architecture_outside_B1_C_D"
    ev = {aid: {seed: evaluation(aid, seed) for seed in cfg["confirmation_seeds"]}}
    rows = costs(cfg, [aid], mappings=("reference_low", "reference_high"))
    for row in rows:
        row.update(Accuracy=.99, q_status="Q_NOT_ADMITTED", eligible=False)
    before = copy.deepcopy(rows)
    result = m.confirmed_points(ev, rows, cfg)
    assert len(result["points"]) == 6
    assert {p["architecture_id"] for p in result["points"]} == {aid}
    assert sorted({p["Accuracy"] for p in result["points"]}) == pytest.approx([.499, .5])
    assert all(p["accuracy_admission"] == "PAIRED_VALIDATION_PASS" and p["n_seeds"] == 3
               for p in result["points"])
    assert rows == before
    assert result["deployment"] == "NOT_CLAIMED"


def test_missing_confirmation_seed_cannot_publish_partial_front():
    cfg = config()
    aid = "fixture_new_architecture"
    ev = {aid: {seed: evaluation(aid, seed) for seed in (42, 43)}}
    with pytest.raises(ValueError, match="Three paired"):
        m.confirmed_points(ev, costs(cfg, [aid]), cfg)


def test_existing_paired_admission_failure_cannot_publish_points():
    cfg = config()
    aid = "fixture_failed_admission"
    ev = {aid: {seed: evaluation(aid, seed, fp32=.5, uniform=.45, mixed=.45)
                for seed in cfg["confirmation_seeds"]}}
    result = m.confirmed_points(ev, costs(cfg, [aid]), cfg)
    assert not result["points"] and not result["pareto"]
    assert len(result["rejected"]) == 3
    assert {r["reason"] for r in result["rejected"]} == {"PAIRED_ACCURACY_GATE"}


@pytest.mark.parametrize("field,value", [("seed", 99), ("split", "test"), ("test_evaluated", True)])
def test_every_confirmation_run_requires_matching_seed_and_validation_identity(field, value):
    cfg = config()
    aid = "fixture_confirmation_identity"
    ev = {aid: {seed: evaluation(aid, seed) for seed in cfg["confirmation_seeds"]}}
    ev[aid][43][field] = value
    with pytest.raises(ValueError, match="identity|Validation|validation"):
        m.confirmed_points(ev, costs(cfg, [aid]), cfg)
