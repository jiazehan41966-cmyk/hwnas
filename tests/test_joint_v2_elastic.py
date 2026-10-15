"""New-space V2 contract tests; fixtures are synthetic and are not experiments."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.modnas_comparison import joint_v2_elastic as adapter
from tools.sonar_joint_v2 import policy


def test_72_unique_candidates_canonicalize_inactive_pan():
    pool = adapter.architectures()
    assert len(pool) == 72
    assert sum(not arch["pan"] for arch in pool.values()) == 8
    assert sum(arch["pan"] for arch in pool.values()) == 64
    assert len({adapter.architecture_hash(arch) for arch in pool.values()}) == 72
    assert all(not arch["pan_kernels"] for arch in pool.values() if not arch["pan"])
    with pytest.raises(ValueError):
        adapter.validate_architecture(dict(pan=False, td_kernels=[3]*3, pan_kernels=[3]*3))


def test_training_models_are_real_independent_new_models_with_class_specific_heads():
    aid = next(iter(adapter.architectures()))
    left = adapter.build_training_model(aid, 42, 2)
    right = adapter.build_training_model(aid, 42, 2)
    changed = adapter.build_training_model(aid, 42, 10)
    assert isinstance(left, adapter.StaticElasticNeckDetector)
    assert left is not right and all(parameter.requires_grad for parameter in left.parameters())
    assert all(a.data_ptr() != b.data_ptr() and torch.equal(a, b)
               for a, b in zip(left.parameters(), right.parameters()))
    assert left.head.prediction_channels == 21
    assert changed.head.prediction_channels == 45
    changed.eval()
    with torch.no_grad():
        predictions = changed(torch.zeros(1, 1, 640, 640))["predictions"]
    assert [list(value.shape) for value in predictions] == [
        [1,45,160,160], [1,45,80,80], [1,45,40,40], [1,45,20,20]]


def test_all_three_dimensions_are_kept_without_precision_or_mapping_aliases():
    cfg = adapter.load_config()
    rows = adapter.triple_requirements(cfg)
    assert len(rows) == 1296
    assert len({(r["architecture_id"], r["profile"], r["mapping_policy"]) for r in rows}) == 1296
    assert {r["profile"] for r in rows} == set(policy.PROFILES)
    assert {r["mapping_policy"] for r in rows} == set(policy.POLICIES)
    assert all(r["status"] == "NEEDS_MEASUREMENT" and r["Latency"] is None for r in rows)


def test_freeze_rejects_unknown_cost_unsupported_profile_and_budget_before_data(monkeypatch, tmp_path):
    cfg = adapter.load_config()
    monkeypatch.setattr(adapter, "_backend_module", lambda _: SimpleNamespace(SUPPORTED_PROFILES=("W8A8",)))
    from tools.modnas_elastic import protocol
    monkeypatch.setattr(protocol, "dataset_binding", lambda *_: pytest.fail("Blocked freeze must not read dataset"))
    result = adapter.preflight(cfg)
    assert result["status"] == "BLOCKED_NOT_LAUNCHABLE"
    assert result["dataset_images_opened"] is False
    assert result["eligible_triples"] == 0
    assert result["qualified_hardware_triples"] == 0
    assert "UNSUPPORTED_NEWSPACE_PRECISION_PROFILES" in result["blockers"]
    assert "COMMON_GPU_MODEL_AND_TOTAL_BUDGET_NOT_FROZEN" in result["blockers"]
    assert "MODNAS_COMMON_PROFILE_MAPPING_DECISION_PROTOCOL_UNIMPLEMENTED" in result["blockers"]
    with pytest.raises(adapter.CampaignBlocked):
        adapter.freeze(adapter.CONFIG, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_single_profile_or_population_copy_rejected():
    cfg = adapter.load_config()
    cfg["quant_profiles"] = ["W8A8"]
    with pytest.raises(ValueError):
        adapter.validate_config(cfg)
    cfg = adapter.load_config()
    cfg["populations"] = [144, 108, 72]
    with pytest.raises(ValueError):
        adapter.validate_config(cfg)


def test_current_real_software_gate_is_source_bound_and_only_software_ready():
    report = adapter.preflight(adapter.load_config())
    software = report["software_backend"]
    assert software["status"] == "READY"
    assert software["verified_execution_sources"] == 23
    assert software["verified_artifacts"] == 36
    assert software["verified_all_node_rows"] == 361
    assert software["macro_profile_groups"] == 6
    assert software["accuracy_retention"] == "NOT_EVALUATED"
    assert software["hardware_feasibility"] == "UNKNOWN"
    assert len(software["mixed_head_boundary_requirements"]) == 4
    assert all(row["count"] == 4 for row in software["mixed_head_boundary_requirements"])
    assert report["status"] == "BLOCKED_NOT_LAUNCHABLE"
    assert report["training_started"] is False and report["dataset_images_opened"] is False


def test_metadata_frozen_cannot_enable_launch_or_data_access(monkeypatch, tmp_path):
    cfg = adapter.load_config()
    cfg["comparison"].update(common_decision_space_status="FROZEN",
        modnas_decision_interface_status="VERIFIED_SHARED_1296_DECISIONS",
        gpu_model="synthetic_no_launch_unit_fixture", gpu_h_cap_per_method_per_independent_search=1)
    monkeypatch.setattr(adapter, "load_config", lambda _: cfg)
    from tools.modnas_elastic import protocol
    monkeypatch.setattr(protocol, "dataset_binding", lambda *_: pytest.fail("No dataset access allowed"))
    with pytest.raises(adapter.CampaignBlocked, match="COMMON_GPU_ACCOUNTING_LAUNCH_DRIVER_NOT_IMPLEMENTED"):
        adapter.freeze(adapter.CONFIG, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_preregistered_real_gate_hash_rejects_changes_before_backend_import(monkeypatch):
    cfg = adapter.load_config()
    cfg["software_backend_evidence_sha256"] = "0" * 64
    monkeypatch.setattr(adapter, "_backend_module", lambda *_: pytest.fail("Must verify proof before execution import"))
    with pytest.raises(ValueError, match="preregistered proof"):
        adapter.software_readiness(cfg)


def test_source_or_leaf_artifact_drift_rejects_real_gate(monkeypatch):
    cfg = adapter.load_config()
    original = adapter.sha
    monkeypatch.setattr(adapter, "sha", lambda path: "0"*64 if str(path).endswith("quantization_mixed.py") else original(path))
    with pytest.raises(ValueError, match="source drift"):
        adapter.software_readiness(cfg)
    monkeypatch.setattr(adapter, "sha", lambda path: "0"*64 if str(path).endswith("integer_parameters.pt") else original(path))
    with pytest.raises(ValueError, match="PTQ artifact drift"):
        adapter.software_readiness(cfg)


@pytest.mark.parametrize("corruption", ["missing_node", "mismatch"])
def test_real_witness_semantics_reject_missing_node_and_nonzero_mismatch(monkeypatch, corruption):
    cfg = adapter.load_config()
    proof, _ = adapter._source_proof(cfg["software_backend_evidence"])
    target = adapter._workspace_file(cfg["software_backend_evidence"])
    original = adapter.Path.read_text
    def corrupt(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == "software_parity_witness.json":
            value = json.loads(text)
            if corruption == "missing_node":
                value["rows"].pop()
            else:
                value["rows"][0]["mismatch_count"] = 1
            return json.dumps(value)
        return text
    monkeypatch.setattr(adapter.Path, "read_text", corrupt)
    with pytest.raises(ValueError, match="per-node hash parity"):
        adapter._validate_real_software_proof(proof, target, cfg)


def _stage_fixture(cfg):
    # Explicit pure-policy unit fixtures, never written as experimental results.
    ids = sorted(adapter.architectures())
    evaluations, costs = [], []
    for index, aid in enumerate(ids):
        accuracy = .5 + ((index * 13) % 72) * .004
        evaluations.append(dict(architecture_id=aid, seed=cfg["search_seed"], budget=40,
            split="valid", test_evaluated=False, fp32={"mAP50_95": accuracy},
            profiles={profile: dict(mAP50_95=accuracy, profile_sha256="unit_fixture")
                      for profile in policy.PROFILES},
            checkpoint_sha256="unit_fixture", calibration_sha256="unit_fixture"))
        for profile in policy.PROFILES:
            for mapping in policy.POLICIES:
                costs.append(dict(architecture_id=aid, profile=profile, mapping_policy=mapping,
                    status="LOOKED_UP", measurement_valid=True, timing_eligible=True,
                    Latency=100., LUT=100., DSP=1., BRAM=1.))
    return ids, evaluations, costs


def test_promotion_executes_original_policy_and_audits_E0_E1():
    cfg = adapter.load_config()
    ids, evaluations, costs = _stage_fixture(cfg)
    expected = policy.promote(policy.joint_points(evaluations, costs, cfg["hardware_budget"]),
                              54, cfg["promotion"])
    actual = adapter.promote_stage(evaluations, costs, cfg, stage=0, previous=ids)
    assert actual["selected"] == expected["selected"]
    assert actual["boundary"] == expected["boundary"]
    assert actual["ranked"] == expected["ranked"]
    assert len(actual["selected"]) == 54
    assert {adapter.macro(aid) for aid in actual["rejection_audits"]} == {"E0", "E1"}
    assert set(actual["rejection_audits"]) <= set(ids) - set(actual["selected"])
    assert actual["rejection_audits"] == adapter.select_rejection_audits(
        ids, actual["selected"], stage=0, search_seed=cfg["search_seed"])


def test_quantized_rejection_above_top5_aborts():
    final = [dict(architecture_id=str(i), Accuracy=.8 - .01*i) for i in range(6)]
    with pytest.raises(adapter.CampaignBlocked):
        adapter.check_rejection_audit(final, [dict(architecture_id="discarded", Accuracy=.79)])
    assert adapter.check_rejection_audit(final, [])['quantized_top5_threshold'] == pytest.approx(.76)


def test_confirmation_executes_original_paired_admission_without_relabeling_searches():
    cfg = adapter.load_config()
    ids, staged, costs = _stage_fixture(cfg)
    aid = ids[0]
    rows = {}
    for seed in cfg["confirmation_seeds"]:
        row = deepcopy(staged[0])
        row.update(seed=seed, budget=80)
        row["fp32"]["mAP50_95"] = .7
        row["profiles"]["W8A8"]["mAP50_95"] = .7
        row["profiles"]["HEAD_W8A16"]["mAP50_95"] = .701
        # Actual paired improvement gate: no gain over W8A8 -> reject this profile.
        row["profiles"]["HEAD_W16A16"]["mAP50_95"] = .7
        rows[seed] = row
    from tools.sonar_joint_v2.run import confirmed_points as original
    expected = original({aid: rows}, costs, cfg)
    actual = adapter.confirmed_points({aid: rows}, costs, cfg)
    assert all(actual[key] == value for key, value in expected.items())
    assert {p["profile"] for p in actual["points"]} == {"W8A8", "HEAD_W8A16"}
    assert all(p["n_seeds"] == 3 for p in actual["points"])
    assert actual["n_independent_searches_represented"] == 1
    assert actual["confirmation_seeds_are_not_independent_search_repetitions"] is True
    with pytest.raises(ValueError, match="Three paired"):
        adapter.confirmed_points({aid: {42: rows[42]}}, costs, cfg)


def test_complete_freeze_source_closure_rejects_deleted_required_key():
    cfg = adapter.load_config()
    sources = adapter._source_inputs(adapter.CONFIG, cfg)
    binding = dict(protocol=cfg, protocol_sha256=adapter.digest(cfg),
        config_path=adapter.CONFIG.relative_to(adapter.ROOT).as_posix(),
        config_sha256=adapter.sha(adapter.CONFIG), source_sha256=sources,
        source_closure_sha256=adapter.digest(sources))
    assert adapter._verify_freeze_sources(binding) == cfg
    del binding["source_sha256"]["tools/sonar_spec/formal_runner.py"]
    # Even recomputing the subset digest cannot hide a required source deletion.
    binding["source_closure_sha256"] = adapter.digest(binding["source_sha256"])
    with pytest.raises(ValueError, match="complete execution-source closure"):
        adapter._verify_freeze_sources(binding)


def _resume_fixture():
    aid = next(iter(adapter.architectures()))
    state = dict(freeze_sha256="fixture", architecture_id=aid, seed=42,
        architecture=adapter.architectures()[aid], num_classes=2, test_evaluated=False,
        epoch=3, history=[dict(epoch=i+1,valid=dict(mAP50_95=v)) for i,v in enumerate([.2,.3,.3])],
        best_epoch=2, best_score=.3, best_state={"weight":torch.tensor([1.])},
        model_state_dict={"weight":torch.tensor([2.])}, optimizer_state_dict={"fixture":True}, rng={"fixture":True})
    return aid, state


@pytest.mark.parametrize("mutation", ["epoch", "history", "best_epoch", "best_score", "num_classes", "test_evaluated"])
def test_resume_rejects_history_identity_and_non_earliest_best(mutation):
    aid,state = _resume_fixture()
    adapter._validate_training_state(state, aid, 42, 40, "fixture", 2)
    if mutation == "history":
        state["history"][1]["epoch"] = 1
    else:
        state[mutation] = dict(epoch=4, best_epoch=3, best_score=.2, num_classes=10, test_evaluated=True)[mutation]
    with pytest.raises(ValueError):
        adapter._validate_training_state(state, aid, 42, 40, "fixture", 2)


def test_budget_checkpoint_requires_matching_train_receipt_and_epoch(tmp_path):
    aid,state = _resume_fixture()
    state.update(budget=40,epoch=2,valid=dict(mAP50_95=.3))
    receipt = dict(freeze_sha256="fixture",architecture_id=aid,seed=42,budget=40,
        num_classes=2,test_evaluated=False,best_epoch=2,mAP50_95=.3)
    adapter._validate_budget_checkpoint(state, receipt, aid,42,40,"fixture",2)
    receipt["best_epoch"] = 3
    with pytest.raises(ValueError, match="epoch/validation receipt"):
        adapter._validate_budget_checkpoint(state, receipt, aid,42,40,"fixture",2)
    assert adapter._training_receipt(tmp_path, aid,42,40,"fixture",2) is None


def test_training_receipt_rejects_changed_checkpoint_bytes(tmp_path):
    aid,state=_resume_fixture()
    state.update(budget=40,epoch=2,valid=dict(mAP50_95=.3))
    checkpoint=tmp_path/"training"/aid/"seed_42"/"best_40.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(state,checkpoint)
    receipt=dict(freeze_sha256="fixture",architecture_id=aid,seed=42,budget=40,
        num_classes=2,test_evaluated=False,best_epoch=2,mAP50_95=.3,
        checkpoint=checkpoint.relative_to(tmp_path).as_posix(),checkpoint_sha256=adapter.sha(checkpoint))
    adapter.save(tmp_path/"tasks"/f"train_{aid}_42_40.json",receipt)
    assert adapter._training_receipt(tmp_path,aid,42,40,"fixture",2)[0]==receipt
    with checkpoint.open("ab") as handle:
        handle.write(b"unit fixture corruption")
    with pytest.raises(ValueError,match="SHA256 drift"):
        adapter._training_receipt(tmp_path,aid,42,40,"fixture",2)


@pytest.mark.parametrize("group,profile", [(group,profile) for group in adapter.MACROS for profile in adapter.PROFILES])
def test_execution_parity_validation_rejects_failed_or_misbound_real_witness(group, profile):
    cfg=adapter.load_config()
    leaf=(adapter.ROOT/cfg["software_backend_evidence"]).parent/group/profile
    record=json.loads((leaf/"profile.json").read_text())
    graph=json.loads((leaf/"numeric_graph.json").read_text())
    witness=json.loads((leaf/"software_parity_witness.json").read_text())
    prepared=SimpleNamespace(profile=record,integer_reference=SimpleNamespace(graph=graph))
    kwargs=dict(profile=profile,architecture=record["architecture"],num_classes=2,
        source_state_sha=record["original_model_state_sha256"],calibration_sha=record["calibration"]["calibration_sha256"])
    adapter._validate_ptq_witness(prepared,witness,**kwargs)
    broken=deepcopy(witness)
    broken.update(status="FAIL_SOFTWARE_INTEGER_FAKE_PARITY",mismatches=1)
    with pytest.raises(ValueError,match="Actual integer parity failed"):
        adapter._validate_ptq_witness(prepared,broken,**kwargs)
    broken=deepcopy(witness)
    broken["original_model_state_sha256"]="0"*64
    with pytest.raises(ValueError,match="source/calibration"):
        adapter._validate_ptq_witness(prepared,broken,**kwargs)


@pytest.mark.parametrize("search_seed", [42, 43, 44])
def test_independent_search_seed_is_distinct_from_internal_confirmation(search_seed):
    cfg = adapter.load_config()
    cfg["search_seed"] = search_seed
    assert adapter.validate_config(cfg)["confirmation_seeds"] == [42,43,44]
    aid = next(iter(adapter.architectures()))
    assert adapter.task_requests(cfg, [aid], 40, search_seed)[0]["seed"] == search_seed
    other = next(value for value in [42,43,44] if value != search_seed)
    with pytest.raises(ValueError):
        adapter.task_requests(cfg, [aid], 40, other)
    assert adapter.task_requests(cfg, [aid], 80, other)[0]["budget"] == 80
