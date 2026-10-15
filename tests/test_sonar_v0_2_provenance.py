"""Synthetic admission fixtures; never experiment results or trained packages."""
from copy import deepcopy
import json

import pytest
import torch

from tools.detector_v0_2.provenance import sha, verify_execution_manifest, verify_completed_training


@pytest.fixture
def manifest_fixture(tmp_path,monkeypatch):
    from tools.detector_v0_2 import train as runner
    for name in ("config.yaml","spec.yaml","training_source.py","train_image","data.yaml"):
        (tmp_path/name).write_text(name)
    cfg = dict(model_spec="spec.yaml",training={"epochs":80})
    def sources(_cfg):
        return {"training_source.py":sha(tmp_path/"training_source.py")}
    def inputs(_cfg,_baseline):
        return dict(datasets={"synthetic_dependency_witness":sha(tmp_path/"train_image")},
            dataset_metadata_sha256=sha(tmp_path/"data.yaml"),historical_frozen_files_verified=42,
            test_performance_accessed=False,additional_glob_files=["unrelated_new_deploy.py"])
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"CONFIG",tmp_path/"config.yaml")
    monkeypatch.setattr(runner,"load_config",lambda:(cfg,{}))
    monkeypatch.setattr(runner,"source_hashes",sources)
    monkeypatch.setattr(runner,"verify_inputs",inputs)
    payload=dict(config_sha256=sha(runner.CONFIG),model_spec_sha256=sha(tmp_path/"spec.yaml"),
        source_sha256=sources(cfg),training=cfg["training"],test_evaluated=False,
        input_verification=inputs(cfg,{}))
    payload["input_verification"]["additional_glob_files"]=[]
    path=tmp_path/"execution_manifest.json"
    path.write_text(json.dumps(payload))
    binding=dict(execution_manifest_sha256=sha(path),prototype="R1",seed=42,epochs=80,
        namespace="sonar_candidate_v0_2",formal_eligible=True,test_evaluated=False)
    return tmp_path,path,binding


def test_manifest_checks_actual_bytes_and_all_dependencies(manifest_fixture):
    _,path,binding=manifest_fixture
    assert verify_execution_manifest(path,binding)["test_evaluated"] is False


@pytest.mark.parametrize("name",["config.yaml","spec.yaml","training_source.py","train_image","data.yaml"])
def test_manifest_rejects_actual_dependency_drift(manifest_fixture,name):
    root,path,binding=manifest_fixture
    (root/name).write_text("changed")
    with pytest.raises(ValueError,match="binding mismatch"):
        verify_execution_manifest(path,binding)


def test_semantically_identical_reencoded_manifest_has_different_identity(manifest_fixture):
    _,path,binding=manifest_fixture
    path.write_text(json.dumps(json.loads(path.read_text()),indent=2))
    with pytest.raises(ValueError,match="manifest SHA256"):
        verify_execution_manifest(path,binding)


@pytest.fixture
def complete_fixture(manifest_fixture):
    root,path,binding=manifest_fixture
    folder=root/"SYNTHETIC_UNIT_FIXTURE_NOT_A_TRAINING_RESULT";folder.mkdir()
    history=[]
    for epoch in range(1,81):
        # Tied maximum at epochs20/30 must select20, not the final checkpoint.
        value=.04 if epoch in (20,30) else .001
        history.append(dict(epoch=epoch,batches=83,gradient_finite=True,
            valid=dict(num_images=99,mAP50_95=value,AP50=value,AP75=value,Precision=.02,Recall=.3)))
    state=dict(binding=binding,epoch=20,architecture={"prototype_id":"R1"},history=history[:20],
               model_state_dict={"fixture.num_batches_tracked":torch.tensor(20*83)})
    torch.save(state,folder/"best.pt")
    latest=deepcopy(state);latest.update(epoch=80,history=history)
    torch.save(latest,folder/"latest.pt")
    (folder/"epoch_log.json").write_text(json.dumps(history))
    (folder/"status.json").write_text(json.dumps(dict(status="COMPLETE",epochs=80,binding=binding,test_evaluated=False)))
    result=dict(prototype="R1",seed=42,best_epoch=20,formal_eligible=True,test_evaluated=False,
        best_checkpoint_sha256=sha(folder/"best.pt"),latest_checkpoint_sha256=sha(folder/"latest.pt"),
        history_sha256=sha(folder/"epoch_log.json"),execution_manifest_sha256=sha(path),
        **{name:history[19]["valid"][name] for name in ("mAP50_95","AP50","AP75","Precision","Recall")})
    (folder/"result.json").write_text(json.dumps(result))
    provenance=dict(execution_manifest_path=str(path),training_run_dir=str(folder))
    return folder,state,provenance


def test_complete_selected_checkpoint_can_precede_epoch80(complete_fixture):
    folder,state,provenance=complete_fixture
    assert verify_completed_training(state,folder/"best.pt",provenance)["formal_result_complete"] is True


def test_running_status_is_rejected_even_if_checkpoint_self_reports_formal(complete_fixture):
    folder,state,provenance=complete_fixture
    path=folder/"status.json";status=json.loads(path.read_text());status["status"]="RUNNING"
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError,match="not a matching complete"):
        verify_completed_training(state,folder/"best.pt",provenance)


def test_untrained_counters_are_rejected_even_with_complete_claims(complete_fixture):
    folder,state,provenance=complete_fixture
    state["model_state_dict"]["fixture.num_batches_tracked"].zero_()
    torch.save(state,folder/"best.pt")
    result=json.loads((folder/"result.json").read_text())
    result["best_checkpoint_sha256"]=sha(folder/"best.pt")
    (folder/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError,match="BatchNorm counters"):
        verify_completed_training(state,folder/"best.pt",provenance)


def test_checkpoint_history_must_match_selected_prefix(complete_fixture):
    folder,state,provenance=complete_fixture
    state["history"]=state["history"][:-1]
    with pytest.raises(ValueError,match="identity mismatch"):
        verify_completed_training(state,folder/"best.pt",provenance)
