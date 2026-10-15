"""Recovery must refuse unrelated edits and altered committed evidence."""
import copy

import numpy as np
import pytest
import torch

from tools.sonar_fcos_pilot import recovery
from tools.sonar_fcos_pilot.common import sha256_file, write_json
from tools.sonar_fcos_pilot.recovery import (
    NEW_SOURCE, PARENT, _verify_migration_files, state_digest,
    verify_config_delta, verify_source_delta,
)


def original_config():
    return {"output_dir": "original/output", "checkpoint_root": "original/checkpoints",
            "epochs": 30, "max_formal_runs": 14, "learning_rate": .0025,
            "amp": True, "amp_initial_scale": 1024., "seeds": [42, 43, 44]}


def new_config():
    value = original_config()
    value.update(output_dir="recovery/output", checkpoint_root="recovery/checkpoints")
    value["recovery_parent"] = {"root": PARENT, "source_freeze_sha256": "a" * 64,
        "config_sha256": "b" * 64, "last_checkpoint_sha256": "c" * 64,
        "completed_epochs": 7, "resumed_run_id": "AAA_seed42", "new_candidate_run": False}
    return value


def test_only_paths_and_provenance_may_change():
    assert verify_config_delta(original_config(), new_config()) == ["checkpoint_root", "output_dir", "recovery_parent"]


@pytest.mark.parametrize("key,value", [("epochs", 31), ("learning_rate", .001),
    ("amp", False), ("amp_initial_scale", 512.), ("seeds", [42, 44, 45]), ("max_formal_runs", 15)])
def test_refuse_any_training_parameter_change(key, value):
    config = new_config(); config[key] = value
    with pytest.raises(RuntimeError, match="Only isolated"):
        verify_config_delta(original_config(), config)


def test_refuse_unrelated_model_or_data_change():
    original = {"tools/sonar_fcos_pilot/train.py": "old", "tools/sonar_fcos_pilot/model.py": "model",
                "tools/sonar_fcos_pilot/data.py": "data", "tools/sonar_fcos_pilot/README.md": "old-doc"}
    current = {**original, "tools/sonar_fcos_pilot/train.py": "repair", **{name: "new" for name in NEW_SOURCE}}
    verify_source_delta(original, current)
    for name in ["model.py", "data.py"]:
        damaged = {**current, "tools/sonar_fcos_pilot/" + name: "modified"}
        with pytest.raises(RuntimeError, match="Unrelated"):
            verify_source_delta(original, damaged)


def test_refuse_extra_executable_source():
    original = {"tools/sonar_fcos_pilot/train.py": "old"}
    current = {**original, **{name: "new" for name in NEW_SOURCE}, "tools/sonar_fcos_pilot/secret_extra.py": "x"}
    with pytest.raises(RuntimeError, match="Unexpected"):
        verify_source_delta(original, current)


def test_state_digest_preserves_nested_state_and_detects_tensor_or_rng_change():
    state = {"model": {"weight": torch.tensor([1., 2.])}, "optimizer": {0: {"momentum_buffer": torch.tensor([.4])}},
             "scaler": {"scale": 1024.}, "rng": {"numpy": ("MT19937", np.array([1, 2], dtype=np.uint32)),
             "torch": torch.tensor([5, 6], dtype=torch.uint8)}, "history": [{"epoch": 1, "validation_map": 32.19}]}
    digest = state_digest(state)
    assert state_digest(copy.deepcopy(state)) == digest
    changed = copy.deepcopy(state); changed["model"]["weight"][0] = 9
    assert state_digest(changed) != digest
    changed = copy.deepcopy(state); changed["rng"]["torch"][0] = 7
    assert state_digest(changed) != digest


def test_migration_refuses_parent_or_copied_artifact_tamper(tmp_path):
    report = {"artifacts": []}
    for name in ["parent_config", "recovery_config", "data_manifest", "acknowledgment"]:
        path = tmp_path / (name + ".json"); path.write_text(name)
        path_key = "manifest_path" if name == "data_manifest" else name + "_path"
        report[path_key] = str(path); report[name + "_sha256"] = sha256_file(path)
    original = tmp_path / "parent.pt"; original.write_bytes(b"parent-state")
    cloned = tmp_path / "cloned.pt"; cloned.write_bytes(b"cloned-state")
    report["artifacts"] = [{"parent_path": str(original), "parent_sha256": sha256_file(original),
                            "recovery_path": str(cloned), "recovery_sha256": sha256_file(cloned)}]
    _verify_migration_files(report)
    cloned.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Migration artifact changed"):
        _verify_migration_files(report)
    cloned.write_bytes(b"cloned-state"); original.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Migration artifact changed"):
        _verify_migration_files(report)


def test_launch_refuses_source_freeze_changed_after_independent_review(monkeypatch, tmp_path):
    root = tmp_path / "recovery"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    parent = tmp_path / "parent"
    marker = root / "source_freeze.json"
    marker.write_text("changed-source-marker", encoding="utf-8")
    check = {key: "reviewed-value" for key in recovery.REVIEW_KEYS}
    check["status"] = "RECOVERY_CHECK_PASS"
    write_json(evidence / "check.json", check)
    review = {**check, "status": "PASS", "check_sha256": sha256_file(evidence / "check.json")}
    write_json(evidence / "independent_doc_review.json", review)
    monkeypatch.setattr(recovery, "_layout", lambda: (root, parent, evidence))
    with pytest.raises(RuntimeError, match="Reviewed recovery source freeze changed"):
        recovery.remote_launch()
    assert not (evidence / "formal_launch.claim").exists()
    assert not (evidence / "formal_launch.json").exists()
