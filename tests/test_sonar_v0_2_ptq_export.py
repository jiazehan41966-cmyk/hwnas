"""V0.2 export identity, graph coverage and strict provenance boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hwnas_fpga.deploy.sonar_v0_2_ptq import prepare_v0_2_ptq
from hwnas_fpga.models.sonar_candidate_v0_2 import build_sonar_candidate_v0_2
from tools.detector_v0_2.ptq_export import (
    DEPLOYMENT, export_ptq_package, profile_digest, restore_v0_2_ptq,
    sha256, state_digest, validate_v0_2_profile,
)
from tools.sonar_deploy.ptq_export import IntegerGraphModel


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def prepared():
    torch.manual_seed(42)
    model = build_sonar_candidate_v0_2(architecture="R1").eval()
    images = [torch.full((1, 1, 640, 640), value) for value in (.5, .25)]
    _, _, _, profile = prepare_v0_2_ptq(model, images, device="cpu")
    return model, profile


def smoke_provenance(profile):
    return dict(purpose="untrained_contract_smoke", profile_sha256=profile_digest(profile),
                parameter_preparation_device="cpu")


@pytest.mark.parametrize("prototype,count,bytes_count", [("R0", 26, 705600), ("R1", 26, 705600),
                                                       ("R7", 36, 714000), ("R8", 32, 705600)])
def test_graph_consumes_all_DS_and_linear_prediction_records(tmp_path, prototype, count, bytes_count):
    torch.manual_seed(42)
    model = build_sonar_candidate_v0_2(architecture=prototype).eval()
    _, _, _, profile = prepare_v0_2_ptq(model, [torch.ones(1, 1, 640, 640)], device="cpu")
    folder = tmp_path / prototype
    graph = export_ptq_package(model, profile, folder, smoke_provenance(profile))
    convs = [node for node in graph["nodes"] if node["op"] == "conv"]
    assert len(convs) == count
    assert {node["id"] for node in convs} == {record["name"] for record in profile["layers"]}
    assert graph["deployment_version"] == DEPLOYMENT
    assert graph["synthetic_weights"] and not graph["real_weights_bound"]
    assert graph["hardware_layout_binding"] == "NOT_IMPLEMENTED"
    assert graph["runtime_interface"]["status"] == "NOT_IMPLEMENTED"
    assert graph["runtime_interface"]["target_ddr_bits"] == 128
    assert "interface_contract_hash" not in graph
    assert sum(record["real_bytes"] for record in graph["descriptors"]) == bytes_count
    for level, path in graph["outputs"].items():
        node = next(node for node in convs if node["id"] == path)
        assert node["Cout"] == 21 and node["kernel"] == 1 and node["activation"] == "linear"
    assert graph["fpn_calibration"]["node_ids"] == [f"neck.fuse.{i}" for i in range(len(model.levels) - 1)]
    assert graph["fpn_calibration"]["unconsumed_records"] == 0
    IntegerGraphModel(folder, model)
    records = np.fromfile(folder / "layers.bin", dtype="<i4").reshape(-1, 32)
    assert records.shape == (count, 32)


@pytest.mark.parametrize("mutation", ["legacy", "missing", "extra", "reordered", "batch", "level", "nonfinite"])
def test_rejects_misbound_calibration(prepared, mutation):
    model, original = prepared
    profile = deepcopy(original)
    if mutation == "legacy":
        profile.pop("calibration_contract")
    elif mutation == "missing":
        profile["fpn_adds"].pop()
    elif mutation == "extra":
        profile["fpn_adds"].append(deepcopy(profile["fpn_adds"][0]))
    elif mutation == "reordered":
        profile["fpn_adds"].reverse()
    elif mutation == "batch":
        profile["fpn_adds"][0]["observed_batches"] = 1
    elif mutation == "level":
        profile["fpn_adds"][0]["level"] = "P3"
    elif mutation == "nonfinite":
        profile["diagnostic_max_abs"] = float("inf")
    with pytest.raises(ValueError):
        validate_v0_2_profile(model, profile)


def test_numeric_contract_is_still_frozen(prepared):
    model, original = prepared
    profile = deepcopy(original)
    profile["rounding"] = "away_from_zero"
    with pytest.raises(ValueError, match="uniform INT8 contract"):
        restore_v0_2_ptq(model, profile)


def test_training_mode_is_rejected(prepared, tmp_path):
    original, profile = prepared
    model = deepcopy(original).train()
    with pytest.raises(ValueError, match="eval-mode"):
        export_ptq_package(model, profile, tmp_path / "train_mode", smoke_provenance(profile))


@pytest.mark.parametrize("mutation", ["missing_purpose", "profile_hash", "device", "test", "checkpoint_claim"])
def test_rejects_provenance_before_creating_output(prepared, tmp_path, mutation):
    model, profile = prepared
    provenance = smoke_provenance(profile)
    if mutation == "missing_purpose":
        provenance.pop("purpose")
    elif mutation == "profile_hash":
        provenance["profile_sha256"] = "0" * 64
    elif mutation == "device":
        provenance["parameter_preparation_device"] = "cuda"
    elif mutation == "test":
        provenance["test_accessed"] = True
    elif mutation == "checkpoint_claim":
        provenance["checkpoint_sha256"] = "0" * 64
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError):
        export_ptq_package(model, profile, destination, provenance)
    assert not destination.exists()


def test_checkpoint_requires_actual_file_hash_and_identical_state(prepared, tmp_path):
    # Synthetic fixture tests binding mechanics only; it is never used as a
    # trained experiment artifact or reported as trained model evidence.
    model, profile = prepared
    checkpoint = tmp_path / "identity_fixture.pt"
    changed = deepcopy(model.state_dict())
    changed["head.towers.0.0.bias"][0] += 1
    torch.save(dict(model_state_dict=changed), checkpoint)
    provenance = dict(purpose="trained_checkpoint", profile_sha256=profile_digest(profile),
        parameter_preparation_device="cpu", checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint.read_bytes()))
    with pytest.raises(ValueError, match="checkpoint state differs"):
        export_ptq_package(model, profile, tmp_path / "rejected_state", provenance)
    provenance["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="actual checkpoint"):
        export_ptq_package(model, profile, tmp_path / "rejected_hash", provenance)


def test_checkpoint_eligibility_is_checked_even_with_identical_state(prepared, tmp_path):
    model, profile = prepared
    checkpoint = tmp_path / "identity_fixture.pt"
    torch.save(dict(model_state_dict=model.state_dict()), checkpoint)
    provenance = dict(purpose="trained_checkpoint", profile_sha256=profile_digest(profile),
        parameter_preparation_device="cpu", checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint.read_bytes()))
    with pytest.raises(ValueError, match="formal V0.2 training binding"):
        export_ptq_package(model, profile, tmp_path / "rejected_eligibility", provenance)


def test_package_tampering_and_overwrite_are_rejected(prepared, tmp_path):
    model, profile = prepared
    folder = tmp_path / "package"
    export_ptq_package(model, profile, folder, smoke_provenance(profile))
    with pytest.raises(FileExistsError):
        export_ptq_package(model, profile, folder, smoke_provenance(profile))
    payload = bytearray((folder / "operator_weights_axi128.bin").read_bytes())
    payload[0] ^= 1
    (folder / "operator_weights_axi128.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="parameter package SHA"):
        IntegerGraphModel(folder, model)


def test_state_and_profile_digest_are_order_stable(prepared, tmp_path):
    model, profile = prepared
    state = model.state_dict()
    assert state_digest(state) == state_digest(dict(reversed(list(state.items()))))
    assert profile_digest(profile) == profile_digest(dict(reversed(list(profile.items()))))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert profile_digest(path) == sha256(path.read_bytes())
