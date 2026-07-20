from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.benchmarks.sure import (
    SureAuthorRecipeConfig,
    apply_author_cosine_classifier,
    indexed_training_loader,
    load_sure_author_components,
)
from hwnas_fpga.models.backbones import SimpleCNN
from scripts.run_sure_s_a_stage1_guarded import validate_sure_stage1_authorization


REPO_ROOT = Path(__file__).resolve().parents[1]
SURE_COMMIT = "5ce0193bc93e73b1c7f1f53aeda8854e997011e2"


def test_sure_author_components_are_loaded_from_pinned_isolated_checkout():
    components = load_sure_author_components(
        REPO_ROOT / "reference/_local/SURE", pinned_commit=SURE_COMMIT
    )
    assert components.commit == SURE_COMMIT
    assert set(components.source_hashes) == {
        "train.py",
        "utils/sam.py",
        "model/classifier.py",
        "run/CIFAR10/wideresnet.sh",
    }
    assert all(len(value) == 64 for value in components.source_hashes.values())


def test_sure_indexed_loader_and_author_cosine_head():
    loader = DataLoader(
        TensorDataset(torch.randn(5, 1, 32, 32), torch.arange(5) % 2),
        batch_size=2,
        shuffle=False,
    )
    indexed = indexed_training_loader(loader)
    _images, _targets, indices = next(iter(indexed))
    assert indices.tolist() == [0, 1]

    components = load_sure_author_components(
        REPO_ROOT / "reference/_local/SURE", pinned_commit=SURE_COMMIT
    )
    model = SimpleCNN(input_channels=1, num_classes=2)
    metadata = apply_author_cosine_classifier(
        model,
        components=components,
        num_classes=2,
        temperature=8.0,
    )
    assert model(torch.randn(3, 1, 32, 32)).shape == (3, 2)
    assert metadata["head"] == "author_cosine_classifier"


def test_sure_recipe_frozen_defaults_match_author_ours_block():
    recipe = SureAuthorRecipeConfig()
    recipe.validate()
    payload = recipe.to_dict()
    assert payload["mixup_weight"] == 0.5
    assert payload["crl_weight"] == 0.5
    assert payload["mixup_beta"] == 10.0
    assert payload["author_recipe"].startswith("FMFP")


def test_sure_stage1_authorization_template_is_not_executable(tmp_path: Path):
    template = REPO_ROOT / (
        "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/manifests/"
        "sure_s_a_stage1_authorization_template_20260719.json.txt"
    )
    import json

    payload = json.loads(template.read_text(encoding="utf-8"))
    errors = validate_sure_stage1_authorization(payload)
    assert errors
    assert any("status mismatch" in value for value in errors)

    freeze = tmp_path / "source_freeze_manifest.json"
    freeze.write_text("{}", encoding="utf-8")
    payload.update(
        {
            "status": "AUTHORIZED",
            "source_freeze_manifest": str(freeze),
            "source_freeze_manifest_sha256": "a" * 64,
            "source_freeze_archive_sha256": "b" * 64,
        }
    )
    assert validate_sure_stage1_authorization(payload) == []


def test_staged_execution_control_is_not_part_of_experiment_fingerprint():
    import inspect
    import run_eval_protocol

    source = inspect.getsource(run_eval_protocol.main)
    immutable_block = source.split("immutable_config = {", 1)[1].split(
        "run_fingerprint =", 1
    )[0]
    assert "max_new_units" not in immutable_block
    assert '"execution_control"' in source


def test_sure_stage1_audit_archives_runtime_measurement():
    import inspect
    from scripts import run_sure_s_a_stage1_guarded

    source = inspect.getsource(run_sure_s_a_stage1_guarded.audit_one_unit)
    assert '"runtime_measurement"' in source
