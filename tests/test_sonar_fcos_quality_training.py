"""CPU-only bounded replay witnesses; no research data, downloads, or GPU work."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_fcos_pilot import numerics
from tools.sonar_fcos_pilot.common import read_json, sha256_file, write_json
from tools.sonar_fcos_quality import train


class TinyDataset(Dataset):
    def __init__(self, path, split, image_size):
        self.split = split
        self.records = [{"image_id": index + 1} for index in range(3 if split == "train" else 2)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # All three RNGs affect updates, making the resume parity test sensitive
        # to missing Python, NumPy, or Torch checkpoint state restoration.
        image = torch.tensor([index + random.random() + float(np.random.random())]) + torch.rand(1)
        target = {"boxes": torch.tensor([[0., 0., 1., 1.]]),
                  "labels": torch.tensor([1]), "image_id": torch.tensor([index + 1])}
        return image, target


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.register_buffer("training_calls", torch.zeros((), dtype=torch.long))

    def forward(self, images, targets):
        self.training_calls.add_(1)
        loss = sum(((self.weight * image + torch.rand(1) * .1) - .5).square().sum() for image in images)
        return {"tiny_loss": loss}


def cpu_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(), "cuda": []}


def restore_cpu_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    assert state["cuda"] == []


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    source = {"fixture": "tiny_cpu_only", "pretrained_checkpoint_sha256": "fixture-hash"}
    environment = {"device": "cpu", "fixture": True}
    input_checks = []
    manifest_path = tmp_path / "manifest.json"
    manifest = {"classes": ["object"], "samples": []}
    write_json(manifest_path, manifest)
    monkeypatch.setattr(train, "require_manifest", lambda config: (manifest_path, manifest))
    monkeypatch.setattr(train, "source_binding", lambda config: source.copy())
    monkeypatch.setattr(train, "environment", lambda: environment.copy())
    monkeypatch.setattr(train, "verify_development_inputs", lambda value: input_checks.append(value))
    monkeypatch.setattr(train, "SonarDataset", TinyDataset)
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(train, "model_contract", lambda model: {"tiny_cpu_fixture": True,
                        "pretrained": {"checkpoint_sha256": "fixture-hash"}})
    monkeypatch.setattr(train, "seed_everything", seed_cpu)
    for module in (train, numerics):
        monkeypatch.setattr(module, "rng_state", cpu_rng_state)
        monkeypatch.setattr(module, "restore_rng", restore_cpu_rng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("The bounded CPU witness must not call a CUDA runtime API")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden_cuda)

    def fake_predict(model, dataset, config, **kwargs):
        model.eval()
        return [{"image_id": record["image_id"], "weight": float(model.weight.detach()),
                 "training_calls": int(model.training_calls)} for record in dataset.records], None

    def fake_evaluate(records, predictions, classes):
        value = 1.0 / (1.0 + abs(predictions[0]["weight"]))
        return {"map": value, "ap75": value * .8, "ap85": value * .6, "wrong_class_fp": 0}

    monkeypatch.setattr(train, "predict", fake_predict)
    monkeypatch.setattr(train, "evaluate_quality_predictions", fake_evaluate)

    def create(name):
        output = tmp_path / name
        config = {"variants": ["AAA", "QFL_DFL"], "seeds": [42, 43, 44],
            "output_dir": str(output), "checkpoint_root": str(output / "checkpoints"),
            "torch_threads": 1, "device": "cpu", "image_size": 640,
            "pretrained": False, "weight_cache": str(tmp_path / "unused"),
            "learning_rate": .01, "momentum": .9, "weight_decay": .0001,
            "amp": False, "amp_initial_scale": 128., "gradient_accumulation": 2,
            "batch_size": 1, "workers": 0, "learning_rate_drop_epochs": [20, 25],
            "warmup_epochs": 1, "warmup_start_factor": .1}
        execution_path = output / "execution_manifest.json"
        write_json(execution_path, {"source_binding": source, "environment": environment})
        write_json(output / "smoke_gate.json", {"status": "PASS", "source_binding": source,
            "environment": environment, "execution_manifest_sha256": sha256_file(execution_path)})
        return config

    yield create, input_checks
    torch.set_num_threads(previous_threads)


def checkpoint_path(config):
    return Path(config["checkpoint_root"]) / "QFL_DFL_seed42" / "last.pt"


def load_checkpoint(config):
    return torch.load(checkpoint_path(config), map_location="cpu", weights_only=False)


def assert_exact_nested(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact_nested(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_exact_nested(first, second)
    else:
        assert actual == expected


def test_two_epochs_equal_one_epoch_plus_resume_with_actual_optimizer_and_rng(tiny_campaign):
    create, input_checks = tiny_campaign
    uninterrupted, interrupted = create("uninterrupted"), create("interrupted")
    assert train.train_one(uninterrupted, "QFL_DFL", 42, epoch_budget=2) == {"status": "PARTIAL", "completed_epochs": 2}
    assert train.train_one(interrupted, "QFL_DFL", 42, epoch_budget=1) == {"status": "PARTIAL", "completed_epochs": 1}
    before = load_checkpoint(interrupted)
    assert before["model"]["training_calls"] == 3
    # Deliberately perturb all RNGs between calls; the checkpoint must override.
    random.random()
    np.random.random(5)
    torch.rand(7)
    assert train.train_one(interrupted, "QFL_DFL", 42, epoch_budget=1) == {"status": "PARTIAL", "completed_epochs": 2}
    actual, expected = load_checkpoint(interrupted), load_checkpoint(uninterrupted)
    for key in ["model", "optimizer", "scaler", "rng", "best_map", "best_epoch", "epoch", "binding"]:
        assert_exact_nested(actual[key], expected[key])
    for checkpoint in (actual, expected):
        for epoch in checkpoint["history"]:
            epoch.pop("seconds")
    assert_exact_nested(actual["history"], expected["history"])
    assert actual["model"]["training_calls"] == 6
    assert [row["optimizer_steps"] for row in actual["history"]] == [2, 2]
    assert len(input_checks) == 2  # Only fresh runs reread all development inputs.
    receipt = read_json(Path(interrupted["output_dir"]) / "runs/QFL_DFL_seed42/run_manifest.json")
    assert receipt["epochs"] == 30
    assert receipt["test_evaluated"] is False


def test_pause_before_next_epoch_preserves_last_checkpoint(tiny_campaign):
    create, _ = tiny_campaign
    config = create("paused")
    train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    before = sha256_file(checkpoint_path(config))
    pause = Path(config["output_dir"]) / "pause.request"
    pause.write_text("primary owns GPU", encoding="utf-8")
    with pytest.raises(train.YieldToPrimary):
        train.train_one(config, "QFL_DFL", 42, epoch_budget=1, pause_file=pause)
    assert sha256_file(checkpoint_path(config)) == before
    status = read_json(Path(config["output_dir"]) / "runs/QFL_DFL_seed42/status.json")
    assert status["status"] == "PAUSED_FOR_PRIMARY"
    assert status["completed_epochs"] == 1


def test_failed_or_stale_smoke_gate_rejects_before_training(tiny_campaign):
    create, input_checks = tiny_campaign
    for name, field, value in [("failed", "status", "FAIL"),
                                ("stale", "execution_manifest_sha256", "not-current")]:
        config = create(name)
        path = Path(config["output_dir"]) / "smoke_gate.json"
        gate = read_json(path)
        gate[field] = value
        write_json(path, gate)
        with pytest.raises(RuntimeError, match="smoke"):
            train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
        assert not checkpoint_path(config).exists()
    assert input_checks == []


def test_resume_rejects_checkpoint_from_different_binding(tiny_campaign):
    create, _ = tiny_campaign
    config = create("wrong_binding")
    train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    saved = load_checkpoint(config)
    saved["binding"]["seed"] = 43
    torch.save(saved, checkpoint_path(config))
    before = sha256_file(checkpoint_path(config))
    with pytest.raises(RuntimeError, match="Resume checkpoint binding changed"):
        train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    assert sha256_file(checkpoint_path(config)) == before
