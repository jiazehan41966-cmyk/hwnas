"""Synthetic CPU checks only: no sonar data, downloads or formal model training."""
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tools.sonar_fcos_pilot import common, train


class TinyDataset:
    def __init__(self, manifest_path, split, image_size):
        assert split in {"train", "val"}, "The training entrypoint must never request test data."
        self.split = split
        self.records = [{"image_id": 1, "width": 100, "height": 100,
                         "boxes": [[0, 0, 10, 10]], "labels": [1]}]

    def __len__(self):
        return 5 if self.split == "train" else 1

    def __getitem__(self, index):
        return torch.tensor([float(index + 1)]), {"boxes": torch.tensor([[0., 0., 10., 10.]]),
                                                "labels": torch.tensor([1])}


class TinyModel(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.stochastic = stochastic
        self.samples = []

    def forward(self, images, targets):
        assert self.training
        value = torch.stack(images).mean()
        if self.stochastic:
            value = value * (random.random() + float(np.random.rand()) + torch.rand(()))
        self.samples.append(float(value))
        return {"loss": self.weight * value}


def setup_synthetic(monkeypatch, tmp_path, stochastic=False):
    manifest_path = tmp_path / "manifest.json"
    common.write_json(manifest_path, {"fixture": "synthetic_only"})
    manifest = {"classes": ["fixture"], "small_area_threshold": 0.1, "samples": []}
    instances = []

    def build(*args, **kwargs):
        model = TinyModel(stochastic=stochastic)
        instances.append(model)
        return model

    monkeypatch.setattr(train, "require_manifest", lambda config: (manifest_path, manifest))
    monkeypatch.setattr(train, "SonarDataset", TinyDataset)
    monkeypatch.setattr(train, "build_model", build)
    monkeypatch.setattr(train, "model_contract", lambda model: {"kind": "synthetic_cpu_fixture"})
    monkeypatch.setattr(train, "environment", lambda: {"fixture": "synthetic_cpu"})
    monkeypatch.setattr(train, "source_binding", lambda config: {"config_sha256": "fixture", "sources": {}})
    monkeypatch.setattr(train, "predict", lambda model, dataset, config:
                        [{"image_id": 1, "weight": float(model.weight.detach())}])
    monkeypatch.setattr(train, "evaluate_predictions", lambda records, predictions, *args:
                        {"map": 50.0 - predictions[0]["weight"], "diagnostics": {"threshold": 0.25}})
    monkeypatch.setattr(train, "calibrate_threshold", lambda *args:
                        {"threshold": 0.5, "selection_set": "validation_only"})
    monkeypatch.setattr(train, "write_error_examples", lambda *args: [])
    config = {
        "output_dir": str(tmp_path / "outputs"), "checkpoint_root": str(tmp_path / "checkpoints"),
        "weight_cache": str(tmp_path / "unused_cache"), "batch_size": 1, "gradient_accumulation": 4,
        "workers": 0, "device": "cpu", "amp": False, "amp_initial_scale": 128.0, "torch_threads": 1,
        "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0,
        "learning_rate_drop_epochs": [], "warmup_epochs": 0, "warmup_start_factor": 0.001,
        "epochs": 2, "image_size": 640, "pretrained": False,
        "combinations": ["AAA", "BBB"], "seeds": [42],
        "max_formal_runs": 14,
        "diagnostic_score_threshold": 0.25, "validation_fp_budget_per_image": 0.1,
    }
    return config, instances


def test_last_partial_accumulation_group_is_scaled_by_actual_steps(monkeypatch, tmp_path):
    config, instances = setup_synthetic(monkeypatch, tmp_path)
    config["epochs"] = 1
    result = train.train_one(config, "AAA", 42)
    samples = instances[-1].samples
    assert len(samples) == 5
    expected = 1.0 - config["learning_rate"] * (sum(samples[:4]) / 4 + samples[4])
    assert float(instances[-1].weight.detach()) == pytest.approx(expected, abs=1e-7)
    history = common.read_json(Path(config["output_dir"]) / "runs" / "AAA_seed42" / "history.json")
    assert history[0]["optimizer_steps"] == 2
    assert result["effective_batch_size"] == 4
    assert result["trained_epochs"] == 1
    assert result["test_used"] is False


def test_checkpoint_resume_replays_python_numpy_and_torch_rng(monkeypatch, tmp_path):
    config, instances = setup_synthetic(monkeypatch, tmp_path, stochastic=True)
    train.train_one(config, "AAA", 42)
    expected_weight = float(instances[-1].weight.detach())
    expected_second_epoch_samples = instances[-1].samples[5:]

    interrupted_config = {**config, "output_dir": str(tmp_path / "interrupted_outputs"),
                          "checkpoint_root": str(tmp_path / "interrupted_checkpoints")}
    original_write_json = train.write_json

    def interrupt_after_first_committed_epoch(path, value):
        original_write_json(path, value)
        if Path(path).name == "history.json" and len(value) == 1:
            raise KeyboardInterrupt("synthetic interruption after completed epoch")

    monkeypatch.setattr(train, "write_json", interrupt_after_first_committed_epoch)
    with pytest.raises(KeyboardInterrupt):
        train.train_one(interrupted_config, "AAA", 42)
    monkeypatch.setattr(train, "write_json", original_write_json)
    result = train.train_one(interrupted_config, "AAA", 42)
    assert result["trained_epochs"] == 2
    assert len(instances[-1].samples) == 5
    assert instances[-1].samples == pytest.approx(expected_second_epoch_samples, abs=0)
    assert float(instances[-1].weight.detach()) == pytest.approx(expected_weight, abs=0)


def test_checkpoint_rng_roundtrip_on_cpu():
    common.seed_everything(123)
    state = train._rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    train._restore_rng(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_checkpoint_from_other_combo_is_rejected_even_with_same_tensor_shapes(monkeypatch, tmp_path):
    config, _ = setup_synthetic(monkeypatch, tmp_path)
    config["epochs"] = 1
    train.train_one(config, "AAA", 42)
    source = Path(config["checkpoint_root"]) / "AAA_seed42" / "last.pt"
    destination = Path(config["checkpoint_root"]) / "BBB_seed42" / "last.pt"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    with pytest.raises(RuntimeError, match="does not belong"):
        train.train_one(config, "BBB", 42)


@pytest.mark.parametrize("artifact_name", ["checkpoint", "predictions", "metrics"])
def test_completed_result_reuse_rejects_modified_best_artifacts(monkeypatch, tmp_path, artifact_name):
    config, instances = setup_synthetic(monkeypatch, tmp_path)
    config["epochs"] = 1
    result = train.train_one(config, "AAA", 42)
    artifact_path = Path(result["best_artifacts"][artifact_name]["path"])
    with artifact_path.open("ab") as stream:
        stream.write(b"\nmodified fixture")
    with pytest.raises(RuntimeError, match="Completed run artifact changed"):
        train.train_one(config, "AAA", 42)
    assert len(instances) == 1  # Cached-result validation precedes rebuilding.


def test_fourteen_existing_run_receipts_reject_a_fifteenth_run(monkeypatch, tmp_path):
    config, instances = setup_synthetic(monkeypatch, tmp_path)
    runs_dir = Path(config["output_dir"]) / "runs"
    for index in range(14):
        common.write_json(runs_dir / f"existing_{index}" / "run_manifest.json",
                          {"fixture": "synthetic_existing_receipt"})
    with pytest.raises(RuntimeError, match="fourteen-run budget"):
        train.train_one(config, "AAA", 42)
    assert not instances[-1].samples  # No optimizer updates before rejection.
    assert not (runs_dir / "AAA_seed42" / "run_manifest.json").exists()


def test_nonfinite_gradient_is_not_reported_as_completed_optimizer_step(monkeypatch, tmp_path):
    config, _ = setup_synthetic(monkeypatch, tmp_path)

    class InfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            return weight * 0  # Finite loss, deliberately nonfinite derivative.

        @staticmethod
        def backward(ctx, gradient):
            return gradient * float("inf")

    class InvalidModel(TinyModel):
        def forward(self, images, targets):
            return {"loss": InfiniteGradient.apply(self.weight)}

    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: InvalidModel())
    with pytest.raises(RuntimeError, match="Nonfinite gradients"):
        train.train_one(config, "AAA", 42)
    status = common.read_json(Path(config["output_dir"]) / "runs" / "AAA_seed42" / "status.json")
    assert status["status"] == "ERROR"
    assert not status["resume_from_last_complete_epoch"]
    assert not list(Path(config["checkpoint_root"]).rglob("last.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_rng_roundtrip_on_cuda():
    common.seed_everything(123)
    state = train._rng_state()
    expected = torch.rand(4, device="cuda")
    train._restore_rng(state)
    actual = torch.rand(4, device="cuda")
    assert torch.equal(actual, expected)


def test_manifest_blocks_invalid_annotations_even_when_split_is_pass(tmp_path):
    common.write_json(tmp_path / "data_audit" / "manifest.json",
                      {"split_status": "PASS", "audit": {"data_integrity_pass": False},
                       "small_area_threshold": 0.1})
    with pytest.raises(RuntimeError, match="正式训练未启动"):
        common.require_manifest({"output_dir": str(tmp_path)})


def test_formal_config_uses_percentage_point_map_gain_and_frozen_budget():
    config = json.loads(common.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["minimum_map_gain"] == 1.0
    assert config["epochs"] == 30
    assert config["batch_size"] * config["gradient_accumulation"] == 4
    assert config["seeds"] == [42, 43, 44]
    assert config["max_formal_runs"] == 14
    assert config["diagnostic_score_threshold"] == 0.25
