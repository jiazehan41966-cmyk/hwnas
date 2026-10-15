"""Synthetic overflow/replay checks; never run a formal sonar training job."""
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from tools.sonar_fcos_pilot import common, numerics


class MatrixModel(nn.Module):
    def __init__(self, factor=128., stochastic=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1)))
        self.factor, self.stochastic = factor, stochastic
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64))
        self.values = []

    def forward(self, images, targets):
        self.forward_count.add_(1)
        factor = self.factor
        if self.stochastic:
            factor *= .5 + .2 * (random.random() + float(np.random.rand()) + float(torch.rand(())))
        self.values.append(factor)
        return {"loss": (self.weight @ (images[0] * factor)).sum()}


def batch(image_id):
    return [torch.ones((1, 1))], [{"boxes": torch.zeros((1, 4)),
                                  "labels": torch.ones(1, dtype=torch.int64),
                                  "image_id": torch.tensor([image_id])}]


def execute(scale, *, sizes=(4,), stochastic=False, factor=128., max_retries=8, growth_tracker=0):
    common.seed_everything(31415)
    model = MatrixModel(stochastic=stochastic, factor=factor)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=scale, growth_interval=2000)
    state = scaler.state_dict()
    state["_growth_tracker"] = growth_tracker
    scaler.load_state_dict(state)
    optimizer_calls, events, overflow_proofs, reports = [], [], [], []
    optimizer.register_step_post_hook(lambda *args: optimizer_calls.append(True))

    def callback(event):
        events.append(event)
        if event["event"] == "AMP_GRADIENT_OVERFLOW":
            overflow_proofs.append((torch.equal(model.weight, before_weight),
                                    optimizer.state_dict() == before_optimizer,
                                    len(optimizer_calls) == before_calls))

    step = 0
    for index, size in enumerate(sizes, 1):
        before_weight = model.weight.detach().clone()
        before_optimizer = optimizer.state_dict()
        before_calls = len(optimizer_calls)
        group = numerics.AccumulationGroup(model, optimizer, scaler, "cpu", size, epoch=8,
                                           group_index=index, use_amp=True, event_callback=callback,
                                           max_retries=max_retries)
        for _ in range(size):
            images, targets = batch(step + 100)
            # Simulate stochastic loading between model forwards. Replay must
            # reuse the recorded pre-forward state rather than loading again.
            random.random(), np.random.rand(), torch.rand(())
            assert math.isfinite(group.backward(images, targets, step=step, learning_rate=.001))
            step += 1
        reports.append(group.finish())
    continuation = (random.random(), float(np.random.rand()), torch.rand(4))
    return {"model": model, "optimizer": optimizer, "scaler": scaler, "events": events,
            "reports": reports, "optimizer_calls": optimizer_calls,
            "proofs": overflow_proofs, "continuation": continuation}


def test_real_scaled_half_gradient_overflow_replays_identical_inputs_and_one_update():
    recovered = execute(4096., growth_tracker=37)
    reference = execute(128.)
    assert recovered["reports"][0]["retries"] == 2
    assert len(recovered["optimizer_calls"]) == 1
    assert all(all(proof) for proof in recovered["proofs"])
    assert torch.equal(recovered["model"].weight, reference["model"].weight)
    assert torch.equal(recovered["optimizer"].state[recovered["model"].weight]["momentum_buffer"],
                       reference["optimizer"].state[reference["model"].weight]["momentum_buffer"])
    assert recovered["scaler"].state_dict()["scale"] == 1024.
    assert recovered["scaler"].state_dict()["_growth_tracker"] == 1
    overflows = [event for event in recovered["events"] if event["event"] == "AMP_GRADIENT_OVERFLOW"]
    assert [event["failed_scale"] for event in overflows] == [4096., 2048.]
    assert [event["reduced_scale"] for event in overflows] == [2048., 1024.]
    assert overflows[0]["epoch"] == 8
    assert overflows[0]["image_ids"] == [100, 101, 102, 103]
    assert overflows[0]["steps"] == [1, 2, 3, 4]


def test_replay_restores_buffers_and_model_rng_without_rereading_stochastic_data():
    recovered = execute(4096., stochastic=True)
    reference = execute(128., stochastic=True)
    assert recovered["reports"][0]["retries"] >= 1
    assert int(recovered["model"].forward_count) == int(reference["model"].forward_count) == 4
    assert recovered["model"].values[:4] == reference["model"].values
    assert recovered["model"].values[-4:] == reference["model"].values
    assert torch.equal(recovered["model"].weight, reference["model"].weight)
    assert recovered["continuation"][:2] == reference["continuation"][:2]
    assert torch.equal(recovered["continuation"][2], reference["continuation"][2])


def test_partial_last_group_still_applies_exactly_ceil_n_over_four_updates():
    recovered = execute(4096., sizes=(4, 4, 1))
    reference = execute(128., sizes=(4, 4, 1))
    assert len(recovered["optimizer_calls"]) == 3
    assert [report["optimizer_steps"] for report in recovered["reports"]] == [1, 1, 1]
    assert int(recovered["model"].forward_count) == 9
    assert torch.equal(recovered["model"].weight, reference["model"].weight)


def test_finite_first_attempt_does_not_restore_rng_or_buffers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The normal finite path must not restore state.")

    monkeypatch.setattr(numerics, "restore_rng", forbidden)
    monkeypatch.setattr(numerics.AccumulationGroup, "_restore_buffers", forbidden)
    result = execute(128., stochastic=True)
    assert result["reports"][0]["retries"] == 0
    assert not result["events"]
    assert len(result["optimizer_calls"]) == 1


def test_retry_exhaustion_diagnoses_float32_without_applying_parameter_update():
    model = MatrixModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=4096.)
    events, calls = [], []
    optimizer.register_step_post_hook(lambda *args: calls.append(True))
    group = numerics.AccumulationGroup(model, optimizer, scaler, "cpu", 4, epoch=8,
                                       group_index=2, use_amp=True, event_callback=events.append,
                                       max_retries=1)
    for step in range(4):
        group.backward(*batch(step), step=step, learning_rate=.001)
    with pytest.raises(RuntimeError, match="retry limit.*float32 diagnostic=finite"):
        group.finish()
    assert not calls
    assert float(model.weight) == 1.
    assert not optimizer.state
    assert model.weight.grad is None
    assert int(model.forward_count) == 0
    assert events[-1]["event"] == "AMP_RETRY_LIMIT"
    assert events[-1]["float32_diagnostic"] == "finite"


def test_intrinsically_nonfinite_gradient_is_identified_by_float32_diagnostic():
    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * 0

        @staticmethod
        def backward(ctx, grad):
            return grad * float("inf")

    class BadModel(MatrixModel):
        def forward(self, images, targets):
            self.forward_count.add_(1)
            return {"loss": BadGradient.apply(self.weight).sum()}

    model = BadModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=4096.)
    group = numerics.AccumulationGroup(model, optimizer, scaler, "cpu", 1, epoch=8,
                                       group_index=1, use_amp=True, max_retries=0)
    group.backward(*batch(5), step=0, learning_rate=.001)
    with pytest.raises(RuntimeError, match="float32 diagnostic=nonfinite_gradients"):
        group.finish()
    assert float(model.weight) == 1.
    assert not optimizer.state


def test_loaded_scaler_growth_state_continues_and_backoff_resets_tracker():
    old = torch.amp.GradScaler("cpu", init_scale=1024.)
    old_state = old.state_dict()
    old_state.update({"scale": 65536., "_growth_tracker": 37})
    resumed = execute(old_state["scale"], growth_tracker=old_state["_growth_tracker"])
    assert resumed["reports"][0]["retries"] > 0
    assert resumed["scaler"].state_dict()["_growth_tracker"] == 1
    assert len(resumed["optimizer_calls"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A real CUDA kernel witness is required")
def test_cuda_fp16_kernel_witness(tmp_path):
    result = numerics.recovery_witness(tmp_path)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["optimizer_updates"] == 3
    assert common.read_json(tmp_path / "amp_recovery_witness.json") == result
