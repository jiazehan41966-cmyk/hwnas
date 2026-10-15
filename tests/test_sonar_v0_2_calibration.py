"""Regression checks for node identity, multi-batch accumulation and cleanup."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.sonar_v0_2_ptq import (
    CALIBRATION_CONTRACT, observe_v0_2_extrema, prepare_v0_2_ptq,
)
from hwnas_fpga.models.sonar_candidate_v0_2 import build_sonar_candidate_v0_2


class ToyNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(1, 1, 1, bias=False) for _ in range(3)])
        for conv in self.lateral_convs:
            nn.init.ones_(conv.weight)
        self.output_convs = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, features):
        fused = [conv(x) for conv, x in zip(self.lateral_convs, features)]
        for index in range(1, -1, -1):
            fused[index] = fused[index] + F.interpolate(fused[index+1], size=fused[index].shape[-2:], mode="nearest")
        return tuple(fused)


class ToyModel(nn.Module):
    levels = ("P2", "P3", "P4")

    def __init__(self):
        super().__init__()
        self.neck = ToyNeck()

    def forward(self, images):
        features = [F.avg_pool2d(images[:, index:index+1], 2**index) for index in range(3)]
        return self.neck(features)


def batch(values):
    return torch.tensor(values, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 8, 8).clone()


def assert_clean(model):
    assert "forward" not in model.neck.__dict__
    for conv in model.neck.lateral_convs:
        assert not conv._forward_hooks and not conv._forward_pre_hooks


def test_fusion_stats_are_accumulated_by_node_across_all_batches():
    model = ToyModel().eval()
    inputs = [batch([1, 10, 100]), batch([7, 3, 4])]
    before = model(inputs[0])
    observed = observe_v0_2_extrema(model, inputs, device="cpu")
    assert observed["calibration_contract"] == CALIBRATION_CONTRACT
    assert observed["calibration_batches"] == observed["calibration_images"] == 2
    assert observed["adds"] == [
        dict(node_id="neck.fuse.0", level="P2", left_max_abs=7., right_max_abs=110., output_max_abs=111., observed_batches=2),
        dict(node_id="neck.fuse.1", level="P3", left_max_abs=10., right_max_abs=100., output_max_abs=110., observed_batches=2),
    ]
    for index, expected in enumerate((7., 10., 100.)):
        assert observed["convs"][f"neck.lateral_convs.{index}"] == dict(input_max_abs=expected, output_max_abs=expected)
    for old, new in zip(before, model(inputs[0])):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert_clean(model)


@pytest.mark.parametrize("inputs, error", [
    ([], "empty calibration iterable"),
    ([torch.zeros(3, 8, 8)], "NCHW"),
    ([torch.zeros(0, 3, 8, 8)], "empty calibration batch"),
    ([batch([1, float("nan"), 3])], "nonfinite"),
    ([batch([1, 2, float("inf")])], "nonfinite"),
])
def test_failed_observation_restores_hooks_and_forward(inputs, error):
    model = ToyModel().eval()
    with pytest.raises(ValueError, match=error):
        observe_v0_2_extrema(model, inputs, device="cpu")
    assert_clean(model)
    assert all(torch.isfinite(x).all() for x in model(batch([1, 2, 3])))


def test_preexisting_instance_forward_is_restored_even_on_error():
    model = ToyModel().eval()
    original = model.neck.forward
    model.neck.forward = original
    with pytest.raises(ValueError, match="empty"):
        observe_v0_2_extrema(model, [], device="cpu")
    assert model.neck.forward is original
    for conv in model.neck.lateral_convs:
        assert not conv._forward_hooks and not conv._forward_pre_hooks


def test_training_mode_is_rejected_without_mutation():
    model = ToyModel()
    with pytest.raises(ValueError, match="eval mode"):
        observe_v0_2_extrema(model, [batch([1, 2, 3])], device="cpu")
    assert_clean(model)


def test_real_r1_prepare_folds_independent_copy_and_binds_fusion_ids():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        model = build_sonar_candidate_v0_2(architecture="R1").eval()
        images = torch.randn(1, 1, 640, 640)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        folded, quantized, extrema, profile = prepare_v0_2_ptq(model, [images], device="cpu")
        assert profile["calibration_contract"] == CALIBRATION_CONTRACT
        assert profile["fpn_add_node_ids"] == ["neck.fuse.0", "neck.fuse.1"]
        assert [row["node_id"] for row in profile["fpn_adds"]] == profile["fpn_add_node_ids"]
        assert len(profile["fpn_adds"]) == len(extrema["adds"]) == 2
        assert profile["calibration"] == dict(calibration_batches=1, calibration_images=1)
        assert profile["parity_ready"] is False
        assert profile["hardware_cost_status"] == "V0_2_NEW_ENGINE_NEEDS_MEASUREMENT"
        assert not any(isinstance(module, nn.BatchNorm2d) for module in folded.modules())
        assert any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
        for key, value in original.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            for expected, actual in zip(model(images)["predictions"], folded(images)["predictions"]):
                torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)
            predictions = quantized(images)["predictions"]
        assert all(torch.isfinite(value).all() for value in predictions)
    finally:
        torch.set_num_threads(old_threads)
