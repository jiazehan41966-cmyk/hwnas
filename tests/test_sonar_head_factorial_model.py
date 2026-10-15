"""Behavior checks for the isolated detector head factorial adapter."""
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.detector_elastic_pilot.model import build_model as build_e0
from tools.sonar_head_factorial.model import (
    BACKBONES, HEADS, build_model, build_criterion, decode, loss_from_output,
)

BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def assert_equal_state(first, second):
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_e0_legacy_exact_state_and_preserved_cpu_rng(seed):
    before = torch.get_rng_state().clone()
    original = build_e0("E0", seed)
    model = build_model("E0", "obj_softmax", seed)
    assert_equal_state(original.state_dict(), model.state_dict())
    assert torch.equal(before, torch.get_rng_state())


def test_e0_legacy_exact_full_size_output():
    original, model = build_e0("E0", 42).eval(), build_model("E0", "obj_softmax", 42).eval()
    with torch.no_grad():
        images = torch.linspace(0, 1, 640 * 640).reshape(1, 1, 640, 640)
        expected, actual = original(images), model(images)
    for key in expected:
        assert len(expected[key]) == len(actual[key])
        assert all(torch.equal(x, y) for x, y in zip(expected[key], actual[key]))


@pytest.mark.parametrize("backbone", BACKBONES)
def test_trunk_initialization_paired_across_heads(backbone):
    before = torch.get_rng_state().clone()
    legacy, quality = [build_model(backbone, head, 42) for head in HEADS]
    assert_equal_state(legacy.backbone.state_dict(), quality.backbone.state_dict())
    assert_equal_state(legacy.neck.state_dict(), quality.neck.state_dict())
    for key in ("backbone_sha256", "neck_sha256"):
        assert legacy.initialization_identity()[key] == quality.initialization_identity()[key]
    assert legacy.initialization_identity()["head_sha256"] != quality.initialization_identity()["head_sha256"]
    assert torch.equal(before, torch.get_rng_state())
    assert quality.metadata()["quality_package"]["no_auxiliary_branch_or_PGI"]
    assert quality.metadata()["quality_hidden_channels"]["classification"] == quality.feature_channels[0]
    for module in quality.head.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var, torch.ones_like(module.running_var))
            assert int(module.num_batches_tracked) == 0


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("empty", [False, True])
def test_native_head_loss_finite_and_gradients(backbone, head, empty):
    model = build_model(backbone, head, 42).train()
    criterion = build_criterion(model, BASELINE)
    # Synthetic feature tensors exercise the full head, native target adapter,
    # assigner and loss at reduced spatial size; these are not training evidence.
    features = [torch.randn(1, channels, 128 // stride, 128 // stride)
                for channels, stride in zip(model.feature_channels, model.feature_strides)]
    predictions = model.head(features)
    boxes = torch.empty((0, 4)) if empty else torch.tensor([[32., 32., 64., 64.]])
    labels = torch.empty(0, dtype=torch.long) if empty else torch.tensor([1])
    loss, parts = loss_from_output(model, criterion, {"predictions": predictions}, [{"boxes": boxes, "labels": labels}])
    assert torch.isfinite(loss) and loss.requires_grad
    assert all(torch.isfinite(torch.tensor(value)) for value in parts.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.head.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    if head == "quality_aware" and not empty:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in model.head.cv2.parameters())


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("head", HEADS)
def test_full_image_geometry_and_frozen_decode(backbone, head):
    model = build_model(backbone, head, 42).eval()
    criterion = build_criterion(model, BASELINE)
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 640, 640))
        predictions = decode(model, criterion, output, score_threshold=0.,
                             nms_iou_threshold=.5, pre_nms_topk=20, max_detections=10)
    assert [tensor.shape[1] for tensor in output["neck_features"]] == list(model.feature_channels)
    assert [tensor.shape[-1] for tensor in output["neck_features"]] == [640 // stride for stride in model.feature_strides]
    assert len(predictions) == 1
    prediction = predictions[0]
    assert len(prediction["scores"]) <= 10
    assert all(torch.isfinite(value).all() for value in prediction.values())
    assert (prediction["scores"] >= 0).all() and (prediction["scores"] <= 1).all()
    assert (prediction["boxes"] >= 0).all() and (prediction["boxes"] <= 640).all()


def test_quality_geometry_moves_with_dtype():
    model = build_model("E0", "quality_aware", 42).double()
    assert model.head.stride.dtype == torch.float64
    assert model.head.anchors.dtype == torch.float64


def test_hidden_assigner_override_rejected(monkeypatch):
    monkeypatch.setenv("YOLOM", "9")
    with pytest.raises(ValueError, match="Nondefault"):
        build_model("E0", "quality_aware", 42)
