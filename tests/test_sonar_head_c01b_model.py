"""C01b contract checks: paired initialization, 6 channels, GIoU and geometry."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sonar_head_factorial.model import build_model as build_c01
from tools.sonar_head_c01b.model import (
    DirectGIoUBboxLoss, build_model, build_criterion, criterion_metadata,
    decode, loss_from_output,
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
def test_exact_c01_shared_initialization_and_rng_preservation(seed):
    before = torch.get_rng_state().clone()
    reference, direct = build_c01("E0", "quality_aware", seed), build_model(seed)
    assert_equal_state(reference.backbone.state_dict(), direct.backbone.state_dict())
    assert_equal_state(reference.neck.state_dict(), direct.neck.state_dict())
    assert_equal_state(reference.head.cv3.state_dict(), direct.head.cv3.state_dict())
    for old, new in zip(reference.head.cv2, direct.head.cv2):
        assert_equal_state(old[:2].state_dict(), new[:2].state_dict())
        assert old[-1].out_channels == 64 and new[-1].out_channels == 4
        assert new[-1].in_channels == 64 and new[-1].groups == 4
        assert torch.equal(new[-1].bias, torch.ones_like(new[-1].bias))
    identity = direct.initialization_identity()
    assert identity["c01_reference"] == reference.initialization_identity()
    assert identity["shared_head_sha256"] == identity["c01_shared_head_sha256"]
    assert identity["head_sha256"] != identity["c01_reference"]["head_sha256"]
    assert torch.equal(before, torch.get_rng_state())
    assert direct.metadata()["quality_hidden_channels"] == {"regression": 64, "classification": 48}
    assert direct.head.reg_max == 1 and direct.head.no == 6
    assert isinstance(direct.head.dfl, nn.Identity)


def test_seeded_final_layer_and_dtype_geometry():
    first, second, other = build_model(42), build_model(42), build_model(43)
    assert_equal_state(first.state_dict(), second.state_dict())
    assert not torch.equal(first.head.cv2[0][-1].weight, other.head.cv2[0][-1].weight)
    first.double()
    assert first.head.stride.dtype == torch.float64
    assert first.head.anchors.dtype == torch.float64


@pytest.mark.parametrize("empty", [False, True])
def test_direct_giou_loss_finite_with_box_and_cls_gradients(empty):
    model = build_model(42).train()
    criterion = build_criterion(model, BASELINE)
    features = [torch.randn(1, channels, 128 // stride, 128 // stride)
                for channels, stride in zip(model.feature_channels, model.feature_strides)]
    predictions = model.head(features)
    boxes = torch.empty((0, 4)) if empty else torch.tensor([[32., 32., 64., 64.]])
    labels = torch.empty(0, dtype=torch.long) if empty else torch.tensor([1])
    loss, parts = loss_from_output(model, criterion, {"predictions": predictions}, [{"boxes": boxes, "labels": labels}])
    assert torch.isfinite(loss) and loss.requires_grad
    assert parts["dfl_loss"] == 0.0
    assert not criterion.use_dfl and not criterion.bbox_loss.use_dfl
    assert criterion_metadata(model, criterion)["box"] == "GIoU"
    assert all(tensor.shape[1] == 6 and (tensor[:, :4] >= 0).all() for tensor in predictions)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.head.cv3.parameters())
    if not empty:
        assert parts["box_loss"] > 0
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in model.head.cv2.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_giou_uses_enclosing_area_and_tal_quality_weights():
    # Non-overlapping 2x2 boxes in a 6x2 enclosure: IoU=0, GIoU=-1/3.
    model = build_model(42)
    criterion = build_criterion(model, BASELINE)
    predicted = torch.tensor([[[0., 0., 2., 2.], [10., 10., 12., 12.]]], requires_grad=True)
    target = torch.tensor([[[4., 0., 6., 2.], [0., 0., 2., 2.]]])
    scores = torch.tensor([[[0.25, 0.], [1., 0.]]])
    foreground = torch.tensor([[True, False]])
    loss, dfl, giou = criterion.bbox_loss(torch.zeros(1, 2, 4), predicted, torch.zeros(2, 2),
                                        target, scores, torch.tensor(1.), foreground)
    assert loss.item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert giou.item() == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert dfl.item() == 0.0
    loss.backward()
    assert predicted.grad[0, 0].abs().sum() > 0
    assert predicted.grad[0, 1].abs().sum() == 0


def test_training_and_inference_share_nonnegative_distances_and_geometry():
    model = build_model(42)
    # A constant final predictor includes negative distances to exercise ReLU.
    with torch.no_grad():
        for tower in model.head.cv2:
            tower[-1].weight.zero_()
            tower[-1].bias.copy_(torch.tensor([-2., 2., 3., -4.]))
    features = [torch.zeros(1, channels, 64 // stride, 64 // stride)
                for channels, stride in zip(model.feature_channels, model.feature_strides)]
    model.train()
    raw_train = model.head(features)
    model.eval()
    criterion = build_criterion(model, BASELINE)
    with torch.no_grad():
        decoded, raw_eval = model.head(features)
    for train, evaluation in zip(raw_train, raw_eval):
        assert torch.equal(train[:, :4], evaluation[:, :4])
        assert torch.equal(evaluation[0, :4, 0, 0], torch.tensor([0., 2., 3., 0.]))
    # First anchor center=(.5,.5), stride4, ltrb=(0,2,3,0).
    assert torch.equal(decoded[0, :4, 0], torch.tensor([8., -2., 12., 8.]))
    points = model.head.anchors.transpose(0, 1)
    distances = torch.cat([raw[:, :4].reshape(1, 4, -1) for raw in raw_eval], dim=2).transpose(1, 2)
    xyxy_train = criterion.bbox_decode(points, distances) * model.head.strides.transpose(0, 1)
    xywh = decoded[:, :4].transpose(1, 2)
    xyxy_eval = torch.cat((xywh[:, :, :2] - xywh[:, :, 2:] / 2,
                          xywh[:, :, :2] + xywh[:, :, 2:] / 2), dim=2)
    assert torch.equal(xyxy_train, xyxy_eval)


def test_full_image_geometry_and_frozen_postprocessing():
    model = build_model(42).eval()
    criterion = build_criterion(model, BASELINE)
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 640, 640))
        predictions = decode(model, criterion, output, score_threshold=0., nms_iou_threshold=.5,
                             pre_nms_topk=20, max_detections=10)
    assert [raw.shape[1] for raw in output["predictions"][1]] == [6, 6, 6, 6]
    assert [raw.shape[-1] for raw in output["predictions"][1]] == [160, 80, 40, 20]
    assert len(predictions) == 1 and len(predictions[0]["scores"]) <= 10
    assert all(torch.isfinite(tensor).all() for tensor in predictions[0].values())
    assert (predictions[0]["boxes"] >= 0).all() and (predictions[0]["boxes"] <= 640).all()
    assert (predictions[0]["scores"] >= 0).all() and (predictions[0]["scores"] <= 1).all()


def test_hidden_assignment_override_is_rejected(monkeypatch):
    monkeypatch.setenv("YOLOM", "9")
    with pytest.raises(ValueError, match="Nondefault"):
        build_model(42)


@pytest.mark.parametrize("seed", [True, 41, 45, "42"])
def test_unregistered_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="C01b seed"):
        build_model(seed)
