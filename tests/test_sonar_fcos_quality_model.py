"""Offline CPU witnesses for the quality/DFL adaptation and control parity."""
from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.fcos import FCOS, FCOSHead
from torchvision.models.detection.image_list import ImageList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_fcos_pilot.model import build_model as build_baseline, state_digest
from tools.sonar_fcos_quality.model import (
    DFL_WEIGHT, REG_MAX, QualityFCOS, QualityFCOSHead, aligned_box_iou,
    build_model, distribution_focal_loss, model_contract, quality_focal_loss,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_qfl_continuous_quality_optimum_and_manual_value():
    logits = torch.tensor([0.0, torch.logit(torch.tensor(0.25))], requires_grad=True)
    targets = torch.tensor([0.75, 0.25])
    losses = quality_focal_loss(logits, targets)
    torch.testing.assert_close(losses[0], torch.tensor(0.0625) * torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(losses[1], torch.tensor(0.0), atol=1e-10, rtol=0)
    losses.sum().backward()
    assert logits.grad[0] < 0  # The quality score must move toward .75.
    assert abs(logits.grad[1]) < 1e-7


def test_dfl_fractional_and_inclusive_endpoint_targets_without_saturation():
    probabilities = torch.tensor([[0.1, 0.6, 0.3], [0.1, 0.2, 0.7]])
    logits = probabilities.log().requires_grad_()
    actual = distribution_focal_loss(logits, torch.tensor([1.25, 2.0]))
    expected = torch.stack([-0.75 * probabilities[0, 1].log() - 0.25 * probabilities[0, 2].log(),
                            -probabilities[1, 2].log()])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0
    assert logits.grad[1, 2] < 0
    with pytest.raises(ValueError, match="outside"):
        distribution_focal_loss(logits, torch.tensor([1.0, 2.01]))


def test_control_is_exact_baseline_and_quality_preserves_shared_initial_tensors():
    reference = build_baseline("AAA", 3, 42, pretrained=False)
    control = build_model("AAA", 3, 42, pretrained=False)
    assert type(control) is FCOS
    assert state_digest(reference) == state_digest(control)
    del control
    quality = build_model("QFL_DFL", 3, 42, pretrained=False)
    reference_state, quality_state = reference.state_dict(), quality.state_dict()
    kept = [name for name in reference_state
            if not name.startswith(("head.regression_head.bbox_reg.", "head.regression_head.bbox_ctrness."))]
    assert kept
    assert all(torch.equal(reference_state[name], quality_state[name]) for name in kept)
    assert not any("ctrness" in name for name in quality_state)
    contract = model_contract(quality)
    assert contract["variant"] == "QFL_DFL"
    assert contract["quality"]["reg_max"] == 16
    assert contract["quality"]["giou_weight"] == 1.0
    assert contract["quality"]["out_of_support_policy"] == "fail_without_clipping"
    assert quality.compute_loss.__func__ is FCOS.compute_loss
    with pytest.raises(ValueError, match="variant"):
        build_model("OTHER", 3, 42, pretrained=False)


def test_dfl_expectation_and_fp32_under_autocast():
    head = QualityFCOSHead(FCOSHead(64, 1, 3))
    # A known symmetric distribution has expectation 8; this also tests layout.
    nn.init.zeros_(head.regression_head.bbox_distribution.weight)
    nn.init.zeros_(head.regression_head.bbox_distribution.bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = head([torch.randn(2, 64, 4, 4), torch.randn(2, 64, 2, 2)])
    assert outputs["bbox_distribution"].shape == (2, 20, 68)
    assert outputs["bbox_regression"].dtype == torch.float32
    torch.testing.assert_close(outputs["bbox_regression"], torch.full((2, 20, 4), 8.0))


def _single_match_outputs():
    return {"cls_logits": torch.zeros((1, 1, 3), requires_grad=True),
            "bbox_regression": torch.tensor([[[2.0, 2.0, 2.0, 2.0]]], requires_grad=True),
            "bbox_distribution": torch.zeros((1, 1, 4 * (REG_MAX + 1)), requires_grad=True)}


def test_quality_target_is_actual_iou_and_does_not_backprop_into_box_target():
    head = QualityFCOSHead(FCOSHead(64, 1, 3))
    outputs = _single_match_outputs()
    anchors = [torch.tensor([[0.0, 0.0, 8.0, 8.0]])]
    targets = [{"boxes": torch.tensor([[-4.0, -4.0, 12.0, 12.0]]), "labels": torch.tensor([1])}]
    # Predicted side distance=16 px -> area1024; GT area256, so quality=.25.
    losses = head.compute_loss(targets, outputs, anchors, [torch.tensor([0])])
    expected_targets = torch.tensor([[[0.0, 0.25, 0.0]]])
    torch.testing.assert_close(losses["classification"], quality_focal_loss(outputs["cls_logits"], expected_targets).sum())
    losses["classification"].backward()
    assert outputs["bbox_regression"].grad is None
    assert outputs["bbox_distribution"].grad is None
    torch.testing.assert_close(losses["bbox_distribution"], torch.tensor(17.0).log() * DFL_WEIGHT)
    assert head.last_dfl_target_stats["maximum"] == 1.0
    assert head.last_dfl_target_stats["clipped_count"] == 0


def test_assignment_on_full_640_grid_has_representable_targets():
    # Exercise inherited FCOS matching on all five real 640-pixel feature grids.
    head = QualityFCOSHead(FCOSHead(64, 1, 3))
    model = QualityFCOS(TinyBackbone(), 3, head=head, min_size=640, max_size=640)
    sides = [80, 40, 20, 10, 5]
    features = [torch.empty(1, 1, side, side) for side in sides]
    anchors = model.anchor_generator(ImageList(torch.empty(1, 3, 640, 640), [(640, 640)]), features)
    counts = [side * side for side in sides]
    length = sum(counts)
    outputs = {"cls_logits": torch.zeros(1, length, 3),
               "bbox_regression": torch.ones(1, length, 4),
               "bbox_distribution": torch.zeros(1, length, 68)}
    targets = [{"boxes": torch.tensor([[12., 12., 36., 36.], [100., 120., 220., 250.],
                                       [40., 60., 430., 510.], [0., 0., 640., 640.]]),
                "labels": torch.tensor([1, 2, 1, 2])}]
    losses = model.compute_loss(targets, outputs, anchors, counts)
    assert all(torch.isfinite(value) for value in losses.values())
    stats = head.last_dfl_target_stats
    assert stats["positive_count"] > 0
    assert 0 < stats["maximum"] < 8
    assert stats["out_of_support_count"] == stats["clipped_count"] == 0


class TinyBackbone(nn.Module):
    """Small CPU fixture exercising real FCOS transform, anchors and heads."""
    out_channels = 64

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, images):
        features = self.conv(images)
        return OrderedDict((str(level), F.adaptive_avg_pool2d(features, (max(1, images.shape[-2] // stride),
                                                                           max(1, images.shape[-1] // stride))))
                           for level, stride in enumerate([8, 16, 32, 64, 128]))


def test_standard_training_and_eval_contract_with_finite_gradients_and_empty_gt():
    model = QualityFCOS(TinyBackbone(), 3, head=QualityFCOSHead(FCOSHead(64, 1, 3)),
                        min_size=64, max_size=64, score_thresh=.001)
    image = torch.rand(3, 64, 64)
    target = {"boxes": torch.tensor([[10., 10., 54., 54.]]), "labels": torch.tensor([1])}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = model([image], [target])
    assert set(losses) == {"classification", "bbox_regression", "bbox_distribution"}
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    for layer in [model.backbone.conv, model.head.classification_head.cls_logits,
                  model.head.regression_head.bbox_distribution]:
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    losses = model([image], [empty])
    assert losses["classification"] > 0
    assert losses["bbox_regression"] == losses["bbox_distribution"] == 0
    sum(losses.values()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with torch.no_grad():
        predictions = model.eval()([image])
    assert len(predictions) == 1
    assert set(predictions[0]) == {"boxes", "scores", "labels"}
    assert all(torch.isfinite(value).all() for value in predictions[0].values())


def test_postprocessing_uses_joint_sigmoid_directly_with_baseline_limits():
    model = QualityFCOS(TinyBackbone(), 3, head=QualityFCOSHead(FCOSHead(64, 1, 3)),
                        min_size=64, max_size=64, score_thresh=.001)
    logits = torch.full((1, 2, 3), -20.0)
    logits[0, :, 1] = torch.logit(torch.tensor([.9, .8]))
    anchors = [[torch.tensor([[4., 4., 12., 12.], [20., 20., 28., 28.]])]]
    outputs = {"cls_logits": [logits], "bbox_regression": [torch.full((1, 2, 4), .5)]}
    predictions = model.postprocess_detections(outputs, anchors, [(64, 64)])[0]
    torch.testing.assert_close(predictions["scores"], torch.tensor([.9, .8]))
    assert predictions["labels"].tolist() == [1, 1]
    torch.testing.assert_close(predictions["boxes"], anchors[0][0])


def test_aligned_iou_matches_hand_computed_pairs():
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 4., 4.]])
    targets = torch.tensor([[1., 1., 3., 3.], [1., 1., 3., 3.]])
    torch.testing.assert_close(aligned_box_iou(boxes, targets), torch.tensor([1 / 7, 1 / 4]))
