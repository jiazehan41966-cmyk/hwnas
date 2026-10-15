"""Gradient, sampling and baseline invariants for the isolated adaptations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch
from hwnas_fpga.training.detection_trainer import DetectionLossConfig, DenseAnchorDetectionLoss
from tools.sonar_background_pilot.losses import (
    BackgroundDetectionLoss, BackgroundOptions, exterior_cells, local_negative_weights,
    prototype_separation, quality_positive_weights, rank_pairs,
)


def test_rank_gradient_pushes_background_down_and_foreground_up():
    foreground = torch.tensor([-2.0], requires_grad=True)
    background = torch.tensor([-0.2], requires_grad=True)
    rank_pairs(foreground, background).backward()
    assert foreground.grad.item() < 0
    assert background.grad.item() > 0


def test_rank_empty_groups_are_finite_and_differentiable():
    positive = torch.empty(0, requires_grad=True)
    negative = torch.ones(2, requires_grad=True)
    value = rank_pairs(positive, negative)
    value.backward()
    assert value.item() == 0 and negative.grad.abs().sum() == 0


def test_pisa_prioritizes_reliable_boxes_with_same_gt_mass():
    quality = torch.tensor([0.2, 0.8, 0.4, 0.4])
    group = torch.tensor([[0.], [0.], [1.], [1.]])
    weights = quality_positive_weights(quality, group, 0.5)
    assert weights[1] > weights[0]
    torch.testing.assert_close(weights[:2].sum(), torch.tensor(2.))
    torch.testing.assert_close(weights[2:], torch.ones(2))
    assert not weights.requires_grad


def test_pisa_local_clusters_preserve_negative_mass():
    score = torch.tensor([[[[0.1, 0.9], [0.2, 0.3]]]])
    mask = torch.ones_like(score, dtype=torch.bool)
    weights = local_negative_weights(score, mask, 2, 0.5)
    assert weights[0, 0, 0, 1] > weights[0, 0, 0, 0]
    torch.testing.assert_close(weights.sum(), torch.tensor(4.))


def test_background_excludes_target_and_surrounding_margin():
    target = {"boxes": torch.tensor([[16., 16., 32., 32.]])}
    outside = exterior_cells(target, 8, 8, 8, torch.device("cpu"))
    assert not outside[2, 2] and not outside[1, 1]
    assert outside[7, 7]


def test_prototype_loss_rewards_separation_and_has_feature_gradients():
    foreground = torch.tensor([[1., 0.], [0.9, 0.1]], requires_grad=True)
    near = torch.tensor([[0.9, 0.1], [0.8, 0.2]], requires_grad=True)
    far = torch.tensor([[-1., 0.], [-0.9, -0.1]], requires_grad=True)
    bad = prototype_separation([foreground], [near], 0.2)
    good = prototype_separation([foreground], [far], 0.2)
    assert good < bad
    bad.backward()
    assert torch.isfinite(foreground.grad).all() and foreground.grad.abs().sum() > 0
    assert torch.isfinite(near.grad).all() and near.grad.abs().sum() > 0


def test_baseline_and_disabled_pisa_preserve_loss_and_gradient():
    cfg = DetectionLossConfig(num_classes=2, image_size=64)
    target = [{"boxes": torch.tensor([[20., 20., 40., 40.]]), "labels": torch.tensor([1])}]
    predictions = [torch.randn(1, 21, n, n, requires_grad=True) for n in (8, 4, 2)]
    original = DenseAnchorDetectionLoss(cfg)(predictions, target)[0]
    original_grad = torch.autograd.grad(original, predictions)
    for option in (BackgroundOptions(), BackgroundOptions(variant="pisa", pisa_strength=0)):
        actual = BackgroundDetectionLoss(cfg, option)(predictions, target)[0]
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        for a, b in zip(torch.autograd.grad(actual, predictions), original_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_pisa_empty_image_background_gradient_has_correct_direction():
    cfg = DetectionLossConfig(num_classes=2, image_size=64)
    target = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
    predictions = [torch.zeros(1, 21, n, n, requires_grad=True) for n in (8, 4, 2)]
    criterion = BackgroundDetectionLoss(cfg, BackgroundOptions(variant="pisa"))
    loss, _ = criterion(predictions, target)
    loss.backward()
    for pred in predictions:
        grad = pred.grad.reshape(1, 3, 7, pred.shape[-2], pred.shape[-1])[:, :, 4]
        assert (grad >= 0).all() and torch.isfinite(grad).all()
