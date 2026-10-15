"""Check heterogeneous author encoding, canonical mass and explicit HW refusal."""
import pytest
import torch

from tools.modnas_comparison.joint_encoding import (
    CHOICE_SIZES, JointConditionalHypernetwork, ValidatedJointCosts,
    canonical_points, discrete_point, joint_probabilities, sample_decisions)
from tools.modnas_elastic.author import AuthorSampler


def test_all_1296_triples_unique_and_pan_tail_omitted():
    points = canonical_points()
    assert len(points) == 1296
    assert len({(p['architecture_id'], p['profile'], p['mapping_policy']) for p in points}) == 1296
    assert sum(not p['architecture']['pan'] for p in points) == 8 * 18
    assert all(not p['architecture']['pan_kernels'] for p in points if not p['architecture']['pan'])


def test_pan_off_probability_ignores_inactive_kernel_choices():
    weights = [torch.full((n,), 1 / n, dtype=torch.float64) for n in CHOICE_SIZES]
    weights[0] = torch.tensor([1., 0.], dtype=torch.float64)
    weights[4:7] = [torch.tensor([.99, .01], dtype=torch.float64) for _ in range(3)]
    original = joint_probabilities(weights)
    weights[4:7] = [torch.tensor([.01, .99], dtype=torch.float64) for _ in range(3)]
    assert torch.equal(original, joint_probabilities(weights))
    assert original.shape == (72, 3, 6)
    assert torch.allclose(original[:8], torch.full((8, 3, 6), 1 / 144, dtype=torch.float64))
    assert original[8:].count_nonzero() == 0


def test_true_author_gdas_all_joint_choices_backpropagate():
    torch.manual_seed(42)
    model = JointConditionalHypernetwork(num_random_devices=2)
    logits = model(torch.tensor([[.4, .6]]), torch.ones(1, 10))
    weights = sample_decisions(AuthorSampler(), logits)
    joint = joint_probabilities(weights)
    assert joint.shape == (72, 3, 6)
    assert joint.count_nonzero() == 1
    # Synthetic unique tuple indices test encoding gradients only, not HW costs.
    (joint * torch.arange(1296).reshape(72, 3, 6)).sum().backward()
    embedding_grads = torch.stack([x.embedding.weight.grad for x in model.hpn_factory])
    assert torch.isfinite(embedding_grads).all()
    flat_grad = embedding_grads.abs().sum(dim=(0, 1))
    assert flat_grad[:14].sum() > 0 and flat_grad[14:17].sum() > 0 and flat_grad[17:].sum() > 0
    point = discrete_point(weights)
    assert point in canonical_points()


def test_unknown_cost_table_and_invalid_condition_fail_closed():
    with pytest.raises(RuntimeError, match='source-bound'):
        ValidatedJointCosts(torch.zeros(72, 3, 6))
    model = JointConditionalHypernetwork(num_random_devices=1)
    with pytest.raises(ValueError, match='sum to one'):
        model(torch.tensor([[.4, .7]]), torch.zeros(1, 10))
