"""Check the pinned author mechanisms, not a restatement of adapter wiring."""
import numpy as np
import pytest
import torch

from tools.modnas_elastic.author import ConditionalHypernetwork, AuthorSampler, combine_gradients


def test_preferences_and_hardware_reach_logits_and_parameters():
    torch.manual_seed(42)
    network = ConditionalHypernetwork(num_random_devices=3)
    hardware = torch.randn(1, 10)
    left = network(torch.tensor([[.2, .8]]), hardware)
    right = network(torch.tensor([[.8, .2]]), hardware)
    other = network(torch.tensor([[.2, .8]]), hardware + 1)
    assert left.shape == (7, 2)
    assert not torch.equal(left, right)
    assert not torch.equal(left, other)
    AuthorSampler(total_epochs=1).sample(left).square().sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in network.parameters())


def test_author_mgd_solves_opposing_conditions_and_singleton():
    combined, weights, _ = combine_gradients([[torch.tensor([1., 0.])], [torch.tensor([-1., 0.])]])
    assert np.allclose(weights, [.5, .5])
    assert torch.allclose(combined[0], torch.zeros(2))
    combined, weights, _ = combine_gradients([[torch.tensor([3., 4.])]])
    assert weights.tolist() == [1.]
    assert torch.allclose(combined[0], torch.tensor([.6, .8]))
    with pytest.raises(ValueError):
        combine_gradients([[torch.tensor([float("nan")])]])


def test_author_sampler_returns_hard_choices_with_straight_through_gradient():
    torch.manual_seed(42)
    logits = torch.randn(7, 2, requires_grad=True)
    sampler = AuthorSampler(total_epochs=2)
    sampler.before_epoch()
    weights = sampler.sample(logits)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7))
    assert torch.all((weights.detach() == 0) | (weights.detach() == 1))
    (weights * torch.tensor([1., 2.])).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad)
