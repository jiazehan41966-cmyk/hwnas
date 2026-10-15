"""Shared new-space architecture / precision / mapping encoding, no cost guess.

This software bridge extends the pinned author's preference table and hardware
expert mixture output from seven binary decisions to seven binaries, one
three-way precision decision and one six-way mapping decision. It does not yet
provide a trainable mixed-precision supernet, cost labels or a formal driver.
"""
from __future__ import annotations

from functools import partial
import math

import torch
from torch import nn

from tools.modnas_elastic.author import _AUTHOR_HPN, AuthorSampler, verify_author_sources
from tools.modnas_elastic.hardware import architecture_probabilities
from tools.modnas_elastic.model import architecture_id, enumerate_architectures, validate_architecture
from tools.sonar_joint_v2.policy import PROFILES, POLICIES

VERSION = "elastic72_joint23_encoding_v1"
CHOICE_SIZES = (2,) * 7 + (3, 6)
LOGIT_COUNT = sum(CHOICE_SIZES)


def _validate_weights(weights):
    if len(weights) != len(CHOICE_SIZES):
        raise ValueError("Seven architecture, precision and mapping rows required")
    device, dtype = weights[0].device, weights[0].dtype
    for value, size in zip(weights, CHOICE_SIZES):
        if value.shape != (size,) or not value.is_floating_point():
            raise ValueError("Joint decision dimension mismatch")
        if value.device != device or value.dtype != dtype:
            raise ValueError("Joint probabilities must share device and dtype")
        if not torch.isfinite(value.detach()).all() or (value.detach() < 0).any():
            raise ValueError("Joint probabilities must be finite and nonnegative")
        if not torch.allclose(value.detach().sum(), value.new_tensor(1), atol=1e-6, rtol=1e-6):
            raise ValueError("Each joint decision row must sum to one")
    return weights


def architecture_weights(weights):
    _validate_weights(weights)
    return dict(pan_weights=weights[0], td_kernel_weights=torch.stack(weights[1:4]),
                pan_kernel_weights=torch.stack(weights[4:7]))


def joint_probabilities(weights):
    """72 x 3 x 6 canonical distribution; inactive PAN choices have zero bias."""
    arch = architecture_probabilities(architecture_weights(weights))
    return arch[:, None, None] * weights[7][None, :, None] * weights[8][None, None, :]


def discrete_point(weights):
    _validate_weights(weights)
    indices = [int(value.detach().argmax()) for value in weights]
    pan = bool(indices[0])
    arch = validate_architecture(dict(pan=pan, td_kernels=[3 + 2 * i for i in indices[1:4]],
             pan_kernels=[3 + 2 * i for i in indices[4:7]] if pan else []))
    return dict(architecture=arch, architecture_id=architecture_id(arch),
                profile=PROFILES[indices[7]], mapping_policy=POLICIES[indices[8]])


def canonical_points():
    return [dict(architecture=arch, architecture_id=architecture_id(arch),
                 profile=profile, mapping_policy=mapping)
            for arch in enumerate_architectures() for profile in PROFILES for mapping in POLICIES]


class _JointExpert(_AUTHOR_HPN.HyperNetwork):
    def __init__(self, num_intervals=101):
        super().__init__(num_intervals=num_intervals)
        self.embedding = nn.Embedding(num_intervals, LOGIT_COUNT)

    def forward(self, preference):
        index = (preference[0, 0] * (self.num_intervals - 1)).round().long().reshape(1)
        return self.embedding(index).reshape(LOGIT_COUNT)


class JointConditionalHypernetwork(_AUTHOR_HPN.MetaHyperNetwork):
    """Retain author expert mixture and lookup; adapt only output dimensions."""
    adaptation_scope = "CHOICE_ENCODING_ONLY_NOT_FULL_JOINT_SEARCH"

    def __init__(self, num_intervals=101, num_random_devices=50, hw_embed_dim=10):
        verify_author_sources()
        if (type(num_intervals) is not int or num_intervals < 2 or
                type(num_random_devices) is not int or num_random_devices < 1 or
                type(hw_embed_dim) is not int or hw_embed_dim < 1):
            raise ValueError("Positive expert/device dimensions and at least two intervals required")
        super().__init__(num_random_devices=num_random_devices, hw_embed_dim=hw_embed_dim,
                         hpn_type=_JointExpert, num_intervals=num_intervals)

    def forward(self, preference, hardware):
        if preference.shape != (1, 2) or hardware.shape != (1, self.hw_embed_dim):
            raise ValueError("Preference/device descriptor shape mismatch")
        if preference.device != hardware.device or hardware.device != self.hardware_embedding.weight.device:
            raise ValueError("All conditions and model must share device")
        if not torch.isfinite(preference).all() or not torch.isfinite(hardware).all():
            raise ValueError("Nonfinite preference/device descriptor")
        if (preference < 0).any() or not torch.allclose(preference.sum(), preference.new_tensor(1), atol=1e-6):
            raise ValueError("Preference must be nonnegative and sum to one")
        similarity = (hardware @ self.hardware_embedding.weight.T).reshape(-1) / math.sqrt(self.hw_embed_dim)
        mixing = _AUTHOR_HPN.sin_softmax(similarity, dim=-1)
        flat = (mixing[:, None] * torch.stack([expert(preference) for expert in self.hpn_factory])).sum(dim=0)
        return list(torch.split(flat, CHOICE_SIZES))


def sample_decisions(sampler: AuthorSampler, logits):
    if len(logits) != len(CHOICE_SIZES) or any(x.shape != (n,) for x, n in zip(logits, CHOICE_SIZES)):
        raise ValueError("Joint logits dimension mismatch")
    # Call the untouched author's sampler separately for heterogeneous rows.
    return [sampler.sample(value[None, :]).reshape(-1) for value in logits]


class ValidatedJointCosts(nn.Module):
    """No public loader until complete source-bound tuple costs exist.

    Tensor expectations alone are not evidence. Creating an unverified table
    here is refused, including finite toy tensors, to avoid a numerical stub
    becoming a hardware-aware objective.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("BLOCKED: complete source-bound 72x3x6 hardware table validator not implemented")
