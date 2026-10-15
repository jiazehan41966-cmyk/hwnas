"""Phase 4A P0 operator-pool unit tests. No test split access."""

from __future__ import annotations

import torch
import pytest

from hwnas_fpga.models.sonar_operator_pool_p0 import (
    P0_OPERATORS,
    P0_STAGES,
    apply_p0_operator,
    build_d_baseline,
    build_p0_candidate,
    get_stage_block,
    p0_architecture_id,
    parse_p0_architecture_id,
)


def test_all_p0_operators_forward_backward_stage7_shape():
    for operator in P0_OPERATORS:
        model = build_p0_candidate(operator, 7).train()
        x = torch.randn(2, 1, 640, 640)
        output = model(x)
        assert [tuple(t.shape) for t in output["predictions"]] == [(2, 21, 80, 80), (2, 21, 40, 40), (2, 21, 20, 20)]
        loss = sum(t.mean() for t in output["predictions"])
        loss.backward()
        block = get_stage_block(model, 7)
        grads = [p.grad for p in block.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("stage,expected_feature", [(1, (1, 24, 160, 160)), (2, (1, 32, 80, 80)), (5, (1, 96, 40, 40)), (7, (1, 320, 20, 20))])
def test_representative_stage_feature_shapes_with_frequency_lite(stage, expected_feature):
    model = build_p0_candidate("FrequencyLite", stage).eval()
    with torch.no_grad():
        features = model(torch.randn(1, 1, 640, 640))["backbone_features"]
    # P3/P4/P5 features are always collected for D; verify whole detector remains valid.
    assert [tuple(t.shape) for t in features] == [(1, 32, 80, 80), (1, 96, 40, 40), (1, 320, 20, 20)]

    block = get_stage_block(model, stage)
    spatial = block.dw_conv
    channels = getattr(spatial, "channels", getattr(spatial, "in_channels", None))
    assert channels is not None
    stride = 2 if stage in (1, 2) else 1
    out_h, out_w = expected_feature[2], expected_feature[3]
    x = torch.randn(1, channels, out_h * stride, out_w * stride)
    with torch.no_grad():
        y = spatial(x)
    assert tuple(y.shape) == (1, channels, out_h, out_w)


def test_dw3_noop_is_exact_same_module_state():
    left = build_d_baseline().eval()
    right = build_d_baseline().eval()
    right.load_state_dict(left.state_dict())
    apply_p0_operator(right, "DW3", 7)
    with torch.no_grad():
        x = torch.randn(1, 1, 640, 640)
        a = left(x)["predictions"]
        b = right(x)["predictions"]
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_architecture_id_roundtrip_and_rejection():
    for operator in P0_OPERATORS:
        for stage in P0_STAGES:
            aid = p0_architecture_id(operator, stage)
            assert parse_p0_architecture_id(aid) == (operator, stage)
    with pytest.raises(ValueError):
        p0_architecture_id("Attention", 7)
    with pytest.raises(ValueError):
        p0_architecture_id("DW5", 4)

