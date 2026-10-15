"""Meaningful operator/neck checks independent of optional author packages.

The small-image tests execute the real E0 trunk and real new modules. They are
not a substitute for the remote C01b 640-image loss/decode smoke.
"""
from copy import deepcopy
import io
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.detector_elastic_pilot.model import build_model as build_e0
from tools.sonar_operator_probe.model import (
    ALIAS_VARIANTS, VARIANTS, NAGR, ProbeNeck, WeightedDualRFDepthwise,
    _apply_backbone_variant, _verify_change_scope, allowed_change_prefixes,
    build_model, parameter_groups, project_nonnegative_, variant_spec,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _small_trunk(name):
    pilot = build_e0("E0", 42)
    trunk = nn.Module()
    trunk.backbone, trunk.neck = pilot.backbone, pilot.neck
    _apply_backbone_variant(trunk, name, 42)
    if name in ("nagr_p2", "nagr_p2_ungated", "dmdc_p2", "cfem_p2",
                "p2_lateral_only", "p2_extra_p4", "p2_concat", "p2_gated", "p3_weighted"):
        trunk.neck = ProbeNeck(trunk.neck, 42, name)
    return trunk


def test_contract_has_no_duplicated_training_alias_and_correct_reference_kernels():
    assert len(VARIANTS) == len(set(VARIANTS)) == 21
    assert ALIAS_VARIANTS == ("dw5_stage7",)
    assert "dw5_stage7" not in VARIANTS
    assert variant_spec("dw5_stage7")["alias_of"] == "control"
    reference = build_e0("E0", 42)
    assert [reference.backbone.stages[index][0].dw_conv.kernel_size[0] for index in (1, 2, 5, 7)] == [3, 5, 5, 5]
    assert variant_spec("dilated_dw3_stage2")["parent_comparators"] == ["dw3_stage2", "control"]
    assert variant_spec("nagr_p2")["parent_comparators"] == ["nagr_p2_ungated", "control"]
    for name in VARIANTS:
        assert variant_spec(name)["allowed_change_prefixes"] == list(allowed_change_prefixes(name))


@pytest.mark.parametrize("name", VARIANTS)
def test_actual_trunk_shapes_backward_and_registered_state_roundtrip(name):
    before = torch.get_rng_state().clone()
    trunk = _small_trunk(name).train()
    assert torch.equal(before, torch.get_rng_state()), name
    generator = torch.Generator().manual_seed(920)
    images = torch.rand(2, 1, 128, 128, generator=generator)
    features = trunk.backbone(images)
    outputs = trunk.neck(features)
    assert [tuple(tensor.shape) for tensor in outputs] == [(2, 48, side, side) for side in (32, 16, 8, 4)]
    assert all(torch.isfinite(tensor).all() for tensor in outputs)
    sum(tensor.square().mean() for tensor in outputs).backward()
    grads = [parameter.grad for parameter in trunk.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    buffer = io.BytesIO()
    torch.save(trunk.state_dict(), buffer)
    buffer.seek(0)
    reconstructed = _small_trunk(name).eval()
    reconstructed.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    trunk.eval()
    with torch.no_grad():
        expected = trunk.neck(trunk.backbone(images))
        actual = reconstructed.neck(reconstructed.backbone(images))
    assert all(torch.equal(left, right) for left, right in zip(expected, actual)), name


def test_location_guard_rejects_unregistered_head_and_other_stage_mutation():
    control = {"head.weight": torch.ones(2), "backbone.stages.1.0.dw_conv.weight": torch.ones(3),
               "backbone.stages.2.0.dw_conv.weight": torch.ones(3)}
    actual = deepcopy(control)
    actual["backbone.stages.1.0.dw_conv.weight"].zero_()
    changed, added = _verify_change_scope(control, actual, allowed_change_prefixes("dcn_stage1"))
    assert changed == ["backbone.stages.1.0.dw_conv.weight"] and not added
    actual["head.weight"].zero_()
    with pytest.raises(ValueError, match="outside"):
        _verify_change_scope(control, actual, allowed_change_prefixes("dcn_stage1"))
    actual = deepcopy(control)
    actual["backbone.stages.2.0.dw_conv.weight"].zero_()
    with pytest.raises(ValueError, match="outside"):
        _verify_change_scope(control, actual, allowed_change_prefixes("dcn_stage1"))


def test_sonarop_weighted_and_plain_share_branches_initial_forward_and_fold():
    plain = _small_trunk("sonarop_stage7").backbone.stages[7][0].dw_conv.eval()
    weighted = _small_trunk("sonarop_weighted_stage7").backbone.stages[7][0].dw_conv.eval()
    for branch in ("branch3", "branch5"):
        for key, value in getattr(plain, branch).state_dict().items():
            assert torch.equal(value, getattr(weighted, branch).state_dict()[key])
    inputs = torch.randn(2, plain.channels, 4, 4)
    with torch.no_grad():
        assert torch.equal(plain(inputs), weighted(inputs))
        weighted.alpha.fill_(0.7)
        expected = weighted(inputs)
        weighted.switch_to_deploy()
        assert torch.allclose(expected, weighted(inputs), atol=2e-6, rtol=2e-5)


def test_p2_fusion_and_p3_scalar_start_at_equal_sum_and_receive_gradients():
    reference = build_e0("E0", 42).neck.eval()
    features = [torch.randn(2, channels, side, side) for channels, side in zip(reference.in_channels, (16, 8, 4, 2))]
    with torch.no_grad():
        expected = reference(features)
    for name in ("p2_concat", "p2_gated", "p3_weighted"):
        neck = ProbeNeck(deepcopy(reference), 42, name).eval()
        outputs = neck(features)
        assert all(torch.allclose(left, right, atol=1e-6, rtol=1e-5) for left, right in zip(expected, outputs))
        outputs[0].square().mean().backward()
        added = [parameter for key, parameter in neck.named_parameters()
                 if key.startswith("p2_fusion.") or key == "alpha"]
        assert added and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in added)
        assert any(parameter.grad.abs().sum() > 0 for parameter in added)
        if name == "p2_gated":
            values = [torch.randn(2, 48, 16, 16), torch.randn(2, 48, 16, 16)]
            weights = neck.p2_fusion.source_weights(values)
            assert (weights >= 0).all() and torch.equal(weights.sum(1), torch.full_like(weights[:, 0], 2.))


def test_nagr_gate_is_bounded_and_residual_is_paired_with_ungated():
    gated, ungated = NAGR(42, True).eval(), NAGR(42, False).eval()
    for key, value in gated.residual.state_dict().items():
        assert torch.equal(value, ungated.residual.state_dict()[key])
    inputs = torch.randn(2, 48, 16, 16)
    weights = gated.gate_map(inputs)
    assert weights.shape == (2, 1, 16, 16)
    assert torch.allclose(weights, torch.full_like(weights, torch.sigmoid(torch.tensor(-3.))))
    assert ((weights >= 0) & (weights <= 1)).all()
    gated(inputs).square().mean().backward()
    assert gated.gate.weight.grad is not None and gated.gate.weight.grad.abs().sum() > 0


def test_dilation_and_downsample_controls_share_trainable_tensors():
    dw3, dilated = _small_trunk("dw3_stage2"), _small_trunk("dilated_dw3_stage2")
    assert torch.equal(dw3.backbone.stages[2][0].dw_conv.weight, dilated.backbone.stages[2][0].dw_conv.weight)
    assert dilated.backbone.stages[2][0].dw_conv.dilation == (2, 2)
    ordinary, dense, blur = [_small_trunk(name) for name in ("control", "dense_decimate_stage1", "blur_stage1")]
    expected = dict(ordinary.backbone.stages[1][0].named_parameters())
    for candidate in (dense, blur):
        for key, value in candidate.backbone.stages[1][0].named_parameters():
            assert torch.equal(value, expected[key]), key
    value = torch.zeros(1, blur.backbone.stages[1][0].dw_conv.out_channels, 8, 8)
    value[:, :, 3, 3] = 1.
    assert not torch.equal(dense.backbone.stages[1][0].downsample(value), blur.backbone.stages[1][0].downsample(value))


def test_scalar_constraint_and_optimizer_coverage():
    model = _small_trunk("p3_weighted")
    groups = parameter_groups(model)
    flattened = [parameter for group in groups for parameter in group["params"]]
    assert len(flattened) == len({id(parameter) for parameter in flattened})
    assert {id(parameter) for parameter in flattened} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert groups[-1]["params"] == [model.neck.alpha] and groups[-1]["weight_decay"] == 0.
    with torch.no_grad():
        model.neck.alpha.fill_(-1.)
    project_nonnegative_(model)
    assert model.neck.alpha.item() == 0.
    with torch.no_grad():
        model.neck.alpha.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        project_nonnegative_(model)


def test_optional_real_c01b_pairing_rng_and_head_preservation():
    pytest.importorskip("IPython", reason="Local optional author runtime absent; remote full smoke is required")
    before = torch.get_rng_state().clone()
    reference = build_model("control", 42)
    candidate = build_model("dilated_dw3_stage2", 42)
    assert torch.equal(before, torch.get_rng_state())
    assert reference.head.no == candidate.head.no == 6
    identity = candidate.initialization_identity()
    assert identity["head_sha256"] == reference.initialization_identity()["head_sha256"]
    for key, digest in identity["shared_tensor_sha256"].items():
        assert digest == reference.initialization_identity()["shared_tensor_sha256"][key]
    assert identity["unchanged_tensor_sha256"] == identity["control_unchanged_tensor_sha256"]
