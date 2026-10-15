"""Paired initialization, P3-only influence, gradient and constraint checks."""

from copy import deepcopy

import pytest
import torch

from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from hwnas_fpga.models.sonar_op.candidate import A1_ARCHITECTURE
from hwnas_fpga.models.sonar_p3_fusion import (
    build_p3_fusion_candidate,
    p3_fusion_parameter_groups,
    project_nonnegative_alpha_,
    register_p3_alpha_projection,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def seeded_build(factory):
    torch.manual_seed(42)
    model = factory()
    return model, torch.get_rng_state().clone()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_both_variants_match_original_a1_initialization_and_forward_exactly(device):
    original, original_rng = seeded_build(lambda: build_sonar_candidate(SPEC_PATH, deepcopy(A1_ARCHITECTURE)))
    control, control_rng = seeded_build(lambda: build_p3_fusion_candidate("equal_add"))
    weighted, weighted_rng = seeded_build(lambda: build_p3_fusion_candidate("weighted"))
    assert torch.equal(original_rng, control_rng)
    assert torch.equal(original_rng, weighted_rng)
    for key, value in original.state_dict().items():
        assert torch.equal(value, control.state_dict()[key]), key
        assert torch.equal(value, weighted.state_dict()[key]), key
    assert control.neck.alpha.item() == weighted.neck.alpha.item() == 1.0
    original_count = sum(p.numel() for p in original.parameters())
    assert sum(p.numel() for p in control.parameters()) == original_count
    assert sum(p.numel() for p in weighted.parameters()) == original_count + 1
    assert all(p.requires_grad for p in weighted.parameters())
    # A separate generator keeps the model-initialization RNG comparison intact.
    image = torch.randn((1, 1, 640, 640), generator=torch.Generator().manual_seed(904)).to(device)
    original.eval().to(device); control.eval().to(device); weighted.eval().to(device)
    with torch.no_grad():
        expected = original(image)
        for candidate in (control, weighted):
            actual = candidate(image)
            for group in ("backbone_features", "neck_features", "predictions"):
                assert len(expected[group]) == len(actual[group]) == 3
                for left, right in zip(expected[group], actual[group]):
                    assert torch.equal(left, right), (device, group)


def test_alpha_changes_only_p3_neck_and_predictions():
    weighted, _ = seeded_build(lambda: build_p3_fusion_candidate("weighted"))
    weighted.eval()
    image = torch.randn((1, 1, 640, 640), generator=torch.Generator().manual_seed(905))
    with torch.no_grad():
        before = weighted(image)
        weighted.neck.alpha.fill_(0.0)
        after = weighted(image)
    for left, right in zip(before["backbone_features"], after["backbone_features"]):
        assert torch.equal(left, right)
    for group in ("neck_features", "predictions"):
        assert not torch.equal(before[group][0], after[group][0]), group
        assert torch.equal(before[group][1], after[group][1]), group
        assert torch.equal(before[group][2], after[group][2]), group


def test_alpha_receives_nonzero_finite_gradient_from_p3_features():
    weighted, _ = seeded_build(lambda: build_p3_fusion_candidate("weighted"))
    weighted.eval()
    features = [torch.randn((1, channels, side, side), generator=torch.Generator().manual_seed(906 + i))
                for i, (channels, side) in enumerate(((32, 8), (96, 4), (320, 2)))]
    p3, p4, p5 = weighted.neck(features)
    p3.square().mean().backward()
    assert weighted.neck.alpha.grad is not None
    assert torch.isfinite(weighted.neck.alpha.grad)
    assert weighted.neck.alpha.grad.abs().item() > 0
    assert p4.shape[-2:] == (4, 4) and p5.shape[-2:] == (2, 2)


@pytest.mark.parametrize("variant", ["equal_add", "weighted"])
def test_parameter_groups_cover_trainable_parameters_once_with_original_decay(variant):
    model = build_p3_fusion_candidate(variant)
    groups = p3_fusion_parameter_groups(model, weight_decay=1e-4)
    flattened = [p for group in groups for p in group["params"]]
    assert len(flattened) == len({id(p) for p in flattened})
    assert {id(p) for p in flattened} == {id(p) for p in model.parameters() if p.requires_grad}
    assert groups[0]["weight_decay"] == 1e-4
    if variant == "weighted":
        assert len(groups) == 2
        assert groups[1]["params"] == [model.neck.alpha]
        assert groups[1]["weight_decay"] == 0
    else:
        assert len(groups) == 1


def test_optimizer_post_step_projects_negative_alpha_and_preserves_other_weights():
    model = build_p3_fusion_candidate("weighted")
    ordinary_before = {name: p.detach().clone() for name, p in model.named_parameters() if name != "neck.alpha"}
    optimizer = torch.optim.SGD(p3_fusion_parameter_groups(model), lr=1.0)
    handle = register_p3_alpha_projection(optimizer, model)
    model.neck.alpha.grad = torch.tensor(2.0)
    optimizer.step()
    assert model.neck.alpha.item() == 0.0
    assert all(torch.equal(ordinary_before[name], p) for name, p in model.named_parameters() if name != "neck.alpha")
    handle.remove()
    with torch.no_grad():
        model.neck.alpha.fill_(0.7)
    project_nonnegative_alpha_(model)
    assert model.neck.alpha.item() == pytest.approx(0.7)
    with torch.no_grad():
        model.neck.alpha.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        project_nonnegative_alpha_(model)


@pytest.mark.parametrize("kwargs", [{"num_classes": 3}, {"image_size": 320}, {"variant": "other"}])
def test_builder_rejects_changes_outside_fixed_a1_comparison(kwargs):
    with pytest.raises(ValueError):
        build_p3_fusion_candidate(**kwargs)
