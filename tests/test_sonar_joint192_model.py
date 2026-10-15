"""Numerical/reference/gradient checks for the independent joint192 adapter."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.detector_elastic_pilot.model import build_model as build_pilot
from tools.sonar_joint192_validation.model import (
    REFERENCE_ARCHITECTURES, BlurDownMBConv, architecture_hash,
    build_model, enumerate_architectures, validate_architecture,
)
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _variant(blur=False, identity=False, pan=False, kernel=3, stage2=5, neck5=False):
    arch = deepcopy(REFERENCE_ARCHITECTURES["E1" if pan else "E0"])
    arch["backbone"]["stage1"] = dict(
        kernel=kernel, downsample="blur3_after_dense_dw" if blur else "strided_dw")
    arch["backbone"]["stage2"]["kernel"] = stage2
    if neck5:
        arch["neck"]["td_ops"] = dict(P2="mbconv_k5", P3="mbconv_k5", P4="mbconv_k5")
    if identity:
        arch["neck"]["td_ops"]["P2"] = "identity"
    return arch


def test_192_unique_architectures_all_strictly_valid_and_reference_coverage():
    architectures = list(enumerate_architectures())
    assert len(architectures) == 192
    assert len({architecture_hash(a) for a in architectures}) == 192
    assert all(validate_architecture(a) == a for a in architectures)
    assert all(a in architectures for a in REFERENCE_ARCHITECTURES.values())
    assert sum(a["neck"]["pan"] for a in architectures) == 96


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(extra=1),
    lambda a: a["backbone"]["stage1"].update(kernel=True),
    lambda a: a["backbone"]["stage1"].update(downsample="reflect_blur"),
    lambda a: a["backbone"]["stage2"].update(kernel=3.0),
    lambda a: a["neck"].update(pan=1),
    lambda a: a["neck"].update(pan_kernels=[3, 3, 3]),
    lambda a: a["neck"]["td_ops"].update(P3="identity"),
])
def test_invalid_architecture_cannot_silently_change_contract(mutate):
    arch = deepcopy(REFERENCE_ARCHITECTURES["E0"])
    mutate(arch)
    with pytest.raises(ValueError):
        validate_architecture(arch)


@pytest.mark.parametrize("prototype", ["E0", "E1"])
def test_reference_bit_exact_predictions_features_state_and_loss(prototype):
    reference = build_pilot(prototype, seed=42).eval()
    adapter = build_model(REFERENCE_ARCHITECTURES[prototype], seed=42).eval()
    assert adapter.architecture == reference.architecture
    assert adapter.loss_config(BASELINE) == reference.loss_config(BASELINE)
    assert set(adapter.state_dict()) == set(reference.state_dict())
    assert all(torch.equal(t, reference.state_dict()[n]) for n, t in adapter.state_dict().items())
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(123)
        images = torch.rand(1, 1, 640, 640)
        first, second = reference(images), adapter(images)
    for key in first:
        for left, right in zip(first[key], second[key]):
            assert torch.equal(left, right), key


@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_blur_and_strided_parameter_pairing_and_rng_preservation(kernel, seed):
    before = torch.random.get_rng_state().clone()
    ordinary = build_model(_variant(kernel=kernel), seed)
    blurred = build_model(_variant(blur=True, kernel=kernel), seed)
    assert torch.equal(before, torch.random.get_rng_state())
    first = dict(ordinary.named_parameters())
    second = dict(blurred.named_parameters())
    assert set(first) == set(second)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert "backbone.stages.1.0.blur_kernel" in dict(blurred.named_buffers())
    assert "backbone.stages.1.0.blur_kernel" not in second


def test_fixed_lowpass_exact_impulse_response_zero_boundary_and_gradient():
    block = BlurDownMBConv(1, 2, 3)
    assert block.dw_conv.stride == (1, 1) and not block.use_residual
    impulse = torch.zeros(1, 3, 7, 7, requires_grad=True)
    with torch.no_grad():
        impulse[:, :, 3, 3] = 1
    output = block.downsample(impulse)
    # Sampling centers are 0,2,4,6; impulse at3 contributes1/16 to four centers.
    expected = torch.zeros(1, 3, 4, 4)
    expected[:, :, 1:3, 1:3] = 1 / 16
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.isfinite(impulse.grad).all()
    constant = block.downsample(torch.ones(1, 3, 8, 8))
    assert constant[0, 0, 0, 0].item() == 9 / 16
    assert constant[0, 0, 1, 1].item() == 1
    assert not block.blur_kernel.requires_grad


@pytest.mark.parametrize("variant", [
    dict(blur=True),
    dict(identity=True),
    dict(blur=True, identity=True),
    dict(blur=True, identity=True, pan=True, kernel=5, stage2=3, neck5=True),
])
def test_four_representative_variants_real_640_loss_backward_geometry(variant):
    model = build_model(_variant(**variant), seed=42).train()
    cfg = model.loss_config(BASELINE)
    targets = [dict(boxes=torch.tensor([[112., 112., 124., 128.]]),
                    labels=torch.tensor([0]))]
    result = model(torch.rand(1, 1, 640, 640))
    for stride, channels, feature, neck, pred in zip(
            (4, 8, 16, 32), (24, 32, 96, 320), result["backbone_features"],
            result["neck_features"], result["predictions"]):
        assert tuple(feature.shape) == (1, channels, 640 // stride, 640 // stride)
        assert tuple(neck.shape) == (1, 48, 640 // stride, 640 // stride)
        assert tuple(pred.shape) == (1, 21, 640 // stride, 640 // stride)
    loss, parts = DenseAnchorDetectionLoss(cfg)(result, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.backbone.stages[1][0].dw_conv.weight.grad.abs().sum() > 0
    assert model.backbone.stages[2][0].dw_conv.weight.grad.abs().sum() > 0
    if variant.get("identity"):
        assert isinstance(model.neck.top_down_blocks["P2"], nn.Identity)
    if variant.get("pan"):
        assert len(model.neck.bottom_up_blocks) == 3
        assert all(block.dw_conv.kernel_size == (3, 3)
                   for block in model.neck.bottom_up_blocks.values())
    else:
        assert not model.neck.bottom_up_blocks
        assert not model.neck.bottom_up_projections
