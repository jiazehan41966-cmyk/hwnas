"""New-space membership, selected-network export, and real architecture gradients."""
import math
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
from tools.modnas_elastic.model import (
    ARCHITECTURE_COUNT, PAN_LEVELS, TD_LEVELS,
    ElasticNeckSupernet, StaticElasticNeckDetector, architecture_hash,
    architecture_id, enumerate_architectures, one_hot_weights, validate_architecture,
)
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def _arch(pan, td=(3, 3, 3), bottom=(3, 3, 3)):
    return dict(pan=pan, td_kernels=list(td), pan_kernels=list(bottom) if pan else [])


def _assert_outputs_equal(first, second):
    assert set(first) == set(second) == {"backbone_features", "neck_features", "predictions"}
    for key in first:
        assert len(first[key]) == len(second[key]) == 4
        for left, right in zip(first[key], second[key]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_exactly_72_new_neck_architectures_and_no_inactive_pan_choices():
    architectures = list(enumerate_architectures())
    assert len(architectures) == len({architecture_id(a) for a in architectures}) == ARCHITECTURE_COUNT == 72
    assert len({architecture_hash(a) for a in architectures}) == 72
    assert sum(not arch["pan"] for arch in architectures) == 8
    assert sum(arch["pan"] for arch in architectures) == 64
    for arch in architectures:
        assert validate_architecture(arch) == arch
        assert len(arch["pan_kernels"]) == (3 if arch["pan"] else 0)
        assert "kernels" not in arch and "output_combination" not in arch
    with pytest.raises(ValueError, match="0 active"):
        validate_architecture(dict(pan=False, td_kernels=[3, 3, 3], pan_kernels=[3, 3, 3]))
    with pytest.raises(ValueError, match="fields"):
        validate_architecture(dict(output_combination="P2_P3_P4", kernels={}))
    for field, value in (("pan", 1), ("td_kernels", [3, 7, 3]), ("td_kernels", [True, 3, 3])):
        bad = _arch(False)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_architecture(bad)


@pytest.mark.parametrize("pan", [False, True])
def test_real_640_k3_onehot_export_and_original_pilot_are_identical(pan):
    model = ElasticNeckSupernet(seed=42).eval()
    arch = _arch(pan)
    candidate = model.export_subnet(arch).eval()
    pilot = build_pilot("E1" if pan else "E0", 42).eval()
    assert candidate.loss_config(BASELINE) == pilot.loss_config(BASELINE) == model.loss_config(BASELINE)
    assert not any("branches" in key for key in candidate.state_dict())
    if not pan:
        assert not any("bottom_up" in key for key in candidate.state_dict())
        assert not hasattr(candidate.neck, "bottom_up_pool")
    mapping = candidate.supernet_state_mapping
    assert set(mapping) == set(candidate.state_dict())
    for key, value in candidate.state_dict().items():
        source = model.state_dict()[mapping[key]]
        assert torch.equal(value, source)
        assert value.data_ptr() != source.data_ptr()
        assert torch.equal(value, pilot.state_dict()[key])
    images = torch.zeros(1, 1, 640, 640)
    with torch.no_grad():
        selected = model(images, one_hot_weights(arch))
        _assert_outputs_equal(selected, candidate(images))
        _assert_outputs_equal(selected, pilot(images))


@pytest.mark.parametrize("pan,td,bottom", [
    (False, (5, 3, 5), ()),
    (True, (3, 5, 5), (5, 3, 5)),
])
def test_mixed_kernel_exact_mapping_physical_pan_omission_and_640_export(pan, td, bottom):
    arch = _arch(pan, td, bottom)
    model = ElasticNeckSupernet(seed=43).eval()
    candidate = model.export_subnet(arch).eval()
    assert [candidate.neck.top_down_blocks[level].dw_conv.kernel_size[0]
            for level in TD_LEVELS] == list(td)
    assert [candidate.neck.bottom_up_blocks[level].dw_conv.kernel_size[0]
            for level in PAN_LEVELS if pan] == list(bottom)
    assert candidate.architecture_contract()["fixed_backbone_architecture"]["kernels"] == {
        "stage1": 3, "stage2": 5, "stage4": 5, "stage5": 5, "stage6": 5, "stage7": 5}
    assert len(candidate.backbone.stages) == 8
    images = torch.randn(1, 1, 640, 640)
    with torch.no_grad():
        _assert_outputs_equal(model(images, one_hot_weights(arch)), candidate(images))


@pytest.mark.parametrize("num_classes", [1, 2, 3, 10])
def test_category_count_head_bias_loss_and_geometry_on_meta(num_classes):
    # Actual 640 geometry without allocating repeated full-resolution activations.
    with torch.device("meta"):
        candidate = StaticElasticNeckDetector(_arch(True, (5, 3, 5), (3, 5, 3)),
                                              seed=44, num_classes=num_classes).eval()
        outputs = candidate(torch.empty(1, 1, 640, 640, device="meta"))
    channels = 3 * (num_classes + 5)
    assert candidate.loss_config(BASELINE).num_classes == num_classes
    assert candidate.loss_config(BASELINE).class_loss_weights == (1.0,) * num_classes
    assert candidate.architecture_contract()["prediction_channels"] == channels
    for prediction, stride in zip(outputs["predictions"], (4, 8, 16, 32)):
        assert tuple(prediction.shape) == (1, channels, 640 // stride, 640 // stride)
    head = ElasticNeckSupernet(seed=44, num_classes=num_classes).head
    for tower in head.towers:
        assert len(tower) == 1 and isinstance(tower[0], nn.Conv2d)
        assert tower[0].kernel_size == (1, 1)
        bias = tower[0].bias.detach().view(3, num_classes + 5)
        torch.testing.assert_close(bias[:, 4], torch.full((3,), math.log(0.01 / 0.99)))
        torch.testing.assert_close(bias[:, 5:], torch.zeros(3, num_classes))


def test_real_640_detection_architecture_gradients_and_inactive_pan_is_zero():
    model = ElasticNeckSupernet(seed=42).eval()
    # Frozen backbone avoids a large unnecessary backward graph for this contract.
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    gate_logits = torch.tensor([0.4, -0.2], requires_grad=True)
    td_logits = torch.zeros(3, 2, requires_grad=True)
    pan_logits = torch.zeros(3, 2, requires_grad=True)
    probabilities = gate_logits.softmax(-1)
    hard = torch.tensor([1.0, 0.0])
    gate = hard + probabilities - probabilities.detach()
    weights = dict(pan_weights=gate, td_kernel_weights=td_logits.softmax(-1),
                   pan_kernel_weights=pan_logits.softmax(-1))
    outputs = model.forward_combinations(torch.rand(1, 1, 640, 640), weights)
    assert outputs["E0"] is not outputs["E1"]
    assert len(outputs["E0"]["predictions"]) == len(outputs["E1"]["predictions"]) == 4
    target = [dict(boxes=torch.tensor([[112.0, 112.0, 124.0, 128.0]]), labels=torch.tensor([0]))]
    criterion = DenseAnchorDetectionLoss(model.loss_config(BASELINE))
    loss0, parts0 = criterion(outputs["E0"], target)
    loss1, parts1 = criterion(outputs["E1"], target)
    loss = gate[0] * loss0 + gate[1] * loss1
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in (*parts0.values(), *parts1.values()))
    loss.backward()
    for logits in (gate_logits, td_logits, pan_logits):
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert gate_logits.grad.abs().sum() > 0
    assert (td_logits.grad.abs().sum(-1) > 0).all()
    torch.testing.assert_close(pan_logits.grad, torch.zeros_like(pan_logits), atol=0, rtol=0)
    for group in (model.neck.bottom_up_blocks, model.neck.bottom_up_projections):
        for parameter in group.parameters():
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), atol=0, rtol=0)
    assert model.neck.top_down_blocks["P2"].branches["k3"].dw_conv.weight.grad.abs().sum() > 0


def test_constant_E0_onehot_skips_pan_and_constructor_preserves_CPU_rng():
    before = torch.random.get_rng_state().clone()
    model = ElasticNeckSupernet(seed=42).eval()
    assert torch.equal(before, torch.random.get_rng_state())
    called = []
    handles = [module.register_forward_hook(lambda *_args: called.append(True))
               for module in model.neck.bottom_up_blocks.values()]
    with torch.no_grad():
        model(torch.zeros(1, 1, 640, 640), one_hot_weights(_arch(False)))
    for handle in handles:
        handle.remove()
    assert not called
    with pytest.raises(ValueError, match="640"):
        model(torch.zeros(1, 1, 320, 320), one_hot_weights(_arch(False)))
    bad = one_hot_weights(_arch(True))
    bad["td_kernel_weights"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="sum to one"):
        model(torch.zeros(1, 1, 640, 640), bad)

