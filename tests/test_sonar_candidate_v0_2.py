"""V0.2 contracts, true truncation, computation geometry and numerical API."""
import hashlib
import math
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from hwnas_fpga.models.sonar_candidate_v0_2 import (
    SPEC_PATH, LightweightFPNNeck, architecture_id,
    build_sonar_candidate_v0_2, load_spec, validate_sonar_architecture_v0_2,
)
from hwnas_fpga.training.detection_trainer import (
    DenseAnchorDetectionLoss, _assign_targets, decode_detection_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))
EXPECTED = {
    "R0": (63975, 444825600, 6, 32, 0),
    "R1": (71975, 519526400, 6, 48, 0),
    "R7": (406460, 684761600, 8, 48, 0),
    "R8": (80759, 611456000, 6, 48, 1),
}
FROZEN_HASHES = {
    "configs/search_space_sonar_v0_1.yaml": "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4",
    "configs/baseline_sonar_frozen.yaml": "23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e",
    "src/hwnas_fpga/models/sonar_candidate/__init__.py": "528ee2ed45682de90971ee05e69a30d7347422b18aeea9fab0148c94b2453e62",
    "src/hwnas_fpga/models/detection.py": "6e8e4cd62a0e0148ebb03a2edbafce88d43f2ac66a95135e8171ba8a912e956a",
    "src/hwnas_fpga/models/builder.py": "e24d3592424be37045bcddda3790957759affced03fc529554197b3356062496",
}


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("prototype", EXPECTED)
def test_meta_shapes_physical_truncation_and_independent_MAC_count(prototype):
    params, expected_MACs, stage_count, width, extra_DS = EXPECTED[prototype]
    with torch.device("meta"):
        model = build_sonar_candidate_v0_2(SPEC_PATH, prototype).eval()
    computed_MACs = []

    def record_conv(conv, _inputs, output):
        computed_MACs.append(output.numel() * (conv.in_channels // conv.groups)
                             * conv.kernel_size[0] * conv.kernel_size[1])

    handles = [module.register_forward_hook(record_conv) for module in model.modules()
               if isinstance(module, nn.Conv2d)]
    result = model(torch.empty(1, 1, 640, 640, device="meta"))
    for handle in handles:
        handle.remove()
    assert sum(p.numel() for p in model.parameters()) == params
    assert sum(computed_MACs) == expected_MACs
    assert len(model.backbone.stages) == stage_count
    if stage_count == 6:
        assert not any(key.startswith(("backbone.stages.6.", "backbone.stages.7."))
                       for key in model.state_dict())
    assert model.neck.out_channels == width
    assert model.head.extra_tower_blocks == extra_DS
    assert set(result) == {"backbone_features", "neck_features", "predictions"}
    feature_channels = {"P2": 24, "P3": 32, "P4": 96, "P5": 320}
    for level, stride, backbone, neck, pred in zip(
            model.levels, model.feature_strides,
            result["backbone_features"], result["neck_features"], result["predictions"]):
        size = 640 // stride
        assert tuple(backbone.shape) == (1, feature_channels[level], size, size)
        assert tuple(neck.shape) == (1, width, size, size)
        assert tuple(pred.shape) == (1, 21, size, size)
    for stage in model.backbone.stages:
        block = stage[0]
        if hasattr(block, "dw_conv"):
            assert block.dw_conv.kernel_size == (3, 3)
            assert block.expand_conv.out_channels == 3 * block.expand_conv.in_channels
    assert architecture_id(prototype) == model.prototype_name


@pytest.mark.parametrize("prototype", EXPECTED)
def test_linear_prediction_bias_and_preserved_stride_anchors(prototype):
    model = build_sonar_candidate_v0_2(architecture={"prototype_id": prototype})
    cfg = model.loss_config(BASELINE)
    assert cfg.strides == model.feature_strides
    assert cfg.anchor_sizes == model.anchor_sizes
    assert cfg.class_loss_weights == (1.0, 1.0)
    assert len(model.head.towers) == len(model.levels) == len(model.anchor_sizes)
    for level, tower, anchors in zip(model.levels, model.head.towers, model.anchor_sizes):
        baseline_index = {"P2": 0, "P3": 0, "P4": 1, "P5": 2}[level]
        assert anchors == tuple(map(tuple, BASELINE["detection"]["anchor_sizes"][baseline_index]))
        prediction = tower[-1]
        assert isinstance(prediction, nn.Conv2d)
        assert prediction.kernel_size == (1, 1)
        assert prediction.out_channels == 21 and prediction.bias is not None
        bias = prediction.bias.reshape(3, 7)
        torch.testing.assert_close(bias[:, 4], torch.full((3,), math.log(0.01 / 0.99)))
        torch.testing.assert_close(bias[:, 5:], torch.zeros(3, 2))
        # An explicitly negative final output must remain negative (linear).
        with torch.no_grad():
            prediction.weight.zero_()
            prediction.bias.fill_(-2.0)
        torch.testing.assert_close(tower(torch.ones(1, model.neck.out_channels, 4, 4)),
                                   torch.full((1, 21, 4, 4), -2.0))
        assert len(tower) == (7 if prototype == "R8" else 1)


class _Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x * self.value


def test_top_down_add_uses_unsmoothed_fused_features():
    neck = LightweightFPNNeck([1, 1, 1], 1)
    neck.lateral_convs = nn.ModuleList([nn.Identity() for _ in range(3)])
    neck.output_convs = nn.ModuleList([_Scale(2), _Scale(3), _Scale(5)])
    out = neck([torch.ones(1, 1, 4, 4), torch.full((1, 1, 2, 2), 2.0),
                torch.full((1, 1, 1, 1), 3.0)])
    for output, expected in zip(out, [12.0, 15.0, 15.0]):
        torch.testing.assert_close(output, torch.full_like(output, expected))


@pytest.mark.parametrize("architecture", [None, {}, {"prototype_id": "R2"},
    {"prototype_id": True}, {"prototype_id": "R1", "neck_channels": 64},
    {"prototype_id": "R1", "kernels": {"stage1": 5}}, "R1_P234_U48_direct"])
def test_unknown_or_widened_architecture_is_rejected(architecture):
    with pytest.raises(ValueError):
        validate_sonar_architecture_v0_2(architecture)


@pytest.mark.parametrize("section,field,value", [
    ("head", "prediction_activation", "relu"), ("head", "prediction_channels", 21.0),
    ("neck", "top_down_source", "smoothed"), ("anchors", "recluster", True),
    ("prototypes", "R1", dict(levels=["P3", "P4", "P5"])),
])
def test_modified_spec_rejected(section, field, value):
    spec = load_spec()
    spec["architecture_space"][section][field] = value
    with pytest.raises(ValueError, match="Specification differs"):
        load_spec(spec)


def test_unknown_spec_field_and_type_alias_are_rejected():
    spec = load_spec()
    spec["new_arbitrary_option"] = 1
    with pytest.raises(ValueError):
        load_spec(spec)
    spec = load_spec()
    spec["architecture_space"]["stages"][0]["stride"] = True
    with pytest.raises(ValueError):
        load_spec(spec)


@pytest.mark.parametrize("kwargs", [{"num_classes": 3}, {"num_classes": 2.0},
    {"image_size": 320}, {"image_size": 640.0}])
def test_fixed_input_and_class_constructor_contract(kwargs):
    with pytest.raises(ValueError):
        build_sonar_candidate_v0_2(**kwargs)


def test_bad_input_shape_rejected_before_backbone_compute():
    model = build_sonar_candidate_v0_2()
    with pytest.raises(ValueError, match="640"):
        model(torch.zeros(1, 3, 320, 320))


@pytest.mark.parametrize("prototype", ["R1", "R8"])
def test_real_FP32_loss_backward_and_extra_tower_gradients(prototype):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = build_sonar_candidate_v0_2(architecture=prototype).train()
        images = torch.rand(1, 1, 640, 640)
        targets = [dict(boxes=torch.tensor([[112.0, 112.0, 124.0, 128.0]]),
                        labels=torch.tensor([0]))]
        result = model(images)
        loss, parts = DenseAnchorDetectionLoss(model.loss_config(BASELINE))(result, targets)
        assert torch.isfinite(loss)
        assert all(math.isfinite(float(value)) for value in parts.values())
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert model.backbone.stem.conv.conv.weight.grad.abs().sum() > 0
        if prototype == "R8":
            assert model.head.towers[0][0].weight.grad.abs().sum() > 0


def test_assignment_tie_and_decoded_coordinates_preserve_old_protocol():
    model = build_sonar_candidate_v0_2()
    cfg = model.loss_config(BASELINE)
    raw = [torch.zeros(1, 21, 640 // stride, 640 // stride) for stride in cfg.strides]
    # A 2x2 box is below the positive IoU threshold for all anchors, invoking
    # the best-match fallback; repeated P2/P3 anchors then tie on the first P2.
    target = [dict(boxes=torch.tensor([[99.0, 99.0, 101.0, 101.0]]), labels=torch.tensor([0]))]
    assigned = _assign_targets(raw, target, cfg, device=torch.device("cpu"))
    assert (assigned[0]["objectness"] > 0.5).any()
    assert all(not (a["objectness"] > 0.5).any() for a in assigned[1:])
    for pred in raw:
        pred[:, [4, 11, 18]] = -20.0
    raw[0][0, 4, 10, 10] = 10.0
    raw[0][0, 5, 10, 10] = 10.0
    prediction = decode_detection_predictions(
        raw, config=cfg, score_threshold=0.25, pre_nms_topk=500,
        max_detections=100, nms_iou_threshold=0.5,
    )[0]
    aw, ah = cfg.anchor_sizes[0][0]
    center = 10.5 * cfg.strides[0]
    torch.testing.assert_close(prediction["boxes"][0],
                               torch.tensor([center - aw / 2, center - ah / 2,
                                             center + aw / 2, center + ah / 2]))


def test_no_V0_1_source_or_frozen_config_changes():
    for path, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
