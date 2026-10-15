"""Controlled ELASTIC/PAN-inspired four-scale pilot geometry and numerical tests."""
from copy import deepcopy
import hashlib
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

from tools.detector_elastic_pilot.model import (
    B1_ARCH, B1_SPEC_PATH, INITIALIZATION_POLICY, SPEC_PATH,
    ElasticFourScaleNeck, build_model, load_spec,
)
from hwnas_fpga.models.builder import MBConvBlock
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.training.detection_trainer import (
    DenseAnchorDetectionLoss, _assign_targets, decode_detection_predictions,
)

BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))
EXPECTED = {"E0": (460172, 1119430400, 0), "E1": (514748, 1265792000, 3)}
FROZEN_SHA256 = {
    "configs/search_space_sonar_v0_1.yaml": "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4",
    "configs/baseline_sonar_frozen.yaml": "23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e",
    "src/hwnas_fpga/models/sonar_candidate/__init__.py": "528ee2ed45682de90971ee05e69a30d7347422b18aeea9fab0148c94b2453e62",
    "src/hwnas_fpga/models/detection.py": "6e8e4cd62a0e0148ebb03a2edbafce88d43f2ac66a95135e8171ba8a912e956a",
    "tools/detector_b1_light/model.py": "a2fb31610bd9a889978c1f4f6ed6f2bc5f07452abf9ead158ef8c43606ac245a",
    "configs/detector_b1_light_model.yaml": "fb8f663e2901cd09abd723a738eb36c70c6eaa73b9a65397ef446a922f50fb77",
}


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _assert_state_equal(first, second):
    assert set(first) == set(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_legacy_B1_backbone_and_paired_common_neck_prediction_initialization(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        legacy = build_sonar_candidate(B1_SPEC_PATH, deepcopy(B1_ARCH))
    topdown, pan = build_model("E0", seed), build_model("E1", seed)
    assert topdown.architecture == pan.architecture == legacy.architecture
    _assert_state_equal(topdown.backbone.state_dict(), legacy.backbone.state_dict())
    _assert_state_equal(pan.backbone.state_dict(), legacy.backbone.state_dict())
    assert [(name, type(module), module.extra_repr())
            for name, module in topdown.backbone.named_modules()] == [
        (name, type(module), module.extra_repr())
        for name, module in legacy.backbone.named_modules()]
    _assert_state_equal(topdown.neck.common_state_dict(), pan.neck.common_state_dict())
    _assert_state_equal(topdown.head.state_dict(), pan.head.state_dict())
    first, second = topdown.initialization_identity(), pan.initialization_identity()
    for field in ("backbone_sha256", "common_neck_sha256", "prediction_sha256"):
        assert first[field] == second[field]
    assert first["full_neck_sha256"] != second["full_neck_sha256"]
    assert first["pan_sha256"] != second["pan_sha256"]
    assert first["policy"] == second["policy"] == INITIALIZATION_POLICY
    assert all(p.requires_grad for p in topdown.parameters())
    assert all(p.requires_grad for p in pan.parameters())


@pytest.mark.parametrize("prototype", EXPECTED)
def test_constructor_CPU_rng_restored_without_seeding_CUDA(prototype, monkeypatch):
    def forbidden_cuda_seed(*_args, **_kwargs):
        raise AssertionError("CPU construction must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda_seed)
    before = torch.random.get_rng_state().clone()
    build_model(prototype, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_identity_records_construction_state_and_distinguishes_seeds():
    first, second = build_model("E1", 42), build_model("E1", 43)
    for field in ("backbone_sha256", "common_neck_sha256", "prediction_sha256", "pan_sha256"):
        assert first.initialization_identity()[field] != second.initialization_identity()[field]
    expected = first.initialization_identity()
    returned = first.initialization_identity()
    returned["backbone_sha256"] = "caller-change"
    with torch.no_grad():
        first.backbone.stem.conv.conv.weight.add_(1.0)
    assert first.initialization_identity() == expected


@pytest.mark.parametrize("prototype", EXPECTED)
def test_meta_640_geometry_independent_MACs_and_all_convs_are_executed(prototype):
    parameters, expected_macs, pan_blocks = EXPECTED[prototype]
    with torch.device("meta"):
        model = build_model(prototype).eval()
    counts = {}
    handles = []
    conv_names = {name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(
                lambda conv, inputs, output, name=name: counts.__setitem__(
                    name, output.numel() * (conv.in_channels // conv.groups)
                    * conv.kernel_size[0] * conv.kernel_size[1],
                )
            ))
    result = model(torch.empty(1, 1, 640, 640, device="meta"))
    for handle in handles:
        handle.remove()
    assert set(counts) == conv_names
    assert sum(counts.values()) == expected_macs
    assert sum(value for name, value in counts.items() if name.startswith("backbone.")) == 524288000
    assert sum(value for name, value in counts.items() if name.startswith("head.")) == 34272000
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert len(model.backbone.stages) == 8
    for index, kernel in [(1, 3), (2, 5), (4, 5), (5, 5), (6, 5), (7, 5)]:
        assert model.backbone.stages[index][0].dw_conv.kernel_size == (kernel, kernel)
    for channels, stride, back, neck, pred in zip(
        [24, 32, 96, 320], [4, 8, 16, 32],
        result["backbone_features"], result["neck_features"], result["predictions"],
    ):
        spatial = 640 // stride
        assert tuple(back.shape) == (1, channels, spatial, spatial)
        assert tuple(neck.shape) == (1, 48, spatial, spatial)
        assert tuple(pred.shape) == (1, 21, spatial, spatial)
    assert list(model.neck.top_down_blocks) == ["P2", "P3", "P4"]
    assert len(model.neck.bottom_up_blocks) == pan_blocks
    assert len(model.neck.bottom_up_projections) == pan_blocks
    assert all(len(tower) == 1 for tower in model.head.towers)
    for block in model.neck.top_down_blocks.values():
        assert isinstance(block, MBConvBlock) and block.use_residual
        assert block.expand_conv.out_channels == 3 * block.expand_conv.in_channels
        assert isinstance(block.expand_relu, nn.ReLU6) and isinstance(block.dw_relu, nn.ReLU6)
        assert block.project_conv.kernel_size == (1, 1)
        assert isinstance(block.project_bn, nn.BatchNorm2d)
        assert not hasattr(block, "project_relu")
    contract = model.architecture_contract()
    assert contract["backbone_architecture"] == B1_ARCH
    assert contract["top_down_MBConv_count"] == 3
    assert contract["bottom_up_MBConv_count"] == pan_blocks
    assert contract["top_down_source"] == "post_MBConv_fused_feature"
    assert contract["board_latency_status"] == "NOT_MEASURED"
    assert "not_paper_reproduction" in contract["relation_to_papers"]
    with pytest.raises(ValueError, match="Meta"):
        model.initialization_identity()


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


@pytest.mark.parametrize("pan,expected", [
    (False, [245.0, 48.0, 14.0, 4.0]),
    (True, [245.0, 2051.0, 22715.0, 295347.0]),
])
def test_topdown_uses_post_MBConv_source_and_PAN_is_progressive(pan, expected):
    neck = ElasticFourScaleNeck([1, 1, 1, 1], 1, pan)
    neck.lateral_convs = nn.ModuleDict({level: nn.Identity() for level in neck.levels})
    neck.top_down_blocks = nn.ModuleDict({
        "P2": _Scale(5), "P3": _Scale(3), "P4": _Scale(2),
    })
    if pan:
        neck.bottom_up_projections = nn.ModuleDict({level: nn.Identity() for level in neck.levels[1:]})
        neck.bottom_up_blocks = nn.ModuleDict({"P3": _Scale(7), "P4": _Scale(11), "P5": _Scale(13)})
    outputs = neck([torch.full((1, 1, size, size), float(index + 1))
                    for index, size in enumerate([16, 8, 4, 2])])
    for output, value in zip(outputs, expected):
        torch.testing.assert_close(output, torch.full_like(output, value))


@pytest.mark.parametrize("prototype", EXPECTED)
def test_original_B1_loss_anchors_assignment_decode_and_prediction_prior(prototype):
    model = build_model(prototype)
    with torch.random.fork_rng(devices=[]):
        legacy = build_sonar_candidate(B1_SPEC_PATH, deepcopy(B1_ARCH))
    cfg = model.loss_config(BASELINE)
    assert cfg == legacy.loss_config(BASELINE)
    assert model.levels == legacy.levels == ("P2", "P3", "P4", "P5")
    assert model.feature_strides == legacy.feature_strides == (4, 8, 16, 32)
    assert model.anchor_sizes == legacy.anchor_sizes
    raw = [torch.zeros(1, 21, 640 // stride, 640 // stride) for stride in cfg.strides]
    target = [dict(boxes=torch.tensor([[99.0, 99.0, 101.0, 101.0]]), labels=torch.tensor([0]))]
    assigned = _assign_targets(raw, target, cfg, device=torch.device("cpu"))
    assert (assigned[0]["objectness"] > 0.5).any()
    assert all(not (part["objectness"] > 0.5).any() for part in assigned[1:])
    for pred in raw:
        pred[:, [4, 11, 18]] = -20.0
    raw[0][0, 4, 10, 10] = 10.0
    raw[0][0, 5, 10, 10] = 10.0
    decoded = decode_detection_predictions(
        raw, config=cfg, score_threshold=0.25, pre_nms_topk=500,
        max_detections=100, nms_iou_threshold=0.5,
    )[0]
    aw, ah = cfg.anchor_sizes[0][0]
    center = 10.5 * cfg.strides[0]
    torch.testing.assert_close(decoded["boxes"][0],
                               torch.tensor([center - aw / 2, center - ah / 2,
                                             center + aw / 2, center + ah / 2]))
    for prediction in model.head.prediction_convs:
        bias = prediction.bias.detach().view(3, 7)
        torch.testing.assert_close(bias[:, 4], torch.full((3,), math.log(0.01 / 0.99)))
        torch.testing.assert_close(bias[:, 5:], torch.zeros(3, 2))
        with torch.no_grad():
            prediction.weight.zero_()
            prediction.bias.fill_(-2.0)
        torch.testing.assert_close(prediction(torch.ones(1, 48, 3, 3)),
                                   torch.full((1, 21, 3, 3), -2.0))


@pytest.mark.parametrize("prototype", EXPECTED)
def test_real_640_FP32_loss_backward_and_PAN_branch_gradients(prototype):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261005)
        model = build_model(prototype).train()
        initial = model.initialization_identity()
        pan_calls = []
        handles = [module.register_forward_hook(
            lambda module, inputs, output, name=name: pan_calls.append(name)
        ) for name, module in model.neck.named_modules()
            if name.startswith(("bottom_up_projections.", "bottom_up_blocks."))
            and isinstance(module, nn.Conv2d)]
        outputs = model(torch.rand(1, 1, 640, 640))
        for handle in handles:
            handle.remove()
        target = [dict(boxes=torch.tensor([[112.0, 112.0, 124.0, 128.0]]),
                       labels=torch.tensor([0]))]
        loss, parts = DenseAnchorDetectionLoss(model.loss_config(BASELINE))(outputs, target)
        assert torch.isfinite(loss)
        assert all(math.isfinite(float(value)) for value in parts.values())
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.requires_grad, name
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert model.backbone.stages[7][0].dw_conv.weight.grad.abs().sum() > 0
        for module in model.neck.top_down_blocks.values():
            assert module.dw_conv.weight.grad.abs().sum() > 0
        if prototype == "E1":
            assert len(pan_calls) == 12 and len(set(pan_calls)) == 12
            for module in model.neck.bottom_up_blocks.values():
                assert module.dw_conv.weight.grad.abs().sum() > 0
            for module in model.neck.bottom_up_projections.values():
                assert module[0].weight.grad.abs().sum() > 0
        else:
            assert pan_calls == []
        assert model.initialization_identity() == initial


@pytest.mark.parametrize("kwargs", [dict(prototype="E2"), dict(prototype="L1"),
                                    dict(prototype=None), dict(seed=41),
                                    dict(seed=42.0), dict(seed=True)])
def test_unregistered_prototype_or_seed_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)


@pytest.mark.parametrize("section,key,value", [
    ("backbone", "weights", "frozen"), ("neck", "channels", 96),
    ("head", "prediction_activation", "relu"), ("initialization", "seeds", [42]),
    ("features", "levels", ["P2", "P3", "P4"]),
    ("pilot_search_space", "backbone_options", ["all_144"]),
])
def test_unregistered_spec_changes_rejected(section, key, value):
    spec = load_spec()
    spec[section][key] = value
    with pytest.raises(ValueError, match="Specification differs"):
        load_spec(spec)


def test_spec_type_alias_unknown_field_bad_input_and_wrong_feature_count_rejected():
    assert load_spec(SPEC_PATH) == load_spec()
    spec = load_spec()
    spec["num_classes"] = 2.0
    with pytest.raises(ValueError):
        load_spec(spec)
    spec = load_spec()
    spec["unknown"] = "new"
    with pytest.raises(ValueError):
        load_spec(spec)
    with pytest.raises(ValueError, match="640"):
        build_model()(torch.zeros(1, 3, 320, 320))
    with pytest.raises(ValueError, match="four"):
        ElasticFourScaleNeck([48, 48, 48], 48, False)
    with pytest.raises(ValueError, match="four"):
        build_model().neck([torch.zeros(1, 24, 8, 8)])


def test_old_V0_1_and_B1_light_sources_configs_unchanged():
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
