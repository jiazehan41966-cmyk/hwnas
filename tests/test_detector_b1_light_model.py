"""B1 preservation, paired initialization and real trainable detector behavior."""
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

from tools.detector_b1_light.model import (
    B1_ARCH, B1_SPEC_PATH, INITIALIZATION_POLICY, SPEC_PATH, build_model, load_spec,
)
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.training.detection_trainer import (
    DenseAnchorDetectionLoss, _assign_targets, decode_detection_predictions,
)

BASELINE = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))
EXPECTED = {"L1": (436220, 797446400, 1), "L0": (424508, 704422400, 0)}
FROZEN_SHA256 = {
    "configs/search_space_sonar_v0_1.yaml": "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4",
    "configs/baseline_sonar_frozen.yaml": "23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e",
    "src/hwnas_fpga/models/sonar_candidate/__init__.py": "528ee2ed45682de90971ee05e69a30d7347422b18aeea9fab0148c94b2453e62",
    "src/hwnas_fpga/models/detection.py": "6e8e4cd62a0e0148ebb03a2edbafce88d43f2ac66a95135e8171ba8a912e956a",
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
def test_backbone_exact_legacy_B1_modules_initial_weights_and_features(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        legacy = build_sonar_candidate(B1_SPEC_PATH, deepcopy(B1_ARCH)).eval()
    model = build_model("L1", seed).eval()
    assert model.architecture == legacy.architecture
    assert model.levels == legacy.levels == ("P2", "P3", "P4", "P5")
    assert model.feature_indices == legacy.feature_indices == (1, 3, 5, 7)
    assert model.feature_strides == legacy.feature_strides == (4, 8, 16, 32)
    assert model.anchor_sizes == legacy.anchor_sizes
    assert [(name, type(module), module.extra_repr())
            for name, module in model.backbone.named_modules()] == [
        (name, type(module), module.extra_repr())
        for name, module in legacy.backbone.named_modules()]
    _assert_state_equal(model.backbone.state_dict(), legacy.backbone.state_dict())
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(2101)
        image = torch.rand(1, 1, 64, 64)
        for actual, expected in zip(model.backbone(image), legacy.backbone(image)):
            assert torch.equal(actual, expected)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_paired_shared_tensor_identity_and_only_one_extra_DS_tower(seed):
    direct = build_model("L0", seed)
    ds = build_model("L1", seed)
    _assert_state_equal(direct.backbone.state_dict(), ds.backbone.state_dict())
    _assert_state_equal(direct.neck.state_dict(), ds.neck.state_dict())
    for direct_tower, ds_tower in zip(direct.head.towers, ds.head.towers):
        assert len(direct_tower) == 1 and len(ds_tower) == 7
        _assert_state_equal(direct_tower[-1].state_dict(), ds_tower[-1].state_dict())
        assert isinstance(ds_tower[0], nn.Conv2d)
        assert ds_tower[0].groups == ds_tower[0].in_channels == 48
        assert ds_tower[0].kernel_size == (3, 3)
        assert ds_tower[3].kernel_size == (1, 1)
    first, second = direct.initialization_identity(), ds.initialization_identity()
    assert first["policy"] == second["policy"] == INITIALIZATION_POLICY
    for field in ("backbone_sha256", "neck_sha256", "prediction_sha256"):
        assert first[field] == second[field]
    assert first["tower_sha256"] != second["tower_sha256"]
    assert sum(p.numel() for p in ds.parameters()) - sum(p.numel() for p in direct.parameters()) == 11712


@pytest.mark.parametrize("prototype", EXPECTED)
def test_construct_preserves_global_CPU_rng_and_does_not_seed_CUDA(prototype, monkeypatch):
    def forbidden_cuda_seed(*_args, **_kwargs):
        raise AssertionError("CPU model construction must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda_seed)
    before = torch.random.get_rng_state().clone()
    build_model(prototype, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_identity_is_construction_time_evidence_and_not_mutable_by_caller():
    model = build_model()
    expected = model.initialization_identity()
    returned = model.initialization_identity()
    returned["backbone_sha256"] = "caller-change"
    with torch.no_grad():
        model.backbone.stem.conv.conv.weight.add_(1.0)
    assert model.initialization_identity() == expected


def test_seeds_change_initial_weights_while_module_policy_stays_fixed():
    first, second = build_model(seed=42), build_model(seed=43)
    for field in ("backbone_sha256", "neck_sha256", "prediction_sha256", "tower_sha256"):
        assert first.initialization_identity()[field] != second.initialization_identity()[field]


@pytest.mark.parametrize("prototype", EXPECTED)
def test_meta_shapes_full_B1_kernels_and_independent_MAC_count(prototype):
    parameters, expected_macs, extra = EXPECTED[prototype]
    with torch.device("meta"):
        model = build_model(prototype).eval()
    counts = []
    backbone_counts = []

    def record(name, conv, _inputs, output):
        macs = output.numel() * (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
        counts.append(macs)
        if name.startswith("backbone."):
            backbone_counts.append(macs)

    handles = [module.register_forward_hook(
        lambda module, inputs, output, name=name: record(name, module, inputs, output)
    ) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    result = model(torch.empty(1, 1, 640, 640, device="meta"))
    for handle in handles:
        handle.remove()
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert sum(counts) == expected_macs
    assert sum(backbone_counts) == 524288000
    assert len(model.backbone.stages) == 8
    for index, kernel in [(1, 3), (2, 5), (4, 5), (5, 5), (6, 5), (7, 5)]:
        assert model.backbone.stages[index][0].dw_conv.kernel_size == (kernel, kernel)
    for channels, stride, back, neck, prediction in zip(
        [24, 32, 96, 320], [4, 8, 16, 32],
        result["backbone_features"], result["neck_features"], result["predictions"],
    ):
        spatial = 640 // stride
        assert tuple(back.shape) == (1, channels, spatial, spatial)
        assert tuple(neck.shape) == (1, 48, spatial, spatial)
        assert tuple(prediction.shape) == (1, 21, spatial, spatial)
    contract = model.architecture_contract()
    assert contract["physical_backbone_stage_count"] == 8
    assert contract["backbone_architecture"] == B1_ARCH
    assert contract["extra_DS_tower_blocks"] == extra
    assert contract["board_latency_status"] == "NOT_MEASURED"
    with pytest.raises(ValueError, match="Meta"):
        model.initialization_identity()


@pytest.mark.parametrize("prototype", EXPECTED)
def test_exact_B1_loss_anchor_assignment_decode_and_linear_bias(prototype):
    model = build_model(prototype)
    with torch.random.fork_rng(devices=[]):
        legacy = build_sonar_candidate(B1_SPEC_PATH, deepcopy(B1_ARCH))
    cfg = model.loss_config(BASELINE)
    assert cfg == legacy.loss_config(BASELINE)
    raw = [torch.zeros(1, 21, 640 // stride, 640 // stride) for stride in cfg.strides]
    target = [dict(boxes=torch.tensor([[99.0, 99.0, 101.0, 101.0]]), labels=torch.tensor([0]))]
    assignment = _assign_targets(raw, target, cfg, device=torch.device("cpu"))
    assert (assignment[0]["objectness"] > 0.5).any()
    assert all(not (part["objectness"] > 0.5).any() for part in assignment[1:])
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
    for tower in model.head.towers:
        prediction = tower[-1]
        assert isinstance(prediction, nn.Conv2d) and prediction.kernel_size == (1, 1)
        bias = prediction.bias.detach().reshape(3, 7)
        torch.testing.assert_close(bias[:, 4], torch.full((3,), math.log(0.01 / 0.99)))
        torch.testing.assert_close(bias[:, 5:], torch.zeros(3, 2))
        with torch.no_grad():
            prediction.weight.zero_()
            prediction.bias.fill_(-2.0)
        torch.testing.assert_close(tower(torch.ones(1, 48, 4, 4)),
                                   torch.full((1, 21, 4, 4), -2.0))


@pytest.mark.parametrize("prototype", EXPECTED)
def test_real_640_FP32_loss_backward_all_backbone_and_head_trainable(prototype):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261003)
        model = build_model(prototype).train()
        initial = model.initialization_identity()
        outputs = model(torch.rand(1, 1, 640, 640))
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
        for index in [1, 2, 4, 5, 6, 7]:
            assert model.backbone.stages[index][0].dw_conv.weight.grad.abs().sum() > 0
        assert model.backbone.stem.conv.conv.weight.grad.abs().sum() > 0
        if prototype == "L1":
            assert model.head.towers[0][0].weight.grad.abs().sum() > 0
        assert model.initialization_identity() == initial


@pytest.mark.parametrize("kwargs", [dict(prototype="L2"), dict(prototype=None),
                                    dict(seed=41), dict(seed=42.0), dict(seed=True)])
def test_unregistered_prototype_or_seed_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)


@pytest.mark.parametrize("section,key,value", [
    ("backbone", "weights", "frozen"), ("neck", "channels", 96),
    ("head", "prediction_activation", "relu"), ("initialization", "seeds", [42]),
    ("features", "levels", ["P2", "P3", "P4"]),
])
def test_unregistered_spec_change_rejected(section, key, value):
    spec = load_spec()
    spec[section][key] = value
    with pytest.raises(ValueError, match="Specification differs"):
        load_spec(spec)


def test_spec_type_alias_unknown_field_and_bad_input_rejected():
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


def test_frozen_V0_1_sources_and_configs_unchanged():
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
