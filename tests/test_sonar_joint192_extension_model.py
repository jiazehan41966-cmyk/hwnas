"""CPU correctness checks for the isolated 2026-10-13 controlled extension."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hwnas_fpga.models.builder import MBConvBlock, DMDCConvSonarBlock, CFEMSonarBlock
from hwnas_fpga.models.sonar_op import ReparamDualRFDepthwise
from hwnas_fpga.models.sonar_operator_pool_p0 import FrequencyLiteDepthwise
from tools.sonar_joint192_validation.model import build_model as build_base, BlurDownMBConv
from tools.sonar_joint192_extension_20261013.model import (
    REFERENCE_ARCHITECTURES, Joint192ExtensionDetector, DMDCChannelAdapted,
    CFEMChannelAdapted, P2SourceFusion, architecture_hash, validate_architecture,
    build_model, profile_model,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _arch(prototype="E0", *, blur=False, **changes):
    architecture = deepcopy(REFERENCE_ARCHITECTURES[prototype])
    if blur:
        architecture["base_architecture"]["backbone"]["stage1"]["downsample"] = "blur3_after_dense_dw"
    architecture["changes"].update(changes)
    return architecture


def _assert_same_state(first, second):
    left, right = first.state_dict(), second.state_dict()
    assert set(left) == set(right)
    assert all(torch.equal(left[name], right[name]) for name in left)


@pytest.mark.parametrize("prototype,blur", [("E0", False), ("E1", True)])
def test_unchanged_reference_bit_exact_state_features_and_predictions(prototype, blur):
    architecture = _arch(prototype, blur=blur)
    reference = build_base(architecture["base_architecture"], seed=42).eval()
    extension = build_model(architecture, seed=42).eval()
    _assert_same_state(reference, extension)
    inputs = torch.linspace(0, 1, 640 * 640).reshape(1, 1, 640, 640)
    with torch.no_grad():
        expected, actual = reference(inputs), extension(inputs)
    assert set(expected) == set(actual)
    for name in expected:
        assert all(torch.equal(left, right) for left, right in zip(expected[name], actual[name]))
    assert architecture_hash(architecture) == architecture_hash(validate_architecture(architecture))
    contract = extension.architecture_contract()
    assert contract["feature_strides"] == [4, 8, 16, 32]
    assert contract["hardware_admission"] is False


@pytest.mark.parametrize("operation", ["sonarop", "frequencylite"])
@pytest.mark.parametrize("stage,blur", [(1, False), (1, True), (2, False)])
def test_stage_spatial_preserves_stride_blur_and_outer_weights(operation, stage, blur):
    architecture = _arch(blur=blur, **{f"stage{stage}_spatial": operation})
    model = build_model(architecture, 42).train()
    reference = build_base(architecture["base_architecture"], 42)
    block, old = model.backbone.stages[stage][0], reference.backbone.stages[stage][0]
    for name in ("expand_conv", "expand_bn", "project_conv", "project_bn"):
        _assert_same_state(getattr(block, name), getattr(old, name))
    assert block.use_residual == old.use_residual
    assert isinstance(block, BlurDownMBConv) == (stage == 1 and blur)
    if blur:
        assert torch.equal(block.blur_kernel, old.blur_kernel)
        assert block.dw_conv.stride == 1
    else:
        assert block.dw_conv.stride == 2
    if operation == "sonarop":
        assert isinstance(block.dw_bn, nn.Identity) and isinstance(block.dw_relu, nn.Identity)
        branch = block.dw_conv.branch3 if old.dw_conv.kernel_size == (3, 3) else block.dw_conv.branch5
        _assert_same_state(branch[0], old.dw_conv)
        _assert_same_state(branch[1], old.dw_bn)
    else:
        _assert_same_state(block.dw_bn, old.dw_bn)
        assert isinstance(block.dw_relu, nn.ReLU6)
        if old.dw_conv.kernel_size == (3, 3):
            _assert_same_state(block.dw_conv.high_dw, old.dw_conv)
    value = torch.randn(2, old.expand_conv.in_channels, 17, 19, requires_grad=True)
    result = block(value)
    assert result.shape == (2, old.project_conv.out_channels, 9, 10)
    result.square().mean().backward()
    assert torch.isfinite(value.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.parametrize("operation", ["keep", "identity", "mbconv_k3", "mbconv_k5",
                                       "sonarop", "frequencylite", "dmdc48", "cfem48"])
def test_p2_blocks_small_shape_backward_and_no_dead_replaced_parameters(operation):
    model = build_model(_arch(p2_op=operation), 42).train()
    block = model.neck.top_down_blocks["P2"]
    if operation in ("sonarop", "frequencylite"):
        reference = build_base(_arch()["base_architecture"], 42).neck.top_down_blocks["P2"]
        assert block.use_residual
        for name in ("expand_conv", "expand_bn", "project_conv", "project_bn"):
            _assert_same_state(getattr(block, name), getattr(reference, name))
    inputs = torch.randn(2, 48, 9, 11, requires_grad=True)
    outputs = block(inputs)
    assert outputs.shape == inputs.shape and torch.isfinite(outputs).all()
    outputs.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    if operation == "identity":
        assert outputs is inputs
        assert not list(block.parameters())
        assert not any(name.startswith("neck.top_down_blocks.P2.") for name, _ in model.named_parameters())


@pytest.mark.parametrize("old_type,new_type", [
    (DMDCConvSonarBlock, DMDCChannelAdapted), (CFEMSonarBlock, CFEMChannelAdapted)])
def test_channel_adaptation_exact32_reference_and_unchanged48_core(old_type, new_type):
    original = old_type(32, 32, 1).eval()
    adapted = new_type(32).eval()
    adapted.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 32, 9, 11)
    with torch.no_grad():
        assert torch.equal(original(inputs), adapted(inputs))
    widened = new_type(48)
    original_modules, widened_modules = dict(original.named_modules()), dict(widened.named_modules())
    assert set(original_modules) == set(widened_modules)
    for name in original_modules:
        if not name:
            continue  # The wrapper is deliberately a distinct class.
        first, second = original_modules[name], widened_modules[name]
        assert type(first) is type(second), name
        if isinstance(first, nn.Conv2d):
            assert (first.kernel_size, first.stride, first.padding, first.dilation,
                    first.bias is None) == (second.kernel_size, second.stride,
                                           second.padding, second.dilation, second.bias is None)
        if isinstance(first, nn.BatchNorm2d):
            assert (first.eps, first.momentum) == (second.eps, second.momentum)
    assert widened.dilation_rates == original.dilation_rates
    with pytest.raises(ValueError, match="32->32"):
        old_type(48, 48, 1)


class _Scale(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, value):
        return self.amount * value


@pytest.mark.parametrize("sources,expected", [
    ("lateral_only", 1.), ("adjacent", 6631.), ("direct_p4", 8831.)])
def test_p2_routes_use_preserved_L2_and_postblock_T3_T4(sources, expected):
    model = build_model(_arch(p2_op="identity", p2_inputs=sources), 42)
    neck = model.neck
    neck.lateral_convs = nn.ModuleDict({level: nn.Identity() for level in neck.levels})
    neck.top_down_blocks["P4"] = _Scale(2.)
    neck.top_down_blocks["P3"] = _Scale(3.)
    values = [torch.full((1, 48, size, size), value)
              for size, value in zip((16, 8, 4, 2), (1., 10., 100., 1000.))]
    outputs = neck(values)
    assert torch.equal(outputs[0], torch.full_like(outputs[0], expected))
    assert outputs[1][0, 0, 0, 0] == 6630
    assert outputs[2][0, 0, 0, 0] == 2200
    assert not list(neck.p2_fusion.parameters())


@pytest.mark.parametrize("sources", [2, 3])
@pytest.mark.parametrize("mode", ["sum", "concat_project", "gated_sum"])
def test_fusions_start_as_sum_and_have_finite_gradients(sources, mode):
    fusion = P2SourceFusion(4, sources, mode)
    inputs = [torch.randn(2, 4, 5, 7, requires_grad=True) for _ in range(sources)]
    expected = inputs[0]
    for value in inputs[1:]:
        expected = expected + value
    actual = fusion(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.square().mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in inputs)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
    if mode == "gated_sum":
        weights = fusion.source_weights(inputs)
        torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)
        assert any(p.grad.abs().sum() > 0 for p in fusion.parameters())
        with torch.no_grad():
            fusion.gates[0].weight.fill_(0.5)
        varied = fusion.source_weights(inputs)
        assert not torch.equal(varied[:, 0, :, :1, :1], varied[:, 0, :, 1:2, 1:2])
        torch.testing.assert_close(varied.sum(dim=1), torch.full_like(varied[:, 0], float(sources)))
    if mode == "concat_project":
        assert fusion.projection.bias is None
        assert not any(isinstance(m, (nn.BatchNorm2d, nn.ReLU6)) for m in fusion.modules())


def test_single_source_identity_and_rejected_dead_modules():
    fusion = P2SourceFusion(48, 1, "sum")
    value = torch.randn(1, 48, 3, 5)
    assert fusion([value]) is value and not list(fusion.parameters())
    for mode in ("concat_project", "gated_sum"):
        with pytest.raises(ValueError):
            build_model(_arch(p2_inputs="lateral_only", p2_fusion=mode))


def test_initialization_reproducible_preserves_RNG_head_lateral_and_unmodified_blocks():
    architecture = _arch(blur=True, stage1_spatial="sonarop", stage2_spatial="frequencylite",
                         p2_op="cfem48", p2_inputs="direct_p4", p2_fusion="gated_sum")
    before = torch.random.get_rng_state().clone()
    first = Joint192ExtensionDetector(architecture, 43)
    assert torch.equal(before, torch.random.get_rng_state())
    second = build_model(architecture, 43)
    assert torch.equal(before, torch.random.get_rng_state())
    _assert_same_state(first, second)
    reference = build_base(architecture["base_architecture"], 43)
    _assert_same_state(first.head, reference.head)
    _assert_same_state(first.neck.lateral_convs, reference.neck.lateral_convs)
    for stage in (0, 3, 4, 5, 6, 7):
        _assert_same_state(first.backbone.stages[stage], reference.backbone.stages[stage])
    for level in ("P3", "P4"):
        _assert_same_state(first.neck.top_down_blocks[level], reference.neck.top_down_blocks[level])
    third = build_model(architecture, 44)
    assert first.initialization_identity()["model_sha256"] != third.initialization_identity()["model_sha256"]


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(unknown=True),
    lambda a: a["changes"].pop("stage1_spatial"),
    lambda a: a["changes"].update(stage1_spatial=True),
    lambda a: a["changes"].update(stage2_spatial="dmdc48"),
    lambda a: a["changes"].update(p2_op="dmdc32"),
    lambda a: a["changes"].update(p2_inputs="all"),
    lambda a: a["changes"].update(p2_fusion="sigmoid"),
    lambda a: a["base_architecture"]["neck"]["td_ops"].update(P2="unknown"),
])
def test_invalid_architectures_fail_closed(mutate):
    architecture = _arch()
    mutate(architecture)
    with pytest.raises(ValueError):
        validate_architecture(architecture)


@pytest.mark.parametrize("base_operation", ["identity", "mbconv_k5"])
@pytest.mark.parametrize("probe", ["sonarop", "frequencylite"])
def test_p2_spatial_requires_original_k3_reference(base_operation, probe):
    architecture = _arch(p2_op=probe)
    architecture["base_architecture"]["neck"]["td_ops"]["P2"] = base_operation
    with pytest.raises(ValueError, match="original mbconv_k3"):
        build_model(architecture)


def test_extended_full640_eval_geometry():
    architecture = _arch("E1", blur=True, stage1_spatial="sonarop", stage2_spatial="frequencylite",
                         p2_op="dmdc48", p2_inputs="direct_p4", p2_fusion="concat_project")
    model = build_model(architecture, 42).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 1, 640, 640))
    for index, (stride, channels) in enumerate(zip((4, 8, 16, 32), (24, 32, 96, 320))):
        spatial = 640 // stride
        assert outputs["backbone_features"][index].shape == (1, channels, spatial, spatial)
        assert outputs["neck_features"][index].shape == (1, 48, spatial, spatial)
        assert outputs["predictions"][index].shape == (1, 21, spatial, spatial)
        assert torch.isfinite(outputs["predictions"][index]).all()
    assert model.architecture_contract()["dmdc_training_requires_batch_at_least_2"]


def test_profile_counts_fixed_convolution_MACs_and_preserves_state_and_RNG():
    frequency = FrequencyLiteDepthwise(3, stride=2)
    before = torch.random.get_rng_state().clone()
    report = profile_model(frequency, (2, 3, 7, 9))
    assert torch.equal(before, torch.random.get_rng_state())
    assert report["conv_macs"] == 2 * 3 * 4 * 5 * 9
    assert report["frequencylite_fixed_macs"] == 9 * (2 * 3 * 7 * 9 + 2 * 3 * 4 * 5)
    assert report["params_total"] == 27
    blur = BlurDownMBConv(2, 4, 3, expand_ratio=3)
    report = profile_model(blur, (2, 2, 7, 9))
    assert report["fixed_blur_macs"] == 2 * 6 * 4 * 5 * 9
    assert report["macs_total"] == report["conv_macs"] + report["functional_macs"]
    assert "interpolation" in report["excluded_operations"]


def test_profile_train_deploy_SonarOp_and_fusion_parity():
    model = build_model(_arch(blur=True, stage1_spatial="sonarop"), 42)
    operator = model.backbone.stages[1][0].dw_conv.eval()
    state = deepcopy(operator.state_dict())
    training = profile_model(operator, (2, 48, 9, 11))
    deployed = profile_model(operator, (2, 48, 9, 11), deploy=True)
    assert training["params_total"] == 38 * 48
    assert deployed["params_total"] == 26 * 48
    assert training["conv_macs"] == 2 * 48 * 9 * 11 * 34
    assert deployed["conv_macs"] == 2 * 48 * 9 * 11 * 25
    assert set(operator.state_dict()) == set(state)
    assert all(torch.equal(operator.state_dict()[name], value) for name, value in state.items())
    folded = deepcopy(operator).switch_to_deploy()
    value = torch.randn(2, 48, 9, 11)
    with torch.no_grad():
        torch.testing.assert_close(operator(value), folded(value), rtol=1e-5, atol=1e-5)

