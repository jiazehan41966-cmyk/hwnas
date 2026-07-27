from __future__ import annotations

import torch

from hwnas_fpga.deploy.mixconv_v2 import (
    MIXCONV_V2_BIAS_LAYOUT,
    MIXCONV_V2_INTEGER_SCHEMA,
    MIXCONV_V2_WEIGHT_LAYOUT,
    build_mixconv_v2_integer_package,
    simulate_mixconv_v2_int8,
)
from hwnas_fpga.deploy.reparam import (
    FoldedMixConvV2Block,
    fold_mixconv_v2_block,
)
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec
from hwnas_fpga.models.builder import MixConvV2Block, build_block
from hwnas_fpga.search_space import BlockSpec, ResolvedBlockSpec


def _warm_up_bn(block: torch.nn.Module, channels: int) -> None:
    block.train()
    for _ in range(3):
        block(torch.randn(4, channels, 11, 13))
    block.eval()


def test_even_and_odd_channel_split_is_deterministic() -> None:
    even = MixConvV2Block(8, 8)
    odd = MixConvV2Block(7, 9)
    assert even.group_sizes == [4, 4]
    assert odd.group_sizes == [3, 4]
    assert even.dw_convs[0].kernel_size == (3, 3)
    assert even.dw_convs[1].kernel_size == (5, 5)
    assert even(torch.randn(2, 8, 9, 11)).shape == (2, 8, 9, 11)
    assert odd(torch.randn(2, 7, 9, 11)).shape == (2, 9, 9, 11)


def test_projection_is_linear_and_residual_preserves_negative_values() -> None:
    block = MixConvV2Block(8, 8).eval()
    with torch.no_grad():
        block.pw_conv.weight.zero_()
        block.pw_bn.weight.fill_(1.0)
        block.pw_bn.bias.zero_()
        block.pw_bn.running_mean.zero_()
        block.pw_bn.running_var.fill_(1.0)
    inputs = torch.linspace(-2.0, 2.0, 8 * 5 * 5).reshape(1, 8, 5, 5)
    outputs = block(inputs)
    assert torch.allclose(outputs, inputs, atol=1e-7, rtol=0.0)
    assert float(outputs.min()) < 0.0


def test_build_block_registers_mixconv_v2() -> None:
    spec = BlockSpec(op="mixconv_v2", kernel_size=3, expand_ratio=1, stride=1)
    assert isinstance(build_block(spec, in_channels=8, out_channels=8), MixConvV2Block)


def test_bn_folding_matches_eval_graph() -> None:
    torch.manual_seed(17)
    block = MixConvV2Block(7, 9, stride=1)
    _warm_up_bn(block, 7)
    folded = fold_mixconv_v2_block(block)
    inputs = torch.randn(2, 7, 11, 13)
    with torch.no_grad():
        expected = block(inputs)
        actual = folded(inputs)
    assert isinstance(folded, FoldedMixConvV2Block)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in folded.modules())


def test_integer_package_has_canonical_weight_and_bias_order() -> None:
    torch.manual_seed(23)
    block = fold_mixconv_v2_block(MixConvV2Block(7, 9).eval())
    package = build_mixconv_v2_integer_package(
        block,
        input_scale=0.05,
        branch_output_scale=0.04,
        output_scale=0.03,
    )
    assert package["schema"] == MIXCONV_V2_INTEGER_SCHEMA
    assert package["weight_layout"] == MIXCONV_V2_WEIGHT_LAYOUT
    assert package["bias_layout"] == MIXCONV_V2_BIAS_LAYOUT
    expected_weights = torch.cat(
        [
            package["dw_weights"][0].reshape(-1),
            package["dw_weights"][1].reshape(-1),
            package["pw_weight"].reshape(-1),
        ]
    )
    expected_biases = torch.cat(
        [package["dw_biases"][0], package["dw_biases"][1], package["pw_bias"]]
    )
    assert torch.equal(package["weights_flat"], expected_weights)
    assert torch.equal(package["biases_flat"], expected_biases)
    assert package["weights_flat"].numel() == 3 * 9 + 4 * 25 + 9 * 7
    assert package["biases_flat"].numel() == 7 + 9


def test_integer_reference_applies_projection_before_residual() -> None:
    block = fold_mixconv_v2_block(MixConvV2Block(8, 8).eval())
    with torch.no_grad():
        for conv in block.dw_convs:
            conv.weight.zero_()
            conv.bias.zero_()
        block.pw_conv.weight.zero_()
        block.pw_conv.bias.zero_()
    package = build_mixconv_v2_integer_package(
        block,
        input_scale=0.05,
        branch_output_scale=0.05,
        output_scale=0.05,
    )
    boundary = torch.tensor([-127, -1, 0, 1, 127], dtype=torch.int8)
    inputs = boundary.repeat(8 * 5 * 5 // 5).reshape(1, 8, 5, 5)
    outputs = simulate_mixconv_v2_int8(inputs, package)
    assert torch.equal(outputs, inputs)


def test_cost_model_matches_exact_split_and_macs() -> None:
    estimator = FPGACostEstimator(HardwareSpec(name="test", clock_mhz=200))
    block = ResolvedBlockSpec(
        stage_index=3,
        block_index=0,
        op="mixconv_v2",
        kernel_size=3,
        expand_ratio=1,
        stride=1,
        in_channels=32,
        out_channels=32,
        input_resolution=28,
        output_resolution=28,
    )
    cost = estimator._estimate_block(block)
    expected_params = 16 * 3 * 3 + 16 * 5 * 5 + 32 * 32
    assert cost.params == expected_params
    assert cost.macs == expected_params * 28 * 28
