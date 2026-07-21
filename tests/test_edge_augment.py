"""EdgeAugmentBlock (edge_v2, 信息保留加性边缘) 的单元测试。"""

from __future__ import annotations

import pytest
import torch

from hwnas_fpga.models.builder import EdgeAugmentBlock, DepthwiseConvBlock, build_block
from hwnas_fpga.search_space import BlockSpec


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "in_channels,out_channels,stride",
    [
        (8, 8, 1),   # 残差路径
        (8, 12, 1),  # 无残差（通道不匹配）
        (8, 8, 2),   # 无残差（stride=2）
    ],
)
def test_output_shape(in_channels, out_channels, stride, kernel_size):
    block = EdgeAugmentBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    x = torch.randn(2, in_channels, 17, 17)
    out = block(x)
    expected_hw = (17 + stride - 1) // stride
    assert out.shape == (2, out_channels, expected_hw, expected_hw)


def test_gamma_zero_init_makes_edge_contribution_vanish():
    """初始化时 edge_gamma=0，块等价于 DW+BN→PW→BN(→残差)→ReLU，边缘旁支无贡献。"""
    torch.manual_seed(0)
    block = EdgeAugmentBlock(8, 8, kernel_size=3, stride=1).eval()
    assert float(block.edge_gamma) == 0.0

    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        out_full = block(x)
        # 手动去掉边缘旁支：只走主路
        feat = block.dw_bn(block.dw_conv(x))
        main = block.bn(block.pw_conv(feat))
        main = main + x  # residual
        main = block.relu(main)
    assert torch.allclose(out_full, main, atol=1e-6), "gamma=0 时边缘旁支不应有任何贡献"


def test_edge_becomes_active_when_gamma_nonzero():
    """gamma≠0 时边缘旁支产生实际贡献（区别于纯 dw_pw）。"""
    torch.manual_seed(0)
    block = EdgeAugmentBlock(8, 8, kernel_size=3, stride=1).eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        out0 = block(x).clone()
        block.edge_gamma.fill_(1.0)
        out1 = block(x)
    assert not torch.allclose(out0, out1), "gamma 抬起后输出应改变（边缘信息注入）"


def test_gradients_flow_to_gamma_and_grad_convs():
    block = EdgeAugmentBlock(4, 4, kernel_size=3, stride=1)
    block.edge_gamma.data.fill_(0.5)  # 让边缘旁支有梯度通路
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    block(x).sum().backward()
    assert block.edge_gamma.grad is not None and block.edge_gamma.grad.abs().item() > 0
    assert block.grad_convs[0].weight.grad is not None
    assert block.grad_convs[0].weight.grad.abs().sum() > 0
    assert x.grad is not None


def test_information_preserving_vs_old_edge():
    """信息保留性质：输入全为常数强度时，主路仍保留该强度（梯度旁支为 0）。

    旧 EdgeAwareBlock 对常数输入输出恒为 0（丢弃 DC）；新块保留强度信息。
    """
    torch.manual_seed(0)
    block = EdgeAugmentBlock(8, 8, kernel_size=3, stride=1).eval()
    const = torch.full((1, 8, 16, 16), 0.7)
    with torch.no_grad():
        out = block(const)
    # 常数图梯度为 0，输出应来自强度主路 + 残差，不应塌缩为 0
    assert out.abs().mean() > 1e-3, "常数强度输入不应被清零（DC 信息须保留）"


def test_build_block_registration():
    spec = BlockSpec(op="edge_v2", kernel_size=3, expand_ratio=1, stride=1)
    block = build_block(spec, in_channels=8, out_channels=8)
    assert isinstance(block, EdgeAugmentBlock)
    out = block(torch.randn(1, 8, 16, 16))
    assert out.shape == (1, 8, 16, 16)
