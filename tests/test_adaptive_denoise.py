"""AdaptiveDenoiseBlock (denoise_v2, Lee 式门控) 的单元测试。"""

from __future__ import annotations

import pytest
import torch

from hwnas_fpga.models.builder import AdaptiveDenoiseBlock, build_block
from hwnas_fpga.search_space import BlockSpec


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "in_channels,out_channels,stride",
    [
        (8, 8, 1),  # 残差路径
        (8, 12, 1),  # 无残差（通道不匹配）
        (8, 8, 2),  # 无残差（stride=2）
    ],
)
def test_output_shape(in_channels, out_channels, stride, kernel_size):
    block = AdaptiveDenoiseBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    x = torch.randn(2, in_channels, 17, 17)  # 奇数尺寸检验 stride 网格对齐
    out = block(x)
    expected_hw = (17 + stride - 1) // stride
    assert out.shape == (2, out_channels, expected_hw, expected_hw)


def test_gradients_flow_to_gate_and_smooth():
    block = AdaptiveDenoiseBlock(4, 4, kernel_size=3, stride=1)
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    block(x).sum().backward()
    assert block.gate_alpha.grad is not None and block.gate_alpha.grad.abs().sum() > 0
    assert block.gate_beta.grad is not None and block.gate_beta.grad.abs().sum() > 0
    assert block.smooth_weight.grad is not None and block.smooth_weight.grad.abs().sum() > 0
    assert x.grad is not None


def test_adaptive_smooth_denoises_flat_and_preserves_edge():
    """Lee 门控的核心性质：均匀区方差下降，阶跃边缘幅度保留优于纯平滑。"""
    torch.manual_seed(0)
    block = AdaptiveDenoiseBlock(1, 1, kernel_size=5, stride=1)
    block.eval()

    # 左半 0.2、右半 0.8 的阶跃 + 斑点状乘性噪声
    clean = torch.full((1, 1, 64, 64), 0.2)
    clean[..., 32:] = 0.8
    noisy = (clean * (1.0 + 0.3 * torch.randn_like(clean))).clamp(0.0, 1.0)

    with torch.no_grad():
        lee = block.adaptive_smooth(noisy)
        # 纯平滑参照：把门控完全关掉（g→0 即输出 = 局部均值 mu）
        block.gate_beta.fill_(-20.0)
        mu_only = block.adaptive_smooth(noisy)

    # 1) 均匀区（远离边缘的左侧内部）噪声方差显著下降
    flat_in = noisy[..., 8:24, 8:24].var()
    flat_lee = lee[..., 8:24, 8:24].var()
    assert flat_lee < flat_in * 0.7

    # 2) 阶跃幅度：Lee 输出的边缘落差保留应优于纯局部均值
    def edge_amplitude(t: torch.Tensor) -> float:
        left = t[..., 16:48, 28:31].mean()
        right = t[..., 16:48, 34:37].mean()
        return float((right - left).abs())

    assert edge_amplitude(lee) > edge_amplitude(mu_only)
    # 且不低于干净阶跃幅度的 80%
    assert edge_amplitude(lee) >= 0.8 * edge_amplitude(clean)


def test_build_block_registration():
    spec = BlockSpec(op="adaptive_denoise", kernel_size=3, expand_ratio=1, stride=1)
    block = build_block(spec, in_channels=8, out_channels=8)
    assert isinstance(block, AdaptiveDenoiseBlock)
    out = block(torch.randn(1, 8, 16, 16))
    assert out.shape == (1, 8, 16, 16)


def test_gate_monotonic_in_residual_magnitude():
    """|d| 越大门开得越大（保边机制的单调性）。"""
    block = AdaptiveDenoiseBlock(1, 1, kernel_size=3, stride=1)
    alpha = block.gate_alpha.detach()
    beta = block.gate_beta.detach()
    small = torch.sigmoid(alpha * 0.05 + beta)
    large = torch.sigmoid(alpha * 0.5 + beta)
    assert (large > small).all()
