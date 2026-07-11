"""推理期重参数化折叠的数值等价测试（eval 模式）。"""

from __future__ import annotations

import pytest
import torch

from hwnas_fpga.deploy.reparam import (
    FoldedDenoiseBlock,
    FoldedEdgeBlock,
    fold_denoise_block,
    fold_edge_block,
    fold_sonar_blocks,
)
from hwnas_fpga.models.builder import DenoiseBlock, EdgeAwareBlock, build_model
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec


def _warm_up_bn(block: torch.nn.Module, in_channels: int, steps: int = 3) -> None:
    """train 模式跑几个 batch，让 BN running stats 变成非平凡值。"""
    block.train()
    for _ in range(steps):
        block(torch.randn(4, in_channels, 16, 16))
    block.eval()


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "in_channels,out_channels,stride",
    [
        (8, 8, 1),  # 残差路径
        (8, 12, 1),  # 无残差（通道不匹配）
        (8, 8, 2),  # 无残差（stride=2）
    ],
)
def test_fold_denoise_block_matches_original(in_channels, out_channels, stride, kernel_size):
    torch.manual_seed(0)
    block = DenoiseBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    _warm_up_bn(block, in_channels)

    folded = fold_denoise_block(block)
    assert isinstance(folded, FoldedDenoiseBlock)
    assert folded.use_residual == block.use_residual

    x = torch.randn(2, in_channels, 16, 16)
    with torch.no_grad():
        expected = block(x)
        actual = folded(x)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
        f"max abs diff {(actual - expected).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "in_channels,out_channels,stride",
    [
        (8, 8, 1),
        (8, 12, 1),
        (8, 8, 2),
    ],
)
def test_fold_edge_block_matches_original(in_channels, out_channels, stride, kernel_size):
    torch.manual_seed(0)
    block = EdgeAwareBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    _warm_up_bn(block, in_channels)

    folded = fold_edge_block(block)
    assert isinstance(folded, FoldedEdgeBlock)
    assert folded.use_residual == block.use_residual

    x = torch.randn(2, in_channels, 16, 16)
    with torch.no_grad():
        expected = block(x)
        actual = folded(x)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
        f"max abs diff {(actual - expected).abs().max().item():.3e}"
    )


def test_fold_sonar_blocks_replaces_and_preserves_model_output():
    torch.manual_seed(0)
    architecture = ArchitectureSpec(
        input_channels=1,
        stem_channels=16,
        stem_stride=2,
        stages=(
            StageSpec(
                channels=16,
                depth=2,
                stride=1,
                blocks=(
                    BlockSpec(op="denoise", kernel_size=3, expand_ratio=1, stride=1),
                    BlockSpec(op="edge", kernel_size=3, expand_ratio=1, stride=1),
                ),
            ),
            StageSpec(
                channels=24,
                depth=1,
                stride=2,
                blocks=(
                    BlockSpec(op="dw_pw_conv", kernel_size=3, expand_ratio=1, stride=2),
                ),
            ),
        ),
    )
    model = build_model(architecture, num_classes=4)
    model.train()
    for _ in range(2):
        model(torch.randn(2, 1, 32, 32))
    model.eval()

    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        expected = model(x)

    replaced = fold_sonar_blocks(model)
    assert replaced == 2
    assert not any(isinstance(m, (DenoiseBlock, EdgeAwareBlock)) for m in model.modules())

    with torch.no_grad():
        actual = model(x)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_folded_denoise_is_plain_dw_pw():
    """折叠后的 denoise 部署成本应等同标准 dw_pw：只含两个普通卷积。"""
    block = DenoiseBlock(8, 8, kernel_size=3, stride=1)
    block.eval()
    folded = fold_denoise_block(block)
    convs = [m for m in folded.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 2
    assert convs[0].groups == 8  # depthwise
    assert convs[1].kernel_size == (1, 1)  # pointwise
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())


def test_folded_edge_is_single_dense_conv():
    """折叠后的 edge 是单个标准稠密卷积，HLS 侧无需 4 分支结构。"""
    block = EdgeAwareBlock(8, 12, kernel_size=3, stride=1)
    block.eval()
    folded = fold_edge_block(block)
    convs = [m for m in folded.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 1
    assert convs[0].groups == 1
    assert convs[0].weight.shape == (12, 8, 3, 3)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())


def test_folding_rejects_training_mode():
    block = DenoiseBlock(8, 8, kernel_size=3, stride=1).train()
    with pytest.raises(RuntimeError, match="requires eval mode"):
        fold_denoise_block(block)
