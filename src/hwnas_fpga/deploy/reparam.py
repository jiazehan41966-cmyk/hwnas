"""推理期结构重参数化：把声呐先验块折叠为标准卷积。

训练期 DenoiseBlock / EdgeAwareBlock 保留可学习先验结构（双分支平滑、
4 方向 Sobel），部署期通过本模块折叠为普通 depthwise / dense 卷积，
使 HLS 端只需实现标准算子即可与 PyTorch 位级对齐：

- DenoiseBlock: DW特征分支(含BN) + softmax平滑分支 在 eval 模式下都是
  对输入的线性算子，且 kernel/stride/groups 完全相同，可合并为单个带
  bias 的 depthwise 卷积；PW 后的 BN 同样离线折叠。
- EdgeAwareBlock: 4 方向 depthwise(含BN) → concat → 1x1 融合(含BN)
  等价于一个标准 KxK 稠密卷积（每个输出核是 4 个方向核的线性组合）。

折叠仅在 eval 模式语义下等价（BN 使用 running statistics）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.models.builder import DenoiseBlock, EdgeAwareBlock


def _bn_scale_bias(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 eval 模式 BN 等价的逐通道 scale/bias: y = scale*x + bias。"""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias - bn.running_mean * scale
    return scale, bias


class FoldedDenoiseBlock(nn.Module):
    """DenoiseBlock 的部署等价形式: DW(带bias) → ReLU → PW(带bias) → 残差。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_residual: bool,
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw_conv(x)
        x = self.relu(x)
        x = self.pw_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class FoldedEdgeBlock(nn.Module):
    """EdgeAwareBlock 的部署等价形式: 稠密 KxK conv(带bias) → 残差 → ReLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_residual: bool,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        if self.use_residual:
            x = x + identity
        x = self.relu(x)
        return x


@torch.no_grad()
def fold_denoise_block(block: DenoiseBlock) -> FoldedDenoiseBlock:
    """把 DenoiseBlock 折叠为单 DW + PW（eval 语义等价）。"""
    dw = block.dw_conv
    in_channels = dw.in_channels
    kernel_size = dw.kernel_size[0]
    stride = dw.stride[0]

    # 特征分支: BN 折叠进 DW 权重
    dw_scale, dw_bias = _bn_scale_bias(block.dw_bn)
    dw_weight = dw.weight * dw_scale.view(-1, 1, 1, 1)

    # 平滑分支: softmax 归一化核在推理期是常量，直接算出
    smooth = block.smooth_weight
    smooth_weight = F.softmax(smooth.view(in_channels, -1), dim=1).view_as(smooth)

    # feat + smooth = (W_dw_folded + W_smooth) ⊛ x + b_dw
    merged_weight = dw_weight + smooth_weight

    # PW 分支: pw_bn 折叠进 pw 权重
    pw_scale, pw_bias = _bn_scale_bias(block.pw_bn)
    pw_weight = block.pw_conv.weight * pw_scale.view(-1, 1, 1, 1)

    folded = FoldedDenoiseBlock(
        in_channels=in_channels,
        out_channels=block.pw_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        use_residual=block.use_residual,
    )
    folded.dw_conv.weight.copy_(merged_weight)
    folded.dw_conv.bias.copy_(dw_bias)
    folded.pw_conv.weight.copy_(pw_weight)
    folded.pw_conv.bias.copy_(pw_bias)
    folded.eval()
    return folded


@torch.no_grad()
def fold_edge_block(block: EdgeAwareBlock) -> FoldedEdgeBlock:
    """把 EdgeAwareBlock 折叠为单个稠密 KxK 卷积（eval 语义等价）。"""
    first_conv = block.edge_convs[0]
    in_channels = first_conv.in_channels
    kernel_size = first_conv.kernel_size[0]
    stride = first_conv.stride[0]
    out_channels = block.fusion_conv.out_channels
    num_directions = block.NUM_DIRECTIONS

    # 各方向 BN 折叠: y_{d,c} = W'_{d,c} ⊛ x_c + b'_{d,c}
    direction_weights: list[torch.Tensor] = []  # 每项 (C, k, k)
    direction_biases: list[torch.Tensor] = []  # 每项 (C,)
    for conv, bn in zip(block.edge_convs, block.edge_bns):
        scale, bias = _bn_scale_bias(bn)
        direction_weights.append(conv.weight.squeeze(1) * scale.view(-1, 1, 1))
        direction_biases.append(bias)

    # 融合 1x1 (无bias) 与 fusion_bn 折叠: F'[o, d*C+c]
    fusion_scale, fusion_bias = _bn_scale_bias(block.fusion_bn)
    fusion = block.fusion_conv.weight.squeeze(-1).squeeze(-1)  # (C_out, 4C)
    fusion = fusion * fusion_scale.view(-1, 1)

    # out_o = Σ_c (Σ_d F'[o,dC+c] W'_{d,c}) ⊛ x_c + Σ_{d,c} F'[o,dC+c] b'_{d,c} + b_f[o]
    dense_weight = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    dense_bias = fusion_bias.clone()
    for d in range(num_directions):
        coeff = fusion[:, d * in_channels : (d + 1) * in_channels]  # (C_out, C)
        dense_weight += torch.einsum("oc,ckl->ockl", coeff, direction_weights[d])
        dense_bias += coeff @ direction_biases[d]

    folded = FoldedEdgeBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        use_residual=block.use_residual,
    )
    folded.conv.weight.copy_(dense_weight)
    folded.conv.bias.copy_(dense_bias)
    folded.eval()
    return folded


def fold_sonar_blocks(module: nn.Module) -> int:
    """递归把模型中的 DenoiseBlock/EdgeAwareBlock 原位替换为折叠形式。

    返回替换的块数。要求模型已处于 eval 模式（BN 使用 running stats）。
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, DenoiseBlock):
            setattr(module, name, fold_denoise_block(child))
            replaced += 1
        elif isinstance(child, EdgeAwareBlock):
            setattr(module, name, fold_edge_block(child))
            replaced += 1
        else:
            replaced += fold_sonar_blocks(child)
    return replaced
