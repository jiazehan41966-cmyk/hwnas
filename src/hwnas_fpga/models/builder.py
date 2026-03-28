from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec


class ConvBlock(nn.Module):
    """基础卷积块: Conv + BN + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseConvBlock(nn.Module):
    """Depthwise Separable卷积: DW + BN + ReLU + PW + BN + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)

        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_relu(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class MBConvBlock(nn.Module):
    """MobileNetV2风格MBConv: 1x1 PW(expand) -> DW(kernel) -> 1x1 PW(project)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        hidden_channels = in_channels * expand_ratio

        # 1x1 pointwise expand (只有当 expand_ratio > 1 时才有效)
        self.use_expand = expand_ratio > 1
        if self.use_expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_channels)
            self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise conv
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            hidden_channels if self.use_expand else in_channels,
            hidden_channels if self.use_expand else in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_channels if self.use_expand else in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(hidden_channels if self.use_expand else in_channels)
        self.dw_relu = nn.ReLU6(inplace=True)

        # 1x1 pointwise project
        self.project_conv = nn.Conv2d(
            hidden_channels if self.use_expand else in_channels,
            out_channels,
            1,
            1,
            0,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Residual connection条件
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.use_expand:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_relu(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_relu(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            x += identity

        return x


class FusedMBConvBlock(nn.Module):
    """Fused MBConv: 直接在DW上做expand，省略一个PW"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        hidden_channels = in_channels * expand_ratio

        padding = kernel_size // 2

        # Fused expand + depthwise conv
        self.fused_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.fused_bn = nn.BatchNorm2d(hidden_channels)
        self.fused_relu = nn.ReLU6(inplace=True)

        # Pointwise project
        self.project_conv = nn.Conv2d(
            hidden_channels,
            out_channels,
            1,
            1,
            0,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Residual connection条件
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.fused_conv(x)
        x = self.fused_bn(x)
        x = self.fused_relu(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            x += identity

        return x


class SkipBlock(nn.Module):
    """跳过连接块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 只在通道数不匹配时使用1x1 conv
        self.use_conv = in_channels != out_channels
        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x)
            x = self.bn(x)
        return x


class MixConvBlock(nn.Module):
    """声呐专用多尺度卷积块 (MixConv风格)

    并行使用不同kernel_size的深度可分离卷积，捕获不同尺度的特征。
    适用于声呐图像中不同尺寸的目标和模糊边界处理。

    结构: Split → 多分支 DW(k=3,5,7) → Concat → PW → BN → ReLU6
    当 in_channels < num_kernels 时，自动退化为单一最大kernel的DW卷积。

    Reference: MixConv: Mixed Depthwise Convolutional Kernels (Tan & Le, 2019)
    """

    KERNEL_SIZES = (3, 5, 7)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = KERNEL_SIZES,
        stride: int = 1,
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        num_kernels = len(kernel_sizes)

        # 当通道数不够分时，减少分支数
        if in_channels < num_kernels:
            # 退化为单kernel（取最大的以保持多尺度语义）
            kernel_sizes = (max(kernel_sizes),)
            num_kernels = 1

        self.num_kernels = num_kernels

        # 均匀分配通道到各分支，余数分给最后一个分支
        base = in_channels // num_kernels
        self.group_sizes: list[int] = []
        remaining = in_channels
        for i in range(num_kernels):
            if i == num_kernels - 1:
                self.group_sizes.append(remaining)
            else:
                self.group_sizes.append(base)
                remaining -= base

        # 为每个kernel size创建depthwise conv分支
        self.dw_convs = nn.ModuleList()
        self.dw_bns = nn.ModuleList()

        for i, ks in enumerate(kernel_sizes):
            ch = self.group_sizes[i]
            self.dw_convs.append(
                nn.Conv2d(ch, ch, ks, stride=stride, padding=ks // 2,
                          groups=ch, bias=False)
            )
            self.dw_bns.append(nn.BatchNorm2d(ch))

        self.dw_relu = nn.ReLU6(inplace=True)

        # Pointwise融合
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU6(inplace=True)

        # 残差连接
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Split → 多分支 DW
        splits = torch.split(x, self.group_sizes, dim=1)
        branch_outputs = []
        for i, x_branch in enumerate(splits):
            x_branch = self.dw_convs[i](x_branch)
            x_branch = self.dw_bns[i](x_branch)
            branch_outputs.append(x_branch)

        x = torch.cat(branch_outputs, dim=1)
        x = self.dw_relu(x)

        # Pointwise融合
        x = self.pw_conv(x)
        x = self.pw_bn(x)

        if self.use_residual:
            x = x + identity

        x = self.pw_relu(x)
        return x


class DenoiseBlock(nn.Module):
    """声呐专用去噪/平滑先验块

    双分支结构：
    1. 特征分支：DW卷积提取局部特征
    2. 平滑分支：可学习的高斯初始化smooth核抑制斑点噪声

    两分支相加后通过PW投影到输出通道，兼顾去噪和特征保持。

    结构: x → DW(特征) + Smooth(去噪) → ReLU → PW → BN
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        use_gaussian: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        # 分支1: 特征提取 DW 卷积
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        # 分支2: 可学习平滑核（每通道独立的 kernel_size x kernel_size 核）
        # 形状: (in_channels, 1, kernel_size, kernel_size) — depthwise 卷积权重格式
        smooth_weight = torch.zeros(in_channels, 1, kernel_size, kernel_size)
        if use_gaussian:
            # 用高斯核初始化
            sigma = kernel_size / 4.0
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            gaussian_2d = g.unsqueeze(0) * g.unsqueeze(1)
            gaussian_2d = gaussian_2d / gaussian_2d.sum()
            smooth_weight[:, 0] = gaussian_2d
        else:
            # 均值核初始化
            smooth_weight.fill_(1.0 / (kernel_size * kernel_size))
        self.smooth_weight = nn.Parameter(smooth_weight)
        self.smooth_stride = stride
        self.smooth_padding = padding

        self.relu = nn.ReLU(inplace=True)

        # PW 投影
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 分支1: 特征 DW
        feat = self.dw_conv(x)
        feat = self.dw_bn(feat)

        # 分支2: 可学习平滑（softmax 归一化保证核权重和为1）
        C = self.smooth_weight.shape[0]
        w = self.smooth_weight.view(C, -1)
        w = F.softmax(w, dim=1).view_as(self.smooth_weight)
        smooth = F.conv2d(
            x, w,
            stride=self.smooth_stride, padding=self.smooth_padding,
            groups=C,
        )

        # 双分支融合
        x = feat + smooth
        x = self.relu(x)

        # PW 投影
        x = self.pw_conv(x)
        x = self.pw_bn(x)

        if self.use_residual:
            x = x + identity

        return x


class EdgeAwareBlock(nn.Module):
    """声呐专用轮廓/形状感知块

    使用类 Sobel 算子进行多方向边缘检测，结合 1x1 融合卷积提取形状特征。
    适用于声呐图像中目标轮廓和边界识别。

    结构: x → 4方向DW(Sobel初始化, 可学习) → Concat → PW融合 → BN → ReLU
    对于 kernel_size > 3，在 kernel 中心嵌入 3x3 Sobel 核。
    """

    NUM_DIRECTIONS = 4  # 水平、垂直、两条对角线

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        padding = kernel_size // 2

        # 4方向 depthwise 边缘检测卷积，Sobel 初始化
        self.edge_convs = nn.ModuleList()
        self.edge_bns = nn.ModuleList()

        sobel_3x3 = [
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),   # 水平
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32),    # 垂直
            torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32),    # 对角1
            torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32),    # 对角2
        ]

        for i in range(self.NUM_DIRECTIONS):
            conv = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=False,
            )
            # 初始化权重: 将3x3 Sobel嵌入任意大小kernel的中心
            with torch.no_grad():
                conv.weight.zero_()
                sobel = sobel_3x3[i]
                offset = (kernel_size - 3) // 2
                for c in range(in_channels):
                    conv.weight[c, 0, offset:offset + 3, offset:offset + 3] = sobel

            self.edge_convs.append(conv)
            self.edge_bns.append(nn.BatchNorm2d(in_channels))

        # 1x1 融合卷积
        self.fusion_conv = nn.Conv2d(
            in_channels * self.NUM_DIRECTIONS, out_channels, 1, bias=False,
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差连接
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        edge_outputs = [
            self.edge_bns[i](self.edge_convs[i](x))
            for i in range(self.NUM_DIRECTIONS)
        ]
        x = torch.cat(edge_outputs, dim=1)

        x = self.fusion_conv(x)
        x = self.fusion_bn(x)

        if self.use_residual:
            x = x + identity

        x = self.relu(x)
        return x


class StemBlock(nn.Module):
    """Stem: 初始3x3卷积"""

    def __init__(self, in_channels: int, stem_channels: int, stride: int = 2):
        super().__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HeadBlock(nn.Module):
    """Head: 全局平均池化 + 分类层"""

    def __init__(self, in_channels: int, num_classes: int, head_channels: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 可选的额外全连接层
        if head_channels and head_channels > 0:
            self.fc1 = nn.Linear(in_channels, head_channels)
            self.fc2 = nn.Linear(head_channels, num_classes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = x.flatten(1)

        if hasattr(self, "fc1"):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)

        return x


def build_block(block_spec: BlockSpec, in_channels: int, out_channels: int) -> nn.Module:
    """根据BlockSpec构建对应的PyTorch模块"""
    op = block_spec.op

    if op == "conv":
        return ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
        )

    elif op == "dw_pw_conv":
        return DepthwiseConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
        )

    elif op == "mbconv":
        return MBConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
            expand_ratio=block_spec.expand_ratio,
        )

    elif op == "fused_mbconv":
        return FusedMBConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
            expand_ratio=block_spec.expand_ratio,
        )

    elif op == "skip":
        return SkipBlock(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    elif op == "mixconv":
        # 声呐专用多尺度卷积，默认使用(3,5,7)三种kernel size
        return MixConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=(3, 5, 7),
            stride=block_spec.stride,
        )

    elif op == "denoise":
        # 声呐专用去噪/平滑块
        return DenoiseBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
            use_gaussian=True,
        )

    elif op == "edge":
        # 声呐专用轮廓/边缘感知块
        return EdgeAwareBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=block_spec.kernel_size,
            stride=block_spec.stride,
        )

    else:
        raise ValueError(f"Unsupported op type: {op}")


class HWNASModel(nn.Module):
    """HW-NAS搜索出的模型"""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        num_classes: int,
        head_channels: Optional[int] = None,
    ):
        super().__init__()

        arch = architecture

        # Stem
        self.stem = StemBlock(
            in_channels=arch.input_channels,
            stem_channels=arch.stem_channels,
            stride=arch.stem_stride,
        )

        # Stages
        self.stages = nn.ModuleList()
        current_channels = arch.stem_channels

        for stage_spec in arch.stages:
            stage_blocks = nn.ModuleList()

            for i, block_spec in enumerate(stage_spec.blocks):
                block_in_channels = current_channels if i == 0 else stage_spec.channels
                block = build_block(block_spec, block_in_channels, stage_spec.channels)
                stage_blocks.append(block)

            self.stages.append(stage_blocks)
            current_channels = stage_spec.channels

        # Head
        self.head = HeadBlock(
            in_channels=current_channels,
            num_classes=num_classes,
            head_channels=head_channels,
        )

        # 保存架构信息
        self.architecture = architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)

        x = self.head(x)
        return x

    def get_output_channels(self) -> int:
        """获取最后一个stage的输出通道数"""
        if self.stages:
            last_stage = self.stages[-1]
            if hasattr(last_stage, "__iter__"):
                last_block = last_stage[-1]
                if isinstance(last_block, ConvBlock):
                    return last_block.conv.out_channels
                elif isinstance(last_block, DepthwiseConvBlock):
                    return last_block.pw_conv.out_channels
                elif isinstance(last_block, MBConvBlock):
                    return last_block.project_conv.out_channels
                elif isinstance(last_block, FusedMBConvBlock):
                    return last_block.project_conv.out_channels
                elif isinstance(last_block, SkipBlock):
                    if last_block.use_conv:
                        return last_block.conv.out_channels
                elif isinstance(last_block, MixConvBlock):
                    return last_block.pw_conv.out_channels
                elif isinstance(last_block, DenoiseBlock):
                    return last_block.pw_conv.out_channels
                elif isinstance(last_block, EdgeAwareBlock):
                    return last_block.fusion_conv.out_channels
        return self.stem.conv.out_channels


def build_model(
    architecture: ArchitectureSpec,
    num_classes: int,
    head_channels: Optional[int] = None,
) -> HWNASModel:
    """构建模型工厂函数"""
    return HWNASModel(
        architecture=architecture,
        num_classes=num_classes,
        head_channels=head_channels,
    )
