"""Fixed B1 primitive subset, migrated verbatim from legacy builder.py."""
from __future__ import annotations
import torch
from torch import nn

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
