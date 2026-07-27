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


class MixConvV2Block(nn.Module):
    """Versioned 3x3/5x5 mixed-depthwise block with MBConv semantics.

    The first ``floor(C/2)`` channels use a 3x3 depthwise kernel and the
    remaining channels use a 5x5 depthwise kernel.  Projection is linear and
    the optional residual is applied after projection, matching the project's
    MBConv block and the HLS contract for ``mixconv_v2``.
    """

    KERNEL_SIZES = (3, 5)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if int(in_channels) < 2:
            raise ValueError("mixconv_v2 requires at least two input channels")
        first = int(in_channels) // 2
        self.group_sizes = [first, int(in_channels) - first]
        self.kernel_sizes = self.KERNEL_SIZES
        self.dw_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=channels,
                    bias=False,
                )
                for channels, kernel_size in zip(
                    self.group_sizes, self.kernel_sizes
                )
            ]
        )
        self.dw_bns = nn.ModuleList(
            [nn.BatchNorm2d(channels) for channels in self.group_sizes]
        )
        self.dw_relu = nn.ReLU6(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branch_outputs = []
        for branch, batch_norm in zip(
            torch.split(x, self.group_sizes, dim=1), self.dw_bns
        ):
            index = len(branch_outputs)
            branch_outputs.append(
                self.dw_relu(batch_norm(self.dw_convs[index](branch)))
            )
        x = torch.cat(branch_outputs, dim=1)
        x = self.pw_bn(self.pw_conv(x))
        if self.use_residual:
            x = x + identity
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
            # Forward applies softmax to keep each kernel non-negative and
            # normalized. Store logits so the effective initialization is
            # actually Gaussian.
            smooth_weight[:, 0] = torch.log(gaussian_2d.clamp_min(1e-12))
        else:
            # Equal logits produce an exactly uniform effective kernel.
            smooth_weight.zero_()
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


class AdaptiveDenoiseBlock(nn.Module):
    """声呐自适应去噪块 (denoise_v2, Lee 式门控)

    经典 Lee 滤波的硬件友好近似：均匀区收敛到局部均值（强去斑点），
    高梯度区收敛到原信号（保边缘）。修复 v1 固定低通核的边缘保持短板。

        mu  = smooth(x)              # softmax 归一化可学习核（高斯初始化），即局部均值
        d   = x - mu                 # 高通残差
        e   = avgpool3(|d|)          # 空间聚合的边缘证据（斑点在聚合中被平均掉）
        g   = sigmoid(alpha*e + beta)    # 逐通道可学习门控（对应 Lee 的 k 系数）
        lee = mu + g * d             # g→0 平滑，g→1 保留
        out = PW(ReLU(feat(x) + lee))，残差条件同 v1

    门控证据用池化聚合而非逐像素 |d|：2026-07-10 的合成斑点扫描表明，
    重斑点下逐像素残差门控会把噪声尖峰当边缘保留（经典 Lee 的失败模式，
    EPI 反而低于纯高斯），空间聚合是门控不帮倒忙的前提。beta 可学习意味着
    训练可将门完全关死退化回 v1 的纯平滑分支，因此 v2 的行为空间包含 v1。

    相对 v1 仅新增逐元素 sub/abs/mul/add、3x3 平均池化与逐通道 sigmoid
    （INT8 部署可用 256 项查表），无新增卷积。门控依赖输入，故不可折叠为
    单一静态卷积——这是换取自适应性的确定代价。
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

        # 特征分支（同 v1）
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        # 平滑核（同 v1：softmax 归一化，高斯初始化）
        smooth_weight = torch.zeros(in_channels, 1, kernel_size, kernel_size)
        if use_gaussian:
            sigma = kernel_size / 4.0
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            gaussian_2d = g.unsqueeze(0) * g.unsqueeze(1)
            gaussian_2d = gaussian_2d / gaussian_2d.sum()
            smooth_weight[:, 0] = torch.log(gaussian_2d.clamp_min(1e-12))
        self.smooth_weight = nn.Parameter(smooth_weight)
        self.smooth_padding = padding
        self.stride = stride

        # 逐通道门控参数：beta<0 使默认偏向平滑，alpha>0 使高残差区开门保边
        self.gate_alpha = nn.Parameter(torch.full((1, in_channels, 1, 1), 4.0))
        self.gate_beta = nn.Parameter(torch.full((1, in_channels, 1, 1), -2.0))

        self.relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def adaptive_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Lee 式自适应平滑分支（stride 在此分支内通过网格子采样对齐）。"""
        C = self.smooth_weight.shape[0]
        w = self.smooth_weight.view(C, -1)
        w = F.softmax(w, dim=1).view_as(self.smooth_weight)
        # 局部均值先按 stride 采样（与 conv 采样网格一致：中心 = i*stride）
        mu = F.conv2d(x, w, stride=self.stride, padding=self.smooth_padding, groups=C)
        x_sub = x[..., ::self.stride, ::self.stride] if self.stride > 1 else x
        d = x_sub - mu
        # 空间聚合的门控证据：斑点残差在 3x3 平均中互相抵消，真实边缘证据保留
        evidence = F.avg_pool2d(d.abs(), kernel_size=3, stride=1, padding=1)
        gate = torch.sigmoid(self.gate_alpha * evidence + self.gate_beta)
        return mu + gate * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        feat = self.dw_conv(x)
        feat = self.dw_bn(feat)
        lee = self.adaptive_smooth(x)

        x = feat + lee
        x = self.relu(x)
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


class EdgeAugmentBlock(nn.Module):
    """声呐边缘增强块 v2 — 信息保留的加性边缘增强。

    修复 EdgeAwareBlock 的致命缺陷：旧版只输出 4 方向梯度、把强度/DC 信息
    全丢了，E1 消融显示它拖累分类约 9 个点（Δ macro_f1 = -0.092，CI 全负）。

    重新设计的三个要点：
    1. **强度主路 + 边缘旁支**：以标准 DW 特征为主路，边缘作为**加性**旁支叠加，
       绝不替换主路，信息不丢失。
    2. **小初始化门控 gamma≈0**：初始时边缘旁支权重≈0，块退化为普通 dw_pw
       （E1 证明其无害），训练只会在边缘确有增益时把 gamma 抬起——**保底不劣**。
    3. **2 方向替代 4 方向**：只用 Gx/Gy + 幅值，融合输入降到 2C，显著降低 LUT
       （旧版 4C 融合是它进 HIGH_LUT_OPS 的主因）。

        feat = DW(x)+BN                          # 强度主路
        gx,gy = SobelDW(x)  (2 方向, Sobel 初始化, 可学习)
        mag  = sqrt(gx^2 + gy^2 + eps)           # 逐通道边缘幅值
        out  = PW(feat) + gamma * PW_edge(mag)   # gamma 初始 0
        → BN → (+residual) → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        # 强度主路
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        # 2 方向 Sobel 边缘 DW（可学习，Sobel 初始化）
        sobel = [
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),  # Gx
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32),  # Gy
        ]
        self.grad_convs = nn.ModuleList()
        offset = (kernel_size - 3) // 2
        for direction in range(2):
            conv = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=False,
            )
            with torch.no_grad():
                conv.weight.zero_()
                for c in range(in_channels):
                    conv.weight[c, 0, offset:offset + 3, offset:offset + 3] = sobel[direction]
            self.grad_convs.append(conv)

        # 主路 PW 与边缘旁支 PW
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.edge_pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # 小初始化门控：初始边缘贡献为 0，保底退化为 dw_pw
        self.edge_gamma = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        feat = self.dw_bn(self.dw_conv(x))

        gx = self.grad_convs[0](x)
        gy = self.grad_convs[1](x)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        out = self.pw_conv(feat) + self.edge_gamma * self.edge_pw(mag)
        out = self.bn(out)

        if self.use_residual:
            out = out + identity

        out = self.relu(out)
        return out


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

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        head_channels: Optional[int] = None,
        conv_head_channels: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        classifier_in_channels = in_channels
        if conv_head_channels and conv_head_channels > 0:
            self.conv_head = nn.Sequential(
                nn.Conv2d(in_channels, conv_head_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(conv_head_channels),
                nn.ReLU(inplace=True),
            )
            classifier_in_channels = conv_head_channels
        else:
            self.conv_head = None

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 可选的额外全连接层
        if head_channels and head_channels > 0:
            self.fc1 = nn.Linear(classifier_in_channels, head_channels)
            self.fc2 = nn.Linear(head_channels, num_classes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.fc = nn.Linear(classifier_in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_head is not None:
            x = self.conv_head(x)
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

    elif op == "mixconv_v2":
        return MixConvV2Block(
            in_channels=in_channels,
            out_channels=out_channels,
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

    elif op == "adaptive_denoise":
        # 声呐自适应去噪块 (denoise_v2, Lee 式门控)；G5 准入前不得进入正式搜索
        return AdaptiveDenoiseBlock(
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

    elif op == "edge_v2":
        # 声呐边缘增强块 v2（信息保留加性边缘）；G5 准入前不得进入正式搜索
        return EdgeAugmentBlock(
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
        self.post_stem_downsample = (
            nn.MaxPool2d(kernel_size=3, stride=arch.post_stem_downsample_stride, padding=1)
            if arch.post_stem_downsample_stride > 1
            else None
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
        resolved_head_channels = arch.head_channels if head_channels is None else head_channels
        self.head = HeadBlock(
            in_channels=current_channels,
            num_classes=num_classes,
            head_channels=resolved_head_channels,
            conv_head_channels=arch.head_conv_channels,
        )

        # 保存架构信息
        self.architecture = architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.post_stem_downsample is not None:
            x = self.post_stem_downsample(x)

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
                elif isinstance(last_block, (DenoiseBlock, AdaptiveDenoiseBlock)):
                    return last_block.pw_conv.out_channels
                elif isinstance(last_block, EdgeAwareBlock):
                    return last_block.fusion_conv.out_channels
                elif isinstance(last_block, EdgeAugmentBlock):
                    return last_block.pw_conv.out_channels
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
