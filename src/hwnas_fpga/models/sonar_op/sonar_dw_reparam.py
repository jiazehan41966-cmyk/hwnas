"""Reparameterizable dual-receptive-field depthwise operator V0.2-P0."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReparamDualRFDepthwise(nn.Module):
    """Train: BN(DW3)+BN(DW5)->ReLU6. Deploy: one DW5->ReLU6."""

    sonarop_version = "v0.2-p0"
    deploy_primitive = "EXISTING_DW5"

    def __init__(self, channels: int, stride: int = 1, *, deploy: bool = False) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.channels = int(channels)
        self.stride = int(stride)
        self.deploy = bool(deploy)
        self.activation = nn.ReLU6(inplace=False)
        if self.deploy:
            self.reparam_conv = nn.Conv2d(
                channels, channels, 5, stride=stride, padding=2,
                groups=channels, bias=True,
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=stride, padding=1,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.branch5 = nn.Sequential(
                nn.Conv2d(channels, channels, 5, stride=stride, padding=2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.reparam_conv(x))
        return self.activation(self.branch3(x) + self.branch5(x))

    @staticmethod
    def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        if conv.bias is None:
            conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device,
                                    dtype=conv.weight.dtype)
        else:
            conv_bias = conv.bias
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias + (conv_bias - bn.running_mean) * scale
        return weight, bias

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam_conv.weight.detach().clone(), self.reparam_conv.bias.detach().clone()
        if self.training:
            raise RuntimeError("fusion requires eval mode with frozen BN running statistics")
        w3, b3 = self._fold_conv_bn(self.branch3[0], self.branch3[1])
        w5, b5 = self._fold_conv_bn(self.branch5[0], self.branch5[1])
        return w5 + F.pad(w3, (1, 1, 1, 1)), b5 + b3

    @torch.no_grad()
    def switch_to_deploy(self) -> "ReparamDualRFDepthwise":
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.channels, self.channels, 5, stride=self.stride, padding=2,
            groups=self.channels, bias=True, device=kernel.device, dtype=kernel.dtype,
        )
        conv.weight.copy_(kernel)
        conv.bias.copy_(bias)
        self.reparam_conv = conv
        del self.branch3
        del self.branch5
        self.deploy = True
        return self
