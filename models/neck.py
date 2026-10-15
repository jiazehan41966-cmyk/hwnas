"""Selected ELASTIC top-down/PAN path; no search mixtures."""
from __future__ import annotations
import hashlib
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import MBConvBlock

LEVELS=("P2", "P3", "P4", "P5")
PILOT_INITIALIZATION_POLICY="B1_LEGACY_ELASTIC_NECK_HEAD_SEMANTIC_CPU_SEED_V1"

def _cpu_seed(seed: int) -> None:
    torch.random.default_generator.manual_seed(seed)


def _semantic_seed(seed: int, name: str) -> int:
    encoded = f"{PILOT_INITIALIZATION_POLICY}:{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _initialize_semantic_modules(module: nn.Module, seed: int, prefix: str) -> None:
    for name, child in module.named_modules():
        if isinstance(child, (nn.Conv2d, nn.BatchNorm2d)):
            _cpu_seed(_semantic_seed(seed, f"{prefix}.{name}"))
            child.reset_parameters()


def _pointwise(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class ElasticFourScaleNeck(nn.Module):
    """MBConv after each top-down fusion, with optional progressive bottom-up PAN."""

    def __init__(self, in_channels: Sequence[int], out_channels: int,
                 bottom_up_pan: bool) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.levels = LEVELS
        self.bottom_up_pan = bottom_up_pan
        if len(self.in_channels) != 4:
            raise ValueError("ELASTIC pilot neck requires all four B1 feature levels")
        self.lateral_convs = nn.ModuleDict({
            level: _pointwise(channels, out_channels)
            for level, channels in zip(self.levels, self.in_channels)
        })
        # Only P2/P3/P4 need fusion blocks; P5 is a lateral projection alone.
        self.top_down_blocks = nn.ModuleDict({
            level: MBConvBlock(out_channels, out_channels, 3, stride=1, expand_ratio=3)
            for level in self.levels[:-1]
        })
        self.bottom_up_projections = nn.ModuleDict()
        self.bottom_up_blocks = nn.ModuleDict()
        if bottom_up_pan:
            self.bottom_up_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottom_up_projections.update({
                level: _pointwise(out_channels, out_channels) for level in self.levels[1:]
            })
            self.bottom_up_blocks.update({
                level: MBConvBlock(out_channels, out_channels, 3, stride=1, expand_ratio=3)
                for level in self.levels[1:]
            })

    def common_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(("lateral_convs.", "top_down_blocks."))}

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != 4:
            raise ValueError("ELASTIC pilot neck expects exactly four feature maps")
        fused = [self.lateral_convs[level](feature)
                 for level, feature in zip(self.levels, features)]
        for index in range(2, -1, -1):
            merged = fused[index] + F.interpolate(
                fused[index + 1], size=fused[index].shape[-2:], mode="nearest",
            )
            fused[index] = self.top_down_blocks[self.levels[index]](merged)
        if not self.bottom_up_pan:
            return tuple(fused)
        outputs = [fused[0]]
        for index in range(1, 4):
            level = self.levels[index]
            reduced = self.bottom_up_projections[level](self.bottom_up_pool(outputs[-1]))
            if reduced.shape[-2:] != fused[index].shape[-2:]:
                raise ValueError("PAN requires adjacent feature sizes with exact factor two")
            outputs.append(self.bottom_up_blocks[level](fused[index] + reduced))
        return tuple(outputs)
