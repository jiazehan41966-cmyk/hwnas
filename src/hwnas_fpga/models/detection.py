"""Fixed first-stage multi-scale detector built on the HW-NAS backbone."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.models.builder import StemBlock, build_block
from hwnas_fpga.search_space import ArchitectureSpec


DEFAULT_DETECTION_FEATURE_INDICES = (3, 5, 7)


def _normalize_feature_indices(
    feature_indices: Sequence[int],
    stage_count: int,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in feature_indices:
        resolved = int(index)
        if resolved < 0:
            resolved += stage_count
        if resolved < 0 or resolved >= stage_count:
            raise ValueError(
                f"detection feature index {index} is outside {stage_count} stages"
            )
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ValueError("at least one detection feature index is required")
    return tuple(normalized)


def detection_feature_channels(
    architecture: ArchitectureSpec,
    feature_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
) -> tuple[int, ...]:
    indices = _normalize_feature_indices(feature_indices, len(architecture.stages))
    return tuple(int(architecture.stages[index].channels) for index in indices)


class HWNASFeatureBackbone(nn.Module):
    """Backbone-only copy of the classification search architecture."""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        feature_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.feature_indices = _normalize_feature_indices(
            feature_indices,
            len(architecture.stages),
        )

        self.stem = StemBlock(
            in_channels=architecture.input_channels,
            stem_channels=architecture.stem_channels,
            stride=architecture.stem_stride,
        )
        self.post_stem_downsample = (
            nn.MaxPool2d(
                kernel_size=3,
                stride=architecture.post_stem_downsample_stride,
                padding=1,
            )
            if architecture.post_stem_downsample_stride > 1
            else None
        )

        self.stages = nn.ModuleList()
        current_channels = architecture.stem_channels
        for stage_spec in architecture.stages:
            stage_blocks = nn.ModuleList()
            for block_index, block_spec in enumerate(stage_spec.blocks):
                block_in_channels = (
                    current_channels if block_index == 0 else stage_spec.channels
                )
                stage_blocks.append(
                    build_block(block_spec, block_in_channels, stage_spec.channels)
                )
            self.stages.append(stage_blocks)
            current_channels = stage_spec.channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        if self.post_stem_downsample is not None:
            x = self.post_stem_downsample(x)

        features: list[torch.Tensor] = []
        requested = set(self.feature_indices)
        for stage_index, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            if stage_index in requested:
                features.append(x)
        return tuple(features)


class FixedFPNNeck(nn.Module):
    """Fixed top-down FPN neck; not part of the stage-1 search space."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 96,
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("FixedFPNNeck requires at least one input scale")
        self.in_channels = tuple(int(channels) for channels in in_channels)
        self.out_channels = int(out_channels)
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
            for channels in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.in_channels
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature maps, got {len(features)}"
            )
        lateral = [
            conv(feature)
            for conv, feature in zip(self.lateral_convs, features)
        ]
        fused = list(lateral)
        for index in range(len(fused) - 2, -1, -1):
            fused[index] = fused[index] + F.interpolate(
                fused[index + 1],
                size=fused[index].shape[-2:],
                mode="nearest",
            )
        return tuple(conv(feature) for conv, feature in zip(self.output_convs, fused))


class FixedDetectionHead(nn.Module):
    """Fixed dense detection head for each FPN scale."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        anchors_per_location: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.anchors_per_location = int(anchors_per_location)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive for detection")
        if self.anchors_per_location <= 0:
            raise ValueError("anchors_per_location must be positive")
        self.outputs_per_anchor = self.num_classes + 5
        self.prediction_channels = self.anchors_per_location * self.outputs_per_anchor
        self.towers = nn.ModuleList()

    def build(self, num_levels: int) -> None:
        if self.towers:
            if len(self.towers) != num_levels:
                raise ValueError("fixed detection head level count cannot change")
            return
        for _ in range(int(num_levels)):
            self.towers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.in_channels,
                        self.prediction_channels,
                        kernel_size=1,
                    ),
                )
            )
            self._initialize_prediction_bias(self.towers[-1][-1])

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.build(len(features))
        return tuple(tower(feature) for tower, feature in zip(self.towers, features))

    def _initialize_prediction_bias(self, prediction_conv: nn.Conv2d) -> None:
        with torch.no_grad():
            bias = prediction_conv.bias.view(self.anchors_per_location, self.outputs_per_anchor)
            bias[:, 4].fill_(math.log(0.01 / 0.99))
            bias[:, 5:].zero_()


class HWNASMultiScaleDetector(nn.Module):
    """Detection task branch with searchable backbone and fixed Neck/Head."""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        num_classes: int,
        *,
        feature_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
        neck_channels: int = 96,
        anchors_per_location: int = 3,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.feature_indices = _normalize_feature_indices(
            feature_indices,
            len(architecture.stages),
        )
        self.backbone = HWNASFeatureBackbone(
            architecture,
            feature_indices=self.feature_indices,
        )
        in_channels = detection_feature_channels(architecture, self.feature_indices)
        self.neck = FixedFPNNeck(in_channels, out_channels=neck_channels)
        self.head = FixedDetectionHead(
            neck_channels,
            num_classes=num_classes,
            anchors_per_location=anchors_per_location,
        )
        self.head.build(len(self.feature_indices))

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        predictions = self.head(neck_features)
        return {
            "backbone_features": backbone_features,
            "neck_features": neck_features,
            "predictions": predictions,
        }


def build_detection_model(
    architecture: ArchitectureSpec,
    num_classes: int,
    *,
    feature_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
    neck_channels: int = 96,
    anchors_per_location: int = 3,
) -> HWNASMultiScaleDetector:
    return HWNASMultiScaleDetector(
        architecture=architecture,
        num_classes=num_classes,
        feature_indices=feature_indices,
        neck_channels=neck_channels,
        anchors_per_location=anchors_per_location,
    )
