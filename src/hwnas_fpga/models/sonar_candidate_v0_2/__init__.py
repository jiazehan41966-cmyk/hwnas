"""Independent V0.2 detector prototypes; no shared weights or V0.1 edits.

This module builds software models only. Its architecture contract neither
estimates FPGA latency nor claims quantized, routed, or board correctness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from hwnas_fpga.models.detection import HWNASFeatureBackbone
from hwnas_fpga.search_space import ArchitectureSpec
from hwnas_fpga.training.detection_trainer import DetectionLossConfig

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / "configs/sonar_detector_v0_2.yaml"

_STAGES = [
    dict(index=0, op="conv", channels=16, stride=1, depth=1, kernel=1),
    dict(index=1, op="mbconv", channels=24, stride=2, depth=1, expansion=3, kernel=3),
    dict(index=2, op="mbconv", channels=32, stride=2, depth=1, expansion=3, kernel=3),
    dict(index=3, op="skip", channels=32, stride=1, depth=1, kernel=1),
    dict(index=4, op="mbconv", channels=64, stride=2, depth=1, expansion=3, kernel=3),
    dict(index=5, op="mbconv", channels=96, stride=1, depth=1, expansion=3, kernel=3),
    dict(index=6, op="mbconv", channels=160, stride=2, depth=1, expansion=3, kernel=3),
    dict(index=7, op="mbconv", channels=320, stride=1, depth=1, expansion=3, kernel=3),
]
_FEATURES = {
    "P2": dict(stage=1, stride=4, channels=24, spatial=160),
    "P3": dict(stage=3, stride=8, channels=32, spatial=80),
    "P4": dict(stage=5, stride=16, channels=96, spatial=40),
    "P5": dict(stage=7, stride=32, channels=320, spatial=20),
}
_ANCHORS = {
    "units": "input_pixels",
    "protocol": "preserve_stride_association_copy_P3_to_P2",
    "P2": [[9.25, 13.5], [12.0, 16.25], [15.0, 16.25]],
    "P3": [[9.25, 13.5], [12.0, 16.25], [15.0, 16.25]],
    "P4": [[12.62, 20.25], [19.75, 15.25], [15.0, 22.75]],
    "P5": [[16.5, 22.38], [21.25, 19.5], [21.25, 27.5]],
    "recluster": False,
}
_PROTOTYPES = {
    "R0": dict(name="R0_P234_U32_direct", levels=["P2", "P3", "P4"],
               last_active_stage=5, neck_channels=32, extra_DS_tower_blocks=0),
    "R1": dict(name="R1_P234_U48_direct", levels=["P2", "P3", "P4"],
               last_active_stage=5, neck_channels=48, extra_DS_tower_blocks=0),
    "R7": dict(name="R7_P2345_U48_direct", levels=["P2", "P3", "P4", "P5"],
               last_active_stage=7, neck_channels=48, extra_DS_tower_blocks=0),
    "R8": dict(name="R8_P234_U48_extraDS", levels=["P2", "P3", "P4"],
               last_active_stage=5, neck_channels=48, extra_DS_tower_blocks=1),
}


def _canonical_spec() -> dict[str, Any]:
    # Validate against an independent contract, never against a mutable file
    # used as its own source of truth. JSON also distinguishes True/1/1.0.
    return dict(
        schema_version=2,
        name="sonar_detector_v0_2_prototypes",
        status="executable_software_prototype_contract_not_trained_not_board_measured",
        preserve_v0_1=True,
        executable_prototype_config=True,
        executable_search_config=False,
        architecture_space=dict(
            input=[1, 640, 640], num_classes=2,
            stem=dict(channels=32, kernel=3, stride=2),
            stages=deepcopy(_STAGES), features=deepcopy(_FEATURES),
            inactive_tail="omit_after_last_active_stage",
            neck=dict(lateral="conv1_bn_relu_per_level",
                      fusion="fixed_top_down_nearest_add",
                      top_down_source="fused_before_output_smoothing",
                      smoothing="dw3_bn_relu_then_pw1_bn_relu_per_level"),
            head=dict(prediction_kernel=1, prediction_bias=True,
                      prediction_activation="linear", anchors_per_location=3,
                      prediction_channels=21, objectness_bias_prior=0.01,
                      class_bias=0.0),
            anchors=deepcopy(_ANCHORS), prototypes=deepcopy(_PROTOTYPES),
        ),
        fixed_protocol=dict(
            reference="configs/baseline_sonar_frozen.yaml",
            split_policy="official_train_valid_only", test_policy="sealed",
            checkpoint_selection="valid_mAP50_95",
            postprocessing="frozen_thresholds_and_max_detections",
            repeated_P2_P3_anchor_tie="preserve_first_level_and_report",
        ),
    )


def load_spec(spec: str | Path | Mapping[str, Any] = SPEC_PATH) -> dict[str, Any]:
    """Load exactly the registered four-prototype V0.2 contract."""
    if isinstance(spec, (str, Path)):
        value = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
    elif isinstance(spec, Mapping):
        value = deepcopy(dict(spec))
    else:
        raise ValueError("V0.2 specification must be a path or mapping")
    try:
        actual = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(_canonical_spec(), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("V0.2 specification contains unsupported values") from error
    if actual != expected:
        raise ValueError("Specification differs from the registered V0.2 prototype contract")
    return value


def validate_sonar_architecture_v0_2(
    architecture: str | Mapping[str, Any],
    spec: str | Path | Mapping[str, Any] = SPEC_PATH,
) -> dict[str, str]:
    """Resolve a prototype; all levels, kernels, widths and anchors are derived."""
    load_spec(spec)
    if isinstance(architecture, str):
        value = {"prototype_id": architecture}
    elif isinstance(architecture, Mapping):
        value = dict(architecture)
    else:
        raise ValueError("V0.2 architecture must be a prototype ID or mapping")
    if set(value) != {"prototype_id"}:
        raise ValueError("V0.2 architecture supports only prototype_id; other fields are derived")
    prototype = value["prototype_id"]
    if not isinstance(prototype, str) or prototype not in _PROTOTYPES:
        raise ValueError(f"Unknown V0.2 prototype_id: {prototype!r}")
    return {"prototype_id": prototype}


def architecture_id(
    architecture: str | Mapping[str, Any],
    spec: str | Path | Mapping[str, Any] = SPEC_PATH,
) -> str:
    arch = validate_sonar_architecture_v0_2(architecture, spec)
    return _PROTOTYPES[arch["prototype_id"]]["name"]


def _ds_layers(channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
    ]


class LightweightFPNNeck(nn.Module):
    """Nearest/Add FPN, smoothed only after every top-down Add is complete."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for channels in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Sequential(*_ds_layers(out_channels)) for _ in self.in_channels
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(features)}")
        fused = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        for index in range(len(fused) - 2, -1, -1):
            fused[index] = fused[index] + F.interpolate(
                fused[index + 1], size=fused[index].shape[-2:], mode="nearest"
            )
        return tuple(conv(feature) for conv, feature in zip(self.output_convs, fused))


class LightweightDetectionHead(nn.Module):
    """Per-level optional DS tower followed by a biased, linear 1x1 prediction."""

    def __init__(self, in_channels: int, num_levels: int, extra_tower_blocks: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = 2
        self.anchors_per_location = 3
        self.outputs_per_anchor = 7
        self.prediction_channels = 21
        self.extra_tower_blocks = extra_tower_blocks
        self.towers = nn.ModuleList()
        for _ in range(num_levels):
            layers: list[nn.Module] = []
            for _ in range(extra_tower_blocks):
                layers.extend(_ds_layers(in_channels))
            prediction = nn.Conv2d(in_channels, self.prediction_channels, 1, bias=True)
            with torch.no_grad():
                bias = prediction.bias.view(self.anchors_per_location, self.outputs_per_anchor)
                bias[:, 4].fill_(math.log(0.01 / 0.99))
                bias[:, 5:].zero_()
            self.towers.append(nn.Sequential(*layers, prediction))

    @property
    def prediction_convs(self) -> tuple[nn.Conv2d, ...]:
        return tuple(tower[-1] for tower in self.towers)

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.towers):
            raise ValueError(f"expected {len(self.towers)} feature maps, got {len(features)}")
        return tuple(tower(feature) for tower, feature in zip(self.towers, features))


class SonarCandidateDetectorV02(nn.Module):
    """Physically truncated, independently initialized V0.2 software detector."""

    def __init__(self, spec: Mapping[str, Any], architecture: Mapping[str, Any]) -> None:
        super().__init__()
        self.spec = load_spec(spec)
        self.candidate_architecture = validate_sonar_architecture_v0_2(architecture, self.spec)
        self.prototype_id = self.candidate_architecture["prototype_id"]
        space = self.spec["architecture_space"]
        prototype = space["prototypes"][self.prototype_id]
        self.prototype_name = prototype["name"]
        self.input_channels = 1
        self.image_size = 640
        self.levels = tuple(prototype["levels"])
        self.last_active_stage = prototype["last_active_stage"]
        self.feature_indices = tuple(space["features"][level]["stage"] for level in self.levels)
        self.feature_strides = tuple(space["features"][level]["stride"] for level in self.levels)
        self.anchor_sizes = tuple(tuple(map(tuple, space["anchors"][level])) for level in self.levels)
        stages = []
        for stage in space["stages"][:self.last_active_stage + 1]:
            block = dict(op=stage["op"], kernel_size=stage["kernel"],
                         expand_ratio=stage.get("expansion", 1), stride=stage["stride"])
            stages.append(dict(channels=stage["channels"], depth=1, stride=stage["stride"], blocks=[block]))
        self.architecture = ArchitectureSpec.from_dict(dict(
            input_channels=1, stem_channels=32, stem_stride=2,
            post_stem_downsample_stride=1, stages=stages,
        ))
        self.backbone = HWNASFeatureBackbone(self.architecture, self.feature_indices)
        neck_channels = prototype["neck_channels"]
        self.neck = LightweightFPNNeck(
            [space["features"][level]["channels"] for level in self.levels], neck_channels
        )
        self.head = LightweightDetectionHead(
            neck_channels, len(self.levels), prototype["extra_DS_tower_blocks"]
        )

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 640, 640):
            raise ValueError("V0.2 requires [B,1,640,640] input")
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        return dict(backbone_features=backbone_features, neck_features=neck_features,
                    predictions=self.head(neck_features))

    def loss_config(self, baseline_config: Mapping[str, Any]) -> DetectionLossConfig:
        base = DetectionLossConfig.from_config(baseline_config)
        return replace(base, num_classes=2, image_size=640,
                       strides=self.feature_strides, anchor_sizes=self.anchor_sizes,
                       class_loss_weights=(1.0, 1.0))

    def architecture_contract(self) -> dict[str, Any]:
        """JSON-compatible description for provenance and future graph export."""
        return dict(
            schema_version=2, prototype_id=self.prototype_id, name=self.prototype_name,
            architecture=deepcopy(self.candidate_architecture), input_chw=[1, 640, 640],
            levels=list(self.levels), feature_indices=list(self.feature_indices),
            strides=list(self.feature_strides), anchors=deepcopy(list(self.anchor_sizes)),
            physical_backbone_stage_count=len(self.backbone.stages),
            neck_channels=self.neck.out_channels,
            extra_DS_tower_blocks=self.head.extra_tower_blocks,
            top_down_source="fused_before_output_smoothing",
            prediction=dict(kernel=1, channels=21, bias=True, activation="linear"),
        )


def build_sonar_candidate_v0_2(
    spec: str | Path | Mapping[str, Any] = SPEC_PATH,
    architecture: str | Mapping[str, Any] = "R1",
    *, num_classes: int = 2, image_size: int = 640,
) -> SonarCandidateDetectorV02:
    if type(num_classes) is not int or num_classes != 2:
        raise ValueError("V0.2 prototype contract requires num_classes=2")
    if type(image_size) is not int or image_size != 640:
        raise ValueError("V0.2 prototype contract requires image_size=640")
    space = load_spec(spec)
    arch = validate_sonar_architecture_v0_2(architecture, space)
    return SonarCandidateDetectorV02(space, arch)


__all__ = ["SPEC_PATH", "load_spec", "validate_sonar_architecture_v0_2",
           "architecture_id", "build_sonar_candidate_v0_2", "SonarCandidateDetectorV02",
           "LightweightFPNNeck", "LightweightDetectionHead"]
