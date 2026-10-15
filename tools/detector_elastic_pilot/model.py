"""Four-scale ELASTIC/PAN-inspired neck pilots with the unchanged B1 backbone.

This is a registered engineering adaptation, not an original-paper reproduction.
E0 and E1 differ only by an executed bottom-up PAN path. No pretrained weight,
accuracy, quantization correctness or FPGA latency is asserted by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from hwnas_fpga.models.builder import MBConvBlock
from hwnas_fpga.models.detection import detection_feature_channels
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_candidate_v0_2 import LightweightDetectionHead
from hwnas_fpga.training.detection_trainer import DetectionLossConfig

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs/detector_elastic_pilot_model.yaml"
B1_SPEC_PATH = ROOT / "configs/search_space_sonar_v0_1.yaml"
B1_ARCH = {
    "output_combination": "P2_P3_P4_P5",
    "kernels": {"stage1": 3, "stage2": 5, "stage4": 5,
                "stage5": 5, "stage6": 5, "stage7": 5},
}
INITIALIZATION_POLICY = "B1_LEGACY_ELASTIC_NECK_HEAD_SEMANTIC_CPU_SEED_V1"
_LEVELS = ("P2", "P3", "P4", "P5")
_PROTOTYPES = {
    "E0": dict(name="B1_P2345_U48_MB3E3_topdown", bottom_up_pan=False),
    "E1": dict(name="B1_P2345_U48_MB3E3_topdown_PAN", bottom_up_pan=True),
}


def _canonical_spec() -> dict[str, Any]:
    return dict(
        schema_version=1, name="detector_elastic_four_scale_pilot",
        status="software_pilots_not_trained_not_board_measured",
        relation_to_papers="ELASTIC_and_PAN_inspired_four_scale_engineering_adaptation_not_paper_reproduction",
        preserve_previous_runs=True, executable_search_config=False,
        backbone=dict(
            source="configs/search_space_sonar_v0_1.yaml",
            source_sha256="ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4",
            architecture=deepcopy(B1_ARCH), structure="complete_B1_8_stages",
            weights="train_from_scratch_all_trainable",
        ),
        input=[1, 640, 640], num_classes=2,
        features=dict(levels=list(_LEVELS), indices=[1, 3, 5, 7],
                      strides=[4, 8, 16, 32], channels=[24, 32, 96, 320]),
        neck=dict(
            channels=48, lateral="conv1_bn_relu6_per_level",
            top_down="nearest_add_then_MBConv_at_P4_P3_P2",
            top_down_source="post_MBConv_fused_feature",
            upsample_preprojection="omitted_equal_width_engineering_adaptation",
            P5_top_down="lateral_only_no_unused_MBConv",
            fusion_block=dict(kernel=3, expansion=3, stride=1, depth=1,
                              expand="conv1_bn_relu6", depthwise="dw3_bn_relu6",
                              project="linear_conv1_bn", residual=True),
            batch_norm=dict(eps=1e-5, momentum=0.1,
                            source="local_builder_defaults_not_author_BN_reproduction"),
            bottom_up="maxpool2_stride2_conv1_bn_relu6_add_then_MBConv_at_P3_P4_P5",
            bottom_up_source="previous_bottom_up_output",
            bottom_up_P2="unchanged_top_down_P2",
        ),
        head=dict(prediction_kernel=1, prediction_channels=21,
                  prediction_bias=True, prediction_activation="linear",
                  anchors_per_location=3, extra_DS_tower_blocks=0,
                  objectness_bias_prior=0.01, class_bias=0.0),
        prototypes=deepcopy(_PROTOTYPES),
        initialization=dict(
            seeds=[42, 43, 44], policy=INITIALIZATION_POLICY,
            paired_components=["backbone", "lateral", "top_down", "prediction"],
            PAN="independent_semantic_module_seeds",
            global_rng="preserve_CPU_and_do_not_touch_CUDA",
        ),
        fixed_protocol=dict(
            reference="configs/baseline_sonar_frozen.yaml",
            anchors="exact_B1_stride_association_including_repeated_P2_P3",
            assignment_loss_decode="unchanged_frozen_detection_protocol",
            split_policy="official_train_valid_only", test_policy="sealed",
            checkpoint_selection="valid_mAP50_95_earliest_epoch",
            historical_B1_control="screening_reference_not_new_paired_control",
        ),
        pilot_search_space=dict(
            backbone_options=["fixed_B1"], neck_options=["E0", "E1"],
            candidate_count=2, original_144_backbone_pool="unchanged_not_activated",
            integration="independent_neck_backend_candidate_adapter",
            future_backbone_extension="NOT_IMPLEMENTED_OR_EXECUTED",
        ),
    )


def load_spec(spec: str | Path | Mapping[str, Any] = SPEC_PATH) -> dict[str, Any]:
    if isinstance(spec, (str, Path)):
        value = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
    elif isinstance(spec, Mapping):
        value = deepcopy(dict(spec))
    else:
        raise ValueError("ELASTIC pilot specification must be a path or mapping")
    try:
        actual = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(_canonical_spec(), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("ELASTIC pilot specification contains unsupported values") from error
    if actual != expected:
        raise ValueError("Specification differs from registered ELASTIC pilot contract")
    return value


def _cpu_seed(seed: int) -> None:
    torch.random.default_generator.manual_seed(seed)


def _semantic_seed(seed: int, name: str) -> int:
    encoded = f"{INITIALIZATION_POLICY}:{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _hash_named_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            separators=(",", ":")).encode("utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


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
        self.levels = _LEVELS
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


class ElasticPilotDetector(nn.Module):
    def __init__(self, prototype: str, seed: int, spec: Mapping[str, Any]) -> None:
        super().__init__()
        self.spec = load_spec(spec)
        self.prototype_id = prototype
        self.prototype_name = _PROTOTYPES[prototype]["name"]
        self.seed = seed
        self.input_channels = 1
        self.image_size = 640
        self.last_active_stage = 7
        _cpu_seed(seed)
        legacy = build_sonar_candidate(B1_SPEC_PATH, deepcopy(B1_ARCH),
                                       num_classes=2, image_size=640)
        self.architecture = legacy.architecture
        self.backbone = legacy.backbone
        self.levels = legacy.levels
        self.feature_indices = legacy.feature_indices
        self.feature_strides = legacy.feature_strides
        self.anchor_sizes = legacy.anchor_sizes
        _cpu_seed(_semantic_seed(seed, "neck_container"))
        self.neck = ElasticFourScaleNeck(
            detection_feature_channels(self.architecture, self.feature_indices), 48,
            _PROTOTYPES[prototype]["bottom_up_pan"],
        )
        _initialize_semantic_modules(self.neck, seed, "neck")
        _cpu_seed(_semantic_seed(seed, "head_container"))
        self.head = LightweightDetectionHead(48, len(self.levels), extra_tower_blocks=0)
        for level, tower in zip(self.levels, self.head.towers):
            _cpu_seed(_semantic_seed(seed, f"prediction.{level}"))
            prediction = tower[-1]
            prediction.reset_parameters()
            # Keep the original linear detection output and bias prior.
            with torch.no_grad():
                bias = prediction.bias.view(3, 7)
                bias[:, 4].fill_(math.log(0.01 / 0.99))
                bias[:, 5:].zero_()
        self.candidate_architecture = dict(
            prototype_id=prototype, backbone_architecture=deepcopy(B1_ARCH),
            neck=dict(**deepcopy(self.spec["neck"]), bottom_up_pan=self.neck.bottom_up_pan),
            head=deepcopy(self.spec["head"]),
        )
        self._initialization_identity = None
        if not any(value.is_meta for value in self.state_dict().values()):
            prediction_tensors = [
                (f"{level}.{name}", tensor)
                for level, tower in zip(self.levels, self.head.towers)
                for name, tensor in tower[-1].state_dict().items()
            ]
            self._initialization_identity = dict(
                schema_version=1, seed=seed, policy=INITIALIZATION_POLICY,
                backbone_sha256=_hash_named_tensors(self.backbone.state_dict().items()),
                common_neck_sha256=_hash_named_tensors(self.neck.common_state_dict().items()),
                prediction_sha256=_hash_named_tensors(prediction_tensors),
                full_neck_sha256=_hash_named_tensors(self.neck.state_dict().items()),
                pan_sha256=_hash_named_tensors(
                    (name, value) for name, value in self.neck.state_dict().items()
                    if name.startswith(("bottom_up_projections.", "bottom_up_blocks."))
                ),
            )

    def initialization_identity(self) -> dict[str, Any]:
        if self._initialization_identity is None:
            raise ValueError("Meta construction has no numerical initialization identity")
        return deepcopy(self._initialization_identity)

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 640, 640):
            raise ValueError("ELASTIC pilot requires [B,1,640,640] input")
        features = self.backbone(x)
        neck = self.neck(features)
        return dict(backbone_features=features, neck_features=neck,
                    predictions=self.head(neck))

    def loss_config(self, baseline_config: Mapping[str, Any]) -> DetectionLossConfig:
        base = DetectionLossConfig.from_config(baseline_config)
        return replace(base, num_classes=2, image_size=640,
                       strides=self.feature_strides, anchor_sizes=self.anchor_sizes,
                       class_loss_weights=(1.0, 1.0))

    def architecture_contract(self) -> dict[str, Any]:
        return dict(
            schema_version=1, prototype_id=self.prototype_id, name=self.prototype_name,
            architecture=deepcopy(self.candidate_architecture), input_chw=[1, 640, 640],
            levels=list(self.levels), feature_indices=list(self.feature_indices),
            strides=list(self.feature_strides), anchors=deepcopy(list(self.anchor_sizes)),
            physical_backbone_stage_count=len(self.backbone.stages),
            backbone_reference="complete_B1_no_stage_kernel_channel_change",
            backbone_architecture=deepcopy(B1_ARCH), backbone_weights_trainable=True,
            neck_channels=self.neck.out_channels,
            top_down_MBConv_count=len(self.neck.top_down_blocks),
            bottom_up_MBConv_count=len(self.neck.bottom_up_blocks),
            bottom_up_projection_count=len(self.neck.bottom_up_projections),
            bottom_up_pan=self.neck.bottom_up_pan, extra_DS_tower_blocks=0,
            top_down_source="post_MBConv_fused_feature",
            prediction=dict(kernel=1, channels=21, bias=True, activation="linear"),
            initialization_policy=INITIALIZATION_POLICY,
            relation_to_papers=self.spec["relation_to_papers"],
            pilot_search_space=deepcopy(self.spec["pilot_search_space"]),
            board_latency_status="NOT_MEASURED",
        )


def build_model(prototype: str = "E0", seed: int = 42,
                *, spec: str | Path | Mapping[str, Any] = SPEC_PATH) -> ElasticPilotDetector:
    contract = load_spec(spec)
    if not isinstance(prototype, str) or prototype not in _PROTOTYPES:
        raise ValueError(f"Unknown ELASTIC pilot prototype: {prototype!r}")
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError("ELASTIC pilot requires registered integer seed 42, 43 or 44")
    with torch.random.fork_rng(devices=[]):
        return ElasticPilotDetector(prototype, seed, contract)


__all__ = ["ROOT", "SPEC_PATH", "B1_SPEC_PATH", "B1_ARCH", "INITIALIZATION_POLICY",
           "load_spec", "build_model", "ElasticFourScaleNeck", "ElasticPilotDetector"]
