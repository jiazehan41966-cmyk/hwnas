"""Controlled neck/head substitutions on the complete, unchanged B1 backbone.

This independent namespace is outside the completed V0.2 training dependencies.
All parameters remain trainable; the backbone structure is fixed, not its weights.
No trained checkpoint, FPGA timing or deployment behavior is implied here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml

from hwnas_fpga.models.detection import detection_feature_channels
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_candidate_v0_2 import (
    LightweightDetectionHead, LightweightFPNNeck,
)
from hwnas_fpga.training.detection_trainer import DetectionLossConfig

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs/detector_b1_light_model.yaml"
B1_SPEC_PATH = ROOT / "configs/search_space_sonar_v0_1.yaml"
B1_ARCH = {
    "output_combination": "P2_P3_P4_P5",
    "kernels": {"stage1": 3, "stage2": 5, "stage4": 5,
                "stage5": 5, "stage6": 5, "stage7": 5},
}
INITIALIZATION_POLICY = "B1_LEGACY_BACKBONE_SEMANTIC_NECK_HEAD_CPU_SEED_V1"
_PROTOTYPES = {
    "L1": {"name": "B1_P2345_U48_DS1", "extra_DS_tower_blocks": 1},
    "L0": {"name": "B1_P2345_U48_direct", "extra_DS_tower_blocks": 0},
}


def _canonical_spec() -> dict[str, Any]:
    return dict(
        schema_version=1, name="detector_b1_light_controlled_prototypes",
        status="software_prototypes_not_board_measured", preserve_previous_runs=True,
        executable_search_config=False,
        backbone=dict(
            source="configs/search_space_sonar_v0_1.yaml",
            source_sha256="ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4",
            architecture=deepcopy(B1_ARCH), structure="complete_B1_8_stages",
            weights="train_from_scratch_all_trainable",
        ),
        input=[1, 640, 640], num_classes=2,
        features=dict(levels=["P2", "P3", "P4", "P5"],
                      indices=[1, 3, 5, 7], strides=[4, 8, 16, 32],
                      channels=[24, 32, 96, 320]),
        neck=dict(channels=48, lateral="conv1_bn_relu_per_level",
                  fusion="fixed_top_down_nearest_add",
                  top_down_source="fused_before_output_smoothing",
                  smoothing="dw3_bn_relu_then_pw1_bn_relu_per_level"),
        head=dict(prediction_kernel=1, prediction_channels=21,
                  prediction_bias=True, prediction_activation="linear",
                  anchors_per_location=3, objectness_bias_prior=0.01, class_bias=0.0),
        prototypes=deepcopy(_PROTOTYPES),
        initialization=dict(
            seeds=[42, 43, 44], policy=INITIALIZATION_POLICY,
            paired_components=["backbone", "neck", "prediction"],
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
    )


def load_spec(spec: str | Path | Mapping[str, Any] = SPEC_PATH) -> dict[str, Any]:
    """Reject any unregistered structural, anchor or initialization change."""
    if isinstance(spec, (str, Path)):
        value = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
    elif isinstance(spec, Mapping):
        value = deepcopy(dict(spec))
    else:
        raise ValueError("B1-light specification must be a path or mapping")
    try:
        actual = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(_canonical_spec(), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("B1-light specification contains unsupported values") from error
    if actual != expected:
        raise ValueError("Specification differs from registered B1-light contract")
    return value


def _cpu_seed(seed: int) -> None:
    # torch.manual_seed also seeds CUDA. This build only uses the CPU generator.
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
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        # Flatten first: scalar BN counters cannot be viewed directly as bytes.
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


class B1LightDetector(nn.Module):
    """Complete B1 feature backbone with one of two paired lightweight heads."""

    def __init__(self, prototype: str, seed: int,
                 spec: Mapping[str, Any]) -> None:
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
        # The legacy dense neck/head are discarded; no weights are transferred.
        _cpu_seed(_semantic_seed(seed, "neck"))
        self.neck = LightweightFPNNeck(
            detection_feature_channels(self.architecture, self.feature_indices), 48,
        )
        _cpu_seed(_semantic_seed(seed, "head_container"))
        self.head = LightweightDetectionHead(
            48, len(self.levels), _PROTOTYPES[prototype]["extra_DS_tower_blocks"],
        )
        # DS constructors consume different random numbers. Reset by semantic
        # level/component names so prediction tensors are paired despite indexes.
        for level, tower in zip(self.levels, self.head.towers):
            for index, layer in enumerate(tower[:-1]):
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    _cpu_seed(_semantic_seed(seed, f"tower.{level}.{index}"))
                    layer.reset_parameters()
            _cpu_seed(_semantic_seed(seed, f"prediction.{level}"))
            prediction = tower[-1]
            prediction.reset_parameters()
            with torch.no_grad():
                bias = prediction.bias.view(3, 7)
                bias[:, 4].fill_(math.log(0.01 / 0.99))
                bias[:, 5:].zero_()

        self.candidate_architecture = dict(
            prototype_id=prototype, backbone_architecture=deepcopy(B1_ARCH),
            neck=deepcopy(self.spec["neck"]),
            head=dict(**deepcopy(self.spec["head"]),
                      extra_DS_tower_blocks=self.head.extra_tower_blocks),
        )
        self._initialization_identity = None
        if not any(value.is_meta for value in self.state_dict().values()):
            self._initialization_identity = self._capture_initialization_identity()

    def _capture_initialization_identity(self) -> dict[str, Any]:
        prediction_tensors = []
        tower_tensors = []
        for level, tower in zip(self.levels, self.head.towers):
            prediction_tensors.extend(
                (f"{level}.{name}", tensor) for name, tensor in tower[-1].state_dict().items()
            )
            for index, layer in enumerate(tower[:-1]):
                tower_tensors.extend(
                    (f"{level}.{index}.{name}", tensor)
                    for name, tensor in layer.state_dict().items()
                )
        return dict(
            schema_version=1, seed=self.seed, policy=INITIALIZATION_POLICY,
            backbone_sha256=_hash_named_tensors(self.backbone.state_dict().items()),
            neck_sha256=_hash_named_tensors(self.neck.state_dict().items()),
            prediction_sha256=_hash_named_tensors(prediction_tensors),
            tower_sha256=_hash_named_tensors(tower_tensors),
        )

    def initialization_identity(self) -> dict[str, Any]:
        """Return construction-time evidence, unchanged by subsequent training."""
        if self._initialization_identity is None:
            raise ValueError("Meta construction has no numerical initialization identity")
        return deepcopy(self._initialization_identity)

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 640, 640):
            raise ValueError("B1-light requires [B,1,640,640] input")
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
            extra_DS_tower_blocks=self.head.extra_tower_blocks,
            top_down_source="fused_before_output_smoothing",
            prediction=dict(kernel=1, channels=21, bias=True, activation="linear"),
            initialization_policy=INITIALIZATION_POLICY,
            board_latency_status="NOT_MEASURED",
        )


def build_model(prototype: str = "L1", seed: int = 42,
                *, spec: str | Path | Mapping[str, Any] = SPEC_PATH) -> B1LightDetector:
    """Construct registered prototypes without consuming global random state."""
    contract = load_spec(spec)
    if not isinstance(prototype, str) or prototype not in _PROTOTYPES:
        raise ValueError(f"Unknown B1-light prototype: {prototype!r}")
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError("B1-light requires registered integer seed 42, 43 or 44")
    with torch.random.fork_rng(devices=[]):
        return B1LightDetector(prototype, seed, contract)


__all__ = ["ROOT", "SPEC_PATH", "B1_SPEC_PATH", "B1_ARCH", "load_spec",
           "INITIALIZATION_POLICY", "B1LightDetector", "build_model"]
