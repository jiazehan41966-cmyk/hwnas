"""Differentiable adapter for the registered 72-candidate ELASTIC neck space.

The complete B1 backbone, four scales, 48-channel neck, expansion three,
depth one, anchors and detection loss remain fixed. Only the three top-down
MBConv kernels (3/5) and the whole PAN path are architecture choices; PAN has
three further kernel choices only when present. This module implements the
project supernet and export contract, not the MODNAS optimization algorithm.

Differentiable mixtures execute multiple branches. Their real search runtime
must be charged separately from the cost of a selected static candidate.
Training dense branches also updates their BN buffers; final candidates need
train-only BN calibration before any independent evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.models.builder import MBConvBlock
from hwnas_fpga.training.detection_trainer import DetectionLossConfig
from tools.detector_elastic_pilot.model import B1_ARCH, build_model as build_pilot


SEARCH_SPACE_VERSION = "elastic_neck72_v1"
LEVELS = ("P2", "P3", "P4", "P5")
TD_LEVELS = LEVELS[:-1]
PAN_LEVELS = LEVELS[1:]
KERNEL_CHOICES = (3, 5)
ARCHITECTURE_COUNT = 72
INITIALIZATION_POLICY = "ELASTIC_NECK72_PILOT_K3_PAIRED_K5_DEPTHWISE_V1"


def validate_architecture(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize only this new space; absent PAN cannot carry choices."""
    if not isinstance(architecture, Mapping):
        raise ValueError("architecture must be a mapping")
    if set(architecture) != {"pan", "td_kernels", "pan_kernels"}:
        raise ValueError("architecture fields must be pan, td_kernels, pan_kernels")
    pan = architecture["pan"]
    if type(pan) is not bool:
        raise ValueError("pan must be a bool controlling the whole PAN path")
    result = {"pan": pan}
    for field, count in (("td_kernels", 3), ("pan_kernels", 3 if pan else 0)):
        kernels = architecture[field]
        if not isinstance(kernels, (tuple, list)) or len(kernels) != count:
            raise ValueError(f"{field} requires {count} active kernel choices")
        if any(type(kernel) is not int or kernel not in KERNEL_CHOICES for kernel in kernels):
            raise ValueError(f"{field} choices must be integer 3 or 5")
        result[field] = list(kernels)
    return result


def enumerate_architectures():
    """Yield exactly 8 E0 + 64 E1 architectures, never an inactive-PAN tail."""
    for pan in (False, True):
        for td in itertools.product(KERNEL_CHOICES, repeat=3):
            for bottom in itertools.product(KERNEL_CHOICES, repeat=3 if pan else 0):
                yield dict(pan=pan, td_kernels=list(td), pan_kernels=list(bottom))


def architecture_id(architecture: Mapping[str, Any]) -> str:
    arch = validate_architecture(architecture)
    td = "_".join(f"{level}k{kernel}" for level, kernel in zip(TD_LEVELS, arch["td_kernels"]))
    suffix = ""
    if arch["pan"]:
        suffix = "_PAN_" + "_".join(
            f"{level}k{kernel}" for level, kernel in zip(PAN_LEVELS, arch["pan_kernels"]))
    return ("E1_" if arch["pan"] else "E0_") + "TD_" + td + suffix


def architecture_hash(architecture: Mapping[str, Any]) -> str:
    encoded = json.dumps(validate_architecture(architecture), sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def one_hot_weights(architecture: Mapping[str, Any], *, device=None,
                    dtype=torch.float32) -> dict[str, torch.Tensor]:
    arch = validate_architecture(architecture)
    def kernel_weights(kernels):
        return torch.tensor([[float(kernel == 3), float(kernel == 5)] for kernel in kernels],
                            device=device, dtype=dtype)
    return dict(
        pan_weights=torch.tensor([float(not arch["pan"]), float(arch["pan"])],
                                 device=device, dtype=dtype),
        td_kernel_weights=kernel_weights(arch["td_kernels"]),
        # These placeholders are not architecture variables when PAN is absent.
        pan_kernel_weights=kernel_weights(arch["pan_kernels"] or [3, 3, 3]),
    )


def _validate_classes(num_classes: int) -> None:
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")


def _kernel_five(k3: MBConvBlock, seed: int, name: str) -> MBConvBlock:
    """Pair matching parameters with k3; independently seed the new DW shape."""
    encoded = f"{INITIALIZATION_POLICY}:{seed}:{name}:k5".encode("utf-8")
    branch_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(branch_seed)
        k5 = MBConvBlock(48, 48, 5, stride=1, expand_ratio=3)
    old_state, state = k3.state_dict(), k5.state_dict()
    paired = {key: old_state[key] if value.shape == old_state[key].shape else value
              for key, value in state.items()}
    k5.load_state_dict(paired, strict=True)
    return k5


class _PredictionHead(nn.Module):
    """The pilot's direct, linear 1x1 prediction generalized to class count."""
    def __init__(self, num_classes: int, seed: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_location = 3
        self.outputs_per_anchor = num_classes + 5
        self.prediction_channels = 3 * self.outputs_per_anchor
        self.extra_tower_blocks = 0
        self.towers = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            for level in LEVELS:
                encoded = f"{INITIALIZATION_POLICY}:{seed}:prediction.{level}:nc{num_classes}".encode("utf-8")
                torch.random.default_generator.manual_seed(
                    int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1))
                prediction = nn.Conv2d(48, self.prediction_channels, 1, bias=True)
                with torch.no_grad():
                    bias = prediction.bias.view(3, self.outputs_per_anchor)
                    bias[:, 4].fill_(math.log(0.01 / 0.99))
                    bias[:, 5:].zero_()
                self.towers.append(nn.Sequential(prediction))

    @property
    def prediction_convs(self):
        return tuple(tower[-1] for tower in self.towers)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("prediction head requires four scales")
        return tuple(tower(feature) for tower, feature in zip(self.towers, features))


def _loss_config(model, baseline_config) -> DetectionLossConfig:
    return replace(DetectionLossConfig.from_config(baseline_config),
                   num_classes=model.num_classes, image_size=640,
                   strides=model.feature_strides, anchor_sizes=model.anchor_sizes,
                   class_loss_weights=tuple(1.0 for _ in range(model.num_classes)))


def _check_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
        raise ValueError("new ELASTIC adapter requires [B,1,640,640] input")


class StaticElasticNeckDetector(nn.Module):
    """Independent selected network with no mixture branches or PAN tail."""
    def __init__(self, architecture, seed=42, num_classes=2) -> None:
        super().__init__()
        _validate_classes(num_classes)
        self.candidate_architecture = validate_architecture(architecture)
        self.num_classes = num_classes
        self.seed = seed
        self.image_size, self.input_channels = 640, 1
        self.search_space_version = SEARCH_SPACE_VERSION
        pilot = build_pilot("E1" if self.candidate_architecture["pan"] else "E0", seed)
        self.architecture = pilot.architecture
        self.backbone, self.neck = pilot.backbone, pilot.neck
        self.head = pilot.head if num_classes == 2 else _PredictionHead(num_classes, seed)
        self.levels, self.feature_indices = pilot.levels, pilot.feature_indices
        self.feature_strides, self.anchor_sizes = pilot.feature_strides, pilot.anchor_sizes
        for level, kernel in zip(TD_LEVELS, self.candidate_architecture["td_kernels"]):
            if kernel == 5:
                self.neck.top_down_blocks[level] = _kernel_five(
                    self.neck.top_down_blocks[level], seed, f"top_down.{level}")
        for level, kernel in zip(PAN_LEVELS, self.candidate_architecture["pan_kernels"]):
            if kernel == 5:
                self.neck.bottom_up_blocks[level] = _kernel_five(
                    self.neck.bottom_up_blocks[level], seed, f"bottom_up.{level}")

    def forward(self, images):
        _check_images(images)
        features = self.backbone(images)
        neck = self.neck(features)
        return dict(backbone_features=features, neck_features=neck,
                    predictions=self.head(neck))

    def loss_config(self, baseline_config):
        return _loss_config(self, baseline_config)

    def architecture_contract(self):
        return dict(
            search_space_version=SEARCH_SPACE_VERSION,
            architecture=deepcopy(self.candidate_architecture),
            architecture_id=architecture_id(self.candidate_architecture),
            architecture_sha256=architecture_hash(self.candidate_architecture),
            fixed_backbone_architecture=deepcopy(B1_ARCH),
            input_chw=[1, 640, 640], num_classes=self.num_classes,
            levels=list(self.levels), feature_indices=list(self.feature_indices),
            feature_strides=list(self.feature_strides), anchors=deepcopy(self.anchor_sizes),
            neck_channels=48, expansion=3, block_depth=1,
            prediction_channels=3 * (self.num_classes + 5),
            prediction_kernel=1, prediction_activation="linear", prediction_bias=True,
            top_down_source="post_MBConv_fused_feature",
            bottom_up_source="previous_bottom_up_output",
            initialization_policy=INITIALIZATION_POLICY,
            relation_to_papers="new_space_engineering_adapter_not_author_reproduction",
            board_latency_status="NOT_MEASURED",
        )


def _constant_one_hot_index(weights: torch.Tensor):
    if weights.requires_grad:
        return None
    detached = weights.detach()
    if bool(torch.equal(detached, detached.new_tensor([1.0, 0.0]))):
        return 0
    if bool(torch.equal(detached, detached.new_tensor([0.0, 1.0]))):
        return 1
    return None


class _MixedMBConv(nn.Module):
    """Mix complete conv/BN/activation/residual blocks, preserving BN pairing."""
    def __init__(self, k3: MBConvBlock, seed: int, name: str) -> None:
        super().__init__()
        self.branches = nn.ModuleDict({"k3": k3, "k5": _kernel_five(k3, seed, name)})

    def forward(self, features, weights):
        selected = _constant_one_hot_index(weights)
        if selected is not None:
            return self.branches[f"k{KERNEL_CHOICES[selected]}"](features)
        return (weights[0] * self.branches["k3"](features)
                + weights[1] * self.branches["k5"](features))


class ElasticNeckSupernet(nn.Module):
    """Fixed B1 plus differentiable k3/k5 neck and a whole-path PAN gate."""
    def __init__(self, seed=42, num_classes=2) -> None:
        super().__init__()
        _validate_classes(num_classes)
        pilot = build_pilot("E1", seed)
        self.seed, self.num_classes = seed, num_classes
        self.input_channels, self.image_size = 1, 640
        self.search_space_version = SEARCH_SPACE_VERSION
        self.architecture = pilot.architecture
        self.backbone, self.neck = pilot.backbone, pilot.neck
        self.head = pilot.head if num_classes == 2 else _PredictionHead(num_classes, seed)
        self.levels, self.feature_indices = pilot.levels, pilot.feature_indices
        self.feature_strides, self.anchor_sizes = pilot.feature_strides, pilot.anchor_sizes
        for level in TD_LEVELS:
            self.neck.top_down_blocks[level] = _MixedMBConv(
                self.neck.top_down_blocks[level], seed, f"top_down.{level}")
        for level in PAN_LEVELS:
            self.neck.bottom_up_blocks[level] = _MixedMBConv(
                self.neck.bottom_up_blocks[level], seed, f"bottom_up.{level}")

    @staticmethod
    def _weights(architecture_weights):
        expected = dict(pan_weights=(2,), td_kernel_weights=(3, 2),
                        pan_kernel_weights=(3, 2))
        if not isinstance(architecture_weights, Mapping) or set(architecture_weights) != set(expected):
            raise ValueError("architecture_weights requires pan_weights, td_kernel_weights, pan_kernel_weights")
        for name, shape in expected.items():
            value = architecture_weights[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating tensor of shape {shape}")
            check = value.detach()
            if not bool(torch.isfinite(check).all()) or bool((check < -1e-6).any()):
                raise ValueError(f"{name} requires finite nonnegative normalized weights")
            if not bool(torch.allclose(check.sum(-1), torch.ones_like(check.sum(-1)), atol=1e-5, rtol=1e-5)):
                raise ValueError(f"{name} last dimension must sum to one")
        return architecture_weights

    def _top_down(self, images, weights):
        _check_images(images)
        features = self.backbone(images)
        fused = [self.neck.lateral_convs[level](feature)
                 for level, feature in zip(LEVELS, features)]
        for index in range(2, -1, -1):
            merged = fused[index] + F.interpolate(
                fused[index + 1], size=fused[index].shape[-2:], mode="nearest")
            fused[index] = self.neck.top_down_blocks[LEVELS[index]](
                merged, weights["td_kernel_weights"][index])
        return features, tuple(fused)

    def _bottom_up(self, fused, weights):
        outputs = [fused[0]]
        for index, level in enumerate(PAN_LEVELS, start=1):
            reduced = self.neck.bottom_up_projections[level](
                self.neck.bottom_up_pool(outputs[-1]))
            if reduced.shape[-2:] != fused[index].shape[-2:]:
                raise ValueError("PAN requires exact factor-two adjacent feature sizes")
            outputs.append(self.neck.bottom_up_blocks[level](
                fused[index] + reduced, weights["pan_kernel_weights"][index - 1]))
        return tuple(outputs)

    def _result(self, features, neck):
        return dict(backbone_features=features, neck_features=neck,
                    predictions=self.head(neck))

    def forward_combinations(self, images, architecture_weights):
        """Two independent outputs for PAN-weighted losses in the search runner.

        Both paths execute even for a zero PAN weight. Applying that weight to
        E1's loss gives zero PAN parameter/kernel gradient when E0 is selected.
        BN buffers still follow dense training; use eval mode on validation.
        """
        weights = self._weights(architecture_weights)
        features, fused = self._top_down(images, weights)
        pan = self._bottom_up(fused, weights)
        return {"E0": self._result(features, fused), "E1": self._result(features, pan)}

    def forward(self, images, architecture_weights):
        """Same four-scale interface; exact one-hot gates export as one subnet.

        For architecture optimization prefer forward_combinations and the
        PAN-weighted two criterion losses over a loss of soft mixed logits.
        """
        weights = self._weights(architecture_weights)
        selected = _constant_one_hot_index(weights["pan_weights"])
        if selected is not None:
            features, fused = self._top_down(images, weights)
            neck = fused if selected == 0 else self._bottom_up(fused, weights)
            return self._result(features, neck)
        outputs = self.forward_combinations(images, weights)
        gate = weights["pan_weights"]
        return dict(backbone_features=outputs["E0"]["backbone_features"],
                    neck_features=tuple(gate[0] * left + gate[1] * right for left, right in
                                        zip(outputs["E0"]["neck_features"], outputs["E1"]["neck_features"])),
                    predictions=tuple(gate[0] * left + gate[1] * right for left, right in
                                      zip(outputs["E0"]["predictions"], outputs["E1"]["predictions"])))

    def loss_config(self, baseline_config):
        return _loss_config(self, baseline_config)

    def candidate_state_mapping(self, architecture):
        arch = validate_architecture(architecture)
        candidate = StaticElasticNeckDetector(arch, self.seed, self.num_classes)
        mapping = {}
        for key in candidate.state_dict():
            source = key
            for group, levels, kernels in (
                    ("top_down_blocks", TD_LEVELS, arch["td_kernels"]),
                    ("bottom_up_blocks", PAN_LEVELS, arch["pan_kernels"])):
                prefix = f"neck.{group}."
                if key.startswith(prefix):
                    level, tail = key[len(prefix):].split(".", 1)
                    kernel = kernels[levels.index(level)]
                    source = f"{prefix}{level}.branches.k{kernel}.{tail}"
                    break
            mapping[key] = source
        return candidate, mapping

    def export_subnet(self, architecture):
        candidate, mapping = self.candidate_state_mapping(architecture)
        parameter = next(self.parameters())
        candidate.to(device=parameter.device, dtype=parameter.dtype)
        state = self.state_dict()
        candidate.load_state_dict({target: state[source].detach().clone()
                                   for target, source in mapping.items()}, strict=True)
        candidate.train(self.training)
        candidate.supernet_state_mapping = deepcopy(mapping)
        return candidate


__all__ = ["SEARCH_SPACE_VERSION", "ARCHITECTURE_COUNT", "INITIALIZATION_POLICY",
           "LEVELS", "TD_LEVELS", "PAN_LEVELS", "KERNEL_CHOICES",
           "validate_architecture", "enumerate_architectures", "architecture_id",
           "architecture_hash", "one_hot_weights", "StaticElasticNeckDetector",
           "ElasticNeckSupernet"]
