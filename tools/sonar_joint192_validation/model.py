"""Independent static adapter for the proposed sonar_backbone_neck192_v1 space.

Reuse canonical E0/E1 initialization, head, anchors and loss. This implements
architectures, not a NAS optimizer, detection-benefit claim or FPGA contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.models.builder import MBConvBlock
from hwnas_fpga.training.detection_trainer import DetectionLossConfig
from tools.detector_elastic_pilot.model import (
    B1_ARCH, build_model as build_pilot, _hash_named_tensors,
)

SEARCH_SPACE_VERSION = "sonar_backbone_neck192_v1"
ARCHITECTURE_COUNT = 192
INITIALIZATION_POLICY = "JOINT192_PILOT_SAME_SHAPE_PAIRED_DW_SEMANTIC_V1"
REFERENCE_ARCHITECTURES = {
    prototype: dict(
        backbone=dict(stage1=dict(kernel=3, downsample="strided_dw"),
                      stage2=dict(kernel=5)),
        neck=dict(td_ops=dict(P2="mbconv_k3", P3="mbconv_k3", P4="mbconv_k3"),
                  pan=prototype == "E1"),
    ) for prototype in ("E0", "E1")
}


def _mapping(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly {', '.join(keys)}")
    return value


def validate_architecture(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Reject implicit defaults, type aliases, inactive choices and extra fields."""
    arch = _mapping(architecture, ("backbone", "neck"), "architecture")
    backbone = _mapping(arch["backbone"], ("stage1", "stage2"), "backbone")
    first = _mapping(backbone["stage1"], ("kernel", "downsample"), "stage1")
    second = _mapping(backbone["stage2"], ("kernel",), "stage2")
    for name, stage in (("stage1", first), ("stage2", second)):
        if type(stage["kernel"]) is not int or stage["kernel"] not in (3, 5):
            raise ValueError(f"{name}.kernel requires integer 3 or 5")
    if type(first["downsample"]) is not str or first["downsample"] not in (
            "strided_dw", "blur3_after_dense_dw"):
        raise ValueError("stage1.downsample requires strided_dw or blur3_after_dense_dw")
    neck = _mapping(arch["neck"], ("td_ops", "pan"), "neck")
    td = _mapping(neck["td_ops"], ("P2", "P3", "P4"), "td_ops")
    for level in ("P2", "P3", "P4"):
        choices = ("identity", "mbconv_k3", "mbconv_k5") if level == "P2" else (
            "mbconv_k3", "mbconv_k5")
        if type(td[level]) is not str or td[level] not in choices:
            raise ValueError(f"{level} operation requires one of {choices}")
    if type(neck["pan"]) is not bool:
        raise ValueError("neck.pan requires bool")
    return dict(backbone=dict(stage1=dict(kernel=first["kernel"], downsample=first["downsample"]),
                             stage2=dict(kernel=second["kernel"])),
                neck=dict(td_ops={level: td[level] for level in ("P2", "P3", "P4")},
                          pan=neck["pan"]))


def enumerate_architectures():
    for k1, down, k2, p2, p3, p4, pan in itertools.product(
            (3, 5), ("strided_dw", "blur3_after_dense_dw"), (3, 5),
            ("identity", "mbconv_k3", "mbconv_k5"),
            ("mbconv_k3", "mbconv_k5"), ("mbconv_k3", "mbconv_k5"),
            (False, True)):
        yield dict(backbone=dict(stage1=dict(kernel=k1, downsample=down),
                                stage2=dict(kernel=k2)),
                   neck=dict(td_ops=dict(P2=p2, P3=p3, P4=p4), pan=pan))


def architecture_hash(architecture):
    return hashlib.sha256(json.dumps(validate_architecture(architecture),
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def architecture_id(architecture):
    arch = validate_architecture(architecture)
    first, second = arch["backbone"]["stage1"], arch["backbone"]["stage2"]
    down = "blur3" if first["downsample"] == "blur3_after_dense_dw" else "strided"
    td = "_".join(f"{level}_{op}" for level, op in arch["neck"]["td_ops"].items())
    return f"S1_{down}_k{first['kernel']}_S2_k{second['kernel']}_{td}_PAN{int(arch['neck']['pan'])}"


class BlurDownMBConv(MBConvBlock):
    """Dense DW -> BN -> ReLU6 -> fixed triangle3 stride2 -> linear projection.

    The lowpass uses zero padding. It is not trainable, introduces no extra
    normalization/activation and intentionally attenuates DC at boundaries.
    """
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio=3):
        super().__init__(in_channels, out_channels, kernel_size, stride=1,
                         expand_ratio=expand_ratio)
        self.use_residual = False  # Overall downsampling stride is two.
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        self.register_buffer("blur_kernel", kernel.reshape(1, 1, 3, 3).repeat(
            self.dw_conv.out_channels, 1, 1, 1))

    def downsample(self, value):
        return F.conv2d(value, self.blur_kernel, stride=2, padding=1,
                        groups=self.dw_conv.out_channels)

    def forward(self, value):
        if self.use_expand:
            value = self.expand_relu(self.expand_bn(self.expand_conv(value)))
        value = self.dw_relu(self.dw_bn(self.dw_conv(value)))
        value = self.downsample(value)
        return self.project_bn(self.project_conv(value))


class DenseDecimateMBConv(BlurDownMBConv):
    """Attribution control, outside the 192-space: dense DW then exact decimation."""
    def downsample(self, value):
        return value[:, :, ::2, ::2]


def _paired_block(reference, kernel, seed, name, *, blur=False):
    """Copy every matching tensor; new DW shapes receive a location-specific seed.

    Blur and ordinary stride2 blocks with the same kernel share all trainable
    tensors. Semantic seed excludes the downsampling choice intentionally.
    """
    if not blur and reference.dw_conv.kernel_size == (kernel, kernel):
        return reference
    encoded = f"{INITIALIZATION_POLICY}:{seed}:{name}:k{kernel}".encode()
    branch_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(branch_seed)
        cls = BlurDownMBConv if blur else MBConvBlock
        kwargs = dict(expand_ratio=3) if blur else dict(
            stride=reference.dw_conv.stride[0], expand_ratio=3)
        result = cls(reference.expand_conv.in_channels,
                     reference.project_conv.out_channels, kernel, **kwargs)
    previous, current = reference.state_dict(), result.state_dict()
    result.load_state_dict({name: previous[name] if name in previous and
                            previous[name].shape == tensor.shape else tensor
                            for name, tensor in current.items()}, strict=True)
    return result


class StaticJoint192Detector(nn.Module):
    """One selected candidate; no supernet branches or dormant PAN modules."""
    def __init__(self, architecture, seed=42):
        super().__init__()
        self.candidate_architecture = validate_architecture(architecture)
        if type(seed) is not int or seed not in (42, 43, 44):
            raise ValueError("registered integer seed must be 42, 43 or 44")
        self.seed, self.num_classes = seed, 2
        self.image_size, self.input_channels = 640, 1
        self.search_space_version = SEARCH_SPACE_VERSION
        arch = self.candidate_architecture
        pilot = build_pilot("E1" if arch["neck"]["pan"] else "E0", seed)
        self.backbone, self.neck, self.head = pilot.backbone, pilot.neck, pilot.head
        self.levels, self.feature_indices = pilot.levels, pilot.feature_indices
        self.feature_strides, self.anchor_sizes = pilot.feature_strides, pilot.anchor_sizes
        self.last_active_stage = 7
        self.architecture = deepcopy(pilot.architecture)
        stage_specs = list(self.architecture.stages)
        for index in (1, 2):
            choice = arch["backbone"][f"stage{index}"]
            is_blur = index == 1 and choice["downsample"] == "blur3_after_dense_dw"
            self.backbone.stages[index][0] = _paired_block(
                self.backbone.stages[index][0], choice["kernel"], seed,
                f"backbone.stage{index}", blur=is_blur)
            stage_specs[index] = replace(stage_specs[index], blocks=(replace(
                stage_specs[index].blocks[0], kernel_size=choice["kernel"],
                op="mbconv_blur3" if is_blur else "mbconv"),))
        self.architecture = replace(self.architecture, stages=tuple(stage_specs))
        self.backbone.architecture = self.architecture
        for level, operation in arch["neck"]["td_ops"].items():
            if operation == "identity":
                self.neck.top_down_blocks[level] = nn.Identity()
            else:
                self.neck.top_down_blocks[level] = _paired_block(
                    self.neck.top_down_blocks[level], int(operation[-1]), seed,
                    f"neck.top_down.{level}")
        self._initialization_identity = dict(
            seed=seed, policy=INITIALIZATION_POLICY,
            architecture_sha256=architecture_hash(arch),
            model_sha256=_hash_named_tensors(self.state_dict().items()),
            head_sha256=_hash_named_tensors(self.head.state_dict().items()),
            lateral_sha256=_hash_named_tensors(self.neck.lateral_convs.state_dict().items()),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("joint192 adapter requires [B,1,640,640] input")
        features = self.backbone(images)
        neck = self.neck(features)
        return dict(backbone_features=features, neck_features=neck,
                    predictions=self.head(neck))

    def loss_config(self, baseline_config):
        return replace(DetectionLossConfig.from_config(baseline_config),
                       num_classes=2, image_size=640, strides=self.feature_strides,
                       anchor_sizes=self.anchor_sizes, class_loss_weights=(1., 1.))

    def initialization_identity(self):
        return deepcopy(self._initialization_identity)

    def architecture_contract(self):
        return dict(search_space_version=SEARCH_SPACE_VERSION,
                    architecture=deepcopy(self.candidate_architecture),
                    architecture_id=architecture_id(self.candidate_architecture),
                    architecture_sha256=architecture_hash(self.candidate_architecture),
                    input_chw=[1, 640, 640], num_classes=2,
                    levels=list(self.levels), feature_indices=list(self.feature_indices),
                    feature_strides=list(self.feature_strides),
                    anchors=deepcopy(self.anchor_sizes), neck_channels=48,
                    PAN_kernels=[3, 3, 3] if self.neck.bottom_up_pan else [],
                    reference_backbone_architecture=deepcopy(B1_ARCH),
                    initialization_policy=INITIALIZATION_POLICY,
                    fixed_lowpass_padding="zeros", board_latency_status="NOT_MEASURED")


def build_model(architecture=None, seed=42):
    architecture = REFERENCE_ARCHITECTURES["E0"] if architecture is None else architecture
    with torch.random.fork_rng(devices=[]):
        return StaticJoint192Detector(architecture, seed)


__all__ = ["SEARCH_SPACE_VERSION", "ARCHITECTURE_COUNT", "INITIALIZATION_POLICY",
           "REFERENCE_ARCHITECTURES", "validate_architecture", "enumerate_architectures",
           "architecture_id", "architecture_hash", "BlurDownMBConv", "DenseDecimateMBConv",
           "StaticJoint192Detector", "build_model"]
