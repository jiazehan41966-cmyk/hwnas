"""Static elastic_neck72_v1 builder with exact historical tensor names and seeds.

Only selected kernels and the active PAN path are instantiated. The fixed B1
backbone is independent of all search, training, quantization and hardware code.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import math
import torch
from torch import nn
from space import Genotype, SPACE_VERSION
from .blocks import ConvBlock, StemBlock, SkipBlock, MBConvBlock
from .neck import (ElasticFourScaleNeck, LEVELS, _cpu_seed, _semantic_seed,
                   _initialize_semantic_modules)

SEARCH_SPACE_VERSION = SPACE_VERSION
TD_LEVELS = LEVELS[:-1]
PAN_LEVELS = LEVELS[1:]
INITIALIZATION_POLICY = "ELASTIC_NECK72_PILOT_K3_PAIRED_K5_DEPTHWISE_V1"
B1_ARCH = {"output_combination": "P2_P3_P4_P5", "kernels":
           {"stage1": 3, "stage2": 5, "stage4": 5,
            "stage5": 5, "stage6": 5, "stage7": 5}}
ANCHORS = (((9.25, 13.5), (12., 16.25), (15., 16.25)),
           ((9.25, 13.5), (12., 16.25), (15., 16.25)),
           ((12.62, 20.25), (19.75, 15.25), (15., 22.75)),
           ((16.5, 22.38), (21.25, 19.5), (21.25, 27.5)))

@dataclass(frozen=True)
class _Block:
    op: str
    kernel_size: int
    expand_ratio: int
    stride: int
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class _Stage:
    channels: int
    depth: int
    stride: int
    blocks: tuple[_Block, ...]
    def to_dict(self):
        return dict(channels=self.channels, depth=self.depth, stride=self.stride,
                    blocks=[b.to_dict() for b in self.blocks])

@dataclass(frozen=True)
class FixedB1Architecture:
    """Read-only B1 metadata, retained for graph/checkpoint compatibility."""
    stages: tuple[_Stage, ...]
    input_channels: int = 1
    stem_channels: int = 32
    stem_stride: int = 2
    post_stem_downsample_stride: int = 1
    head_conv_channels: None = None
    head_channels: None = None
    num_classes: None = None
    def to_dict(self):
        value = asdict(self)
        value["stages"] = [stage.to_dict() for stage in self.stages]
        return value


def _b1_architecture():
    # Source: frozen search_space_sonar_v0_1.yaml, fixed B1 only.
    rows = (("conv",16,1,1,1), ("mbconv",24,3,3,2),
            ("mbconv",32,5,3,2), ("skip",32,1,1,1),
            ("mbconv",64,5,3,2), ("mbconv",96,5,3,1),
            ("mbconv",160,5,3,2), ("mbconv",320,5,3,1))
    return FixedB1Architecture(tuple(_Stage(c,1,s,(_Block(op,k,e,s),))
                                    for op,c,k,e,s in rows))


class FixedB1Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = _b1_architecture()
        self.feature_indices = (1,3,5,7)
        self.stem = StemBlock(1,32,stride=2)
        self.post_stem_downsample = None
        self.stages = nn.ModuleList()
        current = 32
        for stage in self.architecture.stages:
            block = stage.blocks[0]
            if block.op == "conv":
                module = ConvBlock(current,stage.channels,block.kernel_size,block.stride)
            elif block.op == "skip":
                module = SkipBlock(current,stage.channels)
            else:
                module = MBConvBlock(current,stage.channels,block.kernel_size,
                                     block.stride,block.expand_ratio)
            self.stages.append(nn.ModuleList([module]))
            current = stage.channels

    def forward(self, images):
        value = self.stem(images)
        features = []
        for index, stage in enumerate(self.stages):
            for block in stage: value = block(value)
            if index in self.feature_indices: features.append(value)
        return tuple(features)


class PredictionHead(nn.Module):
    def __init__(self, num_classes, seed):
        super().__init__()
        self.in_channels = 48
        self.num_classes = num_classes
        self.anchors_per_location = 3
        self.outputs_per_anchor = num_classes + 5
        self.prediction_channels = 3 * self.outputs_per_anchor
        self.extra_tower_blocks = 0
        self.towers = nn.ModuleList()
        for level in LEVELS:
            if num_classes == 2:
                module_seed = _semantic_seed(seed, f"prediction.{level}")
            else:
                encoded = f"{INITIALIZATION_POLICY}:{seed}:prediction.{level}:nc{num_classes}".encode("utf-8")
                module_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)
            _cpu_seed(module_seed)
            prediction = nn.Conv2d(48,self.prediction_channels,1,bias=True)
            # The pilot constructs, then resets each nc=2 prediction under its
            # semantic seed; Conv2d.__init__ already invokes this exact reset.
            with torch.no_grad():
                bias = prediction.bias.view(3,self.outputs_per_anchor)
                bias[:,4].fill_(math.log(.01/.99)); bias[:,5:].zero_()
            self.towers.append(nn.Sequential(prediction))

    @property
    def prediction_convs(self): return tuple(tower[-1] for tower in self.towers)

    def forward(self, features):
        if len(features) != 4: raise ValueError("Prediction head requires four scales")
        return tuple(tower(feature) for tower,feature in zip(self.towers,features))


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

def _check_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
        raise ValueError("new ELASTIC adapter requires [B,1,640,640] input")

class StaticElasticNeckDetector(nn.Module):
    def __init__(self, genotype: Genotype, seed=42):
        super().__init__()
        if not isinstance(genotype, Genotype): raise TypeError("Genotype required")
        if type(seed) is not int or seed not in (42,43,44):
            raise ValueError("Registered integer seed 42, 43 or 44 required")
        self.genotype = genotype
        self.arch_hash = genotype.arch_hash
        self.candidate_architecture = genotype.legacy_architecture()
        self.num_classes, self.seed = genotype.num_classes, seed
        self.image_size, self.input_channels = 640, 1
        self.search_space_version = SEARCH_SPACE_VERSION
        self.levels, self.feature_indices = LEVELS, (1,3,5,7)
        self.feature_strides, self.anchor_sizes = (4,8,16,32), ANCHORS
        with torch.random.fork_rng(devices=[]):
            _cpu_seed(seed)
            self.backbone = FixedB1Backbone()
            self.architecture = self.backbone.architecture
            _cpu_seed(_semantic_seed(seed,"neck_container"))
            self.neck = ElasticFourScaleNeck((24,32,96,320),48,genotype.pan)
            _initialize_semantic_modules(self.neck,seed,"neck")
            self.head = PredictionHead(genotype.num_classes, seed)
            for level,kernel in zip(TD_LEVELS,genotype.td_kernels):
                if kernel == 5:
                    self.neck.top_down_blocks[level] = _kernel_five(
                        self.neck.top_down_blocks[level],seed,f"top_down.{level}")
            for level,kernel in zip(PAN_LEVELS,genotype.pan_kernels):
                if kernel == 5:
                    self.neck.bottom_up_blocks[level] = _kernel_five(
                        self.neck.bottom_up_blocks[level],seed,f"bottom_up.{level}")

    def forward(self, images):
        _check_images(images)
        features = self.backbone(images)
        neck = self.neck(features)
        return dict(backbone_features=features,neck_features=neck,predictions=self.head(neck))

    def architecture_contract(self):
        return dict(search_space_version=SEARCH_SPACE_VERSION,
            genotype=self.genotype.to_dict(),arch_hash=self.arch_hash,
            architecture=deepcopy(self.candidate_architecture),
            architecture_id=self.genotype.architecture_id,
            architecture_sha256=self.genotype.legacy_arch_hash,
            fixed_backbone_architecture=deepcopy(B1_ARCH), input_chw=[1,640,640],
            num_classes=self.num_classes, levels=list(self.levels),
            feature_indices=list(self.feature_indices),feature_strides=list(self.feature_strides),
            anchors=deepcopy(self.anchor_sizes),neck_channels=48,expansion=3,block_depth=1,
            prediction_channels=3*(self.num_classes+5),prediction_kernel=1,
            prediction_activation="linear",prediction_bias=True,
            top_down_source="post_MBConv_fused_feature",
            bottom_up_source="previous_bottom_up_output",
            initialization_policy=INITIALIZATION_POLICY,
            relation_to_papers="new_space_engineering_adapter_not_author_reproduction",
            board_latency_status="NOT_MEASURED")


def build_model(genotype: Genotype, seed=42):
    return StaticElasticNeckDetector(genotype,seed)
