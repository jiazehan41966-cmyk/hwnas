"""Isolated 2x2 detector adapter: native trunks crossed with head/loss packages.

The quality package is the pinned author's single DDetect + TAL soft-label BCE,
CIoU and DFL implementation. This is not a dual/PGI YOLOv9 reproduction, and the
backbone factor includes the native neck, feature levels and channel widths.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import yaml

from hwnas_fpga.training.detection_trainer import (
    DenseAnchorDetectionLoss, DetectionLossConfig, decode_detection_predictions,
)
from tools.detector_elastic_pilot.model import (
    ElasticPilotDetector, _hash_named_tensors, load_spec,
)
from tools.sonar_yolov9_baseline.author_adapter import (
    AUTHOR_LOSS_GAINS, _load_author_modules,
    _validate_assigner_environment, decode_author, gray_to_three_channels,
    targets_to_yolo,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "reference/yolov9"
BACKBONES = ("E0", "YOLOv9_T")
HEADS = ("obj_softmax", "quality_aware")
INITIALIZATION_POLICY = "FACTORIAL_NATIVE_TRUNK_PAIRED_CPU_SEED_V1"


def _seed(seed: int, component: str) -> None:
    digest = hashlib.sha256(f"{INITIALIZATION_POLICY}:{seed}:{component}".encode()).digest()
    torch.random.default_generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


class LegacyVariableWidthHead(nn.Module):
    """The existing linear 3-anchor/7-value predictor on native YOLO widths."""

    def __init__(self, channels: Sequence[int], levels: Sequence[str], seed: int):
        super().__init__()
        self.towers = nn.ModuleList()
        for channel, level in zip(channels, levels):
            _seed(seed, f"legacy_prediction.{level}")
            prediction = nn.Conv2d(channel, 21, 1, bias=True)
            with torch.no_grad():
                bias = prediction.bias.view(3, 7)
                bias[:, 4].fill_(math.log(0.01 / 0.99))
                bias[:, 5:].zero_()
            self.towers.append(nn.Sequential(prediction))

    def forward(self, features):
        if len(features) != len(self.towers):
            raise ValueError("Head/feature level count differs")
        return tuple(tower(feature) for tower, feature in zip(self.towers, features))


class FactorialDetector(nn.Module):
    def __init__(self, backbone: str, head: str, seed: int, source_root: Path):
        super().__init__()
        self.backbone_type, self.head_type, self.seed = backbone, head, seed
        self.image_size, self.input_channels, self.nc = 640, 1, 2
        self.names = ["cylider", "manta"]
        self._author_adapter_main_only = True
        self.hyp = {**AUTHOR_LOSS_GAINS, "cls_pw": 1.0, "fl_gamma": 0.0,
                    "label_smoothing": 0.0}
        self.source_root = source_root
        model_module = None
        if backbone == "YOLOv9_T" or head == "quality_aware":
            _validate_assigner_environment()
            _, model_module, _ = _load_author_modules(source_root)

        if backbone == "E0":
            # Direct original constructor preserves historical E0 initialization
        # exactly. The fixed five-seed E0 protocol uses the same initializer
        # for seeds 45/46 without editing the E0 architecture.
            original = ElasticPilotDetector("E0", seed, load_spec())
            self.backbone, self.neck = original.backbone, original.neck
            self.levels = tuple(original.levels)
            self.feature_strides = tuple(original.feature_strides)
            self.feature_channels = (48, 48, 48, 48)
            self.anchor_sizes = tuple(original.anchor_sizes)
            self.head = original.head
            self._legacy_initialization = original.initialization_identity()
        else:
            config = yaml.safe_load((source_root / "models/detect/yolov9-t.yaml").read_text(encoding="utf-8"))
            config["head"] = config["head"][:13] + [[[15, 18, 21], 1, "DDetect", ["nc"]]]
            _seed(seed, "yolov9_main_model")
            native = model_module.DetectionModel(deepcopy(config), ch=3, nc=2)
            if len(native.model) != 23 or type(native.model[-1]).__name__ != "DDetect":
                raise RuntimeError("Unexpected pinned YOLOv9 main graph")
            self.backbone = nn.ModuleList(list(native.model[:9]))
            self.neck = nn.ModuleList(list(native.model[9:22]))
            self.levels, self.feature_strides = ("P3", "P4", "P5"), (8, 16, 32)
            self.feature_channels = (64, 96, 128)
            # Use the exact original E0/B1 anchor association for retained levels.
            anchor_reference = ElasticPilotDetector("E0", seed, load_spec())
            self.anchor_sizes = tuple(anchor_reference.anchor_sizes[1:])
            self._legacy_initialization = None
            self.head = native.model[-1]
            if head == "obj_softmax":
                self.head = LegacyVariableWidthHead(self.feature_channels, self.levels, seed)

        if head == "quality_aware":
            # Recreate the same semantic head initializer on BOTH trunks. The
            # native DetectionModel constructor's dummy stride pass otherwise
            # advances only the YOLO head's BatchNorm running statistics.
            _seed(seed, "quality_head")
            self.head = model_module.DDetect(nc=2, ch=self.feature_channels)
            self.head.stride = torch.tensor(self.feature_strides, dtype=torch.float32)
            self.head.bias_init()
            # Match author DDetect BatchNorm semantics on both trunks, while
            # retaining the native E0 trunk BatchNorm settings unchanged.
            importlib.import_module("utils.torch_utils").initialize_weights(self.head)
        self._initialization = {
            "policy": INITIALIZATION_POLICY, "seed": seed,
            "backbone_sha256": _hash_named_tensors(self.backbone.state_dict().items()),
            "neck_sha256": _hash_named_tensors(self.neck.state_dict().items()),
            "head_sha256": _hash_named_tensors(self.head.state_dict().items()),
        }

    @property
    def model(self):
        """Minimal native ComputeLoss interface; do not register head twice."""
        return (self.head,)

    def _apply(self, fn):
        super()._apply(fn)
        if self.head_type == "quality_aware":
            # The author's detection geometry tensors are plain attributes.
            for name in ("stride", "anchors", "strides"):
                setattr(self.head, name, fn(getattr(self.head, name)))
        return self

    def extract_features(self, images):
        if self.backbone_type == "E0":
            features = self.backbone(images)
            return features, self.neck(features)
        x = gray_to_three_channels(images)
        saved = []
        for module in (*self.backbone, *self.neck):
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if index == -1 else saved[index] for index in module.f]
            x = module(x)
            saved.append(x)
        return tuple(saved[index] for index in (4, 6, 8)), tuple(saved[index] for index in (15, 18, 21))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("Factorial detector requires [B,1,640,640] input")
        features, neck = self.extract_features(images)
        # Native DDetect mutates the supplied list; preserve neck feature refs.
        predictions = self.head(list(neck)) if self.head_type == "quality_aware" else self.head(neck)
        return {"backbone_features": features, "neck_features": neck, "predictions": predictions}

    def loss_config(self, baseline: Mapping[str, Any]):
        return replace(DetectionLossConfig.from_config(dict(baseline)), num_classes=2,
                       image_size=640, strides=self.feature_strides,
                       anchor_sizes=self.anchor_sizes, class_loss_weights=(1.0, 1.0))

    def initialization_identity(self):
        return deepcopy(self._initialization)

    def metadata(self):
        quality = self.head_type == "quality_aware"
        source_files = ("models/yolo.py", "models/common.py", "models/detect/yolov9-t.yaml",
                        "utils/loss_tal.py", "utils/tal/assigner.py", "utils/tal/anchor_generator.py")
        result = {
            "schema_version": 1, "backbone": self.backbone_type, "head": self.head_type,
            "init_condition": "scratch", "input_chw": [1, 640, 640],
            "input_adapter": "unchanged grayscale" if self.backbone_type == "E0" else "exact grayscale repeat to 3 channels",
            "levels": list(self.levels), "strides": list(self.feature_strides),
            "native_neck_channels": list(self.feature_channels),
            "parameters": {name: sum(p.numel() for p in module.parameters()) for name, module in
                           (("backbone", self.backbone), ("neck", self.neck), ("head", self.head))},
            "initialization": self.initialization_identity(),
            "legacy_e0_initialization": deepcopy(self._legacy_initialization),
            "quality_package": {
                "head": "pinned DDetect", "assignment": {"topk": 10, "alpha": 0.5, "beta": 6.0},
                "classification": "BCEWithLogitsLoss against TAL quality soft targets",
                "box": "CIoU", "distribution": "DFL reg_max=16", "gains": dict(AUTHOR_LOSS_GAINS),
                "reduction": "native sum(box,cls,dfl) * batch_size",
                "no_varifocal_loss": True, "no_auxiliary_branch_or_PGI": True,
            } if quality else None,
            "score": "max sigmoid(class quality logits)" if quality else "sigmoid(obj) * max softmax(class)",
            "factor_scope": "head + loss + assignment + score together; not individual component attribution",
            "structural_confound": "E0 P2-P5 at width48; YOLOv9-T P3-P5 at widths64/96/128; backbone factor includes native neck/levels/widths",
            "head_capacity": "native input channels alter prediction parameters and DDetect classification hidden width",
            "board_latency_status": "NOT_MEASURED",
        }
        if quality:
            result["quality_hidden_channels"] = {"regression": self.head.cv2[0][0].conv.out_channels,
                                                 "classification": self.head.cv3[0][0].conv.out_channels}
        if self.backbone_type == "YOLOv9_T" or quality:
            result["author_source_sha256"] = {relative: hashlib.sha256((self.source_root / relative).read_bytes()).hexdigest()
                                               for relative in source_files}
        return result


def build_model(backbone: str, head: str, seed: int, *, source_root: str | Path | None = None):
    if backbone not in BACKBONES or head not in HEADS:
        raise ValueError(f"Unknown factorial cell: {backbone!r}, {head!r}")
    if type(seed) is not int or seed not in (42, 43, 44, 45, 46):
        raise ValueError("Factorial seed must be an integer from 42 through 46")
    with torch.random.fork_rng(devices=[]):
        return FactorialDetector(backbone, head, seed, Path(source_root or DEFAULT_SOURCE_ROOT).resolve())


def build_criterion(model: FactorialDetector, baseline: Mapping[str, Any]):
    if model.head_type == "obj_softmax":
        return DenseAnchorDetectionLoss(model.loss_config(baseline)).to(next(model.parameters()).device)
    _validate_assigner_environment()
    return importlib.import_module("utils.loss_tal").ComputeLoss(model)


def loss_from_output(model, criterion, output, targets):
    if model.head_type == "obj_softmax":
        return criterion(output, targets)
    _validate_assigner_environment()
    device = next(model.parameters()).device
    if criterion.device != device:
        raise ValueError("Build the quality criterion after moving model to the training device")
    native_targets = targets_to_yolo(targets, image_size=640, device=device)
    loss, items = criterion(output["predictions"], native_targets)
    if not torch.isfinite(loss) or not torch.isfinite(items).all():
        raise ValueError("Nonfinite factorial quality loss")
    return loss, {"loss": float(loss.detach()), **{name: float(items[index]) for index, name in
                                                  enumerate(("box_loss", "class_loss", "dfl_loss"))}}


def batch_loss(model, criterion, images, targets):
    return loss_from_output(model, criterion, model(images), targets)


def decode(model, criterion, output, **kwargs):
    if model.training:
        raise ValueError("Decode requires eval mode without switching BatchNorm state")
    if model.head_type == "obj_softmax":
        return decode_detection_predictions(output, config=criterion.config, **kwargs)
    return decode_author(model, output["predictions"], image_size=640, **kwargs)


def criterion_metadata(model, criterion):
    return asdict(criterion.config) if model.head_type == "obj_softmax" else model.metadata()["quality_package"]
