from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any
import warnings

import torch
import torch.nn as nn

try:
    from torchvision import models as tv_models
except ImportError:  # pragma: no cover - handled at runtime
    tv_models = None


@dataclass(frozen=True)
class BackboneCandidate:
    """Configuration for one backbone baseline candidate."""

    arch_id: str
    name: str
    display_name: str
    pretrained: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackboneCandidate":
        return cls(
            arch_id=str(payload["arch_id"]),
            name=str(payload["name"]),
            display_name=str(payload.get("display_name", payload["arch_id"])),
            pretrained=bool(payload.get("pretrained", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "name": self.name,
            "display_name": self.display_name,
            "pretrained": self.pretrained,
        }


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """A lightweight CNN baseline close to the ~100K parameter target."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        num_classes: int = 8,
        hidden_dim: int = 80,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(input_channels, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNormAct(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNormAct(32, 64),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


FBNET_LIKE_STAGE_SPECS = (
    {"channels": 16, "depth": 1, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
    {"channels": 24, "depth": 4, "stride": 2, "expand": 3, "kernel": 3, "groups": 2},
    {"channels": 32, "depth": 4, "stride": 2, "expand": 3, "kernel": 5, "groups": 2},
    {"channels": 64, "depth": 4, "stride": 2, "expand": 4, "kernel": 5, "groups": 2},
    {"channels": 112, "depth": 4, "stride": 1, "expand": 4, "kernel": 5, "groups": 2},
    {"channels": 184, "depth": 4, "stride": 2, "expand": 6, "kernel": 5, "groups": 4},
)

MOBILENET_V2_LIKE_STAGE_SPECS = (
    {"channels": 24, "depth": 2, "stride": 1, "expand": 1, "kernel": 3},
    {"channels": 32, "depth": 3, "stride": 2, "expand": 6, "kernel": 3},
    {"channels": 64, "depth": 4, "stride": 2, "expand": 6, "kernel": 5},
    {"channels": 96, "depth": 3, "stride": 2, "expand": 6, "kernel": 5},
)

FBNET_V1_STAGE_SPECS = (
    {"channels": 24, "depth": 2, "stride": 1, "expand": 3, "kernel": 3},
    {"channels": 32, "depth": 4, "stride": 2, "expand": 3, "kernel": 5},
    {"channels": 64, "depth": 4, "stride": 2, "expand": 4, "kernel": 5},
    {"channels": 112, "depth": 4, "stride": 2, "expand": 4, "kernel": 5},
)

REFERENCE_FBNET_A_STAGE_BLOCK_SPECS = (
    (
        {"op": "skip", "channels": 16, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
    ),
    (
        {"op": "ir", "channels": 24, "stride": 2, "expand": 3, "kernel": 3, "groups": 1},
        {"op": "ir", "channels": 24, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
        {"op": "skip", "channels": 24, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
        {"op": "skip", "channels": 24, "stride": 1, "expand": 1, "kernel": 3, "groups": 1},
    ),
    (
        {"op": "ir", "channels": 32, "stride": 2, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 32, "stride": 1, "expand": 3, "kernel": 3, "groups": 1},
        {"op": "ir", "channels": 32, "stride": 1, "expand": 1, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 32, "stride": 1, "expand": 3, "kernel": 3, "groups": 1},
    ),
    (
        {"op": "ir", "channels": 64, "stride": 2, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 64, "stride": 1, "expand": 3, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 64, "stride": 1, "expand": 1, "kernel": 5, "groups": 2},
        {"op": "ir", "channels": 64, "stride": 1, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 112, "stride": 1, "expand": 6, "kernel": 3, "groups": 1},
        {"op": "ir", "channels": 112, "stride": 1, "expand": 1, "kernel": 5, "groups": 2},
        {"op": "ir", "channels": 112, "stride": 1, "expand": 3, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 112, "stride": 1, "expand": 1, "kernel": 3, "groups": 2},
    ),
    (
        {"op": "ir", "channels": 184, "stride": 2, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 184, "stride": 1, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 184, "stride": 1, "expand": 3, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 184, "stride": 1, "expand": 6, "kernel": 5, "groups": 1},
        {"op": "ir", "channels": 352, "stride": 1, "expand": 6, "kernel": 5, "groups": 1},
    ),
)

MACRO_TEMPLATES: dict[str, dict[str, Any]] = {
    "mobilenet_v2_like": {
        "display_name": "MobileNetV2-like",
        "stem_channels": 16,
        "stem_stride": 2,
        "head_channels": 1280,
        "stages": MOBILENET_V2_LIKE_STAGE_SPECS,
    },
    "fbnet_like": {
        "display_name": "FBNet-like",
        "stem_channels": 16,
        "stem_stride": 2,
        "head_channels": 1280,
        "stages": FBNET_V1_STAGE_SPECS,
    },
}


def normalize_macro_template_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mobilenetv2_like": "mobilenet_v2_like",
        "mobilenet_like": "mobilenet_v2_like",
        "mobile_anchor": "mobilenet_v2_like",
        "mobilenet_v2_like": "mobilenet_v2_like",
        "fbnet": "fbnet_like",
        "fbnet_like": "fbnet_like",
    }
    return aliases.get(normalized, normalized)


def get_macro_template(name: str) -> dict[str, Any]:
    normalized = normalize_macro_template_name(name)
    if normalized not in MACRO_TEMPLATES:
        raise ValueError(f"Unsupported macro template: {name}")
    template = MACRO_TEMPLATES[normalized]
    return {
        "name": normalized,
        "display_name": template["display_name"],
        "stem_channels": int(template["stem_channels"]),
        "stem_stride": int(template["stem_stride"]),
        "head_channels": int(template["head_channels"]),
        "stages": tuple(dict(stage) for stage in template["stages"]),
    }


def list_macro_templates() -> dict[str, dict[str, Any]]:
    return {name: get_macro_template(name) for name in MACRO_TEMPLATES}


def _resolve_group_count(desired_groups: int, *channel_dims: int) -> int:
    if desired_groups <= 1:
        return 1

    shared_divisor = 0
    for channel_dim in channel_dims:
        if channel_dim <= 0:
            raise ValueError("channel dimensions must be positive")
        shared_divisor = channel_dim if shared_divisor == 0 else gcd(shared_divisor, channel_dim)
    return max(1, min(int(desired_groups), shared_divisor))


class FBNetBottleneckBlock(nn.Module):
    """Simplified FBNet-style bottleneck with grouped pointwise convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        groups: int,
    ) -> None:
        super().__init__()
        hidden_channels = max(out_channels, in_channels * int(expand_ratio))
        expand_groups = _resolve_group_count(groups, in_channels, hidden_channels)
        project_groups = _resolve_group_count(groups, hidden_channels, out_channels)
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                groups=expand_groups,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                groups=project_groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.use_residual = stride == 1 and in_channels == out_channels
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.activation(out)


class FBNetLike(nn.Module):
    """A stage-based FBNet-style baseline for macro-architecture screening."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        num_classes: int = 8,
        stem_channels: int = 16,
        head_channels: int = 1280,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        current_channels = stem_channels
        for stage_spec in FBNET_LIKE_STAGE_SPECS:
            blocks: list[nn.Module] = []
            stage_channels = int(stage_spec["channels"])
            stage_depth = int(stage_spec["depth"])
            stage_stride = int(stage_spec["stride"])
            stage_expand = int(stage_spec["expand"])
            stage_kernel = int(stage_spec["kernel"])
            stage_groups = int(stage_spec["groups"])
            for block_index in range(stage_depth):
                block_stride = stage_stride if block_index == 0 else 1
                blocks.append(
                    FBNetBottleneckBlock(
                        current_channels,
                        stage_channels,
                        expand_ratio=stage_expand,
                        kernel_size=stage_kernel,
                        stride=block_stride,
                        groups=stage_groups,
                    )
                )
                current_channels = stage_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.Conv2d(current_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(head_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


class ReferenceFBNetA(nn.Module):
    """Reference FBNet-A macro architecture adapted to the local training pipeline."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        num_classes: int = 8,
        stem_channels: int = 16,
        head_channels: int = 1504,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        current_channels = stem_channels
        for stage_specs in REFERENCE_FBNET_A_STAGE_BLOCK_SPECS:
            blocks: list[nn.Module] = []
            for block_spec in stage_specs:
                op_name = str(block_spec["op"])
                stage_channels = int(block_spec["channels"])
                stage_stride = int(block_spec["stride"])
                if op_name == "skip":
                    if stage_stride != 1 or current_channels != stage_channels:
                        raise ValueError("Reference FBNet-A skip block expects matching channels and stride=1")
                    blocks.append(nn.Identity())
                else:
                    blocks.append(
                        FBNetBottleneckBlock(
                            current_channels,
                            stage_channels,
                            expand_ratio=int(block_spec["expand"]),
                            kernel_size=int(block_spec["kernel"]),
                            stride=stage_stride,
                            groups=int(block_spec.get("groups", 1)),
                        )
                    )
                current_channels = stage_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.Conv2d(current_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(head_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def default_backbone_candidates() -> tuple[BackboneCandidate, ...]:
    return (
        BackboneCandidate(
            arch_id="simplecnn",
            name="simplecnn",
            display_name="SimpleCNN",
            pretrained=False,
        ),
        BackboneCandidate(
            arch_id="mobilenet_v2_pretrained",
            name="mobilenet_v2",
            display_name="MobileNetV2 (pretrained)",
            pretrained=True,
        ),
        BackboneCandidate(
            arch_id="mobilenet_v2_scratch",
            name="mobilenet_v2",
            display_name="MobileNetV2 (scratch)",
            pretrained=False,
        ),
        BackboneCandidate(
            arch_id="fbnet_a",
            name="fbnet_a",
            display_name="FBNet-A",
            pretrained=False,
        ),
        BackboneCandidate(
            arch_id="shufflenet_v2",
            name="shufflenet_v2",
            display_name="ShuffleNetV2",
            pretrained=True,
        ),
        BackboneCandidate(
            arch_id="efficientnet_b0",
            name="efficientnet_b0",
            display_name="EfficientNet-B0",
            pretrained=True,
        ),
    )


def normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "simple_cnn": "simplecnn",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "shufflenetv2": "shufflenet_v2",
        "shufflenet_v2": "shufflenet_v2",
        "efficientnetb0": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "fbnet": "fbnet_a",
        "fbnet_a": "fbnet_a",
        "reference_fbnet_a": "fbnet_a",
        "fbnet_a_reference": "fbnet_a",
        "fbnet_like": "fbnet_like",
    }
    return aliases.get(normalized, normalized)


def _instantiate_torchvision_model(
    normalized_name: str,
    *,
    pretrained: bool,
    strict_pretrained: bool,
) -> tuple[nn.Module, bool]:
    if tv_models is None:
        raise ImportError(
            "torchvision is required for MobileNetV2, ShuffleNetV2, and EfficientNet-B0"
        )

    builders = {
        "mobilenet_v2": (tv_models.mobilenet_v2, "MobileNet_V2_Weights"),
        "shufflenet_v2": (tv_models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights"),
        "efficientnet_b0": (tv_models.efficientnet_b0, "EfficientNet_B0_Weights"),
    }
    if normalized_name not in builders:
        raise ValueError(f"Unsupported backbone: {normalized_name}")

    builder, weights_attr = builders[normalized_name]
    if pretrained:
        try:
            weights_enum = getattr(tv_models, weights_attr, None)
            if weights_enum is not None:
                return builder(weights=weights_enum.DEFAULT), True
            return builder(pretrained=True), True
        except Exception as exc:  # pragma: no cover - network/cache dependent
            if strict_pretrained:
                raise RuntimeError(
                    f"Failed to load pretrained weights for {normalized_name}"
                ) from exc
            warnings.warn(
                f"Pretrained weights for {normalized_name} unavailable ({exc}); "
                "falling back to random initialization.",
                RuntimeWarning,
            )

    try:
        return builder(weights=None), False
    except TypeError:
        return builder(pretrained=False), False


def _find_first_conv(model: nn.Module) -> tuple[str, nn.Conv2d]:
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return module_name, module
    raise ValueError("Unable to find a Conv2d layer for input-channel adaptation")


def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _adapt_conv_weight(weight: torch.Tensor, input_channels: int) -> torch.Tensor:
    current_channels = weight.shape[1]
    if current_channels == input_channels:
        return weight.clone()
    if input_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if current_channels == 1:
        return weight.repeat(1, input_channels, 1, 1) / float(input_channels)
    averaged = weight.mean(dim=1, keepdim=True)
    return averaged.repeat(1, input_channels, 1, 1)


def adapt_model_input_channels(model: nn.Module, input_channels: int) -> nn.Module:
    if input_channels <= 0:
        raise ValueError("input_channels must be positive")

    first_conv_name, first_conv = _find_first_conv(model)
    if first_conv.in_channels == input_channels:
        return model
    if first_conv.groups != 1:
        raise ValueError(
            "Automatic input adaptation currently supports only standard first convolutions"
        )

    new_conv = nn.Conv2d(
        input_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(_adapt_conv_weight(first_conv.weight.data, input_channels))
        if first_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias.data)
    _set_module(model, first_conv_name, new_conv)
    return model


def _replace_classifier(
    model: nn.Module,
    normalized_name: str,
    *,
    num_classes: int,
    dropout: float,
) -> None:
    if normalized_name in {"mobilenet_v2", "efficientnet_b0"}:
        classifier = getattr(model, "classifier", None)
        if not isinstance(classifier, nn.Sequential):
            raise ValueError(f"{normalized_name} classifier is not a Sequential module")
        last_layer = classifier[-1]
        if not isinstance(last_layer, nn.Linear):
            raise ValueError(f"{normalized_name} classifier does not end with Linear")
        dropout_layer = classifier[0] if len(classifier) > 1 and isinstance(classifier[0], nn.Dropout) else nn.Dropout(p=dropout)
        model.classifier = nn.Sequential(dropout_layer, nn.Linear(last_layer.in_features, num_classes))
        return

    if normalized_name == "shufflenet_v2":
        if not isinstance(model.fc, nn.Linear):
            raise ValueError("ShuffleNetV2 head is not a Linear layer")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    raise ValueError(f"Unsupported classifier replacement target: {normalized_name}")


def build_backbone(
    *,
    name: str,
    num_classes: int,
    input_channels: int = 1,
    pretrained: bool = False,
    strict_pretrained: bool = False,
    dropout: float = 0.2,
) -> tuple[nn.Module, dict[str, Any]]:
    normalized_name = normalize_backbone_name(name)
    metadata = {
        "name": normalized_name,
        "num_classes": num_classes,
        "input_channels": input_channels,
        "pretrained_requested": pretrained,
        "pretrained_loaded": False,
    }

    if normalized_name == "simplecnn":
        model = SimpleCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        return model, metadata

    if normalized_name == "fbnet_like":
        model = FBNetLike(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        return model, metadata

    if normalized_name == "fbnet_a":
        model = ReferenceFBNetA(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        return model, metadata

    model, pretrained_loaded = _instantiate_torchvision_model(
        normalized_name,
        pretrained=pretrained,
        strict_pretrained=strict_pretrained,
    )
    metadata["pretrained_loaded"] = pretrained_loaded
    model = adapt_model_input_channels(model, input_channels)
    _replace_classifier(
        model,
        normalized_name,
        num_classes=num_classes,
        dropout=dropout,
    )
    return model, metadata
