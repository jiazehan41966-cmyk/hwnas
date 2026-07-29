"""Frozen four-stage operator experiment architecture helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch
import torch.nn as nn

from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec


FROZEN_CHANNELS = (16, 24, 32, 32)
FROZEN_STRIDES = (1, 2, 2, 1)
FROZEN_DEPTHS = (1, 1, 1, 1)
STAGE2_CHOICES = (
    ("k3_e3", 3, 3),
    ("k3_e6", 3, 6),
    ("k5_e3", 5, 3),
    ("k5_e6", 5, 6),
)
BASE_STAGE4_CHOICES = ("mbconv_k3_e3", "skip")


@dataclass(frozen=True)
class FourStageFactorRow:
    arch_id: str
    kernel: int
    expansion: int
    stage4: str
    architecture: ArchitectureSpec

    def factors(self) -> dict[str, object]:
        return {
            "kernel": f"K{self.kernel}",
            "expansion": f"e{self.expansion}",
            "stage4": self.stage4,
        }


def build_fourstage_architecture(
    *,
    stage2_kernel: int,
    stage2_expansion: int,
    stage4_op: str,
) -> ArchitectureSpec:
    stage4_key = str(stage4_op).strip().lower()
    if stage4_key == "mbconv_k3_e3":
        stage4_block = BlockSpec(
            op="mbconv", kernel_size=3, expand_ratio=3, stride=1
        )
    elif stage4_key == "skip":
        stage4_block = BlockSpec(
            op="skip", kernel_size=1, expand_ratio=1, stride=1
        )
    else:
        raise ValueError(f"unsupported base stage4_op: {stage4_op}")
    architecture = ArchitectureSpec(
        input_channels=1,
        stem_channels=32,
        stem_stride=2,
        post_stem_downsample_stride=1,
        head_conv_channels=None,
        head_channels=None,
        num_classes=8,
        stages=(
            StageSpec(
                channels=16,
                depth=1,
                stride=1,
                blocks=(
                    BlockSpec(
                        op="conv", kernel_size=1, expand_ratio=1, stride=1
                    ),
                ),
            ),
            StageSpec(
                channels=24,
                depth=1,
                stride=2,
                blocks=(
                    BlockSpec(
                        op="mbconv",
                        kernel_size=int(stage2_kernel),
                        expand_ratio=int(stage2_expansion),
                        stride=2,
                    ),
                ),
            ),
            StageSpec(
                channels=32,
                depth=1,
                stride=2,
                blocks=(
                    BlockSpec(
                        op="mbconv", kernel_size=3, expand_ratio=3, stride=2
                    ),
                ),
            ),
            StageSpec(
                channels=32,
                depth=1,
                stride=1,
                blocks=(stage4_block,),
            ),
        ),
    )
    validate_frozen_fourstage(architecture)
    return architecture


def enumerate_base8() -> tuple[FourStageFactorRow, ...]:
    rows: list[FourStageFactorRow] = []
    for stage2_name, kernel, expansion in STAGE2_CHOICES:
        for stage4 in BASE_STAGE4_CHOICES:
            arch_id = f"fourstage_s2_{stage2_name}_s4_{stage4}"
            rows.append(
                FourStageFactorRow(
                    arch_id=arch_id,
                    kernel=kernel,
                    expansion=expansion,
                    stage4="MBConv" if stage4 == "mbconv_k3_e3" else "Skip",
                    architecture=build_fourstage_architecture(
                        stage2_kernel=kernel,
                        stage2_expansion=expansion,
                        stage4_op=stage4,
                    ),
                )
            )
    if len(rows) != 8:
        raise AssertionError(f"base factorial must contain 8 rows, got {len(rows)}")
    return tuple(rows)


def validate_frozen_fourstage(architecture: ArchitectureSpec) -> None:
    errors: list[str] = []
    if architecture.input_channels != 1:
        errors.append("input_channels must be 1")
    if architecture.stem_channels != 32 or architecture.stem_stride != 2:
        errors.append("stem must be Conv3x3 1->32 stride2")
    if architecture.post_stem_downsample_stride != 1:
        errors.append("post-stem downsample must be disabled")
    if architecture.head_conv_channels is not None:
        errors.append("conv head must be disabled")
    if architecture.num_classes != 8:
        errors.append("head must be FC8")
    if len(architecture.stages) != 4:
        errors.append("architecture must contain exactly four stages")
    channels = tuple(stage.channels for stage in architecture.stages)
    strides = tuple(stage.stride for stage in architecture.stages)
    depths = tuple(stage.depth for stage in architecture.stages)
    if channels != FROZEN_CHANNELS:
        errors.append(f"stage channels changed: {channels}")
    if strides != FROZEN_STRIDES:
        errors.append(f"stage strides changed: {strides}")
    if depths != FROZEN_DEPTHS:
        errors.append(f"stage depths changed: {depths}")
    if architecture.stages:
        if architecture.stages[0].blocks[0] != BlockSpec(
            op="conv", kernel_size=1, expand_ratio=1, stride=1
        ):
            errors.append("Stage1 must be Conv1x1 32->16 stride1")
        if architecture.stages[2].blocks[0] != BlockSpec(
            op="mbconv", kernel_size=3, expand_ratio=3, stride=2
        ):
            errors.append("Stage3 must be MBConv-k3-e3 24->32 stride2")
    if errors:
        raise ValueError("; ".join(errors))


def architecture_sha256(architecture: ArchitectureSpec) -> str:
    encoded = json.dumps(
        architecture.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_payload(row: FourStageFactorRow) -> dict[str, object]:
    return {
        "candidate": {
            "arch_id": row.arch_id,
            "backbone": "nksid_fourstage_operator_v2",
            "factorial_design": row.factors(),
            "encoding": row.architecture.to_dict(),
            "architecture_sha256": architecture_sha256(row.architecture),
        },
        "claim_boundary": (
            "Architecture encoding only. Training, INT8, HLS, route, board, "
            "and power evidence are separate."
        ),
    }


def parameter_and_mac_count(
    architecture: ArchitectureSpec,
    *,
    input_shape: Iterable[int] = (1, 1, 224, 224),
) -> dict[str, object]:
    model = build_model(architecture=architecture, num_classes=8).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    macs = 0
    layer_rows: list[dict[str, object]] = []
    hooks = []

    def hook(module: nn.Module, inputs, output) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            output_tensor = output
            batch, out_channels, out_height, out_width = output_tensor.shape
            kernel_height, kernel_width = module.kernel_size
            operations = (
                batch
                * out_channels
                * out_height
                * out_width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
            kind = "conv2d"
        elif isinstance(module, nn.Linear):
            batch = output.shape[0]
            operations = batch * module.in_features * module.out_features
            kind = "linear"
        else:
            return
        macs += int(operations)
        layer_rows.append(
            {
                "kind": kind,
                "module": module.__class__.__name__,
                "macs": int(operations),
                "output_shape": list(output.shape),
            }
        )

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        output = model(torch.zeros(tuple(int(value) for value in input_shape)))
    for handle in hooks:
        handle.remove()
    return {
        "parameter_count": int(params),
        "trainable_parameter_count": int(trainable_params),
        "macs": int(macs),
        "input_shape": list(input_shape),
        "output_shape": list(output.shape),
        "layers": layer_rows,
    }
