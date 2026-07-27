#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.models.mobilenetv2 import InvertedResidual


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.models.backbones import build_backbone


@dataclass(frozen=True)
class StageBlueprint:
    stage_index: int
    stage_name: str
    feature_h: int
    feature_w: int
    in_channels: int
    out_channels: int
    stride: int

    @property
    def out_h(self) -> int:
        return self.feature_h if self.stride == 1 else self.feature_h // self.stride

    @property
    def out_w(self) -> int:
        return self.feature_w if self.stride == 1 else self.feature_w // self.stride


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate_kernels.yaml from MobileNetV2 blueprint")
    parser.add_argument(
        "--manifest",
        default=str(BUILDER_ROOT / "configs" / "operator_manifest.yaml"),
        help="Path to operator manifest YAML",
    )
    parser.add_argument(
        "--output-config",
        default=str(BUILDER_ROOT / "configs" / "candidate_kernels.yaml"),
        help="Path to generated candidate_kernels.yaml",
    )
    parser.add_argument(
        "--output-shapes",
        default=str(BUILDER_ROOT / "configs" / "mobile_anchor_shapes.yaml"),
        help="Path to generated extracted shape YAML",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML: {yaml_path}")
    return payload


def extract_mobile_anchor_blueprint(
    *,
    backbone: str,
    input_channels: int,
    num_classes: int,
    image_size: int,
) -> dict[str, Any]:
    model, _ = build_backbone(
        name=backbone,
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=False,
    )
    model.eval()

    if not hasattr(model, "features") or not hasattr(model, "classifier"):
        raise ValueError(f"Backbone {backbone} does not expose MobileNetV2-style features/classifier")

    stem_conv = model.features[0][0]
    head_conv = model.features[18][0]
    fc = model.classifier[1]

    module_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    hooks = []

    def register(name: str, module: nn.Module) -> None:
        def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            module_shapes[name] = (tuple(inputs[0].shape), tuple(output.shape))

        hooks.append(module.register_forward_hook(_hook))

    register("stem", stem_conv)
    register("head_pw", head_conv)
    register("fc", fc)

    blocks: list[tuple[str, InvertedResidual]] = []
    for index, module in enumerate(model.features):
        if isinstance(module, InvertedResidual):
            name = f"features.{index}"
            blocks.append((name, module))
            register(name, module)

    with torch.no_grad():
        model(torch.zeros(1, input_channels, image_size, image_size))

    for hook in hooks:
        hook.remove()

    stem_input, stem_output = module_shapes["stem"]
    stage_blueprints = extract_stage_blueprints(blocks, module_shapes)
    skip_blueprints = extract_skip_blueprints(blocks, module_shapes)
    primitive_shapes = derive_primitive_shapes(stage_blueprints, skip_blueprints)

    head_input, head_output = module_shapes["head_pw"]
    fc_input, fc_output = module_shapes["fc"]
    last_stage = stage_blueprints[-1]
    head_shape = OrderedDict(
        feature_h=int(head_input[2]),
        feature_w=int(head_input[3]),
        in_channels=int(head_input[1]),
        out_channels=int(head_output[1]),
        stride=1,
    )
    gap_shape = OrderedDict(
        feature_h=int(last_stage.out_h),
        feature_w=int(last_stage.out_w),
        in_channels=int(last_stage.out_channels),
        out_channels=int(last_stage.out_channels),
        stride=1,
        padding=0,
        out_h=1,
        out_w=1,
    )
    fc_shape = OrderedDict(
        feature_h=1,
        feature_w=1,
        in_channels=int(fc_input[1]),
        out_channels=int(fc_output[1]),
        stride=1,
        padding=0,
        out_h=1,
        out_w=1,
    )

    shapes = OrderedDict()
    shapes["stem_224_1_32_s2"] = OrderedDict(
        feature_h=int(stem_input[2]),
        feature_w=int(stem_input[3]),
        in_channels=int(stem_input[1]),
        out_channels=int(stem_output[1]),
        stride=2,
    )
    for name, payload in primitive_shapes.items():
        shapes[name] = payload
    shapes["pw_head_7_320_1280"] = head_shape
    shapes["gap_7_320"] = gap_shape
    shapes["fc_1_1280_8"] = fc_shape

    return {
        "source": OrderedDict(
            backbone=backbone,
            input_channels=input_channels,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "stem": OrderedDict(
            input_shape=list(stem_input),
            output_shape=list(stem_output),
        ),
        "stages": [
            OrderedDict(
                stage_index=item.stage_index,
                stage_name=item.stage_name,
                feature_h=item.feature_h,
                feature_w=item.feature_w,
                in_channels=item.in_channels,
                out_channels=item.out_channels,
                stride=item.stride,
            )
            for item in stage_blueprints
        ],
        "representative_shapes": shapes,
        "skip_shapes": [
            OrderedDict(
                feature_h=item["feature_h"],
                feature_w=item["feature_w"],
                channels=item["channels"],
            )
            for item in skip_blueprints
        ],
    }


def extract_stage_blueprints(
    blocks: list[tuple[str, InvertedResidual]],
    module_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[StageBlueprint]:
    seen: set[tuple[int, int, int]] = set()
    blueprints: list[StageBlueprint] = []
    stage_index = 0
    for name, block in blocks:
        input_shape, output_shape = module_shapes[name]
        out_key = (int(output_shape[2]), int(output_shape[3]), int(output_shape[1]))
        if out_key in seen:
            continue
        seen.add(out_key)
        stage_index += 1
        blueprints.append(
            StageBlueprint(
                stage_index=stage_index,
                stage_name=f"ir{int(output_shape[1])}",
                feature_h=int(input_shape[2]),
                feature_w=int(input_shape[3]),
                in_channels=int(input_shape[1]),
                out_channels=int(output_shape[1]),
                stride=int(block.stride),
            )
        )
    return blueprints


def extract_skip_blueprints(
    blocks: list[tuple[str, InvertedResidual]],
    module_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[dict[str, int]]:
    seen: set[tuple[int, int, int]] = set()
    skip_shapes: list[dict[str, int]] = []
    for name, block in blocks:
        input_shape, output_shape = module_shapes[name]
        if int(block.stride) != 1 or int(input_shape[1]) != int(output_shape[1]):
            continue
        key = (int(output_shape[2]), int(output_shape[3]), int(output_shape[1]))
        if key in seen:
            continue
        seen.add(key)
        skip_shapes.append(
            {
                "feature_h": int(output_shape[2]),
                "feature_w": int(output_shape[3]),
                "channels": int(output_shape[1]),
            }
        )
    return skip_shapes


def derive_primitive_shapes(
    stage_blueprints: list[StageBlueprint],
    skip_blueprints: list[dict[str, int]],
) -> OrderedDict[str, OrderedDict[str, int]]:
    shapes: OrderedDict[str, OrderedDict[str, int]] = OrderedDict()

    stage1 = stage_blueprints[0]
    shapes[f"dw_{stage1.stage_name}_e1_{stage1.feature_h}_c{stage1.in_channels}_s{stage1.stride}"] = OrderedDict(
        feature_h=stage1.feature_h,
        feature_w=stage1.feature_w,
        in_channels=stage1.in_channels,
        stride=stage1.stride,
        expand_ratio=1,
    )
    shapes[f"pw_{stage1.stage_name}_proj_{stage1.feature_h}_{stage1.in_channels}_{stage1.out_channels}"] = OrderedDict(
        feature_h=stage1.feature_h,
        feature_w=stage1.feature_w,
        in_channels=stage1.in_channels,
        out_channels=stage1.out_channels,
        stride=1,
        apply_relu=0,
        expand_ratio=1,
    )

    for stage in stage_blueprints[1:]:
        stage_label = stage.stage_name
        shapes[f"mbconv_stage{stage.stage_index}_{stage.feature_h}_{stage.in_channels}_{stage.out_channels}_s{stage.stride}"] = OrderedDict(
            feature_h=stage.feature_h,
            feature_w=stage.feature_w,
            in_channels=stage.in_channels,
            out_channels=stage.out_channels,
            stride=stage.stride,
        )

        for ratio in (3, 6):
            hidden = stage.in_channels * ratio
            shapes[f"dw_{stage_label}_e{ratio}_{stage.feature_h}_c{hidden}_s{stage.stride}"] = OrderedDict(
                feature_h=stage.feature_h,
                feature_w=stage.feature_w,
                in_channels=hidden,
                stride=stage.stride,
                expand_ratio=ratio,
            )
            shapes[f"pw_{stage_label}_expand_e{ratio}_{stage.feature_h}_{stage.in_channels}_{hidden}"] = OrderedDict(
                feature_h=stage.feature_h,
                feature_w=stage.feature_w,
                in_channels=stage.in_channels,
                out_channels=hidden,
                stride=1,
                apply_relu=1,
                expand_ratio=ratio,
            )
            shapes[f"pw_{stage_label}_proj_e{ratio}_{stage.out_h}_{hidden}_{stage.out_channels}"] = OrderedDict(
                feature_h=stage.out_h,
                feature_w=stage.out_w,
                in_channels=hidden,
                out_channels=stage.out_channels,
                stride=1,
                apply_relu=0,
                expand_ratio=ratio,
            )

    for skip in skip_blueprints:
        shapes[f"skip_{skip['feature_h']}_{skip['channels']}"] = OrderedDict(
            feature_h=skip["feature_h"],
            feature_w=skip["feature_w"],
            in_channels=skip["channels"],
            out_channels=skip["channels"],
            stride=1,
        )

    return shapes


def build_candidate_config(manifest: dict[str, Any], blueprint: dict[str, Any]) -> OrderedDict[str, Any]:
    cfg = OrderedDict()
    for section in ("toolchain", "workspace", "measurement_contract", "defaults", "clock_profiles", "implementation_profiles"):
        cfg[section] = manifest[section]
    cfg["representative_shapes"] = blueprint["representative_shapes"]
    cfg["operators"] = build_operator_entries(manifest, blueprint["representative_shapes"])
    return cfg


def build_operator_entries(shapes: dict[str, Any], representative_shapes: dict[str, Any]) -> OrderedDict[str, Any]:
    operator_entries: OrderedDict[str, Any] = OrderedDict()

    mbconv_stage_refs = [name for name in representative_shapes if name.startswith("mbconv_stage")]
    dw_shape_refs = [name for name in representative_shapes if name.startswith("dw_")]
    pw_shape_refs = [name for name in representative_shapes if name.startswith("pw_")]
    skip_shape_refs = [name for name in representative_shapes if name.startswith("skip_")]
    ext_shape_refs = [
        ref for ref in mbconv_stage_refs
        if any(token in ref for token in ("stage2_", "stage4_", "stage6_"))
    ]

    operator_entries["stem_conv_k3_s2"] = OrderedDict(
        enabled=True,
        template="../templates/stem_conv_k3_s2.cpp.tmpl",
        top_function="stem_conv_k3_s2_kernel",
        op_type="stem_conv_k3_s2",
        shape_refs=["stem_224_1_32_s2"],
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=3, padding=1, apply_relu=1),
        op_spec_defaults=OrderedDict(op="stem_conv_k3_s2", groups=1, expand_ratio=1),
    )
    operator_entries["dw_conv_k3"] = OrderedDict(
        enabled=True,
        template="../templates/dw_conv.cpp.tmpl",
        top_function="dw_conv_k3_kernel",
        op_type="dw_conv_k3",
        shape_refs=dw_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=3, padding=1, apply_relu=1),
        op_spec_defaults=OrderedDict(op="dw_conv_k3"),
    )
    operator_entries["dw_conv_k5"] = OrderedDict(
        enabled=True,
        template="../templates/dw_conv.cpp.tmpl",
        top_function="dw_conv_k5_kernel",
        op_type="dw_conv_k5",
        shape_refs=dw_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=5, padding=2, apply_relu=1),
        op_spec_defaults=OrderedDict(op="dw_conv_k5"),
    )
    operator_entries["pw_conv"] = OrderedDict(
        enabled=True,
        template="../templates/pw_conv.cpp.tmpl",
        top_function="pw_conv_kernel",
        op_type="pw_conv",
        shape_refs=pw_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=1, padding=0, apply_relu=1),
        op_spec_defaults=OrderedDict(op="pw_conv", groups=1),
    )

    for expand_ratio, kernel_size in ((3, 3), (3, 5), (6, 3), (6, 5)):
        name = f"mbconv_e{expand_ratio}_k{kernel_size}"
        operator_entries[name] = OrderedDict(
            enabled=True,
            template="../templates/mbconv.cpp.tmpl",
            top_function=f"{name}_kernel",
            op_type=name,
            shape_refs=mbconv_stage_refs,
            implementation_refs=["baseline_pi1_po1_u1"],
            clock_profile_refs=["main_5ns"],
            parameters=OrderedDict(kernel_size=kernel_size, padding=kernel_size // 2, expand_ratio=expand_ratio),
            op_spec_defaults=OrderedDict(op=name, groups=1, expand_ratio=expand_ratio),
        )

    operator_entries["skip"] = OrderedDict(
        enabled=True,
        template="../templates/skip.cpp.tmpl",
        top_function="skip_kernel",
        op_type="skip",
        shape_refs=skip_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=1, padding=0),
        op_spec_defaults=OrderedDict(op="skip", groups=1, expand_ratio=1),
    )
    operator_entries["global_avg_pool"] = OrderedDict(
        enabled=True,
        template="../templates/global_avg_pool.cpp.tmpl",
        top_function="global_avg_pool_kernel",
        op_type="global_avg_pool",
        shape_refs=["gap_7_320"],
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=7, padding=0),
        op_spec_defaults=OrderedDict(op="global_avg_pool", groups=1, expand_ratio=1),
    )
    operator_entries["fc_layer"] = OrderedDict(
        enabled=True,
        template="../templates/fc_layer.cpp.tmpl",
        top_function="fc_layer_kernel",
        op_type="fc_layer",
        shape_refs=["fc_1_1280_8"],
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=1, padding=0),
        op_spec_defaults=OrderedDict(op="fc_layer", groups=1, expand_ratio=1),
    )

    operator_entries["fused_mbconv_e3_k3"] = OrderedDict(
        enabled=False,
        template="../templates/fused_mbconv.cpp.tmpl",
        top_function="fmb_e3_k3_kernel",
        op_type="fused_mbconv_e3_k3",
        shape_refs=ext_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=3, padding=1, expand_ratio=3),
        op_spec_defaults=OrderedDict(op="fused_mbconv_e3_k3", groups=1, expand_ratio=3),
    )
    operator_entries["fused_mbconv_e6_k3"] = OrderedDict(
        enabled=False,
        template="../templates/fused_mbconv.cpp.tmpl",
        top_function="fmb_e6_k3_kernel",
        op_type="fused_mbconv_e6_k3",
        shape_refs=ext_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=3, padding=1, expand_ratio=6),
        op_spec_defaults=OrderedDict(op="fused_mbconv_e6_k3", groups=1, expand_ratio=6),
    )
    for name in ("mixconv", "denoise", "edge"):
        template = f"../templates/{name}.cpp.tmpl"
        params = OrderedDict(kernel_size=5 if name == "mixconv" else 3, padding=2 if name == "mixconv" else 1)
        operator_entries[name] = OrderedDict(
            enabled=False,
            template=template,
            top_function=f"{name}_kernel",
            op_type=name,
            shape_refs=ext_shape_refs,
            implementation_refs=["baseline_pi1_po1_u1"],
            clock_profile_refs=["main_5ns"],
            parameters=params,
            op_spec_defaults=OrderedDict(op=name, groups=1, expand_ratio=1),
        )

    operator_entries["mixconv_v2"] = OrderedDict(
        enabled=False,
        template="../templates/mixconv_v2.cpp.tmpl",
        top_function="mixconv_v2_kernel",
        op_type="mixconv_v2",
        shape_refs=ext_shape_refs,
        implementation_refs=["baseline_pi1_po1_u1"],
        clock_profile_refs=["main_5ns"],
        parameters=OrderedDict(kernel_size=5, padding=2),
        op_spec_defaults=OrderedDict(op="mixconv_v2", groups=1, expand_ratio=1),
    )

    return operator_entries


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    yaml_path = Path(path).expanduser().resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(_plain_yaml(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _plain_yaml(value: Any) -> Any:
    if isinstance(value, OrderedDict):
        return {key: _plain_yaml(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_yaml(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest)
    shape_export = manifest["shape_export"]
    blueprint = extract_mobile_anchor_blueprint(
        backbone=str(shape_export["backbone"]),
        input_channels=int(shape_export["input_channels"]),
        num_classes=int(shape_export["num_classes"]),
        image_size=int(shape_export["image_size"]),
    )
    candidate_config = build_candidate_config(manifest, blueprint)
    write_yaml(args.output_shapes, blueprint)
    write_yaml(args.output_config, candidate_config)
    print(f"Wrote shape blueprint: {Path(args.output_shapes).resolve()}")
    print(f"Wrote candidate config: {Path(args.output_config).resolve()}")
    print(f"Core shape count: {len(candidate_config['representative_shapes'])}")
    print(f"Operator count: {len(candidate_config['operators'])}")


if __name__ == "__main__":
    main()
