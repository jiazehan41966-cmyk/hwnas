from __future__ import annotations

import argparse
import json
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _status_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"status entries must be a list: {path}")
    return entries


def _shape_key(spec: dict[str, Any]) -> tuple[int, int, int, int]:
    resolution = spec.get("input_resolution") or []
    if len(resolution) != 2 or int(resolution[0]) != int(resolution[1]):
        raise ValueError(f"only square input_resolution is supported: {resolution}")
    return (
        int(resolution[0]),
        int(spec["in_channels"]),
        int(spec["out_channels"]),
        int(spec.get("stride", 1)),
    )


def _block_choice_from_entry(entry: dict[str, Any]) -> dict[str, int | str] | None:
    op_type = str(entry.get("op_type") or "")
    spec = entry.get("op_spec") or {}
    op = str(spec.get("op") or "")
    kernel_size = int(spec.get("kernel_size", 1))
    expand_ratio = int(spec.get("expand_ratio", 1))

    if op_type.startswith("mbconv") or op.startswith("mbconv"):
        return {
            "op": "mbconv",
            "kernel_size": kernel_size,
            "expand_ratio": expand_ratio,
        }
    if op == "conv" and kernel_size == 1 and expand_ratio == 1:
        return {
            "op": "conv",
            "kernel_size": 1,
            "expand_ratio": 1,
        }
    return None


def _choice_sort_key(choice: dict[str, int | str]) -> tuple[str, int, int]:
    return (
        str(choice["op"]),
        int(choice["expand_ratio"]),
        int(choice["kernel_size"]),
    )


def build_measured_indices(
    entries: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, int, int, int], list[dict[str, int | str]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    block_choices_by_shape: dict[tuple[int, int, int, int], list[dict[str, int | str]]] = {}
    head_entries: list[dict[str, Any]] = []
    primitive_entries: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("status") != "measured":
            continue
        spec = entry.get("op_spec") or {}
        op_type = str(entry.get("op_type") or "")
        op = str(spec.get("op") or "")

        if op_type in {"fc_layer", "global_avg_pool"} or "head" in str(entry.get("case_name") or ""):
            head_entries.append(entry)

        choice = _block_choice_from_entry(entry)
        if choice is None:
            if op in {"dw_conv_k3", "dw_conv_k5", "conv"} or op_type == "pw_conv":
                primitive_entries.append(entry)
            continue

        key = _shape_key(spec)
        block_choices_by_shape.setdefault(key, [])
        if choice not in block_choices_by_shape[key]:
            block_choices_by_shape[key].append(choice)

    for choices in block_choices_by_shape.values():
        choices.sort(key=_choice_sort_key)

    return block_choices_by_shape, head_entries, primitive_entries


def _stage_shapes(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_cfg = config["dataset"]
    search_cfg = config["search_space"]
    image_size = int(dataset_cfg.get("image_size", 224))
    stem_stride = int(search_cfg.get("stem_stride", 2))
    post_stem_stride = int(search_cfg.get("post_stem_downsample_stride", 1))
    stage_strides = [int(v) for v in search_cfg["stage_strides"]]
    base_channels = [int(v) for v in search_cfg["stage_base_channels"]]
    width_multipliers = [float(v) for v in search_cfg.get("width_multipliers", [1.0])]
    if len(width_multipliers) != 1:
        raise ValueError("expanded config generation expects a single width multiplier")

    current_resolution = ceil(image_size / stem_stride)
    current_resolution = ceil(current_resolution / post_stem_stride)
    current_channels = int(search_cfg["stem_channels"])
    shapes: list[dict[str, Any]] = []

    for stage_index, (stride, base_channel) in enumerate(zip(stage_strides, base_channels)):
        out_channels = max(1, int(round(base_channel * width_multipliers[0])))
        shapes.append(
            {
                "stage_index": stage_index,
                "key": (current_resolution, current_channels, out_channels, stride),
                "input_resolution": current_resolution,
                "in_channels": current_channels,
                "out_channels": out_channels,
                "stride": stride,
            }
        )
        current_resolution = ceil(current_resolution / stride)
        current_channels = out_channels

    return shapes


def expand_config(
    *,
    base_config: dict[str, Any],
    block_choices_by_shape: dict[tuple[int, int, int, int], list[dict[str, int | str]]],
    run_name: str,
    project_name: str,
    episodes: int,
    smoke: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = deepcopy(base_config)
    config["project"]["name"] = project_name
    config["project"]["run_name"] = run_name
    if smoke:
        config["dataset"]["name"] = "dummy"
        config["dataset"]["data_dir"] = None
        config["dataset"]["num_workers"] = 0
        config["search"]["num_candidates"] = int(episodes)
        config["search"]["episodes"] = int(episodes)
        config["search"]["eval_epochs"] = 0
        config["search"]["eval_early_stopping_patience"] = 0
        config["training"]["epochs"] = 0
        config["training"]["batch_size"] = 16

    stage_choices: list[list[dict[str, int | str]]] = []
    stage_report: list[dict[str, Any]] = []
    for shape in _stage_shapes(config):
        key = shape["key"]
        choices = deepcopy(block_choices_by_shape.get(key, []))
        if not choices:
            raise ValueError(f"no measured block choices for stage shape {key}")
        stage_choices.append(choices)
        stage_report.append(
            {
                **shape,
                "choices": choices,
                "choice_count": len(choices),
            }
        )

    config["search_space"]["stage_block_choices"] = stage_choices
    config["search_space"]["kernel_choices"] = sorted(
        {int(choice["kernel_size"]) for choices in stage_choices for choice in choices}
    )
    config["search_space"]["expand_choices"] = sorted(
        {int(choice["expand_ratio"]) for choices in stage_choices for choice in choices}
    )
    config["search_space"]["op_choices"] = list(
        dict.fromkeys(str(choice["op"]) for choices in stage_choices for choice in choices)
    )
    config["search_space"]["head_conv_channels"] = None
    return config, stage_report


def _write_report(
    path: Path,
    *,
    outputs: list[dict[str, Any]],
    head_entries: list[dict[str, Any]],
    primitive_entries: list[dict[str, Any]],
) -> None:
    lines = ["# Measured LUT expanded config report\n\n"]
    for output in outputs:
        lines.append(f"## {output['name']}\n\n")
        lines.append(f"- Config: `{output['config_path']}`\n")
        lines.append(f"- Run name: `{output['run_name']}`\n")
        lines.append("\n| Stage | Shape (res,in,out,stride) | Choices |\n")
        lines.append("|---:|---|---|\n")
        for stage in output["stage_report"]:
            choices = ", ".join(
                f"{choice['op']}_e{choice['expand_ratio']}_k{choice['kernel_size']}"
                for choice in stage["choices"]
            )
            shape = (
                stage["input_resolution"],
                stage["in_channels"],
                stage["out_channels"],
                stage["stride"],
            )
            lines.append(f"| {stage['stage_index']} | {shape} | {choices} |\n")
        lines.append("\n")

    lines.append("## Measured entries not used as stage block choices\n\n")
    lines.append(
        "- Primitive measured entries are kept out of `stage_block_choices` unless they exactly match a NAS block query.\n"
    )
    lines.append(f"- Primitive measured entries observed: {len(primitive_entries)}\n")
    lines.append(f"- Head/GAP/FC measured entries observed: {len(head_entries)}\n")
    if head_entries:
        lines.append("\nHead/GAP/FC entries:\n")
        for entry in head_entries:
            spec = entry.get("op_spec") or {}
            lines.append(
                f"- `{entry.get('case_name')}`: {spec.get('op')} "
                f"{spec.get('input_resolution')} {spec.get('in_channels')}->{spec.get('out_channels')}\n"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="hls_lut_builder/results/formal_lut_status_v1.json")
    parser.add_argument("--base-4stage", default="configs/search/formal_lut_compact_4stage_smoke_av7k325.yaml")
    parser.add_argument("--base-5stage", default="configs/search/formal_lut_compact_5stage_smoke_av7k325.yaml")
    parser.add_argument(
        "--base-4stage-nksid",
        default="configs/search/formal_lut_compact_4stage_nksid_main_av7k325.yaml",
    )
    parser.add_argument(
        "--base-5stage-nksid",
        default="configs/search/formal_lut_compact_5stage_nksid_aux_av7k325.yaml",
    )
    parser.add_argument("--out-dir", default="configs/search")
    parser.add_argument("--report", default="results/formal_lut_capacity_diagnosis/measured_lut_expansion_report.md")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    entries = _status_entries(Path(args.status))
    block_choices_by_shape, head_entries, primitive_entries = build_measured_indices(entries)

    specs = [
        {
            "name": "4stage_expanded",
            "base": Path(args.base_4stage),
            "filename": "formal_lut_expanded_4stage_smoke_av7k325.yaml",
            "run_name": "formal_lut_expanded_4stage_smoke_20ep_100ms",
            "project_name": "formal-lut-expanded-4stage-smoke-av7k325",
            "smoke": True,
        },
        {
            "name": "5stage_expanded",
            "base": Path(args.base_5stage),
            "filename": "formal_lut_expanded_5stage_smoke_av7k325.yaml",
            "run_name": "formal_lut_expanded_5stage_smoke_20ep_150ms",
            "project_name": "formal-lut-expanded-5stage-smoke-av7k325",
            "smoke": True,
        },
        {
            "name": "4stage_expanded_nksid",
            "base": Path(args.base_4stage_nksid),
            "filename": "formal_lut_expanded_4stage_nksid_main_av7k325.yaml",
            "run_name": "formal_lut_expanded_4stage_nksid_main_20ep_100ms",
            "project_name": "formal-lut-expanded-4stage-nksid-main-av7k325",
            "smoke": False,
        },
        {
            "name": "5stage_expanded_nksid",
            "base": Path(args.base_5stage_nksid),
            "filename": "formal_lut_expanded_5stage_nksid_aux_av7k325.yaml",
            "run_name": "formal_lut_expanded_5stage_nksid_aux_20ep_150ms",
            "project_name": "formal-lut-expanded-5stage-nksid-aux-av7k325",
            "smoke": False,
        },
    ]

    outputs: list[dict[str, Any]] = []
    for spec in specs:
        base_config = _load_yaml(spec["base"])
        config, stage_report = expand_config(
            base_config=base_config,
            block_choices_by_shape=block_choices_by_shape,
            run_name=spec["run_name"],
            project_name=spec["project_name"],
            episodes=args.episodes,
            smoke=bool(spec["smoke"]),
        )
        output_path = Path(args.out_dir) / spec["filename"]
        _write_yaml(output_path, config)
        outputs.append(
            {
                "name": spec["name"],
                "config_path": str(output_path),
                "run_name": spec["run_name"],
                "stage_report": stage_report,
            }
        )

    _write_report(
        Path(args.report),
        outputs=outputs,
        head_entries=head_entries,
        primitive_entries=primitive_entries,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
