#!/usr/bin/env python3
"""Generate an analytical LUT aligned with the active HW-NAS search space."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.hardware import (
    BackboneCostEstimator,
    FPGACostEstimator,
    LutEntry,
    LutQueryEngine,
    LutTable,
)
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.models.builder import build_block
from hwnas_fpga.runtime import (
    build_constraints,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, ResolvedBlockSpec, SearchSpace


DEFAULT_JSON_PATH = "artifacts/kintex7_analytical_lut.json"
DEFAULT_PKL_PATH = "artifacts/kintex7_analytical_lut.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an analytical block LUT compatible with the HW-NAS search runtime."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--board", type=str, default=None, help="Optional board override")
    parser.add_argument("--clock-mhz", type=int, default=None, help="Optional clock override")
    parser.add_argument("--image-size", type=int, default=None, help="Optional dataset image size override")
    parser.add_argument("--input-channels", type=int, default=None, help="Optional input channel override")
    parser.add_argument("--num-classes", type=int, default=None, help="Optional num_classes override")
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_JSON_PATH,
        help="Human-readable analytical LUT JSON path",
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=DEFAULT_PKL_PATH,
        help="Binary LUT pickle path for the existing runtime",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to write generation/validation summary JSON",
    )
    parser.add_argument(
        "--measure-cpu-latency",
        action="store_true",
        help="Also measure CPU latency while building the per-block estimates",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=6,
        help="Number of random architectures used for LUT smoke validation",
    )
    return parser.parse_args()


def _ceil_div(value: int, divisor: int) -> int:
    return int(ceil(value / divisor))


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    payload = dict(config)
    dataset_cfg = dict(payload.get("dataset", {}))
    hardware_cfg = dict(payload.get("hardware", {}))

    if args.image_size is not None:
        dataset_cfg["image_size"] = int(args.image_size)
    if args.input_channels is not None:
        dataset_cfg["input_channels"] = int(args.input_channels)
    if args.num_classes is not None:
        dataset_cfg["num_classes"] = int(args.num_classes)
    if dataset_cfg:
        payload["dataset"] = dataset_cfg

    if args.board is not None:
        hardware_cfg["board"] = str(args.board)
    if args.clock_mhz is not None:
        hardware_cfg["clock_mhz"] = int(args.clock_mhz)
    if hardware_cfg:
        payload["hardware"] = hardware_cfg

    return payload


def _iter_legal_blocks(
    *,
    search_space: SearchSpace,
    stage_index: int,
    block_index: int,
    input_resolution: int,
    output_resolution: int,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> Iterable[ResolvedBlockSpec]:
    config = search_space.config
    valid_ops = search_space.available_ops(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    )
    for op in valid_ops:
        kernel_choices = (1,) if op == "skip" else config.kernel_choices
        expand_choices = config.expand_choices if op in {"mbconv", "fused_mbconv"} else (1,)
        for kernel_size in kernel_choices:
            for expand_ratio in expand_choices:
                yield ResolvedBlockSpec(
                    stage_index=stage_index,
                    block_index=block_index,
                    op=op,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    stride=stride,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    input_resolution=input_resolution,
                    output_resolution=output_resolution,
                )


def enumerate_search_space_blocks(
    search_space: SearchSpace,
    analytical_estimator: FPGACostEstimator,
) -> dict:
    """Enumerate unique legal block placements seen by the active search space."""
    config = search_space.config
    max_depth = max(config.depth_choices)
    blocks_by_spec = {}

    stem_output_resolution = _ceil_div(config.image_size, config.stem_stride)
    stem_block = ResolvedBlockSpec(
        stage_index=-1,
        block_index=-1,
        op="conv",
        kernel_size=3,
        expand_ratio=1,
        stride=config.stem_stride,
        in_channels=config.input_channels,
        out_channels=config.stem_channels,
        input_resolution=config.image_size,
        output_resolution=stem_output_resolution,
    )
    blocks_by_spec[analytical_estimator._block_to_op_spec(stem_block)] = stem_block

    stage_input_resolution = stem_output_resolution

    for stage_index, stage_stride in enumerate(config.stage_strides):
        stage_output_resolution = _ceil_div(stage_input_resolution, stage_stride)
        first_block_inputs = (config.stem_channels,) if stage_index == 0 else config.channel_choices

        for in_channels in first_block_inputs:
            for out_channels in config.channel_choices:
                for block in _iter_legal_blocks(
                    search_space=search_space,
                    stage_index=stage_index,
                    block_index=0,
                    input_resolution=stage_input_resolution,
                    output_resolution=stage_output_resolution,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stage_stride,
                ):
                    blocks_by_spec[analytical_estimator._block_to_op_spec(block)] = block

        for block_index in range(1, max_depth):
            for channels in config.channel_choices:
                for block in _iter_legal_blocks(
                    search_space=search_space,
                    stage_index=stage_index,
                    block_index=block_index,
                    input_resolution=stage_output_resolution,
                    output_resolution=stage_output_resolution,
                    in_channels=channels,
                    out_channels=channels,
                    stride=1,
                ):
                    blocks_by_spec[analytical_estimator._block_to_op_spec(block)] = block

        stage_input_resolution = stage_output_resolution

    return blocks_by_spec


def generate_lut_table(
    *,
    search_space: SearchSpace,
    hardware_spec: HardwareSpec,
    constraints: SearchConstraints,
    quantization_bits: int,
    measure_cpu_latency: bool,
) -> tuple[LutTable, dict]:
    backbone_estimator = BackboneCostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=quantization_bits,
    )
    analytical_estimator = FPGACostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=quantization_bits,
    )

    candidate_blocks = enumerate_search_space_blocks(search_space, analytical_estimator)
    lut_table = LutTable()
    failures = []
    op_counter: Counter[str] = Counter()

    print(
        f"Generating analytical LUT for {len(candidate_blocks)} unique block/operator specs "
        f"on {hardware_spec.name}..."
    )

    for index, (op_spec, block) in enumerate(candidate_blocks.items(), start=1):
        if index % 50 == 0 or index == len(candidate_blocks):
            print(f"  [{index}/{len(candidate_blocks)}] {block.op} cin={block.in_channels} cout={block.out_channels}")

        block_spec = BlockSpec(
            op=block.op,
            kernel_size=block.kernel_size,
            expand_ratio=block.expand_ratio,
            stride=block.stride,
        )

        try:
            module = build_block(block_spec, block.in_channels, block.out_channels)
            block_cost = backbone_estimator.estimate(
                module,
                input_shape=(1, block.in_channels, block.input_resolution, block.input_resolution),
                measure_cpu_latency=measure_cpu_latency,
            )
        except Exception as exc:  # pragma: no cover - only exercised on invalid operator construction.
            failures.append(
                {
                    "op_spec": op_spec.to_dict(),
                    "error": str(exc),
                }
            )
            continue

        total_dsp = sum(layer.allocated_dsp for layer in block_cost.per_layer)
        total_bram = sum(layer.bram_blocks for layer in block_cost.per_layer)
        total_lut = sum(layer.lut for layer in block_cost.per_layer)

        lut_table.add(
            LutEntry(
                op_spec=op_spec,
                latency_ms=block_cost.latency_ms,
                cycles=block_cost.latency_cycles,
                dsp=total_dsp,
                bram=total_bram,
                lut=total_lut,
                power_w=block_cost.power_w,
                energy_mj=block_cost.energy_mj,
            )
        )
        op_counter[block.op] += 1

    summary = {
        "board": hardware_spec.__dict__,
        "search_space": {
            "image_size": search_space.config.image_size,
            "input_channels": search_space.config.input_channels,
            "stem_channels": search_space.config.stem_channels,
            "stem_stride": search_space.config.stem_stride,
            "stage_strides": list(search_space.config.stage_strides),
            "channel_choices": list(search_space.config.channel_choices),
            "depth_choices": list(search_space.config.depth_choices),
            "kernel_choices": list(search_space.config.kernel_choices),
            "expand_choices": list(search_space.config.expand_choices),
            "op_choices": list(search_space.config.op_choices),
        },
        "entries_requested": len(candidate_blocks),
        "entries_built": len(lut_table),
        "table_stats": lut_table.get_stats(),
        "entries_per_op": dict(sorted(op_counter.items())),
        "failures": failures,
    }
    return lut_table, summary


def validate_lut(
    *,
    lut_table: LutTable,
    search_space: SearchSpace,
    hardware_spec: HardwareSpec,
    constraints: SearchConstraints,
    quantization_bits: int,
    num_random_samples: int,
) -> dict:
    estimator = FPGACostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=quantization_bits,
        lut_query_engine=LutQueryEngine(lut_table),
    )

    architectures: list[tuple[str, ArchitectureSpec]] = [
        ("baseline", search_space.baseline_architecture())
    ]
    for seed in range(max(0, num_random_samples)):
        architectures.append(
            (
                f"sample_{seed}",
                search_space.sample(seed=seed, apply_pruning=False),
            )
        )

    results = []
    for name, architecture in architectures:
        before_hits = estimator.lut_hits
        before_misses = estimator.lut_misses
        estimate = estimator.estimate(architecture, search_space)
        results.append(
            {
                "arch_id": name,
                "lut_hits": estimator.lut_hits - before_hits,
                "lut_misses": estimator.lut_misses - before_misses,
                "latency_ms": estimate.latency_ms,
                "dsp": estimate.resource_dsp,
                "bram": estimate.resource_bram,
                "lut": estimate.resource_lut,
                "violations": list(estimate.violations),
            }
        )

    return {
        "architectures": results,
        "lut_stats": estimator.get_lut_stats(),
    }


def main() -> None:
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    constraints = build_constraints(config)
    hardware_cfg = config.get("hardware", {})
    quantization_bits = int(hardware_cfg.get("quantization_bits", 8))

    dataset_cfg = config.get("dataset", {})
    image_size = int(dataset_cfg.get("image_size", 224))
    input_channels = int(dataset_cfg.get("input_channels", 1))
    num_classes = dataset_cfg.get("num_classes")

    hardware_spec = build_hardware_spec(config)
    search_space = build_search_space(
        config,
        image_size=image_size,
        input_channels=input_channels,
        num_classes=num_classes,
        constraints=constraints,
    )

    lut_table, summary = generate_lut_table(
        search_space=search_space,
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=quantization_bits,
        measure_cpu_latency=args.measure_cpu_latency,
    )
    validation = validate_lut(
        lut_table=lut_table,
        search_space=search_space,
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=quantization_bits,
        num_random_samples=args.validation_samples,
    )

    output_json = Path(args.output_json).expanduser().resolve()
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    lut_table.save_json(
        str(output_json),
        metadata={
            "kind": "analytical_lut",
            "board": hardware_spec.__dict__,
            "search_space": summary["search_space"],
            "entries_requested": summary["entries_requested"],
            "entries_built": summary["entries_built"],
        },
    )
    lut_table.save(str(output_pkl))

    summary_payload = {
        **summary,
        "output_json": str(output_json),
        "output_pkl": str(output_pkl),
        "validation": validation,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Analytical LUT JSON: {output_json}")
    print(f"Analytical LUT PKL:  {output_pkl}")
    print(
        "Validation LUT hit-rate: "
        f"{validation['lut_stats']['hit_rate']:.3f} "
        f"({validation['lut_stats']['hits']} hits / {validation['lut_stats']['total']} queries)"
    )
    if summary["failures"]:
        print(f"Warnings: {len(summary['failures'])} blocks failed during analytical profiling")


if __name__ == "__main__":
    main()
