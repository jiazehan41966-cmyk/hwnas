#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.runtime import (  # noqa: E402
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec, StageSpec  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "formal_lut_strict40_nksid_short_av7k325.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "strict40_deterministic_4candidate_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate the exact-covered strict40 search-space candidates and "
            "run hardware-cost/LUT-integrity checks without RL sampling."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="search config path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    return parser.parse_args()


def block_label(block: Any) -> str:
    return f"{block.op}_k{block.kernel_size}_e{block.expand_ratio}_s{block.stride}"


def _single_choice(values: tuple[Any, ...], *, name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{name} must have exactly one choice for deterministic smoke: {values}")
    return values[0]


def enumerate_architectures(config_path: Path) -> tuple[list[ArchitectureSpec], Any, Any]:
    config = load_config(str(config_path))
    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    search_space = build_search_space(
        config,
        image_size=config["dataset"]["image_size"],
        input_channels=config["dataset"]["input_channels"],
        num_classes=config["dataset"]["num_classes"],
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )
    search_space, _ = apply_operator_policies_to_search_space(
        search_space,
        estimator.operator_policies,
    )
    search_space = search_space.pre_prune(estimator)

    stage_channel_choices = search_space.config.stage_channel_choices
    stage_depth_choices = search_space.config.stage_depth_choices
    if stage_channel_choices is None or stage_depth_choices is None:
        raise ValueError("deterministic strict40 smoke requires stage-level channel/depth choices")

    stage_channels = [
        int(_single_choice(tuple(choices), name=f"stage {idx} channels"))
        for idx, choices in enumerate(stage_channel_choices)
    ]
    stage_depths = [
        int(_single_choice(tuple(choices), name=f"stage {idx} depths"))
        for idx, choices in enumerate(stage_depth_choices)
    ]
    if any(depth != 1 for depth in stage_depths):
        raise ValueError(f"deterministic strict40 smoke expects one block per stage: {stage_depths}")

    block_choice_groups = []
    current_channels = search_space.config.stem_channels
    for stage_idx, channels in enumerate(stage_channels):
        stride = search_space.config.stage_strides[stage_idx]
        choices = search_space.concrete_block_choices_for_stage(
            stage_index=stage_idx,
            in_channels=current_channels,
            out_channels=channels,
            stride=stride,
        )
        if choices is None:
            raise ValueError("deterministic strict40 smoke requires explicit stage_block_choices")
        block_choice_groups.append(choices)
        current_channels = channels

    architectures: list[ArchitectureSpec] = []
    for block_combo in product(*block_choice_groups):
        stages = tuple(
            StageSpec(
                channels=stage_channels[stage_idx],
                depth=1,
                stride=search_space.config.stage_strides[stage_idx],
                blocks=(block_combo[stage_idx],),
            )
            for stage_idx in range(search_space.config.stage_count)
        )
        architectures.append(
            ArchitectureSpec(
                input_channels=search_space.config.input_channels,
                stem_channels=search_space.config.stem_channels,
                stem_stride=search_space.config.stem_stride,
                post_stem_downsample_stride=search_space.config.post_stem_downsample_stride,
                head_conv_channels=search_space.config.head_conv_channels,
                head_channels=search_space.config.head_channels,
                num_classes=search_space.config.num_classes,
                stages=stages,
            )
        )

    return architectures, estimator, search_space


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    architectures, estimator, search_space = enumerate_architectures(config_path)

    rows: list[dict[str, Any]] = []
    for idx, architecture in enumerate(architectures):
        estimator.reset_lut_stats()
        cost = estimator.estimate(architecture, search_space)
        lut_stats = estimator.get_lut_stats()
        stage_blocks = [block_label(stage.blocks[0]) for stage in architecture.stages]
        rows.append(
            {
                "candidate_id": f"strict40_enum_{idx}",
                "stage0": stage_blocks[0],
                "stage1": stage_blocks[1],
                "stage2": stage_blocks[2],
                "stage3": stage_blocks[3],
                "feasible": len(cost.violations) == 0,
                "violations": list(cost.violations),
                "latency_ms": cost.latency_ms,
                "dsp": cost.resource_dsp,
                "bram": cost.resource_bram,
                "lut": cost.resource_lut,
                "power_w": cost.power_w,
                "energy_mj": cost.energy_mj,
                "memory_bandwidth_gbps": cost.memory_bandwidth_gbps,
                "offchip_mem_mb": cost.offchip_mem_mb,
                "lut_hits": lut_stats["hits"],
                "lut_misses": lut_stats["misses"],
                "true_misses": lut_stats["true_misses"],
                "deferred_hits": lut_stats["deferred_hits"],
                "hit_rate": lut_stats["hit_rate"],
                "fallback_rate": lut_stats["fallback_rate"],
            }
        )

    summary = {
        "config": str(config_path),
        "candidate_count": len(rows),
        "feasible": sum(1 for row in rows if row["feasible"]),
        "infeasible": sum(1 for row in rows if not row["feasible"]),
        "strict_lut_ok": all(
            row["true_misses"] == 0
            and row["deferred_hits"] == 0
            and row["lut_misses"] == 0
            for row in rows
        ),
        "unique_stage1_choices": sorted({row["stage1"] for row in rows}),
        "candidates": rows,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Enumerated strict40 candidates: "
        f"{summary['candidate_count']} total, "
        f"{summary['feasible']} feasible, "
        f"strict_lut_ok={summary['strict_lut_ok']} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
