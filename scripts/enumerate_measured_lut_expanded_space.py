#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively enumerate explicit stage_block_choices and evaluate formal LUT coverage."
    )
    parser.add_argument("--config", required=True, help="Search config with explicit stage_block_choices")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.json and candidates.csv")
    parser.add_argument(
        "--skip-pre-prune",
        action="store_true",
        help="Do not apply the same pre-prune step used by the search pipeline.",
    )
    return parser.parse_args()


def block_label(block: Any) -> str:
    return f"{block.op}_e{block.expand_ratio}_k{block.kernel_size}_s{block.stride}"


def _single_choice(values: tuple[Any, ...], *, name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{name} must have exactly one value for exhaustive block-choice enumeration")
    return values[0]


def _build_runtime(config_path: Path, *, skip_pre_prune: bool) -> tuple[Any, Any, Any, Any]:
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
    if not skip_pre_prune:
        search_space = search_space.pre_prune(estimator)
    return config, constraints, estimator, search_space


def enumerate_architectures(config_path: Path, *, skip_pre_prune: bool) -> tuple[list[ArchitectureSpec], Any, Any]:
    _config, _constraints, estimator, search_space = _build_runtime(
        config_path,
        skip_pre_prune=skip_pre_prune,
    )

    stage_channel_choices = search_space.config.stage_channel_choices
    stage_depth_choices = search_space.config.stage_depth_choices
    if stage_channel_choices is None or stage_depth_choices is None:
        raise ValueError("explicit stage-level channel/depth choices are required")

    stage_channels = [
        int(_single_choice(tuple(choices), name=f"stage {idx} channels"))
        for idx, choices in enumerate(stage_channel_choices)
    ]
    stage_depths = [
        int(_single_choice(tuple(choices), name=f"stage {idx} depths"))
        for idx, choices in enumerate(stage_depth_choices)
    ]
    if any(depth != 1 for depth in stage_depths):
        raise ValueError(f"one block per stage is required for this enumerator: {stage_depths}")

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
            raise ValueError("explicit stage_block_choices are required")
        block_choice_groups.append(tuple(choices))
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


def evaluate_architectures(
    architectures: list[ArchitectureSpec],
    estimator: Any,
    search_space: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, architecture in enumerate(architectures):
        estimator.reset_lut_stats()
        cost = estimator.estimate(architecture, search_space)
        lut_stats = estimator.get_lut_stats()
        stage_blocks = [block_label(stage.blocks[0]) for stage in architecture.stages]
        row = {
            "candidate_id": f"expanded_enum_{idx}",
            "architecture": architecture.to_dict(),
            "stage_blocks": "|".join(stage_blocks),
            "feasible": len(cost.violations) == 0,
            "violations": "|".join(cost.violations),
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
        for stage_idx, block in enumerate(stage_blocks):
            row[f"stage{stage_idx}"] = block
        rows.append(row)
    return rows


def _range(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def _violation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not row["violations"]:
            continue
        for violation in str(row["violations"]).split("|"):
            counts[violation] = counts.get(violation, 0) + 1
    return counts


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    config_path: Path,
    output_dir: Path,
    skip_pre_prune: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    (output_dir / "feasible_candidates.json").write_text(
        json.dumps(feasible_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [key for key in rows[0].keys() if key != "architecture"] if rows else []
    with (output_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fieldnames}
            for row in rows
        )

    totals = {
        "lut_hits": sum(int(row["lut_hits"]) for row in rows),
        "lut_misses": sum(int(row["lut_misses"]) for row in rows),
        "true_misses": sum(int(row["true_misses"]) for row in rows),
        "deferred_hits": sum(int(row["deferred_hits"]) for row in rows),
    }
    summary = {
        "config": str(config_path),
        "skip_pre_prune": skip_pre_prune,
        "candidate_count": len(rows),
        "feasible": sum(1 for row in rows if row["feasible"]),
        "infeasible": sum(1 for row in rows if not row["feasible"]),
        "strict_lut_ok": (
            totals["lut_misses"] == 0
            and totals["true_misses"] == 0
            and totals["deferred_hits"] == 0
        ),
        "lut_totals": totals,
        "latency_ms": _range(rows, "latency_ms"),
        "bram": _range(rows, "bram"),
        "lut": _range(rows, "lut"),
        "power_w": _range(rows, "power_w"),
        "violation_counts": _violation_counts(rows),
        "unique_arch": len({row["stage_blocks"] for row in rows}),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    architectures, estimator, search_space = enumerate_architectures(
        config_path,
        skip_pre_prune=args.skip_pre_prune,
    )
    rows = evaluate_architectures(architectures, estimator, search_space)
    write_outputs(
        rows=rows,
        config_path=config_path,
        output_dir=output_dir,
        skip_pre_prune=args.skip_pre_prune,
    )
    feasible = sum(1 for row in rows if row["feasible"])
    strict_lut_ok = all(
        int(row["lut_misses"]) == 0
        and int(row["true_misses"]) == 0
        and int(row["deferred_hits"]) == 0
        for row in rows
    )
    print(
        f"Enumerated {len(rows)} candidates from {config_path.name}: "
        f"{feasible}/{len(rows)} feasible, strict_lut_ok={strict_lut_ok} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
