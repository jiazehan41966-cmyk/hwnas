"""Enumerate MobileNetA ZCU104 proxy feasibility without training candidates."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec, StageSpec


def _violation_category(message: str) -> str:
    if ":" in message:
        return message.split(":", 1)[0]
    return message


def _block_label(block: Any, stage_index: int) -> str:
    return f"s{stage_index}:{block.op}_k{block.kernel_size}_e{block.expand_ratio}"


def enumerate_proxy(config_path: Path, *, topk: int) -> dict[str, Any]:
    config = load_config(str(config_path))
    constraints = build_constraints(config)
    dataset = config["dataset"]
    space = build_search_space(
        config,
        image_size=int(dataset["image_size"]),
        input_channels=int(dataset["input_channels"]),
        num_classes=int(dataset["num_classes"]),
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=build_hardware_spec(config),
        constraints=constraints,
    )

    search_config = space.config
    per_stage_choices = []
    in_channels = int(search_config.stem_channels)
    for stage_index, channel_choices in enumerate(search_config.stage_channel_choices):
        out_channels = int(channel_choices[0])
        stride = int(search_config.stage_strides[stage_index])
        choices = space.concrete_block_choices_for_stage(
            stage_index=stage_index,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
        per_stage_choices.append(tuple(choices or ()))
        in_channels = out_channels

    rows: list[dict[str, Any]] = []
    violation_counter: Counter[str] = Counter()
    feasible_count = 0
    for index, selected_blocks in enumerate(itertools.product(*per_stage_choices), start=1):
        stages = tuple(
            StageSpec(
                channels=int(search_config.stage_channel_choices[stage_index][0]),
                depth=1,
                stride=int(search_config.stage_strides[stage_index]),
                blocks=(block,),
            )
            for stage_index, block in enumerate(selected_blocks)
        )
        architecture = ArchitectureSpec(
            input_channels=int(search_config.input_channels),
            stem_channels=int(search_config.stem_channels),
            stages=stages,
            stem_stride=int(search_config.stem_stride),
            post_stem_downsample_stride=int(search_config.post_stem_downsample_stride),
            head_conv_channels=search_config.head_conv_channels,
            head_channels=search_config.head_channels,
            num_classes=search_config.num_classes,
        )
        estimate = estimator.estimate(architecture, space)
        feasible = not estimate.violations
        feasible_count += int(feasible)
        for violation in estimate.violations:
            violation_counter[_violation_category(violation)] += 1
        physical = estimate.physical_metrics
        rows.append(
            {
                "candidate_index": index,
                "feasible": feasible,
                "ops": [
                    _block_label(block, stage_index)
                    for stage_index, block in enumerate(selected_blocks)
                ],
                "latency_ms": estimate.latency_ms,
                "energy_mj": estimate.energy_mj,
                "lut": estimate.resource_lut,
                "bram": estimate.resource_bram,
                "dsp": estimate.resource_dsp,
                "zcu104_operator_lutram_heuristic_pressure": physical.get(
                    "zcu104_operator_lutram_heuristic_pressure"
                ),
                "zcu104_predicted_lut_as_memory_pressure": physical.get(
                    "zcu104_predicted_lut_as_memory_pressure"
                ),
                "zcu104_predicted_bram18_pressure": physical.get(
                    "zcu104_predicted_bram18_pressure"
                ),
                "physical_risk": physical.get("physical_risk"),
                "violations": list(estimate.violations),
            }
        )

    def best_by(key: str) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda item: float(item.get(key) or 0.0))[:topk]

    return {
        "schema_version": 1,
        "artifact_type": "zcu104_mobilenetA_proxy_feasibility_probe",
        "config_path": str(config_path),
        "search_space_version": config["search_space"].get("search_space_version"),
        "head_conv_channels": config["search_space"].get("head_conv_channels"),
        "constraints": config.get("constraints", {}),
        "hardware": config.get("hardware", {}),
        "candidate_count": len(rows),
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / len(rows) if rows else 0.0,
        "violation_top": violation_counter.most_common(),
        "lowest_predicted_lut_as_memory_pressure": best_by(
            "zcu104_predicted_lut_as_memory_pressure"
        ),
        "lowest_predicted_bram18_pressure": best_by("zcu104_predicted_bram18_pressure"),
        "lowest_lut": best_by("lut"),
        "lowest_bram": best_by("bram"),
        "claim_boundary": (
            "Proxy feasibility only; no candidate training, full-network route, "
            "board classification, FPS, or measured power is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    result = enumerate_proxy(Path(args.config), topk=max(1, int(args.topk)))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
