#!/usr/bin/env python3
"""Freeze four diverse semantic-safe full-network calibration probes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.hardware.calibration_v2 import canonical_sha256
from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)


def block_choice_count(space, stage_index: int, in_channels: int, out_channels: int, stride: int) -> int:
    explicit = space.concrete_block_choices_for_stage(
        stage_index=stage_index,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    )
    if explicit is not None:
        return len(explicit)
    count = 0
    for op in space.available_ops(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    ):
        if op == "skip":
            count += 1
        elif op in {"mbconv", "fused_mbconv"}:
            count += len(space.config.kernel_choices) * len(space.config.expand_choices)
        else:
            count += len(space.config.kernel_choices)
    return count


def exact_cardinality(space) -> int:
    states = {space.config.stem_channels: 1}
    for stage_index, stride in enumerate(space.config.stage_strides):
        next_states: dict[int, int] = defaultdict(int)
        for in_channels, prior_count in states.items():
            for out_channels in space.config.channel_choices_for_stage(stage_index):
                for depth in space.config.depth_choices_for_stage(stage_index):
                    ways = block_choice_count(
                        space,
                        stage_index,
                        in_channels,
                        out_channels,
                        stride,
                    )
                    for _ in range(1, depth):
                        ways *= block_choice_count(
                            space,
                            stage_index,
                            out_channels,
                            out_channels,
                            1,
                        )
                    next_states[out_channels] += prior_count * ways
        states = dict(next_states)
    return sum(states.values())


def architecture_features(architecture, cost) -> dict[str, float]:
    blocks = [block for stage in architecture.stages for block in stage.blocks]
    skip_count = sum(block.op == "skip" for block in blocks)
    return {
        "depth": float(len(blocks)),
        "channel_sum": float(sum(stage.channels for stage in architecture.stages)),
        "skip_fraction": skip_count / max(1, len(blocks)),
        "latency_ms": float(cost.latency_ms),
        "dsp": float(cost.resource_dsp),
        "lut": float(cost.resource_lut),
        "bram": float(cost.resource_bram),
    }


def normalized_vectors(rows: list[dict[str, Any]]) -> list[list[float]]:
    keys = list(rows[0]["features"])
    minima = {key: min(row["features"][key] for row in rows) for key in keys}
    maxima = {key: max(row["features"][key] for row in rows) for key in keys}
    vectors = []
    for row in rows:
        vectors.append(
            [
                (row["features"][key] - minima[key])
                / max(1e-12, maxima[key] - minima[key])
                for key in keys
            ]
        )
    return vectors


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def select_maximin(rows: list[dict[str, Any]], count: int) -> list[int]:
    vectors = normalized_vectors(rows)
    centroid = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]
    selected = [max(range(len(rows)), key=lambda idx: distance(vectors[idx], centroid))]
    while len(selected) < min(count, len(rows)):
        remaining = [idx for idx in range(len(rows)) if idx not in selected]
        selected.append(
            max(
                remaining,
                key=lambda idx: min(
                    distance(vectors[idx], vectors[chosen]) for chosen in selected
                ),
            )
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml",
    )
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="artifacts/hw_surrogate_calibration_v2/probes",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    constraints = build_constraints(config)
    hardware = build_hardware_spec(config)
    space = build_search_space(
        config,
        image_size=config["dataset"]["image_size"],
        input_channels=config["dataset"]["input_channels"],
        num_classes=config["dataset"]["num_classes"],
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware,
        constraints=constraints,
    )
    space, policy = apply_operator_policies_to_search_space(
        space,
        estimator.operator_policies,
    )
    space = space.pre_prune(estimator)
    rng = random.Random(args.seed)
    rows_by_hash: dict[str, dict[str, Any]] = {}
    for _ in range(max(args.count, args.sample_count)):
        architecture = space.sample(rng=rng, apply_pruning=False)
        encoding = architecture.to_dict()
        architecture_hash = canonical_sha256(encoding)
        if architecture_hash in rows_by_hash:
            continue
        cost = estimator.estimate(architecture, space)
        rows_by_hash[architecture_hash] = {
            "architecture_hash": architecture_hash,
            "architecture": encoding,
            "features": architecture_features(architecture, cost),
            "analytic_metrics": {
                "latency_ms": cost.latency_ms,
                "dsp": cost.resource_dsp,
                "lut": cost.resource_lut,
                "bram": cost.resource_bram,
            },
            "analytic_violations": list(cost.violations),
        }
    rows = list(rows_by_hash.values())
    selected_rows = [rows[index] for index in select_maximin(rows, args.count)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = []
    for index, row in enumerate(selected_rows, start=1):
        probe_id = f"calibration_probe_{index:02d}"
        payload = {
            "schema_version": 1,
            "role": "hardware_calibration_probe_not_search_result",
            "frozen_before_route_measurement": True,
            "candidate": {
                "arch_id": probe_id,
                "encoding": row["architecture"],
                "metrics": row["analytic_metrics"],
            },
            "architecture_sha256": row["architecture_hash"],
            "selection_features": row["features"],
            "analytic_violations": row["analytic_violations"],
        }
        path = output_dir / f"{probe_id}.candidate.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        probes.append(
            {
                "probe_id": probe_id,
                "path": str(path.resolve()),
                "payload_sha256": canonical_sha256(payload),
                "architecture_sha256": row["architecture_hash"],
                "features": row["features"],
            }
        )
    manifest = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "config_sha256": canonical_sha256(config),
        "seed": args.seed,
        "sample_count_requested": args.sample_count,
        "unique_sample_count": len(rows),
        "exact_semantic_safe_space_cardinality": exact_cardinality(space),
        "operator_policy": policy,
        "selection": "maximin over normalized depth/channel/skip/analytic-cost features",
        "measurement_status": "frozen_unmeasured",
        "probes": probes,
    }
    (output_dir / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir.resolve()),
                "probes": len(probes),
                "space_cardinality": manifest["exact_semantic_safe_space_cardinality"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
