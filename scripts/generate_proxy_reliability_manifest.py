#!/usr/bin/env python3
"""Freeze the stratified architecture sample and Gate 0 run matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.analysis.proxy_manifest import (  # noqa: E402
    build_prefix_work_units,
    build_work_units,
    sample_candidate_pool,
    select_stratified_architectures,
    sha256_path,
    write_manifest,
)
from hwnas_fpga.runtime import (  # noqa: E402
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)


DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "audit" / "proxy_reliability_gate0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    source_config = Path(protocol["source_search_config"]).expanduser()
    if not source_config.is_absolute():
        source_config = (REPO_ROOT / source_config).resolve()
    config = load_config(str(source_config))

    sampling = protocol.get("sampling", {})
    pool_size = int(args.pool_size or sampling.get("pool_size", 4096))
    target_count = int(args.target_count or sampling.get("target_architectures", 48))
    seed = int(args.seed if args.seed is not None else sampling.get("seed", 20260704))
    output_dir = Path(
        args.output_dir
        or protocol.get(
            "output_dir",
            "results/proxy_reliability_gate0/manifest_v1",
        )
    )
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    dataset_cfg = config.get("dataset", {})
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size", 224)),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=int(dataset_cfg.get("num_classes", 8)),
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )
    search_space, operator_policy = apply_operator_policies_to_search_space(
        search_space,
        estimator.operator_policies,
    )

    pool = sample_candidate_pool(
        search_space,
        estimator,
        pool_size=pool_size,
        seed=seed,
    )
    selected, sampling_summary = select_stratified_architectures(
        pool,
        target_count=target_count,
        seed=seed,
    )
    sampling_summary["pool_feasibility_counts"] = {
        bucket: sum(
            1
            for candidate in pool
            if candidate["descriptors"]["feasibility_class"] == bucket
        )
        for bucket in ("feasible_interior", "near_boundary", "infeasible")
    }
    sampling_summary["operator_policy"] = operator_policy

    execution = protocol["execution"]
    scheduler_policy = str(
        execution.get("scheduler_policy", "independent_exact_budget_constant_lr")
    )
    if scheduler_policy == "prefix_consistent_single_trajectory_constant_lr":
        work_units = build_prefix_work_units(
            selected,
            stages=execution["stages"],
            budgets=execution["budgets"],
            truth_budget=int(execution["truth_budget"]),
            proxy_name=str(execution["proxy_name"]),
            zero_cost_proxy_name=str(
                execution.get("zero_cost_proxy_name", "naswot_v1")
            ),
        )
        planned_cells = {
            (int(unit["outer_fold"]), int(unit["seed"])) for unit in work_units
        }
        expected_cells = {
            (int(fold), int(seed))
            for fold in execution["outer_folds"]
            for seed in execution["seeds"]
        }
        if planned_cells != expected_cells:
            raise ValueError(
                "prefix stages do not exactly cover the registered fold/seed grid: "
                f"missing={sorted(expected_cells - planned_cells)}, "
                f"extra={sorted(planned_cells - expected_cells)}"
            )
    else:
        work_units = build_work_units(
            selected,
            seeds=execution["seeds"],
            outer_folds=execution["outer_folds"],
            budgets=execution["budgets"],
            truth_budget=int(execution["truth_budget"]),
            proxy_name=str(execution["proxy_name"]),
            zero_cost_proxy_name=str(
                execution.get("zero_cost_proxy_name", "naswot_v1")
            ),
        )
    outputs = write_manifest(
        output_dir,
        source_config=source_config,
        source_config_sha256=sha256_path(source_config),
        pool_size=pool_size,
        pool_seed=seed,
        candidates=selected,
        sampling_summary=sampling_summary,
        work_units=work_units,
        protocol=protocol,
    )
    print(
        json.dumps(
            {
                "candidate_count": len(selected),
                "work_unit_count": len(work_units),
                "scheduler_policy": scheduler_policy,
                **outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
