"""Deterministic architecture and work-unit manifests for Gate 0."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hwnas_fpga.hardware.cost import CostEstimate, FPGACostEstimator
from hwnas_fpga.search_space import ArchitectureSpec, SearchSpace
from hwnas_fpga.training.protocol_reporting import canonical_sha256


MANIFEST_SCHEMA_VERSION = 1


def architecture_id(architecture: ArchitectureSpec) -> str:
    digest = canonical_sha256(architecture.to_dict())[:16]
    return f"gate0_{digest}"


def _operator_counts(architecture: ArchitectureSpec) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in architecture.stages:
        for block in stage.blocks:
            counts[block.op] = counts.get(block.op, 0) + 1
    return counts


def _constraint_limit(
    name: str,
    *,
    estimator: FPGACostEstimator,
) -> float | None:
    constraints = estimator.constraints
    hardware = estimator.hardware_spec
    configured = getattr(constraints, name, None) if constraints is not None else None
    hardware_name = {
        "max_latency_ms": None,
        "max_dsp": "max_dsp",
        "max_bram": "max_bram",
        "max_lut": "max_lut",
    }[name]
    physical = getattr(hardware, hardware_name, None) if hardware_name else None
    values = [
        float(value)
        for value in (configured, physical)
        if value is not None and float(value) > 0
    ]
    return min(values) if values else None


def describe_architecture(
    architecture: ArchitectureSpec,
    cost: CostEstimate,
    *,
    estimator: FPGACostEstimator,
    boundary_band: float = 0.15,
) -> dict[str, Any]:
    operator_counts = _operator_counts(architecture)
    channels = [stage.channels for stage in architecture.stages]
    depths = [stage.depth for stage in architecture.stages]
    resources = {
        "latency_ms": float(cost.latency_ms),
        "dsp": int(cost.resource_dsp),
        "bram": int(cost.resource_bram),
        "lut": int(cost.resource_lut),
    }
    limits = {
        "latency_ms": _constraint_limit("max_latency_ms", estimator=estimator),
        "dsp": _constraint_limit("max_dsp", estimator=estimator),
        "bram": _constraint_limit("max_bram", estimator=estimator),
        "lut": _constraint_limit("max_lut", estimator=estimator),
    }
    utilizations = {
        key: (resources[key] / limit if limit else None)
        for key, limit in limits.items()
    }
    finite_utilizations = [
        float(value) for value in utilizations.values() if value is not None
    ]
    max_utilization = max(finite_utilizations, default=0.0)
    feasibility_margin = 1.0 - max_utilization
    feasible = not cost.violations
    if abs(feasibility_margin) <= float(boundary_band):
        feasibility_class = "near_boundary"
    elif feasible:
        feasibility_class = "feasible_interior"
    else:
        feasibility_class = "infeasible"
    dominant_op = (
        max(sorted(operator_counts), key=lambda op: operator_counts[op])
        if operator_counts
        else "none"
    )
    return {
        "architecture_id": architecture_id(architecture),
        "encoding": architecture.to_dict(),
        "architecture_sha256": canonical_sha256(architecture.to_dict()),
        "descriptors": {
            "stage_count": len(architecture.stages),
            "total_depth": sum(depths),
            "mean_channels": statistics_mean(channels),
            "max_channels": max(channels, default=0),
            "stage_channels": channels,
            "stage_depths": depths,
            "operator_counts": operator_counts,
            "dominant_op": dominant_op,
            "feasible": feasible,
            "feasibility_class": feasibility_class,
            "feasibility_margin": feasibility_margin,
            "max_resource_utilization": max_utilization,
        },
        "hardware_proxy": {
            **resources,
            "power_w": float(cost.power_w),
            "energy_mj": float(cost.energy_mj),
            "memory_bandwidth_gbps": float(cost.memory_bandwidth_gbps),
            "offchip_mem_mb": float(cost.offchip_mem_mb),
            "utilization": utilizations,
            "violations": list(cost.violations),
            "truth_status": "not_measured",
        },
    }


def statistics_mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def sample_candidate_pool(
    search_space: SearchSpace,
    estimator: FPGACostEstimator,
    *,
    pool_size: int,
    seed: int,
    max_attempt_multiplier: int = 20,
) -> list[dict[str, Any]]:
    """Uniformly sample and deduplicate a raw, policy-filtered architecture pool."""
    rng = random.Random(seed)
    candidates: dict[str, dict[str, Any]] = {}
    attempts = 0
    maximum_attempts = max(pool_size, pool_size * max_attempt_multiplier)
    while len(candidates) < pool_size and attempts < maximum_attempts:
        attempts += 1
        architecture = search_space.sample(
            rng=rng,
            apply_pruning=False,
            prefer_lightweight=False,
        )
        candidate_id = architecture_id(architecture)
        if candidate_id in candidates:
            continue
        estimator.reset_lut_stats()
        cost = estimator.estimate(architecture, search_space)
        candidates[candidate_id] = describe_architecture(
            architecture,
            cost,
            estimator=estimator,
        )
    if len(candidates) < pool_size:
        raise RuntimeError(
            f"only sampled {len(candidates)} unique architectures after {attempts} attempts"
        )
    return [candidates[key] for key in sorted(candidates)]


def _feature_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    operator_names = sorted(
        {
            op
            for candidate in candidates
            for op in candidate["descriptors"]["operator_counts"]
        }
    )
    numeric: list[list[float]] = []
    for candidate in candidates:
        descriptor = candidate["descriptors"]
        hardware = candidate["hardware_proxy"]
        depth = max(1, int(descriptor["total_depth"]))
        row = [
            math.log1p(float(hardware["latency_ms"])),
            math.log1p(float(hardware["dsp"])),
            math.log1p(float(hardware["bram"])),
            math.log1p(float(hardware["lut"])),
            float(descriptor["total_depth"]),
            float(descriptor["mean_channels"]),
            float(descriptor["max_channels"]),
            float(descriptor["max_resource_utilization"]),
        ]
        row.extend(
            float(descriptor["operator_counts"].get(op, 0)) / depth
            for op in operator_names
        )
        row.extend(
            1.0 if descriptor["feasibility_class"] == bucket else 0.0
            for bucket in ("feasible_interior", "near_boundary", "infeasible")
        )
        numeric.append(row)
    array = np.asarray(numeric, dtype=float)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std == 0] = 1.0
    return (array - mean) / std


def _quota_counts(target_count: int) -> dict[str, int]:
    near = max(1, round(target_count / 3))
    infeasible = max(1, round(target_count / 6))
    interior = target_count - near - infeasible
    return {
        "feasible_interior": interior,
        "near_boundary": near,
        "infeasible": infeasible,
    }


def select_stratified_architectures(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic maximin sample with feasibility-class quotas."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(candidates) < target_count:
        raise ValueError(
            f"candidate pool has {len(candidates)} rows, below target {target_count}"
        )
    features = _feature_matrix(candidates)
    rng = random.Random(seed)
    quotas = _quota_counts(target_count)
    available = {
        bucket: [
            index
            for index, candidate in enumerate(candidates)
            if candidate["descriptors"]["feasibility_class"] == bucket
        ]
        for bucket in quotas
    }

    allocated = {
        bucket: min(quotas[bucket], len(available[bucket]))
        for bucket in quotas
    }
    shortfall = target_count - sum(allocated.values())
    while shortfall > 0:
        expandable = [
            bucket
            for bucket in quotas
            if allocated[bucket] < len(available[bucket])
        ]
        if not expandable:
            raise RuntimeError("unable to allocate target across feasibility strata")
        bucket = max(
            expandable,
            key=lambda name: (len(available[name]) - allocated[name], name),
        )
        allocated[bucket] += 1
        shortfall -= 1

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    # Seed the maximin design with global extrema, preserving quota capacity.
    extrema: list[int] = []
    for column in range(features.shape[1]):
        extrema.extend(
            [int(np.argmin(features[:, column])), int(np.argmax(features[:, column]))]
        )
    extrema = list(dict.fromkeys(extrema))
    rng.shuffle(extrema)

    def bucket_for(index: int) -> str:
        return str(candidates[index]["descriptors"]["feasibility_class"])

    def selected_count(bucket: str) -> int:
        return sum(1 for index in selected if bucket_for(index) == bucket)

    for index in extrema:
        bucket = bucket_for(index)
        if (
            index in remaining
            and bucket in allocated
            and selected_count(bucket) < allocated[bucket]
        ):
            selected.append(index)
            remaining.remove(index)
        if len(selected) == target_count:
            break

    while len(selected) < target_count:
        eligible = [
            index
            for index in remaining
            if bucket_for(index) in allocated
            and selected_count(bucket_for(index)) < allocated[bucket_for(index)]
        ]
        if not eligible:
            raise RuntimeError("maximin selection exhausted eligible candidates")
        if not selected:
            chosen = rng.choice(eligible)
        else:
            selected_features = features[selected, :]
            chosen = max(
                eligible,
                key=lambda index: (
                    float(
                        np.square(selected_features - features[index]).sum(axis=1).min()
                    ),
                    str(candidates[index]["architecture_id"]),
                ),
            )
        selected.append(chosen)
        remaining.remove(chosen)

    selected_candidates = [dict(candidates[index]) for index in selected]
    selected_candidates.sort(key=lambda row: str(row["architecture_id"]))
    actual = {
        bucket: sum(
            1
            for candidate in selected_candidates
            if candidate["descriptors"]["feasibility_class"] == bucket
        )
        for bucket in quotas
    }
    operator_coverage = sorted(
        {
            op
            for candidate in selected_candidates
            for op, count in candidate["descriptors"]["operator_counts"].items()
            if count
        }
    )
    return selected_candidates, {
        "target_count": target_count,
        "requested_feasibility_quotas": quotas,
        "allocated_feasibility_quotas": allocated,
        "actual_feasibility_counts": actual,
        "operator_coverage": operator_coverage,
        "selection_method": (
            "policy-filtered uniform pool; feasibility quota; standardized "
            "hardware/width/depth/operator maximin"
        ),
        "selection_seed": seed,
        "selection_independent_of_classification_proxy": True,
    }


def build_work_units(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    outer_folds: Sequence[int],
    budgets: Sequence[int],
    truth_budget: int,
    proxy_name: str,
    zero_cost_proxy_name: str = "naswot_v1",
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["architecture_id"])):
        for fold in sorted(set(int(value) for value in outer_folds)):
            for seed in sorted(set(int(value) for value in seeds)):
                for budget in sorted(set(int(value) for value in budgets)):
                    if budget < 0:
                        raise ValueError("budgets must be non-negative")
                    payload = {
                        "architecture_id": candidate["architecture_id"],
                        "outer_fold": fold,
                        "seed": seed,
                        "budget": budget,
                        "proxy_name": (
                            proxy_name if budget > 0 else zero_cost_proxy_name
                        ),
                        "work_type": "zero_cost" if budget == 0 else "train",
                        "evaluation_scope": (
                            "inner_select_outer_once"
                            if budget == truth_budget
                            else "inner_only"
                        ),
                        "truth_budget": truth_budget,
                    }
                    payload["work_id"] = "gate0_" + canonical_sha256(payload)[:20]
                    units.append(payload)
    return units


def build_prefix_work_units(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stages: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    truth_budget: int,
    proxy_name: str,
    zero_cost_proxy_name: str = "naswot_v1",
) -> list[dict[str, Any]]:
    """Build one prefix-consistent trajectory per architecture/fold/seed.

    Stage definitions must partition the requested fold/seed cells.  A work
    unit trains once through ``truth_budget`` and snapshots best-so-far inner
    metrics at every positive budget; outer validation remains a one-time
    truth-budget operation.
    """
    normalized_budgets = sorted({int(value) for value in budgets})
    if not normalized_budgets or normalized_budgets[0] < 0:
        raise ValueError("budgets must be a non-empty set of non-negative integers")
    if int(truth_budget) not in normalized_budgets:
        raise ValueError("truth_budget must be present in budgets")
    positive = [value for value in normalized_budgets if value > 0]
    if not positive or positive[-1] != int(truth_budget):
        raise ValueError("truth_budget must be the largest positive budget")

    stage_cells: list[tuple[str, int, int, int]] = []
    seen_cells: set[tuple[int, int]] = set()
    for stage_order, stage in enumerate(stages):
        name = str(stage.get("name", "")).strip()
        if not name:
            raise ValueError("every prefix stage requires a non-empty name")
        folds = sorted({int(value) for value in stage.get("outer_folds", ())})
        seeds = sorted({int(value) for value in stage.get("seeds", ())})
        if not folds or not seeds:
            raise ValueError(f"stage {name!r} requires folds and seeds")
        for fold in folds:
            for seed in seeds:
                cell = (fold, seed)
                if cell in seen_cells:
                    raise ValueError(
                        f"fold/seed cell {cell} appears in more than one stage"
                    )
                seen_cells.add(cell)
                stage_cells.append((name, stage_order, fold, seed))

    units: list[dict[str, Any]] = []
    for stage_name, stage_order, fold, seed in stage_cells:
        for candidate in sorted(
            candidates,
            key=lambda row: str(row["architecture_id"]),
        ):
            payload = {
                "architecture_id": candidate["architecture_id"],
                "outer_fold": fold,
                "seed": seed,
                "budgets": normalized_budgets,
                "truth_budget": int(truth_budget),
                "proxy_name": proxy_name,
                "zero_cost_proxy_name": zero_cost_proxy_name,
                "work_type": "prefix_train",
                "evaluation_scope": "inner_milestones_outer_once",
                "stage": stage_name,
                "stage_order": stage_order,
            }
            payload["work_id"] = "gate0v2_" + canonical_sha256(payload)[:20]
            units.append(payload)
    return units


def write_manifest(
    output_dir: str | Path,
    *,
    source_config: str | Path,
    source_config_sha256: str,
    pool_size: int,
    pool_seed: int,
    candidates: Sequence[Mapping[str, Any]],
    sampling_summary: Mapping[str, Any],
    work_units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    destination = Path(output_dir)
    candidate_dir = destination / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths: dict[str, str] = {}
    for candidate in candidates:
        path = candidate_dir / f"{candidate['architecture_id']}.json"
        artifact = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "candidate": {
                "arch_id": candidate["architecture_id"],
                "encoding": candidate["encoding"],
                "metrics": candidate["hardware_proxy"],
            },
            "gate0": {
                "architecture_sha256": candidate["architecture_sha256"],
                "descriptors": candidate["descriptors"],
                "selection_independent_of_classification_proxy": True,
            },
        }
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        candidate_paths[str(candidate["architecture_id"])] = str(path.resolve())

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "audit": str(protocol.get("audit", "proxy_reliability_gate0")),
        "source_config": str(Path(source_config).resolve()),
        "source_config_sha256": source_config_sha256,
        "pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "sampling_summary": dict(sampling_summary),
        "protocol": dict(protocol),
        "candidate_count": len(candidates),
        "candidates": [
            {
                **dict(candidate),
                "candidate_artifact": candidate_paths[str(candidate["architecture_id"])],
            }
            for candidate in candidates
        ],
        "work_unit_count": len(work_units),
    }
    manifest["manifest_fingerprint"] = canonical_sha256(manifest)
    manifest_path = destination / "architecture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plan_path = destination / "run_matrix.jsonl"
    with plan_path.open("w", encoding="utf-8") as handle:
        for unit in work_units:
            resolved = {
                **dict(unit),
                "candidate_artifact": candidate_paths[str(unit["architecture_id"])],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_fingerprint": manifest["manifest_fingerprint"],
            }
            handle.write(json.dumps(resolved, ensure_ascii=False) + "\n")

    return {
        "manifest": str(manifest_path),
        "run_matrix": str(plan_path),
        "candidate_dir": str(candidate_dir),
    }


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
