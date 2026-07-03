from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from statistics import median
from typing import TYPE_CHECKING, Any, Optional

from .space import ArchitectureSpec, SearchSpace

if TYPE_CHECKING:
    from hwnas_fpga.hardware import CostEstimate, FPGACostEstimator


@dataclass(frozen=True)
class ProbeRecord:
    arch_id: str
    architecture: ArchitectureSpec
    cost_estimate: "CostEstimate"
    feasible: bool
    violations: tuple[str, ...]


def probe_search_space(
    search_space: SearchSpace,
    cost_estimator: "FPGACostEstimator",
    *,
    num_samples: int = 200,
    seed: int = 42,
    prefer_lightweight: bool = False,
) -> tuple[list[ProbeRecord], dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = Random(seed)
    records: list[ProbeRecord] = []

    for index in range(num_samples):
        architecture = search_space.sample(
            seed=None,
            rng=rng,
            cost_estimator=cost_estimator,
            apply_pruning=False,
            require_feasible=False,
            prefer_lightweight=prefer_lightweight,
        )
        cost_estimate = cost_estimator.estimate(architecture, search_space)
        records.append(
            ProbeRecord(
                arch_id=f"probe_{index:04d}",
                architecture=architecture,
                cost_estimate=cost_estimate,
                feasible=len(cost_estimate.violations) == 0,
                violations=tuple(cost_estimate.violations),
            )
        )

    return records, summarize_probe_records(records)


def summarize_probe_records(records: list[ProbeRecord]) -> dict[str, Any]:
    feasible_records = [record for record in records if record.feasible]
    infeasible_records = [record for record in records if not record.feasible]

    violation_counter: Counter[str] = Counter()
    violation_pattern_counter: Counter[str] = Counter()
    for record in infeasible_records:
        violation_counter.update(record.violations)
        violation_pattern_counter[" | ".join(record.violations)] += 1

    return {
        "total_samples": len(records),
        "feasible_samples": len(feasible_records),
        "infeasible_samples": len(infeasible_records),
        "feasible_ratio": (len(feasible_records) / len(records)) if records else 0.0,
        "feasible_ratio_pct": ((len(feasible_records) * 100.0) / len(records)) if records else 0.0,
        "violation_counts": dict(violation_counter.most_common()),
        "violation_patterns": dict(violation_pattern_counter.most_common()),
        "metrics_all": _summarize_metric_group(records),
        "metrics_feasible": _summarize_metric_group(feasible_records),
        "metrics_infeasible": _summarize_metric_group(infeasible_records),
    }


def _summarize_metric_group(records: list[ProbeRecord]) -> dict[str, Optional[dict[str, float]]]:
    return {
        "latency_ms": _numeric_summary([record.cost_estimate.latency_ms for record in records]),
        "energy_mj": _numeric_summary([record.cost_estimate.energy_mj for record in records]),
        "dsp": _numeric_summary([record.cost_estimate.resource_dsp for record in records]),
        "bram": _numeric_summary([record.cost_estimate.resource_bram for record in records]),
        "lut": _numeric_summary([record.cost_estimate.resource_lut for record in records]),
        "memory_bandwidth_gbps": _numeric_summary(
            [record.cost_estimate.memory_bandwidth_gbps for record in records]
        ),
        "offchip_mem_mb": _numeric_summary(
            [record.cost_estimate.offchip_mem_mb for record in records]
        ),
    }


def _numeric_summary(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None

    normalized = [float(value) for value in values]
    return {
        "min": min(normalized),
        "max": max(normalized),
        "mean": sum(normalized) / len(normalized),
        "median": float(median(normalized)),
    }
