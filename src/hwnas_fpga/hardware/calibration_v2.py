"""Evidence-tiered hardware calibration and interval screening."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


HARDWARE_METRICS = ("latency_ms", "dsp", "lut", "bram")
RESOURCE_METRICS = ("dsp", "lut", "bram")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def architecture_family(architecture: Mapping[str, Any]) -> str:
    ops: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            op = value.get("op")
            if op is not None:
                ops.add(str(op))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(architecture)
    sonar = sorted(ops & {"denoise", "edge", "mixconv"})
    if sonar:
        return "semantic_mismatch:" + "+".join(sonar)
    if ops <= {"mbconv", "skip", "conv"}:
        return "mainline_mbconv_skip"
    return "other:" + "+".join(sorted(ops))


def evidence_fingerprint(
    identity: Mapping[str, Any],
    *,
    operator_profile: str = "baseline_pi1_po1_u1",
    bit_width: int = 8,
    fpga_part: str = "xc7k325t-ffg676-2",
    target_clock_mhz: float = 200.0,
    tool_version: str = "unknown",
    harness_version: str = "unknown",
) -> str:
    return canonical_sha256(
        {
            "identity": identity,
            "operator_profile": operator_profile,
            "bit_width": int(bit_width),
            "fpga_part": fpga_part,
            "target_clock_mhz": float(target_clock_mhz),
            "tool_version": tool_version,
            "harness_version": harness_version,
        }
    )


def _metrics_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for metric in HARDWARE_METRICS:
        a = left.get(metric)
        b = right.get(metric)
        if a is None or b is None:
            continue
        if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9):
            return False
    return True


def deduplicate_pairs(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated implementations with identical evidence fingerprints."""
    grouped: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        fingerprint = str(row["fingerprint"])
        alias = {
            "run": row.get("run"),
            "arch_id": row.get("arch_id"),
            "source": row.get("source"),
        }
        if fingerprint not in grouped:
            row["aliases"] = [alias]
            row["repeat_count"] = 1
            grouped[fingerprint] = row
            continue
        existing = grouped[fingerprint]
        if not _metrics_close(existing.get("estimated", {}), row.get("estimated", {})):
            raise ValueError(f"conflicting estimates for fingerprint {fingerprint}")
        if not _metrics_close(existing.get("measured", {}), row.get("measured", {})):
            raise ValueError(f"conflicting measurements for fingerprint {fingerprint}")
        existing["aliases"].append(alias)
        existing["repeat_count"] += 1
    return sorted(grouped.values(), key=lambda item: item["fingerprint"])


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0 else None


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def fit_ratio_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    minimum_family_rows: int = 2,
) -> dict[str, Any]:
    ratios: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        estimated = row.get("estimated", {}).get(metric)
        measured = row.get("measured", {}).get(metric)
        if (
            estimated is None
            or measured is None
            or float(estimated) <= 0
            or float(measured) <= 0
        ):
            continue
        ratio = float(measured) / float(estimated)
        ratios.append(ratio)
        by_family[str(row.get("family", "unknown"))].append(ratio)
    if not ratios:
        return {"metric": metric, "available": False, "n": 0}

    def interval(values: Sequence[float]) -> dict[str, float]:
        # With small hardware samples, empirical extrema plus 5% expansion are
        # more honest than pretending a parametric confidence interval.
        return {
            "point_factor": statistics.median(values),
            "lower_factor": max(0.0, min(values) * 0.95),
            "upper_factor": max(values) * 1.05,
        }

    family_models = {
        family: interval(values)
        for family, values in by_family.items()
        if len(values) >= minimum_family_rows
    }
    return {
        "metric": metric,
        "available": True,
        "n": len(ratios),
        "global": interval(ratios),
        "families": family_models,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
    }


def predict_interval(
    model: Mapping[str, Any],
    *,
    estimated: float,
    family: str,
    source: str,
) -> dict[str, Any]:
    if model.get("available") is not True:
        return {
            "lower": None,
            "point": float(estimated),
            "upper": None,
            "source": source,
            "calibrated": False,
        }
    factors = model.get("families", {}).get(family) or model["global"]
    return {
        "lower": float(estimated) * float(factors["lower_factor"]),
        "point": float(estimated) * float(factors["point_factor"]),
        "upper": float(estimated) * float(factors["upper_factor"]),
        "source": source,
        "calibrated": True,
        "factor_scope": "family" if family in model.get("families", {}) else "global",
    }


def validate_ratio_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    budget: float | None = None,
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("estimated", {}).get(metric) is not None
        and row.get("measured", {}).get(metric) is not None
        and float(row["estimated"][metric]) > 0
        and float(row["measured"][metric]) > 0
    ]
    predictions: list[float] = []
    measured_values: list[float] = []
    absolute_percentage_errors: list[float] = []
    false_rejects = 0
    false_accepts = 0
    decisions = 0
    details = []
    for held_out in usable:
        training = [
            row for row in usable if row["fingerprint"] != held_out["fingerprint"]
        ]
        model = fit_ratio_model(training, metric=metric)
        interval = predict_interval(
            model,
            estimated=float(held_out["estimated"][metric]),
            family=str(held_out.get("family", "unknown")),
            source="leave_one_architecture_out",
        )
        measured = float(held_out["measured"][metric])
        point = float(interval["point"])
        predictions.append(point)
        measured_values.append(measured)
        absolute_percentage_errors.append(abs(point - measured) / measured)
        if budget is not None and interval["lower"] is not None and interval["upper"] is not None:
            actual_feasible = measured <= float(budget)
            predicted_reject = float(interval["lower"]) > float(budget)
            predicted_safe = float(interval["upper"]) <= float(budget)
            false_rejects += int(predicted_reject and actual_feasible)
            false_accepts += int(predicted_safe and not actual_feasible)
            decisions += 1
        details.append(
            {
                "fingerprint": held_out["fingerprint"],
                "family": held_out.get("family"),
                "estimated": held_out["estimated"][metric],
                "measured": measured,
                "predicted": interval,
            }
        )
    if not usable:
        return {"metric": metric, "available": False, "n": 0}
    return {
        "metric": metric,
        "available": True,
        "n": len(usable),
        "mape": statistics.fmean(absolute_percentage_errors),
        "mdape": statistics.median(absolute_percentage_errors),
        "p90_ape": percentile(absolute_percentage_errors, 0.90),
        "spearman": spearman(predictions, measured_values),
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "false_reject_rate": false_rejects / decisions if decisions else None,
        "false_accept_rate": false_accepts / decisions if decisions else None,
        "budget": budget,
        "details": details,
    }


def validation_gate(metric: str, validation: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if validation.get("available") is not True:
        reasons.append("validation unavailable")
    if int(validation.get("n", 0)) < 4:
        reasons.append("fewer than 4 unique validation groups")
    if validation.get("false_rejects") not in (0, None):
        reasons.append("observed false rejection")
    p90 = validation.get("p90_ape")
    if p90 is None or float(p90) > 0.25:
        reasons.append("P90 APE exceeds 25%")
    if metric == "latency_ms":
        rho = validation.get("spearman")
        if rho is None or float(rho) < 0.8:
            reasons.append("Spearman ranking below 0.8")
    return {
        "metric": metric,
        "hard_screening_enabled": not reasons,
        "mode": "interval_hard_screen" if not reasons else "pass_through_to_hls",
        "reasons": reasons,
    }


def classify_intervals(
    intervals: Mapping[str, Mapping[str, Any]],
    budgets: Mapping[str, float | int | None],
    gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checked: list[str] = []
    reject_metrics: list[str] = []
    uncertain_metrics: list[str] = []
    for metric, budget in budgets.items():
        if budget is None:
            continue
        checked.append(metric)
        gate = gates.get(metric, {})
        interval = intervals.get(metric, {})
        if gate.get("hard_screening_enabled") is not True:
            uncertain_metrics.append(metric)
            continue
        lower = interval.get("lower")
        upper = interval.get("upper")
        if lower is None or upper is None:
            uncertain_metrics.append(metric)
        elif float(lower) > float(budget):
            reject_metrics.append(metric)
        elif float(upper) > float(budget):
            uncertain_metrics.append(metric)
    if reject_metrics:
        status = "certified_reject"
    elif uncertain_metrics:
        status = "uncertain"
    else:
        status = "safe_feasible"
    return {
        "status": status,
        "checked_metrics": checked,
        "reject_metrics": reject_metrics,
        "uncertain_metrics": uncertain_metrics,
        "rule": (
            "reject only when a validated optimistic lower bound exceeds budget; "
            "unvalidated metrics pass through to HLS"
        ),
    }
