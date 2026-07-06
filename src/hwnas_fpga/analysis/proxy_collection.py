"""Validate Gate 0 observation records and collect the long analysis table."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hwnas_fpga.training.protocol_reporting import canonical_sha256


CLASSIFICATION_COLUMNS = (
    "architecture_id",
    "proxy_name",
    "budget",
    "seed",
    "outer_fold",
    "metric",
    "proxy_value",
    "truth_value",
    "proxy_direction",
    "status",
    "recipe_id",
    "run_path",
    "work_id",
)


def load_run_matrix(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"run matrix line {line_number} is not an object")
            rows.append(row)
    work_ids = [str(row.get("work_id", "")) for row in rows]
    duplicate_ids = sorted(
        work_id for work_id, count in Counter(work_ids).items() if count > 1
    )
    if not rows or duplicate_ids or any(not work_id for work_id in work_ids):
        raise ValueError(
            "invalid run matrix: "
            f"rows={len(rows)}, duplicate_work_ids={duplicate_ids[:10]}"
        )
    return rows


def _validate_record(
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    metrics: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    identity_fields = (
        "work_id",
        "architecture_id",
        "proxy_name",
        "budget",
        "seed",
        "outer_fold",
        "truth_budget",
    )
    for field in identity_fields:
        observed = record.get(field)
        expected = unit.get(field)
        if str(observed) != str(expected):
            errors.append(f"{field}: observed={observed!r}, expected={expected!r}")
    if record.get("status") != "completed":
        errors.append(f"status={record.get('status')!r}, expected='completed'")
    if not str(record.get("work_fingerprint", "")):
        errors.append("work_fingerprint is missing")
    if record.get("work_unit") != dict(unit):
        errors.append("embedded work_unit does not match run matrix")

    proxy_values = record.get("proxy_values")
    if not isinstance(proxy_values, Mapping):
        errors.append("proxy_values is not a mapping")
        proxy_values = {}
    for metric in metrics:
        if metric not in proxy_values:
            errors.append(f"proxy_values missing metric={metric}")

    budget = int(unit["budget"])
    truth_budget = int(unit["truth_budget"])
    truth_values = record.get("truth_values")
    if not isinstance(truth_values, Mapping):
        errors.append("truth_values is not a mapping")
        truth_values = {}
    outer_performed = bool(record.get("outer_evaluation_performed"))
    if budget == truth_budget:
        if not outer_performed:
            errors.append("truth-budget record did not perform outer evaluation")
        for metric in metrics:
            if metric not in truth_values:
                errors.append(f"truth_values missing metric={metric}")
    else:
        if outer_performed:
            errors.append("short-budget record illegally performed outer evaluation")
        if truth_values:
            errors.append("short-budget record contains outer truth values")
    return errors


def _validate_prefix_record(
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    metrics: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    for field in (
        "work_id",
        "architecture_id",
        "seed",
        "outer_fold",
        "truth_budget",
        "stage",
    ):
        if str(record.get(field)) != str(unit.get(field)):
            errors.append(
                f"{field}: observed={record.get(field)!r}, "
                f"expected={unit.get(field)!r}"
            )
    if record.get("status") != "completed" or not record.get("formal_eligible"):
        errors.append("prefix record is not a completed formal trajectory")
    if not str(record.get("work_fingerprint", "")):
        errors.append("work_fingerprint is missing")
    if record.get("work_unit") != dict(unit):
        errors.append("embedded work_unit does not match run matrix")
    if [int(value) for value in record.get("budgets", ())] != [
        int(value) for value in unit["budgets"]
    ]:
        errors.append("record budgets do not match run matrix")

    milestones = record.get("milestones")
    if not isinstance(milestones, Mapping):
        errors.append("milestones is not a mapping")
        milestones = {}
    for budget in unit["budgets"]:
        milestone = milestones.get(str(int(budget)))
        if not isinstance(milestone, Mapping):
            errors.append(f"milestones missing budget={budget}")
            continue
        proxy_values = milestone.get("proxy_values")
        if not isinstance(proxy_values, Mapping):
            errors.append(f"budget={budget} proxy_values is not a mapping")
            continue
        for metric in metrics:
            if metric not in proxy_values:
                errors.append(f"budget={budget} missing metric={metric}")

    truth_values = record.get("truth_values")
    if not isinstance(truth_values, Mapping):
        errors.append("truth_values is not a mapping")
        truth_values = {}
    for metric in metrics:
        if metric not in truth_values:
            errors.append(f"truth_values missing metric={metric}")
    if not record.get("outer_evaluation_performed"):
        errors.append("formal prefix trajectory did not perform outer evaluation")
    return errors


def collect_observations(
    *,
    run_matrix: str | Path,
    observations_dir: str | Path,
    stages: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect valid records; return long rows plus a strict completeness audit."""
    matrix_path = Path(run_matrix).resolve()
    all_units = load_run_matrix(matrix_path)
    selected_stages = {
        str(value).strip() for value in (stages or ()) if str(value).strip()
    }
    units = [
        unit
        for unit in all_units
        if not selected_stages
        or str(unit.get("stage", "unphased")) in selected_stages
    ]
    if not units:
        raise ValueError(f"no work units matched stages={sorted(selected_stages)}")
    manifest_path = Path(str(units[0]["manifest_path"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = [str(metric) for metric in manifest["protocol"]["execution"]["metrics"]]
    observation_root = Path(observations_dir).resolve()
    long_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    completed_by_budget: Counter[int] = Counter()
    completed_by_stage: Counter[str] = Counter()
    completed_work_units = 0
    positive_recipe_signatures: set[str] = set()
    prefix_mode = all(
        str(unit.get("work_type")) == "prefix_train" for unit in units
    )
    if prefix_mode != any(
        str(unit.get("work_type")) == "prefix_train" for unit in units
    ):
        raise ValueError("run matrix mixes prefix and independent work units")

    for unit in units:
        work_id = str(unit["work_id"])
        record_path = observation_root / f"{work_id}.json"
        if not record_path.exists():
            missing.append(work_id)
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("record must be a JSON object")
            errors = (
                _validate_prefix_record(record, unit, metrics=metrics)
                if prefix_mode
                else _validate_record(record, unit, metrics=metrics)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(
                {
                    "work_id": work_id,
                    "path": str(record_path),
                    "errors": [str(exc)],
                }
            )
            continue
        if errors:
            invalid.append(
                {
                    "work_id": work_id,
                    "path": str(record_path),
                    "errors": errors,
                }
            )
            continue

        completed_work_units += 1
        completed_by_stage[str(unit.get("stage", "unphased"))] += 1
        if prefix_mode:
            positive_recipe_signatures.add(
                canonical_sha256(dict(record["recipe"]))
            )
            truth_budget = int(unit["truth_budget"])
            for budget_value in unit["budgets"]:
                budget = int(budget_value)
                completed_by_budget[budget] += 1
                milestone = record["milestones"][str(budget)]
                for metric in metrics:
                    truth = (
                        record["truth_values"].get(metric)
                        if budget == truth_budget
                        else None
                    )
                    long_rows.append(
                        {
                            "architecture_id": unit["architecture_id"],
                            "proxy_name": milestone["proxy_name"],
                            "budget": budget,
                            "seed": int(unit["seed"]),
                            "outer_fold": int(unit["outer_fold"]),
                            "metric": metric,
                            "proxy_value": float(
                                milestone["proxy_values"][metric]
                            ),
                            "truth_value": (
                                "" if truth is None else float(truth)
                            ),
                            "proxy_direction": str(
                                milestone.get("proxy_direction", "max")
                            ),
                            "status": "completed",
                            "recipe_id": str(record.get("recipe_id", "")),
                            "run_path": str(record_path),
                            "work_id": work_id,
                        }
                    )
            continue

        budget = int(unit["budget"])
        completed_by_budget[budget] += 1
        if budget > 0:
            recipe = dict(record["recipe"])
            recipe.pop("epochs", None)
            positive_recipe_signatures.add(canonical_sha256(recipe))
        for metric in metrics:
            truth = record["truth_values"].get(metric)
            long_rows.append(
                {
                    "architecture_id": unit["architecture_id"],
                    "proxy_name": unit["proxy_name"],
                    "budget": budget,
                    "seed": int(unit["seed"]),
                    "outer_fold": int(unit["outer_fold"]),
                    "metric": metric,
                    "proxy_value": float(record["proxy_values"][metric]),
                    "truth_value": "" if truth is None else float(truth),
                    "proxy_direction": str(
                        record.get("proxy_direction", "max")
                    ),
                    "status": "completed",
                    "recipe_id": str(record.get("recipe_id", "")),
                    "run_path": str(record_path),
                    "work_id": work_id,
                }
            )

    recipe_consistent = len(positive_recipe_signatures) <= 1
    if not recipe_consistent:
        invalid.append(
            {
                "work_id": "__recipe__",
                "path": str(observation_root),
                "errors": [
                    "positive-budget recipes differ beyond the registered epoch budget"
                ],
            }
        )
    expected_by_budget: Counter[int] = Counter()
    for unit in units:
        if prefix_mode:
            expected_by_budget.update(int(value) for value in unit["budgets"])
        else:
            expected_by_budget[int(unit["budget"])] += 1
    expected_by_stage = Counter(
        str(unit.get("stage", "unphased")) for unit in units
    )
    summary = {
        "schema_version": 1,
        "audit": "proxy_reliability_gate0_collection",
        "run_matrix": str(matrix_path),
        "manifest": str(manifest_path),
        "observations_dir": str(observation_root),
        "selected_stages": sorted(selected_stages),
        "scheduler_policy": manifest["protocol"]["execution"].get(
            "scheduler_policy"
        ),
        "expected_work_units": len(units),
        "completed_work_units": completed_work_units,
        "missing_work_units": len(missing),
        "invalid_work_units": len(invalid),
        "expected_by_budget": {
            str(key): value for key, value in sorted(expected_by_budget.items())
        },
        "completed_by_budget": {
            str(key): value for key, value in sorted(completed_by_budget.items())
        },
        "expected_by_stage": {
            key: value for key, value in sorted(expected_by_stage.items())
        },
        "completed_by_stage": {
            key: value for key, value in sorted(completed_by_stage.items())
        },
        "missing_work_ids": missing,
        "invalid_records": invalid,
        "positive_budget_recipe_consistent": recipe_consistent,
        "ready_for_selected_scope": not missing and not invalid,
        "ready_for_formal_analysis": (
            not selected_stages
            and len(units) == len(all_units)
            and not missing
            and not invalid
        ),
        "classification_row_count": len(long_rows),
    }
    return long_rows, summary


def write_collection(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "classification_observations.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASSIFICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    summary_path = destination / "collection_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return {
        "classification_csv": str(csv_path),
        "collection_summary": str(summary_path),
    }
