#!/usr/bin/env python3
"""Summarize RL-only search efficiency and convergence records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_results_dir(run_dir: Path) -> Path:
    root = run_dir.expanduser().resolve()
    if (root / "candidates.jsonl").is_file():
        return root
    if (root / "results" / "candidates.jsonl").is_file():
        return root / "results"
    raise FileNotFoundError(f"cannot find candidates.jsonl under {root}")


def _metric(candidate_record: dict[str, Any], name: str) -> float | None:
    candidate = candidate_record.get("candidate")
    metrics = candidate.get("metrics") if isinstance(candidate, dict) else None
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _extra(candidate_record: dict[str, Any], name: str) -> float | None:
    extra = candidate_record.get("extra")
    if not isinstance(extra, dict):
        return None
    value = extra.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _arch_id(candidate_record: dict[str, Any]) -> str | None:
    candidate = candidate_record.get("candidate")
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("arch_id")
    return None if value is None else str(value)


def load_candidates(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = results_dir / "candidates.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def summarize(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = load_candidates(results_dir)
    efficiency_path = results_dir / "search_efficiency.json"
    efficiency = _read_json(efficiency_path) if efficiency_path.is_file() else {}
    pareto_path = results_dir / "pareto_selection.json"
    pareto = _read_json(pareto_path) if pareto_path.is_file() else {}
    if efficiency.get("search_method") not in {None, "rl"}:
        raise ValueError(
            "this summary is RL-only; use method-specific tooling for "
            f"{efficiency.get('search_method')!r}"
        )

    curve: list[dict[str, Any]] = []
    best_macro = None
    best_reward = None
    for index, record in enumerate(candidates, start=1):
        macro_f1 = _metric(record, "macro_f1")
        top1 = _metric(record, "top1")
        reward = _extra(record, "reward")
        reward_with_bonus = _extra(record, "reward_with_bonus")
        reward_for_best = reward if reward is not None else reward_with_bonus
        if macro_f1 is not None:
            best_macro = macro_f1 if best_macro is None else max(best_macro, macro_f1)
        if reward_for_best is not None:
            best_reward = (
                reward_for_best
                if best_reward is None
                else max(best_reward, reward_for_best)
            )
        curve.append(
            {
                "evaluation": index,
                "arch_id": _arch_id(record),
                "feasible": bool(record.get("feasible")),
                "macro_f1": macro_f1,
                "top1": top1,
                "reward": reward,
                "reward_with_bonus": reward_with_bonus,
                "best_so_far_macro_f1": best_macro,
                "best_so_far_reward": best_reward,
            }
        )

    final_best_macro = best_macro
    target_95 = None if final_best_macro is None else 0.95 * final_best_macro
    candidates_to_95 = None
    if target_95 is not None:
        for row in curve:
            value = row["best_so_far_macro_f1"]
            if value is not None and float(value) >= target_95:
                candidates_to_95 = int(row["evaluation"])
                break

    total = len(candidates)
    feasible = sum(1 for record in candidates if bool(record.get("feasible")))
    gpu_hours = efficiency.get("gpu_reserved_hours")
    gpu_hours_float = (
        float(gpu_hours)
        if gpu_hours is not None and math.isfinite(float(gpu_hours))
        else None
    )
    feasible_per_gpu_hour = (
        feasible / gpu_hours_float if gpu_hours_float and gpu_hours_float > 0 else None
    )
    summary = {
        "schema_version": 1,
        "source_results_dir": str(results_dir),
        "method": "rl",
        "candidate_evaluations": total,
        "valid_candidate_count": feasible,
        "hardware_feasible_rate": feasible / total if total else None,
        "best_so_far_macro_f1": final_best_macro,
        "best_so_far_reward": best_reward,
        "sample_efficiency_candidates_to_95pct_final_best_macro_f1": candidates_to_95,
        "convergence_speed_fraction_of_budget_to_95pct": (
            candidates_to_95 / total if candidates_to_95 is not None and total else None
        ),
        "valid_candidates_per_gpu_hour": feasible_per_gpu_hour,
        "search_time_wall_seconds": efficiency.get("wall_clock_seconds"),
        "gpu_reserved_hours": gpu_hours_float,
        "cpu_wall_time_seconds": (
            efficiency.get("wall_clock_seconds")
            if not bool(efficiency.get("cuda_used", False))
            else None
        ),
        "pareto_front_size": pareto.get("pareto_front_size"),
        "hypervolume": pareto.get("hypervolume"),
        "hypervolume_metadata": pareto.get("hypervolume_metadata"),
        "curve_csv": "rl_search_curve.csv",
        "claim_boundary": (
            "RL search efficiency only; this file does not compare RL against "
            "Random, Aging Evolution, Proxyless, or SPOS."
        ),
    }
    return summary, curve


def write_outputs(output_dir: Path, summary: dict[str, Any], curve: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rl_search_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "evaluation",
        "arch_id",
        "feasible",
        "macro_f1",
        "top1",
        "reward",
        "reward_with_bonus",
        "best_so_far_macro_f1",
        "best_so_far_reward",
    ]
    with (output_dir / "rl_search_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Search run root or its results/ directory.")
    parser.add_argument("--output-dir", help="Defaults to the search results directory.")
    args = parser.parse_args()
    results_dir = _resolve_results_dir(Path(args.run_dir))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir
    summary, curve = summarize(results_dir)
    write_outputs(output_dir, summary, curve)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

