import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate
from scripts.compare_search_methods import (
    exact_hv_curve_for_candidates,
    _holm_adjust,
    build_paired_method_comparison,
    add_joint_pareto_metrics,
    summarize_run,
    write_outputs,
)


def _write_run(
    root: Path,
    method: str,
    candidates: list[tuple[str, float | None, float]],
) -> dict[str, object]:
    results = root / "results"
    results.mkdir(parents=True)
    (results / "pareto_selection.json").write_text(
        json.dumps(
            {
                "objectives": ["macro_f1", "latency_ms"],
                "directions": ["max", "min"],
            }
        ),
        encoding="utf-8",
    )
    (results / "search_efficiency.json").write_text(
        json.dumps({"search_method": method}),
        encoding="utf-8",
    )
    records = []
    for arch_id, macro_f1, latency_ms in candidates:
        records.append(
            json.dumps(
                {
                    "feasible": True,
                    "candidate": {
                        "arch_id": arch_id,
                        "encoding": {"arch_id": arch_id},
                        "metrics": {
                            "macro_f1": macro_f1,
                            "latency_ms": latency_ms,
                        },
                    },
                }
            )
        )
    (results / "candidates.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    return {
        "run_root": str(root),
        "method": method,
        "seed": 42,
        "run_label": f"{method}:seed=42:{root.name}",
    }


def test_joint_pareto_contribution_and_pairwise_coverage(tmp_path: Path) -> None:
    rows = [
        _write_run(
            tmp_path / "aging",
            "aging_evolution",
            [("a1", 0.90, 10.0), ("a2", 0.80, 12.0)],
        ),
        _write_run(tmp_path / "rl", "rl", [("r1", 0.85, 11.0)]),
    ]

    summary = add_joint_pareto_metrics(rows)

    assert summary["available"] is True
    assert summary["joint_pareto_size"] == 1
    assert rows[0]["joint_pareto_contribution_count"] == 1
    assert rows[1]["joint_pareto_contribution_count"] == 0
    aging_label = str(rows[0]["run_label"])
    rl_label = str(rows[1]["run_label"])
    assert summary["pairwise_coverage"][aging_label][rl_label] == 1.0
    assert summary["pairwise_coverage"][rl_label][aging_label] == 0.5


def test_joint_pareto_missing_quality_keeps_stable_row_schema(tmp_path: Path) -> None:
    rows = [
        _write_run(tmp_path / "aging", "aging_evolution", [("a1", None, 10.0)]),
        _write_run(tmp_path / "rl", "rl", [("r1", None, 11.0)]),
    ]

    summary = add_joint_pareto_metrics(rows)

    assert summary["available"] is False
    assert summary["reason"] == "missing_complete_objective_vectors"
    for row in rows:
        assert row["complete_objective_candidate_count"] == 0
        assert row["joint_pareto_contribution_count"] == 0
        assert row["joint_pareto_contribution_fraction"] is None


def test_paired_method_comparison_uses_aging_minus_rl_and_matched_seeds() -> None:
    rows = []
    for seed, aging_quality, rl_quality, aging_gpu, rl_gpu in (
        (42, 0.82, 0.80, 1.0, 1.2),
        (43, 0.84, 0.81, 1.1, 1.3),
        (44, 0.83, 0.82, 0.9, 1.0),
    ):
        rows.extend(
            [
                {
                    "method": "aging_evolution",
                    "seed": seed,
                    "best_macro_f1": aging_quality,
                    "joint_pareto_contribution_count": 3,
                    "wall_clock_seconds": aging_gpu * 3600,
                    "gpu_reserved_hours": aging_gpu,
                    "job_wall_clock_seconds": aging_gpu * 4000,
                    "job_gpu_reserved_hours": aging_gpu * 1.1,
                },
                {
                    "method": "rl",
                    "seed": seed,
                    "best_macro_f1": rl_quality,
                    "joint_pareto_contribution_count": 2,
                    "wall_clock_seconds": rl_gpu * 3600,
                    "gpu_reserved_hours": rl_gpu,
                    "job_wall_clock_seconds": rl_gpu * 4000,
                    "job_gpu_reserved_hours": rl_gpu * 1.1,
                },
            ]
        )

    result = build_paired_method_comparison(rows)

    assert result["available"] is True
    assert result["inference_ready"] is True
    quality = result["metrics"]["best_macro_f1"]
    gpu = result["metrics"]["gpu_reserved_hours"]
    assert quality["difference_aging_minus_rl"]["mean"] == pytest.approx(0.02)
    assert quality["favored_by_mean"] == "aging_evolution"
    assert gpu["difference_aging_minus_rl"]["mean"] == pytest.approx(-1 / 6)
    assert gpu["favored_by_mean"] == "aging_evolution"
    assert quality["holm_adjusted_p_value"] >= quality["exact_two_sided_sign_test"]["p_value"]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def _write_completed_comparison_run(
    root: Path,
    *,
    method: str,
    macro_f1: float,
    latency_ms: float,
    job_gpu_hours: float,
) -> None:
    results = root / "results"
    results.mkdir(parents=True)
    config = {
        "project": {"seed": 42},
        "dataset": {"name": "dummy", "image_size": 32},
        "search_space": {"op_choices": ["conv"]},
        "constraints": {"max_latency_ms": 100.0},
        "hardware": {"board": "test"},
        "training": {"batch_size": 8},
        "search": {
            "method": method,
            "num_candidates": 1,
            "episodes": 1,
            "eval_epochs": 1,
            "selection_metric": "macro_f1",
            "objective_weights": {"accuracy": 1.0, "latency": 1.0},
            "pareto": {
                "objectives": ["macro_f1", "latency_ms"],
                "directions": ["max", "min"],
            },
        },
    }
    (root / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    candidate = {
        "arch_id": f"{method}_arch_0",
        "encoding": {"op": "conv"},
        "metrics": {"macro_f1": macro_f1, "top1": macro_f1, "latency_ms": latency_ms},
    }
    (results / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "total_evaluated": 1,
                "feasible": 1,
                "best_candidate": candidate,
                "extra": {
                    "search_method": method,
                    "train_epochs": 1,
                    "num_candidates": 1,
                    "num_episodes": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "search_efficiency.json").write_text(
        json.dumps(
            {
                "search_method": method,
                "wall_clock_seconds": job_gpu_hours * 3000,
                "gpu_reserved_hours": job_gpu_hours * 0.8,
                "gpu_event_seconds": job_gpu_hours * 2000,
                "peak_cuda_memory_bytes": 1000,
            }
        ),
        encoding="utf-8",
    )
    (results / "job_efficiency.json").write_text(
        json.dumps(
            {
                "wall_clock_seconds": job_gpu_hours * 3600,
                "gpu_reserved_hours": job_gpu_hours,
                "cuda_used": True,
                "exclusive_gpu_required": True,
                "segment_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (results / "pareto_selection.json").write_text(
        json.dumps(
            {
                "objectives": ["macro_f1", "latency_ms"],
                "directions": ["max", "min"],
                "pareto_front_size": 1,
                "hypervolume": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (results / "candidates.json").write_text(
        json.dumps([candidate]), encoding="utf-8"
    )
    (results / "candidates.jsonl").write_text(
        json.dumps({"feasible": True, "candidate": candidate}) + "\n",
        encoding="utf-8",
    )


def test_full_job_gpu_ledger_is_required_for_formal_comparison(tmp_path: Path) -> None:
    aging_root = tmp_path / "aging"
    rl_root = tmp_path / "rl"
    _write_completed_comparison_run(
        aging_root,
        method="aging_evolution",
        macro_f1=0.82,
        latency_ms=10.0,
        job_gpu_hours=1.0,
    )
    _write_completed_comparison_run(
        rl_root,
        method="rl",
        macro_f1=0.80,
        latency_ms=11.0,
        job_gpu_hours=1.2,
    )
    rows = [summarize_run(aging_root), summarize_run(rl_root)]
    payload = write_outputs(rows, tmp_path / "comparison")
    assert payload["comparison_ready"] is True
    assert payload["gpu_efficiency_ready"] is True
    assert payload["primary_compute_metric"] == "job_gpu_reserved_hours"
    paired = payload["paired_method_comparison"]["metrics"]
    assert paired["job_gpu_reserved_hours"]["favored_by_mean"] == "aging_evolution"


def test_exact_hv_curve_uses_frozen_three_objective_definition() -> None:
    candidates = [
        SearchCandidate(
            arch_id="a",
            encoding={},
            metrics=CandidateMetrics(f_clean=0.8, f_robust=0.7, latency_ms=5.0),
        ),
        SearchCandidate(
            arch_id="b",
            encoding={},
            metrics=CandidateMetrics(f_clean=0.9, f_robust=0.6, latency_ms=2.0),
        ),
    ]
    curve = exact_hv_curve_for_candidates(candidates, latency_limit_ms=10.0)
    assert curve[0]["exact_normalized_hypervolume"] == pytest.approx(0.8 * 0.7 * 0.5)
    assert curve[1]["exact_normalized_hypervolume"] == pytest.approx(0.472)
