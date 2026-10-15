from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_rl_search_metrics import summarize


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(arch_id: str, macro_f1: float, reward: float, feasible: bool = True) -> str:
    return json.dumps(
        {
            "feasible": feasible,
            "candidate": {
                "arch_id": arch_id,
                "metrics": {"macro_f1": macro_f1, "top1": macro_f1 + 0.05},
            },
            "extra": {"reward": reward},
        }
    )


def test_summarize_rl_search_metrics_computes_efficiency_curve(tmp_path: Path) -> None:
    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    (results / "candidates.jsonl").write_text(
        "\n".join(
            [
                _candidate("a", 0.40, 0.10, feasible=True),
                _candidate("b", 0.60, 0.20, feasible=False),
                _candidate("c", 0.80, 0.30, feasible=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        results / "search_efficiency.json",
        {
            "search_method": "rl",
            "cuda_used": True,
            "wall_clock_seconds": 360.0,
            "gpu_reserved_hours": 0.1,
        },
    )
    _write_json(
        results / "pareto_selection.json",
        {
            "pareto_front_size": 2,
            "hypervolume": 0.25,
            "hypervolume_metadata": {"kind": "exact_normalized_hypervolume"},
        },
    )

    summary, curve = summarize(results)

    assert summary["candidate_evaluations"] == 3
    assert summary["valid_candidate_count"] == 2
    assert summary["hardware_feasible_rate"] == pytest.approx(2 / 3)
    assert summary["best_so_far_macro_f1"] == pytest.approx(0.80)
    assert summary["sample_efficiency_candidates_to_95pct_final_best_macro_f1"] == 3
    assert summary["valid_candidates_per_gpu_hour"] == pytest.approx(20.0)
    assert curve[-1]["best_so_far_reward"] == pytest.approx(0.30)

