from __future__ import annotations

import math

import numpy as np
import pytest

from hwnas_fpga.benchmarks.metrics import (
    ObjectiveSpec,
    bidirectional_coverage,
    calibration_summary,
    exact_hypervolume,
    ndcg_at_k,
    normalize_objective_rows,
    open_set_summary,
)


def test_exact_hypervolume_known_1d_2d_3d_fronts() -> None:
    assert exact_hypervolume([[0.2]], [1.0]) == pytest.approx(0.8)
    assert exact_hypervolume([[0.2, 0.3]], [1.0, 1.0]) == pytest.approx(0.56)
    # Union of [0.2,1]x[0.8,1] and [0.8,1]x[0.2,1].
    assert exact_hypervolume([[0.2, 0.8], [0.8, 0.2]], [1.0, 1.0]) == pytest.approx(0.28)
    assert exact_hypervolume([[0.2, 0.3, 0.4]], [1.0, 1.0, 1.0]) == pytest.approx(
        0.8 * 0.7 * 0.6
    )


def test_exact_hypervolume_ignores_dominated_and_outside_points() -> None:
    base = exact_hypervolume([[0.2, 0.3], [0.4, 0.5]], [1.0, 1.0])
    assert base == pytest.approx(0.56)
    assert exact_hypervolume([[0.2, 0.3], [0.4, 0.5], [1.2, 0.1]], [1.0, 1.0]) == pytest.approx(base)


def test_objective_normalization_respects_direction() -> None:
    rows = [{"quality": 0.8, "latency": 20.0}]
    specs = [
        ObjectiveSpec("quality", "max", 0.0, 1.0),
        ObjectiveSpec("latency", "min", 0.0, 100.0),
    ]
    assert np.allclose(normalize_objective_rows(rows, specs), [[0.2, 0.2]])


def test_bidirectional_coverage_and_ndcg() -> None:
    coverage = bidirectional_coverage([[0.1, 0.1]], [[0.2, 0.2], [0.3, 0.3]])
    assert coverage == {"c_left_right": 1.0, "c_right_left": 0.0}
    assert ndcg_at_k([3.0, 2.0, 1.0], [0, 1, 2], 3) == pytest.approx(1.0)


def test_calibration_summary_perfect_predictions() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    summary = calibration_summary(probabilities, [0, 1, 0], bins=5)
    assert summary["top1"] == 1.0
    assert summary["nll"] > 0.0
    assert summary["brier"] > 0.0
    assert summary["aurc"] == 0.0
    assert math.isnan(summary["failure_auroc"])


def test_open_set_summary_separates_unknown_class() -> None:
    summary = open_set_summary(
        targets=[0, 1, 2, 3],
        known_predictions=[0, 1, 0, 1],
        known_confidences=[0.9, 0.8, 0.2, 0.1],
        known_class_ids=[0, 1],
        confidence_threshold=0.5,
    )
    assert summary["known_macro_f1"] == pytest.approx(1.0)
    assert summary["nma"] == pytest.approx(1.0)
    assert summary["osfm"] == pytest.approx(1.0)
    assert summary["unknown_auroc"] == pytest.approx(1.0)
    assert summary["oscr_mac"] == pytest.approx(1.0)
