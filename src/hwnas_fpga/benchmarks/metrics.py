"""Exact multi-objective, calibration, and open-set benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: str
    lower: float
    upper: float

    def validate(self) -> None:
        if self.direction not in {"min", "max"}:
            raise ValueError(f"{self.name}: direction must be min/max")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError(f"{self.name}: normalization bounds must be finite")
        if self.upper <= self.lower:
            raise ValueError(f"{self.name}: upper must exceed lower")


def normalize_objective_rows(
    rows: Sequence[Mapping[str, float]], specs: Sequence[ObjectiveSpec]
) -> np.ndarray:
    """Normalize objectives to minimization values in [0, 1]."""

    if not specs:
        raise ValueError("at least one objective is required")
    for spec in specs:
        spec.validate()
    matrix = np.empty((len(rows), len(specs)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, spec in enumerate(specs):
            value = float(row[spec.name])
            if not math.isfinite(value):
                raise ValueError(f"row {row_index} objective {spec.name} is not finite")
            normalized = (value - spec.lower) / (spec.upper - spec.lower)
            normalized = float(np.clip(normalized, 0.0, 1.0))
            matrix[row_index, column] = (
                normalized if spec.direction == "min" else 1.0 - normalized
            )
    return matrix


def _nondominated_min(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, points.shape[-1] if points.ndim == 2 else 0)
    unique = np.unique(points, axis=0)
    keep = np.ones(len(unique), dtype=bool)
    for index, point in enumerate(unique):
        dominated = np.all(unique <= point, axis=1) & np.any(unique < point, axis=1)
        dominated[index] = False
        if np.any(dominated):
            keep[index] = False
    return unique[keep]


def exact_hypervolume(
    points: Sequence[Sequence[float]] | np.ndarray,
    reference: Sequence[float],
) -> float:
    """Exact dominated hypervolume for minimization objectives.

    Each point defines an axis-aligned box from the point to ``reference``.
    A recursive slab sweep computes the exact union volume. The implementation
    is intended for the frozen three-objective benchmark but is exact for any
    modest dimension and candidate count.
    """

    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.ndim != 1 or reference_array.size == 0:
        raise ValueError("reference must be a non-empty 1-D vector")
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return 0.0
    if point_array.ndim != 2 or point_array.shape[1] != reference_array.size:
        raise ValueError("points shape must be [n, len(reference)]")
    if not np.all(np.isfinite(point_array)) or not np.all(np.isfinite(reference_array)):
        raise ValueError("points and reference must be finite")

    # Points outside the dominated reference box contribute no volume. Points
    # equal to the reference are harmless zero-volume boxes.
    valid = np.all(point_array <= reference_array, axis=1)
    point_array = _nondominated_min(point_array[valid])
    if point_array.size == 0:
        return 0.0

    def sweep(active_points: np.ndarray, ref: np.ndarray) -> float:
        if active_points.size == 0:
            return 0.0
        if ref.size == 1:
            return max(0.0, float(ref[0] - np.min(active_points[:, 0])))
        coordinates = sorted({float(value) for value in active_points[:, 0] if value < ref[0]})
        if not coordinates:
            return 0.0
        coordinates.append(float(ref[0]))
        volume = 0.0
        for left, right in zip(coordinates[:-1], coordinates[1:]):
            if right <= left:
                continue
            slab_points = active_points[active_points[:, 0] <= left, 1:]
            volume += (right - left) * sweep(_nondominated_min(slab_points), ref[1:])
        return float(volume)

    return sweep(point_array, reference_array)


def bidirectional_coverage(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> dict[str, float]:
    """Return C(left,right) and C(right,left) for minimization vectors."""

    def coverage(source: np.ndarray, target: np.ndarray) -> float:
        if len(target) == 0:
            return 0.0
        count = 0
        for point in target:
            if np.any(np.all(source <= point, axis=1) & np.any(source < point, axis=1)):
                count += 1
        return count / len(target)

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("coverage inputs must be 2-D")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("coverage inputs must have equal objective dimension")
    return {
        "c_left_right": coverage(left_array, right_array),
        "c_right_left": coverage(right_array, left_array),
    }


def ndcg_at_k(relevance: Sequence[float], ranking: Sequence[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevance_array = np.asarray(relevance, dtype=np.float64)
    indices = np.asarray(ranking, dtype=np.int64)[:k]
    if np.any(indices < 0) or np.any(indices >= len(relevance_array)):
        raise ValueError("ranking contains an invalid index")

    def dcg(values: np.ndarray) -> float:
        discounts = np.log2(np.arange(2, len(values) + 2, dtype=np.float64))
        return float(np.sum((np.power(2.0, values) - 1.0) / discounts))

    observed = dcg(relevance_array[indices])
    ideal = dcg(np.sort(relevance_array)[::-1][: min(k, len(relevance_array))])
    return observed / ideal if ideal > 0 else 0.0


def _validate_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray, targets: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.size or probs.shape[1] < 2:
        raise ValueError("probabilities must be [n, classes] and match targets")
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("targets must be non-empty valid class indices")
    if np.any(probs < 0) or not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return probs, labels


def _binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    indices = np.flatnonzero(np.r_[distinct[1:], True])
    tpr = np.r_[0.0, cumulative_tp[indices] / positives, 1.0]
    fpr = np.r_[0.0, cumulative_fp[indices] / negatives, 1.0]
    area = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, area


def calibration_summary(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[int],
    *,
    bins: int = 15,
) -> dict[str, object]:
    """Classification, calibration, and failure-prediction metrics."""

    if bins <= 1:
        raise ValueError("bins must exceed one")
    probs, labels = _validate_probabilities(probabilities, targets)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(float(accuracy) - float(mean_confidence))
        bin_rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )

    order = np.argsort(-confidence, kind="mergesort")
    sorted_errors = (~correct[order]).astype(np.float64)
    coverage = np.arange(1, len(labels) + 1, dtype=np.float64) / len(labels)
    risk = np.cumsum(sorted_errors) / np.arange(1, len(labels) + 1)
    aurc = float(np.mean(risk))

    failure_labels = (~correct).astype(np.int64)
    failure_scores = 1.0 - confidence
    fpr, tpr, failure_auroc = _binary_roc(failure_labels, failure_scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(np.min(fpr[eligible])) if eligible.size else math.nan
    return {
        "top1": float(correct.mean()),
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "aurc": aurc,
        "failure_auroc": failure_auroc,
        "failure_fpr95": fpr95,
        "reliability_bins": bin_rows,
        "risk_coverage": [
            {"coverage": float(c), "risk": float(r)} for c, r in zip(coverage, risk)
        ],
    }


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int]) -> float:
    scores = []
    for class_id in class_ids:
        tp = int(np.sum((labels == class_id) & (predictions == class_id)))
        fp = int(np.sum((labels != class_id) & (predictions == class_id)))
        fn = int(np.sum((labels == class_id) & (predictions != class_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def open_set_summary(
    targets: Sequence[int],
    known_predictions: Sequence[int],
    known_confidences: Sequence[float],
    *,
    known_class_ids: Sequence[int],
    confidence_threshold: float,
) -> dict[str, object]:
    """Evaluate a frozen known-vs-unknown sonar recognition protocol.

    ``known_predictions`` use original dataset class IDs. Samples whose target
    is outside ``known_class_ids`` are pooled into one unknown class. OSCR is
    the area under correct-known-classification-rate versus unknown false
    positive rate, using known confidence as the decision score.
    """

    labels = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(known_predictions, dtype=np.int64)
    confidence = np.asarray(known_confidences, dtype=np.float64)
    if labels.size == 0 or labels.shape != predictions.shape or labels.shape != confidence.shape:
        raise ValueError("targets, predictions, and confidences must be equal non-empty vectors")
    if not 0.0 <= confidence_threshold <= 1.0 or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidences and threshold must be in [0, 1]")
    known_ids = tuple(int(value) for value in known_class_ids)
    if not known_ids or len(known_ids) != len(set(known_ids)):
        raise ValueError("known_class_ids must be unique and non-empty")

    unknown_id = max(max(known_ids), int(labels.max()), int(predictions.max())) + 1
    target_known = np.isin(labels, known_ids)
    accepted = confidence >= confidence_threshold
    final_targets = np.where(target_known, labels, unknown_id)
    final_predictions = np.where(accepted, predictions, unknown_id)
    known_macro_f1 = _macro_f1(
        labels[target_known], predictions[target_known], known_ids
    )
    osfm = _macro_f1(final_targets, final_predictions, [*known_ids, unknown_id])

    recalls = []
    for class_id in [*known_ids, unknown_id]:
        mask = final_targets == class_id
        recalls.append(float(np.mean(final_predictions[mask] == class_id)) if np.any(mask) else 0.0)
    nma = float(np.mean(recalls))

    unknown_labels = (~target_known).astype(np.int64)
    fpr, tpr, unknown_auroc = _binary_roc(unknown_labels, 1.0 - confidence)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(np.min(fpr[eligible])) if eligible.size else math.nan

    known_count = max(1, int(target_known.sum()))
    unknown_count = max(1, int((~target_known).sum()))
    thresholds = np.r_[math.inf, np.sort(np.unique(confidence))[::-1], -math.inf]
    oscr_rows = []
    for threshold in thresholds:
        accept = confidence >= threshold
        ccr = float(np.sum(target_known & accept & (predictions == labels)) / known_count)
        unknown_fpr = float(np.sum((~target_known) & accept) / unknown_count)
        oscr_rows.append((unknown_fpr, ccr))
    oscr_rows = sorted(set(oscr_rows))
    oscr_mac = float(
        np.trapezoid(
            np.asarray([row[1] for row in oscr_rows]),
            np.asarray([row[0] for row in oscr_rows]),
        )
    )
    return {
        "known_macro_f1": known_macro_f1,
        "nma": nma,
        "osfm": osfm,
        "oscr_mac": oscr_mac,
        "unknown_auroc": unknown_auroc,
        "unknown_fpr95": fpr95,
        "confidence_threshold": float(confidence_threshold),
        "known_count": int(target_known.sum()),
        "unknown_count": int((~target_known).sum()),
        "definition": {
            "nma": "macro recall over each known class plus one pooled unknown class",
            "osfm": "macro F1 over each known class plus one pooled unknown class",
            "oscr_mac": "area under CCR-versus-unknown-FPR curve",
        },
    }
