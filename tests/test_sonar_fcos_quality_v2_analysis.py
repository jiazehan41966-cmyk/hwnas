"""Synthetic exactness checks; never read evolving campaign AP or predictions."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sonar_fcos_quality.diagnostics import evaluate_quality_predictions
from tools.sonar_fcos_quality_v2.analysis import (
    CachedCocoAP75, bootstrap_suite, counts_at_threshold, equal_fppi_workpoint,
    evaluate_checkpoint, filtered_stage_diagnostics,
)


def record(image_id, boxes=None, labels=None):
    boxes = [[0, 0, 10, 10]] if boxes is None else boxes
    return {"image_id": image_id, "width": 100, "height": 100,
            "boxes": boxes, "labels": [1] * len(boxes) if labels is None else labels}


def prediction(image_id, boxes=None, scores=None, labels=None):
    boxes = [] if boxes is None else boxes
    return {"image_id": image_id, "boxes": boxes,
            "scores": [.8] * len(boxes) if scores is None else scores,
            "labels": [1] * len(boxes) if labels is None else labels}


def test_equal_fppi_uses_integer_budget_and_whole_tie_groups():
    records = [record(1), record(2), record(3)]
    preds = [prediction(1, [[0, 0, 10, 10], [20, 20, 30, 30]], [.9, .8]),
             prediction(2, [[20, 20, 30, 30], [30, 30, 40, 40]], [.7, .7])]
    point = equal_fppi_workpoint(records, preds, 1)
    assert point["integer_fp_budget"] == 1
    assert point["threshold"] == .8
    assert point["counts"]["false_positive"] == 1
    assert point["achieved_fppi"] == 1 / 3
    assert point["fppi_shortfall"] == pytest.approx(.5 - 1 / 3)
    assert point["false_positives_if_next_score_group_included"] == 3
    assert not point["exact_target_reached"] and not point["interpolation"]


def test_workpoint_includes_zero_fp_groups_and_handles_unreachable_extremes():
    records = [record(1), record(2)]
    preds = [prediction(1, [[20, 20, 30, 30], [0, 0, 10, 10]], [.9, .1])]
    point = equal_fppi_workpoint(records, preds, 1)
    assert point["threshold"] == 0
    assert point["exact_target_reached"]
    assert point["counts"]["true_positive"] == point["counts"]["false_positive"] == 1
    bad = [prediction(1, [[20, 20, 30, 30]] * 2, [1., 1.])]
    empty = equal_fppi_workpoint(records, bad, 1)
    assert empty["threshold"] > 1
    assert empty["counts"]["detections"] == 0
    assert empty["selection_reason"] == "highest_score_tie_group_exceeds_budget"
    missing = equal_fppi_workpoint(records, [], 1)
    assert missing["target_unreachable_in_saved_cache"]
    assert missing["counts"]["false_positive"] == 0


def test_greedy_matching_is_per_class_and_wrong_class_is_wrong_only_overlap():
    records = [record(1, [[0, 0, 10, 10], [1, 1, 11, 11]], [1, 2])]
    preds = [prediction(1, [[0, 0, 10, 10]] * 3 + [[1, 1, 11, 11]],
                        [.9, .8, .7, .6], [1, 1, 3, 2])]
    counts = counts_at_threshold(records, preds, 3, .25)
    assert counts["true_positive"] == 2
    assert counts["false_positive"] == 2
    assert counts["wrong_class_fp"] == 1  # Duplicate class 1 is not wrong-only.
    assert counts["false_negative"] == 0


def stage_row(image_id=1):
    stages = {}
    for stage, score in zip(("raw", "threshold", "topk", "nms", "top100", "final"),
                            (.9, .9, .8, .3, .3, .3)):
        stages[stage] = {"count": 1, "best_iou": 1., "good75_count": 1,
                         "good85_count": 1, "good75_max_score": score}
    return {"image_id": image_id, "gt_index": 0, "label": 1,
            "stages": stages, "mechanism_first_zero": "covered_final"}


def test_filtering_native_stages_preserves_raw_and_does_not_relabel_primary():
    records, stages = [record(1)], [stage_row()]
    diagnostic = filtered_stage_diagnostics(stages, records, .5)
    assert diagnostic["nms_first_loss"] == 1
    assert diagnostic["final_no_good_box"] == 1
    assert diagnostic["raw_no_good_box"] == 0
    assert diagnostic["protocol"]["nms_or_topk_recomputed"] is False
    preds = [prediction(1, [[0, 0, 10, 10]], [.3])]
    result = evaluate_checkpoint(records, preds, 1, stages, selection="epoch30", epoch=30)
    assert result["diagnostics"]["nms_first_loss"] == 0
    assert result["diagnostics"]["final_no_good_box"] == 0
    assert result["fixed_score_025"]["counts"]["true_positive"] == 1
    assert result["metrics"]["ap75"] == pytest.approx(100)
    assert result["ap_uses_full_native_prediction_cache"]
    del stages[0]["stages"]["nms"]["good75_max_score"]
    assert filtered_stage_diagnostics(stages, records, .5)["status"] == "UNAVAILABLE"


def explicit_resample(records, predictions, indices, classes):
    predicted = {item["image_id"]: item for item in predictions}
    duplicated_records, duplicated_predictions = [], []
    for image_id, index in enumerate(indices, 1):
        original = records[int(index)]
        copied = deepcopy(original)
        copied["image_id"] = image_id
        duplicated_records.append(copied)
        copied_pred = deepcopy(predicted.get(original["image_id"], prediction(original["image_id"])))
        copied_pred["image_id"] = image_id
        duplicated_predictions.append(copied_pred)
    return evaluate_quality_predictions(duplicated_records, duplicated_predictions, classes)["ap75"]


@pytest.mark.parametrize("indices", [[0, 0, 1], [2, 1, 2], [1, 0, 1], [2, 2, 2], [0, 1, 2]])
def test_cached_coco_matches_explicit_duplicates_with_ties_and_absent_classes(indices):
    records = [record(101), record(203, labels=[2]), record(305)]
    preds = [prediction(101, [[20, 20, 30, 30], [0, 0, 10, 10]] * 2, [.8] * 4),
             prediction(203, [[0, 0, 10, 10], [20, 20, 30, 30]], [.8, .8], [2, 2])]
    expected = explicit_resample(records, preds, indices, 2)
    actual = CachedCocoAP75(records, preds, 2).evaluate(indices)
    assert actual == pytest.approx(expected, abs=2e-13)


def test_cached_coco_obeys_100_per_image_per_class_and_zero_predictions():
    records = [record(1), record(2)]
    preds = [prediction(1, [[20, 20, 30, 30]] * 100 + [[0, 0, 10, 10]],
                        list(np.linspace(.99, .6, 100)) + [.5])]
    cached = CachedCocoAP75(records, preds, 1)
    assert cached.evaluate([0, 0, 1]) == explicit_resample(records, preds, [0, 0, 1], 1) == 0
    assert CachedCocoAP75(records, [], 1).evaluate() == 0
    with pytest.raises(ValueError, match="indices"):
        cached.evaluate([0, 2])


def run_predictions():
    records = [record(11), record(12), record(13)]
    a = [prediction(11, [[20, 20, 30, 30], [0, 0, 10, 10]], [.9, .8]),
         prediction(12, [[0, 0, 10, 10]], [.8])]
    b = [prediction(11, [[0, 0, 10, 10]], [.8]),
         prediction(12, [[0, 0, 10, 10]], [.8]),
         prediction(13, [[0, 0, 10, 10]], [.8])]
    runs = {(variant, seed): deepcopy(a if variant == "AAA" else b)
            for seed in (42, 43, 44) for variant in ("AAA", "QFL_DFL")}
    return records, runs


def test_paired_bootstrap_same_draw_all_six_and_linear_quantiles_exact():
    records, runs = run_predictions()
    # One seed differs to catch accidental ensemble or cross-seed concatenation.
    runs["QFL_DFL", 44] = deepcopy(runs["AAA", 44])
    actual = bootstrap_suite(records, runs, 1, n_resamples=12)
    rng = np.random.Generator(np.random.PCG64(20261012))
    differences = []
    for _ in range(12):
        indices = rng.integers(0, len(records), size=len(records), dtype=np.int64)
        differences.append([explicit_resample(records, runs["QFL_DFL", seed], indices, 1)
                            - explicit_resample(records, runs["AAA", seed], indices, 1)
                            for seed in (42, 43, 44)])
    np.testing.assert_allclose(actual["bootstrap_paired_seed_differences_pp"], differences, atol=2e-13, rtol=0)
    expected = np.asarray(differences).mean(axis=1)
    np.testing.assert_allclose(actual["bootstrap_mean_differences_pp"], expected, atol=2e-13, rtol=0)
    lo, hi = np.quantile(expected, [.025, .975], method="linear")
    assert actual["ci95_low_pp"] == pytest.approx(lo, abs=2e-13)
    assert actual["ci95_high_pp"] == pytest.approx(hi, abs=2e-13)
    assert actual["seed_sign_counts"] == {"positive": 2, "zero": 1, "negative": 0}


def test_stability_averages_five_individual_epoch_metrics_not_predictions():
    records, runs = run_predictions()
    single = bootstrap_suite(records, runs, 1, n_resamples=7)
    views = {key: [deepcopy(value)] * 5 for key, value in runs.items()}
    # Four quality epochs equal the AAA cache; the mean difference is one fifth.
    for seed in (42, 43, 44):
        views["QFL_DFL", seed] = [runs["QFL_DFL", seed]] + [runs["AAA", seed]] * 4
    stable = bootstrap_suite(records, views, 1, n_resamples=7)
    assert stable["checkpoint_views_per_run"] == 5
    assert stable["resampled_indices_sha256"] == single["resampled_indices_sha256"]
    np.testing.assert_allclose(stable["bootstrap_mean_differences_pp"],
                               np.asarray(single["bootstrap_mean_differences_pp"]) / 5, atol=2e-13)
    del views["AAA", 44]
    with pytest.raises(ValueError, match="exactly"):
        bootstrap_suite(records, views, 1, n_resamples=7)
