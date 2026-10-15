"""Synthetic-only checks for postdecode error diagnostics. No dataset access."""

from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "sonar_operator_pool_p0"))
from error_breakdown import analyze_records, input_size_bin


def record(boxes=(), scores=(), labels=(), gt_boxes=(), gt_labels=()):
    return dict(image="synthetic", prediction=dict(boxes=list(boxes), scores=list(scores), labels=list(labels)),
                target=dict(boxes=list(gt_boxes), labels=list(gt_labels)))


def analyze(item):
    return analyze_records([item], ["cylider", "manta"], "valid", 42, "synthetic_best")


def test_tp_is_one_to_one_and_duplicate_uses_score_order():
    item = record([[0,0,20,20], [0,0,20,20]], [.6,.9], [0,0], [[0,0,20,20]], [0])
    before = deepcopy(item)
    out = analyze(item)
    assert item == before
    assert (out["summary"]["tp"], out["summary"]["fp"], out["summary"]["fn"]) == (1,1,0)
    assert out["summary"]["precision"] == .5
    assert [r["kind"] for r in out["prediction_rows"]] == ["duplicate", "TP"]
    assert out["gt_rows"][0]["matched_prediction_index"] == 1
    json.dumps(out, allow_nan=False)


def test_low_score_matching_box_is_fn_not_background_fp():
    out = analyze(record([[0,0,20,20]], [.2], [0], [[0,0,20,20]], [0]))
    assert out["gt_rows"][0]["reason"] == "score_below_threshold"
    assert out["summary"]["fn"] == 1
    assert out["summary"]["fp"] == 0
    assert out["prediction_rows"][0]["kind"] == "below_operating_threshold"


@pytest.mark.parametrize("score,reason", [(.5,"detected"),(.01,"score_below_threshold"),(.009,"no_retained_candidate")])
def test_score_boundaries(score, reason):
    out = analyze(record([[0,0,20,20]], [score], [0], [[0,0,20,20]], [0]))
    assert out["gt_rows"][0]["reason"] == reason


def test_wrong_class_and_localization_are_distinct():
    wrong = analyze(record([[0,0,20,20]], [.9], [1], [[0,0,20,20]], [0]))
    assert wrong["gt_rows"][0]["reason"] == "class_confusion"
    assert wrong["prediction_rows"][0]["kind"] == "classification"
    local = analyze(record([[10,0,30,20]], [.9], [0], [[0,0,20,20]], [0]))
    assert local["gt_rows"][0]["reason"] == "localization"
    assert local["prediction_rows"][0]["kind"] == "localization"


def test_wrong_class_low_iou_is_classification_and_localization_not_pure_background():
    out = analyze(record([[10,0,30,20]], [.9], [1], [[0,0,20,20]], [0]))
    assert out["prediction_rows"][0]["kind"] == "classification_and_localization"
    # No same-class candidate or wrong-class high-overlap candidate explains this GT.
    assert out["gt_rows"][0]["reason"] == "no_retained_candidate"


def test_background_and_missing_gt():
    out = analyze(record([[50,50,70,70]], [.9], [0], [[0,0,20,20]], [0]))
    assert out["prediction_rows"][0]["kind"] == "background"
    assert out["gt_rows"][0]["reason"] == "no_retained_candidate"
    assert out["summary"]["fp_background"] == 1


def test_assignment_conflict_is_not_low_score_or_no_candidate():
    out = analyze(record([[0,0,20,20]], [.9], [0], [[0,0,20,20],[1,0,21,20]], [0,0]))
    assert [r["reason"] for r in out["gt_rows"]] == ["detected", "assignment_conflict"]
    assert out["summary"]["tp"] == 1 and out["summary"]["fn"] == 1


def test_matching_uses_best_available_gt_not_already_claimed_gt():
    out = analyze(record([[0,0,20,20],[1,0,21,20]], [.9,.8], [0,0], [[0,0,20,20],[2,0,22,20]], [0,0]))
    assert out["summary"]["tp"] == 2
    assert out["summary"]["fp_duplicate"] == 0


def test_iou75_matching_is_recomputed_independently():
    # Higher-score prediction has IoU .6, lower-score has IoU 1.0.
    # Independent .75 matching must allow the second prediction to become TP.
    out = analyze(record([[5,0,25,20],[0,0,20,20]], [.9,.8], [0,0], [[0,0,20,20]], [0]))
    assert [r["tp_iou50"] for r in out["prediction_rows"]] == [True, False]
    assert [r["tp_iou75"] for r in out["prediction_rows"]] == [False, True]
    assert out["summary"]["tp_iou75"] == 1


def test_empty_records_and_empty_classes_preserve_denominators():
    out = analyze(record())
    assert out["summary"]["gt_count"] == out["summary"]["fp"] == 0
    assert out["summary"]["precision"] is None and out["summary"]["recall"] is None
    assert all(r["recall"] is None for r in out["class_size_rows"])
    empty = analyze_records([], ["a"], "train", 42, "synthetic")
    assert empty["summary"]["images"] == 0
    json.dumps(empty, allow_nan=False)


def test_no_gt_predictions_are_background_but_not_recall_zero():
    out = analyze(record([[1,1,10,10]], [.8], [1]))
    assert out["summary"]["fp_background"] == 1
    assert out["summary"]["precision"] == 0
    assert out["summary"]["recall"] is None


def test_gt_size_denominators_do_not_include_empty_class_or_wrong_pred_size():
    out = analyze(record([[0,0,20,20]], [.9], [0], [[0,0,20,20],[50,50,90,90]], [0,0]))
    by_key = {(r["class_id"],r["size_bin"]):r for r in out["class_size_rows"]}
    assert by_key[0,"all"]["recall"] == .5
    assert by_key[0,"side_16_to_lt32"]["recall"] == 1
    assert by_key[0,"side_ge32"]["recall"] == 0
    assert by_key[1,"all"]["recall"] is None


@pytest.mark.parametrize("side,expected", [(15.99,"side_lt16"),(16,"side_16_to_lt32"),(31.99,"side_16_to_lt32"),(32,"side_ge32")])
def test_size_boundary_is_equivalent_side_not_coco(side, expected):
    assert input_size_bin([0,0,side,side]) == expected
    # Same area, asymmetric width/height, same bin.
    assert input_size_bin([0,0,side/2,side*2]) == expected


def test_test_split_rejected_before_records_read():
    with pytest.raises(ValueError, match="sealed"):
        analyze_records([None], ["a"], "test", 42, "synthetic")


@pytest.mark.parametrize("item", [
    record([[0,0,0,20]], [.9], [0]),
    record([[0,0,20,20]], [float("nan")], [0]),
    record([[0,0,20,20]], [.9], [2]),
    record([[0,0,20,20]], [.9], [.5]),
    record([[0,0,20,20]], [], [0]),
])
def test_malformed_records_rejected(item):
    with pytest.raises(ValueError):
        analyze(item)
