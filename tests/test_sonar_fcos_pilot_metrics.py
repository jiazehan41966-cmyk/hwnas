"""Independent, tiny detection fixtures for the new sonar pilot protocol."""
from pathlib import Path

import pytest

from tools.sonar_fcos_pilot.metrics import (
    baseline_gate,
    calibrate_threshold,
    evaluate_predictions,
    write_error_examples,
)


def record(boxes, labels=None, image_id="one", **kwargs):
    return {"image_id": image_id, "width": 100, "height": 100,
            "boxes": boxes, "labels": [1] * len(boxes) if labels is None else labels, **kwargs}


def prediction(boxes, scores=None, labels=None, image_id="one"):
    return {"image_id": image_id, "boxes": boxes,
            "labels": [1] * len(boxes) if labels is None else labels,
            "scores": [0.9] * len(boxes) if scores is None else scores}


def evaluate(records, predictions, **kwargs):
    pytest.importorskip("pycocotools")
    return evaluate_predictions(records, predictions, num_classes=2,
                                small_area_threshold=0.035, **kwargs)


def test_perfect_coco_ap_is_100_and_independent_of_diagnostic_threshold():
    box = [[0, 0, 10, 10]]
    result = evaluate([record(box)], [prediction(box, scores=[0.2])], score_threshold=0.25)
    assert result["map"] == pytest.approx(100)
    assert result["map50"] == pytest.approx(100)
    assert result["map75"] == pytest.approx(100)
    assert result["diagnostics"]["missed"] == 1
    assert result["diagnostics"]["true_positive"] == 0


def test_empty_predictions_with_nonempty_gt_have_zero_ap():
    result = evaluate([record([[0, 0, 10, 10]])], [])
    assert result["map"] == 0
    assert result["diagnostics"]["missed"] == 1
    assert result["diagnostics"]["false_positive"] == 0


def test_no_ground_truth_has_undefined_ap_but_counts_background_predictions():
    result = evaluate([record([])], [prediction([[10, 10, 20, 20]])])
    assert result["map"] is None
    assert result["coco_status"] == "NO_GROUND_TRUTH"
    assert result["diagnostics"]["recall"] is None
    assert result["diagnostics"]["false_positive"] == 1


def test_duplicate_prediction_does_not_match_gt_twice():
    box = [0, 0, 10, 10]
    diagnostic = evaluate([record([box])], [prediction([box, box])])["diagnostics"]
    assert diagnostic["true_positive"] == 1
    assert diagnostic["false_positive"] == 1
    assert diagnostic["localization"] == diagnostic["missed"] == 0
    assert diagnostic["false_positives_per_image"] == 1


def test_wrong_class_is_missed_object_and_remaining_prediction():
    box = [[0, 0, 10, 10]]
    diagnostic = evaluate([record(box)], [prediction(box, labels=[2])])["diagnostics"]
    assert diagnostic["true_positive"] == 0
    assert diagnostic["missed"] == 1
    assert diagnostic["false_positive"] == 1
    assert diagnostic["localization"] == 0


def test_near_localization_is_exclusive_and_counts_in_total_fp_budget():
    diagnostic = evaluate([record([[0, 0, 10, 10]])],
                          [prediction([[4, 0, 14, 10]])])["diagnostics"]
    assert diagnostic["localization"] == 1
    assert diagnostic["true_positive"] == diagnostic["missed"] == diagnostic["false_positive"] == 0
    assert diagnostic["false_positives_per_image"] == 1
    assert diagnostic["background_false_positives_per_image"] == 0
    assert baseline_gate(diagnostic)["status"] == "STOP_LOCALIZATION"


def test_small_recall_uses_full_gt_matching():
    # The prediction overlaps both objects above IoU 0.5, but matches the
    # large GT better. Re-evaluating only small GT would produce a false TP.
    gt = [[0, 0, 20, 20], [1, 1, 19, 19]]
    diagnostic = evaluate([record(gt)], [prediction([gt[0]])])["diagnostics"]
    assert diagnostic["small_gt"] == diagnostic["nonsmall_gt"] == 1
    assert diagnostic["small_recall"] == 0
    assert diagnostic["nonsmall_recall"] == 1
    assert baseline_gate(diagnostic)["status"] == "GO"


@pytest.mark.parametrize("bad_gt", [True, False])
def test_label_zero_is_rejected_instead_of_silently_remapped(bad_gt):
    box = [[0, 0, 10, 10]]
    with pytest.raises(ValueError, match="label 0"):
        evaluate([record(box, labels=[0] if bad_gt else [1])],
                 [prediction(box, labels=[1] if bad_gt else [0])])


def test_threshold_calibration_maximizes_recall_and_chooses_higher_tie():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predictions = [prediction(gt + [[60, 60, 70, 70]], scores=[0.9, 0.7, 0.6])]
    result = calibrate_threshold([record(gt)], predictions, 2, 0.035, max_fp_per_image=0.1)
    assert result["threshold"] == 0.7
    assert result["diagnostics"]["recall"] == 1
    assert result["diagnostics"]["false_positives_per_image"] == 0
    assert result["selection_set"] == "validation_only"


def test_threshold_calibration_tied_bad_and_good_scores_remain_atomic():
    result = calibrate_threshold([record([[0, 0, 10, 10]])],
                                 [prediction([[0, 0, 10, 10], [70, 70, 80, 80]], scores=[0.9, 0.9])],
                                 2, 0.035, max_fp_per_image=0)
    assert result["all_predictions_disabled"]
    assert result["threshold"] > 1
    assert result["diagnostics"]["recall"] == 0


def test_calibration_does_not_allow_localization_to_hide_false_predictions():
    result = calibrate_threshold([record([[0, 0, 10, 10]])],
                                 [prediction([[4, 0, 14, 10]])], 2, 0.035, max_fp_per_image=0)
    assert result["all_predictions_disabled"]


def test_baseline_gate_rejects_calibrated_threshold():
    diagnostic = evaluate([record([[0, 0, 10, 10]])], [], score_threshold=0.5)["diagnostics"]
    with pytest.raises(ValueError, match="0.25"):
        baseline_gate(diagnostic)


def test_baseline_gate_has_no_targeted_problem_for_perfect_detections():
    box = [[0, 0, 10, 10]]
    diagnostic = evaluate([record(box)], [prediction(box)])["diagnostics"]
    assert baseline_gate(diagnostic)["status"] == "STOP_NO_TARGETED_ERROR"


def test_original_image_bounds_and_prediction_id_are_validated():
    with pytest.raises(ValueError, match="original-image"):
        evaluate([record([[0, 0, 10, 10]])], [prediction([[0, 0, 640, 640]])])
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        evaluate([record([])], [prediction([], image_id="other")])


def test_error_examples_preserve_original_image_and_write_overlay(tmp_path):
    from PIL import Image
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 100)).save(image_path)
    paths = write_error_examples([record([[0, 0, 10, 10]], image_path=str(image_path))], [],
                                 tmp_path / "examples", 0.035)
    assert len(paths) == 1
    assert Path(paths[0]).is_file()
    with Image.open(paths[0]) as image:
        assert image.size == (100, 132)
