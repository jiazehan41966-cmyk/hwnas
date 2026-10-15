"""Semantic fixtures and frozen-evidence replay for the quality campaign."""
import copy
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from tools.sonar_fcos_quality.diagnostics import (
    StageDiagnosticTracer, evaluate_quality_predictions, stage_rows_for_image,
    summarize_stage_rows, wrong_class_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "results/fcos_full_validation_diagnosis_20261008_01/remote/output"


def record(boxes, labels=None, image_id=1):
    return {"image_id": image_id, "width": 100, "height": 100,
            "boxes": boxes, "labels": labels or [1] * len(boxes)}


def prediction(boxes, labels=None, scores=None, image_id=1):
    return {"image_id": image_id, "boxes": boxes,
            "labels": labels or [1] * len(boxes),
            "scores": scores if scores is not None else [.9] * len(boxes)}


def test_nms_loss_is_gt_first_loss_not_removed_box_count():
    gt = record([[0, 0, 10, 10]])
    rows = stage_rows_for_image(gt, [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 12, 10]],
                               [[0, .9], [0, .8], [0, .95]],
                               [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 12, 10]],
                               [1, 1, 1], [2], [2], .001)
    # IoU .833 still survives: removal of two perfect boxes is not first loss.
    assert summarize_stage_rows(rows, [gt])["nms_first_loss"] == 0
    rows = stage_rows_for_image(gt, [[0, 0, 10, 10], [0, 0, 14, 10]],
                               [[0, .8], [0, .95]], [[0, 0, 10, 10], [0, 0, 14, 10]],
                               [1, 1], [1], [1], .001)
    assert summarize_stage_rows(rows, [gt])["nms_first_loss"] == 1


def test_raw_geometry_is_score_independent_and_threshold_is_strict():
    gt = record([[0, 0, 10, 10]])
    rows = stage_rows_for_image(gt, [[0, 0, 10, 10]], [[.9, .001]],
                               [[0, 0, 10, 10]], [0], [0], [0], .001)
    result = summarize_stage_rows(rows, [gt])
    assert result["raw_no_good_box"] == result["nms_first_loss"] == 0
    assert result["first_zero"]["threshold"] == 1
    assert result["stage_covered_gt"]["raw"] == 1


def test_first_zero_raw_and_top100_are_distinct():
    gt = record([[0, 0, 10, 10], [50, 50, 60, 60]])
    rows = stage_rows_for_image(gt, [[0, 0, 10, 10]], [[0, .9]],
                               [[0, 0, 10, 10]], [1], [0], [], .001)
    summary = summarize_stage_rows(rows, [gt])
    assert summary["raw_no_good_box"] == 1
    assert summary["first_zero"]["top100"] == 1
    assert summary["nms_first_loss"] == 0


def test_good_iou_boundary_is_inclusive():
    gt = record([[0, 0, 4, 4]])
    rows = stage_rows_for_image(gt, [[0, 0, 3, 4]], [[0, .9]],
                               [[0, 0, 3, 4]], [1], [0], [0], .001)
    assert summarize_stage_rows(rows, [gt])["first_zero"]["covered_final"] == 1


def test_summary_rejects_missing_duplicate_or_non_nested_rows():
    gt = record([[0, 0, 4, 4]])
    rows = stage_rows_for_image(gt, [[0, 0, 4, 4]], [[0, .9]],
                               [[0, 0, 4, 4]], [1], [0], [0], .001)
    for invalid in ([], rows + rows):
        with pytest.raises(ValueError, match="exactly once"):
            summarize_stage_rows(invalid, [gt])
    altered = copy.deepcopy(rows)
    altered[0]["stages"]["nms"]["good75_count"] = 2
    with pytest.raises(ValueError, match="non-nested"):
        summarize_stage_rows(altered, [gt])


def test_wrong_class_excludes_overlapping_same_class_and_duplicate_predictions():
    gt = record([[0, 0, 10, 10], [0, 0, 10, 10]], labels=[1, 2])
    pred = prediction([[0, 0, 10, 10]] * 4, labels=[1, 1, 2, 3])
    result = wrong_class_diagnostics([gt], [pred], 3)
    assert result["wrong_class_fp"] == 1
    assert result["per_image"][0]["prediction_indices"] == [3]


def test_wrong_class_score_and_iou_boundaries_count_prediction_boxes():
    gt = record([[0, 0, 10, 10]])
    pred = prediction([[0, 0, 5, 10]] * 3, labels=[2, 2, 2], scores=[.25, .25, .2499])
    assert wrong_class_diagnostics([gt], [pred], 2)["wrong_class_fp"] == 2


def test_true_coco_ap85_and_standard_evaluator_parity():
    pytest.importorskip("pycocotools")
    from tools.sonar_fcos_pilot.metrics import evaluate_predictions
    gt = record([[0, 0, 10, 10]])
    pred = prediction([[0, 0, 8, 10]], scores=[.02])
    result = evaluate_quality_predictions([gt], [pred], 2)
    standard = evaluate_predictions([gt], [pred], 2, .1)
    assert result["map"] == standard["map"]
    assert result["ap75"] == standard["map75"] == pytest.approx(100)
    assert result["ap85"] == 0
    assert result["wrong_class_fp"] == 0
    assert result["map"] == pytest.approx(70)


def test_no_detections_zero_ap_and_no_gt_undefined_ap():
    pytest.importorskip("pycocotools")
    result = evaluate_quality_predictions([record([[0, 0, 10, 10]])], [], 2)
    assert result["map"] == result["ap75"] == result["ap85"] == 0
    result = evaluate_quality_predictions([record([])], [], 2)
    assert result["map"] is None


def fake_baseline():
    from torchvision.models.detection.fcos import FCOS
    from torchvision.models.detection._utils import BoxLinearCoder
    model = types.SimpleNamespace(box_coder=BoxLinearCoder(normalize_by_size=True),
                                  score_thresh=.001, topk_candidates=1000,
                                  nms_thresh=.6, detections_per_img=100)
    model.postprocess_detections = types.MethodType(FCOS.postprocess_detections, model)
    return model


def test_trace_exactly_preserves_official_outputs_and_restores_after_exception():
    model = fake_baseline()
    heads = {"cls_logits": [torch.tensor([[[-9., 2.], [-9., 1.]]])],
             "bbox_regression": [torch.tensor([[[.5, .5, .5, .5], [.6, .6, .6, .6]]])],
             "bbox_ctrness": [torch.tensor([[[2.], [1.]]])]}
    anchors = [[torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])]]
    shapes = [(100, 100)]
    target = {"image_id": torch.tensor([1]), "resize_scale": torch.tensor([1., 1.]),
              "pad_offset": torch.tensor([0., 0.]), "original_size": torch.tensor([100, 100])}
    gt = record([[0, 0, 10, 10]])
    original = model.postprocess_detections
    expected = original(heads, anchors, shapes)
    with StageDiagnosticTracer(model) as tracer:
        tracer.begin_batch([target], [gt])
        actual = model.postprocess_detections(heads, anchors, shapes)
        assert all(torch.equal(expected[0][key], actual[0][key]) for key in expected[0])
        summary = tracer.summary([gt])
        assert summary["protocol"]["postprocess_parity"] == "EXACT"
        assert summary["first_zero"]["covered_final"] == 1
    assert model.postprocess_detections == original
    with pytest.raises(RuntimeError, match="purposeful"):
        with StageDiagnosticTracer(model):
            raise RuntimeError("purposeful")
    assert model.postprocess_detections == original


def test_quality_trace_preserves_fp32_sigmoid_and_decode_for_half_inputs():
    from tools.sonar_fcos_quality.model import QualityFCOS
    model = fake_baseline()
    model._sonar_quality_score_mode = "quality_sigmoid"
    model.postprocess_detections = types.MethodType(QualityFCOS.postprocess_detections, model)
    heads = {"cls_logits": [torch.tensor([[[-9., 2.01], [-9., 1.02]]], dtype=torch.float16)],
             "bbox_regression": [torch.tensor([[[.5, .5, .5, .5], [.6, .6, .6, .6]]], dtype=torch.float16)]}
    anchors = [[torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], dtype=torch.float16)]]
    target = {"image_id": torch.tensor([1]), "resize_scale": torch.tensor([1., 1.]),
              "pad_offset": torch.tensor([0., 0.]), "original_size": torch.tensor([100, 100])}
    gt = record([[0, 0, 10, 10]])
    with StageDiagnosticTracer(model) as tracer:
        tracer.begin_batch([target], [gt])
        result = model.postprocess_detections(heads, anchors, [(100, 100)])
        assert result[0]["scores"].dtype == torch.float32
        assert result[0]["boxes"].dtype == torch.float32
        assert tracer.summary([gt])["protocol"]["postprocess_parity"] == "EXACT"


def test_historical_stage_totals_and_wrong_class_replay():
    if not (HISTORY / "gt_stage_rows.json").exists():
        pytest.skip("Frozen local historical diagnosis is unavailable.")
    records = json.loads((HISTORY / "validation_records.json").read_text(encoding="utf-8"))
    rows = json.loads((HISTORY / "gt_stage_rows.json").read_text(encoding="utf-8"))
    preds = json.loads((HISTORY / "predictions_baseline.json").read_text(encoding="utf-8"))
    summary = summarize_stage_rows(rows, records)
    assert summary["images"] == 1517 and summary["ground_truths"] == 2692
    assert summary["raw_no_good_box"] == 448 and summary["nms_first_loss"] == 333
    assert summary["first_zero"] == {"raw": 448, "threshold": 1, "topk": 37,
                                     "nms": 333, "top100": 9, "final": 0, "covered_final": 1864}
    assert wrong_class_diagnostics(records, preds, 10)["wrong_class_fp"] == 577


def test_historical_raw_native_stage_replay():
    raw_files = sorted((HISTORY / "raw").glob("*.npz"))
    if not raw_files:
        pytest.skip("Frozen native raw traces are unavailable.")
    records = {row["image_id"]: row for row in json.loads(
        (HISTORY / "validation_records.json").read_text(encoding="utf-8"))}
    archived = {(row["image_id"], row["gt_index"]): row for row in json.loads(
        (HISTORY / "gt_stage_rows.json").read_text(encoding="utf-8"))}
    for path in raw_files:
        with np.load(path) as raw:
            target = {key: torch.from_numpy(raw[key]) for key in
                      ("resize_scale", "pad_offset", "original_size")}
            raw_boxes = np.concatenate([raw[f"P{level}_boxes_original_native"] for level in range(3, 8)])
            raw_scores = np.concatenate([raw[f"P{level}_scores_native"] for level in range(3, 8)])
            candidates = StageDiagnosticTracer._inverse(torch.from_numpy(raw["candidate_boxes_canvas"]), target)
            nms_keep = raw["nms_keep_baseline"]
            rows = stage_rows_for_image(records[int(path.stem)], raw_boxes, raw_scores, candidates,
                                       raw["candidate_labels"], nms_keep, nms_keep[:100], .001)
            for row in rows:
                expected = archived[(row["image_id"], row["gt_index"])]
                assert row["mechanism_first_zero"] == expected["mechanism_first_zero"]
                for stage, value in row["stages"].items():
                    for key, actual in value.items():
                        assert actual == expected["stages"][stage][key]
