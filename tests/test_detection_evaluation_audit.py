from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hwnas_fpga.data import (
    YoloDetectionDataset,
    network_boxes_to_original,
    yolo_normalized_to_original_xyxy,
    yolo_normalized_to_xyxy,
)
from hwnas_fpga.training import evaluate_detection_predictions
from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco
from hwnas_fpga.training.detection_trainer import DetectionLossConfig

from audit_detection_evaluation import (
    audit_grid_alignment,
    audit_metric_definition,
    audit_synthetic_closed_loop,
)
from diagnose_detection_errors import classify_record, miss_reason


def test_orig_size_records_source_hw_not_network_size(tmp_path) -> None:
    image_root = tmp_path / "train" / "images"
    label_root = tmp_path / "train" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    from PIL import Image

    Image.new("L", (32, 16), color=128).save(image_root / "sample.png")
    (label_root / "sample.txt").write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    _image, target = YoloDetectionDataset(tmp_path, split="train", image_size=640, input_channels=1)[0]
    assert target["orig_size"].tolist() == [16, 32]
    assert target["input_size"].tolist() == [640, 640]


def test_original_roundtrip_uses_independent_axes() -> None:
    rows = [[0, 0.25, 0.75, 0.10, 0.20]]
    network, labels = yolo_normalized_to_xyxy(rows, image_size=640)
    original, _ = yolo_normalized_to_original_xyxy(rows, orig_width=32, orig_height=16)
    restored = network_boxes_to_original(network, image_size=640, orig_width=32, orig_height=16)
    assert labels.tolist() == [0]
    assert torch.allclose(restored, original, atol=1e-5)
    assert torch.allclose(original, torch.tensor([[6.4, 10.4, 9.6, 13.6]]), atol=1e-5)


def test_project_map_is_mean_of_ap50_ap75_not_coco() -> None:
    predictions = [
        {
            "boxes": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
            "labels": [0, 1],
            "scores": [0.9, 0.8],
        }
    ]
    targets = [{"boxes": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]], "labels": [0, 1]}]
    project = evaluate_detection_predictions(predictions, targets, num_classes=2)
    coco = evaluate_detection_predictions_coco(predictions, targets, num_classes=2)
    assert project["map_definition"]["is_standard_coco_map_50_95"] is False
    assert abs(project["map"] - (project["map50"] + project["map75"]) / 2.0) < 1e-12
    assert coco["metric_definition"]["is_standard_coco_map_50_95"] is True
    assert project["map"] == 1.0
    assert coco["coco_ap"] == 1.0


def test_synthetic_and_grid_alignment_close() -> None:
    config = DetectionLossConfig(num_classes=2, image_size=640)
    synthetic = audit_synthetic_closed_loop(640, config)
    grid = audit_grid_alignment(config)
    assert synthetic["dataset_load_max_abs_error_px"] < 1e-3
    assert synthetic["original_restore_max_abs_error_px"] < 1e-3
    assert synthetic["assigner_to_decoder_closed_loop"]["min_recovered_iou"] > 0.95
    assert synthetic["metric_sanity"]["identity_map"] == 1.0
    assert synthetic["metric_sanity"]["xy_swapped_map"] < synthetic["metric_sanity"]["identity_map"]
    assert synthetic["metric_sanity"]["xy_swapped_map"] < 0.25
    assert grid["max_corner_abs_error_px"] < 1e-3
    assert grid["max_interior_center_abs_error_px"] < 1e-3


def test_metric_definition_audit_separates_coco() -> None:
    payload = audit_metric_definition(DetectionLossConfig(num_classes=2, image_size=640))
    assert payload["project_metric"]["map_equals_mean_of_two_thresholds"] is True
    assert payload["project_metric"]["definition"]["is_standard_coco_map_50_95"] is False
    assert payload["standard_coco_supplement"]["definition"]["is_standard_coco_map_50_95"] is True


def test_miss_reasons_split_candidate_score_and_localization() -> None:
    no_box = {
        "prediction": {"boxes": [[80.0, 80.0, 90.0, 90.0]], "labels": [0], "scores": [0.9]},
        "target": {"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [0]},
    }
    low_score = {
        "prediction": {"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [0], "scores": [0.2]},
        "target": {"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [0]},
    }
    localized = {
        "prediction": {"boxes": [[5.0, 0.0, 15.0, 10.0]], "labels": [0], "scores": [0.8]},
        "target": {"boxes": [[0.0, 0.0, 10.0, 10.0]], "labels": [0]},
    }
    assert miss_reason(classify_record(no_box, 0.3)["targets"][0]) == "no_candidate"
    assert miss_reason(classify_record(low_score, 0.3)["targets"][0]) == "score_too_low"
    assert miss_reason(classify_record(localized, 0.3)["targets"][0]) == "localization"
