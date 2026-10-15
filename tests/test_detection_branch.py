from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "hls_lut_builder" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "hls_lut_builder" / "scripts"))

from build_detection_640_feature_cost_table import build_payload
from build_detection_640_measured_lut_from_reports import build_payload as build_measured_payload
from common import build_cases as build_hls_cases, load_config as load_hls_config
from hwnas_fpga.data import YoloDetectionDataset, detection_collate_fn, yolo_normalized_to_xyxy
from hwnas_fpga.data import create_yolo_detection_dataloader
from hwnas_fpga.hardware import (
    FPGACostEstimator,
    FixedMultiscaleDetectionCostConfig,
    detection_feature_shape_table,
    estimate_multiscale_detection_cost,
)
from hwnas_fpga.interfaces import CandidateMetrics, HardwareSpec, SearchCandidate
from hwnas_fpga.models import build_detection_model, build_model
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config
from hwnas_fpga.search import resolve_pareto_objectives
from hwnas_fpga.training import evaluate_detection_predictions
from hwnas_fpga.training import (
    DenseAnchorDetectionLoss,
    DetectionLossConfig,
    decode_detection_predictions,
)


DETECTION_CONFIG = (
    ROOT
    / "configs"
    / "search"
    / "zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml"
)
DETECTION_HLS_CONFIG = (
    ROOT
    / "hls_lut_builder"
    / "configs"
    / "candidate_kernels_detection_multiscale_640.yaml"
)
CLASSIFICATION_CONFIG = (
    ROOT / "configs" / "search" / "zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml"
)


def _load_detection_space():
    config = load_config(str(DETECTION_CONFIG))
    constraints = build_constraints(config)
    space = build_search_space(
        config,
        image_size=int(config["dataset"]["image_size"]),
        input_channels=int(config["dataset"]["input_channels"]),
        num_classes=int(config["dataset"]["num_classes"]),
        constraints=constraints,
    )
    return config, constraints, space


def test_detection_branch_uses_three_multiscale_feature_maps() -> None:
    config, _constraints, space = _load_detection_space()
    architecture = space.baseline_architecture()
    model = build_detection_model(
        architecture,
        num_classes=int(config["dataset"]["num_classes"]),
        feature_indices=tuple(config["detection"]["feature_stage_indices"]),
        neck_channels=int(config["detection"]["fixed_neck"]["out_channels"]),
        anchors_per_location=int(config["detection"]["fixed_head"]["anchors_per_location"]),
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 1, 640, 640))

    assert [tuple(feature.shape[-2:]) for feature in output["backbone_features"]] == [
        (80, 80),
        (40, 40),
        (20, 20),
    ]
    assert [tuple(pred.shape) for pred in output["predictions"]] == [
        (1, 24, 80, 80),
        (1, 24, 40, 40),
        (1, 24, 20, 20),
    ]
    assert not model.neck.training
    assert model.head.prediction_channels == 24


def test_detection_search_space_keeps_classification_head_out_of_backbone() -> None:
    _config, _constraints, detection_space = _load_detection_space()
    assert detection_space.config.num_classes is None
    assert detection_space.config.head_conv_channels is None

    classification_config = load_config(str(CLASSIFICATION_CONFIG))
    classification_space = build_search_space(
        classification_config,
        image_size=int(classification_config["dataset"]["image_size"]),
        input_channels=int(classification_config["dataset"]["input_channels"]),
        num_classes=int(classification_config["dataset"]["num_classes"]),
        constraints=build_constraints(classification_config),
    )
    architecture = classification_space.baseline_architecture()
    model = build_model(
        architecture,
        num_classes=int(classification_config["dataset"]["num_classes"]),
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 224, 224))
    assert tuple(logits.shape) == (1, 8)


def test_detection_cost_table_tracks_640_pyramid_and_fixed_head() -> None:
    config, constraints, space = _load_detection_space()
    architecture = space.baseline_architecture()
    estimator = FPGACostEstimator(
        hardware_spec=HardwareSpec(
            name="ZCU104 test",
            clock_mhz=200,
            max_lut=200_000,
            max_bram=2_000,
            max_dsp=2_000,
        ),
        constraints=constraints,
    )
    cost_config = FixedMultiscaleDetectionCostConfig(
        num_classes=int(config["dataset"]["num_classes"]),
        feature_stage_indices=tuple(config["detection"]["feature_stage_indices"]),
        neck_channels=int(config["detection"]["fixed_neck"]["out_channels"]),
        anchors_per_location=int(config["detection"]["fixed_head"]["anchors_per_location"]),
    )

    backbone_only = estimator.estimate(architecture, space)
    detection_cost = estimate_multiscale_detection_cost(
        architecture,
        space,
        estimator,
        cost_config,
    )
    shapes = detection_feature_shape_table(
        architecture,
        space,
        cost_config.feature_stage_indices,
    )

    assert [(row["feature_h"], row["channels"]) for row in shapes] == [
        (80, 32),
        (40, 96),
        (20, 320),
    ]
    assert detection_cost.params > backbone_only.params
    assert detection_cost.macs > backbone_only.macs
    assert any(layer.op == "detection_fpn_lateral_1x1" for layer in detection_cost.per_layer)
    assert any(layer.op == "detection_head_pred_1x1" for layer in detection_cost.per_layer)


def test_detection_metrics_and_pareto_use_map_precision_recall_f1() -> None:
    summary = evaluate_detection_predictions(
        predictions=[
            {
                "boxes": [[0.0, 0.0, 10.0, 10.0]],
                "scores": [0.99],
                "labels": [0],
            }
        ],
        targets=[
            {
                "boxes": [[0.0, 0.0, 10.0, 10.0]],
                "labels": [0],
            }
        ],
        num_classes=1,
        iou_thresholds=(0.5, 0.75),
    )
    assert summary["map"] == 1.0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["f1"] == 1.0

    objectives, directions = resolve_pareto_objectives(
        {"objectives": ["map", "latency_ms"], "directions": ["max", "min"]},
        selection_metric="map",
    )
    candidate = SearchCandidate(
        "det",
        {},
        CandidateMetrics(map=summary["map"], precision=1.0, recall=1.0, f1=1.0),
    )
    assert objectives == ["map", "latency_ms"]
    assert directions == ["max", "min"]
    assert candidate.metrics.map == 1.0


def test_detection_640_cost_payload_is_analytic_and_fixed_scope() -> None:
    payload = build_payload(DETECTION_CONFIG)
    assert payload["artifact_type"] == "detection_multiscale_640_hardware_cost_table"
    assert payload["input_image_size"] == 640
    assert payload["fixed_neck_head"]["searchable"] is False
    assert payload["claim_boundary"]["hls_route_cost"] == "not_established_by_this_table"
    assert len(payload["feature_map_shapes"]) == 3
    assert len(payload["rows"]) == 14


def test_yolo_detection_dataset_converts_boxes_and_collates(tmp_path) -> None:
    image_root = tmp_path / "train" / "images"
    label_root = tmp_path / "train" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    from PIL import Image

    Image.new("L", (32, 16), color=128).save(image_root / "sample.png")
    (label_root / "sample.txt").write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    dataset = YoloDetectionDataset(tmp_path, split="train", image_size=640, input_channels=1)

    image, target = dataset[0]
    assert tuple(image.shape) == (1, 640, 640)
    assert target["labels"].tolist() == [2]
    assert target["boxes"].tolist() == [[240.0, 160.0, 400.0, 480.0]]

    images, targets = detection_collate_fn([dataset[0]])
    assert tuple(images.shape) == (1, 1, 640, 640)
    assert len(targets) == 1

    boxes, labels = yolo_normalized_to_xyxy([[1, 0.5, 0.5, 1.0, 1.0]], image_size=10)
    assert boxes.tolist() == [[0.0, 0.0, 10.0, 10.0]]
    assert labels.tolist() == [1]


def test_yolo_detection_dataset_applies_horizontal_flip_augmentation(tmp_path) -> None:
    image_root = tmp_path / "train" / "images"
    label_root = tmp_path / "train" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    from PIL import Image

    Image.new("L", (32, 32), color=128).save(image_root / "sample.png")
    (label_root / "sample.txt").write_text("0 0.25 0.5 0.25 0.5\n", encoding="utf-8")
    dataset = YoloDetectionDataset(
        tmp_path,
        split="train",
        image_size=640,
        input_channels=1,
        augment_config={"horizontal_flip_prob": 1.0},
    )

    _image, target = dataset[0]

    assert target["boxes"].tolist() == [[400.0, 160.0, 560.0, 480.0]]


def test_yolo_detection_dataloader_supports_class_balanced_sampler(tmp_path) -> None:
    image_root = tmp_path / "train" / "images"
    label_root = tmp_path / "train" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    from PIL import Image

    for index, label in enumerate([0, 0, 1]):
        Image.new("L", (32, 32), color=128).save(image_root / f"sample{index}.png")
        (label_root / f"sample{index}.txt").write_text(
            f"{label} 0.5 0.5 0.25 0.25\n",
            encoding="utf-8",
        )

    loader = create_yolo_detection_dataloader(
        tmp_path,
        split="train",
        batch_size=1,
        image_size=640,
        input_channels=1,
        class_balanced_sampler=True,
    )

    assert loader.sampler is not None
    assert loader.sampler.weights.tolist() == [0.5, 0.5, 1.0]


def test_detection_measured_lut_config_enumerates_fixed_neck_head_cases() -> None:
    hls_config = load_hls_config(DETECTION_HLS_CONFIG)
    cases = build_hls_cases(hls_config)
    roles = {case.op_spec["detection_role"] for case in cases}

    assert hls_config["toolchain"]["part"] == "xczu7ev-ffvc1156-2-e"
    assert "detection_multiscale_640_zcu104" in hls_config["workspace"]["manifest_path"]
    assert len(cases) == 14
    assert roles == {
        "p3_lateral_1x1",
        "p4_lateral_1x1",
        "p5_lateral_1x1",
        "p4_to_p3_nearest_upsample_add",
        "p5_to_p4_nearest_upsample_add",
        "p3_output_3x3",
        "p4_output_3x3",
        "p5_output_3x3",
        "p3_tower_3x3",
        "p4_tower_3x3",
        "p5_tower_3x3",
        "p3_pred_1x1",
        "p4_pred_1x1",
        "p5_pred_1x1",
    }
    short_dirs = set(hls_config["workspace"]["case_dir_names"].values())
    assert all(case.project_dir.name in short_dirs for case in cases)


def test_detection_measured_lut_payload_blocks_without_reports(tmp_path) -> None:
    payload = build_measured_payload(
        DETECTION_HLS_CONFIG,
        manifest_override=tmp_path / "missing_hls_manifest.yaml",
        lut_json_override=tmp_path / "formal_lut.json",
        status_json_override=tmp_path / "formal_lut_status.json",
    )

    assert payload["status"] == "BLOCKED_MISSING_REPORTS"
    assert payload["summary"]["rows_total"] == 14
    assert payload["summary"]["rows_measured"] == 0
    assert set(payload["summary"]["status_counts"]) == {"missing_manifest"}


def test_dense_anchor_detection_loss_assigns_targets_and_decodes() -> None:
    loss_config = DetectionLossConfig(
        num_classes=3,
        image_size=32,
        strides=(8,),
        anchor_sizes=(((8.0, 8.0), (16.0, 16.0), (24.0, 24.0)),),
    )
    criterion = DenseAnchorDetectionLoss(loss_config)
    predictions = [torch.zeros(1, 24, 4, 4)]
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
        "labels": torch.tensor([1]),
    }

    loss, parts = criterion(predictions, [target])

    assert float(loss.item()) > 0.0
    assert parts["positive_anchors"] >= 1.0

    scored = torch.zeros(1, 24, 4, 4)
    scored[:, 4, 1, 1] = 10.0
    scored[:, 5 + 1, 1, 1] = 10.0
    decoded = decode_detection_predictions(
        [scored],
        config=loss_config,
        score_threshold=0.5,
        pre_nms_topk=10,
        max_detections=5,
    )

    assert len(decoded) == 1
    assert decoded[0]["labels"].tolist() == [1]
    assert decoded[0]["scores"].numel() == 1
    assert decoded[0]["boxes"].shape == (1, 4)


def test_detection_head_starts_with_low_objectness_bias() -> None:
    config, _constraints, space = _load_detection_space()
    model = build_detection_model(
        space.baseline_architecture(),
        num_classes=int(config["dataset"]["num_classes"]),
        feature_indices=tuple(config["detection"]["feature_stage_indices"]),
        neck_channels=int(config["detection"]["fixed_neck"]["out_channels"]),
        anchors_per_location=int(config["detection"]["fixed_head"]["anchors_per_location"]),
    )
    prediction_conv = model.head.towers[0][-1]
    bias = prediction_conv.bias.view(
        model.head.anchors_per_location,
        model.head.outputs_per_anchor,
    )

    assert torch.sigmoid(bias[:, 4]).mean().item() < 0.02


def test_detection_loss_accepts_class_weights() -> None:
    loss_config = DetectionLossConfig(
        num_classes=3,
        image_size=32,
        strides=(8,),
        anchor_sizes=(((8.0, 8.0), (16.0, 16.0), (24.0, 24.0)),),
        class_loss_weights=(0.5, 2.5, 1.3),
    )
    criterion = DenseAnchorDetectionLoss(loss_config)
    predictions = [torch.zeros(1, 24, 4, 4)]
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
        "labels": torch.tensor([1]),
    }

    loss, parts = criterion(predictions, [target])

    assert torch.allclose(
        criterion.class_loss_weight_tensor,
        torch.tensor([0.5, 2.5, 1.3]),
    )
    assert float(loss.item()) > 0.0
    assert parts["positive_anchors"] >= 1.0


def test_detection_loss_accepts_ciou_box_regression_type() -> None:
    config = {
        "dataset": {"num_classes": 1, "image_size": 32},
        "detection": {
            "input_size": 32,
            "feature_strides": [8],
            "anchor_sizes": [[[8.0, 8.0]]],
        },
        "training": {
            "detection_loss": {
                "box_iou_loss_type": "ciou",
                "box_iou_weight": 1.0,
            }
        },
    }
    loss_config = DetectionLossConfig.from_config(config)
    criterion = DenseAnchorDetectionLoss(loss_config)
    predictions = [torch.zeros(1, 6, 4, 4)]
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
        "labels": torch.tensor([0]),
    }

    loss, parts = criterion(predictions, [target])

    assert loss_config.box_iou_loss_type == "ciou"
    assert float(loss.item()) > 0.0
    assert parts["positive_anchors"] >= 1.0


def test_detection_loss_rejects_unknown_box_regression_type() -> None:
    loss_config = DetectionLossConfig(
        num_classes=1,
        image_size=32,
        strides=(8,),
        anchor_sizes=(((8.0, 8.0),),),
        box_iou_loss_type="made_up",
    )
    criterion = DenseAnchorDetectionLoss(loss_config)
    predictions = [torch.zeros(1, 6, 4, 4)]
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
        "labels": torch.tensor([0]),
    }

    with pytest.raises(ValueError, match="unsupported box_iou_loss_type"):
        criterion(predictions, [target])
