"""Supernet training, retraining, and evaluation workflows."""

from .recipe import (
    LogitAdjustedCrossEntropy,
    RecipeConfig,
    RecipeResult,
    build_train_criterion,
    build_warmup_cosine_scheduler,
    train_with_recipe,
)
from .protocol_reporting import (
    canonical_sha256,
    hierarchical_paired_bootstrap,
    protocol_claimability,
    sha256_file,
)
from .retrain import (
    evaluate_model,
    load_architecture_from_artifact,
    load_best_candidate_artifact,
    retrain_architecture,
)
from .trainer import Trainer, create_optimizer, train_model
from .robustness import (
    DEFAULT_SONAR_ROBUSTNESS_CONDITIONS,
    apply_sonar_corruption,
    evaluate_sonar_robustness,
    resolve_sonar_robustness_config,
)
from .detection_metrics import box_iou_xyxy, evaluate_detection_predictions
from .detection_coco_eval import (
    COCO_AREA_RANGES,
    COCO_IOU_THRESHOLDS,
    evaluate_detection_predictions_coco,
)
from .detection_trainer import (
    DEFAULT_ANCHOR_SIZES,
    DenseAnchorDetectionLoss,
    DetectionLossConfig,
    decode_detection_predictions,
    evaluate_detection_model,
    train_detection_model,
)

__all__ = [
    "LogitAdjustedCrossEntropy",
    "RecipeConfig",
    "RecipeResult",
    "Trainer",
    "DEFAULT_SONAR_ROBUSTNESS_CONDITIONS",
    "apply_sonar_corruption",
    "build_train_criterion",
    "build_warmup_cosine_scheduler",
    "canonical_sha256",
    "create_optimizer",
    "evaluate_model",
    "evaluate_detection_predictions",
    "evaluate_detection_model",
    "evaluate_sonar_robustness",
    "hierarchical_paired_bootstrap",
    "load_architecture_from_artifact",
    "load_best_candidate_artifact",
    "retrain_architecture",
    "protocol_claimability",
    "sha256_file",
    "train_model",
    "train_detection_model",
    "train_with_recipe",
    "resolve_sonar_robustness_config",
    "box_iou_xyxy",
    "COCO_AREA_RANGES",
    "COCO_IOU_THRESHOLDS",
    "evaluate_detection_predictions_coco",
    "DEFAULT_ANCHOR_SIZES",
    "DenseAnchorDetectionLoss",
    "DetectionLossConfig",
    "decode_detection_predictions",
]
