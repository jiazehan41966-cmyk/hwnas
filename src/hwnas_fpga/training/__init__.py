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
    "evaluate_sonar_robustness",
    "hierarchical_paired_bootstrap",
    "load_architecture_from_artifact",
    "load_best_candidate_artifact",
    "retrain_architecture",
    "protocol_claimability",
    "sha256_file",
    "train_model",
    "train_with_recipe",
    "resolve_sonar_robustness_config",
]
