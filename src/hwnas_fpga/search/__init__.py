"""Search algorithms and candidate selection."""

from .searcher import BaseSearcher, RandomSearcher, candidate_selection_score, create_searcher
from .constrained import (
    ConfigurableSearcher,
    SearchConfig,
    run_ablation_study,
    run_early_pruning_comparison,
    run_fused_vs_standard_mbconv,
)
from .pareto import (
    ParetoFrontSelector,
    build_pareto_objectives,
    build_pareto_ranked_rows,
    build_pareto_selection_summary,
    compute_hypervolume,
    compute_pareto_front,
    compute_pareto_front_numpy,
    compute_pareto_ranks,
    pareto_encoding_signature,
    plot_pareto_front,
    write_pareto_selection_artifacts,
)
from .rl_searcher import (
    ActionSpace,
    ArchitectureAction,
    Controller,
    RLSearcher,
    RewardFunction,
)
from .proxyless_searcher import ProxylessSearcher
from .supernet import (
    MobileAnchorSupernet,
    evaluate_candidate as evaluate_supernet_candidate,
    train_supernet,
)

__all__ = [
    "BaseSearcher",
    "RandomSearcher",
    "candidate_selection_score",
    "create_searcher",
    "ConfigurableSearcher",
    "SearchConfig",
    "run_ablation_study",
    "run_early_pruning_comparison",
    "run_fused_vs_standard_mbconv",
    "ParetoFrontSelector",
    "build_pareto_objectives",
    "build_pareto_ranked_rows",
    "build_pareto_selection_summary",
    "compute_hypervolume",
    "compute_pareto_front",
    "compute_pareto_front_numpy",
    "compute_pareto_ranks",
    "pareto_encoding_signature",
    "plot_pareto_front",
    "write_pareto_selection_artifacts",
    "ActionSpace",
    "ArchitectureAction",
    "Controller",
    "RLSearcher",
    "RewardFunction",
    "ProxylessSearcher",
    "MobileAnchorSupernet",
    "evaluate_supernet_candidate",
    "train_supernet",
]
