#!/usr/bin/env python3
"""HW-NAS 搜索入口脚本."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate
from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    create_data_pipeline,
    load_anchor_profile_from_pool,
    load_config,
    pick,
)
from hwnas_fpga.research_gates import require_stage3_search_gate
from hwnas_fpga.search import (
    ParetoFrontSelector,
    SearchEfficiencyMonitor,
    merge_search_efficiency,
    build_pareto_representative_roles,
    build_pareto_selection_summary,
    candidate_selection_score,
    compute_hypervolume,
    compute_pareto_front,
    compute_pareto_ranks,
    create_searcher,
    resolve_pareto_objectives,
    write_pareto_selection_artifacts,
)
from hwnas_fpga.search_space import list_family_profiles


def _format_optional(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def _candidate_to_dict(candidate):
    return {
        "arch_id": candidate.arch_id,
        "encoding": candidate.encoding,
        "metrics": candidate.metrics.__dict__,
    }


def _candidate_from_dict(payload: dict | None) -> SearchCandidate | None:
    if payload is None:
        return None
    return SearchCandidate(
        arch_id=payload["arch_id"],
        encoding=payload["encoding"],
        metrics=CandidateMetrics(**payload.get("metrics", {})),
    )


def _resolve_family_profile(args: argparse.Namespace, config: dict) -> tuple[str | None, str | None]:
    search_space_cfg = config.setdefault("search_space", {})

    if args.family_profile is not None:
        search_space_cfg["family_profile"] = args.family_profile
        return str(args.family_profile), "cli"

    if args.backbone_pool is not None and search_space_cfg.get("family_profile") is None:
        resolved_profile = load_anchor_profile_from_pool(args.backbone_pool, args.anchor_role)
        if resolved_profile is not None:
            search_space_cfg["family_profile"] = resolved_profile
            return resolved_profile, "pool"

    existing_profile = search_space_cfg.get("family_profile")
    if existing_profile is not None:
        return str(existing_profile), "config"
    return None, None


def _restore_rl_resume_state(searcher, tracker: ExperimentTracker, device: str) -> int:
    candidates_path = tracker.results_dir / "candidates.jsonl"
    search_state_path = tracker.checkpoints_dir / "search_state.json"
    controller_latest_path = tracker.checkpoints_dir / "controller_latest.pt"
    controller_best_path = tracker.checkpoints_dir / "controller_best.pt"
    controller_reward_best_path = tracker.checkpoints_dir / "controller_reward_best.pt"

    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing resume candidates log: {candidates_path}")
    if not search_state_path.exists():
        raise FileNotFoundError(f"Missing resume search state: {search_state_path}")
    if not controller_latest_path.exists():
        raise FileNotFoundError(f"Missing resume controller checkpoint: {controller_latest_path}")

    records = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    search_state = json.loads(search_state_path.read_text(encoding="utf-8"))
    latest_ckpt = torch.load(controller_latest_path, map_location=device)
    best_ckpt = (
        torch.load(controller_best_path, map_location=device)
        if controller_best_path.exists()
        else {}
    )
    reward_best_ckpt = (
        torch.load(controller_reward_best_path, map_location=device)
        if controller_reward_best_path.exists()
        else {}
    )

    searcher.evaluated_candidates.clear()
    searcher.feasible_candidates.clear()
    searcher.infeasible_candidates.clear()
    for record in records:
        candidate = _candidate_from_dict(record.get("candidate"))
        if candidate is None:
            continue
        searcher.evaluated_candidates.append(candidate)
        if record.get("feasible", True):
            searcher.feasible_candidates.append(candidate)
        else:
            searcher.infeasible_candidates.append(candidate)

    searcher.best_candidate = _candidate_from_dict(search_state.get("best_candidate"))
    searcher.best_reward = float(
        reward_best_ckpt.get(
            "best_reward",
            best_ckpt.get("best_reward", float("-inf")),
        )
    )
    reward_best_arch_id = (
        reward_best_ckpt.get("best_reward_candidate") or {}
    ).get("arch_id")
    searcher.best_reward_candidate = next(
        (
            candidate
            for candidate in searcher.evaluated_candidates
            if candidate.arch_id == reward_best_arch_id
        ),
        None,
    )
    searcher.best_selection_score = float(
        best_ckpt.get(
            "best_selection_score",
            float("-inf"),
        )
    )
    if (
        searcher.best_selection_score == float("-inf")
        and searcher.best_candidate is not None
    ):
        selection_metric = str(
            search_state.get("extra", {}).get("selection_metric", "macro_f1")
        )
        searcher.best_selection_score = candidate_selection_score(
            searcher.best_candidate,
            selection_metric,
        )
    searcher.baseline = float(latest_ckpt.get("baseline", searcher.baseline))
    searcher.controller.load_state_dict(latest_ckpt["controller_state_dict"])
    searcher.controller_optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])

    resumed_episode = int(latest_ckpt.get("episode", len(records) - 1)) + 1
    return resumed_episode


def _restore_aging_resume_state(searcher, tracker: ExperimentTracker) -> int:
    candidates_path = tracker.results_dir / "candidates.jsonl"
    checkpoint_path = tracker.checkpoints_dir / "aging_latest.pt"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing resume candidates log: {candidates_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing aging evolution checkpoint: {checkpoint_path}")
    records = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return searcher.restore_state(records, checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW-NAS FPGA Sonar Search")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--dataset", type=str, default=None, choices=("dummy", "nksid"))
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--fold", type=int, default=None, help="NKSID k-fold split index")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--timeout-minutes", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lut", type=str, default=None, help="Path to LUT json file for zero-latency evaluation")
    parser.add_argument(
        "--family-profile",
        type=str,
        default=None,
        choices=tuple(sorted(list_family_profiles().keys())),
    )
    parser.add_argument(
        "--backbone-pool",
        type=str,
        default=None,
        help=(
            "Path to selected_backbone_pool.json produced by run_backbone_baseline. "
            "If provided, resolves --anchor-role to a family_profile automatically."
        ),
    )
    parser.add_argument(
        "--anchor-role",
        type=str,
        default="search_anchor",
        choices=("search_anchor", "accuracy_anchor", "lightweight_anchor"),
        help="Which anchor role to use from the backbone pool (default: search_anchor).",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default=None,
        choices=("random", "rl", "proxyless", "aging_evolution"),
    )
    parser.add_argument("--controller-hidden", type=int, default=None)
    parser.add_argument("--controller-lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Run artifact root directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    family_profile, family_profile_source = _resolve_family_profile(args, config)

    if args.lut is not None:
        config.setdefault("hardware", {})
        config["hardware"]["lut_path"] = args.lut

    dataset_cfg = config.get("dataset", {})
    project_cfg = config.get("project", {})
    training_cfg = config.get("training", {})
    search_cfg = config.get("search", {})

    seed = pick(args.seed, project_cfg.get("seed"), 42)
    torch.manual_seed(seed)

    dataset_name = pick(args.dataset, dataset_cfg.get("name"), "dummy")
    if dataset_name not in {"dummy", "nksid"}:
        dataset_name = "dummy"
    search_method = pick(args.search_method, search_cfg.get("method"), "random")
    if search_method == "aging":
        search_method = "aging_evolution"
    if search_method not in {"random", "rl", "proxyless", "aging_evolution"}:
        raise ValueError(f"Unsupported search method: {search_method}")
    config.setdefault("search", {})["method"] = search_method
    search_cfg = config["search"]
    # Gate the final resolved request, including any CLI method override.  The
    # approval cannot be checked against the source YAML and then silently
    # replaced with a different method afterward.
    stage3_gate = require_stage3_search_gate(
        Path(__file__).resolve().parent,
        dataset_name=dataset_name,
        config=config,
    )

    if search_method == "proxyless":
        _default_device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        device = pick(args.device, None, _default_device)
        if not args.config:
            raise ValueError("--config is required for official ProxylessNAS search")
        from hwnas_fpga.search.official_proxyless_runner import run_official_proxyless_search

        print("[run_search] proxyless method delegates to reference/ProxylessNAS clone")
        run_official_proxyless_search(
            config_path=args.config,
            device=device,
            resume=args.resume,
        )
        return

    _default_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = pick(args.device, None, _default_device)
    image_size = pick(args.image_size, dataset_cfg.get("image_size"), 224)
    input_channels = dataset_cfg.get("input_channels", 1)
    batch_size = pick(args.batch_size, training_cfg.get("batch_size"), 32)
    num_workers = pick(args.num_workers, dataset_cfg.get("num_workers"), 0)
    fold = pick(args.fold, dataset_cfg.get("fold"), 0)
    data_dir = pick(args.data_dir, dataset_cfg.get("data_dir"), None)
    use_kfold = bool(dataset_cfg.get("use_kfold", True))
    valid_size = dataset_cfg.get("valid_size")
    split_seed = int(dataset_cfg.get("split_seed", seed))
    image_error_policy = str(dataset_cfg.get("image_error_policy", "raise"))
    split_mode = str(dataset_cfg.get("split_mode", "legacy_kfold"))
    inner_val_fraction = float(dataset_cfg.get("inner_val_fraction", 0.15))
    num_candidates = pick(args.num_candidates, search_cfg.get("num_candidates"), 20)
    num_episodes = pick(args.episodes, search_cfg.get("episodes"), 20)
    train_epochs = pick(args.train_epochs, search_cfg.get("eval_epochs"), 3)
    eval_early_stopping_patience = search_cfg.get("eval_early_stopping_patience", 2)
    timeout_minutes = pick(args.timeout_minutes, search_cfg.get("timeout_minutes"), None)
    num_classes = pick(args.num_classes, dataset_cfg.get("num_classes"), None)
    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)
    controller_hidden = pick(args.controller_hidden, search_cfg.get("controller_hidden"), 64)
    controller_lr = pick(args.controller_lr, search_cfg.get("controller_lr"), 0.01)
    controller_temperature = float(search_cfg.get("controller_temperature", 1.0))
    entropy_coef = float(search_cfg.get("controller_entropy_coef", search_cfg.get("entropy_coef", 0.0)))
    exploration_epsilon_start = float(search_cfg.get("exploration_epsilon_start", 0.0))
    exploration_epsilon_end = search_cfg.get("exploration_epsilon_end")
    if exploration_epsilon_end is not None:
        exploration_epsilon_end = float(exploration_epsilon_end)
    exploration_epsilon_decay_episodes = int(search_cfg.get("exploration_epsilon_decay_episodes", 0))
    exploration_bonus = float(search_cfg.get("exploration_bonus", 0.0))

    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    # Persist the resolved protocol, not only the source YAML.  This is
    # essential for matched-seed/matched-budget comparisons launched through
    # CLI overrides.
    resolved_project_cfg = config.setdefault("project", {})
    resolved_project_cfg["seed"] = int(seed)
    resolved_project_cfg["output_dir"] = str(output_root)
    if run_name is not None:
        resolved_project_cfg["run_name"] = str(run_name)
    resolved_dataset_cfg = config.setdefault("dataset", {})
    resolved_dataset_cfg.update(
        {
            "name": dataset_name,
            "data_dir": data_dir,
            "image_size": int(image_size),
            "num_workers": int(num_workers),
            "fold": int(fold),
            "split_mode": split_mode,
            "inner_val_fraction": inner_val_fraction,
        }
    )
    if num_classes is not None:
        resolved_dataset_cfg["num_classes"] = int(num_classes)
    config.setdefault("training", {})["batch_size"] = int(batch_size)
    resolved_search_cfg = config.setdefault("search", {})
    resolved_search_cfg.update(
        {
            "method": search_method,
            "num_candidates": int(num_candidates),
            "episodes": int(num_episodes),
            "eval_epochs": int(train_epochs),
            "timeout_minutes": timeout_minutes,
        }
    )
    if family_profile is not None:
        config.setdefault("search_space", {})["family_profile"] = family_profile

    tracker = ExperimentTracker(
        output_root=output_root,
        project_name=project_cfg.get("name", "hwnas"),
        script_name="run_search",
        search_method=search_method,
        dataset_name=dataset_name,
        run_name=run_name,
        resume=args.resume,
    )
    tracker.save_config(config, vars(args))

    try:
        with tracker.capture_console():
            print(f"Run dir: {tracker.root_dir}")
            print(f"Using device: {device}")
            print(f"Dataset: {dataset_name}")
            print(f"Search method: {search_method}")
            print(
                "Stage-3 gate: "
                f"{stage3_gate['status']} "
                f"(space={stage3_gate['semantic_safe_search_space_cardinality']:,})"
            )
            print(f"Family profile: {family_profile or 'default'}")
            if family_profile_source == "pool":
                print(
                    "[backbone-pool] "
                    f"anchor_role={args.anchor_role!r} -> family_profile={family_profile!r}"
                )
            if data_dir:
                print(f"Data dir: {data_dir}")

            constraints = build_constraints(config)
            hardware_spec = build_hardware_spec(config)

            print("\n=== Creating Search Space ===")
            search_space = build_search_space(
                config,
                image_size=image_size,
                input_channels=input_channels,
                num_classes=num_classes,
                constraints=constraints,
            )
            print(f"Search space created with {search_space.config.stage_count} stages")
            tracker.write_search_space_summary(
                {
                    "input_channels": search_space.config.input_channels,
                    "image_size": search_space.config.image_size,
                    "stem_channels": search_space.config.stem_channels,
                    "stem_stride": search_space.config.stem_stride,
                    "post_stem_downsample_stride": search_space.config.post_stem_downsample_stride,
                    "stage_strides": list(search_space.config.stage_strides),
                    "stage_base_channels": (
                        None
                        if search_space.config.stage_base_channels is None
                        else list(search_space.config.stage_base_channels)
                    ),
                    "width_multipliers": (
                        None
                        if search_space.config.width_multipliers is None
                        else list(search_space.config.width_multipliers)
                    ),
                    "stage_channel_choices": (
                        None
                        if search_space.config.stage_channel_choices is None
                        else [list(group) for group in search_space.config.stage_channel_choices]
                    ),
                    "stage_depth_choices": (
                        None
                        if search_space.config.stage_depth_choices is None
                        else [list(group) for group in search_space.config.stage_depth_choices]
                    ),
                    "channel_choices": list(search_space.config.channel_choices),
                    "depth_choices": list(search_space.config.depth_choices),
                    "kernel_choices": list(search_space.config.kernel_choices),
                    "expand_choices": list(search_space.config.expand_choices),
                    "op_choices": list(search_space.config.op_choices),
                    "head_conv_channels": search_space.config.head_conv_channels,
                    "family_profile": search_space.config.family_profile,
                }
            )

            print("\n=== Creating Hardware Estimator ===")
            estimator = build_cost_estimator(
                config,
                hardware_spec=hardware_spec,
                constraints=constraints,
            )
            print(
                "Hardware profile: "
                f"{hardware_spec.name}, clock={hardware_spec.clock_mhz}MHz, "
                f"LUT={hardware_spec.max_lut}, DSP={hardware_spec.max_dsp}, "
                f"BRAM={hardware_spec.max_bram}, BW={hardware_spec.memory_bandwidth_gbps}GB/s, "
                f"Off-chip={hardware_spec.offchip_mem_mb}MB"
            )
            if estimator.lut_query_engine is not None:
                print("LUT query engine enabled")
            search_space, policy_summary = apply_operator_policies_to_search_space(
                search_space,
                estimator.operator_policies,
            )
            tracker.write_operator_policy_summary(policy_summary)
            if policy_summary["removed_ops"]:
                print("Operator policy filter removed:")
                for item in policy_summary["removed_ops"]:
                    print(
                        f"  - {item['op']} "
                        f"(status={item.get('deployment_status')}, reason={item.get('reason') or 'n/a'})"
                    )
            if policy_summary["conditional_ops"]:
                print("Conditional hardware-cost operators:")
                for item in policy_summary["conditional_ops"]:
                    print(
                        f"  - {item['op']}: "
                        f"latency_rule={item.get('latency_rule')}, "
                        f"post_route_fmax={item.get('achieved_post_route_fmax_mhz')}MHz, "
                        f"deployable_at_200mhz={item.get('deployable_at_200mhz')}"
                    )
            print(f"Search ops after policy filter: {search_space.config.op_choices}")

            print("\n=== Hardware-Driven Pruning ===")
            search_space = search_space.pre_prune(estimator)
            tracker.write_pruned_search_space_summary(
                {
                    "input_channels": search_space.config.input_channels,
                    "image_size": search_space.config.image_size,
                    "stem_channels": search_space.config.stem_channels,
                    "stem_stride": search_space.config.stem_stride,
                    "post_stem_downsample_stride": search_space.config.post_stem_downsample_stride,
                    "stage_strides": list(search_space.config.stage_strides),
                    "stage_base_channels": (
                        None
                        if search_space.config.stage_base_channels is None
                        else list(search_space.config.stage_base_channels)
                    ),
                    "width_multipliers": (
                        None
                        if search_space.config.width_multipliers is None
                        else list(search_space.config.width_multipliers)
                    ),
                    "stage_channel_choices": (
                        None
                        if search_space.config.stage_channel_choices is None
                        else [list(group) for group in search_space.config.stage_channel_choices]
                    ),
                    "stage_depth_choices": (
                        None
                        if search_space.config.stage_depth_choices is None
                        else [list(group) for group in search_space.config.stage_depth_choices]
                    ),
                    "channel_choices": list(search_space.config.channel_choices),
                    "depth_choices": list(search_space.config.depth_choices),
                    "kernel_choices": list(search_space.config.kernel_choices),
                    "expand_choices": list(search_space.config.expand_choices),
                    "op_choices": list(search_space.config.op_choices),
                    "head_conv_channels": search_space.config.head_conv_channels,
                    "family_profile": search_space.config.family_profile,
                    "pruned": True,
                }
            )

            print("\n=== Creating Data Loaders ===")
            train_loader, val_loader, class_weights, resolved_num_classes = create_data_pipeline(
                dataset_name=dataset_name,
                data_dir=data_dir,
                batch_size=batch_size,
                image_size=image_size,
                num_classes=num_classes,
                input_channels=input_channels,
                fold=fold,
                num_workers=num_workers,
                device=device,
                use_kfold=use_kfold,
                valid_size=valid_size,
                split_seed=split_seed,
                image_error_policy=image_error_policy,
                split_mode=split_mode,
                inner_val_fraction=inner_val_fraction,
            )
            print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
            print(f"Num classes: {resolved_num_classes}")
            if dataset_name == "nksid":
                displayed_split_mode = (
                    split_mode
                    if split_mode == "frozen_inner"
                    else "kfold" if use_kfold and valid_size is None else "random_valid_split"
                )
                print(
                    f"NKSID split mode: {displayed_split_mode}, "
                    f"fold={fold}, valid_size={valid_size}, split_seed={split_seed}"
                )
            tracker.write_dataset_summary(
                {
                    "dataset": dataset_name,
                    "data_dir": data_dir,
                    "fold": fold,
                    "use_kfold": use_kfold,
                    "valid_size": valid_size,
                    "split_seed": split_seed,
                    "split_mode": split_mode,
                    "inner_val_fraction": inner_val_fraction,
                    "outer_validation_consumed": False,
                    "image_error_policy": image_error_policy,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "num_classes": resolved_num_classes,
                    "class_weights": None if class_weights is None else class_weights.tolist(),
                }
            )

            print("\n=== Creating Searcher ===")
            objective_weights = search_cfg.get("objective_weights", {})
            reward_cfg = dict(search_cfg.get("reward_cfg") or {})
            selection_metric = str(search_cfg.get("selection_metric", "macro_f1"))
            proxyless_cfg = search_cfg.get("proxyless", {})
            aging_cfg = dict(search_cfg.get("aging") or {})
            pareto_cfg = dict(search_cfg.get("pareto") or {})
            robustness_cfg = dict(search_cfg.get("robustness") or {})
            searcher = create_searcher(
                search_space=search_space,
                cost_estimator=estimator,
                constraints=constraints,
                method=search_method,
                seed=seed,
                controller_hidden_dim=controller_hidden,
                controller_lr=controller_lr,
                train_epochs_per_arch=train_epochs,
                eval_early_stopping_patience=eval_early_stopping_patience,
                device=device,
                reward_weights=objective_weights,
                reward_cfg=reward_cfg,
                proxyless_cfg=proxyless_cfg,
                aging_cfg=aging_cfg,
                pareto_cfg=pareto_cfg,
                robustness_cfg=robustness_cfg,
                selection_metric=selection_metric,
                controller_temperature=controller_temperature,
                entropy_coef=entropy_coef,
                exploration_epsilon_start=exploration_epsilon_start,
                exploration_epsilon_end=exploration_epsilon_end,
                exploration_epsilon_decay_episodes=exploration_epsilon_decay_episodes,
                exploration_bonus=exploration_bonus,
            )
            print(f"Searcher created (method={search_method}, seed={seed}, selection_metric={selection_metric})")
            if search_method == "rl":
                print(
                    "RL exploration config: "
                    f"temperature={controller_temperature}, "
                    f"entropy_coef={entropy_coef}, "
                    f"epsilon_start={exploration_epsilon_start}, "
                    f"epsilon_end={exploration_epsilon_end}, "
                    f"epsilon_decay_episodes={exploration_epsilon_decay_episodes}, "
                    f"exploration_bonus={exploration_bonus}"
                )
            if search_method == "rl" and reward_cfg:
                print(
                    "RL reward config: "
                    f"constraint_penalty={reward_cfg.get('constraint_penalty', 10.0)}, "
                    f"mode={reward_cfg.get('infeasible_penalty_mode', 'fixed')}, "
                    f"base={reward_cfg.get('infeasible_base_penalty', 1.0)}, "
                    f"scale={reward_cfg.get('infeasible_penalty_scale', 5.0)}"
                )
            if search_method == "aging_evolution":
                print(
                    "Aging evolution config: "
                    f"population_size={searcher.population_size}, "
                    f"sample_size={searcher.sample_size}, "
                    f"random_injection_probability={searcher.random_injection_probability}, "
                    f"crossover_probability={searcher.crossover_probability}, "
                    f"survivor_selection={searcher.survivor_selection}, "
                    f"max_mutation_attempts={searcher.max_mutation_attempts}, "
                    f"objectives={searcher.objectives}"
                )
            resume_episode = 0
            resume_iteration = 0
            if args.resume:
                if search_method == "rl":
                    resume_episode = _restore_rl_resume_state(searcher, tracker, device)
                    print(
                        "Resume state restored: "
                        f"{len(searcher.evaluated_candidates)} evaluated, "
                        f"next episode={resume_episode}"
                    )
                elif search_method == "aging_evolution":
                    resume_iteration = _restore_aging_resume_state(searcher, tracker)
                    print(
                        "Aging evolution state restored: "
                        f"{len(searcher.evaluated_candidates)} evaluated, "
                        f"population={len(searcher.population)}, "
                        f"next iteration={resume_iteration}"
                    )
                else:
                    raise ValueError(
                        "Resume is currently supported for RL and aging_evolution search"
                    )

            print("\n=== Testing Baseline Architecture ===")
            baseline_arch = search_space.baseline_architecture()
            baseline_cost = estimator.estimate(baseline_arch, search_space)
            print("Baseline architecture:")
            print(f"  Params: {baseline_cost.params:,}")
            print(f"  MACs: {baseline_cost.macs:,}")
            print(f"  Latency: {baseline_cost.latency_ms:.2f}ms")
            print(f"  Total DSP: {baseline_cost.resource_dsp}")
            print(f"  Total BRAM: {baseline_cost.resource_bram}")
            print(f"  Total LUT: {baseline_cost.resource_lut}")
            print(f"  Peak DSP: {baseline_cost.peak_dsp}")
            print(f"  Peak BRAM: {baseline_cost.peak_bram}")
            print(f"  Peak LUT: {baseline_cost.peak_lut}")
            print(f"  Power: {baseline_cost.power_w:.2f}W")
            print(f"  Memory BW: {baseline_cost.memory_bandwidth_gbps:.3f}GB/s")
            print(f"  Off-chip Mem: {baseline_cost.offchip_mem_mb:.3f}MB")
            print(f"  Violations: {baseline_cost.violations}")
            conditional_layers = [
                layer for layer in baseline_cost.per_layer if layer.effective_clock_mhz is not None
            ]
            if conditional_layers:
                print("  Conditional operator latency sources:")
                for layer in conditional_layers:
                    print(
                        f"    - {layer.op}: latency={layer.resolved_latency_ms(hardware_spec.clock_mhz):.4f}ms, "
                        f"effective_clock={layer.effective_clock_mhz:.2f}MHz, cycles={layer.latency_cycles}"
                    )
            tracker.write_baseline(
                architecture=baseline_arch.to_dict(),
                cost_estimate=baseline_cost,
            )
            estimator.reset_lut_stats()

            print("\n=== Starting Search ===")
            previous_efficiency = tracker.read_search_efficiency() if args.resume else None
            evaluated_before_segment = len(searcher.evaluated_candidates)
            feasible_before_segment = len(searcher.feasible_candidates)
            search_monitor = SearchEfficiencyMonitor(device)
            try:
                with search_monitor:
                    if search_method == "rl":
                        print(f"Episodes: {num_episodes}")
                        best_candidate = searcher.search(
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_classes=resolved_num_classes,
                            num_episodes=num_episodes,
                            start_episode=resume_episode,
                            timeout_minutes=timeout_minutes,
                            device=device,
                            verbose=True,
                            class_weights=class_weights,
                            artifact_tracker=tracker,
                        )
                    elif search_method == "aging_evolution":
                        print(f"Number of candidates to evaluate: {num_candidates}")
                        if timeout_minutes:
                            print(f"Search timeout: {timeout_minutes} minutes")
                        best_candidate = searcher.search(
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_classes=resolved_num_classes,
                            num_candidates=num_candidates,
                            train_epochs=train_epochs,
                            device=device,
                            verbose=True,
                            class_weights=class_weights,
                            artifact_tracker=tracker,
                            start_iteration=resume_iteration,
                            timeout_minutes=timeout_minutes,
                        )
                    elif timeout_minutes:
                        print(f"Search timeout: {timeout_minutes} minutes")
                        best_candidate = searcher.search_with_timeout(
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_classes=resolved_num_classes,
                            timeout_minutes=timeout_minutes,
                            train_epochs=train_epochs,
                            device=device,
                            verbose=True,
                            class_weights=class_weights,
                            artifact_tracker=tracker,
                        )
                    else:
                        print(f"Number of candidates to evaluate: {num_candidates}")
                        best_candidate = searcher.search(
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_classes=resolved_num_classes,
                            num_candidates=num_candidates,
                            train_epochs=train_epochs,
                            device=device,
                            verbose=True,
                            class_weights=class_weights,
                            artifact_tracker=tracker,
                        )
            finally:
                search_efficiency_segment = search_monitor.summary(
                    candidate_count=(
                        len(searcher.evaluated_candidates) - evaluated_before_segment
                    ),
                    feasible_count=(
                        len(searcher.feasible_candidates) - feasible_before_segment
                    ),
                    search_method=search_method,
                )
                search_efficiency = merge_search_efficiency(
                    previous_efficiency,
                    search_efficiency_segment,
                )
                tracker.write_search_efficiency(search_efficiency)
                print(
                    "Search efficiency: "
                    f"wall={search_efficiency['wall_clock_seconds']:.3f}s, "
                    f"gpu_reserved_hours={search_efficiency['gpu_reserved_hours']:.6f}, "
                    f"gpu_event_seconds={search_efficiency['gpu_event_seconds']}, "
                    f"segments={search_efficiency['segment_count']}"
                )

            pareto_topk = int(pareto_cfg.get("topk", 5))
            pareto_method = pareto_cfg.get("selection_method", "rank")
            pareto_objectives, pareto_directions = resolve_pareto_objectives(
                pareto_cfg,
                objective_weights,
                constraints,
                selection_metric=selection_metric,
            )
            pareto_front = compute_pareto_front(
                searcher.feasible_candidates,
                objectives=pareto_objectives,
                directions=pareto_directions,
            )
            selector = ParetoFrontSelector(
                objectives=pareto_objectives,
                directions=pareto_directions,
                selection_method=pareto_method,
            )
            pareto = selector.select(searcher.feasible_candidates, k=pareto_topk)
            pareto_ranks = compute_pareto_ranks(
                searcher.feasible_candidates,
                objectives=pareto_objectives,
                directions=pareto_directions,
            )
            hypervolume = 0.0
            if pareto_front:
                reference_point = selector._default_reference_point(pareto_front)
                hypervolume = compute_hypervolume(
                    pareto_front,
                    reference_point=reference_point,
                    objectives=pareto_objectives,
                    directions=pareto_directions,
                )
            pareto_summary = build_pareto_selection_summary(
                candidates=searcher.feasible_candidates,
                pareto_front=pareto_front,
                selected_candidates=pareto,
                ranks=pareto_ranks,
                objectives=pareto_objectives,
                directions=pareto_directions,
                selection_method=pareto_method,
                topk=pareto_topk,
                hypervolume=hypervolume,
            )
            write_pareto_selection_artifacts(tracker.results_dir, pareto_summary)
            representative_roles = build_pareto_representative_roles(
                pareto_front,
                objectives=pareto_objectives,
                directions=pareto_directions,
            )
            (tracker.results_dir / "pareto_representatives.json").write_text(
                json.dumps(representative_roles, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            lut_stats = estimator.get_lut_stats()
            (tracker.results_dir / "lut_stats.json").write_text(
                json.dumps(lut_stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            deferred_summary = {
                "deferred_hits": lut_stats.get("deferred_hits", 0),
                "deferred_hits_by_op": lut_stats.get("deferred_hits_by_op", {}),
                "deferred_hits_by_case": lut_stats.get("deferred_hits_by_case", {}),
                "deferred_hits_by_reason": lut_stats.get("deferred_hits_by_reason", {}),
                "true_misses": lut_stats.get("true_misses", 0),
                "true_misses_by_op": lut_stats.get("true_misses_by_op", {}),
                "true_misses_by_shape": lut_stats.get("true_misses_by_shape", {}),
                "fallback_rate": lut_stats.get("fallback_rate", 0.0),
                "strict_formal_lut": lut_stats.get("strict_formal_lut", False),
            }
            (tracker.results_dir / "deferred_hit_summary.json").write_text(
                json.dumps(deferred_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print("\n=== Search Results ===")
            if best_candidate:
                if search_method == "aging_evolution":
                    print(
                        "Compatibility accuracy-first representative "
                        f"(not a single overall best): {best_candidate.arch_id}"
                    )
                else:
                    print(f"Best architecture: {best_candidate.arch_id}")
                best_score = candidate_selection_score(best_candidate, selection_metric)
                print(f"  Score ({selection_metric}): {best_score:.4f}")
                if best_candidate.metrics.macro_f1 is not None:
                    print(f"  Macro-F1: {best_candidate.metrics.macro_f1:.4f}")
                if best_candidate.metrics.top1 is not None:
                    print(f"  Top-1: {best_candidate.metrics.top1:.4f}")
                if best_candidate.metrics.top5 is not None:
                    print(f"  Top-5: {best_candidate.metrics.top5:.4f}")
                print(f"  Latency: {best_candidate.metrics.latency_ms:.2f}ms")
                print(f"  DSP: {best_candidate.metrics.dsp}")
                print(f"  BRAM: {best_candidate.metrics.bram}")
                print(f"  LUT: {best_candidate.metrics.lut}")
                print(f"  Power: {_format_optional(best_candidate.metrics.power_w, '.2f')}W")
                print(f"  Energy: {_format_optional(best_candidate.metrics.energy_mj, '.2f')}mJ")
                print(
                    f"  Memory BW: "
                    f"{_format_optional(best_candidate.metrics.memory_bandwidth_gbps, '.3f')}GB/s"
                )
                print(
                    f"  Off-chip Mem: "
                    f"{_format_optional(best_candidate.metrics.offchip_mem_mb, '.3f')}MB"
                )
                if pareto_front:
                    print(
                        f"\nPareto front: {len(pareto_front)} candidates, "
                        f"selection={pareto_method}, hypervolume={hypervolume:.4f}"
                    )
                if pareto:
                    print(f"\n=== Top {len(pareto)} Candidates (Pareto Selected) ===")
                    for index, candidate in enumerate(pareto, start=1):
                        pareto_score = candidate_selection_score(candidate, selection_metric)
                        print(
                            f"{index}. {candidate.arch_id}: "
                            f"{selection_metric}={pareto_score:.4f}, "
                            f"Lat={candidate.metrics.latency_ms:.2f}ms"
                        )
                if search_method == "aging_evolution":
                    print("\n=== Pareto Representative Roles ===")
                    for role_name, payload in representative_roles.get("roles", {}).items():
                        if payload is None:
                            print(f"{role_name}: unavailable")
                        else:
                            print(
                                f"{role_name}: {payload['arch_id']} "
                                f"objectives={payload['objective_values']}"
                            )
            else:
                print("No feasible architecture found!")

            print(
                "LUT usage summary: "
                f"hits={lut_stats['hits']}, misses={lut_stats['misses']}, "
                f"hit_rate={lut_stats['hit_rate']:.3f}"
            )

            tracker.finalize(
                status="completed",
                candidates=searcher.evaluated_candidates,
                feasible_candidates=searcher.feasible_candidates,
                infeasible_candidates=searcher.infeasible_candidates,
                best_candidate=(None if search_method == "aging_evolution" else best_candidate),
                pareto_candidates=pareto_front,
                extra={
                    "device": device,
                    "search_method": search_method,
                    "search_summary": searcher.get_search_summary(),
                    "timeout_minutes": timeout_minutes,
                    "num_candidates": num_candidates,
                    "num_episodes": num_episodes,
                    "train_epochs": train_epochs,
                    "eval_early_stopping_patience": eval_early_stopping_patience,
                    "selection_metric": selection_metric,
                    "objective_weights": objective_weights,
                    "reward_cfg": reward_cfg if search_method == "rl" else None,
                    "aging": aging_cfg if search_method == "aging_evolution" else None,
                    "robustness": robustness_cfg,
                    "pareto_representative_roles": representative_roles,
                    "accuracy_first_compatibility_representative": (
                        _candidate_to_dict(best_candidate)
                        if search_method == "aging_evolution" and best_candidate is not None
                        else None
                    ),
                    "single_overall_best_declared": search_method != "aging_evolution",
                    "search_efficiency": search_efficiency,
                    "rl_exploration": (
                        {
                            "controller_temperature": controller_temperature,
                            "entropy_coef": entropy_coef,
                            "exploration_epsilon_start": exploration_epsilon_start,
                            "exploration_epsilon_end": exploration_epsilon_end,
                            "exploration_epsilon_decay_episodes": exploration_epsilon_decay_episodes,
                            "exploration_bonus": exploration_bonus,
                        }
                        if search_method == "rl"
                        else None
                    ),
                    "proxyless": proxyless_cfg if search_method == "proxyless" else None,
                    "hardware_spec": {
                        "name": hardware_spec.name,
                        "clock_mhz": hardware_spec.clock_mhz,
                        "max_lut": hardware_spec.max_lut,
                        "max_ff": hardware_spec.max_ff,
                        "max_bram": hardware_spec.max_bram,
                        "max_dsp": hardware_spec.max_dsp,
                        "max_power_w": hardware_spec.max_power_w,
                        "memory_bandwidth_gbps": hardware_spec.memory_bandwidth_gbps,
                        "offchip_mem_mb": hardware_spec.offchip_mem_mb,
                    },
                    "lut_stats": lut_stats,
                    "pareto_summary": pareto_summary,
                },
            )

            print("\n=== Done ===")
    except Exception as exc:
        tracker.mark_failed(str(exc))
        raise


if __name__ == "__main__":
    main()
