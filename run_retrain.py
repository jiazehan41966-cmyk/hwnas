#!/usr/bin/env python3
"""Final retraining entrypoint for the best searched architecture."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.models import build_model
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    create_data_pipeline,
    load_config,
    pick,
)
from hwnas_fpga.training import (
    load_architecture_from_artifact,
    load_best_candidate_artifact,
)
from hwnas_fpga.fourstage_operator import architecture_sha256
from hwnas_fpga.training.protocol_reporting import canonical_sha256, sha256_file
from hwnas_fpga.training.recipe import RecipeConfig, train_with_recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the best searched architecture")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--run-dir", type=str, default=None, help="Search run directory containing checkpoints/")
    parser.add_argument("--candidate-path", type=str, default=None, help="Path to best_candidate.json")
    parser.add_argument("--dataset", type=str, default=None, choices=("dummy", "nksid"))
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--fold", type=int, default=None, help="NKSID k-fold split index")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Run artifact root directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run directory name")
    return parser.parse_args()


def resolve_candidate_path(args: argparse.Namespace) -> Path:
    if args.candidate_path:
        return Path(args.candidate_path).expanduser().resolve()
    if args.run_dir:
        return (Path(args.run_dir).expanduser().resolve() / "checkpoints" / "best_candidate.json")
    raise ValueError("Provide either --candidate-path or --run-dir")


def resolve_config_path(args: argparse.Namespace) -> str | None:
    if args.config:
        return args.config
    if args.run_dir:
        candidate_config = Path(args.run_dir).expanduser().resolve() / "config.yaml"
        if candidate_config.exists():
            return str(candidate_config)
    return None


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_path(args))

    dataset_cfg = config.get("dataset", {})
    project_cfg = config.get("project", {})
    training_cfg = config.get("training", {})
    search_cfg = config.get("search", {})

    candidate_path = resolve_candidate_path(args)
    candidate_payload = load_best_candidate_artifact(candidate_path)
    architecture = load_architecture_from_artifact(candidate_path)
    candidate_sha256 = sha256_file(candidate_path)
    architecture_hash = architecture_sha256(architecture)

    seed = pick(args.seed, project_cfg.get("seed"), 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_name = pick(args.dataset, dataset_cfg.get("name"), "dummy")
    device = pick(args.device, None, "cuda" if torch.cuda.is_available() else "cpu")
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
    epochs = pick(args.epochs, training_cfg.get("epochs"), 100)
    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)
    recipe = RecipeConfig(
        epochs=int(epochs),
        optimizer=str(training_cfg.get("optimizer", "adamw")),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
        warmup_epochs=int(training_cfg.get("warmup_epochs", 0)),
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        logit_adjust_tau=float(training_cfg.get("logit_adjust_tau", 0.0)),
        selection_metric=str(search_cfg.get("selection_metric", "macro_f1")),
        early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
        topk=int(training_cfg.get("topk", 5)),
        gradient_accumulation_steps=int(
            training_cfg.get("gradient_accumulation_steps", 1)
        ),
        amp=bool(training_cfg.get("amp", False)),
    )
    resolved_training_config = {
        **recipe.to_dict(),
        "scheduler": "cosine_with_warmup",
        "batch_size": batch_size,
        "image_size": image_size,
        "input_channels": input_channels,
        "device": device,
        "seed": seed,
    }
    data_split_parameters = {
        "dataset": dataset_name,
        "dataset_provenance": None,
        "fold": fold,
        "use_kfold": use_kfold,
        "valid_size": valid_size,
        "split_seed": split_seed,
        "image_error_policy": image_error_policy,
        "split_mode": split_mode,
        "inner_val_fraction": inner_val_fraction,
        "image_size": image_size,
    }

    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    tracker = ExperimentTracker(
        output_root=output_root,
        project_name=project_cfg.get("name", "hwnas"),
        script_name="run_retrain",
        search_method="retrain",
        dataset_name=dataset_name,
        run_name=run_name,
    )
    tracker.save_config(config, vars(args))

    try:
        with tracker.capture_console():
            print(f"Run dir: {tracker.root_dir}")
            print(f"Using device: {device}")
            print(f"Dataset: {dataset_name}")
            print(f"Retraining candidate from: {candidate_path}")

            (
                train_loader,
                val_loader,
                class_weights,
                resolved_num_classes,
                data_pipeline_metadata,
            ) = create_data_pipeline(
                dataset_name=dataset_name,
                data_dir=data_dir,
                batch_size=batch_size,
                image_size=image_size,
                num_classes=dataset_cfg.get("num_classes"),
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
                return_metadata=True,
            )
            data_split_config = {
                **data_split_parameters,
                "resolved": data_pipeline_metadata,
                "outer_validation_consumed": False,
            }
            if dataset_name == "nksid" and data_dir:
                resolved_data_root = Path(data_dir).expanduser().resolve()
                split_files = {}
                for split_name in ("train_abs.txt", "kfold_train.txt", "kfold_val.txt"):
                    split_path = resolved_data_root / split_name
                    if not split_path.is_file():
                        nested_path = resolved_data_root / "NKSID" / split_name
                        split_path = nested_path if nested_path.is_file() else split_path
                    if not split_path.is_file():
                        raise FileNotFoundError(
                            f"retrain provenance requires frozen split file: {split_path}"
                        )
                    split_files[split_name] = {
                        "path": str(split_path),
                        "sha256": sha256_file(split_path),
                    }
                data_split_config["dataset_provenance"] = {
                    "dataset_root": str(resolved_data_root),
                    "split_files": split_files,
                }

            tracker.write_dataset_summary(
                {
                    "dataset": dataset_name,
                    "data_dir": data_dir,
                    "fold": fold,
                    "use_kfold": use_kfold,
                    "valid_size": valid_size,
                    "split_seed": split_seed,
                    "image_error_policy": image_error_policy,
                    "split_mode": split_mode,
                    "inner_val_fraction": inner_val_fraction,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "num_classes": resolved_num_classes,
                }
            )

            constraints = build_constraints(config)
            hardware_spec = build_hardware_spec(config)
            estimator = build_cost_estimator(
                config,
                hardware_spec=hardware_spec,
                constraints=constraints,
            )
            cost_estimate = estimator.estimate(
                architecture=architecture,
                search_space=build_search_space(
                    config=config,
                    image_size=image_size,
                    input_channels=input_channels,
                    num_classes=resolved_num_classes,
                    constraints=constraints,
                ),
            )
            tracker.write_baseline(
                architecture=architecture.to_dict(),
                cost_estimate=cost_estimate,
            )

            model = build_model(
                architecture=architecture,
                num_classes=resolved_num_classes,
                head_channels=architecture.head_channels,
            )
            train_class_counts = (
                data_pipeline_metadata.get("split", {})
                .get("metadata", {})
                .get("train_class_counts")
            )
            if recipe.logit_adjust_tau > 0 and train_class_counts is None:
                raise ValueError(
                    "logit_adjust_tau > 0 requires exact frozen-inner train class counts"
                )
            result = train_with_recipe(
                model,
                train_loader=train_loader,
                inner_val_loader=val_loader,
                num_classes=resolved_num_classes,
                recipe=recipe,
                device=device,
                class_counts=train_class_counts,
            )
            model.load_state_dict(result.best_state)
            history = result.history
            metrics = {
                "best_epoch": result.best_epoch,
                "selection_metric": recipe.selection_metric,
                "best_eval": result.best_inner_eval,
                "final_eval": result.best_inner_eval,
                "epochs_completed": len(history.get("train_loss", [])),
            }

            checkpoint_payload = {
                "schema_version": 2,
                "saved_at": candidate_payload.get("saved_at"),
                "source_candidate": candidate_payload.get("candidate"),
                "source_candidate_file_sha256": candidate_sha256,
                "source_search_config": candidate_payload.get("search_config"),
                "architecture": architecture.to_dict(),
                "architecture_sha256": architecture_hash,
                "training_config": resolved_training_config,
                "training_config_sha256": canonical_sha256(resolved_training_config),
                "data_split_config": data_split_config,
                "data_split_config_sha256": canonical_sha256(data_split_config),
                "data_split_sha256": canonical_sha256(
                    data_pipeline_metadata.get("split", data_pipeline_metadata)
                ),
                "seed": seed,
                "metrics": metrics,
                "history": history,
                "model_state_dict": model.state_dict(),
            }
            checkpoint_path = tracker.checkpoints_dir / "final_best_model.pt"
            tracker.save_named_checkpoint("final_best_model.pt", checkpoint_payload)
            checkpoint_sha256 = sha256_file(checkpoint_path)

            retrain_summary = {
                "candidate_path": str(candidate_path),
                "source_candidate_file_sha256": candidate_sha256,
                "architecture_sha256": architecture_hash,
                "training_config_sha256": checkpoint_payload["training_config_sha256"],
                "data_split_config_sha256": checkpoint_payload["data_split_config_sha256"],
                "data_split_sha256": checkpoint_payload["data_split_sha256"],
                "checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha256,
                },
                "metrics": metrics,
                "cost_estimate": {
                    "latency_ms": cost_estimate.latency_ms,
                    "energy_mj": cost_estimate.energy_mj,
                    "peak_dsp": cost_estimate.peak_dsp,
                    "peak_bram": cost_estimate.peak_bram,
                    "peak_lut": cost_estimate.peak_lut,
                    "power_w": cost_estimate.power_w,
                    "memory_bandwidth_gbps": cost_estimate.memory_bandwidth_gbps,
                    "offchip_mem_mb": cost_estimate.offchip_mem_mb,
                },
            }
            (tracker.results_dir / "retrain_summary.json").write_text(
                json.dumps(retrain_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            tracker.finalize(
                status="completed",
                candidates=[],
                feasible_candidates=[],
                infeasible_candidates=[],
                best_candidate=None,
                pareto_candidates=[],
                extra={
                    "mode": "retrain",
                    "candidate_path": str(candidate_path),
                    "metrics": metrics,
                },
            )

            print("Retraining completed")
            print(f"Best val acc: {metrics['best_val_acc']:.4f}")
            print(f"Epochs completed: {metrics['epochs_completed']}")
            print(f"Checkpoint: {tracker.checkpoints_dir / 'final_best_model.pt'}")
    except Exception as exc:
        tracker.mark_failed(str(exc))
        raise

if __name__ == "__main__":
    main()
