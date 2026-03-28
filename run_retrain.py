#!/usr/bin/env python3
"""Final retraining entrypoint for the best searched architecture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.experiment import ExperimentTracker
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
    retrain_architecture,
)


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

    candidate_path = resolve_candidate_path(args)
    candidate_payload = load_best_candidate_artifact(candidate_path)
    architecture = load_architecture_from_artifact(candidate_path)

    seed = pick(args.seed, project_cfg.get("seed"), 42)
    torch.manual_seed(seed)

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
    epochs = pick(args.epochs, training_cfg.get("epochs"), 100)
    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)

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

            train_loader, val_loader, class_weights, resolved_num_classes = create_data_pipeline(
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
            )

            tracker.write_dataset_summary(
                {
                    "dataset": dataset_name,
                    "data_dir": data_dir,
                    "fold": fold,
                    "use_kfold": use_kfold,
                    "valid_size": valid_size,
                    "split_seed": split_seed,
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

            model, metrics, history = retrain_architecture(
                architecture=architecture,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=resolved_num_classes,
                epochs=epochs,
                optimizer_name=training_cfg.get("optimizer", "adamw"),
                lr=training_cfg.get("lr", 0.001),
                weight_decay=training_cfg.get("weight_decay", 0.0001),
                device=device,
                class_weights=class_weights,
                early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
            )

            checkpoint_payload = {
                "saved_at": candidate_payload.get("saved_at"),
                "source_candidate": candidate_payload.get("candidate"),
                "architecture": architecture.to_dict(),
                "metrics": metrics,
                "history": history,
                "model_state_dict": model.state_dict(),
            }
            tracker.save_named_checkpoint("final_best_model.pt", checkpoint_payload)

            retrain_summary = {
                "candidate_path": str(candidate_path),
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
