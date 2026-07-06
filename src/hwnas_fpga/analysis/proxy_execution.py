"""Execute one frozen Proxy Reliability Audit work unit.

The module deliberately exposes no loop over the complete audit matrix.  One
invocation owns one architecture x outer-fold x seed x budget cell, which
makes scheduler retries and provenance checks explicit.  Budgets below the
truth budget never iterate the outer validation loader.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from hwnas_fpga.data.dataset import create_protocol_dataloaders
from hwnas_fpga.models import build_model
from hwnas_fpga.training.protocol_reporting import (
    canonical_sha256,
    sha256_file,
)
from hwnas_fpga.training.retrain import (
    load_architecture_from_artifact,
    retrain_architecture,
)
from hwnas_fpga.training.trainer import (
    _resolve_selection_score,
    create_optimizer,
    evaluate_classifier,
)


OBSERVATION_SCHEMA_VERSION = 1


def load_work_unit(
    run_matrix: str | Path,
    *,
    work_id: Optional[str] = None,
    work_index: Optional[int] = None,
) -> dict[str, Any]:
    """Resolve exactly one JSONL work unit by id or zero-based index."""
    if (work_id is None) == (work_index is None):
        raise ValueError("provide exactly one of work_id or work_index")
    rows: list[dict[str, Any]] = []
    with Path(run_matrix).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("run matrix rows must be JSON objects")
                rows.append(payload)
    if work_index is not None:
        if not 0 <= int(work_index) < len(rows):
            raise IndexError(
                f"work_index={work_index} outside [0, {len(rows)})"
            )
        return rows[int(work_index)]
    matches = [row for row in rows if str(row.get("work_id")) == str(work_id)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one work unit for work_id={work_id!r}, found {len(matches)}"
        )
    return matches[0]


def validate_manifest_for_work(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Read the manifest and verify its canonical fingerprint."""
    manifest_path = Path(str(unit["manifest_path"])).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(unit["manifest_fingerprint"])
    stored = str(manifest.get("manifest_fingerprint", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    calculated = canonical_sha256(unsigned)
    if not expected or expected != stored or stored != calculated:
        raise ValueError(
            "manifest fingerprint mismatch: "
            f"unit={expected!r}, stored={stored!r}, calculated={calculated!r}"
        )
    candidate_ids = {
        str(candidate["architecture_id"]) for candidate in manifest["candidates"]
    }
    if str(unit["architecture_id"]) not in candidate_ids:
        raise ValueError("work-unit architecture is absent from the manifest")
    return manifest


def set_reproducible_seed(seed: int) -> dict[str, Any]:
    """Set common RNG seeds and deterministic backend preferences."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "python": int(seed),
        "numpy": int(seed),
        "torch": int(seed),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def inverse_frequency_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Match the legacy search-time inverse-frequency weighting."""
    values = counts.detach().to(dtype=torch.float32).clamp_min(1.0)
    weights = values.sum() / (len(values) * values)
    return weights / weights.sum() * len(values)


@torch.no_grad()
def naswot_score(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    """Compute the NASWOT binary-activation log-determinant score.

    This is a classification zero-cost proxy.  It is intentionally distinct
    from latency/resource estimates and never enters a scalar hardware reward.
    """
    kernel: Optional[torch.Tensor] = None
    activation_modules = 0

    def activation_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal kernel, activation_modules
        if not torch.is_tensor(output) or output.ndim < 2:
            return
        binary = (output.detach().reshape(output.shape[0], -1) > 0).to(
            dtype=torch.float64,
            device="cpu",
        )
        contribution = binary @ binary.T + (1.0 - binary) @ (1.0 - binary).T
        kernel = contribution if kernel is None else kernel + contribution
        activation_modules += 1

    handles = [
        module.register_forward_hook(activation_hook)
        for module in model.modules()
        if isinstance(module, (nn.ReLU, nn.ReLU6))
    ]
    try:
        model = model.to(device)
        model.train()
        model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()

    if kernel is None or activation_modules == 0:
        raise RuntimeError("NASWOT found no ReLU/ReLU6 activation modules")
    diagonal_scale = float(torch.diag(kernel).mean().item())
    ridge = max(1e-12, diagonal_scale * 1e-6)
    regularized = kernel + ridge * torch.eye(
        kernel.shape[0],
        dtype=kernel.dtype,
    )
    sign, logabsdet = torch.linalg.slogdet(regularized)
    if float(sign.item()) <= 0:
        raise RuntimeError("NASWOT activation kernel is not positive definite")
    return {
        "score": float(logabsdet.item()),
        "activation_module_count": activation_modules,
        "batch_size": int(inputs.shape[0]),
        "kernel_ridge": ridge,
        "definition": "logdet(sum_l H_l H_l^T + (1-H_l)(1-H_l)^T)",
    }


def _resolve_dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    nested = root / "NKSID"
    if (nested / "kfold_val.txt").exists():
        return nested
    return root


def _dataset_provenance(data_dir: str | Path) -> dict[str, Any]:
    root = _resolve_dataset_root(data_dir)
    files: dict[str, Any] = {}
    for name in ("train_abs.txt", "kfold_train.txt", "kfold_val.txt"):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Gate 0 requires frozen dataset file {path}")
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {"dataset_root": str(root), "files": files}


def _git_commit(repo_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _calibration_batch(
    bundle: Mapping[str, Any],
    *,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    indices = list(bundle["split"].train_indices)
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[: min(int(batch_size), len(indices))]
    if len(selected) < 2:
        raise RuntimeError("NASWOT requires at least two inner-training samples")
    loader = DataLoader(
        Subset(bundle["eval_dataset"], selected),
        batch_size=len(selected),
        shuffle=False,
        num_workers=0,
    )
    inputs, _targets = next(iter(loader))
    return inputs


def _clone_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_prefix_trajectory(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    num_classes: int,
    positive_budgets: list[int],
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    selection_metric: str,
    device: str,
    verbose: bool = True,
) -> tuple[nn.Module, dict[int, dict[str, Any]], dict[str, Any]]:
    """Train once and snapshot best-so-far inner metrics at each budget."""
    budgets = sorted({int(value) for value in positive_budgets})
    if not budgets or budgets[0] <= 0:
        raise ValueError("positive_budgets must contain positive integers")
    maximum_epoch = budgets[-1]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = create_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    best_score = float("-inf")
    best_epoch = 0
    best_eval: dict[str, Any] = {}
    best_state = _clone_state_to_cpu(model)
    milestones: dict[int, dict[str, Any]] = {}
    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "inner_val_loss": [],
        "inner_val_top1": [],
        "inner_val_top5": [],
        "inner_val_macro_f1": [],
        "inner_val_weighted_f1": [],
        "epoch_seconds": [],
    }

    for epoch in range(1, maximum_epoch + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

        inner = evaluate_classifier(
            model,
            inner_val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            topk=5,
        )
        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)
        score = _resolve_selection_score(inner, selection_metric)
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_eval = dict(inner)
            best_state = _clone_state_to_cpu(model)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["inner_val_loss"].append(float(inner["loss"]))
        history["inner_val_top1"].append(float(inner["top1"]))
        history["inner_val_top5"].append(float(inner["top5"]))
        history["inner_val_macro_f1"].append(float(inner["macro_f1"]))
        history["inner_val_weighted_f1"].append(float(inner["weighted_f1"]))
        history["epoch_seconds"].append(time.perf_counter() - started)

        if epoch in budgets:
            milestones[epoch] = {
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "best_eval": dict(best_eval),
            }
        if verbose:
            marker = " *" if best_epoch == epoch else ""
            print(
                f"Epoch {epoch}/{maximum_epoch}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"inner_{selection_metric}={score:.4f}{marker}"
            )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_eval"] = best_eval
    history["total_seconds"] = float(sum(history["epoch_seconds"]))
    return model, milestones, history


def execute_work_unit(
    unit: Mapping[str, Any],
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    data_dir_override: Optional[str | Path] = None,
    device_override: Optional[str] = None,
    num_workers_override: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Execute and atomically persist one work-unit observation."""
    root = Path(repo_root).resolve()
    manifest = validate_manifest_for_work(unit)
    protocol = manifest["protocol"]
    execution = protocol["execution"]
    budget = int(unit["budget"])
    truth_budget = int(unit["truth_budget"])
    expected_scope = (
        "inner_select_outer_once" if budget == truth_budget else "inner_only"
    )
    if str(unit["evaluation_scope"]) != expected_scope:
        raise ValueError(
            f"invalid evaluation_scope for budget={budget}: "
            f"{unit['evaluation_scope']!r}"
        )
    if budget < 0 or budget not in {int(value) for value in execution["budgets"]}:
        raise ValueError(f"unregistered budget={budget}")

    source_config = Path(str(manifest["source_config"])).resolve()
    source = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    dataset_cfg = source.get("dataset", {})
    configured_data_dir = data_dir_override or dataset_cfg.get(
        "data_dir", "data/NKSID"
    )
    data_dir = Path(configured_data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(
        num_workers_override
        if num_workers_override is not None
        else execution.get("num_workers", dataset_cfg.get("num_workers", 4))
    )
    seed = int(unit["seed"])
    fold = int(unit["outer_fold"])
    candidate_path = Path(str(unit["candidate_artifact"])).resolve()
    candidate_sha256 = sha256_file(candidate_path)
    dataset_provenance = _dataset_provenance(data_dir)
    reproducibility = set_reproducible_seed(seed)

    recipe = {
        "optimizer": "adamw",
        "lr": float(execution.get("lr", 0.001)),
        "weight_decay": float(execution.get("weight_decay", 0.0001)),
        "epochs": budget,
        "lr_schedule": "constant",
        "early_stopping_patience": execution.get("early_stopping_patience"),
        "selection_metric": str(execution.get("selection_metric", "macro_f1")),
        "class_weighting": "inverse_frequency_mean_one",
        "batch_size": int(execution.get("batch_size", 16)),
        "image_size": int(dataset_cfg.get("image_size", 224)),
        "input_channels": int(dataset_cfg.get("input_channels", 1)),
        "inner_val_fraction": float(execution.get("inner_val_fraction", 0.15)),
        "short_budget_outer_evaluation": "forbidden",
    }
    recipe_id = canonical_sha256(recipe)
    work_fingerprint = canonical_sha256(
        {
            "unit": dict(unit),
            "recipe": recipe,
            "candidate_sha256": candidate_sha256,
            "dataset": dataset_provenance,
        }
    )
    destination = Path(output_dir).resolve() / f"{unit['work_id']}.json"
    if destination.exists() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("work_fingerprint") == work_fingerprint
        ):
            return destination
        raise RuntimeError(
            f"refusing to overwrite incompatible observation {destination}; "
            "use --force"
        )

    bundle = create_protocol_dataloaders(
        data_dir,
        fold=fold,
        seed=seed,
        batch_size=recipe["batch_size"],
        image_size=recipe["image_size"],
        inner_val_fraction=recipe["inner_val_fraction"],
        num_workers=num_workers,
        output_channels=recipe["input_channels"],
    )
    architecture = load_architecture_from_artifact(candidate_path)
    record: dict[str, Any] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "audit": "proxy_reliability_gate0",
        "status": "completed",
        "work_id": str(unit["work_id"]),
        "work_fingerprint": work_fingerprint,
        "work_unit": dict(unit),
        "architecture_id": str(unit["architecture_id"]),
        "proxy_name": str(unit["proxy_name"]),
        "proxy_direction": "max",
        "budget": budget,
        "seed": seed,
        "outer_fold": fold,
        "truth_budget": truth_budget,
        "recipe_id": recipe_id,
        "recipe": recipe,
        "split": bundle["split"].to_dict(),
        "split_sha256": canonical_sha256(bundle["split"].to_dict()),
        "provenance": {
            "git_commit": _git_commit(root),
            "manifest_path": str(Path(str(unit["manifest_path"])).resolve()),
            "manifest_fingerprint": str(unit["manifest_fingerprint"]),
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha256,
            "dataset": dataset_provenance,
            "rng": reproducibility,
            "device": device,
            "torch_version": torch.__version__,
        },
    }

    if budget == 0:
        model = build_model(
            architecture=architecture,
            num_classes=int(bundle["num_classes"]),
            head_channels=architecture.head_channels,
        )
        calibration = _calibration_batch(
            bundle,
            seed=seed,
            batch_size=int(execution.get("zero_cost_calibration_batch_size", 16)),
        )
        zero_cost = naswot_score(model, calibration, device=device)
        record["proxy_values"] = {
            metric: zero_cost["score"] for metric in execution["metrics"]
        }
        record["truth_values"] = {}
        record["zero_cost"] = zero_cost
        record["outer_evaluation_performed"] = False
    else:
        class_weights = inverse_frequency_class_weights(
            bundle["train_class_counts"]
        )
        model, metrics, history = retrain_architecture(
            architecture=architecture,
            train_loader=bundle["train_loader"],
            val_loader=bundle["inner_val_loader"],
            num_classes=int(bundle["num_classes"]),
            epochs=budget,
            optimizer_name=recipe["optimizer"],
            lr=recipe["lr"],
            weight_decay=recipe["weight_decay"],
            device=device,
            class_weights=class_weights,
            early_stopping_patience=recipe["early_stopping_patience"],
            selection_metric=recipe["selection_metric"],
            lr_schedule=recipe["lr_schedule"],
        )
        inner = dict(metrics["best_eval"] or metrics["final_eval"])
        record["proxy_values"] = {
            metric: float(inner[metric]) for metric in execution["metrics"]
        }
        record["best_epoch"] = int(metrics["best_epoch"])
        record["training_metrics"] = metrics
        record["training_history"] = history
        record["class_weights"] = class_weights.tolist()
        record["truth_values"] = {}
        record["outer_evaluation_performed"] = False
        if budget == truth_budget:
            outer = evaluate_classifier(
                model,
                bundle["outer_val_loader"],
                criterion=nn.CrossEntropyLoss().to(device),
                device=device,
                num_classes=int(bundle["num_classes"]),
                topk=5,
            )
            record["truth_values"] = {
                metric: float(outer[metric]) for metric in execution["metrics"]
            }
            record["outer_summary"] = outer
            record["outer_evaluation_performed"] = True

    _atomic_write_json(destination, record)
    return destination


def execute_prefix_work_unit(
    unit: Mapping[str, Any],
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    data_dir_override: Optional[str | Path] = None,
    device_override: Optional[str] = None,
    num_workers_override: Optional[int] = None,
    max_budget_override: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Execute one v2 prefix trajectory and persist all budget milestones."""
    root = Path(repo_root).resolve()
    manifest = validate_manifest_for_work(unit)
    protocol = manifest["protocol"]
    execution = protocol["execution"]
    if (
        str(execution.get("scheduler_policy"))
        != "prefix_consistent_single_trajectory_constant_lr"
    ):
        raise ValueError("manifest is not a prefix-consistent v2 protocol")
    if str(unit.get("work_type")) != "prefix_train":
        raise ValueError("execute_prefix_work_unit requires work_type=prefix_train")
    if str(unit.get("evaluation_scope")) != "inner_milestones_outer_once":
        raise ValueError("invalid prefix evaluation_scope")

    budgets = sorted({int(value) for value in unit["budgets"]})
    truth_budget = int(unit["truth_budget"])
    if budgets != sorted({int(value) for value in execution["budgets"]}):
        raise ValueError("work-unit budgets differ from the frozen protocol")
    effective_max = (
        truth_budget
        if max_budget_override is None
        else int(max_budget_override)
    )
    if effective_max not in budgets:
        raise ValueError(
            f"max_budget_override={effective_max} is not a registered budget"
        )
    formal_eligible = effective_max == truth_budget

    source_config = Path(str(manifest["source_config"])).resolve()
    source = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    dataset_cfg = source.get("dataset", {})
    configured_data_dir = data_dir_override or dataset_cfg.get(
        "data_dir", "data/NKSID"
    )
    data_dir = Path(configured_data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(
        num_workers_override
        if num_workers_override is not None
        else execution.get("num_workers", dataset_cfg.get("num_workers", 4))
    )
    seed = int(unit["seed"])
    fold = int(unit["outer_fold"])
    candidate_path = Path(str(unit["candidate_artifact"])).resolve()
    candidate_sha256 = sha256_file(candidate_path)
    dataset_provenance = _dataset_provenance(data_dir)
    reproducibility = set_reproducible_seed(seed)

    recipe = {
        "optimizer": "adamw",
        "lr": float(execution.get("lr", 0.001)),
        "weight_decay": float(execution.get("weight_decay", 0.0001)),
        "epochs": truth_budget,
        "milestone_budgets": budgets,
        "lr_schedule": "constant",
        "scheduler_policy": "prefix_consistent_single_trajectory_constant_lr",
        "early_stopping_patience": None,
        "selection_metric": str(execution.get("selection_metric", "macro_f1")),
        "class_weighting": "inverse_frequency_mean_one",
        "batch_size": int(execution.get("batch_size", 32)),
        "image_size": int(dataset_cfg.get("image_size", 224)),
        "input_channels": int(dataset_cfg.get("input_channels", 1)),
        "inner_val_fraction": float(execution.get("inner_val_fraction", 0.15)),
        "short_budget_outer_evaluation": "forbidden",
        "truth_budget_outer_evaluation": "once_after_inner_selection",
    }
    recipe_id = canonical_sha256(recipe)
    work_fingerprint = canonical_sha256(
        {
            "unit": dict(unit),
            "recipe": recipe,
            "effective_max_budget": effective_max,
            "candidate_sha256": candidate_sha256,
            "dataset": dataset_provenance,
        }
    )
    destination = Path(output_dir).resolve() / f"{unit['work_id']}.json"
    if destination.exists() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("status") in {"completed", "benchmark_completed"}
            and existing.get("work_fingerprint") == work_fingerprint
        ):
            return destination
        raise RuntimeError(
            f"refusing to overwrite incompatible observation {destination}; "
            "use --force"
        )

    bundle = create_protocol_dataloaders(
        data_dir,
        fold=fold,
        seed=seed,
        batch_size=recipe["batch_size"],
        image_size=recipe["image_size"],
        inner_val_fraction=recipe["inner_val_fraction"],
        num_workers=num_workers,
        output_channels=recipe["input_channels"],
    )
    architecture = load_architecture_from_artifact(candidate_path)
    model = build_model(
        architecture=architecture,
        num_classes=int(bundle["num_classes"]),
        head_channels=architecture.head_channels,
    )
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    calibration = _calibration_batch(
        bundle,
        seed=seed,
        batch_size=int(execution.get("zero_cost_calibration_batch_size", 32)),
    )
    zero_cost = naswot_score(deepcopy(model), calibration, device=device)
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)

    metrics = [str(metric) for metric in execution["metrics"]]
    milestone_records: dict[str, Any] = {
        "0": {
            "proxy_name": str(unit["zero_cost_proxy_name"]),
            "proxy_direction": "max",
            "proxy_values": {
                metric: float(zero_cost["score"]) for metric in metrics
            },
            "best_epoch": 0,
        }
    }
    history: dict[str, Any] = {}
    class_weights = inverse_frequency_class_weights(
        bundle["train_class_counts"]
    )
    positive_budgets = [
        value for value in budgets if 0 < value <= effective_max
    ]
    if positive_budgets:
        model, trained_milestones, history = train_prefix_trajectory(
            model,
            train_loader=bundle["train_loader"],
            inner_val_loader=bundle["inner_val_loader"],
            num_classes=int(bundle["num_classes"]),
            positive_budgets=positive_budgets,
            optimizer_name=recipe["optimizer"],
            lr=recipe["lr"],
            weight_decay=recipe["weight_decay"],
            class_weights=class_weights,
            selection_metric=recipe["selection_metric"],
            device=device,
        )
        for budget, milestone in trained_milestones.items():
            best_eval = milestone["best_eval"]
            milestone_records[str(budget)] = {
                "proxy_name": str(unit["proxy_name"]),
                "proxy_direction": "max",
                "proxy_values": {
                    metric: float(best_eval[metric]) for metric in metrics
                },
                "best_epoch": int(milestone["best_epoch"]),
                "best_selection_score": float(
                    milestone["best_selection_score"]
                ),
            }

    truth_values: dict[str, float] = {}
    outer_summary: Optional[dict[str, Any]] = None
    outer_performed = False
    if formal_eligible:
        outer_summary = evaluate_classifier(
            model,
            bundle["outer_val_loader"],
            criterion=nn.CrossEntropyLoss().to(device),
            device=device,
            num_classes=int(bundle["num_classes"]),
            topk=5,
        )
        truth_values = {
            metric: float(outer_summary[metric]) for metric in metrics
        }
        outer_performed = True

    record: dict[str, Any] = {
        "schema_version": 2,
        "audit": "proxy_reliability_gate0_v2",
        "status": "completed" if formal_eligible else "benchmark_completed",
        "formal_eligible": formal_eligible,
        "effective_max_budget": effective_max,
        "work_id": str(unit["work_id"]),
        "work_fingerprint": work_fingerprint,
        "work_unit": dict(unit),
        "architecture_id": str(unit["architecture_id"]),
        "stage": str(unit["stage"]),
        "seed": seed,
        "outer_fold": fold,
        "budgets": budgets,
        "truth_budget": truth_budget,
        "recipe_id": recipe_id,
        "recipe": recipe,
        "milestones": milestone_records,
        "truth_values": truth_values,
        "outer_evaluation_performed": outer_performed,
        "outer_summary": outer_summary,
        "zero_cost": zero_cost,
        "training_history": history,
        "class_weights": class_weights.tolist(),
        "split": bundle["split"].to_dict(),
        "split_sha256": canonical_sha256(bundle["split"].to_dict()),
        "provenance": {
            "git_commit": _git_commit(root),
            "manifest_path": str(Path(str(unit["manifest_path"])).resolve()),
            "manifest_fingerprint": str(unit["manifest_fingerprint"]),
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha256,
            "dataset": dataset_provenance,
            "rng": reproducibility,
            "device": device,
            "torch_version": torch.__version__,
        },
    }
    _atomic_write_json(destination, record)
    return destination
