"""Final retraining workflow for the selected NAS architecture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
import json

import torch
import torch.nn as nn

from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec
from hwnas_fpga.training.trainer import (
    _resolve_selection_score,
    create_optimizer,
    evaluate_classifier,
)


def load_best_candidate_artifact(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path).expanduser().resolve()
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "candidate" not in payload or "encoding" not in payload["candidate"]:
        raise ValueError(f"Invalid best candidate artifact: {candidate_path}")
    return payload


def load_architecture_from_artifact(path: str | Path) -> ArchitectureSpec:
    payload = load_best_candidate_artifact(path)
    return ArchitectureSpec.from_dict(payload["candidate"]["encoding"])


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_samples += targets.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def retrain_architecture(
    *,
    architecture: ArchitectureSpec,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    optimizer_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    device: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: Optional[int] = None,
    selection_metric: str = "macro_f1",
    topk: int = 5,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        head_channels=architecture.head_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )
    optimizer = create_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top1": [],
        "val_top5": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "selection_metric": selection_metric,
        "best_epoch": None,
        "best_eval": None,
    }
    best_state = deepcopy(model.state_dict())
    best_metrics = {
        "best_epoch": 0,
        "best_selection_score": float("-inf"),
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "best_eval": None,
    }
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)
        val_summary = evaluate_classifier(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            topk=topk,
        )
        val_loss = val_summary["loss"]
        val_acc = val_summary["top1"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top1"].append(val_summary["top1"])
        history["val_top5"].append(val_summary["top5"])
        history["val_macro_f1"].append(val_summary["macro_f1"])
        history["val_weighted_f1"].append(val_summary["weighted_f1"])

        selection_score = _resolve_selection_score(val_summary, selection_metric)
        improved = selection_score > best_metrics["best_selection_score"]
        if improved:
            best_state = deepcopy(model.state_dict())
            best_eval = dict(val_summary)
            best_metrics = {
                "best_epoch": epoch + 1,
                "best_selection_score": selection_score,
                "best_val_acc": val_acc,
                "best_val_loss": val_loss,
                "best_eval": best_eval,
            }
            history["best_epoch"] = epoch + 1
            history["best_eval"] = best_eval
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                break

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_macro_f1={val_summary['macro_f1']:.6f}, "
            f"val_top1={val_summary['top1']:.6f}, "
            f"val_weighted_f1={val_summary['weighted_f1']:.6f}, "
            f"selection_score={selection_score:.6f}, "
            f"best_epoch={best_metrics['best_epoch']}"
        )

    model.load_state_dict(best_state)
    final_eval = evaluate_classifier(
        model,
        val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        topk=topk,
    )
    metrics = {
        **best_metrics,
        "selection_metric": selection_metric,
        "final_eval": final_eval,
        "final_val_loss": final_eval["loss"],
        "final_val_acc": final_eval["top1"],
        "epochs_completed": len(history["train_acc"]),
        "train_acc_last": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "train_loss_last": history["train_loss"][-1] if history["train_loss"] else 0.0,
    }
    return model, metrics, history
