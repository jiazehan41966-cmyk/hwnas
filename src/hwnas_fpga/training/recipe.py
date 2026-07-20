"""Modern training recipe for protocol runs and retraining.

Replaces the legacy fixed-LR AdamW loop with:

- cosine learning-rate decay with linear warmup,
- label smoothing,
- optional logit-adjusted cross entropy for the NKSID long tail
  (Menon et al., "Long-tail learning via logit adjustment", ICLR 2021),
  which supersedes inverse-frequency class weights.

Epoch selection always happens on the *inner* validation loader; the caller
is responsible for evaluating the returned best state exactly once on the
outer validation fold.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from hwnas_fpga.training.trainer import (
    _resolve_selection_score,
    create_optimizer,
    evaluate_classifier,
)


@dataclass
class RecipeConfig:
    epochs: int = 150
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    label_smoothing: float = 0.1
    logit_adjust_tau: float = 1.0
    selection_metric: str = "macro_f1"
    early_stopping_patience: Optional[int] = None
    topk: int = 5
    gradient_accumulation_steps: int = 1
    amp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "min_lr_ratio": self.min_lr_ratio,
            "label_smoothing": self.label_smoothing,
            "logit_adjust_tau": self.logit_adjust_tau,
            "selection_metric": self.selection_metric,
            "early_stopping_patience": self.early_stopping_patience,
            "topk": self.topk,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "amp": self.amp,
        }


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross entropy with additive log-prior logit adjustment.

    Training-only loss: logits are shifted by ``tau * log(class_prior)``
    before the cross entropy, which is equivalent to optimizing the balanced
    error. Evaluation must use the raw logits.
    """

    def __init__(
        self,
        class_counts: Sequence[float],
        *,
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(list(class_counts), dtype=torch.float32).clamp_min(1.0)
        priors = counts / counts.sum()
        self.register_buffer("log_priors", priors.log())
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_priors.to(logits.device)
        return nn.functional.cross_entropy(
            adjusted, targets, label_smoothing=self.label_smoothing
        )


def build_train_criterion(
    recipe: RecipeConfig,
    *,
    class_counts: Optional[Sequence[float]] = None,
) -> nn.Module:
    if recipe.logit_adjust_tau > 0 and class_counts is not None:
        return LogitAdjustedCrossEntropy(
            class_counts,
            tau=recipe.logit_adjust_tau,
            label_smoothing=recipe.label_smoothing,
        )
    return nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    epochs = max(1, int(epochs))
    warmup_epochs = max(0, min(int(warmup_epochs), epochs - 1))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        span = max(1, epochs - warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / span)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class RecipeResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_inner_eval: dict[str, Any]
    history: dict[str, Any] = field(default_factory=dict)


def train_with_recipe(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    num_classes: int,
    recipe: RecipeConfig,
    device: Optional[str] = None,
    class_counts: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> RecipeResult:
    """Train ``model`` and select the best epoch on the inner validation set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    accumulation_steps = max(1, int(recipe.gradient_accumulation_steps))
    amp_enabled = bool(recipe.amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_criterion = build_train_criterion(recipe, class_counts=class_counts).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = create_optimizer(
        model,
        optimizer_name=recipe.optimizer,
        lr=recipe.lr,
        weight_decay=recipe.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=recipe.epochs,
        warmup_epochs=recipe.warmup_epochs,
        min_lr_ratio=recipe.min_lr_ratio,
    )

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "lr": [],
        "inner_val_loss": [],
        "inner_val_top1": [],
        "inner_val_macro_f1": [],
        "inner_val_weighted_f1": [],
        "recipe": recipe.to_dict(),
    }

    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_inner_eval: dict[str, Any] = {}
    patience_counter = 0

    for epoch in range(int(recipe.epochs)):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(train_loader)
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
            ):
                outputs = model(inputs)
                unscaled_loss = train_criterion(outputs, targets)
                loss = unscaled_loss / accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == batch_count
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += unscaled_loss.item() * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        inner_summary = evaluate_classifier(
            model,
            inner_val_loader,
            criterion=eval_criterion,
            device=device,
            num_classes=num_classes,
            topk=recipe.topk,
        )
        history["inner_val_loss"].append(inner_summary["loss"])
        history["inner_val_top1"].append(inner_summary["top1"])
        history["inner_val_macro_f1"].append(inner_summary["macro_f1"])
        history["inner_val_weighted_f1"].append(inner_summary["weighted_f1"])

        score = _resolve_selection_score(inner_summary, recipe.selection_metric)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_inner_eval = dict(inner_summary)
            patience_counter = 0
        elif recipe.early_stopping_patience:
            patience_counter += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{recipe.epochs}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"inner_{recipe.selection_metric}={score:.4f}"
                f"{' *' if improved else ''}"
            )

        if (
            recipe.early_stopping_patience
            and patience_counter >= int(recipe.early_stopping_patience)
        ):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    history["best_epoch"] = best_epoch
    history["best_inner_eval"] = best_inner_eval
    return RecipeResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_inner_eval=best_inner_eval,
        history=history,
    )
