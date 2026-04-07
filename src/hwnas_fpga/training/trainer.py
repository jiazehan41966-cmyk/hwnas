"""Supernet training, retraining, and evaluation workflows."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable, Tuple, Dict, Any
from tqdm import tqdm


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """完整训练流程"""
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [] if val_loader else None,
            "val_acc": [] if val_loader else None,
        }

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            val_loss = val_acc = None
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if verbose:
                log_msg = f"Epoch {epoch+1}/{epochs}: "
                log_msg += f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
                if val_loader:
                    log_msg += f" | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                print(log_msg)

            # Early stopping
            if early_stopping_patience and val_loader:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        return history


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
) -> optim.Optimizer:
    """创建优化器"""
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _topk_accuracy_hits(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    if outputs.ndim != 2:
        raise ValueError("Expected classifier logits with shape [batch, num_classes]")
    k = max(1, min(int(k), outputs.shape[1]))
    topk = outputs.topk(k, dim=1).indices
    return int(topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


def _summarize_confusion_matrix(confusion: torch.Tensor) -> dict[str, float]:
    supports = confusion.sum(dim=1)
    total = int(supports.sum().item())
    macro_f1 = 0.0
    weighted_f1 = 0.0
    num_classes = int(confusion.shape[0])

    for class_index in range(num_classes):
        tp = float(confusion[class_index, class_index].item())
        fp = float(confusion[:, class_index].sum().item() - tp)
        fn = float(confusion[class_index, :].sum().item() - tp)
        support = int(supports[class_index].item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        macro_f1 += f1
        weighted_f1 += f1 * support

    return {
        "macro_f1": macro_f1 / max(1, num_classes),
        "weighted_f1": weighted_f1 / max(1, total),
        "top1": float(confusion.diag().sum().item()) / max(1, total),
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    topk: int = 5,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_topk = 0
    confusion = torch.zeros(int(num_classes), int(num_classes), dtype=torch.long)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += targets.size(0)
        predictions = outputs.argmax(dim=1)
        total_topk += _topk_accuracy_hits(outputs, targets, topk)

        for target, prediction in zip(targets.view(-1).tolist(), predictions.view(-1).tolist()):
            confusion[int(target), int(prediction)] += 1

    summary = _summarize_confusion_matrix(confusion)
    summary.update(
        {
            "loss": total_loss / max(1, total_samples),
            "top5": total_topk / max(1, total_samples),
            "num_samples": float(total_samples),
        }
    )
    return summary


def _resolve_selection_score(summary: dict[str, float], selection_metric: str) -> float:
    normalized = str(selection_metric or "macro_f1").strip().lower()
    aliases = {
        "accuracy": "top1",
        "top1": "top1",
        "macro_f1": "macro_f1",
        "f1": "macro_f1",
        "weighted_f1": "weighted_f1",
    }
    key = aliases.get(normalized, normalized)
    if key not in summary:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    return float(summary[key])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_classes: int,
    epochs: int = 10,
    lr: float = 0.001,
    device: Optional[str] = None,
    val_loader: Optional[DataLoader] = None,
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: Optional[int] = None,
    selection_metric: str = "accuracy",
    topk: int = 5,
) -> Tuple[float, Dict[str, Any]]:
    """Quickly train a model and return the best validation selection score."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = create_optimizer(model, optimizer_name="adamw", lr=lr)

    trainer = Trainer(model, optimizer, criterion, device)
    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [] if val_loader else None,
        "val_acc": [] if val_loader else None,
        "val_top1": [] if val_loader else None,
        "val_top5": [] if val_loader else None,
        "val_macro_f1": [] if val_loader else None,
        "val_weighted_f1": [] if val_loader else None,
        "selection_metric": selection_metric,
        "best_epoch": None,
        "best_eval": None,
    }

    best_score = float("-inf")
    patience_counter = 0

    for epoch in range(int(epochs)):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            val_summary = evaluate_classifier(
                trainer.model,
                val_loader,
                criterion=trainer.criterion,
                device=device,
                num_classes=num_classes,
                topk=topk,
            )
            history["val_loss"].append(val_summary["loss"])
            history["val_acc"].append(val_summary["top1"])
            history["val_top1"].append(val_summary["top1"])
            history["val_top5"].append(val_summary["top5"])
            history["val_macro_f1"].append(val_summary["macro_f1"])
            history["val_weighted_f1"].append(val_summary["weighted_f1"])

            current_score = _resolve_selection_score(val_summary, selection_metric)
            if current_score > best_score:
                best_score = current_score
                history["best_epoch"] = epoch + 1
                history["best_eval"] = dict(val_summary)
                patience_counter = 0
            elif early_stopping_patience:
                patience_counter += 1
                if patience_counter >= int(early_stopping_patience):
                    break
        else:
            current_score = float(train_acc)
            if current_score > best_score:
                best_score = current_score
                history["best_epoch"] = epoch + 1
                history["best_eval"] = {
                    "loss": float(train_loss),
                    "top1": float(train_acc),
                    "top5": float(train_acc),
                    "macro_f1": float(train_acc),
                    "weighted_f1": float(train_acc),
                    "num_samples": float(len(train_loader.dataset)),
                }

    if best_score == float("-inf"):
        best_score = 0.0
    return float(best_score), history
