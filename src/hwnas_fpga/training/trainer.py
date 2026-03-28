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
) -> Tuple[float, Dict[str, Any]]:
    """快速训练模型并返回最佳精度"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = create_optimizer(model, optimizer_name="adamw", lr=lr)

    trainer = Trainer(model, optimizer, criterion, device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=False,
        early_stopping_patience=early_stopping_patience,
    )

    best_acc = max(history["val_acc"]) if history["val_acc"] else max(history["train_acc"])
    return best_acc, history
