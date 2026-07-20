"""Frozen NKSID bridge for the isolated Sonar-OLTR DMCL author loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hwnas_fpga.training.recipe import RecipeResult
from hwnas_fpga.training.trainer import _resolve_selection_score, evaluate_classifier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DmclAuthorRecipeConfig:
    epochs: int = 100
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.001
    lambda_contrast: float = 0.5
    lambda_uncertainty: float = 0.1
    margin_min: float = 0.2
    margin_max: float = 0.8
    milestones: tuple[int, ...] = (35, 75)
    lr_gamma: float = 0.5
    selection_metric: str = "macro_f1"
    topk: int = 5

    def validate(self) -> None:
        if self.epochs <= 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid DMCL optimization configuration")
        if self.lambda_contrast < 0 or self.lambda_uncertainty < 0:
            raise ValueError("invalid DMCL loss weights")
        if not 0 <= self.margin_min < self.margin_max:
            raise ValueError("invalid DMCL margin range")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "author_component": "DynamicMarginLoss",
                "migration_rule": (
                    "Author loss on the project backbone, frozen known-only training split, "
                    "inner-fold selection, and outer MSP unknown scoring."
                ),
                "feature_bridge": "input to final Linear classifier",
                "amp": False,
            }
        )
        return payload


@dataclass(frozen=True)
class DmclAuthorComponents:
    checkout: Path
    code_root: Path
    commit: str
    archive_sha256: str
    module: ModuleType
    source_hashes: dict[str, str]


def load_dmcl_author_components(
    checkout: str | Path,
    *,
    pinned_commit: str,
    archive_sha256: str,
) -> DmclAuthorComponents:
    root = Path(checkout).resolve()
    archive = root / "Sonar-OLTR-main.rar"
    if not archive.is_file() or _sha256(archive) != archive_sha256:
        raise RuntimeError("Sonar-OLTR archive SHA256 mismatch")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or commit != pinned_commit:
        raise RuntimeError(f"Sonar-OLTR checkout pin mismatch: {commit} != {pinned_commit}")
    matches = sorted(root.glob("extracted/*/code/dmcl.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one isolated dmcl.py, found {len(matches)}")
    dmcl_path = matches[0]
    code_root = dmcl_path.parent
    required = {
        "code/dmcl.py": dmcl_path,
        "code/plud.py": code_root / "plud.py",
        "README_zh.md": code_root.parent / "README_zh.md",
        "requirements.txt": code_root.parent / "requirements.txt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sonar-OLTR extracted source missing: {missing}")

    # dmcl.py imports the broad legacy my_utils module, even though the loss
    # class itself does not need it. A temporary empty shim lets us execute the
    # exact author class definition without installing unrelated legacy model
    # and dataset dependencies. No author training function is claimed here.
    previous_my_utils = sys.modules.get("my_utils")
    sys.modules["my_utils"] = ModuleType("my_utils")
    try:
        spec = importlib.util.spec_from_file_location("hwnas_sonar_oltr_dmcl", dmcl_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {dmcl_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_my_utils is None:
            sys.modules.pop("my_utils", None)
        else:
            sys.modules["my_utils"] = previous_my_utils
    if not hasattr(module, "DynamicMarginLoss"):
        raise RuntimeError("pinned dmcl.py has no DynamicMarginLoss")
    return DmclAuthorComponents(
        checkout=root,
        code_root=code_root,
        commit=commit,
        archive_sha256=archive_sha256,
        module=module,
        source_hashes={name: _sha256(path) for name, path in required.items()},
    )


def _final_linear(model: nn.Module) -> nn.Linear:
    candidates = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not candidates:
        raise ValueError("DMCL feature bridge requires a final Linear classifier")
    return candidates[-1]


def train_with_dmcl_author_loss(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    num_classes: int,
    recipe: DmclAuthorRecipeConfig,
    components: DmclAuthorComponents,
    device: str,
    verbose: bool = True,
) -> RecipeResult:
    recipe.validate()
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=recipe.lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(recipe.milestones), gamma=recipe.lr_gamma
    )
    criterion = nn.CrossEntropyLoss().to(device)
    classifier = _final_linear(model)
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs):
        captured["features"] = inputs[0]

    hook = classifier.register_forward_pre_hook(capture_input)
    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "lr": [],
        "inner_val_macro_f1": [],
        "inner_val_top1": [],
        "recipe": recipe.to_dict(),
        "author_commit": components.commit,
        "author_archive_sha256": components.archive_sha256,
        "author_source_hashes": components.source_hashes,
    }
    best_score = float("-inf")
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_inner_eval: dict[str, Any] = {}
    try:
        for epoch_index in range(recipe.epochs):
            model.train()
            dynamic_margin_loss = components.module.DynamicMarginLoss(
                num_classes,
                margin_min=recipe.margin_min,
                margin_max=recipe.margin_max,
            ).to(device)
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.long().to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                features = captured.get("features")
                if features is None:
                    raise RuntimeError("DMCL feature hook did not capture classifier input")
                probabilities = torch.softmax(logits, dim=1)
                uncertainty_loss = -torch.mean(
                    torch.log(probabilities.max(dim=1).values + 1e-6)
                )
                dynamic_margin_loss.update_class_counts(targets)
                dynamic_margin_loss.update_margins()
                ce_loss = criterion(logits, targets)
                contrastive_loss = dynamic_margin_loss(features, targets)
                if not torch.isfinite(contrastive_loss):
                    raise RuntimeError(
                        "DMCL contrastive loss is non-finite; batch lacks valid positive/negative pairs"
                    )
                loss = (
                    ce_loss
                    + recipe.lambda_contrast * contrastive_loss
                    + recipe.lambda_uncertainty * uncertainty_loss
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * targets.size(0)
                total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                total_samples += int(targets.size(0))
            scheduler.step()
            inner_summary = evaluate_classifier(
                model,
                inner_val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                topk=recipe.topk,
            )
            score = _resolve_selection_score(inner_summary, recipe.selection_metric)
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch_index + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_inner_eval = dict(inner_summary)
            history["train_loss"].append(total_loss / max(1, total_samples))
            history["train_acc"].append(total_correct / max(1, total_samples))
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            history["inner_val_macro_f1"].append(float(inner_summary["macro_f1"]))
            history["inner_val_top1"].append(float(inner_summary["top1"]))
            if verbose:
                print(
                    f"DMCL epoch {epoch_index + 1}/{recipe.epochs}: "
                    f"train_loss={history['train_loss'][-1]:.4f} "
                    f"train_acc={history['train_acc'][-1]:.4f} "
                    f"inner_{recipe.selection_metric}={score:.4f}"
                    f"{' *' if improved else ''}"
                )
    finally:
        hook.remove()
    history["best_epoch"] = best_epoch
    history["best_inner_eval"] = best_inner_eval
    return RecipeResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_inner_eval=best_inner_eval,
        history=history,
    )
