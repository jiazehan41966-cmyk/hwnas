"""Frozen NKSID bridge for the isolated Sonar-OLTR PLUD author loss."""

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
class PludAuthorRecipeConfig:
    epochs: int = 100
    lr: float = 0.01
    classifier_lr_multiplier: float = 10.0
    momentum: float = 0.0
    weight_decay: float = 0.001
    alpha: float = 1.5
    gamma: float = 0.5
    milestones: tuple[int, ...] = (35, 75)
    lr_gamma: float = 0.5
    selection_metric: str = "macro_f1"
    topk: int = 5

    def validate(self) -> None:
        if self.epochs <= 0 or self.lr <= 0 or self.classifier_lr_multiplier <= 0:
            raise ValueError("invalid PLUD optimization configuration")
        if self.momentum < 0 or self.weight_decay < 0:
            raise ValueError("invalid PLUD SGD configuration")
        if self.alpha <= 0 or self.gamma <= 0:
            raise ValueError("PLUD alpha and gamma must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "author_method": "PLUD_push_right_logit_up_wrong_logits_down",
                "author_components": ["push_logit_loss", "four_term_training_objective"],
                "migration_rule": (
                    "Frozen project 5-known/3-unknown split, inner-fold selection, "
                    "and outer MSP unknown scoring."
                ),
                "feature_bridge": "input to final Linear classifier",
                "equivalence_note": (
                    "For the supported CNN heads, mixing after linear pooling is "
                    "equivalent to author feature-map mixing before average pooling."
                ),
                "portability_deviation": (
                    "Replace author hard-coded randperm(...).cuda() with the resolved device."
                ),
                "amp": False,
            }
        )
        return payload


@dataclass(frozen=True)
class PludAuthorComponents:
    checkout: Path
    code_root: Path
    commit: str
    archive_sha256: str
    module: ModuleType
    source_hashes: dict[str, str]


def load_plud_author_components(
    checkout: str | Path,
    *,
    pinned_commit: str,
    archive_sha256: str,
) -> PludAuthorComponents:
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
    matches = sorted(root.glob("extracted/*/code/plud.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one isolated plud.py, found {len(matches)}")
    plud_path = matches[0]
    code_root = plud_path.parent
    required = {
        "code/plud.py": plud_path,
        "code/my_utils.py": code_root / "my_utils.py",
        "README_zh.md": code_root.parent / "README_zh.md",
        "requirements.txt": code_root.parent / "requirements.txt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sonar-OLTR extracted PLUD source missing: {missing}")

    # plud.py imports the broad legacy my_utils module at import time. The exact
    # author push_logit_loss function is self-contained, so an empty temporary
    # shim avoids importing unrelated dataset/model dependencies.
    previous_my_utils = sys.modules.get("my_utils")
    sys.modules["my_utils"] = ModuleType("my_utils")
    try:
        spec = importlib.util.spec_from_file_location("hwnas_sonar_oltr_plud", plud_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {plud_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_my_utils is None:
            sys.modules.pop("my_utils", None)
        else:
            sys.modules["my_utils"] = previous_my_utils
    if not hasattr(module, "push_logit_loss"):
        raise RuntimeError("pinned plud.py has no push_logit_loss")
    return PludAuthorComponents(
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
        raise ValueError("PLUD feature bridge requires a final Linear classifier")
    return candidates[-1]


def train_with_plud_author_loss(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    num_classes: int,
    recipe: PludAuthorRecipeConfig,
    components: PludAuthorComponents,
    device: str,
    verbose: bool = True,
) -> RecipeResult:
    recipe.validate()
    model = model.to(device)
    classifier = _final_linear(model)
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    feature_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in classifier_ids
    ]
    parameter_groups: list[dict[str, Any]] = []
    if feature_parameters:
        parameter_groups.append({"params": feature_parameters})
    parameter_groups.append(
        {
            "params": list(classifier.parameters()),
            "lr": recipe.lr * recipe.classifier_lr_multiplier,
        }
    )
    optimizer = torch.optim.SGD(
        parameter_groups,
        lr=recipe.lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(recipe.milestones), gamma=recipe.lr_gamma
    )
    criterion = nn.CrossEntropyLoss().to(device)
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
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            skipped_small_batches = 0
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.long().to(device)
                half_length = int(images.size(0) / 2)
                if half_length <= 0:
                    skipped_small_batches += 1
                    continue
                optimizer.zero_grad(set_to_none=True)

                pre_images = images[:half_length]
                later_images = images[half_length:]
                later_targets = targets[half_length:]
                _ = model(pre_images)
                pre_features = captured.get("features")
                if pre_features is None or pre_features.ndim != 2:
                    raise RuntimeError("PLUD bridge did not capture 2D classifier features")
                beta = torch.distributions.beta.Beta(recipe.alpha, recipe.alpha).sample().item()
                permutation = torch.randperm(half_length, device=pre_features.device)
                mixed_features = beta * pre_features + (1.0 - beta) * pre_features[permutation]
                mixed_logits = classifier(mixed_features)

                later_logits = model(later_images)
                right_logits = later_logits.gather(1, later_targets[:, None]).squeeze(1)
                true_mask = torch.nn.functional.one_hot(
                    later_targets, num_classes=num_classes
                ).bool()
                wrong_logits = later_logits.masked_fill(true_mask, -1e9)
                loss_push_mixed_down = components.module.push_logit_loss(
                    mixed_logits, recipe.gamma
                )
                loss_ce = criterion(later_logits, later_targets)
                loss_push_wrong_down = components.module.push_logit_loss(
                    wrong_logits, recipe.gamma
                )
                loss_push_right_up = components.module.push_logit_loss(
                    -right_logits, recipe.gamma
                )
                loss = (
                    loss_push_mixed_down
                    + loss_ce
                    + loss_push_wrong_down
                    + loss_push_right_up
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("PLUD four-term loss is non-finite")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * later_targets.size(0)
                total_correct += int(
                    later_logits.argmax(dim=1).eq(later_targets).sum().item()
                )
                total_samples += int(later_targets.size(0))
            if total_samples <= 0:
                raise RuntimeError("PLUD epoch contained no trainable half-batches")
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
            history["train_loss"].append(total_loss / total_samples)
            history["train_acc"].append(total_correct / total_samples)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            history["inner_val_macro_f1"].append(float(inner_summary["macro_f1"]))
            history["inner_val_top1"].append(float(inner_summary["top1"]))
            history.setdefault("skipped_small_batches", []).append(skipped_small_batches)
            if verbose:
                print(
                    f"PLUD epoch {epoch_index + 1}/{recipe.epochs}: "
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
