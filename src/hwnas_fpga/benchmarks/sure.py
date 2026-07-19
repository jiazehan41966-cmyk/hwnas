"""NKSID protocol bridge for the pinned, locally isolated SURE author code.

The SURE repository has no explicit redistribution license. This module does
not copy its implementation: it verifies the isolated checkout and loads the
author's loss, SAM, and cosine-classifier components at runtime. Only protocol
bridging, inner-fold model selection, and provenance capture live here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset

from hwnas_fpga.training.recipe import RecipeResult
from hwnas_fpga.training.trainer import _resolve_selection_score, evaluate_classifier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SureAuthorRecipeConfig:
    epochs: int = 150
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    mixup_beta: float = 10.0
    mixup_weight: float = 0.5
    crl_weight: float = 0.5
    rho: float = 0.05
    cosine_temperature: float = 8.0
    swa_start_ratio: float = 0.6
    swa_lr: float = 0.05
    selection_metric: str = "macro_f1"
    topk: int = 5

    def validate(self) -> None:
        if self.epochs <= 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid SURE optimization configuration")
        if self.mixup_beta <= 0 or self.mixup_weight < 0 or self.crl_weight < 0:
            raise ValueError("invalid SURE loss configuration")
        if not 0.0 <= self.swa_start_ratio < 1.0:
            raise ValueError("swa_start_ratio must be in [0, 1)")
        if self.rho < 0 or self.cosine_temperature <= 0:
            raise ValueError("invalid SURE SAM/cosine configuration")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "author_recipe": "FMFP_plus_CRL_plus_RegMixup_plus_cosine_classifier",
                "author_reference": "run/CIFAR10/wideresnet.sh Ours block",
                "migration_rule": "same project backbone and frozen NKSID split",
                "fairness_deviation": (
                    "Use exactly epochs iterations, not the author main.py epochs+1 loop."
                ),
                "amp": False,
                "gradient_accumulation_steps": 1,
            }
        )
        return payload


@dataclass(frozen=True)
class SureAuthorComponents:
    checkout: Path
    commit: str
    train_module: ModuleType
    sam_module: ModuleType
    classifier_module: ModuleType
    source_hashes: dict[str, str]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load author module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sure_author_components(
    checkout: str | Path,
    *,
    pinned_commit: str,
) -> SureAuthorComponents:
    root = Path(checkout).resolve()
    required = {
        "train.py": root / "train.py",
        "utils/sam.py": root / "utils" / "sam.py",
        "model/classifier.py": root / "model" / "classifier.py",
        "run/CIFAR10/wideresnet.sh": root / "run" / "CIFAR10" / "wideresnet.sh",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SURE author checkout missing: {missing}")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or commit != pinned_commit:
        raise RuntimeError(f"SURE checkout pin mismatch: {commit} != {pinned_commit}")

    # train.py imports top-level ``utils.utils``. Temporarily placing the exact
    # checkout first preserves author imports without copying them into src/.
    sys.path.insert(0, str(root))
    try:
        train_module = _load_module("hwnas_sure_author_train", required["train.py"])
        sam_module = _load_module("hwnas_sure_author_sam", required["utils/sam.py"])
        classifier_module = _load_module(
            "hwnas_sure_author_classifier", required["model/classifier.py"]
        )
    finally:
        sys.path.remove(str(root))
    imported_utils = sys.modules.get("utils.utils")
    if imported_utils is not None:
        imported_path = Path(str(getattr(imported_utils, "__file__", ""))).resolve()
        if root not in imported_path.parents:
            raise RuntimeError(f"SURE train.py resolved an unexpected utils module: {imported_path}")
    return SureAuthorComponents(
        checkout=root,
        commit=commit,
        train_module=train_module,
        sam_module=sam_module,
        classifier_module=classifier_module,
        source_hashes={name: _sha256(path) for name, path in required.items()},
    )


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return image, target, int(index)


def indexed_training_loader(loader: DataLoader) -> DataLoader:
    """Rebuild a loader with stable dataset indices and the same batch sampler."""

    return DataLoader(
        IndexedDataset(loader.dataset),
        batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        worker_init_fn=loader.worker_init_fn,
        persistent_workers=loader.persistent_workers if loader.num_workers > 0 else False,
    )


def apply_author_cosine_classifier(
    model: nn.Module,
    *,
    components: SureAuthorComponents,
    num_classes: int,
    temperature: float,
) -> dict[str, Any]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise ValueError("SURE MobileNet/SimpleCNN bridge requires model.classifier Sequential")
    linear_indices = [index for index, layer in enumerate(classifier) if isinstance(layer, nn.Linear)]
    if not linear_indices:
        raise ValueError("SURE bridge could not find a classifier Linear layer")
    index = linear_indices[-1]
    original = classifier[index]
    classifier[index] = components.classifier_module.Classifier(
        original.in_features, int(num_classes), float(temperature)
    )
    return {
        "head": "author_cosine_classifier",
        "replaced_index": index,
        "in_features": original.in_features,
        "num_classes": int(num_classes),
        "temperature": float(temperature),
        "author_source_sha256": components.source_hashes["model/classifier.py"],
    }


def train_with_sure_author_recipe(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    num_classes: int,
    recipe: SureAuthorRecipeConfig,
    components: SureAuthorComponents,
    device: str,
    verbose: bool = True,
) -> RecipeResult:
    """Run the pinned SURE author loss/SAM components under the frozen split."""

    recipe.validate()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Pinned SURE author CRL code is CUDA-only")
    model = model.to(device)
    indexed_loader = indexed_training_loader(train_loader)
    author = components.train_module
    optimizer = components.sam_module.SAM(
        model.parameters(),
        torch.optim.SGD,
        rho=recipe.rho,
        lr=recipe.lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    # The author SAM wrapper performs the actual step through base_optimizer;
    # scheduling that optimizer avoids PyTorch's false "scheduler before step"
    # warning while updating the same shared parameter groups.
    scheduler_optimizer = optimizer.base_optimizer
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        scheduler_optimizer, T_max=max(1, recipe.epochs)
    )
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(scheduler_optimizer, swa_lr=recipe.swa_lr)
    swa_start_epoch = max(1, int(round(recipe.epochs * recipe.swa_start_ratio)))
    args = SimpleNamespace(
        mixup_beta=recipe.mixup_beta,
        mixup_weight=recipe.mixup_weight,
        crl_weight=recipe.crl_weight,
        optim_name="fmfp",
    )
    correct_log = author.Correctness_Log(len(indexed_loader.dataset))
    cls_criterion = nn.CrossEntropyLoss()
    mixup_criterion = author.Mixup_Criterion(recipe.mixup_beta, cls_criterion)
    rank_criterion = author.CRL_Criterion()
    eval_criterion = nn.CrossEntropyLoss().to(device)
    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "lr": [],
        "inner_val_macro_f1": [],
        "inner_val_top1": [],
        "recipe": recipe.to_dict(),
        "author_commit": components.commit,
        "author_source_hashes": components.source_hashes,
        "swa_start_epoch": swa_start_epoch,
    }
    best_score = float("-inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    best_inner_eval: dict[str, Any] = {}

    for epoch_index in range(recipe.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, targets, sample_indices in indexed_loader:
            images = images.to(device)
            targets = targets.long().to(device)
            # The pinned author code expects indices on CPU: it converts them
            # to NumPy for its epoch-wise correctness memory.
            sample_indices = sample_indices.long()
            loss, _ce, _mixup, _crl, outputs = author.compute_loss(
                args,
                model,
                images,
                targets,
                sample_indices,
                correct_log,
                cls_criterion,
                mixup_criterion,
                rank_criterion,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            second_loss = author.compute_loss(
                args,
                model,
                images,
                targets,
                sample_indices,
                correct_log,
                cls_criterion,
                mixup_criterion,
                rank_criterion,
            )[0]
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
            correctness = outputs.detach().argmax(dim=1).eq(targets)
            correct_log.update(sample_indices, correctness)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int(correctness.sum().item())
            total_samples += int(targets.size(0))
        correct_log.max_correctness_update(epoch_index + 1)

        if epoch_index + 1 > swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_bn(indexed_loader, swa_model, device=device)
            eval_model = swa_model
        else:
            cosine_scheduler.step()
            eval_model = model
        inner_summary = evaluate_classifier(
            eval_model,
            inner_val_loader,
            criterion=eval_criterion,
            device=device,
            num_classes=num_classes,
            topk=recipe.topk,
        )
        score = _resolve_selection_score(inner_summary, recipe.selection_metric)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch_index + 1
            source_model = eval_model.module if isinstance(eval_model, AveragedModel) else eval_model
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in source_model.state_dict().items()
            }
            best_inner_eval = dict(inner_summary)
        history["train_loss"].append(total_loss / max(1, total_samples))
        history["train_acc"].append(total_correct / max(1, total_samples))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["inner_val_macro_f1"].append(float(inner_summary["macro_f1"]))
        history["inner_val_top1"].append(float(inner_summary["top1"]))
        if verbose:
            print(
                f"SURE epoch {epoch_index + 1}/{recipe.epochs}: "
                f"train_loss={history['train_loss'][-1]:.4f} "
                f"train_acc={history['train_acc'][-1]:.4f} "
                f"inner_{recipe.selection_metric}={score:.4f}"
                f"{' *' if improved else ''}"
            )
    history["best_epoch"] = best_epoch
    history["best_inner_eval"] = best_inner_eval
    return RecipeResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_inner_eval=best_inner_eval,
        history=history,
    )
