#!/usr/bin/env python3
"""Frozen-protocol evaluation entrypoint.

Trains a model under the frozen NKSID evaluation protocol
(outer 5-fold x multi-seed, inner contiguous-block selection set) and reports
outer-fold metrics with mean +/- std. This is the only entrypoint whose
classification numbers are claimable; fold-0-only results are legacy.

Examples:

  # MobileNetV2 from scratch, all 5 outer folds, 3 seeds
  python run_eval_protocol.py --arch mobilenet_v2 --epochs 150 \
      --folds 0,1,2,3,4 --seeds 42,43,44 --run-name mnv2_scratch

  # Grayscale-adapted ImageNet-pretrained MobileNetV2
  python run_eval_protocol.py --arch mobilenet_v2 --pretrained --epochs 150 \
      --folds 0,1,2,3,4 --seeds 42,43,44 --run-name mnv2_pretrained

  # A searched candidate (ArchitectureSpec artifact)
  python run_eval_protocol.py --candidate-path <best_candidate.json> \
      --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --run-name rl_arch_135
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import (
    NKSID_CLASSES,
    create_protocol_dataloaders,
    protocol_normalization,
)
from hwnas_fpga.benchmarks.metrics import calibration_summary
from hwnas_fpga.benchmarks.open_set import (
    calibrate_msp_threshold,
    create_open_set_protocol_dataloaders,
    evaluate_open_set_classifier,
)
from hwnas_fpga.benchmarks.sure import (
    SureAuthorRecipeConfig,
    apply_author_cosine_classifier,
    load_sure_author_components,
    train_with_sure_author_recipe,
)
from hwnas_fpga.benchmarks.dmcl import (
    DmclAuthorRecipeConfig,
    load_dmcl_author_components,
    train_with_dmcl_author_loss,
)
from hwnas_fpga.benchmarks.plud import (
    PludAuthorRecipeConfig,
    load_plud_author_components,
    train_with_plud_author_loss,
)
from hwnas_fpga.models import build_model
from hwnas_fpga.models.backbones import build_backbone
from hwnas_fpga.experiment_contract import (
    experiment_contract_provenance,
    load_experiment_contract,
    validate_formal_values_against_contract,
)
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.recipe import RecipeConfig, train_with_recipe
from hwnas_fpga.training.protocol_reporting import (
    canonical_sha256,
    protocol_claimability,
    sha256_file,
)
from hwnas_fpga.training.trainer import evaluate_classifier


def parse_int_list(text: str) -> list[int]:
    return [int(token) for token in str(text).replace(";", ",").split(",") if token.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def begin_unit_runtime_measurement(device: str) -> float:
    """Start a fold-seed timer and reset peak CUDA allocation when applicable."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_unit_runtime_measurement(started_at: float, device: str) -> dict[str, float | None]:
    """Return wall time and peak allocated CUDA memory for one fold-seed unit."""
    peak_gpu_memory_mb: float | None = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
    return {
        "wall_clock_seconds": max(0.0, float(time.perf_counter() - started_at)),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


def runtime_provenance() -> dict[str, object]:
    def installed_version(name: str) -> str:
        try:
            return package_version(name)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "interpreter": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "torchvision": installed_version("torchvision"),
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def environment_card_provenance(args: argparse.Namespace) -> dict[str, object] | None:
    requested = args.environment_card
    if requested:
        card_path = Path(requested).expanduser().resolve()
    elif args.adapter_id != "builtin":
        card_path = (
            Path(__file__).resolve().parent
            / "artifacts"
            / "benchmarks"
            / str(args.campaign_id)
            / "environment"
            / f"{args.paper_id}.json"
        ).resolve()
    else:
        return None
    if not card_path.is_file():
        raise FileNotFoundError(
            f"external benchmark adapter requires a verified environment card: {card_path}"
        )
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    if payload.get("isolation_status") != "READY_DEDICATED_ENVIRONMENT":
        raise RuntimeError(f"dedicated environment is not READY in {card_path}")
    dedicated = dict(payload.get("dedicated_environment") or {})
    environment_root = Path(str(dedicated.get("path") or "")).resolve()
    if environment_root != Path(sys.prefix).resolve():
        raise RuntimeError(
            "external adapter interpreter does not match its verified environment card: "
            f"{Path(sys.prefix).resolve()} != {environment_root}"
        )
    freeze = dict(dedicated.get("freeze") or {})
    freeze_path = Path(str(freeze.get("path") or ""))
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze.get("sha256"):
        raise RuntimeError(f"dedicated environment lock is missing or changed: {freeze_path}")
    stable_fingerprint = canonical_sha256(
        {
            "runtime_role": payload.get("runtime_role"),
            "path": dedicated.get("path"),
            "interpreter": dedicated.get("interpreter"),
            "probe": dedicated.get("probe"),
            "freeze": dedicated.get("freeze"),
        }
    )
    if dedicated.get("verification_fingerprint") != stable_fingerprint:
        raise RuntimeError(f"dedicated environment verification fingerprint is invalid: {card_path}")
    return {
        "path": str(card_path),
        "observed_card_sha256": sha256_file(card_path),
        "isolation_status": payload.get("isolation_status"),
        "runtime_role": payload.get("runtime_role"),
        "environment_root": str(environment_root),
        "freeze_path": str(freeze_path.resolve()),
        "freeze_sha256": freeze.get("sha256"),
        "verification_fingerprint": stable_fingerprint,
    }


def source_freeze_provenance(manifest_value: str | None) -> dict[str, object] | None:
    """Verify and bind a formal run to an immutable source snapshot."""

    if not manifest_value:
        return None
    repo_root = Path(__file__).resolve().parent
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source freeze manifest does not exist: {manifest_path}")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(manifest_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr or "no verifier output").strip()
        raise RuntimeError(f"source freeze verification failed: {detail}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = dict(manifest.get("archive") or {})
    archive_path = Path(str(archive.get("path") or "")).expanduser()
    if not archive_path.is_absolute():
        archive_path = manifest_path.parent / archive_path
    archive_path = archive_path.resolve()
    expected_archive_sha = str(archive.get("sha256") or "")
    if not archive_path.is_file():
        raise FileNotFoundError(f"source snapshot archive does not exist: {archive_path}")
    observed_archive_sha = sha256_file(archive_path)
    if not expected_archive_sha or observed_archive_sha != expected_archive_sha:
        raise RuntimeError(
            "source snapshot archive hash mismatch: "
            f"{observed_archive_sha} != {expected_archive_sha or '<missing>'}"
        )
    expected_archive_bytes = archive.get("bytes")
    if expected_archive_bytes is not None and archive_path.stat().st_size != int(expected_archive_bytes):
        raise RuntimeError("source snapshot archive byte size does not match its manifest")

    verification_payload = json.loads(completed.stdout)
    if verification_payload.get("status") != "PASS":
        raise RuntimeError("source freeze verifier returned a non-PASS payload")
    return {
        "path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verification_status": "PASS",
        "verified_file_count": int(verification_payload.get("expected_file_count") or 0),
        "archive_path": str(archive_path),
        "archive_sha256": observed_archive_sha,
        "archive_bytes": archive_path.stat().st_size,
    }


def build_run_model(args: argparse.Namespace, num_classes: int) -> tuple[nn.Module, dict]:
    if args.candidate_path:
        architecture = load_architecture_from_artifact(args.candidate_path)
        model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            head_channels=architecture.head_channels,
        )
        return model, {
            "model_source": "candidate",
            "candidate_path": str(Path(args.candidate_path).resolve()),
            "architecture": architecture.to_dict(),
        }

    model, metadata = build_backbone(
        name=args.arch,
        num_classes=num_classes,
        input_channels=1,
        pretrained=args.pretrained,
        strict_pretrained=args.pretrained,
    )
    metadata["model_source"] = "backbone"
    return model, metadata


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def per_class_f1(confusion: list[list[int]]) -> list[float]:
    num_classes = len(confusion)
    scores = []
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes)) - tp
        fn = sum(confusion[class_index]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
    return scores


def resolve_dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    nested = root / "NKSID"
    if (nested / "train_abs.txt").exists():
        return nested
    return root


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 digest of an in-memory provenance payload."""

    return hashlib.sha256(payload).hexdigest()


def git_code_provenance(run_dir: Path) -> dict:
    root = Path(__file__).resolve().parent
    commit = git_commit()
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=root,
        capture_output=True,
        check=False,
    )
    diff_bytes = diff_result.stdout
    patch_sha256 = sha256_bytes(diff_bytes)
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    patch_path = provenance_dir / f"code_patch_{patch_sha256}.diff"
    if patch_path.exists():
        observed_sha256 = sha256_file(patch_path)
        if observed_sha256 != patch_sha256:
            raise RuntimeError(
                "Refusing to overwrite corrupted provenance patch: "
                f"{patch_path} (expected {patch_sha256}, observed {observed_sha256})"
            )
    else:
        try:
            with patch_path.open("xb") as handle:
                handle.write(diff_bytes)
        except FileExistsError:
            observed_sha256 = sha256_file(patch_path)
            if observed_sha256 != patch_sha256:
                raise RuntimeError(
                    "Concurrent provenance patch creation produced unexpected bytes: "
                    f"{patch_path} (expected {patch_sha256}, observed {observed_sha256})"
                )
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked = []
    allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".toml"}
    for relative in untracked_result.stdout.splitlines():
        path = root / relative
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            untracked.append(
                {
                    "path": relative.replace("\\", "/"),
                    "sha256": sha256_file(path),
                }
            )
    state = {
        "commit": commit,
        "dirty": bool(status_result.stdout.strip()),
        "status_sha256": canonical_sha256(status_result.stdout.splitlines()),
        "tracked_patch": {
            "path": str(patch_path.resolve()),
            "sha256": patch_sha256,
        },
        "untracked_code": untracked,
    }
    state["code_state_sha256"] = canonical_sha256(state)
    return state


def dataset_provenance(data_dir: str | Path) -> dict:
    root = resolve_dataset_root(data_dir)
    files = {}
    for name in ("train_abs.txt", "kfold_train.txt", "kfold_val.txt"):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Frozen protocol requires {path}")
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "dataset_root": str(root),
        "files": files,
    }


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError(f"Unsupported checkpoint payload: {path}")


def _topk_hits(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    if outputs.ndim != 2:
        raise ValueError("Expected classifier logits with shape [batch, num_classes]")
    resolved_k = max(1, min(int(k), int(outputs.shape[1])))
    topk = outputs.topk(resolved_k, dim=1).indices
    return int(topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


def _summarize_confusion(confusion: torch.Tensor) -> dict[str, float]:
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
def evaluate_outer_classifier_with_predictions(
    model: nn.Module,
    data_loader,
    *,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    topk: int,
    eval_samples: list[tuple[str, int]],
    outer_indices: list[int],
    fold: int,
    seed: int,
    class_names: list[str],
) -> tuple[dict, list[dict]]:
    """Evaluate the outer fold once and retain per-sample prediction evidence."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_topk = 0
    cursor = 0
    confusion = torch.zeros(int(num_classes), int(num_classes), dtype=torch.long)
    rows: list[dict] = []

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)

        batch_size = int(targets.size(0))
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_topk += _topk_hits(outputs, targets, topk)

        target_values = [int(value) for value in targets.view(-1).tolist()]
        prediction_values = [int(value) for value in predictions.view(-1).tolist()]
        for offset, (target, prediction) in enumerate(zip(target_values, prediction_values)):
            dataset_index = int(outer_indices[cursor + offset])
            sample_path, dataset_label = eval_samples[dataset_index]
            confusion[target, prediction] += 1
            confidence = float(probabilities[offset].max().item())
            rows.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "split": "outer_val",
                    "sample_index": dataset_index,
                    "outer_position": cursor + offset,
                    "image_path": str(Path(sample_path).resolve()),
                    "sample_id": str(Path(sample_path).resolve()),
                    "target": target,
                    "prediction": prediction,
                    "confidence": confidence,
                    "unknown_score": None,
                    "logits": [float(value) for value in outputs[offset].detach().cpu().tolist()],
                    "correct": target == prediction,
                    "dataset_label": int(dataset_label),
                    "class_name": (
                        class_names[target] if 0 <= target < len(class_names) else str(target)
                    ),
                    "predicted_class_name": (
                        class_names[prediction]
                        if 0 <= prediction < len(class_names)
                        else str(prediction)
                    ),
                }
            )
        cursor += batch_size

    if cursor != len(outer_indices):
        raise RuntimeError(
            f"outer prediction cursor mismatch: saw {cursor}, expected {len(outer_indices)}"
        )
    summary = _summarize_confusion(confusion)
    summary.update(
        {
            "loss": total_loss / max(1, total_samples),
            "top5": total_topk / max(1, total_samples),
            "num_samples": float(total_samples),
            "confusion_matrix": confusion.tolist(),
            "calibration": calibration_summary(
                [torch.softmax(torch.tensor(row["logits"]), dim=0).tolist() for row in rows],
                [int(row["target"]) for row in rows],
            ),
        }
    )
    return summary, rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def staged_limit_reached(max_new_units: int | None, new_units_completed: int) -> bool:
    """Return whether a staged invocation must stop before another new unit.

    Compatible resumed records do not count toward ``new_units_completed``. A
    first invocation can therefore stop after one new fold-seed unit and a
    later invocation can resume the same 15-unit fingerprint without retraining
    that unit.
    """

    if max_new_units is not None and max_new_units <= 0:
        raise ValueError("max_new_units must be a positive integer")
    if new_units_completed < 0:
        raise ValueError("new_units_completed must be non-negative")
    return max_new_units is not None and new_units_completed >= max_new_units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument(
        "--task",
        choices=("closed_set", "open_long_tail"),
        default="closed_set",
        help="frozen closed-set protocol or frozen 5-known/3-unknown protocol",
    )
    parser.add_argument(
        "--adapter-id",
        default="builtin",
        choices=(
            "builtin",
            "sure_author_recipe",
            "dmcl_author_loss",
            "plud_author_loss",
        ),
        help="benchmark adapter identifier executed inside this canonical entrypoint",
    )
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument(
        "--open-set-config",
        default="configs/benchmarks/nksid_open_long_tail_v1.yaml",
    )
    parser.add_argument("--campaign-id", default="nksid_frozen_protocol_v1")
    parser.add_argument("--paper-id", default="project_internal")
    parser.add_argument("--method-id", default=None)
    parser.add_argument(
        "--environment-card",
        default=None,
        help=(
            "verified dedicated-runtime card; external adapters auto-resolve the "
            "campaign/paper card and fail closed if the interpreter does not match"
        ),
    )
    parser.add_argument(
        "--source-freeze-manifest",
        default=None,
        help=(
            "source_freeze_manifest.json produced by freeze_experiment_source.py; "
            "the source tree and retained snapshot archive are reverified before training"
        ),
    )
    parser.add_argument("--sure-checkout", default="reference/_local/SURE")
    parser.add_argument(
        "--sure-commit", default="5ce0193bc93e73b1c7f1f53aeda8854e997011e2"
    )
    parser.add_argument("--sure-lr", type=float, default=0.1)
    parser.add_argument("--sure-momentum", type=float, default=0.9)
    parser.add_argument("--sure-weight-decay", type=float, default=5e-4)
    parser.add_argument("--sure-mixup-beta", type=float, default=10.0)
    parser.add_argument("--sure-mixup-weight", type=float, default=0.5)
    parser.add_argument("--sure-crl-weight", type=float, default=0.5)
    parser.add_argument("--sure-rho", type=float, default=0.05)
    parser.add_argument("--sure-cos-temperature", type=float, default=8.0)
    parser.add_argument("--sure-swa-start-ratio", type=float, default=0.6)
    parser.add_argument("--sure-swa-lr", type=float, default=0.05)
    parser.add_argument("--dmcl-checkout", default="reference/_local/Sonar-OLTR")
    parser.add_argument(
        "--dmcl-commit", default="eea8dc07ce007988150ac208cd09e00daedba2ca"
    )
    parser.add_argument(
        "--dmcl-archive-sha256",
        default="4bd5158c491821bb1de3138856344949ca3ce1747f033601809d30774e7d5a61",
    )
    parser.add_argument("--dmcl-lr", type=float, default=0.01)
    parser.add_argument("--dmcl-momentum", type=float, default=0.9)
    parser.add_argument("--dmcl-weight-decay", type=float, default=0.001)
    parser.add_argument("--dmcl-lambda-contrast", type=float, default=0.5)
    parser.add_argument("--dmcl-lambda-uncertainty", type=float, default=0.1)
    parser.add_argument("--dmcl-margin-min", type=float, default=0.2)
    parser.add_argument("--dmcl-margin-max", type=float, default=0.8)
    parser.add_argument("--plud-lr", type=float, default=0.01)
    parser.add_argument("--plud-classifier-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--plud-momentum", type=float, default=0.0)
    parser.add_argument("--plud-weight-decay", type=float, default=0.001)
    parser.add_argument("--plud-alpha", type=float, default=1.5)
    parser.add_argument("--plud-gamma", type=float, default=0.5)
    parser.add_argument("--arch", default="mobilenet_v2",
                        help="backbone name (mobilenet_v2 / shufflenet_v2 / efficientnet_b0 / simplecnn)")
    parser.add_argument("--pretrained", action="store_true",
                        help="use grayscale-adapted ImageNet weights (backbones only)")
    parser.add_argument("--candidate-path", default=None,
                        help="ArchitectureSpec candidate artifact; overrides --arch")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable CUDA automatic mixed precision (ignored on CPU)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--geometry-mode",
        choices=("stretch_224", "letterbox_224", "fixed_scale_pad_224"),
        default="stretch_224",
    )
    parser.add_argument("--fixed-scale-factor", type=float, default=None)
    parser.add_argument("--geometry-padding-value", type=int, default=0)
    parser.add_argument(
        "--augmentation-profile",
        choices=("frozen_strong", "none"),
        default="frozen_strong",
    )
    parser.add_argument(
        "--experiment-contract",
        default=None,
        help="frozen protocol YAML whose preprocessing and recipe fields are enforced",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--logit-adjust-tau", type=float, default=1.0,
                        help="0 disables logit adjustment (plain smoothed CE)")
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/protocol")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the selected inner-validation checkpoint (required for claimability)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed fold/seed records when the run fingerprint matches",
    )
    parser.add_argument(
        "--max-new-units",
        type=int,
        default=None,
        help=(
            "stop cleanly after this many newly trained fold-seed units; compatible "
            "resumed records do not count and this execution control is excluded from "
            "the experiment fingerprint"
        ),
    )
    parser.add_argument("--force", action="store_true",
                        help="replace incompatible run metadata instead of failing")
    parser.add_argument(
        "--selection-provenance",
        choices=("baseline_predeclared", "legacy_fold0_selected", "new_nested"),
        default=None,
        help="how the architecture/model family was selected",
    )
    args = parser.parse_args()

    if args.adapter_config and not Path(args.adapter_config).exists():
        raise FileNotFoundError(f"adapter config does not exist: {args.adapter_config}")
    staged_limit_reached(args.max_new_units, 0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    folds = parse_int_list(args.folds)
    seeds = parse_int_list(args.seeds)
    if len(folds) != len(set(folds)) or len(seeds) != len(set(seeds)):
        raise ValueError("folds and seeds must not contain duplicates")
    selection_provenance = args.selection_provenance or (
        "legacy_fold0_selected" if args.candidate_path else "baseline_predeclared"
    )
    if args.adapter_id == "sure_author_recipe" and args.paper_id == "project_internal":
        args.paper_id = "sure_2024"
    if args.adapter_id == "dmcl_author_loss" and args.paper_id == "project_internal":
        args.paper_id = "dmcl_sonar_oltr_2025"
    if args.adapter_id == "plud_author_loss" and args.paper_id == "project_internal":
        args.paper_id = "plud_sonar_oltr_2024"
    if args.method_id:
        method_id = args.method_id
    elif args.candidate_path:
        method_id = "nas_candidate"
    elif args.adapter_id == "sure_author_recipe":
        method_id = (
            f"sure_fmfp_crl_regmixup_{args.arch}_"
            f"{'pretrained' if args.pretrained else 'scratch'}"
        )
    elif args.adapter_id == "dmcl_author_loss":
        method_id = (
            f"dmcl_author_loss_{args.arch}_"
            f"{'pretrained' if args.pretrained else 'scratch'}"
        )
    elif args.adapter_id == "plud_author_loss":
        method_id = (
            f"plud_author_loss_{args.arch}_"
            f"{'pretrained' if args.pretrained else 'scratch'}"
        )
    else:
        method_id = f"{args.arch}_{'pretrained' if args.pretrained else 'scratch'}"
    run_name = args.run_name or (
        f"protocol_{args.arch if not args.candidate_path else 'candidate'}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    environment_card = environment_card_provenance(args)
    source_freeze = source_freeze_provenance(args.source_freeze_manifest)
    experiment_contract = None
    experiment_contract_ref = None
    if args.experiment_contract:
        contract_path, experiment_contract = load_experiment_contract(
            args.experiment_contract
        )
        if experiment_contract.get("status") != "FROZEN":
            raise ValueError("formal evaluation requires a FROZEN experiment contract")
        binding_path = Path(
            experiment_contract["source_freeze_binding"]["binding_summary_path"]
        )
        if not binding_path.is_absolute():
            binding_path = (Path.cwd() / binding_path).resolve()
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("status") != "PASS":
            raise ValueError("Protocol V2 source-freeze binding is not PASS")
        if binding.get("contract", {}).get("sha256") != sha256_file(contract_path):
            raise ValueError("Protocol V2 contract hash conflicts with freeze binding")
        if source_freeze is None or source_freeze.get("verification_status") != "PASS":
            raise ValueError("formal Protocol V2 evaluation requires verified source freeze")
        if (
            binding.get("source_freeze", {}).get("manifest_sha256")
            != source_freeze.get("manifest_sha256")
        ):
            raise ValueError(
                "runtime source-freeze manifest conflicts with Protocol V2 binding"
            )
        experiment_contract_ref = experiment_contract_provenance(
            contract_path, experiment_contract
        )
        experiment_contract_ref["source_freeze_binding"] = {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
            "status": binding["status"],
        }

    recipe = RecipeConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        label_smoothing=args.label_smoothing,
        logit_adjust_tau=args.logit_adjust_tau,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        amp=args.amp,
    )
    sure_recipe = None
    sure_components = None
    dmcl_recipe = None
    dmcl_components = None
    plud_recipe = None
    plud_components = None
    if args.adapter_id == "sure_author_recipe":
        if args.task != "closed_set":
            raise ValueError("SURE failure-prediction recipe is only enabled for closed_set")
        sure_recipe = SureAuthorRecipeConfig(
            epochs=args.epochs,
            lr=args.sure_lr,
            momentum=args.sure_momentum,
            weight_decay=args.sure_weight_decay,
            mixup_beta=args.sure_mixup_beta,
            mixup_weight=args.sure_mixup_weight,
            crl_weight=args.sure_crl_weight,
            rho=args.sure_rho,
            cosine_temperature=args.sure_cos_temperature,
            swa_start_ratio=args.sure_swa_start_ratio,
            swa_lr=args.sure_swa_lr,
        )
        sure_recipe.validate()
        sure_components = load_sure_author_components(
            args.sure_checkout,
            pinned_commit=args.sure_commit,
        )
    elif args.adapter_id == "dmcl_author_loss":
        if args.task != "open_long_tail":
            raise ValueError("DMCL author loss is only enabled for open_long_tail")
        dmcl_recipe = DmclAuthorRecipeConfig(
            epochs=args.epochs,
            lr=args.dmcl_lr,
            momentum=args.dmcl_momentum,
            weight_decay=args.dmcl_weight_decay,
            lambda_contrast=args.dmcl_lambda_contrast,
            lambda_uncertainty=args.dmcl_lambda_uncertainty,
            margin_min=args.dmcl_margin_min,
            margin_max=args.dmcl_margin_max,
        )
        dmcl_recipe.validate()
        dmcl_components = load_dmcl_author_components(
            args.dmcl_checkout,
            pinned_commit=args.dmcl_commit,
            archive_sha256=args.dmcl_archive_sha256,
        )
    elif args.adapter_id == "plud_author_loss":
        if args.task != "open_long_tail":
            raise ValueError("PLUD author loss is only enabled for open_long_tail")
        plud_recipe = PludAuthorRecipeConfig(
            epochs=args.epochs,
            lr=args.plud_lr,
            classifier_lr_multiplier=args.plud_classifier_lr_multiplier,
            momentum=args.plud_momentum,
            weight_decay=args.plud_weight_decay,
            alpha=args.plud_alpha,
            gamma=args.plud_gamma,
        )
        plud_recipe.validate()
        plud_components = load_plud_author_components(
            args.dmcl_checkout,
            pinned_commit=args.dmcl_commit,
            archive_sha256=args.dmcl_archive_sha256,
        )

    adapter_source = None
    if sure_components is not None:
        adapter_source = {
            "checkout": str(sure_components.checkout),
            "commit": sure_components.commit,
            "source_hashes": sure_components.source_hashes,
            "license_state": "missing_unverified_local_execution_only",
        }
    elif dmcl_components is not None:
        adapter_source = {
            "checkout": str(dmcl_components.checkout),
            "code_root": str(dmcl_components.code_root),
            "commit": dmcl_components.commit,
            "archive_sha256": dmcl_components.archive_sha256,
            "source_hashes": dmcl_components.source_hashes,
            "license_state": "missing_unverified_local_execution_only",
        }
    elif plud_components is not None:
        adapter_source = {
            "checkout": str(plud_components.checkout),
            "code_root": str(plud_components.code_root),
            "commit": plud_components.commit,
            "archive_sha256": plud_components.archive_sha256,
            "source_hashes": plud_components.source_hashes,
            "license_state": "missing_unverified_local_execution_only",
            "paper_code_correspondence": "verified_for_plud_2024",
        }
    active_recipe = (
        sure_recipe.to_dict()
        if sure_recipe
        else dmcl_recipe.to_dict()
        if dmcl_recipe
        else plud_recipe.to_dict()
        if plud_recipe
        else recipe.to_dict()
    )
    if experiment_contract is not None:
        if args.task != "closed_set" or args.adapter_id != "builtin":
            raise ValueError(
                "the frozen NKSID experiment contract is only supported for "
                "builtin closed-set evaluation"
            )
        validate_formal_values_against_contract(
            contract=experiment_contract,
            observed_dataset={
                "image_size": args.image_size,
                "input_channels": 1,
                "geometry_mode": args.geometry_mode,
                "fixed_scale_factor": args.fixed_scale_factor,
                "geometry_padding_value": args.geometry_padding_value,
                "augmentation_profile": args.augmentation_profile,
                "inner_val_fraction": args.inner_val_fraction,
            },
            observed_recipe={
                **active_recipe,
                "scheduler": "cosine_with_warmup",
                "batch_size": args.batch_size,
            },
        )

    data_provenance = dataset_provenance(args.data_dir)
    candidate_provenance = None
    if args.candidate_path:
        candidate_path = Path(args.candidate_path).expanduser().resolve()
        candidate_provenance = {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        }
    code_provenance = git_code_provenance(run_dir)
    protocol_name = (
        str(experiment_contract["protocol"])
        if experiment_contract is not None
        else "nksid_outer5fold_inner_contiguous_v1"
        if args.task == "closed_set"
        else "nksid_open_long_tail_5known_3unknown_v1"
    )
    open_set_config = None
    if args.task == "open_long_tail":
        open_set_path = Path(args.open_set_config).resolve()
        open_set_config = {
            "path": str(open_set_path),
            "sha256": sha256_file(open_set_path),
        }
    immutable_config = {
        "protocol": protocol_name,
        "task": args.task,
        "adapter_id": args.adapter_id,
        "adapter_source": adapter_source,
        "adapter_config": (
            None
            if not args.adapter_config
            else {
                "path": str(Path(args.adapter_config).resolve()),
                "sha256": sha256_file(args.adapter_config),
            }
        ),
        "campaign_id": args.campaign_id,
        "paper_id": args.paper_id,
        "method_id": method_id,
        "open_set_config": open_set_config,
        "folds": folds,
        "seeds": seeds,
        "recipe": active_recipe,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "geometry": {
            "mode": args.geometry_mode,
            "fixed_scale_factor": args.fixed_scale_factor,
            "padding_value": args.geometry_padding_value,
            "rounding": "round_half_up",
            "alignment": "center_floor_top_left",
        },
        "augmentation_profile": args.augmentation_profile,
        "experiment_contract": experiment_contract_ref,
        "normalization": protocol_normalization(output_channels=1),
        "group_split_available": False,
        "group_generalization_claimable": False,
        "inner_val_fraction": args.inner_val_fraction,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "candidate": candidate_provenance,
        "selection_provenance": selection_provenance,
        "dataset": data_provenance,
        "runtime": runtime_provenance(),
        "environment_card": environment_card,
        "source_freeze": source_freeze,
        "code_state_sha256": code_provenance["code_state_sha256"],
    }
    run_fingerprint = canonical_sha256(immutable_config)
    protocol_context_sha256 = canonical_sha256(
        {key: value for key, value in immutable_config.items() if key != "candidate"}
    )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not args.force:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_fingerprint = existing_manifest.get("run_fingerprint")
        if existing_fingerprint != run_fingerprint:
            completed_records = list(run_dir.glob("run_fold*_seed*.json"))
            if completed_records:
                raise RuntimeError(
                    f"Refusing to resume incompatible run {run_dir}: "
                    f"{existing_fingerprint} != {run_fingerprint}. "
                    "Use --force or a new --run-name."
                )
            print(
                "[protocol] replacing incompatible empty manifest; "
                "no fold/seed records exist"
            )
    manifest = {
        "protocol": protocol_name,
        "run_name": run_name,
        "run_fingerprint": run_fingerprint,
        "protocol_context_sha256": protocol_context_sha256,
        "created_or_checked": datetime.now().isoformat(timespec="seconds"),
        "git_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
        "immutable_config": immutable_config,
        "data_protocol": {
            "group_split_available": False,
            "group_generalization_claimable": False,
            "group_metadata_source": None,
            "normalization": immutable_config["normalization"],
            "geometry": immutable_config["geometry"],
            "augmentation_profile": args.augmentation_profile,
            "experiment_contract": experiment_contract_ref,
        },
        "planned_pairs": [
            {"fold": fold, "seed": seed} for fold in folds for seed in seeds
        ],
        "completed_pairs": [],
        "execution_control": {
            "max_new_units": args.max_new_units,
            "resume": args.resume,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runs: list[dict] = []
    class_names: list[str] | None = None
    new_units_completed = 0
    stage_limit_hit = False
    for fold in folds:
        for seed in seeds:
            if staged_limit_reached(args.max_new_units, new_units_completed):
                stage_limit_hit = True
                break
            run_tag = f"fold{fold}_seed{seed}"
            record_path = run_dir / f"run_{run_tag}.json"
            checkpoint_path = run_dir / f"best_{run_tag}.pt"
            if args.resume and record_path.exists():
                existing_record = json.loads(record_path.read_text(encoding="utf-8"))
                checkpoint_ready = not args.save_checkpoints or checkpoint_path.exists()
                if (
                    existing_record.get("run_fingerprint") == run_fingerprint
                    and checkpoint_ready
                ):
                    print(f"\n=== fold {fold} seed {seed}: resume existing record ===")
                    runs.append(existing_record)
                    continue
            print(f"\n=== fold {fold} seed {seed} ===")
            unit_started_at = begin_unit_runtime_measurement(device)
            set_global_seed(seed)
            loader_kwargs = {
                "fold": fold,
                "seed": seed,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "inner_val_fraction": args.inner_val_fraction,
                "num_workers": args.num_workers,
            }
            if args.task == "closed_set":
                loader_kwargs.update(
                    {
                        "geometry_mode": args.geometry_mode,
                        "fixed_scale_factor": args.fixed_scale_factor,
                        "geometry_padding_value": args.geometry_padding_value,
                        "augmentation_profile": args.augmentation_profile,
                    }
                )
                bundle = create_protocol_dataloaders(args.data_dir, **loader_kwargs)
            else:
                bundle = create_open_set_protocol_dataloaders(
                    args.data_dir,
                    protocol_path=args.open_set_config,
                    **loader_kwargs,
                )
            num_classes = bundle["num_classes"]
            class_names = bundle["classes"]
            model, model_meta = build_run_model(args, num_classes)
            if sure_components is not None and sure_recipe is not None:
                model_meta["sure_cosine_classifier"] = apply_author_cosine_classifier(
                    model,
                    components=sure_components,
                    num_classes=num_classes,
                    temperature=sure_recipe.cosine_temperature,
                )
                model_meta["paper_code"] = immutable_config["adapter_source"]
            elif dmcl_components is not None and dmcl_recipe is not None:
                model_meta["paper_code"] = immutable_config["adapter_source"]
                model_meta["dmcl_feature_bridge"] = "input_to_final_linear_classifier"
            elif plud_components is not None and plud_recipe is not None:
                model_meta["paper_code"] = immutable_config["adapter_source"]
                model_meta["plud_feature_bridge"] = "input_to_final_linear_classifier"
                model_meta["plud_portability_deviation"] = (
                    "author randperm(...).cuda() resolved to configured device"
                )

            if sure_components is not None and sure_recipe is not None:
                result = train_with_sure_author_recipe(
                    model,
                    train_loader=bundle["train_loader"],
                    inner_val_loader=bundle["inner_val_loader"],
                    num_classes=num_classes,
                    recipe=sure_recipe,
                    components=sure_components,
                    device=device,
                )
            elif dmcl_components is not None and dmcl_recipe is not None:
                result = train_with_dmcl_author_loss(
                    model,
                    train_loader=bundle["train_loader"],
                    inner_val_loader=bundle["inner_val_loader"],
                    num_classes=num_classes,
                    recipe=dmcl_recipe,
                    components=dmcl_components,
                    device=device,
                )
            elif plud_components is not None and plud_recipe is not None:
                result = train_with_plud_author_loss(
                    model,
                    train_loader=bundle["train_loader"],
                    inner_val_loader=bundle["inner_val_loader"],
                    num_classes=num_classes,
                    recipe=plud_recipe,
                    components=plud_components,
                    device=device,
                )
            else:
                result = train_with_recipe(
                    model,
                    train_loader=bundle["train_loader"],
                    inner_val_loader=bundle["inner_val_loader"],
                    num_classes=num_classes,
                    recipe=recipe,
                    device=device,
                    class_counts=bundle["train_class_counts"].tolist(),
                )

            model.load_state_dict(result.best_state)
            model = model.to(device)
            threshold_metadata = None
            if args.task == "closed_set":
                outer_summary, outer_prediction_rows = evaluate_outer_classifier_with_predictions(
                    model,
                    bundle["outer_val_loader"],
                    criterion=nn.CrossEntropyLoss().to(device),
                    device=device,
                    num_classes=num_classes,
                    topk=(
                        sure_recipe.topk
                        if sure_recipe
                        else dmcl_recipe.topk
                        if dmcl_recipe
                        else plud_recipe.topk
                        if plud_recipe
                        else recipe.topk
                    ),
                    eval_samples=bundle["eval_dataset"].samples,
                    outer_indices=list(bundle["split"].outer_val_indices),
                    fold=fold,
                    seed=seed,
                    class_names=class_names,
                )
            else:
                threshold, threshold_metadata = calibrate_msp_threshold(
                    model,
                    bundle["inner_val_loader"],
                    device=device,
                    confidence_quantile=bundle["open_set_spec"].confidence_quantile,
                )
                outer_summary, outer_prediction_rows = evaluate_open_set_classifier(
                    model,
                    bundle["outer_val_loader"],
                    device=device,
                    eval_samples=bundle["eval_dataset"].samples,
                    outer_indices=bundle["outer_indices"],
                    spec=bundle["open_set_spec"],
                    threshold=threshold,
                    fold=fold,
                    seed=seed,
                )

            split_payload = bundle["split"].to_dict()
            split_sha256 = canonical_sha256(split_payload)
            record = {
                "task": args.task,
                "fold": fold,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "inner_val": {
                    key: result.best_inner_eval.get(key)
                    for key in ("macro_f1", "top1", "weighted_f1", "loss")
                },
                "outer_val": (
                    {
                        key: outer_summary[key]
                        for key in ("macro_f1", "top1", "weighted_f1", "top5", "loss")
                    }
                    if args.task == "closed_set"
                    else {
                        key: outer_summary[key]
                        for key in (
                            "known_macro_f1",
                            "nma",
                            "osfm",
                            "oscr_mac",
                            "unknown_auroc",
                            "unknown_fpr95",
                        )
                    }
                ),
                "outer_calibration": outer_summary.get("calibration"),
                "outer_confusion_matrix": outer_summary.get("confusion_matrix"),
                "outer_per_class_f1": (
                    per_class_f1(outer_summary["confusion_matrix"])
                    if "confusion_matrix" in outer_summary
                    else None
                ),
                "open_set_spec": (
                    bundle["open_set_spec"].to_dict()
                    if args.task == "open_long_tail"
                    else None
                ),
                "open_set_threshold": threshold_metadata,
                "split": split_payload,
                "model": model_meta,
                "adapter_source": immutable_config["adapter_source"],
                "selection_provenance": selection_provenance,
                "normalization": immutable_config["normalization"],
                "geometry": immutable_config["geometry"],
                "augmentation_profile": args.augmentation_profile,
                "experiment_contract": experiment_contract_ref,
                "group_split_available": False,
                "group_generalization_claimable": False,
                "runtime_measurement": finish_unit_runtime_measurement(
                    unit_started_at, device
                ),
                "run_fingerprint": run_fingerprint,
                "protocol_context_sha256": protocol_context_sha256,
                "split_sha256": split_sha256,
                "provenance": {
                    "git_commit": manifest["git_commit"],
                    "code_state_sha256": code_provenance["code_state_sha256"],
                    "dataset": data_provenance,
                    "candidate": candidate_provenance,
                    "source_freeze": source_freeze,
                },
            }
            if args.save_checkpoints:
                checkpoint_payload = {
                    "schema_version": 2,
                    "protocol": protocol_name,
                    "task": args.task,
                    "fold": fold,
                    "seed": seed,
                    "model_state_dict": result.best_state,
                    "architecture": model_meta.get("architecture"),
                    "source_candidate": candidate_provenance,
                    "model": model_meta,
                    "metrics": record["outer_val"],
                    "best_epoch": result.best_epoch,
                    "selection_provenance": selection_provenance,
                    "run_fingerprint": run_fingerprint,
                    "protocol_context_sha256": protocol_context_sha256,
                    "split_sha256": record["split_sha256"],
                    "normalization": immutable_config["normalization"],
                    "geometry": immutable_config["geometry"],
                    "augmentation_profile": args.augmentation_profile,
                    "experiment_contract": experiment_contract_ref,
                    "group_split_available": False,
                    "group_generalization_claimable": False,
                    "source_freeze": source_freeze,
                }
                torch.save(checkpoint_payload, checkpoint_path)
                record["checkpoint"] = {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": sha256_file(checkpoint_path),
                    "schema_version": 2,
                }
            checkpoint_sha = (record.get("checkpoint") or {}).get("sha256", "NOT_SAVED")
            data_sha = canonical_sha256(data_provenance)
            evaluated_code_commit = (
                sure_components.commit
                if sure_components is not None
                else dmcl_components.commit
                if dmcl_components is not None
                else plud_components.commit
                if plud_components is not None
                else manifest["git_commit"]
            )
            for row in outer_prediction_rows:
                row.update(
                    {
                        "campaign_id": args.campaign_id,
                        "paper_id": args.paper_id,
                        "method": method_id,
                        "checkpoint_sha": checkpoint_sha,
                        "config_sha": run_fingerprint,
                        "data_sha": data_sha,
                        "split_sha": split_sha256,
                        "code_commit": evaluated_code_commit,
                        "project_code_commit": manifest["git_commit"],
                        "code_state_sha": code_provenance["code_state_sha256"],
                        "claimability_status": "PENDING",
                    }
                )
            prediction_path = run_dir / f"outer_predictions_{run_tag}.jsonl"
            write_jsonl(prediction_path, outer_prediction_rows)
            record["outer_predictions"] = {
                "path": str(prediction_path.resolve()),
                "sha256": sha256_file(prediction_path),
                "num_samples": len(outer_prediction_rows),
                "schema_version": 2 if args.save_checkpoints else "2-incomplete-no-checkpoint",
            }
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            runs.append(record)
            new_units_completed += 1
            if args.task == "closed_set":
                print(
                    f"fold {fold} seed {seed}: outer macro_f1={outer_summary['macro_f1']:.4f} "
                    f"top1={outer_summary['top1']:.4f} (best epoch {result.best_epoch})"
                )
            else:
                print(
                    f"fold {fold} seed {seed}: known macro_f1="
                    f"{outer_summary['known_macro_f1']:.4f} OSCRmac="
                    f"{outer_summary['oscr_mac']:.4f} (best epoch {result.best_epoch})"
                )
        if stage_limit_hit:
            break

    class_names = class_names or list(NKSID_CLASSES)
    if args.task == "closed_set":
        claimability = protocol_claimability(
            folds=folds,
            seeds=seeds,
            completed_pairs=[(r["fold"], r["seed"]) for r in runs],
            selection_provenance=selection_provenance,
            outer_validation_used_for_selection=False,
            provenance_complete=(
                args.save_checkpoints
                and bool(manifest["git_commit"])
                and len(str(code_provenance.get("code_state_sha256", ""))) == 64
                and all(
                    len(str((r.get("checkpoint") or {}).get("sha256", ""))) == 64
                    and len(str(r.get("split_sha256", ""))) == 64
                    for r in runs
                )
            ),
            provenance_fingerprints=[str(r.get("run_fingerprint", "")) for r in runs],
            group_split_available=False,
            protocol_context_sha256=protocol_context_sha256,
            provenance_contexts=[str(r.get("protocol_context_sha256", "")) for r in runs],
            source_freeze_verified=bool(
                source_freeze and source_freeze.get("verification_status") == "PASS"
            ),
        )
    else:
        claimability = protocol_claimability(
            folds=folds,
            seeds=seeds,
            completed_pairs=[(r["fold"], r["seed"]) for r in runs],
            selection_provenance=selection_provenance,
            outer_validation_used_for_selection=False,
            provenance_complete=(
                args.save_checkpoints
                and bool(manifest["git_commit"])
                and len(str(code_provenance.get("code_state_sha256", ""))) == 64
                and all(
                    len(str((r.get("checkpoint") or {}).get("sha256", ""))) == 64
                    and len(str(r.get("split_sha256", ""))) == 64
                    for r in runs
                )
                and (
                    args.adapter_id == "builtin"
                    or bool(environment_card and adapter_source)
                )
            ),
            provenance_fingerprints=[str(r.get("run_fingerprint", "")) for r in runs],
            group_split_available=False,
            protocol_context_sha256=protocol_context_sha256,
            provenance_contexts=[str(r.get("protocol_context_sha256", "")) for r in runs],
            source_freeze_verified=bool(
                source_freeze and source_freeze.get("verification_status") == "PASS"
            ),
        )
        claimability["claim_scope"] = (
            "frozen_open_long_tail_protocol_evaluation"
            if claimability["claimable"]
            else "open_long_tail_experimental_pending_protocol_completion"
        )
        claimability["nas_generalization_claimable"] = False
        if not claimability["claimable"]:
            claimability["warnings"].append(
                "Open-set metrics remain PENDING until all 15 frozen fold-seed units and "
                "their checkpoint, adapter, environment, and source-freeze provenance "
                "are complete."
            )
    manifest["completed_pairs"] = [
        {"fold": r["fold"], "seed": r["seed"]} for r in runs
    ]
    manifest["execution_control"].update(
        {
            "new_units_completed": new_units_completed,
            "stage_limit_hit": stage_limit_hit,
        }
    )
    manifest["claimability"] = claimability
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    aggregate = {
        "protocol": protocol_name,
        "task": args.task,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "device": device,
        "execution_control": manifest["execution_control"],
        "run_fingerprint": run_fingerprint,
        "protocol_context_sha256": protocol_context_sha256,
        "normalization": immutable_config["normalization"],
        "group_split_available": False,
        "group_generalization_claimable": False,
        "run_fingerprints": sorted({str(r.get("run_fingerprint", "")) for r in runs}),
        "claimability": claimability,
        "provenance": {
            "git_commit": manifest["git_commit"],
            "code": code_provenance,
            "dataset": data_provenance,
            "candidate": candidate_provenance,
            "source_freeze": source_freeze,
            "experiment_contract": experiment_contract_ref,
            "manifest": str(manifest_path.resolve()),
        },
        "recipe": active_recipe,
        "model": runs[0]["model"] if runs else None,
        "folds": folds,
        "seeds": seeds,
        "outer_macro_f1": summarize(
            [
                r["outer_val"][
                    "macro_f1" if args.task == "closed_set" else "known_macro_f1"
                ]
                for r in runs
            ]
        ),
        "outer_top1": (
            summarize([r["outer_val"]["top1"] for r in runs])
            if args.task == "closed_set"
            else None
        ),
        "outer_weighted_f1": (
            summarize([r["outer_val"]["weighted_f1"] for r in runs])
            if args.task == "closed_set"
            else None
        ),
        "open_set_metrics": (
            {
                key: summarize([float(r["outer_val"][key]) for r in runs])
                for key in (
                    "known_macro_f1",
                    "nma",
                    "osfm",
                    "oscr_mac",
                    "unknown_auroc",
                    "unknown_fpr95",
                )
            }
            if args.task == "open_long_tail"
            else None
        ),
        "per_class_f1_mean": (
            [
                statistics.fmean(r["outer_per_class_f1"][idx] for r in runs)
                for idx in range(len(class_names))
            ]
            if runs and args.task == "closed_set"
            else []
        ),
        "class_names": class_names,
        "runs": [
            {key: r[key] for key in ("fold", "seed", "best_epoch", "inner_val", "outer_val")}
            for r in runs
        ],
    }
    with (run_dir / "protocol_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    lines = [
        f"# Protocol summary: {run_name}",
        "",
        f"- protocol: `{protocol_name}`",
        f"- task: `{args.task}`",
        f"- adapter: `{args.adapter_id}`; paper: `{args.paper_id}`; method: `{method_id}`",
        f"- folds: {folds}, seeds: {seeds}, epochs: {recipe.epochs}, device: {device}",
        f"- claimable: `{claimability['claimable']}`",
        f"- claim_scope: `{claimability['claim_scope']}`",
        f"- nas_generalization_claimable: `{claimability['nas_generalization_claimable']}`",
        f"- group_split_available: `False`",
        f"- group_generalization_claimable: `False`",
        f"- normalization: `grayscale mean=0.5 std=0.5`",
        f"- model: {aggregate['model']}",
        "",
        "| metric | mean | std | n |",
        "|---|---:|---:|---:|",
    ]
    metric_keys = (
        ("outer_macro_f1", "outer_top1", "outer_weighted_f1")
        if args.task == "closed_set"
        else tuple((aggregate["open_set_metrics"] or {}).keys())
    )
    for key in metric_keys:
        stats = (
            aggregate[key]
            if args.task == "closed_set"
            else aggregate["open_set_metrics"][key]
        )
        if stats and stats["mean"] is not None:
            lines.append(f"| {key} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} |")
    if args.task == "closed_set":
        lines += ["", "| class | mean outer F1 |", "|---|---:|"]
        for name, value in zip(class_names, aggregate["per_class_f1_mean"]):
            lines.append(f"| {name} | {value:.4f} |")
    if claimability["warnings"]:
        lines += ["", "## Claim warnings", ""]
        lines.extend(f"- {warning}" for warning in claimability["warnings"])
    (run_dir / "protocol_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSummary written to {run_dir / 'protocol_summary.json'}")
    macro = aggregate["outer_macro_f1"]
    if macro["mean"] is not None:
        metric_label = "outer macro_f1" if args.task == "closed_set" else "outer known_macro_f1"
        print(
            f"{metric_label} = {macro['mean']:.4f} +/- {macro['std']:.4f} "
            f"(n={macro['n']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
