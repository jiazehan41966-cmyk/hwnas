#!/usr/bin/env python3
"""Execute Protocol V2 inner-only bundle and geometry selection gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.dataset import (  # noqa: E402
    NKSIDDataset,
    create_dataloader,
)
from hwnas_fpga.data.protocol import (  # noqa: E402
    build_protocol_split,
    class_counts,
)
from hwnas_fpga.fourstage_operator import (  # noqa: E402
    build_fourstage_architecture,
)
from hwnas_fpga.fourstage_selection import (  # noqa: E402
    canonical_sha256,
    dhash64,
    group_stress_split,
    infer_hash_groups,
    paired_hierarchical_bootstrap,
)
from hwnas_fpga.models import build_model  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402
from hwnas_fpga.training.recipe import RecipeConfig, train_with_recipe  # noqa: E402
from hwnas_fpga.training.trainer import evaluate_classifier  # noqa: E402


DEFAULT_PREREG = (
    ROOT / "configs" / "evaluation" / "nksid_protocol_v2_selection_preregistered.yaml"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_state() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()

    status = output("status", "--porcelain=v1").splitlines()
    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "status_porcelain": status,
        "dirty": bool(status),
    }


def load_prereg(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("outer_validation_access") != "FORBIDDEN":
        raise ValueError("Protocol V2 selection must forbid outer validation access")
    return payload


def dataset_file_manifest(
    samples: list[tuple[str, int]],
    data_dir: Path,
) -> dict[str, Any]:
    rows = []
    for index, (path_value, label) in enumerate(samples):
        path = Path(path_value).resolve()
        rows.append(
            {
                "index": index,
                "path": str(path),
                "relative_path": path.relative_to(data_dir.resolve()).as_posix(),
                "label": int(label),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    split_files = {}
    for name in ("train_abs.txt", "kfold_train.txt", "kfold_val.txt"):
        path = data_dir / name
        split_files[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": 1,
        "dataset": "NKSID",
        "sample_count": len(rows),
        "samples": rows,
        "split_files": split_files,
        "content_sha256": canonical_sha256(
            {
                "samples": [
                    {
                        "index": row["index"],
                        "relative_path": row["relative_path"],
                        "label": row["label"],
                        "bytes": row["bytes"],
                        "sha256": row["sha256"],
                    }
                    for row in rows
                ],
                "split_files": split_files,
            }
        ),
    }


def build_group_manifest(
    *,
    data_dir: Path,
    output_dir: Path,
    threshold: int,
) -> dict[str, Any]:
    view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=224,
        is_training=False,
        use_kfold=False,
        split="full",
        augmentation_profile="none",
    )
    samples = list(view.samples)
    hashes = {index: dhash64(path) for index, (path, _) in enumerate(samples)}
    folds: dict[str, Any] = {}
    for fold in range(5):
        historical = build_protocol_split(
            samples,
            data_dir,
            fold_index=fold,
            seed=42,
            inner_val_fraction=0.15,
            num_classes=8,
        )
        outer_train = tuple(
            sorted(
                set(historical.train_indices)
                | set(historical.inner_val_indices)
            )
        )
        assignments = infer_hash_groups(
            samples,
            outer_train,
            hamming_threshold=threshold,
            precomputed_hashes=hashes,
        )
        group_sizes: dict[str, int] = defaultdict(int)
        for group_id in assignments.values():
            group_sizes[group_id] += 1
        folds[str(fold)] = {
            "outer_train_count": len(outer_train),
            "outer_validation_count": len(historical.outer_val_indices),
            "outer_validation_indices_sha256": canonical_sha256(
                list(historical.outer_val_indices)
            ),
            "assignments": {
                str(index): assignments[index] for index in sorted(assignments)
            },
            "group_count": len(group_sizes),
            "multi_sample_group_count": sum(
                size > 1 for size in group_sizes.values()
            ),
            "max_group_size": max(group_sizes.values(), default=0),
            "group_size_histogram": {
                str(size): sum(value == size for value in group_sizes.values())
                for size in sorted(set(group_sizes.values()))
            },
        }
    manifest = {
        "schema_version": 1,
        "generated_at": now(),
        "method": "same_class_dhash64_connected_components",
        "dhash": {
            "resize": [9, 8],
            "grayscale": "PIL_L",
            "interpolation": "PIL_LANCZOS",
            "comparison": "horizontal_neighbor_gt",
            "hamming_threshold": int(threshold),
        },
        "claim_boundary": (
            "These are inferred perceptual-hash near-duplicate clusters. "
            "NKSID has no real acquisition or mission group metadata."
        ),
        "sample_hashes_hex": {
            str(index): f"{value:016x}" for index, value in hashes.items()
        },
        "folds": folds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "inferred_hash_group_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data_manifest = dataset_file_manifest(samples, data_dir)
    data_path = output_dir / "dataset_manifest.json"
    data_path.write_text(
        json.dumps(data_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "status": "COMPLETE",
        "group_manifest": str(path.resolve()),
        "group_manifest_sha256": sha256_file(path),
        "dataset_manifest": str(data_path.resolve()),
        "dataset_manifest_sha256": sha256_file(data_path),
        "dataset_content_sha256": data_manifest["content_sha256"],
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def geometry_kwargs(prereg: Mapping[str, Any], geometry: str) -> dict[str, Any]:
    config = prereg["geometry_gate"]["candidates"][geometry]
    return {
        "geometry_mode": geometry,
        "fixed_scale_factor": (
            float(config["fixed_scale_factor"])
            if geometry == "fixed_scale_pad_224"
            else None
        ),
        "geometry_padding_value": int(config.get("padding_value_uint8", 0)),
    }


def recipe_from_bundle(
    prereg: Mapping[str, Any],
    bundle_name: str,
) -> RecipeConfig:
    bundle = prereg["bundle_gate"]["candidates"][bundle_name]
    training = prereg["training"]
    return RecipeConfig(
        epochs=int(training["epochs"]),
        optimizer=str(bundle["optimizer"]),
        lr=float(bundle["lr"]),
        weight_decay=float(bundle["weight_decay"]),
        warmup_epochs=int(bundle["warmup_epochs"]),
        min_lr_ratio=float(bundle["min_lr_ratio"]),
        label_smoothing=float(bundle["label_smoothing"]),
        logit_adjust_tau=float(bundle["logit_adjust_tau"]),
        selection_metric="macro_f1",
        early_stopping_patience=training.get("early_stopping_patience"),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        amp=bool(training["amp"]),
    )


def inner_indices(
    *,
    samples: list[tuple[str, int]],
    data_dir: Path,
    fold: int,
    seed: int,
    split_mode: str,
    group_manifest: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    historical = build_protocol_split(
        samples,
        data_dir,
        fold_index=fold,
        seed=seed,
        inner_val_fraction=0.15,
        num_classes=8,
    )
    outer_train = tuple(
        sorted(set(historical.train_indices) | set(historical.inner_val_indices))
    )
    outer_val = set(historical.outer_val_indices)
    if split_mode == "historical_inner":
        train_indices = historical.train_indices
        inner_val_indices = historical.inner_val_indices
        metadata = {
            "method": "per_class_contiguous_filename_block",
            "seed": seed,
            "fraction": 0.15,
        }
    elif split_mode == "inferred_stress_inner":
        assignments = {
            int(index): group_id
            for index, group_id in group_manifest["folds"][str(fold)][
                "assignments"
            ].items()
        }
        train_indices, inner_val_indices, metadata = group_stress_split(
            samples,
            outer_train,
            assignments,
            seed=seed,
            fraction=0.15,
        )
    else:
        raise ValueError(f"unsupported split_mode: {split_mode}")
    if set(train_indices) & outer_val or set(inner_val_indices) & outer_val:
        raise AssertionError("inner-only selection touched outer validation indices")
    if set(train_indices) | set(inner_val_indices) != set(outer_train):
        raise AssertionError("inner-only split does not cover outer-train")
    return train_indices, inner_val_indices, {
        **metadata,
        "outer_validation_accessed": False,
        "outer_validation_count": len(outer_val),
        "outer_validation_indices_sha256": canonical_sha256(sorted(outer_val)),
        "train_indices_sha256": canonical_sha256(list(train_indices)),
        "inner_val_indices_sha256": canonical_sha256(list(inner_val_indices)),
    }


def dataloaders(
    *,
    data_dir: Path,
    prereg: Mapping[str, Any],
    fold: int,
    seed: int,
    split_mode: str,
    geometry: str,
    bundle_name: str,
    group_manifest: Mapping[str, Any],
):
    augmentation = prereg["bundle_gate"]["candidates"][bundle_name][
        "augmentation_profile"
    ]
    geometry_values = geometry_kwargs(prereg, geometry)
    train_view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=224,
        is_training=True,
        use_kfold=False,
        split="full",
        augmentation_profile=augmentation,
        **geometry_values,
    )
    eval_view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=224,
        is_training=False,
        use_kfold=False,
        split="full",
        augmentation_profile="none",
        **geometry_values,
    )
    samples = list(eval_view.samples)
    train_indices, inner_val_indices, split_metadata = inner_indices(
        samples=samples,
        data_dir=data_dir,
        fold=fold,
        seed=seed,
        split_mode=split_mode,
        group_manifest=group_manifest,
    )
    batch_size = int(prereg["training"]["batch_size"])
    train_loader = create_dataloader(
        Subset(train_view, list(train_indices)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        seed=seed + fold * 100_000,
    )
    inner_loader = create_dataloader(
        Subset(eval_view, list(inner_val_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        seed=seed + fold * 100_000 + 1,
    )
    counts = class_counts(samples, train_indices, num_classes=8)
    return {
        "train_loader": train_loader,
        "inner_loader": inner_loader,
        "eval_view": eval_view,
        "train_indices": train_indices,
        "inner_val_indices": inner_val_indices,
        "class_counts": counts,
        "split_metadata": split_metadata,
        "transform_contract": {
            **geometry_values,
            "geometry_mode": geometry,
            "augmentation_profile": augmentation,
            "eval_augmentation_profile": "none",
            "image_size": 224,
        },
    }


@torch.no_grad()
def predict_inner(
    model: nn.Module,
    *,
    loader,
    indices: tuple[int, ...],
    eval_view: NKSIDDataset,
    device: str,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    cursor = 0
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        probabilities = torch.softmax(outputs, dim=1).cpu()
        predictions = outputs.argmax(dim=1).cpu()
        for offset in range(targets.shape[0]):
            sample_index = int(indices[cursor + offset])
            sample_path, dataset_label = eval_view.samples[sample_index]
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_path": str(Path(sample_path).resolve()),
                    "target": int(targets[offset]),
                    "dataset_label": int(dataset_label),
                    "prediction": int(predictions[offset]),
                    "probabilities": [
                        float(value) for value in probabilities[offset].tolist()
                    ],
                }
            )
        cursor += int(targets.shape[0])
    if cursor != len(indices):
        raise AssertionError("prediction count does not match inner split")
    return rows


def run_unit(
    *,
    data_dir: Path,
    prereg_path: Path,
    prereg: Mapping[str, Any],
    group_manifest_path: Path,
    group_manifest: Mapping[str, Any],
    output_dir: Path,
    bundle_name: str,
    geometry: str,
    split_mode: str,
    fold: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    tag = f"fold{fold}_seed{seed}"
    run_dir = output_dir / split_mode / geometry / bundle_name
    record_path = run_dir / f"run_{tag}.json"
    checkpoint_path = run_dir / f"best_{tag}.pt"
    prediction_path = run_dir / f"inner_predictions_{tag}.jsonl"
    unit_contract = {
        "source_git": git_state(),
        "preregistration_sha256": sha256_file(prereg_path),
        "group_manifest_sha256": sha256_file(group_manifest_path),
        "data_dir": str(data_dir.resolve()),
        "bundle": bundle_name,
        "geometry": geometry,
        "split_mode": split_mode,
        "fold": int(fold),
        "seed": int(seed),
        "architecture": build_fourstage_architecture(
            stage2_kernel=3,
            stage2_expansion=3,
            stage4_op="mbconv_k3_e3",
        ).to_dict(),
    }
    unit_fingerprint = canonical_sha256(unit_contract)
    if record_path.is_file() and checkpoint_path.is_file() and prediction_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            record.get("checkpoint", {}).get("sha256")
            == sha256_file(checkpoint_path)
            and record.get("outer_validation_accessed") is False
            and record.get("unit_fingerprint") == unit_fingerprint
        ):
            return record
        raise RuntimeError(
            f"refusing incompatible Protocol V2 selection resume: {record_path}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    bundle = dataloaders(
        data_dir=data_dir,
        prereg=prereg,
        fold=fold,
        seed=seed,
        split_mode=split_mode,
        geometry=geometry,
        bundle_name=bundle_name,
        group_manifest=group_manifest,
    )
    architecture = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="mbconv_k3_e3",
    )
    model = build_model(architecture=architecture, num_classes=8)
    recipe = recipe_from_bundle(prereg, bundle_name)
    result = train_with_recipe(
        model,
        train_loader=bundle["train_loader"],
        inner_val_loader=bundle["inner_loader"],
        num_classes=8,
        recipe=recipe,
        device=device,
        class_counts=bundle["class_counts"],
        verbose=True,
    )
    model.load_state_dict(result.best_state)
    model = model.to(device)
    inner_eval = evaluate_classifier(
        model,
        bundle["inner_loader"],
        criterion=nn.CrossEntropyLoss().to(device),
        device=device,
        num_classes=8,
        topk=5,
    )
    checkpoint_payload = {
        "schema_version": 1,
        "purpose": "protocol_v2_inner_only_selection",
        "outer_validation_accessed": False,
        "unit_fingerprint": unit_fingerprint,
        "unit_contract": unit_contract,
        "fold": fold,
        "seed": seed,
        "split_mode": split_mode,
        "bundle": bundle_name,
        "geometry": geometry,
        "architecture": architecture.to_dict(),
        "recipe": recipe.to_dict(),
        "best_epoch": result.best_epoch,
        "best_inner_eval": result.best_inner_eval,
        "model_state_dict": result.best_state,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    predictions = predict_inner(
        model,
        loader=bundle["inner_loader"],
        indices=bundle["inner_val_indices"],
        eval_view=bundle["eval_view"],
        device=device,
    )
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    record = {
        "schema_version": 1,
        "status": "COMPLETE",
        "purpose": "protocol_v2_inner_only_selection",
        "outer_validation_accessed": False,
        "unit_fingerprint": unit_fingerprint,
        "unit_contract": unit_contract,
        "fold": fold,
        "seed": seed,
        "split_mode": split_mode,
        "bundle": bundle_name,
        "geometry": geometry,
        "recipe": recipe.to_dict(),
        "transform_contract": bundle["transform_contract"],
        "split": bundle["split_metadata"],
        "class_counts": bundle["class_counts"],
        "best_epoch": result.best_epoch,
        "best_inner_eval": result.best_inner_eval,
        "confirmed_inner_eval": inner_eval,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "inner_predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "count": len(predictions),
        },
        "provenance": {
            "preregistration_path": str(prereg_path.resolve()),
            "preregistration_sha256": sha256_file(prereg_path),
            "group_manifest_path": str(group_manifest_path.resolve()),
            "group_manifest_sha256": sha256_file(group_manifest_path),
            "git": git_state(),
        },
        "completed_at": now(),
    }
    record["record_sha256"] = canonical_sha256(record)
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def summarize_pair(
    records_left: list[dict[str, Any]],
    records_right: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    left = {
        (int(row["fold"]), int(row["seed"])): float(
            row["confirmed_inner_eval"]["macro_f1"]
        )
        for row in records_left
    }
    right = {
        (int(row["fold"]), int(row["seed"])): float(
            row["confirmed_inner_eval"]["macro_f1"]
        )
        for row in records_right
    }
    if set(left) != set(right):
        raise ValueError("paired records do not cover identical fold/seed units")
    deltas: dict[int, list[float]] = defaultdict(list)
    for (fold, unit_seed), left_value in sorted(left.items()):
        deltas[fold].append(right[(fold, unit_seed)] - left_value)
    return {
        "left_mean": float(np.mean(list(left.values()))),
        "right_mean": float(np.mean(list(right.values()))),
        "right_minus_left": paired_hierarchical_bootstrap(
            deltas,
            iterations=10_000,
            seed=seed,
        ),
    }


def execute_matrix(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    prereg_path = Path(args.prereg).resolve()
    prereg = load_prereg(prereg_path)
    root = Path(args.output_dir).resolve()
    group_manifest_path = root / "manifests" / "inferred_hash_group_manifest.json"
    group_manifest = json.loads(group_manifest_path.read_text(encoding="utf-8"))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if phase == "bundle":
        matrix = [
            ("v1", "stretch_224", "historical_inner"),
            ("plain_ce", "stretch_224", "historical_inner"),
        ]
    elif phase == "geometry":
        decision = json.loads(
            (root / "bundle_decision.json").read_text(encoding="utf-8")
        )
        selected_bundle = str(decision["selected_bundle"])
        matrix = [
            (selected_bundle, geometry, split_mode)
            for split_mode in ("historical_inner", "inferred_stress_inner")
            for geometry in (
                "stretch_224",
                "letterbox_224",
                "fixed_scale_pad_224",
            )
        ]
    else:
        raise ValueError(phase)

    all_records: dict[str, list[dict[str, Any]]] = {}
    for bundle_name, geometry, split_mode in matrix:
        key = f"{split_mode}/{geometry}/{bundle_name}"
        rows = []
        for fold in range(5):
            for seed in (42, 43, 44):
                rows.append(
                    run_unit(
                        data_dir=Path(args.data_dir).resolve(),
                        prereg_path=prereg_path,
                        prereg=prereg,
                        group_manifest_path=group_manifest_path,
                        group_manifest=group_manifest,
                        output_dir=root / "runs",
                        bundle_name=bundle_name,
                        geometry=geometry,
                        split_mode=split_mode,
                        fold=fold,
                        seed=seed,
                        device=device,
                    )
                )
        all_records[key] = rows

    if phase == "bundle":
        comparison = summarize_pair(
            all_records["historical_inner/stretch_224/v1"],
            all_records["historical_inner/stretch_224/plain_ce"],
            seed=20260729,
        )
        delta = comparison["right_minus_left"]
        selected = (
            "plain_ce"
            if delta["mean_delta"] > 0.0
            and delta["positive_fold_means"] >= 4
            else "v1"
        )
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "outer_validation_accessed": False,
            "comparison": comparison,
            "decision_rule": prereg["bundle_gate"]["decision"],
            "selected_bundle": selected,
            "record_paths": {
                key: [
                    str(
                        (
                            root
                            / "runs"
                            / row["split_mode"]
                            / row["geometry"]
                            / row["bundle"]
                            / f"run_fold{row['fold']}_seed{row['seed']}.json"
                        ).resolve()
                    )
                    for row in rows
                ]
                for key, rows in all_records.items()
            },
        }
        target = root / "bundle_decision.json"
    else:
        decision = json.loads(
            (root / "bundle_decision.json").read_text(encoding="utf-8")
        )
        selected_bundle = str(decision["selected_bundle"])
        means: dict[str, dict[str, float]] = {}
        comparisons: dict[str, Any] = {}
        for split_mode in ("historical_inner", "inferred_stress_inner"):
            means[split_mode] = {}
            stretch_key = f"{split_mode}/stretch_224/{selected_bundle}"
            stretch_rows = all_records[stretch_key]
            for index, geometry in enumerate(
                ("stretch_224", "letterbox_224", "fixed_scale_pad_224")
            ):
                key = f"{split_mode}/{geometry}/{selected_bundle}"
                means[split_mode][geometry] = float(
                    np.mean(
                        [
                            row["confirmed_inner_eval"]["macro_f1"]
                            for row in all_records[key]
                        ]
                    )
                )
                if geometry != "stretch_224":
                    comparisons[f"{split_mode}:{geometry}_vs_stretch"] = summarize_pair(
                        stretch_rows,
                        all_records[key],
                        seed=20260730 + index,
                    )
        consistent = {}
        for geometry in ("letterbox_224", "fixed_scale_pad_224"):
            signs = [
                np.sign(
                    comparisons[f"{split_mode}:{geometry}_vs_stretch"][
                        "right_minus_left"
                    ]["mean_delta"]
                )
                for split_mode in (
                    "historical_inner",
                    "inferred_stress_inner",
                )
            ]
            consistent[geometry] = bool(signs[0] == signs[1])
        eligible = ["stretch_224"] + [
            geometry
            for geometry in ("letterbox_224", "fixed_scale_pad_224")
            if consistent[geometry]
        ]
        selected_geometry = max(
            eligible,
            key=lambda geometry: min(
                means["historical_inner"][geometry],
                means["inferred_stress_inner"][geometry],
            ),
        )
        payload = {
            "schema_version": 1,
            "status": "COMPLETE",
            "outer_validation_accessed": False,
            "selected_bundle": selected_bundle,
            "means": means,
            "comparisons": comparisons,
            "direction_consistent_vs_stretch": consistent,
            "eligible_geometries": eligible,
            "decision_rule": prereg["geometry_gate"]["decision"],
            "selected_geometry": selected_geometry,
        }
        target = root / "geometry_decision.json"
    payload["generated_at"] = now()
    payload["payload_sha256"] = canonical_sha256(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def finalize_protocol(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = Path(args.prereg).resolve()
    prereg = load_prereg(prereg_path)
    root = Path(args.output_dir).resolve()
    bundle = json.loads((root / "bundle_decision.json").read_text(encoding="utf-8"))
    geometry = json.loads(
        (root / "geometry_decision.json").read_text(encoding="utf-8")
    )
    manifest_summary = json.loads(
        (root / "manifests" / "manifest_summary.json").read_text(encoding="utf-8")
    )
    selected_bundle = str(bundle["selected_bundle"])
    selected_geometry = str(geometry["selected_geometry"])
    bundle_values = prereg["bundle_gate"]["candidates"][selected_bundle]
    geometry_values = geometry_kwargs(prereg, selected_geometry)
    payload = {
        "schema_version": 2,
        "protocol": "nksid_outer5fold_inner_only_v2",
        "status": "FROZEN",
        "dataset": {
            "name": "NKSID",
            "image_size": 224,
            "input_channels": 1,
            "geometry_mode": selected_geometry,
            "fixed_scale_factor": geometry_values["fixed_scale_factor"],
            "geometry_padding_value": geometry_values[
                "geometry_padding_value"
            ],
            "augmentation_profile": bundle_values["augmentation_profile"],
            "inner_val_fraction": 0.15,
            "outer_folds": [0, 1, 2, 3, 4],
            "outer_validation_role": "single_final_evaluation_only",
            "group_metadata_available": False,
            "inferred_group_claim_boundary": (
                "Perceptual-hash groups are near-duplicate stress diagnostics, "
                "not acquisition or mission groups."
            ),
        },
        "training_recipe": {
            "epochs": int(prereg["training"]["epochs"]),
            "optimizer": bundle_values["optimizer"],
            "lr": float(bundle_values["lr"]),
            "weight_decay": float(bundle_values["weight_decay"]),
            "scheduler": bundle_values["scheduler"],
            "warmup_epochs": int(bundle_values["warmup_epochs"]),
            "min_lr_ratio": float(bundle_values["min_lr_ratio"]),
            "label_smoothing": float(bundle_values["label_smoothing"]),
            "logit_adjust_tau": float(bundle_values["logit_adjust_tau"]),
            "batch_size": int(prereg["training"]["batch_size"]),
            "gradient_accumulation_steps": int(
                prereg["training"]["gradient_accumulation_steps"]
            ),
            "amp": bool(prereg["training"]["amp"]),
        },
        "formal_reporting": deepcopy(prereg["formal_outer_evaluation"]),
        "frozen_macroarchitecture": {
            "input": [1, 224, 224],
            "stem": {
                "op": "conv",
                "kernel_size": 3,
                "in_channels": 1,
                "out_channels": 32,
                "stride": 2,
            },
            "stage_channels": [16, 24, 32, 32],
            "stage_strides": [1, 2, 2, 1],
            "stage_depths": [1, 1, 1, 1],
            "stage1": "conv_k1_32_to_16_s1",
            "stage3": "mbconv_k3_e3_24_to_32_s2",
            "head": "GAP_to_FC8",
            "head_conv_channels": None,
        },
        "selection_evidence": {
            "preregistration_path": str(prereg_path),
            "preregistration_sha256": sha256_file(prereg_path),
            "bundle_decision_path": str(
                (root / "bundle_decision.json").resolve()
            ),
            "bundle_decision_sha256": sha256_file(
                root / "bundle_decision.json"
            ),
            "geometry_decision_path": str(
                (root / "geometry_decision.json").resolve()
            ),
            "geometry_decision_sha256": sha256_file(
                root / "geometry_decision.json"
            ),
            **manifest_summary,
        },
        "source_freeze_binding": {
            "required": True,
            "manifest_path": (
                "artifacts/sonar_fourstage_operator_v2/source_freeze/"
                "source_freeze_manifest.json"
            ),
            "binding_summary_path": (
                "artifacts/sonar_fourstage_operator_v2/"
                "protocol_v2_freeze_binding.json"
            ),
        },
        "direction_gate": deepcopy(prereg["direction_gate"]),
        "robustness": deepcopy(prereg["robustness"]),
        "dir_admission": deepcopy(prereg["dir_admission"]),
        "hardware": deepcopy(prereg["hardware"]),
        "claim_boundary": (
            "Protocol selection used outer-train and inner validation only. "
            "No outer-validation classification metric was consumed."
        ),
    }
    target = ROOT / "configs" / "evaluation" / "nksid_frozen_protocol_v2.yaml"
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result = {
        "status": payload["status"],
        "protocol_path": str(target.resolve()),
        "protocol_sha256": sha256_file(target),
        "selected_bundle": selected_bundle,
        "selected_geometry": selected_geometry,
    }
    (root / "protocol_v2_selection_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("manifests", "bundle", "geometry", "finalize"),
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prereg", default=str(DEFAULT_PREREG))
    parser.add_argument(
        "--output-dir",
        default="results/sonar_fourstage_operator_v2/protocol_selection",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "manifests":
        prereg = load_prereg(Path(args.prereg).resolve())
        payload = build_group_manifest(
            data_dir=Path(args.data_dir).resolve(),
            output_dir=Path(args.output_dir).resolve() / "manifests",
            threshold=int(
                prereg["selection_units"]["inferred_stress_inner"][
                    "hamming_threshold"
                ]
            ),
        )
    elif args.phase in {"bundle", "geometry"}:
        payload = execute_matrix(args, args.phase)
    else:
        payload = finalize_protocol(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
