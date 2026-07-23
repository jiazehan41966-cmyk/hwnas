#!/usr/bin/env python3
"""Evaluate saved G1 best checkpoints on deterministic training indices.

This is an inference-only mechanism-triage gate.  For every `(fold, seed)` run,
the script rebuilds the frozen protocol split, indexes only
``split.train_indices`` through the no-augmentation evaluation dataset, loads
the saved best checkpoint, switches the model to eval mode, and measures clean
training-set top-1 and macro-F1.  It never consumes inner/outer validation
indices for this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hwnas_fpga.data.dataset import create_dataloader, create_protocol_dataloaders
from hwnas_fpga.models import build_model
from hwnas_fpga.models.backbones import build_backbone
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.trainer import evaluate_classifier

METHODS = {
    "nas_rl_arch_135": "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected",
    "mnv2_scratch": "results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2",
}
DATASET_FILES = ("train_abs.txt", "kfold_train.txt", "kfold_val.txt")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_by_key(path: Path) -> dict[tuple[int, int], tuple[Path, dict]]:
    records = {}
    for record_path in sorted(path.glob("run_fold*_seed*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        key = (int(record["fold"]), int(record["seed"]))
        if key in records:
            raise ValueError(f"Duplicate record key {key} in {path}")
        records[key] = (record_path, record)
    return records


def resolve_record_path(raw: str, fallback_dir: Path, fallback_name: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    fallback = fallback_dir / fallback_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither recorded nor local fallback exists: {path} / {fallback}")


def verify_hash(path: Path, expected: str | None, label: str) -> str:
    observed = sha256(path)
    if expected and observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected}, observed={observed}, path={path}"
        )
    return observed


def verify_dataset(data_dir: Path, records: list[dict]) -> dict:
    output = {}
    for name in DATASET_FILES:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        expected_values = {
            str(record.get("provenance", {}).get("dataset", {}).get("files", {}).get(name, {}).get("sha256"))
            for record in records
        }
        expected_values.discard("None")
        if len(expected_values) > 1:
            raise RuntimeError(f"Records disagree on dataset hash for {name}: {expected_values}")
        expected = next(iter(expected_values), None)
        output[name] = {
            "path": relative(path),
            "sha256": verify_hash(path, expected, f"dataset {name}"),
            "expected_from_records": expected,
        }
    return output


def build_from_record(record: dict, record_dir: Path, num_classes: int) -> tuple[nn.Module, dict]:
    model_meta = record.get("model", {})
    if model_meta.get("model_source") == "candidate":
        recorded_candidate = str(model_meta.get("candidate_path") or "")
        fallback = resolve(
            "hls_lut_builder/board_harness/results/"
            "pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp/candidates/"
            "003_rl_arch_135.candidate.json"
        )
        candidate_path = Path(recorded_candidate)
        if not candidate_path.exists():
            candidate_path = fallback
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate artifact unavailable: {recorded_candidate}")
        expected = record.get("provenance", {}).get("candidate", {}).get("sha256")
        candidate_sha = verify_hash(candidate_path, expected, "candidate")
        architecture = load_architecture_from_artifact(candidate_path)
        model = build_model(
            architecture=architecture,
            num_classes=num_classes,
            head_channels=architecture.head_channels,
        )
        return model, {
            "source": "candidate",
            "candidate_path": relative(candidate_path),
            "candidate_sha256": candidate_sha,
        }

    name = str(model_meta.get("name") or "mobilenet_v2")
    if bool(model_meta.get("pretrained_requested")):
        raise ValueError("This gate is scoped to scratch MobileNetV2, not pretrained weights")
    model, rebuilt_meta = build_backbone(
        name=name,
        num_classes=num_classes,
        input_channels=1,
        pretrained=False,
        strict_pretrained=False,
    )
    return model, {"source": "backbone", **rebuilt_meta}


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError(f"Unsupported checkpoint payload: {path}")


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
        total += 1
    return exceed / total


def paired_summary(nas: list[float], scratch: list[float], seed: int) -> dict:
    nas_values = np.asarray(nas, dtype=float)
    scratch_values = np.asarray(scratch, dtype=float)
    delta = scratch_values - nas_values
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(50_000, len(delta)), replace=True).mean(axis=1)
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else None
    return {
        "n_pairs": len(delta),
        "nas_mean": float(nas_values.mean()),
        "scratch_mean": float(scratch_values.mean()),
        "delta_scratch_minus_nas_mean": float(delta.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "paired_cohens_dz": float(delta.mean() / sd) if sd and sd > 0 else None,
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(delta),
    }


def decide(pair_count: int, top1: dict, macro_f1: dict) -> dict:
    if pair_count != 15:
        return {
            "status": "INCOMPLETE_DIAGNOSTIC",
            "reason": f"formal decision requires all 15 paired runs; observed {pair_count}",
            "next": "complete all pairs before mechanism attribution",
        }
    gap = top1["delta_scratch_minus_nas_mean"]
    nas_top1 = top1["nas_mean"]
    if nas_top1 >= 0.95 and gap <= 0.03:
        return {
            "status": "ONLINE_AUGMENTATION_OR_TRAIN_MODE_INTERACTION_LIKELY",
            "reason": "NAS clean eval-mode training fit is high and within 0.03 of scratch",
            "next": "audit augmentation and train/eval-mode sensitivity before any capacity sweep",
        }
    if nas_top1 < 0.90 or gap >= 0.10:
        return {
            "status": "CLEAN_TRAIN_FIT_GAP_CONFIRMED__MECHANISM_UNRESOLVED",
            "reason": (
                "NAS clean training top1 is below 0.90 or trails scratch by at least 0.10; "
                "these are predeclared triage thresholds, not a capacity proof"
            ),
            "next": "run one fixed-subset micro-overfit and LR/regularisation triage before capacity sweep",
        }
    return {
        "status": "AMBIGUOUS_CLEAN_TRAIN_FIT_GAP",
        "reason": "full result lies between the predeclared operational triage thresholds",
        "next": "run micro-overfit/LR triage; do not claim a capacity mechanism",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_checkpoint_clean_train_fit_v1.json"
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data_dir = resolve(args.data_dir)
    method_records = {name: records_by_key(resolve(path)) for name, path in METHODS.items()}
    common_keys = sorted(set.intersection(*(set(records) for records in method_records.values())))
    if len(common_keys) != 15:
        raise RuntimeError(f"Expected 15 common fold/seed records, found {len(common_keys)}")
    selected_keys = common_keys[: args.max_pairs] if args.max_pairs else common_keys
    all_records = [record for records in method_records.values() for _, record in records.values()]
    dataset_provenance = verify_dataset(data_dir, all_records)

    results: dict[str, dict[str, dict]] = {name: {} for name in METHODS}
    started = time.perf_counter()
    for key in selected_keys:
        fold, seed = key
        print(f"=== fold {fold} seed {seed} ===", flush=True)
        bundle = create_protocol_dataloaders(
            str(data_dir),
            fold=fold,
            seed=seed,
            batch_size=args.batch_size,
            image_size=224,
            inner_val_fraction=0.15,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_indices = list(bundle["split"].train_indices)
        clean_train_loader = create_dataloader(
            Subset(bundle["eval_dataset"], train_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed + fold * 100_000 + 31,
        )
        for method, records in method_records.items():
            record_path, record = records[key]
            checkpoint = resolve_record_path(
                str(record["checkpoint"]["path"]),
                record_path.parent,
                f"best_fold{fold}_seed{seed}.pt",
            )
            checkpoint_sha = verify_hash(
                checkpoint, record["checkpoint"].get("sha256"), f"{method} checkpoint {key}"
            )
            model, model_meta = build_from_record(record, record_path.parent, bundle["num_classes"])
            model.load_state_dict(load_checkpoint(checkpoint), strict=True)
            model.to(device)
            metrics = evaluate_classifier(
                model,
                clean_train_loader,
                criterion=nn.CrossEntropyLoss(),
                device=str(device),
                num_classes=bundle["num_classes"],
                topk=5,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            tag = f"fold{fold}_seed{seed}"
            results[method][tag] = {
                "fold": fold,
                "seed": seed,
                "record": relative(record_path),
                "checkpoint": relative(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "model": model_meta,
                "split_train_count": len(train_indices),
                "metrics": metrics,
            }
            print(
                f"  {method}: top1={metrics['top1']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )

    tags = [f"fold{fold}_seed{seed}" for fold, seed in selected_keys]
    summaries = {}
    for metric in ("top1", "macro_f1"):
        summaries[metric] = paired_summary(
            [results["nas_rl_arch_135"][tag]["metrics"][metric] for tag in tags],
            [results["mnv2_scratch"][tag]["metrics"][metric] for tag in tags],
            seed=20260722,
        )
    decision = decide(len(selected_keys), summaries["top1"], summaries["macro_f1"])
    payload = {
        "schema_version": 1,
        "diagnostic": "deterministic eval-mode clean training-index checkpoint fit",
        "protocol_guards": {
            "dataset_view": "eval_dataset (no random augmentation)",
            "allowed_indices": "split.train_indices only",
            "forbidden_indices": ["split.inner_val_indices", "split.outer_val_indices"],
            "model_mode": "eval",
            "checkpoint": "saved best checkpoint selected by inner validation during original run",
            "formal_pair_count": 15,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "dataset_provenance": dataset_provenance,
        "selected_fold_seed_pairs": tags,
        "methods": results,
        "paired_summaries": summaries,
        "decision": decision,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"paired_summaries": summaries, "decision": decision}, indent=2))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
