#!/usr/bin/env python3
"""Shared Phase 3C fixed-architecture transfer training and evaluation logic."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts", ROOT / "tools/sonar_analysis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import baseline as frozen_baseline
from hwnas_fpga.data.detection import create_yolo_detection_dataloader
from hwnas_fpga.deploy.export import get_model_info
from hwnas_fpga.models import build_detection_model
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import architecture_id
from hwnas_fpga.runtime import build_constraints, build_search_space, load_config
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, DetectionLossConfig
from hwnas_fpga.training.trainer import create_optimizer


B1_ID = "P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
BASELINE_CONFIG = ROOT / "configs/baseline_sonar_frozen.yaml"
V01_SPEC = ROOT / "configs/search_space_sonar_v0_1.yaml"
NETWORK_CANDIDATES = ROOT / "reports/phase2f_multiseed/network_candidate_set.json"
FORMAL_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class DatasetRegistration:
    dataset_id: str
    dataset_name: str
    modality: str
    num_classes: int
    clean_root: Path
    split_manifest: Path
    output_dir: Path
    expected_identity: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    deployed_commit = os.environ.get("HWNAS_SOURCE_GIT_COMMIT", "").strip()
    if deployed_commit:
        if len(deployed_commit) != 40 or any(character not in "0123456789abcdef" for character in deployed_commit.lower()):
            raise ValueError("HWNAS_SOURCE_GIT_COMMIT must be a 40-character hexadecimal commit")
        return deployed_commit.lower()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def parse_architecture_id(value: str) -> dict[str, Any]:
    tokens = value.split("_")
    kernel_tokens = [token for token in tokens if token.startswith("s") and "k" in token]
    kernels = {}
    for token in kernel_tokens:
        stage, kernel = token[1:].split("k", 1)
        kernels[f"stage{int(stage)}"] = int(kernel)
    return {"output_combination": "_".join(tokens[: len(tokens) - len(kernel_tokens)]), "kernels": kernels}


def load_b1() -> dict[str, Any]:
    payload = json.loads(NETWORK_CANDIDATES.read_text(encoding="utf-8"))
    rows = payload.get("architectures", payload.get("candidates", []))
    matches = [row for row in rows if row.get("label") == "B1" and row.get("architecture_id") == B1_ID]
    if len(matches) != 1:
        raise ValueError("Phase2F frozen candidate set does not contain exactly one B1 binding")
    architecture = parse_architecture_id(B1_ID)
    if architecture_id(architecture, spec=V01_SPEC) != B1_ID:
        raise ValueError("B1 architecture ID does not round-trip through the frozen V0.1 representation")
    return {"architecture_id": B1_ID, "architecture": architecture, "source_row": matches[0]}


@lru_cache(maxsize=None)
def validate_clean_manifest(registration: DatasetRegistration) -> dict[str, Any]:
    manifest_path = registration.split_manifest.resolve()
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"WAITING_FOR_CLEAN_SPLIT: {manifest_path}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha256(manifest_path)
    if expected != actual:
        raise ValueError("clean split manifest SHA256 sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != registration.dataset_id:
        raise ValueError("clean split dataset identity mismatch")
    if registration.expected_identity not in json.dumps(manifest["identity"], ensure_ascii=False):
        raise ValueError("clean split dataset version/identity mismatch")
    if manifest.get("raw_data_modified") is not False or manifest.get("duplicate_groups_cross_derived_split") != 0:
        raise ValueError("clean split manifest fails immutability/duplicate admission")
    if manifest.get("test_membership_frozen") is not True or manifest.get("model_metrics_used") is not False:
        raise ValueError("clean split was not frozen independently of model metrics")
    clean_root = registration.clean_root.resolve()
    # The manifest records its creation-host path. A relocated GPU host is admitted
    # only by rechecking the complete filename membership below at the registered root.
    required_classes = set(range(registration.num_classes))
    for split in ("train", "valid", "test"):
        rows = manifest["members"][split]
        image_dir, label_dir = clean_root / split / "images", clean_root / split / "labels"
        images = {path.name for path in image_dir.iterdir() if path.is_file()}
        labels = {path.name for path in label_dir.iterdir() if path.is_file()}
        if images != {row["file"] for row in rows} or labels != {f"{Path(row['file']).stem}.txt" for row in rows}:
            raise ValueError(f"clean materialized {split} membership differs from manifest")
        present = {int(cls) for row in rows for cls in row["classes"]}
        if present != required_classes:
            raise ValueError(f"{split} lacks registered classes: expected={required_classes}, present={present}")
    locations: dict[str, set[str]] = {}
    for split, rows in manifest["members"].items():
        for row in rows:
            locations.setdefault(row["image_sha256"], set()).add(split)
    if any(len(splits) > 1 for splits in locations.values()):
        raise ValueError("exact duplicate group crosses clean split")
    return {"manifest": manifest, "sha256": actual, "path": str(manifest_path), "clean_root": str(clean_root)}


def effective_config(registration: DatasetRegistration) -> dict[str, Any]:
    config = copy.deepcopy(load_config(str(BASELINE_CONFIG)))
    data_yaml = yaml.safe_load((registration.clean_root / "data.yaml").read_text(encoding="utf-8"))
    names_value = data_yaml["names"]
    if isinstance(names_value, dict):
        class_names = [names_value[key] for key in sorted(names_value, key=lambda key: int(key))]
    else:
        class_names = list(names_value)
    if len(class_names) != registration.num_classes:
        raise ValueError("dataset class count differs from registration")
    config["dataset"].update(
        {
            "name": registration.dataset_id,
            "data_dir": str(registration.clean_root.resolve()),
            "num_classes": registration.num_classes,
            "class_names": class_names,
            "num_workers": 0,
            "split_paths": {
                split: {
                    "images": str((registration.clean_root / split / "images").resolve()),
                    "labels": str((registration.clean_root / split / "labels").resolve()),
                }
                for split in ("train", "valid", "test")
            },
        }
    )
    config["training"]["detection_loss"]["class_loss_weights"] = [1.0] * registration.num_classes
    return config


def make_loader(config: Mapping[str, Any], split: str, seed: int, workers: int, *, train: bool, max_samples: int | None = None):
    return create_yolo_detection_dataloader(
        config["dataset"]["data_dir"],
        split=split,
        batch_size=int(config["training"]["batch_size"]),
        image_size=int(config["dataset"]["image_size"]),
        input_channels=int(config["dataset"]["input_channels"]),
        num_workers=int(workers),
        pin_memory=False,
        shuffle=train,
        seed=int(seed),
        max_samples=max_samples,
        augment_config=config["training"]["detection_augmentation"] if train else None,
        class_balanced_sampler=bool(config["training"]["detection_sampling"]["class_balanced"]) if train else False,
    )


def build_model(kind: str, config: Mapping[str, Any], b1: Mapping[str, Any]):
    num_classes = int(config["dataset"]["num_classes"])
    image_size = int(config["dataset"]["image_size"])
    if kind == "baseline":
        constraints = build_constraints(dict(config))
        space = build_search_space(dict(config), image_size=image_size, input_channels=1, num_classes=num_classes, constraints=constraints)
        spec = space.baseline_architecture()
        detection = config["detection"]
        model = build_detection_model(
            spec,
            num_classes=num_classes,
            feature_indices=tuple(detection["feature_stage_indices"]),
            neck_channels=int(detection["fixed_neck"]["out_channels"]),
            anchors_per_location=int(detection["fixed_head"]["anchors_per_location"]),
        )
        loss = replace(
            DetectionLossConfig.from_config(dict(config)),
            num_classes=num_classes,
            image_size=image_size,
            class_loss_weights=tuple(1.0 for _ in range(num_classes)),
        )
        identity = identity_for_kind(kind, num_classes)
    elif kind == "nas":
        model = build_sonar_candidate(V01_SPEC, b1["architecture"], num_classes=num_classes, image_size=image_size)
        loss = model.loss_config(dict(config))
        identity = identity_for_kind(kind, num_classes)
    else:
        raise ValueError(kind)
    return model, DenseAnchorDetectionLoss(loss), identity


def identity_for_kind(kind: str, num_classes: int) -> dict[str, Any]:
    if kind == "baseline":
        return {"architecture_id": "FrozenBaseline_P3_P4_P5_all_k3", "kind": "baseline", "num_classes": int(num_classes)}
    if kind == "nas":
        return {"architecture_id": B1_ID, "kind": "B1 / Transfer Architecture", "num_classes": int(num_classes)}
    raise ValueError(kind)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(loader) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": loader.generator.get_state() if loader.generator is not None else None,
        "sampler_generator": loader.sampler.generator.get_state() if getattr(loader.sampler, "generator", None) is not None else None,
    }


def restore_rng(state: Mapping[str, Any], loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["loader_generator"] is not None and loader.generator is not None:
        loader.generator.set_state(state["loader_generator"])
    if state["sampler_generator"] is not None and getattr(loader.sampler, "generator", None) is not None:
        loader.sampler.generator.set_state(state["sampler_generator"])


def train_epoch(model, loader, criterion, optimizer, device: str, *, max_batches: int | None = None) -> dict[str, Any]:
    model.train()
    sums = Counter()
    images_seen = 0
    start = time.perf_counter()
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        output = model(images.to(device))
        frozen_baseline.finite_outputs(output)
        loss, parts = criterion(output, targets)
        if not torch.isfinite(loss) or not all(math.isfinite(float(value)) for value in parts.values()):
            raise ValueError("non-finite transfer training loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("non-finite or absent transfer gradients")
        optimizer.step()
        batch = int(images.shape[0])
        images_seen += batch
        sums["total_loss"] += float(loss.detach()) * batch
        for name in ("box_loss", "object_loss", "class_loss"):
            sums[name] += float(parts[name]) * batch
    elapsed = time.perf_counter() - start
    if images_seen == 0:
        raise ValueError("empty transfer train epoch")
    return {
        "images": images_seen,
        "seconds": elapsed,
        "images_per_second": images_seen / elapsed,
        **{name: sums[name] / images_seen for name in ("total_loss", "box_loss", "object_loss", "class_loss")},
        "gradient_finite": True,
    }


def model_info(model: torch.nn.Module) -> dict[str, Any]:
    devices = [torch.cuda.current_device()] if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        info = get_model_info(model, (1, 1, 640, 640))
    macs = int(info["estimated_macs"])
    return {"Params": int(info["total_params"]), "MACs": macs, "FLOPs": 2 * macs, "complexity_scope": "Conv2d/Linear estimate; MACs are not latency"}


def build_binding(registration: DatasetRegistration, admission: Mapping[str, Any], kind: str, seed: int, identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": "3C",
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": kind,
        "architecture": dict(identity),
        "seed": int(seed),
        "split_manifest_sha256": admission["sha256"],
        "baseline_config_sha256": sha256(BASELINE_CONFIG),
        "search_space_sha256": sha256(V01_SPEC),
        "network_candidate_set_sha256": sha256(NETWORK_CANDIDATES),
        "transfer_common_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_policy": "epoch80_latest_only",
        "test_policy": "construct test loader and evaluate exactly once after epoch80 latest checkpoint",
        "git_commit": git_commit(),
    }


def validate_formal_result(result: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    """Reject legacy or foreign summaries before they can enter Phase 3C aggregates."""
    required = {
        "artifact_type": "phase3c_architecture_transfer_result",
        "dataset_id": binding["dataset_id"],
        "model": binding["model"],
        "seed": binding["seed"],
        "architecture_id": binding["architecture"]["architecture_id"],
        "checkpoint_policy": "epoch80/latest.pt",
        "split_manifest_sha256": binding["split_manifest_sha256"],
        "test_evaluated": True,
        "test_used_for_selection": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": result.get(key)}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    if result.get("binding") != dict(binding):
        mismatches["binding"] = {"expected": dict(binding), "actual": result.get("binding")}
    if mismatches:
        raise ValueError(f"foreign or legacy formal result rejected: {mismatches}")


def formal_run(registration: DatasetRegistration, args: argparse.Namespace, admission: Mapping[str, Any], config: dict[str, Any], b1: Mapping[str, Any], kind: str, seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("READY_FOR_GPU_EXECUTION: formal 80-epoch transfer requires CUDA")
    output = Path(args.output_dir).resolve()
    summary_path = output / f"{kind}_seed{seed}.json"
    training_summary_path = output / f"{kind}_seed{seed}_training.json"
    identity = identity_for_kind(kind, int(config["dataset"]["num_classes"]))
    binding = build_binding(registration, admission, kind, seed, identity)
    if summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_formal_result(result, binding)
        return result
    if training_summary_path.is_file():
        result = json.loads(training_summary_path.read_text(encoding="utf-8"))
        if result.get("binding") != binding or result.get("epoch") != 80 or result.get("test_evaluated") is not False:
            raise ValueError("existing training-complete summary does not match the frozen Phase 3C binding")
        return result
    run_dir = output / f"{kind}_seed{seed}_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_loader = make_loader(config, "train", seed, args.num_workers, train=True)
    valid_loader = make_loader(config, "valid", seed + 100, args.num_workers, train=False)
    model, criterion, identity = build_model(kind, config, b1)
    model.to(args.device)
    criterion.to(args.device)
    optimizer = create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
    binding = build_binding(registration, admission, kind, seed, identity)
    latest = run_dir / "latest.pt"
    start_epoch, history = 0, []
    if latest.is_file():
        if not args.resume:
            raise FileExistsError(f"{latest} exists; use --resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("resume binding mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, history = int(state["epoch"]), list(state["history"])
        restore_rng(state["rng"], train_loader)
    dump(run_dir / "binding.json", binding)
    info = model_info(model)
    started = time.perf_counter()
    for epoch in range(start_epoch, 80):
        train = train_epoch(model, train_loader, criterion, optimizer, args.device)
        valid, _ = frozen_baseline.evaluate(model, valid_loader, criterion, config, args.device)
        row = {"epoch": epoch + 1, "train": train, "valid": valid, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        state = {
            "binding": binding,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng": rng_state(train_loader),
            "history": history,
            "test_evaluated": False,
        }
        atomic_torch_save(latest, state)
        dump(run_dir / "history.json", history)
        dump(run_dir / "status.json", {"status": "TRAINING", "epoch": epoch + 1, "test_evaluated": False, "binding": binding})
        print(registration.dataset_id, kind, f"seed={seed}", f"epoch={epoch + 1}/80", f"valid_mAP50_95={valid['mAP50_95']:.6f}", flush=True)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if state["epoch"] != 80:
        raise ValueError("formal training-complete gate requires epoch80 latest checkpoint")
    history_path = run_dir / "history.json"
    result = {
        "artifact_type": "phase3c_architecture_transfer_training_complete",
        "dataset": registration.dataset_name,
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": kind,
        "architecture_id": identity["architecture_id"],
        "seed": seed,
        "checkpoint_policy": "epoch80/latest.pt",
        "checkpoint_sha256": sha256(latest),
        "history_sha256": sha256(history_path),
        "config_sha256": binding["baseline_config_sha256"],
        "split_manifest_sha256": admission["sha256"],
        "epoch": 80,
        "training_complete": True,
        "valid_epoch80": history[-1]["valid"],
        **info,
        "training_time_seconds": sum(float(row["train"]["seconds"]) for row in history),
        "wall_time_seconds_this_invocation": time.perf_counter() - started,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "binding": binding,
        "completed_at": now(),
    }
    dump(training_summary_path, result)
    dump(run_dir / "status.json", {"status": "TRAINING_COMPLETE_TEST_SEALED", "epoch": 80, "test_evaluated": False, "binding": binding})
    return result


def smoke(registration: DatasetRegistration, args: argparse.Namespace, admission: Mapping[str, Any], config: dict[str, Any], b1: Mapping[str, Any]) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in selected_models(args.models):
        set_seed(0)
        train_loader = make_loader(config, "train", 0, 0, train=True, max_samples=8)
        valid_loader = make_loader(config, "valid", 100, 0, train=False, max_samples=8)
        model, criterion, identity = build_model(kind, config, b1)
        model.to(args.device)
        criterion.to(args.device)
        optimizer = create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
        train = train_epoch(model, train_loader, criterion, optimizer, args.device, max_batches=1)
        valid, _ = frozen_baseline.evaluate(model, valid_loader, criterion, config, args.device, max_batches=1)
        rows.append({"model": kind, "architecture": identity, "train": train, "valid": valid, "test_evaluated": False})
    dump(output / "smoke_result.json", {"dataset_id": registration.dataset_id, "split_manifest_sha256": admission["sha256"], "rows": rows, "test_evaluated": False, "formal_eligible": False})


def benchmark(registration: DatasetRegistration, args: argparse.Namespace, admission: Mapping[str, Any], config: dict[str, Any], b1: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("READY_FOR_GPU_EXECUTION: timing benchmark requires CUDA")
    benchmark_dir = ROOT / "results/transfer/benchmarks"
    for kind in selected_models(args.models):
        set_seed(0)
        loader = make_loader(config, "train", 0, args.num_workers, train=True)
        valid_loader = make_loader(config, "valid", 100, args.num_workers, train=False)
        model, criterion, identity = build_model(kind, config, b1)
        model.to(args.device)
        criterion.to(args.device)
        optimizer = create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
        train_epoch(model, loader, criterion, optimizer, args.device)
        frozen_baseline.evaluate(model, valid_loader, criterion, config, args.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        measured = train_epoch(model, loader, criterion, optimizer, args.device)
        validation_started = time.perf_counter()
        frozen_baseline.evaluate(model, valid_loader, criterion, config, args.device)
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - validation_started
        full_epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "dataset_id": registration.dataset_id,
            "model": kind,
            "architecture_id": identity["architecture_id"],
            "split_manifest_sha256": admission["sha256"],
            "seconds_per_epoch": full_epoch_seconds,
            "train_seconds_per_epoch": measured["seconds"],
            "validation_seconds_per_epoch": validation_seconds,
            "train_images_per_second": measured["images_per_second"],
            "images_per_second": measured["images"] / full_epoch_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "estimated_80_epoch_hours_per_run": full_epoch_seconds * 80 / 3600,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "accuracy_aggregate_eligible": False,
        }
        dump(benchmark_dir / f"{registration.dataset_id}_{kind}.json", row)
    write_timing_estimate()


def write_timing_estimate() -> None:
    benchmark_dir = ROOT / "results/transfer/benchmarks"
    paths = [benchmark_dir / f"{dataset}_{kind}.json" for dataset in ("uatd_v3", "classic_sctd_1_0") for kind in ("baseline", "nas")]
    if not all(path.is_file() for path in paths):
        return
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    total_gpu_hours = sum(row["seconds_per_epoch"] * 80 * 3 / 3600 for row in rows)
    lines = ["# Phase 3C Training Time Estimate", "", "Benchmark rows use one warmup epoch plus one measured epoch and are excluded from accuracy aggregates.", "", "| dataset | model | sec/epoch | images/s | peak GPU GiB | estimated 80-epoch hours/run |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset_id']} | {row['model']} | {row['seconds_per_epoch']:.3f} | {row['images_per_second']:.3f} | {row['peak_gpu_memory_bytes']/2**30:.3f} | {row['estimated_80_epoch_hours_per_run']:.3f} |")
    lines += ["", f"Estimated total GPU-hours for 12 runs: **{total_gpu_hours:.3f}**."]
    (ROOT / "results/transfer/TRAINING_TIME_ESTIMATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_timing_gate() -> None:
    benchmark_dir = ROOT / "results/transfer/benchmarks"
    paths = [benchmark_dir / f"{dataset}_{kind}.json" for dataset in ("uatd_v3", "classic_sctd_1_0") for kind in ("baseline", "nas")]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"TIMING_BENCHMARK_PENDING: missing {missing}")
    acceptance_path = ROOT / "results/transfer/TIMING_BENCHMARK_ACCEPTED.json"
    if not acceptance_path.is_file():
        raise RuntimeError("TIMING_REVIEW_PENDING: create TIMING_BENCHMARK_ACCEPTED.json after reviewing TRAINING_TIME_ESTIMATE.md")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    expected = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    if acceptance.get("status") != "ACCEPTED" or acceptance.get("benchmark_sha256") != expected:
        raise ValueError("timing benchmark acceptance does not bind the current benchmark records")


def selected_models(value: str) -> tuple[str, ...]:
    return ("baseline", "nas") if value == "both" else (value,)


def aggregate_results(registration: DatasetRegistration, output: Path, seeds: tuple[int, ...]) -> None:
    admission = validate_clean_manifest(registration)
    num_classes = int(effective_config(registration)["dataset"]["num_classes"])
    rows = []
    for kind in ("baseline", "nas"):
        for seed in seeds:
            path = output / f"{kind}_seed{seed}.json"
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                identity = identity_for_kind(kind, num_classes)
                binding = build_binding(registration, admission, kind, seed, identity)
                validate_formal_result(result, binding)
                rows.append(result)
    if not rows:
        return
    fieldnames = ["model", "seed", "Precision", "Recall", "mAP50", "mAP50_95", "Params", "MACs", "FLOPs", "training_time_seconds", "split_manifest_sha256"]
    table = [
        {
            "model": row["model"], "seed": row["seed"],
            "Precision": row["test_metrics"]["Precision"], "Recall": row["test_metrics"]["Recall"],
            "mAP50": row["test_metrics"]["AP50"], "mAP50_95": row["test_metrics"]["mAP50_95"],
            "Params": row["Params"], "MACs": row["MACs"], "FLOPs": row["FLOPs"],
            "training_time_seconds": row["training_time_seconds"], "split_manifest_sha256": row["split_manifest_sha256"],
        }
        for row in rows
    ]
    import csv
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(table)
    complete = len(rows) == 2 * len(seeds)
    lines = [f"# {registration.dataset_name} Architecture Transfer", "", f"Complete formal matrix: `{complete}`.", "", "Checkpoint policy: `epoch80/latest.pt`. Test is evaluated once after training and never selects a checkpoint."]
    if complete:
        baseline = {int(row["seed"]): row for row in rows if row["model"] == "baseline"}
        b1 = {int(row["seed"]): row for row in rows if row["model"] == "nas"}
        lines += ["", "| seed | ΔmAP50 | ΔmAP50:95 |", "|---:|---:|---:|"]
        delta50, delta95 = [], []
        for seed in seeds:
            d50 = b1[seed]["test_metrics"]["AP50"] - baseline[seed]["test_metrics"]["AP50"]
            d95 = b1[seed]["test_metrics"]["mAP50_95"] - baseline[seed]["test_metrics"]["mAP50_95"]
            delta50.append(d50); delta95.append(d95)
            lines.append(f"| {seed} | {d50:+.6f} | {d95:+.6f} |")
        direction50 = all(value > 0 for value in delta50)
        direction95 = all(value > 0 for value in delta95)
        domain_degradation = statistics.mean(delta50) < 0 or statistics.mean(delta95) < 0
        lines += [
            "",
            f"Mean ΔmAP50: {statistics.mean(delta50):+.6f}; sample std: {statistics.stdev(delta50):.6f}; min/max: {min(delta50):+.6f}/{max(delta50):+.6f}.",
            f"Mean ΔmAP50:95: {statistics.mean(delta95):+.6f}; sample std: {statistics.stdev(delta95):.6f}; min/max: {min(delta95):+.6f}/{max(delta95):+.6f}.",
            "",
            f"1. B1 improves mAP50 on average: `{statistics.mean(delta50) > 0}`.",
            f"2. B1 improves mAP50:95 on average: `{statistics.mean(delta95) > 0}`.",
            f"3. Seed 0/1/2 delta direction is consistently positive: mAP50=`{direction50}`, mAP50:95=`{direction95}`.",
            f"4. Possible domain-specific degradation flag: `{domain_degradation}`.",
            f"5. Complete-architecture transfer direction is positive on both metrics: `{statistics.mean(delta50) > 0 and statistics.mean(delta95) > 0}`.",
        ]
    (output / "transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(registration: DatasetRegistration, admission: Mapping[str, Any], config: Mapping[str, Any], b1: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ADMITTED" if admission else "WAITING_FOR_CLEAN_SPLIT",
        "dataset_id": registration.dataset_id,
        "dataset_name": registration.dataset_name,
        "modality": registration.modality,
        "num_classes": registration.num_classes,
        "split_manifest": {
            "path": admission["path"],
            "sha256": admission["sha256"],
            "clean_root": admission["clean_root"],
            "summary": admission["manifest"]["summary"],
            "policy": admission["manifest"]["policy"],
            "duplicate_groups_cross_derived_split": admission["manifest"]["duplicate_groups_cross_derived_split"],
        },
        "B1_architecture_id": b1["architecture_id"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "checkpoint_policy": "epoch80/latest.pt",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "formal_status": "READY_FOR_GPU_EXECUTION" if not torch.cuda.is_available() else "READY_FOR_TIMING_BENCHMARK",
        "test_evaluated": False,
    }
    dump(output / "preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser(registration: DatasetRegistration) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Phase 3C fixed B1 transfer on {registration.dataset_name}")
    parser.add_argument("--mode", choices=("preflight", "smoke", "benchmark", "formal", "aggregate"), default="preflight")
    parser.add_argument("--models", choices=("baseline", "nas", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--defer-aggregate", action="store_true", help="Use for parallel formal workers; run --mode aggregate after both model queues finish")
    parser.add_argument("--output-dir", default=str(registration.output_dir))
    return parser


def run_cli(registration: DatasetRegistration) -> None:
    args = build_parser(registration).parse_args()
    output = Path(args.output_dir).resolve()
    admission = validate_clean_manifest(registration)
    config = effective_config(registration)
    b1 = load_b1()
    if int(config["training"]["epochs"]) != 80 or int(config["training"]["batch_size"]) != 4:
        raise ValueError("frozen transfer protocol must remain 80 epochs and batch size 4")
    if args.mode == "preflight":
        preflight(registration, admission, config, b1, output)
        return
    if args.mode == "smoke":
        smoke(registration, args, admission, config, b1)
        return
    if args.mode == "benchmark":
        benchmark(registration, args, admission, config, b1)
        return
    if args.mode == "aggregate":
        aggregate_results(registration, output, FORMAL_SEEDS)
        return
    requested_seeds = tuple(args.seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds) or any(seed not in FORMAL_SEEDS for seed in requested_seeds):
        raise ValueError("formal seeds must be a unique subset of frozen seeds 0,1,2")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("READY_FOR_GPU_EXECUTION: formal 80-epoch transfer requires CUDA")
    require_timing_gate()
    for kind in selected_models(args.models):
        for seed in requested_seeds:
            formal_run(registration, args, admission, config, b1, kind, seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
