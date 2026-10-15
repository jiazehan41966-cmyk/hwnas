#!/usr/bin/env python3
"""Phase 3D matched structural decomposition: P2 vs selected k5 bundle.

This module must not construct a UATD/SCTD test dataloader. Frozen Phase 3C
D/B1 checkpoints are evaluated on the clean development validation split only.
The only new training runs are B2 × {UATD, SCTD} × seeds {0,1,2}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

import architecture_transfer_common as common
from hwnas_fpga.data.detection import create_yolo_detection_dataloader
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import architecture_id
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from run_sctd_architecture_transfer import REGISTRATION as SCTD
from run_uatd_architecture_transfer import REGISTRATION as UATD


D_ID = "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3"
B2_ID = "P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
B1_ID = common.B1_ID
FORMAL_SEEDS = common.FORMAL_SEEDS
REGISTRATIONS = (UATD, SCTD)
REPORT_DIR = common.ROOT / "reports/phase3d_structural_decomposition"
RESULT_ROOT = common.ROOT / "results/transfer/phase3d"
FREEZE_MANIFEST = common.ROOT / "results/transfer/FORMAL_CHECKPOINT_FREEZE_MANIFEST.json"
FORBIDDEN_SPLITS = {"test", "test1", "test2", "tests"}
FLOAT_TOLERANCE = 1e-9
ADDITIVITY_NOTE = "delta_P2 + delta_K equals delta_TOTAL up to floating-point precision only"
PHASE3C_KIND = {"D": "baseline", "B1": "nas"}
DATASET_FOLDERS = {"uatd_v3": "UATD", "classic_sctd_1_0": "SCTD"}
KERNEL_STAGES = ("stage1", "stage2", "stage4", "stage5", "stage6", "stage7")

PHASE3D_TEST_ACCESSED = False
_ORIGINAL_CREATE_LOADER = create_yolo_detection_dataloader
_BAN_INSTALLED = False


@dataclass(frozen=True)
class EffectRow:
    seed0: float
    seed1: float
    seed2: float

    def values(self) -> tuple[float, float, float]:
        return (self.seed0, self.seed1, self.seed2)

    def mean(self) -> float:
        return statistics.mean(self.values())

    def std(self) -> float:
        return statistics.stdev(self.values())

    def positive_seed_count(self) -> int:
        return sum(value > 0 for value in self.values())


def parse_arch(value: str) -> dict[str, Any]:
    architecture = common.parse_architecture_id(value)
    if architecture_id(architecture, spec=common.V01_SPEC) != value:
        raise ValueError(f"architecture ID does not round-trip: {value}")
    return architecture


def architecture_map() -> dict[str, dict[str, Any]]:
    return {
        "D": {"architecture_id": D_ID, "architecture": parse_arch(D_ID), "phase3c_kind": "baseline", "phase3c_architecture_id": "FrozenBaseline_P3_P4_P5_all_k3"},
        "B2": {"architecture_id": B2_ID, "architecture": parse_arch(B2_ID), "phase3c_kind": None, "phase3c_architecture_id": None},
        "B1": {"architecture_id": B1_ID, "architecture": parse_arch(B1_ID), "phase3c_kind": "nas", "phase3c_architecture_id": B1_ID},
    }


def assert_matched_decomposition() -> dict[str, Any]:
    mapping = architecture_map()
    d, b2, b1 = mapping["D"]["architecture"], mapping["B2"]["architecture"], mapping["B1"]["architecture"]
    if b1["kernels"] != b2["kernels"]:
        raise ValueError("B1 vs B2 must differ only in P2 / four-scale output")
    if b1["output_combination"] != "P2_P3_P4_P5" or b2["output_combination"] != "P3_P4_P5":
        raise ValueError("B1 vs B2 output-combination mismatch")
    if b2["output_combination"] != d["output_combination"]:
        raise ValueError("B2 vs D must share P3_P4_P5")
    if b2["kernels"] == d["kernels"]:
        raise ValueError("B2 vs D must differ in the selected k5 bundle")
    if d["kernels"] != {stage: 3 for stage in KERNEL_STAGES}:
        raise ValueError("D must be the all-k3 P3_P4_P5 control")
    if b2["kernels"] != {"stage1": 3, "stage2": 5, "stage4": 5, "stage5": 5, "stage6": 5, "stage7": 5}:
        raise ValueError("B2 must be the selected k5 bundle on P3_P4_P5")
    return mapping


def is_forbidden_split(split: str) -> bool:
    token = str(split).strip().lower()
    return token in FORBIDDEN_SPLITS or token.endswith("/test") or token.endswith("\\test")


def reject_phase3d_test_access(split: str, *, path: Path | None = None) -> None:
    global PHASE3D_TEST_ACCESSED
    if is_forbidden_split(split):
        PHASE3D_TEST_ACCESSED = True
        raise RuntimeError(f"PHASE3D_TEST_ACCESS_FORBIDDEN: attempted to access split={split!r}")
    if path is not None:
        parts = {part.lower() for part in Path(path).parts}
        if "test" in parts or "test1" in parts or "test2" in parts:
            PHASE3D_TEST_ACCESSED = True
            raise RuntimeError(f"PHASE3D_TEST_ACCESS_FORBIDDEN: attempted to access path={path}")


def guarded_create_yolo_detection_dataloader(*args, split: str, **kwargs):
    reject_phase3d_test_access(split)
    return _ORIGINAL_CREATE_LOADER(*args, split=split, **kwargs)


def install_phase3d_test_ban() -> None:
    global _BAN_INSTALLED
    if _BAN_INSTALLED:
        return
    import hwnas_fpga.data.detection as detection

    detection.create_yolo_detection_dataloader = guarded_create_yolo_detection_dataloader
    common.create_yolo_detection_dataloader = guarded_create_yolo_detection_dataloader
    _BAN_INSTALLED = True


def phase3d_make_loader(config: Mapping[str, Any], split: str, seed: int, workers: int, *, train: bool, max_samples: int | None = None):
    reject_phase3d_test_access(split)
    install_phase3d_test_ban()
    if split != "train" and split != "valid":
        raise RuntimeError(f"PHASE3D_TEST_ACCESS_FORBIDDEN: only train/valid loaders are allowed, got {split!r}")
    return common.make_loader(config, split, seed, workers, train=train, max_samples=max_samples)


def file_sha256(path: Path) -> str:
    return common.sha256(path)


def protocol_hashes() -> dict[str, str]:
    evaluator_files = (
        common.ROOT / "tools/sonar_analysis/baseline.py",
        common.ROOT / "src/hwnas_fpga/training/detection_coco_eval.py",
        common.ROOT / "src/hwnas_fpga/training/detection_metrics.py",
        common.ROOT / "src/hwnas_fpga/training/detection_trainer.py",
    )
    training_files = (
        common.BASELINE_CONFIG,
        Path(__file__).resolve(),
        common.ROOT / "scripts/architecture_transfer_common.py",
    )
    return {
        "evaluator_hash": hashlib.sha256(b"".join(Path(path).read_bytes() for path in evaluator_files)).hexdigest(),
        "training_protocol_hash": hashlib.sha256(b"".join(Path(path).read_bytes() for path in training_files)).hexdigest(),
        "baseline_config_sha256": file_sha256(common.BASELINE_CONFIG),
        "search_space_sha256": file_sha256(common.V01_SPEC),
        "network_candidate_set_sha256": file_sha256(common.NETWORK_CANDIDATES),
        "phase3d_common_sha256": file_sha256(Path(__file__).resolve()),
    }


def admit_phase3d_split(registration: common.DatasetRegistration) -> dict[str, Any]:
    """Admit the frozen Phase 3B split without listing or loading the test directory."""
    install_phase3d_test_ban()
    manifest_path = registration.split_manifest.resolve()
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"WAITING_FOR_CLEAN_SPLIT: {manifest_path}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = file_sha256(manifest_path)
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
    required_classes = set(range(registration.num_classes))
    for split in ("train", "valid"):
        rows = manifest["members"][split]
        image_dir, label_dir = clean_root / split / "images", clean_root / split / "labels"
        images = {path.name for path in image_dir.iterdir() if path.is_file()}
        labels = {path.name for path in label_dir.iterdir() if path.is_file()}
        if images != {row["file"] for row in rows} or labels != {f"{Path(row['file']).stem}.txt" for row in rows}:
            raise ValueError(f"clean materialized {split} membership differs from manifest")
        present = {int(cls) for row in rows for cls in row["classes"]}
        if present != required_classes:
            raise ValueError(f"{split} lacks registered classes: expected={required_classes}, present={present}")
    return {"manifest": manifest, "sha256": actual, "path": str(manifest_path), "clean_root": str(clean_root), "test_directory_listed": False}


def phase3d_effective_config(registration: common.DatasetRegistration) -> dict[str, Any]:
    config = common.effective_config(registration)
    config["dataset"]["split_paths"].pop("test", None)
    config["evaluation_protocol"]["test_policy"] = "PHASE3D_FORBIDDEN"
    config["evaluation_protocol"]["development_splits"] = ["train", "valid"]
    if int(config["training"]["epochs"]) != 80 or int(config["training"]["batch_size"]) != 4:
        raise ValueError("frozen transfer protocol must remain 80 epochs and batch size 4")
    if int(config["dataset"]["image_size"]) != 640:
        raise ValueError("frozen transfer protocol must remain input 640")
    if str(config["training"].get("scheduler", "none")).lower() != "none":
        raise ValueError("frozen transfer protocol must keep scheduler=none")
    return config


def identity_for_label(label: str, num_classes: int) -> dict[str, Any]:
    mapping = architecture_map()[label]
    kinds = {"D": "D / Frozen Baseline", "B2": "B2 / matched P345 k5-bundle control", "B1": "B1 / Transfer Architecture"}
    return {
        "architecture_id": mapping["architecture_id"],
        "kind": kinds[label],
        "num_classes": int(num_classes),
        "phase3c_architecture_id": mapping["phase3c_architecture_id"],
        "phase3c_kind": mapping["phase3c_kind"],
    }


def build_label_model(label: str, config: Mapping[str, Any]):
    num_classes = int(config["dataset"]["num_classes"])
    image_size = int(config["dataset"]["image_size"])
    b1 = common.load_b1()
    if label == "D":
        model, criterion, _ = common.build_model("baseline", config, b1)
    elif label == "B1":
        model, criterion, _ = common.build_model("nas", config, b1)
    elif label == "B2":
        model = build_sonar_candidate(common.V01_SPEC, parse_arch(B2_ID), num_classes=num_classes, image_size=image_size)
        criterion = DenseAnchorDetectionLoss(model.loss_config(dict(config)))
    else:
        raise ValueError(label)
    return model, criterion, identity_for_label(label, num_classes)


def output_dir_for(registration: common.DatasetRegistration) -> Path:
    return RESULT_ROOT / DATASET_FOLDERS[registration.dataset_id]


def b2_run_dir(registration: common.DatasetRegistration, seed: int) -> Path:
    return output_dir_for(registration) / f"b2_seed{seed}_artifacts"


def b2_summary_path(registration: common.DatasetRegistration, seed: int) -> Path:
    return output_dir_for(registration) / f"b2_seed{seed}.json"


def validation_summary_path(registration: common.DatasetRegistration, label: str, seed: int) -> Path:
    return output_dir_for(registration) / f"{label.lower()}_seed{seed}_valid.json"


def build_binding(registration: common.DatasetRegistration, admission: Mapping[str, Any], label: str, seed: int, identity: Mapping[str, Any]) -> dict[str, Any]:
    hashes = protocol_hashes()
    return {
        "phase": "3D",
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": label,
        "architecture": dict(identity),
        "seed": int(seed),
        "split_manifest_sha256": admission["sha256"],
        "baseline_config_sha256": hashes["baseline_config_sha256"],
        "search_space_sha256": hashes["search_space_sha256"],
        "network_candidate_set_sha256": hashes["network_candidate_set_sha256"],
        "phase3d_common_sha256": hashes["phase3d_common_sha256"],
        "evaluator_hash": hashes["evaluator_hash"],
        "training_protocol_hash": hashes["training_protocol_hash"],
        "checkpoint_policy": "epoch80_latest_only",
        "test_policy": "PHASE3D_FORBIDDEN",
        "git_commit": common.git_commit(),
    }


def require_cuda(device: str) -> None:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("READY_FOR_GPU_EXECUTION: Phase 3D formal 80-epoch B2 training and official validation inference require CUDA")


def load_freeze_manifest() -> dict[str, Any]:
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError("Phase 3C FORMAL_CHECKPOINT_FREEZE_MANIFEST.json is missing")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("formal_runs") != "12/12":
        raise ValueError("Phase 3C freeze manifest did not pass")
    return manifest


def freeze_record(registration: common.DatasetRegistration, label: str, seed: int) -> dict[str, Any]:
    if label not in PHASE3C_KIND:
        raise ValueError(label)
    kind = PHASE3C_KIND[label]
    manifest = load_freeze_manifest()
    matches = [
        row for row in manifest["records"]
        if row["dataset_id"] == registration.dataset_id and row["model"] == kind and int(row["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Phase 3C freeze record for {registration.dataset_id}/{label}/seed{seed}")
    return matches[0]


def resolve_phase3c_checkpoint(registration: common.DatasetRegistration, label: str, seed: int) -> Path:
    record = freeze_record(registration, label, seed)
    kind = PHASE3C_KIND[label]
    candidates = [
        registration.output_dir.resolve() / f"{kind}_seed{seed}_artifacts" / "latest.pt",
        Path(record["checkpoint_path"]),
    ]
    mismatches = []
    for path in candidates:
        if not path.is_file():
            continue
        actual = file_sha256(path)
        if actual == record["checkpoint_sha256"]:
            return path
        mismatches.append(f"{path} actual={actual}")
    if mismatches:
        raise FileNotFoundError(
            f"Phase 3C frozen checkpoint not found with matching SHA256 for {registration.dataset_id}/{label}/seed{seed}; "
            f"ignored mismatched files: {mismatches}"
        )
    raise FileNotFoundError(
        f"Phase 3C frozen checkpoint missing for {registration.dataset_id}/{label}/seed{seed}; "
        "copy epoch80/latest.pt from the Phase 3C host before dedicated validation inference"
    )


def extract_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    return {
        "Precision": float(raw["Precision"]),
        "Recall": float(raw["Recall"]),
        "mAP50": float(raw.get("mAP50", raw["AP50"])),
        "mAP50_95": float(raw["mAP50_95"]),
        "AP50": float(raw["AP50"]),
        "AP75": float(raw.get("AP75", raw.get("mAP75", 0.0))),
    }


def train_b2(registration: common.DatasetRegistration, args: argparse.Namespace, admission: Mapping[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    require_cuda(args.device)
    install_phase3d_test_ban()
    assert_matched_decomposition()
    output = output_dir_for(registration)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = b2_summary_path(registration, seed)
    identity = identity_for_label("B2", int(config["dataset"]["num_classes"]))
    binding = build_binding(registration, admission, "B2", seed, identity)
    if summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result.get("binding") != binding or result.get("epoch") != 80 or result.get("test_evaluated") is not False:
            raise ValueError("existing B2 summary does not match the frozen Phase 3D binding")
        return result
    run_dir = b2_run_dir(registration, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    common.set_seed(seed)
    train_loader = phase3d_make_loader(config, "train", seed, args.num_workers, train=True)
    valid_loader = phase3d_make_loader(config, "valid", seed + 100, args.num_workers, train=False)
    model, criterion, identity = build_label_model("B2", config)
    model.to(args.device)
    criterion.to(args.device)
    optimizer = common.create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
    binding = build_binding(registration, admission, "B2", seed, identity)
    latest = run_dir / "latest.pt"
    start_epoch, history = 0, []
    if latest.is_file():
        if not args.resume:
            raise FileExistsError(f"{latest} exists; use --resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("B2 resume binding mismatch")
        if state.get("test_evaluated") is not False:
            raise ValueError("B2 checkpoint must remain test-sealed")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, history = int(state["epoch"]), list(state["history"])
        common.restore_rng(state["rng"], train_loader)
    common.dump(run_dir / "binding.json", binding)
    info = common.model_info(model)
    started = time.perf_counter()
    for epoch in range(start_epoch, 80):
        train = common.train_epoch(model, train_loader, criterion, optimizer, args.device)
        valid, _ = common.frozen_baseline.evaluate(model, valid_loader, criterion, config, args.device)
        history.append({"epoch": epoch + 1, "train": train, "valid": valid, "lr": optimizer.param_groups[0]["lr"]})
        state = {
            "binding": binding,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng": common.rng_state(train_loader),
            "history": history,
            "test_evaluated": False,
            "phase3d_test_accessed": False,
        }
        common.atomic_torch_save(latest, state)
        common.dump(run_dir / "history.json", history)
        common.dump(run_dir / "status.json", {"status": "TRAINING", "epoch": epoch + 1, "test_evaluated": False, "phase3d_test_accessed": False, "binding": binding})
        print(registration.dataset_id, "B2", f"seed={seed}", f"epoch={epoch + 1}/80", f"valid_mAP50_95={valid['mAP50_95']:.6f}", flush=True)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if state["epoch"] != 80:
        raise ValueError("Phase 3D B2 gate requires epoch80 latest checkpoint")
    result = {
        "artifact_type": "phase3d_b2_training_complete",
        "dataset": registration.dataset_name,
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": "B2",
        "architecture_id": B2_ID,
        "seed": seed,
        "checkpoint_policy": "epoch80/latest.pt",
        "checkpoint_sha256": file_sha256(latest),
        "history_sha256": file_sha256(run_dir / "history.json"),
        "config_sha256": binding["baseline_config_sha256"],
        "split_manifest_sha256": admission["sha256"],
        "evaluator_hash": binding["evaluator_hash"],
        "training_protocol_hash": binding["training_protocol_hash"],
        "git_commit": binding["git_commit"],
        "epoch": 80,
        "training_complete": True,
        "valid_epoch80": history[-1]["valid"],
        **info,
        "training_time_seconds": sum(float(row["train"]["seconds"]) for row in history),
        "wall_time_seconds_this_invocation": time.perf_counter() - started,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "phase3d_test_accessed": False,
        "binding": binding,
        "completed_at": common.now(),
    }
    common.dump(summary_path, result)
    common.dump(run_dir / "status.json", {"status": "TRAINING_COMPLETE_TEST_FORBIDDEN", "epoch": 80, "test_evaluated": False, "phase3d_test_accessed": False, "binding": binding})
    return result


def phase3c_result_path(registration: common.DatasetRegistration, label: str, seed: int) -> Path:
    kind = PHASE3C_KIND[label]
    return registration.output_dir.resolve() / f"{kind}_seed{seed}.json"


def import_phase3c_valid_epoch80(registration: common.DatasetRegistration, label: str, seed: int, admission: Mapping[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Reuse Phase 3C epoch80 validation metrics without opening test metrics or a test loader.

    This is the training-loop `evaluate()` result on the clean development validation
    split at the frozen epoch80/latest checkpoint. It is used only when the Phase 3C
    `.pt` file is not present for a dedicated second inference pass.
    """
    if label not in ("D", "B1"):
        raise ValueError(label)
    frozen = freeze_record(registration, label, seed)
    source_path = phase3c_result_path(registration, label, seed)
    if not source_path.is_file():
        raise FileNotFoundError(f"Phase 3C result JSON missing for cached validation: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_sha256") != frozen["checkpoint_sha256"]:
        raise ValueError("Phase 3C result JSON checkpoint hash does not match the freeze manifest")
    if payload.get("checkpoint_policy") != "epoch80/latest.pt":
        raise ValueError("Phase 3C result is not an epoch80/latest checkpoint")
    if payload.get("split_manifest_sha256") != admission["sha256"]:
        raise ValueError("Phase 3C result split manifest does not match Phase 3D admission")
    valid = payload.get("valid_epoch80")
    if not isinstance(valid, dict) or "mAP50_95" not in valid:
        raise ValueError("Phase 3C result lacks valid_epoch80")
    identity = identity_for_label(label, int(config["dataset"]["num_classes"]))
    binding = build_binding(registration, admission, label, seed, identity)
    result = {
        "artifact_type": "phase3d_validation_result",
        "dataset": registration.dataset_name,
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": label,
        "architecture_id": identity["architecture_id"],
        "seed": seed,
        "split": "valid",
        "checkpoint_policy": "epoch80/latest.pt",
        "checkpoint_source": "phase3c_frozen",
        "checkpoint_path": frozen["checkpoint_path"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "config_sha256": binding["baseline_config_sha256"],
        "split_manifest_sha256": admission["sha256"],
        "evaluator_hash": binding["evaluator_hash"],
        "training_protocol_hash": binding["training_protocol_hash"],
        "git_commit": binding["git_commit"],
        "valid_metrics": {**extract_metrics(valid), **valid},
        "Params": payload.get("Params"),
        "MACs": payload.get("MACs"),
        "FLOPs": payload.get("FLOPs"),
        "evaluation_mode": "phase3c_epoch80_valid_cached",
        "test_evaluated": False,
        "test_used_for_selection": False,
        "phase3d_test_accessed": False,
        "binding": binding,
        "completed_at": common.now(),
    }
    if "test_metrics" in result:
        raise ValueError("cached Phase 3D validation must not copy test_metrics")
    return result


def import_b2_valid_epoch80(registration: common.DatasetRegistration, seed: int, admission: Mapping[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary_path = b2_summary_path(registration, seed)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Phase 3D B2 training summary missing: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("epoch") != 80 or payload.get("test_evaluated") is not False:
        raise ValueError("B2 training summary is not an epoch80 test-sealed result")
    valid = payload.get("valid_epoch80")
    if not isinstance(valid, dict) or "mAP50_95" not in valid:
        raise ValueError("B2 training summary lacks valid_epoch80")
    identity = identity_for_label("B2", int(config["dataset"]["num_classes"]))
    binding = build_binding(registration, admission, "B2", seed, identity)
    checkpoint = b2_run_dir(registration, seed) / "latest.pt"
    return {
        "artifact_type": "phase3d_validation_result",
        "dataset": registration.dataset_name,
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": "B2",
        "architecture_id": B2_ID,
        "seed": seed,
        "split": "valid",
        "checkpoint_policy": "epoch80/latest.pt",
        "checkpoint_source": "phase3d_new",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "config_sha256": binding["baseline_config_sha256"],
        "split_manifest_sha256": admission["sha256"],
        "evaluator_hash": binding["evaluator_hash"],
        "training_protocol_hash": binding["training_protocol_hash"],
        "git_commit": binding["git_commit"],
        "valid_metrics": {**extract_metrics(valid), **valid},
        "Params": payload.get("Params"),
        "MACs": payload.get("MACs"),
        "FLOPs": payload.get("FLOPs"),
        "evaluation_mode": "phase3d_epoch80_valid_from_training",
        "test_evaluated": False,
        "test_used_for_selection": False,
        "phase3d_test_accessed": False,
        "binding": binding,
        "completed_at": common.now(),
    }


def evaluate_validation(registration: common.DatasetRegistration, label: str, seed: int, device: str, num_workers: int, admission: Mapping[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    install_phase3d_test_ban()
    path = validation_summary_path(registration, label, seed)
    identity = identity_for_label(label, int(config["dataset"]["num_classes"]))
    binding = build_binding(registration, admission, label, seed, identity)
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("split") != "valid" or result.get("phase3d_test_accessed") is not False:
            raise ValueError(f"existing validation summary rejected: {path}")
        if "test_metrics" in result:
            raise ValueError(f"Phase 3D validation artifact must not contain test_metrics: {path}")
        return result
    if label in ("D", "B1"):
        try:
            checkpoint = resolve_phase3c_checkpoint(registration, label, seed)
            frozen = freeze_record(registration, label, seed)
            source = "phase3c_frozen"
            expected_sha = frozen["checkpoint_sha256"]
        except FileNotFoundError:
            result = import_phase3c_valid_epoch80(registration, label, seed, admission, config)
            common.dump(path, result)
            return result
    elif label == "B2":
        checkpoint = b2_run_dir(registration, seed) / "latest.pt"
        if checkpoint.is_file() and torch.cuda.is_available() and str(device).startswith("cuda"):
            source = "phase3d_new"
            expected_sha = file_sha256(checkpoint)
        else:
            result = import_b2_valid_epoch80(registration, seed, admission, config)
            common.dump(path, result)
            return result
    else:
        raise ValueError(label)
    require_cuda(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 80:
        raise ValueError(f"{label} checkpoint is not epoch80/latest")
    if "best_valid" in str(state.get("binding", {})).lower() or "best.pt" in str(checkpoint):
        raise ValueError("best-valid checkpoints are forbidden in Phase 3D")
    actual_sha = file_sha256(checkpoint)
    if label in ("D", "B1") and actual_sha != expected_sha:
        raise ValueError("frozen Phase 3C checkpoint hash changed")
    model, criterion, actual_identity = build_label_model(label, config)
    if actual_identity["architecture_id"] != identity["architecture_id"]:
        raise ValueError("model identity mismatch")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    criterion.to(device)
    valid_loader = phase3d_make_loader(config, "valid", seed + 100, num_workers, train=False)
    metrics, _ = common.frozen_baseline.evaluate(model, valid_loader, criterion, config, device)
    info = common.model_info(model)
    result = {
        "artifact_type": "phase3d_validation_result",
        "dataset": registration.dataset_name,
        "dataset_id": registration.dataset_id,
        "modality": registration.modality,
        "model": label,
        "architecture_id": identity["architecture_id"],
        "seed": seed,
        "split": "valid",
        "checkpoint_policy": "epoch80/latest.pt",
        "checkpoint_source": source,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "config_sha256": binding["baseline_config_sha256"],
        "split_manifest_sha256": admission["sha256"],
        "evaluator_hash": binding["evaluator_hash"],
        "training_protocol_hash": binding["training_protocol_hash"],
        "git_commit": binding["git_commit"],
        "valid_metrics": {**extract_metrics(metrics), **metrics},
        "Params": info["Params"],
        "MACs": info["MACs"],
        "FLOPs": info["FLOPs"],
        "evaluation_mode": "phase3d_dedicated_valid_inference",
        "test_evaluated": False,
        "test_used_for_selection": False,
        "phase3d_test_accessed": False,
        "binding": binding,
        "completed_at": common.now(),
    }
    common.dump(path, result)
    return result


def load_validation_matrix() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    for registration in REGISTRATIONS:
        for label in ("D", "B2", "B1"):
            for seed in FORMAL_SEEDS:
                path = validation_summary_path(registration, label, seed)
                if not path.is_file():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("split") != "valid" or result.get("phase3d_test_accessed") is not False:
                    raise ValueError(f"validation artifact rejected: {path}")
                if "test_metrics" in result:
                    raise ValueError(f"Phase 3D validation artifact must not contain test_metrics: {path}")
                rows[(registration.dataset_id, label, seed)] = result
    return rows


def summarize_effect(values: tuple[float, float, float]) -> dict[str, Any]:
    row = EffectRow(*values)
    return {
        "seed0": row.seed0,
        "seed1": row.seed1,
        "seed2": row.seed2,
        "mean": row.mean(),
        "sample_std": row.std(),
        "min": min(row.values()),
        "max": max(row.values()),
        "positive_seed_count": row.positive_seed_count(),
        "seed_direction": [value > 0 for value in row.values()],
    }


def classify_cross_dataset(dataset_effects: Mapping[str, Mapping[str, Any]], *, supported: str, domain: str, not_supported: str) -> str:
    supported_flags = {}
    signs = {}
    for dataset_id, effect in dataset_effects.items():
        mean_positive = float(effect["mean"]) > 0
        seeds_ok = int(effect["positive_seed_count"]) >= 2
        supported_flags[dataset_id] = mean_positive and seeds_ok
        signs[dataset_id] = float(effect["mean"]) > 0
    if all(supported_flags.values()):
        return supported
    if any(signs.values()) and not all(signs.values()):
        return domain
    if any(supported_flags.values()) and not all(supported_flags.values()):
        return domain
    return not_supported


def interpret_structure(p2_state: str, k5_state: str, total_effects: Mapping[str, Mapping[str, Any]]) -> str:
    p2_ok = p2_state == "P2_CROSS_DATASET_SUPPORTED"
    k5_ok = k5_state == "K5_BUNDLE_CROSS_DATASET_SUPPORTED"
    if p2_ok and k5_ok:
        return "BOTH_SUPPORTED"
    if p2_ok and k5_state == "K5_BUNDLE_NOT_SUPPORTED":
        return "P2_DOMINANT"
    if k5_ok and p2_state == "P2_NOT_SUPPORTED_CROSS_DATASET":
        return "K5_BUNDLE_DOMINANT"
    total_positive = all(float(effect["mean"]) > 0 for effect in total_effects.values())
    if total_positive and not p2_ok and not k5_ok:
        return "INTERACTION_OR_UNRESOLVED"
    return "INTERACTION_OR_UNRESOLVED"


def sonarop_p1_gate(k5_state: str, k5_effects: Mapping[str, Mapping[str, Any]]) -> str:
    if k5_state != "K5_BUNDLE_CROSS_DATASET_SUPPORTED":
        return "NO"
    all_three = all(int(effect["positive_seed_count"]) == 3 for effect in k5_effects.values())
    strong = all(float(effect["mean"]) > float(effect["sample_std"]) for effect in k5_effects.values())
    if all_three and strong:
        return "YES"
    return "CONDITIONAL"


def stagewise_probe_needed(interpretation: str) -> str:
    # Stage-wise k3/k5 probes would refine operators, but they do not change the
    # coarse next-search-space decision already implied by P2 vs selected k5 bundle.
    if interpretation in {"BOTH_SUPPORTED", "P2_DOMINANT", "K5_BUNDLE_DOMINANT"}:
        return "NO"
    return "NO"


def metric(row: Mapping[str, Any], name: str) -> float:
    metrics = row["valid_metrics"]
    if name == "mAP50":
        return float(metrics.get("mAP50", metrics["AP50"]))
    return float(metrics[name])


def decompose(matrix: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    missing = [
        f"{registration.dataset_id}/{label}/seed{seed}"
        for registration in REGISTRATIONS
        for label in ("D", "B2", "B1")
        for seed in FORMAL_SEEDS
        if (registration.dataset_id, label, seed) not in matrix
    ]
    if missing:
        raise FileNotFoundError(f"Phase 3D validation matrix incomplete: {missing}")
    payload: dict[str, Any] = {"datasets": {}, "phase3d_test_accessed": False}
    for registration in REGISTRATIONS:
        dataset_id = registration.dataset_id
        p2, k5, total = [], [], []
        p2_50, k5_50, total_50 = [], [], []
        per_seed = []
        for seed in FORMAL_SEEDS:
            d = matrix[(dataset_id, "D", seed)]
            b2 = matrix[(dataset_id, "B2", seed)]
            b1 = matrix[(dataset_id, "B1", seed)]
            delta_p2 = metric(b1, "mAP50_95") - metric(b2, "mAP50_95")
            delta_k = metric(b2, "mAP50_95") - metric(d, "mAP50_95")
            delta_total = metric(b1, "mAP50_95") - metric(d, "mAP50_95")
            if abs((delta_p2 + delta_k) - delta_total) > FLOAT_TOLERANCE:
                raise ValueError(f"additivity failed on {dataset_id} seed{seed}")
            p2.append(delta_p2)
            k5.append(delta_k)
            total.append(delta_total)
            d50 = metric(b1, "mAP50") - metric(b2, "mAP50")
            k50 = metric(b2, "mAP50") - metric(d, "mAP50")
            t50 = metric(b1, "mAP50") - metric(d, "mAP50")
            p2_50.append(d50)
            k5_50.append(k50)
            total_50.append(t50)
            per_seed.append(
                {
                    "seed": seed,
                    "D": extract_metrics(d["valid_metrics"]),
                    "B2": extract_metrics(b2["valid_metrics"]),
                    "B1": extract_metrics(b1["valid_metrics"]),
                    "delta_P2": delta_p2,
                    "delta_K": delta_k,
                    "delta_TOTAL": delta_total,
                    "delta_P2_mAP50": d50,
                    "delta_K_mAP50": k50,
                    "delta_TOTAL_mAP50": t50,
                    "D_checkpoint_sha256": d["checkpoint_sha256"],
                    "B2_checkpoint_sha256": b2["checkpoint_sha256"],
                    "B1_checkpoint_sha256": b1["checkpoint_sha256"],
                    "D_checkpoint_source": d["checkpoint_source"],
                    "B2_checkpoint_source": b2["checkpoint_source"],
                    "B1_checkpoint_source": b1["checkpoint_source"],
                }
            )
        payload["datasets"][dataset_id] = {
            "dataset": registration.dataset_name,
            "modality": registration.modality,
            "split_manifest_sha256": matrix[(dataset_id, "D", 0)]["split_manifest_sha256"],
            "per_seed": per_seed,
            "delta_P2": summarize_effect(tuple(p2)),
            "delta_K": summarize_effect(tuple(k5)),
            "delta_TOTAL": summarize_effect(tuple(total)),
            "delta_P2_mAP50": summarize_effect(tuple(p2_50)),
            "delta_K_mAP50": summarize_effect(tuple(k5_50)),
            "delta_TOTAL_mAP50": summarize_effect(tuple(total_50)),
        }
    p2_state = classify_cross_dataset(
        {dataset_id: row["delta_P2"] for dataset_id, row in payload["datasets"].items()},
        supported="P2_CROSS_DATASET_SUPPORTED",
        domain="P2_DOMAIN_DEPENDENT",
        not_supported="P2_NOT_SUPPORTED_CROSS_DATASET",
    )
    k5_state = classify_cross_dataset(
        {dataset_id: row["delta_K"] for dataset_id, row in payload["datasets"].items()},
        supported="K5_BUNDLE_CROSS_DATASET_SUPPORTED",
        domain="K5_BUNDLE_DOMAIN_DEPENDENT",
        not_supported="K5_BUNDLE_NOT_SUPPORTED",
    )
    total_effects = {dataset_id: row["delta_TOTAL"] for dataset_id, row in payload["datasets"].items()}
    interpretation = interpret_structure(p2_state, k5_state, total_effects)
    payload.update(
        {
            "p2_state": p2_state,
            "k5_state": k5_state,
            "structural_interpretation": interpretation,
            "stagewise_probe_needed": stagewise_probe_needed(interpretation),
            "sonarop_p1_evidence_gate": sonarop_p1_gate(k5_state, {dataset_id: row["delta_K"] for dataset_id, row in payload["datasets"].items()}),
            "b2_formal_runs": "6/6",
            "additivity": ADDITIVITY_NOTE,
        }
    )
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(effect: Mapping[str, Any]) -> str:
    return f"{effect['mean']:+.6f} ± {effect['sample_std']:.6f}"


def protocol_text() -> str:
    return """# Phase 3D Protocol

Status: frozen before B2 training.

## Objective

Attribute the Phase 3C cross-dataset B1 gain at coarse grain only:

- A. P2 / four-scale feature output（P2 / 四尺度特征输出）
- B. selected k5 kernel bundle（选定的 k5 感受野算子组合）
- C. BOTH
- D. interaction / unresolved

No architecture re-search. No stage-wise k3/k5 probe is started automatically.

## Frozen architectures

- D / Frozen Baseline: `P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3`
- B2: `P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5`
- B1: `P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5`

Matched differences:

- B1 vs B2 isolates P2 / four-scale output.
- B2 vs D is the selected k5 bundle increment, not a single-stage causal experiment.

## Data and metric policy

- Split: frozen Phase 3B clean development validation only.
- Primary metric: mAP50:95（COCO 风格 mAP@[0.50:0.95]）.
- Secondary: Precision, Recall, mAP50.
- Checkpoint: epoch80/`latest.pt` only. Best-valid selection is forbidden.
- Test: forbidden. Any Phase 3D attempt to build a UATD/SCTD test dataloader raises immediately.
- Phase 3C test metrics are not reused.

## Training protocol inherited from Phase 3C

- input 640, epochs 80, batch size 4
- optimizer AdamW, lr 0.001, weight decay 0.0001, scheduler none
- same augmentation, class-balanced sampler, evaluator, NMS, class adaptation
- independent random initialization for each B2 seed
- D and B1 are not retrained

## New runs

Exactly six formal runs: B2 × {UATD v3, classic SCTD 1.0} × seeds {0,1,2}.

## Decomposition

For each dataset and seed:

- `delta_P2 = mAP50:95(B1) - mAP50:95(B2)`
- `delta_K = mAP50:95(B2) - mAP50:95(D)`
- `delta_TOTAL = mAP50:95(B1) - mAP50:95(D)`

`delta_P2 + delta_K` must equal `delta_TOTAL` except for floating-point precision.

## Cross-dataset states

A factor is `CROSS_DATASET_SUPPORTED` when both datasets have mean delta > 0 and at least 2/3 seeds > 0. Opposite dataset signs or clearly inconsistent direction yield `DOMAIN_DEPENDENT`. Otherwise `NOT_SUPPORTED`.

Structural labels: `P2_DOMINANT`, `K5_BUNDLE_DOMINANT`, `BOTH_SUPPORTED`, `INTERACTION_OR_UNRESOLVED`. Do not force a winner.

## Stop rule

Phase 3D stops at P2 vs selected k5 bundle. Stage-wise probes and SonarOp P1 design wait for human review. Phase 3D does not implement SonarOp P1. This phase does not modify the frozen general search space.
"""


def pending_payload(cuda_available: bool) -> dict[str, Any]:
    hashes = protocol_hashes()
    return {
        "artifact_type": "phase3d_manifest",
        "status": "READY_FOR_GPU_EXECUTION" if not cuda_available else "IMPLEMENTATION_READY",
        "b2_formal_runs": "0/6",
        "phase3d_test_accessed": False,
        "cuda_available": cuda_available,
        "architectures": {key: value["architecture_id"] for key, value in architecture_map().items()},
        "seeds": list(FORMAL_SEEDS),
        "git_commit": common.git_commit(),
        **hashes,
        "freeze_manifest": str(FREEZE_MANIFEST),
        "report_dir": str(REPORT_DIR),
        "created_at": common.now(),
    }


def write_protocol_and_pending_reports(*, cuda_available: bool, analysis: dict[str, Any] | None = None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "PHASE3D_PROTOCOL.md").write_text(protocol_text(), encoding="utf-8")
    if analysis is None:
        common.dump(REPORT_DIR / "PHASE3D_MANIFEST.json", pending_payload(cuda_available))
        write_csv(REPORT_DIR / "B2_MULTISEED_RESULTS.csv", [{"status": "FORMAL_RESULTS_PENDING"}])
        write_csv(REPORT_DIR / "VALIDATION_RESULTS_D_B2_B1.csv", [{"status": "FORMAL_RESULTS_PENDING"}])
        write_csv(REPORT_DIR / "P2_MATCHED_EFFECT.csv", [{"status": "FORMAL_RESULTS_PENDING"}])
        write_csv(REPORT_DIR / "K5_BUNDLE_MATCHED_EFFECT.csv", [{"status": "FORMAL_RESULTS_PENDING"}])
        write_csv(REPORT_DIR / "CROSS_DATASET_STRUCTURE_SUMMARY.csv", [{"status": "FORMAL_RESULTS_PENDING"}])
        pending = "FORMAL_RESULTS_PENDING. Status: `READY_FOR_GPU_EXECUTION`." if not cuda_available else "FORMAL_RESULTS_PENDING."
        (REPORT_DIR / "STRUCTURAL_DECOMPOSITION.md").write_text(
            "# Phase 3D Structural Decomposition\n\n"
            f"{pending}\n\n"
            "Official comparisons use clean development validation only. "
            "`phase3d_test_accessed=false`.\n",
            encoding="utf-8",
        )
        (REPORT_DIR / "GENERAL_SEARCH_SPACE_EVIDENCE_UPDATE.md").write_text(
            "# General Search Space Evidence Update\n\n"
            f"{pending}\n\n"
            "Frozen Search Space V0.1 is unchanged. This file will record candidate-variable suggestions after formal validation deltas exist. No automatic search-space mutation is performed.\n",
            encoding="utf-8",
        )
        (REPORT_DIR / "SONAROP_P1_EVIDENCE_GATE.md").write_text(
            "# SonarOp P1 Evidence Gate\n\n"
            f"{pending}\n\n"
            "SONAROP_P1_EVIDENCE_GATE is not claimed from smoke or CPU setup. Gate values are YES / CONDITIONAL / NO only after formal k5-bundle validation deltas exist.\n",
            encoding="utf-8",
        )
        return
    write_completed_reports(analysis)


def write_completed_reports(analysis: dict[str, Any]) -> None:
    hashes = protocol_hashes()
    b2_rows, valid_rows, p2_rows, k5_rows, summary_rows = [], [], [], [], []
    records = []
    for registration in REGISTRATIONS:
        dataset = analysis["datasets"][registration.dataset_id]
        for seed_row in dataset["per_seed"]:
            seed = seed_row["seed"]
            for label in ("D", "B2", "B1"):
                path = validation_summary_path(registration, label, seed)
                result = json.loads(path.read_text(encoding="utf-8"))
                metrics = extract_metrics(result["valid_metrics"])
                row = {
                    "dataset": registration.dataset_name,
                    "dataset_id": registration.dataset_id,
                    "modality": registration.modality,
                    "seed": seed,
                    "architecture_label": label,
                    "architecture_id": result["architecture_id"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "mAP50": metrics["mAP50"],
                    "mAP50_95": metrics["mAP50_95"],
                    "checkpoint_source": result["checkpoint_source"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "config_sha256": result["config_sha256"],
                    "split_manifest_sha256": result["split_manifest_sha256"],
                    "git_commit": result["git_commit"],
                    "evaluator_hash": result["evaluator_hash"],
                    "training_protocol_hash": result["training_protocol_hash"],
                }
                valid_rows.append(row)
                records.append(row)
                if label == "B2":
                    b2_rows.append(row)
            p2_rows.append(
                {
                    "dataset": registration.dataset_name,
                    "dataset_id": registration.dataset_id,
                    "modality": registration.modality,
                    "seed": seed,
                    "B1_mAP50_95": seed_row["B1"]["mAP50_95"],
                    "B2_mAP50_95": seed_row["B2"]["mAP50_95"],
                    "delta_P2_mAP50_95": seed_row["delta_P2"],
                    "delta_P2_mAP50": seed_row["delta_P2_mAP50"],
                }
            )
            k5_rows.append(
                {
                    "dataset": registration.dataset_name,
                    "dataset_id": registration.dataset_id,
                    "modality": registration.modality,
                    "seed": seed,
                    "B2_mAP50_95": seed_row["B2"]["mAP50_95"],
                    "D_mAP50_95": seed_row["D"]["mAP50_95"],
                    "delta_K_mAP50_95": seed_row["delta_K"],
                    "delta_K_mAP50": seed_row["delta_K_mAP50"],
                }
            )
        summary_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "delta_P2_mean": dataset["delta_P2"]["mean"],
                "delta_P2_std": dataset["delta_P2"]["sample_std"],
                "delta_P2_min": dataset["delta_P2"]["min"],
                "delta_P2_max": dataset["delta_P2"]["max"],
                "delta_P2_positive_seed_count": dataset["delta_P2"]["positive_seed_count"],
                "delta_K_mean": dataset["delta_K"]["mean"],
                "delta_K_std": dataset["delta_K"]["sample_std"],
                "delta_K_min": dataset["delta_K"]["min"],
                "delta_K_max": dataset["delta_K"]["max"],
                "delta_K_positive_seed_count": dataset["delta_K"]["positive_seed_count"],
                "delta_TOTAL_mean": dataset["delta_TOTAL"]["mean"],
                "delta_TOTAL_std": dataset["delta_TOTAL"]["sample_std"],
                "delta_TOTAL_min": dataset["delta_TOTAL"]["min"],
                "delta_TOTAL_max": dataset["delta_TOTAL"]["max"],
                "delta_TOTAL_positive_seed_count": dataset["delta_TOTAL"]["positive_seed_count"],
                "p2_state": analysis["p2_state"],
                "k5_state": analysis["k5_state"],
                "structural_interpretation": analysis["structural_interpretation"],
                "split_manifest_sha256": dataset["split_manifest_sha256"],
            }
        )
    for registration in REGISTRATIONS:
        dataset = analysis["datasets"][registration.dataset_id]
        p2 = dataset["delta_P2"]
        k5 = dataset["delta_K"]
        p2_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "mean",
                "B1_mAP50_95": "",
                "B2_mAP50_95": "",
                "delta_P2_mAP50_95": p2["mean"],
                "delta_P2_mAP50": dataset["delta_P2_mAP50"]["mean"],
            }
        )
        p2_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "sample_std",
                "B1_mAP50_95": "",
                "B2_mAP50_95": "",
                "delta_P2_mAP50_95": p2["sample_std"],
                "delta_P2_mAP50": dataset["delta_P2_mAP50"]["sample_std"],
            }
        )
        p2_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "min",
                "B1_mAP50_95": "",
                "B2_mAP50_95": "",
                "delta_P2_mAP50_95": p2["min"],
                "delta_P2_mAP50": dataset["delta_P2_mAP50"]["min"],
            }
        )
        p2_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "max",
                "B1_mAP50_95": "",
                "B2_mAP50_95": "",
                "delta_P2_mAP50_95": p2["max"],
                "delta_P2_mAP50": dataset["delta_P2_mAP50"]["max"],
            }
        )
        p2_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "positive_seed_count",
                "B1_mAP50_95": "",
                "B2_mAP50_95": "",
                "delta_P2_mAP50_95": p2["positive_seed_count"],
                "delta_P2_mAP50": dataset["delta_P2_mAP50"]["positive_seed_count"],
            }
        )
        k5_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "mean",
                "B2_mAP50_95": "",
                "D_mAP50_95": "",
                "delta_K_mAP50_95": k5["mean"],
                "delta_K_mAP50": dataset["delta_K_mAP50"]["mean"],
            }
        )
        k5_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "sample_std",
                "B2_mAP50_95": "",
                "D_mAP50_95": "",
                "delta_K_mAP50_95": k5["sample_std"],
                "delta_K_mAP50": dataset["delta_K_mAP50"]["sample_std"],
            }
        )
        k5_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "min",
                "B2_mAP50_95": "",
                "D_mAP50_95": "",
                "delta_K_mAP50_95": k5["min"],
                "delta_K_mAP50": dataset["delta_K_mAP50"]["min"],
            }
        )
        k5_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "max",
                "B2_mAP50_95": "",
                "D_mAP50_95": "",
                "delta_K_mAP50_95": k5["max"],
                "delta_K_mAP50": dataset["delta_K_mAP50"]["max"],
            }
        )
        k5_rows.append(
            {
                "dataset": registration.dataset_name,
                "dataset_id": registration.dataset_id,
                "modality": registration.modality,
                "seed": "positive_seed_count",
                "B2_mAP50_95": "",
                "D_mAP50_95": "",
                "delta_K_mAP50_95": k5["positive_seed_count"],
                "delta_K_mAP50": dataset["delta_K_mAP50"]["positive_seed_count"],
            }
        )
    write_csv(REPORT_DIR / "B2_MULTISEED_RESULTS.csv", b2_rows)
    write_csv(REPORT_DIR / "VALIDATION_RESULTS_D_B2_B1.csv", valid_rows)
    write_csv(REPORT_DIR / "P2_MATCHED_EFFECT.csv", p2_rows)
    write_csv(REPORT_DIR / "K5_BUNDLE_MATCHED_EFFECT.csv", k5_rows)
    write_csv(REPORT_DIR / "CROSS_DATASET_STRUCTURE_SUMMARY.csv", summary_rows)

    uatd = analysis["datasets"]["uatd_v3"]
    sctd = analysis["datasets"]["classic_sctd_1_0"]
    p2_state = analysis["p2_state"]
    k5_state = analysis["k5_state"]
    interpretation = analysis["structural_interpretation"]
    gate = analysis["sonarop_p1_evidence_gate"]
    probe = analysis["stagewise_probe_needed"]

    lines = [
        "# Phase 3D Structural Decomposition",
        "",
        "All comparisons use the clean development validation split. `phase3d_test_accessed=false`.",
        "",
        "## Matched identity",
        "",
        "- `delta_P2 = mAP50:95(B1) - mAP50:95(B2)` isolates P2 / four-scale output（P2 / 四尺度特征输出）.",
        "- `delta_K = mAP50:95(B2) - mAP50:95(D)` is the selected k5 bundle increment（选定 k5 感受野组合）, not a single-stage causal effect.",
        "- `delta_TOTAL = mAP50:95(B1) - mAP50:95(D)`.",
        f"- {ADDITIVITY_NOTE}.",
        "",
        "## UATD v3 (MFLS)",
        "",
        f"- delta_P2 mean ± std: `{fmt_pm(uatd['delta_P2'])}`; seeds `{uatd['delta_P2']['seed0']:+.6f}`, `{uatd['delta_P2']['seed1']:+.6f}`, `{uatd['delta_P2']['seed2']:+.6f}`; positive `{uatd['delta_P2']['positive_seed_count']}/3`.",
        f"- delta_K mean ± std: `{fmt_pm(uatd['delta_K'])}`; seeds `{uatd['delta_K']['seed0']:+.6f}`, `{uatd['delta_K']['seed1']:+.6f}`, `{uatd['delta_K']['seed2']:+.6f}`; positive `{uatd['delta_K']['positive_seed_count']}/3`.",
        f"- delta_TOTAL mean ± std: `{fmt_pm(uatd['delta_TOTAL'])}`; seeds `{uatd['delta_TOTAL']['seed0']:+.6f}`, `{uatd['delta_TOTAL']['seed1']:+.6f}`, `{uatd['delta_TOTAL']['seed2']:+.6f}`; positive `{uatd['delta_TOTAL']['positive_seed_count']}/3`.",
        "",
        "## classic SCTD / SCTD 1.0 (SSS)",
        "",
        f"- delta_P2 mean ± std: `{fmt_pm(sctd['delta_P2'])}`; seeds `{sctd['delta_P2']['seed0']:+.6f}`, `{sctd['delta_P2']['seed1']:+.6f}`, `{sctd['delta_P2']['seed2']:+.6f}`; positive `{sctd['delta_P2']['positive_seed_count']}/3`.",
        f"- delta_K mean ± std: `{fmt_pm(sctd['delta_K'])}`; seeds `{sctd['delta_K']['seed0']:+.6f}`, `{sctd['delta_K']['seed1']:+.6f}`, `{sctd['delta_K']['seed2']:+.6f}`; positive `{sctd['delta_K']['positive_seed_count']}/3`.",
        f"- delta_TOTAL mean ± std: `{fmt_pm(sctd['delta_TOTAL'])}`; seeds `{sctd['delta_TOTAL']['seed0']:+.6f}`, `{sctd['delta_TOTAL']['seed1']:+.6f}`, `{sctd['delta_TOTAL']['seed2']:+.6f}`; positive `{sctd['delta_TOTAL']['positive_seed_count']}/3`.",
        "",
        "## States",
        "",
        f"- P2: `{p2_state}`",
        f"- K5 bundle: `{k5_state}`",
        f"- Structural interpretation: `{interpretation}`",
        f"- Stage-wise probe needed: `{probe}`",
        f"- SonarOp P1 evidence gate: `{gate}`",
        "",
        "No stage-wise experiment, SonarOp P1 design, general-search-space mutation, or new NAS is started automatically.",
    ]
    (REPORT_DIR / "STRUCTURAL_DECOMPOSITION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suggestions = ["# General Search Space Evidence Update", "", "Frozen Search Space V0.1 is not modified by Phase 3D.", ""]
    if p2_state == "P2_CROSS_DATASET_SUPPORTED":
        suggestions.append("Suggestion: P2 / four-scale output becomes a future general sonar search-space candidate variable（候选搜索变量）.")
    if k5_state == "K5_BUNDLE_CROSS_DATASET_SUPPORTED":
        suggestions.append("Suggestion: variable receptive-field / k3-k5 operator family becomes a future general sonar search-space candidate dimension（候选搜索维度）.")
    if interpretation == "BOTH_SUPPORTED":
        suggestions.append("Both P2 and the k3-k5 family may enter the next search-space design discussion. This is not an automatic space mutation.")
    if interpretation == "INTERACTION_OR_UNRESOLVED":
        suggestions.append("Retain the interaction hypothesis. Do not inflate a single-factor search-space claim.")
    if p2_state != "P2_CROSS_DATASET_SUPPORTED" and k5_state != "K5_BUNDLE_CROSS_DATASET_SUPPORTED" and interpretation != "INTERACTION_OR_UNRESOLVED":
        suggestions.append("No new general-search-space variable is suggested from Phase 3D.")
    suggestions += ["", f"Structural interpretation: `{interpretation}`.", "", "Human review is required before any search-space change."]
    (REPORT_DIR / "GENERAL_SEARCH_SPACE_EVIDENCE_UPDATE.md").write_text("\n".join(suggestions) + "\n", encoding="utf-8")

    gate_lines = [
        "# SonarOp P1 Evidence Gate",
        "",
        f"SONAROP_P1_EVIDENCE_GATE = `{gate}`",
        "",
        "This gate only authorizes a later prototype discussion. It does not validate SonarOp.",
        "Phase 3D does not implement SonarOp P1.",
        "",
        f"K5 bundle state: `{k5_state}`.",
        f"Seed-consistency/magnitude rule: YES requires cross-dataset support, 3/3 positive seeds on both datasets, and mean > sample std on both datasets; otherwise CONDITIONAL if supported, else NO.",
    ]
    (REPORT_DIR / "SONAROP_P1_EVIDENCE_GATE.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    manifest = {
        "artifact_type": "phase3d_manifest",
        "status": "COMPLETE",
        "b2_formal_runs": "6/6",
        "phase3d_test_accessed": False,
        "p2_state": p2_state,
        "k5_state": k5_state,
        "structural_interpretation": interpretation,
        "stagewise_probe_needed": probe,
        "sonarop_p1_evidence_gate": gate,
        "git_commit": common.git_commit(),
        **hashes,
        "records": records,
        "created_at": common.now(),
    }
    common.dump(REPORT_DIR / "PHASE3D_MANIFEST.json", manifest)
    (REPORT_DIR / "PHASE3D_PROTOCOL.md").write_text(protocol_text(), encoding="utf-8")


def smoke(registration: common.DatasetRegistration, device: str) -> dict[str, Any]:
    install_phase3d_test_ban()
    admission = admit_phase3d_split(registration)
    config = phase3d_effective_config(registration)
    assert_matched_decomposition()
    rows = []
    common.set_seed(0)
    train_loader = phase3d_make_loader(config, "train", 0, 0, train=True, max_samples=8)
    valid_loader = phase3d_make_loader(config, "valid", 100, 0, train=False, max_samples=8)
    for label in ("D", "B2", "B1"):
        model, criterion, identity = build_label_model(label, config)
        model.to(device)
        criterion.to(device)
        if label == "B2":
            optimizer = common.create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
            train = common.train_epoch(model, train_loader, criterion, optimizer, device, max_batches=1)
        else:
            train = {"skipped": True, "reason": "Phase 3D does not retrain D or B1"}
        valid, _ = common.frozen_baseline.evaluate(model, valid_loader, criterion, config, device, max_batches=1)
        rows.append({"model": label, "architecture": identity, "train": train, "valid": extract_metrics(valid), "test_evaluated": False})
    payload = {
        "dataset_id": registration.dataset_id,
        "split_manifest_sha256": admission["sha256"],
        "rows": rows,
        "test_evaluated": False,
        "phase3d_test_accessed": False,
        "formal_eligible": False,
    }
    output = output_dir_for(registration)
    common.dump(output / "smoke_result.json", payload)
    return payload


def preflight() -> dict[str, Any]:
    install_phase3d_test_ban()
    assert_matched_decomposition()
    rows = []
    for registration in REGISTRATIONS:
        admission = admit_phase3d_split(registration)
        config = phase3d_effective_config(registration)
        rows.append(
            {
                "dataset_id": registration.dataset_id,
                "dataset_name": registration.dataset_name,
                "modality": registration.modality,
                "split_manifest_sha256": admission["sha256"],
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "image_size": config["dataset"]["image_size"],
                "test_split_paths_present": "test" in config["dataset"]["split_paths"],
            }
        )
    payload = {
        "status": "READY_FOR_GPU_EXECUTION" if not torch.cuda.is_available() else "READY_FOR_B2_FORMAL_RUNS",
        "architectures": {key: value["architecture_id"] for key, value in architecture_map().items()},
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "datasets": rows,
        "phase3d_test_accessed": False,
        "b2_formal_runs": "0/6",
        **protocol_hashes(),
        "git_commit": common.git_commit(),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    common.dump(RESULT_ROOT / "preflight.json", payload)
    write_protocol_and_pending_reports(cuda_available=torch.cuda.is_available())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def source_forbids_test_loader() -> bool:
    source = inspect.getsource(train_b2) + inspect.getsource(evaluate_validation) + inspect.getsource(smoke) + inspect.getsource(phase3d_make_loader)
    return 'make_loader(config, "test"' not in source and "split=\"test\"" not in inspect.getsource(train_b2)


def count_b2_runs() -> int:
    return sum(b2_summary_path(registration, seed).is_file() for registration in REGISTRATIONS for seed in FORMAL_SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3D matched P2 vs k5-bundle structural decomposition")
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal", "evaluate-valid", "analyze"), default="preflight")
    parser.add_argument("--datasets", nargs="+", choices=("uatd_v3", "classic_sctd_1_0", "both"), default=["both"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    install_phase3d_test_ban()
    selected = tuple(REGISTRATIONS) if args.datasets == ["both"] or "both" in args.datasets else tuple(row for row in REGISTRATIONS if row.dataset_id in args.datasets)
    requested_seeds = tuple(args.seeds)
    if args.mode == "preflight":
        preflight()
        return
    if args.mode == "smoke":
        for registration in selected:
            smoke(registration, args.device)
        write_protocol_and_pending_reports(cuda_available=torch.cuda.is_available())
        return
    if args.mode == "formal":
        if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds) or any(seed not in FORMAL_SEEDS for seed in requested_seeds):
            raise ValueError("formal seeds must be a unique subset of frozen seeds 0,1,2")
        require_cuda(args.device)
        for registration in selected:
            admission = admit_phase3d_split(registration)
            config = phase3d_effective_config(registration)
            for seed in requested_seeds:
                train_b2(registration, args, admission, config, seed)
        return
    if args.mode == "evaluate-valid":
        for registration in selected:
            admission = admit_phase3d_split(registration)
            config = phase3d_effective_config(registration)
            for seed in FORMAL_SEEDS:
                for label in ("D", "B2", "B1"):
                    evaluate_validation(registration, label, seed, args.device, args.num_workers, admission, config)
        return
    analysis = decompose(load_validation_matrix())
    write_completed_reports(analysis)
    print(json.dumps({key: analysis[key] for key in ("b2_formal_runs", "p2_state", "k5_state", "structural_interpretation", "stagewise_probe_needed", "sonarop_p1_evidence_gate", "phase3d_test_accessed")}, indent=2))


if __name__ == "__main__":
    main()
