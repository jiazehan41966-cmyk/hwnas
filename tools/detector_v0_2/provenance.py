"""Verify actual frozen training inputs and complete selected-best artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_execution_manifest(path, binding):
    from tools.detector_v0_2 import train as runner
    path = Path(path)
    if not path.is_file() or sha(path) != binding.get("execution_manifest_sha256"):
        raise ValueError("actual execution manifest SHA256 binding mismatch")
    manifest = json.loads(path.read_bytes())
    cfg, baseline = runner.load_config()
    if (manifest.get("config_sha256") != sha(runner.CONFIG) or
            manifest.get("model_spec_sha256") != sha(runner.ROOT / cfg["model_spec"]) or
            manifest.get("training") != cfg["training"] or
            manifest.get("source_sha256") != runner.source_hashes(cfg) or
            manifest.get("test_evaluated") is not False):
        raise ValueError("execution manifest source/config/model/protocol binding mismatch")
    actual = runner.verify_inputs(cfg,baseline)
    recorded = manifest.get("input_verification",{})
    # The legacy additional glob list is an audit of an old gate, not a list
    # of V0.2 training dependencies. Independent deployment work can add files.
    for key in ("datasets", "dataset_metadata_sha256", "historical_frozen_files_verified",
                "test_performance_accessed"):
        if key not in recorded or recorded[key] != actual[key]:
            raise ValueError("execution manifest data/frozen-input binding mismatch: " + key)
    if actual["test_performance_accessed"] is not False:
        raise ValueError("test seal violated")
    return manifest


def verify_completed_training(saved, checkpoint, provenance):
    binding = saved["binding"]
    manifest_path = Path(provenance.get("execution_manifest_path",""))
    verify_execution_manifest(manifest_path,binding)
    folder = Path(provenance.get("training_run_dir",""))
    required = ("status.json","result.json","epoch_log.json","latest.pt","best.pt")
    if not folder.is_dir() or any(not (folder/name).is_file() for name in required):
        raise ValueError("complete training run artifacts are required")
    status = json.loads((folder/"status.json").read_bytes())
    result = json.loads((folder/"result.json").read_bytes())
    history = json.loads((folder/"epoch_log.json").read_bytes())
    if (status.get("status") != "COMPLETE" or status.get("epochs") != 80 or
            status.get("binding") != binding or status.get("test_evaluated") is not False):
        raise ValueError("training is not a matching complete 80-epoch run")
    if (len(history) != 80 or any(type(row.get("epoch")) is not int for row in history) or
            [row["epoch"] for row in history] != list(range(1,81))):
        raise ValueError("complete 80-epoch history is required")
    for row in history:
        valid = row.get("valid",{})
        ap = valid.get("mAP50_95")
        if (row.get("batches") != 83 or row.get("gradient_finite") is not True or
                valid.get("num_images") != 99 or type(ap) not in (int,float) or
                not math.isfinite(ap) or not 0 <= ap <= 1):
            raise ValueError("history does not demonstrate the frozen train330/valid99 protocol")
    best = max(history,key=lambda row:row["valid"]["mAP50_95"])
    if (sha(folder/"best.pt") != sha(checkpoint) or
            result.get("best_checkpoint_sha256") != sha(checkpoint) or
            result.get("latest_checkpoint_sha256") != sha(folder/"latest.pt") or
            result.get("history_sha256") != sha(folder/"epoch_log.json") or
            result.get("execution_manifest_sha256") != sha(manifest_path) or
            result.get("formal_eligible") is not True or result.get("test_evaluated") is not False or
            result.get("prototype") != binding["prototype"] or result.get("seed") != binding["seed"] or
            result.get("best_epoch") != best["epoch"] or saved["epoch"] != best["epoch"] or
            saved.get("history") != history[:saved["epoch"]]):
        raise ValueError("selected checkpoint/result/history identity mismatch")
    for name in ("mAP50_95","AP50","AP75","Precision","Recall"):
        if result.get(name) != best["valid"].get(name):
            raise ValueError("result metrics differ from selected best epoch")
    latest = torch.load(folder/"latest.pt",map_location="cpu",weights_only=False)
    if (latest.get("binding") != binding or latest.get("epoch") != 80 or
            latest.get("architecture") != saved["architecture"] or latest.get("history") != history):
        raise ValueError("latest checkpoint does not close the complete training history")
    counters = [tensor for name,tensor in saved["model_state_dict"].items()
                if name.endswith("num_batches_tracked")]
    if not counters or any(int(value.item()) != saved["epoch"]*83 for value in counters):
        raise ValueError("checkpoint BatchNorm counters do not match actual training steps")
    return dict(formal_result_complete=True,execution_manifest_sha256=sha(manifest_path),
        training_result_sha256=sha(folder/"result.json"),
        training_status_sha256=sha(folder/"status.json"),
        training_history_sha256=sha(folder/"epoch_log.json"))
