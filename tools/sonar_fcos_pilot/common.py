"""Small reproducibility helpers, independent of the historical NAS project."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "sonar_fcos_pilot_v1.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    config["config_path"] = str(Path(path).resolve())
    config["config_sha256"] = sha256_file(path)
    if config["epochs"] != 30 or config["image_size"] != 640:
        raise ValueError("The approved formal pilot uses 30 epochs and 640-pixel images.")
    if config["combinations"] != ["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"]:
        raise ValueError("The approved search space has exactly eight combinations.")
    if config["seeds"] != [42, 43, 44] or config["max_formal_runs"] != 14:
        raise ValueError("The approved budget is at most 14 runs with seeds 42, 43, 44.")
    if config["batch_size"] * config["gradient_accumulation"] != 4:
        raise ValueError("All candidates must use the fixed effective batch size of four.")
    if config["diagnostic_score_threshold"] != .25 or config["minimum_map_gain"] != 1.0:
        raise ValueError("Diagnostic threshold and one-percentage-point decision gate are frozen.")
    return config


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import torch
    packages = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "pycocotools", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def source_binding(config):
    files = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "config_sha256": config["config_sha256"],
        "sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
    }


def require_manifest(config):
    path = (resolve_path(config["manifest_path"]) if config.get("manifest_path") else
            resolve_path(config["output_dir"]) / "data_audit" / "manifest.json")
    manifest = read_json(path)
    if manifest.get("split_status") != "PASS" or not manifest.get("audit", {}).get("data_integrity_pass"):
        raise RuntimeError(
            f"正式训练未启动：数据分组门槛未通过 ({manifest.get('split_status')})。"
            "需要可核验的图片到采集序列对应记录，不能用随机图片划分替代。"
        )
    if manifest.get("small_area_threshold") is None:
        raise RuntimeError("Training-only small-target threshold has not been frozen.")
    if manifest.get("protocol_id") == "PUBLIC_SPLIT_EXPLORATORY_V1":
        approval = config.get("approved_protocol", {})
        if approval.get("status") != "APPROVED" or approval.get("protocol_id") != manifest["protocol_id"]:
            raise RuntimeError("探索性数据划分尚未确认，不能替代原计划的真实采集序列分组。")
        if approval.get("manifest_sha256") != sha256_file(path):
            raise RuntimeError("The explicitly approved exploratory manifest changed.")
        acknowledgement = resolve_path(approval["acknowledgment_path"])
        if sha256_file(acknowledgement) != approval["acknowledgment_sha256"]:
            raise RuntimeError("Data-protocol acknowledgment changed.")
    return path, manifest
