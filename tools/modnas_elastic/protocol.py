"""Source/data binding for the isolated, non-formal MODNAS integration smoke."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/modnas_elastic_adapter_v1.yaml"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path=CONFIG):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["space"] != dict(name="elastic_neck72_v1",
        backbone="complete_fixed_B1_all_weights_trainable", levels=["P2", "P3", "P4", "P5"],
        neck_channels=48, fusion_expansion=3, fusion_depth=1, fusion_kernel_choices=[3, 5],
        whole_pan_choices=[False, True], top_down_blocks=["P2", "P3", "P4"],
        pan_blocks_when_enabled=["P3", "P4", "P5"], head="direct_1x1_linear_3_anchors",
        unique_candidates=72, inactive_pan_kernel_variables="omitted"):
        raise ValueError("New-space contract changed; register a new adapter version")
    if cfg["dataset"]["weight_split"] != "train" or cfg["dataset"]["architecture_split"] != "valid":
        raise ValueError("Only train weights / valid architecture updates are permitted")
    if cfg["dataset"]["input"] != [1, 640, 640] or cfg["dataset"]["num_classes"] != 2:
        raise ValueError("First adapter run is bound to Dataset A, grayscale640, two classes")
    if cfg["hardware"]["precision"] != "W8A8" or cfg["hardware"]["mapping"] != "reference_min_latency":
        raise ValueError("Quantization/mapping changes require a distinct cost contract")
    if cfg["smoke"]["formal_eligible"] is not False or cfg["smoke"]["test_evaluated"] is not False:
        raise ValueError("An integration smoke cannot claim formal performance")
    baseline = yaml.safe_load((ROOT / cfg["baseline"]).read_text(encoding="utf-8"))
    return cfg, baseline


def source_binding():
    from .author import DEFAULT_SOURCE_MANIFEST, SOURCE_ROOT, verify_author_sources
    author = verify_author_sources()
    paths = {CONFIG, ROOT / "configs/detector_elastic_pilot_model.yaml",
             ROOT / "configs/baseline_sonar_frozen.yaml",
             ROOT / "reports/baseline_sonar_frozen/freeze_manifest.json",
             ROOT / "configs/search_space_sonar_v0_1.yaml",
             ROOT / "scripts/detection_controlled_study.py",
             ROOT / "tools/sonar_analysis/baseline.py",
             ROOT / "tools/sonar_analysis/analyze_dataset.py",
             ROOT / "tools/detector_elastic_pilot/model.py"}
    paths.add(DEFAULT_SOURCE_MANIFEST)
    paths.update(SOURCE_ROOT / path for path in author["sha256"])
    paths.update((ROOT / "tools/modnas_elastic").glob("*.py"))
    paths.update((ROOT / "src/hwnas_fpga").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def dataset_binding(baseline):
    """Hash train/valid only and compare their identities with the existing freeze."""
    root = ROOT / baseline["dataset"]["data_dir"]
    frozen = json.loads((ROOT / baseline["baseline_freeze"]["manifest"]).read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "valid"):
        images = sorted(p for p in (root / split / "images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        if not images:
            raise ValueError(f"Empty {split} split")
        entries = []
        for image in images:
            label = root / split / "labels" / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing annotation: {label}")
            entries.append(dict(image=image.relative_to(root).as_posix(), image_sha256=sha(image),
                annotation=label.relative_to(root).as_posix(), annotation_sha256=sha(label)))
        normalize = lambda rows: sorted((r["image"].replace("\\", "/"), r["image_sha256"],
            r["annotation"].replace("\\", "/"), r["annotation_sha256"]) for r in rows)
        if normalize(entries) != normalize(frozen["datasets"][split]["entries"]):
            raise ValueError(f"Dataset identity differs from the frozen {split} manifest")
        result[split] = dict(images=len(entries), digest=canonical_sha(entries), entries=entries)
    return result


def runtime(device):
    value = dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                 device=device, threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available())
    if device == "cuda":
        value.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                     gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    return value


def git_identity():
    def command(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8",
                                       errors="replace").strip()
    return dict(commit=command("rev-parse", "HEAD"), status_porcelain=command("status", "--short"))


def prepare(output, device="cpu"):
    cfg, baseline = load_config()
    payload = dict(namespace=cfg["namespace"], created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=deepcopy(cfg), config_sha256=sha(CONFIG), source_sha256=source_binding(),
        dataset=dataset_binding(baseline), environment=runtime(device), git=git_identity(),
        test_accessed=False, test_evaluated=False, formal_eligible=False,
        evidence_scope="local_source_binding_and_train_valid_only_integration")
    target = Path(output) / "execution_manifest.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        for key in ("config_sha256", "source_sha256", "dataset", "environment"):
            if previous[key] != payload[key]:
                raise ValueError("Existing execution manifest drift; use a new result directory")
        return previous, baseline
    dump(target, payload)
    return payload, baseline
