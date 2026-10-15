"""Freeze the bounded pilot without changing the original trainer or datasets."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/sonar_background_pilot_20261010"


def main():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    snapshot = CAMPAIGN / "snapshot"
    if snapshot.exists():
        raise FileExistsError("Snapshot already frozen; use a new campaign for edits")
    origin = ROOT / "results/detection_fixed_normalizer_20260912/configs/baseline.yaml"
    config = yaml.safe_load(origin.read_text(encoding="utf-8"))
    config["project"]["output_dir"] = "results/sonar_background_pilot_20261010"
    config["training"]["epochs"] = 30
    config["training"]["amp"] = False
    (CAMPAIGN / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    paths = set((ROOT / "src/hwnas_fpga").rglob("*.py"))
    paths.update((ROOT / "tools/sonar_background_pilot").glob("*.py"))
    paths.update(ROOT / p for p in (
        "scripts/detection_controlled_study.py", "scripts/diagnose_detection_errors.py",
        "tests/test_detection_background_pilot.py"))
    hashes = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        name = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Snapshot copy mismatch: {name}")
        hashes[name] = digest
    (CAMPAIGN / "source_snapshot_manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    plan = {
        "arms": ["baseline", "rank", "pisa", "fg_bg"], "seeds": [42, 43, 44],
        "epochs_per_arm_seed": 30, "training_jobs": 12, "maximum_gpu_parallelism": 4,
        "source_baseline_config": str(origin.relative_to(ROOT)),
        "source_baseline_config_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "initialization": "fresh; identical state hash within seed; no historical checkpoints",
        "splits": {"train": 330, "valid": 99, "test": "never loaded"},
        "input_size": 640, "batch_size": 4, "precision": "float32",
        "selection": "best validation mean(AP50,AP75), existing project all-points AP",
        "primary_comparison": "seed-paired mean(AP50,AP75)",
        "reporting": ["COCO101 AP50:95 on cached outputs", "F1-selected validation P/R", "per-class recall",
                      "background FP count at fixed 0.1/0.3/0.5 thresholds", "recall at 0.1/0.5/1 FP per image"],
        "promotion_gate": {"mean_paired_map_delta": ">0.005", "positive_seeds": ">=2/3",
                           "mean_paired_f1_delta": ">=0", "mean_manta_recall_delta_at_fppi_0_5": ">=-0.02",
                           "mean_all_recall_delta_at_fppi_0_5": ">=0"},
        "gate_note": "Predeclared screening gate, not statistical significance or deployment acceptance",
        "rank": "0.1*(softplus log-score margin0.5, top16 reliable positives/top32 exterior background per image-class-scale +0.25 same-GT quality order)",
        "pisa": "strength0.5; per-GT IoU-rank positive weighting and top representative per4x4 spatial tile; detached mean-normalized sample weights",
        "fg_bg": "0.1*box-center/class foreground vs exterior per-scale hard32 background cosine prototype loss; temperature0.2, cosine margin0.2",
        "paper_status": "Three inspired adaptations; neither exact RS/PISA nor DMT reproduction",
        "fixed": ["model architecture and parameter count", "assignment", "data labels", "augmentations", "sampler", "optimizer", "decoder/NMS"],
        "excluded": ["classification changes", "combined arm", "hyperparameter search", "test selection", "FPGA/HLS/board/power experiments"]}
    (CAMPAIGN / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = CAMPAIGN / "source_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(snapshot, arcname="snapshot")
        for name in ("config.yaml", "source_snapshot_manifest.json", "PLAN.json"):
            stream.add(CAMPAIGN / name, arcname=name)
    print(json.dumps({"status": "FROZEN", "source_files": len(hashes), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
