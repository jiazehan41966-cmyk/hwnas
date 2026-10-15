"""Prepare classic SCTD 1.0 for the local YOLO detection loader.

The source SCTD package is Pascal VOC style. This script keeps
``datasets/SCTD/raw`` immutable and creates a loader-compatible YOLO view under
``datasets/SCTD/processed`` with provenance metadata under
``datasets/SCTD/metadata``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PIL/Pillow is required to inspect SCTD images") from exc


CLASSES = ["aircraft", "human", "ship"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
SOURCE_ID = "classic_sctd_1_0_mingqiangning_github"
RAW_URL = "https://raw.githubusercontent.com/MingqiangNing/SCTD/master/SCTD.zip"
REPO_URL = "https://github.com/MingqiangNing/SCTD"
SCTD3_REPO_URL = "https://github.com/automlresearch/SCTD-3.0"
SCTD3_ARXIV_URL = "https://arxiv.org/abs/2608.08106"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_voc_annotation(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    filename = (root.findtext("filename") or f"{path.stem}.jpg").strip()
    size_node = root.find("size")
    width = int(float(size_node.findtext("width"))) if size_node is not None else None
    height = int(float(size_node.findtext("height"))) if size_node is not None else None
    depth = int(float(size_node.findtext("depth"))) if size_node is not None else None
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        objects.append(
            {
                "name": name,
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "difficult": int(float(obj.findtext("difficult") or 0)),
                "truncated": int(float(obj.findtext("truncated") or 0)),
            }
        )
    return {
        "id": path.stem,
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects,
    }


def yolo_rows(record: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    width = float(record["width"])
    height = float(record["height"])
    for obj in record["objects"]:
        cls = CLASS_TO_ID[obj["name"]]
        xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        rows.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return rows


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def numeric_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    keyed = sorted(
        ((int(record["id"]), record) for record in records if str(record["id"]).isdigit()),
        key=lambda item: item[0],
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: int | None = None
    for number, record in keyed:
        if previous is None or number == previous + 1:
            current.append(record)
        else:
            groups.append(current)
            current = [record]
        previous = number
    if current:
        groups.append(current)
    return groups


def make_group_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create a deterministic split while keeping consecutive numeric runs together."""

    groups = numeric_groups(records)
    buckets = {"train": [], "valid": [], "test": []}
    targets = {"train": 0.70, "valid": 0.15, "test": 0.15}
    total = len(records)
    desired = {name: int(round(total * ratio)) for name, ratio in targets.items()}
    desired["train"] = total - desired["valid"] - desired["test"]
    # Largest groups first reduces overshoot while keeping possible collection runs intact.
    for group in sorted(groups, key=lambda items: (-len(items), items[0]["id"])):
        split = min(
            buckets,
            key=lambda name: (len(buckets[name]) / max(1, desired[name]), len(buckets[name])),
        )
        buckets[split].extend(group)
    return {name: sorted(items, key=lambda item: item["id"]) for name, items in buckets.items()}


def inspect_source(source_root: Path) -> dict[str, Any]:
    image_root = source_root / "JPEGImages"
    ann_root = source_root / "Annotations"
    records = [load_voc_annotation(path) for path in sorted(ann_root.glob("*.xml"))]
    image_paths = sorted(image_root.glob("*.jpg"))
    image_by_name = {path.name: path for path in image_paths}

    class_objects = Counter()
    class_images = Counter()
    sizes = Counter()
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    corrupt: list[dict[str, str]] = []
    bad_boxes: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_annotations = sorted({path.stem for path in image_paths} - {r["id"] for r in records})

    for record in records:
        if record["filename"] not in image_by_name:
            missing_images.append(record["filename"])
        names = set()
        for obj in record["objects"]:
            name = obj["name"]
            class_objects[name] += 1
            names.add(name)
            xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
            if name not in CLASS_TO_ID or xmin >= xmax or ymin >= ymax:
                bad_boxes.append({"id": record["id"], "name": name, "bbox_xyxy": obj["bbox_xyxy"]})
        for name in names:
            class_images[name] += 1

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                sizes[f"{img.width}x{img.height}"] += 1
        except Exception as exc:  # pragma: no cover
            corrupt.append({"file": path.name, "error": str(exc)})
        duplicate_hashes[sha256_file(path)].append(path.name)

    duplicates = {h: names for h, names in duplicate_hashes.items() if len(names) > 1}
    groups = numeric_groups(records)
    official_split_files = {}
    for path in sorted((source_root / "ImageSets").rglob("*.txt")):
        values = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        official_split_files[path.relative_to(source_root).as_posix()] = {
            "count": len(values),
            "first_ids": values[:5],
            "last_ids": values[-5:],
        }

    return {
        "record_count": len(records),
        "image_count": len(image_paths),
        "object_count": sum(class_objects.values()),
        "class_objects": dict(class_objects),
        "class_images": dict(class_images),
        "sizes": dict(sizes),
        "corrupt_images": corrupt,
        "bad_boxes": bad_boxes,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "duplicate_groups": duplicates,
        "numeric_run_count": len(groups),
        "numeric_runs_ge3": [
            [items[0]["id"], items[-1]["id"], len(items)]
            for items in groups
            if len(items) >= 3
        ],
        "official_split_files": official_split_files,
        "records": records,
    }


def prepare_processed(source_root: Path, processed_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = make_group_split(records)
    image_root = source_root / "JPEGImages"
    operations = Counter()
    split_counts: dict[str, int] = {}
    split_class_objects: dict[str, dict[str, int]] = {}
    for split, split_records in splits.items():
        split_counts[split] = len(split_records)
        split_class_counter = Counter()
        for record in split_records:
            source_image = image_root / record["filename"]
            target_image = processed_root / split / "images" / record["filename"]
            operations[link_or_copy(source_image, target_image)] += 1
            label_path = processed_root / split / "labels" / f"{record['id']}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(yolo_rows(record)) + "\n", encoding="utf-8")
            for obj in record["objects"]:
                split_class_counter[obj["name"]] += 1
        split_class_objects[split] = dict(split_class_counter)

    data_yaml = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names:",
    ]
    for index, name in enumerate(CLASSES):
        data_yaml.append(f"  {index}: {name}")
    (processed_root / "data.yaml").write_text("\n".join(data_yaml) + "\n", encoding="utf-8")
    return {
        "split_policy": "project_derived_numeric_run_group_split_v1",
        "split_counts": split_counts,
        "split_class_objects": split_class_objects,
        "image_materialization": dict(operations),
    }


def write_metadata(
    metadata_root: Path,
    raw_zip: Path,
    source_root: Path,
    processed_root: Path,
    inspection: dict[str, Any],
    processed: dict[str, Any],
) -> None:
    metadata_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    raw_sha = sha256_file(raw_zip)
    class_mapping = {
        "source_id": SOURCE_ID,
        "annotation_source": "Pascal VOC XML class names",
        "mapping_policy": "alphabetical stable order over observed classes",
        "classes": [{"id": index, "name": name} for index, name in enumerate(CLASSES)],
    }
    (metadata_root / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at": now,
        "dataset_id": SOURCE_ID,
        "version": "1.0",
        "dataset_family": "SCTD",
        "dataset_variant": "classic SCTD",
        "not_sctd_3_0": True,
        "source": {
            "repository": REPO_URL,
            "raw_zip_url": RAW_URL,
            "raw_zip_path": str(raw_zip),
            "raw_zip_sha256": raw_sha,
            "download_method": "Start-BitsTransfer from GitHub raw URL after curl timeout",
        },
        "raw": {
            "source_root": str(source_root),
            "annotation_format": "Pascal VOC XML",
            "image_count": inspection["image_count"],
            "annotation_count": inspection["record_count"],
            "object_count": inspection["object_count"],
            "classes": inspection["class_objects"],
            "image_resolutions": inspection["sizes"],
            "official_split_files": inspection["official_split_files"],
        },
        "processed": {
            "root": str(processed_root),
            "annotation_format": "YOLO normalized cx cy w h",
            **processed,
        },
        "integrity": {
            "corrupt_images": inspection["corrupt_images"],
            "bad_boxes": inspection["bad_boxes"],
            "missing_images": inspection["missing_images"],
            "missing_annotations": inspection["missing_annotations"],
            "duplicate_group_count": len(inspection["duplicate_groups"]),
            "duplicate_groups": inspection["duplicate_groups"],
            "numeric_runs_ge3": inspection["numeric_runs_ge3"],
        },
    }
    (metadata_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (metadata_root / "download_source.txt").write_text(
        "\n".join(
            [
                "dataset: classic SCTD / SCTD 1.0",
                f"source_repository: {REPO_URL}",
                f"download_url: {RAW_URL}",
                f"downloaded_archive: {raw_zip}",
                f"sha256: {raw_sha}",
                "download_status: success",
                "failed_attempts: GitHub codeload master zip was truncated; curl raw download timed out/reset; final BITS raw download succeeded.",
                f"sctd_3_0_reference_repo_not_used: {SCTD3_REPO_URL}",
                f"sctd_3_0_reference_arxiv_not_used: {SCTD3_ARXIV_URL}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    card = f"""# Classic SCTD Dataset Card

## Identity

- Dataset: classic SCTD / SCTD 1.0
- Full name used by source repository: Sonar Common Target Detection Dataset (SCTD 1.0)
- Source repository: {REPO_URL}
- Source archive: {RAW_URL}
- SHA256: `{raw_sha}`
- Original paper: Self-Trained Target Detection of Radar and Sonar Images Using Automatic Deep Learning
- Imaging mode: side-scan sonar according to recent paper descriptions of SCTD; not SAS-only SCTD 3.0
- Version: 1.0
- Release evidence: source repository history and README identify this package as SCTD 1.0; repository files are dated 2020-09-11 after extraction.

## Scope

This is not SCTD 3.0. SCTD 3.0 is treated only as a version contrast item and was not downloaded or used.

## Raw Data

- Images: {inspection['image_count']}
- Annotations: {inspection['record_count']}
- Objects: {inspection['object_count']}
- Classes: {inspection['class_objects']}
- Image resolutions: {inspection['sizes']}
- Raw annotation format: Pascal VOC XML with pixel-space `xmin ymin xmax ymax`.
- Official split files found: {inspection['official_split_files']}

## Processed View

- Root: `{processed_root}`
- Annotation format: YOLO normalized `class cx cy w h`.
- Class mapping: `{metadata_root / 'class_mapping.json'}`
- Split policy: {processed['split_policy']}; this is project-derived, not an official SCTD split.
- Split counts: {processed['split_counts']}
- Image materialization: {processed['image_materialization']}

## Leakage Risk

The source package has numeric filenames with many consecutive runs: {len(inspection['numeric_runs_ge3'])} runs of length >= 3 were observed. Acquisition metadata is absent, so consecutive-frame leakage cannot be certified away. The processed split keeps consecutive numeric runs together to reduce obvious leakage risk, but this is not a substitute for mission-level metadata.
"""
    (metadata_root / "dataset_card.md").write_text(card, encoding="utf-8")

    investigation = f"""# SCTD Source Investigation

## Conclusion

The downloaded dataset is classic SCTD / SCTD 1.0 from the original author GitHub repository, not SCTD 3.0.

## Original Paper And Recent-Paper Linkage

- Original dataset paper: Self-Trained Target Detection of Radar and Sonar Images Using Automatic Deep Learning.
- Recent references checked by task: ESL-YOLO (JMSE 13(8):1477, 2025) and MFF-YOLOv7 (JMSE 12(12):2326, 2024).
- The recent-paper SCTD identity matches the classic description: 357 images, three target classes, Pascal/VOC-style detection annotations in the original package.

## Original Download Link

- Repository: {REPO_URL}
- Raw archive: {RAW_URL}
- Archive SHA256: `{raw_sha}`
- Current status: official GitHub raw download succeeded via Windows BITS.

## Mirrors

- No third-party mirror was used.
- GitHub codeload master zip attempt failed and was discarded after ZIP truncation evidence.
- SCTD 3.0 official entry points were checked only for version contrast:
  - {SCTD3_REPO_URL}
  - {SCTD3_ARXIV_URL}

## Classic SCTD Versus SCTD 3.0

Classic SCTD 1.0 and SCTD 3.0 are not treated as the same data version. Classic SCTD 1.0 has 357 images and three classes in Pascal VOC XML. SCTD 3.0 is a separate 2026 dataset/release line and is not substituted here.

## Recommendation

Use this official GitHub classic SCTD 1.0 package for the planned classic-SCTD comparison. Do not replace it with SCTD 3.0 unless the experiment is explicitly re-scoped.
"""
    (metadata_root / "source_investigation.md").write_text(investigation, encoding="utf-8")

    report = f"""# SCTD Integrity Report

- Raw ZIP SHA256: `{raw_sha}`
- Images: {inspection['image_count']}
- XML annotations: {inspection['record_count']}
- Objects: {inspection['object_count']}
- Class object counts: {inspection['class_objects']}
- Class image counts: {inspection['class_images']}
- Image resolutions: {inspection['sizes']}
- Corrupted images: {len(inspection['corrupt_images'])}
- Bad boxes: {len(inspection['bad_boxes'])}
- Missing images for annotations: {len(inspection['missing_images'])}
- Missing annotations for images: {len(inspection['missing_annotations'])}
- Duplicate image hash groups: {len(inspection['duplicate_groups'])}
- Official split files found: {inspection['official_split_files']}
- Processed split: {processed['split_counts']}
- Leakage note: {len(inspection['numeric_runs_ge3'])} consecutive numeric runs of length >= 3; mission/frame metadata absent.
"""
    (metadata_root / "integrity_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/SCTD"))
    parser.add_argument("--raw-zip", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.root
    raw_zip = args.raw_zip or dataset_root / "raw" / "SCTD.bits.zip"
    source_root = dataset_root / "raw" / "extracted" / "SCTD"
    processed_root = dataset_root / "processed"
    metadata_root = dataset_root / "metadata"
    if not raw_zip.is_file():
        raise FileNotFoundError(raw_zip)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    inspection = inspect_source(source_root)
    observed_classes = sorted(inspection["class_objects"])
    if observed_classes != CLASSES:
        raise ValueError(f"Unexpected classes: {observed_classes} != {CLASSES}")
    if inspection["image_count"] != inspection["record_count"]:
        raise ValueError("Image/annotation count mismatch")
    if inspection["corrupt_images"] or inspection["bad_boxes"] or inspection["missing_images"]:
        raise ValueError("Integrity check failed; see inspection output")

    processed = prepare_processed(source_root, processed_root, inspection["records"])
    write_metadata(metadata_root, raw_zip, source_root, processed_root, inspection, processed)
    print(json.dumps({
        "dataset_id": SOURCE_ID,
        "raw_zip_sha256": sha256_file(raw_zip),
        "image_count": inspection["image_count"],
        "classes": inspection["class_objects"],
        "processed_split": processed["split_counts"],
        "metadata_root": str(metadata_root),
        "processed_root": str(processed_root),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
