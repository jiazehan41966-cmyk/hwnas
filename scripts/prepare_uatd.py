#!/usr/bin/env python3
"""Acquire, inspect, and optionally convert the official UATD dataset.

The script keeps raw downloads/extractions under datasets/UATD/raw and writes
project-facing metadata under datasets/UATD/metadata. It does not create random
train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import urllib.request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "UATD"

ARTICLE_API_URL = "https://api.figshare.com/v2/articles/21331143"
ARTICLE_HTML_URL = "https://figshare.com/articles/dataset/UATD_Dataset/21331143"
ARTICLE_DOI = "10.6084/m9.figshare.21331143.v3"
PAPER_URL = "https://www.nature.com/articles/s41597-022-01854-w"
PAPER_DOI = "https://doi.org/10.1038/s41597-022-01854-w"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".xml", ".txt", ".json", ".csv"}
SPLIT_PARTS = {
    "UATD_Training": "train",
    "UATD_Test_1": "test1",
    "UATD_Test_2": "test2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; hwnas-uatd-prep/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - public dataset integrity, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_read_only_tree(path: Path) -> None:
    if path.is_file():
        try:
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
        except OSError:
            pass
        return
    for child in path.rglob("*"):
        if child.is_file():
            try:
                child.chmod(child.stat().st_mode & ~stat.S_IWRITE)
            except OSError:
                pass


def ensure_writable_file(path: Path) -> None:
    if path.exists():
        try:
            path.chmod(path.stat().st_mode | stat.S_IWRITE)
        except OSError:
            pass


def download_file(url: str, destination: Path, expected_size: int, expected_md5: str) -> None:
    if destination.is_file():
        observed_size = destination.stat().st_size
        if observed_size == expected_size and md5_file(destination).lower() == expected_md5.lower():
            print(f"[download] verified existing {destination.name}")
            return
        raise RuntimeError(
            f"Existing file failed integrity check: {destination} "
            f"(size={observed_size}, expected={expected_size})"
        )

    tmp = destination.with_suffix(destination.suffix + ".part")
    ensure_writable_file(tmp)
    print(f"[download] {destination.name} ({expected_size} bytes)", flush=True)
    max_attempts = 24
    for attempt in range(1, max_attempts + 1):
        copied = tmp.stat().st_size if tmp.exists() else 0
        if copied > expected_size:
            tmp.unlink()
            copied = 0
        if copied == expected_size:
            break
        headers = {"User-Agent": "Mozilla/5.0 (compatible; hwnas-uatd-prep/1.0)"}
        if copied > 0:
            headers["Range"] = f"bytes={copied}-"
        request = urllib.request.Request(url, headers=headers)
        mode = "ab" if copied > 0 else "wb"
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if copied > 0 and getattr(response, "status", None) != 206:
                    print(
                        f"  {destination.name}: server ignored Range; restarting from byte 0",
                        flush=True,
                    )
                    tmp.unlink(missing_ok=True)
                    copied = 0
                    mode = "wb"
                next_report = max(256 * 1024 * 1024, ((copied // (256 * 1024 * 1024)) + 1) * 256 * 1024 * 1024)
                with tmp.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        copied += len(chunk)
                        if copied >= next_report or copied == expected_size:
                            print(
                                f"  {destination.name}: {copied}/{expected_size} bytes",
                                flush=True,
                            )
                            next_report += 256 * 1024 * 1024
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(
                f"  {destination.name}: attempt {attempt}/{max_attempts} interrupted at "
                f"{tmp.stat().st_size if tmp.exists() else 0} bytes: {exc!r}",
                flush=True,
            )
            continue
        if tmp.exists() and tmp.stat().st_size == expected_size:
            break

    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {destination.name}")
    observed_md5 = md5_file(tmp)
    if observed_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"Downloaded md5 mismatch for {destination.name}: {observed_md5} != {expected_md5}"
        )
    tmp.replace(destination)
    destination.chmod(destination.stat().st_mode & ~stat.S_IWRITE)


def safe_extract_zip(archive: Path, destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    marker = destination / ".extract_complete.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8")).get("files", [])

    print(f"[extract] {archive.name} -> {destination}")
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(part == ".." for part in pure.parts):
                raise RuntimeError(f"Unsafe ZIP member in {archive}: {info.filename}")
            target = destination.joinpath(*pure.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(
                {
                    "path": str(target.relative_to(destination).as_posix()),
                    "bytes": int(info.file_size),
                    "crc": f"{info.CRC:08x}",
                }
            )
    marker.write_text(json.dumps({"archive": str(archive), "files": extracted}, indent=2), encoding="utf-8")
    set_read_only_tree(destination)
    return extracted


def split_for_path(path: Path) -> str | None:
    parts = set(path.parts)
    hits = [split for marker, split in SPLIT_PARTS.items() if marker in parts]
    if hits:
        return hits[0]
    text = path.as_posix().lower()
    if "training" in text or "/train" in text:
        return "train"
    if "test_1" in text:
        return "test1"
    if "test_2" in text:
        return "test2"
    if "/test" in text:
        return "test"
    return None


def source_group_for_path(path: Path) -> str | None:
    parts = set(path.parts)
    for marker in SPLIT_PARTS:
        if marker in parts:
            return marker
    split = split_for_path(path)
    return split


def inspect_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image inspection") from exc
    with Image.open(path) as image:
        image.load()
        return {
            "width": int(image.width),
            "height": int(image.height),
            "mode": image.mode,
            "channels": len(image.getbands()),
            "format": image.format,
        }


def parse_voc_xml(path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    boxes: list[dict[str, Any]] = []
    filename = root.findtext("filename")
    image_stem = Path(filename).stem if filename else path.stem
    size = root.find("size")
    image_size = None
    if size is not None:
        width = size.findtext("width")
        height = size.findtext("height")
        if width and height:
            image_size = {"width": int(float(width)), "height": int(float(height))}
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.findtext("xmin", "nan"))
            y1 = float(box.findtext("ymin", "nan"))
            x2 = float(box.findtext("xmax", "nan"))
            y2 = float(box.findtext("ymax", "nan"))
        except ValueError:
            continue
        boxes.append(
            {
                "annotation_path": path,
                "annotation_format": "pascal_voc_xml",
                "image_stem": image_stem,
                "class_name": name,
                "bbox": [x1, y1, x2, y2],
                "bbox_format": "xyxy_pixel",
                "image_size": image_size,
            }
        )
    return boxes


def parse_yolo_txt(path: Path) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            values = [float(value) for value in parts[:5]]
        except ValueError:
            continue
        label, cx, cy, width, height = values
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            continue
        boxes.append(
            {
                "annotation_path": path,
                "annotation_format": "yolo_txt",
                "image_stem": path.stem,
                "class_name": str(int(label)),
                "source_class_id": int(label),
                "bbox": [cx, cy, width, height],
                "bbox_format": "cxcywh_normalized",
                "image_size": None,
            }
        )
    return boxes


def parse_coco_json(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        return []
    images = {item.get("id"): item for item in payload.get("images", []) if isinstance(item, dict)}
    categories = {
        item.get("id"): str(item.get("name"))
        for item in payload.get("categories", [])
        if isinstance(item, dict)
    }
    boxes: list[dict[str, Any]] = []
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        image = images.get(ann.get("image_id"), {})
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        image_stem = Path(str(image.get("file_name", path.stem))).stem
        category_id = ann.get("category_id")
        boxes.append(
            {
                "annotation_path": path,
                "annotation_format": "coco_json",
                "image_stem": image_stem,
                "class_name": categories.get(category_id, str(category_id)),
                "source_class_id": category_id,
                "bbox": [float(v) for v in bbox[:4]],
                "bbox_format": "xywh_pixel",
                "image_size": {
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                }
                if "width" in image and "height" in image
                else None,
            }
        )
    return boxes


def parse_annotations(annotation_paths: list[Path]) -> tuple[list[dict[str, Any]], Counter[str]]:
    boxes: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    for path in annotation_paths:
        suffix = path.suffix.lower()
        parsed: list[dict[str, Any]] = []
        if suffix == ".xml":
            parsed = parse_voc_xml(path)
        elif suffix == ".txt":
            parsed = parse_yolo_txt(path)
        elif suffix == ".json":
            parsed = parse_coco_json(path)
        if parsed:
            group = source_group_for_path(path)
            for row in parsed:
                row["source_group"] = group
            boxes.extend(parsed)
            format_counts.update(row["annotation_format"] for row in parsed)
    return boxes, format_counts


def summarize_counter(counter: Counter[Any], limit: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"value": str(key), "count": int(value)} for key, value in items]


def find_duplicate_filenames(paths: list[Path]) -> list[dict[str, Any]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_name[path.name].append(str(path))
    return [
        {"filename": name, "paths": values}
        for name, values in sorted(by_name.items())
        if len(values) > 1
    ]


def infer_sequence_risks(image_paths: list[Path]) -> dict[str, Any]:
    parent_counts = Counter(str(path.parent) for path in image_paths)
    sequence_tokens = Counter()
    numeric_stems = 0
    for path in image_paths:
        lower_parts = [part.lower() for part in path.parts]
        for part in lower_parts:
            if any(token in part for token in ("seq", "sequence", "frame", "video", "scene", "mission")):
                sequence_tokens[part] += 1
        stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
        if stem_digits and len(stem_digits) >= max(2, len(path.stem) // 2):
            numeric_stems += 1
    return {
        "sequence_like_path_tokens": summarize_counter(sequence_tokens, 30),
        "numeric_or_frame_like_stems": numeric_stems,
        "largest_parent_directories": summarize_counter(parent_counts, 20),
        "risk_note": (
            "Official train/test archives exist, but no validation split was detected. "
            "Do not create a random validation split until sequence/acquisition grouping "
            "is reviewed."
        ),
    }


def build_inventory(root: Path) -> dict[str, Any]:
    extracted = root / "raw" / "extracted"
    all_files = [path for path in extracted.rglob("*") if path.is_file()]
    all_image_paths = sorted(path for path in all_files if path.suffix.lower() in IMAGE_EXTENSIONS)
    all_annotation_paths = sorted(path for path in all_files if path.suffix.lower() in ANNOTATION_EXTENSIONS)
    dataset_groups = set(SPLIT_PARTS)
    image_paths = [
        path for path in all_image_paths
        if source_group_for_path(path) in dataset_groups
    ]
    annotation_paths = [
        path for path in all_annotation_paths
        if source_group_for_path(path) in dataset_groups
    ]

    image_ext_counts: Counter[str] = Counter(path.suffix.lower() for path in image_paths)
    annotation_ext_counts: Counter[str] = Counter(path.suffix.lower() for path in annotation_paths)
    split_counts: Counter[str] = Counter()
    resolution_counts: Counter[tuple[int, int]] = Counter()
    mode_counts: Counter[str] = Counter()
    channel_counts: Counter[int] = Counter()
    corrupt_images: list[dict[str, str]] = []
    image_records: list[dict[str, Any]] = []
    stem_to_images: dict[str, list[Path]] = defaultdict(list)

    for path in image_paths:
        split = split_for_path(path.relative_to(extracted)) or "unknown"
        split_counts[split] += 1
        stem_to_images[path.stem].append(path)
        try:
            info = inspect_image(path)
        except Exception as exc:
            corrupt_images.append({"path": str(path), "error": repr(exc)})
            continue
        resolution_counts[(info["width"], info["height"])] += 1
        mode_counts[info["mode"]] += 1
        channel_counts[info["channels"]] += 1
        image_records.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root).as_posix()),
                "split": split,
                "source_group": source_group_for_path(path),
                "image_stem": path.stem,
                **info,
            }
        )

    boxes, annotation_format_counts = parse_annotations(annotation_paths)
    class_counts: Counter[str] = Counter(str(row["class_name"]) for row in boxes)
    boxes_by_key: Counter[tuple[str | None, str]] = Counter(
        (row.get("source_group"), str(row["image_stem"])) for row in boxes
    )
    bbox_format_counts: Counter[str] = Counter(str(row["bbox_format"]) for row in boxes)

    empty_images = [
        record["relative_path"]
        for record in image_records
        if boxes_by_key.get((record.get("source_group"), record["image_stem"]), 0) == 0
    ]
    duplicate_names = find_duplicate_filenames(image_paths)
    official_split_info = {
        "provided": bool(split_counts),
        "detected_from": "Figshare archive names and extracted path segments",
        "splits": dict(sorted(split_counts.items())),
        "validation_split_detected": "valid" in split_counts or "val" in split_counts,
        "note": "UATD provides training and two test archives; no fixed validation split was detected.",
    }

    return {
        "generated_at": utc_now(),
        "root": str(root),
        "extracted_root": str(extracted),
        "file_count": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "all_image_count_including_auxiliary": len(all_image_paths),
        "auxiliary_image_count_excluded": len(all_image_paths) - len(image_paths),
        "image_count": len(image_paths),
        "image_extensions": dict(sorted(image_ext_counts.items())),
        "all_annotation_like_files_including_auxiliary": len(all_annotation_paths),
        "auxiliary_annotation_like_files_excluded": len(all_annotation_paths) - len(annotation_paths),
        "annotation_files": len(annotation_paths),
        "annotation_extensions": dict(sorted(annotation_ext_counts.items())),
        "annotation_formats": dict(sorted(annotation_format_counts.items())),
        "bbox_formats": dict(sorted(bbox_format_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "classes": sorted(class_counts),
        "class_count": len(class_counts),
        "box_count": len(boxes),
        "resolution_distribution": [
            {"width": key[0], "height": key[1], "count": value}
            for key, value in resolution_counts.most_common()
        ],
        "image_modes": dict(sorted(mode_counts.items())),
        "channel_counts": {str(key): int(value) for key, value in sorted(channel_counts.items())},
        "corrupted_images": corrupt_images,
        "empty_annotation_images": {
            "count": len(empty_images),
            "examples": empty_images[:100],
        },
        "duplicate_filenames": {
            "count": len(duplicate_names),
            "examples": duplicate_names[:50],
        },
        "official_split": official_split_info,
        "leakage_risks": infer_sequence_risks(image_paths),
        "image_records": image_records,
        "boxes": [
            {
                **row,
                "annotation_path": str(row["annotation_path"]),
            }
            for row in boxes
        ],
    }


def yolo_line_from_box(box: list[float], bbox_format: str, width: int, height: int) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    if bbox_format == "xyxy_pixel":
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        return cx / width, cy / height, bw / width, bh / height
    if bbox_format == "xywh_pixel":
        x, y, bw, bh = box
        return (x + bw / 2.0) / width, (y + bh / 2.0) / height, bw / width, bh / height
    if bbox_format == "cxcywh_normalized":
        return tuple(float(value) for value in box)  # type: ignore[return-value]
    return None


def copy_or_link_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def reset_generated_dir(path: Path, root: Path) -> None:
    def clear_readonly_and_retry(function: Any, target: str, _excinfo: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            function(target)
        except OSError:
            raise

    resolved_root = root.resolve()
    if path.exists():
        resolved = path.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise RuntimeError(f"Refusing to remove generated directory outside dataset root: {path}")
        shutil.rmtree(path, onexc=clear_readonly_and_retry)


def convert_to_yolo(root: Path, inventory: dict[str, Any], link_mode: str = "copy") -> dict[str, Any]:
    processed = root / "processed"
    yolo_root = root / "processed_yolo"
    reset_generated_dir(processed, root)
    reset_generated_dir(yolo_root, root)
    processed.mkdir(parents=True, exist_ok=True)
    yolo_root.mkdir(parents=True, exist_ok=True)

    image_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inventory["image_records"]:
        image_by_stem[Path(record["path"]).stem].append(record)
    boxes_by_key: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for box in inventory["boxes"]:
        boxes_by_key[(box.get("source_group"), str(box["image_stem"]))].append(box)

    classes = sorted(inventory["classes"])
    class_to_id = {name: index for index, name in enumerate(classes)}
    converted_images = 0
    converted_boxes = 0
    combined_test_images = 0
    combined_test_boxes = 0
    skipped_boxes = 0
    copy_modes: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    def write_yolo_sample(record: dict[str, Any], split_name: str, output_name: str) -> tuple[int, int]:
        source = Path(record["path"])
        image_destination = yolo_root / split_name / "images" / output_name
        copy_modes[copy_or_link_image(source, image_destination, link_mode)] += 1

        label_destination = yolo_root / split_name / "labels" / f"{Path(output_name).stem}.txt"
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        local_skipped = 0
        local_boxes = 0
        key = (record.get("source_group"), record["image_stem"])
        for box in boxes_by_key.get(key, []):
            values = yolo_line_from_box(
                [float(value) for value in box["bbox"]],
                str(box["bbox_format"]),
                int(record["width"]),
                int(record["height"]),
            )
            if values is None:
                local_skipped += 1
                continue
            cx, cy, bw, bh = values
            if bw <= 0.0 or bh <= 0.0:
                local_skipped += 1
                continue
            class_id = class_to_id[str(box["class_name"])]
            lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            local_boxes += 1
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return local_boxes, local_skipped

    for record in inventory["image_records"]:
        source = Path(record["path"])
        split = record["split"]
        if split == "unknown":
            split = "unsplit"
        boxes_written, boxes_skipped = write_yolo_sample(record, split, source.name)
        converted_images += 1
        converted_boxes += boxes_written
        skipped_boxes += boxes_skipped
        split_counts[split] += 1
        if split in {"test1", "test2"}:
            prefixed_name = f"{split}_{source.name}"
            combo_boxes, combo_skipped = write_yolo_sample(record, "test", prefixed_name)
            combined_test_images += 1
            combined_test_boxes += combo_boxes
            skipped_boxes += combo_skipped
            split_counts["test"] += 1

    data_yaml = yolo_root / "data.yaml"
    yaml_lines = [
        "train: ../train/images",
        "test: ../test/images",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for name, index in class_to_id.items())
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    mapping = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_annotation_formats": inventory["annotation_formats"],
        "source_classes_sorted": classes,
        "class_name_to_yolo_id": class_to_id,
        "yolo_id_to_class_name": {str(value): key for key, value in class_to_id.items()},
        "bbox_conversion": {
            "source_supported": ["pascal_voc_xml:xyxy_pixel", "coco_json:xywh_pixel", "yolo_txt:cxcywh_normalized"],
            "target": "YOLO normalized class cx cy width height",
            "coordinate_note": "Pixel boxes are interpreted in the image coordinate frame recorded by the file.",
        },
    }
    summary = {
        "processed_yolo_root": str(yolo_root),
        "converted_images": converted_images,
        "converted_boxes": converted_boxes,
        "combined_test_images": combined_test_images,
        "combined_test_boxes": combined_test_boxes,
        "skipped_boxes": skipped_boxes,
        "split_counts": dict(sorted(split_counts.items())),
        "image_materialization": dict(sorted(copy_modes.items())),
        "current_loader_compatibility": {
            "YoloDetectionDataset": True,
            "create_yolo_detection_dataloaders_train_valid_test": False,
            "reason": "No official validation split was detected; the project factory expects train/valid/test.",
        },
    }
    (processed / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"class_mapping": mapping, "conversion_summary": summary}


def write_metadata(root: Path, article: dict[str, Any], downloads: list[dict[str, Any]], inventory: dict[str, Any], conversion: dict[str, Any] | None) -> None:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    source_text = [
        "Official UATD download source",
        f"Paper: {PAPER_URL}",
        f"Paper DOI: {PAPER_DOI}",
        f"Figshare article: {ARTICLE_HTML_URL}",
        f"Figshare API: {ARTICLE_API_URL}",
        f"Figshare DOI: https://doi.org/{ARTICLE_DOI}",
        f"Retrieved at: {utc_now()}",
        "",
        "Files:",
    ]
    for item in downloads:
        source_text.append(
            f"- {item['name']} | {item['download_url']} | bytes={item['bytes']} | "
            f"md5={item['md5']} | sha256={item['sha256']}"
        )
    (metadata / "download_source.txt").write_text("\n".join(source_text) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "official_source": {
            "official_name": article.get("title"),
            "paper_url": PAPER_URL,
            "paper_doi": PAPER_DOI,
            "figshare_url": ARTICLE_HTML_URL,
            "figshare_api_url": ARTICLE_API_URL,
            "figshare_doi": ARTICLE_DOI,
            "license": article.get("license"),
            "version": article.get("version"),
            "citation": article.get("citation"),
            "description": article.get("description"),
        },
        "files": downloads,
        "inventory": {
            key: value
            for key, value in inventory.items()
            if key not in {"image_records", "boxes"}
        },
        "conversion": conversion["conversion_summary"] if conversion else None,
    }
    (metadata / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    class_mapping = conversion["class_mapping"] if conversion else {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_classes": inventory["classes"],
        "note": "No conversion was generated.",
    }
    (metadata / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    conversion_summary = conversion["conversion_summary"] if conversion else None
    dataset_card = build_dataset_card(article, inventory, downloads, conversion_summary)
    (metadata / "dataset_card.md").write_text(dataset_card, encoding="utf-8")
    report = build_integrity_report(inventory, downloads, conversion_summary)
    (metadata / "integrity_report.md").write_text(report, encoding="utf-8")


def build_dataset_card(article: dict[str, Any], inventory: dict[str, Any], downloads: list[dict[str, Any]], conversion_summary: dict[str, Any] | None) -> str:
    resolutions = ", ".join(
        f"{row['width']}x{row['height']} ({row['count']})"
        for row in inventory["resolution_distribution"][:12]
    )
    classes = ", ".join(inventory["classes"])
    return f"""# UATD Dataset Card

- Official name: {article.get('title')}
- Official paper: {PAPER_URL} ({PAPER_DOI})
- Download URL: {ARTICLE_HTML_URL}; API {ARTICLE_API_URL}
- Sonar modality: multibeam forward-looking sonar (MFLS), from the official paper metadata.
- Sonar device: Tritech Gemini 1200ik, from the official paper metadata.
- Number of images: {inventory['image_count']}
- Classes: {classes}
- Class count: {inventory['class_count']}
- Annotation type: {inventory['annotation_formats']}
- BBox format: {inventory['bbox_formats']}
- Image extensions: {inventory['image_extensions']}
- Image resolutions: {resolutions}
- Image channels/modes: channels={inventory['channel_counts']}, modes={inventory['image_modes']}
- Official split information: {inventory['official_split']}
- Detected leakage risks: {inventory['leakage_risks']['risk_note']}
- Current project-compatible format: {'YOLO under datasets/UATD/processed_yolo' if conversion_summary else 'not converted'}
- Conversion procedure: `python scripts/prepare_uatd.py --root datasets/UATD`

## Downloaded Files

{chr(10).join(f"- `{item['name']}`: {item['bytes']} bytes, md5 `{item['md5']}`, sha256 `{item['sha256']}`" for item in downloads)}

## Notes

The raw archives and extracted source tree are kept under `datasets/UATD/raw/`.
No random validation split was created. The official source provides training
and two test archives; validation design needs a separate group/leakage review
before fixed-architecture transfer experiments.
"""


def build_integrity_report(inventory: dict[str, Any], downloads: list[dict[str, Any]], conversion_summary: dict[str, Any] | None) -> str:
    duplicate_count = inventory["duplicate_filenames"]["count"]
    corrupt_count = len(inventory["corrupted_images"])
    empty_count = inventory["empty_annotation_images"]["count"]
    split = inventory["official_split"]
    conversion_text = json.dumps(conversion_summary, indent=2) if conversion_summary else "No conversion generated."
    return f"""# UATD Integrity Report

- Generated at: {inventory['generated_at']}
- Raw file checksums: {len(downloads)} archives downloaded and verified with Figshare MD5.
- Image count: {inventory['image_count']}
- Annotation file count: {inventory['annotation_files']}
- Parsed box count: {inventory['box_count']}
- Classes: {inventory['classes']}
- Corrupted images: {corrupt_count}
- Empty annotation images: {empty_count}
- Duplicate filenames: {duplicate_count}
- Official split: {split}
- Leakage note: {inventory['leakage_risks']['risk_note']}

## Archive Files

{chr(10).join(f"- `{item['name']}`: bytes={item['bytes']}, md5={item['md5']}, sha256={item['sha256']}" for item in downloads)}

## Annotation Summary

- Annotation extensions: {inventory['annotation_extensions']}
- Annotation formats: {inventory['annotation_formats']}
- BBox formats: {inventory['bbox_formats']}
- Class counts: {inventory['class_counts']}

## Image Summary

- Extensions: {inventory['image_extensions']}
- Channel counts: {inventory['channel_counts']}
- Image modes: {inventory['image_modes']}
- Resolution distribution: {inventory['resolution_distribution'][:25]}

## Conversion Summary

```json
{conversion_text}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Download, inspect, and convert UATD.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--no-convert-yolo", action="store_true")
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    raw = root / "raw"
    archives = raw / "archives"
    extracted = raw / "extracted"
    metadata = root / "metadata"
    for directory in (archives, extracted, metadata):
        directory.mkdir(parents=True, exist_ok=True)

    article = read_json_url(ARTICLE_API_URL)
    (metadata / "figshare_article_api.json").write_text(json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8")

    downloads: list[dict[str, Any]] = []
    for file_info in article.get("files", []):
        name = file_info["name"]
        destination = archives / name
        expected_size = int(file_info["size"])
        expected_md5 = str(file_info.get("computed_md5") or file_info.get("supplied_md5"))
        if not args.skip_download:
            download_file(str(file_info["download_url"]), destination, expected_size, expected_md5)
        if not destination.is_file():
            raise FileNotFoundError(destination)
        observed_md5 = md5_file(destination)
        if observed_md5.lower() != expected_md5.lower():
            raise RuntimeError(f"MD5 mismatch for {destination}: {observed_md5} != {expected_md5}")
        downloads.append(
            {
                "figshare_file_id": int(file_info["id"]),
                "name": name,
                "path": str(destination),
                "download_url": str(file_info["download_url"]),
                "bytes": destination.stat().st_size,
                "expected_bytes": expected_size,
                "md5": observed_md5,
                "expected_md5": expected_md5,
                "sha256": sha256_file(destination),
            }
        )
        if not args.skip_extract and destination.suffix.lower() == ".zip":
            safe_extract_zip(destination, extracted / destination.stem)

    inventory = build_inventory(root)
    conversion = None
    if not args.no_convert_yolo:
        conversion = convert_to_yolo(root, inventory, link_mode=args.link_mode)
    write_metadata(root, article, downloads, inventory, conversion)
    print(json.dumps({"root": str(root), "image_count": inventory["image_count"], "classes": inventory["classes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
