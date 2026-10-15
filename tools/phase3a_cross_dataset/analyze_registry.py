"""Train-only cross-dataset evidence audit for Phase 3A.

The script never opens validation or test images.  Split-integrity statements
are derived from already-existing manifests/reports and filename inventories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
QUANTILES = (10, 25, 50, 75, 90)
EVIDENCE_FIELDS = (
    "small_target_evidence",
    "P2_visibility_evidence",
    "low_contrast_proxy",
    "frequency_difference",
    "high_aspect_ratio",
    "dense_scene",
    "background_variability",
    "orientation_label_available",
    "mask_available",
    "sequence_information_available",
)


@dataclass(frozen=True)
class DatasetSpec:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset_id"])

    def path(self, key: str) -> Path | None:
        value = self.raw.get(key)
        return None if value in (None, "") else (self.repo_root / str(value)).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    arr = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, **{f"p{q}": None for q in QUANTILES}}
    result: dict[str, float | int | None] = {
        "count": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
    }
    result.update({f"p{q}": float(np.percentile(arr, q)) for q in QUANTILES})
    return result


def resolve_repo_root(registry_path: Path) -> Path:
    candidate = registry_path.resolve().parent.parent
    if not (candidate / "pyproject.toml").exists():
        raise FileNotFoundError(f"Cannot resolve repository root from {registry_path}")
    return candidate


def load_registry(path: Path) -> tuple[dict[str, Any], list[DatasetSpec], Path]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError("Registry must contain a datasets list")
    repo_root = resolve_repo_root(path)
    specs = [DatasetSpec(dict(item), repo_root) for item in payload["datasets"]]
    ids = [spec.dataset_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset_id in registry")
    return payload, specs, repo_root


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory missing: {path}")
    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def read_yolo_labels(path: Path, class_count: int) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation: {path}")
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = text.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        cls = int(parts[0])
        vals = tuple(float(v) for v in parts[1:])
        if cls < 0 or cls >= class_count:
            raise ValueError(f"{path}:{line_number}: class {cls} outside [0,{class_count})")
        if not all(math.isfinite(v) for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            raise ValueError(f"{path}:{line_number}: invalid normalized bbox")
        rows.append((cls, *vals))
    return rows


def pixel_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(math.floor(x1)))), max(0, min(height, int(math.floor(y1)))),
        max(0, min(width, int(math.ceil(x2)))), max(0, min(height, int(math.ceil(y2)))),
    )


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def contrast_region(arr: np.ndarray, box: tuple[float, float, float, float], all_boxes: list[tuple[float, float, float, float]], alpha: float, epsilon: float) -> dict[str, Any]:
    height, width = arr.shape
    x1, y1, x2, y2 = pixel_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return {"contrast_status": "empty_target", "contrast": None}
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1
    ex1, ey1, ex2, ey2 = pixel_box((cx-alpha*bw/2, cy-alpha*bh/2, cx+alpha*bw/2, cy+alpha*bh/2), width, height)
    local = arr[ey1:ey2, ex1:ex2]
    mask = np.ones(local.shape, dtype=bool)
    for other in all_boxes:
        ox1, oy1, ox2, oy2 = pixel_box(other, width, height)
        ax, bx = max(ox1, ex1)-ex1, min(ox2, ex2)-ex1
        ay, by = max(oy1, ey1)-ey1, min(oy2, ey2)-ey1
        if bx > ax and by > ay:
            mask[ay:by, ax:bx] = False
    target = arr[y1:y2, x1:x2]
    background = local[mask]
    if target.size == 0 or background.size < 8:
        return {"contrast_status": "insufficient_background", "contrast": None}
    mt, st = float(target.mean()), float(target.std())
    mb, sb = float(background.mean()), float(background.std())
    gy, gx = np.gradient(target) if min(target.shape) >= 2 else (np.zeros_like(target), np.zeros_like(target))
    magnitude = np.hypot(gx, gy)
    return {
        "contrast_status": "ok", "target_mean": mt, "target_std": st,
        "background_mean": mb, "background_std": sb,
        "contrast": abs(mt-mb)/(sb+epsilon), "target_local_variance": float(target.var()),
        "target_gradient_energy": float(np.mean(magnitude*magnitude)),
        "target_edge_density": None,
    }


def nearby_background(arr: np.ndarray, box: tuple[float, float, float, float], all_boxes: list[tuple[float, float, float, float]]) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    height, width = arr.shape
    x1, y1, x2, y2 = pixel_box(box, width, height)
    bw, bh = x2-x1, y2-y1
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            bx, by = x1+dx*bw, y1+dy*bh
            candidate = (bx, by, bx+bw, by+bh)
            if bx < 0 or by < 0 or candidate[2] > width or candidate[3] > height:
                continue
            if any(intersects(candidate, pixel_box(other, width, height)) for other in all_boxes):
                continue
            candidates.append((dx*dx*bw*bw + dy*dy*bh*bh, candidate))
    if not candidates:
        return None, None
    bounds = min(candidates, key=lambda item: (item[0], item[1]))[1]
    return arr[bounds[1]:bounds[3], bounds[0]:bounds[2]], bounds


def frequency_energy(crop: np.ndarray, edges: tuple[float, float], epsilon: float = 1e-12) -> dict[str, Any]:
    if crop.size == 0 or min(crop.shape) < 4:
        return {"frequency_status": "crop_too_small", "low": None, "middle": None, "high": None}
    centered = crop.astype(np.float64) - float(crop.mean())
    window = np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1]))
    spectrum = np.abs(np.fft.fft2(centered*window))**2
    spectrum[0, 0] = 0.0
    fy, fx = np.meshgrid(np.fft.fftfreq(crop.shape[0]), np.fft.fftfreq(crop.shape[1]), indexing="ij")
    radius = np.hypot(fx, fy) / math.sqrt(0.5**2 + 0.5**2)
    total = float(spectrum.sum())
    if total <= epsilon:
        return {"frequency_status": "zero_ac_energy", "low": None, "middle": None, "high": None}
    lo, hi = edges
    return {
        "frequency_status": "ok",
        "low": float(spectrum[radius < lo].sum()/total),
        "middle": float(spectrum[(radius >= lo) & (radius < hi)].sum()/total),
        "high": float(spectrum[radius >= hi].sum()/total),
    }


def background_metrics(arr: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> tuple[float | None, float | None]:
    mask = np.ones(arr.shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = pixel_box(box, arr.shape[1], arr.shape[0])
        mask[y1:y2, x1:x2] = False
    pixels = arr[mask]
    background_std = float(pixels.std()) if pixels.size else None
    tile_means = []
    for ys in np.array_split(np.arange(arr.shape[0]), 4):
        for xs in np.array_split(np.arange(arr.shape[1]), 4):
            tile_mask = mask[np.ix_(ys, xs)]
            tile = arr[np.ix_(ys, xs)][tile_mask]
            if tile.size:
                tile_means.append(float(tile.mean()))
    return background_std, float(np.std(tile_means)) if tile_means else None


def count_split_files(spec: DatasetSpec) -> dict[str, dict[str, int | None]]:
    result = {}
    for split in ("train", "valid", "test"):
        images = spec.path(f"{split}_images")
        labels = spec.path(f"{split}_labels")
        result[split] = {
            "images": None if images is None or not images.is_dir() else sum(1 for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            "labels": None if labels is None or not labels.is_dir() else sum(1 for p in labels.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        }
    return result


def analyze_dataset(spec: DatasetSpec, defaults: dict[str, Any], output: Path, max_images: int | None) -> dict[str, Any]:
    images_dir, labels_dir = spec.path("train_images"), spec.path("train_labels")
    if images_dir is None or labels_dir is None:
        raise ValueError(f"{spec.dataset_id}: train paths are not registered")
    images = list_images(images_dir)
    if max_images is not None:
        images = images[:max_images]
    classes = [str(name) for name in spec.raw["classes"]]
    input_size = int(defaults["input_size"])
    alpha = float(defaults["contrast_ring_alpha"])
    epsilon = float(defaults["contrast_epsilon"])
    edges = tuple(float(v) for v in defaults["frequency_edges"])
    edge_threshold = float(defaults["edge_gradient_threshold"])
    small_area = float(defaults["coco_small_area_px2"])
    medium_area = float(defaults["coco_medium_area_px2"])

    image_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    class_images: defaultdict[str, set[str]] = defaultdict(set)
    for image_index, image_path in enumerate(images):
        label_path = labels_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path, len(classes))
        with Image.open(image_path) as image:
            gray = image.convert("L")
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        height, width = arr.shape
        boxes = []
        for cls, cx, cy, bw, bh in labels:
            boxes.append(((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height))
        background_std, background_tile_mean_std = background_metrics(arr, boxes)
        image_rows.append({
            "dataset_id": spec.dataset_id, "image": image_path.name, "width": width, "height": height,
            "bbox_count": len(labels), "gray_mean": float(arr.mean()), "gray_std": float(arr.std()),
            "dynamic_range_p99_p01": float(np.percentile(arr, 99)-np.percentile(arr, 1)),
            "background_std": background_std, "background_tile_mean_std_4x4": background_tile_mean_std,
        })
        centers = np.asarray([(row[1], row[2]) for row in labels], dtype=np.float64)
        nearest = [None] * len(labels)
        if len(centers) > 1:
            distances = np.sqrt(((centers[:, None, :]-centers[None, :, :])**2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            nearest = [float(v) for v in distances.min(axis=1)]
        for box_index, ((cls, cx, cy, bw, bh), box) in enumerate(zip(labels, boxes)):
            class_name = classes[cls]
            class_counts[class_name] += 1
            class_images[class_name].add(image_path.name)
            w_px, h_px = bw*width, bh*height
            input_w, input_h = bw*input_size, bh*input_size
            input_area = input_w*input_h
            contrast = contrast_region(arr, box, boxes, alpha, epsilon)
            if contrast.get("target_edge_density") is None:
                x1, y1, x2, y2 = pixel_box(box, width, height)
                target = arr[y1:y2, x1:x2]
                if min(target.shape, default=0) >= 2:
                    gy, gx = np.gradient(target)
                    contrast["target_edge_density"] = float(np.mean(np.hypot(gx, gy) > edge_threshold))
            target_bounds = pixel_box(box, width, height)
            target_crop = arr[target_bounds[1]:target_bounds[3], target_bounds[0]:target_bounds[2]]
            background_crop, background_bounds = nearby_background(arr, box, boxes)
            target_frequency = frequency_energy(target_crop, edges)
            background_frequency = frequency_energy(background_crop, edges) if background_crop is not None else {"frequency_status": "no_background_crop", "low": None, "middle": None, "high": None}
            common = {"dataset_id": spec.dataset_id, "image": image_path.name, "box_index": box_index, "class_id": cls, "class_name": class_name}
            bbox_rows.append({
                **common, "native_width_px": w_px, "native_height_px": h_px, "native_area_px2": w_px*h_px,
                "width_norm": bw, "height_norm": bh, "area_norm": bw*bh, "equivalent_side_norm": math.sqrt(bw*bh),
                "aspect_ratio_w_h": bw/bh, "input_width_px": input_w, "input_height_px": input_h,
                "input_area_px2": input_area, "input_equivalent_side_px": math.sqrt(input_area),
                "coco_size_bin_at_640": "small" if input_area < small_area else "medium" if input_area < medium_area else "large",
                "nearest_object_center_spacing_norm": nearest[box_index],
                **contrast,
                "target_freq_low": target_frequency.get("low"), "target_freq_middle": target_frequency.get("middle"), "target_freq_high": target_frequency.get("high"),
                "background_freq_low": background_frequency.get("low"), "background_freq_middle": background_frequency.get("middle"), "background_freq_high": background_frequency.get("high"),
                "frequency_pair_status": "ok" if target_frequency.get("frequency_status") == background_frequency.get("frequency_status") == "ok" else f"target={target_frequency.get('frequency_status')};background={background_frequency.get('frequency_status')}",
            })
            for region, result, bounds in (("target", target_frequency, target_bounds), ("background", background_frequency, background_bounds)):
                frequency_rows.append({**common, "region": region, "bounds": None if bounds is None else json.dumps(bounds), **result})
        if (image_index + 1) % 500 == 0:
            print(f"[{spec.dataset_id}] {image_index+1}/{len(images)} train images", flush=True)

    if not bbox_rows:
        raise ValueError(f"{spec.dataset_id}: train split contains no GT boxes")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "image_statistics.csv", image_rows)
    write_csv(output / "bbox_statistics.csv", bbox_rows)
    write_csv(output / "frequency_statistics.csv", frequency_rows)
    per_class_rows = []
    for name in classes:
        rows = [row for row in bbox_rows if row["class_name"] == name]
        per_class_rows.append({"dataset_id": spec.dataset_id, "class_name": name, "instances": len(rows), "images": len(class_images[name]), "input_side_p50": summary([r["input_equivalent_side_px"] for r in rows])["p50"]})
    write_csv(output / "class_statistics.csv", per_class_rows)

    paired = [row for row in bbox_rows if row["frequency_pair_status"] == "ok"]
    size_counts = Counter(row["coco_size_bin_at_640"] for row in bbox_rows)
    visibility = {}
    for stride in (4, 8, 16, 32):
        cells = [min(row["input_width_px"], row["input_height_px"])/stride for row in bbox_rows]
        visibility[f"stride{stride}"] = {"min_side_cells": summary(cells), **{f"fraction_min_side_lt_{n}_cells": float(np.mean(np.asarray(cells) < n)) for n in (1, 2, 3)}}
    gt_per_image = [row["bbox_count"] for row in image_rows]
    stats = {
        "schema_version": "phase3a-dataset-statistics-v1", "generated_at": utc_now(),
        "dataset_id": spec.dataset_id, "dataset_name": spec.raw["dataset_name"], "modality": spec.raw["modality"],
        "scope": {"split": "train", "max_images": max_images, "test_evaluated": False, "validation_pixels_read": False, "test_pixels_read": False},
        "inventory": {"analyzed_train_images": len(image_rows), "train_gt": len(bbox_rows), "class_count": len(classes), "registered_split_counts": count_split_files(spec)},
        "class_counts": dict(class_counts),
        "objects_per_image": summary(gt_per_image),
        "bbox": {key: summary([row[key] for row in bbox_rows]) for key in ("native_width_px", "native_height_px", "native_area_px2", "width_norm", "height_norm", "area_norm", "equivalent_side_norm", "aspect_ratio_w_h", "input_width_px", "input_height_px", "input_area_px2", "input_equivalent_side_px")},
        "coco_size_bins_at_640": {"definition": {"small": "area < 32^2", "medium": "32^2 <= area < 96^2", "large": "area >= 96^2"}, "counts": {name: size_counts.get(name, 0) for name in ("small", "medium", "large")}, "fractions": {name: size_counts.get(name, 0)/len(bbox_rows) for name in ("small", "medium", "large")}},
        "feature_map_visibility": visibility,
        "aspect_ratio_proxy": {"axis_aligned_bbox_only": True, "fraction_ar_gt_2_or_lt_0p5": float(np.mean([(row["aspect_ratio_w_h"] > 2 or row["aspect_ratio_w_h"] < 0.5) for row in bbox_rows])), "orientation_label_available": bool(spec.raw.get("orientation_label_available", False))},
        "contrast": {"definition": f"abs(mu_target-mu_ring)/(sigma_ring+{epsilon}); ring={alpha}x bbox, all GT excluded", "bbox_level_proxy_not_physical_contrast": True, "score": summary([row.get("contrast") for row in bbox_rows]), "fraction_lt_1_background_std": float(np.mean([row["contrast"] < 1 for row in bbox_rows if finite(row.get("contrast"))]))},
        "local_texture": {"target_variance": summary([row.get("target_local_variance") for row in bbox_rows]), "target_gradient_energy": summary([row.get("target_gradient_energy") for row in bbox_rows]), "target_edge_density": summary([row.get("target_edge_density") for row in bbox_rows]), "edge_threshold": edge_threshold},
        "frequency": {"definition": "paired equal-size target/background crops; mean subtraction; 2D Hann; DC removed; radial normalized bands [0,1/3),[1/3,2/3),[2/3,1]", "valid_pairs": len(paired), "target": {band: summary([row[f"target_freq_{band}"] for row in paired]) for band in ("low", "middle", "high")}, "background": {band: summary([row[f"background_freq_{band}"] for row in paired]) for band in ("low", "middle", "high")}, "paired_target_minus_background": {band: summary([row[f"target_freq_{band}"]-row[f"background_freq_{band}"] for row in paired]) for band in ("low", "middle", "high")}},
        "scene_density": {"objects_per_image": summary(gt_per_image), "fraction_images_ge_3_objects": float(np.mean(np.asarray(gt_per_image) >= 3)), "nearest_object_spacing_norm": summary([row.get("nearest_object_center_spacing_norm") for row in bbox_rows])},
        "background_variability": {"background_pixel_std": summary([row.get("background_std") for row in image_rows]), "background_tile_mean_std_4x4": summary([row.get("background_tile_mean_std_4x4") for row in image_rows]), "definition": "GT boxes masked; grayscale [0,1]; per-image background std and std of valid 4x4 tile means"},
        "metadata": {key: spec.raw.get(key) for key in ("license", "source_url", "source_manifest", "acquisition_group_key", "split_policy", "sequence_information_available", "mask_available")},
    }
    dump_json(output / "dataset_statistics.json", stats)
    return {"stats": stats, "bbox_rows": bbox_rows, "image_rows": image_rows}


def evidence_item(value: Any, threshold: str, status: str, interpretation: str, limitation: str) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "status": status, "interpretation": interpretation, "confidence_limitation": limitation}


def build_evidence(stats: dict[str, Any], spec: DatasetSpec) -> dict[str, dict[str, Any]]:
    small = stats["coco_size_bins_at_640"]["fractions"]["small"]
    p2 = stats["feature_map_visibility"]["stride8"]["fraction_min_side_lt_2_cells"]
    contrast_fraction = stats["contrast"]["fraction_lt_1_background_std"]
    freq = stats["frequency"]["paired_target_minus_background"]["low"]["p50"]
    freq_abs = abs(freq) if freq is not None else None
    aspect = stats["aspect_ratio_proxy"]["fraction_ar_gt_2_or_lt_0p5"]
    dense = stats["scene_density"]["fraction_images_ge_3_objects"]
    bg = stats["background_variability"]["background_pixel_std"]["p50"]
    sequence_value = spec.raw.get("sequence_information_available", False)
    return {
        "small_target_evidence": evidence_item(small, ">=0.50 COCO-small boxes at 640", "SUPPORTED" if small >= 0.5 else "NOT_SUPPORTED", "Descriptive size evidence at square 640 input.", "COCO bins are a reporting convention, not physical target size."),
        "P2_visibility_evidence": evidence_item(p2, ">=0.25 boxes with min side <2 stride-8 cells", "SUPPORTED" if p2 >= 0.25 else "NOT_SUPPORTED", "Indicates whether stride-4 should remain a controlled output candidate.", "Visibility proxy does not prove an AP gain from P2."),
        "low_contrast_proxy": evidence_item(contrast_fraction, ">=0.50 boxes with C<1", "SUPPORTED" if contrast_fraction >= 0.5 else "NOT_SUPPORTED", "BBox mean differs from its local ring by less than one ring standard deviation.", "BBox-level proxy; bbox is not a target mask or physical contrast measurement."),
        "frequency_difference": evidence_item(freq, "abs(median paired low-band delta)>=0.05", "SUPPORTED" if freq_abs is not None and freq_abs >= 0.05 else "WEAK_EVIDENCE", "Paired target-minus-background low-band energy difference.", f"Valid paired crops={stats['frequency']['valid_pairs']}; frequency difference does not identify a kernel/operator."),
        "high_aspect_ratio": evidence_item(aspect, ">=0.25 boxes with w/h>2 or <0.5", "SUPPORTED" if aspect >= 0.25 else "NOT_SUPPORTED", "Axis-aligned bbox shape tendency only.", "No true orientation is inferred without rotated boxes or masks."),
        "dense_scene": evidence_item(dense, ">=0.50 train images with >=3 GT", "SUPPORTED" if dense >= 0.5 else "NOT_SUPPORTED", "High multi-object image prevalence under a fixed threshold.", "Does not measure occlusion or physical crowding."),
        "background_variability": evidence_item(bg, ">=0.10 median masked-background grayscale std", "SUPPORTED" if bg is not None and bg >= 0.10 else "WEAK_EVIDENCE", "Non-semantic masked-background intensity variability.", "Sensitive to source dynamic range and compression."),
        "orientation_label_available": evidence_item(bool(spec.raw.get("orientation_label_available", False)), "explicit rotated bbox/orientation/mask metadata", "SUPPORTED" if spec.raw.get("orientation_label_available", False) else "NOT_TESTABLE", "Availability of true orientation supervision.", "Axis-aligned aspect ratio is not orientation."),
        "mask_available": evidence_item(bool(spec.raw.get("mask_available", False)), "source-provided masks", "SUPPORTED" if spec.raw.get("mask_available", False) else "NOT_TESTABLE", "Availability of segmentation supervision.", "Phase 3A does not synthesize masks."),
        "sequence_information_available": evidence_item(sequence_value, "source acquisition/mission/sequence metadata", "SUPPORTED" if sequence_value is True else "WEAK_EVIDENCE" if sequence_value else "NOT_TESTABLE", "Strength of acquisition grouping metadata.", "Filename heuristics are not proven mission identity."),
    }


def flatten_evidence(spec: DatasetSpec, evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = {"dataset_id": spec.dataset_id, "dataset_name": spec.raw["dataset_name"], "modality": spec.raw["modality"]}
    for field in EVIDENCE_FIELDS:
        item = evidence[field]
        row[field] = item["status"]
        for key in ("value", "threshold", "interpretation", "confidence_limitation"):
            value = item[key]
            row[f"{field}_{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
    return row


def aggregate_row(stats: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset_id": stats["dataset_id"], "dataset_name": stats["dataset_name"], "modality": stats["modality"],
        "train_images": stats["inventory"]["analyzed_train_images"], "train_gt": stats["inventory"]["train_gt"], "classes": stats["inventory"]["class_count"],
        "objects_per_image_mean": stats["objects_per_image"]["mean"], "coco_small_fraction_at_640": stats["coco_size_bins_at_640"]["fractions"]["small"],
        "aspect_extreme_fraction": stats["aspect_ratio_proxy"]["fraction_ar_gt_2_or_lt_0p5"], "contrast_C_median": stats["contrast"]["score"]["p50"],
        "contrast_C_lt1_fraction": stats["contrast"]["fraction_lt_1_background_std"], "frequency_valid_pairs": stats["frequency"]["valid_pairs"],
        "target_minus_background_low_median": stats["frequency"]["paired_target_minus_background"]["low"]["p50"],
        "background_std_median": stats["background_variability"]["background_pixel_std"]["p50"], "dense_image_fraction_ge3": stats["scene_density"]["fraction_images_ge_3_objects"],
        "test_evaluated": False,
    }
    for metric in ("input_width_px", "input_height_px", "area_norm", "input_equivalent_side_px"):
        for q in QUANTILES:
            row[f"{metric}_p{q}"] = stats["bbox"][metric][f"p{q}"]
    for stride in (4, 8, 16, 32):
        for n in (1, 2, 3):
            row[f"stride{stride}_min_side_lt{n}_fraction"] = stats["feature_map_visibility"][f"stride{stride}"][f"fraction_min_side_lt_{n}_cells"]
    return row


def integrity_from_existing(specs: list[DatasetSpec], repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        base = {"dataset_id": spec.dataset_id, "split_policy": spec.raw.get("split_policy"), "test_metrics_used": False, "test_pixels_opened_by_phase3a": False}
        if spec.dataset_id == "cylinder2_v6_dataset_a":
            path = repo_root / "reports/sonar_analysis_full/split_integrity_report.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            screening = data.get("near_duplicate_screening", {})
            rows.append({
                **base,
                "source": str(path.relative_to(repo_root)),
                "exact_cross_split_groups": len(data.get("exact_duplicates", [])),
                "high_similarity_cross_split_candidates": screening.get("shortlist_pair_count", 0),
                "raw_dhash_candidates": screening.get("raw_hash_pair_count", len(data.get("possible_near_duplicates", []))),
                "sequence_overlap": "6 weak filename-inferred groups cross splits",
                "status": "POTENTIAL_ACQUISITION_GROUP_DEPENDENCE",
                "limitation": "63 correlation-filtered candidates remain exploratory and are not labeled leakage; 44,268 is the raw dHash collision count, not the reviewed shortlist.",
            })
        elif spec.dataset_id == "classic_sctd_1_0":
            path = spec.path("source_manifest")
            data = json.loads(path.read_text(encoding="utf-8"))
            split_of = {}
            for split in ("train", "valid", "test"):
                image_dir = spec.path(f"{split}_images")
                if image_dir and image_dir.is_dir():
                    split_of.update({p.name: split for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})
            cross = []
            for digest, names in data["integrity"]["duplicate_groups"].items():
                groups = {split_of.get(name) for name in names} - {None}
                if len(groups) > 1:
                    cross.append({"sha256": digest, "files": names, "splits": sorted(groups)})
            rows.append({**base, "source": str(spec.raw.get("source_manifest")), "exact_cross_split_groups": len(cross), "exact_cross_split_detail": cross, "high_similarity_cross_split_candidates": "NOT_TESTED", "sequence_overlap": "numeric-run groups kept together by project split; mission identity unavailable", "status": "EXACT_CROSS_SPLIT_DUPLICATES_CONFIRMED", "limitation": "Eight exact duplicate groups cross the project-derived splits. This blocks treating SCTD validation/test as independent until a deduplicated split is frozen; train-only descriptive evidence remains usable. Mission/voyage metadata is unavailable."})
        elif spec.dataset_id == "uatd_v3":
            audit_path = repo_root / "reports/phase3a_cross_dataset_evidence/integrity/uatd_v3_exact_hash.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                rows.append({
                    **base,
                    "source": str(audit_path.relative_to(repo_root)),
                    "exact_cross_split_groups": audit["exact_cross_split_group_count"],
                    "exact_cross_split_detail": audit["exact_cross_split_groups"],
                    "high_similarity_cross_split_candidates": audit["near_duplicate_status"],
                    "sequence_overlap": "UNKNOWN; numeric stems were renamed by split during conversion and are not acquisition identifiers",
                    "status": "EXACT_CROSS_SPLIT_DUPLICATES_CONFIRMED_NO_VALIDATION_SPLIT",
                    "test_file_bytes_hashed": True,
                    "test_pixels_decoded": False,
                    "limitation": "Two exact image duplicates cross official train/test. No validation or acquisition grouping metadata is registered; perceptual near-duplicate screening was not run to keep test pixels sealed.",
                })
            else:
                rows.append({**base, "source": str(spec.raw.get("source_manifest")), "exact_cross_split_groups": "NOT_TESTED_WITHOUT_HASH_AUDIT", "high_similarity_cross_split_candidates": "NOT_TESTED", "sequence_overlap": "UNKNOWN", "status": "UNRESOLVED_NO_VALIDATION_SPLIT", "limitation": "Official train/test archives exist; no validation or acquisition grouping metadata is registered."})
        else:
            rows.append({**base, "source": spec.raw.get("source_manifest"), "exact_cross_split_groups": "NOT_APPLICABLE", "high_similarity_cross_split_candidates": "NOT_APPLICABLE", "sequence_overlap": "acquisition year only", "status": "BLOCKED_NO_FROZEN_SPLIT", "limitation": "Registered source has no reviewed train/valid/test protocol."})
    return rows


def save_figure(figures: Path, name: str, source_rows: list[dict[str, Any]], plotter) -> None:
    write_csv(figures / f"{name}.csv", source_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotter(ax, source_rows)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=220)
    fig.savefig(figures / f"{name}.svg")
    plt.close(fig)


def render_figures(output: Path, results: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ids = [r["stats"]["dataset_id"] for r in results]
    size_rows = [{"dataset_id": r["stats"]["dataset_id"], "small": r["stats"]["coco_size_bins_at_640"]["fractions"]["small"], "medium": r["stats"]["coco_size_bins_at_640"]["fractions"]["medium"], "large": r["stats"]["coco_size_bins_at_640"]["fractions"]["large"]} for r in results]
    def plot_size(ax, rows):
        bottom = np.zeros(len(rows)); x = np.arange(len(rows))
        for band in ("small", "medium", "large"):
            vals = np.asarray([row[band] for row in rows]); ax.bar(x, vals, bottom=bottom, label=band); bottom += vals
        ax.set_xticks(x, [row["dataset_id"] for row in rows], rotation=15); ax.set_ylim(0, 1); ax.set_ylabel("fraction"); ax.legend()
    save_figure(figures, "coco_size_distribution_at_640", size_rows, plot_size)
    visibility_rows = [{"dataset_id": r["stats"]["dataset_id"], **{f"stride{s}": r["stats"]["feature_map_visibility"][f"stride{s}"]["fraction_min_side_lt_2_cells"] for s in (4,8,16,32)}} for r in results]
    def plot_visibility(ax, rows):
        x=np.arange(len(rows)); width=.2
        for i,s in enumerate((4,8,16,32)): ax.bar(x+(i-1.5)*width,[row[f"stride{s}"] for row in rows],width,label=f"stride{s}")
        ax.set_xticks(x,[row["dataset_id"] for row in rows],rotation=15); ax.set_ylim(0,1); ax.set_ylabel("fraction min side <2 cells"); ax.legend()
    save_figure(figures, "feature_map_visibility", visibility_rows, plot_visibility)
    contrast_rows = [{"dataset_id": r["stats"]["dataset_id"], "median_C": r["stats"]["contrast"]["score"]["p50"], "fraction_C_lt1": r["stats"]["contrast"]["fraction_lt_1_background_std"]} for r in results]
    save_figure(figures, "contrast_comparison", contrast_rows, lambda ax,rows:(ax.bar([r["dataset_id"] for r in rows],[r["median_C"] for r in rows]),ax.set_ylabel("median bbox-ring C"),ax.tick_params(axis="x",rotation=15)))
    freq_rows = [{"dataset_id": r["stats"]["dataset_id"], "low_delta_median": r["stats"]["frequency"]["paired_target_minus_background"]["low"]["p50"], "valid_pairs": r["stats"]["frequency"]["valid_pairs"]} for r in results]
    save_figure(figures, "frequency_low_band_paired_delta", freq_rows, lambda ax,rows:(ax.axhline(0,color="black",lw=.8),ax.bar([r["dataset_id"] for r in rows],[r["low_delta_median"] for r in rows]),ax.set_ylabel("median target-background low-band energy"),ax.tick_params(axis="x",rotation=15)))
    bg_rows = [{"dataset_id": r["stats"]["dataset_id"], "background_std_median": r["stats"]["background_variability"]["background_pixel_std"]["p50"]} for r in results]
    save_figure(figures, "background_variability", bg_rows, lambda ax,rows:(ax.bar([r["dataset_id"] for r in rows],[r["background_std_median"] for r in rows]),ax.set_ylabel("median masked-background std [0,1]"),ax.tick_params(axis="x",rotation=15)))
    density_rows = [{"dataset_id": r["stats"]["dataset_id"], "objects_per_image_mean": r["stats"]["objects_per_image"]["mean"], "fraction_images_ge3": r["stats"]["scene_density"]["fraction_images_ge_3_objects"]} for r in results]
    save_figure(figures, "scene_density", density_rows, lambda ax,rows:(ax.bar([r["dataset_id"] for r in rows],[r["objects_per_image_mean"] for r in rows]),ax.set_ylabel("mean GT/image"),ax.tick_params(axis="x",rotation=15)))


def build_reports(output: Path, registry: dict[str, Any], specs: list[DatasetSpec], results: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], integrity_rows: list[dict[str, Any]]) -> None:
    analyzed = {r["stats"]["dataset_id"]: r for r in results}
    registry_lines = ["# Dataset Registry Audit", "", "Phase 3A statistics use train splits only. Validation and test pixels were not opened by this pipeline.", "", "| Dataset | Modality | Train/Valid/Test | Status | License |", "|---|---|---:|---|---|"]
    for spec in specs:
        counts = count_split_files(spec)
        triplet = "/".join("—" if counts[s]["images"] is None else str(counts[s]["images"]) for s in ("train","valid","test"))
        status = "ANALYZED_TRAIN_ONLY" if spec.dataset_id in analyzed else "REGISTERED_BLOCKED_NO_FROZEN_SPLIT"
        registry_lines.append(f"| {spec.dataset_id} | {spec.raw['modality']} | {triplet} | {status} | {spec.raw['license']} |")
    registry_lines += ["", "Modality is copied only from registered source evidence. Dataset A remains `UNKNOWN`; no visual inference was used.", "", "`test_evaluated=false`."]
    (output / "DATASET_REGISTRY_AUDIT.md").write_text("\n".join(registry_lines)+"\n", encoding="utf-8")

    protocol = """# Phase 3A Protocol

- Evidence construction: train split only; no augmentation, sampling, training, validation inference, or test metrics.
- Integrity exception: UATD test file bytes were SHA256-hashed without image decoding; test pixels never entered evidence construction.
- Common input geometry: normalized YOLO boxes mapped to a 640x640 model input for cross-dataset visibility reporting.
- Contrast: `abs(mu_target-mu_ring)/(sigma_ring+1e-6)`, 2x bbox ring, every GT excluded.
- Frequency: equal-size paired target/background crops; mean subtraction, 2D Hann, DC removal; radial bands 0-1/3, 1/3-2/3, 2/3-1.
- Aspect ratio: axis-aligned bbox shape only; no orientation claim.
- Background variability: GT-masked grayscale standard deviation and 4x4 tile-mean standard deviation.
- Evidence is retained per dataset. No pooled image- or box-weighted aggregate is reported.
- Future evaluation schema: per-domain `mAP_Di`, `MeanDomainAP=mean(mAP_Di)`, and `WorstDomainAP=min(mAP_Di)`; the largest dataset cannot dominate by sample count.
- Frozen `X*={B1,C,D}` and Search Space V0.1 are unchanged.
- `test_evaluated=false`.
"""
    (output / "PHASE3A_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    status_by_var = {
        "P2 output": ("SUPPORTED_SHARED" if sum(row["P2_visibility_evidence"] == "SUPPORTED" for row in matrix_rows) >= 2 else "WEAK_EVIDENCE", "stride-8 visibility proxy"),
        "P3/P4/P5 output": ("SUPPORTED_SHARED", "retained common baseline output; no claim of universal optimality"),
        "3x3 DW": ("SUPPORTED_SHARED", "standard control retained across domains"),
        "5x5 DW": ("WEAK_EVIDENCE", "frequency/scale proxy cannot establish kernel benefit"),
        "larger-kernel candidate": ("NOT_SUPPORTED", "no controlled cross-dataset operator evidence"),
        "directional candidate": ("NOT_TESTABLE", "no true orientation labels"),
        "frequency-domain candidate": ("WEAK_EVIDENCE", "paired spectral differences are descriptive, not operator evidence"),
        "attention candidate": ("NOT_SUPPORTED", "no direct data or architecture evidence"),
        "dilation candidate": ("NOT_SUPPORTED", "no direct controlled evidence"),
        "early-feature preservation": ("SUPPORTED_SHARED" if sum(row["P2_visibility_evidence"] == "SUPPORTED" for row in matrix_rows) >= 2 else "WEAK_EVIDENCE", "same visibility evidence as P2, still requires controlled AP tests"),
        "dense-scene head": ("SUPPORTED_SHARED" if sum(row["dense_scene"] == "SUPPORTED" for row in matrix_rows) >= 2 else "NOT_SUPPORTED", "fixed GT/image threshold"),
        "modality adapter": ("WEAK_EVIDENCE", "one analyzed dataset per known modality confounds dataset and modality; observed shifts are hypotheses only"),
    }
    lines = ["# Search Variable Support Matrix", "", "| Variable | Status | Evidence boundary |", "|---|---|---|"]
    for name,(status,reason) in status_by_var.items(): lines.append(f"| {name} | {status} | {reason} |")
    lines += ["", "Historical `E_arch(Dataset_A)` remains: P2 `CONTEXT_DEPENDENT/UNCERTAIN`; stage7 k5 `UNCERTAIN`; SonarOp V0.2-P0 `NOT_SUPPORTED`.", "", "Data evidence does not override negative architecture evidence."]
    (output / "SEARCH_VARIABLE_SUPPORT_MATRIX.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    shared = [name for name,(status,_) in status_by_var.items() if status == "SUPPORTED_SHARED"]
    modality = [name for name,(status,_) in status_by_var.items() if status == "SUPPORTED_MODALITY_SPECIFIC"]
    modality_lines = [f"- {name}: conditional hypothesis only; no complete modality-specific network is defined." for name in modality]
    if not modality_lines:
        modality_lines = ["- None established. UATD/SCTD differences remain dataset-conditioned because each known modality is represented by only one analyzed dataset."]
    evidence_lines = ["# Shared vs Modality-Specific Evidence", "", "## Shared evidence", ""] + [f"- {name}" for name in shared] + ["", "## Modality-specific evidence", ""] + modality_lines + ["", "Dataset A modality is UNKNOWN and is never used as FLS/SSS/SAS evidence.", "", "P2/high-resolution visibility is supported only by Dataset A and is not replicated in UATD or SCTD.", "", "No directional, shadow, or physical-acoustic claim is derived from axis-aligned boxes or intensity tails."]
    (output / "SHARED_VS_MODALITY_SPECIFIC_EVIDENCE.md").write_text("\n".join(evidence_lines)+"\n", encoding="utf-8")

    draft = """# General Sonar Search Space Draft

This is a specification draft only. No model, NAS run, or search-space implementation is authorized by this artifact.

## Conditional definition

`X_general = X_shared ∪ X_conditional(z)`

### X_shared

- Fixed 640 input for the first cross-dataset control.
- P3/P4/P5 output as the common baseline control.
- 3x3 MBConv depthwise kernel as the common baseline control.
- Fixed channels/depth/expansion, fixed FPN/head semantics per dataset class count.

### X_conditional(z)

- Empty at Phase 3A. One analyzed dataset per known modality cannot separate modality effects from dataset effects.
- The schema reserves a small input/normalization adapter, output-scale configuration, and SonarOp placement, but none is admitted until replicated modality evidence and controlled transfer tests exist.

### Evidence-collection candidates not yet admitted to X_general

- P2/P3/P4/P5 output and early stride-4 preservation: supported only by Dataset A visibility statistics.
- 5x5 depthwise kernel: frequency proxies are shared, but neither the proxy nor Dataset A's uncertain stage7 result establishes cross-dataset AP benefit.

### Excluded

- Directional, frequency-domain, attention, dilation, kernel7, arbitrary FPN topology, dense-scene head, quantization, and FPGA mapping search variables.

## Future evaluation schema

- Record `mAP_D1 ... mAP_Dk` separately.
- `MeanDomainAP = mean(mAP_Di)`.
- `WorstDomainAP = min(mAP_Di)`.
- Never use pooled global mAP as the only objective.

The frozen Phase 2F network candidate set `X*={B1,C,D}` is unchanged.
"""
    (output / "GENERAL_SONAR_SEARCH_SPACE_DRAFT.md").write_text(draft, encoding="utf-8")

    gate = "CONDITIONAL"
    gate_text = """# SonarOp P1 Gate

`P1_GATE = CONDITIONAL`

Only Dataset A supports the P2/high-resolution visibility hypothesis; UATD and SCTD do not replicate it. The datasets differ in size, contrast, frequency proxy, and background statistics, but one analyzed dataset per known modality cannot separate modality effects from dataset effects. No repeatable concrete operator supersedes the failed V0.2-P0 control. Directional evidence is not testable and frequency differences remain proxy evidence.

Minimum additional evidence: pre-registered fixed-architecture transfer controls on at least two datasets with valid development splits, using one operator hypothesis and matched parameter/compute controls. Do not implement P1 from this audit alone.

`test_evaluated=false`.
"""
    (output / "SONAROP_P1_GATE.md").write_text(gate_text, encoding="utf-8")

    integ = ["# Split Integrity Audit", "", "This report reuses existing hashes/manifests and filename inventories. Phase 3A did not open validation/test pixels or compute model metrics.", "", "| Dataset | Exact cross-split | Similarity candidates | Status |", "|---|---:|---:|---|"]
    for row in integrity_rows: integ.append(f"| {row['dataset_id']} | {row['exact_cross_split_groups']} | {row['high_similarity_cross_split_candidates']} | {row['status']} |")
    integ += ["", "Dataset A similarity candidates are not called leakage; use `potential acquisition-group dependence` until source provenance proves otherwise. Exact duplicates are confirmed for UATD (2 train/test groups) and SCTD (8 project-split groups). Their evaluation protocols must be repaired before model comparison. UATD near-duplicate screening was not run because it would require decoding sealed test pixels.", "", "Full machine-readable details: `split_integrity_audit.json`."]
    (output / "split_integrity_audit.md").write_text("\n".join(integ)+"\n", encoding="utf-8")
    dump_json(output / "split_integrity_audit.json", integrity_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/sonar_datasets_registry.yaml")
    parser.add_argument("--output", default="reports/phase3a_cross_dataset_evidence")
    parser.add_argument("--dataset-id", action="append", help="Analyze only selected enabled dataset(s)")
    parser.add_argument("--max-images", type=int, default=None, help="Deterministic prefix for smoke tests only")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Regenerate aggregate reports from existing full train statistics without opening dataset images",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.reports_only and args.max_images is not None:
        parser.error("--reports-only cannot be combined with --max-images")
    registry_path = Path(args.registry).resolve()
    registry, specs, repo_root = load_registry(registry_path)
    output = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    selected = set(args.dataset_id or [])
    enabled = [spec for spec in specs if spec.raw.get("analysis_enabled", False) and (not selected or spec.dataset_id in selected)]
    if selected - {spec.dataset_id for spec in specs}:
        raise ValueError(f"Unknown dataset IDs: {sorted(selected - {spec.dataset_id for spec in specs})}")
    if not enabled:
        raise ValueError("No enabled datasets selected")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in enabled:
        if args.reports_only:
            stats_path = output / "datasets" / spec.dataset_id / "dataset_statistics.json"
            if not stats_path.is_file():
                raise FileNotFoundError(f"Missing existing full statistics for --reports-only: {stats_path}")
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            registered_train = stats["inventory"].get("registered_split_counts", {}).get("train", {}).get("images")
            if registered_train is None or stats["inventory"]["analyzed_train_images"] != registered_train:
                raise ValueError(f"Refusing reports-only finalization from sampled statistics: {stats_path}")
            print(f"Loading existing full train statistics for {spec.dataset_id}", flush=True)
            results.append({"stats": stats})
        else:
            print(f"Analyzing {spec.dataset_id} train split", flush=True)
            results.append(analyze_dataset(spec, registry["analysis_defaults"], output / "datasets" / spec.dataset_id, args.max_images))
    stats_rows = [aggregate_row(result["stats"]) for result in results]
    write_csv(output / "dataset_statistics.csv", stats_rows)
    matrix_rows = []
    for result in results:
        spec = next(item for item in specs if item.dataset_id == result["stats"]["dataset_id"])
        matrix_rows.append(flatten_evidence(spec, build_evidence(result["stats"], spec)))
    write_csv(output / "DATASET_EVIDENCE_MATRIX.csv", matrix_rows)
    integrity_rows = integrity_from_existing(specs, repo_root)
    render_figures(output, results)
    build_reports(output, registry, specs, results, matrix_rows, integrity_rows)
    manifest = {
        "schema_version": "phase3a-cross-dataset-evidence-v1", "generated_at": utc_now(),
        "registry": str(registry_path.relative_to(repo_root)), "registry_sha256": sha256(registry_path),
        "source": str(Path(__file__).resolve().relative_to(repo_root)), "source_sha256": sha256(Path(__file__).resolve()),
        "analyzed_datasets": [r["stats"]["dataset_id"] for r in results], "split": "train",
        "max_images": args.max_images, "reports_only": args.reports_only,
        "evidence_source": "existing_full_train_statistics" if args.reports_only else "direct_train_analysis",
        "test_evaluated": False, "validation_pixels_read": False, "test_pixels_read": False,
        "test_file_bytes_hashed_for_integrity": True,
        "frozen_search_space_modified": False, "frozen_candidate_set_modified": False,
    }
    dump_json(output / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
