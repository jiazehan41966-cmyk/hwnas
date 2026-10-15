"""Train-split-only sonar detection dataset evidence audit.

The module is intentionally dataset-config driven so external sonar datasets
share one statistics implementation and one evidence-vector schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageOps

from hwnas_fpga.data.external_sources import audit_yolo_detection_tree


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


@dataclass(frozen=True)
class EvidenceAuditConfig:
    dataset_id: str
    dataset_root: Path
    output_dir: Path
    split: str = "train"
    data_yaml: Path | None = None
    source_modality: str = "unknown"
    source_task: str = "object_detection"
    allowed_splits: tuple[str, ...] = ("train",)
    forbidden_splits: tuple[str, ...] = ("valid", "val", "test", "test1", "test2")
    class_names: tuple[str, ...] = ()
    fft_size: int = 256
    local_context_scale: float = 1.5
    edge_gradient_threshold: float = 0.12
    small_area_fraction: float = 0.01
    medium_area_fraction: float = 0.10
    max_images: int | None = None
    comparison_artifacts: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path) -> EvidenceAuditConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dataset = payload.get("dataset", {})
    audit = payload.get("audit", {})
    output = payload.get("output", {})

    dataset_root = Path(dataset["root"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    data_yaml = dataset.get("data_yaml")
    data_yaml_path = None
    if data_yaml:
        data_yaml_path = Path(data_yaml).expanduser()
        if not data_yaml_path.is_absolute():
            data_yaml_path = (path.parent / data_yaml_path).resolve()

    class_names = tuple(dataset.get("class_names") or ())
    if not class_names and data_yaml_path is not None:
        class_names = tuple(_load_yolo_names(data_yaml_path))

    output_dir = Path(output.get("dir", "results/evidence")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()

    return EvidenceAuditConfig(
        dataset_id=str(dataset["id"]),
        dataset_root=dataset_root,
        output_dir=output_dir,
        split=str(audit.get("split", "train")),
        data_yaml=data_yaml_path,
        source_modality=str(dataset.get("source_modality", "unknown")),
        source_task=str(dataset.get("source_task", "object_detection")),
        allowed_splits=tuple(audit.get("allowed_splits", ("train",))),
        forbidden_splits=tuple(audit.get("forbidden_splits", ("valid", "val", "test", "test1", "test2"))),
        class_names=class_names,
        fft_size=int(audit.get("fft_size", 256)),
        local_context_scale=float(audit.get("local_context_scale", 1.5)),
        edge_gradient_threshold=float(audit.get("edge_gradient_threshold", 0.12)),
        small_area_fraction=float(audit.get("small_area_fraction", 0.01)),
        medium_area_fraction=float(audit.get("medium_area_fraction", 0.10)),
        max_images=int(audit["max_images"]) if audit.get("max_images") is not None else None,
        comparison_artifacts=payload.get("comparison_artifacts") or {},
    )


def _load_yolo_names(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def run_evidence_audit(config: EvidenceAuditConfig) -> dict[str, Path]:
    if config.split not in config.allowed_splits:
        raise ValueError(f"Refusing split {config.split!r}; allowed_splits={config.allowed_splits!r}")
    if config.split in config.forbidden_splits:
        raise ValueError(f"Refusing forbidden split {config.split!r}")

    split_root = config.dataset_root / config.split
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected YOLO split directories under {split_root}")

    output_dir = config.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_map = {index: name for index, name in enumerate(config.class_names)}
    pairing_audit = audit_yolo_detection_tree(split_root, class_names=class_map)

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if config.max_images is not None:
        image_paths = image_paths[: config.max_images]

    records = _collect_statistics(config, image_paths, labels_dir)
    dataset_stats = _build_dataset_stats(config, pairing_audit, records)
    evidence_vector = _build_evidence_vector(config, dataset_stats)
    report = _render_report(config, dataset_stats, evidence_vector)

    paths = {
        "dataset_stats": output_dir / "dataset_stats.json",
        "bbox_stats": output_dir / "bbox_stats.csv",
        "class_stats": output_dir / "class_stats.csv",
        "frequency_stats": output_dir / "frequency_stats.csv",
        "evidence_report": output_dir / "evidence_report.md",
        "evidence_vector": output_dir / "evidence_vector.json",
    }
    _write_json(paths["dataset_stats"], dataset_stats)
    _write_json(paths["evidence_vector"], evidence_vector)
    paths["evidence_report"].write_text(report, encoding="utf-8")
    _write_csv(paths["bbox_stats"], records["bbox_rows"])
    _write_csv(paths["frequency_stats"], records["frequency_rows"])
    _write_csv(paths["class_stats"], dataset_stats["gt"]["class_stats_rows"])
    _write_figures(figures_dir, records, dataset_stats)
    return paths


def _collect_statistics(
    config: EvidenceAuditConfig,
    image_paths: list[Path],
    labels_dir: Path,
) -> dict[str, Any]:
    image_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    class_image_keys: dict[int, set[str]] = defaultdict(set)
    global_hist = np.zeros(256, dtype=np.int64)
    sha = hashlib.sha256()

    for image_index, image_path in enumerate(image_paths, start=1):
        sha.update(image_path.name.encode("utf-8"))
        with Image.open(image_path) as img:
            gray = ImageOps.grayscale(img)
            arr_u8 = np.asarray(gray, dtype=np.uint8)
        height, width = arr_u8.shape
        hist = np.bincount(arr_u8.ravel(), minlength=256)
        global_hist += hist
        arr = arr_u8.astype(np.float32) / 255.0
        image_min = int(arr_u8.min())
        image_max = int(arr_u8.max())
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = _read_yolo_boxes(label_path, width, height)
        image_rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "resolution": width * height,
                "aspect_ratio": width / height if height else math.nan,
                "gray_mean": float(arr.mean()),
                "gray_std": float(arr.std()),
                "gray_min": image_min,
                "gray_max": image_max,
                "dynamic_range": image_max - image_min,
                "bbox_count": len(boxes),
            }
        )
        frequency_rows.append(_frequency_row(image_path.stem, arr, config))
        bbox_rows.extend(_bbox_rows(image_path.stem, arr, boxes, width, height, config, class_image_keys))

    return {
        "image_rows": image_rows,
        "bbox_rows": bbox_rows,
        "frequency_rows": frequency_rows,
        "class_image_keys": class_image_keys,
        "global_histogram": global_hist,
        "input_digest_sha256": sha.hexdigest(),
    }


def _read_yolo_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed YOLO row in {label_path}:{line_number}: {line!r}")
        cls, cx, cy, bw, bh = (float(value) for value in fields)
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append(
            {
                "class_id": int(cls),
                "cx_norm": cx,
                "cy_norm": cy,
                "w_norm": bw,
                "h_norm": bh,
                "x1": max(0.0, min(float(width), x1)),
                "y1": max(0.0, min(float(height), y1)),
                "x2": max(0.0, min(float(width), x2)),
                "y2": max(0.0, min(float(height), y2)),
            }
        )
    return boxes


def _bbox_rows(
    image_id: str,
    arr: np.ndarray,
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
    config: EvidenceAuditConfig,
    class_image_keys: dict[int, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_area = float(width * height)
    for box_index, box in enumerate(boxes):
        cls = int(box["class_id"])
        class_image_keys[cls].add(image_id)
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        target, background, local_var = _local_target_stats(arr, x1, y1, x2, y2, config.local_context_scale)
        rows.append(
            {
                "image_id": image_id,
                "box_index": box_index,
                "class_id": cls,
                "class_name": config.class_names[cls] if 0 <= cls < len(config.class_names) else str(cls),
                "image_width": width,
                "image_height": height,
                "bbox_width": bw,
                "bbox_height": bh,
                "bbox_area": area,
                "bbox_width_over_image_width": bw / width if width else math.nan,
                "bbox_height_over_image_height": bh / height if height else math.nan,
                "bbox_area_over_image_area": area / image_area if image_area else math.nan,
                "bbox_aspect_ratio": bw / bh if bh > 0 else math.nan,
                "center_x_norm": box["cx_norm"],
                "center_y_norm": box["cy_norm"],
                "size_bucket_fractional": _size_bucket(area / image_area, config),
                "target_mean_intensity": target,
                "local_background_intensity": background,
                "target_background_contrast": target - background if np.isfinite(target) and np.isfinite(background) else math.nan,
                "local_variance": local_var,
            }
        )
    return rows


def _local_target_stats(
    arr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
) -> tuple[float, float, float]:
    height, width = arr.shape
    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(width, int(math.ceil(x2)))
    iy2 = min(height, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return math.nan, math.nan, math.nan
    target = arr[iy1:iy2, ix1:ix2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * scale / 2.0)
    half_h = max(1.0, (y2 - y1) * scale / 2.0)
    wx1 = max(0, int(math.floor(cx - half_w)))
    wy1 = max(0, int(math.floor(cy - half_h)))
    wx2 = min(width, int(math.ceil(cx + half_w)))
    wy2 = min(height, int(math.ceil(cy + half_h)))
    local = arr[wy1:wy2, wx1:wx2]
    mask = np.ones(local.shape, dtype=bool)
    bx1 = max(0, ix1 - wx1)
    by1 = max(0, iy1 - wy1)
    bx2 = min(local.shape[1], ix2 - wx1)
    by2 = min(local.shape[0], iy2 - wy1)
    mask[by1:by2, bx1:bx2] = False
    background = local[mask]
    return (
        float(target.mean()),
        float(background.mean()) if background.size else math.nan,
        float(local.var()) if local.size else math.nan,
    )


def _frequency_row(image_id: str, arr: np.ndarray, config: EvidenceAuditConfig) -> dict[str, Any]:
    image = Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)))
    image = image.resize((config.fft_size, config.fft_size), Image.Resampling.BILINEAR)
    small = np.asarray(image, dtype=np.float32) / 255.0
    small = small - float(small.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(small))) ** 2
    yy, xx = np.indices(power.shape)
    center = (np.array(power.shape) - 1.0) / 2.0
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    radius = radius / max(float(radius.max()), 1.0)
    total = float(power.sum())
    low = float(power[radius <= 0.15].sum()) / total if total else math.nan
    mid = float(power[(radius > 0.15) & (radius <= 0.35)].sum()) / total if total else math.nan
    high = float(power[radius > 0.35].sum()) / total if total else math.nan
    centroid = float((power * radius).sum() / total) if total else math.nan
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "image_id": image_id,
        "fft_size": config.fft_size,
        "psd_low_band_ratio_r_le_0p15": low,
        "psd_mid_band_ratio_0p15_0p35": mid,
        "psd_high_band_ratio_r_gt_0p35": high,
        "high_frequency_energy_ratio": high,
        "psd_radial_centroid": centroid,
        "edge_density_grad_gt_threshold": float(np.mean(grad > config.edge_gradient_threshold)),
        "edge_gradient_threshold": config.edge_gradient_threshold,
    }


def _build_dataset_stats(config: EvidenceAuditConfig, pairing_audit: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    image_rows = records["image_rows"]
    bbox_rows = records["bbox_rows"]
    frequency_rows = records["frequency_rows"]
    hist = records["global_histogram"]
    class_box_counts = Counter(int(row["class_id"]) for row in bbox_rows)
    class_stats_rows = []
    for class_id in range(len(config.class_names)):
        class_stats_rows.append(
            {
                "class_id": class_id,
                "class_name": config.class_names[class_id],
                "box_count": int(class_box_counts.get(class_id, 0)),
                "image_count": len(records["class_image_keys"].get(class_id, set())),
            }
        )
    box_counts = [row["bbox_count"] for row in image_rows]
    center_heatmap = _center_heatmap(bbox_rows)
    return {
        "artifact_type": "sonar_dataset_evidence_audit",
        "schema_version": "sonar_dataset_evidence_audit.v1",
        "generated_at": utc_now(),
        "dataset_id": config.dataset_id,
        "source_modality": config.source_modality,
        "source_task": config.source_task,
        "split_policy": {
            "split_used": config.split,
            "allowed_splits": list(config.allowed_splits),
            "forbidden_splits_not_used": list(config.forbidden_splits),
            "train_only_for_search_space_definition": config.split == "train",
        },
        "input": {
            "dataset_root": str(config.dataset_root),
            "data_yaml": str(config.data_yaml) if config.data_yaml else None,
            "image_count": len(image_rows),
            "box_count": len(bbox_rows),
            "input_digest_sha256": records["input_digest_sha256"],
        },
        "pairing_audit": pairing_audit,
        "image": {
            "resolution": _summary([row["resolution"] for row in image_rows]),
            "width": _summary([row["width"] for row in image_rows]),
            "height": _summary([row["height"] for row in image_rows]),
            "aspect_ratio": _summary([row["aspect_ratio"] for row in image_rows]),
            "gray_mean_per_image": _summary([row["gray_mean"] for row in image_rows]),
            "gray_std_per_image": _summary([row["gray_std"] for row in image_rows]),
            "global_gray_mean": _hist_mean(hist),
            "global_gray_std": _hist_std(hist),
            "global_gray_percentiles_uint8": _hist_percentiles(hist),
            "dynamic_range_per_image_uint8": _summary([row["dynamic_range"] for row in image_rows]),
            "entropy_bits_global": _hist_entropy(hist),
            "histogram_uint8": hist.astype(int).tolist(),
            "highlight_shadow_proxy": _highlight_shadow_proxy(hist),
        },
        "gt": {
            "class_distribution": {
                config.class_names[class_id]: int(class_box_counts.get(class_id, 0))
                for class_id in range(len(config.class_names))
            },
            "gt_per_image": _summary(box_counts),
            "bbox_width_over_image_width": _summary([row["bbox_width_over_image_width"] for row in bbox_rows]),
            "bbox_height_over_image_height": _summary([row["bbox_height_over_image_height"] for row in bbox_rows]),
            "bbox_area_over_image_area": _summary([row["bbox_area_over_image_area"] for row in bbox_rows]),
            "bbox_aspect_ratio": _summary([row["bbox_aspect_ratio"] for row in bbox_rows]),
            "center_x_norm": _summary([row["center_x_norm"] for row in bbox_rows]),
            "center_y_norm": _summary([row["center_y_norm"] for row in bbox_rows]),
            "center_distribution_heatmap_10x10": center_heatmap,
            "small_medium_large_ratio_fractional_area": _bucket_ratios(bbox_rows),
            "small_medium_large_definition": {
                "small": f"bbox_area/image_area < {config.small_area_fraction}",
                "medium": f"{config.small_area_fraction} <= bbox_area/image_area < {config.medium_area_fraction}",
                "large": f"bbox_area/image_area >= {config.medium_area_fraction}",
            },
            "class_stats_rows": class_stats_rows,
        },
        "local_target": {
            "target_mean_intensity": _summary([row["target_mean_intensity"] for row in bbox_rows]),
            "local_background_intensity": _summary([row["local_background_intensity"] for row in bbox_rows]),
            "target_background_contrast": _summary([row["target_background_contrast"] for row in bbox_rows]),
            "local_variance": _summary([row["local_variance"] for row in bbox_rows]),
            "background_definition": f"ring inside {config.local_context_scale}x bbox-centered context window excluding bbox pixels",
        },
        "frequency": {
            "high_frequency_energy_ratio": _summary([row["high_frequency_energy_ratio"] for row in frequency_rows]),
            "edge_density": _summary([row["edge_density_grad_gt_threshold"] for row in frequency_rows]),
            "psd_low_band_ratio": _summary([row["psd_low_band_ratio_r_le_0p15"] for row in frequency_rows]),
            "psd_mid_band_ratio": _summary([row["psd_mid_band_ratio_0p15_0p35"] for row in frequency_rows]),
            "psd_high_band_ratio": _summary([row["psd_high_band_ratio_r_gt_0p35"] for row in frequency_rows]),
            "psd_radial_centroid": _summary([row["psd_radial_centroid"] for row in frequency_rows]),
            "psd_definition": f"2D FFT over deterministic {config.fft_size}x{config.fft_size} grayscale resize; bands use normalized radial frequency",
        },
    }


def _build_evidence_vector(config: EvidenceAuditConfig, stats: dict[str, Any]) -> dict[str, Any]:
    image = stats["image"]
    gt = stats["gt"]
    local = stats["local_target"]
    freq = stats["frequency"]
    return {
        "artifact_type": "sonar_dataset_evidence_vector",
        "schema_version": "sonar_dataset_evidence_vector.v1",
        "generated_at": stats["generated_at"],
        "dataset_id": config.dataset_id,
        "split_used": config.split,
        "train_only": config.split == "train",
        "forbidden_splits_not_used": list(config.forbidden_splits),
        "source_modality": config.source_modality,
        "source_task": config.source_task,
        "n_images": stats["input"]["image_count"],
        "n_boxes": stats["input"]["box_count"],
        "class_count": len(config.class_names),
        "image_resolution_median": image["resolution"]["p50"],
        "image_aspect_ratio_median": image["aspect_ratio"]["p50"],
        "global_gray_mean_uint8": image["global_gray_mean"],
        "global_gray_std_uint8": image["global_gray_std"],
        "entropy_bits_global": image["entropy_bits_global"],
        "bbox_per_image_mean": gt["gt_per_image"]["mean"],
        "bbox_area_fraction_median": gt["bbox_area_over_image_area"]["p50"],
        "bbox_width_fraction_median": gt["bbox_width_over_image_width"]["p50"],
        "bbox_height_fraction_median": gt["bbox_height_over_image_height"]["p50"],
        "bbox_aspect_ratio_median": gt["bbox_aspect_ratio"]["p50"],
        "small_box_fraction": gt["small_medium_large_ratio_fractional_area"].get("small", 0.0),
        "medium_box_fraction": gt["small_medium_large_ratio_fractional_area"].get("medium", 0.0),
        "large_box_fraction": gt["small_medium_large_ratio_fractional_area"].get("large", 0.0),
        "target_background_contrast_median": local["target_background_contrast"]["p50"],
        "local_variance_median": local["local_variance"]["p50"],
        "high_frequency_energy_ratio_median": freq["high_frequency_energy_ratio"]["p50"],
        "edge_density_median": freq["edge_density"]["p50"],
        "psd_radial_centroid_median": freq["psd_radial_centroid"]["p50"],
        "highlight_proxy_ratio_ge_245": image["highlight_shadow_proxy"]["highlight_proxy_ratio_ge_245"],
        "dark_tail_proxy_ratio_le_10": image["highlight_shadow_proxy"]["dark_tail_proxy_ratio_le_10"],
        "shadow_metric_status": image["highlight_shadow_proxy"]["shadow_metric_status"],
        "architecture_search_executed": False,
        "model_training_executed": False,
    }


def _render_report(config: EvidenceAuditConfig, stats: dict[str, Any], vector: dict[str, Any]) -> str:
    class_counts = stats["gt"]["class_distribution"]
    top_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    comparison = _comparison_notes(config, vector)
    return "\n".join(
        [
            f"# {config.dataset_id} Train-Only Sonar Evidence Audit",
            "",
            f"- Generated at: {stats['generated_at']}",
            f"- Split used: `{config.split}` only",
            f"- Forbidden splits not used: {', '.join(config.forbidden_splits)}",
            f"- Images: {stats['input']['image_count']}",
            f"- Boxes: {stats['input']['box_count']}",
            f"- Source modality: {config.source_modality}",
            "",
            "## Scope Boundary",
            "",
            "This audit uses only the train split to describe data properties that may inform a NAS search space. Validation and test splits were not read for these statistics, and no model training or architecture search was executed.",
            "",
            "Acoustic shadow was not automatically segmented. The report exposes reproducible fixed-threshold intensity-tail proxies instead: near-highlight pixels are gray >= 245 and dark-tail pixels are gray <= 10 in 8-bit grayscale. These proxies must not be described as acoustic-shadow segmentation metrics.",
            "",
            "## Image Statistics",
            "",
            f"- Resolution median: {vector['image_resolution_median']:.3f} pixels",
            f"- Aspect-ratio median: {vector['image_aspect_ratio_median']:.6f}",
            f"- Global gray mean/std: {vector['global_gray_mean_uint8']:.3f} / {vector['global_gray_std_uint8']:.3f} uint8",
            f"- Entropy: {vector['entropy_bits_global']:.6f} bits",
            f"- Dynamic range median: {stats['image']['dynamic_range_per_image_uint8']['p50']:.3f} uint8",
            "",
            "## Ground Truth Statistics",
            "",
            f"- GT/image mean: {vector['bbox_per_image_mean']:.6f}",
            f"- BBox width/image-width median: {vector['bbox_width_fraction_median']:.6f}",
            f"- BBox height/image-height median: {vector['bbox_height_fraction_median']:.6f}",
            f"- BBox area/image-area median: {vector['bbox_area_fraction_median']:.6f}",
            f"- Small/medium/large fractions: {vector['small_box_fraction']:.6f} / {vector['medium_box_fraction']:.6f} / {vector['large_box_fraction']:.6f}",
            f"- Top classes by box count: {', '.join(f'{name}={count}' for name, count in top_classes)}",
            "",
            "## Local Target Statistics",
            "",
            f"- Target-background contrast median: {vector['target_background_contrast_median']:.6f}",
            f"- Local variance median: {vector['local_variance_median']:.6f}",
            f"- Background proxy definition: {stats['local_target']['background_definition']}",
            "",
            "## Spatial Frequency",
            "",
            f"- High-frequency energy ratio median: {vector['high_frequency_energy_ratio_median']:.6f}",
            f"- Edge density median: {vector['edge_density_median']:.6f}",
            f"- PSD radial centroid median: {vector['psd_radial_centroid_median']:.6f}",
            f"- PSD definition: {stats['frequency']['psd_definition']}",
            "",
            "## Cylinder2/SIMD Comparison Boundary",
            "",
            comparison,
            "",
            "## Artifact Index",
            "",
            "- `dataset_stats.json`: full aggregate statistics and definitions",
            "- `bbox_stats.csv`: one row per train GT box",
            "- `class_stats.csv`: train class box/image counts",
            "- `frequency_stats.csv`: one row per train image",
            "- `evidence_vector.json`: compact fixed-schema vector",
            "- `figures/`: train-only diagnostic plots",
            "",
        ]
    )


def _comparison_notes(config: EvidenceAuditConfig, vector: dict[str, Any]) -> str:
    notes = [
        "Only UATD train was analyzed in this run. The comparison below uses existing project-documented Cylinder2/SIMD inventory or anchor artifacts when present; it does not inspect UATD validation/test data.",
        "",
        "- Clearly different from Cylinder2/SIMD at dataset-design level: UATD is MFLS with 10 classes and 7,600 train images, while the project documentation records Cylinder2/Roboflow as a small three-label detection source and SIMD/Figshare as side-scan mine/NOMBO-MILCO data with many empty YOLO label files.",
        "- Clearly different from the existing Cylinder2 train-anchor artifact: UATD train median box fractions are reported in `evidence_vector.json`; Cylinder2's train-only anchor artifact records very small 640-space anchors around 9.25-21.25 px wide and 13.5-27.5 px high. Treat this as a label-geometry contrast, not a validation/test result.",
        "- Likely shared sonar properties: low-texture gray imagery, sparse local targets, background interference risk, and non-natural-image frequency structure. These remain descriptive train-set evidence, not proof of operator benefit.",
        "- UATD/MFLS-only properties: tall variable image resolutions, Gemini 1200ik MFLS provenance, 10-class target taxonomy, and the target/background intensity and frequency summaries produced here.",
    ]
    artifacts = config.comparison_artifacts or {}
    cylinder_anchor = artifacts.get("cylinder2_train_anchor_summary")
    if cylinder_anchor:
        notes.append(f"- Cylinder2 train-anchor artifact used for boundary wording: `{cylinder_anchor}`.")
    return "\n".join(notes)


def _size_bucket(area_fraction: float, config: EvidenceAuditConfig) -> str:
    if area_fraction < config.small_area_fraction:
        return "small"
    if area_fraction < config.medium_area_fraction:
        return "medium"
    return "large"


def _bucket_ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(row["size_bucket_fractional"] for row in rows)
    total = sum(counts.values())
    return {name: counts.get(name, 0) / total if total else 0.0 for name in ("small", "medium", "large")}


def _center_heatmap(rows: list[dict[str, Any]], bins: int = 10) -> list[list[int]]:
    heatmap = np.zeros((bins, bins), dtype=int)
    for row in rows:
        x = min(bins - 1, max(0, int(float(row["center_x_norm"]) * bins)))
        y = min(bins - 1, max(0, int(float(row["center_y_norm"]) * bins)))
        heatmap[y, x] += 1
    return heatmap.tolist()


def _summary(values: Iterable[Any]) -> dict[str, float | int]:
    arr = np.asarray([float(value) for value in values if _finite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        result[f"p{pct}"] = float(np.percentile(arr, pct))
    return result


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _hist_mean(hist: np.ndarray) -> float:
    total = int(hist.sum())
    return float(np.dot(np.arange(256), hist) / total) if total else math.nan


def _hist_std(hist: np.ndarray) -> float:
    total = int(hist.sum())
    if total == 0:
        return math.nan
    values = np.arange(256)
    mean = _hist_mean(hist)
    second = float(np.dot(values * values, hist) / total)
    return float(math.sqrt(max(0.0, second - mean * mean)))


def _hist_percentiles(hist: np.ndarray) -> dict[str, int]:
    total = int(hist.sum())
    if total == 0:
        return {}
    cumulative = np.cumsum(hist)
    result = {}
    for pct in PERCENTILES:
        index = int(math.ceil((pct / 100.0) * total))
        index = max(1, min(total, index))
        result[f"p{pct}"] = int(np.searchsorted(cumulative, index, side="left"))
    return result


def _hist_entropy(hist: np.ndarray) -> float:
    total = float(hist.sum())
    if total == 0:
        return math.nan
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _highlight_shadow_proxy(hist: np.ndarray) -> dict[str, Any]:
    total = float(hist.sum())
    if total == 0:
        return {
            "shadow_metric_status": "not_computed_no_pixels",
            "highlight_proxy_definition": "gray >= 245",
            "dark_tail_proxy_definition": "gray <= 10",
        }
    return {
        "shadow_metric_status": "not_computed_no_reliable_automatic_acoustic_shadow_segmentation",
        "highlight_proxy_definition": "gray >= 245 in 8-bit grayscale",
        "dark_tail_proxy_definition": "gray <= 10 in 8-bit grayscale; intensity-tail proxy only, not acoustic-shadow segmentation",
        "highlight_proxy_ratio_ge_245": float(hist[245:].sum() / total),
        "dark_tail_proxy_ratio_le_10": float(hist[:11].sum() / total),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(figures_dir: Path, records: dict[str, Any], stats: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (figures_dir / "FIGURES_NOT_RENDERED.txt").write_text(
            "matplotlib is not installed; JSON/CSV evidence was still generated.\n",
            encoding="utf-8",
        )
        return

    bbox_rows = records["bbox_rows"]
    freq_rows = records["frequency_rows"]
    hist = np.asarray(stats["image"]["histogram_uint8"], dtype=np.int64)

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(256), hist, width=1.0, color="#4C78A8")
    plt.xlabel("gray value")
    plt.ylabel("pixel count")
    plt.tight_layout()
    plt.savefig(figures_dir / "gray_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    class_rows = stats["gt"]["class_stats_rows"]
    plt.bar([row["class_name"] for row in class_rows], [row["box_count"] for row in class_rows], color="#59A14F")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("train boxes")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_box_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist([row["bbox_area_over_image_area"] for row in bbox_rows], bins=60, color="#F28E2B")
    plt.xlabel("bbox area / image area")
    plt.ylabel("box count")
    plt.tight_layout()
    plt.savefig(figures_dir / "bbox_area_fraction_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(np.asarray(stats["gt"]["center_distribution_heatmap_10x10"]), cmap="viridis", origin="lower")
    plt.colorbar(label="box centers")
    plt.xlabel("center x bin")
    plt.ylabel("center y bin")
    plt.tight_layout()
    plt.savefig(figures_dir / "bbox_center_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist([row["high_frequency_energy_ratio"] for row in freq_rows], bins=60, color="#B07AA1")
    plt.xlabel("high-frequency energy ratio")
    plt.ylabel("image count")
    plt.tight_layout()
    plt.savefig(figures_dir / "high_frequency_ratio_histogram.png", dpi=160)
    plt.close()
