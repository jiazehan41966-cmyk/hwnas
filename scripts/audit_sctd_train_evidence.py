from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "SCTD"
PROCESSED_ROOT = DATASET_ROOT / "processed"
TRAIN_IMAGES = PROCESSED_ROOT / "train" / "images"
TRAIN_LABELS = PROCESSED_ROOT / "train" / "labels"
METADATA_DIR = DATASET_ROOT / "metadata"
DATASET_CARD = METADATA_DIR / "dataset_card.md"
MANIFEST = METADATA_DIR / "dataset_manifest.json"
CLASS_MAPPING = METADATA_DIR / "class_mapping.json"
OUT = ROOT / "results" / "evidence" / "SCTD"
FIGURES = OUT / "figures"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ["min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std"]}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def entropy_u8(gray_u8: np.ndarray) -> float:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def resize_for_metrics(gray: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return gray.astype(np.float32) / 255.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(gray, mode="L").resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def box_mean(arr: np.ndarray, radius: int = 3) -> np.ndarray:
    padded = np.pad(arr, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    total = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return total / float(k * k)


def local_variance_mean(gray01: np.ndarray) -> float:
    mu = box_mean(gray01, radius=3)
    mu2 = box_mean(gray01 * gray01, radius=3)
    return float(np.mean(np.maximum(mu2 - mu * mu, 0.0)))


def sobel_edge_density(gray01: np.ndarray) -> tuple[float, float]:
    padded = np.pad(gray01, 1, mode="reflect")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy) / 4.0
    return float(np.mean(mag > 0.10)), float(np.mean(mag))


def psd_summary(gray01: np.ndarray) -> dict[str, float]:
    arr = gray01 - float(np.mean(gray01))
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return {
            "psd_low_ratio": 0.0,
            "psd_mid_ratio": 0.0,
            "psd_high_ratio": 0.0,
            "high_frequency_energy": 0.0,
        }
    h, w = arr.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt(((yy - cy) / max(h, 1)) ** 2 + ((xx - cx) / max(w, 1)) ** 2)
    low = float(np.sum(power[rr < 0.10]) / total)
    mid = float(np.sum(power[(rr >= 0.10) & (rr < 0.25)]) / total)
    high = float(np.sum(power[rr >= 0.25]) / total)
    return {
        "psd_low_ratio": low,
        "psd_mid_ratio": mid,
        "psd_high_ratio": high,
        "high_frequency_energy": high,
    }


def parse_label(path: Path) -> list[dict[str, float | int]]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label {path}:{line_no}: {line!r}")
        cls, cx, cy, bw, bh = parts
        rows.append(
            {
                "class_id": int(cls),
                "cx": float(cx),
                "cy": float(cy),
                "w_norm": float(bw),
                "h_norm": float(bh),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_hist(path: Path, values: list[float], title: str, xlabel: str, bins: int = 32) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    dataset_card_text = DATASET_CARD.read_text(encoding="utf-8")
    manifest = read_json(MANIFEST)
    class_mapping = read_json(CLASS_MAPPING)
    classes = {int(row["id"]): row["name"] for row in class_mapping["classes"]}

    image_paths = sorted(p for p in TRAIN_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f"No train images found under {TRAIN_IMAGES}")

    image_rows: list[dict] = []
    bbox_rows: list[dict] = []
    frequency_rows: list[dict] = []
    class_image_sets: dict[int, set[str]] = defaultdict(set)
    class_counts: Counter[int] = Counter()
    invalid_boxes: list[dict] = []
    missing_labels: list[str] = []

    for image_path in image_paths:
        image_id = image_path.stem
        label_path = TRAIN_LABELS / f"{image_id}.txt"
        if not label_path.exists():
            missing_labels.append(image_id)
        with Image.open(image_path) as img:
            gray_img = img.convert("L")
            width, height = img.size
            gray_u8 = np.asarray(gray_img, dtype=np.uint8)

        gray01 = resize_for_metrics(gray_u8)
        p01, p05, p50, p95, p99 = np.percentile(gray_u8, [1, 5, 50, 95, 99])
        contrast_std = float(np.std(gray_u8.astype(np.float32) / 255.0))
        michelson = float((p99 - p01) / max(p99 + p01, 1e-9))
        ent = entropy_u8(gray_u8)
        local_var = local_variance_mean(gray01)
        edge_density, edge_strength = sobel_edge_density(gray01)
        psd = psd_summary(gray01)

        labels = parse_label(label_path)
        image_rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "pixels": width * height,
                "objects": len(labels),
                "intensity_mean": float(np.mean(gray_u8) / 255.0),
                "intensity_std": contrast_std,
                "intensity_p05": float(p05 / 255.0),
                "intensity_median": float(p50 / 255.0),
                "intensity_p95": float(p95 / 255.0),
                "contrast_std": contrast_std,
                "contrast_michelson_p01_p99": michelson,
                "entropy_bits": ent,
                "local_variance_mean": local_var,
                "edge_density": edge_density,
                "edge_strength_mean": edge_strength,
                **psd,
            }
        )
        frequency_rows.append(
            {
                "image_id": image_id,
                "edge_density": edge_density,
                "edge_strength_mean": edge_strength,
                "high_frequency_energy": psd["high_frequency_energy"],
                "psd_low_ratio": psd["psd_low_ratio"],
                "psd_mid_ratio": psd["psd_mid_ratio"],
                "psd_high_ratio": psd["psd_high_ratio"],
            }
        )

        for idx, box in enumerate(labels):
            cls = int(box["class_id"])
            cx = float(box["cx"])
            cy = float(box["cy"])
            bw = float(box["w_norm"])
            bh = float(box["h_norm"])
            valid = 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
            if not valid:
                invalid_boxes.append({"image_id": image_id, "box_index": idx, **box})
            w_px = bw * width
            h_px = bh * height
            area_norm = bw * bh
            aspect = bw / bh if bh > 0 else math.nan
            class_counts[cls] += 1
            class_image_sets[cls].add(image_id)
            bbox_rows.append(
                {
                    "image_id": image_id,
                    "box_index": idx,
                    "class_id": cls,
                    "class_name": classes.get(cls, f"class_{cls}"),
                    "cx_norm": cx,
                    "cy_norm": cy,
                    "w_norm": bw,
                    "h_norm": bh,
                    "area_norm": area_norm,
                    "w_px": w_px,
                    "h_px": h_px,
                    "area_px": w_px * h_px,
                    "aspect_ratio_w_over_h": aspect,
                    "center_x_bin_4": min(3, max(0, int(cx * 4))),
                    "center_y_bin_4": min(3, max(0, int(cy * 4))),
                }
            )

    class_rows = []
    total_boxes = sum(class_counts.values())
    for cls in sorted(classes):
        count = class_counts.get(cls, 0)
        class_rows.append(
            {
                "class_id": cls,
                "class_name": classes[cls],
                "object_count": count,
                "image_count": len(class_image_sets.get(cls, set())),
                "object_fraction": count / total_boxes if total_boxes else 0.0,
            }
        )

    image_widths = [float(r["width"]) for r in image_rows]
    image_heights = [float(r["height"]) for r in image_rows]
    image_pixels = [float(r["pixels"]) for r in image_rows]
    objects_per_image = [float(r["objects"]) for r in image_rows]
    bbox_w = [float(r["w_norm"]) for r in bbox_rows]
    bbox_h = [float(r["h_norm"]) for r in bbox_rows]
    bbox_area = [float(r["area_norm"]) for r in bbox_rows]
    bbox_aspect = [float(r["aspect_ratio_w_over_h"]) for r in bbox_rows]
    centers_x = [float(r["cx_norm"]) for r in bbox_rows]
    centers_y = [float(r["cy_norm"]) for r in bbox_rows]

    center_grid = np.zeros((4, 4), dtype=int)
    for row in bbox_rows:
        center_grid[int(row["center_y_bin_4"]), int(row["center_x_bin_4"])] += 1

    metric_keys = [
        "intensity_mean",
        "intensity_std",
        "contrast_std",
        "contrast_michelson_p01_p99",
        "entropy_bits",
        "local_variance_mean",
        "edge_density",
        "edge_strength_mean",
        "high_frequency_energy",
        "psd_low_ratio",
        "psd_mid_ratio",
        "psd_high_ratio",
    ]
    image_metric_summary = {key: quantiles([float(r[key]) for r in image_rows]) for key in metric_keys}
    dataset_stats = {
        "schema_version": "dataset-evidence-train-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "SCTD",
            "variant": "classic SCTD / SCTD 1.0",
            "is_sctd_3_0": False,
            "sonar_modality": "side-scan sonar",
            "version_source": "datasets/SCTD/metadata/dataset_card.md and dataset_manifest.json",
            "source_repository": manifest["source"]["repository"],
            "source_archive": manifest["source"]["raw_zip_url"],
            "source_archive_sha256": manifest["source"]["raw_zip_sha256"],
            "dataset_card_sha256": sha256_file(DATASET_CARD),
        },
        "scope": {
            "split": "train",
            "images_dir": str(TRAIN_IMAGES.relative_to(ROOT)),
            "labels_dir": str(TRAIN_LABELS.relative_to(ROOT)),
            "forbidden_splits_not_used": ["valid", "test"],
            "statistics_schema_matches": ["Cylinder2", "SIMD", "UATD"],
        },
        "counts": {
            "images": len(image_rows),
            "label_files": len(list(TRAIN_LABELS.glob("*.txt"))),
            "objects": total_boxes,
            "classes": len(classes),
            "missing_label_images": missing_labels,
            "invalid_boxes": invalid_boxes,
        },
        "image_statistics": {
            "width": quantiles(image_widths),
            "height": quantiles(image_heights),
            "pixels": quantiles(image_pixels),
            "objects_per_image": quantiles(objects_per_image),
            "metric_summary": image_metric_summary,
        },
        "gt_scale_statistics": {
            "bbox_width_norm": quantiles(bbox_w),
            "bbox_height_norm": quantiles(bbox_h),
            "bbox_area_norm": quantiles(bbox_area),
            "bbox_width_px": quantiles([float(r["w_px"]) for r in bbox_rows]),
            "bbox_height_px": quantiles([float(r["h_px"]) for r in bbox_rows]),
            "bbox_area_px": quantiles([float(r["area_px"]) for r in bbox_rows]),
        },
        "class_distribution": class_rows,
        "bbox_aspect_ratio": quantiles(bbox_aspect),
        "center_distribution": {
            "cx_norm": quantiles(centers_x),
            "cy_norm": quantiles(centers_y),
            "grid_4x4_yx": center_grid.tolist(),
        },
        "contrast": image_metric_summary["contrast_std"],
        "entropy": image_metric_summary["entropy_bits"],
        "local_variance": image_metric_summary["local_variance_mean"],
        "high_frequency_energy": image_metric_summary["high_frequency_energy"],
        "edge_density": image_metric_summary["edge_density"],
        "psd_summary": {
            "low_ratio": image_metric_summary["psd_low_ratio"],
            "mid_ratio": image_metric_summary["psd_mid_ratio"],
            "high_ratio": image_metric_summary["psd_high_ratio"],
        },
        "acoustic_statistics_status": {
            "included": False,
            "reason": "No calibrated sonar geometry, shadow/highlight labels, or reliable acoustic proxy was available from the train-only files; intensity tails were not promoted to acoustic statistics.",
        },
        "provenance": {
            "dataset_card_first_read_path": str(DATASET_CARD.relative_to(ROOT)),
            "dataset_card_contains": {
                "classic_sctd": "classic SCTD / SCTD 1.0" in dataset_card_text,
                "not_sctd_3_0": "not SCTD 3.0" in dataset_card_text,
                "side_scan_sonar": "side-scan sonar" in dataset_card_text,
            },
        },
    }

    write_csv(
        OUT / "bbox_stats.csv",
        bbox_rows,
        [
            "image_id",
            "box_index",
            "class_id",
            "class_name",
            "cx_norm",
            "cy_norm",
            "w_norm",
            "h_norm",
            "area_norm",
            "w_px",
            "h_px",
            "area_px",
            "aspect_ratio_w_over_h",
            "center_x_bin_4",
            "center_y_bin_4",
        ],
    )
    write_csv(OUT / "class_stats.csv", class_rows, ["class_id", "class_name", "object_count", "image_count", "object_fraction"])
    write_csv(
        OUT / "frequency_stats.csv",
        frequency_rows,
        ["image_id", "edge_density", "edge_strength_mean", "high_frequency_energy", "psd_low_ratio", "psd_mid_ratio", "psd_high_ratio"],
    )
    (OUT / "dataset_stats.json").write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    save_hist(FIGURES / "image_width_hist.png", image_widths, "SCTD train image width", "pixels")
    save_hist(FIGURES / "image_height_hist.png", image_heights, "SCTD train image height", "pixels")
    save_hist(FIGURES / "bbox_area_norm_hist.png", bbox_area, "SCTD train bbox normalized area", "area")
    save_hist(FIGURES / "bbox_aspect_ratio_hist.png", bbox_aspect, "SCTD train bbox aspect ratio", "w / h")
    save_hist(FIGURES / "entropy_hist.png", [float(r["entropy_bits"]) for r in image_rows], "SCTD train entropy", "bits")
    save_hist(FIGURES / "high_frequency_energy_hist.png", [float(r["high_frequency_energy"]) for r in image_rows], "SCTD train high-frequency energy", "PSD ratio")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar([r["class_name"] for r in class_rows], [r["object_count"] for r in class_rows], color="#72B7B2")
    ax.set_title("SCTD train class distribution")
    ax.set_ylabel("objects")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(center_grid, cmap="viridis", origin="upper")
    ax.set_title("SCTD train bbox center grid")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    fig.colorbar(im, ax=ax, label="objects")
    fig.tight_layout()
    fig.savefig(FIGURES / "center_distribution_4x4.png", dpi=180)
    plt.close(fig)

    vector = {
        "schema_version": "dataset-evidence-vector-v1",
        "dataset": "SCTD",
        "split": "train",
        "variant": dataset_stats["dataset"]["variant"],
        "is_sctd_3_0": False,
        "sonar_modality": "side-scan sonar",
        "image_count": len(image_rows),
        "object_count": total_boxes,
        "class_count": len(classes),
        "class_counts": {classes[k]: class_counts.get(k, 0) for k in sorted(classes)},
        "median_bbox_area_norm": dataset_stats["gt_scale_statistics"]["bbox_area_norm"]["median"],
        "median_aspect_ratio_w_over_h": dataset_stats["bbox_aspect_ratio"]["median"],
        "median_entropy_bits": dataset_stats["entropy"]["median"],
        "median_contrast_std": dataset_stats["contrast"]["median"],
        "median_local_variance": dataset_stats["local_variance"]["median"],
        "median_edge_density": dataset_stats["edge_density"]["median"],
        "median_high_frequency_energy": dataset_stats["high_frequency_energy"]["median"],
        "median_psd_low_ratio": dataset_stats["psd_summary"]["low_ratio"]["median"],
        "median_psd_mid_ratio": dataset_stats["psd_summary"]["mid_ratio"]["median"],
        "median_psd_high_ratio": dataset_stats["psd_summary"]["high_ratio"]["median"],
        "acoustic_statistics_included": False,
        "comparison_inputs": {
            "Cylinder2": {
                "available": (ROOT / "artifacts" / "datasets" / "redownload_audit_20260909" / "roboflow_official_yolov8_anchor_summary_train_only.json").exists(),
                "evidence": "train-only anchor/class summary exists; full image/frequency evidence schema not found in current repo",
            },
            "SIMD": {
                "available": (ROOT / "data" / "external_redownload_audit_20260909" / "figshare_mine_detection_v2" / "dataset_manifest.json").exists(),
                "evidence": "source manifest exists; no train-only full image/frequency evidence schema found in current repo",
            },
            "UATD": {
                "available": (ROOT / "datasets" / "UATD" / "metadata" / "dataset_card.md").exists(),
                "evidence": "dataset card and train split exist; no matching evidence output was reused for SCTD statistics",
            },
        },
    }
    (OUT / "evidence_vector.json").write_text(json.dumps(vector, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(dataset_stats, vector)
    (OUT / "evidence_report.md").write_text(report, encoding="utf-8")


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def build_report(stats: dict, vector: dict) -> str:
    cls_rows = stats["class_distribution"]
    lines = [
        "# SCTD Train-Only Evidence Audit",
        "",
        "## 1. Dataset Identity And Scope",
        "",
        "- Current version: classic SCTD / SCTD 1.0, not SCTD 3.0.",
        "- Sonar modality: side-scan sonar.",
        "- Version source: `datasets/SCTD/metadata/dataset_card.md` plus `datasets/SCTD/metadata/dataset_manifest.json`.",
        "- Evidence scope: train split only. `valid` and `test` were not used for SCTD statistics, search-space definition, or model/NAS execution.",
        "- Source archive SHA256: `{}`.".format(stats["dataset"]["source_archive_sha256"]),
        "",
        "## 2. Output Files",
        "",
        "- `dataset_stats.json`: full train-only aggregate statistics.",
        "- `bbox_stats.csv`: one row per train bounding box.",
        "- `class_stats.csv`: train class counts and image coverage.",
        "- `frequency_stats.csv`: one row per train image for edge and PSD-derived statistics.",
        "- `evidence_vector.json`: compact machine-readable evidence vector.",
        "- `figures/`: distribution plots generated from train only.",
        "",
        "## 3. Train Counts",
        "",
        f"- Images: {stats['counts']['images']}",
        f"- Label files: {stats['counts']['label_files']}",
        f"- Objects: {stats['counts']['objects']}",
        f"- Classes: {stats['counts']['classes']}",
        f"- Missing label images: {len(stats['counts']['missing_label_images'])}",
        f"- Invalid boxes: {len(stats['counts']['invalid_boxes'])}",
        "",
        "## 4. Class Distribution",
        "",
        "| class | objects | images | object fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in cls_rows:
        lines.append(
            f"| {row['class_name']} | {row['object_count']} | {row['image_count']} | {fmt(row['object_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Image Statistics",
            "",
            "| metric | median | p05 | p95 | mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    image_stats = stats["image_statistics"]
    for key in ["width", "height", "pixels", "objects_per_image"]:
        q = image_stats[key]
        lines.append(f"| {key} | {fmt(q['median'])} | {fmt(q['p05'])} | {fmt(q['p95'])} | {fmt(q['mean'])} |")
    for key in [
        "intensity_mean",
        "contrast_std",
        "contrast_michelson_p01_p99",
        "entropy_bits",
        "local_variance_mean",
        "edge_density",
        "high_frequency_energy",
    ]:
        q = image_stats["metric_summary"][key]
        lines.append(f"| {key} | {fmt(q['median'])} | {fmt(q['p05'])} | {fmt(q['p95'])} | {fmt(q['mean'])} |")

    lines.extend(
        [
            "",
            "## 6. GT Scale, Aspect Ratio, And Centers",
            "",
            "| metric | median | p05 | p95 | mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, q in stats["gt_scale_statistics"].items():
        lines.append(f"| {key} | {fmt(q['median'])} | {fmt(q['p05'])} | {fmt(q['p95'])} | {fmt(q['mean'])} |")
    q = stats["bbox_aspect_ratio"]
    lines.append(f"| bbox_aspect_ratio_w_over_h | {fmt(q['median'])} | {fmt(q['p05'])} | {fmt(q['p95'])} | {fmt(q['mean'])} |")
    qx = stats["center_distribution"]["cx_norm"]
    qy = stats["center_distribution"]["cy_norm"]
    lines.append(f"| center_x_norm | {fmt(qx['median'])} | {fmt(qx['p05'])} | {fmt(qx['p95'])} | {fmt(qx['mean'])} |")
    lines.append(f"| center_y_norm | {fmt(qy['median'])} | {fmt(qy['p05'])} | {fmt(qy['p95'])} | {fmt(qy['mean'])} |")
    lines.extend(
        [
            "",
            "Center grid `4x4` is stored as `center_distribution.grid_4x4_yx` in `dataset_stats.json` and visualized in `figures/center_distribution_4x4.png`.",
            "",
            "## 7. Frequency And Texture Statistics",
            "",
            "| metric | median | p05 | p95 | mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, q in [
        ("contrast_std", stats["contrast"]),
        ("entropy_bits", stats["entropy"]),
        ("local_variance_mean", stats["local_variance"]),
        ("edge_density", stats["edge_density"]),
        ("high_frequency_energy", stats["high_frequency_energy"]),
        ("psd_low_ratio", stats["psd_summary"]["low_ratio"]),
        ("psd_mid_ratio", stats["psd_summary"]["mid_ratio"]),
        ("psd_high_ratio", stats["psd_summary"]["high_ratio"]),
    ]:
        lines.append(f"| {label} | {fmt(q['median'])} | {fmt(q['p05'])} | {fmt(q['p95'])} | {fmt(q['mean'])} |")
    lines.extend(
        [
            "",
            "## 8. Acoustic Statistics",
            "",
            "Acoustic shadow/highlight statistics were not included. The train-only files provide image intensities and boxes, but not calibrated sonar geometry, shadow/highlight labels, grazing angle, or a validated proxy. Therefore intensity-tail measurements were not promoted to acoustic statistics.",
            "",
            "## 9. Comparison Boundary",
            "",
            "The comparisons below separate shared evidence, dataset-specific evidence, and modality-specific evidence. They do not use SCTD validation/test data, do not define a search space from validation, do not run NAS, and do not modify the V0.1 architecture.",
            "",
            "### SCTD vs Cylinder2",
            "",
            "- Shared evidence: both are sonar object-detection style evidence sources with train labels and YOLO-compatible boxes available in this workspace.",
            "- Dataset-specific evidence: SCTD train has 249 images and 255 objects across aircraft/human/ship; Cylinder2 has an existing train-only anchor/class summary with 330 train boxes and classes recorded as cylinder/cylider/manta in the Roboflow source material.",
            "- Modality-specific evidence: SCTD is documented here as side-scan sonar. Cylinder2 modality is treated as external sonar evidence in this repo, but this SCTD audit did not reclassify its acoustic modality beyond existing artifacts.",
            "",
            "### SCTD vs SIMD",
            "",
            "- Shared evidence: both are side-scan sonar source families according to their local metadata/manifests; both are detection-oriented rather than NKSID classification evidence.",
            "- Dataset-specific evidence: SCTD train statistics are fully computed in this package; SIMD/Figshare has a source manifest for side-scan sonar mine detection with NOMBO/MILCO task boundaries, but no matching train-only image/frequency statistics artifact was reused here.",
            "- Modality-specific evidence: both are side-scan, so differences are more dataset/task-specific than modality-specific in this report.",
            "",
            "### SCTD vs UATD",
            "",
            "- Shared evidence: both are underwater acoustic target detection datasets with bounding-box annotations and train split material in this workspace.",
            "- Dataset-specific evidence: SCTD train is much smaller (249 images, 3 classes) than UATD train (7600 images, 10 classes per the UATD dataset card). UATD has official train/test archives but no official validation split detected.",
            "- Modality-specific evidence: SCTD is side-scan sonar; UATD is multibeam forward-looking sonar (MFLS). Treating their image statistics as interchangeable would mix modality layers.",
            "",
            "## 10. Evidence Vector",
            "",
            "Key values from `evidence_vector.json`:",
            "",
            f"- median bbox area norm: {fmt(vector['median_bbox_area_norm'])}",
            f"- median entropy bits: {fmt(vector['median_entropy_bits'])}",
            f"- median edge density: {fmt(vector['median_edge_density'])}",
            f"- median high-frequency energy: {fmt(vector['median_high_frequency_energy'])}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
