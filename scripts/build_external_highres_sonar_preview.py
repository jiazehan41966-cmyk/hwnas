#!/usr/bin/env python3
"""Build a high-resolution external sonar qualitative layout preview.

The preview uses local Figshare/Roboflow source images to demonstrate the
cleaner visual layout possible with original-resolution sonar data. It does not
mix these images with NKSID frozen-protocol predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


DEFAULT_ROOTS = [
    Path("data/external/roboflow_cylider2/source_reconstruction/v6"),
    Path("data/external/figshare_mine_detection_v2/extracted"),
]

DISCLAIMER = (
    "High-resolution external-source layout preview only. It is not NKSID frozen-protocol "
    "classification evidence and does not replace macro_f1, top1, latency, LUT, DSP, BRAM, or power."
)


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def saliency(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(normalize(arr))
    mag = np.hypot(gx, gy)
    lo, hi = np.percentile(mag, [60.0, 99.7])
    if hi <= lo:
        return np.zeros_like(mag)
    return np.clip((mag - lo) / (hi - lo), 0.0, 1.0)


def integral_mean(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    padded = np.pad(arr, ((h // 2, h // 2), (w // 2, w // 2)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return (integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]) / float(h * w)


def select_roi(arr: np.ndarray, frac: float) -> tuple[int, int, int, int]:
    height, width = arr.shape
    roi_h = max(96, int(round(height * frac)))
    roi_w = max(96, int(round(width * frac)))
    roi_h = min(roi_h, height)
    roi_w = min(roi_w, width)
    score = integral_mean(saliency(arr), max(9, roi_h // 4), max(9, roi_w // 4))
    margin_y = roi_h // 2
    margin_x = roi_w // 2
    if score.shape[0] > 2 * margin_y and score.shape[1] > 2 * margin_x:
        score[:margin_y, :] = -1
        score[-margin_y:, :] = -1
        score[:, :margin_x] = -1
        score[:, -margin_x:] = -1
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    x0 = int(np.clip(cx - roi_w // 2, 0, width - roi_w))
    y0 = int(np.clip(cy - roi_h // 2, 0, height - roi_h))
    return x0, y0, roi_w, roi_h


def crop(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = roi
    return arr[y0 : y0 + h, x0 : x0 + w]


def image_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files.extend(root.rglob(suffix))
    return files


def pick_images(roots: list[Path], limit: int) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in image_files(roots):
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        if width < 512 or height < 512:
            continue
        candidates.append((width * height, path))
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in candidates[:limit]]


def imshow(ax: plt.Axes, arr: np.ndarray) -> None:
    ax.imshow(normalize(arr), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#0B2F6B")


def dataset_label(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "roboflow_cylider2" in text:
        return "Roboflow source"
    if "figshare_mine_detection_v2" in text:
        return "Figshare source"
    return "External source"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build high-resolution external sonar preview")
    parser.add_argument("--output", type=Path, default=Path("results/qualitative_sonar_panel_external_highres"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--crop-frac", type=float, default=0.22)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    selected = pick_images(DEFAULT_ROOTS, args.num_samples)
    if not selected:
        raise RuntimeError("No high-resolution external sonar images were found")

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(8.2, 1.95 * len(selected) + 0.65),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.7, 0.92, 0.92, 0.92]},
        constrained_layout=True,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.08, 1.0, 0.94))
    for col, title in enumerate(["Original", "ROI", "Saliency", "Plain ROI"]):
        axes[0, col].set_title(title, fontsize=8, fontweight="bold")

    rows: list[dict[str, str]] = []
    for row_index, path in enumerate(selected):
        arr = load_gray(path)
        roi = select_roi(arr, args.crop_frac)
        roi_arr = crop(arr, roi)
        x0, y0, w, h = roi

        imshow(axes[row_index, 0], arr)
        axes[row_index, 0].add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="#D55E00", linewidth=1.1))
        axes[row_index, 0].text(
            0.01,
            0.99,
            dataset_label(path),
            transform=axes[row_index, 0].transAxes,
            va="top",
            ha="left",
            fontsize=6.2,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.42, "pad": 1.4, "linewidth": 0},
        )
        imshow(axes[row_index, 1], roi_arr)
        imshow(axes[row_index, 2], roi_arr)
        sal = saliency(roi_arr)
        axes[row_index, 2].imshow(sal, cmap="magma", alpha=sal * 0.42, interpolation="bilinear")
        axes[row_index, 2].text(
            0.02,
            0.98,
            "texture saliency",
            transform=axes[row_index, 2].transAxes,
            va="top",
            ha="left",
            fontsize=5.8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.38, "pad": 1.2, "linewidth": 0},
        )
        imshow(axes[row_index, 3], roi_arr)
        axes[row_index, 3].text(
            0.02,
            0.98,
            "no method prediction",
            transform=axes[row_index, 3].transAxes,
            va="top",
            ha="left",
            fontsize=5.8,
            color="white",
            bbox={"facecolor": "#555555", "alpha": 0.58, "pad": 1.2, "linewidth": 0},
        )
        rows.append(
            {
                "image_path": relpath(path),
                "image_sha256": sha256_file(path),
                "image_width": str(arr.shape[1]),
                "image_height": str(arr.shape[0]),
                "roi_x": str(x0),
                "roi_y": str(y0),
                "roi_w": str(w),
                "roi_h": str(h),
                "source": dataset_label(path),
            }
        )

    fig.suptitle("High-resolution External Sonar Layout Preview", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.018, DISCLAIMER, fontsize=7, color="#333333", ha="left", va="bottom")
    png = args.output / "external_highres_sonar_preview.png"
    pdf = args.output / "external_highres_sonar_preview.pdf"
    svg = args.output / "external_highres_sonar_preview.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    source_csv = args.output / "source_selection.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "script": "scripts/build_external_highres_sonar_preview.py",
        "outputs": {
            "png": relpath(png),
            "pdf": relpath(pdf),
            "svg": relpath(svg),
            "source_selection_csv": relpath(source_csv),
        },
        "claim_boundary": DISCLAIMER,
        "selection_policy": "largest local external-source images with width and height at least 512 px",
        "method_predictions": "not included; this is a layout and source-quality preview",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    caption_path = args.output / "caption.md"
    caption_path.write_text(
        "# Caption\n\n"
        "High-resolution external-source sonar layout preview using local Roboflow/Figshare images. "
        "The panel demonstrates cleaner original/ROI/saliency rendering when source images are not the "
        "low-resolution NKSID thumbnails. It does not contain NKSID frozen-protocol method predictions "
        "and must not be used as formal classification evidence.\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote {relpath(png)}")
    print(f"[OK] Wrote {relpath(pdf)}")
    print(f"[OK] Wrote {relpath(svg)}")
    print(f"[OK] Wrote {relpath(source_csv)}")
    print(f"[OK] Wrote {relpath(manifest_path)}")
    print(f"[OK] Wrote {relpath(caption_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
