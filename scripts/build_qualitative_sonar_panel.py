#!/usr/bin/env python3
"""Build qualitative sonar image panels from archived outer predictions.

This script creates a publication-style qualitative visualization: the original
sonar image, an automatically selected ROI crop, and method-wise crop panels
annotated with the archived predicted class and confidence. The heat overlay is
an image-gradient saliency proxy, not model Grad-CAM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


DEFAULT_METHODS = {
    "Baseline": "results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2/outer_predictions_fold{fold}_seed{seed}.jsonl",
    "Pretrained": "results/protocol/g1_clean_20260711/g1_mobilenet_v2_grayscale_imagenet/outer_predictions_fold{fold}_seed{seed}.jsonl",
    "RL": "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/outer_predictions_fold{fold}_seed{seed}.jsonl",
    "Ours": "results/protocol/g5v2_adaptive_denoise_edge_v2/outer_predictions_fold{fold}_seed{seed}.jsonl",
}

DISCLAIMER = (
    "Qualitative visualization only. This panel does not replace formal macro_f1, "
    "top1, latency, LUT, DSP, BRAM, or power evidence."
)
METHOD_PANEL_OVERLAY_ALPHA = 0.0


@dataclass(frozen=True)
class MethodSource:
    label: str
    path: Path


@dataclass(frozen=True)
class PredictionRecord:
    sample_key: str
    image_path: Path
    class_name: str
    predicted_class_name: str
    confidence: float | None
    correct: bool | None
    fold: int | None
    seed: int | None
    sample_index: int | None
    raw: dict


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_path(raw_path: str) -> Path:
    text = str(raw_path).replace("\\", "/")
    marker = "/data/NKSID/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        local = Path("data") / "NKSID" / Path(suffix)
        if local.exists():
            return local
    return Path(raw_path)


def sample_key_from_path(raw_path: str) -> str:
    text = str(raw_path).replace("\\", "/")
    marker = "/data/NKSID/"
    if marker in text:
        return text.split(marker, 1)[1].lower()
    path = Path(str(raw_path))
    parts = [p.lower() for p in path.parts]
    if "nksid" in parts:
        idx = parts.index("nksid")
        return "/".join(parts[idx + 1 :])
    return str(raw_path).replace("\\", "/").lower()


def parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def load_predictions(path: Path) -> dict[str, PredictionRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    records: dict[str, PredictionRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_raw = row.get("image_path") or row.get("sample_id")
            if not image_raw:
                raise ValueError(f"{path}:{line_no} has no image_path/sample_id")
            key = sample_key_from_path(str(image_raw))
            records[key] = PredictionRecord(
                sample_key=key,
                image_path=normalize_path(str(image_raw)),
                class_name=str(row.get("class_name", row.get("target", ""))),
                predicted_class_name=str(row.get("predicted_class_name", row.get("prediction", ""))),
                confidence=parse_confidence(row.get("confidence")),
                correct=row.get("correct") if isinstance(row.get("correct"), bool) else None,
                fold=int(row["fold"]) if row.get("fold") is not None else None,
                seed=int(row["seed"]) if row.get("seed") is not None else None,
                sample_index=int(row["sample_index"]) if row.get("sample_index") is not None else None,
                raw=row,
            )
    return records


def parse_method_sources(items: list[str] | None, fold: int, seed: int) -> list[MethodSource]:
    if not items:
        return [
            MethodSource(label, Path(template.format(fold=fold, seed=seed)))
            for label, template in DEFAULT_METHODS.items()
        ]
    sources: list[MethodSource] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH for --method, got: {item}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty method label in --method: {item}")
        sources.append(MethodSource(label, Path(raw_path.strip().format(fold=fold, seed=seed))))
    return sources


def choose_samples(
    per_method: dict[str, dict[str, PredictionRecord]],
    num_samples: int,
    prefer_disagreement: bool,
) -> list[str]:
    methods = list(per_method)
    common = set(per_method[methods[0]])
    for method in methods[1:]:
        common &= set(per_method[method])
    candidates: list[tuple[int, int, str, str]] = []
    for key in common:
        reference = per_method[methods[0]][key]
        if not reference.image_path.exists():
            continue
        predictions = [per_method[method][key].predicted_class_name for method in methods]
        disagreement = len(set(predictions))
        sample_index = reference.sample_index if reference.sample_index is not None else 10**9
        if prefer_disagreement:
            candidates.append((-disagreement, sample_index, reference.class_name, key))
        else:
            candidates.append((0, sample_index, reference.class_name, key))
    candidates.sort()
    selected: list[str] = []
    seen_classes: set[str] = set()
    for _, _, class_name, key in candidates:
        if class_name in seen_classes:
            continue
        selected.append(key)
        seen_classes.add(class_name)
        if len(selected) >= num_samples:
            return selected
    for _, _, _, key in candidates:
        if key in selected:
            continue
        selected.append(key)
        if len(selected) >= num_samples:
            break
    return selected


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = ImageOps.grayscale(img)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale-compatible image: {path}")
    return arr


def normalize_image(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def gradient_saliency(arr: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(arr.astype(np.float32))
    sal = np.hypot(gx, gy)
    lo, hi = np.percentile(sal, [50.0, 99.5])
    if hi <= lo:
        return np.zeros_like(sal)
    return np.clip((sal - lo) / (hi - lo), 0.0, 1.0)


def smooth_box(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ) / float(window * window)


def select_roi(arr: np.ndarray, crop_frac: float) -> tuple[int, int, int, int]:
    h, w = arr.shape
    side_h = max(24, int(round(h * crop_frac)))
    side_w = max(24, int(round(w * crop_frac)))
    side_h = min(side_h, h)
    side_w = min(side_w, w)
    sal = gradient_saliency(normalize_image(arr))
    score = smooth_box(sal, max(3, min(side_h, side_w) // 3))
    margin_y = max(0, side_h // 2)
    margin_x = max(0, side_w // 2)
    valid = score.copy()
    if valid.shape[0] > 2 * margin_y and valid.shape[1] > 2 * margin_x:
        valid[:margin_y, :] = -1
        valid[-margin_y:, :] = -1
        valid[:, :margin_x] = -1
        valid[:, -margin_x:] = -1
    cy, cx = np.unravel_index(int(np.argmax(valid)), valid.shape)
    x0 = int(np.clip(cx - side_w // 2, 0, w - side_w))
    y0 = int(np.clip(cy - side_h // 2, 0, h - side_h))
    return x0, y0, side_w, side_h


def crop(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, rw, rh = roi
    return arr[y0 : y0 + rh, x0 : x0 + rw]


def fmt_prediction(record: PredictionRecord) -> str:
    conf = "conf n/a" if record.confidence is None else f"conf {record.confidence:.2f}"
    status = ""
    if record.correct is True:
        status = " [T]"
    elif record.correct is False:
        status = " [F]"
    return f"pred: {record.predicted_class_name}\n{conf}{status}"


def imshow_clean(ax: plt.Axes, arr: np.ndarray) -> None:
    ax.imshow(normalize_image(arr), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#0B2F6B")


def add_overlay(ax: plt.Axes, arr: np.ndarray, alpha: float) -> None:
    sal = gradient_saliency(normalize_image(arr))
    ax.imshow(sal, cmap="magma", alpha=np.clip(sal, 0.0, 1.0) * alpha, interpolation="bilinear")


def build_panel(
    selected: list[str],
    method_sources: list[MethodSource],
    per_method: dict[str, dict[str, PredictionRecord]],
    output_dir: Path,
    overlay_alpha: float,
    crop_frac: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(selected)
    n_methods = len(method_sources)
    n_cols = 3 + n_methods
    fig_w = 2.0 + 1.18 * n_cols
    fig_h = 1.6 * n_rows + 0.85
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.45, 0.82, 0.82] + [0.92] * n_methods},
        constrained_layout=True,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.965))

    column_titles = ["Original", "ROI", "Saliency"] + [source.label for source in method_sources]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=8, fontweight="bold")

    source_rows: list[dict] = []
    for row_idx, key in enumerate(selected):
        reference = per_method[method_sources[0].label][key]
        image_path = reference.image_path
        arr = load_gray(image_path)
        roi = select_roi(arr, crop_frac=crop_frac)
        roi_arr = crop(arr, roi)

        ax = axes[row_idx, 0]
        imshow_clean(ax, arr)
        x0, y0, rw, rh = roi
        ax.add_patch(Rectangle((x0, y0), rw, rh, fill=False, edgecolor="#D55E00", linewidth=1.2))
        ax.text(
            0.01,
            0.99,
            f"true: {reference.class_name}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.2,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "linewidth": 0},
        )

        imshow_clean(axes[row_idx, 1], roi_arr)
        imshow_clean(axes[row_idx, 2], roi_arr)
        add_overlay(axes[row_idx, 2], roi_arr, overlay_alpha)
        axes[row_idx, 2].text(
            0.02,
            0.98,
            "texture saliency",
            transform=axes[row_idx, 2].transAxes,
            va="top",
            ha="left",
            fontsize=5.8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.38, "pad": 1.2, "linewidth": 0},
        )

        for method_idx, source in enumerate(method_sources):
            record = per_method[source.label][key]
            ax_method = axes[row_idx, 3 + method_idx]
            imshow_clean(ax_method, roi_arr)
            if METHOD_PANEL_OVERLAY_ALPHA > 0:
                add_overlay(ax_method, roi_arr, METHOD_PANEL_OVERLAY_ALPHA)
            bbox_color = "#009E73" if record.correct is True else "#D55E00" if record.correct is False else "black"
            ax_method.text(
                0.02,
                0.98,
                fmt_prediction(record),
                transform=ax_method.transAxes,
                va="top",
                ha="left",
                fontsize=5.8,
                color="white",
                bbox={"facecolor": bbox_color, "alpha": 0.68, "pad": 1.4, "linewidth": 0},
            )
            source_rows.append(
                {
                    "sample_key": key,
                    "image_path": relpath(record.image_path),
                    "image_sha256": sha256_file(record.image_path) if record.image_path.exists() else "",
                    "true_class": record.class_name,
                    "method_label": source.label,
                    "prediction_file": relpath(source.path),
                    "predicted_class": record.predicted_class_name,
                    "confidence": "" if record.confidence is None else f"{record.confidence:.8f}",
                    "correct": "" if record.correct is None else str(record.correct).lower(),
                    "fold": "" if record.fold is None else str(record.fold),
                    "seed": "" if record.seed is None else str(record.seed),
                    "sample_index": "" if record.sample_index is None else str(record.sample_index),
                    "roi_x": str(x0),
                    "roi_y": str(y0),
                    "roi_w": str(rw),
                    "roi_h": str(rh),
                }
            )

    fig.suptitle("Sonar Qualitative Comparison", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.016, DISCLAIMER, fontsize=7, color="#333333", ha="left", va="bottom")

    png = output_dir / "sonar_qualitative_comparison.png"
    pdf = output_dir / "sonar_qualitative_comparison.pdf"
    svg = output_dir / "sonar_qualitative_comparison.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    source_csv = output_dir / "source_selection.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    return {
        "png": relpath(png),
        "pdf": relpath(pdf),
        "svg": relpath(svg),
        "source_selection_csv": relpath(source_csv),
        "row_count": n_rows,
        "method_count": n_methods,
    }


def write_caption(output_dir: Path, method_sources: list[MethodSource]) -> Path:
    caption = output_dir / "caption.md"
    methods = ", ".join(source.label for source in method_sources)
    lines = [
        "# Caption",
        "",
        (
            "Qualitative visualization of NKSID sonar classification examples. Each row shows the original "
            "sonar image, an automatically selected high-texture ROI, and method-wise ROI panels for "
            f"{methods}. Text in each method panel reports the archived predicted class and confidence from "
            "the selected outer-prediction JSONL file."
        ),
        "",
        (
        "The saliency column is an image-gradient proxy used to highlight local sonar texture. It is not "
        "model Grad-CAM and should be treated only as visual context. Method panels intentionally show "
        "the plain grayscale ROI to avoid obscuring sonar texture."
        ),
        "",
        (
            "This qualitative visualization does not replace formal macro_f1, top1, latency, LUT, DSP, BRAM, "
            "or power evidence. Quantitative claims must be made from the frozen protocol metrics and hardware "
            "evidence layers."
        ),
    ]
    caption.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return caption


def write_manifest(
    output_dir: Path,
    method_sources: list[MethodSource],
    selected: list[str],
    outputs: dict,
    overlay_alpha: float,
    crop_frac: float,
) -> Path:
    manifest = {
        "script": "scripts/build_qualitative_sonar_panel.py",
        "outputs": outputs,
        "methods": [
            {
                "label": source.label,
                "prediction_path": relpath(source.path),
                "prediction_sha256": sha256_file(source.path),
            }
            for source in method_sources
        ],
        "selected_sample_keys": selected,
        "selection_policy": "common samples across methods, sorted by disagreement then sample_index",
        "roi_policy": "automatic high-texture crop from image-gradient saliency",
        "heat_overlay": {
            "kind": "image_gradient_saliency_proxy",
            "is_grad_cam": False,
            "overlay_alpha": overlay_alpha,
            "method_panel_overlay_alpha": METHOD_PANEL_OVERLAY_ALPHA,
        },
        "crop_frac": crop_frac,
        "claim_boundary": DISCLAIMER,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build qualitative sonar comparison panels")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--method", action="append", default=None, help="Method source as LABEL=PATH")
    parser.add_argument("--output", type=Path, default=Path("results/qualitative_sonar_panel"))
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument("--crop-frac", type=float, default=0.42)
    parser.add_argument("--no-prefer-disagreement", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    method_sources = parse_method_sources(args.method, fold=args.fold, seed=args.seed)
    per_method = {source.label: load_predictions(source.path) for source in method_sources}
    selected = choose_samples(
        per_method,
        num_samples=args.num_samples,
        prefer_disagreement=not args.no_prefer_disagreement,
    )
    if not selected:
        raise RuntimeError("No common local-image samples were found across the selected methods")

    outputs = build_panel(
        selected,
        method_sources,
        per_method,
        args.output,
        overlay_alpha=args.overlay_alpha,
        crop_frac=args.crop_frac,
    )
    caption = write_caption(args.output, method_sources)
    manifest = write_manifest(args.output, method_sources, selected, outputs, args.overlay_alpha, args.crop_frac)

    print(f"[OK] Wrote {outputs['png']}")
    print(f"[OK] Wrote {outputs['pdf']}")
    print(f"[OK] Wrote {outputs['svg']}")
    print(f"[OK] Wrote {outputs['source_selection_csv']}")
    print(f"[OK] Wrote {relpath(caption)}")
    print(f"[OK] Wrote {relpath(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
