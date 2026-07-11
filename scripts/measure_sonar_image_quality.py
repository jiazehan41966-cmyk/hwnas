#!/usr/bin/env python3
"""Measure PSNR/SSIM/MSE/SNR/EPI/ENL/SSI for sonar image quality or operator-preservation checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.data import NKSIDDataset
from hwnas_fpga.metrics import (
    compute_image_quality,
    equivalent_number_of_looks,
    speckle_suppression_index,
)


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure sonar image PSNR and SSIM")
    parser.add_argument("--data-dir", default="data/NKSID", help="NKSID root for dataset mode")
    parser.add_argument("--split", default="val", choices=("train", "val", "full"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--transforms",
        default="identity,denoise,edge,edge_enhanced",
        help="Comma-separated transforms for dataset mode",
    )
    parser.add_argument("--reference-dir", default=None, help="Reference image root for paired mode")
    parser.add_argument("--candidate-dir", default=None, help="Candidate image root for paired mode")
    parser.add_argument(
        "--output-dir",
        default="results/sonar_image_quality_psnr_ssim_20260622",
        help="Artifact output directory",
    )
    return parser.parse_args()


def image_to_float01(path: Path, image_size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image_size:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float64) / 255.0


def gaussian_denoise(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0), mode="L")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float64) / 255.0


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    max_value = float(np.max(magnitude))
    if max_value > 0:
        magnitude = magnitude / max_value
    return np.clip(magnitude, 0.0, 1.0)


def _box_mean(image: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return window_sum / float(window * window)


def lee_filter(image: np.ndarray, window: int = 5) -> np.ndarray:
    """经典 Lee 斑点滤波：均匀区取局部均值，边缘/目标区保留原值。

    乘性噪声系数 Cu^2 用 var/mean^2 的中值估计（均匀区主导中值），
    无需外部视数参数。out = mean + k*(x-mean)，k = var_signal/var。
    """
    eps = np.finfo(np.float64).eps
    mean = _box_mean(image, window)
    mean_sq = _box_mean(image * image, window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    cu2 = float(np.median(var / np.maximum(mean * mean, eps)))
    noise_var = cu2 * mean * mean
    signal_var = np.maximum(var - noise_var, 0.0)
    k = signal_var / np.maximum(var, eps)
    return np.clip(mean + k * (image - mean), 0.0, 1.0)


def apply_transform(image: np.ndarray, transform_name: str) -> np.ndarray:
    name = transform_name.strip().lower()
    if name == "identity":
        return image.copy()
    if name == "denoise":
        return gaussian_denoise(image)
    if name == "lee":
        return lee_filter(image)
    if name == "edge":
        return sobel_magnitude(image)
    if name == "edge_enhanced":
        return np.clip(0.75 * image + 0.25 * sobel_magnitude(image), 0.0, 1.0)
    raise ValueError(f"unsupported transform: {transform_name}")


METRIC_FIELDS = ("psnr", "ssim", "mse", "snr", "epi", "ssi", "enl_reference", "enl_candidate")


def quality_fields(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    quality = compute_image_quality(reference, candidate)
    return {
        "psnr": quality.psnr,
        "ssim": quality.ssim,
        "mse": quality.mse,
        "snr": quality.snr,
        "epi": quality.epi,
        # SSI 以 reference 为 noisy、candidate 为 filtered；仅在
        # input_as_reference 协议下具有斑点抑制语义。
        "ssi": speckle_suppression_index(reference, candidate),
        "enl_reference": equivalent_number_of_looks(reference),
        "enl_candidate": equivalent_number_of_looks(candidate),
    }


def clean_number(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return None
    if isinstance(value, dict):
        return {key: clean_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_number(item) for item in value]
    return value


def summarize(values: Iterable[float]) -> dict[str, Any]:
    vals = list(values)
    finite = np.asarray([item for item in vals if math.isfinite(item)], dtype=np.float64)
    inf_count = sum(1 for item in vals if math.isinf(item))
    if finite.size:
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "finite_count": int(finite.size),
            "inf_count": int(inf_count),
        }
    return {
        "mean": math.inf if inf_count else None,
        "std": 0.0 if inf_count else None,
        "min": math.inf if inf_count else None,
        "max": math.inf if inf_count else None,
        "finite_count": 0,
        "inf_count": int(inf_count),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_number(row.get(field, "")) for field in fields})


def dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = NKSIDDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        transform=lambda image: image,
        is_training=False,
        fold=args.fold,
        use_kfold=args.split != "full",
        split=args.split,
        output_channels=1,
    )
    transforms = [item.strip() for item in args.transforms.split(",") if item.strip()]
    samples = dataset.samples[: args.max_samples] if args.max_samples else dataset.samples
    rows: list[dict[str, Any]] = []
    for sample_index, (image_path, label) in enumerate(samples):
        path = Path(image_path).resolve()
        reference = image_to_float01(path, image_size=args.image_size)
        class_name = dataset.label_to_class.get(label, str(label))
        for transform_name in transforms:
            candidate = apply_transform(reference, transform_name)
            rows.append(
                {
                    "mode": "dataset_transform",
                    "reference_policy": "input_as_reference",
                    "transform": transform_name,
                    "sample_index": sample_index,
                    "class_name": class_name,
                    "label": label,
                    "image_path": str(path),
                    **quality_fields(reference, candidate),
                }
            )
    metadata = {
        "mode": "dataset_transform",
        "reference_policy": "input_as_reference",
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "fold": args.fold,
        "image_size": args.image_size,
        "transforms": transforms,
        "samples": len(samples),
    }
    return rows, metadata


def paired_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_root = Path(args.reference_dir).resolve()
    candidate_root = Path(args.candidate_dir).resolve()
    reference_files = [
        path for path in reference_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample_index, ref_path in enumerate(sorted(reference_files)):
        rel_path = ref_path.relative_to(reference_root)
        cand_path = candidate_root / rel_path
        if not cand_path.exists():
            missing.append(str(rel_path))
            continue
        reference = image_to_float01(ref_path, image_size=args.image_size)
        candidate = image_to_float01(cand_path, image_size=args.image_size)
        rows.append(
            {
                "mode": "paired",
                "reference_policy": "paired_reference_dir",
                "transform": "paired_candidate",
                "sample_index": sample_index,
                "class_name": rel_path.parts[0] if len(rel_path.parts) > 1 else "",
                "label": "",
                "image_path": str(rel_path),
                **quality_fields(reference, candidate),
            }
        )
    metadata = {
        "mode": "paired",
        "reference_policy": "paired_reference_dir",
        "reference_dir": str(reference_root),
        "candidate_dir": str(candidate_root),
        "image_size": args.image_size,
        "samples": len(rows),
        "missing_candidates": missing,
    }
    return rows, metadata


def metric_stats(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for metric in METRIC_FIELDS:
        summary = summarize(float(row[metric]) for row in group_rows)
        stats[f"{metric}_mean"] = summary["mean"]
        stats[f"{metric}_std"] = summary["std"]
    psnr_summary = summarize(float(row["psnr"]) for row in group_rows)
    ssim_summary = summarize(float(row["ssim"]) for row in group_rows)
    stats["psnr_min"] = psnr_summary["min"]
    stats["psnr_max"] = psnr_summary["max"]
    stats["psnr_inf_count"] = psnr_summary["inf_count"]
    stats["ssim_min"] = ssim_summary["min"]
    stats["ssim_max"] = ssim_summary["max"]
    return stats


def build_summary(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["transform"]), str(row["class_name"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (transform_name, class_name), group_rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "transform": transform_name,
                "class_name": class_name,
                "sample_count": len(group_rows),
                **metric_stats(group_rows),
            }
        )

    overall_rows: list[dict[str, Any]] = []
    by_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_transform[str(row["transform"])].append(row)
    for transform_name, group_rows in sorted(by_transform.items()):
        overall_rows.append(
            {
                "transform": transform_name,
                "sample_count": len(group_rows),
                **metric_stats(group_rows),
            }
        )
    return {"metadata": metadata, "overall": overall_rows, "by_class": summary_rows}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    lines = [
        "# Sonar Image Quality Metrics",
        "",
        f"- mode: `{metadata.get('mode')}`",
        f"- reference_policy: `{metadata.get('reference_policy')}`",
        f"- samples: `{metadata.get('samples')}`",
        "- boundary: PSNR/SSIM/SNR/EPI are image-quality or structure-preservation metrics; they are not classification accuracy.",
        "- boundary: input_as_reference does not prove denoising restoration quality without clean/noisy paired ground truth.",
        "- boundary: SSI/ENL treat the reference as the noisy input; they only have speckle-suppression semantics under input_as_reference.",
        "",
        "| transform | samples | PSNR mean | PSNR std | SSIM mean | SSIM std | MSE mean | SNR mean | EPI mean | SSI mean | ENL cand mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["overall"]:
        def fmt(value: Any) -> Any:
            value = clean_number(value)
            if isinstance(value, float):
                return f"{value:.6g}"
            return value

        lines.append(
            "| {transform} | {sample_count} | {psnr_mean} | {psnr_std} | {ssim_mean} | {ssim_std} | {mse_mean} | {snr_mean} | {epi_mean} | {ssi_mean} | {enl_candidate_mean} |".format(
                transform=row["transform"],
                sample_count=row["sample_count"],
                psnr_mean=fmt(row["psnr_mean"]),
                psnr_std=fmt(row["psnr_std"]),
                ssim_mean=fmt(row["ssim_mean"]),
                ssim_std=fmt(row["ssim_std"]),
                mse_mean=fmt(row["mse_mean"]),
                snr_mean=fmt(row["snr_mean"]),
                epi_mean=fmt(row["epi_mean"]),
                ssi_mean=fmt(row["ssi_mean"]),
                enl_candidate_mean=fmt(row["enl_candidate_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_dir or args.candidate_dir:
        if not args.reference_dir or not args.candidate_dir:
            raise ValueError("--reference-dir and --candidate-dir must be provided together")
        rows, metadata = paired_rows(args)
    else:
        rows, metadata = dataset_rows(args)

    payload = build_summary(rows, metadata)
    (output_dir / "sonar_image_quality_summary.json").write_text(
        json.dumps(clean_number(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_stat_fields = [
        "psnr_mean",
        "psnr_std",
        "psnr_min",
        "psnr_max",
        "psnr_inf_count",
        "ssim_mean",
        "ssim_std",
        "ssim_min",
        "ssim_max",
        "mse_mean",
        "mse_std",
        "snr_mean",
        "snr_std",
        "epi_mean",
        "epi_std",
        "ssi_mean",
        "ssi_std",
        "enl_reference_mean",
        "enl_reference_std",
        "enl_candidate_mean",
        "enl_candidate_std",
    ]
    write_csv(
        output_dir / "sonar_image_quality_per_image.csv",
        rows,
        [
            "mode",
            "reference_policy",
            "transform",
            "sample_index",
            "class_name",
            "label",
            "image_path",
            *METRIC_FIELDS,
        ],
    )
    write_csv(
        output_dir / "sonar_image_quality_overall.csv",
        payload["overall"],
        ["transform", "sample_count", *summary_stat_fields],
    )
    write_csv(
        output_dir / "sonar_image_quality_by_class.csv",
        payload["by_class"],
        ["transform", "class_name", "sample_count", *summary_stat_fields],
    )
    write_markdown(output_dir / "sonar_image_quality_summary.md", payload)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "samples": metadata.get("samples")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
