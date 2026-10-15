#!/usr/bin/env python3
"""Evaluate synthetic-speckle image quality for the trained v2 sonar operator core."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hwnas_fpga.metrics import peak_signal_noise_ratio, structural_similarity_index

from measure_sonar_image_quality import gaussian_denoise, sobel_magnitude


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_DATASETS = {
    "nksid": PROJECT_ROOT / "data" / "NKSID",
    "roboflow": (
        PROJECT_ROOT
        / "data"
        / "external"
        / "roboflow_cylider2"
        / "source_reconstruction"
        / "v6"
    ),
    "figshare": (
        PROJECT_ROOT
        / "data"
        / "external"
        / "figshare_mine_detection_v2"
        / "extracted"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset in name=path form. Repeat for multiple datasets; defaults to all three.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--variances", default="0.01,0.04,0.09")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--adaptive-checkpoint-dir",
        default="results/protocol/g5v2_adaptive_denoise",
    )
    parser.add_argument(
        "--combined-checkpoint-dir",
        default="results/protocol/g5v2_adaptive_denoise_edge_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sonar_v2_synthetic_quality_20260815",
    )
    return parser.parse_args()


def parse_datasets(items: list[str]) -> dict[str, Path]:
    if not items:
        return {name: path.resolve() for name, path in DEFAULT_DATASETS.items()}
    datasets: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--dataset must use name=path")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or name in datasets:
            raise ValueError("dataset names must be non-empty and unique")
        datasets[name] = Path(raw_path.strip()).resolve()
    return datasets


def parse_variances(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("variances must contain positive finite values")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(root: Path) -> list[Path]:
    files = sorted(root.resolve().glob("best_fold*_seed*.pt"))
    if not files:
        raise FileNotFoundError(f"no trained checkpoints found under: {root}")
    return files


def load_trained_adaptive_projection(checkpoint_dir: Path) -> dict[str, Any]:
    effective_kernels: list[np.ndarray] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    sources: list[dict[str, Any]] = []
    for path in _checkpoint_files(checkpoint_dir):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        smooth_keys = [key for key in state if key.endswith("smooth_weight")]
        alpha_keys = [key for key in state if key.endswith("gate_alpha")]
        beta_keys = [key for key in state if key.endswith("gate_beta")]
        if len(smooth_keys) != 1 or len(alpha_keys) != 1 or len(beta_keys) != 1:
            raise ValueError(f"checkpoint does not contain one adaptive core: {path}")
        raw_kernel = state[smooth_keys[0]].detach().float()
        effective = torch.softmax(raw_kernel.flatten(1), dim=1).view_as(raw_kernel)
        effective_kernels.append(effective.mean(dim=0).squeeze(0).numpy().astype(np.float64))
        alpha_values.append(float(state[alpha_keys[0]].detach().float().mean()))
        beta_values.append(float(state[beta_keys[0]].detach().float().mean()))
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "fold": int(payload["fold"]),
                "seed": int(payload["seed"]),
            }
        )
    kernel = np.mean(np.stack(effective_kernels), axis=0)
    kernel = kernel / float(kernel.sum())
    return {
        "kernel": kernel,
        "gate_alpha": float(np.mean(alpha_values)),
        "gate_beta": float(np.mean(beta_values)),
        "checkpoint_count": len(sources),
        "sources": sources,
        "projection_boundary": (
            "The trained 32-channel adaptive core is averaged over channels and checkpoints "
            "to form a one-channel image-domain projection. It is not the full classifier output."
        ),
    }


def load_trained_edge_strength(checkpoint_dir: Path) -> dict[str, Any]:
    values: list[float] = []
    sources: list[dict[str, Any]] = []
    for path in _checkpoint_files(checkpoint_dir):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        gamma_keys = [key for key in state if key.endswith("edge_gamma")]
        if len(gamma_keys) != 1:
            raise ValueError(f"checkpoint does not contain one edge gamma: {path}")
        values.append(float(state[gamma_keys[0]].detach().float()))
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "fold": int(payload["fold"]),
                "seed": int(payload["seed"]),
            }
        )
    return {
        "mean_gamma": float(np.mean(values)),
        "std_gamma": float(np.std(values)),
        "min_gamma": float(np.min(values)),
        "max_gamma": float(np.max(values)),
        "checkpoint_count": len(sources),
        "sources": sources,
        "projection_boundary": (
            "The image visualization keeps the v2 intensity-preserving additive form and "
            "uses the learned mean gamma, but it cannot reproduce feature-channel PW/BN weights."
        ),
    }


def image_files(root: Path, max_samples: int | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_samples:
        files = files[:max_samples]
    if not files:
        raise ValueError(f"no supported images found under: {root}")
    return files


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float64) / 255.0


def add_multiplicative_speckle(
    clean: np.ndarray,
    *,
    variance: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, math.sqrt(variance), size=clean.shape)
    return np.clip(clean + clean * noise, 0.0, 1.0)


def _weighted_local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    result = np.zeros_like(image, dtype=np.float64)
    for row in range(kh):
        for column in range(kw):
            result += kernel[row, column] * padded[row : row + height, column : column + width]
    return result


def _box_mean(image: np.ndarray, window: int = 3) -> np.ndarray:
    kernel = np.full((window, window), 1.0 / float(window * window), dtype=np.float64)
    return _weighted_local_mean(image, kernel)


def adaptive_denoise_v2_projection(
    image: np.ndarray,
    *,
    kernel: np.ndarray,
    gate_alpha: float,
    gate_beta: float,
) -> np.ndarray:
    local_mean = _weighted_local_mean(image, kernel)
    residual = image - local_mean
    evidence = _box_mean(np.abs(residual), window=3)
    gate = 1.0 / (1.0 + np.exp(-(gate_alpha * evidence + gate_beta)))
    return np.clip(local_mean + gate * residual, 0.0, 1.0)


def edge_v2_image_visualization(image: np.ndarray, *, gamma: float) -> np.ndarray:
    return np.clip(image + gamma * sobel_magnitude(image), 0.0, 1.0)


def _summarize(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = parse_datasets(args.dataset)
    variances = parse_variances(args.variances)
    adaptive = load_trained_adaptive_projection(Path(args.adaptive_checkpoint_dir))
    edge = load_trained_edge_strength(Path(args.combined_checkpoint_dir))
    methods = (
        "noisy_input",
        "v1_fixed_smooth",
        "v2_adaptive_projection",
        "v2_adaptive_edge_visualization",
    )

    per_image: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    for dataset_index, (dataset_name, root) in enumerate(datasets.items()):
        files = image_files(root, args.max_samples)
        dataset_counts[dataset_name] = len(files)
        for variance_index, variance in enumerate(variances):
            for sample_index, path in enumerate(files):
                clean = load_image(path, args.image_size)
                noise_seed = (
                    int(args.seed)
                    + dataset_index * 10_000_019
                    + variance_index * 1_000_003
                    + sample_index
                )
                noisy = add_multiplicative_speckle(clean, variance=variance, seed=noise_seed)
                v1 = gaussian_denoise(noisy)
                v2 = adaptive_denoise_v2_projection(
                    noisy,
                    kernel=adaptive["kernel"],
                    gate_alpha=adaptive["gate_alpha"],
                    gate_beta=adaptive["gate_beta"],
                )
                v2_edge = edge_v2_image_visualization(v2, gamma=edge["mean_gamma"])
                candidates = {
                    "noisy_input": noisy,
                    "v1_fixed_smooth": v1,
                    "v2_adaptive_projection": v2,
                    "v2_adaptive_edge_visualization": v2_edge,
                }
                for method, candidate in candidates.items():
                    per_image.append(
                        {
                            "dataset": dataset_name,
                            "variance": variance,
                            "method": method,
                            "sample_index": sample_index,
                            "image_path": str(path.resolve()),
                            "noise_seed": noise_seed,
                            "psnr": peak_signal_noise_ratio(clean, candidate),
                            "ssim": structural_similarity_index(clean, candidate),
                        }
                    )

    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_image:
        grouped[(row["dataset"], float(row["variance"]), row["method"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for variance in variances:
            baseline = grouped[(dataset_name, variance, "noisy_input")]
            baseline_psnr, _ = _summarize(float(row["psnr"]) for row in baseline)
            baseline_ssim, _ = _summarize(float(row["ssim"]) for row in baseline)
            for method in methods:
                rows = grouped[(dataset_name, variance, method)]
                psnr_mean, psnr_std = _summarize(float(row["psnr"]) for row in rows)
                ssim_mean, ssim_std = _summarize(float(row["ssim"]) for row in rows)
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "variance": variance,
                        "noise_level": {0.01: "weak", 0.04: "medium", 0.09: "strong"}.get(
                            variance, "custom"
                        ),
                        "method": method,
                        "sample_count": len(rows),
                        "psnr_mean": psnr_mean,
                        "psnr_std": psnr_std,
                        "ssim_mean": ssim_mean,
                        "ssim_std": ssim_std,
                        "psnr_gain_vs_noisy": psnr_mean - baseline_psnr,
                        "ssim_gain_vs_noisy": ssim_mean - baseline_ssim,
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "per_image.csv",
        per_image,
        ["dataset", "variance", "method", "sample_index", "image_path", "noise_seed", "psnr", "ssim"],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "dataset",
            "variance",
            "noise_level",
            "method",
            "sample_count",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "psnr_gain_vs_noisy",
            "ssim_gain_vs_noisy",
        ],
    )
    payload = {
        "schema_version": 1,
        "protocol": "synthetic_multiplicative_gaussian_speckle_v1",
        "reference_policy": "original_image_as_relative_clean_reference",
        "datasets": {name: {"root": str(path), "samples": dataset_counts[name]} for name, path in datasets.items()},
        "image_size": int(args.image_size),
        "variances": variances,
        "seed": int(args.seed),
        "adaptive_projection": {
            **{key: value for key, value in adaptive.items() if key not in {"kernel", "sources"}},
            "kernel": adaptive["kernel"].tolist(),
            "sources": adaptive["sources"],
        },
        "edge_visualization": edge,
        "summary": summary_rows,
        "claim_boundary": (
            "The v2 adaptive row is a trained one-channel projection of the feature operator. "
            "The edge row is visualization-only. Neither row is the full classifier output; "
            "full-network effect must be judged with noisy classification metrics."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(Path(args.output_dir).resolve()),
                "datasets": {name: row["samples"] for name, row in payload["datasets"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
