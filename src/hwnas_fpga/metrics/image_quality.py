"""PSNR and SSIM image-quality metrics.

The functions here intentionally do not depend on scikit-image so they can run
inside the lightweight project environment. Inputs are converted to float arrays
in [0, 1] and may be 2-D grayscale, HWC, or CHW tensors/arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageQualityResult:
    psnr: float
    ssim: float
    mse: float
    data_range: float


def _as_float01(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    array = array.astype(np.float64, copy=False)
    if np.nanmin(array) < 0:
        raise ValueError("image values must be non-negative")
    max_value = float(np.nanmax(array))
    if max_value > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _to_channel_first(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4):
        return array
    if array.shape[-1] in (1, 3, 4):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot infer channel axis for shape {array.shape}")


def _uniform_filter2d(image: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return window_sum / float(window_size * window_size)


def peak_signal_noise_ratio(reference: Any, candidate: Any, data_range: float = 1.0) -> float:
    ref = _as_float01(reference)
    cand = _as_float01(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"reference and candidate shapes differ: {ref.shape} vs {cand.shape}")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range * data_range) / mse))


def structural_similarity_index(
    reference: Any,
    candidate: Any,
    data_range: float = 1.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    ref = _to_channel_first(_as_float01(reference))
    cand = _to_channel_first(_as_float01(candidate))
    if ref.shape != cand.shape:
        raise ValueError(f"reference and candidate shapes differ: {ref.shape} vs {cand.shape}")

    if min(ref.shape[-2:]) < window_size:
        window_size = min(ref.shape[-2:])
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(1, window_size)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    channel_scores: list[float] = []
    for ref_channel, cand_channel in zip(ref, cand):
        mu_ref = _uniform_filter2d(ref_channel, window_size)
        mu_cand = _uniform_filter2d(cand_channel, window_size)
        mu_ref_sq = mu_ref * mu_ref
        mu_cand_sq = mu_cand * mu_cand
        mu_ref_cand = mu_ref * mu_cand

        sigma_ref_sq = _uniform_filter2d(ref_channel * ref_channel, window_size) - mu_ref_sq
        sigma_cand_sq = _uniform_filter2d(cand_channel * cand_channel, window_size) - mu_cand_sq
        sigma_ref_cand = _uniform_filter2d(ref_channel * cand_channel, window_size) - mu_ref_cand

        numerator = (2.0 * mu_ref_cand + c1) * (2.0 * sigma_ref_cand + c2)
        denominator = (mu_ref_sq + mu_cand_sq + c1) * (sigma_ref_sq + sigma_cand_sq + c2)
        ssim_map = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
        channel_scores.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_scores))


def compute_image_quality(reference: Any, candidate: Any, data_range: float = 1.0) -> ImageQualityResult:
    ref = _as_float01(reference)
    cand = _as_float01(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"reference and candidate shapes differ: {ref.shape} vs {cand.shape}")
    mse = float(np.mean((ref - cand) ** 2))
    return ImageQualityResult(
        psnr=peak_signal_noise_ratio(ref, cand, data_range=data_range),
        ssim=structural_similarity_index(ref, cand, data_range=data_range),
        mse=mse,
        data_range=float(data_range),
    )
