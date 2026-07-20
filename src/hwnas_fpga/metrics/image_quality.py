"""Sonar image-quality metrics: PSNR, SSIM, MSE, SNR, edge preservation, speckle.

The functions here intentionally do not depend on scikit-image so they can run
inside the lightweight project environment. Inputs are converted to float arrays
in [0, 1] and may be 2-D grayscale, HWC, or CHW tensors/arrays.

Reference-based metrics (PSNR/SSIM/MSE/SNR/EPI) require a meaningful reference:
with ``input_as_reference`` they only measure operator effect, not restoration
quality. ENL is no-reference; SSI compares a filtered image against its noisy
input and is meaningful for the input-as-reference protocol.
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
    snr: float
    epi: float
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


def signal_to_noise_ratio(reference: Any, candidate: Any) -> float:
    """SNR in dB: 10*log10(signal power / error power), error = candidate - reference."""
    ref = _as_float01(reference)
    cand = _as_float01(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"reference and candidate shapes differ: {ref.shape} vs {cand.shape}")
    signal_power = float(np.mean(ref * ref))
    noise_power = float(np.mean((cand - ref) ** 2))
    if noise_power == 0.0:
        return math.inf
    if signal_power == 0.0:
        return -math.inf
    return float(10.0 * math.log10(signal_power / noise_power))


def _sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
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
    return np.sqrt(gx * gx + gy * gy)


def edge_preservation_index(reference: Any, candidate: Any) -> float:
    """EPI: Pearson correlation between Sobel gradient magnitudes, in [-1, 1].

    1.0 means the candidate preserves the reference edge structure exactly;
    low-pass smoothing that blurs target contours lowers the score.
    """
    ref = _to_channel_first(_as_float01(reference))
    cand = _to_channel_first(_as_float01(candidate))
    if ref.shape != cand.shape:
        raise ValueError(f"reference and candidate shapes differ: {ref.shape} vs {cand.shape}")

    scores: list[float] = []
    for ref_channel, cand_channel in zip(ref, cand):
        g_ref = _sobel_gradient_magnitude(ref_channel)
        g_cand = _sobel_gradient_magnitude(cand_channel)
        ref_centered = g_ref - g_ref.mean()
        cand_centered = g_cand - g_cand.mean()
        ref_norm = float(np.sqrt(np.sum(ref_centered * ref_centered)))
        cand_norm = float(np.sqrt(np.sum(cand_centered * cand_centered)))
        if ref_norm == 0.0 and cand_norm == 0.0:
            scores.append(1.0)
        elif ref_norm == 0.0 or cand_norm == 0.0:
            scores.append(0.0)
        else:
            scores.append(float(np.sum(ref_centered * cand_centered) / (ref_norm * cand_norm)))
    return float(np.mean(scores))


def equivalent_number_of_looks(image: Any) -> float:
    """ENL = mean^2 / variance over the image (no-reference speckle metric).

    Higher means smoother homogeneous regions; useful on speckled sonar data
    where no clean reference exists. Returns inf for a constant image.
    """
    array = _as_float01(image)
    mean = float(np.mean(array))
    variance = float(np.var(array))
    if variance == 0.0:
        return math.inf
    return mean * mean / variance


def speckle_suppression_index(noisy: Any, filtered: Any) -> float:
    """SSI = (sigma_f/mu_f) / (sigma_n/mu_n); below 1 means speckle suppression.

    Meaningful when ``noisy`` is the raw input and ``filtered`` is the operator
    output, which matches the input-as-reference protocol.
    """
    noisy_array = _as_float01(noisy)
    filtered_array = _as_float01(filtered)
    if noisy_array.shape != filtered_array.shape:
        raise ValueError(
            f"noisy and filtered shapes differ: {noisy_array.shape} vs {filtered_array.shape}"
        )
    noisy_mean = float(np.mean(noisy_array))
    filtered_mean = float(np.mean(filtered_array))
    noisy_cv = float(np.std(noisy_array)) / noisy_mean if noisy_mean > 0 else math.inf
    filtered_cv = float(np.std(filtered_array)) / filtered_mean if filtered_mean > 0 else math.inf
    if noisy_cv == 0.0:
        return math.inf if filtered_cv > 0 else 1.0
    if math.isinf(noisy_cv):
        return 0.0 if not math.isinf(filtered_cv) else 1.0
    return filtered_cv / noisy_cv


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
        snr=signal_to_noise_ratio(ref, cand),
        epi=edge_preservation_index(ref, cand),
        data_range=float(data_range),
    )
