"""Frozen sonar corruption utilities and reference-policy guardrails."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from hwnas_fpga.metrics.image_quality import (
    compute_image_quality,
    edge_preservation_index,
    equivalent_number_of_looks,
    signal_to_noise_ratio,
    speckle_suppression_index,
)


NoiseKind = Literal["awgn", "speckle"]


def _float01(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("image must be non-empty and finite")
    if array.min() < 0:
        raise ValueError("image values must be non-negative")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _scale_noise_for_snr(
    image: np.ndarray,
    base_noise: np.ndarray,
    target_snr_db: float,
    *,
    tolerance_db: float,
) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(image * image))
    if signal_power <= 0:
        raise ValueError("cannot impose an SNR on a zero-power image")
    desired_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    base_power = float(np.mean(base_noise * base_noise))
    if base_power <= 0:
        raise ValueError("base noise has zero power")
    initial = math.sqrt(desired_noise_power / base_power)

    def apply(scale: float) -> tuple[np.ndarray, float]:
        candidate = np.clip(image + scale * base_noise, 0.0, 1.0)
        achieved = signal_to_noise_ratio(image, candidate)
        return candidate, float(achieved)

    candidate, achieved = apply(initial)
    if abs(achieved - target_snr_db) <= tolerance_db:
        return candidate, achieved

    # Clipping changes the error power. Search a scale that matches the SNR
    # after clipping, which is the image actually sent to the model.
    low, high = 0.0, max(initial, 1e-12)
    _, high_snr = apply(high)
    while high_snr > target_snr_db and high < initial * 1024.0:
        high *= 2.0
        _, high_snr = apply(high)
    best_candidate, best_snr = candidate, achieved
    for _ in range(80):
        middle = (low + high) / 2.0
        current, current_snr = apply(middle)
        if abs(current_snr - target_snr_db) < abs(best_snr - target_snr_db):
            best_candidate, best_snr = current, current_snr
        if current_snr > target_snr_db:
            low = middle
        else:
            high = middle
    if abs(best_snr - target_snr_db) > tolerance_db:
        raise ValueError(
            f"target SNR {target_snr_db} dB is unattainable after clipping; "
            f"best={best_snr:.3f} dB"
        )
    return best_candidate, best_snr


def corrupt_at_snr(
    image: Any,
    target_snr_db: float,
    *,
    kind: NoiseKind,
    seed: int,
    tolerance_db: float = 0.2,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    source = _float01(image)
    rng = np.random.default_rng(int(seed))
    gaussian = rng.normal(0.0, 1.0, size=source.shape)
    base_noise = gaussian if kind == "awgn" else source * gaussian
    candidate, achieved = _scale_noise_for_snr(
        source, base_noise, float(target_snr_db), tolerance_db=float(tolerance_db)
    )
    return candidate, {
        "kind": kind,
        "seed": int(seed),
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(achieved),
        "absolute_error_db": abs(float(achieved) - float(target_snr_db)),
    }


def paired_clean_image_quality(clean_target: Any, candidate: Any) -> dict[str, float | str]:
    """Compute restoration metrics only when a paired clean target exists."""

    result = compute_image_quality(clean_target, candidate)
    return {
        "reference_policy": "paired_clean_target",
        "psnr": result.psnr,
        "ssim": result.ssim,
        "mse": result.mse,
        "snr": result.snr,
        "epi": result.epi,
    }


def input_as_reference_quality(raw_input: Any, processed: Any) -> dict[str, float | str]:
    """Return operator-effect metrics without making a restoration claim."""

    return {
        "reference_policy": "input_as_reference",
        "claim_scope": "operator_effect_and_structure_preservation_only",
        "epi": edge_preservation_index(raw_input, processed),
        "raw_enl": equivalent_number_of_looks(raw_input),
        "processed_enl": equivalent_number_of_looks(processed),
        "ssi": speckle_suppression_index(raw_input, processed),
    }


def guarded_reference_metrics(
    reference: Any,
    candidate: Any,
    *,
    reference_policy: str,
) -> dict[str, float | str]:
    if reference_policy == "paired_clean_target":
        return paired_clean_image_quality(reference, candidate)
    if reference_policy == "input_as_reference":
        return input_as_reference_quality(reference, candidate)
    raise ValueError(
        "reference_policy must be paired_clean_target or input_as_reference; "
        "PSNR/SSIM restoration claims require paired_clean_target"
    )
