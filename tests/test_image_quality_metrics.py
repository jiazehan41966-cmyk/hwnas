from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hwnas_fpga.metrics import (
    compute_image_quality,
    edge_preservation_index,
    equivalent_number_of_looks,
    peak_signal_noise_ratio,
    signal_to_noise_ratio,
    speckle_suppression_index,
    structural_similarity_index,
)


def test_identical_images_have_infinite_psnr_and_unit_ssim():
    image = np.full((16, 16), 0.5, dtype=np.float64)

    result = compute_image_quality(image, image)

    assert math.isinf(result.psnr)
    assert result.mse == 0.0
    assert result.ssim == 1.0
    assert math.isinf(result.snr)
    assert result.epi == 1.0


def test_psnr_matches_known_mse():
    reference = np.zeros((8, 8), dtype=np.float64)
    candidate = np.full((8, 8), 0.1, dtype=np.float64)

    assert peak_signal_noise_ratio(reference, candidate) == 20.0


def test_ssim_decreases_for_structural_change():
    reference = np.zeros((32, 32), dtype=np.float64)
    reference[8:24, 8:24] = 1.0
    shifted = np.zeros((32, 32), dtype=np.float64)
    shifted[10:26, 10:26] = 1.0

    assert structural_similarity_index(reference, shifted) < 1.0


def test_snr_matches_known_powers():
    reference = np.full((8, 8), 0.2, dtype=np.float64)
    candidate = np.full((8, 8), 0.3, dtype=np.float64)

    # signal power 0.04, noise power 0.01 -> 10*log10(4)
    assert signal_to_noise_ratio(reference, candidate) == pytest.approx(10.0 * math.log10(4.0))


def test_epi_perfect_for_identity_and_lower_after_blur():
    rng = np.random.default_rng(0)
    image = np.zeros((32, 32), dtype=np.float64)
    image[8:24, 8:24] = 1.0
    image = np.clip(image + rng.normal(0.0, 0.05, image.shape), 0.0, 1.0)

    assert edge_preservation_index(image, image) == pytest.approx(1.0)

    # 3x3 box blur weakens edges
    padded = np.pad(image, 1, mode="reflect")
    blurred = sum(
        padded[dy : dy + 32, dx : dx + 32] for dy in range(3) for dx in range(3)
    ) / 9.0
    assert edge_preservation_index(image, blurred) < 0.999


def test_enl_increases_after_smoothing_speckle():
    rng = np.random.default_rng(1)
    speckled = np.clip(0.5 * rng.rayleigh(scale=0.5, size=(64, 64)), 0.0, 1.0)
    padded = np.pad(speckled, 1, mode="reflect")
    smoothed = sum(
        padded[dy : dy + 64, dx : dx + 64] for dy in range(3) for dx in range(3)
    ) / 9.0

    assert equivalent_number_of_looks(smoothed) > equivalent_number_of_looks(speckled)
    assert math.isinf(equivalent_number_of_looks(np.full((8, 8), 0.5)))


def test_ssi_below_one_after_smoothing():
    rng = np.random.default_rng(2)
    speckled = np.clip(0.5 * rng.rayleigh(scale=0.5, size=(64, 64)), 0.0, 1.0)
    padded = np.pad(speckled, 1, mode="reflect")
    smoothed = sum(
        padded[dy : dy + 64, dx : dx + 64] for dy in range(3) for dx in range(3)
    ) / 9.0

    assert speckle_suppression_index(speckled, smoothed) < 1.0
    assert speckle_suppression_index(speckled, speckled) == pytest.approx(1.0)


def test_measure_script_importable():
    script = Path("scripts/measure_sonar_image_quality.py")
    assert script.exists()
