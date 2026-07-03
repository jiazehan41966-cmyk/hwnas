from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hwnas_fpga.metrics import compute_image_quality, peak_signal_noise_ratio, structural_similarity_index


def test_identical_images_have_infinite_psnr_and_unit_ssim():
    image = np.full((16, 16), 0.5, dtype=np.float64)

    result = compute_image_quality(image, image)

    assert math.isinf(result.psnr)
    assert result.mse == 0.0
    assert result.ssim == 1.0


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


def test_measure_script_importable():
    script = Path("scripts/measure_sonar_image_quality.py")
    assert script.exists()
