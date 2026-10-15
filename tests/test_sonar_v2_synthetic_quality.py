from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("scripts/evaluate_sonar_v2_synthetic_quality.py").resolve()
    spec = importlib.util.spec_from_file_location("evaluate_sonar_v2_synthetic_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speckle_is_deterministic_and_changes_image():
    module = _load_module()
    clean = np.full((16, 16), 0.5, dtype=np.float64)
    first = module.add_multiplicative_speckle(clean, variance=0.04, seed=123)
    second = module.add_multiplicative_speckle(clean, variance=0.04, seed=123)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, clean)
    assert first.min() >= 0.0 and first.max() <= 1.0


def test_adaptive_projection_reduces_flat_region_noise():
    module = _load_module()
    clean = np.full((64, 64), 0.5, dtype=np.float64)
    noisy = module.add_multiplicative_speckle(clean, variance=0.09, seed=456)
    kernel = np.array(
        [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ],
        dtype=np.float64,
    )
    restored = module.adaptive_denoise_v2_projection(
        noisy,
        kernel=kernel,
        gate_alpha=4.0,
        gate_beta=-2.0,
    )
    assert restored.shape == noisy.shape
    assert restored.var() < noisy.var()
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_edge_visualization_preserves_shape_and_range():
    module = _load_module()
    image = np.zeros((32, 32), dtype=np.float64)
    image[:, 16:] = 0.7
    enhanced = module.edge_v2_image_visualization(image, gamma=0.25)
    assert enhanced.shape == image.shape
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    assert not np.array_equal(enhanced, image)
