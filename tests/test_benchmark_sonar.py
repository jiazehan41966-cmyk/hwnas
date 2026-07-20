from __future__ import annotations

import numpy as np
import pytest

from hwnas_fpga.benchmarks.sonar import (
    corrupt_at_snr,
    guarded_reference_metrics,
    input_as_reference_quality,
    paired_clean_image_quality,
)


@pytest.mark.parametrize("kind", ["awgn", "speckle"])
@pytest.mark.parametrize("target", [20.0, 10.0, 5.0, 0.0])
def test_corruption_hits_requested_image_domain_snr(kind: str, target: float) -> None:
    rng = np.random.default_rng(7)
    image = rng.uniform(0.1, 0.9, size=(64, 64))
    corrupted, metadata = corrupt_at_snr(image, target, kind=kind, seed=42)
    assert corrupted.shape == image.shape
    assert metadata["absolute_error_db"] <= 0.2


def test_reference_policy_prevents_false_restoration_claim() -> None:
    image = np.full((16, 16), 0.5)
    candidate = image.copy()
    paired = paired_clean_image_quality(image, candidate)
    assert paired["reference_policy"] == "paired_clean_target"
    assert paired["psnr"] == float("inf")
    structural = input_as_reference_quality(image, candidate)
    assert "psnr" not in structural
    assert structural["claim_scope"] == "operator_effect_and_structure_preservation_only"
    with pytest.raises(ValueError, match="reference_policy"):
        guarded_reference_metrics(image, candidate, reference_policy="unpaired_real")
