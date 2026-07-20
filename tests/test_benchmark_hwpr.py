from pathlib import Path

import pytest
import torch

from hwnas_fpga.benchmarks.hwpr import (
    audit_hwpr_author_runtime,
    fit_paper_spec_surrogate,
    listmle_pareto_loss,
)
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate


def _encoding(index: int) -> dict:
    return {
        "input_channels": 1,
        "stem_channels": 8,
        "stem_stride": 2,
        "post_stem_downsample_stride": 1,
        "stages": [
            {
                "channels": 8 + 4 * index,
                "depth": 1,
                "stride": 1,
                "blocks": [
                    {
                        "op": "conv" if index % 2 == 0 else "mbconv",
                        "kernel_size": 3,
                        "expand_ratio": 1 + index % 2,
                        "stride": 1,
                    }
                ],
            }
        ],
    }


def test_listmle_prefers_scores_already_ordered_by_pareto_rank() -> None:
    ranks = torch.tensor([0, 1, 2])
    good = listmle_pareto_loss(torch.tensor([3.0, 2.0, 1.0]), ranks)
    bad = listmle_pareto_loss(torch.tensor([1.0, 2.0, 3.0]), ranks)
    assert good < bad


def test_paper_spec_surrogate_smoke_returns_all_architectures() -> None:
    candidates = [
        SearchCandidate(
            arch_id=f"a{index}",
            encoding=_encoding(index),
            metrics=CandidateMetrics(
                f_clean=0.70 + index * 0.02,
                f_robust=0.65 + index * 0.01,
                latency_ms=10.0 - index,
            ),
        )
        for index in range(4)
    ]
    result = fit_paper_spec_surrogate(
        candidates,
        objectives=["f_clean", "f_robust", "latency_ms"],
        directions=["max", "max", "min"],
        epochs=20,
        seed=7,
    )
    assert len(result.ranks) == 4
    assert set(result.ordered_arch_ids) == {"a0", "a1", "a2", "a3"}
    assert result.feature_schema["paper_encoder_equivalent"] is False
    assert result.final_loss >= 0.0


def test_pinned_author_checkout_is_explicitly_not_runtime_ready() -> None:
    checkout = Path(__file__).resolve().parents[1] / "reference" / "HW-PR-NAS"
    if not checkout.exists():
        pytest.skip("pinned author checkout not installed")
    audit = audit_hwpr_author_runtime(checkout)
    assert audit["author_runtime_ready"] is False
    assert "readme_listed_runtime_files_missing" in audit["blockers"]
    assert "undefined_valid_loss_call" in audit["blockers"]
