import pytest

from hwnas_fpga.benchmarks.statistics import (
    holm_adjust,
    paired_permutation_test,
    paired_stratified_bootstrap,
)


def test_paired_bootstrap_preserves_direction_and_is_reproducible():
    kwargs = dict(
        left=[0.8, 0.7, 0.9, 0.6],
        right=[0.7, 0.6, 0.8, 0.5],
        strata=[0, 0, 1, 1],
        iterations=1_000,
        seed=42,
    )
    first = paired_stratified_bootstrap(**kwargs)
    second = paired_stratified_bootstrap(**kwargs)
    assert first == second
    assert first["mean_difference"] == pytest.approx(0.1)
    assert first["ci95_low"] == pytest.approx(0.1)
    assert first["ci95_high"] == pytest.approx(0.1)


def test_paired_permutation_is_exact_for_fifteen_units():
    result = paired_permutation_test([1.0] * 15, [0.0] * 15)
    assert result["exact"] is True
    assert result["permutations"] == 2**15
    assert result["p_value"] == pytest.approx(2 / 2**15)


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_inference_rejects_unpaired_data():
    with pytest.raises(ValueError, match="equal-length"):
        paired_permutation_test([1, 2], [1])
