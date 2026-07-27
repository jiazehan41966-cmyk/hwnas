from __future__ import annotations

from hwnas_fpga.hardware.mixconv_v2_gate import (
    audit_mixconv_v2_admission,
    exact_fold_sign_flip_p_value,
    paired_hierarchical_comparison,
)


def _rows(delta: float):
    return [
        {
            "fold": fold,
            "seed": seed,
            "control_macro_f1": 0.70 + fold * 0.01,
            "candidate_macro_f1": 0.70 + fold * 0.01 + delta,
        }
        for fold in range(5)
        for seed in (42, 43, 44)
    ]


def _complete_manifest():
    mix_k3 = paired_hierarchical_comparison(_rows(0.02), iterations=1_000)
    mix_k5 = paired_hierarchical_comparison(_rows(0.006), iterations=1_000)
    k5_k3 = paired_hierarchical_comparison(_rows(0.014), iterations=1_000)
    return {
        "comparisons": {
            "mixconv_v2_vs_mbconv_k3": mix_k3,
            "mixconv_v2_vs_mbconv_k5": mix_k5,
            "mbconv_k5_vs_mbconv_k3": k5_k3,
        },
        "robustness": {"mean_macro_f1_delta_vs_control": -0.005},
        "int8_parity": {
            "real_activation_count": 2,
            "boundary_tensor_count": 2,
            "random_tensor_count": 2,
            "compared_element_count": 100,
            "mismatch_count": 0,
        },
        "hardware": {
            "advantage_vs_mbconv_k5": False,
            "hls_5ns_synthesis_pass": True,
            "complete_route_pass": True,
            "resources_within_search_constraints": True,
            "measured_lut_traceable": True,
        },
        "provenance": {
            "complete_15_run_records": True,
            "checkpoint_hashes_complete": True,
            "source_freeze_verified": True,
        },
    }


def test_five_fold_two_sided_sign_flip_cannot_reach_0p05() -> None:
    assert exact_fold_sign_flip_p_value([1, 1, 1, 1, 1]) == 0.0625


def test_comparison_reports_effect_ci_fold_direction_and_p() -> None:
    result = paired_hierarchical_comparison(_rows(0.02), iterations=1_000)
    assert result["paired_run_count"] == 15
    assert result["macro_f1_mean_delta"] > 0.019
    assert result["stratified_bootstrap_ci95_lower"] > 0.0
    assert result["positive_fold_count"] == 5
    assert result["two_sided_fold_sign_flip_p_value"] == 0.0625


def test_complete_evidence_admits_without_p_value_gate() -> None:
    result = audit_mixconv_v2_admission(_complete_manifest())
    assert result["status"] == "ADMITTED"
    assert result["may_enable_mixconv_v2"] is True
    assert result["p_values_are_hard_gates"] is False


def test_one_integer_mismatch_blocks_custom_operator() -> None:
    manifest = _complete_manifest()
    manifest["int8_parity"]["mismatch_count"] = 1
    result = audit_mixconv_v2_admission(manifest)
    assert result["status"] == "GENERAL_OP_SELECTED"
    assert result["may_enable_mixconv_v2"] is False


def test_missing_evidence_is_not_misclassified_as_negative_result() -> None:
    assert audit_mixconv_v2_admission({})["status"] == "NOT_READY"
