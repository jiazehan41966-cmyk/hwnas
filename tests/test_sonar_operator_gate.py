import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.hardware.sonar_operator_gate import (
    REQUIRED_FOLDS,
    REQUIRED_SEEDS,
    audit_sonar_operator_manifest,
    holm_adjust,
)


def _passing_manifest():
    variants = {
        name: {
            "folds": list(REQUIRED_FOLDS),
            "seeds": list(REQUIRED_SEEDS),
            "completed_runs": 15,
            "claimable": True,
            "outer_leakage": False,
        }
        for name in ("mbconv_control", "denoise", "edge", "denoise_edge")
    }
    operator = {
        "quantization_contract": "per_tensor_symmetric_int8_v1",
        "software_spec_sha256": "a" * 64,
        "weight_export_complete": True,
        "weight_export_sha256": "b" * 64,
        "parity": {
            "real_sample_count": 10,
            "boundary_tensor_count": 5,
            "random_tensor_count": 10,
            "compared_element_count": 1000,
            "mismatch_count": 0,
        },
        "matched_control": {
            "output_shape_equal": True,
            "control_parameters": 1000,
            "operator_parameters": 1040,
            "control_macs": 10000,
            "operator_macs": 9600,
        },
        "hls": {
            "evidence_complete": True,
            "route_feasible": True,
            "hls_spec_sha256": "a" * 64,
            "spec_consumed": True,
            "consumed_spec_path": "hls/quantization_spec.json",
        },
    }
    comparisons = {
        name: {
            "method": "paired_hierarchical_bootstrap_fold_sign_flip",
            "iterations": 10_000,
            "paired_run_count": 15,
            "macro_f1_mean_delta": 0.01,
            "p_value": 0.001,
            "min_meaningful_delta": 0.01,
            "actual_gain": True,
        }
        for name in ("denoise", "edge", "denoise_edge")
    }
    return {
        "operators": {"denoise": copy.deepcopy(operator), "edge": copy.deepcopy(operator)},
        "ablation_variants": variants,
        "comparisons_vs_control": comparisons,
    }


class SonarOperatorGateTests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_complete_matched_evidence_passes(self) -> None:
        result = audit_sonar_operator_manifest(_passing_manifest())
        self.assertTrue(result["overall_pass"])
        self.assertTrue(
            result["operators"]["denoise"]["may_reenter_search_space"]
        )

    def test_one_integer_mismatch_blocks_reentry(self) -> None:
        manifest = _passing_manifest()
        manifest["operators"]["edge"]["parity"]["mismatch_count"] = 1
        result = audit_sonar_operator_manifest(manifest)
        self.assertFalse(result["overall_pass"])
        self.assertFalse(result["operators"]["edge"]["may_reenter_search_space"])

    def test_unmatched_macs_block_reentry(self) -> None:
        manifest = _passing_manifest()
        manifest["operators"]["denoise"]["matched_control"]["operator_macs"] = 10600
        result = audit_sonar_operator_manifest(manifest)
        self.assertFalse(
            result["operators"]["denoise"]["gates"]["macs_within_5pct"]
        )

    def test_v2_requires_actual_consumed_spec_and_evidence_hashes(self) -> None:
        manifest = _passing_manifest()
        manifest["schema_version"] = 2
        context = "c" * 64
        for row in manifest["ablation_variants"].values():
            row["run_fingerprints"] = ["a" * 64]
            row["protocol_context_sha256"] = context
        for comparison in manifest["comparisons_vs_control"].values():
            comparison.update({"fold_count": 5, "folds": [0, 1, 2, 3, 4], "seeds": [42, 43, 44]})

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = root / "quantization_spec.json"
            evidence = root / "hls_evidence.json"
            spec.write_text("{\"schema_version\":2}\n", encoding="utf-8")
            evidence.write_text("{\"tool\":\"vivado\"}\n", encoding="utf-8")
            spec_hash = hashlib.sha256(spec.read_bytes()).hexdigest()
            evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
            for operator in ("denoise", "edge"):
                manifest["operators"][operator]["quantization_contract"] = "per_tensor_symmetric_int8_v2"
                manifest["operators"][operator]["software_spec_sha256"] = spec_hash
                manifest["operators"][operator]["hls"] = {
                    "evidence_complete": True,
                    "route_feasible": True,
                    "spec_consumed": True,
                    "consumed_spec_path": str(spec),
                    "consumed_spec_sha256": spec_hash,
                    "consumed_spec_file_exists": True,
                    "consumed_spec_actual_sha256": spec_hash,
                    "evidence_path": str(evidence),
                    "declared_evidence_sha256": evidence_hash,
                    "evidence_file_exists": True,
                    "evidence_sha256": evidence_hash,
                    "tool": "vivado",
                    "tool_version": "2024.1",
                }
            result = audit_sonar_operator_manifest(manifest)
            self.assertTrue(result["overall_pass"])

            manifest["operators"]["edge"]["hls"]["consumed_spec_actual_sha256"] = "d" * 64
            blocked = audit_sonar_operator_manifest(manifest)
            self.assertFalse(blocked["overall_pass"])


if __name__ == "__main__":
    unittest.main()
