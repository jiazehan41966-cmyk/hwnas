import unittest

from hwnas_fpga.training.protocol_reporting import (
    canonical_sha256,
    hierarchical_paired_bootstrap,
    protocol_claimability,
)


class ClaimabilityTests(unittest.TestCase):
    RUN_FINGERPRINTS = ["a" * 64] * 15

    def test_complete_protocol_is_claimable(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=self.RUN_FINGERPRINTS,
        )
        self.assertTrue(result["claimable"])
        self.assertTrue(result["nas_generalization_claimable"])

    def test_outer_selection_or_missing_provenance_blocks_claim(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        leaked = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            outer_validation_used_for_selection=True,
            provenance_fingerprints=self.RUN_FINGERPRINTS,
        )
        missing = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_complete=False,
            provenance_fingerprints=self.RUN_FINGERPRINTS,
        )
        self.assertFalse(leaked["claimable"])
        self.assertFalse(missing["claimable"])

    def test_missing_source_freeze_blocks_formal_claim(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=self.RUN_FINGERPRINTS,
            source_freeze_verified=False,
        )
        self.assertFalse(result["claimable"])
        self.assertFalse(result["source_freeze_verified"])
        self.assertTrue(any("source-freeze" in value for value in result["warnings"]))

    def test_partial_protocol_is_not_claimable(self) -> None:
        result = protocol_claimability(
            folds=(0,),
            seeds=(42,),
            completed_pairs=[(0, 42)],
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=["a" * 64],
        )
        self.assertFalse(result["claimable"])
        self.assertTrue(result["legacy"])

    def test_legacy_selected_candidate_has_narrow_claim_scope(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="legacy_fold0_selected",
            provenance_fingerprints=self.RUN_FINGERPRINTS,
        )
        self.assertTrue(result["claimable"])
        self.assertFalse(result["nas_generalization_claimable"])
        self.assertTrue(result["warnings"])

    def test_mixed_run_fingerprints_are_not_claimable(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        fingerprints = ["a" * 64] * 14 + ["b" * 64]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=fingerprints,
        )
        self.assertFalse(result["claimable"])
        self.assertFalse(result["fingerprint_complete"])

    def test_duplicate_fold_seed_records_are_not_claimable(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs + [(0, 42)],
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=["a" * 64] * 16,
        )
        self.assertFalse(result["claimable"])
        self.assertTrue(result["duplicate_pairs"])


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_ignores_dict_order(self) -> None:
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [2, 3]}),
            canonical_sha256({"b": [2, 3], "a": 1}),
        )


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_reports_known_difference(self) -> None:
        left = []
        right = []
        for fold in range(5):
            for seed in (42, 43, 44):
                left.append({"fold": fold, "seed": seed, "macro_f1": 0.8})
                right.append({"fold": fold, "seed": seed, "macro_f1": 0.7})
        result = hierarchical_paired_bootstrap(
            left,
            right,
            metric="macro_f1",
            iterations=100,
        )
        self.assertAlmostEqual(result["mean_difference"], 0.1)
        self.assertGreater(result["ci95_low"], 0.09)


if __name__ == "__main__":
    unittest.main()
