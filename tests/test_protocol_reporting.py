import unittest

from hwnas_fpga.training.protocol_reporting import (
    canonical_sha256,
    hierarchical_paired_bootstrap,
    protocol_claimability,
)


class ClaimabilityTests(unittest.TestCase):
    def test_complete_protocol_is_claimable(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
        )
        self.assertTrue(result["claimable"])
        self.assertTrue(result["nas_generalization_claimable"])

    def test_partial_protocol_is_not_claimable(self) -> None:
        result = protocol_claimability(
            folds=(0,),
            seeds=(42,),
            completed_pairs=[(0, 42)],
            selection_provenance="baseline_predeclared",
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
        )
        self.assertTrue(result["claimable"])
        self.assertFalse(result["nas_generalization_claimable"])
        self.assertTrue(result["warnings"])


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
