import unittest

from hwnas_fpga.training.protocol_reporting import protocol_claimability


class ProtocolV2Tests(unittest.TestCase):
    def test_context_hash_is_required_when_protocol_declares_it(self):
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=["a" * 64] * 15,
            protocol_context_sha256="b" * 64,
            provenance_contexts=["b" * 64] * 14 + ["c" * 64],
        )
        self.assertFalse(result["claimable"])
        self.assertFalse(result["protocol_context_complete"])
        self.assertFalse(result["group_generalization_claimable"])

    def test_group_split_unavailable_is_explicit(self):
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=["a" * 64] * 15,
            group_split_available=False,
        )
        self.assertTrue(result["claimable"])
        self.assertFalse(result["group_split_available"])
        self.assertFalse(result["group_generalization_claimable"])


if __name__ == "__main__":
    unittest.main()
