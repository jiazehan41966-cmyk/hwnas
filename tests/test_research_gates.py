import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.research_gates import (
    SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY,
    require_stage3_search_gate,
    stage3_gate_status,
)


class Stage3GateTests(unittest.TestCase):
    def test_frozen_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status = stage3_gate_status(tmpdir)
            self.assertEqual(status["status"], "FROZEN")
            self.assertEqual(
                status["semantic_safe_search_space_cardinality"],
                SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY,
            )
            with self.assertRaisesRegex(RuntimeError, "Stage 3 is frozen"):
                require_stage3_search_gate(tmpdir, dataset_name="nksid")

    def test_dummy_is_smoke_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status = require_stage3_search_gate(tmpdir, dataset_name="dummy")
            self.assertEqual(status["status"], "SMOKE_ONLY")
            self.assertFalse(status["may_start_new_claimable_search"])

    def test_all_three_evidence_classes_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts/hw_surrogate_calibration_v2").mkdir(parents=True)
            (root / "results/g4_rl_arch_193_fold1_seed42").mkdir(parents=True)
            (root / "artifacts/hw_surrogate_calibration_v2/calibration_v2.json").write_text(
                json.dumps({"g2_pass": True}), encoding="utf-8"
            )
            (
                root
                / "results/g4_rl_arch_193_fold1_seed42/board_validation_summary.json"
            ).write_text(
                json.dumps({"claimable": True, "numeric_mismatch_count": 0}),
                encoding="utf-8",
            )
            (root / "artifacts/stage3_replan_approval.json").write_text(
                json.dumps(
                    {"approved": True, "method": "hierarchical_random"}
                ),
                encoding="utf-8",
            )
            status = require_stage3_search_gate(
                root,
                dataset_name="nksid",
            )
            self.assertTrue(status["may_start_new_claimable_search"])


if __name__ == "__main__":
    unittest.main()
