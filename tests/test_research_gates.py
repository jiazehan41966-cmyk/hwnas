import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.research_gates import (
    SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY,
    STAGE3_ALLOWED_METHODS,
    require_stage3_search_gate,
    stage3_gate_status,
)


def _write_gate0(root: Path, *, gate_status: str = "pass", completed: int = 1200) -> None:
    path = root / "artifacts/proxy_reliability_gate0/manifest_summary_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gate_status": gate_status,
                "work_unit_count": 1200,
                "formal_completed_work_units": completed,
            }
        ),
        encoding="utf-8",
    )


def _write_g5(root: Path, *, overall_pass: bool) -> None:
    path = root / "artifacts/sonar_operator_gate/sonar_operator_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"overall_pass": overall_pass}), encoding="utf-8")


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
            _write_gate0(root)
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

    def test_aging_and_rl_can_be_jointly_approved_for_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_gate0(root)
            (root / "artifacts/hw_surrogate_calibration_v2").mkdir(parents=True)
            (root / "results/g4_rl_arch_193_fold1_seed42").mkdir(parents=True)
            (root / "artifacts/hw_surrogate_calibration_v2/calibration_v2.json").write_text(
                json.dumps({"g2_pass": True}), encoding="utf-8"
            )
            (root / "results/g4_rl_arch_193_fold1_seed42/board_validation_summary.json").write_text(
                json.dumps({"claimable": True, "numeric_mismatch_count": 0}),
                encoding="utf-8",
            )
            (root / "artifacts/stage3_replan_approval.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "methods": ["rl", "aging_evolution"],
                    }
                ),
                encoding="utf-8",
            )
            for method in ("rl", "aging_evolution"):
                status = require_stage3_search_gate(
                    root,
                    dataset_name="nksid",
                    config={"search": {"method": method}},
                )
                self.assertTrue(status["may_start_new_claimable_search"])
                self.assertEqual(status["requested_method"], method)

    def test_approval_cannot_authorize_a_different_requested_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_gate0(root)
            (root / "artifacts/hw_surrogate_calibration_v2").mkdir(parents=True)
            (root / "results/g4_rl_arch_193_fold1_seed42").mkdir(parents=True)
            (root / "artifacts/hw_surrogate_calibration_v2/calibration_v2.json").write_text(
                json.dumps({"g2_pass": True}), encoding="utf-8"
            )
            (root / "results/g4_rl_arch_193_fold1_seed42/board_validation_summary.json").write_text(
                json.dumps({"claimable": True, "numeric_mismatch_count": 0}),
                encoding="utf-8",
            )
            (root / "artifacts/stage3_replan_approval.json").write_text(
                json.dumps({"approved": True, "methods": ["aging_evolution"]}),
                encoding="utf-8",
            )
            status = stage3_gate_status(
                root,
                config={"search": {"method": "rl"}},
            )
            self.assertFalse(status["gates"]["stage3_requested_method_approved"])
            with self.assertRaisesRegex(RuntimeError, "requested_method_approved"):
                require_stage3_search_gate(
                    root,
                    dataset_name="nksid",
                    config={"search": {"method": "rl"}},
                )

    def test_allowed_method_registry_contains_retained_rl_and_aging(self) -> None:
        self.assertIn("rl", STAGE3_ALLOWED_METHODS)
        self.assertIn("aging_evolution", STAGE3_ALLOWED_METHODS)

    def test_gate0_must_be_formally_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_gate0(root, gate_status="not_ready", completed=1199)
            status = stage3_gate_status(root)
            self.assertFalse(status["gates"]["gate0_proxy_reliability_pass"])
            self.assertEqual(status["gate0_progress"]["formal_completed_work_units"], 1199)

    def test_g5_is_required_only_when_sonar_operators_are_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status = stage3_gate_status(
                root,
                config={"search_space": {"op_choices": ["mbconv", "denoise"]}},
            )
            self.assertTrue(status["g5_required"])
            self.assertFalse(status["gates"]["g5_sonar_operator_pass_or_not_required"])
            _write_g5(root, overall_pass=True)
            passed_g5 = stage3_gate_status(
                root,
                config={"search_space": {"op_choices": ["mbconv", "denoise"]}},
            )
            self.assertTrue(passed_g5["gates"]["g5_sonar_operator_pass_or_not_required"])
            no_sonar = stage3_gate_status(
                root,
                config={"search_space": {"op_choices": ["mbconv", "skip"]}},
            )
            self.assertFalse(no_sonar["g5_required"])
            self.assertTrue(no_sonar["gates"]["g5_sonar_operator_pass_or_not_required"])


if __name__ == "__main__":
    unittest.main()
