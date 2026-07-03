import json
import importlib.util
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_phase0_full_route_gate.py"
POLL_SCRIPT = REPO_ROOT / "scripts" / "poll_phase0_full_pipeline.py"
SONAR_READINESS_SCRIPT = REPO_ROOT / "scripts" / "prepare_phase0_v4_sonar_launch_readiness.py"
SONAR_PIPELINE_POLL_SCRIPT = REPO_ROOT / "scripts" / "poll_phase0_v4_sonar_pipeline.py"
PARETO_PROOF = REPO_ROOT / "results" / "phase1_pareto_physical_risk_minirepro" / "pareto_selection.json"


def load_poll_module():
    spec = importlib.util.spec_from_file_location("poll_phase0_full_pipeline", POLL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sonar_poll_module():
    spec = importlib.util.spec_from_file_location("poll_phase0_v4_sonar_pipeline", SONAR_PIPELINE_POLL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase0FullPipelineScriptTests(unittest.TestCase):
    def test_remote_status_freshness_marks_old_status_stale(self) -> None:
        poll = load_poll_module()
        freshness = poll.compute_remote_status_freshness(
            "2026-06-02T02:57:03Z",
            now=datetime(2026, 6, 2, 4, 57, 4, tzinfo=timezone.utc),
            stale_after_seconds=3600,
        )

        self.assertEqual(freshness["age_seconds"], 7201)
        self.assertTrue(freshness["is_stale"])

    def test_prepare_route_gate_waits_when_full_pareto_is_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "route_gate"
            missing_pareto = Path(tmpdir) / "missing_pareto.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREPARE_SCRIPT),
                    "--pareto-selection",
                    str(missing_pareto),
                    "--output-root",
                    str(output_root),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "waiting_for_pareto_selection")
            self.assertFalse((output_root / "route_gate_summary.json").exists())
            status_payload = json.loads(
                (output_root / "prepare_phase0_full_route_gate_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["status"], "waiting_for_pareto_selection")

    def test_prepare_route_gate_records_multiple_lowdsp_catalogs_when_waiting(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "route_gate"
            missing_pareto = Path(tmpdir) / "missing_pareto.json"
            v3_catalog = REPO_ROOT / "hls_lut_builder/board_harness/configs/phase0_v3_lowdsp_override_catalog.yaml"
            sonar_catalog = (
                REPO_ROOT
                / "hls_lut_builder/board_harness/configs/phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREPARE_SCRIPT),
                    "--pareto-selection",
                    str(missing_pareto),
                    "--output-root",
                    str(output_root),
                    "--lowdsp-override-catalog",
                    str(v3_catalog),
                    "--lowdsp-override-catalog",
                    str(sonar_catalog),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "waiting_for_pareto_selection")
            self.assertEqual(
                payload["lowdsp_override_catalog"],
                [str(v3_catalog), str(sonar_catalog)],
            )

    def test_sonar_launch_readiness_writes_manual_command_bundle_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "readiness"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SONAR_READINESS_SCRIPT),
                    "--output-dir",
                    str(output_dir),
                    "--python",
                    "python",
                    "--require-ready",
                ],
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "READY_FOR_MANUAL_SERVER_LAUNCH")
            self.assertFalse(payload["runs_server"])
            self.assertFalse(payload["runs_training"])
            command = payload["commands"]["prepare_route_gate_after_search"]
            self.assertEqual(command.count("--lowdsp-override-catalog"), 2)
            self.assertIn("phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml", " ".join(command))
            self.assertTrue(
                (output_dir / "phase0_v4_sonar_stage3_k3_launch_readiness.json").exists()
            )

    def test_sonar_pipeline_poll_next_action_is_manual_launch_when_results_missing(self) -> None:
        sonar_poll = load_sonar_poll_module()
        actions = sonar_poll.determine_next_actions(
            {
                "readiness": {"status": "READY_FOR_MANUAL_SERVER_LAUNCH"},
                "commands": {
                    "manual_server_rl300_launch": ["python", "run_search.py"],
                    "manual_server_rl300_resume": ["python", "run_search.py", "--resume"],
                },
                "local_search_artifacts": {
                    "summary": False,
                    "pareto_selection": False,
                    "pareto_ranked_candidates": False,
                    "lut_stats": False,
                },
                "local_route_gate_artifacts": {},
            }
        )

        self.assertEqual(actions[0]["id"], "manual_server_rl300_launch_or_fetch_results")
        self.assertEqual(actions[0]["kind"], "manual_server")
        self.assertIn("pareto_selection", actions[0]["missing"])

    def test_sonar_pipeline_poll_next_action_prepares_route_gate_after_search(self) -> None:
        sonar_poll = load_sonar_poll_module()
        actions = sonar_poll.determine_next_actions(
            {
                "readiness": {"status": "READY_FOR_MANUAL_SERVER_LAUNCH"},
                "commands": {
                    "prepare_route_gate_after_search": [
                        "python",
                        "scripts/prepare_phase0_full_route_gate.py",
                    ],
                },
                "local_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": False,
                },
            }
        )

        self.assertEqual(actions[0]["id"], "prepare_route_gate_pending_summary")
        self.assertEqual(actions[0]["kind"], "local")

    def test_prepare_route_gate_marks_uncovered_full_route_skipped_when_pareto_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "route_gate"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREPARE_SCRIPT),
                    "--pareto-selection",
                    str(PARETO_PROOF),
                    "--output-root",
                    str(output_root),
                ],
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "prepared")
            summary = json.loads((output_root / "route_gate_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["full_route_clean_count"], 0)
            self.assertEqual(summary["claimable_real_e2e_latency_count"], 0)
            statuses = [
                (
                    row["quick_route_gate"]["gate_status"],
                    row["full_route_gate"]["gate_status"],
                )
                for row in summary["candidates"]
            ]
            self.assertEqual(statuses, [("PENDING", "FAIL")])
            self.assertEqual(summary["full_route_skipped_count"], 1)
            self.assertEqual(summary["lowdsp_coverage_missing_count"], 1)
            self.assertEqual(
                summary["candidates"][0]["full_route_gate"]["failure_reason"],
                "lowdsp_coverage_missing",
            )
            self.assertTrue((output_root / "route_gate_plan.json").exists())
            self.assertTrue((output_root / "final_candidate_manifest.json").exists())

    def test_poll_next_actions_prioritize_remote_fetch_until_full_results_exist(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {
                    "status": "failed",
                    "error_type": "NoValidConnectionsError",
                },
                "remote_status_freshness": {
                    "captured_at_utc": "2026-06-02T02:57:03Z",
                    "age_seconds": 7201,
                    "stale_after_seconds": 3600,
                    "is_stale": True,
                },
                "local_full_search_artifacts": {
                    "summary": False,
                    "pareto_selection": False,
                    "pareto_ranked_candidates": False,
                    "lut_stats": False,
                },
                "local_route_gate_artifacts": {},
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "restore_remote_access")
        self.assertTrue(actions[0]["evidence"]["remote_status_freshness"]["is_stale"])
        self.assertEqual(actions[1]["id"], "fetch_full_phase0_results")
        self.assertIn("pareto_selection", actions[1]["missing"])

    def test_poll_next_actions_prepare_route_gate_after_full_results_exist(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": False,
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "prepare_route_gate_pending_summary")

    def test_poll_next_actions_require_quick_before_full_route(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": True,
                    "quick_statuses": ["PENDING"],
                    "full_statuses": ["PENDING"],
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "execute_vivado_quick_gate")

    def test_poll_next_actions_require_full_route_after_quick_finishes(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": True,
                    "quick_statuses": ["PASS"],
                    "full_statuses": ["PENDING"],
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "execute_vivado_full_route_gate")

    def test_poll_next_actions_require_measurement_after_route_clean(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": True,
                    "quick_statuses": ["PASS"],
                    "full_statuses": ["PASS"],
                    "full_route_clean_count": 1,
                    "claimable_real_e2e_latency_count": 0,
                    "route_clean_action_candidate": {
                        "candidate": {"arch_id": "clean"},
                        "measurement_json": "measurements/clean.measurement.json",
                        "measurement_json_exists": False,
                        "measurement_command": [
                            "python",
                            "hls_lut_builder/board_harness/scripts/rl_best_e2e_board_validation.py",
                            "measure",
                            "--run-name",
                            "full_clean",
                            "--serial-port",
                            "COM5",
                            "--expected-bitstream-sha256",
                            "abc123",
                        ],
                        "final_report_command": [
                            "python",
                            "hls_lut_builder/board_harness/scripts/final_hardware_validation_report.py",
                        ],
                    },
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "run_com5_measurement")
        self.assertIn("--expected-bitstream-sha256", actions[0]["command"])
        self.assertIn("pareto_route_gate.py refresh", actions[0]["after_measurement_command"])
        self.assertEqual(actions[0]["evidence"]["candidate"]["arch_id"], "clean")

    def test_poll_next_actions_refreshes_when_measurement_exists_but_is_not_claimable(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": True,
                    "quick_statuses": ["PASS"],
                    "full_statuses": ["PASS"],
                    "full_route_clean_count": 1,
                    "claimable_real_e2e_latency_count": 0,
                    "route_clean_action_candidate": {
                        "candidate": {"arch_id": "clean"},
                        "measurement_json": "measurements/clean.measurement.json",
                        "measurement_json_exists": True,
                    },
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "refresh_or_review_com5_measurement")
        self.assertIn("pareto_route_gate.py refresh", actions[0]["command"])

    def test_poll_next_actions_never_skip_final_completion_audit(self) -> None:
        poll = load_poll_module()
        actions = poll.determine_next_actions(
            {
                "latest_fetch_error": {},
                "local_full_search_artifacts": {
                    "summary": True,
                    "pareto_selection": True,
                    "pareto_ranked_candidates": True,
                    "lut_stats": True,
                },
                "local_route_gate_artifacts": {
                    "route_gate_summary": True,
                    "quick_statuses": ["PASS"],
                    "full_statuses": ["PASS"],
                    "full_route_clean_count": 1,
                    "claimable_real_e2e_latency_count": 1,
                },
                "acceptance_audit": {"status": "in_progress"},
            }
        )

        self.assertEqual(actions[0]["id"], "refresh_acceptance_audit")

    def test_select_route_clean_action_candidate_uses_route_clean_best_row(self) -> None:
        poll = load_poll_module()
        with TemporaryDirectory() as tmpdir:
            measurement = Path(tmpdir) / "selected.measurement.json"
            measurement.write_text("{}", encoding="utf-8")
            summary = {
                "route_clean_best": {"selection_index": 2, "arch_id": "selected"},
                "candidates": [
                    {
                        "candidate": {"selection_index": 1, "arch_id": "first"},
                        "full_route_gate": {"gate_status": "PASS"},
                        "measurement_json": str(Path(tmpdir) / "first.measurement.json"),
                        "measurement_command": ["measure", "first"],
                    },
                    {
                        "candidate": {"selection_index": 2, "arch_id": "selected"},
                        "full_route_gate": {"gate_status": "PASS"},
                        "measurement_json": str(measurement),
                        "measurement_command": ["measure", "selected"],
                    },
                ],
            }

            selected = poll.select_route_clean_action_candidate(summary)

        self.assertEqual(selected["candidate"]["arch_id"], "selected")
        self.assertTrue(selected["measurement_json_exists"])
        self.assertEqual(selected["measurement_command"], ["measure", "selected"])


if __name__ == "__main__":
    unittest.main()
