import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware.acceptance_audit import (  # noqa: E402
    GapAcceptanceAuditInputs,
    build_gap_acceptance_audit,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def structured_route_gate(status: str = "PASS") -> dict:
    return {
        "gate_status": status,
        "route_success": status == "PASS",
        "timing_clean": status == "PASS",
        "wns_ns": 0.1 if status == "PASS" else None,
        "tns_ns": 0.0 if status == "PASS" else None,
        "global_congestion_level": 3 if status == "PASS" else None,
        "congestion_acceptable": status == "PASS",
        "utilization": {"lut": 1, "ff": 1, "dsp": 1, "bram": 1},
        "fanout": {
            "max_fanout": 8 if status == "PASS" else None,
            "high_fanout_net_count": 0,
            "hints": [],
            "report": "",
        },
        "report_paths": {
            "timing": "",
            "routed_timing": "",
            "utilization": "",
            "fanout": "",
            "stdout_log": "",
            "stderr_log": "",
        },
        "failure_reason": "" if status == "PASS" else "missing_build_result",
    }


class GapAcceptanceAuditTests(unittest.TestCase):
    def test_audit_marks_current_boundary_without_overclaiming(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "physical.yaml"
            legacy_path = root / "legacy_pareto.json"
            lut_stats_path = root / "lut_stats.json"
            main_pareto_path = root / "main_pareto.json"
            proof_pareto_path = root / "proof_pareto.json"
            ranked_csv_path = root / "pareto_ranked_candidates.csv"
            route_summary_path = root / "route_gate_summary.json"
            final_manifest_path = root / "final_candidate_manifest.json"
            final_validation_path = root / "final_validation.json"
            measurement_path = root / "measurement.json"

            config = {
                "dataset": {"image_size": 224, "input_channels": 1, "num_classes": 8},
                "search_space": {
                    "input_channels": 1,
                    "image_size": 224,
                    "stem_channels": 32,
                    "stem_stride": 2,
                    "stage_strides": [1],
                    "stage_base_channels": [24],
                    "width_multipliers": [1.0],
                    "stage_depth_choices": [[1]],
                    "op_choices": ["mbconv"],
                    "kernel_choices": [3],
                    "expand_choices": [3, 6],
                    "stage_block_choices": [
                        [
                            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
                            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
                        ]
                    ],
                },
                "constraints": {
                    "physical": {
                        "enabled": True,
                        "early_expand_limits": [
                            {"min_input_resolution": 112, "max_expand_ratio": 3}
                        ],
                        "pareto_physical_risk": True,
                    }
                },
                "search": {
                    "objective_weights": {
                        "accuracy": 1.0,
                        "physical_risk": 0.0,
                        "early_expand_pressure": 0.0,
                    }
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            encoding = {
                "input_channels": 1,
                "stem_channels": 32,
                "stem_stride": 2,
                "stages": [
                    {
                        "channels": 24,
                        "depth": 1,
                        "stride": 1,
                        "blocks": [
                            {
                                "op": "mbconv",
                                "kernel_size": 3,
                                "expand_ratio": 6,
                                "stride": 1,
                            }
                        ],
                    }
                ],
                "num_classes": 8,
            }
            write_json(
                legacy_path,
                {
                    "selected_topk": [
                        {
                            "arch_id": "rl_arch_299",
                            "encoding": encoding,
                            "metrics": {"macro_f1": 0.65},
                        }
                    ]
                },
            )
            write_json(
                lut_stats_path,
                {"true_misses": 0, "deferred_hits": 0, "strict_formal_lut": True, "hits": 10, "total": 10},
            )
            write_json(
                main_pareto_path,
                {
                    "objectives": ["macro_f1", "latency_ms"],
                    "metric_columns": ["macro_f1", "latency_ms"],
                    "selected_topk": [{"arch_id": "rl_arch_299", "encoding": encoding}],
                },
            )
            write_json(
                proof_pareto_path,
                {
                    "objectives": ["macro_f1", "latency_ms", "physical_risk"],
                    "physical_objectives": ["physical_risk"],
                    "metric_columns": ["macro_f1", "latency_ms", "physical_risk"],
                    "ranked_candidates": [
                        {"arch_id": "high", "objective_values": {"physical_risk": 0.9}},
                        {"arch_id": "low", "objective_values": {"physical_risk": 0.1}},
                    ],
                },
            )
            ranked_csv_path.write_text("arch_id,macro_f1,latency_ms\nrl_arch_299,0.65,7.0\n", encoding="utf-8")
            route_gate = structured_route_gate("PASS")
            write_json(
                route_summary_path,
                {
                    "candidate_count": 1,
                    "candidates": [
                        {
                            "candidate": {"arch_id": "rl_arch_299"},
                            "quick_route_gate": structured_route_gate("PENDING"),
                            "full_route_gate": route_gate,
                        }
                    ],
                },
            )
            write_json(
                final_manifest_path,
                {
                    "selection_stage": "claimable_board_best",
                    "candidate": {"arch_id": "rl_arch_299"},
                    "route_gate": route_gate,
                },
            )
            write_json(
                final_validation_path,
                {
                    "nas_lut_estimate": {
                        "latency_ms": 7.286,
                        "latency_scope": "NAS strict LUT aggregate estimate only; not board e2e latency.",
                    },
                    "real_board_e2e": {
                        "latency_source": "COM5 measurement JSON from full-network bitstream",
                        "serial_port": "COM5",
                        "status_code": 0,
                        "e2e_cycles": 9812402,
                        "e2e_latency_ms": 49.062,
                    },
                },
            )
            write_json(
                measurement_path,
                {
                    "serial_port": "COM5",
                    "frame": {"status_code": 0, "cycles": 9812402},
                    "latency_ms": 49.062,
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(
                    physical_config_path=str(config_path),
                    legacy_candidate_source_path=str(legacy_path),
                    strict_lut_stats_path=str(lut_stats_path),
                    pareto_selection_path=str(main_pareto_path),
                    pareto_proof_selection_path=str(proof_pareto_path),
                    pareto_ranked_csv_path=str(ranked_csv_path),
                    route_gate_summary_path=str(route_summary_path),
                    final_candidate_manifest_path=str(final_manifest_path),
                    final_validation_report_path=str(final_validation_path),
                    measurement_json_path=str(measurement_path),
                )
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        self.assertEqual(criteria["phase0_legacy_expand6_infeasible"]["status"], "PASS")
        self.assertEqual(criteria["physical_proxy_out_of_main_reward"]["status"], "PASS")
        self.assertEqual(criteria["strict_lut_no_misses_or_deferred_hits"]["status"], "PASS")
        self.assertEqual(criteria["pareto_physical_risk_ranking"]["status"], "WEAK")
        self.assertEqual(criteria["vivado_gate_structured_json"]["status"], "WEAK")
        self.assertEqual(criteria["final_candidate_full_route_clean"]["status"], "PASS")
        self.assertEqual(criteria["com5_measurement_status_zero"]["status"], "PASS")
        self.assertEqual(criteria["latency_scope_distinction"]["status"], "PASS")
        self.assertEqual(audit["status"], "in_progress")

    def test_audit_fails_when_physical_proxy_weight_remains_in_reward(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "physical.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"image_size": 224, "input_channels": 1, "num_classes": 8},
                        "search_space": {"stage_strides": [1]},
                        "constraints": {
                            "physical": {
                                "enabled": True,
                                "early_expand_limits": [
                                    {"min_input_resolution": 112, "max_expand_ratio": 3}
                                ],
                                "pareto_physical_risk": True,
                            }
                        },
                        "search": {"objective_weights": {"physical_risk": 1.0}},
                    }
                ),
                encoding="utf-8",
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(physical_config_path=str(config_path))
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        self.assertEqual(criteria["physical_proxy_out_of_main_reward"]["status"], "FAIL")

    def test_vivado_gate_requires_fanout_and_congestion_structured_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route_summary = root / "route_gate_summary.json"
            incomplete_gate = structured_route_gate("PASS")
            incomplete_gate.pop("fanout")
            write_json(
                route_summary,
                {
                    "candidate_count": 1,
                    "candidates": [
                        {
                            "candidate": {"arch_id": "missing_fanout"},
                            "quick_route_gate": structured_route_gate("PASS"),
                            "full_route_gate": incomplete_gate,
                        }
                    ],
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(route_gate_summary_path=str(route_summary))
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        vivado = criteria["vivado_gate_structured_json"]
        self.assertEqual(vivado["status"], "FAIL")
        self.assertIn("fanout", vivado["message"])

    def test_audit_treats_fresh_smoke_pareto_as_weak_not_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fresh_pareto = root / "fresh_pareto.json"
            fresh_csv = root / "fresh_ranked.csv"
            fresh_summary = root / "summary.json"
            write_json(
                fresh_pareto,
                {
                    "objectives": ["macro_f1", "latency_ms", "physical_risk"],
                    "physical_objectives": ["physical_risk"],
                    "metric_columns": ["macro_f1", "latency_ms", "physical_risk"],
                    "selected_count": 1,
                    "ranked_candidates": [
                        {
                            "arch_id": "fresh_arch",
                            "objective_values": {
                                "macro_f1": 0.1,
                                "latency_ms": 5.0,
                                "physical_risk": 0.2,
                            },
                        }
                    ],
                },
            )
            fresh_csv.write_text("arch_id,macro_f1,latency_ms,physical_risk\nfresh_arch,0.1,5.0,0.2\n", encoding="utf-8")
            write_json(
                fresh_summary,
                {
                    "status": "completed",
                    "total_evaluated": 8,
                    "feasible": 8,
                    "infeasible": 0,
                    "extra": {
                        "search_method": "random",
                        "num_candidates": 8,
                        "train_epochs": 1,
                        "search_summary": {
                            "total_evaluated": 8,
                            "feasible": 8,
                            "infeasible": 0,
                        },
                    },
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(
                    fresh_search_pareto_selection_path=str(fresh_pareto),
                    fresh_search_ranked_csv_path=str(fresh_csv),
                    fresh_search_summary_path=str(fresh_summary),
                )
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        pareto = criteria["pareto_physical_risk_ranking"]
        self.assertEqual(pareto["status"], "WEAK")
        self.assertFalse(pareto["details"]["fresh_search_is_full_phase0"])
        self.assertIn("smoke search", pareto["message"])

    def test_full_phase0_requires_actual_evaluated_episodes_not_config_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fresh_pareto = root / "fresh_pareto.json"
            fresh_csv = root / "fresh_ranked.csv"
            fresh_summary = root / "summary.json"
            write_json(
                fresh_pareto,
                {
                    "objectives": ["macro_f1", "latency_ms", "physical_risk"],
                    "physical_objectives": ["physical_risk"],
                    "metric_columns": ["macro_f1", "latency_ms", "physical_risk"],
                    "selected_count": 1,
                    "ranked_candidates": [
                        {
                            "arch_id": "rl_smoke",
                            "objective_values": {
                                "macro_f1": 0.4,
                                "latency_ms": 5.0,
                                "physical_risk": 0.2,
                            },
                        }
                    ],
                },
            )
            fresh_csv.write_text("arch_id,macro_f1,latency_ms,physical_risk\nrl_smoke,0.4,5.0,0.2\n", encoding="utf-8")
            write_json(
                fresh_summary,
                {
                    "status": "completed",
                    "total_evaluated": 8,
                    "feasible": 8,
                    "infeasible": 0,
                    "extra": {
                        "search_method": "rl",
                        "num_candidates": 300,
                        "num_episodes": 8,
                        "train_epochs": 1,
                        "search_summary": {
                            "total_evaluated": 8,
                            "feasible": 8,
                            "infeasible": 0,
                        },
                    },
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(
                    fresh_search_pareto_selection_path=str(fresh_pareto),
                    fresh_search_ranked_csv_path=str(fresh_csv),
                    fresh_search_summary_path=str(fresh_summary),
                )
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        pareto = criteria["pareto_physical_risk_ranking"]
        self.assertEqual(pareto["status"], "WEAK")
        self.assertFalse(pareto["details"]["fresh_search_is_full_phase0"])
        self.assertEqual(pareto["details"]["fresh_search_scope"]["num_candidates"], 300)
        self.assertEqual(pareto["details"]["fresh_search_scope"]["num_episodes"], 8)

    def test_full_phase0_fresh_pareto_can_satisfy_pareto_requirement(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fresh_pareto = root / "fresh_pareto.json"
            fresh_csv = root / "fresh_ranked.csv"
            fresh_summary = root / "summary.json"
            write_json(
                fresh_pareto,
                {
                    "objectives": ["macro_f1", "latency_ms", "physical_risk"],
                    "physical_objectives": ["physical_risk"],
                    "metric_columns": ["macro_f1", "latency_ms", "physical_risk"],
                    "selected_count": 1,
                    "ranked_candidates": [
                        {
                            "arch_id": "rl_full",
                            "objective_values": {
                                "macro_f1": 0.6,
                                "latency_ms": 5.0,
                                "physical_risk": 0.2,
                            },
                        }
                    ],
                },
            )
            fresh_csv.write_text("arch_id,macro_f1,latency_ms,physical_risk\nrl_full,0.6,5.0,0.2\n", encoding="utf-8")
            write_json(
                fresh_summary,
                {
                    "status": "completed",
                    "total_evaluated": 300,
                    "feasible": 200,
                    "infeasible": 100,
                    "extra": {
                        "search_method": "rl",
                        "num_candidates": 300,
                        "num_episodes": 300,
                        "train_epochs": 10,
                        "device": "cuda",
                        "search_summary": {
                            "total_evaluated": 300,
                            "feasible": 200,
                            "infeasible": 100,
                        },
                    },
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(
                    fresh_search_pareto_selection_path=str(fresh_pareto),
                    fresh_search_ranked_csv_path=str(fresh_csv),
                    fresh_search_summary_path=str(fresh_summary),
                )
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        pareto = criteria["pareto_physical_risk_ranking"]
        self.assertEqual(pareto["status"], "PASS")
        self.assertTrue(pareto["details"]["fresh_search_is_full_phase0"])

    def test_running_full_phase0_status_is_weak_not_complete(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "remote_status_latest.json"
            fetch_error_path = root / "fetch_error_latest.json"
            write_json(
                status_path,
                {
                    "captured_at_utc": "2026-06-02T02:57:03Z",
                    "summary_exists": False,
                    "pareto_exists": False,
                    "ranked_csv_exists": False,
                    "lut_stats_exists": False,
                    "route_plan_exists": False,
                    "route_summary_exists": False,
                    "search_state": {
                        "total_evaluated": 55,
                        "feasible": 55,
                        "infeasible": 0,
                        "completed": None,
                    },
                },
            )
            write_json(
                fetch_error_path,
                {
                    "generated_at_utc": None,
                    "status": {
                        "connection_status": "failed",
                        "error_type": "NoValidConnectionsError",
                        "error": "Unable to connect",
                        "latest_reliable_status_exists": True,
                    },
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(
                    fresh_search_status_path=str(status_path),
                    fresh_fetch_error_path=str(fetch_error_path),
                )
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        pareto = criteria["pareto_physical_risk_ranking"]
        self.assertEqual(pareto["status"], "WEAK")
        self.assertEqual(pareto["details"]["fresh_search_status"]["search_state"]["total_evaluated"], 55)
        self.assertEqual(pareto["details"]["fresh_fetch_error"]["connection_status"], "failed")
        self.assertIn("running", pareto["message"])

    def test_final_report_and_measurement_paths_can_be_derived_from_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_manifest = root / "final_candidate_manifest.json"
            final_validation = root / "reports" / "001.final_hardware_validation_report.json"
            measurement = root / "measurements" / "001.measurement.json"
            route_gate = {
                **structured_route_gate("PASS"),
            }
            write_json(
                final_manifest,
                {
                    "selection_stage": "claimable_board_best",
                    "candidate": {"arch_id": "rl_full"},
                    "route_gate": route_gate,
                    "artifacts": {
                        "final_validation_report_json": str(final_validation),
                        "measurement_json": str(measurement),
                    },
                },
            )
            write_json(
                final_validation,
                {
                    "nas_lut_estimate": {
                        "latency_ms": 5.5,
                        "latency_scope": "NAS strict LUT aggregate estimate only; not board e2e latency.",
                    },
                    "real_board_e2e": {
                        "latency_source": "COM5 measurement JSON from full-network bitstream",
                        "serial_port": "COM5",
                        "status_code": 0,
                        "e2e_cycles": 123,
                        "e2e_latency_ms": 0.615,
                    },
                },
            )
            write_json(
                measurement,
                {
                    "serial_port": "COM5",
                    "frame": {"status_code": 0, "cycles": 123},
                    "latency_ms": 0.615,
                },
            )

            audit = build_gap_acceptance_audit(
                GapAcceptanceAuditInputs(final_candidate_manifest_path=str(final_manifest))
            )

        criteria = {item["id"]: item for item in audit["acceptance_criteria"]}
        self.assertEqual(criteria["final_candidate_full_route_clean"]["status"], "PASS")
        self.assertEqual(criteria["com5_measurement_status_zero"]["status"], "PASS")
        self.assertEqual(criteria["latency_scope_distinction"]["status"], "PASS")
        self.assertEqual(audit["inputs"]["final_validation_report_path"], str(final_validation))
        self.assertEqual(audit["inputs"]["measurement_json_path"], str(measurement))


if __name__ == "__main__":
    unittest.main()
