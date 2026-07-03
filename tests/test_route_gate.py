import unittest
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware.route_gate import (
    classify_route_gate_result,
    dedupe_candidates_by_signature,
    pareto_topk_candidates,
    select_route_clean_best,
)


class RouteGateTests(unittest.TestCase):
    def _load_pareto_route_gate_script(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "hls_lut_builder"
            / "board_harness"
            / "scripts"
            / "pareto_route_gate.py"
        )
        spec = importlib.util.spec_from_file_location("pareto_route_gate_script", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _v3_encoding(
        self,
        *,
        stage1_expand: int = 3,
        stage3_op: str = "mbconv",
        stage3_expand: int = 3,
    ) -> dict:
        stage3_block = (
            {"op": "skip", "kernel_size": 1, "expand_ratio": 1, "stride": 1}
            if stage3_op == "skip"
            else {"op": stage3_op, "kernel_size": 3, "expand_ratio": stage3_expand, "stride": 1}
        )
        return {
            "input_channels": 1,
            "stem_channels": 32,
            "stem_stride": 2,
            "post_stem_downsample_stride": 1,
            "num_classes": 8,
            "stages": [
                {
                    "channels": 16,
                    "depth": 1,
                    "stride": 1,
                    "blocks": [{"op": "conv", "kernel_size": 1, "expand_ratio": 1, "stride": 1}],
                },
                {
                    "channels": 24,
                    "depth": 1,
                    "stride": 2,
                    "blocks": [
                        {
                            "op": "mbconv",
                            "kernel_size": 3,
                            "expand_ratio": stage1_expand,
                            "stride": 2,
                        }
                    ],
                },
                {
                    "channels": 32,
                    "depth": 1,
                    "stride": 2,
                    "blocks": [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2}],
                },
                {
                    "channels": 32,
                    "depth": 1,
                    "stride": 1,
                    "blocks": [stage3_block],
                },
            ],
        }

    def test_dedupe_pareto_topk_candidates_by_signature(self) -> None:
        pareto_payload = {
            "selected_topk": [
                {
                    "selection_index": 2,
                    "rank": 0,
                    "arch_id": "arch_b",
                    "encoding": {"layers": [2]},
                    "metrics": {"macro_f1": 0.8},
                },
                {
                    "selection_index": 1,
                    "rank": 0,
                    "arch_id": "arch_a",
                    "encoding": {"layers": [1]},
                    "metrics": {"macro_f1": 0.9},
                },
                {
                    "selection_index": 3,
                    "rank": 0,
                    "arch_id": "arch_c_same_encoding",
                    "encoding": {"layers": [1]},
                    "metrics": {"macro_f1": 0.95},
                },
                {
                    "selection_index": 1,
                    "rank": 0,
                    "arch_id": "arch_a",
                    "encoding": {"layers": [1]},
                    "metrics": {"macro_f1": 0.9},
                },
            ]
        }

        candidates = dedupe_candidates_by_signature(
            pareto_topk_candidates(pareto_payload, topk=3)
        )

        self.assertEqual([item["arch_id"] for item in candidates], ["arch_a", "arch_b"])
        self.assertTrue(candidates[0]["encoding_signature"])

    def test_pareto_topk_backfills_ranked_candidates_by_unique_encoding(self) -> None:
        pareto_payload = {
            "selected_topk": [
                {
                    "selection_index": 5,
                    "rank": 5,
                    "arch_id": "selected_shape",
                    "encoding": {"layers": [1]},
                    "metrics": {"macro_f1": 0.8},
                }
            ],
            "ranked_candidates": [
                {
                    "rank": 0,
                    "arch_id": "same_shape_different_arch_id",
                    "encoding": {"layers": [1]},
                    "metrics": {"macro_f1": 0.9},
                },
                {
                    "rank": 1,
                    "arch_id": "ranked_backfill",
                    "encoding": {"layers": [2]},
                    "metrics": {"macro_f1": 0.7},
                },
                {
                    "rank": 2,
                    "arch_id": "ranked_extra",
                    "encoding": {"layers": [3]},
                    "metrics": {"macro_f1": 0.6},
                },
            ],
        }

        candidates = pareto_topk_candidates(pareto_payload, topk=2)

        self.assertEqual(
            [item["arch_id"] for item in candidates],
            ["selected_shape", "ranked_backfill"],
        )
        self.assertEqual(
            len({item["encoding_signature"] for item in candidates}),
            2,
        )

    def test_classify_route_gate_result_pass_borderline_fail_pending(self) -> None:
        pending = classify_route_gate_result({})
        self.assertEqual(pending["gate_status"], "PENDING")
        self.assertIn("fanout", pending)
        self.assertIn("utilization", pending)

        passed = classify_route_gate_result(
            {
                "status": "success",
                "wns_ns": "0.012",
                "timing_clean": True,
                "global_congestion_level": "4",
                "lut": "1024",
                "ff": "2048",
                "bram_tile": "12",
                "dsp": "34",
                "max_fanout": "87",
                "high_fanout_or_congestion_hints": [
                    "high fanout net rst_n fanout=87",
                    "global congestion level is 4",
                ],
            }
        )
        self.assertEqual(passed["gate_status"], "PASS")
        self.assertEqual(passed["utilization"]["bram"], "12")
        self.assertEqual(passed["fanout"]["max_fanout"], 87)
        self.assertEqual(passed["fanout"]["high_fanout_net_count"], 1)
        self.assertEqual(
            classify_route_gate_result(
                {
                    "status": "success",
                    "wns_ns": "-0.1",
                    "timing_clean": False,
                    "global_congestion_level": "4",
                }
            )["gate_status"],
            "BORDERLINE",
        )
        self.assertEqual(
            classify_route_gate_result(
                {
                    "status": "success",
                    "wns_ns": "-1.0",
                    "timing_clean": False,
                    "global_congestion_level": "6",
                }
            )["gate_status"],
            "FAIL",
        )

    def test_full_route_gate_rejects_actual_dsp_above_limit(self) -> None:
        over_budget = classify_route_gate_result(
            {
                "status": "success",
                "wns_ns": "0.1",
                "timing_clean": True,
                "global_congestion_level": "4",
                "dsp": "701",
            },
            max_actual_dsp=700,
        )
        self.assertEqual(over_budget["gate_status"], "FAIL")
        self.assertEqual(over_budget["actual_dsp"], 701)
        self.assertFalse(over_budget["actual_dsp_acceptable"])
        self.assertIn("actual_dsp_701_exceeds_700", over_budget["failure_reason"])

        in_budget = classify_route_gate_result(
            {
                "status": "success",
                "wns_ns": "0.1",
                "timing_clean": True,
                "global_congestion_level": "4",
                "dsp": "700",
            },
            max_actual_dsp=700,
        )
        self.assertEqual(in_budget["gate_status"], "PASS")
        self.assertTrue(in_budget["actual_dsp_acceptable"])

    def test_select_route_clean_best_requires_full_route_pass(self) -> None:
        rows = [
            {
                "candidate": {
                    "selection_index": 1,
                    "rank": 0,
                    "arch_id": "not_clean",
                    "metrics": {"macro_f1": 0.95},
                },
                "full_route_gate": {"gate_status": "FAIL"},
            },
            {
                "candidate": {
                    "selection_index": 2,
                    "rank": 0,
                    "arch_id": "clean",
                    "metrics": {"macro_f1": 0.90},
                },
                "full_route_gate": {"gate_status": "PASS"},
            },
        ]

        selected = select_route_clean_best(rows)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"]["arch_id"], "clean")

    def test_select_route_clean_best_rejects_pass_rows_above_actual_dsp_limit(self) -> None:
        rows = [
            {
                "candidate": {
                    "selection_index": 1,
                    "rank": 0,
                    "arch_id": "pass_but_dsp_full",
                    "metrics": {"macro_f1": 0.95},
                },
                "full_route_gate": {
                    "gate_status": "PASS",
                    "actual_dsp": 840,
                    "actual_dsp_acceptable": False,
                    "utilization": {"dsp": "840"},
                },
            },
            {
                "candidate": {
                    "selection_index": 2,
                    "rank": 1,
                    "arch_id": "pass_lowdsp",
                    "metrics": {"macro_f1": 0.90},
                },
                "full_route_gate": {
                    "gate_status": "PASS",
                    "actual_dsp": 612,
                    "actual_dsp_acceptable": True,
                    "utilization": {"dsp": "612"},
                },
            },
        ]

        selected = select_route_clean_best(rows, max_actual_dsp=700)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"]["arch_id"], "pass_lowdsp")

    def test_select_route_clean_best_prioritizes_metrics_after_gate(self) -> None:
        rows = [
            {
                "candidate": {
                    "selection_index": 1,
                    "rank": 0,
                    "arch_id": "lower_macro_front",
                    "metrics": {
                        "macro_f1": 0.84,
                        "top1": 0.90,
                        "latency_ms": 5.0,
                        "physical_risk": 0.1,
                    },
                },
                "full_route_gate": {"gate_status": "PASS"},
            },
            {
                "candidate": {
                    "selection_index": 2,
                    "rank": 1,
                    "arch_id": "higher_macro_clean",
                    "metrics": {
                        "macro_f1": 0.88,
                        "top1": 0.89,
                        "latency_ms": 8.0,
                        "physical_risk": 0.4,
                    },
                },
                "full_route_gate": {"gate_status": "PASS"},
            },
        ]

        selected = select_route_clean_best(rows)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"]["arch_id"], "higher_macro_clean")

    def test_pareto_route_gate_imports_implementation_variant_manifest(self) -> None:
        module = self._load_pareto_route_gate_script()
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "harness_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_name": "arch_a_lowdsp",
                        "case_name": "target_arch_a_lowdsp",
                        "timing_variant": "place_altspread_route_more_global",
                        "impl_profile": "fanout_only",
                        "argmax_pipeline": True,
                        "completion_mode": "sink_and_argmax",
                        "harness_fifo_depth": 64,
                        "harness_fifo_style": "lut_registered",
                        "source_candidate": {"arch_id": "arch_a"},
                        "component_overrides": [
                            {
                                "role": "l3_stage1_block0_pw_expand",
                                "override_case_name": "pwlow_case",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            variants = module.load_implementation_variants([str(manifest_path)])

        variant = variants["arch_a"]
        self.assertEqual(variant["run_name"], "arch_a_lowdsp")
        self.assertEqual(variant["impl_profile"], "fanout_only")
        self.assertEqual(
            variant["component_overrides"],
            ["l3_stage1_block0_pw_expand=pwlow_case"],
        )

        args = SimpleNamespace(
            python="python",
            validation_script="validate.py",
            force=False,
        )
        command = module.build_command(
            args=args,
            candidate_path=Path("candidate.json"),
            run_name_value=variant["run_name"],
            timing_variant=variant["timing_variant"],
            impl_profile=variant["impl_profile"],
            timeout_minutes=240,
            component_overrides=variant["component_overrides"],
            argmax_pipeline=variant["argmax_pipeline"],
            completion_mode=variant["completion_mode"],
            fifo_depth=variant["fifo_depth"],
            fifo_style=variant["fifo_style"],
        )
        self.assertIn("--component-override", command)
        self.assertIn("l3_stage1_block0_pw_expand=pwlow_case", command)
        self.assertIn("--argmax-pipeline", command)
        self.assertIn("sink_and_argmax", command)

    def test_pareto_route_gate_plans_separate_quick_and_full_variants(self) -> None:
        module = self._load_pareto_route_gate_script()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pareto_path = tmp_path / "pareto_selection.json"
            pareto_path.write_text(
                json.dumps(
                    {
                        "selected_topk": [
                            {
                                "selection_index": 1,
                                "rank": 0,
                                "arch_id": "arch_a",
                                "encoding": {"layers": [1]},
                                "metrics": {"macro_f1": 0.9},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            quick_manifest = tmp_path / "quick_manifest.json"
            quick_manifest.write_text(
                json.dumps(
                    {
                        "run_name": "arch_a_default_failed",
                        "timing_variant": "default_timing",
                        "impl_profile": "default",
                        "source_candidate": {"arch_id": "arch_a"},
                    }
                ),
                encoding="utf-8",
            )
            full_manifest = tmp_path / "full_manifest.json"
            full_manifest.write_text(
                json.dumps(
                    {
                        "run_name": "arch_a_lowdsp_pass",
                        "timing_variant": "full_timing",
                        "impl_profile": "fanout_only",
                        "source_candidate": {"arch_id": "arch_a"},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_root=str(tmp_path / "route_gate"),
                pareto_selection=str(pareto_path),
                validation_script="validate.py",
                final_report_script="report.py",
                implementation_variant_manifest=[],
                quick_implementation_variant_manifest=[str(quick_manifest)],
                full_implementation_variant_manifest=[str(full_manifest)],
                python="python",
                topk=8,
                full_topk=5,
                serial_port="COM5",
                quick_timing_variant="quick_default",
                full_timing_variant="full_default",
                quick_impl_profile="default",
                full_impl_profile="default",
                quick_timeout_minutes=180,
                full_timeout_minutes=240,
                force=False,
            )

            plan = module.prepare_plan(args)

        item = plan["candidates"][0]
        self.assertEqual(item["quick_run_name"], "arch_a_default_failed")
        self.assertEqual(item["full_run_name"], "arch_a_lowdsp_pass")
        self.assertEqual(item["quick_implementation_variant"]["impl_profile"], "default")
        self.assertEqual(item["full_implementation_variant"]["impl_profile"], "fanout_only")
        self.assertIn("arch_a_default_failed", item["quick_build_result"])
        self.assertIn("arch_a_lowdsp_pass", item["full_build_result"])

    def test_pareto_route_gate_reuses_variant_by_encoding_signature(self) -> None:
        module = self._load_pareto_route_gate_script()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            old_candidate_path = tmp_path / "old_candidate.json"
            old_candidate_path.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "arch_id": "old_arch_id",
                            "encoding": {"layers": [1, 2, 3]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            pareto_path = tmp_path / "pareto_selection.json"
            pareto_path.write_text(
                json.dumps(
                    {
                        "selected_topk": [
                            {
                                "selection_index": 1,
                                "rank": 0,
                                "arch_id": "new_arch_id",
                                "encoding": {"layers": [1, 2, 3]},
                                "metrics": {"macro_f1": 0.9},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            quick_manifest = tmp_path / "quick_manifest.json"
            quick_manifest.write_text(
                json.dumps(
                    {
                        "run_name": "old_quick_same_encoding",
                        "timing_variant": "default_timing",
                        "impl_profile": "default",
                        "source_candidate": {
                            "arch_id": "old_arch_id",
                            "candidate_path": str(old_candidate_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_root=str(tmp_path / "route_gate"),
                pareto_selection=str(pareto_path),
                validation_script="validate.py",
                final_report_script="report.py",
                implementation_variant_manifest=[],
                quick_implementation_variant_manifest=[str(quick_manifest)],
                full_implementation_variant_manifest=[],
                python="python",
                topk=8,
                full_topk=5,
                serial_port="COM5",
                quick_timing_variant="quick_default",
                full_timing_variant="full_default",
                quick_impl_profile="default",
                full_impl_profile="fanout_only",
                quick_timeout_minutes=180,
                full_timeout_minutes=240,
                force=False,
            )

            plan = module.prepare_plan(args)

        item = plan["candidates"][0]
        self.assertEqual(item["arch_id"], "new_arch_id")
        self.assertEqual(item["quick_run_name"], "old_quick_same_encoding")
        self.assertEqual(item["quick_implementation_variant"]["arch_id"], "old_arch_id")
        self.assertEqual(
            item["quick_implementation_variant"]["encoding_signature"],
            item["encoding_signature"],
        )

    def test_lowdsp_override_resolver_covers_v3_stage_combinations(self) -> None:
        module = self._load_pareto_route_gate_script()
        catalog_path = (
            Path(__file__).resolve().parents[1]
            / "hls_lut_builder"
            / "board_harness"
            / "configs"
            / "phase0_v3_lowdsp_override_catalog.yaml"
        )
        catalog = module.load_lowdsp_override_catalog([str(catalog_path)])

        for stage1_expand, stage3_op in (
            (3, "mbconv"),
            (6, "mbconv"),
            (3, "skip"),
            (6, "skip"),
        ):
            with self.subTest(stage1_expand=stage1_expand, stage3_op=stage3_op):
                coverage = module.resolve_lowdsp_overrides(
                    self._v3_encoding(stage1_expand=stage1_expand, stage3_op=stage3_op),
                    catalog,
                )
                self.assertEqual(coverage["status"], "covered")
                self.assertFalse(coverage["missing"])
                self.assertIn(
                    "l2_stage0_block0_direct=pwlow_a84rec_l2_proj_112_32_16_p8_pe8_simd1_ii4_dsp1_c5",
                    coverage["component_overrides"],
                )

        e6_coverage = module.resolve_lowdsp_overrides(
            self._v3_encoding(stage1_expand=6, stage3_op="mbconv"),
            catalog,
        )
        self.assertIn(
            "l3_stage1_block0_pw_expand=pwlow_rlb_l3_pwe_112_16_96_p8_pe8_simd1_ii4_dsp1_c5",
            e6_coverage["component_overrides"],
        )

    def test_lowdsp_override_resolver_covers_stage3_sonar_direct_overrides(self) -> None:
        module = self._load_pareto_route_gate_script()
        root = Path(__file__).resolve().parents[1]
        catalog = module.load_lowdsp_override_catalog(
            [
                str(
                    root
                    / "hls_lut_builder"
                    / "board_harness"
                    / "configs"
                    / "phase0_v3_lowdsp_override_catalog.yaml"
                ),
                str(
                    root
                    / "hls_lut_builder"
                    / "board_harness"
                    / "configs"
                    / "phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml"
                ),
            ]
        )

        denoise_coverage = module.resolve_lowdsp_overrides(
            self._v3_encoding(stage3_op="denoise", stage3_expand=1),
            catalog,
        )
        self.assertEqual(denoise_coverage["status"], "covered")
        self.assertIn(
            "l5_stage3_block0_denoise=denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
            denoise_coverage["component_overrides"],
        )

        edge_coverage = module.resolve_lowdsp_overrides(
            self._v3_encoding(stage3_op="edge", stage3_expand=1),
            catalog,
        )
        self.assertEqual(edge_coverage["status"], "covered")
        self.assertIn(
            "l5_stage3_block0_edge=edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns",
            edge_coverage["component_overrides"],
        )

    def test_pareto_route_gate_skips_full_when_lowdsp_coverage_missing(self) -> None:
        module = self._load_pareto_route_gate_script()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            catalog_path = (
                Path(__file__).resolve().parents[1]
                / "hls_lut_builder"
                / "board_harness"
                / "configs"
                / "phase0_v3_lowdsp_override_catalog.yaml"
            )
            unsupported_encoding = self._v3_encoding(stage1_expand=3, stage3_op="mbconv")
            unsupported_encoding["stages"][1]["blocks"][0]["kernel_size"] = 5
            pareto_path = tmp_path / "pareto_selection.json"
            pareto_path.write_text(
                json.dumps(
                    {
                        "selected_topk": [
                            {
                                "selection_index": 1,
                                "rank": 0,
                                "arch_id": "arch_k5_uncovered",
                                "encoding": unsupported_encoding,
                                "metrics": {"macro_f1": 0.9},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_root=str(tmp_path / "route_gate"),
                pareto_selection=str(pareto_path),
                validation_script="validate.py",
                final_report_script="report.py",
                implementation_variant_manifest=[],
                quick_implementation_variant_manifest=[],
                full_implementation_variant_manifest=[],
                python="python",
                topk=8,
                full_topk=5,
                full_max_actual_dsp=700,
                lowdsp_override_catalog=[str(catalog_path)],
                serial_port="COM5",
                quick_timing_variant="quick_default",
                full_timing_variant="full_default",
                quick_impl_profile="default",
                full_impl_profile="fanout_only",
                quick_timeout_minutes=180,
                full_timeout_minutes=240,
                force=False,
            )

            plan = module.prepare_plan(args)

        item = plan["candidates"][0]
        self.assertTrue(item["full_route_skipped"])
        self.assertEqual(item["full_route_skip_reason"], "lowdsp_coverage_missing")
        self.assertEqual(item["lowdsp_coverage"]["status"], "lowdsp_coverage_missing")
        self.assertEqual(item["full_command"], [])


if __name__ == "__main__":
    unittest.main()
