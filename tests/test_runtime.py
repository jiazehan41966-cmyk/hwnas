import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import (
    LutEntry,
    LutTable,
    OpSpec,
    build_lut_from_manifest,
    get_board_profile,
    parse_hls_report_text,
)
from hwnas_fpga.interfaces import SearchConstraints
from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_cost_estimator,
    build_search_space,
    load_anchor_profile_from_pool,
    load_config,
    load_lut_query_engine,
)


class BoardProfileTests(unittest.TestCase):
    def test_zynq7020_profile_contains_memory_fields(self) -> None:
        profile = get_board_profile("zynq7020")
        self.assertEqual(profile.max_dsp, 220)
        self.assertEqual(profile.max_bram, 140)
        self.assertGreater(profile.memory_bandwidth_gbps, 0.0)
        self.assertGreater(profile.offchip_mem_mb, 0.0)


class SearchSpaceRuntimeTests(unittest.TestCase):
    def test_build_search_space_applies_family_profile_defaults(self) -> None:
        search_space = build_search_space(
            {
                "search_space": {
                    "family_profile": "mobile_anchor",
                }
            },
            image_size=64,
            input_channels=1,
            num_classes=8,
            constraints=SearchConstraints(),
        )
        self.assertEqual(search_space.config.family_profile, "mobile_anchor")
        self.assertEqual(search_space.config.stage_base_channels, (8, 12, 16, 16, 20, 24, 24))
        self.assertEqual(search_space.config.width_multipliers, (0.75, 1.0))
        self.assertEqual(
            search_space.config.op_choices,
            ("mbconv", "denoise", "edge", "skip"),
        )
        self.assertEqual(search_space.config.head_conv_channels, 320)

    def test_build_search_space_explicit_values_override_profile(self) -> None:
        search_space = build_search_space(
            {
                "search_space": {
                    "family_profile": "lightweight_sonar",
                    "stage_channel_choices": [[16, 24], [24, 32], [32, 48]],
                    "stage_depth_choices": [[1], [1], [1]],
                }
            },
            image_size=64,
            input_channels=1,
            num_classes=8,
            constraints=SearchConstraints(),
        )
        self.assertEqual(search_space.config.family_profile, "lightweight_sonar")
        self.assertEqual(search_space.config.stage_channel_choices[0], (16, 24))
        self.assertEqual(search_space.config.stage_channel_choices[2], (32, 48))
        self.assertEqual(search_space.config.depth_choices_for_stage(0), (1,))

    def test_build_search_space_supports_stage_specific_mobile_anchor(self) -> None:
        search_space = build_search_space(
            {
                "search_space": {
                    "family_profile": "mobile_anchor",
                    "stem_channels": 24,
                    "stem_stride": 2,
                    "post_stem_downsample_stride": 2,
                    "stage_strides": [2, 2, 2],
                    "stage_base_channels": [116, 232, 464],
                    "width_multipliers": [0.5, 0.75, 1.0, 1.25],
                    "stage_depth_choices": [[3, 4, 5], [6, 7, 8, 9], [3, 4, 5]],
                    "kernel_choices": [3, 5],
                    "head_conv_channels": 1024,
                }
            },
            image_size=224,
            input_channels=1,
            num_classes=8,
            constraints=SearchConstraints(),
        )
        self.assertEqual(search_space.config.stage_count, 3)
        self.assertEqual(search_space.config.stage_channel_choices[0], (58, 87, 116, 145))
        self.assertEqual(search_space.config.post_stem_downsample_stride, 2)
        self.assertEqual(search_space.config.head_conv_channels, 1024)

    def test_canonical_anchor_configs_build_expected_spaces(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_expectations = [
            (
                repo_root / "configs" / "search" / "nksid_fpga_search_mobile_anchor_av7k325.yaml",
                "mobile_anchor",
                7,
                320,
            ),
            (
                repo_root / "configs" / "search" / "nksid_fpga_search_accuracy_biased_av7k325.yaml",
                "accuracy_biased",
                7,
                1280,
            ),
            (
                repo_root / "configs" / "search" / "nksid_fpga_search_lightweight_sonar_av7k325.yaml",
                "lightweight_sonar",
                3,
                1024,
            ),
        ]
        for config_path, profile_name, expected_stage_count, expected_head in config_expectations:
            with self.subTest(config=config_path.name):
                config = load_config(str(config_path))
                search_space = build_search_space(
                    config,
                    image_size=224,
                    input_channels=1,
                    num_classes=8,
                    constraints=SearchConstraints(),
                )
                self.assertEqual(search_space.config.family_profile, profile_name)
                self.assertEqual(search_space.config.stage_count, expected_stage_count)
                self.assertEqual(search_space.config.head_conv_channels, expected_head)

    def test_load_anchor_profile_from_pool_resolves_expected_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "selected_backbone_pool.json"
            pool_path.write_text(
                json.dumps(
                    {
                        "recommended_profiles": {
                            "mobile_anchor": {"source_role": "search_anchor"},
                            "accuracy_biased": {"source_role": "accuracy_anchor"},
                            "lightweight_sonar": {"source_role": "lightweight_anchor"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_anchor_profile_from_pool(pool_path, "search_anchor"),
                "mobile_anchor",
            )
            self.assertEqual(
                load_anchor_profile_from_pool(pool_path, "accuracy_anchor"),
                "accuracy_biased",
            )

    def test_apply_operator_policies_filters_runtime_search_ops(self) -> None:
        search_space = build_search_space(
            {
                "search_space": {
                    "family_profile": "mobile_anchor",
                    "op_choices": ["mbconv", "fused_mbconv", "mixconv", "denoise", "edge", "skip"],
                }
            },
            image_size=64,
            input_channels=1,
            num_classes=8,
            constraints=SearchConstraints(),
        )
        filtered_space, summary = apply_operator_policies_to_search_space(
            search_space,
            {
                "fused_mbconv": {
                    "search_enabled": False,
                    "deployment_status": "dropped_current_target",
                    "reason": "drop",
                },
                "mixconv": {
                    "search_enabled": False,
                    "deployment_status": "paused",
                    "reason": "pause",
                },
                "denoise": {
                    "search_enabled": True,
                    "deployment_status": "conditional",
                    "achieved_post_route_fmax_mhz": 175.04,
                    "hardware_cost_policy": {
                        "latency_rule": "use_actual_post_route_fmax",
                        "deployable_at_200mhz": False,
                    },
                },
            },
        )

        self.assertEqual(
            filtered_space.config.op_choices,
            ("mbconv", "denoise", "edge", "skip"),
        )
        self.assertEqual(
            [item["op"] for item in summary["removed_ops"]],
            ["fused_mbconv", "mixconv"],
        )
        self.assertEqual(
            [item["op"] for item in summary["conditional_ops"]],
            ["denoise"],
        )


class LutRuntimeTests(unittest.TestCase):
    def test_build_cost_estimator_loads_operator_policy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "operator_manifest.yaml"
            manifest_path.write_text(
                """
search_policy:
  default_mainline_search_ops: [conv, mbconv]
  conditional_deployment_ops: [denoise]
  paused_ops: [mixconv]
  dropped_current_target_ops: [fused_mbconv]
operator_status:
  denoise:
    search_enabled: true
    deployment_status: conditional
    achieved_post_route_fmax_mhz: 175.04
    hardware_cost_policy:
      latency_rule: use_actual_post_route_fmax
      deployable_at_200mhz: false
                """.strip(),
                encoding="utf-8",
            )

            estimator = build_cost_estimator(
                {
                    "hardware": {
                        "operator_manifest_path": str(manifest_path),
                        "require_deployable_operators": True,
                    }
                },
                hardware_spec=get_board_profile("av7k325"),
                constraints=SearchConstraints(),
            )

            self.assertTrue(estimator.require_deployable_operators)
            self.assertIn("denoise", estimator.operator_policies)
            self.assertEqual(
                estimator.operator_policies["mixconv"]["deployment_status"],
                "paused",
            )
            self.assertEqual(
                estimator.operator_policies["fused_mbconv"]["deployment_status"],
                "dropped_current_target",
            )
            self.assertAlmostEqual(
                estimator.operator_policies["denoise"]["achieved_post_route_fmax_mhz"],
                175.04,
                places=2,
            )

    def test_load_lut_query_engine_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = Path(tmpdir) / "fpga_lut.pkl"
            table = LutTable(
                [
                    LutEntry(
                        op_spec=OpSpec(
                            op="conv",
                            kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            input_resolution=(56, 56),
                        ),
                        latency_ms=0.1,
                        cycles=20_000,
                        dsp=8,
                        bram=2,
                        lut=128,
                        power_w=3.0,
                        energy_mj=0.3,
                    )
                ]
            )
            table.save(str(lut_path))
            engine = load_lut_query_engine({"hardware": {"lut_path": str(lut_path)}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=16,
                        input_resolution=(56, 56),
                    )
                )
            )

    def test_strict_formal_lut_requires_status_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = Path(tmpdir) / "fpga_lut.pkl"
            LutTable().save(str(lut_path))

            with self.assertRaisesRegex(ValueError, "formal_lut_status_path"):
                load_lut_query_engine(
                    {
                        "hardware": {
                            "lut_path": str(lut_path),
                            "strict_formal_lut": True,
                        }
                    }
                )

    def test_load_lut_query_engine_from_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = Path(tmpdir) / "fpga_lut.json"
            lut_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "op_spec": {
                                    "op": "conv",
                                    "kernel_size": 3,
                                    "in_channels": 16,
                                    "out_channels": 32,
                                    "stride": 1,
                                    "groups": 1,
                                    "expand_ratio": 1,
                                    "input_resolution": [56, 56],
                                },
                                "latency_ms": 0.2,
                                "cycles": 40_000,
                                "dsp": 12,
                                "bram": 4,
                                "lut": 256,
                                "power_w": 3.2,
                                "energy_mj": 0.64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            engine = load_lut_query_engine({"hardware": {"lut_path": str(lut_path), "clock_mhz": 200}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=32,
                        stride=1,
                        input_resolution=(56, 56),
                    )
                )
            )

    def test_load_lut_query_engine_from_legacy_json_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = Path(tmpdir) / "legacy_lut.json"
            lut_path.write_text(
                json.dumps(
                    {
                        "clock_mhz": 200,
                        "conv_k3_e1_cin16_cout32_res56_s1": {
                            "dsp": 12,
                            "bram": 4,
                            "lut": 256,
                            "latency_ms": 0.2,
                            "energy_mj": 0.64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            engine = load_lut_query_engine({"hardware": {"lut_path": str(lut_path), "clock_mhz": 200}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=32,
                        stride=1,
                        input_resolution=(56, 56),
                    )
                )
            )

    def test_parse_hls_report_text(self) -> None:
        report_text = """
        == Performance Estimates
        Latency (cycles): 12345
        == Utilization Estimates
        BRAM_18K | 12
        DSP48E   | 34
        FF       | 5678
        LUT      | 4321
        """
        metrics = parse_hls_report_text(report_text)
        self.assertEqual(metrics["cycles"], 12345)
        self.assertEqual(metrics["bram"], 12)
        self.assertEqual(metrics["dsp"], 34)
        self.assertEqual(metrics["lut"], 4321)

    def test_build_lut_from_manifest_with_relative_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "reports" / "conv.rpt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                """
                Latency (cycles): 5000
                BRAM_18K | 4
                DSP48E   | 12
                LUT      | 345
                Power    | 1.25
                """,
                encoding="utf-8",
            )
            manifest_path = tmp_path / "lut_manifest.yaml"
            manifest_path.write_text(
                """
clock_mhz: 200
entries:
  - op: conv
    kernel_size: 3
    in_channels: 16
    out_channels: 32
    stride: 1
    input_resolution: [56, 56]
    report: reports/conv.rpt
                """.strip(),
                encoding="utf-8",
            )

            table, summary = build_lut_from_manifest(manifest_path)
            self.assertEqual(summary["entries_built"], 1)

            entry = table.query(
                OpSpec(
                    op="conv",
                    kernel_size=3,
                    in_channels=16,
                    out_channels=32,
                    stride=1,
                    input_resolution=(56, 56),
                )
            )
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.cycles, 5000)
            self.assertAlmostEqual(entry.latency_ms, 0.025, places=6)
            self.assertEqual(entry.dsp, 12)
            self.assertEqual(entry.bram, 4)
            self.assertEqual(entry.lut, 345)

    def test_build_lut_from_manifest_normalizes_hls_op_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "reports" / "mbconv.rpt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                """
                Latency (cycles): 8000
                BRAM_18K | 5
                DSP48E   | 16
                LUT      | 456
                Power    | 1.75
                """,
                encoding="utf-8",
            )
            manifest_path = tmp_path / "lut_manifest.yaml"
            manifest_path.write_text(
                """
clock_mhz: 200
entries:
  - op: inverted_residual
    kernel_size: 3
    in_channels: 16
    out_channels: 24
    stride: 2
    expand_ratio: 6
    input_resolution: [112, 112]
    report: reports/mbconv.rpt
                """.strip(),
                encoding="utf-8",
            )

            table, summary = build_lut_from_manifest(manifest_path)
            self.assertEqual(summary["entries_built"], 1)

            entry = table.query(
                OpSpec(
                    op="mbconv",
                    kernel_size=3,
                    in_channels=16,
                    out_channels=24,
                    stride=2,
                    expand_ratio=6,
                    input_resolution=(112, 112),
                )
            )
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.op_spec.op, "mbconv")
            self.assertEqual(entry.cycles, 8000)
            self.assertEqual(entry.dsp, 16)
            self.assertEqual(entry.bram, 5)
            self.assertEqual(entry.lut, 456)

    def test_build_lut_from_manifest_prefers_vivado_actual_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "lut_manifest.yaml"
            manifest_path.write_text(
                """
clock_mhz: 200
entries:
  - op: pw_conv
    kernel_size: 1
    in_channels: 24
    out_channels: 144
    stride: 1
    input_resolution: [56, 56]
    hls_estimate:
      metrics:
        cycles: 4000
        latency_ms: 0.02
        dsp: 12
        bram: 14
        lut: 900
    vivado_actual:
      metrics:
        post_route:
          lut: 420
          ff: 315
          dsp: 16
          block_ram_tile: 4.5
          ramb18: 9
          ramb36: 0
          fmax_est_mhz: 210.0
                """.strip(),
                encoding="utf-8",
            )

            table, summary = build_lut_from_manifest(manifest_path)
            self.assertEqual(summary["entries_built"], 1)

            entry = table.query(
                OpSpec(
                    op="pw_conv",
                    kernel_size=1,
                    in_channels=24,
                    out_channels=144,
                    stride=1,
                    input_resolution=(56, 56),
                )
            )
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.cycles, 4000)
            self.assertAlmostEqual(entry.latency_ms, 0.02, places=6)
            self.assertEqual(entry.dsp, 16)
            self.assertEqual(entry.bram, 5)
            self.assertEqual(entry.lut, 420)

    def test_run_build_lut_cli_generates_pickle_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mbconv.rpt"
            report_path.write_text(
                """
                Latency (cycles): 12000
                BRAM_18K | 6
                DSP48E   | 24
                LUT      | 789
                Power    | 2.5
                """,
                encoding="utf-8",
            )
            manifest_path = tmp_path / "lut_manifest.json"
            manifest_path.write_text(
                """
{
  "clock_mhz": 200,
  "entries": [
    {
      "op": "mbconv",
      "kernel_size": 3,
      "in_channels": 32,
      "out_channels": 64,
      "stride": 1,
      "expand_ratio": 4,
      "input_resolution": [28, 28],
      "report": "mbconv.rpt"
    }
  ]
}
                """.strip(),
                encoding="utf-8",
            )
            output_path = tmp_path / "fpga_lut.pkl"
            summary_path = tmp_path / "summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "run_build_lut.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--summary-json",
                    str(summary_path),
                ],
                check=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            engine = load_lut_query_engine({"hardware": {"lut_path": str(output_path)}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="mbconv",
                        kernel_size=3,
                        in_channels=32,
                        out_channels=64,
                        stride=1,
                        expand_ratio=4,
                        input_resolution=(28, 28),
                    )
                )
            )

    def test_generate_estimator_lut_cli_generates_json_and_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "lut_smoke.yaml"
            output_json = tmp_path / "analytical_lut.json"
            output_pkl = tmp_path / "analytical_lut.pkl"
            summary_json = tmp_path / "summary.json"
            config_path.write_text(
                """
dataset:
  name: dummy
  input_channels: 1
  image_size: 32
  num_classes: 4
search_space:
  stem_channels: 8
  stem_stride: 2
  stage_strides: [1, 2]
  channel_choices: [8, 16]
  depth_choices: [1, 2]
  kernel_choices: [3]
  expand_choices: [1, 2]
  op_choices:
    - conv
    - dw_pw_conv
    - mbconv
    - fused_mbconv
    - skip
hardware:
  board: zynq7020
  clock_mhz: 200
  quantization_bits: 8
                """.strip(),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "generate_estimator_lut.py"),
                    "--config",
                    str(config_path),
                    "--output-json",
                    str(output_json),
                    "--output-pkl",
                    str(output_pkl),
                    "--summary-json",
                    str(summary_json),
                    "--validation-samples",
                    "2",
                ],
                check=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )

            self.assertTrue(output_json.exists())
            self.assertTrue(output_pkl.exists())
            self.assertTrue(summary_json.exists())

            engine = load_lut_query_engine({"hardware": {"lut_path": str(output_json), "clock_mhz": 200}})
            self.assertIsNotNone(engine)

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertGreater(summary["entries_built"], 0)
            self.assertGreaterEqual(summary["validation"]["lut_stats"]["hit_rate"], 0.99)


if __name__ == "__main__":
    unittest.main()
