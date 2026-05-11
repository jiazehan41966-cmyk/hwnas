import unittest

from hwnas_fpga.hardware import FPGACostEstimator, LutEntry, LutQueryEngine, LutTable, OpSpec
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.search_space import ResolvedBlockSpec, SearchSpace, SearchSpaceConfig


class HardwareEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(SearchSpaceConfig(num_classes=8))
        self.architecture = self.space.baseline_architecture()

    def test_estimate_returns_positive_metrics(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga",
                clock_mhz=200,
                max_lut=120_000,
                max_bram=2_000,
                max_dsp=512,
                max_power_w=20.0,
            )
        )
        estimate = estimator.estimate(self.architecture, self.space)
        self.assertGreater(estimate.macs, 0)
        self.assertGreater(estimate.latency_ms, 0)
        self.assertGreaterEqual(estimate.peak_dsp, 0)
        self.assertGreaterEqual(estimate.total_dsp, estimate.peak_dsp)
        self.assertGreaterEqual(estimate.total_bram, estimate.peak_bram)
        self.assertGreaterEqual(estimate.total_lut, estimate.peak_lut)
        self.assertFalse(estimate.violations)

    def test_estimate_includes_explicit_stem_layer(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga",
                clock_mhz=200,
                max_lut=120_000,
                max_bram=2_000,
                max_dsp=512,
            )
        )
        estimate = estimator.estimate(self.architecture, self.space)
        resolved_blocks = self.space.resolve_blocks(self.architecture)
        stem_layer = estimate.per_layer[0]

        expected_stem_output = self.space.config.image_size // self.architecture.stem_stride
        if self.space.config.image_size % self.architecture.stem_stride:
            expected_stem_output += 1

        self.assertEqual(len(estimate.per_layer), len(resolved_blocks) + 2)
        self.assertEqual(stem_layer.op, "stem_conv")
        self.assertEqual(stem_layer.input_resolution, self.space.config.image_size)
        self.assertEqual(stem_layer.output_resolution, expected_stem_output)
        self.assertGreater(stem_layer.macs, 0)
        self.assertEqual(estimate.per_layer[-1].op, "head_fc")
        self.assertGreaterEqual(estimate.memory_bandwidth_gbps, 0.0)
        self.assertGreaterEqual(estimate.offchip_mem_mb, 0.0)

    def test_constraints_can_flag_violations(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(name="tight-fpga", clock_mhz=200, max_power_w=1.0),
            constraints=SearchConstraints(max_latency_ms=0.001),
        )
        estimate = estimator.estimate(self.architecture, self.space)
        self.assertTrue(estimate.violations)

    def test_strict_formal_lut_marks_missing_queries_infeasible(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga",
                clock_mhz=200,
                max_lut=120_000,
                max_bram=2_000,
                max_dsp=512,
            ),
            lut_query_engine=LutQueryEngine(
                LutTable(),
                strict_formal_lut=True,
                formal_status_entries={},
            ),
            strict_formal_lut=True,
        )

        estimate = estimator.estimate(self.architecture, self.space)
        stats = estimator.get_lut_stats()

        self.assertTrue(
            any(violation.startswith("formal LUT missing:") for violation in estimate.violations)
        )
        self.assertGreater(stats["true_misses"], 0)


class SonarOpsCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(SearchSpaceConfig())
        self.estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga",
                clock_mhz=200,
                max_lut=120_000,
                max_bram=2_000,
                max_dsp=512,
            ),
        )

    def test_sonar_ops_produce_positive_cost(self) -> None:
        for seed in range(20):
            arch = self.space.sample(seed=seed)
            estimate = self.estimator.estimate(arch, self.space)
            self.assertGreater(estimate.macs, 0, f"seed={seed}: MACs should be > 0")
            self.assertGreater(estimate.latency_ms, 0, f"seed={seed}: latency should be > 0")

    def test_mixconv_has_higher_cost_than_dw_pw(self) -> None:
        mixconv_block = ResolvedBlockSpec(
            stage_index=0,
            block_index=0,
            op="mixconv",
            kernel_size=3,
            expand_ratio=1,
            stride=1,
            in_channels=32,
            out_channels=32,
            input_resolution=56,
            output_resolution=56,
        )
        dw_block = ResolvedBlockSpec(
            stage_index=0,
            block_index=0,
            op="dw_pw_conv",
            kernel_size=3,
            expand_ratio=1,
            stride=1,
            in_channels=32,
            out_channels=32,
            input_resolution=56,
            output_resolution=56,
        )

        mix_cost = self.estimator._estimate_block(mixconv_block)
        dw_cost = self.estimator._estimate_block(dw_block)

        self.assertGreater(mix_cost.macs, dw_cost.macs)


class OperatorPolicyCostTests(unittest.TestCase):
    def test_conditional_operator_uses_actual_post_route_fmax_for_latency(self) -> None:
        lut_table = LutTable(
            [
                LutEntry(
                    op_spec=OpSpec(
                        op="denoise",
                        kernel_size=3,
                        in_channels=32,
                        out_channels=32,
                        stride=1,
                        groups=32,
                        input_resolution=(56, 56),
                    ),
                    latency_ms=0.08,
                    cycles=20_000,
                    dsp=16,
                    bram=4,
                    lut=512,
                )
            ]
        )
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(name="test-fpga", clock_mhz=200),
            lut_query_engine=LutQueryEngine(lut_table),
            operator_policies={
                "denoise": {
                    "deployment_status": "conditional",
                    "achieved_post_route_fmax_mhz": 175.04,
                    "hardware_cost_policy": {
                        "latency_rule": "use_actual_post_route_fmax",
                        "deployable_at_200mhz": False,
                    },
                }
            },
        )
        block = ResolvedBlockSpec(
            stage_index=0,
            block_index=0,
            op="denoise",
            kernel_size=3,
            expand_ratio=1,
            stride=1,
            in_channels=32,
            out_channels=32,
            input_resolution=56,
            output_resolution=56,
        )

        layer_cost = estimator._estimate_block(block)

        self.assertAlmostEqual(layer_cost.effective_clock_mhz or 0.0, 175.04, places=2)
        self.assertAlmostEqual(
            layer_cost.resolved_latency_ms(200.0),
            20_000 / (175.04 * 1_000.0),
            places=6,
        )

    def test_paused_operator_adds_policy_violation(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(name="test-fpga", clock_mhz=200),
            operator_policies={
                "mixconv": {
                    "search_enabled": False,
                    "deployment_status": "paused",
                }
            },
        )
        block = ResolvedBlockSpec(
            stage_index=0,
            block_index=0,
            op="mixconv",
            kernel_size=3,
            expand_ratio=1,
            stride=1,
            in_channels=32,
            out_channels=32,
            input_resolution=56,
            output_resolution=56,
        )

        violations = estimator._check_operator_policy((block,))

        self.assertIn("operator mixconv is paused in current target policy", violations)

    def test_conditional_operator_can_be_marked_non_deployable_under_strict_mode(self) -> None:
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(name="test-fpga", clock_mhz=200),
            operator_policies={
                "edge": {
                    "deployment_status": "conditional",
                    "hardware_cost_policy": {
                        "deployable_at_200mhz": False,
                    },
                }
            },
            require_deployable_operators=True,
        )
        block = ResolvedBlockSpec(
            stage_index=0,
            block_index=0,
            op="edge",
            kernel_size=3,
            expand_ratio=1,
            stride=1,
            in_channels=32,
            out_channels=32,
            input_resolution=56,
            output_resolution=56,
        )

        violations = estimator._check_operator_policy((block,))

        self.assertIn(
            "operator edge is not deployable at 200MHz under current policy",
            violations,
        )


if __name__ == "__main__":
    unittest.main()
