import unittest

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


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


class SonarOpsCostTests(unittest.TestCase):
    """声呐算子的硬件代价估计测试"""

    def setUp(self) -> None:
        self.space = SearchSpace(SearchSpaceConfig())
        self.estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga", clock_mhz=200,
                max_lut=120_000, max_bram=2_000, max_dsp=512,
            ),
        )

    def test_sonar_ops_produce_positive_cost(self) -> None:
        """每种声呐算子的cost估计应产生正值"""
        for seed in range(20):
            arch = self.space.sample(seed=seed)
            estimate = self.estimator.estimate(arch, self.space)
            self.assertGreater(estimate.macs, 0, f"seed={seed}: MACs should be > 0")
            self.assertGreater(estimate.latency_ms, 0, f"seed={seed}: latency should be > 0")

    def test_mixconv_has_higher_cost_than_dw_pw(self) -> None:
        """MixConv (多尺度) 应比单一 DW 卷积的资源消耗更高"""
        from hwnas_fpga.search_space.space import ResolvedBlockSpec

        mixconv_block = ResolvedBlockSpec(
            stage_index=0, block_index=0, op="mixconv",
            kernel_size=3, expand_ratio=1, stride=1,
            in_channels=32, out_channels=32,
            input_resolution=56, output_resolution=56,
        )
        dw_block = ResolvedBlockSpec(
            stage_index=0, block_index=0, op="dw_pw_conv",
            kernel_size=3, expand_ratio=1, stride=1,
            in_channels=32, out_channels=32,
            input_resolution=56, output_resolution=56,
        )

        mix_cost = self.estimator._estimate_block(mixconv_block)
        dw_cost = self.estimator._estimate_block(dw_block)

        # MixConv 使用 (3,5,7) 三种kernel，MAC应更高
        self.assertGreater(mix_cost.macs, dw_cost.macs)


if __name__ == "__main__":
    unittest.main()
