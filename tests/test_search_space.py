import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import (
    FAMILY_PROFILES,
    ArchitectureSpec,
    SearchSpace,
    SearchSpaceConfig,
    SONAR_OPS,
    list_family_profiles,
)
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.search_space import probe_search_space


class SearchSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(SearchSpaceConfig())

    def test_sample_is_valid(self) -> None:
        architecture = self.space.sample(seed=7)
        self.assertTrue(self.space.is_valid(architecture))
        self.assertEqual(len(architecture.stages), self.space.config.stage_count)

    def test_roundtrip_architecture_serialization(self) -> None:
        architecture = self.space.baseline_architecture()
        rebuilt = ArchitectureSpec.from_dict(architecture.to_dict())
        self.assertEqual(architecture, rebuilt)

    def test_illegal_skip_is_rejected(self) -> None:
        architecture = self.space.baseline_architecture()
        payload = architecture.to_dict()
        payload["stages"][1]["blocks"][0]["op"] = "skip"
        illegal_arch = ArchitectureSpec.from_dict(payload)
        self.assertFalse(self.space.is_valid(illegal_arch))

    def test_resolve_blocks_starts_from_stem_output_resolution(self) -> None:
        architecture = self.space.baseline_architecture()
        resolved = self.space.resolve_blocks(architecture)
        expected_resolution = self.space.config.image_size // architecture.stem_stride
        if self.space.config.image_size % architecture.stem_stride:
            expected_resolution += 1

        self.assertEqual(resolved[0].input_resolution, expected_resolution)

        model = build_model(architecture, num_classes=8)
        with torch.no_grad():
            stem_output = model.stem(torch.randn(1, 1, self.space.config.image_size, self.space.config.image_size))
        self.assertEqual(stem_output.shape[-1], resolved[0].input_resolution)
        self.assertEqual(stem_output.shape[-2], resolved[0].input_resolution)


class FamilyProfileTests(unittest.TestCase):
    def test_family_profiles_are_listed(self) -> None:
        profiles = list_family_profiles()
        self.assertEqual(set(profiles), set(FAMILY_PROFILES))

    def test_mobile_anchor_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "mobile_anchor"})
        self.assertEqual(config.family_profile, "mobile_anchor")
        self.assertEqual(config.stem_channels, 32)
        self.assertEqual(config.post_stem_downsample_stride, 1)
        self.assertEqual(config.stage_strides, (1, 2, 2, 2, 1, 2, 1))
        self.assertEqual(config.stage_base_channels, (16, 24, 32, 64, 96, 160, 320))
        self.assertEqual(config.width_multipliers, (0.75, 1.0, 1.25))
        self.assertEqual(config.depth_choices_for_stage(3), (3, 4, 5))
        self.assertEqual(config.expand_choices, (1, 3, 6))
        self.assertEqual(config.op_choices, ("dw_pw_conv", "mbconv", "fused_mbconv", "skip"))
        self.assertEqual(config.head_conv_channels, 1280)

    def test_small_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "small"})
        self.assertEqual(config.family_profile, "small")
        self.assertEqual(config.channel_choices, (16, 24, 32))
        self.assertEqual(config.depth_choices, (1, 2))
        self.assertEqual(config.kernel_choices, (3,))
        self.assertEqual(config.op_choices, ("dw_pw_conv", "skip"))

    def test_accuracy_biased_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "accuracy_biased"})
        self.assertEqual(config.stem_channels, 32)
        self.assertEqual(config.post_stem_downsample_stride, 1)
        self.assertEqual(config.stage_strides, (1, 2, 2, 2, 1, 2, 1))
        self.assertEqual(config.stage_base_channels, (16, 24, 32, 64, 96, 160, 320))
        self.assertEqual(config.width_multipliers, (1.0, 1.25))
        self.assertEqual(config.depth_choices_for_stage(3), (4, 5))
        self.assertEqual(config.expand_choices, (3, 6))
        self.assertNotIn("conv", config.op_choices)
        self.assertNotIn("edge", config.op_choices)
        self.assertIn("denoise", config.op_choices)
        self.assertEqual(config.head_conv_channels, 1280)

    def test_lightweight_sonar_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "lightweight_sonar"})
        self.assertEqual(config.stem_channels, 24)
        self.assertEqual(config.post_stem_downsample_stride, 2)
        self.assertEqual(config.stage_strides, (2, 2, 2))
        self.assertEqual(config.stage_base_channels, (24, 48, 96))
        self.assertEqual(config.width_multipliers, (0.5, 0.75, 1.0))
        self.assertEqual(config.depth_choices_for_stage(1), (2, 3, 4))
        self.assertEqual(config.expand_choices, (1, 2, 4))
        self.assertEqual(config.kernel_choices, (3,))
        self.assertNotIn("conv", config.op_choices)
        self.assertNotIn("mixconv", config.op_choices)
        self.assertIn("denoise", config.op_choices)
        self.assertEqual(config.head_conv_channels, 1024)

    def test_explicit_config_overrides_profile_defaults(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "family_profile": "mobile_anchor",
                "stage_channel_choices": [
                    [16, 24],
                    [24, 32],
                    [32, 48],
                    [48, 64],
                    [64, 96],
                    [96, 160],
                    [160, 320],
                ],
                "stage_depth_choices": [[1], [1], [1], [1], [1], [1], [1]],
            }
        )
        self.assertEqual(config.stage_channel_choices[0], (16, 24))
        self.assertEqual(config.stage_channel_choices[-1], (160, 320))
        self.assertEqual(config.depth_choices_for_stage(0), (1,))

    def test_unknown_family_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            SearchSpaceConfig.from_dict({"family_profile": "unknown_profile"})

    def test_stage_specific_choices_and_shuffle_head_are_derived(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "stem_channels": 24,
                "stem_stride": 2,
                "post_stem_downsample_stride": 2,
                "stage_strides": [2, 2, 2],
                "stage_base_channels": [116, 232, 464],
                "width_multipliers": [0.5, 0.75, 1.0, 1.25],
                "stage_depth_choices": [[3, 4, 5], [6, 7, 8, 9], [3, 4, 5]],
                "kernel_choices": [3, 5],
                "expand_choices": [1, 2, 4, 6],
                "op_choices": ["dw_pw_conv", "mbconv", "fused_mbconv", "skip"],
                "head_conv_channels": 1024,
            }
        )
        self.assertEqual(config.stage_count, 3)
        self.assertEqual(config.stage_channel_choices[0], (58, 87, 116, 145))
        self.assertEqual(config.stage_channel_choices[1], (116, 174, 232, 290))
        self.assertEqual(config.stage_channel_choices[2], (232, 348, 464, 580))
        self.assertEqual(config.depth_choices_for_stage(1), (6, 7, 8, 9))
        self.assertEqual(config.post_stem_downsample_stride, 2)
        self.assertEqual(config.head_conv_channels, 1024)


class SonarOpsSearchSpaceTests(unittest.TestCase):
    """声呐专用算子的搜索空间测试"""

    def setUp(self) -> None:
        # 包含全部8个算子（含3个声呐专用）的搜索空间
        self.space = SearchSpace(SearchSpaceConfig())

    def test_sonar_ops_in_default_choices(self) -> None:
        """声呐算子默认包含在搜索空间中"""
        for op in SONAR_OPS:
            self.assertIn(op, self.space.config.op_choices)

    def test_sample_with_sonar_ops_is_valid(self) -> None:
        """含声呐算子的采样架构应能通过验证"""
        for seed in range(20):
            architecture = self.space.sample(seed=seed)
            errors = self.space.validate(architecture)
            self.assertEqual(errors, [], f"seed={seed} failed: {errors}")

    def test_sonar_ops_require_expand_ratio_1(self) -> None:
        """声呐算子的 expand_ratio 必须为 1"""
        arch = self.space.baseline_architecture()
        payload = arch.to_dict()
        # 强制第一个 block 用 mixconv + expand_ratio=2
        payload["stages"][0]["blocks"][0]["op"] = "mixconv"
        payload["stages"][0]["blocks"][0]["expand_ratio"] = 2
        illegal = ArchitectureSpec.from_dict(payload)
        errors = self.space.validate(illegal)
        self.assertTrue(any("sonar op" in e for e in errors), f"Expected sonar validation error, got: {errors}")

    def test_architecture_with_sonar_ops_serialization(self) -> None:
        """含声呐算子的架构可以正常序列化/反序列化"""
        arch = self.space.sample(seed=123)
        rebuilt = ArchitectureSpec.from_dict(arch.to_dict())
        self.assertEqual(arch, rebuilt)


class HardwarePruningTests(unittest.TestCase):
    """硬件驱动搜索空间剪枝测试"""

    def _make_estimator(self, constraints: SearchConstraints) -> FPGACostEstimator:
        return FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga", clock_mhz=200,
                max_lut=120_000, max_bram=2_000, max_dsp=512,
            ),
            constraints=constraints,
        )

    def test_pre_prune_without_constraints_returns_same(self) -> None:
        """无硬件约束时，剪枝后搜索空间与原空间等同"""
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=None))
        estimator = self._make_estimator(SearchConstraints())
        pruned = space.pre_prune(estimator)
        self.assertTrue(pruned.is_pruned())
        self.assertEqual(pruned.config.channel_choices, space.config.channel_choices)

    def test_pre_prune_with_tight_dsp_reduces_channels(self) -> None:
        """紧张的DSP约束会减少通道选择"""
        constraints = SearchConstraints(max_dsp=30)
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints))
        estimator = self._make_estimator(constraints)
        pruned = space.pre_prune(estimator)
        self.assertTrue(pruned.is_pruned())
        # 通道数应被限制
        self.assertLessEqual(
            len(pruned.config.channel_choices),
            len(space.config.channel_choices),
        )

    def test_pre_prune_with_tight_latency_reduces_depth(self) -> None:
        """紧张的延迟约束会减少深度选择"""
        # 使用极小的延迟约束，确保baseline会违反约束从而触发剪枝
        constraints = SearchConstraints(max_latency_ms=0.001)
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints))
        estimator = self._make_estimator(constraints)
        pruned = space.pre_prune(estimator)
        self.assertTrue(pruned.is_pruned())
        max_depth = max(pruned.config.depth_choices)
        self.assertLessEqual(max_depth, 3)

    def test_pruned_space_produces_valid_samples(self) -> None:
        """剪枝后的搜索空间仍能产生合法架构"""
        constraints = SearchConstraints(max_dsp=100, max_latency_ms=10.0)
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints))
        estimator = self._make_estimator(constraints)
        pruned = space.pre_prune(estimator)
        for seed in range(10):
            arch = pruned.sample(seed=seed)
            self.assertTrue(pruned.is_valid(arch), f"seed={seed} produced invalid arch")

    def test_tight_board_prunes_even_when_baseline_is_feasible(self) -> None:
        constraints = SearchConstraints(
            max_latency_ms=50.0,
            max_dsp=220,
            max_bram=140,
            max_lut=53_200,
            max_memory_bandwidth_gbps=4.0,
        )
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints))
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="zynq7020",
                clock_mhz=200,
                max_lut=53_200,
                max_bram=140,
                max_dsp=220,
                memory_bandwidth_gbps=4.2,
            ),
            constraints=constraints,
        )
        pruned = space.pre_prune(estimator)
        self.assertEqual(pruned.config.channel_choices, (16, 24, 32))
        self.assertEqual(pruned.config.depth_choices, (1, 2))
        self.assertEqual(pruned.config.kernel_choices, (3,))
        self.assertNotIn("conv", pruned.config.op_choices)
        self.assertNotIn("mixconv", pruned.config.op_choices)
        self.assertNotIn("edge", pruned.config.op_choices)
        self.assertIn("fused_mbconv", pruned.config.op_choices)

    def test_require_feasible_sampling_falls_back_to_feasible_architecture(self) -> None:
        constraints = SearchConstraints(
            max_latency_ms=50.0,
            max_dsp=220,
            max_bram=140,
            max_lut=53_200,
        )
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints, num_classes=8))
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="zynq7020",
                clock_mhz=200,
                max_lut=53_200,
                max_bram=140,
                max_dsp=220,
            ),
            constraints=constraints,
        )
        architecture = space.sample(
            seed=42,
            cost_estimator=estimator,
            require_feasible=True,
            prefer_lightweight=True,
            max_feasible_attempts=8,
        )
        pruned_space = space.pre_prune(estimator)
        estimate = estimator.estimate(architecture, pruned_space)
        baseline_estimate = estimator.estimate(pruned_space.baseline_architecture(), pruned_space)
        self.assertTrue(pruned_space.is_valid(architecture))
        self.assertLessEqual(len(estimate.violations), len(baseline_estimate.violations))


class SearchSpaceProbeTests(unittest.TestCase):
    def test_probe_summary_counts_all_samples(self) -> None:
        constraints = SearchConstraints(max_latency_ms=50.0, max_dsp=220, max_bram=140, max_lut=53_200)
        space = SearchSpace(SearchSpaceConfig.from_dict({"family_profile": "lightweight_sonar", "num_classes": 8, "hardware_constraints": constraints}))
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="zynq7020",
                clock_mhz=200,
                max_lut=53_200,
                max_bram=140,
                max_dsp=220,
                memory_bandwidth_gbps=4.2,
                offchip_mem_mb=128.0,
            ),
            constraints=constraints,
        )

        records, summary = probe_search_space(space.pre_prune(estimator), estimator, num_samples=12, seed=7)
        self.assertEqual(len(records), 12)
        self.assertEqual(summary["total_samples"], 12)
        self.assertEqual(summary["feasible_samples"] + summary["infeasible_samples"], 12)
        self.assertGreaterEqual(summary["feasible_ratio"], 0.0)
        self.assertLessEqual(summary["feasible_ratio"], 1.0)
        self.assertIsNotNone(summary["metrics_all"]["latency_ms"])

    def test_probe_reports_violations_under_tight_constraints(self) -> None:
        constraints = SearchConstraints(max_latency_ms=0.001, max_dsp=10, max_bram=2, max_lut=500)
        space = SearchSpace(SearchSpaceConfig(hardware_constraints=constraints, num_classes=8))
        estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="tiny-fpga",
                clock_mhz=200,
                max_lut=500,
                max_bram=2,
                max_dsp=10,
                offchip_mem_mb=1.0,
            ),
            constraints=constraints,
        )

        _records, summary = probe_search_space(space, estimator, num_samples=6, seed=11)
        self.assertEqual(summary["feasible_samples"], 0)
        self.assertGreater(summary["infeasible_samples"], 0)
        self.assertTrue(summary["violation_counts"])

    def test_resolve_blocks_after_post_stem_downsample(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "input_channels": 1,
                "image_size": 224,
                "stem_channels": 24,
                "stem_stride": 2,
                "post_stem_downsample_stride": 2,
                "stage_strides": [2, 2, 2],
                "stage_base_channels": [116, 232, 464],
                "width_multipliers": [1.0],
                "stage_depth_choices": [[4], [8], [4]],
                "kernel_choices": [3],
                "expand_choices": [1],
                "op_choices": ["dw_pw_conv"],
                "head_conv_channels": 1024,
                "num_classes": 8,
            }
        )
        space = SearchSpace(config)
        architecture = space.baseline_architecture()
        resolved = space.resolve_blocks(architecture)
        self.assertEqual(resolved[0].input_resolution, 56)
        model = build_model(architecture, num_classes=8)
        with torch.no_grad():
            stem_output = model.stem(torch.randn(1, 1, 224, 224))
            pooled = model.post_stem_downsample(stem_output)
        self.assertEqual(pooled.shape[-1], resolved[0].input_resolution)


if __name__ == "__main__":
    unittest.main()
