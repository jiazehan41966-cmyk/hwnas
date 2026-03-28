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
        self.assertEqual(config.channel_choices, (16, 24, 32, 48, 64))
        self.assertEqual(config.depth_choices, (1, 2, 3))
        self.assertEqual(config.op_choices, ("dw_pw_conv", "mbconv", "fused_mbconv", "skip"))

    def test_accuracy_biased_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "accuracy_biased"})
        self.assertEqual(config.stem_channels, 24)
        self.assertEqual(config.depth_choices, (2, 3, 4))
        self.assertIn("conv", config.op_choices)
        self.assertIn("edge", config.op_choices)

    def test_lightweight_sonar_profile_resolves_expected_choices(self) -> None:
        config = SearchSpaceConfig.from_dict({"family_profile": "lightweight_sonar"})
        self.assertEqual(config.channel_choices, (16, 24, 32))
        self.assertEqual(config.expand_choices, (1, 2))
        self.assertNotIn("conv", config.op_choices)
        self.assertIn("mixconv", config.op_choices)

    def test_explicit_config_overrides_profile_defaults(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "family_profile": "mobile_anchor",
                "channel_choices": [16, 24],
                "depth_choices": [1],
            }
        )
        self.assertEqual(config.channel_choices, (16, 24))
        self.assertEqual(config.depth_choices, (1,))

    def test_unknown_family_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            SearchSpaceConfig.from_dict({"family_profile": "unknown_profile"})


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
        self.assertNotIn("fused_mbconv", pruned.config.op_choices)

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


if __name__ == "__main__":
    unittest.main()
