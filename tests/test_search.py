import unittest
from pathlib import Path
import sys
from types import MethodType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.search import (
    ActionSpace,
    Controller,
    RLSearcher,
    RandomSearcher,
    build_pareto_objectives,
    create_searcher,
)
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


class SearchFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(SearchSpaceConfig(num_classes=8))
        self.constraints = SearchConstraints(max_latency_ms=50.0)
        self.estimator = FPGACostEstimator(
            hardware_spec=HardwareSpec(
                name="test-fpga",
                clock_mhz=200,
                max_lut=120_000,
                max_bram=2_000,
                max_dsp=512,
                max_power_w=20.0,
                memory_bandwidth_gbps=8.0,
                offchip_mem_mb=512.0,
            ),
            constraints=self.constraints,
        )

    def test_create_random_searcher(self) -> None:
        searcher = create_searcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            method="random",
            seed=1,
        )
        self.assertIsInstance(searcher, RandomSearcher)

    def test_create_rl_searcher_with_reward_weights(self) -> None:
        searcher = create_searcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            method="rl",
            seed=1,
            controller_hidden_dim=32,
            controller_lr=0.005,
            train_epochs_per_arch=1,
            device="cpu",
            reward_weights={
                "accuracy": 1.0,
                "latency": 0.2,
                "energy": 0.3,
                "resource": 0.4,
            },
        )
        self.assertIsInstance(searcher, RLSearcher)
        self.assertAlmostEqual(searcher.reward_function.energy_weight, 0.3)
        self.assertAlmostEqual(searcher.reward_function.dsp_weight, 0.4)
        self.assertAlmostEqual(searcher.reward_function.bram_weight, 0.4)
        self.assertAlmostEqual(searcher.reward_function.lut_weight, 0.4)

    def test_create_rl_searcher_with_reward_cfg(self) -> None:
        searcher = create_searcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            method="rl",
            seed=3,
            controller_hidden_dim=16,
            controller_lr=0.001,
            train_epochs_per_arch=1,
            device="cpu",
            reward_cfg={
                "constraint_penalty": 3.5,
                "infeasible_penalty_mode": "gradient",
                "infeasible_base_penalty": 0.8,
                "infeasible_penalty_scale": 2.2,
            },
        )
        self.assertIsInstance(searcher, RLSearcher)
        self.assertAlmostEqual(searcher.reward_function.constraint_penalty, 3.5)
        self.assertEqual(searcher.reward_function.infeasible_penalty_mode, "violation_ratio")
        self.assertAlmostEqual(searcher.reward_function.infeasible_base_penalty, 0.8)
        self.assertAlmostEqual(searcher.reward_function.infeasible_penalty_scale, 2.2)

    def test_create_searcher_with_family_profiled_space(self) -> None:
        profiled_space = SearchSpace(SearchSpaceConfig.from_dict({"family_profile": "mobile_anchor", "num_classes": 8}))
        searcher = create_searcher(
            search_space=profiled_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            method="random",
            seed=7,
        )
        self.assertIsInstance(searcher, RandomSearcher)
        self.assertEqual(profiled_space.config.family_profile, "mobile_anchor")

    def test_build_pareto_objectives_from_weights_and_constraints(self) -> None:
        objectives, directions = build_pareto_objectives(
            {
                "accuracy": 1.0,
                "latency": 0.2,
                "energy": 0.1,
                "resource": 0.4,
            },
            SearchConstraints(
                max_latency_ms=50.0,
                max_dsp=220,
                max_bram=140,
                max_lut=53_200,
                max_memory_bandwidth_gbps=4.0,
            ),
        )
        self.assertEqual(objectives[0], "accuracy")
        self.assertEqual(directions[0], "max")
        self.assertIn("latency_ms", objectives)
        self.assertIn("energy_mj", objectives)
        self.assertIn("dsp", objectives)
        self.assertIn("bram", objectives)
        self.assertIn("lut", objectives)
        self.assertIn("memory_bandwidth_gbps", objectives)


class ControllerMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        action_space = ActionSpace(
            channel_actions=(8, 12, 16),
            depth_actions=(1, 2),
            kernel_actions=(3, 5),
            expand_actions=(1, 2),
            op_actions=("dw_pw_conv", "mbconv", "skip"),
        )
        self.controller = Controller(action_space, hidden_dim=8)

    def _install_fixed_forward(self, channel_logits) -> None:
        def fixed_forward(_self, stage_idx, block_idx, prev_choices=None):
            del stage_idx, block_idx, prev_choices
            return {
                "channel": torch.tensor(channel_logits, dtype=torch.float32),
                "depth": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "kernel": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "expand": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "op": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            }

        self.controller.forward = MethodType(fixed_forward, self.controller)

    def test_sample_respects_allowed_indices(self) -> None:
        self._install_fixed_forward([50.0, 1000.0, 0.0])
        sample = self.controller.sample(
            0,
            0,
            allowed_indices={"channel": (0, 2)},
        )
        self.assertEqual(sample["channel"], 0)

    def test_get_log_prob_uses_masked_logits(self) -> None:
        self._install_fixed_forward([5.0, 1000.0, 0.0])
        log_prob = self.controller.get_log_prob(
            0,
            0,
            {"channel": 0},
            allowed_indices={"channel": (0, 2)},
        )["channel"]
        expected = torch.log_softmax(torch.tensor([5.0, 0.0]), dim=0)[0]
        self.assertTrue(torch.isfinite(log_prob))
        self.assertAlmostEqual(float(log_prob), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
