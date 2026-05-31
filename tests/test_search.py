import unittest
from pathlib import Path
import sys
from types import MethodType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import CandidateMetrics, HardwareSpec, SearchCandidate, SearchConstraints
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

    def test_rl_searcher_samples_multiple_stage_block_choices(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "input_channels": 1,
                "image_size": 224,
                "stem_channels": 16,
                "stem_stride": 2,
                "stage_strides": [2],
                "stage_base_channels": [24],
                "width_multipliers": [1.0],
                "stage_depth_choices": [[1]],
                "op_choices": ["mbconv"],
                "kernel_choices": [3, 5],
                "expand_choices": [3, 6],
                "stage_block_choices": [
                    [
                        {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
                        {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6},
                    ]
                ],
                "num_classes": 8,
            }
        )
        searcher = RLSearcher(
            search_space=SearchSpace(config),
            cost_estimator=self.estimator,
            constraints=self.constraints,
            controller_hidden_dim=8,
            train_epochs_per_arch=0,
            device="cpu",
            seed=11,
        )

        def fixed_forward(_self, stage_idx, block_idx, prev_choices=None):
            del stage_idx, block_idx, prev_choices
            return {
                "channel": torch.tensor([0.0], dtype=torch.float32),
                "depth": torch.tensor([0.0], dtype=torch.float32),
                "kernel": torch.tensor([-1000.0, 1000.0], dtype=torch.float32),
                "expand": torch.tensor([1000.0, -1000.0], dtype=torch.float32),
                "op": torch.tensor([0.0], dtype=torch.float32),
            }

        searcher.controller.forward = MethodType(fixed_forward, searcher.controller)
        architecture = searcher.generate_architecture()
        block = architecture.stages[0].blocks[0]

        self.assertEqual(block.kernel_size, 5)
        self.assertEqual(block.expand_ratio, 6)

    def test_rl_searcher_epsilon_explores_stage_block_choices(self) -> None:
        config = SearchSpaceConfig.from_dict(
            {
                "input_channels": 1,
                "image_size": 224,
                "stem_channels": 16,
                "stem_stride": 2,
                "stage_strides": [2],
                "stage_base_channels": [24],
                "width_multipliers": [1.0],
                "stage_depth_choices": [[1]],
                "op_choices": ["mbconv"],
                "kernel_choices": [3, 5],
                "expand_choices": [3, 6],
                "stage_block_choices": [
                    [
                        {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
                        {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3},
                        {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
                        {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6},
                    ]
                ],
                "num_classes": 8,
            }
        )
        searcher = RLSearcher(
            search_space=SearchSpace(config),
            cost_estimator=self.estimator,
            constraints=self.constraints,
            controller_hidden_dim=8,
            train_epochs_per_arch=0,
            device="cpu",
            seed=11,
            exploration_epsilon_start=1.0,
        )

        sampled = {
            (
                architecture.stages[0].blocks[0].kernel_size,
                architecture.stages[0].blocks[0].expand_ratio,
            )
            for architecture in (searcher.generate_architecture() for _ in range(100))
        }

        self.assertEqual(sampled, {(3, 3), (5, 3), (3, 6), (5, 6)})

    def test_rl_searcher_updates_controller_once_with_exploration_bonus(self) -> None:
        searcher = RLSearcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            controller_hidden_dim=8,
            train_epochs_per_arch=0,
            device="cpu",
            seed=13,
            exploration_bonus=0.25,
        )
        architecture = searcher.generate_architecture()
        metrics = {
            "latency_ms": 1.0,
            "energy_mj": 0.1,
            "dsp": 1,
            "bram": 1,
            "lut": 1,
        }
        base_reward = searcher.reward_function.compute_reward(
            accuracy=0.7,
            latency_ms=metrics["latency_ms"],
            energy_mj=metrics["energy_mj"],
            dsp=metrics["dsp"],
            bram=metrics["bram"],
            lut=metrics["lut"],
            is_feasible=True,
        )
        candidate = SearchCandidate(
            "rl_arch_test",
            {},
            CandidateMetrics(accuracy=0.7, latency_ms=1.0, energy_mj=0.1, dsp=1, bram=1, lut=1),
        )
        update_rewards = []

        def fixed_generate(_self):
            return architecture

        def fixed_evaluate(_self, *_args, **_kwargs):
            _self.evaluated_candidates.append(candidate)
            _self.feasible_candidates.append(candidate)
            _self.best_candidate = candidate
            _self.best_reward = base_reward
            _self.best_selection_score = 0.7
            return 0.7, metrics, True, candidate

        def record_update(_self, _architecture, reward):
            update_rewards.append(reward)
            return 0.0

        searcher.generate_architecture = MethodType(fixed_generate, searcher)
        searcher.evaluate_architecture = MethodType(fixed_evaluate, searcher)
        searcher.update_controller = MethodType(record_update, searcher)

        result = searcher.search([], None, num_classes=8, num_episodes=1, verbose=False)

        self.assertIs(result, candidate)
        self.assertEqual(len(update_rewards), 1)
        self.assertAlmostEqual(update_rewards[0], base_reward + 0.25)
        self.assertEqual(
            searcher.architecture_visit_counts[searcher._architecture_visit_key(architecture)],
            1,
        )

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
            selection_metric="macro_f1",
        )
        self.assertEqual(objectives[0], "macro_f1")
        self.assertEqual(directions[0], "max")
        self.assertIn("latency_ms", objectives)
        self.assertIn("energy_mj", objectives)
        self.assertIn("dsp", objectives)
        self.assertIn("bram", objectives)
        self.assertIn("lut", objectives)
        self.assertIn("memory_bandwidth_gbps", objectives)

    def test_rl_searcher_inherits_base_feasibility_checks(self) -> None:
        constraints = SearchConstraints(max_energy_mj=1.0)
        searcher = RLSearcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=constraints,
            controller_hidden_dim=8,
            train_epochs_per_arch=0,
            device="cpu",
            seed=7,
        )
        from hwnas_fpga.hardware import CostEstimate

        def _minimal_cost(**overrides):
            defaults = {
                "params": 1,
                "macs": 1,
                "model_size_mb": 1.0,
                "peak_activation_bytes": 1,
                "peak_weight_bytes": 1,
                "peak_buffer_bytes": 1,
                "peak_dsp": 1,
                "peak_bram": 1,
                "peak_lut": 1,
                "total_dsp": 1,
                "total_bram": 1,
                "total_lut": 1,
                "latency_cycles": 1,
                "latency_ms": 1.0,
                "power_w": 1.0,
                "energy_mj": 0.5,
                "memory_bandwidth_gbps": 1.0,
                "offchip_mem_mb": 1.0,
                "violations": (),
                "per_layer": (),
            }
            defaults.update(overrides)
            return CostEstimate(**defaults)

        feasible = searcher.check_feasibility(_minimal_cost(energy_mj=0.5))
        infeasible = searcher.check_feasibility(_minimal_cost(energy_mj=2.0))
        self.assertTrue(feasible)
        self.assertFalse(infeasible)

    def test_rl_reward_uses_violation_ratio_for_infeasible(self) -> None:
        searcher = RLSearcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=SearchConstraints(max_lut=100),
            controller_hidden_dim=8,
            train_epochs_per_arch=0,
            device="cpu",
            seed=9,
            reward_cfg={
                "infeasible_penalty_mode": "violation_ratio",
                "infeasible_base_penalty": 1.0,
                "infeasible_penalty_scale": 2.0,
            },
        )
        reward = searcher.reward_function.compute_reward(
            accuracy=0.0,
            latency_ms=10.0,
            energy_mj=1.0,
            dsp=10,
            bram=10,
            lut=200,
            is_feasible=False,
            constraint_violation_ratio=0.5,
        )
        self.assertAlmostEqual(reward, -(1.0 + 2.0 * 0.5))


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
