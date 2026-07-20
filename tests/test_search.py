import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MethodType
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import CandidateMetrics, HardwareSpec, SearchCandidate, SearchConstraints
from hwnas_fpga.search import (
    ActionSpace,
    AgingEvolutionSearcher,
    Controller,
    RLSearcher,
    RewardFunction,
    RandomSearcher,
    ParetoFrontSelector,
    SearchEfficiencyMonitor,
    architecture_signature,
    build_pareto_objectives,
    build_pareto_representative_roles,
    build_pareto_selection_summary,
    compute_pareto_front,
    compute_pareto_ranks,
    crowding_distances,
    create_searcher,
    write_pareto_selection_artifacts,
    merge_search_efficiency,
    resolve_pareto_objectives,
)
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig
from hwnas_fpga.training import evaluate_sonar_robustness, resolve_sonar_robustness_config


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

    def test_create_multiobjective_aging_evolution_searcher(self) -> None:
        searcher = create_searcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            method="aging_evolution",
            seed=5,
            selection_metric="macro_f1",
            reward_weights={"accuracy": 1.0, "latency": 0.2},
            pareto_cfg={
                "objectives": ["macro_f1", "latency_ms", "lut"],
                "directions": ["max", "min", "min"],
            },
            aging_cfg={"population_size": 6, "sample_size": 3},
        )
        self.assertIsInstance(searcher, AgingEvolutionSearcher)
        self.assertEqual(searcher.population_size, 6)
        self.assertEqual(searcher.sample_size, 3)
        self.assertEqual(searcher.objectives[0], "macro_f1")
        self.assertEqual(searcher.objectives, ["macro_f1", "latency_ms", "lut"])
        self.assertEqual(searcher.survivor_selection, "pareto_crowding")

    def test_explicit_pareto_axes_require_primary_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary selection metric"):
            resolve_pareto_objectives(
                {"objectives": ["latency_ms", "lut"]},
                selection_metric="macro_f1",
            )

    def test_crowding_distance_preserves_objective_boundaries(self) -> None:
        candidates = [
            SearchCandidate(
                f"c{index}",
                {},
                CandidateMetrics(macro_f1=score, latency_ms=latency),
            )
            for index, (score, latency) in enumerate(
                ((0.70, 30.0), (0.75, 20.0), (0.80, 10.0))
            )
        ]
        distances = crowding_distances(candidates, ("macro_f1", "latency_ms"))
        self.assertTrue(math.isinf(distances["c0"]))
        self.assertTrue(math.isinf(distances["c2"]))
        self.assertGreater(distances["c1"], 0.0)

    def test_rank_selector_uses_crowding_for_same_rank(self) -> None:
        candidates = [
            SearchCandidate(
                f"c{index}",
                {},
                CandidateMetrics(f_clean=quality, latency_ms=latency),
            )
            for index, (quality, latency) in enumerate(
                ((0.70, 10.0), (0.75, 20.0), (0.80, 30.0), (0.85, 40.0))
            )
        ]
        selected = ParetoFrontSelector(
            objectives=["f_clean", "latency_ms"],
            directions=["max", "min"],
            selection_method="rank",
        ).select(candidates, k=2)
        self.assertEqual({candidate.arch_id for candidate in selected}, {"c0", "c3"})

    def test_pareto_representative_roles_do_not_create_single_reward(self) -> None:
        front = [
            SearchCandidate(
                "accuracy",
                {},
                CandidateMetrics(f_clean=0.92, f_robust=0.60, latency_ms=18.0, energy_mj=2.0),
            ),
            SearchCandidate(
                "robust",
                {},
                CandidateMetrics(f_clean=0.82, f_robust=0.90, latency_ms=16.0, energy_mj=1.8),
            ),
            SearchCandidate(
                "deployment",
                {},
                CandidateMetrics(f_clean=0.86, f_robust=0.80, latency_ms=9.0, energy_mj=1.0),
            ),
        ]
        roles = build_pareto_representative_roles(
            front,
            objectives=["f_clean", "f_robust", "latency_ms", "energy_mj"],
            directions=["max", "max", "min", "min"],
        )
        self.assertEqual(roles["roles"]["accuracy_first"]["arch_id"], "accuracy")
        self.assertEqual(roles["roles"]["sonar_robust"]["arch_id"], "robust")
        self.assertIn(
            roles["roles"]["deployment_balanced"]["arch_id"],
            {"accuracy", "robust", "deployment"},
        )
        self.assertFalse(roles["deployment_balanced_method"]["fixed_scalar_reward_weights"])

    def test_aging_evolution_mutation_is_valid_and_unseen(self) -> None:
        search_space = SearchSpace(
            SearchSpaceConfig.from_dict(
                {
                    "input_channels": 1,
                    "image_size": 32,
                    "stem_channels": 8,
                    "stage_strides": [1],
                    "stage_channel_choices": [[8]],
                    "stage_depth_choices": [[1]],
                    "op_choices": ["conv", "mbconv"],
                    "kernel_choices": [3],
                    "expand_choices": [1, 2],
                    "stage_block_choices": [
                        [
                            {"op": "conv", "kernel_size": 3, "expand_ratio": 1},
                            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 2},
                        ]
                    ],
                    "num_classes": 2,
                }
            )
        )
        searcher = AgingEvolutionSearcher(
            search_space=search_space,
            cost_estimator=self.estimator,
            constraints=self.constraints,
            seed=17,
            population_size=2,
            sample_size=2,
        )
        parent = searcher.sample_candidate()
        searcher.seen_signatures.add(architecture_signature(parent))
        child, mutation = searcher.mutate_architecture(parent)
        self.assertIsNotNone(child)
        self.assertFalse(search_space.validate(child))
        self.assertNotEqual(architecture_signature(parent), architecture_signature(child))
        self.assertIn(mutation["kind"], {"stage", "block", "random_fallback"})

    def test_aging_evolution_two_parent_crossover_is_valid_and_unseen(self) -> None:
        search_space = SearchSpace(
            SearchSpaceConfig.from_dict(
                {
                    "input_channels": 1,
                    "image_size": 32,
                    "stem_channels": 8,
                    "stage_strides": [1, 2, 2],
                    "stage_channel_choices": [[8], [12], [16]],
                    "stage_depth_choices": [[1], [1], [1]],
                    "op_choices": ["conv", "mbconv"],
                    "kernel_choices": [3],
                    "expand_choices": [1, 2],
                    "num_classes": 2,
                }
            )
        )
        searcher = AgingEvolutionSearcher(
            search_space=search_space,
            cost_estimator=self.estimator,
            constraints=None,
            seed=31,
            population_size=2,
            sample_size=2,
            max_mutation_attempts=128,
        )
        parent_a = searcher.sample_candidate()
        parent_b = None
        for _ in range(128):
            candidate = searcher.sample_candidate()
            differences = sum(
                left != right
                for left, right in zip(parent_a.to_dict()["stages"], candidate.to_dict()["stages"])
            )
            if differences >= 2:
                parent_b = candidate
                break
        self.assertIsNotNone(parent_b)
        searcher.seen_signatures.update(
            {architecture_signature(parent_a), architecture_signature(parent_b)}
        )
        child, metadata = searcher.crossover_architectures(parent_a, parent_b)
        self.assertIsNotNone(child)
        self.assertEqual(metadata["kind"], "crossover")
        self.assertFalse(search_space.validate(child))
        self.assertNotIn(architecture_signature(child), searcher.seen_signatures)

    def test_aging_evolution_ages_population_and_keeps_pareto_archive(self) -> None:
        search_space = SearchSpace(
            SearchSpaceConfig.from_dict(
                {
                    "input_channels": 1,
                    "image_size": 32,
                    "stem_channels": 8,
                    "stage_strides": [1, 2],
                    "stage_channel_choices": [[8], [16]],
                    "stage_depth_choices": [[1], [1]],
                    "op_choices": ["conv", "mbconv"],
                    "kernel_choices": [3],
                    "expand_choices": [1, 2],
                    "stage_block_choices": [
                        [
                            {"op": "conv", "kernel_size": 3, "expand_ratio": 1},
                            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 2},
                        ],
                        [
                            {"op": "conv", "kernel_size": 3, "expand_ratio": 1},
                            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 2},
                        ],
                    ],
                    "num_classes": 2,
                }
            )
        )
        searcher = AgingEvolutionSearcher(
            search_space=search_space,
            cost_estimator=self.estimator,
            constraints=None,
            seed=23,
            population_size=2,
            sample_size=2,
            random_injection_probability=0.0,
            max_mutation_attempts=64,
            survivor_selection="oldest",
            objective_weights={"accuracy": 1.0, "latency": 1.0},
        )

        def fake_evaluate(_self, architecture, *_args, **_kwargs):
            index = len(_self.evaluated_candidates)
            candidate = SearchCandidate(
                f"temporary_{index}",
                architecture.to_dict(),
                CandidateMetrics(
                    accuracy=0.70 + index * 0.01,
                    macro_f1=0.60 + index * 0.02,
                    top1=0.70 + index * 0.01,
                    selection_score=0.60 + index * 0.02,
                    latency_ms=20.0 - index,
                    energy_mj=1.0,
                    dsp=10,
                    bram=10,
                    lut=100,
                ),
            )
            _self.evaluated_candidates.append(candidate)
            _self.feasible_candidates.append(candidate)
            _self.last_training_history = {"best_eval": {"macro_f1": candidate.metrics.macro_f1}}
            _self.last_trained_model = None
            _self.last_cost_estimate = None
            return candidate, True

        searcher.evaluate_candidate = MethodType(fake_evaluate, searcher)
        best = searcher.search(
            [],
            None,
            num_classes=2,
            num_candidates=4,
            train_epochs=0,
            verbose=False,
        )
        self.assertEqual(len(searcher.evaluated_candidates), 4)
        self.assertEqual(len(searcher.population), 2)
        self.assertEqual(
            [candidate.arch_id for candidate in searcher.population],
            ["aging_arch_2", "aging_arch_3"],
        )
        self.assertGreaterEqual(len(searcher.pareto_archive), 1)
        self.assertIs(best, searcher.best_candidate)
        self.assertEqual(best.arch_id, "aging_arch_3")

    def test_pareto_crowding_survivor_selection_keeps_boundaries(self) -> None:
        searcher = AgingEvolutionSearcher(
            search_space=self.search_space,
            cost_estimator=self.estimator,
            constraints=None,
            seed=7,
            population_size=2,
            sample_size=2,
            selection_metric="f_clean",
            pareto_config={
                "objectives": ["f_clean", "latency_ms"],
                "directions": ["max", "min"],
            },
            survivor_selection="pareto_crowding",
        )
        candidates = [
            SearchCandidate("fast", {}, CandidateMetrics(f_clean=0.70, latency_ms=10.0)),
            SearchCandidate("middle", {}, CandidateMetrics(f_clean=0.80, latency_ms=20.0)),
            SearchCandidate("accurate", {}, CandidateMetrics(f_clean=0.90, latency_ms=30.0)),
        ]
        searcher.population.extend(candidates)
        searcher.birth_indices.update({candidate.arch_id: index for index, candidate in enumerate(candidates)})
        removed = searcher._trim_population()
        self.assertEqual(removed, ["middle"])
        self.assertEqual({candidate.arch_id for candidate in searcher.population}, {"fast", "accurate"})

    def test_sonar_robustness_protocol_is_deterministic(self) -> None:
        torch.manual_seed(5)
        inputs = torch.rand(12, 1, 8, 8) * 2.0 - 1.0
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 3))
        config = resolve_sonar_robustness_config({"enabled": True, "seed": 123})
        first = evaluate_sonar_robustness(
            model,
            loader,
            device="cpu",
            num_classes=3,
            class_weights=None,
            config=config,
        )
        second = evaluate_sonar_robustness(
            model,
            loader,
            device="cpu",
            num_classes=3,
            class_weights=None,
            config=config,
        )
        self.assertEqual(first["protocol_sha256"], second["protocol_sha256"])
        self.assertEqual(first["condition_results"], second["condition_results"])
        self.assertAlmostEqual(first["f_robust"], second["f_robust"])

    def test_cpu_search_efficiency_monitor_has_explicit_zero_gpu_hours(self) -> None:
        with SearchEfficiencyMonitor("cpu") as monitor:
            sum(range(1000))
        summary = monitor.summary(
            candidate_count=4,
            feasible_count=3,
            search_method="aging_evolution",
        )
        self.assertFalse(summary["cuda_used"])
        self.assertEqual(summary["gpu_reserved_hours"], 0.0)
        self.assertIsNone(summary["gpu_event_seconds"])
        self.assertGreaterEqual(summary["wall_clock_seconds"], 0.0)

    def test_search_efficiency_accumulates_resume_segments(self) -> None:
        first = {
            "search_method": "aging_evolution",
            "cuda_used": True,
            "wall_clock_seconds": 100.0,
            "host_process_seconds": 20.0,
            "gpu_reserved_wall_seconds": 100.0,
            "gpu_reserved_hours": 100.0 / 3600.0,
            "gpu_event_seconds": 70.0,
            "peak_cuda_memory_bytes": 1000,
            "candidate_count": 4,
            "feasible_candidate_count": 3,
        }
        second = {
            **first,
            "wall_clock_seconds": 50.0,
            "host_process_seconds": 10.0,
            "gpu_reserved_wall_seconds": 50.0,
            "gpu_reserved_hours": 50.0 / 3600.0,
            "gpu_event_seconds": 35.0,
            "peak_cuda_memory_bytes": 1200,
            "candidate_count": 2,
            "feasible_candidate_count": 2,
        }
        merged = merge_search_efficiency(first, second)
        self.assertEqual(merged["segment_count"], 2)
        self.assertEqual(merged["candidate_count"], 6)
        self.assertEqual(merged["feasible_candidate_count"], 5)
        self.assertAlmostEqual(merged["wall_clock_seconds"], 150.0)
        self.assertAlmostEqual(merged["gpu_reserved_hours"], 150.0 / 3600.0)
        self.assertAlmostEqual(merged["gpu_event_seconds"], 105.0)
        self.assertEqual(merged["peak_cuda_memory_bytes"], 1200)

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

    def test_reward_is_invariant_to_candidate_evaluation_order(self) -> None:
        reward_kwargs = {
            "normalization_scales": {
                "accuracy": 1.0,
                "latency": 50.0,
                "energy": 100.0,
                "dsp": 512.0,
                "bram": 2000.0,
                "lut": 120000.0,
            }
        }
        first = RewardFunction(**reward_kwargs)
        reward_a_first = first.compute_reward(
            accuracy=0.7, latency_ms=10.0, energy_mj=20.0,
            dsp=100, bram=200, lut=10000,
        )
        first.compute_reward(
            accuracy=0.9, latency_ms=40.0, energy_mj=80.0,
            dsp=400, bram=1000, lut=90000,
        )

        second = RewardFunction(**reward_kwargs)
        second.compute_reward(
            accuracy=0.9, latency_ms=40.0, energy_mj=80.0,
            dsp=400, bram=1000, lut=90000,
        )
        reward_a_second = second.compute_reward(
            accuracy=0.7, latency_ms=10.0, energy_mj=20.0,
            dsp=100, bram=200, lut=10000,
        )

        self.assertAlmostEqual(reward_a_first, reward_a_second)

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

    def test_build_pareto_objectives_can_include_physical_risk(self) -> None:
        objectives, directions = build_pareto_objectives(
            {"physical_risk": 0.0},
            SearchConstraints(physical={"enabled": True, "pareto_physical_risk": True}),
            selection_metric="macro_f1",
        )
        self.assertEqual(objectives[0], "macro_f1")
        self.assertIn("physical_risk", objectives)
        self.assertEqual(directions[objectives.index("physical_risk")], "min")

    def test_build_pareto_objectives_can_include_power_from_pareto_flag(self) -> None:
        objectives, directions = build_pareto_objectives(
            {},
            SearchConstraints(physical={"enabled": True, "pareto_power": True}),
            selection_metric="macro_f1",
        )
        self.assertIn("power_w", objectives)
        self.assertEqual(directions[objectives.index("power_w")], "min")

    def test_physical_risk_changes_pareto_rank(self) -> None:
        objectives = [
            "macro_f1",
            "latency_ms",
            "dsp",
            "bram",
            "lut",
            "power_w",
            "physical_risk",
        ]
        directions = ["max", "min", "min", "min", "min", "min", "min"]
        high_risk = SearchCandidate(
            "high_risk",
            {"layers": [{"op": "mbconv", "expand_ratio": 6}]},
            CandidateMetrics(
                macro_f1=0.82,
                top1=0.9,
                latency_ms=7.0,
                dsp=100,
                bram=40,
                lut=50000,
                power_w=10.0,
                physical_risk=0.9,
            ),
        )
        low_risk = SearchCandidate(
            "low_risk",
            {"layers": [{"op": "mbconv", "expand_ratio": 3}]},
            CandidateMetrics(
                macro_f1=0.82,
                top1=0.9,
                latency_ms=7.0,
                dsp=100,
                bram=40,
                lut=50000,
                power_w=10.0,
                physical_risk=0.1,
            ),
        )

        ranks = compute_pareto_ranks([high_risk, low_risk], objectives, directions)

        self.assertEqual(ranks, [1, 0])

    def test_pareto_artifacts_include_ranked_physical_metrics(self) -> None:
        objectives = ["macro_f1", "latency_ms", "physical_risk"]
        directions = ["max", "min", "min"]
        selected = SearchCandidate(
            "selected_low_risk",
            {"stage": 0, "expand_ratio": 3},
            CandidateMetrics(
                macro_f1=0.81,
                top1=0.88,
                latency_ms=6.5,
                dsp=90,
                bram=35,
                lut=45000,
                power_w=9.2,
                physical_risk=0.2,
            ),
        )
        dominated = SearchCandidate(
            "dominated_high_risk",
            {"stage": 0, "expand_ratio": 6},
            CandidateMetrics(
                macro_f1=0.81,
                top1=0.88,
                latency_ms=6.5,
                dsp=90,
                bram=35,
                lut=45000,
                power_w=9.2,
                physical_risk=0.8,
            ),
        )
        candidates = [dominated, selected]
        pareto_front = compute_pareto_front(candidates, objectives, directions)
        ranks = compute_pareto_ranks(candidates, objectives, directions)
        summary = build_pareto_selection_summary(
            candidates=candidates,
            pareto_front=pareto_front,
            selected_candidates=[selected],
            ranks=ranks,
            objectives=objectives,
            directions=directions,
            selection_method="rank",
            topk=1,
            hypervolume=0.0,
        )

        self.assertIn("physical_risk", summary["objectives"])
        self.assertIn("physical_risk", summary["physical_objectives"])
        self.assertIn("physical_risk", summary["metric_columns"])
        rows = {row["arch_id"]: row for row in summary["ranked_candidates"]}
        self.assertEqual(rows["selected_low_risk"]["rank"], 0)
        self.assertEqual(rows["dominated_high_risk"]["rank"], 1)
        self.assertEqual(
            rows["selected_low_risk"]["encoding"],
            {"stage": 0, "expand_ratio": 3},
        )
        self.assertEqual(
            rows["dominated_high_risk"]["encoding"],
            {"stage": 0, "expand_ratio": 6},
        )
        self.assertEqual(
            rows["selected_low_risk"]["objective_values"]["physical_risk"],
            0.2,
        )

        with TemporaryDirectory() as tmpdir:
            write_pareto_selection_artifacts(tmpdir, summary)
            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "pareto_selection.json").exists())
            csv_text = (output_dir / "pareto_ranked_candidates.csv").read_text(encoding="utf-8")
            self.assertIn("physical_risk", csv_text.splitlines()[0])
            self.assertIn("selected_low_risk", csv_text)

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
