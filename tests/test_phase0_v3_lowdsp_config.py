from pathlib import Path
import unittest

import yaml


class Phase0V3LowDspConfigTests(unittest.TestCase):
    def _load_config(self) -> dict:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "search"
            / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3_lowdsp_av7k325.yaml"
        )
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def test_config_stage_choices_are_exact_lowdsp_v3_space(self) -> None:
        config = self._load_config()
        choices = config["search_space"]["stage_block_choices"]

        self.assertEqual(
            choices,
            [
                [{"op": "conv", "kernel_size": 1, "expand_ratio": 1}],
                [
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
                ],
                [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3}],
                [
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
                    {"op": "skip", "kernel_size": 1, "expand_ratio": 1},
                ],
            ],
        )
        self.assertNotIn(5, config["search_space"]["kernel_choices"])
        self.assertFalse(
            any(
                block.get("kernel_size") == 5
                for stage_choices in choices
                for block in stage_choices
            )
        )

    def test_config_keeps_exploration_and_zero_physical_proxy_reward(self) -> None:
        config = self._load_config()
        search_cfg = config["search"]
        weights = search_cfg["objective_weights"]

        self.assertEqual(config["constraints"]["max_dsp"], 1200)
        self.assertEqual(config["constraints"]["physical"]["board_max_dsp"], 840)
        self.assertEqual(
            config["constraints"]["physical"]["early_expand_limits"][0],
            {"min_input_resolution": 112, "max_expand_ratio": 6},
        )
        self.assertEqual(
            config["constraints"]["physical"]["early_expand_limits"][1],
            {"min_input_resolution": 56, "max_expand_ratio": 6},
        )
        self.assertEqual(search_cfg["pareto"]["topk"], 24)
        self.assertAlmostEqual(weights["resource"], 0.04)
        self.assertAlmostEqual(weights["dsp"], 0.08)
        self.assertAlmostEqual(weights["bram"], 0.04)
        self.assertAlmostEqual(weights["lut"], 0.03)
        self.assertAlmostEqual(weights["latency"], 0.02)
        self.assertAlmostEqual(weights["physical_risk"], 0.0)
        self.assertAlmostEqual(weights["early_expand_pressure"], 0.0)
        self.assertAlmostEqual(search_cfg["controller_lr"], 0.003)
        self.assertAlmostEqual(search_cfg["controller_temperature"], 2.0)
        self.assertAlmostEqual(search_cfg["controller_entropy_coef"], 0.05)
        self.assertAlmostEqual(search_cfg["exploration_epsilon_start"], 0.35)
        self.assertAlmostEqual(search_cfg["exploration_epsilon_end"], 0.10)
        self.assertAlmostEqual(search_cfg["exploration_bonus"], 0.05)


if __name__ == "__main__":
    unittest.main()
