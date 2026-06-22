from pathlib import Path
import unittest

import yaml


class Phase0V2RouteAwareConfigTests(unittest.TestCase):
    def test_config_keeps_physical_proxy_out_of_reward_and_enables_exploration(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "search"
            / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v2_av7k325.yaml"
        )

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        search_cfg = config["search"]
        weights = search_cfg["objective_weights"]

        self.assertEqual(config["constraints"]["max_dsp"], 1200)
        self.assertEqual(config["constraints"]["physical"]["board_max_dsp"], 840)
        self.assertEqual(search_cfg["pareto"]["topk"], 16)
        self.assertAlmostEqual(weights["resource"], 0.0)
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
