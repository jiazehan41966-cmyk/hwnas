import argparse
import unittest
from pathlib import Path

from scripts.run_g1_baselines import DEFAULT_RL135, command_specs


class G1LauncherTests(unittest.TestCase):
    def test_exactly_45_formal_tasks_are_planned(self) -> None:
        args = argparse.Namespace(
            data_dir="data/NKSID",
            output_dir="results/protocol",
            epochs=150,
            rl135_candidate=str(DEFAULT_RL135),
        )
        specs = command_specs(args)
        self.assertEqual(len(specs), 3)
        self.assertEqual(sum(15 for _ in specs), 45)
        for _, command in specs:
            self.assertIn("0,1,2,3,4", command)
            self.assertIn("42,43,44", command)
            self.assertIn("--save-checkpoints", command)


if __name__ == "__main__":
    unittest.main()
