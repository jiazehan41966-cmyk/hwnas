import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BackboneBaselineCLITests(unittest.TestCase):
    def test_backbone_baseline_cli_runs_simplecnn(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "backbone_baseline.yaml"
            output_dir = (tmp_path / "test-results").as_posix()
            config_path.write_text(
                f"""
project:
  name: backbone-baseline-test
  seed: 7
  output_dir: "{output_dir}"
  run_name: cli-smoke

dataset:
  name: dummy
  input_channels: 1
  image_size: 64
  num_classes: 4
  num_workers: 0
  num_train_samples: 32
  num_val_samples: 16

hardware:
  board: zynq7020
  clock_mhz: 200
  quantization_bits: 8
  cpu_latency_warmup: 1
  cpu_latency_runs: 1

training:
  optimizer: adamw
  batch_size: 8
  epochs: 1
  lr: 0.001
  min_lr: 0.00001
  weight_decay: 0.0001
  warmup_epochs: 0
  early_stopping_patience: null
  dropout: 0.2

baseline:
  strict_pretrained: false
  topk_accuracy: 4
  candidates:
    - arch_id: simplecnn
      name: simplecnn
      display_name: SimpleCNN
      pretrained: false
                """.strip(),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "run_backbone_baseline.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=str(repo_root),
                check=True,
            )

            summary_path = tmp_path / "test-results" / "cli-smoke" / "results" / "backbone_summary.json"
            selected_pool_path = tmp_path / "test-results" / "cli-smoke" / "results" / "selected_backbone_pool.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(selected_pool_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["arch_id"], "simplecnn")
            selected_pool = json.loads(selected_pool_path.read_text(encoding="utf-8"))
            self.assertIn("roles", selected_pool)
            self.assertIn("accuracy_anchor", selected_pool["roles"])
            self.assertIn("search_anchor", selected_pool["roles"])
            self.assertIn("lightweight_anchor", selected_pool["roles"])
            self.assertIn("recommended_profiles", selected_pool)
            self.assertIn("mobile_anchor", selected_pool["recommended_profiles"])
            self.assertIn("accuracy_biased", selected_pool["recommended_profiles"])
            self.assertIn("lightweight_sonar", selected_pool["recommended_profiles"])


if __name__ == "__main__":
    unittest.main()
