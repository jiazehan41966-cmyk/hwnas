import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_backbone_baseline import compare_results, select_backbone_pool


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
            protocol_path = tmp_path / "test-results" / "cli-smoke" / "results" / "experiment_protocol.json"
            backbone_result_path = (
                tmp_path / "test-results" / "cli-smoke" / "results" / "backbones" / "simplecnn.json"
            )
            self.assertTrue(summary_path.exists())
            self.assertTrue(selected_pool_path.exists())
            self.assertTrue(protocol_path.exists())
            self.assertTrue(backbone_result_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["arch_id"], "simplecnn")

            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertIn("training_strategy", protocol)
            self.assertIn("candidate_blueprints", protocol)
            self.assertEqual(protocol["candidate_blueprints"][0]["stage_count"], 3)

            backbone_result = json.loads(backbone_result_path.read_text(encoding="utf-8"))
            self.assertIn("architecture_summary", backbone_result)
            self.assertIn("training_strategy", backbone_result)
            self.assertEqual(backbone_result["architecture_summary"]["stage_count"], 3)
            self.assertEqual(
                backbone_result["architecture_summary"]["stages"][0]["output_resolution"],
                32,
            )
            self.assertEqual(
                backbone_result["training_strategy"]["selection"]["selection_metric"],
                "macro_f1",
            )

            selected_pool = json.loads(selected_pool_path.read_text(encoding="utf-8"))
            self.assertIn("roles", selected_pool)
            self.assertIn("accuracy_anchor", selected_pool["roles"])
            self.assertIn("search_anchor", selected_pool["roles"])
            self.assertIn("lightweight_anchor", selected_pool["roles"])
            self.assertIn("recommended_profiles", selected_pool)
            self.assertIn("mobile_anchor", selected_pool["recommended_profiles"])
            self.assertIn("accuracy_biased", selected_pool["recommended_profiles"])
            self.assertIn("lightweight_sonar", selected_pool["recommended_profiles"])


class BackboneBaselineSelectionTests(unittest.TestCase):
    @staticmethod
    def _result(
        arch_id: str,
        backbone: str,
        *,
        macro_f1: float,
        top1: float,
        fpga_latency_ms: float,
        cpu_latency_ms: float,
        params: int = 1_000_000,
        feasible: bool = True,
        pretrained_loaded: bool = False,
    ) -> dict:
        return {
            "arch_id": arch_id,
            "display_name": arch_id,
            "candidate": {
                "arch_id": arch_id,
                "name": backbone,
                "display_name": arch_id,
                "pretrained": pretrained_loaded,
            },
            "build_metadata": {
                "name": backbone,
                "pretrained_loaded": pretrained_loaded,
            },
            "evaluation": {
                "macro_f1": macro_f1,
                "top1": top1,
                "top5": 1.0,
                "weighted_f1": macro_f1,
            },
            "cost_estimate": {
                "params": params,
                "macs": params * 10,
                "cpu_latency_ms": cpu_latency_ms,
                "latency_ms": fpga_latency_ms,
                "peak_dsp": 100,
                "peak_bram": 100,
                "peak_lut": 1000,
                "power_w": 5.0,
                "feasible": feasible,
                "violations": [],
            },
        }

    def test_compare_results_honors_configured_fpga_tiebreak(self) -> None:
        best = self._result(
            "best",
            "mobilenet_v2",
            macro_f1=0.90,
            top1=0.95,
            fpga_latency_ms=3.0,
            cpu_latency_ms=1.0,
        )
        current = self._result(
            "current",
            "mobilenet_v2",
            macro_f1=0.90,
            top1=0.95,
            fpga_latency_ms=2.0,
            cpu_latency_ms=2.0,
        )

        self.assertFalse(compare_results(current, best, latency_tiebreak_metric="cpu_latency_ms"))
        self.assertTrue(compare_results(current, best, latency_tiebreak_metric="fpga_latency_ms"))

    def test_select_backbone_pool_honors_lightweight_metric(self) -> None:
        cpu_fast = self._result(
            "cpu_fast",
            "mobilenet_v2",
            macro_f1=0.90,
            top1=0.95,
            fpga_latency_ms=4.0,
            cpu_latency_ms=1.0,
            params=2_000_000,
        )
        fpga_fast = self._result(
            "fpga_fast",
            "shufflenet_v2",
            macro_f1=0.89,
            top1=0.94,
            fpga_latency_ms=2.0,
            cpu_latency_ms=3.0,
            params=1_500_000,
        )

        pool = select_backbone_pool(
            [cpu_fast, fpga_fast],
            baseline_cfg={
                "selection_metric": "macro_f1",
                "ranking": {
                    "latency_tiebreak_metric": "fpga_latency_ms",
                },
                "pool_selection": {
                    "prefer_pretrained_for_search_anchor": False,
                    "lightweight_metric": "cpu_latency_ms",
                    "lightweight_max_macro_drop": 0.10,
                    "search_anchor_families": ["mobilenet_v2", "shufflenet_v2"],
                },
            },
            dataset_name="nksid",
            board_name="alinx_av7k325",
        )

        self.assertEqual(pool["lightweight_metric"], "cpu_latency_ms")
        self.assertEqual(pool["roles"]["lightweight_anchor"]["arch_id"], "cpu_fast")


if __name__ == "__main__":
    unittest.main()
