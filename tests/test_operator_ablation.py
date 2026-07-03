import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OperatorAblationCLITests(unittest.TestCase):
    def test_operator_ablation_cli_runs_single_candidate(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "operator_ablation.yaml"
            output_dir = (tmp_path / "test-results").as_posix()
            config_path.write_text(
                f"""
project:
  name: operator-ablation-test
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

constraints:
  max_latency_ms: 50.0
  max_dsp: 220
  max_bram: 140
  max_lut: 53200
  max_power_w: 10.0

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

ablation:
  macro_template: mobilenet_v2_like
  single_ops:
    - mbconv
  retention:
    keep_margin: 0.02
    conditional_margin: 0.05
                """.strip(),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "run_operator_ablation.py"),
                    "--config",
                    str(config_path),
                    "--candidate",
                    "mbconv_only",
                ],
                cwd=str(repo_root),
                check=True,
            )

            results_root = tmp_path / "test-results" / "cli-smoke" / "results"
            summary_path = results_root / "operator_ablation_summary.json"
            retention_path = results_root / "operator_retention_table.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(retention_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            retention = json.loads(retention_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["arch_id"], "mbconv_only")
            self.assertEqual(summary[0]["macro_template"], "mobilenet_v2_like")
            self.assertTrue(any(row["op"] == "mbconv" for row in retention))

    def test_selected_backbone_pool_overrides_config_macro_template(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "operator_ablation.yaml"
            pool_path = tmp_path / "selected_backbone_pool.json"
            output_dir = (tmp_path / "test-results").as_posix()

            config_path.write_text(
                f"""
project:
  name: operator-ablation-test
  seed: 7
  output_dir: "{output_dir}"
  run_name: cli-selected-pool

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

constraints:
  max_latency_ms: 50.0
  max_dsp: 220
  max_bram: 140
  max_lut: 53200
  max_power_w: 10.0

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

ablation:
  macro_template: fbnet_like
  single_ops:
    - mbconv
  retention:
    keep_margin: 0.02
    conditional_margin: 0.05
                """.strip(),
                encoding="utf-8",
            )
            pool_path.write_text(
                json.dumps(
                    {
                        "roles": {
                            "search_anchor": {
                                "arch_id": "mobilenet_v2_pretrained",
                                "display_name": "MobileNetV2 (pretrained)",
                                "backbone": "mobilenet_v2",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "run_operator_ablation.py"),
                    "--config",
                    str(config_path),
                    "--selected-backbone-pool",
                    str(pool_path),
                    "--candidate",
                    "mbconv_only",
                ],
                cwd=str(repo_root),
                check=True,
            )

            results_root = tmp_path / "test-results" / "cli-selected-pool" / "results"
            summary_path = results_root / "operator_ablation_summary.json"
            search_space_summary_path = results_root / "search_space_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(search_space_summary_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            search_space_summary = json.loads(search_space_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["macro_template"], "mobilenet_v2_like")
            self.assertEqual(summary[0]["macro_template_source"], "selected_backbone_pool")
            self.assertEqual(summary[0]["selected_pool_role"], "search_anchor")
            self.assertEqual(summary[0]["selected_pool_backbone"], "mobilenet_v2")
            self.assertEqual(search_space_summary["macro_template"], "mobilenet_v2_like")
            self.assertEqual(search_space_summary["macro_template_source"], "selected_backbone_pool")
            self.assertEqual(search_space_summary["selected_backbone_pool_role"], "search_anchor")
            self.assertEqual(search_space_summary["selected_backbone_pool_backbone"], "mobilenet_v2")

    def test_structured_backbone_pool_uses_experiment_protocol_blueprint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "operator_ablation.yaml"
            pool_path = tmp_path / "selected_backbone_pool.json"
            experiment_protocol_path = tmp_path / "experiment_protocol.json"
            output_dir = (tmp_path / "test-results").as_posix()

            config_path.write_text(
                f"""
project:
  name: operator-ablation-test
  seed: 7
  output_dir: "{output_dir}"
  run_name: cli-structured-pool

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

constraints:
  max_latency_ms: 50.0
  max_dsp: 220
  max_bram: 140
  max_lut: 53200
  max_power_w: 10.0

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

ablation:
  macro_template: fbnet_like
  selected_backbone_pool:
    prefer_structured_blueprint: true
  single_ops:
    - mbconv
  retention:
    keep_margin: 0.02
    conditional_margin: 0.05
                """.strip(),
                encoding="utf-8",
            )
            pool_path.write_text(
                json.dumps(
                    {
                        "roles": {
                            "search_anchor": {
                                "arch_id": "search_anchor_arch",
                                "display_name": "Search Anchor Net",
                                "backbone": "custom_anchor",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            experiment_protocol_path.write_text(
                json.dumps(
                    {
                        "candidate_blueprints": [
                            {
                                "arch_id": "search_anchor_arch",
                                "display_name": "Search Anchor Net",
                                "backbone": "custom_anchor",
                                "input_channels": 1,
                                "input_resolution": 64,
                                "stem": {
                                    "out_channels": 20,
                                    "width": 20,
                                    "stride": 2,
                                    "input_resolution": 64,
                                    "output_resolution": 32,
                                },
                                "stages": [
                                    {
                                        "stage_index": 1,
                                        "stage_name": "s1",
                                        "depth": 1,
                                        "out_channels": 24,
                                        "width": 24,
                                        "input_resolution": 32,
                                        "output_resolution": 32,
                                        "downsample_location": None,
                                        "downsample_op": None,
                                        "kernel_size": 3,
                                        "expand_ratio": 1,
                                    },
                                    {
                                        "stage_index": 2,
                                        "stage_name": "s2",
                                        "depth": 2,
                                        "out_channels": 40,
                                        "width": 40,
                                        "input_resolution": 32,
                                        "output_resolution": 16,
                                        "downsample_location": "stage_start",
                                        "downsample_op": None,
                                        "kernel_size": 5,
                                        "expand_ratio": 4,
                                    },
                                    {
                                        "stage_index": 3,
                                        "stage_name": "s3",
                                        "depth": 2,
                                        "out_channels": 64,
                                        "width": 64,
                                        "input_resolution": 16,
                                        "output_resolution": 8,
                                        "downsample_location": "stage_start",
                                        "downsample_op": None,
                                        "kernel_size": 5,
                                        "expand_ratio": 6,
                                    },
                                ],
                                "head": {
                                    "out_channels": 128,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "run_operator_ablation.py"),
                    "--config",
                    str(config_path),
                    "--selected-backbone-pool",
                    str(pool_path),
                    "--candidate",
                    "mbconv_only",
                ],
                cwd=str(repo_root),
                check=True,
            )

            results_root = tmp_path / "test-results" / "cli-structured-pool" / "results"
            summary_path = results_root / "operator_ablation_summary.json"
            search_space_summary_path = results_root / "search_space_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(search_space_summary_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            search_space_summary = json.loads(search_space_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["macro_template_source"], "experiment_protocol")
            self.assertEqual(search_space_summary["macro_template_source"], "experiment_protocol")
            self.assertEqual(search_space_summary["macro_blueprint_arch_id"], "search_anchor_arch")
            self.assertEqual(search_space_summary["macro_template"], "search_anchor_arch_blueprint")
            self.assertEqual(search_space_summary["stage_count"], 3)
            self.assertEqual(search_space_summary["stem_channels"], 20)
            self.assertEqual(search_space_summary["head_channels"], 128)
            self.assertEqual(search_space_summary["stage_specs"][1]["channels"], 40)
            self.assertEqual(search_space_summary["stage_specs"][1]["stride"], 2)


if __name__ == "__main__":
    unittest.main()
