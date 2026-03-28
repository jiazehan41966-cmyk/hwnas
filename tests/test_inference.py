import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch
import yaml
from PIL import Image

from hwnas_fpga.deploy import (
    load_checkpoint_model,
    predict_image,
    resolve_inference_settings,
)
from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


class InferenceTests(unittest.TestCase):
    def test_checkpoint_to_prediction_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            config = {
                "dataset": {
                    "image_size": 64,
                    "input_channels": 1,
                }
            }
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)

            space = SearchSpace(SearchSpaceConfig(image_size=64))
            architecture = space.baseline_architecture()
            model = build_model(architecture, num_classes=8)
            checkpoint_path = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "candidate": {
                        "arch_id": "arch_0",
                        "encoding": architecture.to_dict(),
                        "metrics": {},
                    },
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            image_path = root / "sample.png"
            image = Image.fromarray(np.full((64, 64), 127, dtype=np.uint8), mode="L")
            image.save(image_path)

            image_size, input_channels = resolve_inference_settings(checkpoint_path)
            loaded_model, loaded_architecture, _, class_names = load_checkpoint_model(checkpoint_path)
            result = predict_image(
                loaded_model,
                image_path,
                image_size=image_size,
                class_names=class_names,
                input_channels=input_channels,
            )

            self.assertEqual(image_size, 64)
            self.assertEqual(input_channels, 1)
            self.assertEqual(loaded_architecture, architecture)
            self.assertEqual(len(result["topk"]), 3)
            self.assertIn("predicted_class_name", result)

    def test_retrained_checkpoint_schema_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            config = {
                "dataset": {
                    "image_size": 64,
                    "input_channels": 1,
                }
            }
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)

            space = SearchSpace(SearchSpaceConfig(image_size=64))
            architecture = space.baseline_architecture()
            model = build_model(architecture, num_classes=8)
            checkpoint_path = checkpoints_dir / "final_best_model.pt"
            torch.save(
                {
                    "source_candidate": {
                        "arch_id": "arch_0",
                        "encoding": architecture.to_dict(),
                        "metrics": {},
                    },
                    "architecture": architecture.to_dict(),
                    "metrics": {"best_val_acc": 0.8},
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            loaded_model, loaded_architecture, payload, class_names = load_checkpoint_model(checkpoint_path)
            self.assertEqual(loaded_architecture, architecture)
            self.assertIn("architecture", payload)
            self.assertEqual(len(class_names), 8)
            self.assertIsNotNone(loaded_model)

    def test_resolve_inference_settings_prefers_cli_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump({"dataset": {"image_size": 224, "input_channels": 1}}, handle)
            (run_dir / "cli_args.json").write_text(
                json.dumps({"image_size": 32}),
                encoding="utf-8",
            )

            space = SearchSpace(SearchSpaceConfig(image_size=64))
            architecture = space.baseline_architecture()
            model = build_model(architecture, num_classes=8)
            checkpoint_path = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "candidate": {
                        "arch_id": "arch_0",
                        "encoding": architecture.to_dict(),
                        "metrics": {},
                    },
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            image_size, input_channels = resolve_inference_settings(checkpoint_path)
            self.assertEqual(image_size, 32)
            self.assertEqual(input_channels, 1)


if __name__ == "__main__":
    unittest.main()
