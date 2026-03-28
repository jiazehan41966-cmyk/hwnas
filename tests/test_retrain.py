import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.data import create_dummy_dataloaders
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig
from hwnas_fpga.training import load_architecture_from_artifact, retrain_architecture


class RetrainWorkflowTests(unittest.TestCase):
    def test_load_architecture_and_retrain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            space = SearchSpace(SearchSpaceConfig(input_channels=1, num_classes=4))
            architecture = space.baseline_architecture()
            artifact_path = Path(tmpdir) / "best_candidate.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "arch_id": "arch_0",
                            "encoding": architecture.to_dict(),
                            "metrics": {},
                        }
                    }
                ),
                encoding="utf-8",
            )

            loaded_arch = load_architecture_from_artifact(artifact_path)
            self.assertEqual(loaded_arch.stem_channels, architecture.stem_channels)

            train_loader, val_loader = create_dummy_dataloaders(
                batch_size=4,
                image_size=64,
                num_classes=4,
                input_channels=1,
                num_train=16,
                num_val=8,
            )
            model, metrics, history = retrain_architecture(
                architecture=loaded_arch,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=4,
                epochs=1,
                device="cpu",
                early_stopping_patience=None,
            )

            self.assertEqual(model.architecture.stem_channels, architecture.stem_channels)
            self.assertEqual(metrics["epochs_completed"], 1)
            self.assertEqual(len(history["train_acc"]), 1)
            self.assertEqual(len(history["val_acc"]), 1)


if __name__ == "__main__":
    unittest.main()
