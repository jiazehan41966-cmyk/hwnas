import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hwnas_fpga.data import NKSID_CLASSES, NKSIDDataset, create_nksid_dataloaders


class NKSIDDatasetTests(unittest.TestCase):
    def _create_fake_nksid_root(self, root: Path) -> Path:
        dataset_root = root / "NKSID"
        dataset_root.mkdir()

        samples = []
        for label, class_name in enumerate(NKSID_CLASSES):
            class_dir = dataset_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image_path = class_dir / f"{class_name}_0.png"
            image = Image.new("L", (16, 16), color=64 + label)
            image.save(image_path)
            samples.append((f"{class_name}/{image_path.name}", label))

        with (dataset_root / "train_abs.txt").open("w", encoding="utf-8") as handle:
            for relative_path, label in samples:
                handle.write(f"{relative_path} {label}\n")

        train_indices = " ".join(str(index) for index in range(len(samples) - 2))
        val_indices = " ".join(str(index) for index in range(len(samples) - 2, len(samples)))
        with (dataset_root / "kfold_train.txt").open("w", encoding="utf-8") as handle:
            handle.write(train_indices + "\n")
        with (dataset_root / "kfold_val.txt").open("w", encoding="utf-8") as handle:
            handle.write(val_indices + "\n")

        return dataset_root

    def test_dataset_reads_official_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self._create_fake_nksid_root(Path(tmpdir))
            dataset = NKSIDDataset(
                data_dir=str(dataset_root),
                image_size=32,
                fold=0,
                split="train",
            )

            self.assertEqual(len(dataset), len(NKSID_CLASSES) - 2)
            image, label = dataset[0]
            self.assertEqual(image.shape, (1, 32, 32))
            self.assertGreaterEqual(label, 0)
            self.assertEqual(set(dataset.get_class_distribution().keys()), set(NKSID_CLASSES))

    def test_dataloader_factory_accepts_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._create_fake_nksid_root(repo_root)
            train_loader, val_loader, class_weights = create_nksid_dataloaders(
                data_dir=str(repo_root),
                batch_size=2,
                image_size=32,
                fold=0,
                num_workers=0,
                pin_memory=False,
            )

            self.assertEqual(len(train_loader.dataset) + len(val_loader.dataset), len(NKSID_CLASSES))
            self.assertEqual(class_weights.numel(), len(NKSID_CLASSES))


if __name__ == "__main__":
    unittest.main()
