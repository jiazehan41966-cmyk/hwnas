import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hwnas_fpga.data import NKSID_CLASSES, NKSIDDataset
from hwnas_fpga.data.protocol import (
    ProtocolError,
    build_protocol_split,
    contiguous_inner_split,
    filename_number,
    load_outer_folds,
)


def _write_fake_nksid(root: Path, *, samples_per_class: int = 10) -> tuple[Path, list]:
    """Create a fake NKSID tree with a valid labeled 5-fold partition."""
    dataset_root = root / "NKSID"
    dataset_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for label, class_name in enumerate(NKSID_CLASSES):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for number in range(samples_per_class):
            image_path = class_dir / f"img_{number}.png"
            Image.new("L", (16, 16), color=32 + label).save(image_path)
            samples.append((f"{class_name}/img_{number}.png", label))

    with (dataset_root / "train_abs.txt").open("w", encoding="utf-8") as handle:
        for relative_path, label in samples:
            handle.write(f"{relative_path} {label}\n")

    total = len(samples)
    # Round-robin assignment yields five disjoint validation folds.
    fold_val = {k: [i for i in range(total) if i % 5 == k] for k in range(5)}
    with (dataset_root / "kfold_val.txt").open("w", encoding="utf-8") as val_handle, (
        dataset_root / "kfold_train.txt"
    ).open("w", encoding="utf-8") as train_handle:
        for k in range(5):
            val_indices = fold_val[k]
            train_indices = [i for i in range(total) if i % 5 != k]
            val_handle.write(f"#p0-k{k}\n")
            val_handle.write(" ".join(str(i) for i in val_indices) + "\n")
            train_handle.write(f"#p0-k{k}\n")
            train_handle.write(" ".join(str(i) for i in train_indices) + "\n")

    return dataset_root, samples


class OuterFoldTests(unittest.TestCase):
    def test_loads_and_verifies_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, samples = _write_fake_nksid(Path(tmpdir))
            folds = load_outer_folds(dataset_root, num_samples=len(samples))

            self.assertEqual(len(folds), 5)
            covered = set()
            for fold in folds:
                val_set = set(fold.val_indices)
                self.assertFalse(covered & val_set)
                covered |= val_set
                self.assertEqual(
                    set(fold.train_indices), set(range(len(samples))) - val_set
                )
            self.assertEqual(covered, set(range(len(samples))))

    def test_rejects_overlapping_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, samples = _write_fake_nksid(Path(tmpdir))
            val_file = dataset_root / "kfold_val.txt"
            lines = val_file.read_text(encoding="utf-8").splitlines()
            # Duplicate fold 0 indices into fold 1 to force an overlap.
            lines[3] = lines[1]
            val_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(ProtocolError):
                load_outer_folds(dataset_root, num_samples=len(samples))

    def test_missing_split_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, _ = _write_fake_nksid(Path(tmpdir))
            (dataset_root / "kfold_val.txt").unlink()
            with self.assertRaises(ProtocolError):
                load_outer_folds(dataset_root)


class InnerSplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_respects_fraction(self) -> None:
        samples = [
            (f"floats/img_{number}.png", 0) for number in range(40)
        ] + [
            (f"tire/img_{number}.png", 1) for number in range(20)
        ]
        outer_train = list(range(len(samples)))
        train_indices, val_indices = contiguous_inner_split(
            samples, outer_train, val_fraction=0.2, seed=7
        )

        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertEqual(
            sorted(set(train_indices) | set(val_indices)), outer_train
        )
        class0_val = [i for i in val_indices if samples[i][1] == 0]
        class1_val = [i for i in val_indices if samples[i][1] == 1]
        self.assertEqual(len(class0_val), 8)
        self.assertEqual(len(class1_val), 4)

    def test_validation_block_is_contiguous_by_filename_number(self) -> None:
        samples = [(f"floats/img_{number}.png", 0) for number in range(30)]
        _, val_indices = contiguous_inner_split(
            samples, range(30), val_fraction=0.2, seed=3
        )
        numbers = sorted(filename_number(samples[i][0]) for i in val_indices)
        # A wrap-around contiguous block has at most one gap in sorted order.
        gaps = sum(1 for a, b in zip(numbers, numbers[1:]) if b - a > 1)
        self.assertLessEqual(gaps, 1)

    def test_tiny_class_keeps_at_least_one_train_sample(self) -> None:
        samples = [("fishing_net/img_0.png", 0), ("fishing_net/img_1.png", 0)]
        train_indices, val_indices = contiguous_inner_split(
            samples, [0, 1], val_fraction=0.5, seed=1
        )
        self.assertEqual(len(train_indices), 1)
        self.assertEqual(len(val_indices), 1)

    def test_deterministic_per_seed(self) -> None:
        samples = [(f"cylinder/img_{number}.png", 0) for number in range(25)]
        first = contiguous_inner_split(samples, range(25), seed=11)
        second = contiguous_inner_split(samples, range(25), seed=11)
        third = contiguous_inner_split(samples, range(25), seed=12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class ProtocolSplitTests(unittest.TestCase):
    def test_build_protocol_split_never_touches_outer_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, samples = _write_fake_nksid(Path(tmpdir))
            resolved = [
                (path, label) for path, label in samples
            ]
            split = build_protocol_split(
                resolved,
                dataset_root,
                fold_index=2,
                seed=42,
                inner_val_fraction=0.2,
            )

            train = set(split.train_indices)
            inner = set(split.inner_val_indices)
            outer = set(split.outer_val_indices)
            self.assertFalse(train & inner)
            self.assertFalse(train & outer)
            self.assertFalse(inner & outer)
            self.assertEqual(
                sorted(train | inner | outer), list(range(len(samples)))
            )
            counts = split.metadata["outer_val_class_counts"]
            self.assertEqual(sum(counts), len(outer))


class DatasetSplitPolicyTests(unittest.TestCase):
    def test_missing_kfold_file_raises_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, _ = _write_fake_nksid(Path(tmpdir))
            (dataset_root / "kfold_train.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                NKSIDDataset(
                    data_dir=str(dataset_root),
                    image_size=16,
                    fold=0,
                    split="train",
                )

    def test_missing_kfold_file_fallback_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root, samples = _write_fake_nksid(Path(tmpdir))
            (dataset_root / "kfold_train.txt").unlink()
            dataset = NKSIDDataset(
                data_dir=str(dataset_root),
                image_size=16,
                fold=0,
                split="train",
                split_file_policy="fallback",
            )
            self.assertEqual(len(dataset), int(len(samples) * 0.8))


if __name__ == "__main__":
    unittest.main()
