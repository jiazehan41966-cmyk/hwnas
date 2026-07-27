import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from hwnas_fpga.data import (
    GROUP_MANIFEST_SCHEMA,
    build_group_protocol_split,
    get_sonar_transforms,
    load_group_manifest,
    sonar_preprocessing_contract,
)
from hwnas_fpga.training.protocol_reporting import protocol_claimability


class GeometryContractTests(unittest.TestCase):
    def test_augmentation_none_is_deterministic(self) -> None:
        image = Image.new("L", (20, 10), color=128)
        transform = get_sonar_transforms(
            image_size=32,
            is_training=True,
            normalize=False,
            augmentation_profile="none",
            geometry_mode="stretch_224",
        )
        self.assertTrue(torch.equal(transform(image), transform(image)))

    def test_letterbox_retains_aspect_ratio(self) -> None:
        image = Image.new("L", (20, 10), color=255)
        transform = get_sonar_transforms(
            image_size=32,
            is_training=False,
            normalize=False,
            augmentation_profile="none",
            geometry_mode="letterbox_224",
        )
        tensor = transform(image)[0]
        rows = torch.where(tensor.sum(dim=1) > 0)[0]
        columns = torch.where(tensor.sum(dim=0) > 0)[0]
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(columns), 32)

    def test_fixed_scale_contract_records_frozen_denominator(self) -> None:
        contract = sonar_preprocessing_contract(
            image_size=224,
            augmentation_profile="sonar_light",
            geometry_mode="fixed_scale_pad_224",
        )
        self.assertEqual(contract["fixed_source_side"], 714)
        self.assertIn("relative", contract["claim_boundary"])

    def test_invalid_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "augmentation_profile"):
            get_sonar_transforms(augmentation_profile="mystery")


class GroupContractTests(unittest.TestCase):
    def _provided_manifest(self, root: Path):
        samples = []
        rows = []
        for label in range(2):
            class_dir = root / f"class_{label}"
            class_dir.mkdir()
            for item in range(10):
                path = class_dir / f"img_{item}.png"
                Image.new("L", (8, 8), color=label * 50 + item).save(path)
                sample_index = len(samples)
                samples.append((str(path), label))
                rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_path": path.relative_to(root).as_posix(),
                        "label": label,
                        "group_id": f"provided_c{label}_g{item}",
                        "group_source": "provided",
                        "original_width": 8,
                        "original_height": 8,
                        "sha256": "not-used-by-loader",
                    }
                )
        payload = {
            "schema_version": 1,
            "schema": GROUP_MANIFEST_SCHEMA,
            "group_policy": "provided",
            "samples": rows,
        }
        manifest = root / "groups.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return samples, manifest

    def test_group_split_is_disjoint_and_provided_is_claimable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples, manifest = self._provided_manifest(root)
            resolved = load_group_manifest(
                manifest, samples, root, policy="provided"
            )
            split = build_group_protocol_split(
                samples,
                resolved,
                fold_index=0,
                seed=42,
                inner_val_fraction=0.2,
                num_classes=2,
            )
            groups = resolved.group_ids
            train_groups = {groups[index] for index in split.train_indices}
            inner_groups = {groups[index] for index in split.inner_val_indices}
            outer_groups = {groups[index] for index in split.outer_val_indices}
            self.assertFalse(train_groups & inner_groups)
            self.assertFalse(train_groups & outer_groups)
            self.assertFalse(inner_groups & outer_groups)
            self.assertTrue(split.metadata["group_generalization_claimable"])

    def test_inferred_split_does_not_become_group_claimable(self) -> None:
        pairs = [(fold, seed) for fold in range(5) for seed in (42, 43, 44)]
        result = protocol_claimability(
            folds=range(5),
            seeds=(42, 43, 44),
            completed_pairs=pairs,
            selection_provenance="baseline_predeclared",
            provenance_fingerprints=["a" * 64] * 15,
            group_split_available=True,
            group_generalization_claimable=False,
        )
        self.assertTrue(result["group_split_available"])
        self.assertFalse(result["group_generalization_claimable"])


if __name__ == "__main__":
    unittest.main()
