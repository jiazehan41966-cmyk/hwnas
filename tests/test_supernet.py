import unittest
from random import Random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.search.supernet import (
    MobileAnchorSupernet,
    SlicedBatchNorm2d,
    choice_key,
    evaluate_candidate,
    parse_choice_key,
    train_supernet,
)
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


def _small_config(**overrides) -> SearchSpaceConfig:
    payload = {
        "family_profile": "mobile_anchor",
        "input_channels": 1,
        "image_size": 32,
        "num_classes": 4,
        "op_choices": ("mbconv", "skip"),
    }
    payload.update(overrides)
    return SearchSpaceConfig.from_dict(payload)


def _dummy_loader(num_samples: int = 24, image_size: int = 32, num_classes: int = 4) -> DataLoader:
    torch.manual_seed(0)
    inputs = torch.randn(num_samples, 1, image_size, image_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=8)


class SlicedBatchNormTests(unittest.TestCase):
    def test_slices_track_running_stats_in_place(self) -> None:
        bn = SlicedBatchNorm2d(8)
        bn.train()
        bn(torch.randn(4, 6, 5, 5) + 3.0)
        # Only the first 6 channels were updated.
        self.assertGreater(bn.running_mean[:6].abs().sum().item(), 0.0)
        self.assertEqual(bn.running_mean[6:].abs().sum().item(), 0.0)

    def test_eval_uses_running_stats(self) -> None:
        bn = SlicedBatchNorm2d(4)
        bn.eval()
        x = torch.randn(2, 4, 3, 3)
        out = bn(x)
        self.assertEqual(out.shape, x.shape)


class ChoiceKeyTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        spec = parse_choice_key(choice_key("mbconv", 5, 2))
        self.assertEqual(spec.op, "mbconv")
        self.assertEqual(spec.kernel_size, 5)
        self.assertEqual(spec.expand_ratio, 2)

    def test_skip(self) -> None:
        spec = parse_choice_key("skip")
        self.assertEqual(spec.op, "skip")
        self.assertEqual(spec.kernel_size, 1)


class SupernetPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _small_config()
        self.supernet = MobileAnchorSupernet(self.config, num_classes=4)
        self.space = SearchSpace(self.config)

    def test_sampled_paths_are_valid_architectures(self) -> None:
        for seed in range(20):
            path = self.supernet.sample_path(seed=seed)
            architecture = self.supernet.path_to_architecture(path)
            errors = self.space.validate(architecture)
            self.assertEqual(errors, [], f"seed {seed}: {errors}")

    def test_forward_shapes_for_sampled_paths(self) -> None:
        x = torch.randn(2, 1, 32, 32)
        for seed in (0, 1, 2, 3):
            path = self.supernet.sample_path(seed=seed)
            out = self.supernet(x, path)
            self.assertEqual(out.shape, (2, 4))

    def test_paths_share_weights(self) -> None:
        # Two different paths must be served by the same parameter set.
        param_count = sum(p.numel() for p in self.supernet.parameters())
        path_a = self.supernet.sample_path(seed=0)
        path_b = self.supernet.sample_path(seed=7)
        self.assertNotEqual(path_a, path_b)
        self.assertEqual(
            param_count, sum(p.numel() for p in self.supernet.parameters())
        )

    def test_deterministic_sampling(self) -> None:
        self.assertEqual(
            self.supernet.sample_path(seed=5), self.supernet.sample_path(seed=5)
        )

    def test_rejects_space_without_stage_choices(self) -> None:
        with self.assertRaises(ValueError):
            MobileAnchorSupernet(SearchSpaceConfig(num_classes=4))


class SupernetTrainEvalTests(unittest.TestCase):
    def test_train_and_evaluate_candidate_smoke(self) -> None:
        torch.manual_seed(0)
        config = _small_config()
        supernet = MobileAnchorSupernet(config, num_classes=4)
        loader = _dummy_loader()

        history = train_supernet(
            supernet,
            loader,
            epochs=2,
            lr=1e-3,
            warmup_epochs=1,
            device="cpu",
            verbose=False,
        )
        self.assertEqual(len(history["train_loss"]), 2)

        path = supernet.sample_path(seed=3)
        summary = evaluate_candidate(
            supernet,
            path,
            bn_loader=loader,
            eval_loader=loader,
            num_classes=4,
            bn_batches=2,
            device="cpu",
        )
        self.assertIn("macro_f1", summary)
        self.assertIn("architecture", summary)
        self.assertGreaterEqual(summary["top1"], 0.0)

    def test_bn_recalibration_changes_stats(self) -> None:
        config = _small_config()
        supernet = MobileAnchorSupernet(config, num_classes=4)
        loader = _dummy_loader()
        path = supernet.sample_path(seed=1)

        from hwnas_fpga.search.supernet import recalibrate_bn

        recalibrate_bn(supernet, path, loader, num_batches=2, device="cpu")
        touched = [
            module
            for module in supernet.modules()
            if isinstance(module, SlicedBatchNorm2d)
            and module.num_batches_tracked.item() > 0
        ]
        self.assertTrue(touched)
        # Momentum must be restored after recalibration.
        self.assertTrue(
            all(module.momentum is not None for module, in
                ((m,) for m in supernet.modules() if isinstance(m, SlicedBatchNorm2d)))
        )


if __name__ == "__main__":
    unittest.main()
