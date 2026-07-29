import math
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.training.recipe import (
    LogitAdjustedCrossEntropy,
    RecipeConfig,
    build_train_criterion,
    build_warmup_cosine_scheduler,
    train_with_recipe,
)


class SchedulerTests(unittest.TestCase):
    def test_warmup_then_cosine_decay(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_warmup_cosine_scheduler(
            optimizer, epochs=20, warmup_epochs=5, min_lr_ratio=0.01
        )

        lrs = []
        for _ in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Linear warmup: first epoch is lr/warmup, fifth reaches full lr.
        self.assertAlmostEqual(lrs[0], 1.0 / 5.0, places=6)
        self.assertAlmostEqual(lrs[4], 1.0, places=6)
        # Monotone decay after warmup, ending near min_lr_ratio.
        self.assertTrue(all(a >= b for a, b in zip(lrs[4:], lrs[5:])))
        self.assertLess(lrs[-1], 0.1)
        self.assertGreaterEqual(lrs[-1], 0.01 - 1e-9)

    def test_zero_warmup_is_valid(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
        scheduler = build_warmup_cosine_scheduler(
            optimizer, epochs=3, warmup_epochs=0, min_lr_ratio=0.1
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5, places=6)
        scheduler.step()


class LogitAdjustmentTests(unittest.TestCase):
    def test_validated_sonar_default_is_plain_smoothed_ce(self) -> None:
        self.assertEqual(RecipeConfig().logit_adjust_tau, 0.0)

    def test_adjustment_shifts_loss_toward_rare_classes(self) -> None:
        counts = [900.0, 100.0]
        criterion = LogitAdjustedCrossEntropy(counts, tau=1.0)
        logits = torch.zeros(1, 2)

        rare_loss = criterion(logits, torch.tensor([1]))
        common_loss = criterion(logits, torch.tensor([0]))
        # With equal logits, predicting the rare class must cost more,
        # pushing the model to compensate for the prior.
        self.assertGreater(rare_loss.item(), common_loss.item())

    def test_tau_zero_matches_plain_cross_entropy(self) -> None:
        counts = [900.0, 100.0]
        criterion = LogitAdjustedCrossEntropy(counts, tau=0.0)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        expected = nn.functional.cross_entropy(logits, targets)
        self.assertAlmostEqual(
            criterion(logits, targets).item(), expected.item(), places=6
        )

    def test_build_train_criterion_dispatch(self) -> None:
        recipe = RecipeConfig(logit_adjust_tau=1.0, label_smoothing=0.1)
        adjusted = build_train_criterion(recipe, class_counts=[10, 20])
        self.assertIsInstance(adjusted, LogitAdjustedCrossEntropy)

        plain_recipe = RecipeConfig(logit_adjust_tau=0.0, label_smoothing=0.1)
        plain = build_train_criterion(plain_recipe, class_counts=[10, 20])
        self.assertIsInstance(plain, nn.CrossEntropyLoss)
        self.assertAlmostEqual(plain.label_smoothing, 0.1)


class TrainWithRecipeTests(unittest.TestCase):
    def test_smoke_train_selects_best_epoch(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(48, 6)
        targets = (inputs.sum(dim=1) > 0).long()
        train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=16)
        val_loader = DataLoader(TensorDataset(inputs, targets), batch_size=16)

        model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))
        recipe = RecipeConfig(
            epochs=4, lr=0.01, warmup_epochs=1, logit_adjust_tau=0.0
        )
        result = train_with_recipe(
            model,
            train_loader=train_loader,
            inner_val_loader=val_loader,
            num_classes=2,
            recipe=recipe,
            device="cpu",
            verbose=False,
        )

        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 4)
        self.assertIn("macro_f1", result.best_inner_eval)
        self.assertEqual(len(result.history["train_loss"]), 4)
        self.assertEqual(len(result.history["lr"]), 4)
        # Best state must be loadable.
        model.load_state_dict(result.best_state)
        self.assertFalse(
            any(value.is_cuda for value in result.best_state.values())
        )

    def test_gradient_accumulation_and_cpu_amp_fallback(self) -> None:
        torch.manual_seed(2)
        inputs = torch.randn(20, 4)
        targets = (inputs[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
        model = nn.Linear(4, 2)
        result = train_with_recipe(
            model,
            train_loader=loader,
            inner_val_loader=loader,
            num_classes=2,
            recipe=RecipeConfig(
                epochs=2,
                gradient_accumulation_steps=4,
                amp=True,
                logit_adjust_tau=0.0,
            ),
            device="cpu",
            verbose=False,
        )
        self.assertEqual(result.history["recipe"]["gradient_accumulation_steps"], 4)
        self.assertTrue(result.history["recipe"]["amp"])


if __name__ == "__main__":
    unittest.main()
