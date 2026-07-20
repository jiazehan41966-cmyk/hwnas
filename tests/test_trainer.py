import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig
from hwnas_fpga.training import trainer as trainer_module
from hwnas_fpga.training.trainer import train_model


class TrainModelBestStateTests(unittest.TestCase):
    @staticmethod
    def _loader() -> DataLoader:
        inputs = torch.tensor([[1.0], [-1.0]])
        targets = torch.tensor([0, 1])
        return DataLoader(TensorDataset(inputs, targets), batch_size=2)

    @staticmethod
    def _summary(*, top1: float, macro_f1: float) -> dict[str, object]:
        return {
            "loss": 0.0,
            "top1": top1,
            "top5": top1,
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1,
            "num_samples": 2.0,
        }

    def test_restores_epoch_selected_by_configured_metric(self) -> None:
        loader = self._loader()

        for selection_metric, expected_state in (("top1", 1.0), ("macro_f1", 2.0)):
            model = nn.Linear(1, 2, bias=False)
            states = iter((1.0, 2.0))
            summaries = iter(
                (
                    self._summary(top1=0.9, macro_f1=0.4),
                    self._summary(top1=0.7, macro_f1=0.8),
                )
            )

            def fake_train_epoch(
                trainer: trainer_module.Trainer, _train_loader: DataLoader
            ) -> tuple[float, float]:
                state = next(states)
                with torch.no_grad():
                    trainer.model.weight.fill_(state)
                return 0.0, 0.0

            with (
                patch.object(trainer_module.Trainer, "train_epoch", new=fake_train_epoch),
                patch.object(
                    trainer_module,
                    "evaluate_classifier",
                    side_effect=lambda *args, **kwargs: next(summaries),
                ),
            ):
                best_score, history = train_model(
                    model,
                    train_loader=loader,
                    val_loader=loader,
                    num_classes=2,
                    epochs=2,
                    device="cpu",
                    selection_metric=selection_metric,
                    topk=2,
                )

            self.assertAlmostEqual(
                best_score,
                0.9 if selection_metric == "top1" else 0.8,
            )
            self.assertEqual(history["best_epoch"], 1 if selection_metric == "top1" else 2)
            self.assertTrue(history["best_state_restored"])
            self.assertTrue(torch.all(model.weight == expected_state))

    def test_restores_best_state_without_validation_loader(self) -> None:
        loader = self._loader()
        model = nn.Linear(1, 2, bias=False)
        states = iter((1.0, 2.0, 3.0))
        accuracies = iter((0.2, 0.8, 0.4))

        def fake_train_epoch(
            trainer: trainer_module.Trainer, _train_loader: DataLoader
        ) -> tuple[float, float]:
            state = next(states)
            with torch.no_grad():
                trainer.model.weight.fill_(state)
            return 0.0, next(accuracies)

        with patch.object(trainer_module.Trainer, "train_epoch", new=fake_train_epoch):
            best_score, history = train_model(
                model,
                train_loader=loader,
                val_loader=None,
                num_classes=2,
                epochs=3,
                device="cpu",
            )

        self.assertAlmostEqual(best_score, 0.8)
        self.assertEqual(history["best_epoch"], 2)
        self.assertTrue(history["best_state_restored"])
        self.assertTrue(torch.all(model.weight == 2.0))

    def test_saved_checkpoint_evaluates_at_best_metric(self) -> None:
        loader = self._loader()
        model = nn.Linear(1, 2, bias=False)
        epoch = {"value": 0}

        def fake_train_epoch(
            trainer: trainer_module.Trainer, _train_loader: DataLoader
        ) -> tuple[float, float]:
            weights = (
                torch.tensor([[1.0], [-1.0]]),
                torch.tensor([[-1.0], [1.0]]),
            )
            with torch.no_grad():
                trainer.model.weight.copy_(weights[epoch["value"]])
            epoch["value"] += 1
            return 0.0, 1.0 if epoch["value"] == 1 else 0.0

        with patch.object(trainer_module.Trainer, "train_epoch", new=fake_train_epoch):
            best_score, history = train_model(
                model,
                train_loader=loader,
                val_loader=loader,
                num_classes=2,
                epochs=2,
                device="cpu",
                selection_metric="macro_f1",
                topk=2,
            )

        self.assertAlmostEqual(best_score, history["best_eval"]["macro_f1"])

        search_space = SearchSpace(SearchSpaceConfig())
        candidate = SearchCandidate(
            arch_id="arch_0",
            encoding=search_space.baseline_architecture().to_dict(),
            metrics=CandidateMetrics(
                accuracy=best_score,
                top1=history["best_eval"]["top1"],
                top5=history["best_eval"]["top5"],
                macro_f1=history["best_eval"]["macro_f1"],
                weighted_f1=history["best_eval"]["weighted_f1"],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                output_root=Path(tmpdir),
                project_name="trainer-best-state",
                script_name="test_trainer",
                search_method="test",
                dataset_name="dummy",
                run_name="best-state",
            )
            tracker.save_best_candidate(
                candidate,
                model_state_dict=model.state_dict(),
                history=history,
            )
            payload = torch.load(
                tracker.checkpoints_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )

        loaded_model = nn.Linear(1, 2, bias=False)
        loaded_model.load_state_dict(payload["model_state_dict"])
        summary = trainer_module.evaluate_classifier(
            loaded_model,
            loader,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
            num_classes=2,
            topk=2,
        )
        self.assertAlmostEqual(summary["macro_f1"], history["best_eval"]["macro_f1"])
        self.assertAlmostEqual(summary["top1"], history["best_eval"]["top1"])


if __name__ == "__main__":
    unittest.main()
