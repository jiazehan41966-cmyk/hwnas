import tempfile
import unittest
from pathlib import Path

import torch

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import CandidateMetrics, HardwareSpec, SearchCandidate
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


class ExperimentTrackerTests(unittest.TestCase):
    def test_tracker_writes_results_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                output_root=Path(tmpdir),
                project_name="unit-test",
                script_name="run_search",
                search_method="random",
                dataset_name="dummy",
                run_name="tracker-test",
            )
            tracker.save_config({"project": {"name": "unit-test"}}, {"seed": 42})

            space = SearchSpace(SearchSpaceConfig())
            estimator = FPGACostEstimator(
                hardware_spec=HardwareSpec(
                    name="test-fpga",
                    clock_mhz=200,
                    max_lut=120_000,
                    max_bram=2_000,
                    max_dsp=512,
                )
            )
            baseline = space.baseline_architecture()
            baseline_cost = estimator.estimate(baseline, space)
            tracker.write_baseline(
                architecture=baseline.to_dict(),
                cost_estimate=baseline_cost,
            )

            candidate = SearchCandidate(
                arch_id="arch_0",
                encoding=baseline.to_dict(),
                metrics=CandidateMetrics(
                    accuracy=0.75,
                    latency_ms=baseline_cost.latency_ms,
                    energy_mj=baseline_cost.energy_mj,
                    lut=baseline_cost.peak_lut,
                    bram=baseline_cost.peak_bram,
                    dsp=baseline_cost.peak_dsp,
                    power_w=baseline_cost.power_w,
                ),
            )

            tracker.record_candidate(
                candidate,
                feasible=True,
                cost_estimate=baseline_cost,
                history={"train_acc": [0.7, 0.75], "val_acc": [0.68, 0.75]},
                extra={"iteration": 1},
            )
            tracker.update_search_state(
                total_evaluated=1,
                feasible=1,
                infeasible=0,
                best_candidate=candidate,
                extra={"iteration": 1},
            )
            tracker.save_best_candidate(
                candidate,
                model_state_dict=torch.nn.Linear(4, 2).state_dict(),
                history={"val_acc": [0.75]},
                extra={"selection_metric": "accuracy"},
            )
            tracker.finalize(
                status="completed",
                candidates=[candidate],
                feasible_candidates=[candidate],
                infeasible_candidates=[],
                best_candidate=candidate,
                pareto_candidates=[candidate],
                extra={"search_summary": {"total_evaluated": 1}},
            )

            self.assertTrue((tracker.root_dir / "config.yaml").exists())
            self.assertTrue((tracker.logs_dir / "console.log").exists() or tracker.logs_dir.exists())
            self.assertTrue((tracker.results_dir / "baseline.json").exists())
            self.assertTrue((tracker.results_dir / "candidates.jsonl").exists())
            self.assertTrue((tracker.results_dir / "candidates.json").exists())
            self.assertTrue((tracker.results_dir / "candidates.csv").exists())
            self.assertTrue((tracker.results_dir / "summary.json").exists())
            self.assertTrue((tracker.checkpoints_dir / "search_state.json").exists())
            self.assertTrue((tracker.checkpoints_dir / "best_candidate.json").exists())
            self.assertTrue((tracker.checkpoints_dir / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
