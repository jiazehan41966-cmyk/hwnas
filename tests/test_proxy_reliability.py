from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.analysis.proxy_collection import collect_observations
from hwnas_fpga.analysis.proxy_execution import (
    naswot_score,
    train_prefix_trajectory,
)
from hwnas_fpga.analysis.proxy_manifest import (
    build_prefix_work_units,
    select_stratified_architectures,
)
from hwnas_fpga.analysis.proxy_reliability import (
    ClassificationObservation,
    analyze_hardware_reliability,
    analyze_proxy_reliability,
    balanced_random_effects_variance,
    balanced_two_way_variance,
    load_hardware_observations,
    write_proxy_reliability_bundle,
)


def _truth_rows(
    *,
    architecture_scale: float,
    noise_scale: float,
) -> list[ClassificationObservation]:
    rng = np.random.default_rng(7)
    rows: list[ClassificationObservation] = []
    for architecture in range(8):
        for seed in range(3):
            for fold in range(3):
                value = (
                    0.50
                    + architecture_scale * architecture
                    + 0.003 * seed
                    - 0.002 * fold
                    + noise_scale * float(rng.normal())
                )
                rows.append(
                    ClassificationObservation(
                        architecture_id=f"a{architecture}",
                        proxy_name="trained",
                        budget=150,
                        seed=seed,
                        outer_fold=fold,
                        metric="macro_f1",
                        proxy_value=value,
                        truth_value=value,
                    )
                )
    return rows


def test_variance_decomposition_distinguishes_architecture_signal() -> None:
    strong = balanced_random_effects_variance(
        _truth_rows(architecture_scale=0.04, noise_scale=0.002)
    )
    weak = balanced_random_effects_variance(
        _truth_rows(architecture_scale=0.0001, noise_scale=0.04)
    )
    assert strong["estimable"]
    assert strong["icc_relative_mean_observed"] > 0.95
    assert strong["icc_relative_mean_seeds_single_fold"] > 0.95
    assert (
        weak["icc_relative_mean_observed"]
        < strong["icc_relative_mean_observed"]
    )
    assert "architecture_outer_fold" in strong["variance_components"]


def test_two_way_stage_a_variance_estimates_architecture_seed_signal() -> None:
    rows = [
        row
        for row in _truth_rows(
            architecture_scale=0.04,
            noise_scale=0.002,
        )
        if row.outer_fold == 0
    ]
    result = balanced_two_way_variance(
        rows,
        replicate_axis="seed",
        bootstrap_iterations=5,
    )
    assert result["estimable"]
    assert result["design"] == "architecture_x_seed"
    assert result["icc_relative_mean_observed"] > 0.95

    analysis = analyze_proxy_reliability(
        rows,
        bootstrap_iterations=5,
        gate_config={
            "metric": "macro_f1",
            "min_architectures": 8,
            "min_seeds": 3,
            "min_outer_folds": 5,
            "required_budgets": [150],
        },
    )
    assert analysis["variance_decomposition"]["macro_f1"]["estimable"]
    assert analysis["gate"]["status"] == "not_ready"


def _multifidelity_rows() -> list[ClassificationObservation]:
    rows: list[ClassificationObservation] = []
    for architecture in range(8):
        target = 0.45 + 0.04 * architecture
        for seed in range(3):
            for fold in range(3):
                truth = target + 0.001 * seed - 0.0015 * fold
                for budget in (1, 10, 150):
                    if budget == 1:
                        proxy = 1.0 - target + 0.0005 * seed
                    else:
                        proxy = target + 0.0005 * seed - 0.0005 * fold
                    rows.append(
                        ClassificationObservation(
                            architecture_id=f"a{architecture}",
                            proxy_name="trained",
                            budget=budget,
                            seed=seed,
                            outer_fold=fold,
                            metric="macro_f1",
                            proxy_value=proxy,
                            truth_value=truth if budget == 150 else None,
                        )
                    )
    return rows


def test_multifidelity_ranking_and_gate() -> None:
    analysis = analyze_proxy_reliability(
        _multifidelity_rows(),
        bootstrap_iterations=20,
        gate_config={
            "metric": "macro_f1",
            "min_architectures": 8,
            "min_seeds": 3,
            "min_outer_folds": 3,
            "min_icc_relative_mean": 0.60,
            "min_kendall_tau_b": 0.30,
            "min_pairwise_accuracy": 0.65,
            "max_regret_at5": 0.02,
            "required_budgets": [1, 10, 150],
        },
    )
    by_budget = {
        row["budget"]: row
        for row in analysis["multi_fidelity"]
        if row["metric"] == "macro_f1"
    }
    assert by_budget[1]["kendall_tau_b"] < 0
    assert by_budget[10]["kendall_tau_b"] > 0.99
    assert by_budget[10]["top5_recall"] == 1.0
    assert by_budget[10]["regret_at5"] == pytest.approx(0.0)
    assert analysis["gate"]["status"] == "pass"
    assert analysis["gate"]["earliest_usable_proxy"]["budget"] == 10


def test_incomplete_truth_grid_is_not_ready() -> None:
    rows = _multifidelity_rows()
    rows = [
        row
        for row in rows
        if not (
            row.budget == 150
            and row.architecture_id == "a0"
            and row.seed == 0
            and row.outer_fold == 0
        )
    ]
    analysis = analyze_proxy_reliability(
        rows,
        bootstrap_iterations=5,
        gate_config={
            "metric": "macro_f1",
            "min_architectures": 8,
            "min_seeds": 3,
            "min_outer_folds": 3,
            "required_budgets": [1, 10, 150],
        },
    )
    assert analysis["gate"]["status"] == "not_ready"
    assert not analysis["variance_decomposition"]["macro_f1"]["estimable"]


def test_analysis_bundle_reports_intervals_and_top_regions(
    tmp_path: Path,
) -> None:
    analysis = analyze_proxy_reliability(
        _multifidelity_rows(),
        bootstrap_iterations=5,
        gate_config={
            "metric": "macro_f1",
            "min_architectures": 8,
            "min_seeds": 3,
            "min_outer_folds": 3,
            "required_budgets": [1, 10, 150],
        },
    )
    outputs = write_proxy_reliability_bundle(
        tmp_path,
        classification=analysis,
        classification_source="synthetic.csv",
    )
    report = Path(outputs["analysis_report"]).read_text(encoding="utf-8")
    assert "Kendall tau-b (95% CI)" in report
    assert "Top-10 recall" in report
    assert Path(outputs["audit_summary"]).exists()


def test_hardware_identity_has_perfect_reliability() -> None:
    rows = []
    for index, feasible in enumerate((True, True, False, False)):
        rows.append(
            {
                "architecture_id": f"a{index}",
                "proxy_feasible": feasible,
                "truth_feasible": feasible,
                "truth_latency_source": "COM5 board timing",
                "truth_resource_source": "post-route utilization",
                "truth_feasibility_source": "route status",
                "proxy_latency_ms": 2.0 + index,
                "truth_latency_ms": 2.0 + index,
                "proxy_dsp": 10 + index,
                "truth_dsp": 10 + index,
                "proxy_bram": 20 + index,
                "truth_bram": 20 + index,
                "proxy_lut": 100 + index,
                "truth_lut": 100 + index,
            }
        )
    result = analyze_hardware_reliability(rows, bootstrap_iterations=10)
    assert result["metrics"]["latency_ms"]["mae"] == pytest.approx(0.0)
    assert result["feasibility"]["balanced_accuracy"] == pytest.approx(1.0)
    assert result["pareto"]["precision"] == pytest.approx(1.0)
    assert result["pareto"]["recall"] == pytest.approx(1.0)


def test_hardware_csv_requires_explicit_truth_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hardware.csv"
    source.write_text(
        "architecture_id,proxy_feasible,truth_feasible,"
        "truth_latency_source,truth_resource_source,"
        "truth_feasibility_source\n"
        "a0,true,true,,post-route,route status\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty truth source"):
        load_hardware_observations(source)


def test_naswot_score_is_finite() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    result = naswot_score(model, torch.randn(4, 1, 8, 8), device="cpu")
    assert math.isfinite(result["score"])
    assert result["activation_module_count"] == 1


def _fake_candidates() -> list[dict]:
    rows = []
    buckets = (
        ["feasible_interior"] * 4
        + ["near_boundary"] * 4
        + ["infeasible"] * 4
    )
    for index, bucket in enumerate(buckets):
        rows.append(
            {
                "architecture_id": f"a{index:02d}",
                "descriptors": {
                    "operator_counts": {
                        "mbconv": 1 + index % 3,
                        "denoise": index % 2,
                    },
                    "total_depth": 4 + index % 4,
                    "mean_channels": 12 + index,
                    "max_channels": 24 + index,
                    "max_resource_utilization": 0.2 + index * 0.08,
                    "feasibility_class": bucket,
                },
                "hardware_proxy": {
                    "latency_ms": 2 + index,
                    "dsp": 10 + index,
                    "bram": 20 + index,
                    "lut": 1000 + 10 * index,
                },
            }
        )
    return rows


def test_stratified_selection_is_deterministic() -> None:
    first, first_summary = select_stratified_architectures(
        _fake_candidates(), target_count=6, seed=12
    )
    second, second_summary = select_stratified_architectures(
        _fake_candidates(), target_count=6, seed=12
    )
    assert [row["architecture_id"] for row in first] == [
        row["architecture_id"] for row in second
    ]
    assert first_summary == second_summary
    assert first_summary["actual_feasibility_counts"] == {
        "feasible_interior": 3,
        "near_boundary": 2,
        "infeasible": 1,
    }


def test_prefix_work_units_partition_stages_without_budget_duplication() -> None:
    candidates = [{"architecture_id": "a0"}, {"architecture_id": "a1"}]
    units = build_prefix_work_units(
        candidates,
        stages=[
            {"name": "phase_a", "outer_folds": [0], "seeds": [42, 43]},
            {"name": "phase_b", "outer_folds": [1], "seeds": [42]},
            {"name": "phase_c", "outer_folds": [1], "seeds": [43]},
        ],
        budgets=[0, 1, 3, 150],
        truth_budget=150,
        proxy_name="trained_prefix",
    )
    assert len(units) == 8
    assert len({unit["work_id"] for unit in units}) == 8
    assert all(unit["budgets"] == [0, 1, 3, 150] for unit in units)
    assert {unit["stage"] for unit in units} == {
        "phase_a",
        "phase_b",
        "phase_c",
    }
    with pytest.raises(ValueError, match="more than one stage"):
        build_prefix_work_units(
            candidates,
            stages=[
                {"name": "left", "outer_folds": [0], "seeds": [42]},
                {"name": "right", "outer_folds": [0], "seeds": [42]},
            ],
            budgets=[0, 150],
            truth_budget=150,
            proxy_name="trained_prefix",
        )


def test_prefix_training_emits_registered_milestones() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(12, 1, 4, 4)
    targets = torch.tensor([0, 1] * 6)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    trained, milestones, history = train_prefix_trajectory(
        model,
        train_loader=loader,
        inner_val_loader=loader,
        num_classes=2,
        positive_budgets=[1, 2],
        optimizer_name="adamw",
        lr=0.001,
        weight_decay=0.0001,
        class_weights=torch.ones(2),
        selection_metric="macro_f1",
        device="cpu",
        verbose=False,
    )
    assert isinstance(trained, nn.Module)
    assert set(milestones) == {1, 2}
    assert milestones[1]["best_epoch"] <= 1
    assert milestones[2]["best_epoch"] <= 2
    assert len(history["train_loss"]) == 2


def test_prefix_collection_expands_one_trajectory_into_budgets(
    tmp_path: Path,
) -> None:
    manifest = {
        "protocol": {
            "execution": {
                "metrics": ["macro_f1", "top1"],
                "scheduler_policy": (
                    "prefix_consistent_single_trajectory_constant_lr"
                ),
            }
        }
    }
    manifest_path = tmp_path / "architecture_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    unit = {
        "work_id": "prefix0",
        "architecture_id": "a0",
        "proxy_name": "trained_prefix",
        "zero_cost_proxy_name": "naswot_v1",
        "budgets": [0, 1, 150],
        "seed": 42,
        "outer_fold": 0,
        "truth_budget": 150,
        "stage": "phase_a",
        "stage_order": 0,
        "evaluation_scope": "inner_milestones_outer_once",
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": "test",
        "candidate_artifact": "candidate.json",
        "work_type": "prefix_train",
    }
    matrix = tmp_path / "run_matrix.jsonl"
    matrix.write_text(json.dumps(unit) + "\n", encoding="utf-8")
    observations = tmp_path / "observations"
    observations.mkdir()
    record = {
        "work_id": "prefix0",
        "architecture_id": "a0",
        "seed": 42,
        "outer_fold": 0,
        "truth_budget": 150,
        "stage": "phase_a",
        "budgets": [0, 1, 150],
        "status": "completed",
        "formal_eligible": True,
        "work_fingerprint": "fp",
        "work_unit": unit,
        "recipe_id": "recipe",
        "recipe": {"epochs": 150, "scheduler_policy": "prefix"},
        "milestones": {
            "0": {
                "proxy_name": "naswot_v1",
                "proxy_values": {"macro_f1": 10.0, "top1": 10.0},
            },
            "1": {
                "proxy_name": "trained_prefix",
                "proxy_values": {"macro_f1": 0.3, "top1": 0.5},
            },
            "150": {
                "proxy_name": "trained_prefix",
                "proxy_values": {"macro_f1": 0.7, "top1": 0.8},
            },
        },
        "truth_values": {"macro_f1": 0.68, "top1": 0.79},
        "outer_evaluation_performed": True,
    }
    (observations / "prefix0.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    rows, summary = collect_observations(
        run_matrix=matrix,
        observations_dir=observations,
    )
    assert len(rows) == 6
    assert summary["completed_work_units"] == 1
    assert summary["completed_by_budget"] == {"0": 1, "1": 1, "150": 1}
    assert summary["ready_for_formal_analysis"]
    assert all(
        row["truth_value"] == ""
        for row in rows
        if row["budget"] != 150
    )


def test_collection_rejects_short_budget_outer_evaluation(
    tmp_path: Path,
) -> None:
    manifest = {
        "protocol": {"execution": {"metrics": ["macro_f1", "top1"]}}
    }
    manifest_path = tmp_path / "architecture_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observations = tmp_path / "observations"
    observations.mkdir()
    units = []
    for budget in (1, 150):
        unit = {
            "work_id": f"w{budget}",
            "architecture_id": "a0",
            "proxy_name": "trained",
            "budget": budget,
            "seed": 42,
            "outer_fold": 0,
            "truth_budget": 150,
            "evaluation_scope": (
                "inner_select_outer_once" if budget == 150 else "inner_only"
            ),
            "manifest_path": str(manifest_path),
            "manifest_fingerprint": "test",
            "candidate_artifact": "candidate.json",
            "work_type": "train",
        }
        units.append(unit)
        record = {
            **{key: unit[key] for key in (
                "work_id",
                "architecture_id",
                "proxy_name",
                "budget",
                "seed",
                "outer_fold",
                "truth_budget",
            )},
            "status": "completed",
            "work_fingerprint": f"fp{budget}",
            "work_unit": unit,
            "proxy_values": {"macro_f1": 0.5, "top1": 0.6},
            "truth_values": (
                {"macro_f1": 0.55, "top1": 0.65} if budget == 150 else {}
            ),
            "outer_evaluation_performed": budget == 150,
            "proxy_direction": "max",
            "recipe_id": f"recipe{budget}",
            "recipe": {"epochs": budget, "lr": 0.001},
        }
        (observations / f"w{budget}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    matrix = tmp_path / "run_matrix.jsonl"
    matrix.write_text(
        "".join(json.dumps(unit) + "\n" for unit in units),
        encoding="utf-8",
    )

    rows, summary = collect_observations(
        run_matrix=matrix,
        observations_dir=observations,
    )
    assert len(rows) == 4
    assert summary["ready_for_formal_analysis"]

    short_path = observations / "w1.json"
    short = json.loads(short_path.read_text(encoding="utf-8"))
    short["outer_evaluation_performed"] = True
    short_path.write_text(json.dumps(short), encoding="utf-8")
    _rows, invalid_summary = collect_observations(
        run_matrix=matrix,
        observations_dir=observations,
    )
    assert not invalid_summary["ready_for_formal_analysis"]
    assert invalid_summary["invalid_work_units"] == 1
