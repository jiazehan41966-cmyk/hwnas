"""Validation-only selection and finite campaign budget, using fake results."""
from pathlib import Path

import pytest

from tools.sonar_fcos_pilot import campaign, common


COMBINATIONS = ["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"]
SEEDS = [42, 43, 44]


def summary(combo, validation_map=40.0, seed=42, epochs=30, status="COMPLETE"):
    return {"combo": combo, "seed": seed, "trained_epochs": epochs, "status": status,
            "validation": {"map": validation_map},
            "test_result": {"map": 100.0 if combo == "BBB" else 0.0}}


def decision_config():
    return {"seeds": SEEDS, "minimum_map_gain": 1.0, "minimum_winning_seeds": 2}


def test_architecture_selection_uses_validation_and_ignores_test_values():
    summaries = [summary(combo, validation_map=45.0 if combo == "AAB" else 40.0)
                 for combo in COMBINATIONS]
    assert campaign.select_combination(summaries, COMBINATIONS) == "AAB"


def test_validation_ties_choose_prespecified_first_combo():
    summaries = [summary(combo) for combo in reversed(COMBINATIONS)]
    assert campaign.select_combination(summaries, COMBINATIONS) == "AAA"


def test_architecture_selection_rejects_missing_candidate():
    with pytest.raises(ValueError, match="Every candidate"):
        campaign.select_combination([summary(combo) for combo in COMBINATIONS[:-1]], COMBINATIONS)


def test_architecture_selection_rejects_duplicate_combo():
    summaries = [summary(combo) for combo in COMBINATIONS] + [summary("AAA", validation_map=99)]
    with pytest.raises(ValueError, match="one first-seed"):
        campaign.select_combination(summaries, COMBINATIONS)


def test_architecture_selection_requires_first_seed_42():
    summaries = [summary(combo) for combo in COMBINATIONS]
    summaries[2]["seed"] = 43
    with pytest.raises(ValueError, match="first-seed"):
        campaign.select_combination(summaries, COMBINATIONS)


@pytest.mark.parametrize("invalid_map", [None, float("nan"), float("inf")])
def test_architecture_selection_rejects_undefined_or_nonfinite_ap(invalid_map):
    summaries = [summary(combo) for combo in COMBINATIONS]
    summaries[2]["validation"]["map"] = invalid_map
    with pytest.raises(ValueError, match="finite"):
        campaign.select_combination(summaries, COMBINATIONS)


@pytest.mark.parametrize("epochs,status", [(29, "COMPLETE"), (30, "TRAINING")])
def test_architecture_selection_rejects_partial_training(epochs, status):
    summaries = [summary(combo) for combo in COMBINATIONS]
    summaries[2] = summary(COMBINATIONS[2], epochs=epochs, status=status)
    with pytest.raises(ValueError, match="Partial training"):
        campaign.select_combination(summaries, COMBINATIONS)


@pytest.mark.parametrize("selected,expected_total", [("AAB", 14), ("AAA", 12), ("BBB", 12)])
def test_shortlist_deduplicates_controls_and_stays_within_fourteen(selected, expected_total):
    first_round = {(combo, 42) for combo in COMBINATIONS}
    shortlist = campaign.comparison_runs(selected, SEEDS)
    assert len(shortlist) == len(set(shortlist))
    assert len(first_round | set(shortlist)) == expected_total
    assert len(first_round | set(shortlist)) <= 14


def results(selected="AAB", maps=(41.0, 41.0, 41.0), small_recall=0.8, fp=0.2):
    rows = []
    for combo in dict.fromkeys(["AAA", "BBB", selected]):
        for index, seed in enumerate(SEEDS):
            is_selected = combo == selected
            rows.append({"combo": combo, "seed": seed,
                         "metrics": {"map": maps[index] if is_selected else 40.0,
                                     "diagnostics": {"small_recall": small_recall if is_selected else 0.5,
                                                     "false_positives_per_image": fp if is_selected
                                                     else (0.1 if combo == "AAA" else 0.2)}}})
    return rows


def test_one_percentage_point_and_small_recall_gates_accept_supported_mix():
    result = campaign.decide("AAB", results(), decision_config())
    assert result["outcome"] == "INITIAL_SUPPORT_FOR_MIXED"
    assert all(result["checks"].values())
    assert result["means"]["AAB"]["map"] == 41.0
    assert result["map_unit"] == "percentage_points_0_to_100"


def test_point_nine_percentage_points_is_below_one_point_gate():
    result = campaign.decide("AAB", results(maps=(40.9, 40.9, 40.9)), decision_config())
    assert result["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert not result["checks"]["mean_map_gain_vs_both_at_least_one_point"]


def test_mean_gain_cannot_replace_two_paired_seed_wins():
    result = campaign.decide("AAB", results(maps=(44.0, 39.5, 39.5)), decision_config())
    assert result["checks"]["mean_map_gain_vs_both_at_least_one_point"]
    assert result["paired_seed_wins"] == 1
    assert result["outcome"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("error", ["missing_seed", "duplicate_seed", "extra_run"])
def test_final_decision_rejects_incomplete_duplicate_or_unshortlisted_results(error):
    rows = results()
    if error == "missing_seed":
        rows.pop()
    elif error == "duplicate_seed":
        rows.append(rows[-1])
    else:
        rows.append({**rows[-1], "combo": "BAA"})
    with pytest.raises(ValueError, match="deduplicated shortlist"):
        campaign.decide("AAB", rows, decision_config())


@pytest.mark.parametrize("invalid_map", [None, float("nan"), float("inf")])
def test_final_decision_rejects_undefined_or_nonfinite_ap(invalid_map):
    rows = results()
    rows[0]["metrics"]["map"] = invalid_map
    with pytest.raises(ValueError, match="finite"):
        campaign.decide("AAB", rows, decision_config())


def test_exactly_two_seed_wins_can_pass_with_one_point_average_gain():
    result = campaign.decide("AAB", results(maps=(42.0, 42.0, 39.0)), decision_config())
    assert result["paired_seed_wins"] == 2
    assert result["outcome"] == "INITIAL_SUPPORT_FOR_MIXED"


@pytest.mark.parametrize("small_recall,fp,failed_check", [
    (0.5, 0.2, "small_recall_improved_vs_both"),
    (None, 0.2, "small_recall_improved_vs_both"),
    (0.8, 0.21, "false_positives_no_higher_than_larger_control_mean"),
])
def test_accuracy_alone_cannot_pass_missing_small_gain_or_excess_false_positives(small_recall, fp, failed_check):
    result = campaign.decide("AAB", results(small_recall=small_recall, fp=fp), decision_config())
    assert result["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert not result["checks"][failed_check]


def test_mock_campaign_selects_before_testing_and_launches_no_duplicate_runs(monkeypatch, tmp_path):
    config = {**decision_config(), "output_dir": str(tmp_path), "max_formal_runs": 14,
              "combinations": COMBINATIONS}
    manifest_path = tmp_path / "synthetic_manifest.json"
    common.write_json(manifest_path, {"fixture": "synthetic_only"})
    monkeypatch.setattr(campaign, "require_manifest", lambda config: (manifest_path, {}))
    monkeypatch.setattr(campaign, "source_binding", lambda config: {"fixture": "synthetic_only"})
    monkeypatch.setattr(campaign, "baseline_gate", lambda metrics: {"status": "GO", "reason": "synthetic fixture"})
    training_calls, test_calls = [], []

    def fake_train(config, combo, seed):
        training_calls.append((combo, seed))
        row = summary(combo, validation_map=45.0 if combo == "AAB" else 40.0, seed=seed)
        common.write_json(campaign.run_directory(config, combo, seed) / "summary.json", row)
        return row

    def fake_test(config, combo, seed, authorization):
        assert len(training_calls) == 14  # All repeats completed before test access.
        selection = common.read_json(tmp_path / "selection.json")
        assert selection["combo"] == "AAB"
        assert selection["selection_set"] == "validation_only"
        assert campaign.run_id(combo, seed) in authorization["runs"]
        test_calls.append((combo, seed))
        return next(row for row in results() if row["combo"] == combo and row["seed"] == seed)

    monkeypatch.setattr(campaign, "train_one", fake_train)
    monkeypatch.setattr(campaign, "evaluate_test", fake_test)
    assert campaign.execute_campaign(config) == 0
    assert len(training_calls) == len(set(training_calls)) == 14
    assert training_calls[:8] == [(combo, 42) for combo in COMBINATIONS]
    assert len(test_calls) == 9
    comparison = common.read_json(tmp_path / "comparison.json")
    assert comparison["outcome"] == "INITIAL_SUPPORT_FOR_MIXED"
    assert not (tmp_path / "campaign.lock").exists()
