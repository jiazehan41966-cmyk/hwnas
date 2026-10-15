"""Exploratory drafts must preserve inputs and stay blocked until choice."""

import json
from pathlib import Path

import pytest

from tools.sonar_fcos_pilot.data import SonarDataset, prepare_manifest
from tools.sonar_fcos_pilot.exploratory import prepare_exploratory_draft
from test_sonar_fcos_pilot_data import fixture_dataset, make_sample


def test_draft_preserves_strict_input_and_is_not_training_approved(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    source = tmp_path / "strict" / "manifest.json"
    original_bytes = source.read_bytes()
    draft = prepare_exploratory_draft(source, tmp_path / "draft")
    assert source.read_bytes() == original_bytes
    assert draft["split_status"] == "PENDING_PROTOCOL_APPROVAL"
    assert draft["protocol"]["name"] == "PUBLIC_SPLIT_EXPLORATORY_V1"
    assert draft["protocol"]["status"] == "PENDING_USER_CHOICE"
    assert draft["protocol"]["acquisition_groups_verified"] is False
    assert draft["protocol"]["training_authorized"] is False
    assert draft["protocol"]["no_new_scene_generalization_claim"] is True
    assert draft["split_counts"] == {"test": 2, "train": 6, "val": 2}
    with pytest.raises(RuntimeError, match="blocked"):
        SonarDataset(tmp_path / "draft" / "manifest.json", "train")


def test_entire_invalid_images_are_excluded_not_repaired(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Training", "00000", color=10, boxes=[[0, 0, 0, 0], [1, 2, 11, 12]])
    make_sample(tmp_path / "raw", "UATD_Test_1", "00000", color=100, boxes=[[2, 3, 1, 3]])
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    draft = prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "draft")
    excluded = [sample for sample in draft["samples"] if sample["split"] == "excluded"]
    assert len(excluded) == 2
    original_train = next(sample for sample in excluded if sample["source_split"] == "official_train")
    assert original_train["boxes"] == [[0, 0, 0, 0], [1, 2, 11, 12]]
    assert original_train["excluded_reason"] == ["INVALID_ORIGINAL_ANNOTATION_WHOLE_IMAGE"]
    assert draft["split_counts"]["test"] == 1
    assert draft["audit"]["data_integrity_pass"] is True
    assert draft["audit"]["original_data_integrity_pass"] is False
    assert draft["audit"]["original_annotation_error_count"] == 2
    assert draft["audit"]["annotation_error_count"] == 0
    assert len(draft["audit"]["excluded_official_test_images_not_scored"]) == 1


def test_exact_duplicates_stay_together_and_test_copy_stays_excluded(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Training", "00002", color=10)
    make_sample(tmp_path / "raw", "UATD_Training", "00007", color=100)
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    draft = prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "draft")
    duplicate = [sample for sample in draft["samples"] if sample["sample_key"] in {"official_train/images/00000.bmp", "official_train/images/00002.bmp"}]
    assert len({sample["split"] for sample in duplicate}) == 1
    excluded = next(sample for sample in draft["samples"] if sample["sample_key"] == "official_train/images/00007.bmp")
    assert excluded["excluded_reason"] == ["EXACT_DUPLICATE_OF_OFFICIAL_TEST"]
    assert not draft["audit"]["active_cross_split_exact_hashes"]


def test_test_targets_do_not_change_development_assignment_or_small_boundary(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    first = prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "first")
    make_sample(tmp_path / "raw", "UATD_Test_1", "00000", color=100, boxes=[[0, 0, 101, 53]], label="tyre")
    prepare_manifest(tmp_path / "raw", tmp_path / "second_strict")
    second = prepare_exploratory_draft(tmp_path / "second_strict" / "manifest.json", tmp_path / "second")
    first_dev = [(sample["sample_key"], sample["split"]) for sample in first["samples"] if sample["source_split"] == "official_train"]
    second_dev = [(sample["sample_key"], sample["split"]) for sample in second["samples"] if sample["source_split"] == "official_train"]
    assert first_dev == second_dev
    assert first["small_area_threshold"] == second["small_area_threshold"]


def test_source_fingerprint_and_repeatability(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    first = prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "first")
    second = prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "second")
    assert first == second
    exclusions = json.loads((tmp_path / "first" / "excluded_images.json").read_text(encoding="utf-8"))
    assert exclusions["source_manifest_sha256"] == first["source_manifest_sha256"]


def test_draft_rejects_another_alternative_as_source(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "draft")
    with pytest.raises(ValueError, match="strict source"):
        prepare_exploratory_draft(tmp_path / "draft" / "manifest.json", tmp_path / "nested")


def test_draft_cannot_overwrite_strict_manifest(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    with pytest.raises(ValueError, match="cannot overwrite"):
        prepare_exploratory_draft(tmp_path / "strict" / "manifest.json", tmp_path / "strict")
