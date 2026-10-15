"""Tests for proxy provenance boundaries and similarity graph mechanics."""

import json
import random

import pytest

from tools.sonar_fcos_pilot.data import SonarDataset, prepare_manifest
from tools.sonar_fcos_pilot.similarity import _BKTree, build_similarity_report
from test_sonar_fcos_pilot_data import fixture_dataset


def test_bk_tree_candidates_match_bruteforce_hamming():
    randomizer = random.Random(42)
    values = [randomizer.getrandbits(64) for _ in range(100)]
    values += [values[0], values[0] ^ 3, values[0] ^ 15]
    tree = _BKTree()
    for i, value in enumerate(values):
        tree.add(value, i)
    for query in values[:10]:
        expected = {i for i, value in enumerate(values) if (value ^ query).bit_count() <= 4}
        assert set(tree.query(query, 4)) == expected


def test_proxy_report_is_not_a_verified_sequence_map(tmp_path):
    fixture_dataset(tmp_path / "raw")
    strict = prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    report = build_similarity_report(tmp_path / "strict" / "manifest.json", tmp_path / "similarity", include_test_audit=True)
    assert report["provenance_status"] == "IMAGE_SIMILARITY_PROXY_ONLY"
    assert report["test_used_for_grouping_or_threshold_selection"] is False
    assert report["formal_training_authorized_under_alternative"] is False
    assert report["development_count"] == 8
    assert report["test_similarity_audit"]["test_image_count"] == 2
    assert report["thresholds"] == {"dhash_hamming_max": 4, "normalized_grayscale_rmse_max": .04}
    assert report["largest_group_size"] == 8
    assert report["split_preview"]["group_assignment_status"] == "BLOCKED_GROUP_METADATA"
    current = json.loads((tmp_path / "strict" / "manifest.json").read_text(encoding="utf-8"))
    assert current == strict
    with pytest.raises(ValueError, match="not verified acquisition"):
        prepare_manifest(tmp_path / "raw", tmp_path / "must_block", group_map=tmp_path / "similarity" / "proxy_group_map.json")
    with pytest.raises(RuntimeError, match="blocked"):
        SonarDataset(tmp_path / "similarity" / "split_preview_not_authorized.json", "train")


def test_test_risk_audit_does_not_change_development_groups(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "strict")
    path = tmp_path / "strict" / "manifest.json"
    without = build_similarity_report(path, tmp_path / "without", include_test_audit=False)
    with_audit = build_similarity_report(path, tmp_path / "with", include_test_audit=True)
    assert without["groups"] == with_audit["groups"]
    assert without["split_preview"] == with_audit["split_preview"]
