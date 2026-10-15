"""Tests for acquisition grouping and reversible original-image geometry."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from tools.sonar_fcos_pilot.data import SonarDataset, prepare_manifest, predictions_to_original


def make_sample(root, source, name, *, size=(101, 53), boxes=None, color=0, group=None, label="ball"):
    folder = root / source / source
    images, annotations = folder / "images", folder / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (color, 0, 0)).save(images / f"{name}.bmp")
    width, height = size
    boxes = [[1, 2, 11, 12]] if boxes is None else boxes
    objects = "".join(
        f"<object><name>{label}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for x1, y1, x2, y2 in boxes
    )
    group_xml = f"<sequence_id>{group}</sequence_id>" if group else ""
    xml = f"<annotation><file><filename>{name}</filename><folder>{source}</folder></file><size><width>{width}</width><height>{height}</height></size>{group_xml}{objects}</annotation>"
    (annotations / f"{name}.xml").write_text(xml, encoding="utf-8")
    return images / f"{name}.bmp"


def fixture_dataset(root, *, explicit_groups=False):
    for i in range(8):
        make_sample(root, "UATD_Training", f"{i:05}", color=i+10, group=f"capture-{i//2}" if explicit_groups else None)
    make_sample(root, "UATD_Test_1", "00000", color=100)
    make_sample(root, "UATD_Test_2", "00000", color=101)


def group_map():
    return {
        "evidence": {"kind": "source_log", "reference": "fixture-log.csv", "explanation": "Synthetic test acquisition log"},
        "groups": {f"official_train/images/{i:05}.bmp": f"capture-{i//2}" for i in range(8)},
    }


def test_numeric_names_do_not_establish_sequences(tmp_path):
    fixture_dataset(tmp_path / "raw")
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit")
    assert result["split_status"] == "BLOCKED_GROUP_METADATA"
    assert result["small_area_threshold"] is None
    assert result["split_counts"] == {"unassigned": 8, "test": 2}
    assert result["audit"]["split_grouping"]["missing_group_count"] == 8
    assert (tmp_path / "audit" / "group_map_template.json").exists()
    with pytest.raises(RuntimeError, match="blocked"):
        SonarDataset(tmp_path / "audit" / "manifest.json", "train")


def test_source_groups_are_disjoint_and_reproducible(tmp_path):
    fixture_dataset(tmp_path / "raw")
    first = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    second = prepare_manifest(tmp_path / "raw", tmp_path / "again", group_map=group_map())
    assert first["split_status"] == "PASS"
    assert [r["split"] for r in first["samples"]] == [r["split"] for r in second["samples"]]
    groups = {split: {r["group_id"] for r in first["samples"] if r["split"] == split} for split in ("train", "val")}
    assert groups["train"] and groups["val"] and groups["train"].isdisjoint(groups["val"])
    assert first["small_area_threshold"] == pytest.approx(100 / (101*53))
    assert all(r["split"] == "test" for r in first["samples"] if r["source_split"].startswith("official_test"))


def test_explicit_xml_sequence_fields_are_accepted(tmp_path):
    fixture_dataset(tmp_path / "raw", explicit_groups=True)
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit")
    assert result["split_status"] == "PASS"
    assert all(r["group_id"].startswith("sequence_id:") for r in result["samples"] if r["split"] == "train")


def test_invalid_boxes_block_without_silent_repair(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Test_1", "00000", boxes=[[20, 4, 10, 4]], color=100)
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    assert result["split_status"] == "BLOCKED_DATA_INTEGRITY"
    assert result["audit"]["annotation_error_count"] == 1
    test_sample = next(r for r in result["samples"] if r["source_split"] == "official_test_1")
    assert test_sample["boxes"] == [[20, 4, 10, 4]]
    assert result["small_area_threshold"] is None


def test_duplicate_of_test_is_excluded_from_development_only(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Training", "00000", color=100)
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    assert result["split_status"] == "PASS"
    assert result["audit"]["excluded_official_train_duplicates_of_test"] == ["official_train/images/00000.bmp"]
    assert result["split_counts"]["test"] == 2
    assert result["split_counts"]["excluded"] == 1
    assert not result["audit"]["cross_split_duplicates"]


def test_rounded_resize_and_padding_roundtrip_original_boxes(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    dataset = SonarDataset(tmp_path / "audit" / "manifest.json", "val", image_size=64)
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target["original_size"].tolist() == [53, 101]
    assert target["resize_scale"][0] != target["resize_scale"][1]
    prediction = {"boxes": target["boxes"], "labels": target["labels"], "scores": torch.ones(len(target["boxes"]))}
    restored = predictions_to_original(target, prediction)
    assert torch.allclose(torch.tensor(restored["boxes"]), target["original_boxes"], atol=1e-5)
    assert target["small_object_mask"].all()
    assert restored["image_id"] == int(target["image_id"].item())


def test_group_mapping_requires_citable_source(tmp_path):
    fixture_dataset(tmp_path / "raw")
    with pytest.raises(ValueError, match="requires evidence"):
        prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map={"groups": {}})


def test_small_threshold_does_not_use_test_targets(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Test_1", "00000", boxes=[[0, 0, 101, 53]], color=100)
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    assert result["small_area_threshold"] == pytest.approx(100 / (101*53))


def test_unknown_class_and_missing_official_partition_block(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Training", "00000", color=10, label="invented_class")
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    assert result["split_status"] == "BLOCKED_DATA_INTEGRITY"
    assert any(e["kind"] == "unknown_class" for e in result["audit"]["annotation_errors"])


def test_duplicate_training_images_cannot_cross_train_val(tmp_path):
    fixture_dataset(tmp_path / "raw")
    make_sample(tmp_path / "raw", "UATD_Training", "00002", color=10)
    result = prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    duplicates = [r for r in result["samples"] if r["sample_key"] in {"official_train/images/00000.bmp", "official_train/images/00002.bmp"}]
    assert len({r["split"] for r in duplicates}) == 1
    assert result["audit"]["split_grouping"]["duplicate_group_links"]


def test_prediction_export_removes_reserved_background_label(tmp_path):
    fixture_dataset(tmp_path / "raw")
    prepare_manifest(tmp_path / "raw", tmp_path / "audit", group_map=group_map())
    dataset = SonarDataset(tmp_path / "audit" / "manifest.json", "val", image_size=64)
    _, target = dataset[0]
    prediction = {"boxes": target["boxes"].repeat(2, 1), "labels": torch.tensor([0, 1]), "scores": torch.tensor([.99, .8])}
    result = predictions_to_original(target, prediction)
    assert result["labels"] == [1]
    assert len(result["boxes"]) == 1
