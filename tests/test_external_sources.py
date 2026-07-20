import json
import zipfile
from pathlib import Path

import pytest

from hwnas_fpga.data.external_sources import (
    ExternalDatasetError,
    audit_yolo_detection_tree,
    extract_zip_idempotent,
    fetch_roboflow_cylider2,
    find_download_url,
    hash_file,
    inventory_directory,
    resolve_latest_roboflow_version,
    safe_extract_zip,
    yolo_row_from_pixel_box,
)


def test_safe_extract_zip_and_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/images/sample.png", b"fake-png")
        bundle.writestr("train/labels/sample.txt", "0 0.5 0.5 1 1\n")

    output = tmp_path / "extracted"
    assert safe_extract_zip(archive, output) == 2
    inventory = inventory_directory(output)
    assert inventory["file_count"] == 2
    assert inventory["image_count"] == 1
    assert inventory["annotation_file_count"] == 1


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "no")

    with pytest.raises(ExternalDatasetError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_idempotent_extraction_records_archive_hash(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data/sample.jpg", b"image")

    destination = tmp_path / "dataset"
    first = extract_zip_idempotent(archive, destination)
    second = extract_zip_idempotent(archive, destination)
    assert first == second
    marker = json.loads(
        (destination / ".hwnas_extract_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["archive_sha256"] == hash_file(archive)


def test_yolo_audit_preserves_empty_annotations(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"image-a")
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"image-b")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")

    audit = audit_yolo_detection_tree(
        tmp_path, class_names={0: "MILCO", 1: "NOMBO"}
    )
    assert audit["paired_image_label_count"] == 2
    assert audit["empty_annotation_file_count"] == 1
    assert audit["images_with_boxes"] == 1
    assert audit["class_box_counts"] == {"MILCO": 1, "NOMBO": 0}
    assert audit["malformed_row_count"] == 0


def test_yolo_audit_matches_images_and_labels_directories(tmp_path: Path) -> None:
    image_path = tmp_path / "train" / "images" / "sample.bmp"
    label_path = tmp_path / "train" / "labels" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    label_path.write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    audit = audit_yolo_detection_tree(
        tmp_path, class_names={0: "cylinder", 1: "cylider", 2: "manta"}
    )
    assert audit["paired_image_label_count"] == 1
    assert audit["orphan_label_count"] == 0
    assert audit["missing_label_count"] == 0
    assert audit["class_box_counts"]["manta"] == 1


def test_find_download_url_prefers_export_link() -> None:
    payload = {
        "project": {"url": "https://api.roboflow.com/project"},
        "export": {"link": "https://signed.example/dataset.zip"},
    }
    assert find_download_url(payload) == "https://signed.example/dataset.zip"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"project": {"versions": 6}}, 6),
        ({"versions": [{"version": 1}, {"version": 4}, {"version": 2}]}, 4),
    ],
)
def test_resolve_latest_roboflow_version(payload: dict, expected: int) -> None:
    assert resolve_latest_roboflow_version(payload) == expected


def test_import_manual_roboflow_zip(tmp_path: Path) -> None:
    archive = tmp_path / "manual.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/images/sample.jpg", b"image")
        bundle.writestr("train/labels/sample.txt", "0 0.5 0.5 0.2 0.2\n")

    payload = fetch_roboflow_cylider2(
        tmp_path / "external",
        local_zip=archive,
        export_format="yolov8",
        version=6,
    )
    assert payload["source"]["version"] == 6
    assert payload["acquisition"] == "local_export_zip"
    assert payload["inventory"]["image_count"] == 1
    assert Path(payload["manifest_path"]).is_file()


def test_yolo_row_from_pixel_box() -> None:
    row = yolo_row_from_pixel_box(
        {
            "label": "cylinder",
            "x": "50",
            "y": "25",
            "width": "20",
            "height": "10",
        },
        image_width=100,
        image_height=50,
        label_to_index={"cylinder": 0},
    )
    assert row == "0 0.5000000000 0.5000000000 0.2000000000 0.2000000000"


def test_yolo_row_rejects_out_of_bounds_box() -> None:
    with pytest.raises(ExternalDatasetError, match="exceeds image bounds"):
        yolo_row_from_pixel_box(
            {"label": "cylinder", "x": 2, "y": 2, "width": 10, "height": 10},
            image_width=100,
            image_height=100,
            label_to_index={"cylinder": 0},
        )
