"""Small synthetic ZIP tests; no research archives, image decoding or ML deps."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import warnings
import zipfile
import zlib

import pytest

from scripts import restore_datasets as restore


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(root, *, duplicate=False, member="source/picture.bmp"):
    root.mkdir()
    image = b"synthetic image bytes: no decoder needed\n" * 3
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member, image)
            if duplicate:
                archive.writestr(member, image)
            archive.writestr("toolkit/unselected.py", b"not extracted")
    payload = buffer.getvalue()
    # Cuts through the ZIP header, compressed data and central-directory tail.
    offsets = (0, 7, 43, len(payload) - 13, len(payload))
    parts = []
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        data = payload[start:end]
        relative = f"datasets/Cylinder2/raw/source.zip.part{index:03d}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        parts.append(dict(path=relative, size=len(data), sha256=_hash(data)))
    archive = dict(id="original", parts=parts, size=len(payload), sha256=_hash(payload))
    views = []
    for name in ("processed", "second_view"):
        view = f"datasets/Cylinder2/{name}"
        metadata = []
        for relative, data in (("data.yaml", b"names: [fixture]\n"), ("train/labels/image.txt", b"")):
            path = root / view / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            metadata.append(dict(path=relative, size=len(data), sha256=_hash(data)))
        views.append(dict(path=view, files=metadata, images=[dict(
            path="train/images/image.bmp", archive_id="original", member=member, size=len(image),
            crc32=f"{zlib.crc32(image):08x}", sha256=_hash(image))]))
    manifest = dict(schema_version=1, datasets=dict(Cylinder2=dict(archives=[archive], views=views)))
    manifest_path = root / "datasets/dataset_sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, image, payload, manifest_path


def test_seekable_parts_and_whole_hash_without_concatenating_on_disk(tmp_path):
    root = tmp_path / "repo"
    manifest, image, payload, path = _fixture(root)
    assert restore.read_manifest(path) == manifest
    archive = manifest["datasets"]["Cylinder2"]["archives"][0]
    receipt = restore.verify_archive(archive, repo_root=root)
    assert receipt["sha256"] == _hash(payload)
    paths = [root / p["path"] for p in archive["parts"]]
    with restore.MultipartReader(paths, [p["size"] for p in archive["parts"]]) as reader:
        assert reader.read(19) == payload[:19]
        reader.seek(-17, io.SEEK_END)
        assert reader.read() == payload[-17:]
        reader.seek(3)
        reader.seek(6, io.SEEK_CUR)
        assert reader.tell() == 9 and reader.read(42) == payload[9:51]
        with zipfile.ZipFile(reader) as zipped:
            assert zipped.read("source/picture.bmp") == image
    assert not list(root.rglob("source.zip"))


def test_restore_only_mapped_images_metadata_hardlinks_and_idempotence(tmp_path):
    root = tmp_path / "repo"
    manifest, image, _, _ = _fixture(root)
    output = tmp_path / "restored"
    report = restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=output)
    assert report["images_restored"] == 1
    assert report["images_linked"] == 1
    assert report["image_bytes_written"] == len(image)
    assert report["metadata_copied"] == 4
    targets = [output / v["path"] / "train/images/image.bmp" for v in manifest["datasets"]["Cylinder2"]["views"]]
    assert all(p.read_bytes() == image for p in targets)
    assert os.path.samefile(*targets)
    assert not list(output.rglob("unselected.py"))
    again = restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=output)
    assert again["images_reused"] == 2 and again["metadata_reused"] == 4
    assert again["images_restored"] == 0 and again["images_linked"] == 0


def test_restore_falls_back_to_stream_copy_if_hardlinks_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    manifest, image, _, _ = _fixture(root)
    original_link = restore.os.link
    def no_shared_image_link(source, target):
        if Path(source).name == "image.bmp":
            raise OSError("fixture filesystem does not support shared hardlinks")
        return original_link(source, target)
    monkeypatch.setattr(restore.os, "link", no_shared_image_link)
    report = restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=tmp_path / "out")
    assert report["images_restored"] == 2 and report["images_linked"] == 0
    assert report["image_bytes_written"] == 2 * len(image)


def test_verify_only_never_creates_output_or_reads_metadata(tmp_path):
    root = tmp_path / "repo"
    manifest, _, _, _ = _fixture(root)
    (root / manifest["datasets"]["Cylinder2"]["views"][0]["path"] / "data.yaml").unlink()
    output = tmp_path / "must_not_exist"
    result = restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=output, verify_only=True)
    assert result["status"] == "VERIFIED_ARCHIVES"
    assert not output.exists()


def test_archive_part_drift_whole_digest_and_git_lfs_pointer(tmp_path):
    root = tmp_path / "repo"
    manifest, _, _, _ = _fixture(root)
    archive = manifest["datasets"]["Cylinder2"]["archives"][0]
    wrong_whole = copy.deepcopy(archive)
    wrong_whole["sha256"] = "0" * 64
    with pytest.raises(restore.DatasetRestoreError, match="Whole archive SHA256 drift"):
        restore.verify_archive(wrong_whole, repo_root=root)
    part = root / archive["parts"][1]["path"]
    original = part.read_bytes()
    part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(restore.DatasetRestoreError, match="part SHA256 drift"):
        restore.verify_archive(archive, repo_root=root)
    part.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 99\n")
    with pytest.raises(restore.DatasetRestoreError, match="git lfs pull"):
        restore.verify_archive(archive, repo_root=root)


@pytest.mark.parametrize("field,bad", [("path", "../outside.bmp"), ("path", "/absolute.bmp"),
                                     ("path", "C:/outside.bmp"), ("path", "train\\image.bmp"),
                                     ("path", "train/images/NUL.bmp"), ("member", "../../outside.bmp")])
def test_manifest_path_attacks_rejected_before_zip_reads(tmp_path, field, bad):
    root = tmp_path / "repo"
    manifest, _, _, path = _fixture(root)
    manifest["datasets"]["Cylinder2"]["views"][0]["images"][0][field] = bad
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(restore.DatasetRestoreError, match="Unsafe relative path"):
        restore.read_manifest(path)


def test_duplicate_zip_and_duplicate_destinations_rejected(tmp_path):
    root = tmp_path / "repo"
    manifest, _, _, _ = _fixture(root, duplicate=True)
    with pytest.raises(restore.DatasetRestoreError, match="Duplicate ZIP entry"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=tmp_path / "out")
    duplicated = copy.deepcopy(manifest)
    images = duplicated["datasets"]["Cylinder2"]["views"][0]["images"]
    images.append(copy.deepcopy(images[0]))
    with pytest.raises(restore.DatasetRestoreError, match="Duplicate output entry"):
        restore.restore_dataset("Cylinder2", duplicated, repo_root=root)


def test_wrong_member_digest_leaves_no_image_or_temporary_file(tmp_path):
    root = tmp_path / "repo"
    manifest, _, _, _ = _fixture(root)
    manifest["datasets"]["Cylinder2"]["views"][0]["images"][0]["sha256"] = "0" * 64
    output = tmp_path / "out"
    with pytest.raises(restore.DatasetRestoreError, match="Restored member size/SHA256/CRC32 differs"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=output)
    assert not list(output.rglob("image.bmp"))
    assert not list(output.rglob(".restore-*"))


def test_existing_conflicts_and_metadata_drift_never_overwritten(tmp_path):
    root = tmp_path / "repo"
    manifest, image, _, _ = _fixture(root)
    output = tmp_path / "out"
    view = manifest["datasets"]["Cylinder2"]["views"][0]
    target = output / view["path"] / "train/images/image.bmp"
    target.parent.mkdir(parents=True)
    conflicting = b"x" * len(image)
    target.write_bytes(conflicting)
    with pytest.raises(restore.DatasetRestoreError, match="SHA256 differs"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=output)
    assert target.read_bytes() == conflicting
    metadata = root / view["path"] / "data.yaml"
    metadata.write_bytes(b"changed")
    with pytest.raises(restore.DatasetRestoreError, match="size or type differs"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=tmp_path / "out2")


def test_unselected_unsafe_zip_entry_rejected_without_extracting_it(tmp_path):
    root = tmp_path / "repo"
    manifest, _, _, _ = _fixture(root, member="../outside.bmp")
    # A forged map to a safe name cannot make the unsafe central-directory tree trusted.
    for view in manifest["datasets"]["Cylinder2"]["views"]:
        view["images"][0]["member"] = "safe.bmp"
    with pytest.raises(restore.DatasetRestoreError, match="Unsafe relative path"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root, output_root=tmp_path / "out")
    assert not (tmp_path / "outside.bmp").exists()


def test_single_part_archive_and_crc_only_reuse(tmp_path):
    root = tmp_path / "repo"
    manifest, image, payload, _ = _fixture(root)
    archive = manifest["datasets"]["Cylinder2"]["archives"][0]
    path = root / "datasets/Cylinder2/raw/single.zip"
    path.write_bytes(payload)
    archive.update(path="datasets/Cylinder2/raw/single.zip", parts=[dict(
        path="datasets/Cylinder2/raw/single.zip", size=len(payload), sha256=_hash(payload))])
    for view in manifest["datasets"]["Cylinder2"]["views"]:
        view["images"][0].pop("sha256")
    restored = restore.restore_dataset("Cylinder2", manifest, repo_root=root)
    assert restored["images_restored"] == 1
    assert restored["metadata_reused"] == 4
    image_path = root / manifest["datasets"]["Cylinder2"]["views"][0]["path"] / "train/images/image.bmp"
    image_path.write_bytes(b"!" * len(image))
    with pytest.raises(restore.DatasetRestoreError, match="CRC32 differs"):
        restore.restore_dataset("Cylinder2", manifest, repo_root=root)


def _directory_link(link, target):
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True)
        if result.returncode:
            pytest.skip("Temporary Windows directory junction creation unavailable")
    else:
        link.symlink_to(target, target_is_directory=True)


def test_top_level_dataset_storage_junction_is_trusted_and_nested_links_are_rejected(tmp_path):
    backing = tmp_path / "backing"
    manifest, image, _, _ = _fixture(backing)
    root = tmp_path / "repo"
    root.mkdir()
    _directory_link(root / "datasets", backing / "datasets")
    # Default destination follows the one authorized storage alias.
    report = restore.restore_dataset("Cylinder2", manifest, repo_root=root)
    assert report["images_restored"] == 1 and report["images_linked"] == 1
    view = manifest["datasets"]["Cylinder2"]["views"][0]
    assert (root / view["path"] / "train/images/image.bmp").read_bytes() == image
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = backing / "datasets/Cylinder2/escape"
    _directory_link(nested, outside)
    attacked = copy.deepcopy(manifest)
    attacked["datasets"]["Cylinder2"]["views"][0]["path"] = "datasets/Cylinder2/escape/processed"
    with pytest.raises(restore.DatasetRestoreError, match="Nested symbolic links or directory junctions"):
        restore.restore_dataset("Cylinder2", attacked, repo_root=root)
    assert not list(outside.iterdir())
    _directory_link(root / "other_alias", outside)
    with pytest.raises(restore.DatasetRestoreError, match="Nested symbolic links or directory junctions"):
        restore._inside(root, "other_alias/payload.bmp")
