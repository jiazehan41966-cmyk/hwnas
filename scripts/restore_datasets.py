"""Restore manifest-listed dataset bytes from verified ZIP parts (standard library).

No image decoding, dataset statistics, training, or model selection occurs here.
Training views share hard links where the destination filesystem supports them.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "datasets/dataset_sources.json"
DATASETS = ("Cylinder2", "SCTD", "UATD")
CHUNK = 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CRC = re.compile(r"[0-9a-fA-F]{8}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10)), *(f"COM{i}" for i in "¹²³"),
             *(f"LPT{i}" for i in "¹²³")}


class DatasetRestoreError(ValueError):
    """An archive or destination failed its byte/provenance contract."""


def _relative(name, *, directory=False):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise DatasetRestoreError("Unsafe relative path: " + repr(name))
    value = name[:-1] if directory and name.endswith("/") else name
    parts = value.split("/")
    if any(not p or p in {".", ".."} or p[-1:] in {" ", "."}
           or any(ord(c) < 32 or c in '<>"|?*' for c in p)
           or p.split(".")[0].upper() in _RESERVED for p in parts):
        raise DatasetRestoreError("Unsafe relative path: " + repr(name))
    return Path(*parts)


def _inside(root, relative):
    root = Path(root).resolve()
    local = _relative(relative)
    parts = local.parts
    path = root / local
    # The repository's sole top-level datasets storage alias may deliberately
    # be a directory junction onto another drive. Trust its resolved root, not
    # arbitrary links below it or any other top-level alias.
    if parts[0] == "datasets":
        alias = root / "datasets"
        if (alias.exists() or alias.is_symlink() or getattr(alias, "is_junction", lambda: False)()) and not alias.is_dir():
            raise DatasetRestoreError("The top-level datasets storage alias must be a directory")
        trusted_root = alias.resolve()
        suffix = parts[1:]
    else:
        trusted_root, suffix = root, parts
    current = trusted_root
    for part in suffix:
        current /= part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise DatasetRestoreError("Nested symbolic links or directory junctions are forbidden: " + relative)
    if not path.resolve().is_relative_to(trusted_root):
        raise DatasetRestoreError("Path resolves outside its root: " + relative)
    return path


def _size(value):
    if type(value) is not int or value < 0:
        raise DatasetRestoreError("Expected a nonnegative integer byte size")
    return value


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise DatasetRestoreError("Expected a lowercase SHA256 digest")
    return value


def _validate_archive(archive):
    if not isinstance(archive.get("id"), str) or not archive["id"]:
        raise DatasetRestoreError("Archive id must be a nonempty string")
    parts = archive.get("parts")
    if not isinstance(parts, list) or not parts:
        raise DatasetRestoreError("Archives require a nonempty parts list")
    paths = set()
    for part in parts:
        path = _relative(part["path"]).as_posix()
        if path.casefold() in paths:
            raise DatasetRestoreError("Duplicate archive part path")
        paths.add(path.casefold())
        _size(part["size"])
        _sha(part["sha256"])
    if sum(part["size"] for part in parts) != _size(archive["size"]):
        raise DatasetRestoreError("Archive size differs from its ordered parts")
    _sha(archive["sha256"])
    if archive.get("path") is not None:
        _relative(archive["path"])


def _validate_manifest(manifest):
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise DatasetRestoreError("Unsupported dataset source manifest schema")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets or set(datasets) - set(DATASETS):
        raise DatasetRestoreError("Manifest datasets must be Cylinder2, SCTD or UATD")
    destinations = set()
    for name, dataset in datasets.items():
        archives = dataset.get("archives", [])
        ids = set()
        for archive in archives:
            _validate_archive(archive)
            if archive["id"] in ids:
                raise DatasetRestoreError("Duplicate archive id in " + name)
            ids.add(archive["id"])
        for view in dataset.get("views", []):
            view_path = _relative(view["path"]).as_posix()
            for item in (*view.get("images", []), *view.get("files", [])):
                relative = _relative(item["path"]).as_posix()
                destination = (view_path + "/" + relative).casefold()
                if destination in destinations:
                    raise DatasetRestoreError("Duplicate output entry: " + destination)
                destinations.add(destination)
                _size(item["size"])
            for image in view.get("images", []):
                _relative(image["member"])
                if image["archive_id"] not in ids:
                    raise DatasetRestoreError("Image refers to an unknown archive id")
                if not isinstance(image["crc32"], str) or not _CRC.fullmatch(image["crc32"]):
                    raise DatasetRestoreError("Image CRC32 must contain eight hex digits")
                if image.get("sha256") is not None:
                    _sha(image["sha256"])
            for item in view.get("files", []):
                _sha(item["sha256"])
    return manifest


def read_manifest(path=DEFAULT_MANIFEST):
    """Read and validate all mapping paths before opening any ZIP member."""
    return _validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


class MultipartReader(io.RawIOBase):
    """A seekable concatenation that opens only one physical part at a time."""
    def __init__(self, paths, sizes):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.sizes = list(sizes)
        self.ends = []
        total = 0
        for size in self.sizes:
            total += size
            self.ends.append(total)
        self.length, self.position, self.part_index, self.stream = total, 0, None, None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.length
        elif whence != io.SEEK_SET:
            raise ValueError("Unsupported seek origin")
        if offset < 0:
            raise ValueError("Negative ZIP seek position")
        self.position = offset
        return offset

    def read(self, size=-1):
        self._checkClosed()
        remaining = max(0, self.length - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size:
            index = bisect.bisect_right(self.ends, self.position)
            start = self.ends[index - 1] if index else 0
            if index != self.part_index:
                if self.stream is not None:
                    self.stream.close()
                self.stream = self.paths[index].open("rb")
                self.part_index = index
            self.stream.seek(self.position - start)
            block = self.stream.read(min(size, self.ends[index] - self.position))
            if not block:
                raise DatasetRestoreError("Archive part became truncated while reading ZIP")
            chunks.append(block)
            self.position += len(block)
            size -= len(block)
        return b"".join(chunks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def verify_archive(archive, *, repo_root=ROOT):
    """Stream all parts once, verifying each digest and the complete ZIP digest."""
    _validate_archive(archive)
    whole = hashlib.sha256()
    for part in archive["parts"]:
        path = _inside(repo_root, part["path"])
        if not path.is_file():
            raise DatasetRestoreError("Missing archive part: " + str(path))
        with path.open("rb") as stream:
            prefix = stream.read(128)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise DatasetRestoreError("Archive is a Git LFS pointer, not ZIP bytes: " + str(path)
                                          + "; run git lfs pull first")
            if path.stat().st_size != part["size"]:
                raise DatasetRestoreError("Archive part size drift: " + str(path))
            digest, count = hashlib.sha256(), 0
            block = prefix
            while block:
                digest.update(block)
                whole.update(block)
                count += len(block)
                block = stream.read(CHUNK)
        if count != part["size"] or digest.hexdigest() != part["sha256"]:
            raise DatasetRestoreError("Archive part SHA256 drift: " + str(path))
    if whole.hexdigest() != archive["sha256"]:
        raise DatasetRestoreError("Whole archive SHA256 drift: " + archive["id"])
    return dict(id=archive["id"], parts=len(archive["parts"]), size=archive["size"],
                sha256=archive["sha256"], status="VERIFIED")


def _check_file(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
        raise DatasetRestoreError("Existing destination/source size or type differs: " + str(path))
    digest, crc, count = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            count += len(chunk)
    if count != item["size"]:
        raise DatasetRestoreError("Existing destination/source changed size while reading: " + str(path))
    if item.get("sha256") is not None and digest.hexdigest() != item["sha256"]:
        raise DatasetRestoreError("Existing destination/source SHA256 differs: " + str(path))
    if "crc32" in item and f"{crc:08x}" != item["crc32"].lower():
        raise DatasetRestoreError("Existing destination/source CRC32 differs: " + str(path))


def _publish(temporary, target, item):
    """Atomically publish without ever replacing an existing destination."""
    try:
        if os.name == "nt":
            os.rename(temporary, target)  # Windows rename refuses an existing file.
        else:
            os.link(temporary, target)  # POSIX rename would silently overwrite.
            temporary.unlink()
    except FileExistsError:
        _check_file(target, item)
        return False
    return True


def _write_stream(stream, target, item):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore-", delete=False) as output:
            temporary = Path(output.name)
            digest, crc, count = hashlib.sha256(), 0, 0
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                count += len(chunk)
                if count > item["size"]:
                    raise DatasetRestoreError("Restored member exceeds its declared size")
                output.write(chunk)
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
            if (count != item["size"] or (item.get("sha256") is not None and digest.hexdigest() != item["sha256"])
                    or ("crc32" in item and f"{crc:08x}" != item["crc32"].lower())):
                raise DatasetRestoreError("Restored member size/SHA256/CRC32 differs: " + str(target))
            output.flush()
            os.fsync(output.fileno())
        return _publish(temporary, target, item)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _zip_index(archive):
    result = {}
    folded = set()
    for info in archive.infolist():
        name = _relative(info.filename, directory=info.is_dir()).as_posix()
        if name.casefold() in folded:
            raise DatasetRestoreError("Duplicate ZIP entry: " + name)
        folded.add(name.casefold())
        result[info.filename] = info
    return result


def restore_dataset(dataset_name, manifest, *, repo_root=ROOT, output_root=None, verify_only=False):
    """Verify a dataset's ZIPs and restore only its listed images and metadata."""
    _validate_manifest(manifest)
    if dataset_name not in manifest["datasets"]:
        raise DatasetRestoreError("Dataset is absent from the manifest: " + dataset_name)
    repo_root = Path(repo_root).resolve()
    output_root = repo_root if output_root is None else Path(output_root).resolve()
    dataset = manifest["datasets"][dataset_name]
    receipts = [verify_archive(a, repo_root=repo_root) for a in dataset["archives"]]
    report = dict(dataset=dataset_name, status="VERIFIED_ARCHIVES" if verify_only else "RESTORED",
                  archives=receipts, images_restored=0, images_reused=0, images_linked=0,
                  metadata_copied=0, metadata_reused=0, image_bytes_written=0,
                  output_root=str(output_root), model_operations=False)
    if verify_only:
        return report
    grouped = {a["id"]: [] for a in dataset["archives"]}
    for view in dataset.get("views", []):
        for item in view.get("files", []):
            relative = view["path"] + "/" + item["path"]
            source, target = _inside(repo_root, relative), _inside(output_root, relative)
            _check_file(source, item)
            if target.exists() or target.is_symlink():
                _check_file(target, item)
                report["metadata_reused"] += 1
            else:
                with source.open("rb") as stream:
                    copied = _write_stream(stream, target, item)
                report["metadata_copied" if copied else "metadata_reused"] += 1
        for image in view.get("images", []):
            grouped[image["archive_id"]].append((_inside(output_root, view["path"] + "/" + image["path"]), image))
    for specification in dataset["archives"]:
        paths = [_inside(repo_root, p["path"]) for p in specification["parts"]]
        sizes = [p["size"] for p in specification["parts"]]
        with MultipartReader(paths, sizes) as reader, zipfile.ZipFile(reader) as archive:
            index, restored = _zip_index(archive), {}
            for target, image in grouped[specification["id"]]:
                shared_key = (image["member"], image.get("sha256"))
                info = index.get(image["member"])
                if (info is None or info.is_dir() or info.file_size != image["size"]
                        or f"{info.CRC:08x}" != image["crc32"].lower()
                        or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK):
                    raise DatasetRestoreError("ZIP member missing or metadata differs: " + image["member"])
                if target.exists() or target.is_symlink():
                    _check_file(target, image)
                    report["images_reused"] += 1
                elif shared_key in restored:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(restored[shared_key], target)
                    except FileExistsError:
                        _check_file(target, image)
                        report["images_reused"] += 1
                    except OSError:
                        with archive.open(info) as stream:
                            written = _write_stream(stream, target, image)
                        report["images_restored" if written else "images_reused"] += 1
                        report["image_bytes_written"] += image["size"] if written else 0
                    else:
                        report["images_linked"] += 1
                else:
                    with archive.open(info) as stream:
                        written = _write_stream(stream, target, image)
                    report["images_restored" if written else "images_reused"] += 1
                    report["image_bytes_written"] += image["size"] if written else 0
                restored[shared_key] = target
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true", help="Verify archive parts and whole hashes; write no files")
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        names = list(manifest["datasets"]) if args.dataset == "all" else [args.dataset]
        reports = [restore_dataset(name, manifest, output_root=args.output_root, verify_only=args.verify_only)
                   for name in names]
    except (DatasetRestoreError, OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        parser.exit(2, "Dataset restore failed: " + str(error) + "\n")
    print(json.dumps(dict(datasets=reports, verify_only=args.verify_only), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
