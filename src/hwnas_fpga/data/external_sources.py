"""Reproducible acquisition helpers for third-party sonar datasets.

Downloaded archives and extracted files live under ``data/external`` and are
deliberately kept outside Git.  Every acquisition writes a local manifest with
source metadata, checksums, and an inventory of extracted files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional

from PIL import Image


FIGSHARE_ARTICLE_ID = 24574879
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
FIGSHARE_EXPECTED_VERSION = 2
FIGSHARE_EXPECTED_LICENSE = "CC BY 4.0"

ROBOFLOW_REQUESTED_URL = (
    "https://universe.roboflow.com/yeesonmin-naver-com/cylider2"
)
ROBOFLOW_WORKSPACE = "yeesonmin-naver-com"
ROBOFLOW_PROJECT = "cylider2"
ROBOFLOW_RECONSTRUCTION_CLASSES = ("cylinder", "cylider", "manta")

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIXES = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}


class ExternalDatasetError(RuntimeError):
    """Raised when an external dataset cannot be acquired or verified."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = 60,
) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "hwnas-fpga-dataset-fetcher/1.0",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ExternalDatasetError(f"Failed to read metadata from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExternalDatasetError(f"Expected an object response from {url}")
    return payload


def request_json_post(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = 60,
) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "hwnas-fpga-dataset-fetcher/1.0",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ExternalDatasetError(f"Failed to read metadata from {url}: {exc}") from exc
    if not isinstance(response_payload, dict):
        raise ExternalDatasetError(f"Expected an object response from {url}")
    return response_payload


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: Optional[int] = None,
    expected_md5: Optional[str] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Download a file atomically, with resume and checksum verification."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size_ok = expected_size is None or destination.stat().st_size == expected_size
        md5_ok = expected_md5 is None or hash_file(destination, "md5") == expected_md5
        if size_ok and md5_ok:
            return {
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "md5": hash_file(destination, "md5"),
                "sha256": hash_file(destination),
                "downloaded": False,
            }
        raise ExternalDatasetError(
            f"Existing file failed verification; move it aside before retrying: {destination}"
        )

    partial = destination.with_name(destination.name + ".part")
    resume_at = partial.stat().st_size if partial.exists() else 0
    request_headers = {"User-Agent": "hwnas-fpga-dataset-fetcher/1.0"}
    request_headers.update(headers or {})
    if resume_at:
        request_headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(url, headers=request_headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            append = resume_at > 0 and getattr(response, "status", None) == 206
            mode = "ab" if append else "wb"
            if not append:
                resume_at = 0
            with partial.open(mode) as handle:
                copied = resume_at
                next_report = ((copied // (64 * 1024 * 1024)) + 1) * 64 * 1024 * 1024
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    copied += len(chunk)
                    if copied >= next_report:
                        print(f"[download] {destination.name}: {copied / (1024**2):.1f} MiB")
                        next_report += 64 * 1024 * 1024
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExternalDatasetError(
            f"Download interrupted for {destination.name}; partial file retained: {exc}"
        ) from exc

    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ExternalDatasetError(
            f"Size mismatch for {destination.name}: {actual_size} != {expected_size}"
        )
    actual_md5 = hash_file(partial, "md5")
    if expected_md5 is not None and actual_md5 != expected_md5:
        raise ExternalDatasetError(
            f"MD5 mismatch for {destination.name}: {actual_md5} != {expected_md5}"
        )
    partial.replace(destination)
    return {
        "path": str(destination.resolve()),
        "bytes": actual_size,
        "md5": actual_md5,
        "sha256": hash_file(destination),
        "downloaded": True,
    }


def _validated_zip_member(member: zipfile.ZipInfo) -> Optional[PurePosixPath]:
    normalized = member.filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ExternalDatasetError(f"Unsafe ZIP member path: {member.filename!r}")
    unix_mode = (member.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise ExternalDatasetError(f"ZIP symlinks are not accepted: {member.filename!r}")
    if member.is_dir():
        return None
    return path


def safe_extract_zip(archive: Path, destination: Path) -> int:
    """Extract a ZIP without permitting traversal or symlink members."""

    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = _validated_zip_member(member)
            if relative is None:
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            extracted += 1
    return extracted


def inventory_directory(root: Path) -> dict[str, Any]:
    suffix_counts: dict[str, int] = {}
    file_count = 0
    total_bytes = 0
    image_count = 0
    annotation_count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".hwnas_extract_manifest.json":
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        image_count += int(suffix in IMAGE_SUFFIXES)
        annotation_count += int(suffix in ANNOTATION_SUFFIXES)
    return {
        "root": str(root.resolve()),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "image_count": image_count,
        "annotation_file_count": annotation_count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def audit_yolo_detection_tree(
    root: Path,
    *,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    """Audit image/YOLO-label pairing without inventing train/test semantics."""

    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    labels = [
        path
        for path in root.rglob("*.txt")
        if path.is_file() and path.name not in {"obj.names.txt", "yolov4-custom.txt"}
    ]
    def sample_key(path: Path, structural_directory: str) -> str:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        for index, part in enumerate(parts[:-1]):
            if part.lower() == structural_directory:
                del parts[index]
                break
        return Path(*parts).with_suffix("").as_posix()

    image_keys = {sample_key(path, "images"): path for path in images}
    label_keys = {sample_key(path, "labels"): path for path in labels}
    class_box_counts = {name: 0 for name in class_names.values()}
    class_image_keys = {name: set() for name in class_names.values()}
    empty_labels = 0
    total_boxes = 0
    malformed_rows: list[dict[str, Any]] = []
    out_of_range_rows: list[dict[str, Any]] = []
    unknown_class_rows: list[dict[str, Any]] = []

    for key, label_path in label_keys.items():
        text = label_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            empty_labels += 1
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                malformed_rows.append(
                    {"path": str(label_path), "line": line_number, "value": line}
                )
                continue
            try:
                raw_class, *raw_box = (float(item) for item in fields)
            except ValueError:
                malformed_rows.append(
                    {"path": str(label_path), "line": line_number, "value": line}
                )
                continue
            class_index = int(raw_class)
            if raw_class != class_index or class_index not in class_names:
                unknown_class_rows.append(
                    {"path": str(label_path), "line": line_number, "value": line}
                )
                continue
            if any(value < 0.0 or value > 1.0 for value in raw_box):
                out_of_range_rows.append(
                    {"path": str(label_path), "line": line_number, "value": line}
                )
                continue
            class_name = class_names[class_index]
            class_box_counts[class_name] += 1
            class_image_keys[class_name].add(key)
            total_boxes += 1

    paired_keys = set(image_keys) & set(label_keys)
    images_with_boxes = set().union(*class_image_keys.values())
    return {
        "root": str(root.resolve()),
        "class_index": {str(index): name for index, name in class_names.items()},
        "image_count": len(images),
        "label_file_count": len(labels),
        "paired_image_label_count": len(paired_keys),
        "missing_label_count": len(set(image_keys) - set(label_keys)),
        "orphan_label_count": len(set(label_keys) - set(image_keys)),
        "empty_annotation_file_count": empty_labels,
        "images_with_boxes": len(images_with_boxes),
        "total_boxes": total_boxes,
        "class_box_counts": class_box_counts,
        "class_image_counts": {
            name: len(keys) for name, keys in class_image_keys.items()
        },
        "images_with_multiple_classes": len(
            set.intersection(*class_image_keys.values())
        )
        if class_image_keys
        else 0,
        "malformed_row_count": len(malformed_rows),
        "out_of_range_row_count": len(out_of_range_rows),
        "unknown_class_row_count": len(unknown_class_rows),
        "malformed_rows": malformed_rows[:20],
        "out_of_range_rows": out_of_range_rows[:20],
        "unknown_class_rows": unknown_class_rows[:20],
        "empty_annotation_interpretation": (
            "Not inferred. Empty YOLO files are preserved as source data and must not "
            "be relabeled as a negative classification class without protocol review."
        ),
    }


def extract_zip_idempotent(archive: Path, destination: Path) -> dict[str, Any]:
    archive_sha256 = hash_file(archive)
    marker_name = ".hwnas_extract_manifest.json"
    marker = destination / marker_name
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalDatasetError(f"Invalid extraction marker: {marker}") from exc
        if existing.get("archive_sha256") != archive_sha256:
            raise ExternalDatasetError(
                f"Extracted directory belongs to a different archive: {destination}"
            )
        return inventory_directory(destination)
    if destination.exists():
        raise ExternalDatasetError(
            f"Refusing to overwrite untracked extracted directory: {destination}"
        )

    temporary = destination.parent / f".{destination.name}.extracting"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        extracted_files = safe_extract_zip(archive, temporary)
        marker_payload = {
            "archive": str(archive.resolve()),
            "archive_sha256": archive_sha256,
            "extracted_at": utc_now_iso(),
            "extracted_files": extracted_files,
        }
        (temporary / marker_name).write_text(
            json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return inventory_directory(destination)


def _write_json_file(target: Path, payload: Any) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)
    return target


def _write_local_manifest(directory: Path, payload: Mapping[str, Any]) -> Path:
    return _write_json_file(directory / "dataset_manifest.json", payload)


def fetch_figshare_mine_detection(
    output_root: Path,
    *,
    extract: bool = True,
) -> dict[str, Any]:
    metadata = request_json(FIGSHARE_API_URL)
    if metadata.get("id") != FIGSHARE_ARTICLE_ID:
        raise ExternalDatasetError("Figshare article ID changed unexpectedly")
    if metadata.get("version") != FIGSHARE_EXPECTED_VERSION:
        raise ExternalDatasetError(
            f"Expected Figshare v{FIGSHARE_EXPECTED_VERSION}, got v{metadata.get('version')}"
        )
    license_name = (metadata.get("license") or {}).get("name")
    if license_name != FIGSHARE_EXPECTED_LICENSE:
        raise ExternalDatasetError(
            f"Expected {FIGSHARE_EXPECTED_LICENSE}, got {license_name!r}"
        )

    dataset_dir = output_root / "figshare_mine_detection_v2"
    raw_dir = dataset_dir / "raw"
    extracted_dir = dataset_dir / "extracted"
    files: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {}
    annotation_audits: dict[str, Any] = {}
    remote_files = metadata.get("files")
    if not isinstance(remote_files, list) or not remote_files:
        raise ExternalDatasetError("Figshare metadata did not contain downloadable files")

    for record in remote_files:
        if not isinstance(record, dict):
            raise ExternalDatasetError("Malformed Figshare file metadata")
        name = str(record["name"])
        archive_path = raw_dir / name
        downloaded = download_file(
            str(record["download_url"]),
            archive_path,
            expected_size=int(record["size"]),
            expected_md5=str(record["supplied_md5"]).lower(),
        )
        downloaded.update(
            {
                "figshare_file_id": int(record["id"]),
                "name": name,
                "download_url": str(record["download_url"]),
                "expected_md5": str(record["supplied_md5"]).lower(),
            }
        )
        files.append(downloaded)
        if extract and zipfile.is_zipfile(archive_path):
            stem = Path(name).stem
            extract_root = extracted_dir / stem
            inventories[stem] = extract_zip_idempotent(archive_path, extract_root)
            if stem.isdigit():
                annotation_audits[stem] = audit_yolo_detection_tree(
                    extract_root,
                    class_names={0: "MILCO", 1: "NOMBO"},
                )

    payload = {
        "schema_version": 1,
        "dataset_id": "figshare_mine_detection_v2",
        "retrieved_at": utc_now_iso(),
        "source": {
            "article_id": metadata["id"],
            "title": metadata.get("title"),
            "article_url": metadata.get("url_public_html"),
            "api_url": FIGSHARE_API_URL,
            "doi": metadata.get("doi"),
            "version": metadata.get("version"),
            "license": metadata.get("license"),
            "authors": metadata.get("authors"),
            "published_date": metadata.get("published_date"),
            "description": metadata.get("description"),
        },
        "task_boundary": {
            "native_task": ["object_detection", "classification", "segmentation"],
            "classes": ["NOMBO", "MILCO"],
            "current_hwnas_pipeline_compatible": False,
            "reason": (
                "The main HW-NAS runtime is an NKSID image-classification pipeline. "
                "No object-detection annotations are silently converted or merged."
            ),
        },
        "files": files,
        "extracted": extract,
        "inventories": inventories,
        "annotation_audits": annotation_audits,
    }
    manifest = _write_local_manifest(dataset_dir, payload)
    payload["manifest_path"] = str(manifest.resolve())
    return payload


def _iter_candidate_urls(value: Any, key_path: str = "") -> Iterable[tuple[int, str]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_path = f"{key_path}.{key}" if key_path else str(key)
            yield from _iter_candidate_urls(nested, next_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _iter_candidate_urls(nested, f"{key_path}[{index}]")
    elif isinstance(value, str) and value.startswith(("https://", "http://")):
        lowered = key_path.lower()
        score = 0
        if "link" in lowered:
            score += 3
        if "download" in lowered:
            score += 2
        if "url" in lowered:
            score += 1
        if ".zip" in value.lower() or "signed" in value.lower():
            score += 2
        yield score, value


def find_download_url(payload: Mapping[str, Any]) -> str:
    candidates = sorted(_iter_candidate_urls(payload), reverse=True)
    if not candidates or candidates[0][0] <= 0:
        raise ExternalDatasetError("Roboflow export response contained no download URL")
    return candidates[0][1]


def resolve_latest_roboflow_version(payload: Mapping[str, Any]) -> int:
    project = payload.get("project", payload)
    if not isinstance(project, Mapping):
        raise ExternalDatasetError("Malformed Roboflow project metadata")
    versions = project.get("versions")
    if isinstance(versions, int) and versions > 0:
        return versions
    if isinstance(versions, str) and versions.isdigit() and int(versions) > 0:
        return int(versions)
    if isinstance(versions, list):
        candidates: list[int] = []
        for item in versions:
            if isinstance(item, int):
                candidates.append(item)
            elif isinstance(item, str) and item.isdigit():
                candidates.append(int(item))
            elif isinstance(item, Mapping):
                for key in ("version", "id", "number"):
                    value = item.get(key)
                    if isinstance(value, int):
                        candidates.append(value)
                        break
                    if isinstance(value, str) and value.isdigit():
                        candidates.append(int(value))
                        break
        if candidates:
            return max(candidates)
    raise ExternalDatasetError(
        "Could not resolve the latest Roboflow dataset version; pass "
        "--roboflow-version explicitly"
    )


def fetch_roboflow_cylider2(
    output_root: Path,
    *,
    api_key: Optional[str] = None,
    local_zip: Optional[Path] = None,
    export_format: str = "yolov8",
    version: Optional[int] = None,
    extract: bool = True,
) -> dict[str, Any]:
    if not api_key and local_zip is None:
        raise ExternalDatasetError(
            "Roboflow requires ROBOFLOW_API_KEY or --roboflow-zip pointing to a "
            "manually exported dataset ZIP"
        )

    if version is not None and version <= 0:
        raise ExternalDatasetError("Roboflow version must be a positive integer")
    dataset_dir = output_root / "roboflow_cylider2"
    raw_dir = dataset_dir / "raw"
    project_metadata: Optional[dict[str, Any]] = None
    project_endpoint: Optional[str] = None
    if local_zip is None:
        project_endpoint = (
            f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
        )
        project_metadata = request_json(
            project_endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resolved_version: Optional[int] = (
            version
            if version is not None
            else resolve_latest_roboflow_version(project_metadata)
        )
    else:
        resolved_version = version
    version_label = str(resolved_version) if resolved_version is not None else "manual"
    archive = raw_dir / f"cylider2_v{version_label}_{export_format}.zip"
    if local_zip is not None:
        source_zip = local_zip.expanduser().resolve()
        if not source_zip.is_file():
            raise ExternalDatasetError(f"Roboflow ZIP not found: {source_zip}")
        if not zipfile.is_zipfile(source_zip):
            raise ExternalDatasetError(f"Roboflow export is not a valid ZIP: {source_zip}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        if archive.exists() and hash_file(archive) != hash_file(source_zip):
            raise ExternalDatasetError(
                f"Existing Roboflow archive differs from {source_zip}: {archive}"
            )
        if not archive.exists():
            shutil.copy2(source_zip, archive)
        acquisition = "local_export_zip"
        api_endpoint = None
    else:
        api_endpoint = (
            f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/"
            f"{resolved_version}/{export_format}"
        )
        export_payload = request_json(
            api_endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        signed_url = find_download_url(export_payload)
        download_file(signed_url, archive)
        acquisition = "roboflow_api"

    archive_record = {
        "path": str(archive.resolve()),
        "bytes": archive.stat().st_size,
        "md5": hash_file(archive, "md5"),
        "sha256": hash_file(archive),
    }
    inventory = None
    if extract:
        inventory = extract_zip_idempotent(
            archive,
            dataset_dir / "extracted" / f"v{version_label}_{export_format}",
        )
    payload = {
        "schema_version": 1,
        "dataset_id": f"roboflow_cylider2_v{version_label}",
        "retrieved_at": utc_now_iso(),
        "source": {
            "requested_url": ROBOFLOW_REQUESTED_URL,
            "workspace": ROBOFLOW_WORKSPACE,
            "project": ROBOFLOW_PROJECT,
            "version": resolved_version,
            "license": "CC BY 4.0",
            "task": "object_detection",
            "source_image_count": 476,
            "published_version_count": 6,
            "published_classes": ["cylinder", "cylider", "manta"],
        },
        "acquisition": acquisition,
        "project_api_endpoint": project_endpoint,
        "api_endpoint": api_endpoint,
        "project_metadata_summary": (
            {
                key: project_metadata.get("project", project_metadata).get(key)
                for key in (
                    "id",
                    "name",
                    "type",
                    "images",
                    "versions",
                    "classes",
                    "splits",
                    "license",
                )
                if isinstance(project_metadata.get("project", project_metadata), Mapping)
                and key in project_metadata.get("project", project_metadata)
            }
            if project_metadata is not None
            else None
        ),
        "export_format": export_format,
        "archive": archive_record,
        "extracted": extract,
        "inventory": inventory,
        "task_boundary": {
            "current_hwnas_pipeline_compatible": False,
            "reason": (
                "The dataset is object detection data. The labels 'cylinder' and "
                "'cylider' remain distinct until a reviewed label-mapping policy exists."
            ),
        },
    }
    manifest = _write_local_manifest(dataset_dir, payload)
    payload["manifest_path"] = str(manifest.resolve())
    return payload


def yolo_row_from_pixel_box(
    box: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    label_to_index: Mapping[str, int],
) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ExternalDatasetError("Image dimensions must be positive")
    label = str(box.get("label", ""))
    if label not in label_to_index:
        raise ExternalDatasetError(f"Unknown Roboflow class label: {label!r}")
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalDatasetError(f"Malformed Roboflow box: {box!r}") from exc
    tolerance = 1e-6
    if width <= 0 or height <= 0:
        raise ExternalDatasetError(f"Non-positive Roboflow box: {box!r}")
    if (
        x - width / 2 < -tolerance
        or y - height / 2 < -tolerance
        or x + width / 2 > image_width + tolerance
        or y + height / 2 > image_height + tolerance
    ):
        raise ExternalDatasetError(f"Roboflow box exceeds image bounds: {box!r}")
    values = (
        x / image_width,
        y / image_height,
        width / image_width,
        height / image_height,
    )
    return " ".join(
        [str(label_to_index[label]), *(f"{value:.10f}" for value in values)]
    )


def _roboflow_version_record(
    project_metadata: Mapping[str, Any], version: int
) -> Mapping[str, Any]:
    versions = project_metadata.get("versions")
    if not isinstance(versions, list):
        raise ExternalDatasetError("Roboflow project metadata omitted version records")
    suffix = f"/{version}"
    for record in versions:
        if isinstance(record, Mapping) and str(record.get("id", "")).endswith(suffix):
            return record
    raise ExternalDatasetError(f"Roboflow version {version} was not found")


def fetch_roboflow_cylider2_public_source(
    output_root: Path,
    *,
    api_key: str,
    version: Optional[int] = None,
    workers: int = 6,
    tracked_index_output: Optional[Path] = None,
) -> dict[str, Any]:
    """Reconstruct the public v6 source view without claiming export equivalence.

    This mode uses the public project image index, original-image URLs, current
    split assignments, and pixel-space boxes. It does not apply the Roboflow
    version's resize preprocessing and is therefore deliberately named a source
    reconstruction rather than an official YOLO export.
    """

    if not api_key:
        raise ExternalDatasetError("Public-source reconstruction requires an API key")
    if workers <= 0:
        raise ExternalDatasetError("workers must be positive")
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    project_endpoint = (
        f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
    )
    project_metadata = request_json(project_endpoint, headers=auth_headers)
    latest_version = resolve_latest_roboflow_version(project_metadata)
    resolved_version = latest_version if version is None else int(version)
    if resolved_version != latest_version:
        raise ExternalDatasetError(
            "The public browse index exposes current split assignments only. "
            f"Requested v{resolved_version}, current/latest is v{latest_version}."
        )
    version_record = _roboflow_version_record(project_metadata, resolved_version)
    project = project_metadata.get("project")
    if not isinstance(project, Mapping):
        raise ExternalDatasetError("Malformed Roboflow project metadata")

    dataset_dir = output_root / "roboflow_cylider2"
    reconstruction_dir = (
        dataset_dir / "source_reconstruction" / f"v{resolved_version}"
    )
    metadata_dir = dataset_dir / "metadata" / f"v{resolved_version}"
    detail_dir = metadata_dir / "image_details"

    search_endpoint = f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/search/v1"
    search_results: list[dict[str, Any]] = []
    continuation_token: Optional[str] = None
    expected_total: Optional[int] = None
    seen_ids: set[str] = set()
    while True:
        search_payload: dict[str, Any] = {
            "query": f"dataset:{ROBOFLOW_PROJECT}",
            "pageSize": 500,
            "fields": ["width", "height", "filename", "split", "tags"],
        }
        if continuation_token:
            search_payload["continuationToken"] = continuation_token
        page = request_json_post(
            search_endpoint,
            search_payload,
            headers=auth_headers,
        )
        if expected_total is None:
            expected_total = int(page.get("total", 0))
        page_results = page.get("results")
        if not isinstance(page_results, list):
            raise ExternalDatasetError("Roboflow image search returned no results list")
        new_count = 0
        for item in page_results:
            if not isinstance(item, dict) or not item.get("id"):
                raise ExternalDatasetError("Malformed Roboflow image search record")
            image_id = str(item["id"])
            if image_id not in seen_ids:
                seen_ids.add(image_id)
                search_results.append(item)
                new_count += 1
        if expected_total and len(search_results) >= expected_total:
            break
        continuation_token = page.get("continuationToken")
        if not continuation_token or new_count == 0:
            break

    expected_images = int(version_record.get("images", 0))
    if expected_total != expected_images or len(search_results) != expected_images:
        raise ExternalDatasetError(
            "Roboflow public index count mismatch: "
            f"version={expected_images}, total={expected_total}, "
            f"retrieved={len(search_results)}"
        )
    _write_json_file(
        metadata_dir / "search_index.json",
        {
            "retrieved_at": utc_now_iso(),
            "query": f"dataset:{ROBOFLOW_PROJECT}",
            "total": expected_total,
            "results": search_results,
        },
    )

    def load_image_detail(item: Mapping[str, Any]) -> dict[str, Any]:
        image_id = str(item["id"])
        checkpoint = detail_dir / f"{image_id}.json"
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("id") == image_id:
                return cached
            raise ExternalDatasetError(f"Invalid Roboflow detail checkpoint: {checkpoint}")
        detail_url = (
            f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/"
            f"images/{image_id}"
        )
        payload = request_json(detail_url, headers=auth_headers)
        image = payload.get("image")
        if not isinstance(image, dict) or str(image.get("id")) != image_id:
            raise ExternalDatasetError(f"Malformed image detail for {image_id}")
        image.pop("embedding", None)
        _write_json_file(checkpoint, image)
        return image

    details_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_image_detail, item): str(item["id"])
            for item in search_results
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id = futures[future]
            details_by_id[image_id] = future.result()
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"[roboflow metadata] {completed}/{len(futures)} image records"
                )

    label_to_index = {
        label: index for index, label in enumerate(ROBOFLOW_RECONSTRUCTION_CLASSES)
    }
    split_counts: Counter[str] = Counter()
    class_box_counts: Counter[str] = Counter()
    search_by_id = {str(item["id"]): item for item in search_results}
    source_records: list[dict[str, Any]] = []
    name_keys: Counter[tuple[str, str]] = Counter()
    stem_keys: Counter[tuple[str, str]] = Counter()
    preliminary: list[dict[str, Any]] = []

    for image_id, detail in details_by_id.items():
        search_record = search_by_id[image_id]
        project_data = search_record.get("projectData", {}).get(ROBOFLOW_PROJECT, {})
        split = str(detail.get("split") or project_data.get("split") or "")
        if split not in {"train", "valid", "test"}:
            raise ExternalDatasetError(f"Invalid split for {image_id}: {split!r}")
        filename = str(detail.get("name") or search_record.get("filename") or "")
        if not filename or Path(filename).name != filename:
            raise ExternalDatasetError(f"Unsafe source filename: {filename!r}")
        annotation = detail.get("annotation")
        if not isinstance(annotation, Mapping):
            raise ExternalDatasetError(f"Missing annotation for {image_id}")
        image_width = int(annotation.get("width", search_record.get("width", 0)))
        image_height = int(annotation.get("height", search_record.get("height", 0)))
        boxes = annotation.get("boxes") or []
        if not isinstance(boxes, list):
            raise ExternalDatasetError(f"Malformed boxes for {image_id}")
        urls = detail.get("urls")
        if not isinstance(urls, Mapping) or not str(urls.get("original", "")).startswith(
            "https://"
        ):
            raise ExternalDatasetError(f"Missing original URL for {image_id}")
        split_counts[split] += 1
        for box in boxes:
            if not isinstance(box, Mapping):
                raise ExternalDatasetError(f"Malformed box for {image_id}")
            label = str(box.get("label", ""))
            if label not in label_to_index:
                raise ExternalDatasetError(f"Unknown class {label!r} for {image_id}")
            class_box_counts[label] += 1
        preliminary.append(
            {
                "image_id": image_id,
                "source_filename": filename,
                "split": split,
                "width": image_width,
                "height": image_height,
                "boxes": boxes,
                "source_url": str(urls["original"]),
            }
        )
        name_keys[(split, filename.lower())] += 1
        stem_keys[(split, Path(filename).stem.lower())] += 1

    expected_splits = {
        str(key): int(value)
        for key, value in dict(version_record.get("splits") or {}).items()
    }
    if dict(split_counts) != expected_splits:
        raise ExternalDatasetError(
            f"Roboflow split mismatch: {dict(split_counts)} != {expected_splits}"
        )
    project_level_class_counts = {
        str(key): int(value)
        for key, value in dict(project.get("classes") or {}).items()
    }
    if sum(class_box_counts.values()) < len(search_results):
        raise ExternalDatasetError(
            "Roboflow v6 contains fewer boxes than images: "
            f"{sum(class_box_counts.values())} < {len(search_results)}"
        )

    download_specs: list[dict[str, Any]] = []
    for record in preliminary:
        split = record["split"]
        source_filename = record["source_filename"]
        duplicate_name = name_keys[(split, source_filename.lower())] > 1
        duplicate_stem = (
            stem_keys[(split, Path(source_filename).stem.lower())] > 1
        )
        local_filename = (
            f"{record['image_id']}__{source_filename}"
            if duplicate_name or duplicate_stem
            else source_filename
        )
        image_path = reconstruction_dir / split / "images" / local_filename
        label_path = (
            reconstruction_dir
            / split
            / "labels"
            / Path(local_filename).with_suffix(".txt").name
        )
        label_rows = [
            yolo_row_from_pixel_box(
                box,
                image_width=record["width"],
                image_height=record["height"],
                label_to_index=label_to_index,
            )
            for box in record["boxes"]
        ]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            ("\n".join(label_rows) + "\n") if label_rows else "",
            encoding="utf-8",
        )
        download_specs.append(
            {
                **record,
                "local_filename": local_filename,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    def download_and_verify(spec: Mapping[str, Any]) -> dict[str, Any]:
        image_path = Path(spec["image_path"])
        download_record = download_file(
            str(spec["source_url"]),
            image_path,
            timeout=180,
        )
        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image_format = image.format
        except Exception as exc:
            raise ExternalDatasetError(f"Unreadable image: {image_path}") from exc
        expected_size = (int(spec["width"]), int(spec["height"]))
        if actual_size != expected_size:
            raise ExternalDatasetError(
                f"Image size mismatch for {image_path.name}: "
                f"{actual_size} != {expected_size}"
            )
        return {
            **download_record,
            "image_id": spec["image_id"],
            "image_format": image_format,
        }

    downloads_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_and_verify, spec): str(spec["image_id"])
            for spec in download_specs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_id = futures[future]
            downloads_by_id[image_id] = future.result()
            if completed % 50 == 0 or completed == len(futures):
                print(f"[roboflow images] {completed}/{len(futures)} verified")

    for spec in sorted(
        download_specs,
        key=lambda item: (str(item["split"]), str(item["local_filename"])),
    ):
        image_id = str(spec["image_id"])
        download = downloads_by_id[image_id]
        source_records.append(
            {
                "image_id": image_id,
                "source_filename": spec["source_filename"],
                "local_filename": spec["local_filename"],
                "split": spec["split"],
                "width": spec["width"],
                "height": spec["height"],
                "boxes": spec["boxes"],
                "source_url": spec["source_url"],
                "relative_image_path": str(
                    Path(spec["image_path"]).relative_to(dataset_dir).as_posix()
                ),
                "relative_label_path": str(
                    Path(spec["label_path"]).relative_to(dataset_dir).as_posix()
                ),
                "bytes": download["bytes"],
                "md5": download["md5"],
                "sha256": download["sha256"],
                "image_format": download["image_format"],
            }
        )

    hash_counts = Counter(record["sha256"] for record in source_records)
    exact_duplicate_groups = sum(1 for count in hash_counts.values() if count > 1)
    exact_duplicate_extra_copies = sum(count - 1 for count in hash_counts.values())
    index_payload = {
        "schema_version": 1,
        "dataset_id": f"roboflow_cylider2_v{resolved_version}_source",
        "retrieved_at": utc_now_iso(),
        "source_url": ROBOFLOW_REQUESTED_URL,
        "license": "CC BY 4.0",
        "version": resolved_version,
        "class_index": {
            str(index): label
            for index, label in enumerate(ROBOFLOW_RECONSTRUCTION_CLASSES)
        },
        "records": source_records,
    }
    index_path = _write_json_file(dataset_dir / "source_index.json", index_payload)
    if tracked_index_output is not None:
        _write_json_file(tracked_index_output.resolve(), index_payload)

    data_yaml = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "names:",
            *[
                f"  {index}: {label}"
                for index, label in enumerate(ROBOFLOW_RECONSTRUCTION_CLASSES)
            ],
            "",
        ]
    )
    (reconstruction_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    inventory = inventory_directory(reconstruction_dir)
    annotation_audit = audit_yolo_detection_tree(
        reconstruction_dir,
        class_names={
            index: label
            for index, label in enumerate(ROBOFLOW_RECONSTRUCTION_CLASSES)
        },
    )
    if (
        annotation_audit["paired_image_label_count"] != len(source_records)
        or annotation_audit["total_boxes"] != sum(class_box_counts.values())
        or annotation_audit["malformed_row_count"] != 0
        or annotation_audit["out_of_range_row_count"] != 0
        or annotation_audit["unknown_class_row_count"] != 0
    ):
        raise ExternalDatasetError(
            f"Reconstructed YOLO audit failed: {annotation_audit}"
        )
    payload = {
        "schema_version": 1,
        "dataset_id": f"roboflow_cylider2_v{resolved_version}_source",
        "retrieved_at": utc_now_iso(),
        "source": {
            "url": ROBOFLOW_REQUESTED_URL,
            "workspace": ROBOFLOW_WORKSPACE,
            "project": ROBOFLOW_PROJECT,
            "version": resolved_version,
            "license": project.get("license"),
            "task": project.get("type"),
            "version_record": version_record,
        },
        "acquisition": "public_source_reconstruction",
        "official_export_equivalent": False,
        "reconstruction_boundary": (
            "Original-resolution public images and pixel-space annotations were "
            "reconstructed with the current v6 split. Roboflow v6 auto-orient/1280x1280 "
            "resize preprocessing was not applied."
        ),
        "class_index": index_payload["class_index"],
        "image_count": len(source_records),
        "split_counts": dict(sorted(split_counts.items())),
        "total_boxes": sum(class_box_counts.values()),
        "class_box_counts": dict(sorted(class_box_counts.items())),
        "project_level_class_box_counts": dict(
            sorted(project_level_class_counts.items())
        ),
        "project_vs_version_class_count_match": (
            dict(class_box_counts) == project_level_class_counts
        ),
        "project_vs_version_class_count_note": (
            "Project-level class counts include project records outside the current "
            "v6 dataset. The reconstructed labels use the 478 v6 image-detail records."
        ),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_extra_copies": exact_duplicate_extra_copies,
        "inventory": inventory,
        "annotation_audit": annotation_audit,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": hash_file(index_path),
        "tracked_index_output": (
            str(tracked_index_output.resolve())
            if tracked_index_output is not None
            else None
        ),
        "task_boundary": {
            "current_hwnas_pipeline_compatible": False,
            "reason": (
                "The dataset is object detection data. The labels 'cylinder' and "
                "'cylider' remain distinct until a reviewed label-mapping policy exists."
            ),
        },
    }
    manifest = _write_local_manifest(dataset_dir, payload)
    payload["manifest_path"] = str(manifest.resolve())
    return payload


def resolve_roboflow_api_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    value = os.environ.get("ROBOFLOW_API_KEY")
    return value.strip() if value and value.strip() else None
