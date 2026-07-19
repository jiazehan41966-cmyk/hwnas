#!/usr/bin/env python3
"""Freeze and verify the exact source tree used by a formal experiment cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    ".Xil/",
    ".venv/",
    ".venv_cuda/",
    "artifacts/",
    "data/",
    "logs/",
    "reference/",
    "results/",
    "results_archive/",
    "hls_lut_builder/results/",
    "hls_lut_builder/board_harness/results/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def normalize_relative(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"freeze input is outside repository: {resolved}") from exc


def is_excluded(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def selected_paths(extras: Iterable[str] = ()) -> list[str]:
    raw = git_output("ls-files", "--cached", "--others", "--exclude-standard", "-z", binary=True)
    assert isinstance(raw, bytes)
    selected: set[str] = set()
    for item in raw.decode("utf-8", errors="surrogateescape").split("\0"):
        if not item:
            continue
        relative = normalize_relative(item)
        path = REPO_ROOT / relative
        if path.is_file() and not is_excluded(relative) and "__pycache__" not in path.parts:
            selected.add(relative)
    for item in extras:
        relative = normalize_relative(item)
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(relative)
    return sorted(selected)


def file_records(paths: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in paths:
        path = REPO_ROOT / relative
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_archive(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for record in records:
            relative = str(record["path"])
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, (REPO_ROOT / relative).read_bytes())


def freeze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    extras = [normalize_relative(item) for item in args.extra]
    paths = selected_paths(extras)
    records = file_records(paths)
    archive = output_dir / "source_snapshot.zip"
    build_archive(archive, records)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": str(git_output("status", "--porcelain=v1")).splitlines(),
        },
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "extras": extras,
        "file_count": len(records),
        "files": records,
        "archive": {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
    }
    manifest_path = output_dir / "source_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "manifest": str(manifest_path), "archive": manifest["archive"]}, indent=2))
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    current_paths = set(selected_paths(manifest.get("extras", [])))
    missing = sorted(set(expected) - current_paths)
    unexpected = sorted(current_paths - set(expected))
    changed: list[dict[str, str]] = []
    for relative, expected_hash in expected.items():
        path = REPO_ROOT / relative
        if path.is_file():
            actual = sha256_file(path)
            if actual != expected_hash:
                changed.append({"path": relative, "expected": expected_hash, "actual": actual})
    passed = not missing and not unexpected and not changed
    payload = {
        "status": "PASS" if passed else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "expected_file_count": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", required=True)
    freeze_parser.add_argument("--extra", action="append", default=[])
    freeze_parser.set_defaults(handler=freeze)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.set_defaults(handler=verify)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
