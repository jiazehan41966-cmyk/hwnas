import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from run_eval_protocol import source_freeze_provenance


def _manifest(tmp_path: Path) -> Path:
    archive = tmp_path / "source_snapshot.zip"
    archive.write_bytes(b"source snapshot")
    payload = {
        "schema_version": 1,
        "files": [{"path": "run_eval_protocol.py", "sha256": "unused"}],
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "source_freeze_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_source_freeze_provenance_binds_manifest_and_archive(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    verifier = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"status": "PASS", "expected_file_count": 1}),
        stderr="",
    )
    with patch("run_eval_protocol.subprocess.run", return_value=verifier):
        provenance = source_freeze_provenance(str(manifest))
    assert provenance is not None
    assert provenance["verification_status"] == "PASS"
    assert provenance["verified_file_count"] == 1
    assert provenance["manifest_sha256"]
    assert provenance["archive_sha256"]


def test_source_freeze_provenance_rejects_changed_archive(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(payload["archive"]["path"]).write_bytes(b"changed")
    verifier = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"status": "PASS", "expected_file_count": 1}),
        stderr="",
    )
    with patch("run_eval_protocol.subprocess.run", return_value=verifier):
        with pytest.raises(RuntimeError, match="archive hash mismatch"):
            source_freeze_provenance(str(manifest))


def test_source_freeze_provenance_rejects_failed_verification(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    verifier = subprocess.CompletedProcess(
        args=[],
        returncode=2,
        stdout=json.dumps({"status": "FAIL", "changed": ["run_eval_protocol.py"]}),
        stderr="",
    )
    with patch("run_eval_protocol.subprocess.run", return_value=verifier):
        with pytest.raises(RuntimeError, match="source freeze verification failed"):
            source_freeze_provenance(str(manifest))
