"""Explicit acknowledgment cannot accidentally promote a pending draft."""
from __future__ import annotations

import pytest

from tools.sonar_fcos_pilot.common import read_json, require_manifest, sha256_file, write_json
from tools.sonar_fcos_pilot.protocol import EXPLORATORY_ID, accept_exploratory


def setup_draft(tmp_path):
    original = tmp_path / "data_audit" / "manifest.json"
    write_json(original, {"split_status": "BLOCKED_GROUP_METADATA", "samples": []})
    draft = tmp_path / "exploratory_review" / "manifest.json"
    write_json(draft, {"split_status": "PENDING_PROTOCOL_APPROVAL", "small_area_threshold": .02,
                      "protocol": {"name": EXPLORATORY_ID, "status": "PENDING_USER_CHOICE", "training_authorized": False},
                      "audit": {"data_integrity_pass": True}, "source_manifest_sha256": sha256_file(original)})
    return {"output_dir": str(tmp_path), "checkpoint_root": str(tmp_path / "weights")}, original, draft


def test_pending_draft_is_not_loadable_and_accept_needs_explicit_confirmation(tmp_path):
    config, _, draft = setup_draft(tmp_path)
    with pytest.raises(RuntimeError):
        require_manifest({**config, "manifest_path": str(draft)})
    with pytest.raises(ValueError, match="明确确认"):
        accept_exploratory(config, None)
    assert not (draft.parent / "manifest_approved.json").exists()


def test_acknowledged_copy_preserves_original_and_binds_exact_manifest(tmp_path):
    config, original, draft = setup_draft(tmp_path)
    original_bytes, draft_bytes = original.read_bytes(), draft.read_bytes()
    path = accept_exploratory(config, EXPLORATORY_ID)
    approved_config = read_json(path)
    _, manifest = require_manifest(approved_config)
    assert manifest["protocol"]["training_authorized"] is True
    assert original.read_bytes() == original_bytes and draft.read_bytes() == draft_bytes
    assert accept_exploratory(config, EXPLORATORY_ID) == path
    write_json(approved_config["manifest_path"], {**manifest, "small_area_threshold": .5})
    with pytest.raises(RuntimeError, match="manifest changed"):
        require_manifest(approved_config)


def test_changed_source_audit_cannot_be_acknowledged(tmp_path):
    config, original, _ = setup_draft(tmp_path)
    write_json(original, {"split_status": "CHANGED"})
    with pytest.raises(ValueError, match="Original audit changed"):
        accept_exploratory(config, EXPLORATORY_ID)
