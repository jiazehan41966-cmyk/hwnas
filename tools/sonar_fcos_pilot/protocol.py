"""Explicitly acknowledge an exploratory data protocol; never runs training."""
from __future__ import annotations

import copy
from pathlib import Path

from .common import read_json, resolve_path, sha256_file, utc_now, write_json

EXPLORATORY_ID = "PUBLIC_SPLIT_EXPLORATORY_V1"


def accept_exploratory(config, confirmation):
    if confirmation != EXPLORATORY_ID:
        raise ValueError("必须明确确认探索性协议；不能自动替换原来的真实采集序列划分。")
    output = resolve_path(config["output_dir"])
    directory = output / "exploratory_review"
    draft_path = directory / "manifest.json"
    draft = read_json(draft_path)
    if draft.get("split_status") != "PENDING_PROTOCOL_APPROVAL" or draft.get("protocol", {}).get("name") != EXPLORATORY_ID:
        raise ValueError("This is not the reviewed exploratory draft.")
    if not draft.get("audit", {}).get("data_integrity_pass"):
        raise ValueError("Active images/annotations must pass the exploratory audit.")
    original_manifest_path = output / "data_audit" / "manifest.json"
    if draft["source_manifest_sha256"] != sha256_file(original_manifest_path):
        raise ValueError("Original audit changed after the draft was prepared.")
    approved_path = directory / "manifest_approved.json"
    acknowledgment_path = directory / "protocol_acknowledgment.json"
    config_path = directory / "approved_config.json"
    if approved_path.exists() or acknowledgment_path.exists() or config_path.exists():
        if not all(path.exists() for path in (approved_path, acknowledgment_path, config_path)):
            raise RuntimeError("Partial acknowledgment exists; inspect the records before retrying.")
        existing = read_json(config_path)
        if existing["approved_protocol"]["manifest_sha256"] != sha256_file(approved_path):
            raise ValueError("Approved protocol record changed.")
        return config_path
    acknowledgment = {
        "protocol_id": EXPLORATORY_ID, "accepted_at": utc_now(), "accepted_via": "explicit_command_line_confirmation",
        "draft_manifest_sha256": sha256_file(draft_path),
        "limitations": ["No verified acquisition-sequence separation", "Invalid-annotation images excluded as whole images", "Evaluation is on a documented valid-annotation subset of the official test data", "No claim of performance in new acquisition scenes"],
        "original_data_unchanged": True,
    }
    write_json(acknowledgment_path, acknowledgment)
    approved = copy.deepcopy(draft)
    approved.update(split_status="PASS", protocol_id=EXPLORATORY_ID,
                    protocol_acknowledgment_sha256=sha256_file(acknowledgment_path))
    approved["protocol"]["status"] = "APPROVED"
    approved["protocol"]["training_authorized"] = True
    write_json(approved_path, approved)
    approved_config = {key: value for key, value in config.items() if key not in {"config_path", "config_sha256"}}
    approved_config.update(
        output_dir=str(output / "exploratory"), manifest_path=str(approved_path),
        checkpoint_root=str(resolve_path(config["checkpoint_root"]) / "exploratory"),
        approved_protocol={"status": "APPROVED", "protocol_id": EXPLORATORY_ID,
                           "manifest_sha256": sha256_file(approved_path),
                           "acknowledgment_path": str(acknowledgment_path),
                           "acknowledgment_sha256": sha256_file(acknowledgment_path)},
    )
    write_json(config_path, approved_config)
    return config_path
