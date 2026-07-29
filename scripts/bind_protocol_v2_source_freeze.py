#!/usr/bin/env python3
"""Bind the frozen Protocol V2 config and selection manifests to a source snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/evaluation/nksid_frozen_protocol_v2.yaml",
    )
    parser.add_argument("--source-freeze-manifest", required=True)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/sonar_fourstage_operator_v2/"
            "protocol_v2_freeze_binding.json"
        ),
    )
    args = parser.parse_args()
    contract_path = Path(args.contract).resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN":
        raise ValueError("Protocol V2 contract must be FROZEN before source binding")
    freeze_path = Path(args.source_freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verifier = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(freeze_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    verification = json.loads(verifier.stdout)
    records = {str(row["path"]): row for row in freeze["files"]}
    relative_contract = contract_path.relative_to(ROOT).as_posix()
    frozen_contract = records.get(relative_contract)
    current_contract_sha = sha256_file(contract_path)
    contract_bound = bool(
        frozen_contract and frozen_contract["sha256"] == current_contract_sha
    )
    selection = contract["selection_evidence"]
    evidence_checks = {}
    for name in (
        "preregistration",
        "bundle_decision",
        "geometry_decision",
        "group_manifest",
        "dataset_manifest",
    ):
        path_key = f"{name}_path"
        sha_key = f"{name}_sha256"
        # The frozen config keeps ``*_path`` for decision files and the
        # shorter manifest field names for the two data manifests.
        configured_path = selection.get(path_key, selection.get(name))
        if configured_path is None:
            raise KeyError(
                f"selection_evidence requires {path_key!r} or {name!r}"
            )
        path = Path(configured_path).resolve()
        evidence_checks[name] = {
            "path": str(path),
            "expected_sha256": str(selection[sha_key]),
            "actual_sha256": sha256_file(path),
            "matches": sha256_file(path) == str(selection[sha_key]),
        }
    passed = (
        verifier.returncode == 0
        and verification.get("status") == "PASS"
        and contract_bound
        and all(row["matches"] for row in evidence_checks.values())
        and sha256_file(freeze["archive"]["path"]) == freeze["archive"]["sha256"]
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "protocol": contract["protocol"],
        "contract": {
            "path": str(contract_path),
            "sha256": current_contract_sha,
            "present_in_source_snapshot": frozen_contract is not None,
            "snapshot_sha256": (
                None if frozen_contract is None else frozen_contract["sha256"]
            ),
            "matches_snapshot": contract_bound,
        },
        "source_freeze": {
            "manifest_path": str(freeze_path),
            "manifest_sha256": sha256_file(freeze_path),
            "archive_path": str(Path(freeze["archive"]["path"]).resolve()),
            "archive_sha256": freeze["archive"]["sha256"],
            "verification": verification,
            "frozen_git": freeze["git"],
        },
        "selection_evidence_checks": evidence_checks,
        "dataset_content_sha256": selection["dataset_content_sha256"],
        "claim_boundary": (
            "This binds Protocol V2 preprocessing, training recipe, data/group "
            "manifests, and source snapshot. It contains no outer-validation metric."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
