#!/usr/bin/env python3
"""Audit benchmark smoke artifacts and preserve their non-claimable boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths, sha256_file, validate_prediction_record, write_json
from hwnas_fpga.training.protocol_reporting import canonical_sha256


def _audit_run(
    run_dir: Path, *, require_verified_environment: bool = False
) -> dict[str, Any]:
    summary_path = run_dir / "protocol_summary.json"
    manifest_path = run_dir / "run_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return {"run": run_dir.name, "status": "MISSING", "formal_eligible": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_files = sorted(run_dir.glob("outer_predictions_*.jsonl"))
    validation_errors = []
    prediction_count = 0
    for prediction_path in prediction_files:
        for line_number, line in enumerate(
            prediction_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                validate_prediction_record(json.loads(line))
            except Exception as error:  # audit must preserve the precise row location
                validation_errors.append(f"{prediction_path.name}:{line_number}:{error}")
            prediction_count += 1
    claimability = summary.get("claimability") or {}
    completed_pairs = manifest.get("completed_pairs") or []
    smoke_shape = len(completed_pairs) == 1
    immutable = dict(manifest.get("immutable_config") or {})
    runtime = dict(immutable.get("runtime") or {})
    environment = immutable.get("environment_card")
    environment_verified = False
    environment_error = None
    if isinstance(environment, dict):
        card_path = Path(str(environment.get("path") or ""))
        try:
            if not card_path.is_file():
                raise FileNotFoundError(card_path)
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("isolation_status") != "READY_DEDICATED_ENVIRONMENT":
                raise RuntimeError("environment card is not READY")
            dedicated = dict(card.get("dedicated_environment") or {})
            stable_fingerprint = canonical_sha256(
                {
                    "runtime_role": card.get("runtime_role"),
                    "path": dedicated.get("path"),
                    "interpreter": dedicated.get("interpreter"),
                    "probe": dedicated.get("probe"),
                    "freeze": dedicated.get("freeze"),
                }
            )
            if dedicated.get("verification_fingerprint") != stable_fingerprint:
                raise RuntimeError("environment card verification fingerprint is invalid")
            if environment.get("verification_fingerprint") != stable_fingerprint:
                raise RuntimeError("run environment fingerprint does not match current card")
            if Path(str(runtime.get("sys_prefix") or "")).resolve() != Path(
                str(dedicated.get("path") or "")
            ).resolve():
                raise RuntimeError("run sys_prefix does not match environment card")
            freeze = dict(dedicated.get("freeze") or {})
            freeze_path = Path(str(freeze.get("path") or ""))
            if not freeze_path.is_file() or sha256_file(freeze_path) != environment.get(
                "freeze_sha256"
            ):
                raise RuntimeError("environment lock missing or changed")
            environment_verified = True
        except Exception as error:  # preserve exact environment-chain failure
            environment_error = str(error)
    environment_ok = environment_verified or not require_verified_environment
    return {
        "run": run_dir.name,
        "task": summary.get("task"),
        "status": (
            "PASS"
            if prediction_count and not validation_errors and smoke_shape and environment_ok
            else "FAIL"
        ),
        "fold_seed_pairs": completed_pairs,
        "prediction_count": prediction_count,
        "prediction_schema_errors": validation_errors,
        "checkpoint_count": len(list(run_dir.glob("best_fold*_seed*.pt"))),
        "claimable": bool(claimability.get("claimable")),
        "claim_scope": claimability.get("claim_scope"),
        "verified_dedicated_environment_required": require_verified_environment,
        "verified_dedicated_environment": environment_verified,
        "environment_error": environment_error,
        "formal_eligible": False,
        "summary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
    }


def _audit_method_contract(path: Path, accepted_status: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "artifact": path.parent.name,
            "status": "MISSING",
            "formal_eligible": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed_status = str(payload.get("status"))
    claimability = str(payload.get("claimability_status"))
    passed = observed_status == accepted_status and claimability == "NOT_CLAIMABLE"
    return {
        "artifact": path.parent.name,
        "paper_id": payload.get("paper_id"),
        "status": "PASS" if passed else "FAIL",
        "observed_status": observed_status,
        "claimability_status": claimability,
        "formal_eligible": False,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    args = parser.parse_args()
    paths = CampaignPaths.from_repo(PROJECT_ROOT, args.campaign_id).create()
    smoke_root = paths.raw_root / "smoke"
    runs = [
        _audit_run(smoke_root / "builtin_closed_1x1x1"),
        _audit_run(smoke_root / "builtin_open_msp_1x1x1"),
        _audit_run(
            smoke_root / "sure_dedicated_env_1x1x1",
            require_verified_environment=True,
        ),
        _audit_run(
            smoke_root / "dmcl_dedicated_env_1x1x1",
            require_verified_environment=True,
        ),
        _audit_run(
            smoke_root / "plud_dedicated_env_1x1x1",
            require_verified_environment=True,
        ),
    ]
    method_contracts = [
        _audit_method_contract(
            smoke_root / "hwpr_paper_spec_10" / "paper_spec_smoke.json",
            "PASS_NONCLAIMABLE_PAPER_SPEC_SMOKE",
        ),
        _audit_method_contract(
            smoke_root / "harp_author_graph_contract" / "graph_contract.json",
            "PASS_AUTHOR_GRAPH_CONTRACT_ONLY",
        ),
        _audit_method_contract(
            smoke_root / "esda_author_artifact_contract" / "evidence_chain.json",
            "PASS_AUTHOR_ARTIFACT_CONTRACT_ONLY",
        ),
    ]
    gate_path = PROJECT_ROOT / "artifacts" / "measurement_first_rebuild" / "status.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "method_contracts": method_contracts,
        "all_integration_smokes_pass": all(row["status"] == "PASS" for row in runs),
        "all_method_contract_smokes_pass": all(
            row["status"] == "PASS" for row in method_contracts
        ),
        "all_builtin_integration_smokes_pass": all(
            row["status"] == "PASS" for row in runs[:2]
        ),
        "any_smoke_formal_eligible": False,
        "measurement_first": {
            "overall_status": gate.get("overall_status"),
            "path": str(gate_path.resolve()),
            "sha256": sha256_file(gate_path),
        },
        "claim_boundary": (
            "The 1-fold x 1-seed x 1-epoch simplecnn runs validate only the canonical "
            "closed/open-set pipeline and prediction schema. HW-PR, HARP, and ESDA "
            "contract smokes validate only bounded data/evidence flow. None populate "
            "T2/T3/T5/T6/T7/T8 or support model-performance or cross-board claims."
        ),
    }
    output = write_json(paths.artifact_root / "manifests" / "integration_smoke.json", payload)
    print(output)
    return 0 if (
        payload["all_integration_smokes_pass"]
        and payload["all_method_contract_smokes_pass"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
