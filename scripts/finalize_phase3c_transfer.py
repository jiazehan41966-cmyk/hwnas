#!/usr/bin/env python3
"""Freeze all Phase 3C epoch-80 checkpoints, then evaluate test exactly once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import torch

import architecture_transfer_common as common
from run_sctd_architecture_transfer import REGISTRATION as SCTD
from run_uatd_architecture_transfer import REGISTRATION as UATD


REGISTRATIONS = (UATD, SCTD)
FREEZE_MANIFEST = common.ROOT / "results/transfer/FORMAL_CHECKPOINT_FREEZE_MANIFEST.json"


def _context(registration):
    admission = common.validate_clean_manifest(registration)
    config = common.effective_config(registration)
    b1 = common.load_b1()
    return admission, config, b1


def freeze_all_checkpoints() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for registration in REGISTRATIONS:
        admission, config, _ = _context(registration)
        for kind in ("baseline", "nas"):
            for seed in common.FORMAL_SEEDS:
                identity = common.identity_for_kind(kind, int(config["dataset"]["num_classes"]))
                binding = common.build_binding(registration, admission, kind, seed, identity)
                output = registration.output_dir.resolve()
                run_dir = output / f"{kind}_seed{seed}_artifacts"
                checkpoint = run_dir / "latest.pt"
                history = run_dir / "history.json"
                training_summary_path = output / f"{kind}_seed{seed}_training.json"
                if not checkpoint.is_file() or not history.is_file() or not training_summary_path.is_file():
                    raise FileNotFoundError(f"formal training artifacts incomplete: {registration.dataset_id}/{kind}/seed{seed}")
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
                if state.get("epoch") != 80 or state.get("binding") != binding or state.get("test_evaluated") is not False:
                    raise ValueError(f"invalid sealed checkpoint: {registration.dataset_id}/{kind}/seed{seed}")
                if training_summary.get("binding") != binding or training_summary.get("test_evaluated") is not False:
                    raise ValueError(f"invalid training summary: {registration.dataset_id}/{kind}/seed{seed}")
                if common.sha256(history) != training_summary.get("history_sha256"):
                    raise ValueError(f"history hash mismatch: {registration.dataset_id}/{kind}/seed{seed}")
                records.append(
                    {
                        "dataset": registration.dataset_name,
                        "dataset_id": registration.dataset_id,
                        "modality": registration.modality,
                        "model": kind,
                        "seed": seed,
                        "architecture_id": identity["architecture_id"],
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": common.sha256(checkpoint),
                        "history_path": str(history),
                        "history_sha256": common.sha256(history),
                        "config_sha256": binding["baseline_config_sha256"],
                        "split_manifest_sha256": admission["sha256"],
                        "epoch": 80,
                        "training_complete": True,
                        "test_evaluated": False,
                        "binding": binding,
                    }
                )
    identities = {(row["dataset_id"], row["model"], row["seed"]) for row in records}
    checkpoint_hashes = {row["checkpoint_sha256"] for row in records}
    if len(records) != 12 or len(identities) != 12 or len(checkpoint_hashes) != 12:
        raise ValueError("freeze gate requires 12 unique runs and 12 unique epoch80 checkpoints")
    payload = {
        "artifact_type": "phase3c_formal_checkpoint_freeze_manifest",
        "status": "PASS",
        "formal_runs": "12/12",
        "checkpoint_policy": "epoch80/latest.pt",
        "missing": 0,
        "duplicates": 0,
        "failed": 0,
        "test_evaluated": False,
        "created_at": common.now(),
        "records": records,
    }
    common.dump(FREEZE_MANIFEST, payload)
    return payload


def _verify_frozen_manifest() -> dict[str, Any]:
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError("TEST_SEALED: FORMAL_CHECKPOINT_FREEZE_MANIFEST.json is missing")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("formal_runs") != "12/12" or len(manifest.get("records", [])) != 12:
        raise ValueError("TEST_SEALED: checkpoint freeze manifest did not pass")
    for row in manifest["records"]:
        if common.sha256(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]:
            raise ValueError("TEST_SEALED: frozen checkpoint hash changed")
        if common.sha256(Path(row["history_path"])) != row["history_sha256"]:
            raise ValueError("TEST_SEALED: frozen history hash changed")
    return manifest


def evaluate_all_frozen_checkpoints(device: str, num_workers: int) -> None:
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("CUDA_ENVIRONMENT_NOT_READY")
    manifest = _verify_frozen_manifest()
    records = {(row["dataset_id"], row["model"], int(row["seed"])): row for row in manifest["records"]}
    for registration in REGISTRATIONS:
        admission, config, b1 = _context(registration)
        for kind in ("baseline", "nas"):
            for seed in common.FORMAL_SEEDS:
                frozen = records[(registration.dataset_id, kind, seed)]
                output = registration.output_dir.resolve()
                summary_path = output / f"{kind}_seed{seed}.json"
                run_dir = output / f"{kind}_seed{seed}_artifacts"
                receipt = run_dir / "test_evaluation_receipt.json"
                identity = common.identity_for_kind(kind, int(config["dataset"]["num_classes"]))
                binding = common.build_binding(registration, admission, kind, seed, identity)
                if binding != frozen["binding"]:
                    raise ValueError("current source/protocol binding differs from frozen checkpoint")
                if summary_path.is_file():
                    result = json.loads(summary_path.read_text(encoding="utf-8"))
                    common.validate_formal_result(result, binding)
                    continue
                if receipt.is_file():
                    prior = json.loads(receipt.read_text(encoding="utf-8"))
                    if prior.get("status") != "COMPLETE":
                        raise RuntimeError("test evaluation was started but not completed; manual audit required")
                    raise RuntimeError("completed test receipt exists without immutable result summary")
                model, criterion, actual_identity = common.build_model(kind, config, b1)
                if actual_identity != identity:
                    raise ValueError("model identity differs from frozen record")
                state = torch.load(frozen["checkpoint_path"], map_location="cpu", weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.to(device)
                criterion.to(device)
                common.dump(receipt, {"status": "STARTED", "started_at": common.now(), "checkpoint_sha256": frozen["checkpoint_sha256"], "freeze_manifest_sha256": common.sha256(FREEZE_MANIFEST), "test_used_for_selection": False})
                test_loader = common.make_loader(config, "test", seed + 200, num_workers, train=False)
                test_metrics, _ = common.frozen_baseline.evaluate(model, test_loader, criterion, config, device)
                training_summary = json.loads((output / f"{kind}_seed{seed}_training.json").read_text(encoding="utf-8"))
                result = {
                    "artifact_type": "phase3c_architecture_transfer_result",
                    "dataset": registration.dataset_name,
                    "dataset_id": registration.dataset_id,
                    "modality": registration.modality,
                    "model": kind,
                    "architecture_id": identity["architecture_id"],
                    "seed": seed,
                    "checkpoint_policy": "epoch80/latest.pt",
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "history_sha256": frozen["history_sha256"],
                    "split_manifest_sha256": admission["sha256"],
                    "valid_epoch80": training_summary["valid_epoch80"],
                    "test_metrics": {**test_metrics, "mAP50": test_metrics["AP50"]},
                    "Params": training_summary["Params"],
                    "MACs": training_summary["MACs"],
                    "FLOPs": training_summary["FLOPs"],
                    "complexity_scope": training_summary["complexity_scope"],
                    "training_time_seconds": training_summary["training_time_seconds"],
                    "test_evaluated": True,
                    "test_used_for_selection": False,
                    "binding": binding,
                    "completed_at": common.now(),
                }
                common.dump(summary_path, result)
                common.dump(receipt, {"status": "COMPLETE", "completed_at": common.now(), "checkpoint_sha256": frozen["checkpoint_sha256"], "summary_sha256": common.sha256(summary_path), "test_used_for_selection": False})
                common.dump(run_dir / "status.json", {"status": "COMPLETE", "epoch": 80, "test_evaluated": True, "binding": binding})
        common.aggregate_results(registration, registration.output_dir.resolve(), common.FORMAL_SEEDS)
    write_cross_dataset_outputs()


def _dataset_row(registration) -> dict[str, Any]:
    rows = [json.loads((registration.output_dir / f"{kind}_seed{seed}.json").read_text(encoding="utf-8")) for kind in ("baseline", "nas") for seed in common.FORMAL_SEEDS]
    baseline = {int(row["seed"]): row for row in rows if row["model"] == "baseline"}
    b1 = {int(row["seed"]): row for row in rows if row["model"] == "nas"}
    delta50 = {seed: b1[seed]["test_metrics"]["AP50"] - baseline[seed]["test_metrics"]["AP50"] for seed in common.FORMAL_SEEDS}
    delta95 = {seed: b1[seed]["test_metrics"]["mAP50_95"] - baseline[seed]["test_metrics"]["mAP50_95"] for seed in common.FORMAL_SEEDS}
    sign = "all_positive" if all(value > 0 for value in delta95.values()) else "all_negative" if all(value <= 0 for value in delta95.values()) else "mixed"
    return {
        "dataset": registration.dataset_name,
        "modality": registration.modality,
        "baseline_mean_mAP50": statistics.mean(row["test_metrics"]["AP50"] for row in baseline.values()),
        "B1_mean_mAP50": statistics.mean(row["test_metrics"]["AP50"] for row in b1.values()),
        "delta_mAP50": statistics.mean(delta50.values()),
        "baseline_mean_mAP50_95": statistics.mean(row["test_metrics"]["mAP50_95"] for row in baseline.values()),
        "B1_mean_mAP50_95": statistics.mean(row["test_metrics"]["mAP50_95"] for row in b1.values()),
        "delta_mAP50_95": statistics.mean(delta95.values()),
        "delta_seed0": delta95[0], "delta_seed1": delta95[1], "delta_seed2": delta95[2],
        "delta_mAP50_seed0": delta50[0], "delta_mAP50_seed1": delta50[1], "delta_mAP50_seed2": delta50[2],
        "delta_sign_consistency": sign,
        "baseline_params": baseline[0]["Params"], "B1_params": b1[0]["Params"],
        "baseline_MACs": baseline[0]["MACs"], "B1_MACs": b1[0]["MACs"],
        "split_manifest_sha256": baseline[0]["split_manifest_sha256"],
    }


def write_cross_dataset_outputs() -> None:
    rows = [_dataset_row(registration) for registration in REGISTRATIONS]
    path = common.ROOT / "results/transfer/CROSS_DATASET_TRANSFER_SUMMARY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    positive = all(row["delta_mAP50"] > 0 and row["delta_mAP50_95"] > 0 for row in rows)
    negative = all(row["delta_mAP50"] <= 0 and row["delta_mAP50_95"] <= 0 for row in rows)
    conclusion = "POSITIVE_CROSS_DATASET_TRANSFER_DIRECTION" if positive else "TRANSFER_NOT_SUPPORTED" if negative else "DOMAIN_DEPENDENT_TRANSFER"
    lines = ["# Phase 3C Final Review", "", "## A. CUDA/runtime status", "", f"CUDA available: `{torch.cuda.is_available()}`.", "", "## B. Formal runs", "", "`12/12`; all epoch80/latest checkpoints were frozen before test."]
    for row in rows:
        lines += ["", f"## {row['dataset']}: baseline vs B1", "", f"- mean ΔmAP50: `{row['delta_mAP50']:+.6f}`", f"- mean ΔmAP50:95: `{row['delta_mAP50_95']:+.6f}`", f"- seed ΔmAP50:95: `{row['delta_seed0']:+.6f}`, `{row['delta_seed1']:+.6f}`, `{row['delta_seed2']:+.6f}`", f"- direction consistency: `{row['delta_sign_consistency']}`"]
    timing = (common.ROOT / "results/transfer/TRAINING_TIME_ESTIMATE.md").read_text(encoding="utf-8")
    lines += ["", "## F. Cross-dataset direction", "", f"`{conclusion}`", "", "## G. Training time", "", timing, "", "## H. Limitations", "", "Complete-architecture transfer only; this experiment cannot attribute effects to P2, DW5, modality, or SonarOp. Dataset A results were not added because their best-valid checkpoint protocol is not directly comparable to this epoch80 one-shot test protocol.", "", "## I. Test evaluation status", "", "`test_evaluated=true`; each frozen checkpoint was evaluated once after the global freeze gate passed.", "", "## J. Matched structural probe readiness", "", "Ready for human review; no probe is started automatically."]
    (common.ROOT / "results/transfer/PHASE3C_FINAL_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.mode in ("freeze", "all"):
        freeze_all_checkpoints()
    if args.mode in ("test", "all"):
        evaluate_all_frozen_checkpoints(args.device, args.num_workers)


if __name__ == "__main__":
    main()
