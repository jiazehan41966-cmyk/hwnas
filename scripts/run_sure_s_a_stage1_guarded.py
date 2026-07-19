#!/usr/bin/env python3
"""Fail-closed SURE S-A stage-1 runner.

This runner is intentionally unusable with the checked-in template. It needs a
one-time AUTHORIZED record bound to a fresh source freeze. It runs only a
MobileNetV2 technical smoke and then one new unit of the full 15-unit formal
fingerprint, after which it pauses for the user's decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "ccf_ab_nksid_av7k325_v1"
CAMPAIGN = ROOT / "artifacts/benchmarks" / CAMPAIGN_ID
MANIFESTS = CAMPAIGN / "manifests"
CONFIG = ROOT / "configs/benchmarks/adapters/sure_2024.yaml"
ENV_CARD = CAMPAIGN / "environment/sure_2024.json"
CHECKOUT = ROOT / "reference/_local/SURE"
SMOKE_ROOT = ROOT / "results/benchmarks" / CAMPAIGN_ID / "smoke"
FORMAL_ROOT = ROOT / "results/benchmarks" / CAMPAIGN_ID / "formal/closed"
LOG_ROOT = ROOT / "logs/benchmarks" / CAMPAIGN_ID / "sure_s_a_stage1_20260719"
ACTIVE_AUTH_NAME = "sure_formal_execution_authorization_20260719.json.txt"
AUTHOR_COMMIT = "5ce0193bc93e73b1c7f1f53aeda8854e997011e2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_sure_stage1_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "status": "AUTHORIZED",
        "decision": "S-A",
        "scope": "sure_technical_smoke_then_first_formal_unit_only",
        "paper_id": "sure_2024",
        "adapter_id": "sure_author_recipe",
        "method_id": "sure_same_backbone",
        "author_commit": AUTHOR_COMMIT,
        "backbone": "mobilenet_v2",
        "image_size": 224,
        "batch_size": 8,
        "mandatory_pause_after_first_formal_unit": True,
        "allows_remaining_fourteen_units": False,
        "allows_hls_route_board_power": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"authorization {key} mismatch")
    smoke = dict(payload.get("technical_smoke") or {})
    if smoke.get("folds") != [0] or smoke.get("seeds") != [42] or smoke.get("epochs") != 1:
        errors.append("technical smoke scope mismatch")
    formal = dict(payload.get("first_formal_stage") or {})
    if (
        formal.get("planned_folds") != [0, 1, 2, 3, 4]
        or formal.get("planned_seeds") != [42, 43, 44]
        or formal.get("epochs") != 150
        or formal.get("max_new_units") != 1
        or formal.get("run_name") != "sure_same_backbone"
    ):
        errors.append("first formal stage scope mismatch")
    for key in (
        "source_freeze_manifest_sha256",
        "source_freeze_archive_sha256",
    ):
        value = str(payload.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            errors.append(f"authorization {key} is not a SHA256")
    freeze_path = Path(str(payload.get("source_freeze_manifest") or ""))
    if not freeze_path.is_absolute():
        freeze_path = ROOT / freeze_path
    if not freeze_path.is_file():
        errors.append("authorized source freeze manifest is missing")
    return errors


def command_base(
    interpreter: Path,
    freeze_manifest: Path,
    *,
    method_id: str,
    run_name: str,
    output_dir: Path,
    epochs: int,
    folds: str,
    seeds: str,
) -> list[str]:
    return [
        str(interpreter),
        str(ROOT / "run_eval_protocol.py"),
        "--task", "closed_set",
        "--adapter-id", "sure_author_recipe",
        "--adapter-config", str(CONFIG),
        "--campaign-id", CAMPAIGN_ID,
        "--paper-id", "sure_2024",
        "--method-id", method_id,
        "--arch", "mobilenet_v2",
        "--folds", folds,
        "--seeds", seeds,
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--gradient-accumulation-steps", "1",
        "--no-amp",
        "--image-size", "224",
        "--num-workers", "4",
        "--device", "cuda",
        "--selection-provenance", "baseline_predeclared",
        "--source-freeze-manifest", str(freeze_manifest),
        "--environment-card", str(ENV_CARD),
        "--sure-checkout", str(CHECKOUT),
        "--sure-commit", AUTHOR_COMMIT,
        "--output-dir", str(output_dir),
        "--run-name", run_name,
        "--resume",
    ]


def run_logged(command: list[str], stem: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_ROOT / f"{stem}_stdout.log"
    stderr_path = LOG_ROOT / f"{stem}_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stem} exited {completed.returncode}; logs: {stdout_path}, {stderr_path}"
        )


def patch_integrity(summary: dict[str, Any]) -> bool:
    patch = dict(((summary.get("provenance") or {}).get("code") or {}).get("tracked_patch") or {})
    path = Path(str(patch.get("path") or ""))
    return bool(path.is_file() and len(str(patch.get("sha256") or "")) == 64 and sha256(path) == patch.get("sha256"))


def audit_one_unit(run_dir: Path, *, epochs: int, expect_stage_limit: bool) -> dict[str, Any]:
    summary_path = run_dir / "protocol_summary.json"
    record_path = run_dir / "run_fold0_seed42.json"
    checkpoint_path = run_dir / "best_fold0_seed42.pt"
    prediction_path = run_dir / "outer_predictions_fold0_seed42.jsonl"
    for path in (summary_path, record_path, checkpoint_path, prediction_path):
        if not path.is_file():
            raise RuntimeError(f"required run artifact missing: {path}")
    summary = load_json(summary_path)
    record = load_json(record_path)
    claimability = dict(summary.get("claimability") or {})
    calibration = dict(record.get("outer_calibration") or {})
    required_metrics = ("nll", "brier", "ece", "aurc", "failure_auroc", "failure_fpr95")
    if summary.get("model", {}).get("name") != "mobilenet_v2":
        raise RuntimeError("run did not use MobileNetV2")
    if summary.get("recipe", {}).get("epochs") != epochs:
        raise RuntimeError("run epoch count mismatch")
    if claimability.get("observed_run_count") != 1 or claimability.get("claimable") is not False:
        raise RuntimeError("staged run must contain exactly one non-claimable unit")
    if not patch_integrity(summary):
        raise RuntimeError("tracked patch integrity failed")
    if any(not math.isfinite(float(calibration.get(key, math.nan))) for key in required_metrics):
        raise RuntimeError("calibration or failure-prediction metric is non-finite")
    execution = dict(summary.get("execution_control") or {})
    if expect_stage_limit and not (
        execution.get("max_new_units") == 1
        and execution.get("new_units_completed") == 1
        and execution.get("stage_limit_hit") is True
    ):
        raise RuntimeError("formal stage did not stop cleanly after one new unit")
    if not expect_stage_limit and execution.get("stage_limit_hit") is True:
        raise RuntimeError("technical smoke unexpectedly reported a stage limit")
    return {
        "summary_sha256": sha256(summary_path),
        "run_fingerprint": summary.get("run_fingerprint"),
        "macro_f1": summary.get("outer_macro_f1", {}).get("mean"),
        "top1": summary.get("outer_top1", {}).get("mean"),
        "weighted_f1": summary.get("outer_weighted_f1", {}).get("mean"),
        "per_class_f1": summary.get("per_class_f1_mean"),
        "calibration": {key: calibration[key] for key in required_metrics},
        "runtime_measurement": record.get("runtime_measurement"),
        "execution_control": execution,
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction_sha256": sha256(prediction_path),
        "prediction_rows": sum(1 for _ in prediction_path.open("r", encoding="utf-8")),
    }


def consume_authorization(path: Path, suffix: str) -> Path:
    destination = path.with_name(path.name.replace(".json.txt", f".{suffix}.json.txt"))
    if destination.exists():
        raise RuntimeError(f"authorization consumption target already exists: {destination}")
    path.replace(destination)
    return destination


def write_incident(reason: str, authorization: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": 1,
        "language": "zh-CN",
        "status": "INTERRUPTED_FAIL_CLOSED",
        "reason_zh": reason,
        "formal_records": len(list((FORMAL_ROOT / "sure_same_backbone").glob("run_fold*_seed*.json"))),
        "boundary_zh": "禁止自动重试；由用户决定后续动作。没有授权剩余 14 单元或任何硬件实验。",
    }
    (MANIFESTS / f"sure_s_a_stage1_incident_{timestamp}.json.txt").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (MANIFESTS / f"sure_s_a_stage1_incident_{timestamp}.md").write_text(
        "# SURE S-A 第一阶段中断记录\n\n"
        f"- 状态：`INTERRUPTED_FAIL_CLOSED`。\n- 原因：{reason}\n"
        "- 禁止自动重试；剩余 14 单元、HLS、route、板卡和功耗均未授权。\n",
        encoding="utf-8",
    )
    if authorization.exists():
        consume_authorization(authorization, "failed_consumed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    authorization = Path(args.authorization).resolve()
    if authorization.name != ACTIVE_AUTH_NAME or authorization.parent != MANIFESTS.resolve():
        raise RuntimeError("authorization must be the exact campaign-scoped active file")
    payload = load_json(authorization)
    errors = validate_sure_stage1_authorization(payload)
    if errors:
        raise RuntimeError("; ".join(errors))

    try:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8-sig")) or {}
        if config.get("formal_execution_enabled") is not True:
            raise RuntimeError("SURE formal_execution_enabled is not true")
        freeze_manifest = Path(str(payload["source_freeze_manifest"]))
        if not freeze_manifest.is_absolute():
            freeze_manifest = ROOT / freeze_manifest
        if sha256(freeze_manifest) != payload["source_freeze_manifest_sha256"]:
            raise RuntimeError("authorized source freeze manifest SHA256 mismatch")
        freeze = load_json(freeze_manifest)
        archive = Path(str((freeze.get("archive") or {}).get("path") or ""))
        if not archive.is_file() or sha256(archive) != payload["source_freeze_archive_sha256"]:
            raise RuntimeError("authorized source freeze archive SHA256 mismatch")
        verify = subprocess.run(
            [sys.executable, str(ROOT / "scripts/freeze_experiment_source.py"), "verify", "--manifest", str(freeze_manifest)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            verify_payload = json.loads(verify.stdout)
        except json.JSONDecodeError:
            verify_payload = {}
        if verify.returncode != 0 or verify_payload.get("status") != "PASS":
            raise RuntimeError("live source freeze verification failed")

        environment = load_json(ENV_CARD)
        interpreter = Path(str((environment.get("dedicated_environment") or {}).get("interpreter") or ""))
        if not interpreter.is_file():
            raise RuntimeError("SURE dedicated interpreter is missing")
        observed_commit = subprocess.run(
            ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        ).stdout.strip()
        if observed_commit != AUTHOR_COMMIT:
            raise RuntimeError("SURE author checkout commit changed")

        smoke_name = str((payload.get("technical_smoke") or {})["run_name"])
        smoke_dir = SMOKE_ROOT / smoke_name
        formal_dir = FORMAL_ROOT / "sure_same_backbone"
        if smoke_dir.exists() or formal_dir.exists():
            raise RuntimeError("stage output directory already exists; refusing overwrite")

        smoke_command = command_base(
            interpreter, freeze_manifest,
            method_id="sure_tech_smoke_mobilenet_v2",
            run_name=smoke_name,
            output_dir=SMOKE_ROOT,
            epochs=1,
            folds="0",
            seeds="42",
        )
        run_logged(smoke_command, "technical_smoke")
        smoke_audit = audit_one_unit(smoke_dir, epochs=1, expect_stage_limit=False)

        formal_command = command_base(
            interpreter, freeze_manifest,
            method_id="sure_same_backbone",
            run_name="sure_same_backbone",
            output_dir=FORMAL_ROOT,
            epochs=150,
            folds="0,1,2,3,4",
            seeds="42,43,44",
        ) + ["--max-new-units", "1"]
        run_logged(formal_command, "first_formal_unit")
        formal_audit = audit_one_unit(formal_dir, epochs=150, expect_stage_limit=True)

        result = {
            "schema_version": 1,
            "language": "zh-CN",
            "status": "PAUSED_AFTER_FIRST_FORMAL_UNIT_USER_DECISION_REQUIRED",
            "authorization_sha256": sha256(authorization),
            "source_freeze_manifest_sha256": sha256(freeze_manifest),
            "technical_smoke": smoke_audit,
            "first_formal_unit": formal_audit,
            "remaining_units_authorized": False,
            "boundary_zh": "首个正式单元后强制暂停。不得自动运行剩余 14 单元或任何硬件实验。",
        }
        result_path = MANIFESTS / "sure_s_a_stage1_result_20260719.json.txt"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (MANIFESTS / "sure_s_a_stage1_result_20260719.md").write_text(
            "# SURE S-A 第一阶段结果与强制暂停\n\n"
            f"- 状态：`{result['status']}`。\n"
            f"- 首单元 macro_f1：`{formal_audit['macro_f1']}`。\n"
            f"- 首单元 top1：`{formal_audit['top1']}`。\n"
            f"- 首单元校准与失败预测：`{json.dumps(formal_audit['calibration'], ensure_ascii=False)}`。\n"
            f"- 首单元运行资源：`{json.dumps(formal_audit['runtime_measurement'], ensure_ascii=False)}`。\n"
            "- 剩余 14 单元未授权；必须由用户作出关键决策。\n",
            encoding="utf-8",
        )
        consumed = consume_authorization(authorization, "consumed")
        print(json.dumps({"status": result["status"], "result": str(result_path), "authorization": str(consumed)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        write_incident(str(exc), authorization)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
