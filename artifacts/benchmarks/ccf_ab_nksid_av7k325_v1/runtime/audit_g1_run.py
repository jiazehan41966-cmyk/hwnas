#!/usr/bin/env python3
"""Independent, fail-closed audit for one completed 5-fold x 3-seed G1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path


EXPECTED_PAIRS = {(fold, seed) for fold in range(5) for seed in (42, 43, 44)}
EXPECTED_FOLD_SIZES = {0: 520, 1: 520, 2: 520, 3: 520, 4: 537}
EXPECTED_DATASET_SIZE = 2617


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metric_values(targets: list[int], predictions: list[int], classes: int) -> dict[str, float]:
    supports = [0] * classes
    predicted = [0] * classes
    true_positive = [0] * classes
    correct = 0
    for target, prediction in zip(targets, predictions, strict=True):
        supports[target] += 1
        predicted[prediction] += 1
        if target == prediction:
            correct += 1
            true_positive[target] += 1
    f1 = []
    for index in range(classes):
        denominator = supports[index] + predicted[index]
        f1.append(0.0 if denominator == 0 else (2.0 * true_positive[index]) / denominator)
    total = len(targets)
    return {
        "macro_f1": sum(f1) / classes,
        "top1": correct / total,
        "weighted_f1": sum(value * support for value, support in zip(f1, supports, strict=True)) / total,
    }


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-pretrained", choices=("true", "false"), required=True)
    parser.add_argument("--expected-method", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    expect_pretrained = args.expect_pretrained == "true"
    errors: list[str] = []
    checks: dict[str, object] = {}

    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "protocol_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        errors.append("run_manifest.json or protocol_summary.json is missing")
        manifest = {}
        summary = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))

    fingerprint = str(manifest.get("run_fingerprint") or "")
    claimability = dict(summary.get("claimability") or {})
    completed_pairs = {
        (int(row["fold"]), int(row["seed"])) for row in (manifest.get("completed_pairs") or [])
    }
    checks["manifest_completed_pairs"] = len(completed_pairs)
    if completed_pairs != EXPECTED_PAIRS:
        errors.append(f"manifest pair set mismatch: {sorted(completed_pairs)}")
    if not (
        claimability.get("claimable") is True
        and claimability.get("protocol_complete") is True
        and claimability.get("source_freeze_verified") is True
        and int(claimability.get("observed_run_count") or 0) == 15
        and not claimability.get("missing_pairs")
        and not claimability.get("unexpected_pairs")
    ):
        errors.append("summary claimability is incomplete")
    if len(fingerprint) != 64 or summary.get("run_fingerprint") != fingerprint:
        errors.append("manifest/summary run fingerprint mismatch")

    immutable = dict(manifest.get("immutable_config") or {})
    source_freeze = dict(immutable.get("source_freeze") or {})
    source_manifest_path = Path(str(source_freeze.get("path") or ""))
    source_archive_path = Path(str(source_freeze.get("archive_path") or ""))
    source_manifest: dict[str, object] = {}
    source_manifest_hash_ok = False
    source_archive_hash_ok = False
    try:
        if source_manifest_path.is_file():
            source_manifest_hash_ok = (
                sha256_file(source_manifest_path)
                == str(source_freeze.get("manifest_sha256") or "")
            )
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8-sig")
            )
        if source_archive_path.is_file():
            source_archive_hash_ok = (
                sha256_file(source_archive_path)
                == str(source_freeze.get("archive_sha256") or "")
            )
    except (OSError, TypeError, ValueError):
        source_manifest = {}
        source_manifest_hash_ok = False
        source_archive_hash_ok = False
    expected_source_files = int(source_manifest.get("file_count") or 0)
    checks["source_freeze"] = {
        **source_freeze,
        "observed_manifest_hash_ok": source_manifest_hash_ok,
        "observed_archive_hash_ok": source_archive_hash_ok,
        "manifest_file_count": expected_source_files,
    }
    if not (
        source_freeze.get("verification_status") == "PASS"
        and expected_source_files > 0
        and int(source_freeze.get("verified_file_count") or 0)
        == expected_source_files
        and source_manifest_hash_ok
        and source_archive_hash_ok
        and str((source_manifest.get("archive") or {}).get("sha256") or "")
        == str(source_freeze.get("archive_sha256") or "")
    ):
        errors.append("source-freeze binding is invalid")

    code_state = str((manifest.get("code_provenance") or {}).get("code_state_sha256") or "")
    tracked_patch = dict((manifest.get("code_provenance") or {}).get("tracked_patch") or {})
    tracked_patch_path = Path(str(tracked_patch.get("path") or ""))
    if not tracked_patch_path.is_file() or sha256_file(tracked_patch_path) != tracked_patch.get("sha256"):
        errors.append("tracked code patch hash mismatch")
    expected_data_sha = canonical_sha256(immutable.get("dataset") or {})
    records: list[dict[str, object]] = []
    fold_samples: dict[tuple[int, int], set[str]] = {}
    per_metric: dict[str, list[float]] = {"macro_f1": [], "top1": [], "weighted_f1": []}

    for fold, seed in sorted(EXPECTED_PAIRS):
        record_path = run_dir / f"run_fold{fold}_seed{seed}.json"
        if not record_path.is_file():
            errors.append(f"missing record {record_path.name}")
            continue
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
        prediction_info = dict(record.get("outer_predictions") or {})
        checkpoint_info = dict(record.get("checkpoint") or {})
        prediction_path = Path(str(prediction_info.get("path") or ""))
        checkpoint_path = Path(str(checkpoint_info.get("path") or ""))
        if not prediction_path.is_file() or sha256_file(prediction_path) != prediction_info.get("sha256"):
            errors.append(f"prediction hash mismatch for fold={fold}, seed={seed}")
            continue
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_info.get("sha256"):
            errors.append(f"checkpoint hash mismatch for fold={fold}, seed={seed}")

        rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
        expected_size = EXPECTED_FOLD_SIZES[fold]
        if len(rows) != expected_size or int(prediction_info.get("num_samples") or 0) != expected_size:
            errors.append(f"row count mismatch for fold={fold}, seed={seed}: {len(rows)}")
        sample_ids = [str(row.get("sample_id")) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            errors.append(f"duplicate sample ids for fold={fold}, seed={seed}")
        if {int(row.get("outer_position")) for row in rows} != set(range(expected_size)):
            errors.append(f"outer positions mismatch for fold={fold}, seed={seed}")

        checkpoint_sha = str(checkpoint_info.get("sha256") or "")
        split_sha = str(record.get("split_sha256") or "")
        for row in rows:
            if not (
                int(row.get("fold")) == fold
                and int(row.get("seed")) == seed
                and row.get("split") == "outer_val"
                and row.get("method") == args.expected_method
                and row.get("config_sha") == fingerprint
                and row.get("code_state_sha") == code_state
                and row.get("checkpoint_sha") == checkpoint_sha
                and row.get("split_sha") == split_sha
                and str(row.get("data_sha")) == expected_data_sha
                and bool(row.get("correct")) == (int(row.get("target")) == int(row.get("prediction")))
            ):
                errors.append(f"row provenance mismatch for fold={fold}, seed={seed}")
                break

        model = dict(record.get("model") or {})
        if bool(model.get("pretrained_requested")) != expect_pretrained or bool(model.get("pretrained_loaded")) != expect_pretrained:
            errors.append(f"pretrained state mismatch for fold={fold}, seed={seed}")
        if record.get("run_fingerprint") != fingerprint:
            errors.append(f"record fingerprint mismatch for fold={fold}, seed={seed}")
        provenance = dict(record.get("provenance") or {})
        if provenance.get("code_state_sha256") != code_state or (provenance.get("source_freeze") or {}).get("verification_status") != "PASS":
            errors.append(f"record provenance mismatch for fold={fold}, seed={seed}")

        targets = [int(row["target"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        recomputed = metric_values(targets, predictions, classes=8)
        reported = dict(record.get("outer_val") or {})
        for key, value in recomputed.items():
            if not close(value, float(reported.get(key))):
                errors.append(f"metric mismatch {key} for fold={fold}, seed={seed}")
            per_metric[key].append(value)
        fold_samples[(fold, seed)] = set(sample_ids)
        records.append({
            "fold": fold,
            "seed": seed,
            "rows": len(rows),
            "prediction_sha256": prediction_info.get("sha256"),
            "checkpoint_sha256": checkpoint_info.get("sha256"),
            **recomputed,
        })

    for fold in range(5):
        sets = [fold_samples.get((fold, seed), set()) for seed in (42, 43, 44)]
        if not sets[0] or not (sets[0] == sets[1] == sets[2]):
            errors.append(f"outer samples differ across seeds for fold={fold}")
    for seed in (42, 43, 44):
        sets = [fold_samples.get((fold, seed), set()) for fold in range(5)]
        if any(left & right for left, right in combinations(sets, 2)):
            errors.append(f"outer folds overlap for seed={seed}")
        if len(set().union(*sets)) != EXPECTED_DATASET_SIZE:
            errors.append(f"outer fold union is not {EXPECTED_DATASET_SIZE} for seed={seed}")

    for key, values in per_metric.items():
        summary_metric = dict(summary.get(f"outer_{key}") or {})
        if len(values) != 15 or int(summary_metric.get("n") or 0) != 15:
            errors.append(f"summary {key} count mismatch")
        elif not close(sum(values) / len(values), float(summary_metric.get("mean"))):
            errors.append(f"summary {key} mean mismatch")

    result = {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir),
        "expected_method": args.expected_method,
        "expect_pretrained": expect_pretrained,
        "run_fingerprint": fingerprint,
        "record_count": len(records),
        "checks": checks,
        "records": records,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "errors": errors}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
