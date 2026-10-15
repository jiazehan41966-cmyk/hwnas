"""Replay three-seed quantization admission from immutable predictions.

This module neither trains nor evaluates a checkpoint.  It independently scores
saved full-valid predictions and verifies their checkpoint/export provenance.
Synthetic receipts can exercise every check but never become formal evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from .contract import digest, sha256_file, validate_identity

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sonar192_confirmation_v1"
SEEDS = (42, 43, 44)
DEPLOYMENT_SEED = 42
METRIC = "coco_101point_AP50_95_max100_per_class"
# Explicitly reviewed trust root, populated only after independent real replay.
# A self-signed receipt, caller flag, or CLI option cannot add an entry. The
# current campaign has no completed epoch80 three-seed PTQ evaluations.
REGISTERED_FORMAL_PTQ_EVALUATIONS = ()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(reference, root):
    """Resolve only an explicitly hash-bound file, never a directory or URL."""
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "Artifact requires exactly path and sha256")
    path = (Path(root) / reference["path"]).resolve()
    require(path.is_file(), "Missing artifact: " + str(path))
    require(sha256_file(path) == reference["sha256"], "Artifact SHA256 mismatch: " + str(path))
    return path


def read_json(reference, root):
    return json.loads(artifact(reference, root).read_text(encoding="utf-8"))


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _validated_records(rows, protocol):
    require(isinstance(rows, list) and len(rows) == protocol["validation_count"],
            "Full validation image count differs")
    images, targets = [], []
    for row in rows:
        require(set(row) == {"image", "prediction", "target"}, "Unexpected prediction record fields")
        image = row["image"].replace("\\", "/")
        require(image.startswith("valid/images/") and ".." not in Path(image).parts,
                "Prediction image must identify valid/images")
        images.append(image)
        for kind in ("prediction", "target"):
            item = row[kind]
            fields = {"boxes", "labels", "scores"} if kind == "prediction" else {"boxes", "labels"}
            require(set(item) == fields, "Unexpected prediction/target fields")
            require(all(isinstance(item[key], list) for key in fields), "Detection values must be lists")
            boxes = np.asarray(item["boxes"], dtype=np.float64)
            require(boxes.shape == (len(item["labels"]), 4) or
                    (len(item["labels"]) == 0 and boxes.shape == (0,)), "Invalid boxes")
            boxes = boxes.reshape(-1, 4)
            require(all(type(label) is int and 0 <= label < protocol["num_classes"]
                        for label in item["labels"]), "Invalid integer class label")
            require(np.isfinite(boxes).all() and (boxes[:, 2:] >= boxes[:, :2]).all(),
                    "Invalid or nonfinite box geometry")
            if kind == "prediction":
                scores = np.asarray(item["scores"], dtype=np.float64)
                require(scores.shape == (len(boxes),) and len(scores) <= 100
                        and np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
                        "Invalid scores or global detection cap")
        targets.append({"image": image, "target": row["target"]})
    require(len(set(images)) == len(images), "Duplicated validation image")
    require(digest(sorted(images)) == protocol["validation_images_sha256"], "Validation image set differs")
    require(digest(sorted(targets, key=lambda row: row["image"])) == protocol["validation_targets_sha256"],
            "Validation ground truth differs")
    return rows


def score_records(rows, protocol):
    """Recompute the frozen COCO metric; never consume an asserted mAP value."""
    _validated_records(rows, protocol)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco
    result = evaluate_detection_predictions_coco(
        [row["prediction"] for row in rows], [row["target"] for row in rows],
        num_classes=protocol["num_classes"], max_detections=100,
        area_ranges={"all": (0., float("inf"))})
    return float(result["coco_ap"])


def validate_protocol(protocol):
    required = {"schema_version", "seeds", "profiles", "metric", "selection_policy",
                "validation_count", "validation_images_sha256", "validation_targets_sha256",
                "num_classes", "absolute_drop", "relative_drop", "improved_seeds", "evidence_kind"}
    require(set(protocol) == required and protocol["schema_version"] == SCHEMA, "Confirmation protocol differs")
    require(protocol["seeds"] == list(SEEDS) and protocol["selection_policy"] == "epoch80_ema"
            and protocol["metric"] == METRIC and protocol["num_classes"] == 2, "Unregistered confirmation semantics")
    require(protocol["profiles"] and protocol["profiles"][0] == "W8A8"
            and len(set(protocol["profiles"])) == len(protocol["profiles"])
            and set(protocol["profiles"]) <= {"W8A8", "HEAD_W8A16", "HEAD_W16A16"}, "Invalid profiles")
    require(protocol["absolute_drop"] == .003 and protocol["relative_drop"] == .20
            and protocol["improved_seeds"] == 2, "Paired admission thresholds differ")
    require(type(protocol["validation_count"]) is int and protocol["validation_count"] > 0,
            "Invalid validation count")
    require(protocol["evidence_kind"] in ("real_validation", "synthetic_diagnostic"), "Unknown evidence kind")
    if protocol["evidence_kind"] == "real_validation":
        require(protocol["validation_count"] == 99, "Formal confirmation requires full99")
    return deepcopy(protocol)


def validate_validation_contract(protocol, manifest_path, checkpoint):
    """Derive full99 image/GT identity from the actual checkpoint's execution manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = checkpoint.get("binding", {})
    require(sha256_file(manifest_path) == binding.get("execution_manifest_sha256"),
            "Validation execution manifest does not belong to checkpoint")
    targets = manifest.get("validation_targets")
    require(isinstance(targets, list) and len(targets) == 99, "Execution manifest must carry frozen full99 targets")
    require(all(set(row) == {"image", "target"} for row in targets), "Execution target schema differs")
    require(digest(targets) == manifest.get("validation_target_sha256") == binding.get("validation_target_sha256"),
            "Execution/EMA validation target hash differs")
    normalized = []
    for row in targets:
        name = row["image"]
        require(isinstance(name, str) and Path(name).name == name and name not in (".", ".."),
                "Frozen validation target must use a basename")
        normalized.append({"image": "valid/images/" + name, "target": row["target"]})
    images = [row["image"] for row in normalized]
    require(len(set(images)) == 99 and protocol["validation_count"] == 99
            and protocol["validation_images_sha256"] == digest(sorted(images))
            and protocol["validation_targets_sha256"] == digest(sorted(normalized, key=lambda row: row["image"])),
            "Confirmation protocol GT/image hashes differ from frozen execution targets")
    return {"execution_manifest_sha256": sha256_file(manifest_path),
            "source_validation_target_sha256": manifest["validation_target_sha256"],
            "normalized_validation_targets_sha256": protocol["validation_targets_sha256"]}


def require_registered_evaluation(identity, receipt, profile_receipt, *, root="."):
    """Require an independently replayed, explicitly admitted evaluation artifact.

    Hash self-consistency alone does not prove quantized weights came from an EMA
    model, or that predictions were produced by that model. This reviewed trust
    root is separate from numerical replay and cannot be enrolled by the runner.
    """
    require(profile_receipt.get("independent_replay") is not None,
            "Formal PTQ evaluation needs an independently reviewed real-replay artifact")
    review_path = artifact(profile_receipt["independent_replay"], root)
    checkpoint = identity["checkpoint"]
    entry = {"space_version": identity["space_version"], "full_contract_sha256": identity["full_contract_sha256"],
             "architecture_sha256": identity["architecture_sha256"], "seed": checkpoint["seed"], "epoch": 80,
             "weights": "ema", "checkpoint_sha256": checkpoint["sha256"],
             "model_state_sha256": checkpoint["model_state_sha256"],
             "calibration_sha256": identity["calibration"].get("calibration_sha256", identity["calibration"].get("sha256")),
             "profile": identity["profile"], "export_manifest_sha256": profile_receipt["export_manifest"]["sha256"],
             "fp32_records_sha256": receipt["fp32_records"]["sha256"],
             "quant_records_sha256": profile_receipt["records"]["sha256"],
             "execution_manifest_sha256": receipt["execution_manifest"]["sha256"],
             "independent_replay_sha256": sha256_file(review_path)}
    require(entry in REGISTERED_FORMAL_PTQ_EVALUATIONS,
            "Formal PTQ/validation evidence is not in the independently reviewed evaluation registry")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    bindings = {key: value for key, value in entry.items() if key != "independent_replay_sha256"}
    require(review.get("schema_version") == "sonar192_independent_evaluation_replay_v1"
            and review.get("evidence_kind") == "independent_real_replay" and review.get("bindings") == bindings
            and review.get("status") == "PASS_REAL_REPLAY" and review.get("test_evaluated") is False,
            "Registered independent evaluation replay report differs")
    return entry


def _checkpoint(receipt, root):
    """Verify actual saved EMA tensor identity, not merely a receipt flag."""
    import torch
    identity = receipt["identity"]
    expected = identity["checkpoint"]
    path = artifact(receipt["checkpoint"], root)
    require(receipt["checkpoint"]["sha256"] == expected["sha256"], "Checkpoint file identity differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state.get("epoch") == 80 and state.get("weights") == "ema"
            and state.get("inference_only") is True and state.get("test_evaluated") is False,
            "Confirmation needs actual epoch80 ema_final.pt, not stable mean or raw weights")
    require(state.get("binding", {}).get("seed") == expected["seed"]
            and state["binding"].get("validation_weights") == "ema"
            and state["binding"].get("test_evaluated") is False, "Checkpoint binding differs")
    from .model import inspect_checkpoint, load_checkpoint
    kwargs = dict(architecture=identity["architecture"], seed=expected["seed"], purpose=identity["purpose"])
    if identity["purpose"] == "formal":
        kwargs.update(execution_manifest=artifact(receipt["execution_manifest"], root),
                      training_result=artifact(receipt["training_result"], root))
        model, actual = load_checkpoint(path, **kwargs)
        del model
    else:
        actual, _ = inspect_checkpoint(path, **kwargs)
    require(actual == expected, "Checkpoint metadata differs from independently loaded EMA evidence")


def confirm_candidates(candidate, seed_receipts, protocol, *, root="."):
    """Return independently recomputed paired admission for exactly three seeds.

    seed_receipts are {path,sha256} references. Each receipt contains identity,
    checkpoint, fp32_records, profiles{q:{records,export_manifest}}, and optional
    stable_epoch_records{"76".."80":reference}. The stable mean is a separately
    named statistic and is never the deployed checkpoint's metric.
    """
    validate_identity(candidate)
    if candidate["purpose"] == "formal":
        require(candidate["checkpoint"]["seed"] == DEPLOYMENT_SEED,
                "Deployment checkpoint must be preregistered seed42 epoch80 EMA; no best-seed selection")
    protocol = validate_protocol(protocol)
    require(candidate["profile"] in protocol["profiles"], "Selected profile is absent from confirmation protocol")
    require(len(seed_receipts) == 3, "Exactly three seed receipts required")
    rows, formal = [], candidate.get("purpose") == "formal"
    for ref in seed_receipts:
        receipt = read_json(ref, root)
        require(receipt.get("schema_version") == SCHEMA and receipt.get("test_evaluated") is False,
                "Unregistered confirmation receipt")
        require(receipt.get("evidence_kind") == protocol["evidence_kind"], "Receipt evidence kind differs")
        identity = receipt["identity"]
        validate_identity(identity)
        for key in ("space_version", "full_contract_sha256", "architecture_sha256", "profile", "mapping", "calibration"):
            require(identity[key] == candidate[key], "Candidate identity differs across seeds: " + key)
        checkpoint = identity["checkpoint"]
        require(checkpoint["seed"] in SEEDS and checkpoint["epoch"] == 80
                and checkpoint["weights"] == "ema" and checkpoint["selection_policy"] == "epoch80_ema",
                "Confirmation cannot use an intermediate or stable-mean checkpoint")
        _checkpoint(receipt, root)
        if identity["purpose"] == "formal":
            validate_validation_contract(protocol, artifact(receipt["execution_manifest"], root), checkpoint)
        require(receipt["metric"] == METRIC and receipt["selection_policy"] == "epoch80_ema",
                "Receipt metric/selection mismatch")
        require(set(receipt["profiles"]) == set(protocol["profiles"]), "Missing required quantization profile")
        require(receipt.get("fp32_evaluation_binding") == {
                    "identity_sha256": identity["identity_sha256"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "records_sha256": receipt["fp32_records"]["sha256"],
                    "weights": "ema", "epoch": 80, "split": "valid"}, "FP32 evaluation binding differs")
        fp = score_records(read_json(receipt["fp32_records"], root), protocol)
        scores = {}
        for profile, item in receipt["profiles"].items():
            manifest_path = artifact(item["export_manifest"], root)
            from .handoff import inspect_export
            expected = item.get("identity", identity)
            validate_identity(expected)
            require(expected["profile"] == profile, "Profile export needs its own bound identity")
            for field in ("checkpoint", "architecture_sha256", "full_contract_sha256", "calibration", "purpose"):
                require(expected[field] == identity[field], "Per-profile identity differs: " + field)
            physical_mapping = lambda mapping: {key: value for key, value in mapping.items()
                                                if key not in ("numeric_contract_sha256", "interface_contract_sha256")}
            require(physical_mapping(expected["mapping"]) == physical_mapping(identity["mapping"]),
                    "Per-profile physical mapping differs")
            if identity["purpose"] == "formal":
                require_registered_evaluation(expected, receipt, item, root=root)
            exported = inspect_export(expected, manifest_path)
            require(exported["profile"]["checkpoint_binding"]["seed"] == checkpoint["seed"], "Export seed differs")
            scores[profile] = score_records(read_json(item["records"], root), protocol)
            binding = item.get("evaluation_binding", {})
            require(binding == {"identity_sha256": expected["identity_sha256"],
                                "checkpoint_sha256": checkpoint["sha256"],
                                "export_manifest_sha256": item["export_manifest"]["sha256"],
                                "records_sha256": item["records"]["sha256"],
                                "weights": "ema", "epoch": 80, "split": "valid"}, "Quantized evaluation binding differs")
        stable_mean = None
        if receipt.get("stable_epoch_records") is not None:
            stable = receipt["stable_epoch_records"]
            require(set(stable) == {str(epoch) for epoch in range(76, 81)}, "Stable statistic needs all five epochs")
            stable_mean = statistics.mean(score_records(read_json(stable[str(epoch)], root), protocol)
                                          for epoch in range(76, 81))
            require(stable["80"] == receipt["fp32_records"], "Stable epoch80 reference differs from selected checkpoint")
        formal = formal and identity.get("purpose") == "formal" and checkpoint.get("formal_eligible") is True
        formal = formal and receipt.get("evidence_kind") == protocol["evidence_kind"] == "real_validation"
        rows.append({"seed": checkpoint["seed"], "identity_sha256": identity["identity_sha256"],
                     "checkpoint_sha256": checkpoint["sha256"], "fp32_epoch80_ema_mAP50_95": fp,
                     "profiles": scores, "fp32_epoch76_80_ema_mean_mAP50_95": stable_mean})
    require(sorted(row["seed"] for row in rows) == list(SEEDS), "Duplicate or missing paired seed")
    require(any(row["identity_sha256"] == candidate["identity_sha256"] for row in rows),
            "Selected deployment checkpoint is not one of the confirmed paired identities")
    rows.sort(key=lambda row: row["seed"])
    points = []
    fp = [row["fp32_epoch80_ema_mAP50_95"] for row in rows]
    uniform = [row["profiles"]["W8A8"] for row in rows]
    for profile in protocol["profiles"]:
        values = [row["profiles"][profile] for row in rows]
        drop = statistics.mean(fp) - statistics.mean(values)
        admitted = drop <= protocol["absolute_drop"] and drop <= protocol["relative_drop"] * statistics.mean(fp)
        if profile != "W8A8":
            admitted = admitted and statistics.mean(values) > statistics.mean(uniform) and sum(
                value > baseline for value, baseline in zip(values, uniform)) >= protocol["improved_seeds"]
        points.append({"profile": profile, "admitted": bool(admitted), "mean_mAP50_95": statistics.mean(values),
                       "std_mAP50_95": statistics.stdev(values), "paired_mean_drop": drop,
                       "formal_eligible": bool(formal and admitted)})
    accepted = next(row["admitted"] for row in points if row["profile"] == candidate["profile"])
    result = {"schema_version": SCHEMA, "status": "PAIRED_ADMISSION_PASS" if accepted else "PAIRED_ADMISSION_REJECTED",
              "formal_eligible": bool(formal and accepted), "candidate_identity_sha256": candidate["identity_sha256"],
              "rows": rows, "points": points, "test_evaluated": False, "training_started": False,
              "request": {"candidate": candidate, "seed_receipts": seed_receipts, "protocol": protocol}}
    result["report_sha256"] = digest(result)
    return result


def replay_confirmation(report, *, root="."):
    saved = deepcopy(report)
    expected = saved.pop("report_sha256")
    require(digest(saved) == expected, "Confirmation report hash differs")
    result = confirm_candidates(**report["request"], root=root)
    require(result == report, "Confirmation does not replay from current evidence")
    return result
