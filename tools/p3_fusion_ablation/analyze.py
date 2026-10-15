"""Analyze only the new paired A1/P3 experiment after fetching its artifacts.

This script never imports or runs the trainer, reads historical A1 results,
accesses test data, or changes a frozen protocol. Missing artifacts produce
PENDING; inconsistent evidence produces INVALID_EVIDENCE, never a gate verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/p3_fusion_ablation_20261003_02"
EXPECTED_PROTOCOL_SHA256 = "ce4c0277e3ed52716edffd4827fbbf7863b8ef0f65c9ca5b208ded8ddc417848"
VARIANTS = ("equal_add", "weighted")
SEEDS = (42, 43, 44)
METRICS = ("mAP50_95", "AP50", "AP75", "Precision", "Recall")
ARCHITECTURE_ID = "P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k5"
GATE_TEXT = "all3_best_deltas_positive AND mean_best_delta>sample_std_best_delta AND mean_epoch80_delta>=0"


class PendingResults(Exception):
    """The fetched archive is not yet complete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value: Any, context: str, *, probability: bool = False) -> float:
    require(not isinstance(value, bool), f"Boolean in numeric field: {context}")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite numeric field: {context}")
    if probability:
        require(0 <= result <= 1, f"Probability outside [0,1]: {context}")
    return result


def close(a: Any, b: Any, context: str) -> None:
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), f"Numeric mismatch: {context}: {a} vs {b}")


def stats(values: list[float]) -> dict[str, Any]:
    require(len(values) == 3, "Final aggregates must contain exactly three seeds")
    values = [number(v, "aggregate") for v in values]
    mean, std = statistics.mean(values), statistics.stdev(values)
    return {"n_seeds": 3, "raw_values_fraction": values, "mean_fraction": mean,
            "sample_std_fraction_ddof1": std, "mean_percent": 100 * mean,
            "sample_std_percent_ddof1": 100 * std}


def gate(best_deltas: list[float], endpoint_deltas: list[float]) -> dict[str, Any]:
    """Apply the exact registered inequalities to unrounded raw differences."""
    best, endpoint = stats(best_deltas), stats(endpoint_deltas)
    all_positive = all(value > 0 for value in best_deltas)
    stable = best["mean_fraction"] > best["sample_std_fraction_ddof1"]
    endpoint_ok = endpoint["mean_fraction"] >= 0
    if all_positive and stable and endpoint_ok:
        verdict = "SUPPORTED"
    elif best["mean_fraction"] > 0:
        verdict = "UNCERTAIN"
    else:
        verdict = "NOT_SUPPORTED"
    return {"verdict": verdict, "rule": GATE_TEXT, "all3_best_deltas_positive": all_positive,
            "mean_best_delta_gt_sample_std_ddof1": stable,
            "mean_epoch80_delta_nonnegative": endpoint_ok,
            "paired_best_deltas_fraction": best_deltas,
            "paired_endpoint_deltas_fraction": endpoint_deltas,
            "mean_best_delta_fraction": best["mean_fraction"],
            "sample_std_best_delta_fraction_ddof1": best["sample_std_fraction_ddof1"],
            "mean_endpoint_delta_fraction": endpoint["mean_fraction"],
            "mean_best_delta_percentage_points": best["mean_percent"],
            "sample_std_best_delta_percentage_points_ddof1": best["sample_std_percent_ddof1"],
            "mean_endpoint_delta_percentage_points": endpoint["mean_percent"],
            "kind": "preregistered descriptive stability rule; not a significance test",
            "rounding_used_for_gate": False}


def state_digest(value: Any, torch: Any) -> str:
    """Reproduce the frozen runner's tensor-aware digest without importing it."""
    h = hashlib.sha256()
    def visit(x: Any) -> None:
        if torch.is_tensor(x):
            tensor = x.detach().cpu().contiguous()
            h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def class_ap(valid: dict[str, Any], thresholds: list[str], context: str) -> dict[str, float]:
    per_class = valid["coco_details"]["per_class_ap"]["all"]
    require(len(per_class) == 2, f"Expected both official classes: {context}")
    result = {}
    for name, scores in per_class.items():
        require(set(scores) == set(thresholds), f"Per-class IoU grid mismatch: {context}/{name}")
        values = [number(scores[key], f"{context}/{name}/{key}", probability=True) for key in thresholds]
        result[name] = statistics.mean(values)
    close(statistics.mean(result.values()), valid["mAP50_95"], f"{context}/macro AP50:95")
    for metric, key in (("AP50", "0.50"), ("AP75", "0.75")):
        close(statistics.mean(scores[key] for scores in per_class.values()), valid[metric], f"{context}/{metric}")
    return result


def protocol_check(protocol: dict[str, Any]) -> None:
    require(protocol["version"] == "A1_P3_scalar_fusion_v2", "Wrong experiment version")
    require(protocol["architecture_id"] == ARCHITECTURE_ID, "Not the fixed-A1 experiment")
    require(tuple(protocol["variants"]) == VARIANTS and tuple(protocol["seeds"]) == SEEDS and protocol["epochs"] == 80,
            "Expected two variants, three seeds and 80 epochs")
    require(protocol["gate"]["supported"] == GATE_TEXT, "Registered gate changed")
    require(protocol["gate"]["uncertain"] == "mean_best_delta>0 but full direction/dispersion/endpoint gate fails"
            and protocol["gate"]["unsupported"] == "mean_best_delta<=0", "Registered fallback gate changed")
    require(protocol["test_accessed"] is False and protocol["board_accessed"] is False, "Evidence scope violation")
    require(protocol["training"]["checkpoint_selection"] == "earliest_best_valid_mAP50_95_within80"
            and protocol["training"]["endpoint"] == 80, "Selection contract changed")
    require(protocol["alpha"]["initial"] == 1 and protocol["alpha"]["weight_decay"] == 0,
            "Alpha initialization/regularization contract changed")
    expected_grid = [round(.5 + .05 * i, 2) for i in range(10)]
    require(protocol["evaluation_protocol"]["iou_thresholds"] == expected_grid, "Expected COCO-style AP50:95")


def load_complete(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], Any], dict[str, Any]]:
    protocol_path = input_dir / "protocol.json"
    if not protocol_path.is_file():
        raise PendingResults(f"Missing fetched protocol: {protocol_path}")
    protocol_hash = sha(protocol_path)
    require(protocol_hash == EXPECTED_PROTOCOL_SHA256, "Protocol differs from the preregistered v2 SHA256")
    protocol = read(protocol_path); protocol_check(protocol)
    required = [input_dir / name for name in ("all_results.json", "execution_review.json", "smoke_witness.json")]
    for variant, seed in ((v, s) for v in VARIANTS for s in SEEDS):
        folder = input_dir / "runs" / variant / f"seed_{seed}"
        required += [folder / name for name in ("result.json", "status.json", "run_binding.json", "epoch_log.json",
                    "best.pt", "latest.pt", "best_valid_predictions.json", "endpoint_valid_predictions.json", "model_info.json")]
    required += [input_dir / "initializations" / f"seed_{seed}.pt" for seed in SEEDS]
    missing = [str(path.relative_to(input_dir)) for path in required if not path.is_file()]
    if missing:
        raise PendingResults("Missing completed-run artifacts: " + ", ".join(missing))
    combined = read(input_dir / "all_results.json")
    if combined.get("status") != "COMPLETE":
        raise PendingResults(f"Fetched campaign status is {combined.get('status')!r}")
    require(combined["test_accessed"] is False and len(combined["runs"]) == 6, "Invalid all_results scope/count")
    identity = [(row["variant"], row["seed"]) for row in combined["runs"]]
    require(len(set(identity)) == 6 and set(identity) == {(v, s) for v in VARIANTS for s in SEEDS}, "Wrong run identities")
    review = read(input_dir / "execution_review.json")
    require(review["status"] == "PASS" and review["protocol_sha256"] == protocol_hash, "Execution review not protocol-bound PASS")
    smoke = read(input_dir / "smoke_witness.json")
    require(smoke["status"] == "PASS" and smoke["protocol_sha256"] == protocol_hash
            and smoke["forward_exact"] is True, "Smoke witness not protocol-bound PASS")
    manifest_path = ROOT / "reports/baseline_sonar_frozen/freeze_manifest.json"
    require(sha(manifest_path) == protocol["dataset"]["original_manifest_sha256"], "Original frozen dataset manifest SHA mismatch")
    manifest = read(manifest_path)
    frozen_valid = manifest["datasets"]["valid"]
    expected_images = [entry["image"].replace("\\", "/") for entry in frozen_valid["entries"]]
    expected_classes = set(frozen_valid["class_names"])
    require(len(expected_images) == len(set(expected_images)) == protocol["dataset"]["valid"] == 99,
            "Expected 99 unique frozen validation images")
    require(manifest["datasets"]["train"]["images"] == protocol["dataset"]["train"], "Frozen training coverage mismatch")
    del manifest
    import torch  # Only decode this experiment's trusted checkpoints after completeness checks.
    checks: dict[str, Any] = {"protocol_sha256": protocol_hash, "execution_review_sha256": sha(input_dir / "execution_review.json"),
                             "smoke_witness_sha256": sha(input_dir / "smoke_witness.json"),
                             "all_results_sha256": sha(input_dir / "all_results.json"), "runs": [], "pairing": []}
    initializer_hashes = {}
    for seed in SEEDS:
        path = input_dir / "initializations" / f"seed_{seed}.pt"
        initial = torch.load(path, map_location="cpu", weights_only=False)
        require(initial["seed"] == seed, f"Initialization seed mismatch: {seed}")
        shared = {name: value for name, value in initial["model_state_dict"].items() if name != "neck.alpha"}
        initializer_hashes[seed] = state_digest(shared, torch)
        require(initial["shared_sha256"] == initializer_hashes[seed], f"Initialization tensor digest mismatch: {seed}")
        close(float(initial["model_state_dict"]["neck.alpha"]), 1, f"Initial alpha seed{seed}")
        del initial, shared
    require(len(set(initializer_hashes.values())) == 3, "The three seeds do not have distinct initial shared states")
    histories = {}; validated = []; reference_targets = None
    thresholds = [f"{value:.2f}" for value in protocol["evaluation_protocol"]["iou_thresholds"]]
    n_batches = math.ceil(protocol["dataset"]["train"] / protocol["training"]["batch_size"])
    for row in sorted(combined["runs"], key=lambda item: (item["variant"], item["seed"])):
        variant, seed = row["variant"], row["seed"]
        context = f"{variant}/seed{seed}"
        folder = input_dir / "runs" / variant / f"seed_{seed}"
        require(row == read(folder / "result.json"), f"all_results/result.json differ: {context}")
        require(row["test_accessed"] is False, f"Test access: {context}")
        status, binding = read(folder / "status.json"), read(folder / "run_binding.json")
        expected_binding = {"protocol_sha256": protocol_hash, "architecture_id": ARCHITECTURE_ID,
                            "variant": variant, "seed": seed, "epochs": 80, "test_accessed": False}
        require(status["status"] == "COMPLETE" and status["epochs"] == 80 and status["binding"] == expected_binding
                and status["test_accessed"] is False, f"Incomplete or wrongly bound status: {context}")
        require(status["result_sha256"] == sha(folder / "result.json"), f"Result status SHA mismatch: {context}")
        require(all(binding.get(key) == value for key, value in expected_binding.items()), f"run_binding mismatch: {context}")
        history = read(folder / "epoch_log.json")
        require([entry["epoch"] for entry in history] == list(range(1, 81)), f"Not a complete 80-epoch history: {context}")
        require(sha(folder / "epoch_log.json") == row["history_sha256"], f"History SHA mismatch: {context}")
        for entry in history:
            alpha = number(entry["alpha"], f"{context}/epoch{entry['epoch']}/alpha")
            require(alpha >= 0 and (variant != "equal_add" or alpha == 1), f"Alpha constraint violated: {context}")
            require(entry["batches"] == n_batches and entry["gradient_finite"] is True, f"Training coverage/gradients: {context}")
            close(entry["lr"], protocol["training"]["lr"], f"{context}/learning rate")
            for name, value in entry["losses"].items():
                number(value, f"{context}/epoch{entry['epoch']}/loss/{name}")
            valid = entry["valid"]
            require(valid["num_images"] == protocol["dataset"]["valid"]
                    and valid["protocol"] == protocol["evaluation_protocol"], f"Validation coverage/protocol: {context}")
            for metric in METRICS:
                number(valid[metric], f"{context}/epoch{entry['epoch']}/{metric}", probability=True)
            class_ap(valid, thresholds, f"{context}/epoch{entry['epoch']}")
            require(set(valid["coco_details"]["per_class_ap"]["all"]) == expected_classes,
                    f"Class names differ from the frozen validation split: {context}")
        # Python max returns the first tied maximum: the frozen earliest-epoch rule.
        best = max(history, key=lambda entry: entry["valid"]["mAP50_95"])
        require(row["best_epoch"] == best["epoch"], f"Wrong earliest-best selection: {context}")
        sidecar_predictions = {}
        for slot, entry in (("best", best), ("endpoint", history[-1])):
            for metric in METRICS:
                close(row[f"{slot}_valid"][metric], entry["valid"][metric], f"{context}/{slot}/{metric}")
            close(row[f"{slot}_alpha"], entry["alpha"], f"{context}/{slot}/alpha")
            computed_classes = class_ap(entry["valid"], thresholds, f"{context}/{slot}")
            require(set(computed_classes) == set(row[f"{slot}_class_AP"]), f"Class identity mismatch: {context}")
            for name, value in computed_classes.items():
                close(value, row[f"{slot}_class_AP"][name], f"{context}/{slot}/{name}")
            pred_path = folder / f"{slot if slot == 'best' else 'endpoint'}_valid_predictions.json"
            require(sha(pred_path) == row[f"{slot}_prediction_sha256"], f"Prediction SHA mismatch: {context}/{slot}")
            predictions = read(pred_path)
            sidecar_predictions[slot] = predictions
            require(len(predictions) == protocol["dataset"]["valid"], f"Prediction coverage mismatch: {context}")
            require([item["image"].replace("\\", "/") for item in predictions] == expected_images,
                    f"Prediction image identity/order differs from the frozen validation split: {context}/{slot}")
            targets = [(item["image"], item["target"]) for item in predictions]
            if reference_targets is None:
                reference_targets = targets
            require(targets == reference_targets, f"GT/order differs across arms/checkpoints: {context}")
        for name in ("best", "latest"):
            require(sha(folder / f"{name}.pt") == row[f"{name}_checkpoint_sha256"], f"Checkpoint SHA mismatch: {context}/{name}")
        latest = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
        snapshot = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        schema = protocol["checkpoint_transaction"]["schema"]
        require(latest["commit_schema"] == schema and latest["purpose"] == "training_resume"
                and latest["test_accessed"] is False, f"Wrong authoritative commit schema/scope: {context}")
        require(snapshot["snapshot_schema"] == schema and snapshot["purpose"] == "inference_only_not_resume"
                and snapshot["test_accessed"] is False, f"Wrong inference snapshot schema/scope: {context}")
        require(not any(key in snapshot for key in ("optimizer_state_dict", "rng", "history")),
                f"Best inference snapshot carries resume state: {context}")
        committed_digest = state_digest({key: value for key, value in latest.items() if key != "commit_sha256"}, torch)
        require(latest["commit_sha256"] == committed_digest == status["commit_sha256"],
                f"Authoritative transaction/status digest mismatch: {context}")
        require(status["latest_checkpoint_sha256"] == row["latest_checkpoint_sha256"],
                f"Latest checkpoint/status SHA mismatch: {context}")
        require(latest["binding"] == expected_binding and latest["epoch"] == 80 and latest["history"] == history,
                f"Authoritative latest checkpoint/history mismatch: {context}")
        require(latest["history_sha256"] == state_digest(history, torch), f"Committed history digest mismatch: {context}")
        require(snapshot["binding"] == expected_binding and snapshot["epoch"] == best["epoch"],
                f"Best inference snapshot identity mismatch: {context}")
        require(snapshot["valid"] == best["valid"], f"Best snapshot valid metrics differ: {context}")
        close(snapshot["alpha"], best["alpha"], f"{context}/snapshot alpha")
        require(state_digest(latest["best_snapshot"], torch) == state_digest(snapshot, torch),
                f"best.pt differs from latest's complete committed inference snapshot: {context}")
        for slot, committed in (("best", snapshot), ("endpoint", latest)):
            require(committed["model_state_sha256"] == state_digest(committed["model_state_dict"], torch),
                    f"Committed model digest mismatch: {context}/{slot}")
            require(committed["validation_predictions_sha256"] == state_digest(committed["validation_predictions"], torch),
                    f"Committed prediction digest mismatch: {context}/{slot}")
            require(sidecar_predictions[slot] == committed["validation_predictions"],
                    f"Prediction sidecar differs from authoritative checkpoint: {context}/{slot}")
        close(float(snapshot["model_state_dict"]["neck.alpha"]), best["alpha"], f"{context}/best state alpha")
        close(float(latest["model_state_dict"]["neck.alpha"]), history[-1]["alpha"], f"{context}/endpoint state alpha")
        require(latest["initial_shared_sha256"] == row["initial_shared_sha256"] == initializer_hashes[seed]
                == binding["initial_shared_sha256"], f"Shared initialization mismatch: {context}")
        require(latest["train_start_rng_sha256"] == row["train_start_rng_sha256"] == binding["train_start_rng_sha256"],
                f"Training-start RNG metadata mismatch: {context}")
        require(snapshot["initial_shared_sha256"] == latest["initial_shared_sha256"]
                and snapshot["train_start_rng_sha256"] == latest["train_start_rng_sha256"],
                f"Best inference snapshot initialization/RNG mismatch: {context}")
        require(set(latest["rng"]) == {"python", "numpy", "torch", "cuda", "sampler_generator", "loader_generator"},
                f"Committed resume RNG fields differ: {context}")
        require(state_digest(latest["optimizer_state_dict"], torch) == latest["optimizer_state_sha256"],
                f"Optimizer-state digest mismatch: {context}")
        groups = latest["optimizer_state_dict"]["param_groups"]
        require(len(groups) == (2 if variant == "weighted" else 1), f"Optimizer group count mismatch: {context}")
        for group_index, group in enumerate(groups):
            require(group["name"] == ("ordinary_model" if group_index == 0 else "p3_alpha"),
                    f"Optimizer parameter-group identity mismatch: {context}")
            close(group["lr"], protocol["training"]["lr"], f"{context}/optimizer lr")
            close(group["weight_decay"], .0001 if group_index == 0 else 0, f"{context}/optimizer decay")
            require(tuple(group["betas"]) == (.9, .999), f"Optimizer betas mismatch: {context}")
            close(group["eps"], 1e-8, f"{context}/optimizer eps")
        if variant == "weighted":
            require(len(groups[1]["params"]) == 1, f"Expected one alpha optimizer parameter: {context}")
        model_info = read(folder / "model_info.json")
        require(model_info == latest["model_info"], f"Model-info export differs from authoritative commit: {context}")
        require(row["params"] == model_info["total_params"] and row["conv_linear_MACs"] == model_info["estimated_macs"],
                f"Model info differs: {context}")
        require(row["added_elementwise_multiplications"] == (614400 if variant == "weighted" else 0),
                f"Added scalar-multiplication count differs: {context}")
        close(row["training_seconds"], sum(entry["wall_seconds"] for entry in history), f"{context}/training seconds")
        checks["runs"].append({"variant": variant, "seed": seed, "epochs_verified": 80,
                              "earliest_best_epoch_verified": best["epoch"], "run_binding_verified": True,
                              "checkpoint_sha256_verified": True, "checkpoint_metadata_verified": True,
                              "authoritative_commit_digest_and_status_verified": True,
                              "model_and_optimizer_internal_digests_verified": True,
                              "complete_best_snapshot_binding_verified": True,
                              "history_sha256_verified": True, "prediction_sha256_and_gt_order_verified": True,
                              "prediction_sidecars_equal_committed_payloads": True,
                              "frozen_validation_image_identity_and_order_verified": True,
                              "resume_rng_payload_bound_to_commit": True,
                              "initial_tensor_digest_verified": True,
                              "initialization_file_sha256": sha(input_dir / "initializations" / f"seed_{seed}.pt")})
        histories[variant, seed] = history; validated.append(row)
        del latest, snapshot
    by_id = {(row["variant"], row["seed"]): row for row in validated}
    for seed in SEEDS:
        a, b = by_id["equal_add", seed], by_id["weighted", seed]
        require(a["initial_shared_sha256"] == b["initial_shared_sha256"], f"Unequal initial states: seed{seed}")
        require(a["train_start_rng_sha256"] == b["train_start_rng_sha256"], f"Unequal initial RNG states: seed{seed}")
        a_binding = read(input_dir / "runs/equal_add" / f"seed_{seed}/run_binding.json")
        b_binding = read(input_dir / "runs/weighted" / f"seed_{seed}/run_binding.json")
        require(a_binding["admission"]["cuda_visible_devices"] == b_binding["admission"]["cuda_visible_devices"],
                f"Seed pair ran on different GPU assignments: {seed}")
        require(b["params"] == a["params"] + 1 and b["conv_linear_MACs"] == a["conv_linear_MACs"],
                f"Expected exactly one added scalar, no convolution change: seed{seed}")
        checks["pairing"].append({"seed": seed, "initial_shared_sha256": a["initial_shared_sha256"],
                                  "train_start_rng_sha256": a["train_start_rng_sha256"],
                                  "same_initial_weights": True, "same_training_start_rng_metadata": True,
                                  "same_physical_gpu_assignment": True})
    checks["epochs_total_verified"] = 480
    checks["bitwise_training_determinism_claimed"] = False
    checks["prediction_metrics_recomputed_from_boxes"] = False
    checks["metric_verification"] = "history/selected epoch/per-class ten-IoU averages; full committed checkpoint digests, prediction sidecar-to-checkpoint payload identity and GT/order verified"
    return protocol, validated, histories, checks


def build_analysis(protocol: dict[str, Any], runs: list[dict[str, Any]], histories: dict[Any, Any], checks: dict[str, Any]) -> dict[str, Any]:
    by_id = {(row["variant"], row["seed"]): row for row in runs}
    classes = sorted(runs[0]["best_class_AP"])
    pairs = []
    for seed in SEEDS:
        a, b = by_id["equal_add", seed], by_id["weighted", seed]
        pair = {"seed": seed, "delta_direction": "weighted minus fresh equal_add", "best_epoch": {v: by_id[v, seed]["best_epoch"] for v in VARIANTS}}
        for slot in ("best", "endpoint"):
            raw = {metric: b[f"{slot}_valid"][metric] - a[f"{slot}_valid"][metric] for metric in METRICS}
            pair[f"{slot}_deltas_fraction"] = raw
            pair[f"{slot}_deltas_percentage_points"] = {metric: 100 * value for metric, value in raw.items()}
            cp = {name: b[f"{slot}_class_AP"][name] - a[f"{slot}_class_AP"][name] for name in classes}
            pair[f"{slot}_class_AP_deltas_fraction"] = cp
            pair[f"{slot}_class_AP_deltas_percentage_points"] = {name: 100 * value for name, value in cp.items()}
        pairs.append(pair)
    aggregates = {}
    for variant in VARIANTS:
        aggregates[variant] = {}
        for slot in ("best", "endpoint"):
            aggregates[variant][f"{slot}_valid"] = {metric: stats([by_id[variant, seed][f"{slot}_valid"][metric] for seed in SEEDS]) for metric in METRICS}
            aggregates[variant][f"{slot}_class_AP"] = {name: stats([by_id[variant, seed][f"{slot}_class_AP"][name] for seed in SEEDS]) for name in classes}
    paired_aggregates = {slot: {metric: stats([pair[f"{slot}_deltas_fraction"][metric] for pair in pairs]) for metric in METRICS} for slot in ("best", "endpoint")}
    paired_classes = {slot: {name: stats([pair[f"{slot}_class_AP_deltas_fraction"][name] for pair in pairs]) for name in classes} for slot in ("best", "endpoint")}
    verdict = gate([pair["best_deltas_fraction"]["mAP50_95"] for pair in pairs], [pair["endpoint_deltas_fraction"]["mAP50_95"] for pair in pairs])
    trajectories = [{"variant": variant, "seed": seed, "initial": 1,
                     "epochs": [0] + [entry["epoch"] for entry in histories[variant, seed]],
                     "alpha": [1.] + [entry["alpha"] for entry in histories[variant, seed]],
                     "best_epoch": by_id[variant, seed]["best_epoch"],
                     "best_alpha": by_id[variant, seed]["best_alpha"],
                     "endpoint_alpha": by_id[variant, seed]["endpoint_alpha"]}
                    for variant in VARIANTS for seed in SEEDS]
    return {"status": "COMPLETE", "gate": verdict, "protocol": protocol, "validation": checks,
            "runs": runs, "paired_by_seed": pairs, "aggregates": aggregates,
            "paired_aggregates": paired_aggregates, "paired_per_class_aggregates": paired_classes,
            "alpha_trajectories": trajectories,
            "units": {"raw_metrics": "fraction in [0,1]", "table_metrics": "percent = fraction*100",
                      "delta": "percentage points = difference of fractions*100", "std": "sample standard deviation across three seeds, ddof1"},
            "limits": ["One official train/valid split; 99-image validation set reused for checkpoint selection and reporting.",
                       "Three seeds and a preregistered descriptive gate; no significance or confidence-interval claim.",
                       "Both arms freshly train ordinary weights; alpha behavior does not isolate a fixed-checkpoint causal mechanism.",
                       "Historical A1 results are not read or included in any pair or aggregate.",
                       "Alpha is one global nonnegative scalar at P3, not a spatial attention map or a class-specific weight.",
                       "Conv/Linear MACs omit scalar fusion arithmetic; added 614400 multiplications/image are separately recorded.",
                       "No test, board latency, power, resource-use or hardware-efficiency conclusion."],
            "training_launched_by_analysis": False, "test_accessed": False, "board_accessed": False}


def render_curves(out: Path, histories: dict[Any, Any], runs: list[dict[str, Any]]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_id = {(row["variant"], row["seed"]): row for row in runs}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.1), sharex="col", sharey="row")
    for column, seed in enumerate(SEEDS):
        for variant, color in (("equal_add", "#0072B2"), ("weighted", "#D55E00")):
            h = histories[variant, seed]; epochs = [entry["epoch"] for entry in h]
            axes[0, column].plot(epochs, [100 * entry["valid"]["mAP50_95"] for entry in h], color=color, linewidth=1.2, label=variant)
            row = by_id[variant, seed]
            axes[0, column].plot(row["best_epoch"], 100 * row["best_valid"]["mAP50_95"], "*", color=color, markersize=8)
            axes[1, column].plot([0] + epochs, [1] + [entry["alpha"] for entry in h], color=color,
                                  linestyle="--" if variant == "equal_add" else "-", linewidth=1.3, label=variant)
            axes[1, column].plot(row["best_epoch"], row["best_alpha"], "*", color=color, markersize=8)
        axes[0, column].set_title(f"Seed {seed}")
        axes[1, column].set_xlabel("Epoch")
        for ax in axes[:, column]:
            ax.set_xlim(-2, 82); ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Valid mAP50:95 (%)")
    axes[1, 0].set_ylabel("P3 deep-branch scalar alpha")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fresh matched A1/P3 fusion runs | equal add versus one learned scalar", fontsize=11)
    fig.text(.5, .02, "Stars: independently selected earliest best-valid checkpoint. Alpha starts at 1; ordinary weights train in both arms.\n"
             "One train/valid split, three seeds; descriptive comparison, no significance, test or hardware claim.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    paths = [out / "p3_fusion_curves.png", out / "p3_fusion_curves.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Creator": "p3_fusion_ablation/analyze.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return paths


def mean_std_text(summary: dict[str, Any]) -> str:
    return f"{summary['mean_percent']:.4f} ± {summary['sample_std_percent_ddof1']:.4f}"


def report(result: dict[str, Any]) -> str:
    g = result["gate"]
    lines = ["# 固定 A1/P3 融合配对实验分析", "", f"预登记判定：**{g['verdict']}**。六次新运行均完成 80 个 epoch；数据与协议核验通过。", "",
             "控制组为本次重新训练的 `equal_add`，处理组为单个非负可学习标量 `weighted`；历史 A1 数值未纳入。",
             "指标以百分数报告，差值以百分点报告。± 为三个随机种子的样本标准差（ddof=1），不是置信区间。", "",
             "## 预登记判定", "", "严格使用未四舍五入的 weighted − equal_add 差值：", "",
             f"- 三个 best-valid 差值均 > 0：{g['all3_best_deltas_positive']}。",
             f"- best-valid 平均差值 > 配对差值样本标准差：{g['mean_best_delta_gt_sample_std_ddof1']}。",
             f"- epoch80 平均差值 ≥ 0：{g['mean_epoch80_delta_nonnegative']}。",
             f"- best-valid 平均差值：{g['mean_best_delta_percentage_points']:+.6f} 个百分点；样本标准差：{g['sample_std_best_delta_percentage_points_ddof1']:.6f} 个百分点。",
             f"- epoch80 平均差值：{g['mean_endpoint_delta_percentage_points']:+.6f} 个百分点。", "",
             "这是冻结前声明的描述性稳定性规则，不是显著性检验；未追加 ERF、alpha 或其他结果作为过门条件。", ""]
    for slot, title in (("best", "各运行 best-valid 检查点"), ("endpoint", "固定 epoch80 端点")):
        lines += [f"## {title}", "", "best 检查点按 valid mAP50:95 最大值选择，完全相同得分取最早 epoch；其他指标均来自同一检查点。" if slot == "best" else "端点指标固定取第 80 个 epoch，不再次择优。", "",
                  "| 组别 | mAP50:95 (%) | AP50 (%) | AP75 (%) | Precision (%) | Recall (%) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            summaries = result["aggregates"][variant][f"{slot}_valid"]
            lines.append("| " + variant + " | " + " | ".join(mean_std_text(summaries[metric]) for metric in METRICS) + " |")
        lines += ["", "配对差值（weighted − equal_add，百分点）：", "",
                  "| Seed | mAP50:95 | AP50 | AP75 | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
        for pair in result["paired_by_seed"]:
            lines.append(f"| {pair['seed']} | " + " | ".join(f"{pair[f'{slot}_deltas_percentage_points'][metric]:+.6f}" for metric in METRICS) + " |")
        lines += ["", "| 类别 AP50:95 (%) | equal_add | weighted | 配对平均差值 ± 标准差（百分点） |", "|---|---:|---:|---:|"]
        classes = result["aggregates"]["equal_add"][f"{slot}_class_AP"]
        for name in classes:
            a = result["aggregates"]["equal_add"][f"{slot}_class_AP"][name]
            b = result["aggregates"]["weighted"][f"{slot}_class_AP"][name]
            delta = result["paired_per_class_aggregates"][slot][name]
            lines.append(f"| {name} | {mean_std_text(a)} | {mean_std_text(b)} | {delta['mean_percent']:+.6f} ± {delta['sample_std_percent_ddof1']:.6f} |")
        lines.append("")
    lines += ["## 原始运行与 alpha", "", "| 组别 | Seed | Best epoch | Best mAP (%) | Epoch80 mAP (%) | Best alpha | Epoch80 alpha |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['best_epoch']} | {100*row['best_valid']['mAP50_95']:.6f} | {100*row['endpoint_valid']['mAP50_95']:.6f} | {row['best_alpha']:.6f} | {row['endpoint_alpha']:.6f} |")
    lines += ["", "alpha 从 1 初始化，每次优化器更新后投影到非负范围；alpha 独立参数组不施加 weight decay，普通参数保持 0.0001。",
              "alpha 是 P3 深层支路的全局标量；其终值与变化不能直接解释为空间注意力、背景抑制或机制因果证明。完整逐 epoch 轨迹保存在 paired_results.json 与曲线图中。", "",
              "## 证据核验与局限", "", f"协议 SHA256：`{result['validation']['protocol_sha256']}`。", "",
              "六份完整历史、最早 best 选点、epoch80、各类十个 IoU 阈值平均、初始共享张量 digest、配对训练起始 RNG 哈希、完整事务 digest、检查点和预测文件 SHA 均已核验；预测导出内容与检查点内提交内容一致。",
              "best.pt 支持最小推理快照，不要求其携带 optimizer 或 history；latest.pt 的完整历史是权威提交状态。",
              "输入 RNG 一致按运行与检查点保存的起始哈希核对；不声称 CUDA 训练逐位确定。预测文件已核验字节与 GT/顺序，未在本脚本从框预测重新计算完整 AP。", ""]
    for limitation in result["limits"]:
        lines.append("- " + limitation)
    lines += ["", "Precision/Recall 在冻结的置信度 0.5、IoU 0.5 下按 micro 口径计算。99 张 valid 同时用于选点与报告，因此不能代替封存 test 的泛化评价。",
              "增加一个参数和每图 614400 次逐元素乘法仅描述软件结构；未测量 FPGA latency、资源或功耗，不据此宣称硬件收益。", "",
              "![本次六个运行的验证曲线与 alpha 轨迹](p3_fusion_curves.png)", ""]
    return "\n".join(lines)


def self_check() -> dict[str, Any]:
    require(gate([.01, .02, .025], [0, 0, 0])["verdict"] == "SUPPORTED", "Supported gate check")
    require(gate([.01, .02, -.001], [0, 0, 0])["verdict"] == "UNCERTAIN", "Mixed-direction gate check")
    require(gate([.001, .002, .04], [0, 0, 0])["verdict"] == "UNCERTAIN", "Dispersion gate check")
    require(gate([.01, .02, .025], [-.01, 0, 0])["verdict"] == "UNCERTAIN", "Endpoint gate check")
    require(gate([-.001, 0, .001], [0, 0, 0])["verdict"] == "NOT_SUPPORTED", "Nonpositive mean check")
    require(gate([1e-15, 1e-15, 1e-15], [0, 0, 0])["verdict"] == "SUPPORTED", "Unrounded inequalities check")
    require(stats([.01, .02, .03])["sample_std_fraction_ddof1"] == .01, "Sample std ddof1 check")
    require(max([{"epoch": 1, "ap": .2}, {"epoch": 2, "ap": .2}], key=lambda row: row["ap"])["epoch"] == 1, "Earliest tie check")
    return {"status": "PASS", "checks": "six gate cases, sample std ddof1 and earliest tie; synthetic cases are in-memory tests only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUT / "remote_results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2)); return
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    code = 0
    try:
        protocol, runs, histories, checks = load_complete(args.input.resolve())
        result = build_analysis(protocol, runs, histories, checks)
        paths = render_curves(out, histories, runs)
        result["figure_sha256"] = {path.name: sha(path) for path in paths}
        text = report(result)
    except PendingResults as error:
        result = {"status": "PENDING", "gate": {"verdict": "NOT_EVALUATED"}, "reason": str(error),
                  "input_directory": str(args.input.resolve()), "figures_generated": False,
                  "training_launched_by_analysis": False, "test_accessed": False, "board_accessed": False}
        text = "# P3 融合实验分析：PENDING\n\n六个运行的完整证据尚未齐全；未计算正式配对结论或通过判定。\n\n" + str(error) + "\n\n未生成正式结果曲线；本脚本不等待、不训练、不访问 test。\n"
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        code = 2
        result = {"status": "INVALID_EVIDENCE", "gate": {"verdict": "NOT_EVALUATED"}, "reason": str(error),
                  "input_directory": str(args.input.resolve()), "figures_generated": False,
                  "training_launched_by_analysis": False, "test_accessed": False, "board_accessed": False}
        text = "# P3 融合实验分析：证据校验未通过\n\n未给出效果结论。\n\n" + str(error) + "\n"
    result["analysis_source_sha256"] = sha(Path(__file__))
    (out / "paired_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "analysis-report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]["verdict"], "output_directory": str(out)}, indent=2))
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
