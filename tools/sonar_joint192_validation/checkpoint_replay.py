"""Read-only final checkpoint inference replay after all 12 formal runs finish.

Use the original pinned CUDA interpreter. Recompute all 99 valid predictions
from every best.pt with the frozen evaluator; never train, access test, call
make_run/summarize, or rewrite original campaign files. Only a new independent
output namespace is written. Comparison tolerances are fixed before execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.sonar_joint192_validation import train
from tools.sonar_joint192_validation.model import build_model, validate_architecture
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

POLICY = dict(version="joint192_checkpoint_replay_fixed_absolute_tolerance_v1",
              boxes_absolute_tolerance_input_pixels=1e-4,
              scores_absolute_tolerance=1e-6,
              metrics_absolute_tolerance=1e-8,
              relative_tolerance=0.,
              image_order="exact_frozen_valid_order_99_unique_images",
              prediction_order="exact_no_reordering_or_matching",
              counts_labels_ground_truth="exact",
              exact_bit_pattern_scope="float64_conversion_of_JSON_numeric_values",
              changed_thresholds=False)
COUNT_METRICS = ("num_images", "true_positives", "false_positives", "ground_truths")
RUN_FILES = ("best.pt", "latest.pt", "epoch_log.json", "status.json", "result.json",
             "initialization_identity.json", "run_binding.json", "model_info.json",
             "best_valid_records.json", "best_valid_metrics.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_normalize(value):
    """Compare checkpoint tuples against their intentional JSON list encoding."""
    return json.loads(json.dumps(value, allow_nan=False))


def write_new(path, value):
    """Never overwrite differing evidence, including a prior failed attempt."""
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if path.exists():
        require(path.read_bytes() == encoded, f"Existing evidence differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def inventory(result_root):
    paths = [result_root / "execution_manifest.json"]
    paths += [result_root / f"worker_gpu{gpu}/queue_status.json" for gpu in (2, 3)]
    for seed in train.SEEDS:
        for arm in train.PROTOTYPES:
            paths.extend(train.result_dir(result_root, arm, seed) / name for name in RUN_FILES)
    return {p.relative_to(result_root).as_posix(): train.sha(p) for p in paths}


def complete_gate(result_root, cfg, manifest_sha, artifacts):
    """Both workers must be COMPLETE, with their exact six unique bound runs."""
    rows = []
    for gpu in (2, 3):
        worker = read_json(result_root / f"worker_gpu{gpu}/queue_status.json")
        expected = {(arm, seed) for seed in train.SEEDS for arm in cfg["staging"]["workers"][gpu]}
        completed = worker.get("completed", [])
        observed = [(r["arm"], r["seed"]) for r in completed]
        require(worker.get("status") == "COMPLETE" and worker.get("gpu") == gpu,
                f"Worker GPU{gpu} must be COMPLETE on its assigned GPU")
        require(worker.get("execution_manifest_sha256") == manifest_sha,
                f"Worker GPU{gpu} manifest mismatch")
        require(worker.get("active") is None and worker.get("child_pid") is None,
                f"Worker GPU{gpu} still has an active child")
        require(worker.get("test_evaluated") is False, f"Worker GPU{gpu} test seal mismatch")
        require(len(completed) == 6 and len(set(observed)) == 6 and set(observed) == expected,
                f"Worker GPU{gpu} must contain exactly its six registered completed runs")
        for row in completed:
            rel = f"formal/{row['arm']}/seed_{row['seed']}/result.json"
            require(row.get("result_sha256") == artifacts[rel], f"Worker result SHA mismatch: {rel}")
        rows.append(dict(gpu=gpu, completed=deepcopy(completed), status="COMPLETE"))
    require(len({(r["arm"], r["seed"]) for w in rows for r in w["completed"]}) == 12,
            "The two workers do not cover all twelve unique registered runs")
    return rows


def validate_records(records, expected_images):
    require(isinstance(records, list) and len(records) == 99, "Require exactly 99 prediction records")
    images = [r["image"] for r in records]
    require(images == expected_images and len(set(images)) == 99, "Valid image identity/order mismatch")
    for record in records:
        require(set(record) == {"image", "prediction", "target"}, "Unexpected prediction-record fields")
        for kind in ("prediction", "target"):
            values = record[kind]
            fields = {"boxes", "scores", "labels"} if kind == "prediction" else {"boxes", "labels"}
            require(set(values) == fields, f"Unexpected {kind} fields")
            require(all(isinstance(values[k], list) for k in fields), f"{kind} fields must be lists")
            require(len(values["boxes"]) == len(values["labels"]), f"{kind} box/label count mismatch")
            require(all(type(label) is int and label in (0, 1) for label in values["labels"]),
                    f"{kind} labels must be exact registered integer IDs")
            boxes = values["boxes"]
            require(all(isinstance(box, list) and len(box) == 4 for box in boxes), "Box shape mismatch")
            require(np.isfinite(np.asarray(boxes, dtype=np.float64)).all(), "Nonfinite boxes")
            if kind == "prediction":
                require(len(values["scores"]) == len(boxes), "Prediction score/box count mismatch")
                scores = np.asarray(values["scores"], dtype=np.float64)
                require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
                        "Prediction scores must be finite and in [0,1]")


def array_comparison(actual, expected, tolerance):
    left, right = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        return dict(shape_equal=False, actual_shape=list(left.shape), expected_shape=list(right.shape),
                    exact_equal=False, bit_pattern_equal=False, within_tolerance=False,
                    maximum_absolute_difference=None)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    maximum = float(np.max(np.abs(left - right))) if left.size and finite else (0. if finite else None)
    return dict(shape_equal=True, finite=finite, exact_equal=bool(np.array_equal(left, right)),
                bit_pattern_equal=left.tobytes() == right.tobytes(),
                within_tolerance=finite and maximum <= tolerance,
                maximum_absolute_difference=maximum)


def compare_records(actual, expected, expected_images):
    validate_records(actual, expected_images)
    validate_records(expected, expected_images)
    comparisons = []
    for left, right in zip(actual, expected):
        target = dict(labels_equal=left["target"]["labels"] == right["target"]["labels"],
                      boxes=array_comparison(left["target"]["boxes"], right["target"]["boxes"], 0.))
        pred = dict(labels_equal=left["prediction"]["labels"] == right["prediction"]["labels"],
                    boxes=array_comparison(left["prediction"]["boxes"], right["prediction"]["boxes"],
                                           POLICY["boxes_absolute_tolerance_input_pixels"]),
                    scores=array_comparison(left["prediction"]["scores"], right["prediction"]["scores"],
                                            POLICY["scores_absolute_tolerance"]))
        comparisons.append(dict(image=left["image"], target=target, prediction=pred))
    def passed(row):
        return row["target"]["labels_equal"] and row["target"]["boxes"]["exact_equal"] and (
            row["prediction"]["labels_equal"] and all(row["prediction"][k]["within_tolerance"]
                                                     for k in ("boxes", "scores")))
    exact = all(row["target"]["labels_equal"] and row["prediction"]["labels_equal"] and
                all(row[kind][field]["bit_pattern_equal"] for kind, field in (
                    ("target", "boxes"), ("prediction", "boxes"), ("prediction", "scores")))
                for row in comparisons)
    maxima = {}
    for field in ("boxes", "scores"):
        values = [row["prediction"][field]["maximum_absolute_difference"] for row in comparisons]
        maxima[field] = max(values) if all(v is not None for v in values) else None
    return dict(within_tolerance=all(passed(row) for row in comparisons),
                all_exact_bit_patterns_equal=exact, maximum_prediction_absolute_differences=maxima,
                mismatched_images=[row["image"] for row in comparisons if not passed(row)],
                per_image=comparisons)


def compare_metrics(actual, expected):
    fields = {}
    for name in train.METRICS:
        left, right = float(actual[name]), float(expected[name])
        require(math.isfinite(left) and math.isfinite(right), f"Nonfinite metric {name}")
        difference = abs(left - right)
        fields[name] = dict(actual=left, expected=right, absolute_difference=difference,
                            exact_equal=left == right,
                            within_tolerance=difference <= POLICY["metrics_absolute_tolerance"])
    counts = {name: dict(actual=actual[name], expected=expected[name], equal=actual[name] == expected[name])
              for name in COUNT_METRICS if name in expected}
    return dict(within_tolerance=all(v["within_tolerance"] for v in fields.values()) and
                                 all(v["equal"] for v in counts.values()),
                all_exact_equal=all(v["exact_equal"] for v in fields.values()) and
                                all(v["equal"] for v in counts.values()), fields=fields, counts=counts)


def checkpoint_gate(result_root, arm, seed, cfg, artifacts):
    folder = train.result_dir(result_root, arm, seed)
    result, status = read_json(folder / "result.json"), read_json(folder / "status.json")
    expected = train.binding(result_root, arm, seed, True)
    prefix = folder.relative_to(result_root).as_posix() + "/"
    for key, name in (("best_checkpoint_sha256", "best.pt"), ("latest_checkpoint_sha256", "latest.pt"),
                      ("history_sha256", "epoch_log.json")):
        require(result.get(key) == artifacts[prefix + name], f"{arm}/seed{seed}: {key} mismatch")
    require(result.get("execution_manifest_sha256") == expected["execution_manifest_sha256"], "Result manifest mismatch")
    require(result.get("prototype") == arm and result.get("seed") == seed and
            result.get("formal_eligible") is True and result.get("test_evaluated") is False, "Result identity/seal mismatch")
    require(status.get("status") == "COMPLETE" and status.get("epochs") == 80 and
            status.get("binding") == expected and status.get("test_evaluated") is False, "Run is not COMPLETE")
    latest = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    history = read_json(folder / "epoch_log.json")
    require(latest["epoch"] == 80 and latest["global_step"] == 6640 and latest["history"] == history,
            "Latest checkpoint/history/step mismatch")
    require([r["epoch"] for r in history] == list(range(1, 81)), "Incomplete epoch history")
    require(all(r["batches"] == 83 and r["valid"]["num_images"] == 99 and
                all(math.isfinite(float(r["valid"][k])) for k in train.METRICS) for r in history),
            "Incomplete/nonfinite train/valid history")
    selected = max(history, key=lambda r: r["valid"]["mAP50_95"])
    require(best["epoch"] == selected["epoch"] == result["best_epoch"] and
            best["global_step"] == best["epoch"] * 83 and best["history"] == history[:best["epoch"]],
            "Best checkpoint is not the earliest best epoch/prefix")
    initial = read_json(folder / "initialization_identity.json")
    run = read_json(folder / "run_binding.json")
    info = read_json(folder / "model_info.json")
    canonical = validate_architecture(cfg["arms"][arm])
    for state in (latest, best):
        require(state["binding"] == expected and state["prototype"] == arm and state["seed"] == seed and
                state.get("test_evaluated") is False, "Checkpoint binding/prototype/seed/seal mismatch")
        require(validate_architecture(state["architecture"]) == canonical and
                state["initialization_identity"] == initial, "Checkpoint architecture/initialization mismatch")
    require(best["architecture_contract"] == latest["architecture_contract"] and
            json_normalize(best["architecture_contract"]) == run["architecture_contract"] == info["architecture_contract"],
            "Architecture contract mismatch")
    require(run["binding"] == expected and run["initialization_identity"] == initial == info["initialization_identity"] and
            run.get("test_evaluated") is False, "Run/model-info binding/initialization mismatch")
    require(best["initial_state_sha256"] == latest["initial_state_sha256"] == run["initial_state_sha256"],
            "Initial state SHA mismatch")
    require(result["Params"] == info["total_params"] and result["MACs"] == info["estimated_macs"], "Cost identity mismatch")
    require(result["endpoint_mAP50_95"] == history[-1]["valid"]["mAP50_95"], "Endpoint metric mismatch")
    del latest
    return best, result, selected["valid"], initial, run


def replay_one(result_root, output_root, arm, seed, cfg, baseline, manifest, artifacts, require_exact):
    state, result, selected, initial, run = checkpoint_gate(result_root, arm, seed, cfg, artifacts)
    train.frozen.seed_all(seed)
    model = build_model(architecture=cfg["arms"][arm], seed=seed).to("cuda")
    require(model.initialization_identity() == initial, "Live model initialization differs from original run")
    require(model.architecture_contract() == state["architecture_contract"], "Live model architecture contract drift")
    require(train.frozen.state_hash(model) == state["initial_state_sha256"], "Live initial tensor hash drift")
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to("cuda")
    require(json_normalize(asdict(criterion.config)) == run["loss_config"], "Live loss/anchors/decoder contract drift")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    before = train.frozen.state_hash(model)
    loader = train.frozen.make_loader(baseline, ROOT / baseline["dataset"]["data_dir"], "valid", seed, 0)
    expected_images = [entry["image"] for entry in manifest["input_verification"]["datasets"]["valid"]["entries"]]
    folder = train.result_dir(result_root, arm, seed)
    saved_records = read_json(folder / "best_valid_records.json")
    saved_metrics = read_json(folder / "best_valid_metrics.json")
    with torch.inference_mode():
        metrics, records = train.frozen.evaluate(model, loader, criterion, baseline, "cuda")
    torch.cuda.synchronize()
    require(train.frozen.state_hash(model) == before, "Inference modified model parameters/BN buffers")
    require(not model.training and all(p.grad is None for p in model.parameters()), "Replay entered training/backward")
    out = output_root / arm / f"seed_{seed}"
    write_new(out / "replayed_valid_metrics.json", metrics)
    write_new(out / "replayed_valid_records.json", records)
    record_comparison = compare_records(records, saved_records, expected_images)
    from_saved = train.frozen.metrics_from_records(saved_records, baseline)
    metric_comparisons = dict(replay_vs_result=compare_metrics(metrics, result),
                             replay_vs_checkpoint_history=compare_metrics(metrics, selected),
                             replay_vs_saved_metrics=compare_metrics(metrics, saved_metrics),
                             saved_records_vs_history=compare_metrics(from_saved, selected))
    within = record_comparison["within_tolerance"] and all(v["within_tolerance"] for v in metric_comparisons.values())
    exact = record_comparison["all_exact_bit_patterns_equal"] and all(v["all_exact_equal"] for v in metric_comparisons.values())
    receipt = dict(arm=arm, seed=seed, best_epoch=state["epoch"], valid_images=len(records),
                   best_checkpoint_sha256=train.sha(folder / "best.pt"),
                   loaded_state_sha256=before, model_state_unchanged=True,
                   strict_comparison_pass=exact, tolerance_comparison_pass=within,
                   status="PASS" if within and (exact or not require_exact) else "FAIL",
                   prediction_comparison=record_comparison, metric_comparisons=metric_comparisons,
                   test_evaluated=False, backward_called=False, thresholds_changed=False)
    write_new(out / "comparison.json", receipt)
    del model, criterion, loader, state
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def run(result_root, output_root, physical_gpu=2, require_exact=False):
    result_root, output_root = Path(result_root).resolve(), Path(output_root).resolve()
    require(output_root != result_root and not result_root.is_relative_to(output_root),
            "Replay output needs a separate folder; it cannot replace the campaign root or an ancestor")
    require(not output_root.is_relative_to(result_root / "formal") and
            not output_root.is_relative_to(result_root / "smoke"), "Replay output cannot enter a formal/smoke run")
    receipt_path = output_root / "replay_receipt.json"
    self_sha = train.sha(Path(__file__))
    if receipt_path.exists():
        existing = read_json(receipt_path)
        require(existing.get("status") == "PASS", "Previous replay did not pass; preserve it and choose a new output namespace")
        completed = existing.get("runs", [])
        require(existing.get("all_twelve_replayed") is True and len(completed) == 12 and
                {(r["arm"], r["seed"]) for r in completed} ==
                {(arm, seed) for arm in train.PROTOTYPES for seed in train.SEEDS} and
                all(r.get("status") == "PASS" and r.get("tolerance_comparison_pass") is True and
                    (not require_exact or r.get("strict_comparison_pass") is True) for r in completed),
                "Existing PASS receipt lacks twelve successful bound comparison records")
        manifest, _ = train.load_manifest(result_root, True)
        current = inventory(result_root)
        require(existing["binding"] == dict(execution_manifest_sha256=train.sha(result_root / "execution_manifest.json"),
                replay_script_sha256=self_sha, original_artifact_sha256=current,
                comparison_policy=dict(POLICY, require_exact=require_exact), physical_gpu=physical_gpu),
                "Existing replay receipt binding differs; do not overwrite")
        cfg, _ = train.load_config()
        complete_gate(result_root, cfg, existing["binding"]["execution_manifest_sha256"], current)
        for rel, digest in existing["output_artifact_sha256"].items():
            require(train.sha(output_root / rel) == digest, "Existing replay output SHA mismatch: " + rel)
        return existing
    output_root.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="FAIL", created_at=train.now(), replay_script_sha256=self_sha,
                   comparison_policy=dict(POLICY, require_exact=require_exact), runs=[],
                   test_evaluated=False, training_performed=False, original_results_modified=False)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "CUDA_VISIBLE_DEVICES must equal the explicitly selected physical GPU")
        require(torch.cuda.is_available(), "Original CUDA environment required; no CPU fallback")
        manifest, baseline = train.load_manifest(result_root, True)
        cfg, _ = train.load_config()
        artifacts = inventory(result_root)
        manifest_sha = train.sha(result_root / "execution_manifest.json")
        workers = complete_gate(result_root, cfg, manifest_sha, artifacts)
        train.frozen.seed_all(train.SEEDS[0])
        environment = train.environment("cuda")
        spec = yaml.safe_load((ROOT / cfg["environment_spec"]).read_text(encoding="utf-8"))
        for key, expected in spec["expected"].items():
            require(environment.get(key) == str(expected), "Pinned original environment mismatch: " + key)
        require(environment["gpu"] == spec["gpu"], "Pinned GPU model mismatch")
        receipt.update(binding=dict(execution_manifest_sha256=manifest_sha,
            replay_script_sha256=self_sha, original_artifact_sha256=artifacts,
            comparison_policy=dict(POLICY, require_exact=require_exact), physical_gpu=physical_gpu),
            environment=environment, workers=workers,
            valid_dataset_sha256=manifest["input_verification"]["datasets"]["valid"]["sha256"])
        failures = []
        for seed in train.SEEDS:
            for arm in train.PROTOTYPES:
                try:
                    row = replay_one(result_root, output_root, arm, seed, cfg, baseline, manifest, artifacts, require_exact)
                except Exception as error:
                    row = dict(arm=arm, seed=seed, status="FAIL", error_type=type(error).__name__, error=str(error))
                receipt["runs"].append(row)
                if row["status"] != "PASS":
                    failures.append(f"{arm}/seed{seed}")
                print(json.dumps(dict(arm=arm, seed=seed, status=row["status"],
                    strict_comparison_pass=row.get("strict_comparison_pass")), ensure_ascii=False), flush=True)
        require(inventory(result_root) == artifacts, "Original campaign artifacts changed during replay")
        train.load_manifest(result_root, True)
        receipt.update(completed_at=train.now(), status="PASS" if not failures else "FAIL", failed_runs=failures,
                       all_twelve_replayed=len(receipt["runs"]) == 12,
                       all_strict_comparisons_pass=all(r.get("strict_comparison_pass", False) for r in receipt["runs"]))
    except Exception as error:
        receipt.update(status="FAIL", completed_at=train.now(), error_type=type(error).__name__, error=str(error))
    receipt["output_artifact_sha256"] = {p.relative_to(output_root).as_posix(): train.sha(p)
        for p in sorted(output_root.rglob("*.json")) if p != receipt_path}
    write_new(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=train.DEFAULT_ROOT / "remote")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--physical-gpu", type=int, choices=[2, 3], default=2)
    parser.add_argument("--require-exact", action="store_true",
                        help="Require exact recorded numeric bit patterns in addition to the fixed tolerance checks")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output_root or args.result_root.parent / "checkpoint_replay_final"
    receipt = run(args.result_root, output, args.physical_gpu, args.require_exact)
    print(json.dumps(dict(status=receipt["status"], receipt=str(output / "replay_receipt.json"),
                          completed_runs=len(receipt["runs"])), ensure_ascii=False), flush=True)
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
