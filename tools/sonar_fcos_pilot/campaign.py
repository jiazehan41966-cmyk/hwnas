"""Finite eight-combination search, conditional repeats and sealed final test."""
from __future__ import annotations

import csv
import gc
import math
import statistics
from pathlib import Path

import torch

from .common import read_json, require_manifest, resolve_path, sha256_file, source_binding, utc_now, write_json
from .data import SonarDataset
from .metrics import baseline_gate, evaluate_predictions, write_error_examples
from .model import build_model
from .train import predict, run_directory, run_id, train_one


def select_combination(summaries, combinations):
    """Validation only; ties follow the preregistered order, preferring AAA."""
    by_combo = {item["combo"]: item for item in summaries}
    if len(summaries) != len(by_combo) or any(item.get("seed") != 42 for item in summaries):
        raise ValueError("Selection requires one first-seed (42) result for each combination.")
    if set(by_combo) != set(combinations):
        raise ValueError("Every candidate must have a completed first-seed result before selection.")
    if any(item.get("status") != "COMPLETE" or item.get("trained_epochs") != 30 for item in by_combo.values()):
        raise ValueError("Partial training cannot enter architecture selection.")
    if any(not isinstance(item["validation"].get("map"), (int, float)) or not math.isfinite(item["validation"]["map"]) for item in summaries):
        raise ValueError("Validation mAP must be finite before selection.")
    return max(combinations, key=lambda combo: by_combo[combo]["validation"]["map"])


def comparison_runs(selected, seeds):
    combos = list(dict.fromkeys(["AAA", "BBB", selected]))
    return [(combo, seed) for combo in combos for seed in seeds]


def decide(selected, test_results, config):
    """Prespecified percentage-point gate; no search or tuning on test data."""
    rows = {(result["combo"], result["seed"]): result for result in test_results}
    expected = set(comparison_runs(selected, config["seeds"]))
    if len(rows) != len(test_results) or set(rows) != expected:
        raise ValueError("Final comparison requires exactly the deduplicated shortlist and all three seeds.")
    if any(not isinstance(result["metrics"].get("map"), (int, float)) or not math.isfinite(result["metrics"]["map"]) for result in test_results):
        raise ValueError("Test mAP must be finite.")
    means = {}
    for combo in dict.fromkeys(["AAA", "BBB", selected]):
        values = [rows[(combo, seed)] for seed in config["seeds"]]
        def average(field):
            numbers = [v["metrics"]["diagnostics"][field] for v in values]
            return statistics.mean(numbers) if all(v is not None for v in numbers) else None
        maps = [v["metrics"]["map"] for v in values]
        means[combo] = {"map": statistics.mean(maps), "map_std": statistics.stdev(maps),
                        "small_recall": average("small_recall"),
                        "false_positives_per_image": average("false_positives_per_image")}
    winner = means[selected]
    winning_seeds = sum(all(rows[(selected, seed)]["metrics"]["map"] > rows[(control, seed)]["metrics"]["map"]
                           for control in ("AAA", "BBB")) for seed in config["seeds"])
    small_values = [means[c]["small_recall"] for c in [selected, "AAA", "BBB"]]
    small_improved = all(v is not None for v in small_values) and small_values[0] > max(small_values[1:])
    fp_limit = max(means[c]["false_positives_per_image"] for c in ["AAA", "BBB"])
    checks = {
        "mean_map_gain_vs_both_at_least_one_point": all(winner["map"] - means[c]["map"] >= config["minimum_map_gain"] for c in ["AAA", "BBB"]),
        "at_least_two_paired_seed_wins": winning_seeds >= config["minimum_winning_seeds"],
        "small_recall_improved_vs_both": bool(small_improved),
        "false_positives_no_higher_than_larger_control_mean": winner["false_positives_per_image"] <= fp_limit,
    }
    if selected == "AAA":
        outcome, explanation = "KEEP_ORDINARY", "验证集选中了全普通卷积方案；按层分别选择尚未得到支持。"
    elif selected == "BBB":
        outcome, explanation = "CONTEXT_ONLY_SELECTED", "验证集选中了全扩张方案；结果只能检验扩大查看范围，不能支持按层分别选择。"
    elif all(checks.values()):
        outcome, explanation = "INITIAL_SUPPORT_FOR_MIXED", "混合方案通过预设门槛，初步支持按层分别选择；下一轮连接搜索另行开展。"
    else:
        outcome, explanation = "INSUFFICIENT_EVIDENCE", "混合方案未通过全部门槛，暂不扩大搜索范围。"
    return {"selected_on_validation": selected, "outcome": outcome, "explanation": explanation,
            "checks": checks, "paired_seed_wins": winning_seeds, "means": means,
            "map_unit": "percentage_points_0_to_100", "recall_unit": "fraction_0_to_1"}


def evaluate_test(config, combo, seed, authorization):
    directory = run_directory(config, combo, seed)
    summary = read_json(directory / "summary.json")
    manifest_path, manifest = require_manifest(config)
    if run_id(combo, seed) not in authorization["runs"]:
        raise RuntimeError("This run was not shortlisted using validation data.")
    expected = authorization["runs"][run_id(combo, seed)]
    if sha256_file(directory / "summary.json") != expected["summary_sha256"]:
        raise RuntimeError("Completed training receipt changed after test authorization.")
    if summary["status"] != "COMPLETE" or summary["trained_epochs"] != 30:
        raise RuntimeError("The entire fixed training budget must complete before testing.")
    if sha256_file(summary["best_checkpoint"]) != summary["best_checkpoint_sha256"]:
        raise RuntimeError("Shortlisted checkpoint changed.")
    current = {**source_binding(config), "data_manifest_sha256": sha256_file(manifest_path)}
    if authorization["binding"] != current:
        raise RuntimeError("Source/config/data changed after validation-only selection.")
    for record in manifest["samples"]:
        if record["split"] == "test" and sha256_file(record["image_path"]) != record["sha256"]:
            raise RuntimeError("Official test image changed: " + record["image_path"])
    result_path = directory / "test_result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["authorization_sha256"] != authorization["authorization_sha256"]:
            raise RuntimeError("Test receipt belongs to another shortlist.")
        return result
    dataset = SonarDataset(manifest_path, "test", config["image_size"])
    # Recreate the same pretrained architecture (including FrozenBatchNorm).
    # A weights=None torchvision builder uses a different normalization module.
    model = build_model(combo, len(manifest["classes"]) + 1, seed, pretrained=config["pretrained"],
                        image_size=config["image_size"], cache_dir=resolve_path(config["weight_cache"]))
    saved = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=False)
    if saved["binding"] != summary["binding"]:
        raise RuntimeError("Checkpoint and training receipt identify different runs.")
    model.load_state_dict(saved["model"])
    model.to(config["device"])
    try:
        predictions = predict(model, dataset, config)
        threshold = summary["validation_calibration"]["threshold"]
        metrics = evaluate_predictions(dataset.records, predictions, len(manifest["classes"]), manifest["small_area_threshold"], threshold)
        if metrics["map"] is None:
            raise RuntimeError("Official test mAP is undefined.")
        write_json(directory / "test_predictions.json", predictions)
        examples = write_error_examples(dataset.records, predictions, directory / "test_error_examples",
                                        manifest["small_area_threshold"], threshold)
        result = {"combo": combo, "seed": seed, "metrics": metrics, "test_images": len(dataset),
                  "threshold_selected_on": "validation_only", "threshold": threshold,
                  "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
                  "authorization_sha256": authorization["authorization_sha256"], "error_examples": examples}
        write_json(result_path, result)
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_campaign(config):
    output = resolve_path(config["output_dir"])
    state_path = output / "campaign_status.json"
    try:
        manifest_path, manifest = require_manifest(config)
    except (RuntimeError, FileNotFoundError) as error:
        write_json(state_path, {"status": "BLOCKED_DATA", "reason": str(error), "formal_training_started": False,
                               "updated_at": utc_now(), "max_formal_runs": config["max_formal_runs"]})
        return 2
    lock = output / "campaign.lock"
    # A durable lock prevents two controllers from launching overlapping training.
    try:
        import os
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("A campaign lock exists; verify the old controller has stopped before removing this single lock.") from error
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    state = {"status": "BASELINE", "formal_training_started": True, "completed_runs": [], "updated_at": utc_now()}
    try:
        def train(combo, seed):
            state.update(status="TRAINING", current_run=run_id(combo, seed), updated_at=utc_now())
            write_json(state_path, state)
            result = train_one(config, combo, seed)
            if run_id(combo, seed) not in state["completed_runs"]:
                state["completed_runs"].append(run_id(combo, seed))
            if len(state["completed_runs"]) > config["max_formal_runs"]:
                raise RuntimeError("Formal run budget exceeded.")
            return result
        first_seed = config["seeds"][0]
        baseline = train("AAA", first_seed)
        gate = baseline_gate(baseline["validation"])
        write_json(output / "baseline_gate.json", gate)
        if gate["status"] != "GO":
            state.update(status=gate["status"], reason=gate["reason"], updated_at=utc_now())
            write_json(state_path, state)
            return 0
        summaries = [baseline]
        for combo in config["combinations"][1:]:
            summaries.append(train(combo, first_seed))
        selected = select_combination(summaries, config["combinations"])
        selection = {"combo": selected, "selection_set": "validation_only", "seed": first_seed,
                     "candidate_maps": {r["combo"]: r["validation"]["map"] for r in summaries}, "map_units": "percent_0_to_100"}
        selection_path = output / "selection.json"
        if selection_path.exists() and read_json(selection_path) != selection:
            raise RuntimeError("Frozen validation selection changed; preserve the previous test protocol.")
        write_json(selection_path, selection)
        with (output / "eight_combinations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["组合", "随机编号", "完整训练轮数", "验证平均检测精度_百分数"])
            writer.writerows((r["combo"], first_seed, r["trained_epochs"], r["validation"]["map"]) for r in summaries)
        shortlist = comparison_runs(selected, config["seeds"])
        for combo, seed in shortlist:
            if seed != first_seed:
                train(combo, seed)
        authorization = {"binding": {**source_binding(config), "data_manifest_sha256": sha256_file(manifest_path)},
                         "selection_sha256": sha256_file(selection_path),
                         "runs": {run_id(c, s): {"summary_sha256": sha256_file(run_directory(config, c, s) / "summary.json")} for c, s in shortlist}}
        authorization_path = output / "test_authorization.json"
        if authorization_path.exists() and read_json(authorization_path) != authorization:
            raise RuntimeError("Test shortlist/receipt changed.")
        write_json(authorization_path, authorization)
        authorization["authorization_sha256"] = sha256_file(authorization_path)
        state.update(status="FINAL_TEST", updated_at=utc_now())
        write_json(state_path, state)
        results = [evaluate_test(config, c, s, authorization) for c, s in shortlist]
        result = decide(selected, results, config)
        write_json(output / "comparison.json", result)
        text = ["# 第一轮验证结果", "", result["explanation"], "", "验证集选择的组合：" + selected,
                "", "| 组合 | 三次平均检测精度 | 三次波动 | 小目标找到的比例 | 每图误报 |", "|---|---:|---:|---:|---:|"]
        for combo, row in result["means"].items():
            small = f"{row['small_recall']:.4f}" if row["small_recall"] is not None else "无小目标，无法评价"
            text.append(f"| {combo} | {row['map']:.3f}% | {row['map_std']:.3f} | {small} | {row['false_positives_per_image']:.4f} |")
        text.extend(["", "该结果用于判断研究方向，不能单独证明论文创新或真实拍摄序列之外的泛化能力。", ""])
        (output / "第一轮结果.md").write_text("\n".join(text), encoding="utf-8")
        state.update(status="COMPLETE", outcome=result["outcome"], updated_at=utc_now(), formal_runs=len(state["completed_runs"]))
        write_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="ERROR", reason=str(error), updated_at=utc_now())
        write_json(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)
