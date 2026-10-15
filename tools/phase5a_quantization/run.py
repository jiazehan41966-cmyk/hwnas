"""Phase 5A runner: instantiate Q(x) for frozen B1/C/D. No architecture selection."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "tools"),
    str(ROOT / "tools/sonar_analysis"),
    str(ROOT / "tools/sonar_spec"),
    str(ROOT / "tools/fpga_v0_1"),
    str(ROOT / "scripts"),
]

import baseline as frozen
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from phase5a_quantization.calibrate import build_calibration, calibration_tensor_batches
from phase5a_quantization.common import (
    ARTIFACTS,
    BASELINE_CONFIG,
    NETWORKS,
    QAT_ABSOLUTE_DROP,
    QAT_RELATIVE_DROP,
    REPORT,
    SEARCH_SPACE_SHA256,
    SEEDS,
    admit_network,
    baseline,
    build_network,
    checkpoint_rows,
    dump_json,
    forbid_split,
    require_frozen_xstar,
    resolve_checkpoint,
    sha256_file,
)
from phase5a_quantization.export_int8 import (
    export_fp32_onnx,
    export_int8_onnx,
    operator_coverage,
    write_quantization_profile,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_weights(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()


def evaluate_model(model: torch.nn.Module, split: str, device: str) -> dict[str, Any]:
    forbid_split(split)
    if split != "valid":
        raise ValueError("Phase 5A accuracy uses official valid only")
    config = baseline()
    loader = frozen.make_loader(config, frozen.resolve(config["dataset"]["data_dir"]), "valid", 42, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(config)).to(device)
    metrics, _records = frozen.evaluate(model.to(device).eval(), loader, criterion, config, device)
    return {
        "mAP50": float(metrics["AP50"]),
        "mAP50_95": float(metrics["mAP50_95"]),
        "Precision": float(metrics["Precision"]),
        "Recall": float(metrics["Recall"]),
        "num_images": int(metrics["num_images"]),
        "split": "valid",
        "test_evaluated": False,
    }


def quant_config_payload() -> dict[str, Any]:
    contract = load_numeric_contract_v1()
    return {
        "phase": "phase5a",
        "purpose": "Q(x)_infrastructure_not_quantization_search",
        "admitted_networks": NETWORKS,
        "qat": False,
        "search": {
            "architecture_search": False,
            "quantization_aware_architecture_search": False,
            "joint_nas_quantization": False,
            "quantize_all_144": False,
        },
        "official_policy": {
            "scheme": "symmetric",
            "zero_point": 0,
            "activation_bits": 8,
            "weight_bits": 8,
            "accumulator_bits": 32,
            "requant": "int64_multiplier_integer_shift",
            "rounding": "nearest_even",
            "saturation": [-128, 127],
            "weight_granularity": "per_tensor",
            "activation_granularity": "per_tensor",
            "scale": "positive_power_of_two_exponent_int32",
            "numeric_contract": "NUMERIC_CONTRACT_V1",
            "numeric_contract_hash": EXPECTED_HASH,
        },
        "configurable_not_used_for_fpga_profile": {
            "scheme_choices": ["symmetric", "asymmetric"],
            "weight_granularity_choices": ["per_tensor", "per_channel"],
            "activation_granularity_choices": ["per_tensor", "per_channel"],
            "asymmetric_fpga_status": "rejected_zero_point_must_remain_0",
        },
        "calibration": {
            "split": "train",
            "valid_used_for_selection": False,
            "test_evaluated": False,
            "seed": 42,
            "images": 64,
            "batch_size": 4,
            "augmentation": False,
        },
        "evaluation": {
            "split": "valid",
            "metrics": ["mAP50", "mAP50_95", "Precision", "Recall"],
            "primary_drop": "delta_mAP50_95",
            "test_evaluated": False,
            "network_selection_from_ptq": False,
        },
        "qat_trigger_preregistered": {
            "absolute_drop_mAP50_95": QAT_ABSOLUTE_DROP,
            "relative_drop": QAT_RELATIVE_DROP,
            "rule": "QAT_REQUIRED if any admitted network mean(FP32-INT8) mAP50:95 exceeds 0.003 or 20% of its FP32 value",
        },
        "h2_contract": contract,
    }


def write_protocol() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 5A Protocol

Status: frozen before PTQ evaluation.

Date: {now()}

## Objective

Instantiate network-dependent quantization Q(x) for the frozen candidate set

X* = {{B1, C, D}}

This is PTQ infrastructure, not quantization search, not QAT, and not (x,q,m) joint optimization.

## Immutable inputs

- Search Space V0.1 SHA256 `{SEARCH_SPACE_SHA256}`
- NUMERIC_CONTRACT_V1 hash `{EXPECTED_HASH}`
- B1: `P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5`
- C: `P3_P4_P5_s1k3_s2k5_s4k5_s5k3_s6k5_s7k3`
- D: `P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3`

Phase 5A does not modify X*, B1/C/D, the search space, NAS, or SonarOp.

## Admitted networks

Only B1, C, and D. 144-architecture quantization is forbidden. NAS + quantization joint search is forbidden.

## Numeric contract

FPGA H2 NUMERIC_CONTRACT_V1 is reused; no new arithmetic contract is created.

- INT8 activation, range [-128, 127]
- INT8 weight, range [-128, 127]
- INT32 accumulation
- requant: INT64 multiplier and integer shift
- rounding: nearest-even
- saturation: [-128, 127]
- zero point: 0
- scale: positive power-of-two exponent

Software knobs for asymmetric / per-channel exist in the infrastructure, but an FPGA profile is admitted only when it remains H2-aligned. Asymmetric FPGA profiles are rejected.

## Calibration

- split: train only
- valid does not participate in calibration selection
- test is sealed (`test_evaluated=false`)
- seed: 42
- images: 64
- batch size: 4
- no augmentation

Recorded fields: number of calibration images, dataset hash, calibration seed.

## Evaluation

Compare FP32 versus INT8 PTQ on the official valid split for B1/C/D.

Metrics: mAP50, mAP50:95, Precision, Recall.

Paired delta: ΔmAP50:95 = INT8 − FP32.

Checkpoints are Phase2F `best.pt` (earliest best-valid), hash-verified. PTQ results do not re-select X*.

## Export

PyTorch FP32 → BN-folded INT8 graph / QDQ model → ONNX.

Report operator coverage and the unsupported operator list. Emit a quantization profile for later M(x,q).

## Pre-registered QAT gate

QAT is not run in Phase 5A.

QAT_REQUIRED if any admitted network has mean absolute drop (FP32 − INT8) mAP50:95 > {QAT_ABSOLUTE_DROP} or relative drop > {QAT_RELATIVE_DROP}.

Otherwise PTQ_SUFFICIENT and the next measured step is FPGA cost evaluation with q, still on the same frozen X*.

If checkpoints are missing, accuracy remains unmeasured and neither QAT nor FPGA-q is started automatically.
"""
    (REPORT / "PHASE5A_PROTOCOL.md").write_text(text, encoding="utf-8")
    (REPORT / "quant_config.yaml").write_text(
        yaml.safe_dump(quant_config_payload(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_calibration_report(manifest: dict[str, Any]) -> None:
    lines = [
        "# Calibration Report",
        "",
        f"- split: `{manifest['split']}`",
        f"- number_of_calibration_images: {manifest['number_of_calibration_images']}",
        f"- train_image_count: {manifest.get('train_image_count', '')}",
        f"- dataset_hash: `{manifest['dataset_hash']}`",
        f"- split_hash: `{manifest.get('split_hash', '')}`",
        f"- calibration_seed: {manifest['calibration_seed']}",
        f"- valid_used_for_selection: {manifest['valid_used_for_selection']}",
        f"- test_evaluated: {manifest['test_evaluated']}",
        f"- augmentation: {manifest['augmentation']}",
        f"- order: {manifest['order']}",
        "",
        "Valid images are not in the selection set. Test is sealed.",
        "",
        "| # | image | image_sha256 |",
        "|---|---|---|",
    ]
    for index, entry in enumerate(manifest["entries"], start=1):
        lines.append(f"| {index} | `{entry['image']}` | `{entry['image_sha256']}` |")
    (REPORT / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_net: dict[str, list[dict[str, Any]]] = {label: [] for label in NETWORKS}
    for row in rows:
        if row["status"] == "MEASURED":
            by_net[row["network"]].append(row)
    summary: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for label, items in by_net.items():
        if not items:
            summary[label] = {"status": "WAITING_FOR_CHECKPOINT", "n": 0}
            continue
        ordered = sorted(items, key=lambda item: int(item["seed"]))
        delta_50 = [float(item["delta_mAP50"]) for item in ordered]
        delta_95 = [float(item["delta_mAP50_95"]) for item in ordered]
        fp32_50 = [float(item["fp32_mAP50"]) for item in ordered]
        fp32_95 = [float(item["fp32_mAP50_95"]) for item in ordered]
        fp32_precision = [float(item["fp32_Precision"]) for item in ordered]
        fp32_recall = [float(item["fp32_Recall"]) for item in ordered]
        int8_50 = [float(item["int8_mAP50"]) for item in ordered]
        int8_95 = [float(item["int8_mAP50_95"]) for item in ordered]
        int8_precision = [float(item["int8_Precision"]) for item in ordered]
        int8_recall = [float(item["int8_Recall"]) for item in ordered]
        mean_fp32 = float(mean(fp32_95))
        mean_int8 = float(mean(int8_95))
        mean_delta = float(mean(delta_95))
        absolute_drop = mean_fp32 - mean_int8
        relative = absolute_drop / mean_fp32 if mean_fp32 else float("inf")
        qat_trigger = bool(absolute_drop > QAT_ABSOLUTE_DROP or relative > QAT_RELATIVE_DROP)
        summary[label] = {
            "status": "MEASURED",
            "n": len(items),
            "fp32": {
                "mAP50": {"mean": float(mean(fp32_50)), "sample_std": sample_std(fp32_50)},
                "mAP50_95": {"mean": mean_fp32, "sample_std": sample_std(fp32_95)},
                "Precision": {"mean": float(mean(fp32_precision)), "sample_std": sample_std(fp32_precision)},
                "Recall": {"mean": float(mean(fp32_recall)), "sample_std": sample_std(fp32_recall)},
            },
            "int8": {
                "mAP50": {"mean": float(mean(int8_50)), "sample_std": sample_std(int8_50)},
                "mAP50_95": {"mean": mean_int8, "sample_std": sample_std(int8_95)},
                "Precision": {"mean": float(mean(int8_precision)), "sample_std": sample_std(int8_precision)},
                "Recall": {"mean": float(mean(int8_recall)), "sample_std": sample_std(int8_recall)},
            },
            "paired_delta": {
                "mAP50": {f"seed{int(item['seed'])}": float(item["delta_mAP50"]) for item in ordered},
                "mAP50_95": {f"seed{int(item['seed'])}": float(item["delta_mAP50_95"]) for item in ordered},
                "mean_mAP50": float(mean(delta_50)),
                "sample_std_mAP50": sample_std(delta_50),
                "min_mAP50": float(min(delta_50)),
                "max_mAP50": float(max(delta_50)),
                "mean_mAP50_95": mean_delta,
                "sample_std_mAP50_95": sample_std(delta_95),
                "min_mAP50_95": float(min(delta_95)),
                "max_mAP50_95": float(max(delta_95)),
            },
            "absolute_mean_drop_mAP50_95": absolute_drop,
            "relative_drop": relative,
            "qat_trigger": qat_trigger,
        }
        aggregate_rows.append(
            {
                "network": label,
                "n": len(ordered),
                "fp32_mAP50_mean": float(mean(fp32_50)),
                "fp32_mAP50_sample_std": sample_std(fp32_50),
                "fp32_mAP50_95_mean": mean_fp32,
                "fp32_mAP50_95_sample_std": sample_std(fp32_95),
                "int8_mAP50_mean": float(mean(int8_50)),
                "int8_mAP50_sample_std": sample_std(int8_50),
                "int8_mAP50_95_mean": mean_int8,
                "int8_mAP50_95_sample_std": sample_std(int8_95),
                "delta_mAP50_seed42": next(float(item["delta_mAP50"]) for item in ordered if int(item["seed"]) == 42),
                "delta_mAP50_seed43": next(float(item["delta_mAP50"]) for item in ordered if int(item["seed"]) == 43),
                "delta_mAP50_seed44": next(float(item["delta_mAP50"]) for item in ordered if int(item["seed"]) == 44),
                "delta_mAP50_mean": float(mean(delta_50)),
                "delta_mAP50_sample_std": sample_std(delta_50),
                "delta_mAP50_min": float(min(delta_50)),
                "delta_mAP50_max": float(max(delta_50)),
                "delta_mAP50_95_seed42": next(float(item["delta_mAP50_95"]) for item in ordered if int(item["seed"]) == 42),
                "delta_mAP50_95_seed43": next(float(item["delta_mAP50_95"]) for item in ordered if int(item["seed"]) == 43),
                "delta_mAP50_95_seed44": next(float(item["delta_mAP50_95"]) for item in ordered if int(item["seed"]) == 44),
                "delta_mAP50_95_mean": mean_delta,
                "delta_mAP50_95_sample_std": sample_std(delta_95),
                "delta_mAP50_95_min": float(min(delta_95)),
                "delta_mAP50_95_max": float(max(delta_95)),
                "absolute_mean_drop_mAP50_95": absolute_drop,
                "relative_drop_mAP50_95": relative,
                "qat_gate": "QAT_REQUIRED" if qat_trigger else "PTQ_ACCEPTABLE",
            }
        )
    measured = [item for item in summary.values() if item["status"] == "MEASURED"]
    waiting = [label for label, item in summary.items() if item["status"] != "MEASURED"]
    if not measured:
        decision = "ACCURACY_NOT_MEASURED"
        nxt = "do_not_start_QAT_or_FPGA_q_until_checkpoints_are_present"
        feasible = "PIPELINE_READY_ACCURACY_PENDING"
    elif any(item["qat_trigger"] for item in measured):
        decision = "QAT_REQUIRED"
        nxt = "stop_without_automatic_QAT"
        feasible = "PTQ_RUNS_BUT_DROP_EXCEEDS_PREREGISTERED_GATE"
    else:
        decision = "PTQ_ACCEPTABLE"
        nxt = "FPGA_cost_evaluation_with_q"
        feasible = "PTQ_FEASIBLE"
    return {
        "per_network": summary,
        "aggregate_rows": aggregate_rows,
        "waiting": waiting,
        "decision": decision,
        "next": nxt,
        "feasibility": feasible,
        "test_evaluated": False,
        "network_selection_from_ptq": False,
    }


def write_remaining_reports(
    rows: list[dict[str, Any]],
    exports: list[dict[str, Any]],
    summary: dict[str, Any],
    contract_check: dict[str, Any],
    calibration_manifest: dict[str, Any],
    quant_config_hash: str,
    h4_cost_binding: str,
) -> None:
    export_lines = [
        "# Export Report",
        "",
        "PyTorch FP32 → BN fold → INT8 QDQ graph → ONNX.",
        "Integer/HLS opcode coverage is separate from ONNX helper ops.",
        "",
        f"test_evaluated: {False}",
        "",
    ]
    for item in exports:
        export_lines += [
            f"## {item['network']} seed {item['seed']}",
            "",
            f"- status: {item['status']}",
            f"- fp32_onnx: `{item.get('fp32_onnx', '')}`",
            f"- int8_onnx: `{item.get('int8_onnx', '')}`",
            f"- profile: `{item.get('profile', '')}`",
            f"- H2 operator coverage: {item.get('coverage', {})}",
            f"- ONNX unsupported/software-only: {item.get('onnx_unsupported', [])}",
            "",
        ]
    (REPORT / "export_report.md").write_text("\n".join(export_lines), encoding="utf-8")
    contract_lines = [
        "# Quantization Contract Check",
        "",
        f"- numeric_contract: NUMERIC_CONTRACT_V1",
        f"- hash: `{EXPECTED_HASH}`",
        f"- hash_verified: {contract_check['hash_verified']}",
        f"- new_contract_created: false",
        f"- INT8 activation: true",
        f"- INT8 weight: true",
        f"- INT32 accumulation: true",
        f"- INT64 multiplier + integer shift: true",
        f"- rounding: nearest_even",
        f"- saturation: [-128, 127]",
        f"- zero_point: 0",
        f"- QAT_run: false",
        f"- X_star_modified: false",
        f"- search_space_modified: false",
        f"- sonarop_modified: false",
        f"- test_evaluated: false",
        f"- network_selection_from_ptq: false",
        "",
        "Phase 5A reuses FPGA H2 NUMERIC_CONTRACT_V1. It does not replace rounding, range, or requant rules.",
        "",
        f"Overall: **{'PASS' if contract_check['pass'] else 'FAIL'}**",
    ]
    (REPORT / "quantization_contract_check.md").write_text("\n".join(contract_lines) + "\n", encoding="utf-8")
    dump_json(REPORT / "phase5a_summary.json", summary)
    write_csv(REPORT / "PTQ_PAIRED_RESULTS.csv", rows)
    write_csv(REPORT / "PTQ_AGGREGATE.csv", summary["aggregate_rows"])

    calibration_manifest = dict(calibration_manifest)
    calibration_manifest["quant_config_sha256"] = quant_config_hash
    dump_json(REPORT / "CALIBRATION_MANIFEST.json", calibration_manifest)

    ptq_manifest = {
        "phase": "phase5a_quantization",
        "status": summary["decision"],
        "checkpoint_policy": "ALL_BEST_PT",
        "paired_by": ["network", "seed", "exact_best_pt_sha256"],
        "numeric_contract": "NUMERIC_CONTRACT_V1",
        "numeric_contract_hash": EXPECTED_HASH,
        "quant_config_sha256": quant_config_hash,
        "calibration_manifest_sha256": sha256_file(REPORT / "CALIBRATION_MANIFEST.json"),
        "fp32_evaluations_completed": sum(1 for row in rows if row["fp32_mAP50_95"] != ""),
        "int8_evaluations_completed": sum(1 for row in rows if row["int8_mAP50_95"] != ""),
        "qat_gate": summary["decision"],
        "qat_started": False,
        "test_evaluated": False,
        "network_selection_from_ptq": False,
        "h4_cost_binding": h4_cost_binding,
        "runs": rows,
        "aggregate": summary["per_network"],
    }
    dump_json(REPORT / "PTQ_MANIFEST.json", ptq_manifest)

    reeval_lines = [
        "# FP32 Reevaluation",
        "",
        "Checkpoint policy: `ALL_BEST_PT`.",
        "",
        "These FP32 values are newly evaluated on Dataset A clean valid using the verified best.pt checkpoints. They are not required to equal historical Phase2F aggregate values; any difference is recorded as checkpoint/evaluation binding difference.",
        "",
        "| run | best.pt SHA | Precision | Recall | mAP50 | mAP50:95 | Phase2F mAP50:95 | binding note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reeval_lines.append(
            "| {network} seed{seed} | `{checkpoint_sha256}` | {fp32_Precision} | {fp32_Recall} | {fp32_mAP50} | {fp32_mAP50_95} | {phase2f_recorded_mAP50_95} | {checkpoint_evaluation_binding_note} |".format(**row)
        )
    reeval_lines += ["", "test_evaluated=false"]
    (REPORT / "FP32_REEVALUATION.md").write_text("\n".join(reeval_lines) + "\n", encoding="utf-8")

    review_lines = [
        "# Phase5A Final Review",
        "",
        f"- FP32 evaluation: {ptq_manifest['fp32_evaluations_completed']}/9",
        f"- INT8 PTQ evaluation: {ptq_manifest['int8_evaluations_completed']}/9",
        f"- QAT gate: {summary['decision']}",
        f"- H4 cost binding: {h4_cost_binding}",
        "- test_evaluated=false",
        "- QAT was not started.",
        "- PTQ results did not re-select networks.",
        "",
        "| network | FP32 mAP50:95 mean | FP32 std | INT8 mAP50:95 mean | INT8 std | mean delta mAP50:95 | relative drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_rows"]:
        review_lines.append(
            f"| {row['network']} | {row['fp32_mAP50_95_mean']} | {row['fp32_mAP50_95_sample_std']} | {row['int8_mAP50_95_mean']} | {row['int8_mAP50_95_sample_std']} | {row['delta_mAP50_95_mean']} | {row['relative_drop_mAP50_95']} |"
        )
    (REPORT / "PHASE5A_FINAL_REVIEW.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")


def process_one(
    row: dict[str, str],
    manifest: dict[str, Any],
    device: str,
    *,
    skip_export: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = row["label"]
    seed = int(row["seed"])
    identity = admit_network(label)
    located = resolve_checkpoint(label, seed, row)
    result = {
        "network": label,
        "architecture_id": identity,
        "seed": seed,
        "checkpoint_sha256": located["sha256"],
        "checkpoint_path": located["path"],
        "checkpoint_status": located["status"],
        "checkpoint_policy": "ALL_BEST_PT",
        "phase2f_bound_artifact": row.get("phase2f_bound_artifact", ""),
        "phase2f_metric_source": row.get("phase2f_metric_source", ""),
        "fp32_mAP50": "",
        "fp32_mAP50_95": "",
        "fp32_Precision": "",
        "fp32_Recall": "",
        "int8_mAP50": "",
        "int8_mAP50_95": "",
        "int8_Precision": "",
        "int8_Recall": "",
        "delta_mAP50": "",
        "delta_mAP50_95": "",
        "phase2f_recorded_mAP50_95": row.get("mAP50_95", ""),
        "checkpoint_evaluation_binding_note": "",
        "status": located["status"],
        "test_evaluated": False,
        "split": "valid",
    }
    export_row = {"network": label, "seed": seed, "architecture_id": identity, "status": located["status"]}
    if located["status"] != "FOUND":
        return result, export_row
    model, _identity = build_network(label)
    load_weights(model, Path(located["path"]))
    fp32 = evaluate_model(model, "valid", device)
    batches = list(calibration_tensor_batches(manifest, device))
    folded, quantized, _extrema, profile = prepare_h2_ptq(
        model,
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
    )
    int8 = evaluate_model(quantized, "valid", device)
    delta_50 = float(int8["mAP50"]) - float(fp32["mAP50"])
    delta_95 = float(int8["mAP50_95"]) - float(fp32["mAP50_95"])
    phase2f_recorded = float(row["mAP50_95"])
    binding_note = (
        "matches_phase2f_metric"
        if abs(float(fp32["mAP50_95"]) - phase2f_recorded) <= 1e-12
        else "checkpoint_evaluation_binding_difference"
    )
    result.update(
        {
            "fp32_mAP50": fp32["mAP50"],
            "fp32_mAP50_95": fp32["mAP50_95"],
            "fp32_Precision": fp32["Precision"],
            "fp32_Recall": fp32["Recall"],
            "int8_mAP50": int8["mAP50"],
            "int8_mAP50_95": int8["mAP50_95"],
            "int8_Precision": int8["Precision"],
            "int8_Recall": int8["Recall"],
            "delta_mAP50": delta_50,
            "delta_mAP50_95": delta_95,
            "status": "MEASURED",
            "phase2f_recorded_mAP50_95": phase2f_recorded,
            "checkpoint_evaluation_binding_note": binding_note,
        }
    )
    coverage = operator_coverage(quantized)
    out = ARTIFACTS / label / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / "quantization_profile.json"
    write_quantization_profile(
        profile_path,
        network=label,
        architecture_id=identity,
        seed=seed,
        profile=profile,
        coverage=coverage,
    )
    dump_json(out / "h2_ptq_profile.json", profile)
    export_row.update(
        {
            "coverage": coverage,
            "profile": str(profile_path),
            "status": "MEASURED",
            "numeric_contract_hash": profile.get("numeric_contract_hash"),
            "h2_aligned": profile.get("h2_aligned"),
            "architecture_id": identity,
        }
    )
    if not skip_export:
        fp32_onnx = export_fp32_onnx(folded, out / "fp32.onnx")
        int8_onnx = export_int8_onnx(quantized, out / "int8_qdq.onnx")
        export_row.update(
            {
                "fp32_onnx": fp32_onnx["path"],
                "int8_onnx": int8_onnx["path"],
                "onnx_unsupported": int8_onnx["unsupported_or_software_only"],
                "fp32_onnx_ops": fp32_onnx["onnx_op_counts"],
                "int8_onnx_ops": int8_onnx["onnx_op_counts"],
            }
        )
    return result, export_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0, help="optional cap for debugging; 0 means all B1/C/D seeds")
    args = parser.parse_args()
    require_frozen_xstar()
    load_numeric_contract_v1()
    write_protocol()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest = build_calibration(REPORT)
    quant_config_hash = sha256_file(REPORT / "quant_config.yaml")
    manifest["quant_config_sha256"] = quant_config_hash
    dump_json(REPORT / "CALIBRATION_MANIFEST.json", manifest)
    write_calibration_report(manifest)
    device = device_name()
    rows = []
    exports = []
    jobs = checkpoint_rows()
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    for row in jobs:
        print(row["label"], f"seed={row['seed']}", flush=True)
        result, export_row = process_one(row, manifest, device, skip_export=args.skip_export)
        rows.append(result)
        exports.append(export_row)
        dump_json(REPORT / "ptq_results_partial.json", rows)
    write_csv(REPORT / "ptq_results.csv", rows)
    summary = summarize(rows)
    summary.update(
        {
            "updated_at": now(),
            "device": device,
            "calibration_dataset_hash": manifest["dataset_hash"],
            "quant_config_sha256": quant_config_hash,
            "numeric_contract_hash": EXPECTED_HASH,
            "baseline_config": str(BASELINE_CONFIG),
            "test_evaluated": False,
        }
    )
    contract = load_numeric_contract_v1()
    contract_check = {
        "hash_verified": True,
        "pass": contract["range"] == [-128, 127] and contract["rounding"] == "nearest_even",
        "test_evaluated": False,
    }
    measured_exports = [item for item in exports if item["status"] == "MEASURED"]
    graph_identity_unchanged = {item.get("architecture_id") for item in measured_exports} == set(NETWORKS.values())
    quant_profile_aligned = (
        contract_check["pass"]
        and all(item.get("h2_aligned") is True for item in measured_exports)
        and all(item.get("numeric_contract_hash") == EXPECTED_HASH for item in measured_exports)
        and all(item.get("coverage", {}).get("all_operators_h2_covered") for item in measured_exports)
    )
    h4_cost_binding = (
        "H4_COST_BINDING_COMPATIBLE"
        if len(measured_exports) == 9 and graph_identity_unchanged and quant_profile_aligned
        else "NEEDS_MEASUREMENT"
    )
    summary["h4_cost_binding"] = h4_cost_binding
    write_remaining_reports(rows, exports, summary, contract_check, manifest, quant_config_hash, h4_cost_binding)
    dump_json(REPORT / "phase5a_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
