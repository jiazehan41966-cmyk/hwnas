"""Phase 5C coarse quantization sensitivity audit.

This is a software diagnostic only. It reuses Phase 5A PTQ and temporarily
keeps one coarse module in FP32 while the remaining graph uses the same H2
INT8 fake-quant path. It never evaluates test and never starts QAT.
"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "tools"),
    str(ROOT / "tools/sonar_analysis"),
    str(ROOT / "tools/sonar_spec"),
    str(ROOT / "scripts"),
]

import baseline as frozen
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from phase5a_quantization.calibrate import build_calibration, calibration_tensor_batches
from phase5a_quantization.common import (
    BASELINE_CONFIG,
    NETWORKS,
    REPORT as PHASE5A_REPORT,
    SEARCH_SPACE,
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
from phase5a_quantization.run import load_weights


REPORT = ROOT / "reports/phase5c_quant_sensitivity"
ARTIFACTS = ROOT / "artifacts/phase5c_quant_sensitivity"
BASELINE_TOLERANCE = 1e-12

MODULE_SPECS: dict[str, dict[str, Any]] = {
    "ALL_INT8": {
        "prefixes": (),
        "quantize_neck_adds": True,
        "description": "Phase5A baseline: full H2 INT8 PTQ fake-quant graph.",
    },
    "BACKBONE_FP32": {
        "prefixes": ("backbone",),
        "quantize_neck_adds": True,
        "description": "Backbone FP32; neck/head remain INT8.",
    },
    "NECK_FP32": {
        "prefixes": ("neck",),
        "quantize_neck_adds": False,
        "description": "FPN neck FP32, including lateral/output convs and FPN adds; backbone/head remain INT8.",
    },
    "HEAD_FP32": {
        "prefixes": ("head",),
        "quantize_neck_adds": True,
        "description": "Detection head FP32; backbone/neck remain INT8.",
    },
    "P2_PATH_FP32": {
        "prefixes": ("neck.lateral_convs.0", "neck.output_convs.0", "head.towers.0"),
        "quantize_neck_adds": True,
        "description": "B1-only P2 lateral, P2 output branch, and P2 head tower/prediction path FP32.",
        "networks": ("B1",),
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_phase5a_rows() -> dict[tuple[str, int], dict[str, Any]]:
    path = PHASE5A_REPORT / "PTQ_PAIRED_RESULTS.csv"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["network"]), int(row["seed"]))
            rows[key] = row
    expected = {(label, seed) for label in NETWORKS for seed in SEEDS}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(f"Phase5A PTQ rows are incomplete: {missing}")
    return rows


def evaluate_model(model: torch.nn.Module, device: str) -> dict[str, Any]:
    forbid_split("valid")
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
        "test_evaluated": False,
    }


def build_quantized_variant(
    *,
    label: str,
    checkpoint_path: Path,
    calibration_manifest: dict[str, Any],
    device: str,
    module: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, _identity = build_network(label)
    load_weights(model, checkpoint_path)
    spec = MODULE_SPECS[module]
    batches = list(calibration_tensor_batches(calibration_manifest, device))
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        model.eval(),
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
        fp32_module_prefixes=spec["prefixes"],
        quantize_neck_adds=bool(spec["quantize_neck_adds"]),
    )
    return quantized, profile


def admitted_modules(label: str) -> list[str]:
    result = ["ALL_INT8", "BACKBONE_FP32", "NECK_FP32", "HEAD_FP32"]
    if label == "B1":
        result.append("P2_PATH_FP32")
    return result


def classify_network(module_rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in module_rows if row["module"] != "ALL_INT8"]
    for row in candidates:
        if float(row["mean_recovery_ratio"]) >= 0.50 and int(row["positive_seed_count"]) >= 2:
            return "LOCALIZED_SENSITIVITY"
    for row in candidates:
        ratio = float(row["mean_recovery_ratio"])
        if 0.20 <= ratio < 0.50 and int(row["positive_seed_count"]) >= 2:
            return "PARTIAL_LOCALIZATION"
    return "DISTRIBUTED_OR_UNRESOLVED"


def q_space_gate(network_summaries: dict[str, str], module_summary: list[dict[str, Any]]) -> str:
    localized = [name for name, status in network_summaries.items() if status == "LOCALIZED_SENSITIVITY"]
    if len(localized) >= 2:
        return "YES"
    for row in module_summary:
        if (
            row["module"] != "ALL_INT8"
            and int(row["positive_seed_count"]) == 3
            and float(row["mean_recovery_ratio"]) >= 0.50
        ):
            return "YES"
    partial = [name for name, status in network_summaries.items() if status == "PARTIAL_LOCALIZATION"]
    if localized or partial:
        return "CONDITIONAL"
    return "NO"


def write_protocol() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase5C Protocol

Date: {now()}

## Objective

Audit whether uniform INT8 PTQ accuracy loss is concentrated in coarse modules.

This is not QAT tuning, not mixed-precision training, not INT4, not NAS, and not FPGA mixed-precision cost estimation.

## Frozen Inputs

- Networks: B1, C, D
- Seeds: 42, 43, 44
- Checkpoint policy: Phase5A frozen `ALL_BEST_PT`
- Numeric contract: `NUMERIC_CONTRACT_V1`
- Numeric contract SHA256: `{EXPECTED_HASH}`
- Search Space V0.1 SHA256: `{SEARCH_SPACE_SHA256}`
- Evaluation split: Dataset A clean valid only
- test_evaluated: false

## Runs

Baseline: `ALL_INT8`.

Module rescue variants:

- `BACKBONE_FP32`: backbone FP32, neck/head INT8.
- `NECK_FP32`: neck/FPN FP32, backbone/head INT8.
- `HEAD_FP32`: head FP32, backbone/neck INT8.
- `P2_PATH_FP32`: B1 only, P2 lateral/output/head path FP32.

Each run binds the same network, seed, best.pt SHA, calibration manifest, evaluator, and valid split as Phase5A.

## Classification

`LOCALIZED_SENSITIVITY`: mean recovery ratio >= 0.50 and at least 2/3 seeds have positive rescue.

`PARTIAL_LOCALIZATION`: 0.20 <= mean recovery ratio < 0.50 and at least 2/3 seeds have positive rescue.

Otherwise: `DISTRIBUTED_OR_UNRESOLVED`.

FP32/INT8 hybrid runs are software diagnostics only. Future mixed precision is `NEEDS_MEASUREMENT`.
"""
    (REPORT / "PHASE5C_PROTOCOL.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-jobs", type=int, default=0, help="optional debug cap across sensitivity variants")
    args = parser.parse_args()

    require_frozen_xstar()
    load_numeric_contract_v1()
    if sha256_file(SEARCH_SPACE) != SEARCH_SPACE_SHA256:
        raise ValueError("Search Space V0.1 hash changed")
    write_protocol()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase5a = read_phase5a_rows()
    calibration_manifest = build_calibration(REPORT)
    calibration_sha = sha256_file(REPORT / "calibration_manifest.json")

    rows: list[dict[str, Any]] = []
    jobs_done = 0
    for ckpt_row in checkpoint_rows():
        label = ckpt_row["label"]
        seed = int(ckpt_row["seed"])
        identity = admit_network(label)
        located = resolve_checkpoint(label, seed, ckpt_row)
        if located["status"] != "FOUND":
            raise FileNotFoundError(f"missing checkpoint for {label} seed{seed}")
        expected_sha = ckpt_row["best_checkpoint_sha256"]
        if located["sha256"] != expected_sha:
            raise ValueError(f"checkpoint SHA drift for {label} seed{seed}")
        fp32 = phase5a[(label, seed)]
        for module in admitted_modules(label):
            if args.max_jobs and jobs_done >= args.max_jobs:
                break
            print(label, f"seed={seed}", module, flush=True)
            quantized, profile = build_quantized_variant(
                label=label,
                checkpoint_path=Path(located["path"]),
                calibration_manifest=calibration_manifest,
                device=device,
                module=module,
            )
            metrics = evaluate_model(quantized, device)
            baseline_int8 = float(phase5a[(label, seed)]["int8_mAP50_95"])
            if module == "ALL_INT8":
                diff = abs(float(metrics["mAP50_95"]) - baseline_int8)
                if diff > BASELINE_TOLERANCE:
                    dump_json(REPORT / "PHASE5C_MANIFEST.json", {
                        "status": "SENSITIVITY_BASELINE_MISMATCH",
                        "network": label,
                        "seed": seed,
                        "observed": metrics["mAP50_95"],
                        "phase5a": baseline_int8,
                        "abs_diff": diff,
                        "tolerance": BASELINE_TOLERANCE,
                        "test_evaluated": False,
                    })
                    raise RuntimeError("SENSITIVITY_BASELINE_MISMATCH")
            rescue = float(metrics["mAP50_95"]) - baseline_int8
            gap = float(fp32["fp32_mAP50_95"]) - baseline_int8
            recovery = rescue / gap if gap > 0 else 0.0
            row = {
                "network": label,
                "architecture_id": identity,
                "seed": seed,
                "module": module,
                "checkpoint_sha256": located["sha256"],
                "calibration_manifest_sha256": calibration_sha,
                "numeric_contract": "NUMERIC_CONTRACT_V1",
                "numeric_contract_sha256": EXPECTED_HASH,
                "fp32_mAP50": fp32["fp32_mAP50"],
                "fp32_mAP50_95": fp32["fp32_mAP50_95"],
                "all_int8_mAP50": phase5a[(label, seed)]["int8_mAP50"],
                "all_int8_mAP50_95": phase5a[(label, seed)]["int8_mAP50_95"],
                "hybrid_mAP50": metrics["mAP50"],
                "hybrid_mAP50_95": metrics["mAP50_95"],
                "hybrid_Precision": metrics["Precision"],
                "hybrid_Recall": metrics["Recall"],
                "rescue_mAP50_95": rescue,
                "gap_mAP50_95": gap,
                "recovery_ratio": recovery,
                "fp32_prefixes": json.dumps(profile["fp32_bypass"]["module_prefixes"]),
                "neck_adds_quantized": profile["fp32_bypass"]["neck_adds_quantized"],
                "test_evaluated": False,
                "split": "valid",
                "status": "MEASURED",
            }
            rows.append(row)
            jobs_done += 1
            dump_json(REPORT / "partial_results.json", rows)
        if args.max_jobs and jobs_done >= args.max_jobs:
            break

    expected_jobs = sum(len(admitted_modules(label)) for label in NETWORKS for _seed in SEEDS)
    if not args.max_jobs and len(rows) != expected_jobs:
        raise ValueError(f"incomplete Phase5C matrix: {len(rows)}/{expected_jobs}")
    write_csv(REPORT / "SENSITIVITY_RESULTS.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    network_status: dict[str, str] = {}
    for label in NETWORKS:
        module_names = admitted_modules(label)
        per_network_module_rows: list[dict[str, Any]] = []
        for module in module_names:
            values = [row for row in rows if row["network"] == label and row["module"] == module]
            if not values:
                continue
            rescues = [float(row["rescue_mAP50_95"]) for row in values]
            recoveries = [float(row["recovery_ratio"]) for row in values]
            summary = {
                "network": label,
                "module": module,
                "seed42_rescue": next((row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 42), ""),
                "seed43_rescue": next((row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 43), ""),
                "seed44_rescue": next((row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 44), ""),
                "mean_rescue": float(mean(rescues)),
                "sample_std_rescue": sample_std(rescues),
                "positive_seed_count": sum(1 for value in rescues if value > 0),
                "mean_recovery_ratio": float(mean(recoveries)),
                "sample_std_recovery_ratio": sample_std(recoveries),
            }
            summary_rows.append(summary)
            per_network_module_rows.append(summary)
        network_status[label] = classify_network(per_network_module_rows)

    write_csv(REPORT / "MODULE_RECOVERY_SUMMARY.csv", summary_rows)
    gate = q_space_gate(network_status, summary_rows)
    shared_modules = []
    for module in ("BACKBONE_FP32", "NECK_FP32", "HEAD_FP32"):
        positives = [
            row for row in summary_rows
            if row["module"] == module
            and float(row["mean_recovery_ratio"]) >= 0.20
            and int(row["positive_seed_count"]) >= 2
        ]
        if len(positives) >= 2:
            shared_modules.append(module)

    manifest = {
        "phase": "phase5c_quant_sensitivity",
        "status": "PASS",
        "created_at": now(),
        "device": device,
        "runs": len(rows),
        "expected_runs": expected_jobs,
        "baseline_reproduced": True,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "networks": list(NETWORKS),
        "seeds": list(SEEDS),
        "checkpoint_policy": "ALL_BEST_PT",
        "calibration_manifest_sha256": calibration_sha,
        "numeric_contract": "NUMERIC_CONTRACT_V1",
        "numeric_contract_sha256": EXPECTED_HASH,
        "search_space_sha256": SEARCH_SPACE_SHA256,
        "network_classification": network_status,
        "shared_sensitive_region": shared_modules,
        "q_space_extension_gate": gate,
        "mixed_precision_cost_status": "NEEDS_MEASUREMENT",
        "test_evaluated": False,
    }
    dump_json(REPORT / "PHASE5C_MANIFEST.json", manifest)

    gate_text = [
        "# Q Space Extension Gate",
        "",
        f"Gate: `{gate}`",
        "",
        "Mixed-precision is not implemented in Phase5C.",
        "",
        "Any future mixed precision uses `NEEDS_MEASUREMENT`; H3/H4 uniform INT8 costs must not be extrapolated.",
        "",
        f"test_evaluated=false",
    ]
    (REPORT / "Q_SPACE_EXTENSION_GATE.md").write_text("\n".join(gate_text) + "\n", encoding="utf-8")

    report_lines = [
        "# Quantization Sensitivity Report",
        "",
        f"- ALL_INT8 baseline reproduced: `true`",
        f"- Numeric contract: `NUMERIC_CONTRACT_V1` `{EXPECTED_HASH}`",
        f"- Calibration manifest SHA256: `{calibration_sha}`",
        f"- test_evaluated=false",
        "",
        "## Network Classification",
        "",
        "| Network | Classification |",
        "|---|---|",
    ]
    for label in NETWORKS:
        report_lines.append(f"| {label} | `{network_status[label]}` |")
    report_lines += [
        "",
        "## Module Recovery Summary",
        "",
        "| Network | Module | Mean rescue | Std | Positive seeds | Mean recovery ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['network']} | {row['module']} | {float(row['mean_rescue']):+.6f} | {float(row['sample_std_rescue']):.6f} | {row['positive_seed_count']}/3 | {float(row['mean_recovery_ratio']):.6f} |"
        )
    report_lines += [
        "",
        "## Cross-network Interpretation",
        "",
        f"Shared sensitive region: `{', '.join(shared_modules) if shared_modules else 'none'}`.",
        "",
        f"Q_SPACE_EXTENSION_GATE: `{gate}`.",
        "",
        "This is quantization sensitivity evidence only; no sonar physical mechanism is claimed.",
    ]
    (REPORT / "QUANTIZATION_SENSITIVITY_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
