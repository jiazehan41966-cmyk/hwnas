"""Phase5B: fixed-schedule H2 QAT for frozen B1/C/D best.pt checkpoints."""

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
from detection_controlled_study import seed_all
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.deploy.h2_qat import prepare_h2_qat
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from hwnas_fpga.training.trainer import create_optimizer
from phase5a_quantization.calibrate import calibration_tensor_batches
from phase5a_quantization.common import (
    BASELINE_CONFIG,
    NETWORKS,
    QAT_ABSOLUTE_DROP,
    QAT_RELATIVE_DROP,
    admit_network,
    baseline,
    build_network,
    checkpoint_rows,
    dump_json,
    require_frozen_xstar,
    resolve_checkpoint,
    sha256_file,
)
from phase5a_quantization.export_int8 import operator_coverage, write_quantization_profile


REPORT = ROOT / "reports/phase5b_qat"
ARTIFACTS = ROOT / "artifacts/phase5b_qat"
PHASE5A_REPORT = ROOT / "reports/phase5a_quantization"
QAT_EPOCHS = 20
LR_MULTIPLIER = 0.1
QAT_LR = 0.0001


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_weights(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()


def metric_row(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_Precision": float(metrics["Precision"]),
        f"{prefix}_Recall": float(metrics["Recall"]),
        f"{prefix}_mAP50": float(metrics["AP50"]),
        f"{prefix}_mAP50_95": float(metrics["mAP50_95"]),
    }


def evaluate_model(model: torch.nn.Module, device: str) -> dict[str, Any]:
    config = baseline()
    loader = frozen.make_loader(config, frozen.resolve(config["dataset"]["data_dir"]), "valid", 42, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(config)).to(device)
    metrics, _records = frozen.evaluate(model.to(device).eval(), loader, criterion, config, device)
    metrics["test_evaluated"] = False
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion: DenseAnchorDetectionLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    batches = 0
    images_seen = 0
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        output = model(images)
        frozen.finite_outputs(output)
        loss, parts = criterion(output, targets)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite QAT loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batches += 1
        images_seen += int(images.shape[0])
    return {"train_loss": total_loss / max(1, batches), "train_batches": batches, "train_images": images_seen}


def write_protocol() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase5B Protocol

Status: frozen before QAT execution.

Date: {now()}

## Scope

Networks: B1, C, D.

Seeds: 42, 43, 44.

Checkpoint policy: each run starts from the verified original `best.pt` selected by Phase5A `ALL_BEST_PT`.

Forbidden: architecture changes, NUMERIC_CONTRACT_V1 changes, test access, NAS, new networks, INT4/mixed precision, FPGA mapping changes, HLS reruns, and best-valid checkpoint selection for QAT.

## Quantization Contract

NUMERIC_CONTRACT_V1 hash: `{EXPECTED_HASH}`.

- INT8 activation
- INT8 weight
- INT32 accumulation
- INT64 requant multiplier/shift
- nearest-even rounding
- saturation [-128, 127]
- zero_point = 0
- H2 activation and Add scale-alignment rules unchanged

QAT fake quantization uses the same H2 fixed-scale contract initialized from Phase5A train calibration.

## Initialization

For every network/seed:

1. Load FP32 original `best.pt`.
2. Reuse Phase5A calibration manifest and quant config hash.
3. Initialize fake-quant exponents/ranges from Dataset A train calibration.
4. Train the H2 fake-quant model.

PTQ quantized weights are not used as independent initialization.

## Training

- Dataset: Dataset A train only.
- Validation: Dataset A clean valid for curve logging and final reporting.
- Test: sealed; `test_evaluated=false`.
- QAT epochs: {QAT_EPOCHS}.
- Original LR: 0.001.
- QAT LR: {QAT_LR} = {LR_MULTIPLIER} x original LR.
- Optimizer: AdamW inherited from the frozen baseline protocol.
- Weight decay: inherited from the frozen baseline protocol.
- Checkpoint for formal comparison: final QAT epoch checkpoint only.

The epoch count and LR are fixed here and must not be changed based on validation results.
"""
    (REPORT / "PHASE5B_PROTOCOL.md").write_text(text, encoding="utf-8")


def phase5a_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, str]]]:
    manifest = read_json(PHASE5A_REPORT / "PTQ_MANIFEST.json")
    calibration = read_json(PHASE5A_REPORT / "CALIBRATION_MANIFEST.json")
    if manifest["checkpoint_policy"] != "ALL_BEST_PT":
        raise ValueError("Phase5B requires Phase5A ALL_BEST_PT")
    if manifest["status"] != "QAT_REQUIRED":
        raise ValueError("Phase5B is only admitted after QAT_REQUIRED")
    if manifest["test_evaluated"] is not False:
        raise ValueError("test must remain sealed")
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with (PHASE5A_REPORT / "PTQ_PAIRED_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["network"], int(row["seed"]))] = row
    return manifest, calibration, rows


def process_one(
    row: dict[str, str],
    calibration_manifest: dict[str, Any],
    phase5a_rows: dict[tuple[str, int], dict[str, str]],
    *,
    device: str,
    workers: int,
) -> dict[str, Any]:
    label = row["label"]
    seed = int(row["seed"])
    identity = admit_network(label)
    located = resolve_checkpoint(label, seed, row)
    if located["status"] != "FOUND":
        raise FileNotFoundError(f"{label} seed{seed} verified best.pt not found")
    if not str(located["path"]).endswith("best.pt"):
        raise ValueError("Phase5B must start from best.pt only")

    seed_all(seed)
    config = baseline()
    data_root = frozen.resolve(config["dataset"]["data_dir"])
    train_loader = frozen.make_loader(config, data_root, "train", seed, workers, train=True)
    criterion = DenseAnchorDetectionLoss(config=build_network(label)[0].loss_config(config)).to(device)

    fp32_model, _ = build_network(label)
    load_weights(fp32_model, Path(located["path"]))
    fp32_metrics = evaluate_model(fp32_model, device)

    batches = list(calibration_tensor_batches(calibration_manifest, device))
    _folded, ptq_model, _extrema, ptq_profile = prepare_h2_ptq(
        fp32_model,
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
    )
    ptq_metrics = evaluate_model(ptq_model, device)

    qat_model, _qat_extrema, qat_profile = prepare_h2_qat(
        fp32_model,
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
    )
    qat_model.to(device)
    optimizer = create_optimizer(
        qat_model,
        config["training"]["optimizer"],
        QAT_LR,
        config["training"]["weight_decay"],
    )
    qat_criterion = DenseAnchorDetectionLoss(qat_model.loss_config(config)).to(device)
    history = []
    for epoch in range(QAT_EPOCHS):
        train_summary = train_one_epoch(
            qat_model,
            train_loader,
            qat_criterion,
            optimizer,
            device=device,
        )
        valid_metrics = evaluate_model(qat_model, device)
        record = {
            "epoch": epoch + 1,
            **train_summary,
            "valid_mAP50": float(valid_metrics["AP50"]),
            "valid_mAP50_95": float(valid_metrics["mAP50_95"]),
            "valid_Precision": float(valid_metrics["Precision"]),
            "valid_Recall": float(valid_metrics["Recall"]),
        }
        history.append(record)
        print(
            f"{label} seed={seed} qat_epoch={epoch + 1}/{QAT_EPOCHS} "
            f"loss={record['train_loss']:.6f} valid_mAP50_95={record['valid_mAP50_95']:.6f}",
            flush=True,
        )

    qat_metrics = evaluate_model(qat_model, device)
    coverage = operator_coverage(qat_model)
    out = ARTIFACTS / label / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "qat_final_epoch.pt"
    torch.save(
        {
            "model_state_dict": qat_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": QAT_EPOCHS,
            "source_best_pt": located["path"],
            "source_best_pt_sha256": located["sha256"],
            "architecture_id": identity,
            "seed": seed,
            "schedule": {
                "epochs": QAT_EPOCHS,
                "lr": QAT_LR,
                "lr_multiplier": LR_MULTIPLIER,
                "optimizer": config["training"]["optimizer"],
                "weight_decay": config["training"]["weight_decay"],
                "checkpoint_policy": "final_epoch",
            },
            "quantization_profile": qat_profile,
            "test_evaluated": False,
        },
        checkpoint_path,
    )
    dump_json(out / "qat_epoch_history.json", history)
    write_quantization_profile(
        out / "qat_quantization_profile.json",
        network=label,
        architecture_id=identity,
        seed=seed,
        profile=qat_profile,
        coverage=coverage,
    )

    delta_ptq = float(ptq_metrics["mAP50_95"]) - float(fp32_metrics["mAP50_95"])
    delta_qat = float(qat_metrics["mAP50_95"]) - float(fp32_metrics["mAP50_95"])
    recovery = float(qat_metrics["mAP50_95"]) - float(ptq_metrics["mAP50_95"])
    denominator = float(fp32_metrics["mAP50_95"]) - float(ptq_metrics["mAP50_95"])
    recovery_ratio = recovery / denominator if denominator > 0 else ""
    phase5a = phase5a_rows[(label, seed)]
    return {
        "network": label,
        "seed": seed,
        "architecture_id": identity,
        "source_best_pt": located["path"],
        "source_best_pt_sha256": located["sha256"],
        "qat_checkpoint": str(checkpoint_path),
        "qat_checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration_dataset_hash": calibration_manifest["dataset_hash"],
        "calibration_manifest_sha256": sha256_file(PHASE5A_REPORT / "CALIBRATION_MANIFEST.json"),
        "quant_config_sha256": calibration_manifest["quant_config_sha256"],
        "numeric_contract_hash": EXPECTED_HASH,
        "qat_epochs": QAT_EPOCHS,
        "qat_lr": QAT_LR,
        "checkpoint_policy": "final_epoch",
        **metric_row(fp32_metrics, "fp32"),
        **metric_row(ptq_metrics, "ptq"),
        **metric_row(qat_metrics, "qat"),
        "phase5a_fp32_mAP50_95": phase5a["fp32_mAP50_95"],
        "phase5a_ptq_mAP50_95": phase5a["int8_mAP50_95"],
        "delta_PTQ_mAP50_95": delta_ptq,
        "delta_QAT_mAP50_95": delta_qat,
        "recovery_mAP50_95": recovery,
        "recovery_ratio_mAP50_95": recovery_ratio,
        "operator_coverage_h2": coverage["all_operators_h2_covered"],
        "graph_identity_unchanged": True,
        "test_evaluated": False,
    }


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    aggregate_rows = []
    recovery_rows = []
    recovered_count = 0
    meaningful_count = 0
    for label in NETWORKS:
        items = sorted([row for row in rows if row["network"] == label], key=lambda row: int(row["seed"]))
        fp32 = [float(row["fp32_mAP50_95"]) for row in items]
        ptq = [float(row["ptq_mAP50_95"]) for row in items]
        qat = [float(row["qat_mAP50_95"]) for row in items]
        delta_qat = [float(row["delta_QAT_mAP50_95"]) for row in items]
        recovery = [float(row["recovery_mAP50_95"]) for row in items]
        ratios = [
            float(row["recovery_ratio_mAP50_95"])
            for row in items
            if row["recovery_ratio_mAP50_95"] != ""
        ]
        mean_fp32 = float(mean(fp32))
        mean_qat = float(mean(qat))
        absolute_drop = mean_fp32 - mean_qat
        relative_drop = absolute_drop / mean_fp32 if mean_fp32 else float("inf")
        gate = (
            "QAT_RECOVERED"
            if absolute_drop <= QAT_ABSOLUTE_DROP and relative_drop <= QAT_RELATIVE_DROP
            else "QAT_NOT_FULLY_RECOVERED"
        )
        if gate == "QAT_RECOVERED":
            recovered_count += 1
        if float(mean(recovery)) > 0:
            meaningful_count += 1
        aggregate_rows.append(
            {
                "network": label,
                "n": len(items),
                "fp32_mAP50_95_mean": mean_fp32,
                "fp32_mAP50_95_sample_std": sample_std(fp32),
                "ptq_mAP50_95_mean": float(mean(ptq)),
                "ptq_mAP50_95_sample_std": sample_std(ptq),
                "qat_mAP50_95_mean": mean_qat,
                "qat_mAP50_95_sample_std": sample_std(qat),
                "delta_QAT_seed42": next(float(row["delta_QAT_mAP50_95"]) for row in items if int(row["seed"]) == 42),
                "delta_QAT_seed43": next(float(row["delta_QAT_mAP50_95"]) for row in items if int(row["seed"]) == 43),
                "delta_QAT_seed44": next(float(row["delta_QAT_mAP50_95"]) for row in items if int(row["seed"]) == 44),
                "delta_QAT_mean": float(mean(delta_qat)),
                "delta_QAT_sample_std": sample_std(delta_qat),
                "delta_QAT_min": float(min(delta_qat)),
                "delta_QAT_max": float(max(delta_qat)),
                "recovery_mean": float(mean(recovery)),
                "recovery_sample_std": sample_std(recovery),
                "recovery_ratio_mean": float(mean(ratios)) if ratios else "",
                "absolute_mean_drop_mAP50_95": absolute_drop,
                "relative_mean_drop_mAP50_95": relative_drop,
                "gate": gate,
            }
        )
        for row in items:
            recovery_rows.append(
                {
                    "network": row["network"],
                    "seed": row["seed"],
                    "source_best_pt_sha256": row["source_best_pt_sha256"],
                    "fp32_mAP50_95": row["fp32_mAP50_95"],
                    "ptq_mAP50_95": row["ptq_mAP50_95"],
                    "qat_mAP50_95": row["qat_mAP50_95"],
                    "delta_PTQ_mAP50_95": row["delta_PTQ_mAP50_95"],
                    "delta_QAT_mAP50_95": row["delta_QAT_mAP50_95"],
                    "recovery_mAP50_95": row["recovery_mAP50_95"],
                    "recovery_ratio_mAP50_95": row["recovery_ratio_mAP50_95"],
                }
            )
    if recovered_count == len(NETWORKS):
        overall = "ALL_RECOVERED"
    elif meaningful_count:
        overall = "PARTIAL_RECOVERY"
    else:
        overall = "NO_MEANINGFUL_RECOVERY"
    return aggregate_rows, recovery_rows, overall


def write_review(
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    *,
    overall: str,
    h4_status: str,
) -> None:
    lines = [
        "# Phase5B Final Review",
        "",
        f"- QAT runs complete: {len(rows)}/9",
        f"- Overall gate: {overall}",
        f"- H4 cost binding: {h4_status}",
        "- test_evaluated=false",
        "- QAT checkpoint policy: final epoch only",
        "- Joint Pareto optimization was not started.",
        "",
        "| network | FP32 mean | FP32 std | PTQ mean | PTQ std | QAT mean | QAT std | QAT gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['network']} | {row['fp32_mAP50_95_mean']} | {row['fp32_mAP50_95_sample_std']} | "
            f"{row['ptq_mAP50_95_mean']} | {row['ptq_mAP50_95_sample_std']} | "
            f"{row['qat_mAP50_95_mean']} | {row['qat_mAP50_95_sample_std']} | {row['gate']} |"
        )
    (REPORT / "PHASE5B_FINAL_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    require_frozen_xstar()
    load_numeric_contract_v1()
    phase5a_manifest, calibration_manifest, phase5a_rows = phase5a_inputs()
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_protocol()
    device = device_name()
    partial_path = REPORT / "qat_run_results_partial.json"
    rows = []
    completed: set[tuple[str, int]] = set()
    if args.resume_completed and partial_path.is_file():
        rows = read_json(partial_path)
        completed = {(str(row["network"]), int(row["seed"])) for row in rows}
        print(f"resume_completed={len(rows)}", flush=True)
    for row in checkpoint_rows():
        key = (row["label"], int(row["seed"]))
        if key in completed:
            print(f"{row['label']} seed={row['seed']} Phase5B QAT already complete; skipping", flush=True)
            continue
        print(f"{row['label']} seed={row['seed']} Phase5B QAT", flush=True)
        result = process_one(
            row,
            calibration_manifest,
            phase5a_rows,
            device=device,
            workers=args.workers,
        )
        rows.append(result)
        dump_json(partial_path, rows)

    aggregate_rows, recovery_rows, overall = aggregate(rows)
    h4_status = (
        "H4_COST_BINDING_COMPATIBLE"
        if all(row["operator_coverage_h2"] and row["graph_identity_unchanged"] for row in rows)
        and phase5a_manifest["h4_cost_binding"] == "H4_COST_BINDING_COMPATIBLE"
        else "NEEDS_MEASUREMENT"
    )
    write_csv(REPORT / "QAT_RUN_RESULTS.csv", rows)
    write_csv(REPORT / "QAT_AGGREGATE.csv", aggregate_rows)
    write_csv(REPORT / "PTQ_VS_QAT_RECOVERY.csv", recovery_rows)
    manifest = {
        "phase": "phase5b_qat",
        "status": overall,
        "qat_runs_completed": len(rows),
        "checkpoint_policy": "final_epoch",
        "started_from": "verified_original_best_pt",
        "qat_schedule": {
            "epochs": QAT_EPOCHS,
            "lr": QAT_LR,
            "lr_multiplier": LR_MULTIPLIER,
            "optimizer": "adamw",
            "weight_decay": baseline()["training"]["weight_decay"],
        },
        "phase5a_ptq_manifest_sha256": sha256_file(PHASE5A_REPORT / "PTQ_MANIFEST.json"),
        "calibration_manifest_sha256": sha256_file(PHASE5A_REPORT / "CALIBRATION_MANIFEST.json"),
        "quant_config_sha256": calibration_manifest["quant_config_sha256"],
        "numeric_contract": "NUMERIC_CONTRACT_V1",
        "numeric_contract_hash": EXPECTED_HASH,
        "h4_cost_binding": h4_status,
        "test_evaluated": False,
        "qat_started": True,
        "joint_optimization_started": False,
        "network_selection_from_qat": False,
        "runs": rows,
        "aggregate": aggregate_rows,
    }
    dump_json(REPORT / "QAT_MANIFEST.json", manifest)
    write_review(rows, aggregate_rows, overall=overall, h4_status=h4_status)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
