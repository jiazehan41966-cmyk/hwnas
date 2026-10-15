#!/usr/bin/env python3
"""Evaluate MobileNetA frozen Pareto candidates for complexity and PTQ.

This is a deployment-preflight layer for the ZCU104 MobileNetA classification
campaign.  It reuses the frozen NKSID protocol dataloaders and retrained
checkpoints, calibrates PTQ only from training-side samples, and reports FP32
versus fake-quantized INT8 metrics.  It does not claim HLS, board, or power
parity.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.dataset import create_protocol_dataloaders
from hwnas_fpga.deploy.int8_reference import (
    IntegerReferenceClassifier,
    UnsupportedIntegerOperatorError,
)
from hwnas_fpga.deploy.ptq_eval import (
    apply_ptq,
    prepare_integer_ptq,
    stratified_calibration_indices,
)
from hwnas_fpga.training.protocol_reporting import canonical_sha256, sha256_file
from hwnas_fpga.training.trainer import evaluate_classifier

from run_eval_protocol import build_run_model, load_checkpoint_state, per_class_f1, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain-root",
        default="results/zcu104_mobilenetA_retrain_protocol_v1",
        help="Root containing one completed run_eval_protocol directory per candidate.",
    )
    parser.add_argument(
        "--candidate-dir",
        default=(
            "artifacts/zcu104_full_campaign_v1/frozen_candidates/"
            "mobilenetA_rl200_20260908_155713/candidates"
        ),
        help="Directory containing *.frozen_candidate.json files.",
    )
    parser.add_argument(
        "--summary-json",
        default=(
            "artifacts/zcu104_full_campaign_v1/retrain_protocol_v1_summaries/"
            "zcu104_mobilenetA_retrain_protocol_v1_summary.json"
        ),
        help="Retrain aggregate summary with role, arch_id, and run_name rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/zcu104_full_campaign_v1/int8_complexity/mobilenetA_rl200_20260908_1640",
    )
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on candidate rows.",
    )
    parser.add_argument(
        "--unit-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on fold/seed units per candidate.",
    )
    parser.add_argument(
        "--skip-integer-reference",
        action="store_true",
        help="Skip the fail-closed integer-reference probe.",
    )
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def count_params(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


@torch.no_grad()
def count_macs(model: nn.Module, *, input_shape: tuple[int, int, int, int], device: str) -> int:
    modules_seen: set[int] = set()
    total_macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total_macs
        if id(module) in modules_seen:
            return
        modules_seen.add(id(module))
        batch = max(1, int(output.shape[0]))
        out_elements_per_sample = int(output.numel() // batch)
        kernel_ops = int(module.in_channels // module.groups)
        kernel_ops *= int(module.kernel_size[0]) * int(module.kernel_size[1])
        total_macs += out_elements_per_sample * kernel_ops

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total_macs
        if id(module) in modules_seen:
            return
        modules_seen.add(id(module))
        batch = max(1, int(output.shape[0]))
        out_elements_per_sample = int(output.numel() // batch)
        total_macs += out_elements_per_sample * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model = model.to(device).eval()
    dummy = torch.zeros(input_shape, device=device)
    model(dummy)
    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return int(total_macs)


def build_candidate_model(candidate_path: Path, num_classes: int) -> tuple[nn.Module, dict[str, Any]]:
    args = SimpleNamespace(
        candidate_path=str(candidate_path),
        arch="mobilenet_v2",
        pretrained=False,
    )
    return build_run_model(args, num_classes)


def calibration_loader_from_bundle(
    bundle: dict[str, Any],
    *,
    batch_size: int,
    max_samples: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, dict[str, Any]]:
    labels = [int(label) for _path, label in bundle["eval_dataset"].samples]
    selected = stratified_calibration_indices(
        labels,
        list(bundle["split"].train_indices),
        max_samples=max_samples,
        seed=seed,
    )
    overlap = set(selected) & set(bundle["split"].outer_val_indices)
    if overlap:
        raise RuntimeError(f"PTQ calibration leaked {len(overlap)} outer-val samples")
    loader = DataLoader(
        Subset(bundle["eval_dataset"], selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda"),
    )
    return loader, {
        "source": "training_side_only",
        "max_samples": int(max_samples),
        "selected_samples": len(selected),
        "sha256": canonical_sha256(selected),
    }


def evaluate_one_unit(
    *,
    candidate_path: Path,
    checkpoint_path: Path,
    data_dir: str,
    fold: int,
    seed: int,
    image_size: int,
    inner_val_fraction: float,
    batch_size: int,
    calibration_samples: int,
    calibration_batches: int,
    num_workers: int,
    device: str,
    skip_integer_reference: bool,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    set_global_seed(seed)
    bundle = create_protocol_dataloaders(
        data_dir,
        fold=fold,
        seed=seed,
        batch_size=batch_size,
        image_size=image_size,
        inner_val_fraction=inner_val_fraction,
        num_workers=num_workers,
        geometry_mode="stretch_224",
        augmentation_profile="frozen_strong",
    )
    model, model_meta = build_candidate_model(candidate_path, int(bundle["num_classes"]))
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model = model.to(device).eval()

    criterion = nn.CrossEntropyLoss().to(device)
    fp32 = evaluate_classifier(
        model,
        bundle["outer_val_loader"],
        criterion=criterion,
        device=device,
        num_classes=int(bundle["num_classes"]),
        topk=5,
    )
    fp32_per_class = per_class_f1(fp32["confusion_matrix"])

    calibration_loader, calibration = calibration_loader_from_bundle(
        bundle,
        batch_size=batch_size,
        max_samples=calibration_samples,
        seed=seed,
        num_workers=num_workers,
    )
    int8_model = deepcopy(model)
    ptq_metadata = apply_ptq(
        int8_model,
        calibration_loader,
        num_calibration_batches=calibration_batches,
        bit_width=8,
        device=device,
    )
    int8 = evaluate_classifier(
        int8_model,
        bundle["outer_val_loader"],
        criterion=criterion,
        device=device,
        num_classes=int(bundle["num_classes"]),
        topk=5,
    )
    int8_per_class = per_class_f1(int8["confusion_matrix"])

    integer_reference = {
        "status": "SKIPPED" if skip_integer_reference else "NOT_RUN",
        "parity_pass": False,
    }
    if not skip_integer_reference:
        try:
            integer_loader, _integer_calibration = calibration_loader_from_bundle(
                bundle,
                batch_size=1,
                max_samples=max(8, min(calibration_samples, 64)),
                seed=seed,
                num_workers=0,
            )
            folded, package, metadata = prepare_integer_ptq(
                model,
                integer_loader,
                num_calibration_batches=max(1, min(calibration_batches, 8)),
                device=device,
            )
            classifier = IntegerReferenceClassifier(folded.cpu(), package).to(device).eval()
            first_batch = next(iter(bundle["outer_val_loader"]))
            inputs = first_batch[0][:1].to(device)
            _ = classifier(inputs)
            integer_reference = {
                "status": "PROBE_PASS_GRAPH_EXECUTED",
                "parity_pass": False,
                "metadata": metadata,
                "quantization_spec_sha256": canonical_sha256(package.get("quantization") or {}),
            }
        except (UnsupportedIntegerOperatorError, RuntimeError, ValueError, StopIteration) as exc:
            integer_reference = {
                "status": "FAIL_CLOSED_UNSUPPORTED_OR_UNVERIFIED",
                "parity_pass": False,
                "error": str(exc)[:500],
            }

    return {
        "fold": int(fold),
        "seed": int(seed),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "model": {
            "architecture_sha256": model_meta.get("architecture_sha256"),
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
        },
        "complexity": {
            "params": count_params(model),
            "macs": count_macs(
                model,
                input_shape=(1, 1, int(image_size), int(image_size)),
                device=device,
            ),
        },
        "fp32": {
            "macro_f1": float(fp32["macro_f1"]),
            "top1": float(fp32["top1"]),
            "top5": float(fp32["top5"]),
            "loss": float(fp32["loss"]),
            "per_class_f1": fp32_per_class,
            "confusion_matrix": fp32["confusion_matrix"],
        },
        "int8_ptq_fake_quant": {
            "macro_f1": float(int8["macro_f1"]),
            "top1": float(int8["top1"]),
            "top5": float(int8["top5"]),
            "loss": float(int8["loss"]),
            "per_class_f1": int8_per_class,
            "confusion_matrix": int8["confusion_matrix"],
            "metadata": ptq_metadata,
        },
        "deltas": {
            "macro_f1_drop": float(fp32["macro_f1"]) - float(int8["macro_f1"]),
            "top1_drop": float(fp32["top1"]) - float(int8["top1"]),
        },
        "calibration": calibration,
        "integer_reference_probe": integer_reference,
    }


def aggregate_candidate(row: dict[str, Any], units: list[dict[str, Any]]) -> dict[str, Any]:
    fp32_macro = [float(unit["fp32"]["macro_f1"]) for unit in units]
    fp32_top1 = [float(unit["fp32"]["top1"]) for unit in units]
    int8_macro = [float(unit["int8_ptq_fake_quant"]["macro_f1"]) for unit in units]
    int8_top1 = [float(unit["int8_ptq_fake_quant"]["top1"]) for unit in units]
    macro_drop = [float(unit["deltas"]["macro_f1_drop"]) for unit in units]
    top1_drop = [float(unit["deltas"]["top1_drop"]) for unit in units]
    params = [int(unit["complexity"]["params"]) for unit in units]
    macs = [int(unit["complexity"]["macs"]) for unit in units]
    integer_statuses = sorted({str(unit["integer_reference_probe"]["status"]) for unit in units})
    return {
        "role": row["role"],
        "arch_id": row["arch_id"],
        "run_name": row["run_name"],
        "unit_count": len(units),
        "params": params[0] if len(set(params)) == 1 else None,
        "macs": macs[0] if len(set(macs)) == 1 else None,
        "params_values": params,
        "macs_values": macs,
        "fp32_macro_f1": mean_std(fp32_macro),
        "fp32_top1": mean_std(fp32_top1),
        "int8_macro_f1": mean_std(int8_macro),
        "int8_top1": mean_std(int8_top1),
        "int8_macro_f1_drop": mean_std(macro_drop),
        "int8_top1_drop": mean_std(top1_drop),
        "ptq_type": "post_training_symmetric_per_tensor_float_fake_quant",
        "int8_parity_pass": False,
        "integer_reference_statuses": integer_statuses,
    }


def write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "role",
        "arch_id",
        "run_name",
        "unit_count",
        "params",
        "macs",
        "fp32_macro_f1_mean",
        "fp32_macro_f1_std",
        "fp32_top1_mean",
        "fp32_top1_std",
        "int8_macro_f1_mean",
        "int8_macro_f1_std",
        "int8_top1_mean",
        "int8_top1_std",
        "int8_macro_f1_drop_mean",
        "int8_top1_drop_mean",
        "ptq_type",
        "int8_parity_pass",
        "integer_reference_statuses",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "role": candidate["role"],
                    "arch_id": candidate["arch_id"],
                    "run_name": candidate["run_name"],
                    "unit_count": candidate["unit_count"],
                    "params": candidate["params"],
                    "macs": candidate["macs"],
                    "fp32_macro_f1_mean": candidate["fp32_macro_f1"]["mean"],
                    "fp32_macro_f1_std": candidate["fp32_macro_f1"]["std"],
                    "fp32_top1_mean": candidate["fp32_top1"]["mean"],
                    "fp32_top1_std": candidate["fp32_top1"]["std"],
                    "int8_macro_f1_mean": candidate["int8_macro_f1"]["mean"],
                    "int8_macro_f1_std": candidate["int8_macro_f1"]["std"],
                    "int8_top1_mean": candidate["int8_top1"]["mean"],
                    "int8_top1_std": candidate["int8_top1"]["std"],
                    "int8_macro_f1_drop_mean": candidate["int8_macro_f1_drop"]["mean"],
                    "int8_top1_drop_mean": candidate["int8_top1_drop"]["mean"],
                    "ptq_type": candidate["ptq_type"],
                    "int8_parity_pass": candidate["int8_parity_pass"],
                    "integer_reference_statuses": ";".join(candidate["integer_reference_statuses"]),
                }
            )


def main() -> None:
    args = parse_args()
    args.device = str(args.device).strip()
    args.data_dir = str(args.data_dir).strip()
    started_at = time.perf_counter()
    summary_path = (ROOT / args.summary_json).resolve()
    retrain_root = (ROOT / args.retrain_root).resolve()
    candidate_dir = (ROOT / args.candidate_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(summary.get("rows") or [])
    if args.candidate_limit is not None:
        rows = rows[: max(0, int(args.candidate_limit))]
    candidate_summaries: list[dict[str, Any]] = []
    all_units: list[dict[str, Any]] = []
    for row in rows:
        arch_id = str(row["arch_id"])
        candidate_path = candidate_dir / f"{arch_id}.frozen_candidate.json"
        run_dir = retrain_root / str(row["run_name"])
        protocol_summary = json.loads((run_dir / "protocol_summary.json").read_text(encoding="utf-8"))
        units_plan = [
            {"fold": int(item["fold"]), "seed": int(item["seed"])}
            for item in protocol_summary.get("runs", [])
        ]
        if args.unit_limit is not None:
            units_plan = units_plan[: max(0, int(args.unit_limit))]
        units: list[dict[str, Any]] = []
        for plan in units_plan:
            fold = int(plan["fold"])
            seed = int(plan["seed"])
            checkpoint_path = run_dir / f"best_fold{fold}_seed{seed}.pt"
            print(f"[ptq] {arch_id} fold={fold} seed={seed}", flush=True)
            unit = evaluate_one_unit(
                candidate_path=candidate_path,
                checkpoint_path=checkpoint_path,
                data_dir=args.data_dir,
                fold=fold,
                seed=seed,
                image_size=args.image_size,
                inner_val_fraction=args.inner_val_fraction,
                batch_size=args.batch_size,
                calibration_samples=args.calibration_samples,
                calibration_batches=args.calibration_batches,
                num_workers=args.num_workers,
                device=args.device,
                skip_integer_reference=args.skip_integer_reference,
            )
            unit.update({"role": row["role"], "arch_id": arch_id, "run_name": row["run_name"]})
            units.append(unit)
            all_units.append(unit)
        candidate_summaries.append(aggregate_candidate(row, units))

    result = {
        "schema_version": 1,
        "campaign_id": "zcu104_full_campaign_v1",
        "task": "NKSID_image_classification",
        "input_size": [1, int(args.image_size), int(args.image_size)],
        "source": {
            "summary_json": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "retrain_root": str(retrain_root),
            "candidate_dir": str(candidate_dir),
        },
        "ptq": {
            "type": "PTQ",
            "implementation": "hwnas_fpga.deploy.ptq_eval.apply_ptq",
            "calibration_source": "training_side_only",
            "calibration_samples": int(args.calibration_samples),
            "calibration_batches": int(args.calibration_batches),
            "claim_boundary": (
                "FP32 and PTQ fake-quant validation only; INT8 parity, HLS, "
                "ZCU104 board classification, and power remain separate evidence."
            ),
        },
        "candidate_summaries": candidate_summaries,
        "units": all_units,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    json_path = output_dir / "zcu104_mobilenetA_ptq_complexity_summary.json"
    csv_path = output_dir / "zcu104_mobilenetA_ptq_complexity_summary.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(csv_path, candidate_summaries)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
