#!/usr/bin/env python3
"""INT8 PTQ accuracy evaluation under the frozen protocol.

Takes the checkpoints produced by ``run_eval_protocol.py --save-checkpoints``
and reports FP32 vs simulated-INT8 metrics on each outer validation fold.
Calibration batches come from the fold's training data (eval transforms);
the outer fold is only used for the final paired evaluation.

Example:
  python run_eval_int8.py --protocol-run results/protocol/baseline_rl_arch_135 \
      --candidate-path <best_candidate.json>
  python run_eval_int8.py --protocol-run results/protocol/baseline_mnv2_scratch \
      --arch mobilenet_v2
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import create_protocol_dataloaders
from hwnas_fpga.deploy.ptq_eval import apply_ptq, stratified_calibration_indices
from hwnas_fpga.models import build_model
from hwnas_fpga.models.backbones import build_backbone
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.trainer import evaluate_classifier
from hwnas_fpga.training.protocol_reporting import canonical_sha256, sha256_file

CHECKPOINT_PATTERN = re.compile(r"^best_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.pt$")


def build_run_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.candidate_path:
        architecture = load_architecture_from_artifact(args.candidate_path)
        return build_model(
            architecture=architecture,
            num_classes=num_classes,
            head_channels=architecture.head_channels,
        )
    model, _ = build_backbone(
        name=args.arch,
        num_classes=num_classes,
        input_channels=1,
        pretrained=False,
    )
    return model


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-run", required=True,
                        help="run directory from run_eval_protocol.py --save-checkpoints")
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--arch", default="mobilenet_v2")
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-allowed-drop", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.protocol_run)
    checkpoints = sorted(
        (path, CHECKPOINT_PATTERN.match(path.name))
        for path in run_dir.glob("best_fold*_seed*.pt")
    )
    checkpoints = [(path, match) for path, match in checkpoints if match]
    if args.fold is not None:
        checkpoints = [
            (path, match)
            for path, match in checkpoints
            if int(match.group("fold")) == args.fold
        ]
    if args.seed is not None:
        checkpoints = [
            (path, match)
            for path, match in checkpoints
            if int(match.group("seed")) == args.seed
        ]
    if not checkpoints:
        print(
            f"No best_fold*_seed*.pt checkpoints in {run_dir}; "
            "rerun run_eval_protocol.py with --save-checkpoints.",
            file=sys.stderr,
        )
        return 1

    criterion = nn.CrossEntropyLoss().to(device)
    records: list[dict] = []
    for path, match in checkpoints:
        fold = int(match.group("fold"))
        seed = int(match.group("seed"))
        print(f"\n=== fold {fold} seed {seed}: {path.name} ===")

        bundle = create_protocol_dataloaders(
            args.data_dir,
            fold=fold,
            seed=seed,
            batch_size=args.batch_size,
            image_size=args.image_size,
            inner_val_fraction=args.inner_val_fraction,
            num_workers=args.num_workers,
        )
        num_classes = bundle["num_classes"]

        model = build_run_model(args, num_classes)
        checkpoint_payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            isinstance(checkpoint_payload, dict)
            and isinstance(checkpoint_payload.get("model_state_dict"), dict)
        ):
            state = checkpoint_payload["model_state_dict"]
        elif isinstance(checkpoint_payload, dict) and all(
            torch.is_tensor(value) for value in checkpoint_payload.values()
        ):
            state = checkpoint_payload
        else:
            raise ValueError(f"Unsupported checkpoint payload: {path}")
        model.load_state_dict(state)
        model = model.to(device).eval()

        fp32_summary = evaluate_classifier(
            model,
            bundle["outer_val_loader"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        labels = [int(label) for _, label in bundle["eval_dataset"].samples]
        calibration_indices = stratified_calibration_indices(
            labels,
            bundle["split"].train_indices,
            max_samples=args.calibration_samples,
            seed=seed,
        )
        if set(calibration_indices) & set(bundle["split"].outer_val_indices):
            raise RuntimeError("PTQ calibration leaked into outer validation")
        calibration_loader = DataLoader(
            Subset(bundle["eval_dataset"], calibration_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(device).startswith("cuda"),
        )

        # Calibrate on a fixed, stratified training-side subset only.
        ptq_meta = apply_ptq(
            model,
            calibration_loader,
            num_calibration_batches=len(calibration_loader),
            device=device,
        )
        int8_summary = evaluate_classifier(
            model,
            bundle["outer_val_loader"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        drop_macro_f1 = fp32_summary["macro_f1"] - int8_summary["macro_f1"]
        drop_top1 = fp32_summary["top1"] - int8_summary["top1"]
        gate_pass = (
            drop_macro_f1 <= args.max_allowed_drop
            and drop_top1 <= args.max_allowed_drop
        )
        record = {
            "fold": fold,
            "seed": seed,
            "checkpoint": path.name,
            "fp32": {key: fp32_summary[key] for key in ("macro_f1", "top1", "weighted_f1")},
            "int8": {key: int8_summary[key] for key in ("macro_f1", "top1", "weighted_f1")},
            "delta_macro_f1": int8_summary["macro_f1"] - fp32_summary["macro_f1"],
            "delta_top1": int8_summary["top1"] - fp32_summary["top1"],
            "checkpoint_sha256": sha256_file(path),
            "calibration": {
                "source": "inner_train_eval_transform",
                "sample_count": len(calibration_indices),
                "indices_sha256": canonical_sha256(calibration_indices),
                "outer_overlap": 0,
            },
            "ptq": ptq_meta,
            "ptq_gate": {
                "pass": gate_pass,
                "max_allowed_drop": args.max_allowed_drop,
                "macro_f1_drop": drop_macro_f1,
                "top1_drop": drop_top1,
                "required_action": "proceed_to_parity" if gate_pass else "qat_required",
            },
        }
        records.append(record)
        print(
            f"fp32 macro_f1={fp32_summary['macro_f1']:.4f} -> "
            f"int8 macro_f1={int8_summary['macro_f1']:.4f} "
            f"(delta {record['delta_macro_f1']:+.4f})"
        )

    aggregate = {
        "schema_version": 2,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol_run": str(run_dir),
        "device": device,
        "fp32_macro_f1": summarize([r["fp32"]["macro_f1"] for r in records]),
        "int8_macro_f1": summarize([r["int8"]["macro_f1"] for r in records]),
        "delta_macro_f1": summarize([r["delta_macro_f1"] for r in records]),
        "delta_top1": summarize([r["delta_top1"] for r in records]),
        "records": records,
        "claim_boundary": records[0]["ptq"]["claim_boundary"],
        "ptq_gate": {
            "pass": all(record["ptq_gate"]["pass"] for record in records),
            "max_allowed_drop": args.max_allowed_drop,
            "failure_action": "run matching-scheme QAT before HLS/board validation",
        },
    }
    output_path = run_dir / "int8_ptq_summary.json"
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nSummary written to {output_path}")
    delta = aggregate["delta_macro_f1"]
    if delta["mean"] is not None:
        print(
            f"INT8 - FP32 macro_f1: {delta['mean']:+.4f} +/- {delta['std']:.4f} "
            f"(n={delta['n']})"
        )
    return 0 if aggregate["ptq_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
