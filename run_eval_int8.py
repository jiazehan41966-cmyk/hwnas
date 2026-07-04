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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import create_protocol_dataloaders
from hwnas_fpga.deploy.ptq_eval import apply_ptq
from hwnas_fpga.models import build_model
from hwnas_fpga.models.backbones import build_backbone
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.trainer import evaluate_classifier

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
    parser.add_argument("--calibration-batches", type=int, default=16)
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
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(device).eval()

        fp32_summary = evaluate_classifier(
            model,
            bundle["outer_val_loader"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        # Calibrate on training-side batches only, then re-evaluate.
        ptq_meta = apply_ptq(
            model,
            bundle["train_loader"],
            num_calibration_batches=args.calibration_batches,
            device=device,
        )
        int8_summary = evaluate_classifier(
            model,
            bundle["outer_val_loader"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        record = {
            "fold": fold,
            "seed": seed,
            "checkpoint": path.name,
            "fp32": {key: fp32_summary[key] for key in ("macro_f1", "top1", "weighted_f1")},
            "int8": {key: int8_summary[key] for key in ("macro_f1", "top1", "weighted_f1")},
            "delta_macro_f1": int8_summary["macro_f1"] - fp32_summary["macro_f1"],
            "delta_top1": int8_summary["top1"] - fp32_summary["top1"],
            "ptq": {key: ptq_meta[key] for key in (
                "num_quantized_ops", "num_calibration_batches", "claim_boundary"
            )},
        }
        records.append(record)
        print(
            f"fp32 macro_f1={fp32_summary['macro_f1']:.4f} -> "
            f"int8 macro_f1={int8_summary['macro_f1']:.4f} "
            f"(delta {record['delta_macro_f1']:+.4f})"
        )

    aggregate = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol_run": str(run_dir),
        "device": device,
        "fp32_macro_f1": summarize([r["fp32"]["macro_f1"] for r in records]),
        "int8_macro_f1": summarize([r["int8"]["macro_f1"] for r in records]),
        "delta_macro_f1": summarize([r["delta_macro_f1"] for r in records]),
        "delta_top1": summarize([r["delta_top1"] for r in records]),
        "records": records,
        "claim_boundary": records[0]["ptq"]["claim_boundary"],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
