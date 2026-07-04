#!/usr/bin/env python3
"""SPOS supernet search entrypoint (Phase 3).

Trains a single-path weight-sharing supernet on the frozen-protocol training
split, then scores candidate paths with inherited weights (BN recalibrated on
training-side batches, metrics from the inner validation set) alongside
calibrated hardware estimates. The outer protocol fold is never touched:
final claims still go through run_eval_protocol.py retraining.

Example:
  python run_supernet.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml \
      --fold 0 --seed 42 --epochs 40 --num-candidates 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from random import Random

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import create_protocol_dataloaders
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search.supernet import (
    MobileAnchorSupernet,
    evaluate_candidate,
    train_supernet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-candidates", type=int, default=200)
    parser.add_argument("--bn-batches", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="debug/smoke only: cap batches per epoch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/supernet")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    image_size = int(dataset_cfg.get("image_size", 224))
    num_classes = int(dataset_cfg.get("num_classes", 8))

    hardware_spec = build_hardware_spec(config)
    constraints = build_constraints(config)
    estimator = build_cost_estimator(
        config, hardware_spec=hardware_spec, constraints=constraints
    )
    space = build_search_space(
        config,
        image_size=image_size,
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=num_classes,
        constraints=constraints,
    )

    run_name = args.run_name or (
        f"supernet_fold{args.fold}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_protocol_dataloaders(
        args.data_dir,
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    supernet = MobileAnchorSupernet(space.config, num_classes=num_classes)
    print(
        f"Supernet parameters: {sum(p.numel() for p in supernet.parameters()):,} "
        f"(device={device})"
    )
    train_loader = bundle["train_loader"]
    if args.max_train_batches is not None:
        from itertools import islice

        class _CappedLoader:
            def __init__(self, loader, cap):
                self._loader = loader
                self._cap = int(cap)
                self.dataset = loader.dataset

            def __iter__(self):
                return islice(iter(self._loader), self._cap)

            def __len__(self):
                return min(self._cap, len(self._loader))

        train_loader = _CappedLoader(train_loader, args.max_train_batches)

    history = train_supernet(
        supernet,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    torch.save(supernet.state_dict(), run_dir / "supernet.pt")

    # Sample unique candidate paths and score them with inherited weights.
    rng = Random(args.seed)
    seen: set[str] = set()
    paths: list[dict] = []
    attempts = 0
    while len(paths) < args.num_candidates and attempts < args.num_candidates * 50:
        attempts += 1
        path = supernet.sample_path(rng)
        key = json.dumps(path, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    print(f"Evaluating {len(paths)} unique candidates ({attempts} draws)")

    ranking_path = run_dir / "supernet_ranking.jsonl"
    records: list[dict] = []
    with ranking_path.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(paths):
            architecture = supernet.path_to_architecture(path)
            estimate = estimator.estimate(architecture, space)
            summary = evaluate_candidate(
                supernet,
                path,
                bn_loader=bundle["train_loader"],
                eval_loader=bundle["inner_val_loader"],
                num_classes=num_classes,
                bn_batches=args.bn_batches,
                device=device,
            )
            record = {
                "candidate_index": index,
                "path": path,
                "architecture": summary.pop("architecture"),
                "inner_val": {
                    key: summary[key]
                    for key in ("macro_f1", "top1", "weighted_f1", "loss")
                },
                "hardware_estimate": {
                    "latency_ms": estimate.latency_ms,
                    "dsp": estimate.total_dsp,
                    "lut": estimate.total_lut,
                    "bram": estimate.total_bram,
                    "violations": list(estimate.violations),
                    "feasible": not estimate.violations,
                },
                "evidence": {
                    "accuracy": "supernet-inherited weights, inner validation set",
                    "hardware": "calibrated analytic estimate; needs HLS shortlist",
                },
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            if (index + 1) % 20 == 0:
                print(f"  {index + 1}/{len(paths)} candidates scored")

    feasible = [r for r in records if r["hardware_estimate"]["feasible"]]
    feasible.sort(key=lambda r: r["inner_val"]["macro_f1"], reverse=True)
    summary_payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "config": str(args.config),
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": device,
        "num_candidates": len(records),
        "num_feasible": len(feasible),
        "train_history_tail": {
            "loss": history["train_loss"][-3:],
            "acc": history["train_acc"][-3:],
        },
        "top10_feasible": [
            {
                "candidate_index": r["candidate_index"],
                "inner_macro_f1": r["inner_val"]["macro_f1"],
                "inner_top1": r["inner_val"]["top1"],
                "latency_ms_est": r["hardware_estimate"]["latency_ms"],
                "dsp_est": r["hardware_estimate"]["dsp"],
                "lut_est": r["hardware_estimate"]["lut"],
            }
            for r in feasible[:10]
        ],
        "evidence_boundary": (
            "inner-val supernet ranking only; claimable numbers require "
            "run_eval_protocol.py retraining and HLS shortlist screening"
        ),
    }
    (run_dir / "supernet_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    # Export the top feasible candidates in the *.candidate.json layout the
    # HLS shortlist tooling consumes.
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for rank, record in enumerate(feasible[:10], start=1):
        payload = {
            "candidate": {
                "arch_id": f"supernet_{run_name}_c{record['candidate_index']}",
                "encoding": record["architecture"],
                "metrics": {
                    "macro_f1": record["inner_val"]["macro_f1"],
                    "top1": record["inner_val"]["top1"],
                    "latency_ms": record["hardware_estimate"]["latency_ms"],
                    "dsp": record["hardware_estimate"]["dsp"],
                    "lut": record["hardware_estimate"]["lut"],
                    "bram": record["hardware_estimate"]["bram"],
                },
            },
            "evidence": record["evidence"],
        }
        path = candidates_dir / f"{rank:03d}_c{record['candidate_index']}.candidate.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nRanking written to {ranking_path}")
    print(f"Summary written to {run_dir / 'supernet_summary.json'}")
    if feasible:
        best = feasible[0]
        print(
            f"Best feasible (inner-val): macro_f1={best['inner_val']['macro_f1']:.4f} "
            f"top1={best['inner_val']['top1']:.4f} "
            f"latency_est={best['hardware_estimate']['latency_ms']:.2f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
