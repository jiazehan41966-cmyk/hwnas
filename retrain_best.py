#!/usr/bin/env python3
"""对搜索出的最优架构做完整训练并保存结果."""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.data.dataset import create_nksid_dataloaders
from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec
from hwnas_fpga.training import train_model


def parse_args():
    p = argparse.ArgumentParser(description="对搜索候选做完整训练")
    p.add_argument("--candidate-json", required=True, help="候选架构 JSON 路径")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--image-size",     type=int,   default=224)
    p.add_argument("--fold",           type=int,   default=0)
    p.add_argument("--num-workers",    type=int,   default=0)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--early-stopping", type=int,   default=5)
    p.add_argument("--output-dir",     default="results/retrain")
    p.add_argument("--device",         default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 设备 ──
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── 加载架构 ──
    with open(args.candidate_json) as f:
        payload = json.load(f)
    arch_id  = payload["candidate"]["arch_id"]
    encoding = payload["candidate"]["encoding"]
    arch     = ArchitectureSpec.from_dict(encoding)

    # ── 数据 ──
    train_loader, val_loader, class_weights = create_nksid_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold,
        image_size=args.image_size,
    )

    # ── 建模 ──
    model  = build_model(arch, num_classes=8).to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"Arch      : {arch_id}")
    print(f"Params    : {params:,}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"Train/Val : {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print()

    # ── 训练 ──
    t0 = time.time()
    final_acc, history = train_model(
        model=model,
        train_loader=train_loader,
        num_classes=8,
        epochs=args.epochs,
        device=device,
        val_loader=val_loader,
        class_weights=class_weights,
        early_stopping_patience=args.early_stopping,
        lr=args.lr,
    )
    elapsed = time.time() - t0

    best_acc = max(history.get("val_acc", [final_acc]))
    best_ep  = history.get("val_acc", [final_acc]).index(best_acc) + 1

    print()
    print("=" * 50)
    print(f"Final val acc : {final_acc:.4f}  ({final_acc*100:.1f}%)")
    print(f"Best  val acc : {best_acc:.4f}  ({best_acc*100:.1f}%) @ epoch {best_ep}")
    print(f"Training time : {elapsed:.0f}s  ({elapsed/60:.1f} min)")

    # ── 保存结果 ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "arch_id":    arch_id,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "config": {
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "image_size":  args.image_size,
            "fold":        args.fold,
            "lr":          args.lr,
            "device":      str(device),
            "data_dir":    args.data_dir,
        },
        "model": {
            "params":      params,
            "encoding":    encoding,
        },
        "results": {
            "final_val_acc":   final_acc,
            "best_val_acc":    best_acc,
            "best_epoch":      best_ep,
            "training_time_s": elapsed,
        },
        "history": history,
        "search_metrics": payload["candidate"].get("metrics", {}),
    }

    out_file = out_dir / f"{arch_id}_retrain.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
