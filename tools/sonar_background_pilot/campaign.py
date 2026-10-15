"""Bounded, isolated four-arm comparison using train/valid only.

Run from a frozen snapshot root. The queue owns only its new output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch
from scripts.detection_controlled_study import (
    dataset_binding, dump, infer_records, make_loader, make_model, metric,
    seed_all, select_threshold, sha256, state_hash, summarize_errors,
)
from scripts.diagnose_detection_errors import recall_at_fppi
from hwnas_fpga.runtime import load_config
from hwnas_fpga.training.detection_trainer import (
    DetectionLossConfig, DenseAnchorDetectionLoss, train_detection_model,
)
from hwnas_fpga.training.trainer import create_optimizer
from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco
from tools.sonar_background_pilot.losses import BackgroundDetectionLoss, BackgroundOptions


VARIANTS = ("baseline", "rank", "pisa", "fg_bg")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def verify_snapshot(campaign):
    manifest = json.loads((campaign / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"frozen source mismatch: {name}")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(config, seed, variant, device):
    seed_all(seed)
    model, architecture = make_model(config)
    initial_hash = state_hash(model)
    model.to(device)
    criterion = BackgroundDetectionLoss(DetectionLossConfig.from_config(config), BackgroundOptions(variant=variant)).to(device)
    if variant == "fg_bg":
        model.neck.register_forward_hook(criterion.capture_features)
    return model, criterion, architecture, initial_hash


def preflight(args):
    campaign = Path(args.campaign).resolve()
    source_hash = verify_snapshot(campaign)
    config = load_config(str(campaign / "config.yaml"))
    torch.set_num_threads(4)
    if args.synthetic:
        config["dataset"]["image_size"] = 64
        config["detection"]["input_size"] = 64
        seed_all(42)
        images = torch.randn(2, 1, 64, 64)
        targets = [{"boxes": torch.tensor([[20., 20., 40., 40.]]), "labels": torch.tensor([i % 2])}
                   for i in range(2)]
        bindings = None
    else:
        loader = make_loader(config, args.data_dir, "train", 42, 0, train=True)
        images, targets = next(iter(loader))
        bindings = {"train": dataset_binding(loader),
                    "valid": dataset_binding(make_loader(config, args.data_dir, "valid", 42, 0))}
    evidence = {"source_sha256": source_hash, "synthetic": args.synthetic, "device": args.device,
                "torch": torch.__version__, "python": sys.version, "test_accessed": False,
                "data_bindings": bindings, "variants": {}, "timestamp": timestamp()}
    initial_hashes = []
    for variant in VARIANTS:
        model, criterion, architecture, initial_hash = build(config, 42, variant, args.device)
        initial_hashes.append(initial_hash)
        model.train()
        images_device = images.to(args.device)
        started = time.monotonic()
        output = model(images_device)
        loss, parts = criterion(output, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {variant} loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"nonfinite {variant} gradient")
        optimizer = create_optimizer(model, optimizer_name=config["training"]["optimizer"],
                                     lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"nonfinite {variant} parameter after update")
        evidence["variants"][variant] = {"initial_state_sha256": initial_hash,
             "loss": float(loss.detach()), "parts": parts, "parameters": sum(p.numel() for p in model.parameters()),
             "step_seconds": time.monotonic() - started, "auxiliary": criterion.last_auxiliary}
        if variant == "baseline":
            model.eval()
            with torch.no_grad():
                output = model(images_device)
                actual = criterion(output, targets)[0]
                reference = DenseAnchorDetectionLoss(criterion.config).to(args.device)(output, targets)[0]
            if not torch.equal(actual, reference):
                raise RuntimeError("baseline parity failed")
        del model, criterion, optimizer, output, loss, grads
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(set(initial_hashes)) != 1:
        raise RuntimeError("paired initialization hashes differ")
    evidence["status"] = "PASS"
    dump(campaign / ("preflight_synthetic.json" if args.synthetic else "preflight_cuda.json"), evidence)
    print(json.dumps(evidence), flush=True)


def run(args):
    campaign = Path(args.campaign).resolve()
    source_hash = verify_snapshot(campaign)
    config = load_config(str(campaign / "config.yaml"))
    output = campaign / "runs" / f"{args.variant}_seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config["project"]["seed"] = args.seed
    model, criterion, architecture, initial_hash = build(config, args.seed, args.variant, args.device)
    train = make_loader(config, args.data_dir, "train", args.seed, args.workers, train=True)
    valid = make_loader(config, args.data_dir, "valid", args.seed, args.workers)
    bindings = {"train": dataset_binding(train), "valid": dataset_binding(valid)}
    manifest = {"started_at": timestamp(), "seed": args.seed, "variant": args.variant,
                "epochs": args.epochs, "config_sha256": sha256(campaign / "config.yaml"),
                "source_sha256": source_hash, "architecture": architecture,
                "initial_state_sha256": initial_hash, "data_bindings": bindings,
                "torch": torch.__version__, "python": sys.version,
                "options": asdict(criterion.options), "test_accessed": False,
                "selection": "maximum validation mean(AP50,AP75); same for all arms",
                "precision": "float32; existing trainer does not use the YAML amp flag",
                "paper_status": "paper-inspired adaptation; not full author reproduction",
                "command": sys.argv}
    dump(output / "run_manifest.json", manifest)
    expected = json.loads((campaign / "preflight_cuda.json").read_text())["data_bindings"]
    if bindings != expected:
        raise RuntimeError("dataset binding differs from real-data preflight")
    optimizer = create_optimizer(model, optimizer_name=config["training"]["optimizer"],
                                 lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    started = time.monotonic()
    history = train_detection_model(model, train, valid, criterion=criterion,
        optimizer=optimizer, device=args.device, epochs=args.epochs, score_threshold=0.01,
        nms_iou_threshold=0.5, pre_nms_topk=500, max_detections=50,
        checkpoint_path=output / "checkpoints/best_model.pt", epoch_log_path=output / "epoch_metrics.jsonl")
    dump(output / "history.json", history)
    rows = infer_records(model, valid, criterion, args.device)
    with (output / "valid_predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    names = bindings["valid"]["class_names"]
    sweep, selected = select_threshold(rows, len(names))
    diagnostics = summarize_errors(rows, names, selected["score_threshold"])
    fppi = recall_at_fppi(rows, names)
    coco = evaluate_detection_predictions_coco([r["prediction"] for r in rows],
        [r["target"] for r in rows], num_classes=len(names), class_names=names)
    status = {"status": "COMPLETE", "completed_at": timestamp(), "seed": args.seed,
              "variant": args.variant, "epochs": len(history["epochs"]),
              "best_epoch": history["best_epoch"], "best_validation": history["best_eval"],
              "validation_ap10_allpoints": metric(rows, len(names), ious=tuple(round(.5 + i * .05, 2) for i in range(10))),
              "ap10_note": "all-points integration, NOT official COCO101 mAP",
              "validation_coco101": coco,
              "coco_note": "101-point evaluator on cached score>=0.01, global max50 predictions; not unrestricted COCO output",
              "validation_operating_point": selected, "threshold_sweep": sweep,
              "errors": diagnostics, "recall_at_fppi": fppi["selected"],
              "test_accessed": False, "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": sha256(output / "checkpoints/best_model.pt")}
    dump(output / "status.json", status)
    print("COMPLETE", args.variant, args.seed, selected["f1"], flush=True)


def queue(args):
    campaign = Path(args.campaign).resolve()
    verify_snapshot(campaign)
    import fcntl
    locks = []
    for gpu in args.gpus:
        lock = (Path.home() / f".sonar_background_pilot_gpu{gpu}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        memory = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                              "--format=csv,noheader,nounits"], text=True).strip())
        if memory >= 500:
            raise RuntimeError(f"GPU {gpu} is occupied; no training started")
    jobs = [{"variant": variant, "seed": seed} for seed in args.seeds for variant in VARIANTS]
    state = {"status": "RUNNING", "started_at": timestamp(), "jobs": [], "test_accessed": False,
             "budget": {"epochs_per_job": args.epochs, "jobs": len(jobs), "seeds": args.seeds}}
    active = []
    failed = False
    def save():
        state["updated_at"] = timestamp()
        temp = campaign / "queue_status.tmp"
        dump(temp, state)
        temp.replace(campaign / "queue_status.json")
    while jobs or active:
        for gpu in args.gpus:
            while jobs and sum(item[2]["gpu"] == gpu for item in active) < args.parallel_per_gpu and not failed:
                job = jobs.pop(0)
                name = f"{job['variant']}_seed{job['seed']}"
                path = campaign / "logs" / f"{name}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("x")
                command = [sys.executable, "-m", "tools.sonar_background_pilot.campaign", "run",
                    "--campaign", str(campaign), "--data-dir", args.data_dir,
                    "--variant", job["variant"], "--seed", str(job["seed"]),
                    "--epochs", str(args.epochs), "--workers", "2", "--device", "cuda"]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED="0",
                                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                record = dict(job, gpu=gpu, pid=proc.pid, status="RUNNING", command=command, started_at=timestamp())
                state["jobs"].append(record)
                active.append((proc, handle, record))
                save()
        for entry in list(active):
            proc, handle, record = entry
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            result = campaign / "runs" / f"{record['variant']}_seed{record['seed']}" / "status.json"
            record.update(status="COMPLETE" if rc == 0 and result.is_file() else "FAILED",
                          returncode=rc, completed_at=timestamp())
            failed |= record["status"] == "FAILED"
            active.remove(entry)
            save()
        if failed and not active:
            break
        time.sleep(3)
    state["status"] = "FAILED" if failed else "COMPLETE"
    state["unstarted_jobs"] = jobs
    save()
    if not failed:
        compare(campaign)


def compare(campaign):
    import numpy as np
    statuses = [json.loads(path.read_text()) for path in sorted((campaign / "runs").glob("*/status.json"))]
    baseline = {row["seed"]: row for row in statuses if row["variant"] == "baseline"}
    summary = {"created_at": timestamp(), "test_accessed": False, "variants": {},
               "primary": "paired validation mean(AP50,AP75)", "limitation": "3 seeds, one split, short training pilot"}
    for variant in VARIANTS:
        rows = [row for row in statuses if row["variant"] == variant]
        paired = []
        for row in rows:
            base = baseline[row["seed"]]
            manifest = json.loads((campaign / "runs" / f"{variant}_seed{row['seed']}" / "run_manifest.json").read_text())
            bm = json.loads((campaign / "runs" / f"baseline_seed{row['seed']}" / "run_manifest.json").read_text())
            for key in ("initial_state_sha256", "config_sha256", "source_sha256", "data_bindings", "epochs"):
                if manifest[key] != bm[key]:
                    raise RuntimeError(f"paired invariant failed: {variant}, {key}")
            op, bop = row["validation_operating_point"], base["validation_operating_point"]
            paired.append({"seed": row["seed"], "map": row["best_validation"]["map"],
                           "map_delta": row["best_validation"]["map"] - base["best_validation"]["map"],
                           "precision": op["precision"], "recall": op["recall"], "f1": op["f1"],
                           "f1_delta": op["f1"] - bop["f1"], "best_epoch": row["best_epoch"],
                           "recall_at_fppi": row["recall_at_fppi"]})
        summary["variants"][variant] = {"runs": paired,
            "mean_map": float(np.mean([row["map"] for row in paired])),
            "mean_map_delta": float(np.mean([row["map_delta"] for row in paired])),
            "positive_map_seeds": sum(row["map_delta"] > 0 for row in paired),
            "mean_f1": float(np.mean([row["f1"] for row in paired])),
            "mean_f1_delta": float(np.mean([row["f1_delta"] for row in paired]))}
    dump(campaign / "comparison.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "preflight", "run", "queue", "compare"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--parallel-per-gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps({"source_sha256": verify_snapshot(Path(args.campaign).resolve()), "status": "PASS"}))
    elif args.mode == "preflight":
        preflight(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "queue":
        queue(args)
    else:
        compare(Path(args.campaign).resolve())


if __name__ == "__main__":
    main()
