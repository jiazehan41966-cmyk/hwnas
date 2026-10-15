"""Official-recipe YOLOv8n/YOLO11n train/val references; the test split stays sealed.

Example (24 Cylinder2 runs; use a single item in each list for one run)::

    python scripts/run_reference_baselines.py --data DATA.yaml --name cylinder2 \
        --models yolov8n yolo11n --init coco scratch --imgsz 640 1280 --seeds 42 43 44

The supplied reference script used iou=0.6 only at final evaluation. Here training
validation and both final evaluations use the same NMS IoU (official default 0.7).
Only validation mAP50:95 selects best_map.pt. Unaugmented train evaluation is a
diagnostic, never a selection input. Metrics are fractions, not percentages.
The official best.pt is retained separately; no optimizer/loss is replaced.

No Ultralytics import occurs for --dry-run or --aggregate-only. PyYAML is needed
to inspect a data YAML. Existing incomplete runs require --resume and a usable
checkpoint; they are never silently restarted. Use a new --project for a changed
protocol. Single-device execution is deliberate, so callbacks remain in-process.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback


PINNED_ULTRALYTICS = "8.3.221"
MAP_KEY = "metrics/mAP50-95(B)"
SUMMARY_FIELDS = ["dataset", "run", "model", "init", "imgsz", "seed", "best_epoch",
                  "mAP50_95", "AP50", "AP75", "train_mAP50_95", "train_AP50", "train_AP75",
                  "Precision_maxF1", "Recall_maxF1", "weights", "weights_sha256"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(canonical(value), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--models", nargs="+", choices=["yolov8n", "yolo11n"], default=["yolov8n", "yolo11n"])
    parser.add_argument("--init", nargs="+", choices=["coco", "scratch"], default=["coco", "scratch"])
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16, help="640 batch; >640 uses max(4, batch//4)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", default="results/reference_baselines")
    parser.add_argument("--weights-dir", type=Path, help="Existing MODEL.pt files; absent files fail without downloading")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete matching runs; never restart from zero")
    parser.add_argument("--dry-run", action="store_true", help="Read and print plan without importing Ultralytics or writing files")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild summary from completed run results")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in {".", ".."}:
        parser.error("--name must be a simple dataset identifier")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.patience < 0:
        parser.error("epochs/batch must be positive; workers/patience must be nonnegative")
    if any(size < 32 or size % 32 for size in args.imgsz) or not 0 < args.iou < 1:
        parser.error("imgsz must be positive multiples of 32; iou must be in (0,1)")
    if any(seed < 0 for seed in args.seeds) or "," in args.device:
        parser.error("seeds must be nonnegative; only one device per process is supported")
    return args


def matrix(args):
    seen = set()
    for model, init, size, seed in itertools.product(args.models, args.init, args.imgsz, args.seeds):
        name = f"{model}_{init}_{size}_s{seed}"
        if name in seen:
            continue
        seen.add(name)
        yield {"run": name, "model": model, "init": init, "imgsz": size, "seed": seed,
               "batch": args.batch if size <= 640 else max(4, args.batch // 4)}


def prepare_data(source: Path):
    """Whitelist train/val metadata; never resolve, enumerate or open a test path."""
    import yaml

    source = source.resolve(strict=True)
    original = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(original, dict) or "names" not in original:
        raise ValueError("data YAML must contain train, val and names")
    root = Path(original.get("path") or source.parent)
    root = (source.parent / root).resolve() if not root.is_absolute() else root.resolve()
    safe = {"path": str(root), "names": original["names"]}
    for split in ("train", "val"):
        entries = original.get(split)
        if not entries:
            raise ValueError(f"data YAML missing {split}")
        entries = entries if isinstance(entries, list) else [entries]
        resolved = []
        for entry in entries:
            if not isinstance(entry, str) or "://" in entry:
                raise ValueError("Only explicit local train/val paths are allowed")
            candidate = Path(entry)
            if "test" in {part.lower() for part in candidate.parts}:
                raise ValueError(f"Refusing test path in {split}: {entry}")
            candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            if "test" in {part.lower() for part in candidate.parts}:
                raise ValueError(f"Refusing resolved test path in {split}")
            if not candidate.exists():
                raise FileNotFoundError(f"{split} path does not exist: {candidate}; fix data.yaml before running")
            if candidate.is_file():
                if candidate.suffix.lower() != ".txt":
                    raise ValueError("Split files must be local .txt image manifests")
                for line in candidate.read_text(encoding="utf-8-sig").splitlines():
                    if "://" in line or "test" in {p.lower() for p in Path(line.strip()).parts}:
                        raise ValueError(f"Refusing test/remote entry in {split} image manifest")
            resolved.append(str(candidate))
        safe[split] = resolved if isinstance(original[split], list) else resolved[0]
    if safe["train"] == safe["val"]:
        raise ValueError("train and val point to the same source")
    for key in ("nc", "channels"):
        if key in original:
            safe[key] = original[key]
    content = canonical(safe)  # JSON is valid YAML; deliberately has no test/download keys.
    return {"source_path": str(source), "source_sha256": sha256(source),
            "train_val_yaml_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "train_val_yaml": safe, "test_accessed": False}


def contract(args, item, data):
    return {"schema": 1, "dataset": args.name, **item, "epochs": args.epochs,
            "patience": args.patience, "device": args.device, "workers": args.workers,
            "iou": args.iou, "conf": 0.001, "max_det": 300,
            "data": data, "selection": MAP_KEY, "tie_break": "earliest_epoch",
            "required_ultralytics": PINNED_ULTRALYTICS,
            "weights_dir": str(args.weights_dir.resolve()) if args.weights_dir else None}


def existing_action(run_dir: Path, expected, resume=False):
    if not run_dir.exists() or not any(p.name != ".runner.lock" for p in run_dir.iterdir()):
        return "new"
    provenance = run_dir / "provenance.json"
    if not provenance.is_file() or read_json(provenance)["contract"] != expected:
        raise RuntimeError(f"Existing run has absent/different provenance: {run_dir}; use a new --project")
    result = run_dir / "result.json"
    if result.is_file():
        completed = read_json(result)
        checkpoint = run_dir / "weights" / "best_map.pt"
        if completed.get("status") != "complete" or sha256(checkpoint) != completed["weights_sha256"]:
            raise RuntimeError(f"Invalid completed result/checkpoint: {run_dir}")
        return "skip_complete"
    if not resume:
        raise RuntimeError(f"Incomplete run exists: {run_dir}; explicitly pass --resume")
    if (run_dir / "training_complete.json").is_file():
        if not (run_dir / "weights" / "best_map.pt").is_file():
            raise RuntimeError(f"Completed training has no best_map checkpoint: {run_dir}")
        return "resume_evaluation"
    if not (run_dir / "weights" / "last.pt").is_file():
        raise RuntimeError(f"No resumable last.pt: {run_dir}; use a new --project (no restart/overwrite)")
    return "resume_training"


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """OS releases this advisory lock on normal exit, exceptions and process death."""
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def aggregate(out_dir: Path):
    """Rebuild from all complete per-run files, including runs not in this invocation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out_dir / ".summary.lock"):
        rows = []
        for result_path in sorted(out_dir.glob("*/result.json")):
            result = read_json(result_path)
            if result.get("status") == "complete":
                rows.append({key: result.get(key) for key in SUMMARY_FIELDS})
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        target = out_dir / "summary.csv"
        temporary = target.with_name(f"summary.{os.getpid()}.tmp")
        temporary.write_text(buffer.getvalue(), encoding="utf-8")
        os.replace(temporary, target)
    return rows


class BestMapCheckpoint:
    """Called after last.pt has been saved, using that epoch's validation metric."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.record_path = run_dir / "best_map_selection.json"
        self.record = read_json(self.record_path) if self.record_path.exists() else None
        if self.record and sha256(run_dir / "weights" / "best_map.pt") != self.record["sha256"]:
            raise RuntimeError("best_map checkpoint/selection mismatch; refusing an unsafe resume")

    def __call__(self, trainer):
        score = float(trainer.metrics[MAP_KEY])
        if not math.isfinite(score):
            raise RuntimeError("Non-finite validation mAP; cannot select a checkpoint")
        if self.record and score <= self.record["validation_mAP50_95"]:
            return
        target = self.run_dir / "weights" / "best_map.pt"
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(trainer.last, temporary)
        os.replace(temporary, target)
        self.record = {"metric": MAP_KEY, "validation_mAP50_95": score,
                       "best_epoch": int(trainer.epoch) + 1, "epoch_numbering": "one_based",
                       "selection_split": "val", "sha256": sha256(target)}
        atomic_json(self.record_path, self.record)


def model_state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical([key, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def environment(torch, ultralytics):
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "packages": dict(sorted(versions.items()))}


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.original.write(value)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextlib.contextmanager
def run_logging(path):
    with path.open("a", encoding="utf-8", buffering=1) as log:
        stdout, stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        changed = []
        for handler in logging.getLogger("ultralytics").handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                changed.append((handler, handler.stream))
                handler.setStream(stdout if handler.stream is sys.stdout else stderr)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                yield
        finally:
            for handler, stream in changed:
                handler.setStream(stream)


def metrics_dict(metrics):
    return {"mAP50_95": float(metrics.box.map), "AP50": float(metrics.box.map50),
            "AP75": float(metrics.box.map75), "Precision_maxF1": float(metrics.box.mp),
            "Recall_maxF1": float(metrics.box.mr)}


def train_and_evaluate(args, item, expected, run_dir, action):
    # Disable optional telemetry in this process, without modifying user settings.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import callbacks, checks
    from ultralytics.utils.torch_utils import init_seeds

    if ultralytics.__version__ != PINNED_ULTRALYTICS:
        raise RuntimeError(f"Expected ultralytics=={PINNED_ULTRALYTICS}, got {ultralytics.__version__}")
    callbacks.add_integration_callbacks = lambda instance: None
    checks.check_pip_update_available = lambda: False
    # The imported alias in engine.model also needs suppression (no update query).
    import ultralytics.engine.model as engine_model
    engine_model.check_pip_update_available = lambda: False

    env = environment(torch, ultralytics)
    provenance_path = run_dir / "provenance.json"
    data_path = run_dir / "train_val_only.yaml"
    if action == "new":
        data_path.write_text(canonical(expected["data"]["train_val_yaml"]), encoding="utf-8")
        provenance = {"contract": expected, "environment": env, "argv": sys.argv,
                      "runner_sha256": sha256(Path(__file__)), "created_unix": time.time()}
        atomic_json(provenance_path, provenance)
    else:
        provenance = read_json(provenance_path)
        if provenance["environment"] != env or provenance["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("Environment or runner changed; resume requires the original environment and runner")
        if sha256(data_path) != expected["data"]["train_val_yaml_sha256"]:
            raise RuntimeError("Sanitized train/val YAML changed")

    selector = BestMapCheckpoint(run_dir)
    if action != "resume_evaluation":
        init_seeds(item["seed"], deterministic=True)
        if action == "resume_training":
            source = run_dir / "weights" / "last.pt"
        elif item["init"] == "coco":
            source = args.weights_dir.resolve() / f"{item['model']}.pt" if args.weights_dir else f"{item['model']}.pt"
            if args.weights_dir and not Path(source).is_file():
                raise FileNotFoundError(f"Pre-downloaded initialization is missing: {source}")
        else:
            source = f"{item['model']}.yaml"
        model = YOLO(str(source), task="detect")
        if action == "new":
            checkpoint = Path(getattr(model, "ckpt_path", "") or "__not_a_checkpoint__")
            provenance["initialization"] = {"source": str(source), "pretrained": item["init"] == "coco",
                "checkpoint_path": str(checkpoint.resolve()) if checkpoint.is_file() else None,
                "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
                "architecture": model.model.yaml,
                "architecture_sha256": hashlib.sha256(canonical(model.model.yaml).encode()).hexdigest()}
            if item["init"] == "coco" and not checkpoint.is_file():
                raise RuntimeError("Cannot verify COCO initialization checkpoint")
            atomic_json(provenance_path, provenance)

        def record_setup(trainer):
            record = {"actual_model_state_sha256": model_state_hash(trainer.model),
                      "args": vars(trainer.args), "start_epoch": int(trainer.start_epoch),
                      "optimizer": type(trainer.optimizer).__name__, "ema_enabled": trainer.ema is not None,
                      "train_images": len(trainer.train_loader.dataset),
                      "val_images": len(trainer.test_loader.dataset), "unix": time.time()}
            atomic_json(run_dir / f"trainer_setup_start{trainer.start_epoch}.json", record)

        def finished(trainer):
            atomic_json(run_dir / "training_complete.json", {"last_epoch": int(trainer.epoch) + 1,
                        "epochs_budget": args.epochs, "finished_unix": time.time()})

        model.add_callback("on_pretrain_routine_end", record_setup)
        model.add_callback("on_model_save", selector)
        model.add_callback("on_train_end", finished)
        train_args = dict(data=str(data_path), epochs=args.epochs, imgsz=item["imgsz"], batch=item["batch"],
                          seed=item["seed"], deterministic=True, pretrained=item["init"] == "coco",
                          patience=args.patience, close_mosaic=10, flipud=0.0,
                          project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                          device=args.device, workers=args.workers, plots=True, save=True,
                          val=True, split="val", conf=0.001, iou=args.iou, max_det=300)
        atomic_json(run_dir / f"train_call_{'resume' if action == 'resume_training' else 'initial'}.json", train_args)
        if action == "resume_training":
            model.train(resume=str(source), **train_args)
        else:
            model.train(**train_args)

    best = run_dir / "weights" / "best_map.pt"
    selection = read_json(run_dir / "best_map_selection.json")
    if sha256(best) != selection["sha256"]:
        raise RuntimeError("Selected checkpoint hash changed before final evaluation")
    evaluations = {}
    for split in ("val", "train"):
        model = YOLO(str(best), task="detect")
        evaluations[split] = metrics_dict(model.val(
            data=str(data_path), imgsz=item["imgsz"], batch=item["batch"],
            conf=0.001, iou=args.iou, max_det=300, split=split, device=args.device,
            workers=args.workers, augment=False, plots=False, save_json=False,
            project=str(run_dir), name=f"eval_{split}", exist_ok=True))
        atomic_json(run_dir / f"evaluation_{split}.json", {"split": split, "augment": False,
                    "used_for_selection": False, "metrics": evaluations[split], "weights_sha256": sha256(best)})
    result = {"status": "complete", "dataset": args.name, **item, **evaluations["val"],
              "train_mAP50_95": evaluations["train"]["mAP50_95"],
              "train_AP50": evaluations["train"]["AP50"], "train_AP75": evaluations["train"]["AP75"],
              "best_epoch": selection["best_epoch"], "selection": selection,
              "weights": str(best), "weights_sha256": sha256(best),
              "official_best_weights": str(run_dir / "weights" / "best.pt"),
              "evaluations": evaluations, "test_accessed": False, "finished_unix": time.time()}
    atomic_json(run_dir / "result.json", result)
    print(canonical(result))


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.project).resolve() / args.name
    if args.aggregate_only:
        if args.dry_run:
            raise ValueError("--aggregate-only writes summary; do not combine with --dry-run")
        print(canonical(aggregate(out_dir)))
        return 0
    data = prepare_data(Path(args.data))
    planned = [(item, contract(args, item, data)) for item in matrix(args)]
    # Validate the whole requested matrix before launching any expensive work.
    for item, expected in planned:
        run_dir = out_dir / item["run"]
        action = existing_action(run_dir, expected, args.resume)
        print(canonical({"run_dir": str(run_dir), "action": action, "contract": expected}))
    if args.dry_run:
        return 0
    for item, expected in planned:
        run_dir = out_dir / item["run"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(run_dir / ".runner.lock"):
            action = existing_action(run_dir, expected, args.resume)
            if action == "skip_complete":
                print(f"SKIP complete: {run_dir}")
            else:
                with run_logging(run_dir / "runner.log"):
                    atomic_json(run_dir / "status.json", {"status": "running", "action": action,
                                "pid": os.getpid(), "started_unix": time.time()})
                    try:
                        train_and_evaluate(args, item, expected, run_dir, action)
                    except BaseException as error:
                        atomic_json(run_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                    "error_type": type(error).__name__, "error": str(error), "unix": time.time()})
                        traceback.print_exc()
                        raise
                    atomic_json(run_dir / "status.json", {"status": "complete", "unix": time.time()})
        aggregate(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
