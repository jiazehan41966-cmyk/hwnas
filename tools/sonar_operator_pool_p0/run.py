"""Phase 4A Sonar Operator Probe Pool P0 runner.

This runner is intentionally outside X_V0.1 NAS history. It evaluates single
operator x stage replacements on Dataset A train/valid only. The test split is
never constructed or evaluated here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "sonar_analysis"))

from detection_controlled_study import make_loader, seed_all, state_hash  # noqa: E402
from hwnas_fpga.runtime import load_config  # noqa: E402
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, DetectionLossConfig  # noqa: E402
from hwnas_fpga.training.trainer import create_optimizer  # noqa: E402
from hwnas_fpga.models.sonar_operator_pool_p0 import (  # noqa: E402
    D_ARCHITECTURE,
    D_ARCHITECTURE_ID,
    OPERATOR_DEFINITIONS,
    P0_OPERATORS,
    P0_STAGES,
    SCREENING_OPERATORS,
    build_d_baseline,
    build_p0_candidate,
    get_stage_block,
    model_info_with_p0_ops,
    p0_architecture_id,
)
from baseline import bind_dataset, evaluate, finite_outputs, gradient_summary, metrics_from_records, source_hashes  # noqa: E402

REPORT_DIR = ROOT / "reports" / "sonar_operator_pool_p0"
RESULTS_DIR = ROOT / "results" / "sonar_operator_pool_p0"
BASELINE_CONFIG = ROOT / "configs" / "baseline_sonar_frozen.yaml"
PHASE2F_MULTI = ROOT / "reports" / "phase2f_multiseed" / "multiseed_results.csv"
D_SEED42_FALLBACK = {
    "mAP50_95": 0.029121885943459346,
    "AP50": 0.07796538923867047,
    "AP75": 0.012663151923334226,
    "Precision": 0.02445302445302445,
    "Recall": 0.3838383838383838,
    "Params": 916103,
    "MACs": 1961715200,
    "source": "fallback_from_phase2f_reported_D_seed42",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def source_binding() -> dict[str, str]:
    paths = {
        "tools/sonar_operator_pool_p0/run.py": Path(__file__).resolve(),
        "src/hwnas_fpga/models/sonar_operator_pool_p0.py": ROOT / "src" / "hwnas_fpga" / "models" / "sonar_operator_pool_p0.py",
        "tests/test_sonar_operator_pool_p0.py": ROOT / "tests" / "test_sonar_operator_pool_p0.py",
        "configs/baseline_sonar_frozen.yaml": BASELINE_CONFIG,
    }
    result = {name: sha256(path) for name, path in paths.items() if path.exists()}
    result.update({f"baseline_source::{k}": v for k, v in source_hashes().items()})
    return result


def load_frozen_config() -> dict[str, Any]:
    config = load_config(str(BASELINE_CONFIG))
    if config["dataset"].get("name") != "roboflow_cylider2_v6_official_yolov8_detection":
        raise ValueError("Phase4A P0 is bound to Dataset A frozen baseline config")
    if config["training"].get("epochs") != 80 or config["project"].get("seed") != 42:
        raise ValueError("Unexpected baseline protocol drift")
    return config


def data_loaders(config: dict[str, Any], workers: int, seed: int = 42):
    root = resolve(config["dataset"]["data_dir"])
    train = make_loader(config, root, "train", seed, workers, train=True)
    valid = make_loader(config, root, "valid", seed, workers, train=False)
    return train, valid


def loss_for(config: dict[str, Any], device: str) -> DenseAnchorDetectionLoss:
    criterion = DenseAnchorDetectionLoss(DetectionLossConfig.from_config(config))
    return criterion.to(device)


def set_run_seed(seed: int) -> None:
    seed_all(seed)


def d_baseline_reference() -> dict[str, Any]:
    if PHASE2F_MULTI.exists():
        rows = read_csv(PHASE2F_MULTI)
        candidates = [r for r in rows if r.get("architecture_id") == D_ARCHITECTURE_ID and str(r.get("seed")) == "42"]
        if candidates:
            row = candidates[0]
            def f(*names: str, default: float = 0.0) -> float:
                for name in names:
                    if name in row and row[name] not in ("", None):
                        return float(row[name])
                return default
            return {
                "mAP50_95": f("mAP50_95", "best_mAP50_95", "best_valid_mAP50_95"),
                "AP50": f("AP50"),
                "AP75": f("AP75"),
                "Precision": f("Precision"),
                "Recall": f("Recall"),
                "Params": int(f("Params", "params", default=0.0)),
                "MACs": int(f("MACs", "macs", default=0.0)),
                "source": str(PHASE2F_MULTI.relative_to(ROOT)).replace("\\", "/"),
            }
    return dict(D_SEED42_FALLBACK)


def p0_matrix_rows() -> list[dict[str, Any]]:
    rows = []
    for operator in SCREENING_OPERATORS:
        for stage in P0_STAGES:
            rows.append({
                "architecture_id": p0_architecture_id(operator, stage),
                "baseline_architecture_id": D_ARCHITECTURE_ID,
                "operator": operator,
                "stage": f"stage{stage}",
                "stage_index": stage,
                "seed": 42,
                "epochs": 80,
                "single_replacement_only": True,
                "test_evaluated": False,
            })
    return rows


def operator_def_rows() -> list[dict[str, Any]]:
    return [{"operator": key, **asdict(value)} for key, value in OPERATOR_DEFINITIONS.items()]


def write_protocol_files() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pool_yaml = {
        "phase": "4A",
        "name": "Sonar Operator Probe Pool P0",
        "belongs_to_x_v0_1": False,
        "frozen_baseline": {
            "architecture_id": D_ARCHITECTURE_ID,
            "architecture": D_ARCHITECTURE,
            "dataset": "Dataset A / Cylinder2 train-valid only",
            "test_policy": "sealed; test_evaluated=false",
        },
        "operators": operator_def_rows(),
        "representative_stages": list(P0_STAGES),
        "screening_matrix": "5 non-baseline operators x 4 stages = 20 configurations",
        "training": {
            "seed": 42,
            "epochs": 80,
            "from_scratch": True,
            "protocol_source": str(BASELINE_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (REPORT_DIR / "SONAR_OPERATOR_POOL_P0.yaml").write_text(yaml.safe_dump(pool_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    definitions = [
        "# Operator Definitions\n",
        "These operators are Phase 4A probes only. They are not admitted into X_V0.1 and do not change historical V0.1 results.\n",
    ]
    for row in operator_def_rows():
        definitions.append(f"## {row['operator']}\n")
        definitions.append(f"- Hypothesis: {row['hypothesis']}\n")
        definitions.append(f"- Formula: `{row['formula']}`\n")
        definitions.append(f"- Deploy note: {row['deploy_note']}\n")
        definitions.append(f"- Geometry adaptive: {row['geometry_adaptive']}\n")
    (REPORT_DIR / "OPERATOR_DEFINITIONS.md").write_text("\n".join(definitions), encoding="utf-8")

    protocol = f"""# Phase 4A P0 Protocol

Frozen baseline: `{D_ARCHITECTURE_ID}`.

The experiment replaces exactly one MBConv spatial/depthwise operator at one representative stage. Stem, expansion, projection, channels, strides, neck, head, anchors, loss, train split, valid split and evaluator stay frozen to `{BASELINE_CONFIG.relative_to(ROOT)}`.

Representative stages: stage1, stage2, stage5, stage7.

Formal screening: seed42, 80 epochs, independent random initialization, train/valid only. Dataset A test is not constructed and `test_evaluated=false`.

Screening result is a single-seed probe. It must not be claimed as a stable sonar prior. At most one stage per non-baseline operator family may be recommended for later seed43/44 confirmation.
"""
    (REPORT_DIR / "P0_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    write_csv(REPORT_DIR / "OPERATOR_STAGE_MATRIX.csv", p0_matrix_rows(), ["architecture_id", "baseline_architecture_id", "operator", "stage", "stage_index", "seed", "epochs", "single_replacement_only", "test_evaluated"])
    dump_json(REPORT_DIR / "P0_MANIFEST.json", {
        "phase": "4A",
        "created_at": now(),
        "baseline_config_sha256": sha256(BASELINE_CONFIG),
        "source_sha256": source_binding(),
        "operator_count": len(P0_OPERATORS),
        "screening_configurations": len(p0_matrix_rows()),
        "test_evaluated": False,
        "status": "PROTOCOL_PREPARED",
    })


def dw3_equivalence(device: str = "cpu") -> dict[str, Any]:
    set_run_seed(42)
    left = build_d_baseline().to(device).eval()
    right = build_p0_candidate("DW3", 7).to(device).eval()
    right.load_state_dict(left.state_dict(), strict=True)
    x = torch.randn(1, 1, 640, 640, device=device)
    with torch.inference_mode():
        a = left(x)["predictions"]
        b = right(x)["predictions"]
    max_err = max(float((u - v).abs().max().detach().cpu()) for u, v in zip(a, b))
    mean_err = float(torch.stack([(u - v).abs().mean().detach().cpu() for u, v in zip(a, b)]).mean())
    return {"pass": max_err == 0.0 and mean_err == 0.0, "max_abs_error": max_err, "mean_abs_error": mean_err, "state_dict_strict": True}


def smoke(args: argparse.Namespace) -> None:
    config = load_frozen_config()
    set_run_seed(42)
    train_loader, valid_loader = data_loaders(config, args.workers, 42)
    criterion = loss_for(config, args.device)
    rows: list[dict[str, Any]] = []
    operators = list(P0_OPERATORS) if args.operator == "all" else [args.operator]
    stages = list(P0_STAGES) if args.stage == "all" else [int(args.stage)]
    for operator in operators:
        for stage in stages:
            set_run_seed(42)
            model = build_p0_candidate(operator, stage).to(args.device)
            model.train()
            images, targets = next(iter(train_loader))
            optimizer = create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
            optimizer.zero_grad(set_to_none=True)
            output = model(images.to(args.device))
            finite_outputs(output)
            shapes = [list(t.shape) for t in output["predictions"]]
            loss, parts = criterion(output, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite smoke loss for {operator} stage{stage}")
            loss.backward()
            grads = gradient_summary(model)
            validation, _ = evaluate(model, valid_loader, criterion, config, args.device, max_batches=1)
            info = model_info_with_p0_ops(model, (1, 1, config["dataset"]["image_size"], config["dataset"]["image_size"]))
            rows.append({
                "operator": operator,
                "stage": f"stage{stage}",
                "architecture_id": p0_architecture_id(operator, stage),
                "prediction_shapes": json.dumps(shapes),
                "loss": float(loss.detach().cpu()),
                "box_loss": float(parts.get("box_loss", 0.0)),
                "objectness_loss": float(parts.get("objectness_loss", 0.0)),
                "class_loss": float(parts.get("class_loss", 0.0)),
                "valid_mAP50_95_one_batch": validation["mAP50_95"],
                "Params": info["total_params"],
                "MACs": info["estimated_macs"],
                "finite_gradients": True,
                "test_evaluated": False,
            })
            print(f"SMOKE {operator} stage{stage} loss={float(loss):.6f}", flush=True)
    write_csv(REPORT_DIR / "smoke_validation.csv", rows)
    unit_md = [
        "# Operator Unit Tests\n",
        "Static pytest command: `pytest tests/test_sonar_operator_pool_p0.py -q`.\n",
        "Smoke runner checks shape, forward, loss, backward, gradient finiteness, one-batch validation and MAC/parameter estimation.\n",
        f"DW3 baseline equivalence: `{dw3_equivalence(args.device)}`.\n",
        f"Smoke rows: {len(rows)}. Test accessed: false.\n",
    ]
    (REPORT_DIR / "OPERATOR_UNIT_TESTS.md").write_text("\n".join(unit_md), encoding="utf-8")
    manifest_path = REPORT_DIR / "P0_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({"smoke_status": "PASS", "smoke_rows": len(rows), "dw3_equivalence": dw3_equivalence(args.device), "test_evaluated": False, "updated_at": now()})
    dump_json(manifest_path, manifest)


def run_dir(operator: str, stage: int, output_root: Path = RESULTS_DIR) -> Path:
    return output_root / p0_architecture_id(operator, stage)


def train_one(args: argparse.Namespace) -> None:
    operator = args.operator
    stage = int(args.stage)
    if operator not in SCREENING_OPERATORS:
        raise ValueError("Formal screening train expects one non-baseline operator")
    config = load_frozen_config()
    epochs = int(args.epochs)
    output_root = resolve(args.output_root)
    out = run_dir(operator, stage, output_root)
    out.mkdir(parents=True, exist_ok=True)
    set_run_seed(42)
    train_loader, valid_loader = data_loaders(config, args.workers, 42)
    train_binding = bind_dataset(train_loader)
    valid_binding = bind_dataset(valid_loader)
    model = build_p0_candidate(operator, stage).to(args.device)
    criterion = loss_for(config, args.device)
    optimizer = create_optimizer(model, config["training"]["optimizer"], config["training"]["lr"], config["training"]["weight_decay"])
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best = -float("inf")
    latest = out / "latest.pt"
    binding = {
        "architecture_id": p0_architecture_id(operator, stage),
        "operator": operator,
        "stage": stage,
        "baseline_architecture_id": D_ARCHITECTURE_ID,
        "config_sha256": sha256(BASELINE_CONFIG),
        "source_sha256": source_binding(),
        "train_binding": train_binding,
        "valid_binding": valid_binding,
        "seed": 42,
        "test_evaluated": False,
    }
    if latest.exists() and not args.restart:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("binding") != binding:
            raise ValueError("Resume binding mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        start_epoch = int(state["epoch"])
        best = max((r["valid"]["mAP50_95"] for r in history), default=-float("inf"))
    elif latest.exists() and args.restart:
        raise FileExistsError(f"Refusing to overwrite existing run; remove or choose new output: {out}")

    if not (out / "model_info.json").exists():
        info = model_info_with_p0_ops(model, (1, 1, config["dataset"]["image_size"], config["dataset"]["image_size"]))
        dump_json(out / "model_info.json", info)
    else:
        info = json.loads((out / "model_info.json").read_text(encoding="utf-8"))
    dump_json(out / "run_binding.json", binding)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        parts_acc: dict[str, list[float]] = {}
        epoch_start = time.perf_counter()
        for batch_index, (images, targets) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            output = model(images.to(args.device))
            finite_outputs(output)
            loss, parts = criterion(output, targets)
            if not torch.isfinite(loss) or not all(math.isfinite(float(v)) for v in parts.values()):
                raise ValueError(f"nonfinite training loss for {operator} stage{stage}")
            loss.backward()
            gradient_summary(model)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_acc.setdefault(key, []).append(float(value))
        valid, _records = evaluate(model, valid_loader, criterion, config, args.device, max_batches=args.max_valid_batches)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_batches": len(losses),
            "loss_parts": {k: float(np.mean(v)) for k, v in parts_acc.items()},
            "valid": valid,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (out / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        state = {
            "binding": binding,
            "epoch": epoch + 1,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_info": info,
            "updated_at": now(),
            "test_evaluated": False,
        }
        torch.save(state, latest)
        if valid["mAP50_95"] > best:
            best = valid["mAP50_95"]
            torch.save(state, out / "best.pt")
            dump_json(out / "best_valid_metrics.json", valid)
        print(f"{operator} stage{stage} epoch={epoch+1}/{epochs} mAP50_95={valid['mAP50_95']:.6f}", flush=True)
    best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    final = history[-1]["valid"]
    status = {
        "status": "COMPLETE" if epochs == 80 and args.max_train_batches is None and args.max_valid_batches is None else "SMOKE_OR_PARTIAL",
        "architecture_id": p0_architecture_id(operator, stage),
        "operator": operator,
        "stage": stage,
        "seed": 42,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_state["epoch"],
        "best_valid": best_state["history"][best_state["epoch"] - 1]["valid"],
        "final_valid": final,
        "checkpoint": str((out / "best.pt").resolve()),
        "checkpoint_sha256": sha256(out / "best.pt"),
        "latest_checkpoint_sha256": sha256(latest),
        "model_info": info,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "test_evaluated": False,
    }
    dump_json(out / "status.json", status)


def aggregate(args: argparse.Namespace) -> None:
    output_root = resolve(args.output_root)
    baseline = d_baseline_reference()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for operator in SCREENING_OPERATORS:
        for stage in P0_STAGES:
            aid = p0_architecture_id(operator, stage)
            status_path = output_root / aid / "status.json"
            if not status_path.exists():
                missing.append(aid)
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            best = status["best_valid"]
            info = status["model_info"]
            row = {
                "architecture_id": aid,
                "operator": operator,
                "stage": f"stage{stage}",
                "stage_index": stage,
                "seed": 42,
                "epochs_completed": status["epochs_completed"],
                "run_status": status["status"],
                "best_epoch": status["best_epoch"],
                "mAP50_95": best["mAP50_95"],
                "AP50": best["AP50"],
                "AP75": best["AP75"],
                "Precision": best["Precision"],
                "Recall": best["Recall"],
                "delta_mAP50_95_vs_D_seed42": best["mAP50_95"] - baseline["mAP50_95"],
                "delta_AP50_vs_D_seed42": best["AP50"] - baseline["AP50"],
                "Params": info["total_params"],
                "MACs": info["estimated_macs"],
                "FLOPs_2x_MACs": info["estimated_FLOPs_2x_MACs"],
                "checkpoint_sha256": status["checkpoint_sha256"],
                "test_evaluated": False,
            }
            rows.append(row)
    write_csv(REPORT_DIR / "P0_SCREENING_RESULTS.csv", rows, [
        "architecture_id", "operator", "stage", "stage_index", "seed", "epochs_completed", "run_status", "best_epoch",
        "mAP50_95", "AP50", "AP75", "Precision", "Recall", "delta_mAP50_95_vs_D_seed42", "delta_AP50_vs_D_seed42",
        "Params", "MACs", "FLOPs_2x_MACs", "checkpoint_sha256", "test_evaluated",
    ])
    matrix_rows = []
    for operator in SCREENING_OPERATORS:
        r = {"operator": operator}
        for stage in P0_STAGES:
            match = next((x for x in rows if x["operator"] == operator and x["stage_index"] == stage), None)
            r[f"stage{stage}_delta_mAP50_95"] = None if match is None else match["delta_mAP50_95_vs_D_seed42"]
            r[f"stage{stage}_mAP50_95"] = None if match is None else match["mAP50_95"]
        matrix_rows.append(r)
    write_csv(REPORT_DIR / "OPERATOR_STAGE_MATRIX.csv", matrix_rows)
    recommendations = []
    for operator in SCREENING_OPERATORS:
        family_rows = [r for r in rows if r["operator"] == operator and r["run_status"] == "COMPLETE"]
        if not family_rows:
            continue
        top = max(family_rows, key=lambda r: (r["delta_mAP50_95_vs_D_seed42"], -r["stage_index"]))
        recommendations.append({
            "operator": operator,
            "recommended_pair": top["architecture_id"] if top["delta_mAP50_95_vs_D_seed42"] > 0 else None,
            "top_stage": top["stage"],
            "top_delta_mAP50_95": top["delta_mAP50_95_vs_D_seed42"],
            "recommendation_rule": "top positive single-seed delta only; max one stage per operator family; requires seed43/44 confirmation",
        })
    dump_json(REPORT_DIR / "P0_MANIFEST.json", {
        "phase": "4A",
        "updated_at": now(),
        "baseline_reference": baseline,
        "screening_complete": len(rows) == 20 and not missing and all(r["run_status"] == "COMPLETE" for r in rows),
        "completed_rows": len(rows),
        "missing": missing,
        "recommendations": recommendations,
        "test_evaluated": False,
        "source_sha256": source_binding(),
    })
    if missing:
        print(f"AGGREGATE incomplete; missing={len(missing)}", flush=True)
    else:
        print("AGGREGATE complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "smoke", "train", "aggregate"])
    parser.add_argument("--operator", default="all", help="Operator name, or all for smoke")
    parser.add_argument("--stage", default="all", help="Stage index, or all for smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-root", default=str(RESULTS_DIR))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        write_protocol_files()
        print(f"PREPARED {REPORT_DIR}")
    elif args.mode == "smoke":
        write_protocol_files()
        smoke(args)
        print(f"SMOKE PASS {REPORT_DIR}")
    elif args.mode == "train":
        if args.operator == "all" or args.stage == "all":
            raise SystemExit("train requires one --operator and one --stage")
        train_one(args)
    elif args.mode == "aggregate":
        aggregate(args)


if __name__ == "__main__":
    main()

