#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from enumerate_strict40_candidates import block_label, enumerate_architectures  # noqa: E402
from hwnas_fpga.models import build_model  # noqa: E402
from hwnas_fpga.runtime import create_data_pipeline, load_config, pick  # noqa: E402
from hwnas_fpga.search_space import ArchitectureSpec  # noqa: E402
from hwnas_fpga.training import train_model  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "strict40_proxy_multiseed_eval10_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled multi-seed proxy evaluation for strict40 candidates."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--candidate-manifest",
        default=None,
        help=(
            "Optional JSON manifest containing exact candidate architectures. "
            "When omitted, all architectures enumerated from --config are evaluated."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping so eval-epochs is actually used.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--selection-metric", default=None)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _candidate_label(architecture: Any) -> str:
    return " / ".join(block_label(stage.blocks[0]) for stage in architecture.stages[1:])


def _candidate_key(architecture: Any) -> str:
    return "|".join(
        f"s{stage_index}:{block_label(stage.blocks[0])}"
        for stage_index, stage in enumerate(architecture.stages)
    )


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std_pop": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std_pop": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "candidate_rank",
        "candidate_label",
        "candidate_key",
        "source_method",
        "source_arch_id",
        "observed_macro_f1",
        "observed_top1",
        "seed",
        "eval_epochs",
        "best_epoch",
        "macro_f1",
        "top1",
        "weighted_f1",
        "top5",
        "latency_ms",
        "lut",
        "dsp",
        "bram",
        "power_w",
        "feasible",
        "lut_hits",
        "lut_misses",
        "true_misses",
        "deferred_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def _load_candidate_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"candidate manifest has no candidates: {path}")

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError(f"candidate manifest item {index} is not an object")
        architecture_payload = item.get("architecture") or item.get("encoding")
        if not isinstance(architecture_payload, dict):
            raise ValueError(f"candidate manifest item {index} has no architecture")
        architecture = ArchitectureSpec.from_dict(architecture_payload)
        key = _candidate_key(architecture)
        if key in seen_keys:
            raise ValueError(f"duplicate architecture in candidate manifest: {key}")
        seen_keys.add(key)
        label = str(item.get("candidate_label") or item.get("combo") or _candidate_label(architecture))
        records.append(
            {
                "candidate_id": str(item.get("candidate_id") or f"manifest_candidate_{index}"),
                "candidate_rank": item.get("rank", index + 1),
                "candidate_label": label,
                "candidate_key": key,
                "architecture": architecture,
                "source_method": item.get("source_method") or item.get("best_seen_by"),
                "source_arch_id": item.get("source_arch_id"),
                "observed_macro_f1": item.get("observed_macro_f1") or item.get("best_macro_f1"),
                "observed_top1": item.get("observed_top1") or item.get("best_top1"),
            }
        )
    return records


def _records_from_architectures(architectures: list[ArchitectureSpec]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, architecture in enumerate(architectures):
        records.append(
            {
                "candidate_id": f"strict40_candidate_{index}",
                "candidate_rank": index + 1,
                "candidate_label": _candidate_label(architecture),
                "candidate_key": _candidate_key(architecture),
                "architecture": architecture,
                "source_method": None,
                "source_arch_id": None,
                "observed_macro_f1": None,
                "observed_top1": None,
            }
        )
    return records


def _write_markdown_report(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for label, item in sorted(
        summary["by_candidate"].items(),
        key=lambda kv: (
            -float(kv[1]["macro_f1"]["mean"])
            if kv[1]["macro_f1"]["mean"] is not None
            else float("inf")
        ),
    ):
        hw = item["hardware"]
        rows.append(
            "| {rank} | `{label}` | {macro:.4f} | {macro_std:.4f} | "
            "{top1:.4f} | {top1_std:.4f} | {lat:.3f} | {lut} | {dsp} | {bram} | {power:.4f} |".format(
                rank=len(rows) + 1,
                label=label,
                macro=float(item["macro_f1"]["mean"]),
                macro_std=float(item["macro_f1"]["std_pop"]),
                top1=float(item["top1"]["mean"]),
                top1_std=float(item["top1"]["std_pop"]),
                lat=float(hw["latency_ms"]),
                lut=hw["lut"],
                dsp=hw["dsp"],
                bram=hw["bram"],
                power=float(hw["power_w"]),
            )
        )

    lines = [
        "# Strict40 Expanded Top-8 Multi-Seed Proxy Report",
        "",
        f"Generated: `{summary['finished_at']}`",
        "",
        "## Summary",
        "",
        f"- Config: `{summary['config']}`",
        f"- Candidate manifest: `{summary.get('candidate_manifest')}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Eval epochs: `{summary['eval_epochs']}`",
        f"- Selection metric: `{summary['selection_metric']}`",
        f"- Strict LUT OK: `{summary['strict_lut_ok']}`",
        "- Power caveat: added LUT expansion power coverage remains partial (`4 measured / 3 missing`).",
        "",
        "## Ranking By Macro-F1 Mean",
        "",
        "| rank | candidate | macro_f1_mean | macro_f1_std_pop | top1_mean | top1_std_pop | latency_ms | LUT | DSP | BRAM | power_w |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Decision Rule",
        "",
        "- Select any later Top-3 only from this multi-seed mean ranking.",
        "- Do not launch 240 epoch training from single-run proxy results.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "histories"
    history_dir.mkdir(exist_ok=True)

    config = load_config(str(config_path))
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    search_cfg = config.get("search", {})
    project_cfg = config.get("project", {})

    base_seed = int(project_cfg.get("seed", 42))
    _set_seed(base_seed)
    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    batch_size = int(pick(args.batch_size, training_cfg.get("batch_size"), 16))
    num_workers = int(pick(args.num_workers, dataset_cfg.get("num_workers"), 0))
    selection_metric = str(args.selection_metric or search_cfg.get("selection_metric", "macro_f1"))
    early_stopping = (
        None
        if args.early_stopping_patience is None or args.early_stopping_patience <= 0
        else int(args.early_stopping_patience)
    )

    train_loader, val_loader, class_weights, resolved_num_classes = create_data_pipeline(
        dataset_name=str(dataset_cfg.get("name", "nksid")),
        data_dir=dataset_cfg.get("data_dir"),
        batch_size=batch_size,
        image_size=int(dataset_cfg.get("image_size", 224)),
        num_classes=dataset_cfg.get("num_classes"),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        fold=int(dataset_cfg.get("fold", 0)),
        num_workers=num_workers,
        device=device,
        use_kfold=bool(dataset_cfg.get("use_kfold", True)),
        valid_size=dataset_cfg.get("valid_size"),
        split_seed=int(dataset_cfg.get("split_seed", base_seed)),
        image_error_policy=str(dataset_cfg.get("image_error_policy", "raise")),
    )

    enumerated_architectures, estimator, search_space = enumerate_architectures(config_path)
    candidate_manifest_path = (
        Path(args.candidate_manifest).expanduser().resolve()
        if args.candidate_manifest
        else None
    )
    candidate_records = (
        _load_candidate_manifest(candidate_manifest_path)
        if candidate_manifest_path is not None
        else _records_from_architectures(enumerated_architectures)
    )
    for record in candidate_records:
        errors = search_space.validate(record["architecture"])
        if errors:
            raise ValueError(
                f"candidate {record['candidate_id']} does not match search space: "
                + "; ".join(errors)
            )

    rows: list[dict[str, Any]] = []
    manifest = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "candidate_manifest": str(candidate_manifest_path) if candidate_manifest_path else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": list(args.seeds),
        "eval_epochs": int(args.eval_epochs),
        "early_stopping_patience": early_stopping,
        "selection_metric": selection_metric,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "num_classes": resolved_num_classes,
        "candidate_count": len(candidate_records),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "Strict40 proxy multi-seed eval: "
        f"{len(candidate_records)} candidates x {len(args.seeds)} seeds x {args.eval_epochs} epochs"
    )
    print(f"Output dir: {output_dir}")

    for record in candidate_records:
        architecture = record["architecture"]
        candidate_id = str(record["candidate_id"])
        candidate_label = str(record["candidate_label"])
        estimator.reset_lut_stats()
        cost = estimator.estimate(architecture, search_space)
        lut_stats = estimator.get_lut_stats()
        feasible = len(cost.violations) == 0

        for seed in args.seeds:
            run_id = f"{candidate_id}_seed{seed}"
            _set_seed(int(seed))
            model = build_model(architecture, num_classes=resolved_num_classes)
            score, history = train_model(
                model=model,
                train_loader=train_loader,
                num_classes=resolved_num_classes,
                epochs=int(args.eval_epochs),
                lr=float(training_cfg.get("lr", 0.001)),
                device=device,
                val_loader=val_loader,
                class_weights=class_weights,
                early_stopping_patience=early_stopping,
                selection_metric=selection_metric,
            )
            best_eval = dict(history.get("best_eval") or {})
            history_payload = {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "candidate_label": candidate_label,
                "seed": int(seed),
                "score": score,
                "history": history,
            }
            (history_dir / f"{run_id}.json").write_text(
                json.dumps(history_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            row = {
                "candidate_id": candidate_id,
                "candidate_rank": record["candidate_rank"],
                "candidate_label": candidate_label,
                "candidate_key": record["candidate_key"],
                "source_method": record["source_method"],
                "source_arch_id": record["source_arch_id"],
                "observed_macro_f1": record["observed_macro_f1"],
                "observed_top1": record["observed_top1"],
                "seed": int(seed),
                "eval_epochs": int(args.eval_epochs),
                "best_epoch": history.get("best_epoch"),
                "macro_f1": best_eval.get("macro_f1"),
                "top1": best_eval.get("top1"),
                "weighted_f1": best_eval.get("weighted_f1"),
                "top5": best_eval.get("top5"),
                "latency_ms": cost.latency_ms,
                "lut": cost.resource_lut,
                "dsp": cost.resource_dsp,
                "bram": cost.resource_bram,
                "power_w": cost.power_w,
                "feasible": feasible,
                "violations": list(cost.violations),
                "lut_hits": lut_stats["hits"],
                "lut_misses": lut_stats["misses"],
                "true_misses": lut_stats["true_misses"],
                "deferred_hits": lut_stats["deferred_hits"],
            }
            rows.append(row)
            with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _write_csv(output_dir / "results.csv", rows)
            print(
                f"[{len(rows)}/{len(candidate_records) * len(args.seeds)}] "
                f"{candidate_label} seed={seed}: "
                f"macro_f1={best_eval.get('macro_f1'):.4f}, "
                f"top1={best_eval.get('top1'):.4f}, "
                f"best_epoch={history.get('best_epoch')}"
            )

    by_candidate: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_label"])].append(row)

    for label, group in sorted(grouped.items()):
        by_candidate[label] = {
            "n": len(group),
            "macro_f1": _stats([float(row["macro_f1"]) for row in group if row["macro_f1"] is not None]),
            "top1": _stats([float(row["top1"]) for row in group if row["top1"] is not None]),
            "best_epochs": [row["best_epoch"] for row in group],
            "seeds": [row["seed"] for row in group],
            "hardware": {
                "latency_ms": group[0]["latency_ms"],
                "lut": group[0]["lut"],
                "dsp": group[0]["dsp"],
                "bram": group[0]["bram"],
                "power_w": group[0]["power_w"],
            },
        }

    summary = {
        **manifest,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_runs": len(rows),
        "candidate_count": len(candidate_records),
        "strict_lut_ok": all(
            int(row["lut_misses"]) == 0
            and int(row["true_misses"]) == 0
            and int(row["deferred_hits"]) == 0
            for row in rows
        ),
        "by_candidate": by_candidate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_report(output_dir, summary)
    print(f"Completed proxy multi-seed eval -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
