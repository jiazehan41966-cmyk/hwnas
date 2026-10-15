"""Phase5D: disambiguate detection-head weight vs activation quantization sensitivity."""

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
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from phase5a_quantization.calibrate import build_calibration, calibration_tensor_batches
from phase5a_quantization.common import (
    NETWORKS,
    REPORT as PHASE5A_REPORT,
    SEARCH_SPACE,
    SEARCH_SPACE_SHA256,
    SEEDS,
    admit_network,
    baseline,
    build_network,
    checkpoint_rows,
    dump_json,
    forbid_split,
    require_frozen_xstar,
    resolve_checkpoint,
    sha256_file,
)
from phase5a_quantization.run import load_weights


REPORT = ROOT / "reports/phase5d_head_quantization"
PHASE5C_REPORT = ROOT / "reports/phase5c_quant_sensitivity"
BASELINE_TOLERANCE = 1e-12
HEAD_BOTH_REPRO_TOLERANCE = 1e-12
CLOSE_TO_BOTH_RATIO = 0.80
BOTH_STRONGER_RATIO = 1.25
MIN_BOTH_RECOVERY_RATIO = 0.20

VARIANTS = {
    "HEAD_WEIGHT_FP32": {
        "fp32_module_prefixes": (),
        "fp32_weight_prefixes": ("head",),
        "fp32_activation_prefixes": (),
        "description": "Head weights and bias FP32; head activations INT8; other modules INT8.",
    },
    "HEAD_ACTIVATION_FP32": {
        "fp32_module_prefixes": (),
        "fp32_weight_prefixes": (),
        "fp32_activation_prefixes": ("head",),
        "description": "Head activations FP32; head weights INT8; other modules INT8.",
    },
    "HEAD_BOTH_FP32": {
        "fp32_module_prefixes": ("head",),
        "fp32_weight_prefixes": (),
        "fp32_activation_prefixes": (),
        "description": "Entire head FP32; other modules INT8. Must reproduce Phase5C HEAD_FP32.",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_phase5a_rows() -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with (PHASE5A_REPORT / "PTQ_PAIRED_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(row["network"], int(row["seed"]))] = row
    expected = {(label, seed) for label in NETWORKS for seed in SEEDS}
    if set(rows) != expected:
        raise ValueError("Phase5A PTQ baseline rows are incomplete")
    return rows


def read_phase5c_head_rows() -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with (PHASE5C_REPORT / "SENSITIVITY_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["module"] == "HEAD_FP32":
                rows[(row["network"], int(row["seed"]))] = row
    expected = {(label, seed) for label in NETWORKS for seed in SEEDS}
    if set(rows) != expected:
        raise ValueError("Phase5C HEAD_FP32 rows are incomplete")
    return rows


def evaluate_model(model: torch.nn.Module, device: str) -> dict[str, Any]:
    forbid_split("valid")
    config = baseline()
    loader = frozen.make_loader(config, frozen.resolve(config["dataset"]["data_dir"]), "valid", 42, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(config)).to(device)
    metrics, _records = frozen.evaluate(model.to(device).eval(), loader, criterion, config, device)
    return {
        "Precision": float(metrics["Precision"]),
        "Recall": float(metrics["Recall"]),
        "mAP50": float(metrics["AP50"]),
        "mAP50_95": float(metrics["mAP50_95"]),
        "test_evaluated": False,
    }


def build_variant(
    *,
    label: str,
    checkpoint_path: Path,
    calibration_manifest: dict[str, Any],
    device: str,
    variant: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, _identity = build_network(label)
    load_weights(model, checkpoint_path)
    spec = VARIANTS[variant]
    batches = list(calibration_tensor_batches(calibration_manifest, device))
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        model.eval(),
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
        fp32_module_prefixes=spec["fp32_module_prefixes"],
        fp32_weight_prefixes=spec["fp32_weight_prefixes"],
        fp32_activation_prefixes=spec["fp32_activation_prefixes"],
        quantize_neck_adds=True,
    )
    return quantized, profile


def write_protocol() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase5D Protocol

Date: {now()}

## Objective

Disambiguate whether detection-head quantization sensitivity is dominated by head weights, head activations, both, or remains unresolved.

This is a diagnostic only. It is not mixed-precision search and does not define a hardware profile.

## Frozen Inputs

- Networks: B1, C, D
- Seeds: 42, 43, 44
- Checkpoint policy: `ALL_BEST_PT`
- Baseline: Phase5A `ALL_INT8` PTQ
- Numeric contract: `NUMERIC_CONTRACT_V1` `{EXPECTED_HASH}`
- Search Space V0.1 SHA256: `{SEARCH_SPACE_SHA256}`
- Split: Dataset A clean valid
- test_evaluated=false

## Variants

- `HEAD_WEIGHT_FP32`: head weights and bias FP32; head activations INT8; other modules INT8.
- `HEAD_ACTIVATION_FP32`: head activations FP32; head weights INT8; other modules INT8.
- `HEAD_BOTH_FP32`: entire head FP32; other modules INT8. This must reproduce Phase5C `HEAD_FP32`.

## Metrics

Primary metric: mAP50:95.

For each run:

- rescue = mAP_variant - mAP_ALL_INT8
- recovery_ratio = rescue / (mAP_FP32 - mAP_ALL_INT8)

Recovery ratio is reported only when FP32 > ALL_INT8.

## Frozen Classification Rules

Let W, A, and B be mean rescue ratios for `HEAD_WEIGHT_FP32`, `HEAD_ACTIVATION_FP32`, and `HEAD_BOTH_FP32`.

- If B is non-positive, or fewer than 2 seeds have positive B rescue: `UNRESOLVED`.
- If A >= {CLOSE_TO_BOTH_RATIO} * B and A >= W: `ACTIVATION_DOMINANT`.
- If W >= {CLOSE_TO_BOTH_RATIO} * B and W > A: `WEIGHT_DOMINANT`.
- If B >= {MIN_BOTH_RECOVERY_RATIO} and B >= {BOTH_STRONGER_RATIO} * max(W, A): `BOTH_SENSITIVE`.
- Otherwise: `UNRESOLVED`.

Cross-network:

- Same resolved classification for all networks: that classification.
- Different resolved classifications across networks: `NETWORK_DEPENDENT`.
- Otherwise: `UNRESOLVED`.

`Q_PROFILE_EXTENSION=YES` only when cross-network classification is a single resolved class. `CONDITIONAL` only when cross-network is `NETWORK_DEPENDENT`. Otherwise `NO`.
"""
    (REPORT / "PHASE5D_PROTOCOL.md").write_text(text, encoding="utf-8")


def classify_network(summary: dict[str, dict[str, Any]]) -> str:
    w = float(summary["HEAD_WEIGHT_FP32"]["mean_recovery_ratio"])
    a = float(summary["HEAD_ACTIVATION_FP32"]["mean_recovery_ratio"])
    b = float(summary["HEAD_BOTH_FP32"]["mean_recovery_ratio"])
    b_positive = int(summary["HEAD_BOTH_FP32"]["positive_seed_count"])
    if b <= 0.0 or b_positive < 2:
        return "UNRESOLVED"
    if a >= CLOSE_TO_BOTH_RATIO * b and a >= w:
        return "ACTIVATION_DOMINANT"
    if w >= CLOSE_TO_BOTH_RATIO * b and w > a:
        return "WEIGHT_DOMINANT"
    if b >= MIN_BOTH_RECOVERY_RATIO and b >= BOTH_STRONGER_RATIO * max(w, a):
        return "BOTH_SENSITIVE"
    return "UNRESOLVED"


def cross_network_gate(network_classes: dict[str, str]) -> tuple[str, str]:
    resolved = {value for value in network_classes.values() if value != "UNRESOLVED"}
    if len(resolved) == 1 and len(set(network_classes.values())) == 1:
        return next(iter(resolved)), "YES"
    if len(resolved) >= 2 or (resolved and "UNRESOLVED" in set(network_classes.values())):
        return "NETWORK_DEPENDENT", "CONDITIONAL"
    return "UNRESOLVED", "NO"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()

    require_frozen_xstar()
    load_numeric_contract_v1()
    if sha256_file(SEARCH_SPACE) != SEARCH_SPACE_SHA256:
        raise ValueError("Search Space V0.1 hash changed")
    write_protocol()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase5a_rows = read_phase5a_rows()
    phase5c_head = read_phase5c_head_rows()
    calibration_manifest = build_calibration(REPORT)
    calibration_sha = sha256_file(REPORT / "calibration_manifest.json")

    rows: list[dict[str, Any]] = []
    jobs = 0
    for ckpt_row in checkpoint_rows():
        label = ckpt_row["label"]
        seed = int(ckpt_row["seed"])
        identity = admit_network(label)
        located = resolve_checkpoint(label, seed, ckpt_row)
        if located["status"] != "FOUND":
            raise FileNotFoundError(f"missing checkpoint for {label} seed{seed}")
        if not str(located["path"]).endswith("best.pt"):
            raise ValueError("Phase5D must use best.pt only")
        fp32 = phase5a_rows[(label, seed)]
        baseline = float(fp32["int8_mAP50_95"])
        gap = float(fp32["fp32_mAP50_95"]) - baseline
        for variant in VARIANTS:
            if args.max_jobs and jobs >= args.max_jobs:
                break
            print(label, f"seed={seed}", variant, flush=True)
            model, profile = build_variant(
                label=label,
                checkpoint_path=Path(located["path"]),
                calibration_manifest=calibration_manifest,
                device=device,
                variant=variant,
            )
            metrics = evaluate_model(model, device)
            if variant == "HEAD_BOTH_FP32":
                reference = float(phase5c_head[(label, seed)]["hybrid_mAP50_95"])
                diff = abs(float(metrics["mAP50_95"]) - reference)
                if diff > HEAD_BOTH_REPRO_TOLERANCE:
                    manifest = {
                        "phase": "phase5d_head_quantization",
                        "status": "STOP_HEAD_BOTH_REPRODUCTION_FAILED",
                        "network": label,
                        "seed": seed,
                        "observed": metrics["mAP50_95"],
                        "phase5c_head_fp32": reference,
                        "abs_diff": diff,
                        "tolerance": HEAD_BOTH_REPRO_TOLERANCE,
                        "test_evaluated": False,
                    }
                    dump_json(REPORT / "PHASE5D_MANIFEST.json", manifest)
                    raise RuntimeError("HEAD_BOTH_FP32 did not reproduce Phase5C HEAD_FP32")
            rescue = float(metrics["mAP50_95"]) - baseline
            recovery_ratio = rescue / gap if gap > 0 else ""
            rows.append(
                {
                    "network": label,
                    "architecture_id": identity,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint_sha256": located["sha256"],
                    "calibration_manifest_sha256": calibration_sha,
                    "numeric_contract": "NUMERIC_CONTRACT_V1",
                    "numeric_contract_sha256": EXPECTED_HASH,
                    "fp32_mAP50_95": fp32["fp32_mAP50_95"],
                    "all_int8_mAP50_95": fp32["int8_mAP50_95"],
                    "variant_Precision": metrics["Precision"],
                    "variant_Recall": metrics["Recall"],
                    "variant_mAP50": metrics["mAP50"],
                    "variant_mAP50_95": metrics["mAP50_95"],
                    "rescue_mAP50_95": rescue,
                    "gap_mAP50_95": gap,
                    "recovery_ratio": recovery_ratio,
                    "head_weight_fp32_layers": json.dumps(profile["fp32_bypass"]["weight_fp32_layers"]),
                    "head_activation_fp32_layers": json.dumps(profile["fp32_bypass"]["activation_fp32_layers"]),
                    "head_both_fp32_layers": json.dumps(profile["fp32_bypass"]["conv_layers"]),
                    "test_evaluated": False,
                    "split": "valid",
                    "status": "MEASURED",
                }
            )
            jobs += 1
            dump_json(REPORT / "partial_results.json", rows)
        if args.max_jobs and jobs >= args.max_jobs:
            break

    if not args.max_jobs and len(rows) != len(NETWORKS) * len(SEEDS) * len(VARIANTS):
        raise ValueError("incomplete Phase5D matrix")
    write_csv(REPORT / "HEAD_WEIGHT_ACTIVATION_RESULTS.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    network_classes: dict[str, str] = {}
    for label in NETWORKS:
        per_variant_summary: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            values = [row for row in rows if row["network"] == label and row["variant"] == variant]
            rescues = [float(row["rescue_mAP50_95"]) for row in values]
            ratios = [float(row["recovery_ratio"]) for row in values if row["recovery_ratio"] != ""]
            summary = {
                "network": label,
                "variant": variant,
                "seed42_rescue": next(row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 42),
                "seed43_rescue": next(row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 43),
                "seed44_rescue": next(row["rescue_mAP50_95"] for row in values if int(row["seed"]) == 44),
                "mean_rescue": float(mean(rescues)),
                "sample_std_rescue": sample_std(rescues),
                "positive_seed_count": sum(1 for value in rescues if value > 0),
                "mean_recovery_ratio": float(mean(ratios)) if ratios else 0.0,
                "sample_std_recovery_ratio": sample_std(ratios) if ratios else 0.0,
            }
            per_variant_summary[variant] = summary
            summary_rows.append(summary)
        network_classes[label] = classify_network(per_variant_summary)

    cross_class, q_profile_extension = cross_network_gate(network_classes)
    manifest = {
        "phase": "phase5d_head_quantization",
        "status": "PASS",
        "created_at": now(),
        "device": device,
        "runs": len(rows),
        "expected_runs": len(NETWORKS) * len(SEEDS) * len(VARIANTS),
        "checkpoint_policy": "ALL_BEST_PT",
        "baseline": "Phase5A_ALL_INT8_PTQ",
        "head_both_reproduced_phase5c": True,
        "head_both_repro_tolerance": HEAD_BOTH_REPRO_TOLERANCE,
        "calibration_manifest_sha256": calibration_sha,
        "numeric_contract": "NUMERIC_CONTRACT_V1",
        "numeric_contract_sha256": EXPECTED_HASH,
        "search_space_sha256": SEARCH_SPACE_SHA256,
        "network_classification": network_classes,
        "HEAD_QUANTIZATION_SENSITIVITY": cross_class,
        "Q_PROFILE_EXTENSION": q_profile_extension,
        "mixed_precision_started": False,
        "hls_started": False,
        "qat_started": False,
        "retrain_started": False,
        "test_evaluated": False,
    }
    dump_json(REPORT / "PHASE5D_MANIFEST.json", manifest)

    lines = [
        "# Head Sensitivity Summary",
        "",
        f"- HEAD_BOTH_FP32 reproduced Phase5C HEAD_FP32: true",
        f"- HEAD_QUANTIZATION_SENSITIVITY: `{cross_class}`",
        f"- Q_PROFILE_EXTENSION: `{q_profile_extension}`",
        f"- test_evaluated=false",
        "",
        "| Network | Variant | Mean rescue | Std | Positive seeds | Mean recovery ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['network']} | {row['variant']} | {row['mean_rescue']:+.6f} | {row['sample_std_rescue']:.6f} | {row['positive_seed_count']}/3 | {row['mean_recovery_ratio']:.6f} |"
        )
    lines += [
        "",
        "## Network Classification",
        "",
        "| Network | Classification |",
        "|---|---|",
    ]
    for label, status in network_classes.items():
        lines.append(f"| {label} | `{status}` |")
    (REPORT / "HEAD_SENSITIVITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gate = [
        "# Q Profile Extension Gate",
        "",
        f"Q_PROFILE_EXTENSION = {q_profile_extension}",
        "",
        f"HEAD_QUANTIZATION_SENSITIVITY = {cross_class}",
        "",
        "This gate does not validate mixed precision. It only controls whether a later phase may define a minimal mixed-precision candidate.",
        "",
        "No mixed-precision run, HLS run, QAT, retraining, or test evaluation was started.",
        "",
        "test_evaluated=false",
    ]
    (REPORT / "Q_PROFILE_EXTENSION_GATE.md").write_text("\n".join(gate) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
