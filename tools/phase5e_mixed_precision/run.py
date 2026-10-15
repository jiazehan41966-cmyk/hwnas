"""Phase5E: validate frozen network-conditioned INT16 head profiles."""

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
from phase5a_quantization.calibrate import calibration_tensor_batches
from phase5a_quantization.common import (
    NETWORKS,
    QAT_ABSOLUTE_DROP,
    QAT_RELATIVE_DROP,
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


REPORT = ROOT / "reports/phase5e_mixed_precision"
PHASE5D_REPORT = ROOT / "reports/phase5d_head_quantization"
HARDWARE_COST_STATUS = "NEEDS_MEASUREMENT"

PROFILES: dict[str, dict[str, Any]] = {
    "B1_Q1": {"network": "B1", "head_weight_bits": 8, "head_activation_bits": 16},
    "B1_Q2": {"network": "B1", "head_weight_bits": 16, "head_activation_bits": 16},
    "C_Q1": {"network": "C", "head_weight_bits": 16, "head_activation_bits": 8},
    "C_Q2": {"network": "C", "head_weight_bits": 16, "head_activation_bits": 16},
    "D_Q1": {"network": "D", "head_weight_bits": 16, "head_activation_bits": 16},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) >= 2 else 0.0


def stable_sha256(payload: Any) -> str:
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
        raise ValueError("Phase5A paired FP32/INT8 rows are incomplete")
    return rows


def read_calibration_manifest() -> tuple[dict[str, Any], str, str]:
    path = PHASE5A_REPORT / "CALIBRATION_MANIFEST.json"
    if not path.is_file():
        path = PHASE5A_REPORT / "calibration_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError("Phase5E calibration must reuse train-only Phase5A calibration")
    if manifest.get("valid_used_for_selection") is not False or manifest.get("test_evaluated") is not False:
        raise ValueError("calibration manifest violates split boundary")
    return manifest, str(path), sha256_file(path)


def assert_phase5d_conditional() -> None:
    manifest = json.loads((PHASE5D_REPORT / "PHASE5D_MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        "B1": "ACTIVATION_DOMINANT",
        "C": "WEIGHT_DOMINANT",
        "D": "BOTH_SENSITIVE",
    }
    if manifest.get("network_classification") != expected:
        raise ValueError("Phase5D frozen network classifications do not match Phase5E prerequisites")
    if manifest.get("HEAD_QUANTIZATION_SENSITIVITY") != "NETWORK_DEPENDENT":
        raise ValueError("Phase5E requires Phase5D NETWORK_DEPENDENT conclusion")
    if manifest.get("Q_PROFILE_EXTENSION") != "CONDITIONAL":
        raise ValueError("Phase5E requires Phase5D Q_PROFILE_EXTENSION=CONDITIONAL")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("Phase5D test boundary is not sealed")


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


def head_scale_records(profile: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for layer in profile["layers"]:
        if str(layer["name"]) == "head" or str(layer["name"]).startswith("head."):
            records.append(
                {
                    "name": layer["name"],
                    "input_bits": layer["input_bits"],
                    "weight_bits": layer["weight_bits"],
                    "output_bits": layer["output_bits"],
                    "input_exp": layer["input_exp"],
                    "weight_exps": layer["weight_exps"],
                    "output_exp": layer["output_exp"],
                    "input_scale": layer["input_scale"],
                    "weight_scale": layer["weight_scale"],
                    "output_scale": layer["output_scale"],
                    "input_saturation": layer["input_saturation"],
                    "weight_saturation": layer["weight_saturation"],
                    "output_saturation": layer["output_saturation"],
                }
            )
    return records


def build_mixed_model(
    *,
    network: str,
    checkpoint_path: Path,
    calibration_manifest: dict[str, Any],
    profile_name: str,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = PROFILES[profile_name]
    model, _identity = build_network(network)
    load_weights(model, checkpoint_path)
    batches = list(calibration_tensor_batches(calibration_manifest, device))
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        model.eval(),
        batches,
        device=device,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        scheme="symmetric",
        weight_bits_by_prefix={"head": int(spec["head_weight_bits"])},
        activation_bits_by_prefix={"head": int(spec["head_activation_bits"])},
        quantize_neck_adds=True,
    )
    return quantized, profile


def write_protocol(calibration_path: str, calibration_sha: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    profile_lines = [
        "| Profile | Network | Head weight | Head activation | Body |",
        "|---|---|---:|---:|---|",
    ]
    for name, spec in PROFILES.items():
        profile_lines.append(
            f"| `{name}` | {spec['network']} | W{spec['head_weight_bits']} | A{spec['head_activation_bits']} | W8/A8 |"
        )
    text = f"""# Phase5E Protocol

Date: {now()}

## Objective

Validate the frozen network-conditioned mixed-precision candidates derived from Phase5D. This is software accuracy validation only.

## Frozen Inputs

- Networks: B1, C, D
- Seeds: 42, 43, 44
- Checkpoint policy: `ALL_BEST_PT`
- Uniform baseline: Phase5A `W8/A8` PTQ
- Calibration source: `{calibration_path}`
- Calibration manifest SHA256: `{calibration_sha}`
- Evaluation split: Dataset A clean valid
- test_evaluated=false

## Candidate Profiles

{chr(10).join(profile_lines)}

No additional profile is evaluated.

## INT16 Semantics

- signed symmetric integer
- zero_point = 0
- saturation = [-32768, 32767]
- deterministic power-of-two scale derivation
- nearest-even rounding

INT8 body operators remain bound to `NUMERIC_CONTRACT_V1` `{EXPECTED_HASH}`. Mixed-precision hardware status is always `{HARDWARE_COST_STATUS}` in Phase5E.

## Admission Gate

A network/profile is admitted only when all are true:

1. absolute mean mAP50:95 drop vs FP32 <= {QAT_ABSOLUTE_DROP}
2. relative mean drop vs FP32 <= {QAT_RELATIVE_DROP}
3. mean mixed mAP50:95 > uniform INT8
4. at least 2/3 seeds have mixed > uniform INT8

## Network Decision

- B1 keeps `B1_Q1` if admitted; otherwise keeps `B1_Q2` only if admitted.
- C keeps `C_Q1` if admitted; otherwise keeps `C_Q2` only if admitted.
- D only tests `D_Q1`.

Uniform W8/A8 is always retained as the baseline point.
"""
    (REPORT / "PHASE5E_PROTOCOL.md").write_text(text, encoding="utf-8")


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    by_profile: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILES}
    for row in rows:
        by_profile[str(row["profile"])].append(row)

    for profile_name, items in by_profile.items():
        if len(items) != 3:
            raise ValueError(f"incomplete rows for {profile_name}")
        ordered = sorted(items, key=lambda item: int(item["seed"]))
        fp32 = [float(item["fp32_mAP50_95"]) for item in ordered]
        int8 = [float(item["uniform_int8_mAP50_95"]) for item in ordered]
        mixed = [float(item["mixed_mAP50_95"]) for item in ordered]
        delta_fp32 = [float(item["delta_vs_FP32_mAP50_95"]) for item in ordered]
        delta_int8 = [float(item["delta_vs_INT8_mAP50_95"]) for item in ordered]
        ratios = [float(item["recovery_ratio_mAP50_95"]) for item in ordered if item["recovery_ratio_mAP50_95"] != ""]
        mean_fp32 = float(mean(fp32))
        mean_int8 = float(mean(int8))
        mean_mixed = float(mean(mixed))
        mean_drop = mean_fp32 - mean_mixed
        relative_drop = mean_drop / mean_fp32 if mean_fp32 else float("inf")
        positive_seed_count = sum(1 for item in ordered if float(item["mixed_mAP50_95"]) > float(item["uniform_int8_mAP50_95"]))
        gates = {
            "absolute_drop_pass": bool(mean_drop <= QAT_ABSOLUTE_DROP),
            "relative_drop_pass": bool(relative_drop <= QAT_RELATIVE_DROP),
            "mean_beats_int8_pass": bool(mean_mixed > mean_int8),
            "seed_count_pass": bool(positive_seed_count >= 2),
        }
        admitted = all(gates.values())
        row = {
            "network": PROFILES[profile_name]["network"],
            "profile": profile_name,
            "n": len(ordered),
            "fp32_mAP50_95_mean": mean_fp32,
            "fp32_mAP50_95_sample_std": sample_std(fp32),
            "uniform_int8_mAP50_95_mean": mean_int8,
            "uniform_int8_mAP50_95_sample_std": sample_std(int8),
            "mixed_mAP50_95_mean": mean_mixed,
            "mixed_mAP50_95_sample_std": sample_std(mixed),
            "delta_vs_FP32_seed42": next(float(item["delta_vs_FP32_mAP50_95"]) for item in ordered if int(item["seed"]) == 42),
            "delta_vs_FP32_seed43": next(float(item["delta_vs_FP32_mAP50_95"]) for item in ordered if int(item["seed"]) == 43),
            "delta_vs_FP32_seed44": next(float(item["delta_vs_FP32_mAP50_95"]) for item in ordered if int(item["seed"]) == 44),
            "delta_vs_FP32_mean": float(mean(delta_fp32)),
            "delta_vs_FP32_sample_std": sample_std(delta_fp32),
            "delta_vs_INT8_seed42": next(float(item["delta_vs_INT8_mAP50_95"]) for item in ordered if int(item["seed"]) == 42),
            "delta_vs_INT8_seed43": next(float(item["delta_vs_INT8_mAP50_95"]) for item in ordered if int(item["seed"]) == 43),
            "delta_vs_INT8_seed44": next(float(item["delta_vs_INT8_mAP50_95"]) for item in ordered if int(item["seed"]) == 44),
            "delta_vs_INT8_mean": float(mean(delta_int8)),
            "delta_vs_INT8_sample_std": sample_std(delta_int8),
            "recovery_ratio_mean": float(mean(ratios)) if ratios else "",
            "recovery_ratio_sample_std": sample_std(ratios) if ratios else "",
            "absolute_mean_drop_vs_FP32": mean_drop,
            "relative_mean_drop_vs_FP32": relative_drop,
            "positive_seed_count_vs_INT8": positive_seed_count,
            "admission": "Q_PROFILE_ADMITTED" if admitted else "Q_PROFILE_NOT_ADMITTED",
            "hardware_cost_status": HARDWARE_COST_STATUS,
            "test_evaluated": False,
        }
        aggregate_rows.append(row)
        admission_rows.append(
            {
                "network": row["network"],
                "profile": profile_name,
                **gates,
                "absolute_mean_drop_vs_FP32": mean_drop,
                "relative_mean_drop_vs_FP32": relative_drop,
                "positive_seed_count_vs_INT8": positive_seed_count,
                "admission": row["admission"],
                "hardware_cost_status": HARDWARE_COST_STATUS,
                "test_evaluated": False,
            }
        )

    admitted = {row["profile"]: row["admission"] == "Q_PROFILE_ADMITTED" for row in aggregate_rows}
    q_of_x = {
        "B1": ["W8A8"] + (["HEAD_W8A16"] if admitted.get("B1_Q1") else (["HEAD_W16A16"] if admitted.get("B1_Q2") else [])),
        "C": ["W8A8"] + (["HEAD_W16A8"] if admitted.get("C_Q1") else (["HEAD_W16A16"] if admitted.get("C_Q2") else [])),
        "D": ["W8A8"] + (["HEAD_W16A16"] if admitted.get("D_Q1") else []),
    }
    decisions = {
        "B1": q_of_x["B1"][-1] if len(q_of_x["B1"]) > 1 else "NO_EXTENSION",
        "C": q_of_x["C"][-1] if len(q_of_x["C"]) > 1 else "NO_EXTENSION",
        "D": q_of_x["D"][-1] if len(q_of_x["D"]) > 1 else "NO_EXTENSION",
    }
    return aggregate_rows, admission_rows, {
        "Q": q_of_x,
        "network_decisions": decisions,
        "NETWORK_CONDITIONED_QUANTIZATION_SPACE": any(len(values) > 1 for values in q_of_x.values()),
    }


def write_review(aggregate_rows: list[dict[str, Any]], q_payload: dict[str, Any], completed: int) -> None:
    lines = [
        "# Phase5E Final Review",
        "",
        f"- evaluations_completed: {completed}/15",
        f"- NETWORK_CONDITIONED_QUANTIZATION_SPACE: {q_payload['NETWORK_CONDITIONED_QUANTIZATION_SPACE']}",
        f"- mixed_precision_hardware_status: {HARDWARE_COST_STATUS}",
        "- test_evaluated=false",
        "- HLS was not started.",
        "- Joint Pareto was not started.",
        "",
        "| Network | Profile | mixed mean | mixed std | delta vs FP32 mean | delta vs INT8 mean | Admission |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['network']} | `{row['profile']}` | {row['mixed_mAP50_95_mean']:.12f} | {row['mixed_mAP50_95_sample_std']:.12f} | {row['delta_vs_FP32_mean']:+.12f} | {row['delta_vs_INT8_mean']:+.12f} | `{row['admission']}` |"
        )
    lines += [
        "",
        "## Q(x)",
        "",
        f"- Q(B1) = {q_payload['Q']['B1']}",
        f"- Q(C) = {q_payload['Q']['C']}",
        f"- Q(D) = {q_payload['Q']['D']}",
    ]
    (REPORT / "PHASE5E_FINAL_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-jobs", type=int, default=0, help="optional debug cap; 0 means all 15 evaluations")
    args = parser.parse_args()

    require_frozen_xstar()
    load_numeric_contract_v1()
    assert_phase5d_conditional()
    if sha256_file(SEARCH_SPACE) != SEARCH_SPACE_SHA256:
        raise ValueError("Search Space V0.1 hash changed")

    calibration_manifest, calibration_path, calibration_sha = read_calibration_manifest()
    write_protocol(calibration_path, calibration_sha)
    dump_json(REPORT / "CALIBRATION_MANIFEST_USED.json", calibration_manifest)
    phase5a_rows = read_phase5a_rows()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows: list[dict[str, Any]] = []
    jobs = 0

    checkpoint_by_key = {(row["label"], int(row["seed"])): row for row in checkpoint_rows()}
    for profile_name, profile_spec in PROFILES.items():
        network = str(profile_spec["network"])
        for seed in SEEDS:
            if args.max_jobs and jobs >= args.max_jobs:
                break
            ckpt_row = checkpoint_by_key[(network, seed)]
            identity = admit_network(network)
            located = resolve_checkpoint(network, seed, ckpt_row)
            if located["status"] != "FOUND":
                raise FileNotFoundError(f"missing checkpoint for {network} seed{seed}")
            if not str(located["path"]).endswith("best.pt"):
                raise ValueError("Phase5E must use best.pt only")
            print(profile_name, network, f"seed={seed}", flush=True)
            mixed_model, quant_profile = build_mixed_model(
                network=network,
                checkpoint_path=Path(located["path"]),
                calibration_manifest=calibration_manifest,
                profile_name=profile_name,
                device=device,
            )
            metrics = evaluate_model(mixed_model, device)
            baseline_row = phase5a_rows[(network, seed)]
            fp32_95 = float(baseline_row["fp32_mAP50_95"])
            int8_95 = float(baseline_row["int8_mAP50_95"])
            mixed_95 = float(metrics["mAP50_95"])
            gap = fp32_95 - int8_95
            recovery = mixed_95 - int8_95
            q_config = {
                "profile": profile_name,
                "network": network,
                "body": {"weight_bits": 8, "activation_bits": 8},
                "head": {
                    "weight_bits": int(profile_spec["head_weight_bits"]),
                    "activation_bits": int(profile_spec["head_activation_bits"]),
                },
                "int16_semantics": quant_profile["mixed_precision"]["int16_semantics"],
            }
            row = {
                "network": network,
                "architecture_id": identity,
                "seed": seed,
                "profile": profile_name,
                "checkpoint_sha256": located["sha256"],
                "checkpoint_policy": "ALL_BEST_PT",
                "calibration_manifest_sha256": calibration_sha,
                "quant_profile_config_sha256": stable_sha256(q_config),
                "numeric_contract": "NUMERIC_CONTRACT_V1",
                "numeric_contract_sha256": EXPECTED_HASH,
                "head_weight_bits": int(profile_spec["head_weight_bits"]),
                "head_activation_bits": int(profile_spec["head_activation_bits"]),
                "mixed_Precision": metrics["Precision"],
                "mixed_Recall": metrics["Recall"],
                "mixed_mAP50": metrics["mAP50"],
                "mixed_mAP50_95": mixed_95,
                "fp32_mAP50": baseline_row["fp32_mAP50"],
                "fp32_mAP50_95": fp32_95,
                "uniform_int8_mAP50": baseline_row["int8_mAP50"],
                "uniform_int8_mAP50_95": int8_95,
                "delta_vs_FP32_mAP50_95": mixed_95 - fp32_95,
                "delta_vs_INT8_mAP50_95": mixed_95 - int8_95,
                "recovery_mAP50_95": recovery,
                "recovery_ratio_mAP50_95": recovery / gap if gap > 0 else "",
                "head_layer_scales_json": json.dumps(head_scale_records(quant_profile), sort_keys=True),
                "mixed_precision_hardware_status": HARDWARE_COST_STATUS,
                "hls_started": False,
                "test_evaluated": False,
                "split": "valid",
                "status": "MEASURED",
            }
            rows.append(row)
            jobs += 1
            dump_json(REPORT / "partial_results.json", rows)
        if args.max_jobs and jobs >= args.max_jobs:
            break

    expected = len(PROFILES) * len(SEEDS)
    if not args.max_jobs and len(rows) != expected:
        raise ValueError("incomplete Phase5E matrix")

    write_csv(REPORT / "MIXED_PRECISION_RESULTS.csv", rows)
    aggregate_rows, admission_rows, q_payload = aggregate(rows)
    write_csv(REPORT / "MIXED_PRECISION_AGGREGATE.csv", aggregate_rows)
    write_csv(REPORT / "Q_PROFILE_ADMISSION.csv", admission_rows)

    q_of_x = {
        **q_payload,
        "uniform_baseline_retained": True,
        "hardware_cost_status": HARDWARE_COST_STATUS,
        "test_evaluated": False,
    }
    dump_json(REPORT / "Q_OF_X.json", q_of_x)
    write_review(aggregate_rows, q_of_x, len(rows))

    manifest = {
        "phase": "phase5e_mixed_precision",
        "status": "PASS" if len(rows) == expected else "PARTIAL",
        "created_at": now(),
        "device": device,
        "runs": len(rows),
        "expected_runs": expected,
        "checkpoint_policy": "ALL_BEST_PT",
        "baseline": "Phase5A_W8A8_PTQ",
        "numeric_contract_int8_body": "NUMERIC_CONTRACT_V1",
        "numeric_contract_sha256": EXPECTED_HASH,
        "int16_semantics": {
            "signed": True,
            "symmetric": True,
            "zero_point": 0,
            "saturation": [-32768, 32767],
            "rounding": "nearest_even",
            "scale_derivation": "deterministic_power_of_two_from_train_calibration_or_weight_maxabs",
        },
        "calibration_manifest_path": calibration_path,
        "calibration_manifest_sha256": calibration_sha,
        "search_space_sha256": SEARCH_SPACE_SHA256,
        "admission": {row["profile"]: row["admission"] for row in aggregate_rows},
        "Q_OF_X": q_of_x,
        "NETWORK_CONDITIONED_QUANTIZATION_SPACE": q_payload["NETWORK_CONDITIONED_QUANTIZATION_SPACE"],
        "mixed_precision_hardware_status": HARDWARE_COST_STATUS,
        "hardware_cost_extrapolated": False,
        "hls_started": False,
        "qat_started": False,
        "retrain_started": False,
        "architecture_changed": False,
        "nas_started": False,
        "test_evaluated": False,
    }
    dump_json(REPORT / "PHASE5E_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
