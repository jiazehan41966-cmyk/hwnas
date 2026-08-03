#!/usr/bin/env python3
"""Run training-only activation calibration and Python INT8 reference.

This is the next software gate after real-checkpoint export.  It intentionally
does not run C-sim, HLS, route, bitstream generation, COM5, or power
measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.dataset import NKSIDDataset  # noqa: E402
from hwnas_fpga.data.protocol import build_protocol_split  # noqa: E402
from hwnas_fpga.deploy.inference import load_checkpoint_model  # noqa: E402
from hwnas_fpga.fourstage_int8_reference import (  # noqa: E402
    CALIBRATION_CONTRACT,
    collect_activation_stats,
    full_network_int8_reference,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


DEFAULT_EXPORT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_checkpoint_export_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "results"
    / "sonar_fourstage_operator_v2"
    / "full_network_deployment_closure"
)
DEFAULT_SUMMARY = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "fourstage_int8_reference_summary.json"
)
DEFAULT_CONTRACT = ROOT / "configs" / "evaluation" / "nksid_frozen_protocol_v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "NKSID"))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--checkpoint-export-summary", default=str(DEFAULT_EXPORT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--reference-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def build_training_only_loaders(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = Path(args.contract).expanduser().resolve()
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    dataset_cfg = contract["dataset"]
    recipe = contract["training_recipe"]
    fold = 0
    seed = 42
    dataset = NKSIDDataset(
        data_dir=str(Path(args.data_dir).expanduser().resolve()),
        image_size=int(dataset_cfg["image_size"]),
        is_training=False,
        fold=fold,
        use_kfold=False,
        split="full",
        output_channels=int(dataset_cfg["input_channels"]),
        image_error_policy="raise",
        geometry_mode=str(dataset_cfg["geometry_mode"]),
        fixed_scale_factor=dataset_cfg.get("fixed_scale_factor"),
        geometry_padding_value=int(dataset_cfg["geometry_padding_value"]),
        augmentation_profile=str(dataset_cfg["augmentation_profile"]),
        cache_decoded_images=False,
        cache_geometry_images=False,
    )
    split = build_protocol_split(
        dataset.samples,
        str(Path(args.data_dir).expanduser().resolve()),
        fold_index=fold,
        seed=seed,
        inner_val_fraction=float(dataset_cfg["inner_val_fraction"]),
        num_classes=8,
    )
    train_indices = list(split.train_indices)
    calibration_indices = train_indices[: int(args.calibration_samples)]
    reference_indices = train_indices[: int(args.reference_samples)]
    if not calibration_indices or not reference_indices:
        raise ValueError("calibration/reference index selection is empty")
    calibration_loader = DataLoader(
        Subset(dataset, calibration_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    reference_loader = DataLoader(
        Subset(dataset, reference_indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "dataset": dataset,
        "split": split,
        "fold": fold,
        "seed": seed,
        "calibration_indices": calibration_indices,
        "reference_indices": reference_indices,
        "calibration_loader": calibration_loader,
        "reference_loader": reference_loader,
        "recipe": recipe,
    }


def summarize_reference_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [bool(row["argmax_match"]) for row in records]
    errors = [float(row["max_abs_logit_error"]) for row in records]
    return {
        "sample_count": len(records),
        "argmax_match_count": int(sum(matches)),
        "argmax_match_rate": float(sum(matches) / len(matches)) if matches else 0.0,
        "max_abs_logit_error_max": max(errors) if errors else None,
        "max_abs_logit_error_mean": (
            float(sum(errors) / len(errors)) if errors else None
        ),
    }


def run_one_candidate(
    candidate: dict[str, Any],
    *,
    loaders: dict[str, Any],
    output_root: Path,
    device: str,
    calibration_samples: int,
) -> dict[str, Any]:
    checkpoint = Path(candidate["source_checkpoint"]["path"]).resolve()
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != candidate["source_checkpoint"]["sha256"]:
        raise RuntimeError(
            f"checkpoint SHA mismatch for {candidate['arch_id']}: "
            f"{actual_checkpoint_sha} != {candidate['source_checkpoint']['sha256']}"
        )
    model, architecture, _payload, class_names = load_checkpoint_model(
        checkpoint,
        device=device,
    )
    out_dir = (
        output_root
        / f"{safe_name(candidate['role'])}__{safe_name(candidate['arch_id'])}"
        / "fold0_seed42"
        / "int8_reference"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = collect_activation_stats(
        model,
        loaders["calibration_loader"],
        device=device,
        max_samples=int(calibration_samples),
    )
    calibration.update(
        {
            "role": candidate["role"],
            "arch_id": candidate["arch_id"],
            "source_checkpoint": evidence(checkpoint),
            "candidate_json": candidate["candidate_json"],
            "class_names": list(class_names),
            "calibration_data": {
                "source": "Protocol V2 fold0/seed42 outer-train inner-train only",
                "outer_validation_accessed": False,
                "inner_validation_accessed": False,
                "fold": loaders["fold"],
                "seed": loaders["seed"],
                "indices": loaders["calibration_indices"],
            },
        }
    )
    calibration["payload_sha256"] = canonical_sha256(calibration)
    calibration_path = out_dir / "activation_calibration.json"
    write_json(calibration_path, calibration)

    records: list[dict[str, Any]] = []
    for offset, (inputs, labels) in enumerate(loaders["reference_loader"]):
        result = full_network_int8_reference(model, inputs.to(device), calibration)
        records.append(
            {
                "offset": offset,
                "dataset_index": int(loaders["reference_indices"][offset]),
                "label": int(labels.reshape(-1)[0].item()),
                "fp32_argmax": int(result["fp32_argmax"].reshape(-1)[0].item()),
                "int8_argmax": int(result["int8_argmax"].reshape(-1)[0].item()),
                "argmax_match": bool(result["argmax_match"].reshape(-1)[0].item()),
                "max_abs_logit_error": float(result["max_abs_logit_error"]),
                "logits_int8": [
                    int(value)
                    for value in result["logits_int8"].reshape(-1).tolist()
                ],
                "logits_dequant": [
                    float(value)
                    for value in result["logits_dequant"].reshape(-1).tolist()
                ],
                "fp32_logits": [
                    float(value)
                    for value in result["fp32_logits"].reshape(-1).tolist()
                ],
            }
        )
    reference_payload = {
        "schema_version": 1,
        "contract": CALIBRATION_CONTRACT,
        "status": "PASS",
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "architecture": architecture.to_dict(),
        "source_checkpoint": evidence(checkpoint),
        "activation_calibration": evidence(calibration_path),
        "reference_data": {
            "source": "same training-only calibration subset prefix",
            "outer_validation_accessed": False,
            "inner_validation_accessed": False,
            "indices": loaders["reference_indices"],
        },
        "reference_records": records,
        "summary": summarize_reference_records(records),
        "downstream_gates": {
            "hls_c_sim": "PENDING",
            "pytorch_int8_vs_csim_zero_mismatch": "PENDING",
            "rtl_cosim": "PENDING",
            "full_network_hls": "PENDING",
            "place_and_route_5ns": "PENDING",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "claim_boundary": (
            "Python INT8 reference generated from real checkpoint and "
            "training-only activation calibration. This is not HLS/C-sim "
            "zero-mismatch, route, COM5, or measured power evidence."
        ),
    }
    reference_payload["payload_sha256"] = canonical_sha256(reference_payload)
    reference_path = out_dir / "python_int8_reference.json"
    write_json(reference_path, reference_payload)
    return {
        "role": candidate["role"],
        "arch_id": candidate["arch_id"],
        "status": "PASS",
        "source_checkpoint": evidence(checkpoint),
        "activation_calibration": evidence(calibration_path),
        "python_int8_reference": evidence(reference_path),
        "reference_summary": reference_payload["summary"],
        "scale_count": len(calibration["scales"]),
        "calibration_samples": int(calibration["samples_seen"]),
        "claim_boundary": reference_payload["claim_boundary"],
    }


def main() -> int:
    args = parse_args()
    export_summary_path = Path(args.checkpoint_export_summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    export_summary = read_json(export_summary_path)
    loaders = build_training_only_loaders(args)
    rows = [
        run_one_candidate(
            candidate,
            loaders=loaders,
            output_root=output_root,
            device=str(args.device),
            calibration_samples=int(args.calibration_samples),
        )
        for candidate in export_summary["candidates"]
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "gate": "activation_calibrated_python_int8_reference",
        "contract": CALIBRATION_CONTRACT,
        "checkpoint_export_summary": evidence(export_summary_path),
        "protocol_file": evidence(loaders["contract_path"]),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "calibration_dataset": {
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "fold": loaders["fold"],
            "seed": loaders["seed"],
            "outer_validation_accessed": False,
            "inner_validation_accessed": False,
            "sample_count": int(args.calibration_samples),
            "reference_sample_count": int(args.reference_samples),
            "index_source": "split.train_indices from Protocol V2 build_protocol_split",
            "calibration_indices": loaders["calibration_indices"],
            "reference_indices": loaders["reference_indices"],
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "downstream_gates": {
            "hls_c_sim": "PENDING",
            "pytorch_int8_vs_csim_zero_mismatch": "PENDING",
            "rtl_cosim": "PENDING",
            "full_network_hls": "PENDING",
            "place_and_route_5ns": "PENDING",
            "bitstream": "NOT_GENERATED",
            "com5_board_latency": "NOT_RUN",
            "external_meter_power": "NOT_MEASURED",
        },
        "known_next_work": [
            "Generate a four-stage HLS C-sim harness that consumes this exact calibration/reference contract.",
            "Run C-sim and require PyTorch INT8 reference versus C-sim zero mismatch before RTL or route.",
        ],
        "claim_boundary": (
            "Activation calibration and Python INT8 reference are complete. "
            "No HLS/C-sim zero-mismatch, RTL co-sim, full-network route, "
            "bitstream, COM5, or power claim is made."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
