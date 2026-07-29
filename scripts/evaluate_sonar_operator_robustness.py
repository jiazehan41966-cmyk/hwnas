#!/usr/bin/env python3
"""Re-evaluate frozen formal checkpoints on the inner-only sonar corruption suite.

The formal outer split is deliberately not constructed or iterated by this
script.  Robustness is measured on the same inner-validation indices that
selected each checkpoint, under deterministic synthetic speckle, contrast, and
blur corruptions.  This keeps the outer validation evaluation single-use while
producing a paired robustness gate for an operator comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.data.dataset import create_protocol_dataloaders  # noqa: E402
from hwnas_fpga.models import build_model  # noqa: E402
from hwnas_fpga.training import (  # noqa: E402
    evaluate_sonar_robustness,
    load_architecture_from_artifact,
)
from hwnas_fpga.training.protocol_reporting import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from hwnas_fpga.training.trainer import evaluate_classifier  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def resolve_record_paths(run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        paths.extend(sorted(run_dir.glob("run_fold*_seed*.json")))
    return sorted(paths, key=lambda path: (path.parent.name, path.name))


def checkpoint_path(record: dict[str, Any], record_path: Path) -> Path:
    metadata = dict(record.get("checkpoint") or {})
    raw_path = str(metadata.get("path") or "")
    if not raw_path:
        raise ValueError(f"checkpoint metadata is missing: {record_path}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    expected = str(metadata.get("sha256") or "")
    observed = sha256_file(path)
    if len(expected) != 64 or observed != expected:
        raise RuntimeError(
            f"checkpoint hash mismatch for {record_path}: {observed} != {expected}"
        )
    return path


def candidate_path(record: dict[str, Any], record_path: Path) -> Path:
    provenance = dict(record.get("provenance") or {})
    candidate = dict(provenance.get("candidate") or {})
    raw_path = str(candidate.get("path") or "")
    if not raw_path:
        raw_path = str(dict(record.get("model") or {}).get("candidate_path") or "")
    if not raw_path:
        raise ValueError(f"candidate provenance is missing: {record_path}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact does not exist: {path}")
    expected = str(candidate.get("sha256") or "")
    observed = sha256_file(path)
    if expected and observed != expected:
        raise RuntimeError(
            f"candidate hash mismatch for {record_path}: {observed} != {expected}"
        )
    return path


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_state_dict"), dict
    ):
        raise ValueError(f"unsupported formal checkpoint payload: {path}")
    return payload["model_state_dict"]


def evaluate_record(
    record_path: Path,
    *,
    data_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    record = read_json(record_path)
    if record.get("evaluation_scope") != "formal_outer":
        raise ValueError(f"robustness input is not a formal_outer record: {record_path}")
    if record.get("outer_validation_consumed") is not True:
        raise ValueError(f"formal record has no single-use outer evaluation: {record_path}")

    fold = int(record["fold"])
    seed = int(record["seed"])
    preprocessing = dict(record.get("preprocessing") or {})
    split_record = dict(record.get("split") or {})
    split_metadata = dict(split_record.get("metadata") or {})
    group_policy = str(split_metadata.get("group_policy") or "none")
    if group_policy != "none":
        raise ValueError(
            "robustness replay currently accepts only the formal group_policy=none "
            f"protocol, observed {group_policy!r} in {record_path}"
        )

    bundle = create_protocol_dataloaders(
        str(data_dir),
        fold=fold,
        seed=seed,
        batch_size=batch_size,
        image_size=int(preprocessing.get("image_size") or 224),
        inner_val_fraction=float(split_record.get("inner_val_fraction") or 0.15),
        num_workers=num_workers,
        augmentation_profile=str(
            preprocessing.get("augmentation_profile") or "frozen_strong"
        ),
        geometry_mode=str(preprocessing.get("geometry_mode") or "stretch_224"),
        group_policy="none",
    )
    observed_split_sha = canonical_sha256(bundle["split"].to_dict())
    expected_split_sha = str(record.get("split_sha256") or "")
    if observed_split_sha != expected_split_sha:
        raise RuntimeError(
            f"split replay mismatch for {record_path}: "
            f"{observed_split_sha} != {expected_split_sha}"
        )

    candidate = candidate_path(record, record_path)
    architecture = load_architecture_from_artifact(candidate)
    model = build_model(
        architecture=architecture,
        num_classes=int(bundle["num_classes"]),
        head_channels=architecture.head_channels,
    )
    checkpoint = checkpoint_path(record, record_path)
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    clean = evaluate_classifier(
        model,
        bundle["inner_val_loader"],
        criterion=criterion,
        device=device,
        num_classes=int(bundle["num_classes"]),
        topk=int(dict(record.get("recipe") or {}).get("topk") or 5),
    )
    stored_clean = float(dict(record.get("inner_val") or {})["macro_f1"])
    replay_delta = float(clean["macro_f1"]) - stored_clean
    if abs(replay_delta) > 1e-10:
        raise RuntimeError(
            f"clean inner replay mismatch for {record_path}: "
            f"{clean['macro_f1']} != {stored_clean}"
        )

    robustness = evaluate_sonar_robustness(
        model,
        bundle["inner_val_loader"],
        device=device,
        num_classes=int(bundle["num_classes"]),
        class_weights=None,
        config={"enabled": True},
        topk=int(dict(record.get("recipe") or {}).get("topk") or 5),
    )
    return {
        "schema_version": 1,
        "run_dir": record_path.parent.name,
        "record_path": str(record_path.resolve()),
        "record_sha256": sha256_file(record_path),
        "fold": fold,
        "seed": seed,
        "candidate_path": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_sha256": observed_split_sha,
        "preprocessing": preprocessing,
        "clean_inner": clean,
        "stored_clean_inner_macro_f1": stored_clean,
        "clean_replay_delta": replay_delta,
        "robustness": robustness,
        "outer_loader_constructed_but_not_iterated": True,
        "outer_evaluation_performed_by_this_script": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_dirs = [Path(value).expanduser().resolve() for value in args.run_dir]
    record_paths = resolve_record_paths(run_dirs)
    if args.expect_count is not None and len(record_paths) != args.expect_count:
        raise RuntimeError(
            f"expected {args.expect_count} formal records, found {len(record_paths)}"
        )
    if not record_paths:
        raise RuntimeError("no formal fold/seed records were found")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir).expanduser().resolve()
    rows = [
        evaluate_record(
            path,
            data_dir=data_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for path in record_paths
    ]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "sonar_operator_inner_robustness_replay_v1",
        "claim_boundary": (
            "Synthetic corruption robustness is replayed on inner validation only. "
            "No outer loader is iterated and no image-restoration claim is made."
        ),
        "device": device,
        "data_dir": str(data_dir),
        "run_dirs": [str(path) for path in run_dirs],
        "record_count": len(rows),
        "records": rows,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "record_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
