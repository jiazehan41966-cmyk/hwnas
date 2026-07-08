#!/usr/bin/env python3
"""Create synthetic multiplicative speckle pairs for the G5 E3b protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hwnas_fpga.data import NKSIDDataset
from measure_sonar_image_quality import apply_transform


TRANSFORM_DIR_NAMES = {
    "identity": "identity",
    "denoise": "denoised",
    "edge": "edge",
    "edge_enhanced": "edge_enhanced",
}


def parse_csv_numbers(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one speckle level is required")
    if any(value <= 0 for value in values):
        raise ValueError("Speckle levels must be positive")
    return values


def parse_transforms(text: str) -> list[str]:
    transforms = [item.strip() for item in str(text).split(",") if item.strip()]
    for name in transforms:
        if name not in TRANSFORM_DIR_NAMES:
            raise ValueError(f"Unsupported candidate transform: {name}")
    return transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--split", default="val", choices=("train", "val", "full"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--levels", default="1,2,4", help="Comma-separated looks L values.")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="data/synthetic_speckle")
    parser.add_argument(
        "--candidate-transforms",
        default="denoise",
        help="Optional comma-separated transforms to apply to noisy images.",
    )
    return parser.parse_args()


def image_to_float01(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float64) / 255.0


def save_float_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.uint8(np.rint(np.clip(image, 0.0, 1.0) * 255.0))
    Image.fromarray(payload, mode="L").save(path)


def sample_output_name(sample_index: int, source_path: str | Path) -> str:
    source = Path(source_path)
    stem = source.stem or f"sample_{sample_index:05d}"
    return f"{sample_index:05d}_{stem}.png"


def make_pairs(args: argparse.Namespace) -> dict[str, Any]:
    levels = parse_csv_numbers(args.levels)
    candidate_transforms = parse_transforms(args.candidate_transforms)
    output_root = Path(args.output_dir)
    dataset = NKSIDDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        transform=lambda image: image,
        is_training=False,
        fold=args.fold,
        use_kfold=args.split != "full",
        split=args.split,
        output_channels=1,
        image_error_policy="raise",
    )
    samples = dataset.samples[: args.max_samples] if args.max_samples else dataset.samples

    level_summaries: list[dict[str, Any]] = []
    for level in levels:
        level_name = f"L{int(level) if float(level).is_integer() else str(level).replace('.', 'p')}"
        counts = {
            "ref": 0,
            "noisy": 0,
            **{TRANSFORM_DIR_NAMES[name]: 0 for name in candidate_transforms},
        }
        for sample_index, (image_path, label) in enumerate(samples):
            clean = image_to_float01(image_path, args.image_size)
            rng = np.random.default_rng(int(args.seed) + sample_index)
            speckle = rng.gamma(shape=float(level), scale=1.0 / float(level), size=clean.shape)
            noisy = np.clip(clean * speckle, 0.0, 1.0)

            class_name = dataset.label_to_class.get(int(label), str(label))
            file_name = sample_output_name(sample_index, image_path)
            base = output_root / level_name
            save_float_image(base / "ref" / class_name / file_name, clean)
            save_float_image(base / "noisy" / class_name / file_name, noisy)
            counts["ref"] += 1
            counts["noisy"] += 1

            for transform_name in candidate_transforms:
                transformed = apply_transform(noisy, transform_name)
                directory_name = TRANSFORM_DIR_NAMES[transform_name]
                save_float_image(base / directory_name / class_name / file_name, transformed)
                counts[directory_name] += 1

        level_summaries.append(
            {
                "level": float(level),
                "level_name": level_name,
                "root": str((output_root / level_name).resolve()),
                "counts": counts,
            }
        )

    manifest = {
        "schema_version": 1,
        "protocol": "synthetic_multiplicative_gamma_speckle_v1",
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "fold": int(args.fold),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "per_image_seed_rule": "global_seed + sample_index",
        "samples": len(samples),
        "levels": level_summaries,
        "candidate_transforms": candidate_transforms,
        "boundary": (
            "NKSID images are used as relative clean references for synthetic "
            "speckle only; this is not real paired sonar restoration ground truth."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = make_pairs(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(Path(args.output_dir).resolve()),
                "samples": manifest["samples"],
                "levels": [item["level_name"] for item in manifest["levels"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
