#!/usr/bin/env python3
"""Summarize YOLO detection box sizes and fit anchor clusters."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "external" / "roboflow_cylider2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "detection_multiscale_640" / "anchor_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit detection anchors from YOLO normalized labels.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits used to fit anchors. Default: train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_anchor_summary(
        Path(args.data_dir),
        image_size=int(args.image_size),
        clusters=int(args.clusters),
        seed=int(args.seed),
        splits=tuple(item.strip() for item in str(args.splits).split(",") if item.strip()),
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"anchors {payload['anchors_by_level']}")
    print(f"mean_best_iou {payload['coverage']['mean_best_iou']:.4f}")
    print(f"artifact {output_path}")


def build_anchor_summary(
    data_dir: Path,
    *,
    image_size: int,
    clusters: int,
    seed: int,
    splits: tuple[str, ...] = ("train",),
) -> dict[str, object]:
    root = _resolve_yolo_root(data_dir)
    if not splits:
        raise ValueError("at least one split is required")
    rows = []
    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for split in splits:
        split_rows = []
        label_root = root / split / "labels"
        if not label_root.is_dir():
            raise FileNotFoundError(f"YOLO label split not found: {label_root}")
        for label_path in sorted(label_root.glob("*.txt")):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                width = float(parts[3]) * float(image_size)
                height = float(parts[4]) * float(image_size)
                split_rows.append((width, height, class_id))
                rows.append((width, height, class_id))
                class_counts[str(class_id)] = class_counts.get(str(class_id), 0) + 1
        split_counts[split] = len(split_rows)
    if len(rows) < clusters:
        raise ValueError(f"Need at least {clusters} boxes to fit anchors, got {len(rows)}")

    boxes = np.asarray([[row[0], row[1]] for row in rows], dtype=np.float64)
    anchors = _kmeans_iou(boxes, clusters=clusters, seed=seed)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    best = _iou_wh(boxes, anchors).max(axis=1)
    anchors_rounded = [[round(float(w), 2), round(float(h), 2)] for w, h in anchors]
    return {
        "artifact_type": "detection_anchor_summary",
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(root),
        "image_size": int(image_size),
        "clusters": int(clusters),
        "seed": int(seed),
        "fit_splits": list(splits),
        "split_box_counts": split_counts,
        "class_box_counts": class_counts,
        "anchors": anchors_rounded,
        "anchors_by_level": [
            anchors_rounded[index:index + 3]
            for index in range(0, len(anchors_rounded), 3)
        ],
        "coverage": {
            "mean_best_iou": float(best.mean()),
            "p10_best_iou": float(np.quantile(best, 0.10)),
            "min_best_iou": float(best.min()),
        },
        "box_size_pixels": _size_summary(boxes),
    }


def _resolve_yolo_root(data_dir: Path) -> Path:
    root = data_dir.expanduser().resolve()
    nested = root / "source_reconstruction" / "v6"
    return nested if nested.is_dir() else root


def _kmeans_iou(boxes: np.ndarray, *, clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = boxes[rng.choice(len(boxes), clusters, replace=False)]
    for _ in range(100):
        assignments = _iou_wh(boxes, centers).argmax(axis=1)
        updated = []
        for cluster_index in range(clusters):
            points = boxes[assignments == cluster_index]
            updated.append(np.median(points, axis=0) if len(points) else centers[cluster_index])
        updated_centers = np.asarray(updated, dtype=np.float64)
        if np.allclose(updated_centers, centers):
            break
        centers = updated_centers
    return centers


def _iou_wh(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    inter = np.minimum(boxes[:, None, 0], anchors[None, :, 0]) * np.minimum(boxes[:, None, 1], anchors[None, :, 1])
    union = boxes[:, None, 0] * boxes[:, None, 1] + anchors[None, :, 0] * anchors[None, :, 1] - inter
    return inter / np.maximum(union, 1e-12)


def _size_summary(boxes: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, values in (("width", boxes[:, 0]), ("height", boxes[:, 1])):
        result[name] = {
            "min": float(values.min()),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return result


if __name__ == "__main__":
    main()
