"""Fixed-threshold, validation-only sonar diagnostics for recorded detections.

Match against every GT before grouping it. Contrast thresholds come only from
the train images and use the original-resolution bbox/ring definition in
tools/sonar_analysis/complete_analysis.py. This module never trains or reads test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
GROUPS = ("all", "small_area", "very_small_min_side", "contrast_low",
          "contrast_middle", "contrast_high", "contrast_unavailable", "small_and_low")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_labels(path: Path, num_classes: int) -> list[list[float]]:
    """Mirror the strict audited YOLO reader; empty labels are explicit negatives."""
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = [float(item) for item in line.split()]
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError(f"{path}:{line_no}: expected five finite YOLO values")
        c, x, y, w, h = row
        if c != int(c) or not 0 <= c < num_classes:
            raise ValueError(f"{path}:{line_no}: invalid class")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{line_no}: invalid normalized geometry")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"{path}:{line_no}: out-of-image annotation")
        rows.append(row)
    return rows


def _boxes(rows: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    return np.asarray([
        [max(0., (x-w/2)*width), max(0., (y-h/2)*height),
         min(float(width), (x+w/2)*width), min(float(height), (y+h/2)*height)]
        for _, x, y, w, h in rows
    ], dtype=np.float32).reshape(-1, 4)


def _pixel_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
            min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2))))


def _contrast(array: np.ndarray, box: Sequence[float], all_boxes: np.ndarray) -> float | None:
    """Same alpha=2, epsilon=1e-6, exclusion and ddof=0 as the train audit."""
    height, width = array.shape
    x1, y1, x2, y2 = _pixel_box(box, width, height)
    cx, cy, bw, bh = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
    ex1, ey1, ex2, ey2 = _pixel_box((cx-bw, cy-bh, cx+bw, cy+bh), width, height)
    mask = np.ones((ey2-ey1, ex2-ex1), dtype=bool)
    for other in all_boxes:
        ox1, oy1, ox2, oy2 = _pixel_box(other, width, height)
        ax, bx = max(ox1, ex1)-ex1, min(ox2, ex2)-ex1
        ay, by = max(oy1, ey1)-ey1, min(oy2, ey2)-ey1
        if bx > ax and by > ay:
            mask[ay:by, ax:bx] = False
    target = array[y1:y2, x1:x2].reshape(-1)
    background = array[ey1:ey2, ex1:ex2][mask]
    if len(background) < 8 or not len(target):
        return None
    return float(abs(target.mean()-background.mean())/(background.std(ddof=0)+1e-6))


def _image_contrasts(path: Path, rows: Sequence[Sequence[float]]) -> list[float | None]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=float)/255.
    boxes = _boxes(rows, array.shape[1], array.shape[0])
    return [_contrast(array, box, boxes) for box in boxes]


def _images(root: Path, split: str) -> list[Path]:
    if split not in {"train", "valid"}:
        raise ValueError("Only train/valid are permitted")
    paths = sorted(p for p in (root/split/"images").iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"Empty {split} image set")
    return paths


def _identity(root: Path, image: Path, label: Path) -> dict[str, str]:
    return dict(image=image.relative_to(root).as_posix(), image_sha256=_sha(image),
                annotation=label.relative_to(root).as_posix(), annotation_sha256=_sha(label))


def compute_train_contrast_reference(root: str | Path, num_classes: int) -> dict[str, Any]:
    """Derive and bind immutable diagnostic cutoffs without looking at valid scores."""
    root = Path(root).resolve()
    values, identities, unavailable, targets = [], [], 0, 0
    for image in _images(root, "train"):
        label = root/"train"/"labels"/(image.stem+".txt")
        rows = _read_labels(label, num_classes)
        contrasts = _image_contrasts(image, rows)
        values.extend(value for value in contrasts if value is not None)
        unavailable += sum(value is None for value in contrasts)
        targets += len(rows)
        identities.append(_identity(root, image, label))
    if not values:
        raise ValueError("No train GT has a usable contrast measurement")
    q25, q75 = (float(v) for v in np.quantile(values, [.25, .75], method="linear"))
    return dict(q25=q25, q75=q75, num_images=len(identities), num_targets=targets,
                contrast_valid=len(values), contrast_unavailable=unavailable,
                dataset_entries_sha256=_canonical_sha(identities), entries=identities,
                contrast_values_sha256=_canonical_sha(values), threshold_split="train")


def _arrays(record: Mapping[str, Any], *, prediction: bool, num_classes: int) -> dict[str, np.ndarray]:
    boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
    raw_labels = np.asarray(record["labels"], dtype=np.float64).reshape(-1)
    if len(boxes) != len(raw_labels) or not np.isfinite(boxes).all() or not np.isfinite(raw_labels).all():
        raise ValueError("Inconsistent/nonfinite detection records")
    if np.any(raw_labels != np.floor(raw_labels)) or np.any(raw_labels < 0) or np.any(raw_labels >= num_classes):
        raise ValueError("Invalid detection class labels")
    invalid_geometry = boxes[:, 2:] < boxes[:, :2] if prediction else boxes[:, 2:] <= boxes[:, :2]
    if np.any(invalid_geometry):
        raise ValueError("Invalid detection box geometry")
    result = dict(boxes=boxes, labels=raw_labels.astype(np.int64))
    if prediction:
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)
        if len(scores) != len(boxes) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid detection scores")
        result["scores"] = scores
    return result


def _ious(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)), dtype=np.float32)
    intersection = np.maximum(0., np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
                              - np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(axis=2)
    areas = (boxes[:, 2:]-boxes[:, :2]).prod(axis=1)
    gt_areas = (gt[:, 2:]-gt[:, :2]).prod(axis=1)
    return intersection/np.maximum(areas[:, None]+gt_areas[None, :]-intersection, 1e-12)


def match_full_image(pred: Mapping[str, np.ndarray], gt: Mapping[str, np.ndarray],
                     score_threshold: float = .5, iou_threshold: float = .5) -> dict[str, Any]:
    """Greedy class-aware matching first; background FP is class-agnostic IoU<.1."""
    ious = _ious(pred["boxes"], gt["boxes"])
    matched = np.zeros(len(gt["boxes"]), dtype=bool)
    tp = fp = background_fp = 0
    for index in np.argsort(-pred["scores"], kind="stable"):
        if float(pred["scores"][index]) < score_threshold:
            continue
        candidates = np.flatnonzero((gt["labels"] == pred["labels"][index]) & ~matched)
        best = int(candidates[np.argmax(ious[index, candidates])]) if len(candidates) else -1
        if best >= 0 and float(ious[index, best]) >= iou_threshold:
            matched[best] = True
            tp += 1
        else:
            fp += 1
            if not len(gt["boxes"]) or float(ious[index].max()) < .1:
                background_fp += 1
    return dict(matched=matched, TP=tp, FP=fp, background_FP=background_fp)


def _count_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    count = counts["GT"]
    return dict(GT=count, TP=counts["TP"], FN=count-counts["TP"],
                Recall=counts["TP"]/count if count else None)


def evaluate_domain_metrics(
    records: Sequence[Mapping[str, Any]], baseline_config: Mapping[str, Any], *,
    dataset_root: str | Path | None = None, records_path: str | Path | None = None,
    require_full_valid: bool = True,
) -> dict[str, Any]:
    """Return reproducible domain diagnostics from unthresholded best-valid records."""
    root = Path(dataset_root or baseline_config["dataset"]["data_dir"])
    root = root.resolve() if root.is_absolute() else (ROOT/root).resolve()
    num_classes = int(baseline_config["dataset"]["num_classes"])
    image_size = int(baseline_config["dataset"]["image_size"])
    protocol = baseline_config["evaluation_protocol"]
    score_threshold = float(protocol["precision_recall_score_threshold"])
    iou_threshold = float(protocol["precision_recall_iou"])
    if not records:
        raise ValueError("No validation prediction records")
    reference = compute_train_contrast_reference(root, num_classes)
    q25, q75 = reference["q25"], reference["q75"]
    counts = {name: dict(GT=0, TP=0) for name in GROUPS}
    by_class = {name: {group: dict(GT=0, TP=0) for group in GROUPS}
                for name in baseline_config["dataset"]["class_names"]}
    image_rows, identities, seen = [], [], set()
    total_tp = total_fp = background_fp = 0
    for record in records:
        relative = PurePosixPath(str(record["image"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("valid", "images"):
            raise ValueError("Prediction records must identify valid/images relative paths")
        if relative.as_posix() in seen:
            raise ValueError("Duplicate validation image")
        seen.add(relative.as_posix())
        image = root/Path(*relative.parts)
        label = root/"valid"/"labels"/(image.stem+".txt")
        rows = _read_labels(label, num_classes)
        gt = _arrays(record["target"], prediction=False, num_classes=num_classes)
        expected_boxes = _boxes(rows, image_size, image_size).astype(np.float32)
        if len(gt["boxes"]) != len(rows) or not np.allclose(gt["boxes"], expected_boxes, rtol=0., atol=.002):
            raise ValueError("Recorded GT differs from current validation annotations")
        if gt["labels"].tolist() != [int(row[0]) for row in rows]:
            raise ValueError("Recorded GT class/order differs from annotations")
        pred = _arrays(record["prediction"], prediction=True, num_classes=num_classes)
        match = match_full_image(pred, gt, score_threshold, iou_threshold)
        total_tp += match["TP"]
        total_fp += match["FP"]
        background_fp += match["background_FP"]
        contrasts = _image_contrasts(image, rows)
        gt_rows = []
        for index, (box, contrast) in enumerate(zip(gt["boxes"], contrasts)):
            width, height = (float(v) for v in box[2:]-box[:2])
            small = width*height < 1024.
            low = contrast is not None and contrast < q25
            names = ["all"]
            if small:
                names.append("small_area")
            if min(width, height) < 16.:
                names.append("very_small_min_side")
            names.append("contrast_unavailable" if contrast is None else "contrast_low" if low
                         else "contrast_high" if contrast > q75 else "contrast_middle")
            if small and low:
                names.append("small_and_low")
            is_tp = bool(match["matched"][index])
            class_name = baseline_config["dataset"]["class_names"][int(gt["labels"][index])]
            for name in names:
                counts[name]["GT"] += 1
                counts[name]["TP"] += int(is_tp)
                by_class[class_name][name]["GT"] += 1
                by_class[class_name][name]["TP"] += int(is_tp)
            gt_rows.append(dict(index=index, class_name=class_name, area=width*height,
                                min_side=min(width, height), contrast=contrast, TP=is_tp,
                                groups=names))
        identities.append(_identity(root, image, label))
        image_rows.append(dict(image=relative.as_posix(), TP=match["TP"], FP=match["FP"],
                               background_FP=match["background_FP"], GT=len(rows), targets=gt_rows))
    expected = {p.relative_to(root).as_posix() for p in _images(root, "valid")}
    if not seen.issubset(expected) or (require_full_valid and seen != expected):
        raise ValueError("Validation records do not cover exactly the current valid image set")
    n = len(records)
    return dict(
        scope="train_cutoffs_and_valid_diagnostics_only_not_test_or_physical_weak_target_labels",
        num_images=n, full_valid=seen == expected, train_contrast_reference=reference,
        groups={name: _count_summary(value) for name, value in counts.items()},
        per_class={name: {group: _count_summary(value) for group, value in groups.items()}
                   for name, groups in by_class.items()},
        overall=dict(TP=total_tp, FP=total_fp, FN=counts["all"]["GT"]-total_tp,
                     Precision=total_tp/(total_tp+total_fp) if total_tp+total_fp else 0.,
                     Recall=total_tp/counts["all"]["GT"] if counts["all"]["GT"] else None,
                     FP_per_image=total_fp/n, background_FP=background_fp,
                     background_FP_per_image=background_fp/n),
        definitions=dict(
            score_threshold=score_threshold, matching_iou=iou_threshold,
            matching="class-aware full-image one-to-one greedy score order, stable input-index ties; then GT groups",
            background_FP="unmatched detection with class-agnostic max IoU to all GT < 0.1; not all FP",
            small_area="GT width*height < 1024 square network input pixels",
            very_small_min_side="min(GT width,height) < 16 network input pixels",
            size_coordinate_space=f"{image_size}x{image_size} square network input",
            contrast_preprocessing="original image resolution, PIL.convert('L'), float64 /255, float32 xyxy before floor/ceil, no resize or augmentation",
            contrast_formula="abs(mean_bbox-mean_ring)/(std_ring_ddof0+1e-6)",
            contrast_ring="alpha=2 expanded bbox using floor/ceil; exclude every GT bbox; require >=8 background pixels",
            contrast_low="C < train q25", contrast_high="C > train q75",
            contrast_missing="reported as unavailable; excluded from quantile groups",
            quantile_method="numpy.quantile linear",
            overlapping_groups=True,
            AP="These fixed-threshold diagnostics do not replace unthresholded AP or checkpoint selection"),
        provenance=dict(records_sha256=_canonical_sha(records),
                        records_file_sha256=_sha(Path(records_path)) if records_path is not None else None,
                        baseline_config_sha256=_canonical_sha(baseline_config),
                        valid_entries_sha256=_canonical_sha(identities), valid_entries=identities,
                        analysis_source_sha256=_sha(Path(__file__))),
        per_image=image_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT/"configs/baseline_sonar_frozen.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(args.baseline.read_text(encoding="utf-8"))
    result = evaluate_domain_metrics(records, baseline, dataset_root=args.dataset_root, records_path=args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), overall=result["overall"], groups=result["groups"])))


if __name__ == "__main__":
    main()
