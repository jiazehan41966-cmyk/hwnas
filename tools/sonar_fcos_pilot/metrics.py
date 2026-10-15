"""Independent COCO detection metrics and explicitly defined error diagnostics.

Boxes use original-image xyxy coordinates. Foreground labels are 1..N; label
zero is not remapped. COCO AP uses every prediction, never a calibrated cutoff.
The three diagnostic error counts partition unmatched objects/predictions:
same-class near-localization pairs, remaining ground truths, remaining predictions.
They are a deliberately simple diagnostic, not the TIDE error taxonomy.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np


def _array(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _boxes(value, width, height, name):
    boxes = _array(value, np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name}: boxes must have shape [N,4].")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name}: boxes must be finite.")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(f"{name}: boxes must have positive width and height.")
    if (np.any(boxes[:, :2] < -1e-4)
            or np.any(boxes[:, [0, 2]] > width + 1e-4)
            or np.any(boxes[:, [1, 3]] > height + 1e-4)):
        raise ValueError(f"{name}: boxes must be in original-image coordinates and bounds.")
    return boxes


def _labels(value, n, num_classes, name):
    raw = _array(value, np.float64)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError(f"{name}: labels must have one entry per box.")
    if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
        raise ValueError(f"{name}: labels must be finite integers.")
    labels = raw.astype(np.int64)
    if np.any(labels < 1) or (num_classes is not None and np.any(labels > num_classes)):
        raise ValueError(f"{name}: foreground labels are 1..N; label 0 is background.")
    return labels


def _normalize(records, predictions, num_classes=None):
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 1):
        raise ValueError("num_classes is the positive number of foreground classes.")
    normalized, ids = [], set()
    for record in records:
        image_id = record["image_id"]
        if image_id in ids:
            raise ValueError(f"Duplicate ground-truth image_id: {image_id!r}.")
        ids.add(image_id)
        width, height = int(record["width"]), int(record["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        boxes = _boxes(record["boxes"], width, height, f"GT {image_id}")
        normalized.append({**record, "width": width, "height": height, "boxes": boxes,
                           "labels": _labels(record["labels"], len(boxes), num_classes,
                                             f"GT {image_id}")})
    predicted = {}
    record_by_id = {record["image_id"]: record for record in normalized}
    for prediction in predictions:
        image_id = prediction["image_id"]
        if image_id not in ids or image_id in predicted:
            raise ValueError(f"Unknown or duplicate prediction image_id: {image_id!r}.")
        record = record_by_id[image_id]
        boxes = _boxes(prediction["boxes"], record["width"], record["height"],
                       f"prediction {image_id}")
        labels = _labels(prediction["labels"], len(boxes), num_classes, f"prediction {image_id}")
        scores = _array(prediction["scores"], np.float64)
        if (scores.ndim != 1 or len(scores) != len(boxes) or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)):
            raise ValueError("Prediction scores must be finite probabilities in [0,1].")
        predicted[image_id] = {"image_id": image_id, "boxes": boxes,
                               "labels": labels, "scores": scores}
    for image_id in ids:
        predicted.setdefault(image_id, {"image_id": image_id, "boxes": np.empty((0, 4)),
                                       "labels": np.empty(0, dtype=np.int64),
                                       "scores": np.empty(0)})
    return normalized, predicted


def _iou(box, boxes):
    if not len(boxes):
        return np.empty(0)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])
                              - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / (area + areas - intersection)


def _match_image(record, prediction, threshold):
    """Match all sizes together before counting the small-target subset."""
    active = np.flatnonzero(prediction["scores"] >= threshold)
    active = active[np.argsort(-prediction["scores"][active], kind="stable")]
    remaining_gt = set(range(len(record["boxes"])))
    true_positive, localization = [], []
    remaining_predictions = []
    for prediction_index in active:
        candidates = [index for index in sorted(remaining_gt)
                      if record["labels"][index] == prediction["labels"][prediction_index]]
        overlaps = _iou(prediction["boxes"][prediction_index], record["boxes"][candidates])
        if len(overlaps) and overlaps.max() >= 0.5:
            gt_index = candidates[int(overlaps.argmax())]
            remaining_gt.remove(gt_index)
            true_positive.append((int(prediction_index), gt_index))
        else:
            remaining_predictions.append(int(prediction_index))
    false_positive = []
    for prediction_index in remaining_predictions:
        candidates = [index for index in sorted(remaining_gt)
                      if record["labels"][index] == prediction["labels"][prediction_index]]
        overlaps = _iou(prediction["boxes"][prediction_index], record["boxes"][candidates])
        eligible = np.flatnonzero((overlaps >= 0.1) & (overlaps < 0.5))
        if len(eligible):
            best = eligible[int(overlaps[eligible].argmax())]
            gt_index = candidates[int(best)]
            remaining_gt.remove(gt_index)
            localization.append((prediction_index, gt_index))
        else:
            false_positive.append(prediction_index)
    return {"true_positive": true_positive, "localization": localization,
            "missed": sorted(remaining_gt), "false_positive": false_positive,
            "active_predictions": [int(index) for index in active]}


def _diagnose(records, predicted, small_area_threshold, threshold):
    if not math.isfinite(float(small_area_threshold)) or not 0 <= small_area_threshold <= 1:
        raise ValueError("small_area_threshold is a training-only relative area in [0,1].")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("score_threshold must be finite and nonnegative.")
    counts = {"true_positive": 0, "localization": 0, "missed": 0, "false_positive": 0,
              "small_gt": 0, "nonsmall_gt": 0, "small_true_positive": 0,
              "nonsmall_true_positive": 0}
    per_image = []
    for record in records:
        match = _match_image(record, predicted[record["image_id"]], threshold)
        areas = (record["boxes"][:, 2:] - record["boxes"][:, :2]).prod(axis=1)
        small = areas / (record["width"] * record["height"]) <= small_area_threshold
        small_gt = int(small.sum())
        small_tp = sum(bool(small[gt_index]) for _, gt_index in match["true_positive"])
        image_counts = {key: len(match[key]) for key in
                        ("true_positive", "localization", "missed", "false_positive")}
        image_counts.update({"image_id": record["image_id"], "small_gt": small_gt,
                             "nonsmall_gt": len(small) - small_gt,
                             "small_true_positive": small_tp,
                             "nonsmall_true_positive": image_counts["true_positive"] - small_tp})
        for key in counts:
            counts[key] += image_counts[key]
        per_image.append(image_counts)
    gt_count = counts["small_gt"] + counts["nonsmall_gt"]
    images = len(records)
    total_fp = counts["false_positive"] + counts["localization"]
    return {**counts, "threshold": float(threshold), "images": images, "ground_truths": gt_count,
            "recall": counts["true_positive"] / gt_count if gt_count else None,
            "small_recall": counts["small_true_positive"] / counts["small_gt"]
            if counts["small_gt"] else None,
            "nonsmall_recall": counts["nonsmall_true_positive"] / counts["nonsmall_gt"]
            if counts["nonsmall_gt"] else None,
            "false_positives_per_image": total_fp / images if images else None,
            "background_false_positives_per_image": counts["false_positive"] / images if images else None,
            "per_image": per_image,
            "protocol": {"correct_iou": 0.5, "near_localization_iou": [0.1, 0.5],
                         "matching": "score-descending same-class one-to-one; all GT sizes together",
                         "false_positives_per_image": "(localization + remaining false_positive) / images",
                         "small_definition": "GT relative box area <= frozen training threshold",
                         "recall_units": "fraction_0_to_1"}}


def _coco_metrics(records, predicted, num_classes):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("True COCO mAP requires pycocotools; no approximate fallback is used.") from error
    image_ids = {record["image_id"]: index + 1 for index, record in enumerate(records)}
    images, annotations, detections = [], [], []
    for record in records:
        image_id = image_ids[record["image_id"]]
        images.append({"id": image_id, "width": record["width"], "height": record["height"]})
        for box, label in zip(record["boxes"], record["labels"]):
            x, y, x2, y2 = [float(value) for value in box]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": int(label), "bbox": [x, y, x2 - x, y2 - y],
                                "area": (x2 - x) * (y2 - y), "iscrowd": 0})
        prediction = predicted[record["image_id"]]
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            x, y, x2, y2 = [float(value) for value in box]
            detections.append({"image_id": image_id, "category_id": int(label),
                               "bbox": [x, y, x2 - x, y2 - y], "score": float(score)})
    if not annotations:
        return {"map": None, "coco_map": None, "map50": None, "map75": None,
                "coco_status": "NO_GROUND_TRUTH"}
    categories = [{"id": label, "name": str(label)} for label in range(1, num_classes + 1)]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {"info": {}, "images": images, "annotations": annotations,
                                "categories": categories}
        ground_truth.createIndex()
        if detections:
            result = ground_truth.loadRes(detections)
        else:
            result = COCO()
            result.dataset = {"info": {}, "images": images, "annotations": [], "categories": categories}
            result.createIndex()
        evaluation = COCOeval(ground_truth, result, "bbox")
        evaluation.params.imgIds = list(image_ids.values())
        evaluation.params.catIds = list(range(1, num_classes + 1))
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    values = [float(evaluation.stats[index] * 100) for index in (0, 1, 2)]
    return {"map": values[0], "coco_map": values[0], "map50": values[1], "map75": values[2],
            "coco_status": "OK"}


def evaluate_predictions(records, predictions, num_classes, small_area_threshold, score_threshold=None):
    """Return AP in percentage points and recall in [0,1].

    score_threshold affects diagnostic counts only; the default is fixed 0.25.
    num_classes counts foreground categories, excluding background.
    """
    records, predicted = _normalize(records, predictions, num_classes)
    threshold = 0.25 if score_threshold is None else float(score_threshold)
    return {**_coco_metrics(records, predicted, num_classes),
            "diagnostics": _diagnose(records, predicted, small_area_threshold, threshold),
            "protocol": {"ap": "pycocotools COCO bbox AP@[0.50:0.95], maxDets=100, all scores",
                         "ap_units": "percentage_points_0_to_100",
                         "ap_small": "Not reported; diagnostic small targets use training-relative area."}}


def calibrate_threshold(records, predictions, num_classes, small_area_threshold, max_fp_per_image=0.1):
    """Validation-only threshold selection; test data must never enter here."""
    if not math.isfinite(float(max_fp_per_image)) or max_fp_per_image < 0:
        raise ValueError("max_fp_per_image must be finite and nonnegative.")
    records, predicted = _normalize(records, predictions, num_classes)
    if not records:
        raise ValueError("Cannot calibrate a threshold on an empty validation set.")
    # Greedy correct matches of a high-score prefix are unchanged when lower
    # scores are appended. Sweep tied score events once, rather than diagnosing
    # the whole validation set once per distinct score (quadratic in dataset size).
    events = {}
    gt_count = sum(len(record["boxes"]) for record in records)
    for record in records:
        prediction = predicted[record["image_id"]]
        correct = {index for index, _ in _match_image(record, prediction, 0.0)["true_positive"]}
        for index, score in enumerate(prediction["scores"]):
            score = float(score)
            count, true_positive = events.get(score, (0, 0))
            events[score] = (count + 1, true_positive + int(index in correct))
    # Scores are bounded by one, so this exact sentinel disables every detection.
    best_threshold = float(np.nextafter(1.0, 2.0))
    best_recall = 0.0
    active = true_positive = 0
    for threshold in sorted(events, reverse=True):
        count, positives = events[threshold]
        active += count
        true_positive += positives
        recall = true_positive / gt_count if gt_count else 0.0
        fp_per_image = (active - true_positive) / len(records)
        if fp_per_image <= max_fp_per_image + 1e-12 and recall > best_recall:
            best_recall, best_threshold = recall, threshold
    selected = _diagnose(records, predicted, small_area_threshold, best_threshold)
    return {"threshold": selected["threshold"], "max_fp_per_image": float(max_fp_per_image),
            "diagnostics": selected, "selection_set": "validation_only",
            "selection": "highest overall recall within total FP/image budget; ties choose higher threshold",
            "all_predictions_disabled": selected["threshold"] > 1.0}


def baseline_gate(metrics):
    """Use fixed-threshold diagnostics, not the calibrated low-FP operating point."""
    diagnostic = metrics.get("diagnostics", metrics)
    if abs(diagnostic["threshold"] - 0.25) > 1e-12:
        raise ValueError("baseline_gate requires the prespecified diagnostic threshold 0.25.")
    missed, false_positive, localization = [diagnostic[key]
                                           for key in ("missed", "false_positive", "localization")]
    if localization > missed + false_positive:
        return {"status": "STOP_LOCALIZATION", "reason": "框偏移数量超过漏检和剩余误报之和。"}
    small, nonsmall = diagnostic["small_recall"], diagnostic["nonsmall_recall"]
    small_disadvantage = small is not None and nonsmall is not None and small < nonsmall
    if missed + false_positive + localization > 0 and (small_disadvantage or false_positive > 0):
        return {"status": "GO", "reason": "小目标检出比例较低或存在背景、重复框、错类别误报。"}
    return {"status": "STOP_NO_TARGETED_ERROR", "reason": "当前诊断尚未支持局部细节与上下文搜索方向。"}


def write_error_examples(records, predictions, output_dir, small_area_threshold,
                         score_threshold=0.25, max_examples=12):
    """Write at most 12 original-coordinate overlays; return saved absolute paths.

    Ground truth: green; correct prediction: blue; near-localization: orange;
    remaining prediction: red. Captions use ASCII for portable Pillow fonts.
    """
    from PIL import Image, ImageDraw

    records, predicted = _normalize(records, predictions)
    diagnostics = _diagnose(records, predicted, small_area_threshold, score_threshold)
    counts = {item["image_id"]: item for item in diagnostics["per_image"]}
    selected = sorted(records, key=lambda record: -sum(counts[record["image_id"]][key]
                                                     for key in ("missed", "false_positive", "localization")))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for record in selected:
        count = counts[record["image_id"]]
        if not sum(count[key] for key in ("missed", "false_positive", "localization")):
            continue
        if len(paths) >= min(12, max(0, int(max_examples))):
            break
        if not record.get("image_path"):
            raise ValueError("Error examples require original image_path in each record.")
        image = Image.open(record["image_path"]).convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError("Example image dimensions differ from original-coordinate record.")
        draw = ImageDraw.Draw(image)
        for box, label in zip(record["boxes"], record["labels"]):
            draw.rectangle(tuple(box), outline="lime", width=2)
            draw.text((float(box[0]), float(box[1])), f"GT {int(label)}", fill="lime")
        prediction = predicted[record["image_id"]]
        match = _match_image(record, prediction, score_threshold)
        colored = ([(index, "dodgerblue") for index, _ in match["true_positive"]]
                   + [(index, "orange") for index, _ in match["localization"]]
                   + [(index, "red") for index in match["false_positive"]])
        for index, color in colored:
            box = prediction["boxes"][index]
            draw.rectangle(tuple(box), outline=color, width=2)
            draw.text((float(box[0]), max(0, float(box[1]) - 12)),
                      f"pred {int(prediction['labels'][index])} {prediction['scores'][index]:.2f}", fill=color)
        canvas = Image.new("RGB", (image.width, image.height + 32), "white")
        canvas.paste(image, (0, 32))
        ImageDraw.Draw(canvas).text((4, 4),
                                   f"miss={count['missed']} remaining_fp={count['false_positive']} "
                                   f"localization={count['localization']} threshold={score_threshold:.3f}",
                                   fill="black")
        path = output_dir / f"error_{len(paths) + 1:02d}.png"
        canvas.save(path)
        paths.append(str(path.resolve()))
    return paths
