"""Fixed UATD quality diagnostics shared by AAA and quality-score FCOS.

The six-stage coverage definition comes from the 2026-10-08 full validation
diagnosis. Counts are GT availability, not matched detection recall. Wrong-class
FP counts follow the separate 2026-10-08 class-confusion review.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import io
import types

import numpy as np

from tools.sonar_fcos_pilot.metrics import _normalize


STAGES = ("raw", "threshold", "topk", "nms", "top100", "final")
GOOD_IOU = 0.75
WRONG_CLASS_IOU = 0.5
WRONG_CLASS_SCORE = 0.25


def _iou(boxes, box):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], box[2:])
                              - np.maximum(boxes[:, :2], box[:2])).prod(axis=1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    gt_area = np.maximum(0, box[2:] - box[:2]).prod()
    union = area + gt_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def stage_rows_for_image(record, raw_boxes, raw_scores, candidate_boxes,
                         candidate_labels, nms_keep, top100_keep, score_threshold):
    """Reduce a native postprocess trace in original-image coordinates.

    raw_boxes has one decoded box per point; raw_scores includes every class,
    including background. candidate_* are the actual per-level top-k pool.
    No score cut is applied to raw geometry. NMS/top100 arrays index that pool.
    """
    raw_boxes = np.asarray(raw_boxes).reshape(-1, 4)
    raw_scores = np.asarray(raw_scores)
    boxes = np.asarray(candidate_boxes).reshape(-1, 4)
    labels = np.asarray(candidate_labels, dtype=np.int64)
    if raw_scores.ndim != 2 or len(raw_scores) != len(raw_boxes):
        raise ValueError("Raw scores must have shape [points, classes].")
    if len(labels) != len(boxes):
        raise ValueError("One label is required for each candidate box.")
    nms_mask = np.zeros(len(boxes), dtype=bool)
    top_mask = np.zeros(len(boxes), dtype=bool)
    nms_mask[np.asarray(nms_keep, dtype=np.int64)] = True
    top_mask[np.asarray(top100_keep, dtype=np.int64)] = True
    if np.any(top_mask & ~nms_mask):
        raise ValueError("Top100 must be a subset of NMS survivors.")
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (labels > 0)
    rows = []
    for index, (gt, label) in enumerate(zip(record["boxes"], record["labels"])):
        label = int(label)
        if not 0 < label < raw_scores.shape[1]:
            raise ValueError("GT foreground class absent from raw score columns.")
        raw_iou, candidate_iou = _iou(raw_boxes, gt), _iou(boxes, gt)
        masks = {"raw": np.ones(len(raw_boxes), dtype=bool),
                 "threshold": raw_scores[:, label] > score_threshold,
                 "topk": labels == label,
                 "nms": (labels == label) & nms_mask,
                 "top100": (labels == label) & top_mask,
                 "final": (labels == label) & top_mask & valid}
        stages = {}
        for stage in STAGES:
            values = raw_iou if stage in ("raw", "threshold") else candidate_iou
            mask = masks[stage]
            stages[stage] = {"count": int(mask.sum()),
                             "best_iou": float(values[mask].max()) if mask.any() else 0.0,
                             "good75_count": int((mask & (values >= GOOD_IOU)).sum()),
                             "good85_count": int((mask & (values >= .85)).sum())}
        first_zero = next((stage for stage in STAGES if not stages[stage]["good75_count"]),
                          "covered_final")
        rows.append({"image_id": record["image_id"], "gt_index": index, "label": label,
                     "stages": stages, "mechanism_first_zero": first_zero})
    return rows


def summarize_stage_rows(rows, records):
    """Validate complete, nested GT traces and count the mutually exclusive losses."""
    expected = {(record["image_id"], index): int(label) for record in records
                for index, label in enumerate(record["labels"])}
    actual = {(row["image_id"], row["gt_index"]): row["label"] for row in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("Stage rows must cover every supplied GT exactly once with its label.")
    covered = Counter({stage: 0 for stage in STAGES})
    first_zero = Counter({stage: 0 for stage in (*STAGES, "covered_final")})
    for row in rows:
        previous_count = previous_good = float("inf")
        previous_best = float("inf")
        for stage in STAGES:
            value = row["stages"][stage]
            count, good, best = value["count"], value["good75_count"], value["best_iou"]
            if not (0 <= good <= count <= previous_count and good <= previous_good
                    and 0 <= best <= 1 and best <= previous_best + 1e-6):
                raise ValueError(f"Invalid or non-nested stage: {stage}")
            if bool(good) != (best >= GOOD_IOU):
                raise ValueError("Good-box count disagrees with best IoU.")
            previous_count, previous_good, previous_best = count, good, best
            covered[stage] += int(good > 0)
        first = next((stage for stage in STAGES if row["stages"][stage]["good75_count"] == 0),
                     "covered_final")
        if row.get("mechanism_first_zero", first) != first:
            raise ValueError("Reported first-loss stage disagrees with coverage.")
        first_zero[first] += 1
    return {"images": len(records), "ground_truths": len(rows),
            "raw_no_good_box": first_zero["raw"], "nms_first_loss": first_zero["nms"],
            "first_zero": dict(first_zero), "stage_covered_gt": dict(covered),
            "protocol": {"good_iou": GOOD_IOU, "class_aware": True,
                         "raw_geometry_ignores_class_score": True,
                         "definition": "same-class candidate availability per GT; not detection recall",
                         "stages": list(STAGES)}}


def wrong_class_diagnostics(records, predictions, num_classes):
    """Count final wrong-only class detections, rather than affected GT objects.

    A prediction is eligible at score >= .25 if it overlaps a different-class
    GT at IoU >= .5 and has no same-class GT at IoU >= .5. In particular a
    duplicate same-class box is not a wrong-class FP, even if another-class GT
    overlaps more strongly. The definition does not depend on TP matching order.
    """
    records, predicted = _normalize(records, predictions, num_classes)
    per_image = []
    for record in records:
        pred = predicted[record["image_id"]]
        wrong = []
        for index in np.flatnonzero(pred["scores"] >= WRONG_CLASS_SCORE):
            overlaps = _iou(record["boxes"], pred["boxes"][index])
            same = record["labels"] == pred["labels"][index]
            if (np.any(overlaps[~same] >= WRONG_CLASS_IOU)
                    and not np.any(overlaps[same] >= WRONG_CLASS_IOU)):
                wrong.append(int(index))
        per_image.append({"image_id": record["image_id"], "wrong_class_fp": len(wrong),
                          "prediction_indices": wrong})
    return {"wrong_class_fp": sum(row["wrong_class_fp"] for row in per_image),
            "per_image": per_image,
            "protocol": {"score_gte": WRONG_CLASS_SCORE, "other_class_iou_gte": WRONG_CLASS_IOU,
                         "same_class_iou_lt": WRONG_CLASS_IOU,
                         "unit": "final retained prediction boxes, not unique GT objects"}}


def evaluate_quality_predictions(records, predictions, num_classes):
    """True COCO bbox AP75/AP85/AP50:95, in percentage points, plus wrong-class FP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    records, predicted = _normalize(records, predictions, num_classes)
    wrong = wrong_class_diagnostics(records, list(predicted.values()), num_classes)
    images, annotations, detections = [], [], []
    for image_id, record in enumerate(records, 1):
        images.append({"id": image_id, "width": record["width"], "height": record["height"]})
        for box, label in zip(record["boxes"], record["labels"]):
            x, y, x2, y2 = map(float, box)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": int(label), "bbox": [x, y, x2 - x, y2 - y],
                                "area": (x2 - x) * (y2 - y), "iscrowd": 0})
        pred = predicted[record["image_id"]]
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x, y, x2, y2 = map(float, box)
            detections.append({"image_id": image_id, "category_id": int(label),
                               "bbox": [x, y, x2 - x, y2 - y], "score": float(score)})
    protocol = {"ap_units": "percentage_points_0_to_100", "ap": "pycocotools COCO bbox",
                "iou_thresholds": [round(.5 + .05 * index, 2) for index in range(10)],
                "max_detections": 100, "ap_score_cut": None,
                "wrong_class": wrong["protocol"]}
    result = {"wrong_class_fp": wrong["wrong_class_fp"], "protocol": protocol}
    if not annotations:
        return {**result, "map": None, "coco_map": None, "ap50": None,
                "ap75": None, "ap85": None, "coco_status": "NO_GROUND_TRUTH"}
    categories = [{"id": label, "name": str(label)} for label in range(1, num_classes + 1)]
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"info": {}, "images": images, "annotations": annotations,
                         "categories": categories}
        truth.createIndex()
        if detections:
            detected = truth.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {"info": {}, "images": images, "annotations": [],
                                "categories": categories}
            detected.createIndex()
        evaluation = COCOeval(truth, detected, "bbox")
        evaluation.params.imgIds = [image["id"] for image in images]
        evaluation.params.catIds = list(range(1, num_classes + 1))
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    precision = evaluation.eval["precision"][:, :, :, 0, -1]

    def mean_valid(values):
        valid = values[values > -1]
        return float(valid.mean() * 100) if valid.size else None

    curve = {f"{threshold:.2f}": mean_valid(precision[index])
             for index, threshold in enumerate(evaluation.params.iouThrs)}
    return {**result, "map": mean_valid(precision), "coco_map": mean_valid(precision),
            "ap50": curve["0.50"], "ap75": curve["0.75"], "ap85": curve["0.85"],
            "ap_iou_curve": curve, "coco_status": "OK"}


class StageDiagnosticTracer:
    """Context manager tracing the existing model postprocessor without altering outputs.

    Call begin_batch(CPU_targets, original_records) immediately before each
    inference. The traced reconstruction must match the model's authoritative
    native boxes/scores/labels exactly. Summary requires the complete record set.
    """

    def __init__(self, model):
        self.model = model
        self.rows = []
        self.images = []
        self._batch = None
        self._active = False
        self.score_mode = getattr(model, "_sonar_quality_score_mode", "baseline_centerness")
        if self.score_mode not in ("baseline_centerness", "quality_sigmoid"):
            raise ValueError(f"Unsupported score mode: {self.score_mode}")

    def __enter__(self):
        if self._active:
            raise RuntimeError("Trace context cannot be entered twice.")
        self._original = self.model.postprocess_detections
        self._active = True

        def wrapped(_model, heads, anchors, image_shapes):
            official = self._original(heads, anchors, image_shapes)
            self._trace(heads, anchors, image_shapes, official)
            return official

        self.model.postprocess_detections = types.MethodType(wrapped, self.model)
        return self

    def __exit__(self, *_):
        self.model.postprocess_detections = self._original
        self._active = False
        self._batch = None

    def begin_batch(self, targets, records):
        if not self._active or self._batch is not None:
            raise RuntimeError("Trace must be active with no unconsumed batch.")
        if len(targets) != len(records):
            raise ValueError("Target and record batches must align.")
        if len({record["image_id"] for record in records}) != len(records):
            raise ValueError("Batch image identities must be unique.")
        for target, record in zip(targets, records):
            if int(target["image_id"].reshape(-1)[0]) != record["image_id"]:
                raise ValueError("Target/record image identities differ.")
            if record["image_id"] in self.images:
                raise ValueError("An image cannot be traced twice.")
        self._batch = (targets, records)

    @staticmethod
    def _inverse(boxes, target):
        # Preserve native dtype during the inverse, as in the archived diagnosis.
        boxes = boxes.detach().cpu().clone()
        scale = target["resize_scale"].detach().cpu().to(boxes.dtype)
        pad = target["pad_offset"].detach().cpu().to(boxes.dtype)
        height, width = map(int, target["original_size"])
        boxes = (boxes - pad.repeat(2)) / scale.repeat(2)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        return boxes.numpy()

    def _trace(self, heads, anchors, image_shapes, official):
        import torch
        from torchvision.ops import boxes as box_ops

        if self._batch is None:
            raise RuntimeError("begin_batch must precede traced inference.")
        targets, records = self._batch
        if len(targets) != len(official):
            raise ValueError("Traced inference batch changed size.")
        for image_index, (target, record) in enumerate(zip(targets, records)):
            raw_boxes, raw_scores, boxes, scores, labels = [], [], [], [], []
            for level, anchor in enumerate(anchors[image_index]):
                logits = heads["cls_logits"][level][image_index]
                regression = heads["bbox_regression"][level][image_index]
                # The distribution head integrates in FP32; its authoritative
                # postprocessor also computes the joint sigmoid/decode in FP32.
                # AAA must retain its historical native AMP path unchanged.
                if self.score_mode == "quality_sigmoid":
                    logits, regression, anchor = logits.float(), regression.float(), anchor.float()
                probabilities = logits.sigmoid()
                native_scores = (probabilities if self.score_mode == "quality_sigmoid" else
                                 torch.sqrt(probabilities * heads["bbox_ctrness"][level][image_index].sigmoid()))
                flat_scores = native_scores.flatten()
                passed = flat_scores > self.model.score_thresh
                indices = torch.where(passed)[0]
                selected_scores, order = flat_scores[passed].topk(min(len(indices), self.model.topk_candidates))
                indices = indices[order]
                points = torch.div(indices, logits.shape[-1], rounding_mode="floor")
                candidate_labels = indices % logits.shape[-1]
                candidate_boxes = box_ops.clip_boxes_to_image(
                    self.model.box_coder.decode(regression[points], anchor[points]), image_shapes[image_index])
                raw = box_ops.clip_boxes_to_image(self.model.box_coder.decode(regression, anchor),
                                                 image_shapes[image_index])
                raw_boxes.append(self._inverse(raw, target))
                raw_scores.append(native_scores.detach().cpu().numpy())
                boxes.append(candidate_boxes)
                scores.append(selected_scores)
                labels.append(candidate_labels)
            boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            nms_keep = box_ops.batched_nms(boxes, scores, labels, self.model.nms_thresh)
            top_keep = nms_keep[:self.model.detections_per_img]
            replay = {"boxes": boxes[top_keep], "scores": scores[top_keep], "labels": labels[top_keep]}
            if not all(torch.equal(official[image_index][key], value) for key, value in replay.items()):
                raise RuntimeError("Diagnostic reconstruction differs from authoritative postprocessing.")
            self.rows.extend(stage_rows_for_image(
                record, np.concatenate(raw_boxes), np.concatenate(raw_scores), self._inverse(boxes, target),
                labels.detach().cpu().numpy(), nms_keep.detach().cpu().numpy(),
                top_keep.detach().cpu().numpy(), float(self.model.score_thresh)))
            self.images.append(record["image_id"])
        self._batch = None

    def summary(self, records):
        expected_ids = [record["image_id"] for record in records]
        if (len(set(expected_ids)) != len(expected_ids) or len(self.images) != len(expected_ids)
                or set(expected_ids) != set(self.images)):
            raise ValueError("Every evaluation image must be traced exactly once.")
        if self._batch is not None:
            raise RuntimeError("Cannot summarize an unconsumed batch.")
        value = summarize_stage_rows(self.rows, records)
        value["protocol"].update({"score_mode": self.score_mode,
                                   "score_threshold_strict_gt": float(self.model.score_thresh),
                                   "per_level_topk": int(self.model.topk_candidates),
                                   "nms_iou": float(self.model.nms_thresh),
                                   "detections_per_image": int(self.model.detections_per_img),
                                   "postprocess_parity": "EXACT"})
        return value
