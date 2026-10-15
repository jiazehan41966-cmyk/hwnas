"""Pure statistics for retained detector predictions; no inference or training.

All boxes are xyxy in the frozen 640-input coordinate space. These are operating
point diagnostics, not AP or a replacement evaluator. ``no_retained_candidate``
means none suitable survived the frozen decoder/top500/NMS100 pipeline; it does
not prove the network never generated a proposal. GT reasons use predictions
retained at score >= .01; TP/FP matching uses score >= .5. Reasons are exclusive
and ordered as documented in :func:`analyze_records`.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any, Mapping, Sequence

import torch

from hwnas_fpga.training.detection_metrics import box_iou_xyxy


OPERATING_SCORE = 0.5
RETAINED_SCORE = 0.01
BACKGROUND_IOU = 0.1
SIZE_BINS = ("side_lt16", "side_16_to_lt32", "side_ge32")
GT_REASONS = (
    "detected", "assignment_conflict", "score_below_threshold",
    "class_confusion", "localization", "no_retained_candidate",
)
FP_KINDS = (
    "duplicate", "classification", "localization",
    "classification_and_localization", "background",
)


def input_size_bin(box: Sequence[float]) -> str:
    """Bin sqrt(bbox area) in input pixels; these are explicitly not COCO bins."""
    side = sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    return SIZE_BINS[0] if side < 16 else SIZE_BINS[1] if side < 32 else SIZE_BINS[2]


def _prepare(value: Mapping[str, Any], *, prediction: bool, nclasses: int) -> dict:
    boxes = torch.as_tensor(value.get("boxes", []), dtype=torch.float32).detach().cpu()
    if boxes.numel() == 0:
        boxes = torch.empty((0, 4), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be an N x 4 xyxy array")
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("boxes must be finite and have positive width and height")
    raw_labels = torch.as_tensor(value.get("labels", [])).detach().cpu().reshape(-1)
    if not torch.isfinite(raw_labels).all() or (raw_labels != raw_labels.long()).any():
        raise ValueError("labels must be finite integer class indices")
    labels = raw_labels.long()
    if len(labels) != len(boxes) or ((labels < 0) | (labels >= nclasses)).any():
        raise ValueError("boxes/labels length mismatch or class index outside class_names")
    result = dict(boxes=boxes, labels=labels)
    if prediction:
        scores = torch.as_tensor(value.get("scores", []), dtype=torch.float32).detach().cpu().reshape(-1)
        if len(scores) != len(boxes) or not torch.isfinite(scores).all():
            raise ValueError("prediction scores must be finite and match boxes length")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("prediction scores must be in [0, 1]")
        result["scores"] = scores
    return result


def _match(iou: torch.Tensor, pred: dict, gt: dict, threshold: float) -> dict[int, int]:
    """Score-descending, class-aware greedy matching; ties keep original index."""
    active = torch.where(pred["scores"] >= OPERATING_SCORE)[0].tolist()
    active.sort(key=lambda p: (-float(pred["scores"][p]), p))
    available = set(range(len(gt["boxes"])))
    matched = {}
    for p in active:
        candidates = [g for g in available if pred["labels"][p] == gt["labels"][g]
                      and iou[p, g] >= threshold]
        if candidates:
            g = min(candidates, key=lambda g: (-float(iou[p, g]), g))
            matched[p] = g
            available.remove(g)
    return matched


def _best_index(iou: torch.Tensor, indices: list[int]) -> int | None:
    return min(indices, key=lambda i: (-float(iou[i]), i)) if indices else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    split: str,
    seed: int,
    checkpoint: str,
) -> dict[str, Any]:
    """Return JSON-safe summary and CSV-ready rows for train/valid only.

    GT reason priority: detected; high-score matching conflict; same-class
    high-IoU low-score prediction; wrong-class high-IoU prediction; same-class
    .1 <= IoU < .5 prediction; no retained suitable candidate. The latter
    auxiliary cases use score >= .01 and are descriptive, not causal attributions.

    FP priority: duplicate; classification; localization;
    classification_and_localization; background. Matching is performed
    independently at IoU .5 and .75. Per-class/size recall uses GT count as its
    denominator; empty strata yield None, not artificial zero. Prediction rows
    include below-operating-threshold predictions but these do not enter FP.
    """
    if split not in ("train", "valid", "val"):
        raise ValueError("diagnostics admit only train/valid; test remains sealed")
    if not class_names:
        raise ValueError("class_names must not be empty")
    context = dict(split=split, seed=int(seed), checkpoint=str(checkpoint))
    image_rows, gt_rows, prediction_rows = [], [], []
    for record_index, record in enumerate(records):
        image = str(record.get("image", record_index))
        common = dict(context, image=image, image_index=record_index)
        pred = _prepare(record["prediction"], prediction=True, nclasses=len(class_names))
        gt = _prepare(record["target"], prediction=False, nclasses=len(class_names))
        iou = box_iou_xyxy(pred["boxes"], gt["boxes"])
        match50, match75 = _match(iou, pred, gt, .5), _match(iou, pred, gt, .75)
        reverse50, reverse75 = {g: p for p, g in match50.items()}, {g: p for p, g in match75.items()}
        retained = torch.where(pred["scores"] >= RETAINED_SCORE)[0].tolist()
        active = set(torch.where(pred["scores"] >= OPERATING_SCORE)[0].tolist())
        reasons = Counter()
        fp_kinds = Counter()

        for g, box_tensor in enumerate(gt["boxes"]):
            box = box_tensor.tolist()
            same = [p for p in retained if pred["labels"][p] == gt["labels"][g]]
            high = [p for p in same if iou[p, g] >= .5]
            wrong_high = [p for p in retained if pred["labels"][p] != gt["labels"][g] and iou[p, g] >= .5]
            local = [p for p in same if BACKGROUND_IOU <= iou[p, g] < .5]
            if g in reverse50:
                reason = "detected"
                witness = reverse50[g]
            elif any(p in active for p in high):
                reason = "assignment_conflict"
                witness = _best_index(iou[:, g], [p for p in high if p in active])
            elif high:
                reason = "score_below_threshold"
                witness = _best_index(iou[:, g], high)
            elif wrong_high:
                reason = "class_confusion"
                witness = _best_index(iou[:, g], wrong_high)
            elif local:
                reason = "localization"
                witness = _best_index(iou[:, g], local)
            else:
                reason, witness = "no_retained_candidate", None
            best_same = _best_index(iou[:, g], same)
            best_any = _best_index(iou[:, g], retained)
            reasons[reason] += 1
            gt_rows.append(dict(
                common, gt_index=g, class_id=int(gt["labels"][g]), class_name=class_names[int(gt["labels"][g])],
                x1=box[0], y1=box[1], x2=box[2], y2=box[3], width=box[2]-box[0], height=box[3]-box[1],
                equivalent_side=sqrt((box[2]-box[0])*(box[3]-box[1])), size_bin=input_size_bin(box),
                reason=reason, detected_iou50=g in reverse50, detected_iou75=g in reverse75,
                matched_prediction_index=reverse50.get(g),
                matched_iou=float(iou[reverse50[g], g]) if g in reverse50 else None,
                diagnostic_prediction_index=witness,
                diagnostic_score=float(pred["scores"][witness]) if witness is not None else None,
                diagnostic_iou=float(iou[witness, g]) if witness is not None else None,
                diagnostic_above_operating_score=witness in active if witness is not None else None,
                best_same_class_iou=float(iou[best_same, g]) if best_same is not None else None,
                best_any_class_iou=float(iou[best_any, g]) if best_any is not None else None,
            ))

        for p, box_tensor in enumerate(pred["boxes"]):
            box = box_tensor.tolist()
            same = [g for g in range(len(gt["boxes"])) if gt["labels"][g] == pred["labels"][p]]
            wrong = [g for g in range(len(gt["boxes"])) if gt["labels"][g] != pred["labels"][p]]
            high_same = [g for g in same if iou[p, g] >= .5]
            high_wrong = [g for g in wrong if iou[p, g] >= .5]
            near_same = [g for g in same if BACKGROUND_IOU <= iou[p, g] < .5]
            near_wrong = [g for g in wrong if BACKGROUND_IOU <= iou[p, g] < .5]
            if p not in retained:
                kind, associated_gt = "below_retention_threshold", None
            elif p not in active:
                kind, associated_gt = "below_operating_threshold", None
            elif p in match50:
                kind, associated_gt = "TP", match50[p]
            elif high_same:
                kind, associated_gt = "duplicate", _best_index(iou[p], high_same)
            elif high_wrong:
                kind, associated_gt = "classification", _best_index(iou[p], high_wrong)
            elif near_same:
                kind, associated_gt = "localization", _best_index(iou[p], near_same)
            elif near_wrong:
                kind, associated_gt = "classification_and_localization", _best_index(iou[p], near_wrong)
            else:
                kind, associated_gt = "background", None
            if kind in FP_KINDS:
                fp_kinds[kind] += 1
            prediction_rows.append(dict(
                common, prediction_index=p, class_id=int(pred["labels"][p]),
                class_name=class_names[int(pred["labels"][p])], score=float(pred["scores"][p]),
                x1=box[0], y1=box[1], x2=box[2], y2=box[3], size_bin=input_size_bin(box),
                kind=kind, above_operating_score=p in active, retained_at_diagnostic_score=p in retained,
                tp_iou50=p in match50, tp_iou75=p in match75, associated_gt_index=associated_gt,
                associated_iou=float(iou[p, associated_gt]) if associated_gt is not None else None,
                max_gt_iou=float(iou[p].max()) if len(gt["boxes"]) else None,
            ))
        image_rows.append(dict(
            common, gt_count=len(gt["boxes"]), input_prediction_count=len(pred["boxes"]),
            retained_prediction_count=len(retained), operating_prediction_count=len(active),
            tp=len(match50), fp=len(active)-len(match50), fn=len(gt["boxes"])-len(match50),
            tp_iou75=len(match75), fp_iou75=len(active)-len(match75), fn_iou75=len(gt["boxes"])-len(match75),
            precision=_rate(len(match50), len(active)), recall=_rate(len(match50), len(gt["boxes"])),
            **{"gt_"+name: reasons[name] for name in GT_REASONS},
            **{"fp_"+name: fp_kinds[name] for name in FP_KINDS},
        ))

    totals = {key: sum(row[key] for row in image_rows) for key in (
        "gt_count", "input_prediction_count", "retained_prediction_count", "operating_prediction_count",
        "tp", "fp", "fn", "tp_iou75", "fp_iou75", "fn_iou75",
        *("gt_"+name for name in GT_REASONS), *("fp_"+name for name in FP_KINDS),
    )}
    summary = dict(context, images=len(image_rows), **totals,
                   precision=_rate(totals["tp"], totals["operating_prediction_count"]),
                   recall=_rate(totals["tp"], totals["gt_count"]),
                   recall_iou75=_rate(totals["tp_iou75"], totals["gt_count"]),
                   operating_score=OPERATING_SCORE, auxiliary_score=RETAINED_SCORE,
                   background_iou=BACKGROUND_IOU, test_accessed=False,
                   diagnostic_scope="postdecode retained predictions only; not AP",
                   size_definition="sqrt(bbox area) in input pixels: <16, [16,32), >=32; not COCO")
    class_size_rows = []
    for class_id, class_name in enumerate(class_names):
        for size in ("all", *SIZE_BINS):
            rows = [r for r in gt_rows if r["class_id"] == class_id and (size == "all" or r["size_bin"] == size)]
            n, tp50, tp75 = len(rows), sum(r["detected_iou50"] for r in rows), sum(r["detected_iou75"] for r in rows)
            kinds = Counter(r["reason"] for r in rows)
            matched = [r["matched_iou"] for r in rows if r["matched_iou"] is not None]
            class_size_rows.append(dict(
                context, class_id=class_id, class_name=class_name, size_bin=size,
                gt_count=n, tp=tp50, fn=n-tp50, tp_iou75=tp75, fn_iou75=n-tp75,
                recall=_rate(tp50, n), recall_iou75=_rate(tp75, n),
                mean_matched_iou=sum(matched)/len(matched) if matched else None,
                **{"gt_"+name: kinds[name] for name in GT_REASONS},
            ))
    return dict(summary=summary, image_rows=image_rows, gt_rows=gt_rows,
                prediction_rows=prediction_rows, class_size_rows=class_size_rows)
