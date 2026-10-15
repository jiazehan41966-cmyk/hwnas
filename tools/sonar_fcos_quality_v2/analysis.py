"""Frozen-workpoint diagnostics and exact paired image bootstrap for FCOS v2.

AP always consumes the complete saved native prediction cache.  The FPPI
workpoint and fixed score .25 are separate descriptive views.  Bootstrap
resamples images, never predictions or training seeds, and each duplicate
image occurrence has its own matching.  Per-image matching is delegated to
the installed official pycocotools COCOeval implementation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import math

import numpy as np

from tools.sonar_fcos_pilot.metrics import _normalize
from tools.sonar_fcos_quality.diagnostics import (
    STAGES, _iou, evaluate_quality_predictions, summarize_stage_rows,
)


TARGET_FPPI = 0.5
FIXED_SCORE = 0.25
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261012
SEEDS = (42, 43, 44)
VARIANTS = ("AAA", "QFL_DFL")


def _matching_cache(records, predictions, num_classes):
    """Class-aware IoU .5 greedy matching in stable descending score order."""
    records, predicted = _normalize(records, predictions, num_classes)
    if not records:
        raise ValueError("A workpoint needs at least one evaluation image")
    cache = []
    for record in records:
        pred = predicted[record["image_id"]]
        count = len(pred["scores"])
        true_positive = np.zeros(count, dtype=bool)
        wrong_class = np.zeros(count, dtype=bool)
        used = np.zeros(len(record["labels"]), dtype=bool)
        order = np.argsort(-pred["scores"], kind="stable")
        for index in order:
            overlaps = _iou(record["boxes"], pred["boxes"][index])
            same = record["labels"] == pred["labels"][index]
            candidates = np.flatnonzero(same & ~used & (overlaps >= .5))
            if candidates.size:
                # Stable original GT order resolves equal-IoU ties.
                match = candidates[int(np.argmax(overlaps[candidates]))]
                true_positive[index] = used[match] = True
            wrong_class[index] = bool(np.any(overlaps[~same] >= .5)
                                      and not np.any(overlaps[same] >= .5))
        cache.append({"image_id": record["image_id"], "scores": pred["scores"],
                      "true_positive": true_positive, "wrong_class": wrong_class,
                      "ground_truths": len(record["labels"])})
    return records, cache


def _counts_at_threshold(cache, threshold):
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("Score threshold must be finite and nonnegative")
    rows = []
    for item in cache:
        kept = item["scores"] >= threshold
        tp = int(np.count_nonzero(kept & item["true_positive"]))
        detections = int(np.count_nonzero(kept))
        rows.append({"image_id": item["image_id"], "detections": detections,
                     "true_positive": tp, "false_positive": detections - tp,
                     "false_negative": item["ground_truths"] - tp,
                     "wrong_class_fp": int(np.count_nonzero(kept & item["wrong_class"]))})
    totals = {key: sum(row[key] for row in rows) for key in (
        "detections", "true_positive", "false_positive", "false_negative", "wrong_class_fp")}
    return {**totals, "images": len(cache), "fppi": totals["false_positive"] / len(cache),
            "per_image": rows,
            "protocol": {"score_comparison": ">=", "matching_iou": .5,
                         "matching": "per-image class-aware descending-score greedy",
                         "equal_score_order": "saved prediction order",
                         "equal_iou_gt_order": "original GT order",
                         "wrong_class": "other-class IoU>=.5 and every same-class IoU<.5"}}


def counts_at_threshold(records, predictions, num_classes, threshold=FIXED_SCORE):
    """Integer TP/FP/FN and historical wrong-only class counts at one cutoff."""
    _, cache = _matching_cache(records, predictions, num_classes)
    return _counts_at_threshold(cache, threshold)


def _choose_workpoint(cache, target_fppi):
    if not math.isfinite(float(target_fppi)) or target_fppi < 0:
        raise ValueError("Target FPPI must be finite and nonnegative")
    # Work on integer counts, never interpolate a fraction of a detection.
    real_budget = float(target_fppi) * len(cache)
    budget = math.floor(real_budget)
    scores = np.concatenate([row["scores"] for row in cache])
    fp = np.concatenate([~row["true_positive"] for row in cache])
    next_score = None
    next_fp_count = None
    if not scores.size:
        threshold = 0.0
        reason = "empty_prediction_cache"
    else:
        order = np.argsort(-scores, kind="stable")
        ordered = scores[order]
        ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1]
        cumulative_fp = np.cumsum(fp[order], dtype=np.int64)[ends]
        legal = np.flatnonzero(cumulative_fp <= budget)
        if len(legal) == len(ends):
            threshold = 0.0  # Include the complete saved cache.
            reason = "all_saved_predictions_within_budget"
        elif not legal.size:
            threshold = float(np.nextafter(ordered[0], np.inf))
            next_score, next_fp_count = float(ordered[0]), int(cumulative_fp[0])
            reason = "highest_score_tie_group_exceeds_budget"
        else:
            last = int(legal[-1])
            threshold = float(ordered[ends[last]])
            next_score = float(ordered[ends[last] + 1])
            next_fp_count = int(cumulative_fp[last + 1])
            reason = "next_score_group_exceeds_budget"
    counts = _counts_at_threshold(cache, threshold)
    actual = counts["false_positive"]
    if actual > budget:
        raise AssertionError("Selected workpoint exceeds its integer FP budget")
    exact = actual == real_budget
    return {"target_fppi": float(target_fppi), "threshold": threshold,
            "integer_fp_budget": budget, "real_valued_fp_budget": real_budget,
            "achieved_fppi": counts["fppi"], "exact_target_reached": exact,
            "target_unreachable_in_saved_cache": not exact,
            "fppi_shortfall": float(target_fppi) - counts["fppi"],
            "integer_budget_shortfall": budget - actual,
            "selection_status": "EXACT" if exact else "NEAREST_AT_OR_BELOW_TARGET",
            "selection_reason": reason, "next_excluded_score": next_score,
            "false_positives_if_next_score_group_included": next_fp_count,
            "threshold_candidates": "unique saved final scores plus 0 and nextafter(max_score,+inf)",
            "selection_rule": "most inclusive whole-score-group threshold with FP<=floor(target*images)",
            "interpolation": False, "counts": counts}


def equal_fppi_workpoint(records, predictions, num_classes, target_fppi=TARGET_FPPI):
    """Select a descriptive operating point without splitting equal-score groups."""
    _, cache = _matching_cache(records, predictions, num_classes)
    return _choose_workpoint(cache, target_fppi)


def filtered_stage_diagnostics(stage_rows, records, threshold):
    """Filter the same native stages; raw geometry is score independent.

    No NMS/top-k is rerun.  good75_max_score from the native tracer is required
    for every nonempty stage.  Counts remain GT availability, not matched
    detection recall.  Missing scores explicitly make this extra view absent.
    """
    native = summarize_stage_rows(stage_rows, records)
    missing = [(row["image_id"], row["gt_index"], stage)
               for row in stage_rows for stage in STAGES[1:]
               if "good75_max_score" not in row["stages"][stage]]
    if missing:
        return {"status": "UNAVAILABLE", "reason": "native stage good-box scores not retained",
                "missing_stage_scores": len(missing), "score_threshold_gte": float(threshold)}
    covered = {stage: 0 for stage in STAGES}
    first_loss = {stage: 0 for stage in (*STAGES, "covered_final")}
    for row in stage_rows:
        availability = {"raw": bool(row["stages"]["raw"]["good75_count"])}
        previous_score = math.inf
        for stage in STAGES[1:]:
            value = row["stages"][stage]
            score = value["good75_max_score"]
            if value["good75_count"]:
                if score is None or not math.isfinite(float(score)) or not 0 <= score <= 1:
                    raise ValueError("Nonempty good-box stage needs a finite probability score")
                if score > previous_score:
                    raise ValueError("Good-box scores must be nested across native stages")
                previous_score = score
                availability[stage] = bool(score >= threshold)
            else:
                if score is not None:
                    raise ValueError("Empty good-box stage must have a null maximum score")
                previous_score = -math.inf
                availability[stage] = False
        for stage in STAGES:
            covered[stage] += int(availability[stage])
        first = next((stage for stage in STAGES if not availability[stage]), "covered_final")
        first_loss[first] += 1
    return {"status": "AVAILABLE", "images": len(records), "ground_truths": len(stage_rows),
            "raw_no_good_box": native["raw_no_good_box"],
            "nms_first_loss": first_loss["nms"],
            "final_no_good_box": len(stage_rows) - covered["final"],
            "first_zero": first_loss, "stage_covered_gt": covered,
            "protocol": {**native["protocol"], "score_threshold_gte": float(threshold),
                         "raw_geometry_unchanged": True,
                         "native_candidate_chain_unchanged": True,
                         "nms_or_topk_recomputed": False,
                         "role": "supplementary_workpoint_not_primary_gate"}}


def evaluate_checkpoint(records, predictions, num_classes, stage_rows, *, selection=None, epoch=None):
    """One full-cache AP evaluation and three explicitly separate diagnoses."""
    normalized, cache = _matching_cache(records, predictions, num_classes)
    metrics = evaluate_quality_predictions(records, predictions, num_classes)
    native = summarize_stage_rows(stage_rows, records)
    fixed_counts = _counts_at_threshold(cache, FIXED_SCORE)
    native["wrong_class_fp"] = fixed_counts["wrong_class_fp"]
    native["final_no_good_box"] = len(stage_rows) - native["stage_covered_gt"]["final"]
    native["protocol"].update({"score_threshold_strict_gt": .001,
                                "per_level_topk": 1000, "nms_iou": .6,
                                "detections_per_image": 100,
                                "wrong_class_score_gte": FIXED_SCORE,
                                "role": "primary_native_gate"})
    fixed = {"threshold": FIXED_SCORE, "counts": fixed_counts,
             "diagnostics": filtered_stage_diagnostics(stage_rows, normalized, FIXED_SCORE)}
    fixed["diagnostics"]["wrong_class_fp"] = fixed_counts["wrong_class_fp"]
    workpoint = _choose_workpoint(cache, TARGET_FPPI)
    workpoint["diagnostics"] = filtered_stage_diagnostics(stage_rows, normalized, workpoint["threshold"])
    workpoint["diagnostics"]["wrong_class_fp"] = workpoint["counts"]["wrong_class_fp"]
    return {"selection": selection, "epoch": epoch, "metrics": metrics,
            "diagnostics": native, "fixed_score_025": fixed, "workpoint": workpoint,
            "ap_uses_full_native_prediction_cache": True,
            "workpoint_selected_on_evaluation_images": True,
            "workpoint_scope": "descriptive validation comparison; not deployable threshold calibration"}


class CachedCocoAP75:
    """Official per-image COCO matches with vectorized exact accumulation.

    A draw is a sequence of original image indices.  Each occurrence contributes
    a full independent copy of that image's precomputed matches.  Concatenation
    follows draw order before stable score sorting, exactly as explicit image
    duplication with new consecutive image IDs; equal scores are not regrouped.
    """

    def __init__(self, records, predictions, num_classes):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        records, predicted = _normalize(records, predictions, num_classes)
        if not records:
            raise ValueError("Bootstrap needs at least one evaluation image")
        self.image_count = len(records)
        self.image_ids = [record["image_id"] for record in records]
        self.num_classes = num_classes
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
        categories = [{"id": label, "name": str(label)} for label in range(1, num_classes + 1)]
        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = {"info": {}, "images": images, "annotations": annotations, "categories": categories}
            truth.createIndex()
            if detections:
                detected = truth.loadRes(detections)
            else:
                detected = COCO()
                detected.dataset = {"info": {}, "images": images, "annotations": [], "categories": categories}
                detected.createIndex()
            evaluation = COCOeval(truth, detected, "bbox")
            evaluation.params.imgIds = list(range(1, self.image_count + 1))
            evaluation.params.catIds = list(range(1, num_classes + 1))
            evaluation.params.iouThrs = np.array([.75], dtype=np.float64)
            evaluation.params.areaRng = [[0, 1e10]]
            evaluation.params.areaRngLbl = ["all"]
            evaluation.params.maxDets = [100]
            evaluation.evaluate()
        self.classes = []
        self.recall_grid = evaluation.params.recThrs.copy()
        for category in range(num_classes):
            rows = evaluation.evalImgs[category * self.image_count:(category + 1) * self.image_count]
            scores, true_positive, false_positive = [], [], []
            lengths, ground_truths = [], []
            for row in rows:
                if row is None:
                    lengths.append(0)
                    ground_truths.append(0)
                    continue
                ignored = row["dtIgnore"][0].astype(bool)
                matched = row["dtMatches"][0].astype(bool)
                scores.extend(row["dtScores"])
                true_positive.extend(matched & ~ignored)
                false_positive.extend(~matched & ~ignored)
                lengths.append(len(row["dtScores"]))
                ground_truths.append(int(np.count_nonzero(np.asarray(row["gtIgnore"]) == 0)))
            lengths = np.asarray(lengths, dtype=np.int64)
            self.classes.append({"scores": np.asarray(scores, dtype=np.float64),
                                 "tp": np.asarray(true_positive, dtype=bool),
                                 "fp": np.asarray(false_positive, dtype=bool),
                                 "lengths": lengths, "offsets": np.cumsum(lengths) - lengths,
                                 "gt": np.asarray(ground_truths, dtype=np.int64)})

    def evaluate(self, image_indices=None):
        indices = (np.arange(self.image_count, dtype=np.int64) if image_indices is None
                   else np.asarray(image_indices))
        if (indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.image_count)):
            raise ValueError("Resampled indices must be a nonempty 1D integer sequence in image range")
        values = []
        for category in self.classes:
            count = int(category["gt"][indices].sum())
            if not count:
                continue  # Official COCO excludes absent categories from mean AP.
            lengths = category["lengths"][indices]
            total = int(lengths.sum())
            if not total:
                values.append(0.)
                continue
            starts = np.cumsum(lengths) - lengths
            offsets = np.repeat(category["offsets"][indices] - starts, lengths)
            selected = offsets + np.arange(total, dtype=np.int64)
            order = np.argsort(-category["scores"][selected], kind="mergesort")
            selected = selected[order]
            tp = np.cumsum(category["tp"][selected], dtype=np.float64)
            fp = np.cumsum(category["fp"][selected], dtype=np.float64)
            recall = tp / count
            precision = tp / (fp + tp + np.spacing(1))
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            locations = np.searchsorted(recall, self.recall_grid, side="left")
            interpolated = np.zeros(len(self.recall_grid), dtype=np.float64)
            valid = locations < total
            interpolated[valid] = envelope[locations[valid]]
            values.append(float(interpolated.mean()))
        return float(np.mean(values) * 100) if values else float("nan")


def _prediction_views(value):
    # A single prediction cache is a list of image dictionaries; a stability
    # view is a list of complete per-epoch caches, not a prediction ensemble.
    if not isinstance(value, (list, tuple)):
        raise ValueError("Run predictions must be a cache list or a list of per-epoch cache lists")
    if not value or isinstance(value[0], dict):
        return [value]
    if not all(isinstance(item, (list, tuple)) for item in value):
        raise ValueError("Malformed per-epoch prediction views")
    return list(value)


def bootstrap_suite(records, predictions_by_run, num_classes, *,
                    n_resamples=BOOTSTRAP_REPETITIONS, random_seed=BOOTSTRAP_SEED):
    """Paired three-seed AP75 difference, with one image draw shared by all runs.

    Keys are (variant, seed).  Values are one complete prediction cache, or
    equally many complete per-epoch caches per run.  Epoch metrics are averaged
    after separate COCO evaluation, then the three paired seed differences are
    averaged.  No seed or checkpoint is selected from the bootstrap results.
    """
    expected = {(variant, seed) for seed in SEEDS for variant in VARIANTS}
    if set(predictions_by_run) != expected:
        raise ValueError("Paired bootstrap requires exactly AAA/QFL_DFL x seeds 42,43,44")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not records:
        raise ValueError("Bootstrap needs nonempty records")
    views = {key: _prediction_views(value) for key, value in predictions_by_run.items()}
    view_counts = {len(value) for value in views.values()}
    if len(view_counts) != 1 or next(iter(view_counts)) not in (1, 5):
        raise ValueError("All six runs need one selected checkpoint or five retained stability epochs")
    prepared = {key: [CachedCocoAP75(records, predictions, num_classes) for predictions in value]
                for key, value in views.items()}

    def per_seed_differences(indices):
        values = []
        for seed in SEEDS:
            control = float(np.mean([cache.evaluate(indices) for cache in prepared["AAA", seed]]))
            quality = float(np.mean([cache.evaluate(indices) for cache in prepared["QFL_DFL", seed]]))
            values.append(quality - control)
        return np.asarray(values, dtype=np.float64)

    point = per_seed_differences(None)
    if not np.isfinite(point).all():
        raise ValueError("AP75 is undefined on the complete evaluation set")
    rng = np.random.Generator(np.random.PCG64(random_seed))
    differences = np.empty((n_resamples, len(SEEDS)), dtype=np.float64)
    draw_digest = hashlib.sha256()
    for repetition in range(n_resamples):
        indices = rng.integers(0, len(records), size=len(records), dtype=np.int64)
        draw_digest.update(indices.astype("<i8", copy=False).tobytes())
        differences[repetition] = per_seed_differences(indices)
        if not np.isfinite(differences[repetition]).all():
            raise ValueError("An image resample has undefined AP75; no draws may be silently omitted")
    mean_distribution = differences.mean(axis=1)
    lower, upper = np.quantile(mean_distribution, [.025, .975], method="linear")
    return {"status": "COMPLETE", "metric": "AP75", "units": "percentage_points_0_to_100",
            "seeds": list(SEEDS), "images": len(records), "n_resamples": n_resamples,
            "random_generator": "NumPy PCG64", "random_seed": int(random_seed),
            "quantile_method": "linear", "confidence_level": .95,
            "mean_paired_ap75_gain_pp": float(point.mean()),
            "paired_seed_differences_pp": {str(seed): float(value) for seed, value in zip(SEEDS, point)},
            "seed_signs": {str(seed): "positive" if value > 0 else "negative" if value < 0 else "zero"
                           for seed, value in zip(SEEDS, point)},
            "seed_sign_counts": {"positive": int(np.count_nonzero(point > 0)),
                                 "negative": int(np.count_nonzero(point < 0)),
                                 "zero": int(np.count_nonzero(point == 0))},
            "ci95_low_pp": float(lower), "ci95_high_pp": float(upper),
            "ci95_excludes_zero": bool(lower > 0 or upper < 0),
            "bootstrap_mean_differences_pp": mean_distribution.tolist(),
            "bootstrap_paired_seed_differences_pp": differences.tolist(),
            "resampled_indices_sha256": draw_digest.hexdigest(),
            "resampled_indices_serialization": "successive draws, little-endian signed int64, C order",
            "checkpoint_views_per_run": next(iter(view_counts)),
            "checkpoint_aggregation": "arithmetic mean of separately recomputed per-epoch AP75",
            "resampling": "one common image-index draw for all six runs and all retained epochs",
            "duplicate_images": "independent copies; fresh image IDs in draw order; stable score ties",
            "matching": "cached official pycocotools COCOeval per-image AP75 matches",
            "ci_scope": "conditional image-sampling uncertainty for fixed models and development validation",
            "resampling_assumption": "images treated as exchangeable; acquisition-sequence grouping is unverified",
            "excludes_training_seed_variation": True,
            "does_not_correct_validation_checkpoint_selection_bias": True,
            "ci_is_additional_acceptance_gate": False}
