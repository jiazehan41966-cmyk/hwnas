#!/usr/bin/env python3
"""Validation-only error diagnosis for detection checkpoints.

The script never constructs a test loader. It can read existing
``*_predictions.jsonl`` files, so a completed training run can be diagnosed
without reloading weights. Predictions are assumed to be in square network
input coordinates, matching the current loader and decoder.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco
from hwnas_fpga.training.detection_metrics import box_iou_xyxy, evaluate_detection_predictions

FP_KINDS = (
    "duplicate",
    "classification",
    "localization",
    "classification_and_localization",
    "background",
)
MISS_REASONS = ("no_candidate", "score_too_low", "localization")
SIZE_BINS = ((0.0, 16.0, "sqrt_area_lt16"), (16.0, 32.0, "sqrt_area_16_32"), (32.0, float("inf"), "sqrt_area_ge32"))
FP_LIMITS = (0.1, 0.5, 1.0)
PREFILTER = 0.01


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def size_bin(side: float) -> str:
    for low, high, name in SIZE_BINS:
        if low <= side < high:
            return name
    return SIZE_BINS[-1][2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_image_path(raw: str, data_root: Path | None) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    if data_root is not None:
        candidate = data_root / "valid" / "images" / path.name
        if candidate.is_file():
            return candidate
        candidate = data_root / "train" / "images" / path.name
        if candidate.is_file():
            return candidate
    local = ROOT / "data" / "external_redownload_audit_20260909" / "roboflow_cylider2" / "extracted" / "v6_yolov8"
    for split in ("valid", "train"):
        candidate = local / split / "images" / path.name
        if candidate.is_file():
            return candidate
    return path


def classify_record(record: dict[str, Any], threshold: float, *, prefilter: float = PREFILTER) -> dict[str, Any]:
    pred, gt = record["prediction"], record["target"]
    pb = torch.as_tensor(pred["boxes"], dtype=torch.float32).reshape(-1, 4)
    gb = torch.as_tensor(gt["boxes"], dtype=torch.float32).reshape(-1, 4)
    ious = box_iou_xyxy(pb, gb)
    used: set[int] = set()
    predictions = []
    for index in sorted(range(len(pred["scores"])), key=lambda item: -pred["scores"][item]):
        if pred["scores"][index] < threshold:
            continue
        same = [gt_index for gt_index, label in enumerate(gt["labels"]) if label == pred["labels"][index]]
        available = [gt_index for gt_index in same if gt_index not in used and float(ious[index, gt_index]) >= 0.5]
        best_any = float(ious[index].max()) if gb.numel() else 0.0
        best_same = max((float(ious[index, gt_index]) for gt_index in same), default=0.0)
        match = None
        if available:
            match = max(available, key=lambda gt_index: float(ious[index, gt_index]))
            used.add(match)
            kind = "true_positive"
        elif best_same >= 0.5:
            kind = "duplicate"
        elif best_any >= 0.5:
            kind = "classification"
        elif best_same >= 0.1:
            kind = "localization"
        elif best_any >= 0.1:
            kind = "classification_and_localization"
        else:
            kind = "background"
        predictions.append(
            {
                "pred_index": index,
                "class_id": int(pred["labels"][index]),
                "score": float(pred["scores"][index]),
                "kind": kind,
                "matched_gt": match,
                "best_any_iou": best_any,
                "best_same_class_iou": best_same,
                "box": [float(value) for value in pred["boxes"][index]],
            }
        )

    targets = []
    for gt_index, box in enumerate(gt["boxes"]):
        same = [
            pred_index
            for pred_index, label in enumerate(pred["labels"])
            if label == gt["labels"][gt_index]
        ]
        prefilter_same = [pred_index for pred_index in same if pred["scores"][pred_index] >= prefilter]
        operating_same = [pred_index for pred_index in same if pred["scores"][pred_index] >= threshold]
        best_prefilter = max((float(ious[pred_index, gt_index]) for pred_index in prefilter_same), default=0.0)
        best_operating = max((float(ious[pred_index, gt_index]) for pred_index in operating_same), default=0.0)
        side = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) ** 0.5
        detected = gt_index in used
        if detected:
            reason = "detected"
        elif best_prefilter >= 0.5:
            reason = "score_too_low"
        elif best_prefilter >= 0.1:
            reason = "localization"
        else:
            reason = "no_candidate"
        targets.append(
            {
                "gt_index": gt_index,
                "class_id": int(gt["labels"][gt_index]),
                "box": [float(value) for value in box],
                "size_bin": size_bin(side),
                "sqrt_area": side,
                "detected": detected,
                "best_same_class_iou_prefilter": best_prefilter,
                "best_same_class_iou_operating": best_operating,
                "reason": reason,
                "has_same_class_iou75": best_operating >= 0.75,
            }
        )
    return {"predictions": predictions, "targets": targets}


def miss_reason(target: dict[str, Any]) -> str:
    return str(target["reason"])


def score_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "q10": None, "median": None, "q90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
    }


def summarize(records: list[dict[str, Any]], names: list[str], threshold: float) -> dict[str, Any]:
    counts = {name: Counter() for name in names}
    scores: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in names}
    size_counts: dict[str, Counter[str]] = {}
    image_rows = []
    for record in records:
        classified = classify_record(record, threshold)
        image_fp = 0
        for pred in classified["predictions"]:
            name = names[pred["class_id"]]
            counts[name][pred["kind"]] += 1
            scores[name][pred["kind"]].append(pred["score"])
            if pred["kind"] != "true_positive":
                image_fp += 1
        for target in classified["targets"]:
            name = names[target["class_id"]]
            counts[name]["targets"] += 1
            counts[name]["missed"] += int(not target["detected"])
            counts[name][f"miss_{target['reason']}"] += 1
            key = f"{name}/{target['size_bin']}"
            bucket = size_counts.setdefault(key, Counter())
            bucket["targets"] += 1
            bucket["detected_iou50"] += int(target["detected"])
            bucket["same_class_overlap75"] += int(target["has_same_class_iou75"])
            bucket[f"miss_{target['reason']}"] += 1
        image_rows.append(
            {
                "path": record["path"],
                "image": Path(record["path"]).name,
                "predictions": classified["predictions"],
                "targets": classified["targets"],
                "fp_count": image_fp,
            }
        )
    per_class = {}
    for name, counter in counts.items():
        tp = counter["true_positive"]
        fp = sum(counter[kind] for kind in FP_KINDS)
        per_class[name] = {
            **{key: int(counter[key]) for key in list(counter)},
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, counter["targets"]),
        }
    return {
        "threshold": threshold,
        "images": len(records),
        "per_class": per_class,
        "confidence": {name: {kind: score_summary(values) for kind, values in groups.items()} for name, groups in scores.items()},
        "score_lists": {name: {kind: values for kind, values in groups.items()} for name, groups in scores.items()},
        "size_breakdown": {key: dict(value) for key, value in size_counts.items()},
        "images_detail": image_rows,
        "taxonomy_note": (
            "False positives are disjoint descriptive categories, not TIDE dAP. "
            "Misses use the prefilter pool (score>=0.01 after NMS) to split "
            "no_candidate / score_too_low / localization. Appearance proxies are "
            "not sonar-environment labels."
        ),
    }


def recall_at_fppi(records: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    thresholds = [round(value, 4) for value in np.linspace(0.01, 0.99, 99)]
    rows = []
    for threshold in thresholds:
        summary = summarize(records, names, float(threshold))
        row: dict[str, Any] = {"threshold": float(threshold)}
        total_fp = 0
        total_tp = 0
        total_gt = 0
        for name in names:
            item = summary["per_class"][name]
            fp = sum(item.get(kind, 0) for kind in FP_KINDS)
            tp = item.get("true_positive", 0)
            gt = item.get("targets", 0)
            row[name] = {
                "fp_per_image": fp / max(1, len(records)),
                "recall": tp / max(1, gt),
                "tp": tp,
                "fp": fp,
                "targets": gt,
            }
            total_fp += fp
            total_tp += tp
            total_gt += gt
        row["all"] = {
            "fp_per_image": total_fp / max(1, len(records)),
            "recall": total_tp / max(1, total_gt),
            "tp": total_tp,
            "fp": total_fp,
            "targets": total_gt,
        }
        rows.append(row)

    selected: dict[str, Any] = {}
    for scope in ("all", *names):
        selected[scope] = {}
        for limit in FP_LIMITS:
            eligible = [row for row in rows if row[scope]["fp_per_image"] <= limit]
            if not eligible:
                selected[scope][str(limit)] = None
                continue
            best = max(eligible, key=lambda row: (row[scope]["recall"], -row[scope]["fp_per_image"], row["threshold"]))
            selected[scope][str(limit)] = {
                "threshold": best["threshold"],
                "fp_per_image": best[scope]["fp_per_image"],
                "recall": best[scope]["recall"],
                "tp": best[scope]["tp"],
                "fp": best[scope]["fp"],
                "targets": best[scope]["targets"],
            }
    return {"sweep": rows, "selected": selected}


def plot_confidence(score_lists: dict[str, dict[str, list[float]]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ("true_positive", "background", "localization", "classification", "duplicate")
    colors = {
        "true_positive": "#0072B2",
        "background": "#D55E00",
        "localization": "#E69F00",
        "classification": "#009E73",
        "duplicate": "#CC79A7",
    }
    names = list(score_lists)
    fig, axes = plt.subplots(1, max(1, len(names)), figsize=(5.2 * max(1, len(names)), 3.6), squeeze=False)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, name in zip(axes[0], names):
        for kind in kinds:
            values = score_lists.get(name, {}).get(kind, [])
            if not values:
                continue
            axis.hist(values, bins=bins, histtype="step", linewidth=1.6, label=f"{kind} n={len(values)}", color=colors[kind])
        axis.set_title(name)
        axis.set_xlabel("score (objectness x class probability)")
        axis.set_ylabel("count")
        axis.set_xlim(0.0, 1.0)
        axis.legend(frameon=False, fontsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def draw_atlas(records: list[dict[str, Any]], summary: dict[str, Any], names: list[str], output_dir: Path, data_root: Path | None) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {record["path"]: record for record in records}
    entries = []
    number = 0
    for image in summary["images_detail"]:
        errors = [pred for pred in image["predictions"] if pred["kind"] != "true_positive"]
        misses = [target for target in image["targets"] if not target["detected"]]
        if not errors and not misses:
            continue
        number += 1
        image_id = f"E{number:03d}"
        source = resolve_image_path(image["path"], data_root)
        with Image.open(source) as raw:
            panel = raw.convert("RGB").resize((640, 640))
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        record = lookup[image["path"]]
        for box in record["target"]["boxes"]:
            draw.rectangle(tuple(box), outline=(0, 200, 0), width=2)
        for pred in errors:
            color = {
                "background": (220, 50, 47),
                "duplicate": (200, 120, 0),
                "classification": (0, 90, 180),
                "localization": (230, 160, 0),
                "classification_and_localization": (140, 0, 160),
            }.get(pred["kind"], (220, 50, 47))
            draw.rectangle(tuple(pred["box"]), outline=color, width=2)
            draw.text((pred["box"][0], max(0.0, pred["box"][1] - 10)), f"{names[pred['class_id']]} {pred['score']:.2f} {pred['kind']}", fill=color, font=font)
        for miss in misses:
            draw.rectangle(tuple(miss["box"]), outline=(255, 255, 0), width=1)
            draw.text((miss["box"][0], min(628.0, miss["box"][3] + 1)), f"MISS {names[miss['class_id']]} {miss['reason']}", fill=(255, 220, 0), font=font)
        filename = f"{image_id}_{source.stem}.png"
        panel.save(output_dir / filename)
        entries.append(
            {
                "id": image_id,
                "file": filename,
                "source_image": source.name,
                "source_path": str(source),
                "fp_kinds": dict(Counter(pred["kind"] for pred in errors)),
                "miss_reasons": dict(Counter(miss["reason"] for miss in misses)),
                "fp_count": len(errors),
                "miss_count": len(misses),
                "predictions": errors,
                "misses": misses,
                "human_review": {
                    "status": "pending_visual_review",
                    "background_type": None,
                    "cannot_judge": None,
                    "note": "Automatic texture bins are not sonar-environment labels.",
                },
            }
        )
    dump(output_dir / "index.json", {"count": len(entries), "entries": entries})
    lines = [
        "# Numbered validation error atlas",
        "",
        "Green = ground truth. Colored boxes = operating-threshold false positives. Yellow text = misses.",
        "Background-type fields are filled only after visual review; patch std/mean bins are not used as labels.",
        "",
        "| ID | Image | FP | Miss | FP kinds | Miss reasons |",
        "|---|---|---:|---:|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['id']} | `{entry['source_image']}` | {entry['fp_count']} | {entry['miss_count']} | "
            f"{dict(entry['fp_kinds'])} | {dict(entry['miss_reasons'])} |"
        )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"count": len(entries), "entries": entries}


def markdown_tables(summary: dict[str, Any], fppi: dict[str, Any], coco: dict[str, Any], names: list[str]) -> str:
    lines = [
        "## Per-class operating-point errors",
        "",
        "| Class | GT | TP | Miss | background FP | class FP | duplicate FP | loc FP | no_candidate | score_too_low | loc miss | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        item = summary["per_class"][name]
        lines.append(
            f"| {name} | {item.get('targets', 0)} | {item.get('true_positive', 0)} | {item.get('missed', 0)} | "
            f"{item.get('background', 0)} | {item.get('classification', 0)} | {item.get('duplicate', 0)} | "
            f"{item.get('localization', 0)} | {item.get('miss_no_candidate', 0)} | {item.get('miss_score_too_low', 0)} | "
            f"{item.get('miss_localization', 0)} | {item['precision']:.4f} | {item['recall']:.4f} |"
        )
    lines += [
        "",
        "## Size-bin miss split",
        "",
        "| Group | GT | detected@0.5 | no_candidate | score_too_low | localization | IoU>=0.75 same-class |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(summary["size_breakdown"].items()):
        lines.append(
            f"| {key} | {item.get('targets', 0)} | {item.get('detected_iou50', 0)} | "
            f"{item.get('miss_no_candidate', 0)} | {item.get('miss_score_too_low', 0)} | "
            f"{item.get('miss_localization', 0)} | {item.get('same_class_overlap75', 0)} |"
        )
    lines += ["", "## Recall at bounded FP/image (validation only)", "", "| Scope | FP/image cap | threshold | observed FP/image | Recall | TP/GT |", "|---|---:|---:|---:|---:|---|"]
    for scope, limits in fppi["selected"].items():
        for limit, row in limits.items():
            if row is None:
                lines.append(f"| {scope} | {limit} | — | — | — | — |")
                continue
            lines.append(
                f"| {scope} | {limit} | {row['threshold']:.2f} | {row['fp_per_image']:.4f} | {row['recall']:.4f} | "
                f"{row['tp']}/{row['targets']} |"
            )
    lines += [
        "",
        "## Standard COCO supplement (not used for checkpoint selection)",
        "",
        f"- coco_ap = {coco['coco_ap']:.6f}",
        f"- coco_ap50 = {coco['coco_ap50']:.6f}",
        f"- coco_ap75 = {coco['coco_ap75']:.6f}",
        f"- coco_ap_small = {coco['coco_ap_small']:.6f}",
        f"- coco_ap_medium = {coco['coco_ap_medium']:.6f}",
        f"- coco_ap_large = {coco['coco_ap_large']:.6f}",
    ]
    return "\n".join(lines) + "\n"


def bottleneck(summary: dict[str, Any], fppi: dict[str, Any], names: list[str]) -> dict[str, Any]:
    misses = {reason: 0 for reason in MISS_REASONS}
    fps = {kind: 0 for kind in FP_KINDS}
    detected = 0
    targets = 0
    for name in names:
        item = summary["per_class"][name]
        detected += item.get("true_positive", 0)
        targets += item.get("targets", 0)
        for reason in MISS_REASONS:
            misses[reason] += item.get(f"miss_{reason}", 0)
        for kind in FP_KINDS:
            fps[kind] += item.get(kind, 0)
    manta = fppi["selected"].get("manta", {})
    evidence = {
        "misses": misses,
        "false_positives": fps,
        "detected": detected,
        "targets": targets,
        "manta_recall_at_fp": manta,
    }
    dominant_miss = max(misses, key=lambda key: misses[key]) if any(misses.values()) else "none"
    dominant_fp = max(fps, key=lambda key: fps[key]) if any(fps.values()) else "none"
    if misses["no_candidate"] >= misses["score_too_low"] and misses["no_candidate"] >= misses["localization"] and fps["background"] >= sum(fps[k] for k in fps if k != "background"):
        verdict = "background_interference_and_missing_candidates"
        sentence = (
            "Remaining errors are dominated by missing candidates plus background false positives, "
            "not by fine box refinement at IoU 0.50-0.75."
        )
    elif misses["score_too_low"] >= misses["localization"] and misses["score_too_low"] >= misses["no_candidate"]:
        verdict = "confidence_discrimination"
        sentence = (
            "Most remaining ground-truth objects already have a same-class candidate at IoU>=0.50 "
            "below the operating threshold, so the bottleneck is score separation rather than localization."
        )
    elif misses["localization"] > misses["score_too_low"] and misses["localization"] > misses["no_candidate"]:
        verdict = "fine_localization"
        sentence = (
            "Most remaining misses have a same-class box with 0.10<=IoU<0.50, so the bottleneck is "
            "box placement rather than candidate generation or ranking."
        )
    elif dominant_fp == "background":
        verdict = "background_interference"
        sentence = (
            "False positives are mostly unmatched background boxes. Misses are mixed; ranking against "
            "background remains the main operating-point constraint."
        )
    else:
        verdict = "mixed"
        sentence = (
            f"No single bin dominates: leading miss={dominant_miss}, leading FP={dominant_fp}."
        )
    evidence["verdict"] = verdict
    evidence["sentence"] = sentence
    return evidence


def diagnose_records(
    records: list[dict[str, Any]],
    *,
    names: list[str],
    output_dir: Path,
    group: str,
    threshold: float,
    data_root: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    project = evaluate_detection_predictions(
        [row["prediction"] for row in records],
        [row["target"] for row in records],
        num_classes=len(names),
        score_threshold=0.0,
    )
    coco = evaluate_detection_predictions_coco(
        [row["prediction"] for row in records],
        [row["target"] for row in records],
        num_classes=len(names),
        class_names=names,
    )
    operating = summarize(records, names, threshold)
    fppi = recall_at_fppi(records, names)
    plot_confidence(operating["score_lists"], output_dir / "confidence_distribution.png")
    atlas = draw_atlas(records, operating, names, output_dir / "error_atlas", data_root)
    evidence = bottleneck(operating, fppi, names)
    payload = {
        "artifact_type": "detection_error_diagnosis",
        "generated_at": datetime.now().astimezone().isoformat(),
        "group": group,
        "split": "valid",
        "test_accessed": False,
        "class_names": names,
        "operating_threshold": threshold,
        "prefilter_score": PREFILTER,
        "project_metric": project,
        "coco_supplement": coco,
        "operating_errors": {key: value for key, value in operating.items() if key not in {"images_detail", "score_lists"}},
        "validation_recall_at_fp_per_image": fppi["selected"],
        "bottleneck": evidence,
        "atlas_count": atlas["count"],
        "map_definition_note": project["map_definition"],
    }
    dump(output_dir / "diagnosis.json", payload)
    dump(output_dir / "per_class_errors.json", operating["per_class"])
    dump(output_dir / "size_breakdown.json", operating["size_breakdown"])
    dump(output_dir / "recall_at_fppi.json", fppi["selected"])
    tables = markdown_tables(operating, fppi, coco, names)
    (output_dir / "per_class_errors.md").write_text(
        f"# {group} validation error tables\n\nOperating threshold = {threshold:.2f}\n\n" + tables,
        encoding="utf-8",
    )
    (output_dir / "size_breakdown.md").write_text(
        "# Size-bin miss split\n\n" + "\n".join(tables.split("## Size-bin miss split", 1)[-1].split("## Recall")[0].strip().splitlines()) + "\n",
        encoding="utf-8",
    )
    return payload


def discover_groups(study_root: Path) -> list[dict[str, Any]]:
    groups = []
    runs = study_root / "runs"
    if runs.is_dir():
        for folder in sorted(runs.iterdir()):
            pred = folder / "valid_predictions.jsonl"
            status = folder / "status.json"
            if pred.is_file() and status.is_file():
                record = json.loads(status.read_text(encoding="utf-8-sig"))
                groups.append(
                    {
                        "name": folder.name,
                        "predictions": pred,
                        "threshold": float(record.get("validation_operating_point", {}).get("score_threshold", 0.7)),
                        "status": record,
                        "kind": "new_checkpoint",
                    }
                )
    legacy_root = ROOT / "results" / "detection_controlled_20260912"
    for name in ("legacy_baseline_diagnostics", "legacy_noobj02_diagnostics"):
        pred = legacy_root / name / "diagnostics" / "valid_predictions.jsonl"
        diag = legacy_root / name / "diagnostics" / "diagnostics.json"
        if pred.is_file():
            threshold = 0.7
            if diag.is_file():
                payload = json.loads(diag.read_text(encoding="utf-8-sig"))
                threshold = float(payload.get("selected_validation_threshold", {}).get("score_threshold", 0.7))
            groups.append(
                {
                    "name": name.replace("_diagnostics", ""),
                    "predictions": pred,
                    "threshold": threshold,
                    "status": {"status": "HISTORICAL", "legacy": True},
                    "kind": "legacy_checkpoint",
                }
            )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", default=str(ROOT / "results" / "detection_fixed_normalizer_20260912"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "detection_eval_diagnosis_20260913"))
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--class-names", default="cylider,manta")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--include-legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    data_root = Path(args.data_dir) if args.data_dir else None
    groups = []
    if args.predictions_jsonl:
        groups.append(
            {
                "name": args.group or Path(args.predictions_jsonl).parent.name,
                "predictions": Path(args.predictions_jsonl),
                "threshold": float(args.threshold if args.threshold is not None else 0.7),
                "status": {},
                "kind": "explicit",
            }
        )
    else:
        groups = discover_groups(Path(args.study_root))
        if not args.include_legacy:
            groups = [group for group in groups if group["kind"] != "legacy_checkpoint"]
    if not groups:
        raise SystemExit("No completed validation prediction files found. Pass --include-legacy or --predictions-jsonl.")

    catalog = []
    for group in groups:
        dest = output / "groups" / group["name"]
        records = load_jsonl(Path(group["predictions"]))
        payload = diagnose_records(
            records,
            names=names,
            output_dir=dest,
            group=group["name"],
            threshold=float(group["threshold"]),
            data_root=data_root,
        )
        catalog.append(
            {
                "name": group["name"],
                "kind": group["kind"],
                "n_images": len(records),
                "threshold": group["threshold"],
                "project_map": payload["project_metric"]["map"],
                "project_map50": payload["project_metric"]["map50"],
                "project_map75": payload["project_metric"]["map75"],
                "coco_ap": payload["coco_supplement"]["coco_ap"],
                "bottleneck": payload["bottleneck"]["verdict"],
                "output": str(dest),
            }
        )
        print(f"diagnosed {group['name']} map={payload['project_metric']['map']:.4f} verdict={payload['bottleneck']['verdict']}", flush=True)
    dump(output / "group_catalog.json", {"groups": catalog, "test_accessed": False})


if __name__ == "__main__":
    main()
