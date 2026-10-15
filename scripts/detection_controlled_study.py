"""Validation-only, seeded detection controls and per-image error diagnostics.

This entry point never constructs a test dataset. Existing training/metric
implementations and the classification branch remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hwnas_fpga.data.detection import create_yolo_detection_dataloader
from hwnas_fpga.models import build_detection_model
from hwnas_fpga.runtime import load_config, build_search_space, build_constraints
from hwnas_fpga.training.detection_metrics import box_iou_xyxy, evaluate_detection_predictions
from hwnas_fpga.training.detection_trainer import (
    DetectionLossConfig, DenseAnchorDetectionLoss, decode_detection_predictions,
    train_detection_model,
)
from hwnas_fpga.training.trainer import create_optimizer


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some CUDA kernels lack a deterministic implementation; record this limit.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_model(config):
    dc, det = config["dataset"], config["detection"]
    space = build_search_space(config, image_size=dc["image_size"],
                               input_channels=dc["input_channels"], num_classes=dc["num_classes"],
                               constraints=build_constraints(config))
    arch = space.baseline_architecture()
    model = build_detection_model(arch, num_classes=dc["num_classes"],
        feature_indices=tuple(det["feature_stage_indices"]),
        neck_channels=det["fixed_neck"]["out_channels"],
        anchors_per_location=det["fixed_head"]["anchors_per_location"])
    return model, asdict(arch)


def make_loader(config, data_dir, split, seed, workers, train=False):
    if split not in {"train", "valid"}:
        raise ValueError("This study only permits train/valid; test is forbidden")
    dc, tc = config["dataset"], config["training"]
    loader = create_yolo_detection_dataloader(data_dir, split=split,
        image_size=dc["image_size"], input_channels=dc["input_channels"],
        batch_size=tc["batch_size"], num_workers=workers, shuffle=train, seed=seed,
        augment_config=tc.get("detection_augmentation", {}) if train else None,
        class_balanced_sampler=train and tc.get("detection_sampling", {}).get("class_balanced", False))
    if train and hasattr(loader.sampler, "generator"):
        loader.sampler.generator = torch.Generator().manual_seed(seed)
    return loader


def dataset_binding(loader):
    dataset = loader.dataset
    h = hashlib.sha256()
    for path in dataset.samples:
        label = dataset.labels_dir / (path.stem + ".txt")
        h.update(path.name.encode())
        h.update(sha256(path).encode())
        h.update(label.read_bytes() if label.exists() else b"MISSING_LABEL")
    return {"images": len(dataset), "sha256_images_and_labels": h.hexdigest(),
            "class_names": dataset.class_names}


def infer_records(model, loader, criterion, device):
    model.eval()
    records = []
    with torch.inference_mode():
        for images, targets in loader:
            output = model(images.to(device))
            predictions = decode_detection_predictions(output, config=criterion.config,
                score_threshold=0.01, nms_iou_threshold=0.5, pre_nms_topk=500, max_detections=50)
            for pred, target in zip(predictions, targets):
                records.append({"path": str(target["path"]),
                    "prediction": {k: pred[k].tolist() for k in ("boxes", "scores", "labels")},
                    "target": {k: target[k].tolist() for k in ("boxes", "labels")}})
    return records


def metric(records, num_classes, threshold=0.0, ious=(0.5, 0.75)):
    return evaluate_detection_predictions([r["prediction"] for r in records],
        [r["target"] for r in records], num_classes=num_classes,
        score_threshold=threshold, iou_thresholds=ious)


def select_threshold(valid_records, num_classes):
    rows = [metric(valid_records, num_classes, t) for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)]
    return rows, max(rows, key=lambda r: (r["f1"], -r["score_threshold"]))


def score_summary(values):
    if not values:
        return {"n": 0, "mean": None, "q10": None, "median": None, "q90": None,
                "histogram_0_to_1_step_01": [0] * 10}
    a = np.asarray(values)
    return {"n": len(values), "mean": float(a.mean()), "q10": float(np.quantile(a, .1)),
            "median": float(np.median(a)), "q90": float(np.quantile(a, .9)),
            "histogram_0_to_1_step_01": np.histogram(a, bins=np.linspace(0, 1, 11))[0].tolist()}


def classify_image(record, threshold):
    """One-to-one score-ordered matching plus disjoint diagnostic FP categories.

    Error taxonomy is descriptive (not TIDE's counterfactual dAP). A GT whose
    best same-class box overlaps below .1 is not assigned a localization error.
    """
    pred, gt = record["prediction"], record["target"]
    pb = torch.tensor(pred["boxes"], dtype=torch.float32).reshape(-1, 4)
    gb = torch.tensor(gt["boxes"], dtype=torch.float32).reshape(-1, 4)
    ious = box_iou_xyxy(pb, gb)
    used = set()
    predictions = []
    for j in sorted(range(len(pred["scores"])), key=lambda j: -pred["scores"][j]):
        if pred["scores"][j] < threshold:
            continue
        same = [i for i, label in enumerate(gt["labels"]) if label == pred["labels"][j]]
        available = [i for i in same if i not in used and float(ious[j, i]) >= .5]
        best_any = float(ious[j].max()) if len(gb) else 0.0
        best_same = max((float(ious[j, i]) for i in same), default=0.0)
        match = None
        if available:
            match = max(available, key=lambda i: float(ious[j, i]))
            used.add(match)
            kind = "true_positive"
        elif best_same >= .5:
            kind = "duplicate"
        elif best_any >= .5:
            kind = "classification"
        elif best_same >= .1:
            kind = "localization"
        elif best_any >= .1:
            kind = "classification_and_localization"
        else:
            kind = "background"
        predictions.append({"pred_index": j, "class_id": pred["labels"][j],
            "score": pred["scores"][j], "kind": kind, "matched_gt": match,
            "best_any_iou": best_any, "best_same_class_iou": best_same,
            "box": pred["boxes"][j]})
    targets = []
    for i, box in enumerate(gt["boxes"]):
        eligible = [j for j, lab in enumerate(pred["labels"])
                    if lab == gt["labels"][i] and pred["scores"][j] >= threshold]
        best = max((float(ious[j, i]) for j in eligible), default=0.0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        side = area ** .5
        size = "sqrt_area_lt16" if side < 16 else "sqrt_area_16_32" if side < 32 else "sqrt_area_ge32"
        found = i in used
        reason = "detected" if found else "localization" if .1 <= best < .5 else "miss_or_wrong_class_or_duplicate_conflict"
        targets.append({"class_id": gt["labels"][i], "size_bin": size, "detected": found,
                        "best_same_class_iou": best, "reason": reason,
                        "has_same_class_iou75": best >= .75})
    return predictions, targets


def background_descriptor(path, box, size=640):
    from PIL import Image
    with Image.open(path) as im:
        array = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32) / 255
    x1, y1, x2, y2 = [max(0, min(size, int(v))) for v in box]
    patch = array[max(0, y1 - 4):min(size, y2 + 4), max(0, x1 - 4):min(size, x2 + 4)]
    std = float(patch.std()) if patch.size else 0.0
    mean = float(patch.mean()) if patch.size else 0.0
    return {"cx_fraction": (x1+x2)/2/size, "cy_fraction": (y1+y2)/2/size,
            "patch_mean": mean, "patch_std": std,
            "appearance_proxy": "dark" if mean < .15 else "low_variation" if std < .06 else "textured",
            "semantic_type": "unlabelled_requires_visual_review"}


def summarize_errors(records, names, threshold, with_background=False):
    counts = {n: Counter() for n in names}
    scores = {n: {} for n in names}
    size_counts = {}
    errors = []
    background = []
    for rec in records:
        preds, targets = classify_image(rec, threshold)
        for p in preds:
            name = names[p["class_id"]]
            counts[name][p["kind"]] += 1
            scores[name].setdefault(p["kind"], []).append(p["score"])
            if p["kind"] != "true_positive":
                errors.append({"path": rec["path"], **p})
            if with_background and p["kind"] == "background":
                background.append({"path": rec["path"], **p,
                    **background_descriptor(rec["path"], p["box"])})
        for t in targets:
            name = names[t["class_id"]]
            counts[name]["targets"] += 1
            counts[name]["missed"] += int(not t["detected"])
            key = name + "/" + t["size_bin"]
            s = size_counts.setdefault(key, Counter())
            s["targets"] += 1
            s["detected_iou50"] += int(t["detected"])
            s["same_class_overlap75"] += int(t["has_same_class_iou75"])
            s[t["reason"]] += 1
    for name, c in counts.items():
        tp = c["true_positive"]
        total_pred = sum(c[k] for k in ("true_positive", "duplicate", "classification", "localization",
                                       "classification_and_localization", "background"))
        c["precision"] = tp / max(1, total_pred)
        c["recall"] = tp / max(1, c["targets"])
    return {"threshold": threshold, "images": len(records), "per_class": counts,
        "confidence": {n: {k: score_summary(v) for k, v in groups.items()} for n, groups in scores.items()},
        "size_breakdown": size_counts, "false_positive_examples": sorted(errors, key=lambda p:-p["score"])[:30],
        "background_appearance": dict(Counter(p["appearance_proxy"] for p in background)),
        "background_records": background,
        "taxonomy_note": "Disjoint descriptive categories; not TIDE dAP. Appearance bins are proxies, not semantic annotations."}


def make_gallery(records, errors, output):
    from PIL import Image, ImageDraw
    paths = list(dict.fromkeys(e["path"] for e in errors))[:12]
    if not paths:
        return
    canvas = Image.new("RGB", (960, 340 * ((len(paths)+2)//3)), "white")
    lookup = {r["path"]: r for r in records}
    for index, path in enumerate(paths):
        with Image.open(path) as im:
            panel = im.convert("RGB").resize((320, 320))
        draw = ImageDraw.Draw(panel)
        for box in lookup[path]["target"]["boxes"]:
            draw.rectangle(tuple(v/2 for v in box), outline="lime", width=2)
        for error in errors:
            if error["path"] == path:
                b = tuple(v/2 for v in error["box"])
                draw.rectangle(b, outline="red", width=1)
                draw.text((b[0], max(0, b[1]-10)), f"{error['class_id']} {error['score']:.2f}", fill="red")
        x,y = (index%3)*320,(index//3)*340
        canvas.paste(panel,(x,y))
        ImageDraw.Draw(canvas).text((x+2,y+321), f"{index+1}: GT green; FP red", fill="black")
    canvas.save(output)
    dump(Path(output).with_suffix(".json"), {"panel_paths": paths})


def diagnose(model, config, data_dir, criterion, device, workers, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    bindings = {}
    for split in ("train", "valid"):
        loader = make_loader(config, data_dir, split, 42, workers)
        records[split] = infer_records(model, loader, criterion, device)
        bindings[split] = dataset_binding(loader)
        with (output / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as f:
            for r in records[split]:
                f.write(json.dumps(r) + "\n")
    names = bindings["valid"]["class_names"]
    sweep, selected = select_threshold(records["valid"], len(names))
    summary = {"schema_version": 1, "test_accessed": False, "data_bindings": bindings,
        "postprocess": {"prefilter_score": .01, "pre_nms_topk": 500, "max_detections": 50, "nms_iou": .5},
        "metric_note": "Legacy map averages .50/.75; AP10 averages .50:.95 with all-points integration, not COCO101.",
        "validation_threshold_sweep": sweep, "selected_validation_threshold": selected,
        "splits": {}}
    for split, rows in records.items():
        summary["splits"][split] = {
            "legacy_ap": metric(rows, len(names)),
            "ap10_allpoints": metric(rows, len(names), ious=tuple(round(.5+i*.05,2) for i in range(10))),
            "low_threshold_errors": summarize_errors(rows, names, .01),
            "operating_errors": summarize_errors(rows, names, selected["score_threshold"], with_background=True)}
    # Select operating points from validation only for the FP/image constraints.
    fppi_rows = []
    for threshold in np.linspace(.01, 1, 100):
        errors = summarize_errors(records["valid"], names, float(threshold))
        c = errors["per_class"]
        fp = sum(sum(v.get(k,0) for k in ("duplicate", "classification", "localization", "classification_and_localization", "background")) for v in c.values())
        tp = sum(v.get("true_positive",0) for v in c.values())
        gt = sum(v.get("targets",0) for v in c.values())
        fppi_rows.append({"threshold":float(threshold), "fp_per_image":fp/len(records["valid"]), "recall":tp/max(1,gt)})
    summary["validation_recall_at_fp_per_image"] = {
        str(limit): max((r for r in fppi_rows if r["fp_per_image"] <= limit), key=lambda r:(r["recall"],-r["fp_per_image"]))
        for limit in (.1, .5, 1.)}
    dump(output / "diagnostics.json", summary)
    make_gallery(records["valid"], summary["splits"]["valid"]["operating_errors"]["false_positive_examples"], output/"valid_false_positives.png")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("diagnose", "train"))
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    config = load_config(args.config)
    config["project"]["seed"] = args.seed
    model, arch = make_model(config)
    criterion = DenseAnchorDetectionLoss(DetectionLossConfig.from_config(config)).to(args.device)
    manifest = {"started_at":datetime.now().astimezone().isoformat(), "args":vars(args),
        "config_sha256":sha256(args.config), "architecture":arch,
        "initial_state_sha256":state_hash(model), "torch_version":torch.__version__,
        "python_version":sys.version, "seed_controls":["python", "numpy", "torch", "cuda", "worker", "sampler"],
        "determinism":"cudnn deterministic; torch deterministic algorithms warn_only; not bitwise guarantee",
        "test_accessed":False, "selection_metric":"validation legacy map(.50,.75)",
        "code_sha256":{str(path.relative_to(ROOT)):sha256(path) for base in (ROOT/"src", ROOT/"scripts")
                        for path in sorted(base.rglob("*.py")) if "__pycache__" not in str(path)}}
    dump(output/"run_manifest.json", manifest)
    dump(output/"resolved_config.json", config)
    if args.mode == "train":
        train_loader = make_loader(config, args.data_dir, "train", args.seed, args.workers, train=True)
        valid_loader = make_loader(config, args.data_dir, "valid", args.seed, args.workers)
        manifest["data_bindings"] = {"train":dataset_binding(train_loader), "valid":dataset_binding(valid_loader)}
        dump(output/"run_manifest.json", manifest)
        optimizer = create_optimizer(model, optimizer_name=config["training"]["optimizer"],
            lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
        history = train_detection_model(model, train_loader, valid_loader, criterion=criterion,
            optimizer=optimizer, device=args.device, epochs=args.epochs, score_threshold=.01,
            nms_iou_threshold=.5, pre_nms_topk=500, max_detections=50,
            checkpoint_path=output/"checkpoints/best_model.pt", epoch_log_path=output/"epoch_metrics.jsonl")
        dump(output/"history.json", history)
        checkpoint = output/"checkpoints/best_model.pt"
        # Only validation inference is needed for the repeated comparison.
        rows = infer_records(model, valid_loader, criterion, args.device)
        sweep, selected = select_threshold(rows, config["dataset"]["num_classes"])
        with (output/"valid_predictions.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
        dump(output/"status.json", {"status":"COMPLETE", "test_accessed":False,
            "seed":args.seed, "epochs":len(history["epochs"]), "best_epoch":history["best_epoch"],
            "best_validation":history["best_eval"], "validation_operating_point":selected,
            "validation_threshold_sweep":sweep, "checkpoint_sha256":sha256(checkpoint),
            "completed_at":datetime.now().astimezone().isoformat()})
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for diagnose")
        cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(cp.get("model_state_dict",cp))
        model.to(args.device)
        diagnose(model, config, args.data_dir, criterion, args.device, args.workers, output/"diagnostics")
        dump(output/"status.json", {"status":"COMPLETE", "test_accessed":False,
            "checkpoint_sha256":sha256(args.checkpoint), "completed_at":datetime.now().astimezone().isoformat()})
    print("COMPLETE", str(output), flush=True)


if __name__ == "__main__":
    main()
