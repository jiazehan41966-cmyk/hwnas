"""Read-only frozen-D assignment audit using existing train/valid target records.

No checkpoint, image inference, optimizer, augmented data or test data is loaded.
Zero-valued tensors provide output shapes to the original assignment function;
zero-logit gradient figures are explicitly mathematical references, not measured
checkpoint gradients. Frozen source hashes must match the original audit.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hwnas_fpga.data.detection import _build_class_balanced_sampler
from hwnas_fpga.training.detection_trainer import (
    DetectionLossConfig, _assign_targets, _anchor_iou,
    _decode_boxes_for_loss, _weighted_objectness_loss,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csvwrite(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_records(input_dir, split, manifest):
    if split not in ("train", "valid"):
        raise ValueError("Only train and valid diagnostic records are permitted")
    tag = f"{split}_D_seed42_best"
    meta = next(x for x in manifest["artifacts"] if x["tag"] == tag)
    p = input_dir / "predictions" / (tag + ".jsonl.gz")
    if sha(p) != meta["predictions_sha256"]:
        raise ValueError("Diagnostic record binding changed")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    # This dataset has one annotated object per image; class grouping of negative
    # slots below means images bearing that class, not a semantic negative class.
    if any(len(r["target"]["labels"]) != 1 for r in records):
        raise ValueError("One-GT-per-image assumption no longer holds")
    return records, p


def zero_reference(assigned, cfg):
    """Exact autograd at z=0; not actual trained logits, loss, or gradients."""
    y = assigned["objectness"]
    z = torch.zeros_like(y, requires_grad=True)
    loss = _weighted_objectness_loss(z, y, valid_obj_mask=y >= 0,
        positive_weight=cfg.object_weight, negative_weight=cfg.no_object_weight,
        focal_gamma=cfg.object_focal_gamma, focal_alpha=cfg.object_focal_alpha,
        normalization=cfg.objectness_normalization)
    grad, = torch.autograd.grad(loss, z)
    return dict(reference_zero_logit_obj_loss=float(loss.detach()),
        reference_zero_logit_positive_gradient_l1=float(grad[y > .5].abs().sum()),
        reference_zero_logit_negative_gradient_l1=float(grad[y == 0].abs().sum()),
        reference_zero_logit_ignored_gradient_l1=float(grad[y < 0].abs().sum()),
        positive_slots=int((y>.5).sum()),negative_slots=int((y==0).sum()),
        reference_zero_logit_positive_slot_mean_abs_gradient=float(grad[y > .5].abs().mean()) if (y>.5).any() else None,
        reference_zero_logit_negative_slot_mean_abs_gradient=float(grad[y == 0].abs().mean()) if (y==0).any() else None)


def run(input_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/baseline_sonar_frozen.yaml"
    audit_path = ROOT / "reports/sonar_operator_pool_p0_confirmation/BASELINE_COMPARABILITY.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or sha(config_path) != audit["baseline_sha256"]:
        raise ValueError("Baseline config binding failed")
    for rel, expected in audit["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Frozen source drift: {rel}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = DetectionLossConfig.from_config(config)
    infer_path = input_dir / "INFERENCE_MANIFEST.json"
    inference = json.loads(infer_path.read_text(encoding="utf-8"))
    classes = config["dataset"]["class_names"]
    flat = [(level, ai, aw, ah) for level, anchors in enumerate(cfg.anchor_sizes)
            for ai, (aw, ah) in enumerate(anchors)]
    inputs = {str(config_path.relative_to(ROOT)): sha(config_path),
              str(audit_path.relative_to(ROOT)): sha(audit_path),
              str(infer_path.relative_to(ROOT)): sha(infer_path)}
    per_image, gtrows, checks, samprows, refrows = [], [], [], [], []
    refs = defaultdict(list)
    for split, expected_n in (("train", 330), ("valid", 99)):
        records, p = read_records(input_dir, split, inference)
        inputs[str(p.relative_to(ROOT))] = sha(p)
        if len(records) != expected_n:
            raise ValueError("Wrong split cardinality")
        for image_idx, r in enumerate(records):
            target = r["target"]
            box = torch.tensor(target["boxes"], dtype=torch.float32)[0]
            label = int(target["labels"][0]); w, h = (box[2:] - box[:2]).tolist()
            predictions = [torch.zeros(1, 3 * (cfg.num_classes + 5), cfg.image_size // s,
                                       cfg.image_size // s) for s in cfg.strides]
            assigned = _assign_targets(predictions, [target], cfg, device=torch.device("cpu"))
            ious = [_anchor_iou(max(1,w), max(1,h), aw, ah) for _, _, aw, ah in flat]
            bestidx = int(np.argmax(ious)); bestlevel = cfg.strides[flat[bestidx][0]]
            pos_total, roundtrip_err = 0, 0.0
            for li, a in enumerate(assigned):
                obj = a["objectness"]
                pos = int((obj > .5).sum()); neg = int((obj == 0).sum()); ign = int((obj < 0).sum())
                pos_total += pos
                wp = cfg.object_weight * (cfg.object_focal_alpha if cfg.object_focal_gamma > 0 else 1)
                wn = cfg.no_object_weight * (1-cfg.object_focal_alpha if cfg.object_focal_gamma > 0 else 1)
                den = wp*pos + wn*neg
                mean_iou = None
                if pos:
                    ai = torch.nonzero(obj > .5)[:,1]
                    mean_iou = float(np.mean([ious[li*3 + int(j)] for j in ai]))
                    decoded = _decode_boxes_for_loss(a["box"], level_index=li, config=cfg)
                    err = (decoded[obj > .5] - a["box_xyxy"][obj > .5]).abs().max().item()
                    roundtrip_err = max(roundtrip_err, err)
                    if not torch.all(a["label"][obj > .5] == label):
                        raise ValueError("Assigned class mismatch")
                row = dict(split=split, image_index=image_idx, image=r.get("path", r.get("image", "")),
                    class_id=label, class_name=classes[label], level=f"P{int(np.log2(cfg.strides[li]))}",
                    stride=cfg.strides[li], total_slots=obj.numel(), positive=pos, ignored=ign, negative=neg,
                    min_side_cells=min(w,h)/cfg.strides[li],
                    min_side_below_one_cell=min(w,h)/cfg.strides[li]<1,
                    min_side_below_two_cells=min(w,h)/cfg.strides[li]<2,
                    gt_covered=int(pos>0), assigned_anchor_shape_iou_mean=mean_iou,
                    positive_fixed_weight_mass=wp*pos, negative_fixed_weight_mass=wn*neg,
                    fixed_denominator=den, positive_fixed_weight_fraction=wp*pos/den,
                    per_positive_coefficient=wp/den, per_negative_coefficient=wn/den)
                per_image.append(row)
                refs[(split, li)].append(a)
            gtrows.append(dict(split=split,image_index=image_idx,image=r.get("path", r.get("image", "")),
                class_id=label,class_name=classes[label],width=w,height=h,best_anchor_iou=max(ious),
                best_anchor_level=f"P{int(np.log2(bestlevel))}",positive_slots=pos_total,
                fallback_best_anchor_below_positive_threshold=max(ious)<cfg.positive_anchor_iou_threshold,
                best_anchor_below_ignore_threshold=max(ious)<cfg.ignore_anchor_iou_threshold,
                gt_covered=pos_total>0,assigned_box_decode_max_abs_error=roundtrip_err))
            if pos_total < 1 or roundtrip_err > .002:
                raise ValueError("GT coverage or assignment-encode/decode failure")
        if split == "train":
            ds = SimpleNamespace(sample_labels=[r["target"]["labels"] for r in records])
            sampler = _build_class_balanced_sampler(ds)
            counts = Counter(l[0] for l in ds.sample_labels)
            weights = sampler.weights.numpy(); p_image = weights/weights.sum()
            for label in sorted(counts):
                idxs = [i for i,l in enumerate(ds.sample_labels) if l[0]==label]
                samprows.append(dict(class_id=label,class_name=classes[label],unique_train_images=counts[label],
                    raw_image_fraction=counts[label]/len(records),weight_per_image=float(weights[idxs[0]]),
                    expected_class_probability=float(p_image[idxs].sum()),expected_draws_per_epoch=float(p_image[idxs].sum()*len(records)),
                    expected_draws_per_unique_image=float(p_image[idxs[0]]*len(records)),
                    expected_unique_images_seen_per_epoch=float(sum(1-(1-p_image[i])**len(records) for i in idxs)),
                    replacement=sampler.replacement,num_samples=sampler.num_samples,
                    class_loss_weight=cfg.class_loss_weights[label]))
            # CPU reference only: identical API, not claimed to reconstruct
            # historical epoch draws (resume/RNG logs are not loaded here).
            for seed in (42,43,44):
                sampler.generator = torch.Generator().manual_seed(seed)
                selected = list(iter(sampler))
                checks.append(dict(check="sampler_one_epoch_reference",seed=seed,draws=len(selected),
                    unique_images=len(set(selected)),class_counts=dict(Counter(ds.sample_labels[i][0] for i in selected))))
    # Aggregating targets by groups of <=4 recreates loss reduction at deterministic
    # unaugmented loader batching. This is not the historical sampled train batch.
    for (split, li), items in refs.items():
        for start in range(0,len(items),4):
            a = {"objectness": torch.cat([x["objectness"] for x in items[start:start+4]],dim=0)}
            rr=zero_reference(a,cfg)
            refrows.append(dict(split=split,level=f"P{int(np.log2(cfg.strides[li]))}",batch_start_index=start,
                batch_size=len(items[start:start+4]),**rr))
    summary=[]
    for split in ("train","valid"):
        for classname in ("ALL",*classes):
            gs=[r for r in gtrows if r["split"]==split and (classname=="ALL" or r["class_name"]==classname)]
            for level in ("P3","P4","P5"):
                rows=[r for r in per_image if r["split"]==split and r["level"]==level and (classname=="ALL" or r["class_name"]==classname)]
                summary.append(dict(split=split,class_name=classname,level=level,images=len(rows),
                    positives=sum(r["positive"] for r in rows),ignored=sum(r["ignored"] for r in rows),
                    negatives=sum(r["negative"] for r in rows),gt_covered=sum(r["gt_covered"] for r in rows),
                    coverage=sum(r["gt_covered"] for r in rows)/len(rows),
                    positives_per_image=float(np.mean([r["positive"] for r in rows])),
                    positive_fixed_mass_fraction=float(sum(r["positive_fixed_weight_mass"] for r in rows)/sum(r["fixed_denominator"] for r in rows)),
                    positive_images_min_side_lt1=sum(r["gt_covered"] and r["min_side_below_one_cell"] for r in rows),
                    positive_images_min_side_lt1_fraction=(sum(r["gt_covered"] and r["min_side_below_one_cell"] for r in rows)/sum(r["gt_covered"] for r in rows)) if sum(r["gt_covered"] for r in rows) else None,
                    positive_images_min_side_cells_median=float(np.median([r["min_side_cells"] for r in rows if r["gt_covered"]])) if any(r["gt_covered"] for r in rows) else None,
                    best_anchor_iou_median=float(np.median([r["best_anchor_iou"] for r in gs])),
                    fallback_below_065=sum(r["fallback_best_anchor_below_positive_threshold"] for r in gs)))
    csvwrite(out/"assignment_per_image_level.csv",per_image)
    csvwrite(out/"assignment_gt.csv",gtrows)
    csvwrite(out/"assignment_summary.csv",summary)
    csvwrite(out/"sampler_expectation.csv",samprows)
    csvwrite(out/"zero_logit_reference.csv",refrows)
    dump(out/"assignment_checks.json",dict(checks=checks,
        gt_covered=sum(r["gt_covered"] for r in gtrows),gt_total=len(gtrows),
        maximum_assignment_encode_decode_error=max(r["assigned_box_decode_max_abs_error"] for r in gtrows),
        frozen_source_hashes_verified=len(audit["source_sha256"]),
        test_accessed=False,new_training_runs=0))
    dump(out/"ASSIGNMENT_MANIFEST.json",dict(status="COMPLETE",source_sha256=sha(Path(__file__)),
        input_sha256=inputs,loss_config=asdict(cfg),source_binding=audit["source_sha256"],
        record_scope="Existing unaugmented train330/valid99 targets; assignment independent of prediction values",
        reference_warning="zero-logit gradients are mathematical references only, not checkpoint gradients",
        historical_sampler_draws_reconstructed=False,test_accessed=False,new_training_runs=0,
        output_sha256={p.name:sha(p) for p in out.glob("*.csv")}))
    print(json.dumps(dict(status="COMPLETE",images=len(gtrows),positives=sum(r["positive_slots"] for r in gtrows),
        max_assignment_roundtrip_error=max(r["assigned_box_decode_max_abs_error"] for r in gtrows)),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"reports/sonar_operator_pool_p0_diagnosis")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/sonar_operator_pool_p0_followup/assignment")
    args=parser.parse_args();torch.set_num_threads(2);run(args.input,args.output)
