"""Reviewable public-split exploratory draft; never approves its own protocol.

The original sequence-based plan stays blocked. This module groups only exact
image content, excludes entire invalidly annotated images in a derived list,
and labels every output as pending a user's explicit protocol choice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import _write_json

PROTOCOL_NAME = "PUBLIC_SPLIT_EXPLORATORY_V1"
CLASS_NAMES_ZH = {
    "ball": "球", "circle cage": "圆形笼", "cube": "立方体", "cylinder": "圆柱体",
    "human body": "人体", "metal bucket": "金属桶", "plane": "飞机形目标",
    "rov": "遥控水下机器人", "square cage": "方形笼", "tyre": "轮胎",
}


def _balanced_exact_split(samples, class_count, seed, val_fraction):
    """Greedily fit image, object-class, and class-presence counts to 20%.

    Only active development labels are used. The objective and seed are fixed;
    official test labels do not participate in assigning the development split.
    """
    by_hash = defaultdict(list)
    for sample in samples:
        by_hash[sample["sha256"]].append(sample)
    keys = sorted(by_hash)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        return {"status": "INSUFFICIENT_EXACT_CONTENT_GROUPS", "group_count": len(keys)}
    features = np.zeros((len(keys), 1 + 2*class_count), dtype=np.float64)
    for i, key in enumerate(keys):
        members = by_hash[key]
        features[i, 0] = len(members)
        for sample in members:
            for label, count in Counter(sample["labels"]).items():
                features[i, label] += count
                features[i, class_count + label] += 1
    total = features.sum(axis=0)
    targets = total * val_fraction
    target_images = max(1, min(len(samples)-1, round(len(samples)*val_fraction)))
    targets[0] = target_images
    denominators = np.maximum(targets, 1.)
    active_box_classes = max(1, np.count_nonzero(total[1:class_count+1]))
    active_presence_classes = max(1, np.count_nonzero(total[class_count+1:]))
    weights = 1. / denominators**2
    weights[0] *= 2.
    weights[1:class_count+1] /= active_box_classes
    weights[class_count+1:] /= active_presence_classes
    costs = np.sum(features**2 * weights, axis=1)
    current = np.zeros(features.shape[1], dtype=np.float64)
    available = np.ones(len(keys), dtype=bool)
    validation_groups = []
    while current[0] < target_images and len(validation_groups) < len(keys)-1:
        eligible = available.copy()
        remaining = target_images-current[0]
        fitting = eligible & (features[:, 0] <= remaining)
        if fitting.any():
            eligible = fitting
        changes = np.sum(features * (2*(current-targets)*weights), axis=1) + costs
        changes[~eligible] = np.inf
        selected = int(np.argmin(changes))
        validation_groups.append(selected)
        available[selected] = False
        current += features[selected]
    chosen = set(validation_groups)
    for i, key in enumerate(keys):
        for sample in by_hash[key]:
            sample["split"] = "val" if i in chosen else "train"
            sample["group_id"] = f"exact_image_sha256:{key}"
    return {
        "status": "DRAFT_SPLIT_CREATED", "group_count": len(keys),
        "validation_group_count": len(chosen), "target_validation_images": target_images,
        "actual_validation_images": int(current[0]), "validation_fraction_observed": float(current[0]/len(samples)),
        "algorithm": "Seeded greedy fitting of image count, per-class object counts, and per-class image-presence counts; exact-image SHA256 groups kept intact",
        "objective": "Normalized squared target error: image count weight 2; mean over active object classes weight 1; mean over active presence classes weight 1",
        "uses_official_test_labels": False,
    }


def prepare_exploratory_draft(strict_manifest_path, output_dir, seed=42, val_fraction=.2):
    """Build a pending alternative from the strict manifest without raw edits."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    source_path = Path(strict_manifest_path).resolve()
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    original = json.loads(source_bytes.decode("utf-8"))
    if original.get("schema_version") != "sonar-fcos-pilot-data-v1":
        raise ValueError("Expected the original strict sonar-fcos-pilot-data-v1 manifest")
    if original.get("protocol") or original.get("provenance_status") == "IMAGE_SIMILARITY_PROXY_ONLY":
        raise ValueError("The draft must derive from the strict source manifest, not a previous alternative split")
    manifest = copy.deepcopy(original)
    output = Path(output_dir).resolve()
    if (output / "manifest.json").resolve() == source_path:
        raise ValueError("The draft output cannot overwrite the original strict source manifest")
    output.mkdir(parents=True, exist_ok=True)
    issues_by_image = defaultdict(list)
    for issue in original.get("audit", {}).get("annotation_errors", []):
        issues_by_image[issue.get("annotation_path")].append(issue)
    excluded = []
    development = []
    for sample in manifest["samples"]:
        reasons = []
        if not sample.get("annotation_valid", False):
            reasons.append("INVALID_ORIGINAL_ANNOTATION_WHOLE_IMAGE")
        if sample.get("excluded"):
            if sample["source_split"] != "official_train":
                raise ValueError("Unexpected prior exclusion of an official test image")
            reasons.append("EXACT_DUPLICATE_OF_OFFICIAL_TEST")
        if reasons:
            sample["excluded"] = True
            sample["excluded_reason"] = reasons
            sample["split"] = "excluded"
            sample["group_id"] = "excluded_from_exploratory_scoring"
            excluded.append({
                "image_id": sample["image_id"], "sample_key": sample["sample_key"],
                "source_split": sample["source_split"], "image_path": sample["image_path"],
                "annotation_path": sample["annotation_path"], "sha256": sample["sha256"],
                "annotation_sha256": sample["annotation_sha256"], "excluded_reason": reasons,
                "annotation_issues": issues_by_image.get(sample["annotation_path"], []),
            })
        elif sample["source_split"] == "official_train":
            sample["split"] = "unassigned"
            sample["group_id"] = "exact_group_assignment_pending"
            development.append(sample)
        else:
            sample["split"] = "test"
            sample["group_id"] = f"official_membership:{sample['source_split']}"
    grouping = _balanced_exact_split(development, len(manifest["classes"]), seed, val_fraction)
    train_areas = [
        (box[2]-box[0])*(box[3]-box[1])/(sample["width"]*sample["height"])
        for sample in development if sample["split"] == "train" for box in sample["boxes"]
    ]
    threshold = float(np.quantile(train_areas, .25)) if train_areas else None
    counts = dict(Counter(sample["split"] for sample in manifest["samples"]))
    class_counts = {}
    for split in ("train", "val", "test"):
        labels = Counter(label for sample in manifest["samples"] if sample["split"] == split for label in sample["labels"])
        class_counts[split] = {name: labels[class_id] for name, class_id in manifest["class_to_id"].items()}
    excluded_source_counts = dict(Counter(item["source_split"] for item in excluded))
    excluded_reason_counts = dict(Counter(reason for item in excluded for reason in item["excluded_reason"]))
    protocol = {
        "name": PROTOCOL_NAME, "status": "PENDING_USER_CHOICE",
        "acquisition_groups_verified": False, "no_new_scene_generalization_claim": True,
        "permitted_claim_after_approval": "Controlled architecture comparison under this documented public-data exploratory split only",
        "training_authorized": False, "source_manifest_sha256": source_sha256,
        "official_test_membership_policy": "Preserve original official test membership; all explicitly listed invalidly annotated test images excluded from scoring",
        "exclusion_policy": "Exclude entire invalidly annotated images and inherited exact development duplicates of test; never repair or delete original targets",
        "development_grouping": "Exact image SHA256 only; not source acquisitions or appearance-similarity components",
    }
    active = [sample for sample in manifest["samples"] if sample["split"] in {"train", "val", "test"}]
    cross_hash = defaultdict(set)
    for sample in active:
        cross_hash[sample["sha256"]].add(sample["split"])
    leaking_hashes = [key for key, splits in cross_hash.items() if len(splits) > 1]
    integrity = all(sample["annotation_valid"] for sample in active) and not leaking_hashes and grouping["status"] == "DRAFT_SPLIT_CREATED"
    manifest.update({
        "schema_version": "sonar-fcos-pilot-data-v1", "split_status": "PENDING_PROTOCOL_APPROVAL",
        "protocol": protocol, "source_manifest_path": str(source_path), "source_manifest_sha256": source_sha256,
        "seed": seed, "val_fraction_requested": val_fraction, "split_counts": counts,
        "small_area_threshold": threshold, "active_class_box_counts": class_counts,
        "excluded_source_counts": excluded_source_counts, "excluded_reason_counts": excluded_reason_counts,
    })
    manifest["audit"].update({
        "data_integrity_pass": integrity, "split_grouping": grouping,
        "original_data_integrity_pass": original["audit"].get("data_integrity_pass", False),
        "integrity_evidence": "Only active valid-annotation samples assessed; original invalid annotations remain unchanged in excluded source records",
        "original_strict_split_status": original["split_status"],
        "active_cross_split_exact_hashes": leaking_hashes,
        "original_annotation_error_count": original["audit"].get("annotation_error_count"),
        "original_annotation_errors": original["audit"].get("annotation_errors", []),
        "annotation_error_count": 0, "annotation_errors": [],
        "active_annotation_error_count": 0, "excluded_images": excluded,
        "excluded_official_test_images_not_scored": [item["sample_key"] for item in excluded if item["source_split"].startswith("official_test")],
        "acquisition_group_evidence": "NOT_AVAILABLE; this exploratory draft does not meet the original sequence-group protocol",
        "test_use": "Existing annotation-integrity findings only; test labels never drive development balancing or the small-object threshold",
    })
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "excluded_images.json", {"source_manifest_sha256": source_sha256, "protocol": protocol, "images": excluded})
    _write_json(output / "validation_class_counts.json", {"source_manifest_sha256": source_sha256, "protocol": protocol, "counts": class_counts["val"]})
    _write_json(output / "draft_summary.json", {
        "source_manifest_sha256": source_sha256, "protocol": protocol, "split_counts": counts,
        "eligible_development_count": len(development), "active_test_count": counts.get("test", 0),
        "excluded_source_counts": excluded_source_counts, "excluded_reason_counts": excluded_reason_counts,
        "active_class_box_counts": class_counts, "small_area_threshold": threshold,
        "grouping": grouping, "active_data_integrity_pass": integrity,
    })
    text = [
        "# 公开数据下的探索比较：待确认的替代方案", "",
        "**状态：等待用户选择，本草稿不能启动正式训练。原先按真实采集序列分组的方案仍未满足。**", "",
        "## 这次改变什么", "",
        "没有真实采集序列对应表，因此改成只把完全相同的原图副本放在同一部分。",
        "再尽量让训练和验证中的各类目标数量接近 8:2。这能支持公开数据下的方案比较，不能证明换到新场景仍有效。",
        "不采用外观相似性大组，也不复用以前实验的数据划分。", "",
        "## 数据如何处理", "",
        f"有原始框异常的图整张退出本轮训练或评分：{excluded_reason_counts.get('INVALID_ORIGINAL_ANNOTATION_WHOLE_IMAGE', 0)} 张。",
        f"开发图与官方测试原图完全重复的副本退出训练：{excluded_reason_counts.get('EXACT_DUPLICATE_OF_OFFICIAL_TEST', 0)} 张。",
        f"退出图按官方部分统计：{excluded_source_counts}。",
        "退出只发生在派生清单中。原始图像、目标框和压缩包全部不改；异常框不自动修正，不删除目标后把图当作背景。",
        "官方测试原有归属不改；标注异常的测试图明确不计分。因此新成绩不能直接冒充完整官方测试集成绩。", "",
        "## 划分预览", "",
        f"剩余开发图 {len(development)} 张；训练 {counts.get('train', 0)} 张、验证 {counts.get('val', 0)} 张。",
        f"最后评分的测试图 {counts.get('test', 0)} 张；所有退出图共 {counts.get('excluded', 0)} 张。",
        f"完全相同图像的分组数：{grouping['group_count']}。固定随机编号：{seed}。",
        "类别平衡只使用开发图标注；不看测试结果选择划分。",
        "小目标界线仅由新训练部分的原图目标框面积占比计算，取最小四分之一的界线；缩放和补边不参与。", "",
        "| 类别 | 训练目标框数 | 验证目标框数 | 测试目标框数 |", "|---|---:|---:|---:|"]
    for name in manifest["classes"]:
        label = f"{name}（{CLASS_NAMES_ZH.get(name, name)}）"
        text.append(f"| {label} | {class_counts['train'][name]} | {class_counts['val'][name]} | {class_counts['test'][name]} |")
    text.extend(["", "## 可以怎样解释最终结果", "",
        "允许：在这个清楚记录的公开数据划分下，比较八个网络结构，判断按层选择普通或扩张卷积是否有帮助。",
        "不能声称：已经控制了真实采集序列，或证明模型能推广到没有见过的新场景。",
        "仍可能有相邻帧或相似场景跨集合，这是本方案的已知限制。", "",
        "## 可追溯记录", "",
        f"原始严格清单校验值（SHA256，即识别文件内容是否变化的校验编码）：`{source_sha256}`。",
        "排除图清单见 excluded_images.json，验证各类数量见 validation_class_counts.json。",
        "派生清单保留全部图像记录、原始标注校验值和退出理由。", "",
        "只有用户明确选择此替代协议后，才能另建批准副本；本草稿及原严格清单保持当前状态。", ""])
    (output / "探索划分方案.md").write_text("\n".join(text), encoding="utf-8")
    return manifest
