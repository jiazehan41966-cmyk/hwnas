"""Original UATD annotations, evidence-backed splits, and reversible letterboxing.

This module intentionally does not import the repository's previous datasets,
networks, or split manifests. A numeric image filename is never a sequence ID.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

UATD_CLASSES = sorted([
    "ball", "circle cage", "cube", "cylinder", "human body", "metal bucket",
    "plane", "rov", "square cage", "tyre",
])
SOURCE_FOLDERS = {
    "UATD_Training": "official_train",
    "UATD_Test_1": "official_test_1",
    "UATD_Test_2": "official_test_2",
}
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GROUP_FIELDS = ("sequence_id", "acquisition_id", "video_id", "session_id")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> str | None:
    matches = {value for key, value in SOURCE_FOLDERS.items() if key in path.parts}
    return next(iter(matches)) if len(matches) == 1 else None


def _canonical_key(source: str, image: Path) -> str:
    return f"{source}/images/{image.name}"


def _load_group_map(group_map: str | Path | dict | None) -> tuple[dict, dict | None]:
    if group_map is None:
        return {}, None
    if not isinstance(group_map, dict):
        group_map = json.loads(Path(group_map).read_text(encoding="utf-8"))
    if group_map.get("provenance_status") == "IMAGE_SIMILARITY_PROXY_ONLY":
        raise ValueError("Image similarity is not verified acquisition evidence; the strict sequence split cannot accept this proxy")
    evidence = group_map.get("evidence", {})
    allowed = {"author_metadata", "source_log", "manual_verified_acquisition_map"}
    if evidence.get("kind") not in allowed or not evidence.get("reference") or not evidence.get("explanation"):
        raise ValueError("group_map requires evidence.kind, reference, and explanation; numeric filenames alone are not evidence")
    groups = group_map.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("group_map.groups must map canonical image keys to documented acquisition group IDs")
    for key, value in groups.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid acquisition group ID for {key}")
    return groups, evidence


def _read_annotation(path: Path, image_path: Path, class_to_id: dict) -> tuple[dict, list]:
    errors: list[dict] = []
    root = ET.parse(path).getroot()
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    xml_width = int(root.findtext("size/width", "0"))
    xml_height = int(root.findtext("size/height", "0"))
    if (xml_width, xml_height) != (width, height):
        errors.append({"kind": "size_mismatch", "xml": [xml_width, xml_height], "image": [width, height]})
    filename = root.findtext("file/filename") or root.findtext("filename")
    if filename and Path(filename).stem != image_path.stem:
        errors.append({"kind": "filename_mismatch", "xml_filename": filename})
    boxes, labels = [], []
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            errors.append({"kind": "unknown_class", "object_index": index, "name": name})
            continue
        try:
            box = [float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            errors.append({"kind": "malformed_box", "object_index": index, "name": name})
            continue
        if not all(math.isfinite(v) for v in box):
            errors.append({"kind": "nonfinite_box", "object_index": index, "name": name})
            continue
        boxes.append(box)
        labels.append(class_to_id[name])
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            reasons = []
            if x1 == x2 or y1 == y2:
                reasons.append("zero_area")
            if x1 > x2:
                reasons.append("reversed_x")
            if y1 > y2:
                reasons.append("reversed_y")
            if any(v < 0 for v in box) or max(x1, x2) > width or max(y1, y2) > height:
                reasons.append("outside_image")
            errors.append({"kind": "invalid_box", "object_index": index, "name": name, "box": box, "reasons": reasons})
    explicit_group = None
    for field in GROUP_FIELDS:
        values = [node.text.strip() for node in root.iter(field) if node.text and node.text.strip()]
        if values:
            if len(set(values)) != 1:
                errors.append({"kind": "ambiguous_group_field", "field": field, "values": values})
            else:
                explicit_group = f"{field}:{values[0]}"
            break
    return {
        "width": width, "height": height, "boxes": boxes, "labels": labels,
        "annotation_valid": not errors, "explicit_group": explicit_group,
        "xml_fields": sorted({node.tag for node in root.iter()}),
    }, errors


def _assign_groups(samples: list, groups: dict, seed: int, fraction: float) -> dict:
    """Split entire acquisition groups; exact duplicates link groups together."""
    development = [sample for sample in samples if sample["source_split"] == "official_train" and not sample.get("excluded")]
    missing = []
    for sample in development:
        group = groups.get(sample["sample_key"]) or sample.pop("explicit_group", None)
        if not group:
            missing.append(sample["sample_key"])
        sample["group_id"] = group or "UNVERIFIED"
    if missing:
        return {"status": "BLOCKED_GROUP_METADATA", "missing_group_count": len(missing), "missing_examples": missing[:20]}
    unique_groups = sorted({sample["group_id"] for sample in development})
    if len(unique_groups) < 2:
        return {"status": "BLOCKED_GROUP_METADATA", "reason": "At least two documented acquisition groups are required"}
    parent = {group: group for group in unique_groups}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    by_hash = defaultdict(list)
    for sample in development:
        by_hash[sample["sha256"]].append(sample)
    links = []
    for duplicates in by_hash.values():
        if len(duplicates) < 2:
            continue
        first = duplicates[0]["group_id"]
        for sample in duplicates[1:]:
            other = sample["group_id"]
            if find(first) != find(other):
                parent[find(other)] = find(first)
                links.append([first, other])
    components = defaultdict(list)
    for sample in development:
        components[find(sample["group_id"])].append(sample)
    if len(components) < 2:
        return {"status": "BLOCKED_GROUP_METADATA", "reason": "Exact duplicate links leave fewer than two independent groups", "duplicate_group_links": links}
    order = sorted(components)
    random.Random(seed).shuffle(order)
    target = len(development) * fraction
    chosen, count = [], 0
    for group in order:
        if len(chosen) == len(order) - 1:
            break
        size = len(components[group])
        if abs(count + size - target) < abs(count - target):
            chosen.append(group)
            count += size
    if not chosen:
        nearest = min(order, key=lambda group: abs(len(components[group]) - target))
        chosen = [nearest]
        count = len(components[nearest])
    selected = set(chosen)
    for group, members in components.items():
        for sample in members:
            sample["split"] = "val" if group in selected else "train"
    return {
        "status": "PASS", "group_count": len(unique_groups), "independent_group_count": len(components),
        "validation_fraction_observed": count / len(development), "duplicate_group_links": links,
    }


def prepare_manifest(dataset_root, output_dir, seed=42, val_fraction=.2, group_map=None) -> dict:
    """Audit raw UATD and create a sequence-group split only with source evidence.

    group_map is a JSON path or dictionary with `evidence` and `groups` fields.
    The generated group_map_template.json defines all required image keys.
    Original images, annotations, and official test membership remain untouched.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    groups, group_evidence = _load_group_map(group_map)
    class_to_id = {name: i + 1 for i, name in enumerate(UATD_CLASSES)}
    samples, errors, fields = [], [], Counter()
    annotations = sorted(path for path in root.rglob("*.xml") if _source(path) and not path.name.startswith("._"))
    if not annotations:
        raise ValueError(f"No official UATD annotations were found under {root}")
    for image_id, annotation in enumerate(annotations, 1):
        source = _source(annotation)
        image_dir = annotation.parent.parent / "images"
        matches = sorted(path for path in image_dir.glob(f"{annotation.stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(matches) != 1:
            errors.append({"annotation_path": str(annotation), "kind": "missing_or_ambiguous_image", "matches": [str(p) for p in matches]})
            continue
        image = matches[0]
        try:
            parsed, annotation_errors = _read_annotation(annotation, image, class_to_id)
        except (ET.ParseError, OSError, ValueError) as exc:
            errors.append({"annotation_path": str(annotation), "kind": "unreadable_source", "message": str(exc)})
            continue
        fields.update(parsed.pop("xml_fields"))
        errors.extend({"annotation_path": str(annotation), "image_path": str(image), "source_split": source, **error} for error in annotation_errors)
        sample = {
            "image_id": image_id, "image_path": str(image.resolve()), "annotation_path": str(annotation.resolve()),
            "sample_key": _canonical_key(source, image), "source_split": source,
            "split": "unassigned" if source == "official_train" else "test",
            "group_id": "UNVERIFIED" if source == "official_train" else f"official_membership:{source}",
            **parsed, "sha256": _sha256(image), "annotation_sha256": _sha256(annotation),
        }
        samples.append(sample)
    by_hash = defaultdict(list)
    for sample in samples:
        by_hash[sample["sha256"]].append(sample)
    duplicate_groups = [[sample["sample_key"] for sample in members] for members in by_hash.values() if len(members) > 1]
    excluded = []
    for members in by_hash.values():
        if not any(sample["split"] == "test" for sample in members):
            continue
        for sample in members:
            if sample["source_split"] == "official_train":
                sample["excluded"] = True
                sample["split"] = "excluded"
                sample["group_id"] = "excluded_exact_duplicate_of_official_test"
                excluded.append(sample["sample_key"])
    split_audit = _assign_groups(samples, groups, seed, val_fraction)
    status = split_audit["status"]
    if status == "PASS" and errors:
        status = "BLOCKED_DATA_INTEGRITY"
    for sample in samples:
        sample.pop("explicit_group", None)
    cross_split_duplicates = []
    for members in by_hash.values():
        active = [sample for sample in members if not sample.get("excluded")]
        if len({sample["split"] for sample in active}) > 1:
            cross_split_duplicates.append([sample["sample_key"] for sample in active])
    if status == "PASS" and cross_split_duplicates:
        status = "BLOCKED_DATA_INTEGRITY"
    train_areas = [
        (box[2] - box[0]) * (box[3] - box[1]) / (sample["width"] * sample["height"])
        for sample in samples if sample["split"] == "train" and sample["annotation_valid"]
        for box in sample["boxes"]
    ]
    if status == "PASS" and not train_areas:
        status = "BLOCKED_DATA_INTEGRITY"
    threshold = float(np.quantile(train_areas, .25)) if status == "PASS" else None
    counts = dict(Counter(sample["source_split"] for sample in samples))
    missing_sources = sorted(set(SOURCE_FOLDERS.values()) - set(counts))
    if missing_sources:
        errors.append({"kind": "missing_official_partition", "source_splits": missing_sources})
        if status == "PASS":
            status = "BLOCKED_DATA_INTEGRITY"
            threshold = None
    audit = {
        "data_integrity_pass": not errors and not cross_split_duplicates,
        "annotation_error_count": len(errors), "annotation_errors": errors,
        "original_source_counts": counts, "box_count": sum(len(sample["boxes"]) for sample in samples),
        "xml_fields": dict(sorted(fields.items())), "split_grouping": split_audit,
        "group_evidence": group_evidence, "exact_duplicate_groups": duplicate_groups,
        "excluded_official_train_duplicates_of_test": excluded,
        "cross_split_duplicates": cross_split_duplicates,
        "official_train_test_acquisition_disjointness": "UNVERIFIED: official membership preserved; acquisition metadata unavailable unless supplied separately",
        "box_coordinate_convention": "Original xyxy pixel values; no inferred one-pixel offset, clipping, sorting, or deletion",
        "test_use": "Integrity checks only; test labels are not used to select models, thresholds, or small-object definition",
    }
    manifest = {
        "schema_version": "sonar-fcos-pilot-data-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root), "seed": seed, "val_fraction_requested": val_fraction,
        "classes": UATD_CLASSES, "class_to_id": class_to_id, "samples": samples,
        "split_status": status, "small_area_threshold": threshold,
        "small_area_definition": "Original box area / original image area; training-only 25th percentile",
        "split_counts": dict(Counter(sample["split"] for sample in samples)), "audit": audit,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "annotation_issues.json", errors)
    template = {
        "evidence": {"kind": "author_metadata", "reference": "", "explanation": ""},
        "instructions_zh": "由作者采集记录、原始日志或经过核验的人工对应表填写；数字文件名、相同参数、相似图片不能直接当作同一采集序列的证据。",
        "groups": {sample["sample_key"]: None for sample in samples if sample["source_split"] == "official_train" and not sample.get("excluded")},
    }
    _write_json(output / "group_map_template.json", template)
    examples = render_annotation_examples(manifest, output / "annotation_examples")
    _write_json(output / "annotation_examples.json", examples)
    text = [
        "# 声呐检测第一轮：原始数据核验", "", f"当前状态：**{status}**。", "",
        f"官方部分图像数：{counts}。原始目标框数：{audit['box_count']}。",
        f"标注问题：{len(errors)} 条，完整记录见 annotation_issues.json。",
        f"从开发数据排除与官方测试图像完全相同的副本：{len(excluded)} 张；官方测试成员不变。", "",
        "## 采集序列证据", "",
        "普通数字文件名不等于采集序列。相同声呐参数也不能证明来自同一段采集。",
        "没有可核验的序列对应关系时，训练与验证不能按本计划分组，正式训练会停止。",
        "需要的最小材料：每张官方训练图像对应的采集序列编号，以及编号来源的说明。",
        "可填写 group_map_template.json；必须给出 evidence.reference 和 evidence.explanation。", "",
        "## 标注规则", "",
        "保留原始框坐标，不自动交换反向坐标、不把零面积目标删除成背景。异常需核验后修正，原始数据保持不变。",
        "示例图上的橙色框或十字表示坐标异常，供人工核验；图不能证明全部标注正确。", "",
        "## 小目标界线", "",
        "只在数据门槛通过后，使用训练目标框占原图面积的比例计算最小四分之一的界线。",
        "缩放和补边不参与这个定义；测试目标不会参与界线计算。", "",
    ]
    if errors:
        reasons_zh = {"zero_area": "宽度或高度为零", "reversed_x": "左右坐标反向", "reversed_y": "上下坐标反向", "outside_image": "坐标超出图像"}
        text.extend(["## 原始标注问题清单", "", "| 官方部分 | 标注文件 | 类别 | 问题 | 原始坐标 |", "|---|---|---|---|---|"])
        for error in errors:
            description = "、".join(reasons_zh.get(reason, reason) for reason in error.get("reasons", [])) or error["kind"]
            text.append(f"| {error.get('source_split', '')} | {Path(error.get('annotation_path', '')).name} | {error.get('name', '')} | {description} | {error.get('box', '')} |")
    (output / "数据核验说明.md").write_text("\n".join(text), encoding="utf-8")
    return manifest


def render_annotation_examples(manifest: dict, output_dir, per_source=2) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected, counts = [], Counter()
    for sample in manifest["samples"]:
        source = sample["source_split"]
        if counts[source] < per_source or not sample["annotation_valid"]:
            selected.append(sample)
            counts[source] += 1
    results = []
    for sample in selected:
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB")
            ratio = min(1., 1000 / max(image.size))
            image = image.resize((round(image.width * ratio), round(image.height * ratio)))
        draw = ImageDraw.Draw(image)
        for box, label in zip(sample["boxes"], sample["labels"]):
            valid = 0 <= box[0] < box[2] <= sample["width"] and 0 <= box[1] < box[3] <= sample["height"]
            coords = [value * ratio for value in box]
            color = "lime" if valid else "orange"
            x1, y1, x2, y2 = coords
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=color, width=2)
            if not valid:
                draw.line((x1-5, y1-5, x1+5, y1+5), fill=color, width=2)
                draw.line((x1-5, y1+5, x1+5, y1-5), fill=color, width=2)
            draw.text((max(0, min(x1, x2)), max(0, min(y1, y2)-12)), manifest["classes"][label-1], fill=color)
        path = output / f"{sample['source_split']}_{sample['image_id']:05d}.png"
        image.save(path)
        results.append({"image_id": sample["image_id"], "source_split": sample["source_split"], "image_path": sample["image_path"], "rendered_path": str(path.resolve()), "annotation_valid": sample["annotation_valid"]})
    return results


class SonarDataset:
    """640-square tensors with original geometry retained in each target."""

    def __init__(self, manifest_path, split, image_size=640):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if self.manifest.get("split_status") != "PASS" or not self.manifest.get("audit", {}).get("data_integrity_pass"):
            raise RuntimeError(f"Formal dataset loading blocked: {self.manifest.get('split_status')}; inspect the data audit")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.samples = [sample for sample in self.manifest["samples"] if sample["split"] == split]
        self.records = self.samples
        if not self.samples:
            raise ValueError(f"The {split} split is empty")
        self.image_size = int(image_size)
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as original:
            original = original.convert("RGB")
            width, height = original.size
            nominal_scale = self.image_size / max(width, height)
            resized_width = max(1, min(self.image_size, round(width * nominal_scale)))
            resized_height = max(1, min(self.image_size, round(height * nominal_scale)))
            resized = original.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        canvas.paste(resized, (left, top))
        tensor = torch.from_numpy(np.array(canvas, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.
        original_boxes = torch.tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4)
        scale = torch.tensor([resized_width / width, resized_height / height], dtype=torch.float32)
        pad = torch.tensor([left, top], dtype=torch.float32)
        boxes = original_boxes * scale.repeat(2) + pad.repeat(2)
        original_area = (original_boxes[:, 2]-original_boxes[:, 0]) * (original_boxes[:, 3]-original_boxes[:, 1])
        target = {
            "image_id": torch.tensor([sample["image_id"]], dtype=torch.int64),
            "boxes": boxes, "labels": torch.tensor(sample["labels"], dtype=torch.int64),
            "original_boxes": original_boxes, "original_size": torch.tensor([height, width], dtype=torch.int64),
            "resize_scale": scale, "pad_offset": pad,
            "area": (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "small_object_mask": original_area / (width * height) <= self.manifest["small_area_threshold"],
        }
        return tensor, target


def predictions_to_original(target, prediction) -> dict:
    """Undo rounded resize and padding; retain predictions inside image bounds."""
    import torch
    boxes = prediction["boxes"].detach().cpu().clone()
    scale = torch.as_tensor(target["resize_scale"]).detach().cpu().to(boxes.dtype)
    pad = torch.as_tensor(target["pad_offset"]).detach().cpu().to(boxes.dtype)
    height, width = [int(value) for value in target["original_size"]]
    boxes = (boxes - pad.repeat(2)) / scale.repeat(2)
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    labels = prediction["labels"].detach().cpu()
    # Class 0 is reserved for background, never an evaluated target class.
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (labels > 0)
    return {
        "image_id": int(torch.as_tensor(target["image_id"]).reshape(-1)[0]),
        "boxes": boxes[keep].tolist(),
        "labels": labels[keep].tolist(),
        "scores": prediction["scores"].detach().cpu()[keep].tolist(),
    }
