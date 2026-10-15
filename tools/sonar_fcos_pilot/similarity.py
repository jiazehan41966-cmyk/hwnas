"""Image-similarity proxy groups, explicitly distinct from acquisition groups.

Thresholds are declared before reading test imagery. Official test images are
used only in a separate risk audit, never to form development groups or choose
the 80:20 development split preview. Nothing here authorizes formal training.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import _assign_groups, _write_json

MAX_HAMMING = 4
MAX_NORMALIZED_RMSE = .04


class _BKTree:
    def __init__(self):
        self.root = None

    def add(self, value: int, index: int):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int):
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.extend(node[1])
            stack.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        return found


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a
            return True
        return False


def _features(path):
    with Image.open(path) as image:
        gray = image.convert("L")
        small = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
        differences = small[:, 1:] > small[:, :-1]
        dhash = int.from_bytes(np.packbits(differences.reshape(-1)).tobytes(), "big")
        thumbnail = np.asarray(gray.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.
    return dhash, thumbnail


def _pair_examples(edges, development, output, limit=12):
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    # Include an ordinary and an almost-identical example rather than tuning
    # thresholds from these pictures. The selection never affects grouping.
    ordered = sorted(edges, key=lambda edge: edge["rmse"])
    selected = ordered[:limit//2] + ordered[-limit//2:]
    seen = set()
    for edge in selected:
        identity = (edge["a"], edge["b"])
        if identity in seen:
            continue
        seen.add(identity)
        a, b = development[edge["a"]], development[edge["b"]]
        canvas = Image.new("RGB", (960, 520), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, sample in [(0, a), (480, b)]:
            with Image.open(sample["image_path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((470, 470))
                canvas.paste(image, (offset + (480-image.width)//2, 35))
            draw.text((offset+5, 5), sample["sample_key"], fill="black")
        draw.text((5, 505), f"dHash distance={edge['hamming']}; normalized grayscale RMSE={edge['rmse']:.5f}; similarity is not acquisition evidence", fill="black")
        path = output / f"pair_{len(examples)+1:02d}.png"
        canvas.save(path)
        examples.append({"a": a["sample_key"], "b": b["sample_key"], "hamming": edge["hamming"], "rmse": edge["rmse"], "path": str(path.resolve())})
    return examples


def build_similarity_report(manifest_path, output_dir, *, include_test_audit=False, seed=42, val_fraction=.2):
    """Create a reviewable proxy alternative without changing the strict split.

    Every edge must satisfy both fixed dHash Hamming <= 4 and grayscale 64x64
    normalized RMSE <= .04. Connected components may contain endpoint images
    farther apart than these limits; the report explicitly discloses chaining.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    declaration = {
        "provenance_status": "IMAGE_SIMILARITY_PROXY_ONLY",
        "thresholds": {"dhash_hamming_max": MAX_HAMMING, "normalized_grayscale_rmse_max": MAX_NORMALIZED_RMSE},
        "feature_definition": "dHash: grayscale bilinear 9x8 horizontal comparisons; RMSE: bilinear grayscale 64x64 normalized by 255",
        "development_only_grouping": True, "test_used_for_grouping_or_threshold_selection": False,
        "formal_training_authorized_under_alternative": False,
    }
    _write_json(output / "protocol_declaration.json", declaration)
    development = sorted((sample for sample in manifest["samples"] if sample["source_split"] == "official_train"), key=lambda sample: sample["sample_key"])
    if not development:
        raise ValueError("Manifest contains no original official training images")
    hashes, thumbnails = [], []
    for i, sample in enumerate(development):
        dhash, thumbnail = _features(sample["image_path"])
        hashes.append(dhash)
        thumbnails.append(thumbnail)
        if (i+1) % 1000 == 0:
            print(f"[similarity] development thumbnails {i+1}/{len(development)}", flush=True)
    features = np.stack(thumbnails)
    del thumbnails
    means = features.mean(axis=1)
    norms = np.sqrt(np.mean(features**2, axis=1))
    union = _UnionFind(len(development))
    exact = defaultdict(list)
    for i, sample in enumerate(development):
        exact[sample["sha256"]].append(i)
    edges = []
    for indices in exact.values():
        for b in indices[1:]:
            if union.union(indices[0], b):
                edges.append({"a": indices[0], "b": b, "hamming": 0, "rmse": 0., "kind": "exact_content"})
    tree = _BKTree()
    candidate_count, rmse_count, skipped_connected = 0, 0, 0
    for i, dhash in enumerate(hashes):
        candidates = tree.query(dhash, MAX_HAMMING)
        candidate_count += len(candidates)
        pending = []
        for j in candidates:
            if union.find(j) == union.find(i):
                skipped_connected += 1
                continue
            # Both differences are mathematical lower bounds on pixel RMSE.
            if abs(means[i]-means[j]) <= MAX_NORMALIZED_RMSE and abs(norms[i]-norms[j]) <= MAX_NORMALIZED_RMSE:
                pending.append(j)
        for start in range(0, len(pending), 64):
            chunk = pending[start:start+64]
            distances = np.sqrt(np.mean((features[chunk]-features[i])**2, axis=1))
            rmse_count += len(chunk)
            for j, distance in zip(chunk, distances):
                if float(distance) <= MAX_NORMALIZED_RMSE and union.union(i, j):
                    edges.append({"a": i, "b": j, "hamming": (hashes[i]^hashes[j]).bit_count(), "rmse": float(distance), "kind": "near_similarity"})
        tree.add(dhash, i)
        if (i+1) % 1000 == 0:
            print(f"[similarity] group checks {i+1}/{len(development)}, pixel comparisons {rmse_count}", flush=True)
    components = defaultdict(list)
    for i in range(len(development)):
        components[union.find(i)].append(i)
    ordered_components = sorted(components.values(), key=lambda indices: min(development[i]["sample_key"] for i in indices))
    group_map, groups_summary = {}, []
    for group_index, indices in enumerate(ordered_components, 1):
        group_id = f"image_similarity_proxy_{group_index:05d}"
        members = [development[i] for i in indices]
        for sample in members:
            group_map[sample["sample_key"]] = group_id
        groups_summary.append({"group_id": group_id, "size": len(indices), "members": [sample["sample_key"] for sample in members]})
    proxy_map = {
        **declaration,
        "evidence": {"kind": "image_similarity_proxy", "reference": str((output / "similarity_report.json").resolve()), "explanation": "Appearance-only grouping; not verified source acquisition metadata"},
        "groups": group_map,
    }
    _write_json(output / "proxy_group_map.json", proxy_map)
    preview = copy.deepcopy(manifest)
    preview_audit = _assign_groups(preview["samples"], group_map, seed, val_fraction)
    preview["split_status"] = "PROXY_PREVIEW_NOT_AUTHORIZED"
    preview["small_area_threshold"] = None
    preview["provenance_status"] = "IMAGE_SIMILARITY_PROXY_ONLY"
    preview["audit"]["split_grouping"] = preview_audit
    preview["audit"]["formal_training_authorized_under_alternative"] = False
    preview_counts = dict(Counter(sample["split"] for sample in preview["samples"]))
    preview["split_counts"] = preview_counts
    class_coverage = {}
    for split in ("train", "val"):
        label_counts = Counter(label for sample in preview["samples"] if sample["split"] == split for label in sample["labels"])
        class_coverage[split] = {name: label_counts[class_id] for name, class_id in manifest["class_to_id"].items()}
    _write_json(output / "split_preview_not_authorized.json", preview)
    sizes = sorted((len(indices) for indices in ordered_components), reverse=True)
    test_audit = {"status": "NOT_CHECKED", "used_for_grouping": False}
    if include_test_audit:
        test_audit = _test_risk_audit(manifest, tree, hashes, features, means, norms, development)
        _write_json(output / "official_test_similarity_risk.json", test_audit)
    example_paths = _pair_examples(edges, development, output / "pair_examples")
    report = {
        **declaration, "development_count": len(development), "eligible_development_count": sum(not sample.get("excluded") for sample in development),
        "group_count": len(sizes), "singleton_count": sum(size == 1 for size in sizes),
        "largest_group_size": sizes[0], "largest_group_fraction": sizes[0] / len(development),
        "largest_20_group_sizes": sizes[:20], "group_size_histogram": dict(sorted(Counter(sizes).items())),
        "dhash_candidate_pair_count": candidate_count, "rmse_comparison_count": rmse_count,
        "already_connected_candidate_pairs_skipped": skipped_connected,
        "saved_connecting_edge_count": len(edges), "edge_count_note": "Only connecting edges saved; this is not the total number of qualifying image pairs",
        "groups": groups_summary, "connecting_edges": edges, "pair_examples": example_paths,
        "split_preview": {"status": "NOT_AUTHORIZED_FOR_TRAINING", "group_assignment_status": preview_audit["status"], "counts": preview_counts, "class_box_counts": class_coverage},
        "test_similarity_audit": test_audit,
        "limitations": [
            "Appearance similarity does not establish acquisition identity, time order, or scene independence",
            "A chain of similar images may produce a component whose endpoints are not similar",
            "Grayscale resizing ignores original aspect ratio and color; unrelated images can be grouped",
            "Singletons may still be adjacent frames; a low similarity score does not prove independence",
            "Official test findings only disclose risk; they do not select or change development groups",
            "Raw annotation anomalies remain unresolved and formal training remains blocked",
        ],
    }
    _write_json(output / "similarity_report.json", report)
    text = [
        "# 图像相似性分组：替代方案预览", "", "**这是外观相似性分组，不是真实采集序列分组；尚未授权用它正式训练。**", "",
        "固定判断规则：灰度图缩小后，相邻亮度变化编码最多相差 4 位；64×64 灰度缩略图的平均像素差（归一化均方根差）不超过 0.04。",
        "规则在读取官方测试图像之前锁定，测试只做独立风险审计，不参与开发分组或阈值选择。", "",
        f"原始开发图像 {len(development)} 张，形成 {len(sizes)} 组，其中只有一张图的组 {report['singleton_count']} 个。",
        f"最大组 {sizes[0]} 张，占开发数据 {100*sizes[0]/len(development):.2f}%。",
        f"8:2 分配预览：{preview_counts}；分组是否具备分配条件：{preview_audit['status']}。", "",
        "## 这个办法能做什么", "", "把非常相似的图放在同一部分，降低训练和验证出现近重复图的机会。", "",
        "## 这个办法不能证明什么", "", "它不能还原真实采集时间、序列和场景。连续帧可能仍被分开，不同采集也可能被合并。",
        "连通组有链式效应：A像B、B像C时，A和C也会被放在一组，但A和C不一定很像。", "",
        "原始标注中的异常仍需核验，本方案不会删除目标或改动原始文件。", "",
        "## 预览中的类别覆盖", "", "| 类别 | 训练目标框数 | 验证目标框数 |", "|---|---:|---:|"]
    for name in manifest["classes"]:
        text.append(f"| {name} | {class_coverage['train'][name]} | {class_coverage['val'][name]} |")
    text.extend(["", "## 官方测试风险", "", f"独立审计状态：{test_audit['status']}。", "",
        f"至少匹配一张开发图的测试图数量：{test_audit.get('matched_test_image_count', '未检查')}。",
        "正式训练需要用户明确接受外观分组这个协议变更，并解决原始标注异常；严格采集序列协议仍保持阻断。", ""])
    (output / "相似分组预览说明.md").write_text("\n".join(text), encoding="utf-8")
    return report


def _test_risk_audit(manifest, tree, hashes, features, means, norms, development):
    matches = []
    tests = [sample for sample in manifest["samples"] if sample["source_split"].startswith("official_test")]
    for i, sample in enumerate(tests):
        dhash, thumbnail = _features(sample["image_path"])
        candidates = tree.query(dhash, MAX_HAMMING)
        mean, norm = thumbnail.mean(), np.sqrt(np.mean(thumbnail**2))
        candidates = [j for j in candidates if abs(means[j]-mean) <= MAX_NORMALIZED_RMSE and abs(norms[j]-norm) <= MAX_NORMALIZED_RMSE]
        best = None
        for start in range(0, len(candidates), 64):
            chunk = candidates[start:start+64]
            distances = np.sqrt(np.mean((features[chunk]-thumbnail)**2, axis=1))
            j = int(np.argmin(distances))
            if float(distances[j]) <= MAX_NORMALIZED_RMSE and (best is None or float(distances[j]) < best[0]):
                best = (float(distances[j]), chunk[j])
        if best:
            distance, j = best
            matches.append({"test_sample": sample["sample_key"], "development_sample": development[j]["sample_key"], "rmse": distance, "hamming": (hashes[j]^dhash).bit_count(), "exact_content": sample["sha256"] == development[j]["sha256"]})
        if (i+1) % 400 == 0:
            print(f"[similarity] independent test risk audit {i+1}/{len(tests)}", flush=True)
    return {"status": "READ_ONLY_RISK_AUDIT", "test_image_count": len(tests), "matched_test_image_count": len(matches), "best_match_per_test_image": matches, "used_for_grouping": False, "development_split_changed_by_findings": False}
