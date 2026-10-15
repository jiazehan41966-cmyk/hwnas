"""Read original YOLO annotations; produce reproducible geometry statistics only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hwnas_fpga.data.detection import YoloDetectionDataset, yolo_normalized_to_original_xyxy
from hwnas_fpga.runtime import load_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(path, num_classes):
    """Strict compatible reader: empty files are negatives; missing files are errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing annotation (not an explicit negative): {path}')
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(x) for x in line.split()]
            if len(row) != 5 or not np.isfinite(row).all():
                raise ValueError('expected five finite values: class cx cy w h')
            c, x, y, w, h = row
            if c != int(c) or not 0 <= c < num_classes:
                raise ValueError('invalid class id')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid normalized geometry')
            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                raise ValueError('bbox outside image; refusing silent clipping')
        except ValueError as exc:
            raise ValueError(f'{path}:{line_number}: {exc}') from exc
        rows.append(row)
    return rows


def bbox_statistics(rows, width, height, input_size, square_factor=1.25):
    """Pixel geometry uses original W,H; evaluation size uses square input space."""
    boxes, labels = yolo_normalized_to_original_xyxy(rows, orig_width=width, orig_height=height)
    records = []
    for index, (box, label) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x1, y1, x2, y2 = box
        w, h = x2-x1, y2-y1
        if w <= 0 or h <= 0:
            raise ValueError(f'Degenerate box {index}')
        area, ratio = w*h, w/h
        nw, nh = w/width*input_size, h/height*input_size
        records.append(dict(bbox_index=index, class_id=label, x1=x1, y1=y1, x2=x2, y2=y2,
            width_px=w, height_px=h, area_px2=area, normalized_area=area/(width*height),
            aspect_ratio=ratio, input_width_px=nw, input_height_px=nh, input_area_px2=nw*nh,
            shape='horizontal' if ratio > square_factor else 'vertical' if ratio < 1/square_factor else 'near_square',
            size_bin='small' if nw*nh < 32**2 else 'medium' if nw*nh < 96**2 else 'large'))
    return records


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(a, [0,.1,.5,.9,1]))), mean=float(a.mean()))


def summarize(boxes, images, names):
    n = len(boxes)
    result = {'images': len(images), 'instances': n, 'negative_images': sum(r['object_count']==0 for r in images)}
    for field in ('width_px', 'height_px', 'area_px2', 'normalized_area', 'aspect_ratio', 'input_width_px', 'input_height_px'):
        result[field] = describe([r[field] for r in boxes])
    result['objects_per_image'] = describe([r['object_count'] for r in images])
    result['size_counts'] = {k: sum(r['size_bin']==k for r in boxes) for k in ('small','medium','large')}
    result['shape_counts'] = {k: sum(r['shape']==k for r in boxes) for k in ('horizontal','vertical','near_square')}
    result['size_proportions'] = {k: v/n if n else None for k,v in result['size_counts'].items()}
    result['per_class'] = {name: {'instances': sum(r['class_id']==i for r in boxes),
        **{field: describe([r[field] for r in boxes if r['class_id']==i]) for field in
           ('area_px2','normalized_area','input_width_px','input_height_px','aspect_ratio')}} for i,name in enumerate(names)}
    return result


def plot_results(boxes, output, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = output/'figures'
    figdir.mkdir()
    def save(name):
        plt.tight_layout()
        plt.savefig(figdir/f'{name}.png', dpi=300)
        plt.savefig(figdir/f'{name}.svg')
        plt.close()
    for key, label in [('area_px2','Original bbox area (px²)'), ('normalized_area','BBox / image area'), ('aspect_ratio','Original bbox width / height')]:
        values = [r[key] for r in boxes]
        counts, edges = np.histogram(values, bins=20)
        write_csv(figdir/f'{key}_histogram.csv', [dict(left=float(a),right=float(b),count=int(c)) for a,b,c in zip(edges[:-1],edges[1:],counts)])
        plt.figure(figsize=(6,4))
        plt.stairs(counts, edges, fill=True, color='#0072B2')
        plt.xlabel(label); plt.ylabel('Instances'); save(f'{key}_histogram')
    plt.figure(figsize=(6,4))
    for i,name in enumerate(names):
        subset = [r for r in boxes if r['class_id']==i]
        plt.scatter([r['width_px'] for r in subset], [r['height_px'] for r in subset], label=name, alpha=.65)
    plt.xlabel('Original bbox width (px)'); plt.ylabel('Original bbox height (px)'); plt.legend()
    save('width_height_scatter')
    write_csv(figdir/'width_height_scatter.csv', [{k:r[k] for k in ('split','image','bbox_index','class_name','width_px','height_px')} for r in boxes])
    for field in ('area_px2','aspect_ratio'):
        plt.figure(figsize=(6,4))
        active = [(name,[r[field] for r in boxes if r['class_id']==i]) for i,name in enumerate(names)]
        active = [(name,v) for name,v in active if v]
        plt.boxplot([v for _,v in active], tick_labels=[name for name,_ in active], showfliers=True)
        plt.ylabel('Original bbox '+field); save('per_class_'+field)
        write_csv(figdir/f'per_class_{field}.csv', [{k:r[k] for k in ('split','image','bbox_index','class_name',field)} for r in boxes])
    rows = [dict(size_bin=k,count=sum(r['size_bin']==k for r in boxes)) for k in ('small','medium','large')]
    for row in rows:
        row['proportion'] = row['count']/len(boxes)
    write_csv(figdir/'size_proportions.csv', rows)
    plt.figure(figsize=(6,4)); plt.bar([r['size_bin'] for r in rows],[r['proportion'] for r in rows],color='#0072B2')
    plt.ylabel('Fraction of instances'); plt.title('COCO-style bins at network input resolution'); save('size_proportions')


def make_report(summary):
    s = summary['statistics']
    train = summary['by_split'].get('train')
    lines = ['# SONAR DATA ANALYSIS — 第一轮基础统计', '', '## 1. Dataset Overview',
        f"本次模式：{summary['scope']}；实际分析 {s['images']} 张、{s['instances']} 个 bbox。类别：{', '.join(summary['class_names'])}。",
        f"各划分可用图片数（只枚举文件）：{summary['available_images']}；实际分析数：{summary['sampled_images']}。",
        f"每类实例数：{ {k:v['instances'] for k,v in s['per_class'].items()} }。",
        '原始标签直接读取；抽样不经过增强或加权采样。清单包含选中图片和标签的 SHA256。', '',
        '## 2. Object Size Characteristics',
        f"原图 bbox 面积统计（px²）：{s['area_px2']}。",
        f"归一化 bbox 面积统计：{s['normalized_area']}。",
        f"small / medium / large：{s['size_counts']}；比例：{s['size_proportions']}。",
        '数据集未提供物理尺度定义。为与现有 detection_coco_eval.py 对照，使用网络输入尺度面积 <32²、[32²,96²)、≥96² 像素分箱；这是明确的统计约定，不是声呐物理小目标定义。原图面积与归一化面积同时保留。', '',
        '## 3. Shape Characteristics',
        f"原图宽高比：{s['aspect_ratio']}；形状计数：{s['shape_counts']}。",
        f"横向 r>{summary['square_factor']}；纵向 r<1/{summary['square_factor']}；其余近方形。阈值可配置，仅描述 axis-aligned bbox（轴对齐框）形状，不能推断真实方向。", '',
        '## 4. Intensity / Contrast Characteristics', '本轮未执行；没有低对比度或弱目标定量结论。', '',
        '## 5. Frequency Characteristics', '本轮未执行；没有频率能量或上下文依赖结论。', '',
        '## 6. Implications for NAS Search Space',
        '仅训练集结果可用于下列建议；小规模抽样结论待完整训练集复核。', '',
        '| Data Evidence | Possible Search-space Design |', '| --- | --- |']
    if train and train['instances']:
        small = train['size_proportions']['small']
        lines.append(f'| 训练子集在输入尺度 small 比例 {small:.1%} | '+('将高分辨率输出 / P2 候选列入后续验证，不代表已证实收益' if small>=.5 else '暂不据此优先加入 P2')+' |')
    else:
        lines.append('| 未分析训练集 | 不据验证/测试集设计搜索变量 |')
    lines += ['| 已保留各类别尺度分布；尚未验证尺度变化带来的检测误差 | 多尺度融合变量暂缓定案 |',
        '| 只有轴对齐框形状统计 | 暂不足以引入专用方向算子 |',
        '| 对比度、FFT、上下文依赖未分析 | 暂不增加专用增强、空洞或大感受野算子 |', '',
        '所有统计图的对应数据位于 figures/*.csv；逐框与逐图原始统计分别见 bbox_statistics.csv 和 image_statistics.csv。',
        '本报告没有进行模型训练、性能测量或声影标注。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-dir', help='Override dataset.data_dir; relative paths are repository-relative')
    p.add_argument('--output', required=True, help='New or empty output directory')
    p.add_argument('--splits', nargs='+', choices=['train','valid','val','test'], default=['train'])
    p.add_argument('--max-images-per-split', type=int, help='Seeded uniform sample; omit for all images')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--square-factor', type=float, default=1.25)
    args = p.parse_args(argv)
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        p.error('--max-images-per-split must be positive')
    if not np.isfinite(args.square_factor) or args.square_factor <= 1:
        p.error('--square-factor must be finite and >1')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    dc = cfg['dataset']
    if cfg.get('task',{}).get('name') != 'detection' or dc.get('annotation_format') != 'yolo':
        raise ValueError('Requires detection task with YOLO bbox annotations')
    root = Path(args.data_dir or dc['data_dir']).expanduser()
    root = root if root.is_absolute() else ROOT/root
    splits = list(dict.fromkeys('valid' if x=='val' else x for x in args.splits))
    images, boxes, available, sampled = [], [], {}, {}
    for split in splits:
        dataset = YoloDetectionDataset(root, split=split, image_size=dc['image_size'], input_channels=dc['input_channels'])
        names = dataset.class_names
        if len(names) != dc['num_classes'] or len(set(names)) != len(names):
            raise ValueError('data.yaml class names missing, duplicated, or inconsistent with config')
        available[split] = len(dataset)
        if args.max_images_per_split is None:
            selected = range(len(dataset))  # Full enumeration does not use RNG.
        else:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, ['train','valid','test'].index(split)]))
            selected = sorted(rng.choice(len(dataset), min(args.max_images_per_split,len(dataset)), replace=False))
        sampled[split] = len(selected)
        for index in selected:
            path = dataset.samples[index]
            label_path = dataset.labels_dir/(path.stem+'.txt')
            rows = read_labels(label_path, len(names))
            with Image.open(path) as im:
                width, height = im.size
                im.load()
            common = dict(split=split, image=str(path.relative_to(dataset.root)), annotation=str(label_path.relative_to(dataset.root)), image_width=width, image_height=height)
            records = bbox_statistics(rows,width,height,dc['image_size'],args.square_factor)
            images.append(dict(**common, object_count=len(records), image_sha256=sha256(path), annotation_sha256=sha256(label_path)))
            boxes.extend(dict(**common, **r, class_name=names[r['class_id']]) for r in records)
    if not boxes:
        raise ValueError('Selected dataset contains no GT boxes; bbox distributions cannot be computed')
    summary = dict(schema_version=1, scope='sample' if args.max_images_per_split else 'full_requested_splits',
        config=str(config_path), config_sha256=sha256(config_path), data_root=str(dataset.root),
        data_yaml_sha256=sha256(dataset.root/'data.yaml'), seed=args.seed, square_factor=args.square_factor,
        input_size=dc['image_size'], class_names=names, available_images=available, sampled_images=sampled,
        size_definition='COCO-style network-input area: <1024, [1024,9216), >=9216 px2; not physical sonar size',
        command=sys.argv, python=sys.version, numpy=np.__version__, source_sha256=sha256(__file__),
        statistics=summarize(boxes,images,names),
        by_split={sp:summarize([r for r in boxes if r['split']==sp],[r for r in images if r['split']==sp],names) for sp in splits})
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'bbox_statistics.csv',boxes)
    write_csv(output/'image_statistics.csv',images)
    plot_results(boxes,output,names)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (output/'SONAR_DATA_ANALYSIS.md').write_text(make_report(summary),encoding='utf-8')
    print(json.dumps(dict(output=str(output),images=len(images),boxes=len(boxes),statistics=summary['statistics']),ensure_ascii=False))


if __name__ == '__main__':
    main()
