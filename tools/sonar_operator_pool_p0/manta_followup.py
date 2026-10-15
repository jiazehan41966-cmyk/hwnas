"""Existing train/valid distribution and class-recall audit; no model execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/sonar_analysis'))
from complete_analysis import contrast_region
from hwnas_fpga.data.detection import yolo_normalized_to_original_xyxy

FIELDS = ['width_640', 'height_640', 'side_640', 'normalized_area', 'aspect_ratio',
          'center_x', 'center_y', 'image_mean', 'image_std', 'target_mean',
          'target_std', 'background_mean', 'background_std', 'contrast']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No output rows: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def bootstrap_mean(values, rng, repeats=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)


def ci(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def group_proxy(stem):
    original = stem.split('.rf.')[0]
    matched = re.fullmatch(r'(\d{4}_\d{2}_\d{2}_exp\d+)_(\d+)_bmp', original)
    return (matched.group(1), int(matched.group(2))) if matched else ('UNRESOLVED', None)


def collect(config):
    classes = config['dataset']['class_names']
    rows, files = [], []
    for split in ('train', 'valid'):
        paths = config['dataset']['split_paths'][split]
        image_root = (ROOT / paths['images']).resolve()
        label_root = (ROOT / paths['labels']).resolve()
        images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.bmp'})
        if not images:
            raise ValueError(f'No {split} images')
        for image in images:
            label = label_root / (image.stem + '.txt')
            if not label.is_file():
                raise ValueError(f'Missing annotation: {label}')
            with Image.open(image) as im:
                a = np.asarray(im.convert('L'), dtype=np.float64) / 255.
            height, width = a.shape
            annotations = []
            for line in label.read_text().splitlines():
                fields = [float(v) for v in line.split()]
                if len(fields) != 5 or not np.isfinite(fields).all():
                    raise ValueError(f'Invalid YOLO row: {label}')
                cid, cx, cy, bw, bh = fields
                if cid != int(cid) or not 0 <= cid < len(classes) or not (bw > 0 and bh > 0):
                    raise ValueError(f'Invalid class/size: {label}')
                # Reuse frozen float32 geometry before floor/ceil pixel cropping.
                # Python float geometry can cross an integer boundary by rounding.
                frozen_boxes, _ = yolo_normalized_to_original_xyxy(
                    [fields], orig_width=width, orig_height=height)
                box = tuple(frozen_boxes[0].tolist())
                if min(box) < -1e-5 or box[2] > width+1e-5 or box[3] > height+1e-5:
                    raise ValueError(f'Out-of-bound bbox: {label}')
                annotations.append((int(cid), cx, cy, bw, bh, box))
            # Dataset A has one GT per image. This condition ensures that image bootstrap
            # is genuinely image-level, never an accidental instance-level resample.
            if len(annotations) != 1:
                raise ValueError('Image-level bootstrap contract requires one GT per image for this audit')
            proxy, index = group_proxy(image.stem)
            for gt_index, (cid, cx, cy, bw, bh, box) in enumerate(annotations):
                contrast = contrast_region(a, box, [r[-1] for r in annotations], alpha=2., epsilon=1e-6)
                if contrast['status'] != 'ok':
                    raise ValueError(f'Unavailable ring contrast: {image}')
                side = float(np.sqrt(bw*bh)*640)
                rows.append(dict(split=split, image=f'{split}/images/{image.name}', gt_index=gt_index,
                    class_id=cid, class_name=classes[cid], native_width=width, native_height=height,
                    width_640=bw*640, height_640=bh*640, side_640=side,
                    normalized_area=bw*bh, aspect_ratio=bw/bh, center_x=cx, center_y=cy,
                    image_mean=float(a.mean()), image_std=float(a.std()), **contrast,
                    size_bin='side_lt16' if side < 16 else 'side_16_to_lt32' if side < 32 else 'side_ge32',
                    filename_proxy=proxy, filename_index=index))
            files.extend([dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in (image, label)])
    expected = {('train', 'cylider'):234, ('train','manta'):96, ('valid','cylider'):70, ('valid','manta'):29}
    for key, count in expected.items():
        if sum((r['split'],r['class_name']) == key for r in rows) != count:
            raise ValueError(f'Dataset binding count mismatch: {key}')
    return rows, files


def run(config_path, diagnosis, output):
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    protocol = dict(version='train_valid_manta_followup_v1', splits=['train','valid'], test_accessed=False,
        new_training_runs=0, checkpoint_selection='existing D best and epoch80; unchanged',
        classes=['cylider','manta'], contrast='reuse complete_analysis.contrast_region; native PIL gray/255, alpha=2, epsilon=1e-6; exclude all GT',
        geometry='640x640 square resize; no augmentation', fields=FIELDS,
        size_bins='sqrt bbox area in input pixels: <16 / [16,32) / >=32; not COCO',
        bootstrap=dict(unit='image', repeats=2000, seed=42, interval='percentile95',
                       limitation='descriptive; possible acquisition dependence; no multiplicity correction or causal claim'),
        filename='strip Roboflow suffix; YYYY_MM_DD_expN weak proxy only; not verified acquisition group',
        baseline_config_sha256=sha(config_path), gt_rows_sha256=sha(diagnosis/'gt_rows.csv'))
    protocol_path = output/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Diagnostic protocol drift')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows, files = collect(config)
    write_csv(output/'image_bbox_statistics.csv', rows)
    historical = {r['image'].replace('\\','/'):r for r in read_csv(ROOT/'reports/sonar_analysis_full/contrast_statistics.csv')}
    reproduction = max(abs(r['contrast']-float(historical[r['image']]['contrast'])) for r in rows if r['split']=='train')
    if reproduction > 1e-10:
        raise ValueError(f'Frozen train contrast did not reproduce: {reproduction}')
    rng = np.random.default_rng(42)
    descriptives, effects = [], []
    for name in ('cylider','manta'):
        groups = {split:[r for r in rows if r['split']==split and r['class_name']==name] for split in ('train','valid')}
        for field in FIELDS:
            boots = {}
            for split, group in groups.items():
                values = np.array([r[field] for r in group], dtype=float)
                boots[split] = bootstrap_mean(values, rng)
                lo, hi = ci(boots[split]); q = np.quantile(values,[.1,.25,.5,.75,.9])
                descriptives.append(dict(class_name=name, split=split, field=field, n_images=len(values),
                    mean=float(values.mean()), std=float(values.std(ddof=1)), p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4],
                    mean_ci_low=lo, mean_ci_high=hi))
            lo, hi = ci(boots['valid']-boots['train'])
            a=np.array([r[field] for r in groups['train']]); b=np.array([r[field] for r in groups['valid']])
            effects.append(dict(class_name=name, field=field, n_train=len(a), n_valid=len(b),
                mean_valid_minus_train=float(b.mean()-a.mean()), mean_delta_ci_low=lo, mean_delta_ci_high=hi,
                median_valid_minus_train=float(np.median(b)-np.median(a))))
    write_csv(output/'distribution_summary.csv',descriptives); write_csv(output/'split_effects.csv',effects)
    proxy_rows=[]
    for name in ('cylider','manta'):
        for proxy in sorted({r['filename_proxy'] for r in rows}):
            result=dict(class_name=name, filename_proxy=proxy)
            for split in ('train','valid'):
                group=[r for r in rows if r['split']==split and r['class_name']==name]
                subset=[r for r in group if r['filename_proxy']==proxy]
                result[split+'_count']=len(subset);result[split+'_fraction']=len(subset)/len(group)
            proxy_rows.append(result)
    write_csv(output/'filename_proxy_distribution.csv',proxy_rows)
    gt_rows=read_csv(diagnosis/'gt_rows.csv'); metadata={r['image']:r for r in rows}
    recalls=[]
    for split in ('train','valid'):
        for seed in (42,43,44):
            for checkpoint in ('best','epoch80'):
                selected=[r for r in gt_rows if r['split']==split and int(r['seed'])==seed and r['checkpoint']==checkpoint]
                if len(selected) != (330 if split=='train' else 99):raise ValueError('Missing diagnostic GT rows')
                for name in ('cylider','manta'):
                    group=[r for r in selected if r['class_name']==name]
                    for stratifier in ('all','size_bin','filename_proxy'):
                        strata=['all'] if stratifier=='all' else (['side_lt16','side_16_to_lt32','side_ge32'] if stratifier=='size_bin' else sorted({r['filename_proxy'] for r in rows}))
                        for stratum in strata:
                            sub=[r for r in group if stratifier=='all' or metadata[r['image']][stratifier]==stratum]
                            for iou in (50,75):
                                values=np.array([r[f'detected_iou{iou}']=='True' for r in sub], dtype=float)
                                interval=ci(bootstrap_mean(values,rng)) if len(values) else [None,None]
                                recalls.append(dict(split=split,seed=seed,checkpoint=checkpoint,class_name=name,
                                    stratifier=stratifier,stratum=stratum,iou=iou/100,gt_count=len(values),
                                    detected_count=int(values.sum()),recall=float(values.mean()) if len(values) else None,
                                    recall_ci_low=interval[0],recall_ci_high=interval[1]))
    write_csv(output/'stratified_recall.csv',recalls)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax, field in zip(axes.ravel(),['side_640','aspect_ratio','center_x','center_y','contrast','image_std']):
        for split,color in [('train','#2274A5'),('valid','#E07A36')]:
            values=np.sort([r[field] for r in rows if r['class_name']=='manta' and r['split']==split])
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{split} n={len(values)}',color=color)
        ax.set_xlabel(field); ax.set_ylabel('Empirical CDF');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Manta: fixed train/valid distribution audit');fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(figures/f'manta_distributions.{suffix}',dpi=170)
    plt.close(fig);write_csv(figures/'manta_distributions.csv',[r for r in rows if r['class_name']=='manta'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    plotted=[]
    for ax,name in zip(axes,['cylider','manta']):
        for split,color in [('train','#2274A5'),('valid','#E07A36')]:
            values=[]
            for seed in (42,43,44):
                r=next(r for r in recalls if r['class_name']==name and r['split']==split and r['seed']==seed and r['checkpoint']=='epoch80' and r['stratifier']=='all' and r['iou']==.5)
                plotted.append(r);values.append(r['recall'])
            ax.plot([42,43,44],values,'o-',color=color,label=split)
        ax.set_title(name);ax.set_xlabel('Seed');ax.set_xticks([42,43,44]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Recall: score >= 0.5, IoU >= 0.5');axes[0].legend();axes[0].set_ylim(0,1);fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(figures/f'endpoint_class_recall.{suffix}',dpi=170)
    plt.close(fig);write_csv(figures/'endpoint_class_recall.csv',plotted)
    manifest=dict(status='COMPLETE',test_accessed=False,new_training_runs=0,
        input_images=429,input_gt=429,historical_train_contrast_max_abs_error=reproduction,
        protocol_sha256=sha(protocol_path),source_sha256=sha(Path(__file__)),data_files=files,
        outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',images=429,train_contrast_reproduction_error=reproduction,test_accessed=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline_sonar_frozen.yaml')
    parser.add_argument('--diagnosis',type=Path,default=ROOT/'reports/sonar_operator_pool_p0_diagnosis')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/sonar_operator_pool_p0_followup/manta')
    args=parser.parse_args();run(args.config,args.diagnosis,args.output)
