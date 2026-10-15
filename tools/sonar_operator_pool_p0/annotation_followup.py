"""Train/valid annotation integrity and train-only cross-seed FP review; no training."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def iou(a, b):
    x = max(0., min(a[2], b[2])-max(a[0], b[0]))
    y = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = x*y
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter, 1e-12)


def allowed_path(root, image, split):
    if split not in ('train', 'valid'):
        raise ValueError('Only train/valid are allowed')
    relative = Path(image.replace('\\', '/'))
    target = (root/relative).resolve()
    if not target.is_relative_to((root/split/'images').resolve()):
        raise ValueError(f'Image outside allowed split: {image}')
    return target


def audit_labels(root):
    rows, boxes, inputs = [], {}, []
    for split in ('train', 'valid'):
        directory = root/split/'images'
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        if len(images) != {'train': 330, 'valid': 99}[split]:
            raise ValueError('Unexpected Dataset A split cardinality')
        labels = root/split/'labels'
        image_stems = {p.stem for p in images}
        orphan_labels = sorted(p.name for p in labels.glob('*.txt') if p.stem not in image_stems)
        for path in images:
            label = labels/(path.stem+'.txt')
            name = path.relative_to(root).as_posix()
            with Image.open(path) as im:
                im.load()
                width, height = im.size
            errors = []
            parsed = []
            if not label.is_file():
                errors.append('missing_label')
            else:
                for line_idx, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    try:
                        if len(fields) != 5:
                            raise ValueError('expected_five_yolo_fields')
                        values = np.asarray([float(v) for v in fields])
                        if not np.isfinite(values).all():
                            raise ValueError('nonfinite')
                        cls, cx, cy, w, h = values
                        if cls != int(cls) or int(cls) not in (0, 1):
                            raise ValueError('invalid_class')
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            raise ValueError('invalid_normalized_coordinates')
                        box = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                        parsed.append(dict(class_id=int(cls), box=box))
                    except ValueError as exc:
                        errors.append(f'line{line_idx}:{exc}')
            bounds = sum(any(v < -1e-7 or v > 1+1e-7 for v in p['box']) for p in parsed)
            touching = sum(any(abs(v) < 1e-7 or abs(v-1) < 1e-7 for v in p['box']) for p in parsed)
            duplicate_rows = len(parsed)-len({(p['class_id'], *p['box']) for p in parsed})
            rows.append(dict(split=split, image=name, width=width, height=height,
                             label_exists=label.is_file(), gt_count=len(parsed),
                             cylider_count=sum(p['class_id']==0 for p in parsed),
                             manta_count=sum(p['class_id']==1 for p in parsed),
                             invalid_errors=';'.join(errors), bbox_exceeds_boundary=bounds,
                             bbox_touches_boundary=touching, duplicate_annotation_rows=duplicate_rows,
                             split_orphan_labels=';'.join(orphan_labels)))
            boxes[name] = [dict(class_id=p['class_id'], box=[float(v*640) for v in p['box']]) for p in parsed]
            inputs.append(dict(image=name, image_sha256=sha(path), label_sha256=sha(label) if label.is_file() else None))
    return rows, boxes, inputs


def review_candidates(pred_dir, boxes, source_manifest):
    by_image = defaultdict(dict)
    inputs = []
    for seed in (42, 43, 44):
        name = f'train_D_seed{seed}_epoch80'
        path = pred_dir/(name+'.jsonl.gz')
        expected = next(a for a in source_manifest['artifacts'] if a['tag']==name)['predictions_sha256']
        if sha(path) != expected:
            raise ValueError('Prediction hash mismatch')
        inputs.append(dict(path=str(path), sha256=expected))
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        if len(records) != 330:
            raise ValueError('Expected complete train predictions')
        for r in records:
            name = r['image']
            if Path(name).parts[0] != 'train':
                raise ValueError('Non-train prediction in review set')
            ground = boxes[name]
            if len(ground) != len(r['target']['boxes']):
                raise ValueError('Annotation/prediction GT mismatch')
            for g, b, c in zip(ground, r['target']['boxes'], r['target']['labels']):
                if g['class_id'] != c or not np.allclose(g['box'], b, atol=1e-3, rtol=0):
                    raise ValueError('Annotation coordinate mismatch')
            p = r['prediction']
            selected = []
            for idx, (b, s, c) in enumerate(zip(p['boxes'], p['scores'], p['labels'])):
                if s >= .5 and max((iou(b, g['box']) for g in ground), default=0) < .1:
                    selected.append(dict(box=b, score=s, class_id=c, prediction_index=idx))
            by_image[name][seed] = selected
    rows = []
    for image, seeds in sorted(by_image.items()):
        for ref in sorted(seeds[42], key=lambda r: (-r['score'], r['prediction_index'])):
            matches = [ref]
            for seed in (43, 44):
                candidates = [p for p in seeds[seed] if p['class_id']==ref['class_id'] and iou(p['box'], ref['box']) >= .5]
                if not candidates:
                    break
                matches.append(max(candidates, key=lambda p: (iou(p['box'],ref['box']),p['score'],-p['prediction_index'])))
            if len(matches) == 3:
                rows.append(dict(image=image, class_id=ref['class_id'], class_name=('cylider','manta')[ref['class_id']],
                                 x1=ref['box'][0], y1=ref['box'][1], x2=ref['box'][2], y2=ref['box'][3],
                                 score42=matches[0]['score'], score43=matches[1]['score'], score44=matches[2]['score'],
                                 minimum_seed_score=min(p['score'] for p in matches),
                                 cross_seed_iou43=iou(ref['box'],matches[1]['box']),
                                 cross_seed_iou44=iou(ref['box'],matches[2]['box']),
                                 semantic_status='UNRESOLVED_NOT_EXPERT_ANNOTATED'))
    # Review at most one reference region per image, then up to eight per class.
    rows.sort(key=lambda r: (-r['minimum_seed_score'], r['image'],r['x1'],r['y1']))
    chosen, used, counts = [], set(), Counter()
    for r in rows:
        if r['image'] in used or counts[r['class_id']]>=8:
            continue
        chosen.append(dict(r,review_id=f'R{len(chosen)+1:02d}'))
        used.add(r['image']); counts[r['class_id']]+=1
    return rows, chosen, inputs


def render_review(root, out, chosen, boxes):
    sheets = []
    for r in chosen:
        im = Image.open(allowed_path(root,r['image'],'train')).convert('RGB').resize((640,640))
        draw = ImageDraw.Draw(im)
        for gt in boxes[r['image']]:
            draw.rectangle(gt['box'], outline='lime', width=3)
        box = [r[k] for k in ('x1','y1','x2','y2')]
        draw.rectangle(box,outline='red',width=3)
        boxed_only = im.copy()
        draw.text((5,5),f"{r['review_id']} {r['class_name']} FP proxy, min score {r['minimum_seed_score']:.3f}",fill='yellow')
        draw.text((5,20),'Green=GT, red=prediction; NOT a verified missing annotation',fill='yellow')
        im.save(out/'examples'/(r['review_id']+'.png'))
        # Enlarge a 96px-minimum neighborhood for inspection, without altering image content.
        center=((box[0]+box[2])/2,(box[1]+box[3])/2)
        side=max(96,box[2]-box[0]+48,box[3]-box[1]+48)
        region=(max(0,int(center[0]-side/2)),max(0,int(center[1]-side/2)),min(640,int(center[0]+side/2)),min(640,int(center[1]+side/2)))
        tile=Image.new('RGB',(640,360),'white')
        tile.paste(im.resize((320,320)),(0,30))
        tile.paste(boxed_only.crop(region).resize((320,320)),(320,30))
        ImageDraw.Draw(tile).text((5,5),f"{r['review_id']} full image + FP neighborhood | {r['class_name']}",fill='black')
        sheets.append(tile)
    for start in range(0,len(sheets),4):
        page=Image.new('RGB',(1280,720),'white')
        for n,tile in enumerate(sheets[start:start+4]):
            page.paste(tile,((n%2)*640,(n//2)*360))
        page.save(out/f'montage_{start//4+1}.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8'))
    parser.add_argument('--diagnosis',type=Path,default=Path('reports/sonar_operator_pool_p0_diagnosis'))
    parser.add_argument('--output',type=Path,default=Path('reports/sonar_operator_pool_p0_followup/annotation'))
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True); (out/'examples').mkdir(exist_ok=True)
    rows,boxes,inputs=audit_labels(args.data_root)
    manifest=json.loads((args.diagnosis/'INFERENCE_MANIFEST.json').read_text(encoding='utf-8'))
    candidates,chosen,pred_inputs=review_candidates(args.diagnosis/'predictions',boxes,manifest)
    dump_csv(out/'annotation_integrity.csv',rows)
    dump_csv(out/'cross_seed_background_candidates.csv',candidates)
    dump_csv(out/'review_index.csv',chosen)
    render_review(args.data_root,out,chosen,boxes)
    summary=[]
    for split in ('train','valid'):
        group=[r for r in rows if r['split']==split]
        summary.append(dict(split=split,images=len(group),gt=sum(r['gt_count'] for r in group),
                            missing=sum(not r['label_exists'] for r in group),invalid=sum(bool(r['invalid_errors']) for r in group),
                            orphan_label_files=len(set(n for r in group for n in r['split_orphan_labels'].split(';') if n)),
                            bbox_exceeds_boundary=sum(r['bbox_exceeds_boundary'] for r in group),
                            bbox_touches_boundary=sum(r['bbox_touches_boundary'] for r in group),
                            duplicate_annotation_rows=sum(r['duplicate_annotation_rows'] for r in group),
                            one_gt_per_image=sum(r['gt_count']==1 for r in group),
                            class_counts={name:sum(r[name+'_count'] for r in group) for name in ('cylider','manta')}))
    report=['# Background annotation completeness audit','',
            'This script opened only train/valid image and label folders. This script did not read a combined source index, test metadata, or test files, and performed no training or annotation editing. These access statements apply to this script, not to earlier unrelated tool calls.',
            '', '## Technical integrity', '', '```json',json.dumps(summary,indent=2),'```','',
            'A technically valid file is not evidence that every physical target in that image was annotated. Boundary-touching boxes identify possible clipping, not incorrect labels.',
            '', '## Deterministic train-only review queue','',
            f'{len(candidates)} seed42-reference regions reproduce across seeds42/43/44 at epoch80; {len(chosen)} distinct-image examples are shown.',
            f"These regions occur in {len({r['image'] for r in candidates})} train images; predicted class counts are {dict(Counter(r['class_name'] for r in candidates))}.",
            'Frozen review rule: each seed score >=0.5; max IoU to annotation <0.1; same predicted class and IoU >=0.5 to seed42 region. Rank by minimum score across seeds, then image/coordinates; at most one per image and eight per class.',
            'This queue is deliberately enriched for recurrent errors. It cannot estimate dataset missing-label prevalence. Shared models can repeat the same background error.',
            '', '## Semantic completeness status','',
            '**UNRESOLVED.** No expert exhaustive annotation or second independent ground truth is available. The repeated proposals are review candidates, not newly discovered ground truth.',
            'All 429 images carry exactly one GT if confirmed by the table above. This supports a single-target annotation convention; it does not prove that other bright structures are targets.',
            'Manual/expert review must distinguish: genuine unlabeled target, structured background, ambiguous target-like region, and annotation boundary issue. No labels or training protocol are changed.',
            '', '## Scope', '', '`new_training_runs=0`; `test_accessed=false`; `labels_modified=false`.']
    (out/'ANNOTATION_COMPLETENESS_AUDIT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    binding=dict(status='TECHNICAL_AUDIT_COMPLETE_SEMANTIC_COMPLETENESS_UNRESOLVED',splits=['train','valid'],summary=summary,
                 source_prediction_hashes=pred_inputs,data_binding=inputs,script_sha256=sha(Path(__file__)),
                 source_index_read=False,test_accessed=False,new_training_runs=0,labels_modified=False,
                 recurrent_reference_regions=len(candidates),review_images=len(chosen))
    (out/'ANNOTATION_AUDIT_MANIFEST.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in binding.items() if k not in ('data_binding','source_prediction_hashes')},indent=2))


if __name__=='__main__':
    main()
