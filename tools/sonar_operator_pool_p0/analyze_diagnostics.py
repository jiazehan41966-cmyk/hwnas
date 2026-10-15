"""Analyze existing D train/valid predictions; no training or test access."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import run as p0
from error_breakdown import analyze_records, FP_KINDS, GT_REASONS


def load_records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def texture(a):
    if a.size==0:raise ValueError('Empty crop')
    gradients=np.concatenate([np.abs(np.diff(a,axis=0)).ravel(),np.abs(np.diff(a,axis=1)).ravel()])
    return dict(mean=float(a.mean()),std=float(a.std()),
        gradient_energy=float(gradients.mean()) if len(gradients) else 0.,
        edge_density=float((gradients>0.1).mean()) if len(gradients) else 0.)


def crop_box(box,size=640):
    x1,y1,x2,y2=box
    return (max(0,min(size-1,int(np.floor(x1)))),max(0,min(size-1,int(np.floor(y1)))),
            max(1,min(size,int(np.ceil(x2)))),max(1,min(size,int(np.ceil(y2)))))


def overlap(box,gt):
    return any(min(box[2],g[2])>max(box[0],g[0]) and min(box[3],g[3])>max(box[1],g[1]) for g in gt)


def background_proxies(out,records_by_key,pred_rows,data_root):
    rows=[];examples=[];cache={};groups=defaultdict(list)
    for r in pred_rows:
        if r['split']=='train' and r['kind']=='background':groups[(r['seed'],r['checkpoint'],r['image'])].append(r)
    for (seed,ck,image),preds in groups.items():
        relative=Path(image.replace('\\','/'))
        if relative.parts[0]!='train':raise ValueError('Only train images allowed for visual/proxy analysis')
        full=(data_root/relative).resolve()
        if not full.is_relative_to((data_root/'train').resolve()):raise ValueError('Unsafe image path')
        if image not in cache:
            with Image.open(full) as im:cache[image]=np.asarray(im.convert('L').resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        a=cache[image];record=records_by_key[('train',seed,ck)][image];gt=record['target']['boxes']
        for r in preds:
            box=crop_box([r[k] for k in ('x1','y1','x2','y2')]);x1,y1,x2,y2=box
            w,h=x2-x1,y2-y1
            if w<=0 or h<=0:continue
            raw=f'{seed}:{ck}:{image}:{r["prediction_index"]}'.encode()
            rng=np.random.default_rng(int(hashlib.sha256(raw).hexdigest()[:16],16))
            control=None
            for attempt in range(32):
                cx=int(rng.integers(0,641-w));cy=int(rng.integers(0,641-h))
                proposed=(cx,cy,cx+w,cy+h)
                if not overlap(proposed,gt):control=proposed;break
            stat=texture(a[y1:y2,x1:x2])
            row=dict(seed=seed,checkpoint=ck,image=image,prediction_index=r['prediction_index'],
                score=r['score'],x1=x1,y1=y1,x2=x2,y2=y2,control_available=control is not None,
                overlaps_annotated_gt_any=overlap(box,gt),**{'fp_'+k:v for k,v in stat.items()})
            if control:
                cx,cy,cx2,cy2=control
                row.update(control_x1=cx,control_y1=cy,control_x2=cx2,control_y2=cy2)
                ref=texture(a[cy:cy2,cx:cx2])
                for k,v in ref.items():row['control_'+k]=v;row['delta_'+k]=stat[k]-v
            rows.append(row)
    p0.write_csv(out/'background_fp_statistics.csv',rows)
    summaries=[]
    for seed in (42,43,44):
        for ck in ('best','epoch80'):
            group=[r for r in rows if r['seed']==seed and r['checkpoint']==ck]
            paired=[r for r in group if r['control_available'] and not r['overlaps_annotated_gt_any']]
            summary=dict(seed=seed,checkpoint=ck,background_fp=len(group),gt_disjoint_paired_crops=len(paired))
            for k in ('mean','std','gradient_energy','edge_density'):
                for prefix in ('fp_','control_','delta_'):
                    summary[prefix+k]=float(np.mean([r[prefix+k] for r in paired])) if paired else None
            summaries.append(summary)
            chosen=[];used=set()
            for r in sorted(group,key=lambda r:(-r['score'],r['image'],r['prediction_index'])):
                if r['image'] not in used:chosen.append(r);used.add(r['image'])
                if len(chosen)==2:break
            for i,r in enumerate(chosen):
                image=r['image'];im=Image.fromarray((cache[image]*255).astype('uint8')).convert('RGB');draw=ImageDraw.Draw(im)
                for b in records_by_key[('train',seed,ck)][image]['target']['boxes']:draw.rectangle(tuple(b),outline='lime',width=2)
                draw.rectangle((r['x1'],r['y1'],r['x2'],r['y2']),outline='red',width=3)
                draw.text((5,5),f'D seed{seed} {ck}: red FP, green GT; score={r["score"]:.3f}',fill='yellow')
                name=f'examples/train_seed{seed}_{ck}_fp{i+1}.png';(out/'examples').mkdir(exist_ok=True)
                im.save(out/name);examples.append(dict(r,example=name))
    p0.write_csv(out/'background_proxy_summary.csv',summaries)
    p0.write_csv(out/'examples/index.csv',examples)
    return summaries


def summarize(out,data_root):
    manifest=json.loads((out/'INFERENCE_MANIFEST.json').read_text())
    protocol=json.loads((out/'DIAGNOSTIC_PROTOCOL.json').read_text())
    expected={(s,n,c) for s in ('train','valid') for n in (42,43,44) for c in ('best','epoch80')}
    keys=[(a['split'],a['seed'],a['checkpoint']) for a in manifest['artifacts']]
    if len(keys)!=12 or set(keys)!=expected:raise ValueError('Expected 12 unique evaluations')
    if manifest['test_accessed'] or manifest['new_training_runs']!=0:raise ValueError('Scope violation')
    if p0.sha256(out/'DIAGNOSTIC_PROTOCOL.json')!=manifest['protocol_sha256']:raise ValueError('Protocol drift')
    classes=p0.load_frozen_config()['dataset']['class_names'];all_rows=defaultdict(list);records_by_key={};raw_metrics={}
    for a in manifest['artifacts']:
        key=(a['split'],a['seed'],a['checkpoint']);path=out/'predictions'/f'{a["tag"]}.jsonl.gz'
        if p0.sha256(path)!=a['predictions_sha256']:raise ValueError('Prediction hash mismatch')
        records=load_records(path)
        if len(records)!=(330 if a['split']=='train' else 99):raise ValueError('Incomplete split')
        if len({r['image'] for r in records})!=len(records):raise ValueError('Repeated image')
        if any(Path(r['image'].replace('\\','/')).parts[0]!=a['split'] for r in records):raise ValueError('Wrong split record')
        report=analyze_records(records,classes,*key)
        report['summary'].update({k:a['metrics'][k] for k in ('mAP50_95','AP50','AP75','Precision','Recall','validation_loss')})
        for k in ('true_positives','false_positives','ground_truths'):
            source={'true_positives':'tp','false_positives':'fp','ground_truths':'gt_count'}[k]
            if report['summary'][source]!=a['metrics'][k]:raise ValueError('Operating metric disagreement')
        all_rows['summary'].append(report['summary'])
        for table in ('image_rows','gt_rows','prediction_rows','class_size_rows'):all_rows[table].extend(report[table])
        per=a['metrics']['coco_details']['per_class_ap']['all']
        for name,v in per.items():
            all_rows['per_class_ap'].append(dict(split=key[0],seed=key[1],checkpoint=key[2],class_name=name,
                AP50=v['0.50'],AP75=v['0.75'],mAP50_95=float(np.mean(list(v.values())))))
        records_by_key[key]={r['image']:r for r in records};raw_metrics[key]=a['metrics']
    for name,rows in all_rows.items():p0.write_csv(out/(name+'.csv'),rows)
    paired=[];transitions=[];boot=[]
    for split in ('train','valid'):
        for seed in (42,43,44):
            tags=[r for r in all_rows['summary'] if r['split']==split and r['seed']==seed]
            by={r['checkpoint']:r for r in tags};left,right=by['best'],by['epoch80']
            paired.append(dict(split=split,seed=seed,**{k+'_delta':right[k]-left[k] for k in
                ('mAP50_95','AP50','AP75','validation_loss','tp','fp','fn','tp_iou75','recall',*(f'fp_{k}' for k in FP_KINDS))}))
            gs={ck:{(r['image'],r['gt_index']):r for r in all_rows['gt_rows'] if r['split']==split and r['seed']==seed and r['checkpoint']==ck} for ck in ('best','epoch80')}
            if gs['best'].keys()!=gs['epoch80'].keys():raise ValueError('Unpaired GT')
            for ident,l in gs['best'].items():
                r=gs['epoch80'][ident];transitions.append(dict(split=split,seed=seed,image=ident[0],gt_index=ident[1],
                    class_name=l['class_name'],size_bin=l['size_bin'],best_reason=l['reason'],endpoint_reason=r['reason'],
                    best_detected=l['detected_iou50'],endpoint_detected=r['detected_iou50'],
                    best_iou=l['best_same_class_iou'],endpoint_iou=r['best_same_class_iou']))
            ims={ck:{r['image']:r for r in all_rows['image_rows'] if r['split']==split and r['seed']==seed and r['checkpoint']==ck} for ck in ('best','epoch80')}
            names=sorted(ims['best']);gt=np.array([ims['best'][n]['gt_count'] for n in names]);delta=np.array([ims['epoch80'][n]['tp']-ims['best'][n]['tp'] for n in names])
            rng=np.random.default_rng(42);idx=rng.integers(0,len(names),size=(2000,len(names)));den=gt[idx].sum(axis=1)
            samples=delta[idx].sum(axis=1)/np.maximum(1,den);samples=samples[den>0]
            lo,hi=np.quantile(samples,[.025,.975]);boot.append(dict(split=split,seed=seed,n_images=len(names),
                recall_delta=float(delta.sum()/gt.sum()),ci95_low=float(lo),ci95_high=float(hi),unit='paired_image',resamples=2000,
                limitation='conditional on selected checkpoints; image exchangeability and acquisition independence unproven; exploratory only'))
    p0.write_csv(out/'paired_checkpoint_deltas.csv',paired);p0.write_csv(out/'gt_transitions.csv',transitions);p0.write_csv(out/'recall_bootstrap.csv',boot)
    # Ratio denominators remain visible; no pooled-domain AP and no threshold sweep.
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,('train','valid')):
        for seed in (42,43,44):
            vals=[raw_metrics[(split,seed,c)]['mAP50_95'] for c in ('best','epoch80')]
            ax.plot([0,1],vals,marker='o',label=f'seed{seed}')
        ax.set_xticks([0,1],['best valid checkpoint','epoch80']);ax.set_title(split);ax.set_ylabel('mAP50:95');ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(figdir/f'checkpoint_comparison.{ext}',dpi=180)
    plt.close(fig);p0.write_csv(figdir/'checkpoint_comparison.csv',all_rows['summary'])
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    error_rows=[]
    for row,split in enumerate(('train','valid')):
        for col,ck in enumerate(('best','epoch80')):
            subset=[r for r in all_rows['summary'] if r['split']==split and r['checkpoint']==ck]
            kinds=list(FP_KINDS);values=[sum(r['fp_'+k] for r in subset) for k in kinds]
            axes[row,col].bar(range(len(kinds)),values);axes[row,col].set_xticks(range(len(kinds)),kinds,rotation=25,ha='right')
            axes[row,col].set_title(f'{split} {ck}: FP counts summed over 3 seeds');axes[row,col].set_ylabel('FP count; repeated images')
            error_rows.extend(dict(split=split,checkpoint=ck,kind=k,count=v) for k,v in zip(kinds,values))
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(figdir/f'false_positive_breakdown.{ext}',dpi=180)
    plt.close(fig);p0.write_csv(figdir/'false_positive_breakdown.csv',error_rows)
    background_proxies(out,records_by_key,all_rows['prediction_rows'],data_root)
    p0.dump_json(out/'ANALYSIS_MANIFEST.json',dict(status='COMPLETE',evaluations=12,new_training_runs=0,test_accessed=False,
        inference_manifest_sha256=p0.sha256(out/'INFERENCE_MANIFEST.json'),analysis_sources={str(p.relative_to(p0.ROOT)):p0.sha256(p) for p in
            [Path(__file__),Path(__file__).with_name('error_breakdown.py'),Path(__file__).with_name('curve_diagnostics.py')]},
        protocol_boundaries=protocol))
    print('ANALYSIS COMPLETE; 12 evaluations; no training; test accessed false')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=p0.ROOT/'reports/sonar_operator_pool_p0_diagnosis')
    parser.add_argument('--data-root',type=Path,default=p0.resolve(p0.load_frozen_config()['dataset']['data_dir']))
    args=parser.parse_args();summarize(args.input,args.data_root)
