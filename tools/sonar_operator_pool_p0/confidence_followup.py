"""Diagnose retained-score ordering without training, tuning, or test reads."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('cylider', 'manta')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ious(boxes, gt):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt, dtype=float).reshape(1, 4)
    inter = np.maximum(0, np.minimum(boxes[:, 2:], gt[:, 2:]) - np.maximum(boxes[:, :2], gt[:, :2])).prod(1)
    union = (boxes[:, 2:] - boxes[:, :2]).clip(0).prod(1) + (gt[:, 2:] - gt[:, :2]).clip(0).prod(1) - inter
    return inter / np.maximum(union, 1e-12)


def probability_bounds(score):
    """score=obj*max(2-class softmax); exact algebra, not raw-logit inference."""
    score = float(score)
    if not 0 <= score <= 1:
        raise ValueError('score outside [0,1]')
    return score, min(1., 2 * score)


def analyze_record(record, common):
    if common['split'] not in ('train', 'valid'):
        raise ValueError('Only train/valid permitted')
    image = record['image'].replace('\\', '/')
    if image.split('/')[0] != common['split']:
        raise ValueError('Image split mismatch')
    target = record['target']
    # The audited dataset has exactly one GT per image. Fail if that changes.
    if len(target['boxes']) != 1:
        raise ValueError('Single-GT diagnostic assumption changed')
    gt_class = int(target['labels'][0])
    pred = record['prediction']
    scores = np.asarray(pred['scores'], float)
    labels = np.asarray(pred['labels'], int)
    overlap = ious(pred['boxes'], target['boxes'])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid prediction scores')
    if len(scores) != len(labels) or len(scores) != len(overlap):
        raise ValueError('Inconsistent prediction lengths')
    order = np.argsort(-scores, kind='stable')
    is_tp50 = np.zeros(len(scores), bool)
    is_tp75 = np.zeros(len(scores), bool)
    for threshold, flag in ((.5, is_tp50), (.75, is_tp75)):
        hits = order[(labels[order] == gt_class) & (overlap[order] >= threshold)]
        if len(hits):
            flag[hits[0]] = True
    rows = []
    for j in range(len(scores)):
        lo, hi = probability_bounds(scores[j])
        rows.append(dict(common, image=image, prediction_index=j, class_name=CLASSES[labels[j]],
                         gt_class_name=CLASSES[gt_class], score=float(scores[j]), gt_iou=float(overlap[j]),
                         tp50=bool(is_tp50[j]), tp75=bool(is_tp75[j]),
                         background=bool(overlap[j] < .1), objectness_lower_bound=lo, objectness_upper_bound=hi))
    gtrows = []
    for threshold in (.5, .75):
        good = np.where((labels == gt_class) & (overlap >= threshold))[0]
        best = min(good, key=lambda j: (-scores[j], j)) if len(good) else None
        background = (overlap < .1)
        threshold_score = float(scores[best]) if best is not None else None
        same_bg = background & (labels == gt_class)
        bgmax = float(scores[same_bg].max()) if same_bg.any() else None
        gtrows.append(dict(common, image=image, class_name=CLASSES[gt_class], iou_threshold=threshold,
                           correct_retained_exists=best is not None, correct_max_score=threshold_score,
                           correct_score_below_05=best is not None and threshold_score < .5,
                           objectness_definitely_below_05=best is not None and threshold_score < .25,
                           same_class_background_max_score=bgmax,
                           background_outranking_correct=int((same_bg & (scores > threshold_score)).sum()) if best is not None else None,
                           correct_minus_background_max=threshold_score-bgmax if best is not None and bgmax is not None else None,
                           correct_global_image_rank=int((scores > threshold_score).sum()+1) if best is not None else None))
    return rows, gtrows


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='D_confidence_followup_v1', seed=42, splits=['train','valid'],
                    checkpoints=['best','epoch80'], seeds=[42,43,44],
                    fixed_iou=[.5,.75], fixed_score_descriptions=[.5,.75,.9],
                    score_bins=list(np.linspace(0,1,11)),
                    raw_logits_available=False, threshold_selection=False, new_training_runs=0,
                    test_images_labels_read=False, test_evaluated=False,
                    selection_caveat='Retained top500/NMS100 only; no pre-NMS coverage inference',
                    source_manifest_sha256=sha(source/'FINAL_DIAGNOSTIC_MANIFEST.json'))
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    predrows=[]; gtrows=[]; bindings={}
    for split in ('train','valid'):
        for seed in (42,43,44):
            for ck in ('best','epoch80'):
                tag=f'{split}_D_seed{seed}_{ck}'
                path=source/'predictions'/f'{tag}.jsonl.gz'
                metric=json.loads((source/'metrics'/f'{tag}.json').read_text(encoding='utf-8'))
                if sha(path)!=metric['predictions_sha256']:
                    raise ValueError('Prediction hash mismatch')
                bindings[tag]=sha(path)
                with gzip.open(path,'rt',encoding='utf-8') as handle:
                    records=[json.loads(line) for line in handle if line.strip()]
                if len(records)!=(330 if split=='train' else 99):
                    raise ValueError('Unexpected split size')
                for record in records:
                    p,g=analyze_record(record,dict(split=split,seed=seed,checkpoint=ck))
                    predrows.extend(p);gtrows.extend(g)
    p=pd.DataFrame(predrows);g=pd.DataFrame(gtrows)
    p.to_csv(out/'retained_score_rows.csv.gz',index=False,compression='gzip')
    g.to_csv(out/'gt_score_order.csv',index=False)
    keys=['split','seed','checkpoint','class_name']
    official=pd.read_csv(source/'score_ranking.csv')
    summary=[];bins=[]
    for key,rows in p.groupby(keys):
        base=dict(zip(keys,key))
        refs=official
        for k,v in base.items():refs=refs[refs[k]==v]
        for iou,col in ((.5,'tp50'),(.75,'tp75')):
            ref=refs[refs.iou==iou]
            if len(ref)!=1 or int(rows[col].sum())!=int(ref.iloc[0].total_tp):
                raise ValueError('Frozen evaluator retained TP-count mismatch')
        tp=rows[rows.tp50];bg=rows[rows.background]
        summary.append(dict(base,retained=len(rows),tp50=len(tp),background=len(bg),
                            tp_score_median=tp.score.median() if len(tp) else None,
                            background_score_median=bg.score.median() if len(bg) else None,
                            **{f'background_score_ge_{int(t*100)}':int((bg.score>=t).sum()) for t in (.5,.75,.9)},
                            operating_precision=float(rows[rows.score>=.5].tp50.mean()) if (rows.score>=.5).any() else None))
        for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            b=rows[(rows.score>=low)&((rows.score<high) if high<1 else (rows.score<=high))]
            bins.append(dict(base,bin_low=low,bin_high=high,count=len(b),mean_score=b.score.mean() if len(b) else None,
                             fraction_matched_iou50=b.tp50.mean() if len(b) else None,background_count=int(b.background.sum())))
    s=pd.DataFrame(summary);s.to_csv(out/'score_summary.csv',index=False)
    b=pd.DataFrame(bins);b.to_csv(out/'score_bins.csv',index=False)
    gr=[]
    for key,rows in g.groupby(keys+['iou_threshold']):
        exists=rows[rows.correct_retained_exists]
        gr.append(dict(zip(keys+['iou_threshold'],key),gt_count=len(rows),correct_retained=len(exists),
                       low_score_correct=int(exists.correct_score_below_05.sum()),
                       definite_low_objectness=int(exists.objectness_definitely_below_05.sum()),
                       correct_score_median=exists.correct_max_score.median() if len(exists) else None,
                       images_background_outranks=int((exists.background_outranking_correct>0).sum()),
                       median_background_outranking=exists.background_outranking_correct.median() if len(exists) else None,
                       median_correct_minus_background=exists.correct_minus_background_max.median() if len(exists) else None))
    gr=pd.DataFrame(gr);gr.to_csv(out/'gt_score_summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,split in enumerate(('train','valid')):
        for col,cl in enumerate(CLASSES):
            ax=axes[row,col]
            for seed in (42,43,44):
                data=b[(b.split==split)&(b.class_name==cl)&(b.checkpoint=='epoch80')&(b.seed==seed)&(b['count']>0)]
                ax.plot(data.mean_score,data.fraction_matched_iou50,marker='o',label=f'seed{seed}')
            ax.plot([0,1],[0,1],color='grey',linestyle=':',label='score=value reference')
            ax.set(xlim=(0,1),ylim=(0,1),title=f'{split} {cl}: retained detections',xlabel='Mean score in fixed bin',ylabel='Fraction one-to-one TP at IoU .5')
            ax.legend(fontsize=8)
    fig.suptitle('Selected detections only; descriptive reliability, no calibration fit')
    fig.tight_layout();fig.savefig(out/'score_reliability.png',dpi=160);fig.savefig(out/'score_reliability.svg');plt.close(fig)
    notes=['# Confidence and score-order follow-up','',
           'No training, score selection, AP protocol changes or test evaluation. All 12 cached train/valid prediction hashes verified; 48 class/IoU retained TP counts agree with the frozen AP evaluator.',
           '', '## Frozen score semantics',
           '`score = sigmoid(objectness_logit) * max(softmax(class_logits))` with two classes. Classification CE is applied only on assigned positives. Class confidence is conditional on a target, not a background-class probability.',
           'Given emitted score s, objectness is in [s, min(2s,1)]. Therefore a background FP with s>=0.9 necessarily has objectness>=0.9. This is exact algebra; the two individual probabilities cannot be reconstructed uniquely from their product.',
           'A correct retained box with s<0.25 necessarily has objectness<0.5. For 0.25<=s<0.5, objectness/class uncertainty cannot be separated from cached scores alone. No raw objectness claim is made for this ambiguous interval.',
           '', '## Per-seed endpoint observations',
           'Fixed score counts below are descriptions, not tuned operating thresholds. TP/FP here uses all retained scores for one-to-one matching.', '']
    for split in ('train','valid'):
        notes.append(f'### {split}')
        for seed in (42,43,44):
            r=s[(s.split==split)&(s.seed==seed)&(s.checkpoint=='epoch80')]
            t=gr[(gr.split==split)&(gr.seed==seed)&(gr.checkpoint=='epoch80')&(gr.iou_threshold==.5)]
            notes.append(f'- seed{seed}: background score>=.5/.75/.9 = {r.background_score_ge_50.sum()}/{r.background_score_ge_75.sum()}/{r.background_score_ge_90.sum()}; correct retained GT {t.correct_retained.sum()}, of which {t.images_background_outranks.sum()} have a higher-scored same-class background box. Correct low-score boxes {t.low_score_correct.sum()}, with definitely objectness<.5: {t.definite_low_objectness.sum()}.')
    notes += ['', '## Limits and next decision',
              'Score bins describe a decoder-selected population, not calibration over all 25,200 raw anchors. NMS duplicates and one-to-one matching affect fractions. No crop-level significance test is valid as an independent sample test. Background is relative to annotations; missing annotations remain unresolved.',
              'This supports examining objectness/background discrimination and ordering before adding an operator. It does not show the cause is loss normalization, prove a coding bug, or warrant tuning the score threshold. Exact pre-topK coverage and objectness/class decomposition remain unmeasured because raw outputs were not saved; the existing remote checkpoint connection was unavailable in this continuation.',
              'AP and ranking can decline even with more correct boxes when false positives precede them. See the earlier score_ranking.csv for exact best-versus-endpoint AP reproduction.']
    (out/'CONFIDENCE_AUDIT.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    manifest=dict(status='COMPLETE_FOR_RETAINED_PREDICTIONS',raw_logit_decomposition='NOT_AVAILABLE',
                  source_sha256=sha(Path(__file__)),predictions_sha256=bindings,retained_rows=len(p),gt_rows=len(g),
                  tp_count_parity_checks=48,new_training_runs=0,test_evaluated=False,test_images_labels_read=False)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PASS',len(p),'retained predictions; 48 TP-count checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'reports/sonar_operator_pool_p0_diagnosis')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/sonar_operator_pool_p0_followup/confidence')
    args=parser.parse_args();run(args.input,args.output)
