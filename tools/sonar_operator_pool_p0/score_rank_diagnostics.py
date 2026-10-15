"""Inspect frozen AP score ordering; no new inference or threshold selection."""
import argparse
import gzip
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as p0
from hwnas_fpga.training.detection_coco_eval import _stack, _evaluate_class_area, _interpolate_101


def run(out):
    torch.set_num_threads(2)
    summaries=[];curves=[]
    for path in sorted((out/'predictions').glob('*.jsonl.gz')):
        metric=json.loads((out/'metrics'/(path.name.replace('.jsonl.gz','.json'))).read_text())
        if metric['split'] not in ('train','valid'):raise ValueError('Forbidden split')
        if p0.sha256(path)!=metric['predictions_sha256']:raise ValueError('Predictions changed')
        with gzip.open(path,'rt') as f:records=[json.loads(l) for l in f]
        preds=_stack([r['prediction'] for r in records],with_scores=True)
        targets=_stack([r['target'] for r in records],with_scores=False)
        for class_id,name in enumerate(('cylider','manta')):
            for iou in (.5,.75):
                scores,tp,fp,n=_evaluate_class_area(preds,targets,class_index=class_id,iou_threshold=iou,area_range=(0.,float('inf')),max_detections=100)
                order=scores.argsort(descending=True);scores=scores[order];tp=tp[order];fp=fp[order]
                cumtp=tp.cumsum(0);cumfp=fp.cumsum(0);rec=cumtp/n;prec=cumtp/(cumtp+cumfp).clamp_min(1e-12)
                ap=_interpolate_101(rec,prec)
                original=metric['metrics']['coco_details']['per_class_ap']['all'][name][f'{iou:.2f}']
                if abs(ap-original)>1e-7:raise ValueError('AP helper reproduction mismatch')
                ranks=torch.where(tp>0)[0]+1
                common=dict(split=metric['split'],seed=metric['seed'],checkpoint=metric['checkpoint'],class_name=name,iou=iou)
                summaries.append(dict(common,gt_count=n,predictions=len(tp),total_tp=int(tp.sum()),AP=ap,
                    first_tp_rank=int(ranks[0]) if len(ranks) else None,
                    median_tp_rank=float(ranks.float().median()) if len(ranks) else None,
                    tp_in_first100=int(tp[:100].sum()),tp_in_first500=int(tp[:500].sum()),
                    fp_before_first_tp=int(ranks[0]-1) if len(ranks) else len(tp)))
                # Exact curve turning points, plus fixed ranks; avoids redundant flat rows.
                wanted=set(torch.where(tp>0)[0].tolist())|{0,len(tp)-1}
                wanted|={i for i in (9,49,99,499,999) if i<len(tp)}
                for i in sorted(wanted):
                    if i>=0:curves.append(dict(common,rank=i+1,score=float(scores[i]),recall=float(rec[i]),precision=float(prec[i]),tp_cumulative=int(cumtp[i]),fp_cumulative=int(cumfp[i])))
    p0.write_csv(out/'score_ranking.csv',summaries)
    p0.write_csv(out/'figures/precision_recall_curves.csv',curves)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,name in enumerate(('cylider','manta')):
        for col,seed in enumerate((42,43,44)):
            ax=axes[row,col]
            for ck in ('best','epoch80'):
                r=[r for r in curves if r['split']=='valid' and r['seed']==seed and r['checkpoint']==ck and r['class_name']==name and r['iou']==.75]
                ax.step([r['recall'] for r in r],[r['precision'] for r in r],where='post',label=ck)
            ax.set_title(f'valid {name} seed{seed}, IoU .75');ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'figures/precision_recall_curves.{ext}',dpi=180)
    plt.close(fig)
    p0.dump_json(out/'SCORE_RANKING_MANIFEST.json',dict(source_sha256=p0.sha256(Path(__file__)),rows=len(summaries),
        official_per_class_AP_reproduced=True,threshold_tuning=False,new_training_runs=0,test_accessed=False,
        note='fixed IoU50/75 score-order inspection using frozen evaluator helpers'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=p0.ROOT/'reports/sonar_operator_pool_p0_diagnosis')
    args=parser.parse_args();run(args.input)
