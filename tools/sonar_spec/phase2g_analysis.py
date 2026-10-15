"""Controlled A1/A2/A3 multi-seed comparison and preregistered prototype gate."""
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from formal_protocol import ROOT,dump
from phase2g_runner import REPORT,folder

A1='P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k5'
A2='P3_P4_P5_s1k5_s2k3_s4k3_s5k3_s6k5_s7k3'
METRICS=('mAP50_95','AP50','AP75','Recall')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect():
    with (ROOT/'reports/phase2f_multiseed/multiseed_results.csv').open(newline='',encoding='utf-8') as f:
        old=list(csv.DictReader(f))
    rows=[]
    for label,aid in [('A1',A1),('A2',A2)]:
        group=[r for r in old if r['architecture_id']==aid]
        if len(group)!=3:raise ValueError(f'Missing immutable {label} three-seed controls')
        for r in group:rows.append(dict(architecture=label,architecture_id=aid,seed=int(r['seed']),best_epoch=int(r['best_epoch']),
            mAP50_95=float(r['mAP50_95']),AP50=float(r['AP50']),AP75=float(r['AP75']),Precision=float(r['Precision']),
            Recall=float(r['Recall']),endpoint_mAP50_95=float(r['endpoint_mAP50_95']),
            peak_endpoint_gap=float(r['mAP50_95'])-float(r['endpoint_mAP50_95']),Params=int(r['Params']),MACs=int(r['MACs']),
            provenance='phase2f_immutable_control',test_evaluated=False))
    for seed in (42,43,44):
        r=json.loads((folder(seed)/'result.json').read_text(encoding='utf-8'))
        rows.append(dict(architecture='A3',architecture_id=r['architecture_id'],seed=seed,best_epoch=r['best_epoch'],
            **{k:r[k] for k in ('mAP50_95','AP50','AP75','Precision','Recall','endpoint_mAP50_95','peak_endpoint_gap','Params','MACs')},
            provenance='phase2g_independent_from_scratch',test_evaluated=False))
    rows.sort(key=lambda r:(r['architecture'],r['seed']));return rows


def aggregate(rows,label):
    g=[r for r in rows if r['architecture']==label];v=[r['mAP50_95'] for r in g]
    return dict(architecture=label,mean_mAP50_95=mean(v),sample_std_mAP50_95=stdev(v),min_mAP50_95=min(v),max_mAP50_95=max(v),
        endpoint_mean=mean(r['endpoint_mAP50_95'] for r in g),peak_endpoint_gap_mean=mean(r['peak_endpoint_gap'] for r in g),
        Params=g[0]['Params'],MACs=g[0]['MACs'])


def compare(rows,left,right,path):
    result=[]
    for seed in (42,43,44):
        a=next(r for r in rows if r['architecture']==left and r['seed']==seed);b=next(r for r in rows if r['architecture']==right and r['seed']==seed)
        result.append(dict(seed=seed,comparison=f'{left}-{right}',**{f'delta_{k}':a[k]-b[k] for k in METRICS},
            delta_endpoint_mAP50_95=a['endpoint_mAP50_95']-b['endpoint_mAP50_95'],
            delta_peak_endpoint_gap=a['peak_endpoint_gap']-b['peak_endpoint_gap']))
    summary=[]
    for kind,fn in [('mean',mean),('sample_std',stdev),('min',min),('max',max)]:
        summary.append(dict(seed=kind,comparison=f'{left}-{right}',**{k:fn([r[k] for r in result]) for k in result[0] if k.startswith('delta_')}))
    write_csv(path,result+summary);return result


def figures(rows,agg,k3,k5):
    out=REPORT/'figures';out.mkdir(exist_ok=True)
    labels=['A1','A2','A3']
    fig,ax=plt.subplots(figsize=(6,4));
    for i,label in enumerate(labels):
        g=[r for r in rows if r['architecture']==label];ax.scatter([i-.12,i,i+.12],[r['mAP50_95'] for r in g],label=label)
    ax.set_xticks(range(3),labels);ax.set_ylabel('best valid mAP50:95');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'a1_a2_a3_seed_map.png',dpi=180);fig.savefig(out/'a1_a2_a3_seed_map.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.errorbar(range(3),[r['mean_mAP50_95'] for r in agg],yerr=[r['sample_std_mAP50_95'] for r in agg],fmt='o',capsize=4)
    ax.set_xticks(range(3),labels);ax.set_ylabel('mean +/- sample std');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'mean_std.png',dpi=180);fig.savefig(out/'mean_std.svg');plt.close(fig)
    for name,data,title in [('a3_minus_a2',k3,'A3 - A2'),('a3_minus_a1',k5,'A3 - A1')]:
        fig,ax=plt.subplots(figsize=(5,4));ax.bar([str(r['seed']) for r in data],[r['delta_mAP50_95'] for r in data]);ax.axhline(0,color='black',lw=.8);ax.set_ylabel(title+' mAP50:95');ax.set_xlabel('seed');fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));
    for r in rows:ax.scatter(r['endpoint_mAP50_95'],r['mAP50_95'],label=f"{r['architecture']}-{r['seed']}")
    lo=min(min(r['endpoint_mAP50_95'],r['mAP50_95']) for r in rows);hi=max(max(r['endpoint_mAP50_95'],r['mAP50_95']) for r in rows);ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.set_xlabel('epoch80 endpoint');ax.set_ylabel('best');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'best_vs_endpoint.png',dpi=180);fig.savefig(out/'best_vs_endpoint.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3));ax.axis('off');ax.text(.03,.55,'Train:  DW3+BN  ─┐\n                    ├─ Add → ReLU6\n        DW5+BN  ─┘',fontsize=14,family='monospace',va='center');ax.text(.58,.55,'Deploy: DW5 fused → ReLU6',fontsize=14,family='monospace',va='center');ax.annotate('',xy=(.56,.55),xytext=(.43,.55),arrowprops={'arrowstyle':'->','lw':2});fig.tight_layout();fig.savefig(out/'sonarop_train_deploy_schematic.png',dpi=180);fig.savefig(out/'sonarop_train_deploy_schematic.svg');plt.close(fig)


def main():
    rows=collect();write_csv(REPORT/'multiseed_sonarop_results.csv',rows)
    agg=[aggregate(rows,x) for x in ('A1','A2','A3')];k3=compare(rows,'A3','A2',REPORT/'sonarop_vs_k3.csv');k5=compare(rows,'A3','A1',REPORT/'sonarop_vs_k5.csv')
    a={r['architecture']:r for r in agg};d3=[r['delta_mAP50_95'] for r in k3];d5=[r['delta_mAP50_95'] for r in k5]
    severe=(a['A3']['peak_endpoint_gap_mean']>max(a['A1']['peak_endpoint_gap_mean'],a['A2']['peak_endpoint_gap_mean'])+.005 or
        (a['A3']['endpoint_mean']<min(a['A1']['endpoint_mean'],a['A2']['endpoint_mean']) and a['A3']['mean_mAP50_95']>max(a['A1']['mean_mAP50_95'],a['A2']['mean_mAP50_95'])))
    strong=(a['A3']['mean_mAP50_95']>a['A1']['mean_mAP50_95'] and a['A3']['mean_mAP50_95']>a['A2']['mean_mAP50_95'] and
        sum(x>0 for x in d3)>=2 and sum(x>0 for x in d5)>=2 and not severe)
    partial=(not strong and a['A3']['mean_mAP50_95']>min(a['A1']['mean_mAP50_95'],a['A2']['mean_mAP50_95']) and not severe)
    gate='STRONG_SUPPORT' if strong else 'PARTIAL_SUPPORT' if partial else 'NOT_SUPPORTED'
    overall={'STRONG_SUPPORT':'PROVISIONAL_YES','PARTIAL_SUPPORT':'CONDITIONAL','NOT_SUPPORTED':'NO_FOR_V0.2_P0'}[gate]
    dump(REPORT/'prototype_gate.json',dict(prototype_gate=gate,overall_gate=overall,severe_endpoint_collapse=severe,
        aggregates=agg,a3_vs_k3=d3,a3_vs_k5=d5,test_evaluated=False))
    table='\n'.join(f"| {r['architecture']} | {r['mean_mAP50_95']:.6f} | {r['sample_std_mAP50_95']:.6f} | {r['endpoint_mean']:.6f} | {r['peak_endpoint_gap_mean']:.6f} |" for r in agg)
    fmt=lambda xs:', '.join(f'{x:+.6f}' for x in xs)
    (REPORT/'SONAROP_CONTROLLED_COMPARISON.md').write_text(f'''# SonarOp Controlled Comparison

| Model | Mean mAP50:95 | Sample std | Endpoint mean | Mean peak-end gap |
|---|---:|---:|---:|---:|
{table}

## A3 versus A2 DW3

Seed deltas: {fmt(d3)}. Mean {mean(d3):+.6f}, sample std {stdev(d3):.6f}, min {min(d3):+.6f}, max {max(d3):+.6f}.

## A3 versus A1 DW5

Seed deltas: {fmt(d5)}. Mean {mean(d5):+.6f}, sample std {stdev(d5):.6f}, min {min(d5):+.6f}, max {max(d5):+.6f}.

The comparison uses the frozen Phase 2F A1/A2 controls and three independent A3 runs. Test was not evaluated.
''',encoding='utf-8')
    (REPORT/'SONAROP_PROTOTYPE_GATE.md').write_text(f'''# SonarOp Prototype Gate

Prototype gate: **{gate}**

Overall gate: **{overall}**

Severe endpoint collapse: `{str(severe).lower()}`.

This is controlled prototype evidence only. V0.1 and the Phase 2F candidate set remain unchanged. Test evaluated: false.
''',encoding='utf-8')
    figures(rows,agg,k3,k5)


if __name__=='__main__':main()
