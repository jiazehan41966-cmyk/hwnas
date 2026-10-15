"""Phase2C software-only structure and witness ranking analysis."""
import argparse
import json
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau,pearsonr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def correlations(x,y,bootstrap_samples=10000,seed=42):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)!=12 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Need12 finite paired witnesses')
    functions={'spearman':lambda a,b:spearmanr(a,b).statistic,'kendall_tau_b':lambda a,b:kendalltau(a,b,variant='b').statistic,
               'pearson_supplementary':lambda a,b:pearsonr(a,b).statistic}
    rng=np.random.default_rng(seed);samples={k:[] for k in functions}
    for _ in range(bootstrap_samples):
        ids=rng.integers(0,len(x),len(x));a=x[ids];b=y[ids]
        if np.ptp(a)==0 or np.ptp(b)==0:continue
        for name,fn in functions.items():samples[name].append(float(fn(a,b)))
    result={}
    for name,fn in functions.items():
        result[name]=dict(value=float(fn(x,y)) if np.ptp(x)>0 and np.ptp(y)>0 else None,
            percentile_ci95=np.quantile(samples[name],[.025,.975]).tolist() if samples[name] else None,
            valid_bootstrap_replicates=len(samples[name]))
    return result


def figure(out,name,data,draw):
    data.to_csv(out/(name+'.csv'),index=False)
    fig,ax=plt.subplots(figsize=(7,4));draw(ax);fig.tight_layout()
    for extension in ('png','svg'):fig.savefig(out/(name+'.'+extension),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[2]/'reports/phase2c_supernet')
    p.add_argument('--standalone-root',type=Path);args=p.parse_args();out=args.directory
    ranking=out/'subnet_inherited_ranking.csv'
    if not ranking.exists():raise FileNotFoundError('No real144-architecture inherited results; analysis cannot run')
    df=pd.read_csv(ranking)
    if len(df)!=144 or df.architecture_id.nunique()!=144:raise ValueError('Require144 unique architecture evaluations')
    provenance=json.loads((out/'ranking_provenance.json').read_text())
    if provenance['status']!='COMPLETE':raise ValueError('Ranking evaluation incomplete')
    bindings={'ranking_sha256':ranking,'witness_manifest_sha256':out/'ranking_witness_manifest.json',
        'shared_bn_sha256':out/'witness_shared_bn.csv','calibration_manifest_sha256':out/'calibration_manifest.json'}
    for key,path in bindings.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance[key]:raise ValueError(f'Ranking provenance drift: {key}')
    figs=out/'figures';figs.mkdir(exist_ok=True)
    df=df.sort_values(['mAP50_95','architecture_id'],ascending=[False,True]);df['rank']=np.arange(1,145)
    group=df.groupby('output_combination').mAP50_95.agg(['count','min','median','max'])
    group['P25']=df.groupby('output_combination').mAP50_95.quantile(.25);group['P75']=df.groupby('output_combination').mAP50_95.quantile(.75)
    group.to_csv(out/'combination_statistics.csv')
    pref=[]
    for fraction in (.1,.25):
        top=df.head(math.ceil(len(df)*fraction))
        for stage in (1,2,4,5,6,7):
            active=top[f'stage{stage}'].dropna()
            pref.append(dict(top_fraction=fraction,stage=stage,active_count=len(active),p_k3=float((active==3).mean()),p_k5=float((active==5).mean())))
    prefs=pd.DataFrame(pref)
    figure(figs,'inherited_map_distribution',df[['architecture_id','mAP50_95']],lambda ax:ax.hist(df.mAP50_95,bins=20))
    combos=['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5']
    def boxplot(ax):
        ax.boxplot([df[df.output_combination==c].mAP50_95 for c in combos]);ax.set_xticks([1,2,3],combos);ax.set_ylabel('Inherited mAP50:95')
    figure(figs,'output_combination_boxplot',df[['output_combination','mAP50_95']],boxplot)
    def kernelplot(ax):
        for f in (.1,.25):
            s=prefs[prefs.top_fraction==f];ax.plot(s.stage,s.p_k5,marker='o',label=f'Top {f:.0%}')
        ax.set(xlabel='Active MBConv stage',ylabel='P(k5)',ylim=(0,1));ax.legend()
    figure(figs,'kernel_preference_by_stage',prefs,kernelplot)
    df['compute_pareto']=[not ((df.MACs<=r.MACs)&(df.mAP50_95>=r.mAP50_95)&((df.MACs<r.MACs)|(df.mAP50_95>r.mAP50_95))).any() for _,r in df.iterrows()]
    def pareto(ax):
        for c in combos:
            s=df[df.output_combination==c];ax.scatter(s.MACs/1e9,s.mAP50_95,label=c,s=14)
        ax.set(xlabel='Conv/Linear MACs (G), not latency',ylabel='Inherited mAP50:95',title='Accuracy–Compute diagnostic Pareto');ax.legend()
    figure(figs,'map_vs_macs_diagnostic_pareto',df[['architecture_id','MACs','mAP50_95','compute_pareto']],pareto)
    paired=df[df.output_combination=='P3_P4_P5'].merge(df[df.output_combination=='P2_P3_P4_P5'],on=[f'stage{s}' for s in (1,2,4,5,6,7)],suffixes=('_P345','_P2345'))
    paired['p2_increment_map']=paired.mAP50_95_P2345-paired.mAP50_95_P345;paired.to_csv(out/'paired_p2_increment.csv',index=False)
    top_counts={str(f):df.head(math.ceil(144*f)).output_combination.value_counts().to_dict() for f in (.1,.25)}
    (out/'SEARCH_BEHAVIOR_ANALYSIS.md').write_text('# Search Behavior Analysis\n\n'+group.to_string()+'\n\nTop composition: '+json.dumps(top_counts)+
        f'\n\n64 matched-kernel +P2 comparisons: median delta={paired.p2_increment_map.median():.6f}. '
        'P2/P3/P4 simultaneously removes P5 and two stages; it is not a pure P2 effect. '
        'Preference counts are conditional on active stages. A single checkpoint/seed is not stable operator-design evidence.\n',encoding='utf-8')
    if args.standalone_root is None:return
    manifest=json.loads((out/'ranking_witness_manifest.json').read_text());shared=pd.read_csv(out/'witness_shared_bn.csv').set_index('architecture_id')
    rows=[]
    for index,w in enumerate(manifest['witnesses']):
        run=args.standalone_root/f'witness_{index:02d}'
        state=json.loads((run/'status.json').read_text())
        if state['status']!='COMPLETE' or state['epochs']!=80:raise ValueError(f'Incomplete standalone witness{index}')
        binding=state['binding']
        if binding['mode']!='standalone' or binding['witness_index']!=index or binding['protocol_sha256']!=manifest['protocol_sha256'] or binding['witness_sha256']!=hashlib.sha256((out/'ranking_witness_manifest.json').read_bytes()).hexdigest():
            raise ValueError(f'Standalone witness binding mismatch: {index}')
        metrics=json.loads((run/'best_valid_metrics.json').read_text());inherited=df.set_index('architecture_id').loc[w['architecture_id']]
        rows.append(dict(architecture_id=w['architecture_id'],output_combination=w['architecture']['output_combination'],
            **{f'stage{s}':w['architecture']['kernels'].get(f'stage{s}') for s in (1,2,4,5,6,7)},
            inherited_mAP50_95=float(inherited.mAP50_95),shared_bn_mAP50_95=float(shared.loc[w['architecture_id']].mAP50_95),
            standalone_mAP50_95=metrics['mAP50_95'],standalone_AP50=metrics['AP50'],standalone_AP75=metrics['AP75'],Precision=metrics['Precision'],Recall=metrics['Recall']))
    w=pd.DataFrame(rows);w['inherited_rank']=rankdata(-w.inherited_mAP50_95,method='average');w['standalone_rank']=rankdata(-w.standalone_mAP50_95,method='average');w['rank_error']=w.inherited_rank-w.standalone_rank
    w.to_csv(out/'standalone_witness_results.csv',index=False)
    calibrated=correlations(w.inherited_mAP50_95,w.standalone_mAP50_95);shared_results=correlations(w.shared_bn_mAP50_95,w.standalone_mAP50_95)
    overlaps={}
    for k in (3,5):
        top=lambda column:set(w.sort_values([column,'architecture_id'],ascending=[False,True]).head(k).architecture_id)
        overlaps[f'top{k}']=dict(count=len(top('inherited_mAP50_95')&top('standalone_mAP50_95')),denominator=k)
    def scatter(ax,x,y):
        ax.scatter(w[x],w[y]);ax.set(xlabel=x,ylabel=y)
        for i,row in w.iterrows():ax.annotate(str(i), (row[x],row[y]))
    figure(figs,'inherited_vs_standalone',w,lambda ax:scatter(ax,'inherited_mAP50_95','standalone_mAP50_95'))
    figure(figs,'rank_scatter',w,lambda ax:scatter(ax,'inherited_rank','standalone_rank'))
    result=dict(calibrated_bn=calibrated,shared_bn=shared_results,overlaps=overlaps,macro_rank_error=w.groupby('output_combination').rank_error.mean().to_dict(),
        kernel_rank_error={f'stage{s}':w.groupby(f'stage{s}').rank_error.agg(['count','mean','median']).reset_index().to_dict('records') for s in (1,2,4,5,6,7)},
        gate='REQUIRES_INDEPENDENT_INTERPRETATION_NO_AUTOMATIC_SELECTION',ci_scope='paired percentile bootstrap conditional on purposive12-witness set; not population144 CI')
    (out/'reliability_statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'RANKING_RELIABILITY.md').write_text('# Ranking Reliability\n\n'+json.dumps(result,indent=2)+'\n\n'+w.to_string(index=False)+
        '\n\nNo universal NAS correlation cutoff is assumed. Review positive correlation/CI, top-set inversions and macro-path rank errors before any final selection. '
        'If scores tie/variance vanishes, correlations are undefined and ranking evidence is insufficient. '
        'Near-zero/negative correlations or severe systematic inversion require STOP and bias diagnosis. '
        'This stratified witness design and one seed cannot establish a stable SonarOp design preference.\n',encoding='utf-8')


if __name__=='__main__':main()
