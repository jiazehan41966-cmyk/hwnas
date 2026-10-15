"""Independent Phase2D statistics; never select a replacement witness or protocol."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau,pearsonr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def association(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return dict(spearman=None,kendall=None,pearson=None)
    return dict(spearman=float(spearmanr(x,y).statistic),kendall=float(kendalltau(x,y,variant='b').statistic),pearson=float(pearsonr(x,y).statistic))


def bootstrap(x,y,repeats=10000):
    x=np.asarray(x,float);y=np.asarray(y,float);rng=np.random.default_rng(42);values={k:[] for k in association(x,y)}
    for _ in range(repeats):
        indices=rng.integers(len(x),size=len(x))
        for k,v in association(x[indices],y[indices]).items():
            if v is not None:values[k].append(v)
    return {k:dict(estimate=v,ci95=np.percentile(values[k],[2.5,97.5]).tolist() if values[k] else None,
                   valid_replicates=len(values[k])) for k,v in association(x,y).items()}


def distribution(values):
    x=np.sort(np.asarray(values,float));gaps=np.diff(x);pairgaps=np.array([abs(a-b) for a,b in itertools.combinations(x,2)])
    row=dict(n=len(x),min=float(x.min()),max=float(x.max()),range=float(np.ptp(x)),mean=float(x.mean()),
        std_sample=float(x.std(ddof=1)),median=float(np.median(x)),P25=float(np.quantile(x,.25)),P75=float(np.quantile(x,.75)),
        min_adjacent_gap=float(gaps.min()),exact_tie_pairs=int((pairgaps==0).sum()))
    for eps in (.0001,.0005,.001):
        row[f'near_tie_pairs_le_{eps}']=int((pairgaps<=eps).sum())
        row[f'adjacent_gaps_le_{eps}']=int((gaps<=eps).sum())
    return row


def figure(out,name,data,draw):
    data.to_csv(out/(name+'.csv'),index=False);fig,ax=plt.subplots(figsize=(7,4.5));draw(ax);fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(out/(name+'.'+suffix),dpi=170)
    plt.close(fig)


def verified_rows(c,inputs):
    provenance=json.loads((c/'ranking_provenance.json').read_text());manifest=json.loads((c/'ranking_witness_manifest.json').read_text())
    for k,f in [('ranking_sha256','subnet_inherited_ranking.csv'),('witness_manifest_sha256','ranking_witness_manifest.json'),
                ('shared_bn_sha256','witness_shared_bn.csv')]:assert digest(c/f)==provenance[k],f
    inherited=pd.read_csv(c/'subnet_inherited_ranking.csv');reported=pd.read_csv(c/'standalone_witness_results.csv')
    shared=pd.read_csv(c/'witness_shared_bn.csv').set_index('architecture_id');expected=[r['architecture_id'] for r in manifest['witnesses']]
    assert len(inherited)==144 and inherited.architecture_id.nunique()==144
    assert len(expected)==len(set(expected))==len(reported)==reported.architecture_id.nunique()==12
    assert set(expected)==set(reported.architecture_id)==set(shared.index)
    rows=[];bindings=[]
    for index,item in enumerate(manifest['witnesses']):
        folder=inputs/'results'/f'witness_{index:02d}';status=json.loads((folder/'status.json').read_text());b=status['binding']
        assert status['epochs']==80 and status['status']=='COMPLETE' and status['test_evaluated'] is False
        assert b['witness_index']==index and b['witness_sha256']==digest(c/'ranking_witness_manifest.json')
        assert b['protocol_sha256']==manifest['protocol_sha256'] and status['initialization']=='from_scratch_seed42'
        assert b['source_sha256']==provenance['source_sha256']
        metrics=json.loads((folder/'best_valid_metrics.json').read_text());a=item['architecture'];aid=item['architecture_id']
        computed=a['output_combination']+'_'+'_'.join(f's{s[5:]}k{k}' for s,k in sorted(a['kernels'].items()))
        assert computed==aid
        source=inherited.set_index('architecture_id').loc[aid];old=reported.set_index('architecture_id').loc[aid]
        for s in (1,2,4,5,6,7):
            v=a['kernels'].get(f'stage{s}')
            assert (pd.isna(source[f'stage{s}']) if v is None else source[f'stage{s}']==v)
        assert abs(old.standalone_mAP50_95-metrics['mAP50_95'])<1e-14
        assert abs(old.inherited_mAP50_95-source.mAP50_95)<1e-14
        row=source.to_dict();row.update(architecture_id=aid,inherited=source.mAP50_95,standalone=metrics['mAP50_95'],shared_bn=shared.loc[aid,'mAP50_95'])
        row['k5_count']=sum(k==5 for k in a['kernels'].values())
        row['early_k5_count']=sum(a['kernels'].get(f'stage{s}')==5 for s in (1,2))
        row['middle_k5_count']=sum(a['kernels'].get(f'stage{s}')==5 for s in (4,5))
        row['deep_k5_count']=sum(a['kernels'].get(f'stage{s}')==5 for s in (6,7))
        row['heads']=len(a['output_combination'].split('_'));row['P2']=int('P2' in a['output_combination']);row['P5']=int('P5' in a['output_combination'])
        rows.append(row);bindings.append(dict(index=index,architecture_id=aid,status_sha256=digest(folder/'status.json'),metrics_sha256=digest(folder/'best_valid_metrics.json')))
    df=pd.DataFrame(rows);df['inherited_rank']=rankdata(-df.inherited);df['standalone_rank']=rankdata(-df.standalone)
    # Positive means inherited ranked it worse (underestimated), negative means overestimated.
    df['rank_error']=df.inherited_rank-df.standalone_rank;df['score_bias']=df.inherited-df.standalone
    return df,inherited,bindings


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path(__file__).resolve().parents[2]
    p.add_argument('--phase2c',type=Path,default=root/'reports/phase2c_supernet')
    p.add_argument('--output',type=Path,default=root/'reports/phase2d_ranking_bias');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    df,all144,bindings=verified_rows(args.phase2c,out/'input_snapshot');df.to_csv(out/'verified_witness_scores.csv',index=False)
    save(out/'identity_verification.json',dict(status='PASS',n=12,witnesses=bindings,selection_changed=False))
    corrs=bootstrap(df.inherited,df.standalone);shared=bootstrap(df.shared_bn,df.standalone)
    tops={}
    for k in (3,5):
        a=set(df.sort_values(['inherited','architecture_id'],ascending=[False,True]).head(k).architecture_id)
        b=set(df.sort_values(['standalone','architecture_id'],ascending=[False,True]).head(k).architecture_id)
        tops[str(k)]=dict(overlap=len(a&b),k=k,ids=sorted(a&b))
    save(out/'recheck_statistics.json',dict(n=12,calibrated=corrs,shared_bn=shared,topk=tops,
        bootstrap='paired resampling of12 fixed purposive witnesses; percentile95%;10000;seed42; not144-population or multi-seed CI'))
    dist=pd.DataFrame([dict(score='inherited_witness12',**distribution(df.inherited)),dict(score='standalone_witness12',**distribution(df.standalone)),
        dict(score='inherited_all144',**distribution(all144.mAP50_95))]);dist.to_csv(out/'score_distributions.csv',index=False)
    macros=[]
    for combo,g in df.groupby('output_combination'):
        macros.append(dict(output_combination=combo,n=len(g),inherited_mean=g.inherited.mean(),inherited_median=g.inherited.median(),
            standalone_mean=g.standalone.mean(),standalone_median=g.standalone.median(),mean_score_bias=g.score_bias.mean(),
            mean_rank_error=g.rank_error.mean(),mean_absolute_rank_error=g.rank_error.abs().mean(),**association(g.inherited,g.standalone)))
    macro=pd.DataFrame(macros);macro.to_csv(out/'MACRO_BIAS_ANALYSIS.csv',index=False)
    pairs=[];kernelpairs=[]
    for (_,a),(_,b) in itertools.combinations(df.iterrows(),2):
        di=b.inherited-a.inherited;ds=b.standalone-a.standalone;same=a.output_combination==b.output_combination
        pairs.append(dict(architecture_a=a.architecture_id,architecture_b=b.architecture_id,same_macro=same,
            delta_inherited=di,delta_standalone=ds,concordant=bool(di*ds>0),discordant=bool(di*ds<0)))
        diff=[s for s in (1,2,4,5,6,7) if not(pd.isna(a[f'stage{s}']) and pd.isna(b[f'stage{s}'])) and a[f'stage{s}']!=b[f'stage{s}']]
        if same and len(diff)==1:
            sign=1 if b[f'stage{diff[0]}']==5 else -1
            kernelpairs.append(dict(architecture_a=a.architecture_id,architecture_b=b.architecture_id,stage=diff[0],
                k5_minus_k3_inherited=sign*di,k5_minus_k3_standalone=sign*ds,direction_agrees=di*ds>0))
    pairdf=pd.DataFrame(pairs);pairdf.to_csv(out/'pairwise_rank_agreement.csv',index=False)
    pd.DataFrame(kernelpairs,columns=['architecture_a','architecture_b','stage','k5_minus_k3_inherited','k5_minus_k3_standalone','direction_agrees']).to_csv(out/'matched_kernel_pairs.csv',index=False)
    effects=[]
    for combo,g in [('ALL',df),*list(df.groupby('output_combination'))]:
        for variable in ('MACs','Params','heads','P2','P5','k5_count','early_k5_count','middle_k5_count','deep_k5_count'):
            for target in ('rank_error','inherited_rank','standalone_rank'):
                effects.append(dict(group=combo,n=len(g),variable=variable,target=target,**association(g[variable],g[target])))
    pd.DataFrame(effects).to_csv(out/'structural_bias_associations.csv',index=False)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    for column in ('inherited','standalone'):
        def hist(ax,col=column):ax.hist(df[col],bins=8);ax.set(xlabel='mAP50:95 (0–1)',ylabel='Witness count',title=col+' n=12')
        figure(figs,column+'_score_histogram',df[['architecture_id',column]],hist)
    def scatter(ax):
        for c,g in df.groupby('output_combination'):ax.scatter(g.inherited,g.standalone,label=c)
        for i,r in df.iterrows():ax.annotate(str(i), (r.inherited,r.standalone),fontsize=8)
        ax.set(xlabel='Inherited mAP50:95',ylabel='Standalone mAP50:95');ax.legend(fontsize=8)
    figure(figs,'inherited_vs_standalone',df[['architecture_id','output_combination','inherited','standalone']],scatter)
    def rankplot(ax):ax.bar(range(12),df.rank_error);ax.set(xlabel='Frozen witness index',ylabel='Inherited rank − standalone rank');ax.axhline(0,color='black',lw=.5)
    figure(figs,'rank_error',df[['architecture_id','rank_error']],rankplot)
    text='# Ranking Reliability Recheck\n\nIdentity:12/12 unique IDs, explicit kernels, immutable witness SHA and each original standalone status/metric verified. No reselection.\n\n'
    text+='n=12; scores0–1. Positive rank_error means inherited underestimated the architecture. Rank ties: average; topK ties: ID ascending.\n\n'
    text+=json.dumps(dict(calibrated=corrs,shared_bn=shared,topk=tops),indent=2)+'\n\n'
    text+='Bootstrap:10000 paired architecture resamples, seed42, percentile95% intervals. Constant resamples omitted and counts recorded. Conditional on purposive12 witnesses; not144-population uncertainty or standalone training-seed uncertainty. Pearson supplementary; no normality claim or asymptotic p-value used.\n\n'
    text+='## Score distribution\n\n'+dist.to_string(index=False)+'\n\nNear-tie thresholds1e-4/5e-4/1e-3 are descriptive tolerances (0.01/0.05/0.1 AP percentage points), not measured training-noise boundaries.\n\n'
    text+='## Macro groups\n\n'+macro.to_string(index=False)+'\n\nSmall groups cannot establish causal macro or kernel effects. Macro membership is categorical, not assigned a spurious ordinal correlation.\n\n'
    text+='Pair concordance:\n'+pairdf.groupby('same_macro')[['concordant','discordant']].sum().to_string()+'\n\nSingle-kernel witness pairs:\n'+str(kernelpairs)+'\n'
    (out/'RANKING_RELIABILITY_RECHECK.md').write_text(text,encoding='utf-8')
    bnfile=out/'BN_CALIBRATION_SENSITIVITY.csv'
    if bnfile.exists():
        bn=pd.read_csv(bnfile);budgetrows=[];baseline=None
        for budget,g in bn.groupby('batches'):
            assert len(g)==12 and set(g.architecture_id)==set(df.architecture_id)
            merged=df[['architecture_id','standalone','inherited']].merge(g,on='architecture_id',validate='one_to_one')
            if budget==16:
                baseline=float(np.max(np.abs(merged.mAP50_95-merged.inherited)));assert baseline<1e-10,baseline
            corr=bootstrap(merged.mAP50_95,merged.standalone)
            row=dict(batches=int(budget),images=int(g.images.iloc[0]),n=12,
                top3_overlap=len(set(merged.nlargest(3,'mAP50_95').architecture_id)&set(merged.nlargest(3,'standalone').architecture_id)))
            for k,v in corr.items():row.update({k:v['estimate'],k+'_ci_low':v['ci95'][0],k+'_ci_high':v['ci95'][1]})
            budgetrows.append(row)
        budgets=pd.DataFrame(budgetrows);budgets.to_csv(out/'bn_budget_correlations.csv',index=False)
        save(out/'bn16_reproduction.json',dict(max_abs_error=baseline,formal_budget_unchanged=True))
        def bnplot(ax):ax.plot(budgets.batches,budgets.spearman,'o-',label='Spearman');ax.plot(budgets.batches,budgets.kendall,'o-',label='Kendall');ax.axhline(0,color='gray');ax.set(xlabel='Train calibration batches',ylabel='Correlation');ax.legend()
        figure(figs,'bn_budget_sensitivity',budgets,bnplot)
        stability=[]
        for a,b in itertools.combinations(sorted(bn.batches.unique()),2):
            ga=bn[bn.batches==a].set_index('architecture_id').loc[df.architecture_id];gb=bn[bn.batches==b].set_index('architecture_id').loc[df.architecture_id]
            stability.append(dict(budget_a=int(a),budget_b=int(b),**association(ga.mAP50_95,gb.mAP50_95)))
        pd.DataFrame(stability).to_csv(out/'bn_cross_budget_rank_stability.csv',index=False)
        trajectory=[dict(epoch=e,status='AVAILABLE' if e==80 else 'NOT_SAVED',n=12 if e==80 else 0,
            spearman=corrs['spearman']['estimate'] if e==80 else None,kendall=corrs['kendall']['estimate'] if e==80 else None,
            top3_overlap=tops['3']['overlap'] if e==80 else None) for e in (20,40,60,80)]
        pd.DataFrame(trajectory).to_csv(out/'CHECKPOINT_RANKING_TRAJECTORY.csv',index=False)
        if (out/'diagnostic_provenance.json').exists():
            assert json.loads((out/'diagnostic_provenance.json').read_text())['status']=='COMPLETE'
            exposure=pd.read_csv(out/'SAMPLING_EXPOSURE_AUDIT.csv');updates=json.loads((out/'exposure_verification.json').read_text())
            grad=pd.read_csv(out/'GRADIENT_SCALE_DIAGNOSTICS.csv');cos=pd.read_csv(out/'shared_gradient_cosines.csv')
            grad_summary=grad.groupby('output_combination')[['loss','box_loss','object_loss','class_loss','positive_anchors','negative_anchors',
                'ignore_anchors','backbone_gradient_L2','neck_gradient_L2','head_gradient_L2','P2_specific_gradient_L2','P5_specific_gradient_L2']].mean()
            grad_summary.to_csv(out/'gradient_macro_means.csv')
            def gp(ax):
                for name in ('backbone','neck','head'):
                    ax.plot(grad_summary.index,grad_summary[name+'_gradient_L2'],'o-',label=name)
                ax.set(ylabel='Gradient L2 norm',title='Two identical batches × three kernel patterns');ax.legend()
            figure(figs,'gradient_macro_comparison',grad_summary.reset_index(),gp)
            exp_rows=[]
            for combo,n in [('P3_P4_P5',64),('P2_P3_P4',16),('P2_P3_P4_P5',64)]:
                count=updates['sampling_statistics'][combo]
                exp_rows.append(dict(output_combination=combo,architectures=n,actual_macro_iterations=count,
                    macro_probability=count/6640,mean_iterations_per_architecture=count/n))
            expdf=pd.DataFrame(exp_rows);expdf.to_csv(out/'exposure_macro_summary.csv',index=False)
            best=budgets.sort_values(['spearman','batches'],ascending=[False,True]).iloc[0]
            stdratio=float(df.inherited.std()/df.standalone.std());rangeratio=float(np.ptp(df.inherited)/np.ptp(df.standalone))
            corrtable=budgets.to_string(index=False)
            signed=pairdf.groupby('same_macro')[['concordant','discordant']].sum()
            causes=[
                dict(cause='A. BN_CALIBRATION_BIAS',status='PLAUSIBLE',evidence='Budget-dependent rank changes are measured below; no budget supplies reliable independent confirmation. Sensitivity is observed, but it does not identify BN as the sufficient cause. No per-architecture budget tuning.'),
                dict(cause='B. MACRO_PATH_WEIGHT_SHARING_BIAS',status='PLAUSIBLE',evidence='P345 mean rank error+4.0; P234 -2.25; P2345 -0.6. Descriptive macro calibration offset, but within-group ranking is not uniformly better; no causal intervention.'),
                dict(cause='C. SAMPLING_EXPOSURE_BIAS',status='PLAUSIBLE',evidence='Approximately4x per-architecture exposure is verified. Actual parameter steps match intended paths; P2/P5 module exposure differs only modestly. Unequal exact-architecture exposure is not proof of unequal per-parameter training quality.'),
                dict(cause='D. GRADIENT_SCALE_BIAS',status='NOT_SUPPORTED',evidence='Two fixed real batches show moderate differences, no order-of-magnitude P2 gradient amplification; finite gradients and no optimizer step. Long-run persistent differences were not measured.'),
                dict(cause='E. KERNEL_BRANCH_INTERFERENCE',status='NOT_SUPPORTED',evidence=f'{len(kernelpairs)} available single-kernel witness pairs all agree in delta sign; sparse coverage cannot rule out interference. Negative gradient cosines are diagnostic observations, not causal proof.'),
                dict(cause='F. CHECKPOINT_OVERTRAINING_OR_UNDERTRAINING',status='NOT_TESTABLE',evidence='Only final epoch80 exists. Atomic latest retention overwrote intermediate checkpoints. No trajectory or best diagnostic checkpoint selection possible.'),
                dict(cause='G. STANDALONE_TRAINING_NOISE',status='NOT_TESTABLE',evidence='Only seed42 per witness. Descriptive near ties do not estimate training variance; bootstrap across witnesses does not capture seed noise.'),
                dict(cause='H. SCORE_COMPRESSION',status='NOT_SUPPORTED',evidence=f'Comparable witness12 SD ratio inherited/standalone={stdratio:.4f}; range ratio={rangeratio:.4f}. Smaller SD across all144 is not a matched comparison to a stratified extreme witness sample.'),
                dict(cause='I. MIXED / UNRESOLVED',status='SUPPORTED',evidence='Current observations fail to isolate a single mechanism. Macro/exposure hypotheses are plausible; neither BN, gradients nor sparse kernel pairs identify a sufficient cause.')]
            pd.DataFrame(causes).to_csv(out/'cause_classification.csv',index=False)
            report='# Phase2D — Supernet Ranking Bias Diagnosis\n\n'
            report+='Most plausible lead: macro-path sharing/calibration differences together with unequal architecture exposure; causal root remains MIXED / UNRESOLVED. No search-space, baseline, loss, anchor, sampler or test changes.\n\n'
            report+='## Identity and independent recheck\n\n12/12 immutable witness IDs and original best-checkpoint active architectures match; source/protocol/witness hashes verified. No reselection. See RANKING_RELIABILITY_RECHECK.md for all score moments, ties, point table and paired-bootstrap95% CI.\n\n'
            report+=json.dumps(dict(n=12,calibrated=corrs,shared_bn=shared,topk=tops),indent=2)+'\n\n'
            report+='## Macro vs global\n\n'+macro.to_string(index=False)+'\n\n'+signed.to_string()+'\n\n'
            report+='Within-macro pair concordance10/19 and cross-macro24/47 are both near half. The four-level group is negatively correlated; therefore MACRO_CALIBRATION_BIAS is a hypothesis, not a proven explanation based on strong within-group reliability. Group sizes3/4/5 are too small for a robust general claim. Positive rank_error means inherited underestimation; negative means overestimation.\n\n'
            report+='## Exposure\n\n'+expdf.to_string(index=False)+'\n\n'
            report+='Total6640 iterations: stem/stages0–5/shared P3/P4 modules6640; stage6/7/P5 modules4352; P2 modules4435. Theory:6640,4426.67,4426.67 respectively. Full per-parameter AdamW step and BN counters in SAMPLING_EXPOSURE_AUDIT.csv agree with replay; replay agrees with every epoch macro/kernel count and final RNG. Architecture IDs were reconstructed, not originally logged. All144 appeared.\n\n'
            report+='Hierarchical sampler theory:1/192,1/48,1/192 per architecture, expected34.583/138.333/34.583 appearances. Active stage1/2/4/5 DW3/DW5 expected3320 each; stage6/7 expected2213.33 each. Actual counts:\n'+json.dumps(updates['sampling_statistics'],indent=2)+'\n\n'
            report+='## Fixed-batch gradient probe\n\n'+grad_summary.to_string()+'\n\n'
            report+='Nine declared architectures (allk3/allk5/fixed mixed within each macro) × first two real unaugmented train batches. Each starts from the same epoch80 state in train mode; no optimizer step, weights verified unchanged, original BN restored. P2-specific includes P2 lateral/output/head; P5-specific likewise, with backbone tail reported under backbone. Group norms overlap and must not be added as disjoint components. L2 and parameter-normalized RMS are both saved.\n\n'
            report+='Shared-backbone gradient cosines:\n'+cos.groupby('same_macro').shared_backbone_gradient_cosine.agg(['count','mean','min','max']).to_string()+'\n\n'
            report+='No observed order-of-magnitude gradient inflation. Two batches cannot establish persistent optimization bias; instantaneous direction disagreement is not proof of destructive weight sharing.\n\n'
            report+='## BN sensitivity (diagnostic only)\n\n'+corrtable+'\n\n'
            report+=f'Largest observed Spearman: budget{int(best.batches)} ({int(best.images)} train images), rho={best.spearman:.6f}; this is a diagnostic maximum, not a new protocol. Budget16 reproduces formal inherited scores with max_abs_error={baseline}.\n\n'
            report+='All budgets use prefixes of the same sorted330 train images, no augmentation, batch4/workers0/seed42. Full budget83 includes a final batch2; the existing cumulative BN method weights batches equally rather than pooling population moments. Each evaluation uses a fresh exported candidate and full valid99. No test. Pairwise cross-budget stability is in bn_cross_budget_rank_stability.csv; one run per budget cannot demonstrate repeat-run stability.\n\n'
            stable=max(stability,key=lambda r:r['spearman'])
            report+=f'Most concordant tested budget pair: {stable["budget_a"]}/{stable["budget_b"]}, Spearman={stable["spearman"]:.6f}, Kendall={stable["kendall"]:.6f}. The reliability curve is non-monotonic; every budget correlation CI crosses zero and Top3 overlap remains at most1/3. Larger calibration does not establish a reliable selector. Shared-BN rho0.02797/tau-0.06061 versus formal16 rho0.04895/tau0.03030 is a small numerical improvement, not demonstrated reliability improvement.\n\n'
            report+='## Kernel and complexity\n\nSingle-choice matched witness pairs:\n'+str(kernelpairs)+'\n\n'
            report+='All three observed single-kernel contrasts have matching inherited/standalone delta signs, at stages1,2,5. No evidence of systematic reversed k3/k5 preference in these pairs; they are different architectures and stages, all seed42, so they do not establish stable general sonar-operator design evidence. Other kernel/macro variables are confounded.\n\n'
            report+='Kernel summaries define early=stages1/2, middle=4/5, deep=6/7; inactive stages contribute zero to counts. Within-macro constant features have undefined correlation and are left empty.\n\n'
            report+=pd.DataFrame(effects).query("group == 'ALL' and target == 'rank_error'").to_string(index=False)+'\n\n'
            report+='Negative rank-error correlation with MACs/P2 describes relative inherited overestimation in this sample. MACs is software complexity, not latency; no hardware claim. No causal ordinal coding of output combination, no multiple-regression attribution with only12 cases.\n\n'
            report+='## Checkpoint and standalone noise limits\n\nOnly epoch80 is available; NOT_TESTABLE for formation/decay/undertraining/overtraining. Missing checkpoints are explicitly represented in CHECKPOINT_RANKING_TRAJECTORY.csv. No retraining to fabricate a trajectory.\n\n'
            report+='Standalone minimum adjacent gap0.00023836 (0.02384 AP percentage points); 2 adjacent gaps≤0.0005 and3≤0.001, with no exact ties. These are descriptive tolerances, not estimated noise. The12 runs suffice to reject using the current ranking for selection, but are not noise-free ground truth and cannot isolate a mechanism or establish fine ordering. Inherited rank1 falls to standalone11, a substantial inversion requiring caution beyond near-tie reordering.\n\n'
            report+='No additional seed runs were necessary to maintain the STOP gate and none were launched. Before attributing the bias to sharing rather than training noise, the smallest targeted optional addition is seeds43/44 for frozen witness indices0,3,9 (P345 reference, inherited top, standalone top), six runs total. Preserve all12 existing witnesses; never replace them or treat this selected trio as an unbiased population sample.\n\n'
            report+='## Cause classification\n\n'+pd.DataFrame(causes).to_string(index=False)+'\n\nSUPPORTED/NOT_SUPPORTED refer to present evidence, not universal causal proof or exclusion.\n\n'
            report+='## Next routes: B + D only\n\nRoute labels used here: A=strict exposure-fair sampling; B=separate fixed-macro supernets; C=kernel-sharing strategy revision; D=independent multi-fidelity training.\n\n'
            report+='B: a bounded fixed-macro sharing control can test the macro-interference hypothesis; cross-macro comparisons must use independent training, not uncalibrated scores from separate supernets. Do not presume its within-macro ranking will be reliable.\n\n'
            report+='D: prepare independent multi-fidelity screening over the144 fixed architectures, with promotion reliability separately checked against full80-epoch runs. Low-budget ranking is not automatically reliable either. This is the preferred fallback if sharing controls remain unreliable. Neither route was launched in Phase2D.\n\n'
            report+='Can we design V0.2 SonarOp now? **NO**. No stable standalone multi-seed structural evidence. Architecture selection remains stopped.\n'
            (out/'RANKING_BIAS_DIAGNOSIS.md').write_text(report,encoding='utf-8')
    print('Identity PASS; n=12;',association(df.inherited,df.standalone),flush=True)


if __name__=='__main__':main()
