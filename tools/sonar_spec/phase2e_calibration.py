"""Calibrate independent-training budgets using immutable standalone histories only."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from phase2d_analysis import association,bootstrap,digest,save,figure,verified_rows

BUDGETS=(5,10,20,30,40,60,80)


def ordered(scores,ids):
    return sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))


def fidelity(x,y,ids):
    ranks=rankdata(-np.asarray(x));final=rankdata(-np.asarray(y));a=ordered(x,ids);b=ordered(y,ids)
    signs=[np.sign((x[i]-x[j])*(y[i]-y[j])) for i,j in itertools.combinations(range(len(ids)),2)]
    return dict(n=len(ids),**association(x,y),rank_MAE=float(np.mean(abs(ranks-final))),
        top3_recall=len(set(a[:3])&set(b[:3]))/min(3,len(ids)),top5_recall=len(set(a[:5])&set(b[:5]))/min(5,len(ids)),
        pairwise_concordance=float(np.mean(np.asarray(signs)>0)),discordant_pairs=int(sum(s<0 for s in signs)),
        tied_pairs=int(sum(s==0 for s in signs)),pair_count=len(signs),final_best_budget_rank=a.index(b[0])+1)


def simulate(scores,ids,b1,b2,f1,f2):
    first=ordered(scores[:,b1-1],ids)[:math.ceil(len(ids)*f1)]
    second=sorted(first,key=lambda i:(-scores[i,b2-1],ids[i]))[:math.ceil(len(first)*f2)]
    final=ordered(scores[:,-1],ids)
    return dict(B1=b1,B2=b2,stage1_fraction=f1,stage2_fraction=f2,n_stage2=len(first),n_final=len(second),
        stage1_final_top3_recall=len(set(first)&set(final[:3]))/3,stage1_final_top5_recall=len(set(first)&set(final[:5]))/5,
        final_top3_survival=len(set(second)&set(final[:3]))/3,final_top5_survival=len(set(second)&set(final[:5]))/5,
        best_survives=final[0] in second,stage2_ids=[ids[i] for i in first],final_ids=[ids[i] for i in second])


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path(__file__).resolve().parents[2]
    p.add_argument('--output',type=Path,default=root/'reports/phase2e_multifidelity');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    c=root/'reports/phase2c_supernet';df,_,identity=verified_rows(c,out/'input_snapshot');ids=df.architecture_id.tolist()
    arrays=[];audits=[];historyrows=[];bindings={}
    for i,aid in enumerate(ids):
        path=out/'input_snapshot/results'/f'witness_{i:02d}'/'epoch_log.json';history=json.loads(path.read_text());bindings[str(path.relative_to(out)).replace('\\','/')]=digest(path)
        assert [r['epoch'] for r in history]==list(range(1,81))
        assert all(r['valid']['num_images']==99 for r in history)
        for row in history:
            assert all(np.isfinite(row['valid'][k]) for k in ('AP50','AP75','mAP50_95','Precision','Recall','validation_loss'))
            assert all(np.isfinite(v) for v in row['losses'].values())
            historyrows.append(dict(architecture_id=aid,output_combination=df.iloc[i].output_combination,epoch=row['epoch'],
                **{k:row['valid'][k] for k in ('AP50','AP75','mAP50_95','Precision','Recall','validation_loss')},
                **{'train_'+k:v for k,v in row['losses'].items()},train_seconds=row['elapsed_seconds']))
        values=[r['valid']['mAP50_95'] for r in history];assert abs(max(values)-df.iloc[i].standalone)<1e-14;arrays.append(values)
        audits.append(dict(index=i,architecture_id=aid,epoch_records=80,epochs='1..80',valid_metrics_complete=True,
            train_loss_complete=True,valid_loss_complete=True,best_epoch=int(np.argmax(values))+1,endpoint80=values[-1],best80=max(values),
            checkpoints='best.pt/latest.pt only; intermediate epoch weights not retained',history_sha256=digest(path)))
    raw=np.array(arrays);prefix=np.maximum.accumulate(raw,axis=1);curves=pd.DataFrame(historyrows)
    curves.to_csv(out/'standalone_learning_curves.csv',index=False);pd.DataFrame(audits).to_csv(out/'standalone_history_audit.csv',index=False)
    save(out/'calibration_inputs.json',dict(n=12,witness_sha256=digest(c/'ranking_witness_manifest.json'),histories=bindings,
        witnesses=identity,source_sha256=digest(Path(__file__)),population='fixed purposive inherited-stratified witnesses, not random144 sample'))
    rows=[];macro=[];crossings=[];ranks=[]
    for semantics,values in [('epoch_endpoint',raw),('best_valid_up_to_budget',prefix)]:
        for b in BUDGETS:
            row=dict(score_semantics=semantics,budget=b,**fidelity(values[:,b-1],values[:,-1],ids))
            if semantics=='best_valid_up_to_budget' and b!=80:
                ci=bootstrap(values[:,b-1],values[:,-1],repeats=10000)
                for k,v in ci.items():row[k+'_CI95_low']=v['ci95'][0];row[k+'_CI95_high']=v['ci95'][1]
            rows.append(row)
            for combo,g in df.groupby('output_combination'):
                ix=g.index.to_numpy();x=values[ix,b-1];y=values[ix,-1]
                global_error=rankdata(-values[:,b-1])-rankdata(-values[:,-1])
                macro.append(dict(score_semantics=semantics,budget=b,output_combination=combo,n=len(ix),
                    mean_score=x.mean(),final_mean_score=y.mean(),mean_global_rank_error=global_error[ix].mean(),**association(x,y)))
            if b in (10,20,40):
                for i,j in itertools.combinations(range(12),2):
                    early=values[i,b-1]-values[j,b-1];late=values[i,-1]-values[j,-1]
                    crossings.append(dict(score_semantics=semantics,budget=b,architecture_a=ids[i],architecture_b=ids[j],
                        same_macro=df.iloc[i].output_combination==df.iloc[j].output_combination,
                        early_delta=early,final_delta=late,inversion=early*late<0,tied=early*late==0))
        for epoch in range(80):
            for i,r in enumerate(rankdata(-values[:,epoch])):ranks.append(dict(score_semantics=semantics,epoch=epoch+1,architecture_id=ids[i],rank=r))
    scores=pd.DataFrame(rows);scores.to_csv(out/'BUDGET_FIDELITY.csv',index=False);pd.DataFrame(macro).to_csv(out/'MACRO_BUDGET_FIDELITY.csv',index=False)
    pd.DataFrame(crossings).to_csv(out/'learning_curve_crossings.csv',index=False);rankdf=pd.DataFrame(ranks);rankdf.to_csv(out/'rank_trajectory.csv',index=False)
    plans=[]
    for b1,b2 in itertools.combinations(BUDGETS[:-1],2):
        for f1,f2 in itertools.product((.5,2/3,.75,5/6),repeat=2):
            r=simulate(prefix,ids,b1,b2,f1,f2);r.pop('stage2_ids');r.pop('final_ids')
            n2=math.ceil(144*f1);nf=math.ceil(n2*f2);r['nominal_epochs']=144*b1+n2*(b2-b1)+nf*(80-b2)
            r['nominal_saving']=1-r['nominal_epochs']/11520;plans.append(r)
    pd.DataFrame(plans).to_csv(out/'promotion_schedule_calibration.csv',index=False)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def curvesplot(ax):
        for i,aid in enumerate(ids):ax.plot(range(1,81),raw[i],label=f'{i}: '+df.iloc[i].output_combination,lw=.8)
        ax.set(xlabel='Epoch',ylabel='Endpoint valid mAP50:95');ax.legend(fontsize=6,ncol=2)
    figure(figs,'independent_learning_curves',curves,curvesplot)
    def rankplot(ax):
        for i,aid in enumerate(ids):ax.plot(range(1,81),rankdata(-prefix,axis=0)[i],label=str(i),lw=.8)
        ax.invert_yaxis();ax.set(xlabel='Epoch budget',ylabel='Best-prefix rank');ax.legend(ncol=4,fontsize=7)
    figure(figs,'budget_rank_trajectory',rankdf.query("score_semantics=='best_valid_up_to_budget'"),rankplot)
    for b in (10,20,40,60):
        frame=pd.DataFrame(dict(architecture_id=ids,budget_score=prefix[:,b-1],final_score=prefix[:,-1]))
        def scatter(ax,f=frame,budget=b):
            ax.scatter(f.budget_score,f.final_score)
            for i,r in f.iterrows():ax.annotate(str(i),(r.budget_score,r.final_score),fontsize=8)
            ax.set(xlabel=f'Best valid through epoch{budget}',ylabel='Best valid through epoch80')
        figure(figs,f'budget_{b}_vs_final',frame,scatter)
    (out/'STANDALONE_HISTORY_AUDIT.md').write_text('# Standalone History Audit\n\n12/12 immutable witness identities and source/seed/protocol binding pass. Each has80 complete epochs of AP50/AP75/mAP50:95/Precision/Recall, train component losses and valid loss. No rerun required.\n\n'+pd.DataFrame(audits).to_string(index=False)+'\n\nOriginal checkpoints retain best/latest only; metric histories permit retrospective fidelity but not intermediate-state resumption. Original seed42 witnesses may be reused as completed independent runs, with only budget-prefix metrics supplied to promotion. Their known future metrics must never enter promotion.\n',encoding='utf-8')
    primary=scores.query("score_semantics=='best_valid_up_to_budget'")
    (out/'BUDGET_FIDELITY_ANALYSIS.md').write_text('# Budget Fidelity Analysis\n\nPrimary decision semantics follow frozen baseline checkpoint selection: best valid mAP50:95 observed by budget b versus best valid observed by80. This is explicitly different from the literal epoch-b/epoch80 endpoint; both are recorded in BUDGET_FIDELITY.csv. No metric-semantic selection based on which correlation looks better.\n\n'+primary.to_string(index=False)+'\n\nEndpoint comparisons:\n'+scores.query("score_semantics=='epoch_endpoint'").to_string(index=False)+'\n\nCrossings:\n'+pd.DataFrame(crossings).groupby(['score_semantics','budget']).inversion.sum().to_string()+'\n\nThe12 runs are a purposive calibration set, not unbiased evidence for all144 architectures. Bootstrap CI is paired,10000 resamples, seed42, conditional on this set. Schedule selection on these curves is calibration, not independent validation. TopK recall at equal K differs from recall under a broad promotion quota. Final best drops to rank9 at budget30; aggressive early halving can discard it.\n\nNo candidate ranking, hardware claim, test performance or SonarOp conclusion follows from this calibration alone.\n',encoding='utf-8')
    print(primary[['budget','spearman','kendall','top3_recall','top5_recall']].to_string(index=False))


if __name__=='__main__':main()
