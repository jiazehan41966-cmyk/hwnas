"""Report independent survivors and rejection audits without claiming all144 truth."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from phase2e_runner import ROOT,REPORT,load,binding
from phase2e_calibration import fidelity
from phase2d_analysis import figure,save


def main():
    cfg,b=load();first=pd.read_csv(REPORT/'stage1_results.csv');second=pd.read_csv(REPORT/'stage2_results.csv');final=pd.read_csv(REPORT/'final_results.csv')
    assert len(first)==first.architecture_id.nunique()==144
    assert len(second)==second.architecture_id.nunique()==108
    assert len(final)==final.architecture_id.nunique()==72
    final=final.sort_values(['mAP50_95','architecture_id'],ascending=[False,True]);final['rank']=range(1,73)
    final.to_csv(REPORT/'final_independent_ranking.csv',index=False)
    merged=final[['architecture_id','mAP50_95','rank']].merge(first[['architecture_id','mAP50_95']],on='architecture_id',suffixes=('_80','_B1'),validate='one_to_one')
    merged=merged.merge(second[['architecture_id','mAP50_95']].rename(columns={'mAP50_95':'mAP50_95_B2'}),on='architecture_id',validate='one_to_one')
    stats=[]
    for budget,column in [(40,'mAP50_95_B1'),(60,'mAP50_95_B2')]:
        stats.append(dict(budget=budget,**fidelity(merged[column].to_numpy(),merged.mAP50_95_80.to_numpy(),merged.architecture_id.tolist())))
        merged[f'rank_at_{budget}_among_finalists']=rankdata(-merged[column])
    merged.to_csv(REPORT/'finalist_rank_trajectories.csv',index=False)
    audit_path=REPORT/'rejection_audit_results.csv';audits=pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    threshold=float(final.iloc[4].mAP50_95);violations=audits[audits.mAP50_95>threshold] if len(audits) else audits
    # Known12 provide retrospective sanity only; they calibrated the schedule and are not independent proof.
    original=json.loads((ROOT/'reports/phase2c_supernet/ranking_witness_manifest.json').read_text())['witnesses'];known=[]
    for i,a in enumerate(original):
        h=json.loads((REPORT/'input_snapshot/results'/f'witness_{i:02d}'/'epoch_log.json').read_text())
        known.append(dict(architecture_id=a['architecture_id'],mAP50_95=max(r['valid']['mAP50_95'] for r in h),
                          survived_stage1=a['architecture_id'] in set(second.architecture_id),survived_stage2=a['architecture_id'] in set(final.architecture_id)))
    pd.DataFrame(known).to_csv(REPORT/'original12_survival.csv',index=False)
    save(REPORT/'promotion_reliability.json',dict(binding=binding(),conditional_finalist_fidelity=stats,
        audited_rejections=len(audits),audit_top5_violation=bool(len(violations)),violations=violations.architecture_id.tolist() if len(violations) else [],
        limitation='Finalist-only correlations condition on promotion; small random rejection audits cannot prove no false negatives among144.',
        test_evaluated=False,final_selection_executed=False))
    figs=REPORT/'figures';figs.mkdir(exist_ok=True)
    for budget in (40,60):
        frame=merged[['architecture_id',f'mAP50_95_B{1 if budget==40 else 2}','mAP50_95_80']]
        def draw(ax,f=frame,budget=budget):ax.scatter(f.iloc[:,1],f.iloc[:,2],s=12);ax.set(xlabel=f'Budget{budget}',ylabel='Budget80',title='Promoted finalists only')
        figure(figs,f'finalist_budget_{budget}',frame,draw)
    def metrics(r):return json.loads(r.kernels)
    pairs=[];p2=[]
    for (_,a),(_,z) in itertools.combinations(final.iterrows(),2):
        ka,kz=metrics(a),metrics(z)
        if a.output_combination==z.output_combination:
            changed=[k for k in ka if ka[k]!=kz[k]]
            if len(changed)==1:
                stage=changed[0];sign=1 if kz[stage]==5 else -1
                pairs.append(dict(architecture_a=a.architecture_id,architecture_b=z.architecture_id,stage=stage,
                    macro=a.output_combination,k5_minus_k3_mAP=sign*(z.mAP50_95-a.mAP50_95)))
        if {a.output_combination,z.output_combination}=={'P3_P4_P5','P2_P3_P4_P5'} and ka==kz:
            sign=1 if z.output_combination=='P2_P3_P4_P5' else -1
            p2.append(dict(architecture_a=a.architecture_id,architecture_b=z.architecture_id,P2_increment_mAP=sign*(z.mAP50_95-a.mAP50_95)))
    pd.DataFrame(pairs,columns=['architecture_a','architecture_b','stage','macro','k5_minus_k3_mAP']).to_csv(REPORT/'independent_matched_kernel_pairs.csv',index=False)
    pd.DataFrame(p2,columns=['architecture_a','architecture_b','P2_increment_mAP']).to_csv(REPORT/'independent_matched_p2_pairs.csv',index=False)
    groups=final.groupby('output_combination').mAP50_95.agg(['count','mean','median','min','max']);groups.to_csv(REPORT/'final_macro_summary.csv')
    kernel=[]
    for stage in (1,2,4,5,6,7):
        for k in (3,5):
            g=final[final.kernels.map(lambda v:json.loads(v).get(f'stage{stage}')==k)]
            kernel.append(dict(stage=stage,kernel=k,n=len(g),mean_mAP=float(g.mAP50_95.mean()) if len(g) else None))
    pd.DataFrame(kernel).to_csv(REPORT/'independent_kernel_summary.csv',index=False)
    (REPORT/'STRUCTURAL_PREFERENCE_ANALYSIS.md').write_text('# Independent Structural Evidence\n\nPromoted seed42 architectures only; no inherited weights or test scores. This is a selected population, not an unbiased144-space comparison.\n\n'+groups.to_string()+'\n\nP2-only contrasts require identical kernels in P345 vs P2345. P234 simultaneously adds P2, removes P5 and prunes stages6/7. Kernel causal comparisons prioritize one-choice matched pairs; aggregate means are confounded.\n\nMatched kernel pairs:\n'+str(pairs)+'\n\nMatched P2 pairs:\n'+str(p2)+'\n\nSingle-seed selected results do not establish stable multi-seed operator design evidence. SonarOp implementation remains forbidden; gate NO pending interpretation and independent replication.\n',encoding='utf-8')
    nominal=144*40+108*20+72*20
    audit_epochs=sum(80-(40 if aid in json.loads((REPORT/'promotion_stage1.json').read_text())['audit'] else 60) for aid in audits.architecture_id) if len(audits) else 0
    actual_new_epochs=0;actual_new_seconds=0
    for folder in (ROOT/'results/phase2e_independent').iterdir():
        if (folder/'epoch_log.json').exists():
            h=json.loads((folder/'epoch_log.json').read_text());actual_new_epochs+=len(h);actual_new_seconds+=sum(r['wall_seconds'] for r in h)
    save(REPORT/'compute_accounting.json',dict(exhaustive_epochs=11520,nominal_schedule_epochs=nominal,rejection_audit_extra_epochs=audit_epochs,
        nominal_saving_fraction=1-(nominal+audit_epochs)/11520,new_training_epochs=actual_new_epochs,new_gpu_training_seconds=actual_new_seconds,
        historical_witness_epochs=960,actual_lifetime_epochs=actual_new_epochs+960,
        timing_note='Sum measured train+valid epoch wall time on dedicated GPU per run; exhaustive GPU-time saving cannot be directly measured without exhaustive execution.'))
    (REPORT/'PROMOTION_RELIABILITY.md').write_text('# Promotion Reliability\n\n'+json.dumps(stats,indent=2)+
        f'\n\nFinalist-only metrics, n=72, condition on promotion and cannot measure false negatives for untrained rejects. Random rejected audits n={len(audits)}; audited rejects above promoted final Top5 cutoff: {len(violations)}. '
        'If nonzero, STOP selection. Even if zero, the small audit has limited power; no assertion that the global best among144 was retained. Original12 survival is calibration reuse, not an independent generalization test.\n\nNo test evaluation or SonarOp implementation.\n',encoding='utf-8')


if __name__=='__main__':main()
