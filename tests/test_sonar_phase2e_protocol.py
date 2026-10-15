"""Promotion must not see future epochs or inherited scores."""
import sys
from pathlib import Path
import copy
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/sonar_spec'))
from phase2e_runner import prefix_record,promote,rejected_audit,state_hash
from phase2e_calibration import simulate,fidelity


def hist(values):
    return [dict(epoch=i+1,elapsed_seconds=1,valid=dict(mAP50_95=v,AP50=v,AP75=v,Precision=v,Recall=v)) for i,v in enumerate(values)]


def test_prefix_never_reads_future_best_and_keeps_earliest_tie():
    original=hist([.2,.2,.1,.99]);changed=copy.deepcopy(original);changed[-1]['valid']['mAP50_95']=999
    assert prefix_record(original,3,'a')==prefix_record(changed,3,'a')
    assert prefix_record(original,3,'a')['best_epoch']==1
    assert prefix_record(original,3,'a')['mAP50_95']==.2
    with pytest.raises(ValueError):prefix_record(original,5,'a')


def test_promotion_ties_use_id_not_endpoint_or_inherited():
    rows=[dict(architecture_id='b',mAP50_95=.2,endpoint_mAP50_95=9),dict(architecture_id='a',mAP50_95=.2,endpoint_mAP50_95=0),dict(architecture_id='c',mAP50_95=.1)]
    assert [r['architecture_id'] for r in promote(rows,1,2)]==['a','b']
    with pytest.raises(ValueError):promote(rows+[rows[0]],1,2)
    with pytest.raises(ValueError):promote([dict(architecture_id='a',mAP50_95=float('nan'))],1,2)


def test_rejection_audit_excludes_known_calibration_and_is_order_invariant():
    rows=[dict(architecture_id=str(i),output_combination=c,mAP50_95=.1) for i,c in enumerate(['P3_P4_P5']*3+['P2_P3_P4']*3+['P2_P3_P4_P5']*3)]
    selected=rejected_audit(rows,[rows[0]],1,{'1':0})
    assert len(selected)==3 and '0' not in selected and '1' not in selected
    assert selected==rejected_audit(list(reversed(rows)),[rows[0]],1,{'1':0})


def test_final_top_recall_can_differ_from_topk_rank_recall():
    scores=np.tile(np.arange(12,dtype=float)[:,None],(1,80));ids=[str(i) for i in range(12)]
    scores[:,39]=np.array([0,1,2,3,4,5,6,7,11,10,9,8])
    point=fidelity(scores[:,39],scores[:,-1],ids)
    assert point['top3_recall']<1
    plan=simulate(scores,ids,40,60,.75,2/3)
    assert plan['stage1_final_top5_recall']==1 and plan['final_top5_survival']==1


def test_optimizer_hash_detects_tensor_or_hyperparameter_changes():
    import torch
    a={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.ones(3)}},'param_groups':[{'lr':.001}]}
    assert state_hash(a)==state_hash(copy.deepcopy(a))
    b=copy.deepcopy(a);b['state'][0]['exp_avg'][0]=0
    assert state_hash(a)!=state_hash(b)


def test_source_binding_includes_statistics_dependency():
    from phase2e_runner import sources
    assert 'tools/sonar_spec/phase2d_analysis.py' in sources()


def test_witness_manifest_drift_rejected(monkeypatch):
    import phase2e_runner as r
    original=r.sha
    monkeypatch.setattr(r,'sha',lambda p:'tampered' if Path(p).name=='ranking_witness_manifest.json' else original(p))
    with pytest.raises(ValueError,match='witness manifest drift'):r.load(False)


def test_committed_budget_recovers_stale_history_and_missing_status(tmp_path,monkeypatch):
    import json
    import torch
    import phase2e_runner as r
    from formal_runner import rng_state
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4,1)),batch_size=2,generator=torch.Generator().manual_seed(42))
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    folder=tmp_path/'results/phase2e_independent/a';folder.mkdir(parents=True)
    history=hist([.1]*40)
    for row in history:row['wall_seconds']=1
    state=dict(binding={'fixture':1},architecture_id='a',smoke=False,epoch=40,model_state_dict=model.state_dict(),
        optimizer_state_dict=opt.state_dict(),rng=rng_state(loader),history=history,initial_state_sha256='fixture',parents=[],
        optimizer_state_sha256=state_hash(opt.state_dict()),git_commit='fixture',invocation_start='start',updated_at='end')
    torch.save(state,folder/'latest.pt');(folder/'epoch_log.json').write_text(json.dumps(history[:1]))
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'binding',lambda:{'fixture':1})
    monkeypatch.setattr(r,'load',lambda require_admission=True:({'search_space':'space','budgets':[40,60,80]},{'dataset':{'data_dir':'fixture'}}))
    monkeypatch.setattr(r,'verify_baseline',lambda b:None);monkeypatch.setattr(r,'existing_map',lambda:{})
    monkeypatch.setattr(r,'arch_map',lambda:{'a':{'output_combination':'P3_P4_P5','kernels':{'stage1':3}}})
    monkeypatch.setattr(r,'build_sonar_candidate',lambda *args:torch.nn.Linear(1,1))
    monkeypatch.setattr(r.frozen,'make_loader',lambda *args,**kwargs:loader)
    monkeypatch.setattr(r.frozen,'get_model_info',lambda *args:{'total_params':2,'estimated_macs':2})
    result=r.train('a',40,'cpu')
    assert result['budget']==40 and len(json.loads((folder/'epoch_log.json').read_text()))==40
    assert (folder/'budget_40.pt').exists()
    assert json.loads((folder/'budget_40_status.json').read_text())['derived_metadata_recovered']


def test_result_reporting_stops_on_strong_audited_rejection(tmp_path,monkeypatch):
    import json
    import pandas as pd
    import phase2e_results as m
    from phase2e_runner import arch_map
    architectures=arch_map();ids=sorted(architectures)
    report=tmp_path/'reports/phase2e_multifidelity';report.mkdir(parents=True)
    rows=[dict(architecture_id=aid,output_combination=architectures[aid]['output_combination'],
        kernels=json.dumps(architectures[aid]['kernels']),mAP50_95=(i+1)/1000) for i,aid in enumerate(ids)]
    pd.DataFrame(rows).to_csv(report/'stage1_results.csv',index=False)
    pd.DataFrame(rows[36:]).to_csv(report/'stage2_results.csv',index=False)
    pd.DataFrame(rows[72:]).to_csv(report/'final_results.csv',index=False)
    audit=dict(rows[0],mAP50_95=.99);pd.DataFrame([audit]).to_csv(report/'rejection_audit_results.csv',index=False)
    (report/'promotion_stage1.json').write_text(json.dumps({'audit':[ids[0]]}))
    original=tmp_path/'reports/phase2c_supernet';original.mkdir()
    (original/'ranking_witness_manifest.json').write_text(json.dumps({'witnesses':[{'architecture_id':a} for a in ids[:12]]}))
    for i in range(12):
        folder=report/'input_snapshot/results'/f'witness_{i:02d}';folder.mkdir(parents=True)
        (folder/'epoch_log.json').write_text(json.dumps(hist([.1]*80)))
    (tmp_path/'results/phase2e_independent').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'REPORT',report)
    monkeypatch.setattr(m,'load',lambda:({},{}));monkeypatch.setattr(m,'binding',lambda:{'fixture':1})
    m.main()
    result=json.loads((report/'promotion_reliability.json').read_text())
    assert result['audit_top5_violation'] and result['violations']==[ids[0]]
    assert len(pd.read_csv(report/'final_independent_ranking.csv'))==72
    assert (report/'STRUCTURAL_PREFERENCE_ANALYSIS.md').exists()
