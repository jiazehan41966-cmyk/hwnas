from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools/sonar_spec'))
from hwnas_fpga.deploy.detection_gate import relative_accuracy_gate,action_decision,guarded_execute
from deployment_pipeline import load_evidence,validate_policy
import deployment_pipeline as pipeline
import yaml


def rows(network='D',precision='W8A8',m=.02,recall=.3):
    return [dict(network=network,architecture_id=network+'_arch',seed=s,precision=precision,
        checkpoint_sha256='a'*64,evaluator_sha256='b'*64,calibration_sha256='c'*64,
        split_manifest_sha256='d'*64,
        checkpoint_policy='ALL_BEST_PT',training_budget=80,split='valid',test_evaluated=False,
        mAP50_95=m,Recall=recall) for s in (42,43,44)]


def test_and_gate_equal_control_and_no_absolute_quality_claim():
    g=relative_accuracy_gate(rows(),rows(),precision='W8A8')
    assert g['passed'] and g['status']=='CONTROL_PASS'
    assert g['application_accuracy_sufficient']=='NOT_ESTABLISHED_RELATIVE_ONLY'
    assert relative_accuracy_gate(rows('B1',m=.03),rows(),precision='W8A8')['status']=='PASS'


@pytest.mark.parametrize('metric',['mAP50_95','Recall'])
def test_either_metric_below_D_stops_all_downstream(metric):
    candidate=rows('B1',m=.04,recall=.8)
    for r in candidate:r[metric]=.001
    failed=relative_accuracy_gate(candidate,rows(),precision='W8A8')
    assert failed['status']=='STOP_ACCURACY_BELOW_D'
    good=relative_accuracy_gate(rows(),rows(),precision='W8A8')
    for action in ['parameter_export','integer_parity','hls','route','board']:
        assert not action_decision(action,good,failed)['allowed']
    assert not action_decision('ptq',failed,good)['allowed']


def test_negative_single_seed_is_reported_not_extra_gate():
    c=rows('B1',m=.04,recall=.5);c[0]['mAP50_95']=.001
    g=relative_accuracy_gate(c,rows(),precision='W8A8')
    assert g['passed'] and g['metrics']['mAP50_95']['positive_seed_count']==2


@pytest.mark.parametrize('mutation',[
    'missing','duplicate','nan','wrong_split','test','precision','evaluator','checkpoint','budget'])
def test_invalid_or_unmatched_evidence_blocks(mutation):
    c=rows('B1',m=.03)
    if mutation=='missing':c.pop()
    elif mutation=='duplicate':c[2]['seed']=42
    elif mutation=='nan':c[0]['mAP50_95']=float('nan')
    elif mutation=='wrong_split':c[0]['split']='train'
    elif mutation=='test':c[0]['test_evaluated']=True
    elif mutation=='precision':c[0]['precision']='FP32'
    elif mutation=='evaluator':c[0]['evaluator_sha256']='d'*64
    elif mutation=='checkpoint':c[0]['checkpoint_sha256']=''
    elif mutation=='budget':c[0]['training_budget']=40
    g=relative_accuracy_gate(c,rows(),precision='W8A8')
    assert not g['passed'] and g['status']=='BLOCKED_EVIDENCE'


def test_search_budget_and_hardware_are_distinct_from_accuracy():
    g=relative_accuracy_gate(rows(),rows(),precision='W8A8')
    assert action_decision('search',g,g)['status']=='STOP_SEARCH_BUDGET'
    assert action_decision('parameter_export',g,g)['allowed']
    assert action_decision('board',g,g,hardware={'witness_pass':True})['status']=='BLOCKED_HARDWARE_EVIDENCE'


def test_dispatch_does_not_run_when_accuracy_or_journal_fails():
    called=[];events=[]
    runner=lambda *a,**k:called.append(a) or SimpleNamespace(returncode=0)
    with pytest.raises(RuntimeError,match='stopped'):
        guarded_execute(['not-run'],{'allowed':False,'status':'STOP_ACCURACY_BELOW_D'},journal=events.append,runner=runner)
    assert not called and events[0]['event']=='ACTION_GATE'
    def broken(event):raise OSError('disk full')
    with pytest.raises(OSError,match='disk full'):
        guarded_execute(['not-run'],{'allowed':True,'action':'parameter_export'},journal=broken,runner=runner)
    assert not called
    guarded_execute(['placeholder'],{'allowed':True,'action':'parameter_export'},journal=events.append,runner=runner)
    assert len(called)==1 and events[-1]['returncode']==0


def test_real_manifest_integrity_and_same_precision_results():
    cfg=yaml.safe_load((ROOT/'configs/sonar_deployment_gate_v1.yaml').read_text())
    data,_,_,_=load_evidence(cfg)
    assert len(data)==18
    for p in ['FP32','W8A8']:
        ref=[r for r in data if r['network']=='D' and r['precision']==p]
        for n in ['B1','C','D']:
            c=[r for r in data if r['network']==n and r['precision']==p]
            assert relative_accuracy_gate(c,ref,precision=p)['passed']
    cfg['reference_precision']='FP32'
    with pytest.raises(ValueError,match='semantics'):validate_policy(cfg)


@pytest.mark.parametrize('failure',[ValueError('late mixed identity mismatch'),OSError('metrics disk full')])
def test_late_audit_failure_clears_prior_pass_and_ready_queue(tmp_path,monkeypatch,failure):
    pipeline.write(tmp_path/'ADMISSION.json',{'status':'ACCURACY_PASS_HARDWARE_PENDING'})
    pipeline.write(tmp_path/'ACTION_QUEUE.json',[{'allowed':True}])
    def fail_late(config,out):
        assert pipeline.read(out/'ADMISSION.json')['status']=='IN_PROGRESS'
        assert pipeline.read(out/'ACTION_QUEUE.json')==[]
        pipeline.write(out/'ACTION_QUEUE.json',[{'allowed':True}])
        raise failure
    monkeypatch.setattr(pipeline,'_run_audit',fail_late)
    result=pipeline.run_audit(tmp_path/'config.yaml',tmp_path)
    assert result['status']=='BLOCKED_EVIDENCE'
    assert not pipeline.read(tmp_path/'ADMISSION.json')['allowed']
    assert pipeline.read(tmp_path/'ACTION_QUEUE.json')==[]
    assert pipeline.read(tmp_path/'MANIFEST.json')['status']=='BLOCKED_EVIDENCE'


def test_audit_admission_commits_after_manifest(tmp_path,monkeypatch):
    def succeed(config,out):
        assert pipeline.read(out/'ADMISSION.json')['status']=='IN_PROGRESS'
        pipeline.write(out/'MANIFEST.json',{'complete':True})
        return {'status':'ACCURACY_PASS_HARDWARE_PENDING'}
    monkeypatch.setattr(pipeline,'_run_audit',succeed)
    result=pipeline.run_audit(tmp_path/'config.yaml',tmp_path)
    assert result['manifest_sha256']==pipeline.sha(tmp_path/'MANIFEST.json')
    assert pipeline.read(tmp_path/'ADMISSION.json')==result


def test_concurrent_audits_are_serialized(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    active=[]
    def work(config,out):
        active.append(1)
        assert len(active)==1
        time.sleep(.05)
        pipeline.write(out/'MANIFEST.json',{'complete':True})
        active.pop()
        return {'status':'ACCURACY_PASS_HARDWARE_PENDING'}
    monkeypatch.setattr(pipeline,'_run_audit',work)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:pipeline.run_audit(tmp_path/'config.yaml',tmp_path),range(2)))
    assert all(r['status']=='ACCURACY_PASS_HARDWARE_PENDING' for r in results)


def test_board_alias_requires_original_manifest_sha_and_never_masks_mismatch(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    alternate=tmp_path/'original.bin';alternate.write_bytes(b'original')
    pipeline.write(bundle/'manifest.json',{'scope':'operator witness','full_detector':False,
        'files':{'vector.bin':pipeline.sha(alternate)}})
    pipeline.write(tmp_path/'hw.json',{'total_candidates':2592,'hardware_measured':2592,
        'architectures':144,'measurement_scope':'OP_LEVEL_HLS'})
    pipeline.write(tmp_path/'cases.json',{})
    paths={'board_manifest':bundle/'manifest.json','hardware_summary':tmp_path/'hw.json',
        'board_results':tmp_path/'cases.json'}
    aliases={'vector.bin':'original.bin'}
    result=pipeline.evidence_layers(paths,aliases,root=tmp_path)
    assert result['board_witness']['artifact_verification']=='COMPLETE'
    assert result['board_witness']['resolved_files']['vector.bin']['sha256']==pipeline.sha(alternate)
    (bundle/'vector.bin').write_bytes(b'incorrect primary')
    result=pipeline.evidence_layers(paths,aliases,root=tmp_path)
    assert result['board_witness']['mismatched_files']==['vector.bin']
    assert not result['board_witness']['artifact_aliases_used']
