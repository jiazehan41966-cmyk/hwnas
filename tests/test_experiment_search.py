from copy import deepcopy
import json
from pathlib import Path
import pytest

from experiment import ROOT,load_config,ResultStore,make_record,validate_record,verify_run
from space import decode,canonical_hash
from quant import Q
from hw import HardwareContext,evaluate
from search import select_stage,pareto_records


def test_plan_enumerates_real_space_without_importing_model_or_opening_dataset(tmp_path,monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*a,**kw):
        if name.split('.')[0] in {'torch','models','train'}:
            raise AssertionError('Planning cannot import ML executors')
        return original(name,*a,**kw)
    from scripts.search import plan
    monkeypatch.setattr(builtins,'__import__',guard)
    result=plan(ROOT/'configs/search.yaml',tmp_path/'plan')
    assert result['architectures']==72 and result['decisions']==1296
    assert result['dataset_opened'] is False and result['training_started'] is False
    rows=ResultStore(tmp_path/'plan').read()
    assert len(rows)==1296 and all(r['accuracy'] is None for r in rows)
    assert {r['hardware']['status'] for r in rows}=={'NEEDS_MEASUREMENT'}
    assert pareto_records(rows,load_config(ROOT/'configs/search.yaml'))==[]
    with pytest.raises(FileExistsError):plan(ROOT/'configs/search.yaml',tmp_path/'plan')


def unknown(g,cfg):
    q=Q(g)
    c=HardwareContext(cfg['hardware']['device'],200.,'reference_low','hls_report','UNMEASURED','operator_hls')
    run=dict(git=dict(commit='a'*40),config_hash=canonical_hash(cfg),source_manifest_hash='b'*64)
    return make_record(g,q,evaluate(g,q,c).to_dict(),run=run,seed=42,budget=40,stage='search',context=c.to_dict())


def test_unknown_hardware_never_promotes_and_wrong_identity_rejected():
    cfg=load_config(ROOT/'configs/search.yaml')
    rows=[unknown(decode(i),cfg) for i in range(72)]
    result=select_stage(rows,cfg,stage=0,previous=[r['arch_hash'] for r in rows])
    assert result['status']=='NEEDS_MEASUREMENT' and result['selected']==[]
    bad=deepcopy(rows[0]);bad['architecture_id']=rows[8]['architecture_id']
    with pytest.raises(ValueError,match='display/legacy'):validate_record(bad)
    bad=deepcopy(rows[0]);bad['hardware_context']['clock_mhz']=100.
    with pytest.raises(ValueError,match='context hash'):validate_record(bad)
    with pytest.raises(ValueError,match='Every previous'):
        select_stage(rows[:-1],cfg,stage=0,previous=[r['arch_hash'] for r in rows])


def test_unknown_tables_retain_missing_values_and_empty_pareto(tmp_path):
    from scripts.search import plan
    from scripts.tables import generate
    plan(ROOT/'configs/search.yaml',tmp_path/'run')
    result=generate(tmp_path/'run',tmp_path/'tables')
    assert result['pareto_rows']==0 and result['accuracy_rows']==216 and result['hardware_rows']==1296
    import csv
    with (tmp_path/'tables/table2_hardware.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert all(r['latency_us']=='' and r['power_w']=='' for r in rows)


def test_runtime_blocks_missing_hardware_before_any_dataset_or_model_access(tmp_path):
    from scripts.runtime import CandidateEvaluator
    cfg=load_config(ROOT/'configs/search.yaml')
    with pytest.raises(ValueError,match='before training'):
        CandidateEvaluator(cfg,{},tmp_path,config_path=ROOT/'configs/search.yaml',device='cpu')


def test_run_source_freeze_uses_dataset_index_without_reading_lfs_archives(tmp_path,monkeypatch):
    import experiment
    from experiment import create_run
    dataset=tmp_path/'datasets'
    (dataset/'UATD/raw').mkdir(parents=True)
    (dataset/'UATD/raw/archive.zip').write_bytes(b'fixture archive must not be read by source freeze')
    (dataset/'dataset_sources.json').write_text('{"schema_version":1}',encoding='utf-8')
    (tmp_path/'module.py').write_text('pass\n',encoding='utf-8')
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(experiment,'git_info',lambda:dict(commit='a'*40,branch='test',dirty=False))
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *a,**k:
        b'module.py\0datasets/dataset_sources.json\0datasets/UATD/raw/archive.zip\0')
    original_sha=experiment.sha_file
    def guarded(path):
        if str(path).endswith('archive.zip'):raise AssertionError('Planning must not read dataset archives')
        return original_sha(path)
    monkeypatch.setattr(experiment,'sha_file',guarded)
    _,run=create_run(ROOT/'configs/search.yaml',tmp_path/'run',command='fixture source freeze')
    assert 'datasets/dataset_sources.json' in run['source_files']
    assert 'datasets/UATD/raw/archive.zip' not in run['source_files']
