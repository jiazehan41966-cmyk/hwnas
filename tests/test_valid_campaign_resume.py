"""Offline recovery gates: never duplicate an already completed FPGA transaction."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_deploy import valid_campaign as v
from tools.sonar_deploy import board_result


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(v.c, 'NETWORKS', ('D',))
    project = '/home/jiajia/experiments/unit'
    run = tmp_path/'original'
    run.mkdir()
    performance = tmp_path/'performance'
    v.a.atomic_json(performance/'STATUS.json', {'status':'FORMAL_PERFORMANCE_COMPLETE'})
    v.a.atomic_json(performance/'D/PERFORMANCE_REPORT.json', {'status':'PERFORMANCE_PASS'})
    binding = 'b'*64
    row = dict(network='D', architecture_id='D_arch', checkpoint_sha256='c'*64,
               package_manifest_sha256='p'*64, graph_sha256='g'*64,
               build=project+'/original_build', regression=project+'/regression',
               action={'allowed':True})
    state = dict(deployment_binding_sha256=binding, runs=[row], test_evaluated=False)
    protocol = dict(order=['D'], seed=42, train_frames=8, valid_frames=99,
                    historical_driver_benchmark=dict(warmup=1, repetitions=10),
                    deployment_binding_sha256=binding, source_sha256='historical_source',
                    test_evaluated=False,
                    performance_reports={'D':v.a.sha256(performance/'D/PERFORMANCE_REPORT.json')})
    for name, value in [('PROTOCOL.json',protocol), ('UPSTREAM_SNAPSHOT.json',state),
                        ('STATUS.json',{'network':'D','status':'BOARD_TRAIN8_VALID99_RUNNING'}),
                        ('FAILURE.json',{'status':'FAILED_NO_AUTORETRY','error':'Disconnected'})]:
        v.a.atomic_json(run/name, value)
    folder = run/'D/graph_build'
    folder.mkdir(parents=True)
    files = {}
    for name in ('h2_graph.cpp', 'numeric_v1.h', 'pynq_graph_runner.py', 'graph.json'):
        (folder/name).write_text('{}')
        files[name] = v.a.sha256(folder/name)
    v.a.atomic_json(folder/'manifest.json',dict(real_weights_bound=True,graph_sha256=row['graph_sha256'], files=files))
    v.a.atomic_json(folder/'frames.json',{})
    remote = project+'/results/sonar_real_int8_deployment_v1/'+run.name+'/D/graph_build'
    v.a.atomic_json(folder/'remote_launch.json',dict(remote=remote, network='D', deployment_binding_sha256=binding, test_evaluated=False))
    (folder/'board_evidence').mkdir()
    tx = SimpleNamespace(REMOTE=project, PYTHON='/python', HOST='unit@host')
    monkeypatch.setattr(v.c, 'transport', lambda:tx)
    monkeypatch.setattr(v.c, 'snapshot',lambda *args:deepcopy(state))
    monkeypatch.setattr(v.c,'_local_command',lambda *args:pytest.fail('Recovered board must never execute again'))
    monkeypatch.setattr(v.c,'_remote_command',lambda *args,**kw:pytest.fail('Recovered frames must not be regenerated'))
    return run, performance, state, protocol


def test_completed_board_recovered_without_reexecution_preserves_original_failure(tmp_path,monkeypatch):
    run, performance, state, protocol = prepare(tmp_path,monkeypatch)
    failure_hash = v.a.sha256(run/'FAILURE.json')
    calls = []
    monkeypatch.setattr(board_result,'validate_actual_frames',lambda *args:calls.append('validate107'))
    monkeypatch.setattr(v,'upload_or_verify_evidence',lambda *args:calls.append('upload_or_verify'))
    monkeypatch.setattr(v.c,'remote_json',lambda *args,**kwargs:calls.append('evaluate') or {})
    monkeypatch.setattr(v.c,'_scp',lambda *args:calls.append('fetch_report'))
    v.run(run,performance,resume=True)
    assert calls == ['validate107','validate107','upload_or_verify','evaluate','fetch_report']
    assert v.a.read_json(run/'STATUS.json')['status']=='ALL_VALID99_COMPLETE'
    assert v.a.sha256(run/'FAILURE.json')==failure_hash
    claim = v.a.read_json(run/'RESUME_CLAIM.json')
    assert claim['original_source_sha256']=='historical_source'
    assert claim['resume_source_sha256']==v.a.sha256(v.__file__)
    with pytest.raises(FileExistsError):v.run(run,performance,resume=True)


def test_partial_board_transaction_never_retried(tmp_path,monkeypatch):
    run,performance,_,_ = prepare(tmp_path,monkeypatch)
    def incomplete(*args):raise ValueError('missing/extra frames: requires train8 and valid99')
    monkeypatch.setattr(board_result,'validate_actual_frames',incomplete)
    monkeypatch.setattr(v.c,'remote_json',lambda *args,**kwargs:pytest.fail('No upload/evaluation of partial evidence'))
    with pytest.raises(ValueError,match='missing/extra frames'):
        v.run(run,performance,resume=True)


@pytest.mark.parametrize('field,value',[('seed',43),('train_frames',7),('valid_frames',98),
    ('test_evaluated',True),('deployment_binding_sha256','changed'),('order',['B1','D'])])
def test_changed_protocol_rejected_before_claim(tmp_path,monkeypatch,field,value):
    run,performance,state,protocol = prepare(tmp_path,monkeypatch)
    protocol[field]=value
    v.a.atomic_json(run/'PROTOCOL.json',protocol)
    with pytest.raises(ValueError,match='protocol changed'):
        v.run(run,performance,resume=True)
    assert not (run/'RESUME_CLAIM.json').exists()


def test_checkpoint_changed_rejected_before_claim(tmp_path,monkeypatch):
    run,performance,state,protocol = prepare(tmp_path,monkeypatch)
    state['runs'][0]['checkpoint_sha256']='other'
    with pytest.raises(ValueError,match='candidate identity changed'):
        v.run(run,performance,resume=True)
    assert not (run/'RESUME_CLAIM.json').exists()


def test_recovered_source_tampering_rejected(tmp_path,monkeypatch):
    run,performance,_,_ = prepare(tmp_path,monkeypatch)
    (run/'D/graph_build/h2_graph.cpp').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):
        v.run(run,performance,resume=True)


def test_existing_remote_evidence_is_not_uploaded_twice(tmp_path,monkeypatch):
    folder=tmp_path/'build';(folder/'board_evidence').mkdir(parents=True)
    (folder/'board_evidence/completion.json').write_text('{}')
    tx=SimpleNamespace(REMOTE='/remote')
    monkeypatch.setattr(v.c,'remote_json',lambda *args:dict(status='IDENTICAL_EXISTING_EVIDENCE'))
    monkeypatch.setattr(v.c,'upload_evidence',lambda *args:pytest.fail('Must not overwrite identical evidence'))
    v.upload_or_verify_evidence(tx,{'build':'/remote/results/D/graph_build'},folder)
