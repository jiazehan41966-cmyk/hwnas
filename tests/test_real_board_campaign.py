"""No network/device calls: fail-closed campaign sequencing and identity tests."""
from copy import deepcopy
from contextlib import nullcontext
import json
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.sonar_deploy import board_campaign as c
from tools.detector_preflight.run_graph_board import board_directory,result_upload_limit


@pytest.fixture
def plan(monkeypatch):
    project='/home/jiajia/experiments/real_test'
    remote_run=project+'/'+c.DEFAULT_RUN
    rows=[]
    for index,network in enumerate(c.NETWORKS):
        rows.append(dict(network=network,architecture_id=network+'_arch',checkpoint_sha256=str(index)*64,
            package_manifest_sha256='a'*64,graph_sha256='b'*64,accuracy='PASS',
            action=dict(allowed=True),build=remote_run+'/'+network+'/seed_42/hardware/graph_build',
            regression=remote_run+'/'+network+'/seed_42/hardware/numeric_regression',
            graph_session_available=True,board_complete=False,attempt_exists=False))
    state=dict(deployment_binding_sha256='c'*64,runs=rows,test_evaluated=False)
    manifest=dict(remote_project=project,remote_run=remote_run,networks=list(c.NETWORKS),
        candidates=[{k:r[k] for k in ('network','architecture_id','checkpoint_sha256','package_manifest_sha256','graph_sha256')} for r in rows],
        deployment_binding_sha256='c'*64,node_timeout_seconds=900,board_ip='192.168.31.217')
    monkeypatch.setattr(c,'transport',lambda:SimpleNamespace(REMOTE=project,PYTHON='/venv/python',HOST='test@host'))
    monkeypatch.setattr(c,'snapshot',lambda *args:deepcopy(state))
    return manifest,state


def test_graph_build_paths_do_not_collide():
    paths=[f'/home/jiajia/experiments/root/{n}/seed_42/hardware/graph_build' for n in c.NETWORKS]
    assert len({board_directory(p) for p in paths})==3
    assert board_directory(paths[0])==board_directory(paths[0])
    assert board_directory(paths[0],'retry1')!=board_directory(paths[0])


def test_b1_return_capacity_does_not_assume_compression():
    m=dict(plan=dict(outputs=[dict(real_bytes=21*s*s,tail_zero_bytes=0) for s in (160,80,40,20)]))
    assert result_upload_limit(m)>150*1024*1024
    m['plan']['outputs'][0]['real_bytes']=3*1024**3
    with pytest.raises(ValueError,match='capacity'):result_upload_limit(m)


@pytest.mark.parametrize('path',['relative/root','/etc/results','/home/jiajia/experiments/real_test/results/sonar_real_int8_deployment_v1/../other'])
def test_unbound_remote_roots_rejected(path,plan):
    with pytest.raises(ValueError):c.validate_remote_run(path,plan[0]['remote_project'])


def test_prepare_is_immutable_and_has_no_programming(tmp_path,plan):
    manifest,state=plan
    first,_=c.prepare(tmp_path,manifest['remote_run'],c.NETWORKS)
    second,_=c.prepare(tmp_path,manifest['remote_run'],c.NETWORKS)
    assert first==second and first['one_attempt_per_network'] and first['node_timeout_seconds']==900
    with pytest.raises(ValueError,match='Frozen campaign differs'):
        c.prepare(tmp_path,manifest['remote_run'],c.NETWORKS,node_timeout=180)


@pytest.mark.parametrize('key',['architecture_id','checkpoint_sha256','package_manifest_sha256','graph_sha256'])
def test_identity_changed_rejected(plan,key):
    manifest,state=plan;state['runs'][0][key]='changed'
    with pytest.raises(ValueError,match='identity changed'):c._verify_snapshot(manifest,state,'D')


@pytest.mark.parametrize('key,value',[('deployment_binding_sha256','changed'),('test_evaluated',True)])
def test_root_or_seal_changed_rejected(plan,key,value):
    manifest,state=plan;state[key]=value
    with pytest.raises(ValueError,match='identity changed'):c._verify_snapshot(manifest,state,'D')


@pytest.mark.parametrize('field,value,expected',[
    ('action',dict(allowed=False,reason='route pending'),'WAITING_STAGE_EVIDENCE'),
    ('attempt_exists',True,'STOP_EXISTING_ATTEMPT_REVIEW_REQUIRED'),
    ('accuracy','STOP_ACCURACY_BELOW_D','STOP_ACCURACY_BELOW_D'),
    ('board_complete',True,'ALREADY_VERIFIED')])
def test_blocked_or_complete_never_dispatches(tmp_path,plan,monkeypatch,field,value,expected):
    manifest,state=plan;state['runs'][0][field]=value
    monkeypatch.setattr(c,'_remote_command',lambda *a,**k:pytest.fail('Must not dispatch'))
    monkeypatch.setattr(c,'claim_attempt',lambda *a,**k:pytest.fail('Must not claim board'))
    assert c.run_network(tmp_path,manifest,'D')['status']==expected


def test_missing_frozen_session_interface_never_claims(tmp_path,plan):
    manifest,state=plan;state['runs'][0]['graph_session_available']=False
    with pytest.raises(ValueError,match='GraphSession'):c.run_network(tmp_path,manifest,'D')


@pytest.mark.parametrize('ready',[False,True])
def test_hardware_failure_stops_immediately_without_attempt_even_with_gate_wait(tmp_path,plan,monkeypatch,ready):
    manifest,state=plan
    state['hardware_execution']=dict(status='FAILED',reason='route timing did not meet constraint')
    state['hardware_execution_sha256']='f'*64
    state['runs'][0]['action']['allowed']=ready
    monkeypatch.setattr(c,'_remote_command',lambda *a,**k:pytest.fail('No hardware dispatch after FAILED'))
    monkeypatch.setattr(c,'claim_attempt',lambda *a,**k:pytest.fail('No board attempt after FAILED'))
    monkeypatch.setattr(c.time,'sleep',lambda *a:pytest.fail('Do not wait90min after a known failure'))
    result=c.run_network(tmp_path,manifest,'D',gate_wait_seconds=5400)
    assert result['status']=='STOP_HARDWARE_FAILURE' and result['hardware_execution_sha256']=='f'*64


def setup_fake_execution(monkeypatch,tmp_path,state):
    calls=[]
    def remote(project,argv,**kwargs):
        label='frames' if 'tools/sonar_deploy/frames.py' in argv else 'final_verify'
        calls.append(label)
        if label=='final_verify':state['runs'][0]['board_complete']=True
    def fetch(manifest,row,folder):
        calls.append('fetch');folder.mkdir(parents=True)
        for name in ('graph.json','frames.json','manifest.json'):(folder/name).write_text('{}')
    def claim(manifest,row,output):
        calls.append('claim');state['runs'][0]['attempt_exists']=True
        c.a.atomic_json(output/'D/ATTEMPT.json',dict(network='D'))
    def local(argv,log):
        calls.append('train8_valid99');assert argv[-2:]==['--node-timeout','900']
    monkeypatch.setattr(c,'_remote_command',remote)
    monkeypatch.setattr(c,'fetch_small_build',fetch)
    monkeypatch.setattr(c,'claim_attempt',claim)
    monkeypatch.setattr(c,'_local_command',local)
    monkeypatch.setattr(c,'upload_evidence',lambda *a:calls.append('upload'))
    monkeypatch.setattr(c,'e2e_transaction',lambda *a:calls.append('e2e'))
    from tools.sonar_deploy import board_result
    monkeypatch.setattr(board_result,'validate_actual_frames',lambda *a:calls.append('verify107'))
    return calls


def test_full107_verified_before_e2e_and_final_board_gate(tmp_path,plan,monkeypatch):
    manifest,state=plan;calls=setup_fake_execution(monkeypatch,tmp_path,state)
    assert c.run_network(tmp_path,manifest,'D')['status']=='COMPLETE_VERIFIED'
    assert calls==['frames','fetch','claim','train8_valid99','verify107','upload','e2e','final_verify']


def test_board_failure_retained_stops_e2e_and_no_auto_retry(tmp_path,plan,monkeypatch):
    manifest,state=plan;calls=setup_fake_execution(monkeypatch,tmp_path,state)
    def fail(*args):raise RuntimeError('integer mismatch')
    monkeypatch.setattr(c,'_local_command',fail)
    with pytest.raises(RuntimeError,match='integer mismatch'):c.run_network(tmp_path,manifest,'D')
    assert c.a.read_json(tmp_path/'D/STATUS.json')['status']=='FAILED_RETAINED_NO_AUTORETRY'
    assert 'e2e' not in calls
    assert c.run_network(tmp_path,manifest,'D')['status']=='STOP_EXISTING_ATTEMPT_REVIEW_REQUIRED'
    assert calls==['frames','fetch','claim']


def test_incomplete107_does_not_advance(tmp_path,plan,monkeypatch):
    manifest,state=plan;calls=setup_fake_execution(monkeypatch,tmp_path,state)
    from tools.sonar_deploy import board_result
    def missing(*a):raise ValueError('requires train8 and valid99')
    monkeypatch.setattr(board_result,'validate_actual_frames',missing)
    with pytest.raises(ValueError,match='train8'):c.run_network(tmp_path,manifest,'D')
    assert 'e2e' not in calls and 'upload' not in calls


def test_stop_helper_has_no_null_byte_and_verifies_pid_owner(monkeypatch):
    code=[]
    monkeypatch.setattr(c,'remote_json',lambda text,**k:code.append(text))
    c._stop_transfer('/project',dict(pid=123,start_ticks='456',directory='/isolated/path'))
    compile(code[0],'<remote-stop>','exec')
    assert '\0' not in code[0] and 'start_ticks' not in code[0]  # literal expected start-time is frozen in command
    assert "ticks!='456'" in code[0] and "'/isolated/path'.encode() not in args" in code[0]


def test_remote_command_uses_python_argv_without_shell(monkeypatch):
    code=[];monkeypatch.setattr(c,'remote_json',lambda text,**k:code.append(text))
    c._remote_command('/project',['/python','script.py','argument with spaces'])
    compile(code[0],'<remote-command>','exec')
    assert 'shell=True' not in code[0] and "'argument with spaces'" in code[0]


def rpc_fixture(tmp_path,status='PASS'):
    rows=[dict(call_index=i,graph_seconds=.1,input_graph_capture_seconds=.2,board_clock_mhz=200.,test_evaluated=False) for i in range(11)]
    identity=dict(graph_sha256='a'*64,package_manifest_sha256='b'*64,board_manifest_sha256='c'*64)
    c.a.atomic_json(tmp_path/'e2e.json',dict(status='PASS',rows=rows,**identity))
    c.a.atomic_json(tmp_path/'e2e_rpc_session.json',dict(status=status,successful_calls=11,weights_resident=True,test_evaluated=False,rows=rows,**identity))


def test_rpc_host_pass_board_quarantine_is_rejected(tmp_path):
    rpc_fixture(tmp_path,'QUARANTINED')
    with pytest.raises(ValueError,match='release buffers'):c.validate_rpc_completion(tmp_path,0)


def test_rpc_cell_failure_is_rejected_after_preserving_session(tmp_path):
    rpc_fixture(tmp_path)
    with pytest.raises(RuntimeError,match='cell failed'):c.validate_rpc_completion(tmp_path,1)
    assert (tmp_path/'e2e_rpc_session.json').exists()


def test_rpc_pass_requires_corresponding_eleven_responses(tmp_path):
    rpc_fixture(tmp_path);c.validate_rpc_completion(tmp_path,0)
    value=c.a.read_json(tmp_path/'e2e.json');value['rows'][5]['graph_seconds']=.9
    c.a.atomic_json(tmp_path/'e2e.json',value)
    with pytest.raises(ValueError,match='differs'):c.validate_rpc_completion(tmp_path,0)


@pytest.mark.parametrize('version',['serial_ddr_graph_v2_requant','serial_ddr_graph_v2_requant_timing02'])
def test_driver_accepts_only_bound_requant_implementations(version):
    from tools.detector_preflight.pynq_graph_runner import validate_real_executor
    value=dict(executor_version=version,requant_pipeline='off')
    assert validate_real_executor(dict(executor=value),dict(implementation=deepcopy(value)))==value
    with pytest.raises(ValueError,match='differs'):
        validate_real_executor(dict(executor=value),dict(implementation={**value,'extra':'different'}))


@pytest.mark.parametrize('value',[dict(executor_version='unknown'),dict(executor_version='serial_ddr_graph_v2_requant_timing02',requant_pipeline='on')])
def test_driver_rejects_unbound_or_wrong_timing02(value):
    from tools.detector_preflight.pynq_graph_runner import validate_real_executor
    with pytest.raises(ValueError):validate_real_executor(dict(executor=value),dict(implementation=deepcopy(value)))


def test_cli_all_stops_at_d_pending(tmp_path,plan,monkeypatch):
    manifest,_=plan;c.a.atomic_json(tmp_path/'CAMPAIGN_MANIFEST.json',manifest)
    seen=[]
    def run(output,protocol,network,**kwargs):
        seen.append(network);return dict(status='WAITING_STAGE_EVIDENCE',network=network)
    monkeypatch.setattr(c,'run_network',run)
    import hwnas_fpga.metrics_ledger as ledger
    monkeypatch.setattr(ledger,'_ledger_lock',lambda *a,**k:nullcontext())
    monkeypatch.setattr(sys,'argv',['campaign','run','--network','all','--output',str(tmp_path)])
    assert c.main()==2 and seen==['D']


def test_cli_rejects_run_root_different_from_frozen_manifest(tmp_path,plan,monkeypatch):
    manifest,_=plan;c.a.atomic_json(tmp_path/'CAMPAIGN_MANIFEST.json',manifest)
    monkeypatch.setattr(sys,'argv',['campaign','run','--output',str(tmp_path),'--remote-run',manifest['remote_run']+'_route02'])
    with pytest.raises(ValueError,match='independent output'):c.main()


def test_new_route_root_uses_separate_frozen_campaign(tmp_path,plan):
    manifest,_=plan
    new=manifest['remote_run']+'_route02'
    actual,_=c.prepare(tmp_path,new,c.NETWORKS)
    assert actual['remote_run']==new


def rpc_tar(path,files):
    with tarfile.open(path,'w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data)
            archive.addfile(member,io.BytesIO(data))


def test_rpc_return_preserves_quarantine_and_power_with_hashes(tmp_path):
    archive=tmp_path/'return.tar.gz'
    rpc_tar(archive,{'e2e_rpc_session.json':b'{"status":"QUARANTINED","successful_calls":11}',
        'power/power.json':b'{"status":"NOT_MEASURED"}','power/power_samples.csv':b'watts\n1.0\n'})
    output=tmp_path/'evidence';result=c.extract_rpc_return(archive,output)
    assert result['status']=='QUARANTINED' and len(result['files'])==3
    assert result['files']['power/power_samples.csv']==c.a.sha256(output/'power/power_samples.csv')
    with pytest.raises(ValueError,match='replace'):c.extract_rpc_return(archive,output)


@pytest.mark.parametrize('name',['../escaping.json','unknown.txt','power/../../outside','/absolute'])
def test_rpc_return_rejects_unregistered_paths(tmp_path,name):
    archive=tmp_path/'return.tar.gz'
    rpc_tar(archive,{'e2e_rpc_session.json':b'{}',name:b'bad'})
    with pytest.raises(ValueError,match='unregistered'):c.extract_rpc_return(archive,tmp_path/'evidence')


def test_rpc_heartbeat_keeps_fpga_body_on_main_thread_and_joins():
    scope={'expected_thread':threading.get_ident()}
    exec(c.with_rpc_heartbeat("assert threading.get_ident()==expected_thread\nbody_complete=True\n"),scope)
    assert scope['body_complete'] and scope['_e2e_heartbeat_stop'].is_set()
    assert not scope['_e2e_heartbeat_worker'].is_alive()


def test_rpc_heartbeat_is_stopped_on_body_failure():
    scope={}
    with pytest.raises(RuntimeError,match='device retained'):
        exec(c.with_rpc_heartbeat("raise RuntimeError('device retained')\n"),scope)
    assert scope['_e2e_heartbeat_stop'].is_set() and not scope['_e2e_heartbeat_worker'].is_alive()


def test_rpc_heartbeat_emits_output_while_main_thread_waits():
    # Scale only Event.wait's 10s interval; the production wrapper retains 10s.
    original_event=threading.Event
    seen=original_event();finished=original_event();intervals=[]
    class Event:
        def wait(self,seconds):
            intervals.append(seconds);return finished.wait(.01)
        def set(self):finished.set()
    fake_threading=SimpleNamespace(Event=Event,Thread=threading.Thread)
    def custom_import(name,*args,**kwargs):
        assert name=='threading';return fake_threading
    def output(message,**kwargs):
        assert message=='E2E_RUNNING' and kwargs.get('flush');seen.set()
    scope={'__builtins__':{'__import__':custom_import,'print':output},'seen':seen}
    exec(c.with_rpc_heartbeat("assert seen.wait(1)\n"),scope)
    assert seen.is_set() and intervals and all(value==10 for value in intervals)
    assert not scope['_e2e_heartbeat_worker'].is_alive()
