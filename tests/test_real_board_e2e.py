from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.sonar_deploy import board_e2e as b


@pytest.fixture
def fixture():
    image=np.zeros((1,640,640),dtype=np.int8)
    heads={level:bytes(21*side*side) for level,side in [('P3',80),('P4',40),('P5',20)]}
    manifest=dict(schema_version=b.SCHEMA,network='D',seed=42,
        deployment_binding_sha256='a'*64,package_manifest_sha256='b'*64,graph_sha256='c'*64,
        board_manifest_sha256='d'*64,calibration_sha256='e'*64,
        rpc_source_sha256=b.sha_file(b.__file__),
        input_shape=[1,640,640],input_sha256=b.sha_bytes(image.tobytes()),
        sample=dict(split='train',calibration_index=0,image='train/images/first.png'),
        heads=[dict(level=level,shape=[21,side,side],sha256=b.sha_bytes(heads[level]))
               for level,side in [('P3',80),('P4',40),('P5',20)]],
        warmup=1,repetitions=10,batch_size=1,test_evaluated=False)
    return manifest,image,heads


class Session:
    def __init__(self,heads,fail=False):
        self.heads=heads;self.calls=0;self.closed=False;self.threads=[];self.fail=fail
    def infer(self,image):
        self.calls+=1;self.threads.append(threading.get_ident())
        if self.fail:raise TimeoutError('fake device timeout')
        return self.heads,dict(graph_seconds=.000001,input_graph_capture_seconds=.000002,board_clock_mhz=200.)
    def close(self):self.closed=True


def background(server,**kwargs):
    errors=[];results=[]
    def run():
        try:results.append(b.serve_session(server,**kwargs))
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=run,daemon=True);thread.start()
    return thread,errors,results


def test_real_loopback_transport_eleven_calls_single_thread_then_stops(fixture,tmp_path):
    manifest,image,heads=fixture
    session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1')
    thread,errors,results=background(server,idle_timeout=2,output=tmp_path/'rpc.json')
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest,timeout=2)
    for index in range(11):
        actual,timing=client.infer(image,index)
        assert actual==heads and timing['call_index']==index
    thread.join(3)
    assert not thread.is_alive() and not errors
    assert session.calls==11 and len(set(session.threads))==1 and session.closed
    assert results[0]['successful_calls']==11
    with pytest.raises(urllib.error.URLError):client.infer(image,11)


def test_continuous_clock_includes_preprocess_transfer_and_decode(fixture):
    manifest,image,heads=fixture
    clock=[0.];order=[]
    def preprocess():
        order.append('preprocess');clock[0]+=2;return image
    def infer(image,index):
        order.append('infer');clock[0]+=4
        return heads,dict(graph_seconds=1.,input_graph_capture_seconds=2.,board_clock_mhz=200.)
    def decode(heads):
        order.append('decode');clock[0]+=3;return [{'scores':[.1]}]
    result=b.benchmark(preprocess,decode,infer,manifest,clock=lambda:clock[0])
    assert order==['preprocess','infer','decode']*11
    assert len(result['rows'])==11 and len(result['samples'])==10
    assert result['timing']['end_to_end_seconds']==[9.]*10
    assert result['timing']['graph_seconds']==[1.]*10
    assert result['board_latency_ms']==9000
    assert result['throughput_fps']==pytest.approx(1/9)
    assert result['rows'][0]['warmup'] and not any(row['warmup'] for row in result['samples'])


@pytest.mark.parametrize('key,value', [('network','OTHER'),('seed',43),('warmup',0),('repetitions',11),
    ('batch_size',4),('test_evaluated',True),('input_shape',[3,640,640]),('graph_sha256','invalid')])
def test_manifest_contract_rejects_changes(fixture,key,value):
    manifest,_,_=fixture;manifest[key]=value
    with pytest.raises(ValueError):b.validate_manifest(manifest)


def test_test_image_never_allowed(fixture):
    manifest,_,_=fixture;manifest['sample']['split']='test'
    with pytest.raises(ValueError,match='train image'):b.validate_manifest(manifest)


def test_three_four_level_identity_and_order(fixture):
    manifest,_,_=fixture
    manifest['heads'].reverse()
    with pytest.raises(ValueError,match='order'):b.validate_manifest(manifest)
    manifest['heads'].reverse();manifest['network']='B1'
    with pytest.raises(ValueError,match='order'):b.validate_manifest(manifest)
    manifest['heads'].insert(0,dict(level='P2',shape=[21,160,160],sha256='a'*64))
    assert b.validate_manifest(manifest)


def test_device_timeout_quarantines_without_releasing_buffer(fixture):
    manifest,image,heads=fixture;session=Session(heads,fail=True)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1')
    thread,errors,_=background(server,idle_timeout=2)
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest,timeout=2)
    with pytest.raises(urllib.error.HTTPError):client.infer(image,0)
    thread.join(3)
    assert isinstance(errors[0],TimeoutError) and session in b.QUARANTINE and not session.closed


def test_idle_timeout_retains_session_and_service_closes(fixture):
    manifest,_,heads=fixture;session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1')
    thread,errors,_=background(server,idle_timeout=.03)
    thread.join(2)
    assert not thread.is_alive() and isinstance(errors[0],TimeoutError)
    assert session.calls==0 and not session.closed and session in b.QUARANTINE


def test_unauthorized_source_never_reaches_infer(fixture):
    manifest,image,heads=fixture;session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest)
    thread,errors,_=background(server,idle_timeout=.1)
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest,timeout=1)
    with pytest.raises(Exception):client.infer(image,0)
    thread.join(2)
    assert session.calls==0 and isinstance(errors[0],TimeoutError)


@pytest.mark.parametrize('header,value',[('X-Call-Index','1'),('X-Package-Sha256','f'*64),('X-Input-Sha256','f'*64)])
def test_request_identity_or_replay_never_executes(fixture,header,value):
    manifest,image,heads=fixture;session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1')
    thread,errors,_=background(server,idle_timeout=2)
    headers={**b._binding_headers(manifest),'X-Call-Index':'0'};headers[header]=value
    req=urllib.request.Request('http://127.0.0.1:'+str(server.server_port)+'/infer',data=image.tobytes(),headers=headers)
    # A rejected header can close before a large unread body drains; Windows
    # legitimately reports connection-aborted instead of the HTTP error body.
    with pytest.raises((urllib.error.HTTPError,ConnectionError)):urllib.request.urlopen(req,timeout=2)
    thread.join(3)
    assert session.calls==0 and isinstance(errors[0],ValueError) and not session.closed


def test_output_mismatch_stops_without_fake_board_pass(fixture):
    manifest,image,heads=fixture;heads['P3']=b'x'+heads['P3'][1:]
    session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1')
    thread,errors,_=background(server,idle_timeout=2)
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest,timeout=2)
    with pytest.raises(urllib.error.HTTPError):client.infer(image,0)
    thread.join(3)
    assert 'Raw head differs' in str(errors[0]) and session in b.QUARANTINE


@pytest.mark.parametrize('url',['https://127.0.0.1:12','http://user:password@127.0.0.1:12','http://example.com','http://8.8.8.8','http://127.0.0.1/path'])
def test_not_an_arbitrary_remote_service(fixture,url):
    with pytest.raises(ValueError):b.GraphClient(url,fixture[0])


def test_board_entry_refuses_disposable_interpreter(fixture):
    with pytest.raises(RuntimeError,match='retained main-thread'):
        b.serve_board('unused','unused')


def test_host_copies_bound_manifest_and_never_overwrites_measurement(fixture,tmp_path,monkeypatch):
    manifest,image,heads=fixture
    (tmp_path/'DEPLOYMENT_BINDING.json').write_text('{}')
    manifest['deployment_binding_sha256']=b.sha_file(tmp_path/'DEPLOYMENT_BINDING.json')
    manifest_path=tmp_path/'source_manifest.json';b.write(manifest_path,manifest)
    result={'package_manifest_sha256':manifest['package_manifest_sha256']}
    decode=lambda heads:[{'scores':[.1]}]
    monkeypatch.setattr(b,'_host_components',lambda *args:({},result,{}, {},lambda:image,decode))
    session=Session(heads)
    class Client:
        def __init__(self,*args):pass
        def infer(self,image,index):return session.infer(image)
    monkeypatch.setattr(b,'GraphClient',Client)
    output=tmp_path/'board_evidence/e2e.json'
    report=b.run_host(tmp_path,'D',manifest_path,'http://127.0.0.1:1',output)
    assert report['status']=='PASS' and session.calls==11
    assert (output.parent/'benchmark_manifest.json').read_bytes()==manifest_path.read_bytes()
    assert b.read(output.parent/'e2e_progress.json')['calls']==11
    assert report['benchmark_manifest_sha256']==b.sha_file(manifest_path)
    with pytest.raises(ValueError,match='already exists'):
        b.run_host(tmp_path,'D',manifest_path,'http://127.0.0.1:1',output)
    assert session.calls==11


def test_driver_verified_before_import(fixture,tmp_path,monkeypatch):
    from types import SimpleNamespace
    manifest,_,_=fixture
    monkeypatch.setitem(sys.modules,'IPython',SimpleNamespace(get_ipython=lambda:SimpleNamespace(kernel=True)))
    driver=tmp_path/'pynq_graph_runner.py';driver.write_text('raise AssertionError("must not execute")')
    b.write(tmp_path/'board_manifest.json',{'files':{'pynq_graph_runner.py':'f'*64}})
    manifest['board_manifest_sha256']=b.sha_file(tmp_path/'board_manifest.json')
    path=tmp_path/'benchmark_manifest.json';b.write(path,manifest)
    with pytest.raises(ValueError,match='before driver import'):
        b.serve_board(tmp_path,path)


def test_optional_power_markers_include_warmup_and_all_ten_calls(fixture,tmp_path):
    manifest,image,heads=fixture;events=[]
    class Power:
        def begin_inference(self,index):events.append(('begin',index))
        def end_inference(self,index):events.append(('end',index))
        def finish(self):events.append(('finish',None))
        def close(self):events.append(('close',None))
    class MeasuredSession(Session):
        def infer(self,value):
            events.append(('infer',self.calls));return super().infer(value)
    session=MeasuredSession(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1',power_monitor=Power())
    thread,errors,results=background(server,output=tmp_path/'e2e_rpc_session.json')
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest)
    for index in range(11):client.infer(image,index)
    thread.join(timeout=5)
    assert not errors and results[0]['status']=='PASS'
    assert events==[(name,index) for index in range(11) for name in ('begin','infer','end')]+[('finish',None),('close',None)]


def test_power_sensor_failure_is_not_measured_and_does_not_change_graph(fixture,tmp_path):
    manifest,image,heads=fixture
    class Power:
        def begin_inference(self,index):raise OSError('sensor unavailable')
        def end_inference(self,index):pass
        def finish(self):raise OSError('no samples')
        def close(self):pass
    session=Session(heads)
    server=b.BoundedGraphServer(('127.0.0.1',0),session,manifest,allowed_host='127.0.0.1',power_monitor=Power())
    thread,errors,results=background(server,output=tmp_path/'e2e_rpc_session.json')
    client=b.GraphClient('http://127.0.0.1:'+str(server.server_port),manifest)
    for index in range(11):client.infer(image,index)
    thread.join(timeout=5)
    assert not errors and results[0]['status']=='PASS' and session.calls==11
    assert b.read(tmp_path/'power/power.json')['status']=='NOT_MEASURED'


def test_power_monitor_contract_is_explicit_and_fixed(fixture):
    manifest,_,_=fixture
    manifest['power_monitor']=dict(enabled=True,source_sha256='f'*64,sample_rate_hz=10,idle_seconds=5,
        sensor_path='/sys/bus/iio/devices/iio:device1',scope='whole_board_12v_input')
    b.validate_manifest(manifest)
    manifest['power_monitor']['scope']='sum_of_all_rails'
    with pytest.raises(ValueError,match='sensor protocol'):b.validate_manifest(manifest)
