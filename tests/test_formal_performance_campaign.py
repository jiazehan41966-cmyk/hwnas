import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sonar_deploy import performance_target as t, performance_campaign as c
from tools.sonar_deploy.performance_admission import check


def test_generated_board_program_compiles():
    code=c.board_source('/home/xilinx/old','/home/xilinx/new','a'*64)
    compile(code,'board.py','exec')
    assert 'native.run(root,adapter)' in code and 'rpc.serve_board(' in code
    assert 'execute_frames' not in code


def test_incomplete_smoke_cannot_admit_performance(tmp_path):
    (tmp_path/'VERIFIED.json').write_text(json.dumps(dict(status='PENDING',test_evaluated=False)))
    with pytest.raises(ValueError,match='Verified single-train'):check(tmp_path,'D',tmp_path)


def test_native_exactly_eleven_calls_no_layer_trace(tmp_path, monkeypatch):
    root=tmp_path/'board';root.mkdir();adapter=tmp_path/'adapter';adapter.mkdir()
    image=np.array([1,2],dtype=np.int8);digest=hashlib.sha256(image.tobytes()).hexdigest()
    manifest={k:'a'*64 for k in ['board_manifest_sha256','deployment_binding_sha256','package_manifest_sha256','graph_sha256']}
    manifest.update(network='D',power_monitor={'source_sha256':'a'*64},input_sha256=digest,heads=[dict(level='P3',sha256=digest)])
    calls=[]
    session=SimpleNamespace(source={'plan':{'allocations':{'image':{'shape':[1,1,2],'offset':0}}}},
        arena=image,clock_mhz=187.5,owned=[],overlay=None,
        infer=lambda x:(calls.append('infer') or ({'P3':image.tobytes()},dict(graph_seconds=.1,input_graph_capture_seconds=.2,board_clock_mhz=187.5))),
        close=lambda:calls.append('close'))
    driver=SimpleNamespace(GraphSession=lambda *a,**k:session,QUARANTINE=[])
    monitor=SimpleNamespace(start=lambda:None,begin_inference=lambda i:None,end_inference=lambda i:None,
        finish=lambda:dict(status='NOT_MEASURED'),close=lambda:None)
    power=SimpleNamespace(IioPowerMonitor=lambda *a,**k:monitor)
    rpc=SimpleNamespace(validate_manifest=lambda m:m,read=lambda p:manifest if p.name=='benchmark_manifest.json' else {'files':{}},
        sha_file=lambda p:'a'*64,sha_bytes=lambda b:hashlib.sha256(b).hexdigest(),
        write=lambda p,v:p.write_text(json.dumps(v)),_summary=lambda v:{'samples':len(v)})
    monkeypatch.setattr(t,'module',lambda name,path:rpc if path.name=='board_e2e.py' else power if path.name=='power.py' else driver)
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'boot' if p.as_posix()=='/proc/sys/kernel/random/boot_id' else original(p,*a,**k))
    t.run(root,adapter)
    result=json.loads((root/'native_results/completion.json').read_text())
    assert calls==['infer']*11+['close']
    assert result['graph_latency']['samples']==10 and result['warmup']==1
    assert result['test_evaluated'] is False and result['throughput_fps']>0
