import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sonar_deploy import board_smoke as host
from tools.sonar_deploy import board_smoke_target as target


@pytest.mark.parametrize('bad', ['../escape', '/absolute', 'layers.json'])
def test_reject_unsafe_return(tmp_path, bad):
    archive=tmp_path/'return.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name in ['completion.json','layers.json','power.json',bad]:
            row=tarfile.TarInfo(name);row.size=2;tf.addfile(row,io.BytesIO(b'{}'))
    with pytest.raises(ValueError):host.extract_smoke(archive,tmp_path/'out')


def test_minimal_smoke_return_without_full_valid(tmp_path):
    archive=tmp_path/'return.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name in ['completion.json','layers.json','power.json']:
            row=tarfile.TarInfo(name);row.size=2;tf.addfile(row,io.BytesIO(b'{}'))
    host.extract_smoke(archive,tmp_path/'out')
    assert (tmp_path/'out/completion.json').exists()


@pytest.mark.parametrize('mismatch', [False,True])
def test_single_path_parity_and_quarantine(tmp_path, monkeypatch, mismatch):
    digest=hashlib.sha256(bytes([1,2])).hexdigest()
    binding=dict(test_evaluated=False,scope='single_fixed_train_image',files={},network='D',
                 deployment_binding_sha256='a'*64,package_manifest_sha256='b'*64,graph_sha256='c'*64)
    (tmp_path/'SMOKE_BINDING.json').write_text(json.dumps(binding))
    (tmp_path/'board_manifest.json').write_text('{}')
    (tmp_path/'expected_layers.bin').write_bytes(bytes([1,2]))
    words=np.zeros(32,dtype=np.int32);words[25]=2
    calls=[]
    session=SimpleNamespace(descriptors=[words],arena=np.array([1,3 if mismatch else 2],dtype=np.int8),
        source=dict(layers=[dict(node='head',output_sha256=digest)],input_sha256='d'*64,
                    plan=dict(outputs=[dict(level='P3',edge='head',arena_offset=0,real_bytes=2)])),
        manifest={'abi':{}},core=SimpleNamespace(read=lambda _:4),descriptor=None,
        timeout=900,clock_mhz=187.5,overlay=None,owned=[],failed=False,
        close=lambda:calls.append('close'))
    driver=SimpleNamespace(GraphSession=lambda *a,**k:session,QUARANTINE=[],
                           execute_node=lambda *a,**k:(calls.append('node') or .1))
    monitor=SimpleNamespace(start=lambda:dict(status='NOT_MEASURED'),error='no sensor',
                            close=lambda:None)
    power=SimpleNamespace(IioPowerMonitor=lambda *a,**k:monitor)
    monkeypatch.setattr(target,'load_module',lambda name,path:driver if name=='smoke_bound_driver' else power)
    original=target.Path.read_text
    monkeypatch.setattr(target.Path,'read_text',lambda p,*a,**k:'boot' if p.as_posix()=='/proc/sys/kernel/random/boot_id' else original(p,*a,**k))
    if mismatch:
        with pytest.raises(RuntimeError,match='Layer mismatch'):target.run(tmp_path)
        assert driver.QUARANTINE and 'close' not in calls
        assert not (tmp_path/'smoke_results/completion.json').exists()
    else:
        result=target.run(tmp_path)
        assert result['status']=='SINGLE_TRAIN_SMOKE_PASS'
        assert result['valid_evaluated'] is False and result['test_evaluated'] is False
        assert calls==['node','close']
        assert result['power']['status']=='NOT_MEASURED'


def test_dispatch_contains_no_full_driver_run():
    code=host.board_code({'network':'D','package_sha256':'a'*64},'/home/xilinx/smoke')
    compile(code,'board.py','exec')
    assert 'execute_frames' not in code and 'driver.run(' not in code
    assert '18774' in code and 'smoke_module.run(folder)' in code


def test_verified_smoke_records_measured_fields_before_completion(tmp_path):
    folder=tmp_path/'D/evidence';folder.mkdir(parents=True)
    digest=hashlib.sha256(b'ab').hexdigest()
    binding={'checkpoint_sha256':'a'*64}
    spec=dict(network='D',smoke=binding,source=dict(
        layers=[dict(node='head',output_sha256=digest)],
        plan=dict(outputs=[dict(level='P3',edge='head',real_bytes=2)])))
    result=dict(status='SINGLE_TRAIN_SMOKE_PASS',binding=binding,
        heads=[dict(level='P3',bytes=2,sha256=digest)],board_clock_mhz=187.5,
        graph_with_layer_checks_seconds=2,power=dict(status='NOT_MEASURED'))
    (folder/'completion.json').write_text(json.dumps(result))
    (folder/'layers.json').write_text(json.dumps([dict(node='head',output_sha256=digest,mismatch_count=0)]))
    (folder/'P3.bin').write_bytes(b'ab')
    host.verify_and_record(folder,spec)
    assert (folder/'METRICS.jsonl').is_file()
    assert json.loads((folder/'VERIFIED.json').read_text())['full_board_acceptance'] is False
