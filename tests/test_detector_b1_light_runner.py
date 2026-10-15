"""B1-light fail-closed queue, manifest, resume and equal-RNG contracts."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.detector_b1_light import queue_runner as queue
from tools.detector_b1_light import train


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def gates_root(tmp_path):
    write(tmp_path/'execution_manifest.json', {'test_evaluated':False})
    identity = dict(schema_version=1, seed=42, backbone_sha256='a'*64,
                    neck_sha256='b'*64, prediction_sha256='c'*64, tower_sha256='d'*64)
    for prototype in train.PROTOTYPES:
        folder = train.result_dir(tmp_path, prototype, 42, False)
        write(folder/'initialization_identity.json', identity)
        write(folder/'training_rng_identity.json', dict(seed=42, torch='same', numpy='same', python='same'))
        write(folder/'smoke_report.json', dict(status='PASS', binding=train.binding(tmp_path, prototype, 42, False),
            formal_eligible=False, test_evaluated=False, initialization_identity=identity,
            environment=dict(device='cpu'), train_batches=[dict(size=4),dict(size=4)],
            validation_smoke=dict(num_images=4)))
    return tmp_path


def test_queue_gate_binds_two_smokes_and_seed42_only(gates_root):
    result = queue.smoke_gate(gates_root, 'cpu')
    assert result['order'] == ['L1','L0'] and result['seeds'] == [42]
    assert result['namespace'] == 'sonar_detector_b1_light'
    assert result['no_automatic_hardware_or_nas_promotion']
    assert len(result['smoke_records']) == 2


@pytest.mark.parametrize('mutation', ['missing','failed','test','binding','device','valid_subset','one_batch','rng','backbone'])
def test_queue_rejects_missing_or_mismatched_smoke(gates_root, mutation):
    folder = train.result_dir(gates_root,'L0',42,False)
    path = folder/'smoke_report.json'
    record = json.loads(path.read_text())
    if mutation == 'missing': path.unlink()
    elif mutation == 'failed': record['status']='FAILED'
    elif mutation == 'test': record['test_evaluated']=True
    elif mutation == 'binding': record['binding']['namespace']='sonar_candidate_v0_2'
    elif mutation == 'device': record['environment']['device']='cuda'
    elif mutation == 'valid_subset': record['validation_smoke']['num_images']=3
    elif mutation == 'one_batch': record['train_batches'].pop()
    elif mutation == 'rng': write(folder/'training_rng_identity.json',dict(seed=99))
    elif mutation == 'backbone':
        record['initialization_identity']['backbone_sha256']='e'*64
        write(folder/'initialization_identity.json',record['initialization_identity'])
    if mutation != 'missing': write(path,record)
    with pytest.raises((ValueError,FileNotFoundError)):
        queue.smoke_gate(gates_root,'cpu')


def test_explicit_resume_requires_own_existing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError,match='own existing latest'):
        train.train(tmp_path,'L1',42,'cpu',True)


def test_existing_invocation_cannot_be_silently_overwritten(tmp_path):
    folder=train.result_dir(tmp_path,'L1',42)
    write(folder/'run_binding.json',{})
    with pytest.raises(RuntimeError,match='independent audit'):
        train.train(tmp_path,'L1',42,'cpu',False)
    (folder/'latest.pt').write_bytes(b'owned')
    with pytest.raises(FileExistsError,match='requires --resume'):
        train.train(tmp_path,'L1',42,'cpu',False)


def test_legacy_glob_additions_are_audit_not_dataset_identity():
    original=dict(datasets={'train':{'images':330}},historical_frozen_files_verified=42,
                  additional_glob_files=['old'],historical_source_gate='NOT_PASSED_ADDITIONAL_GLOB_FILES')
    changed=deepcopy(original); changed['additional_glob_files'].append('later_independent_deploy.py')
    assert train.input_identity(original)==train.input_identity(changed)
    changed['datasets']['train']['images']=329
    assert train.input_identity(original)!=train.input_identity(changed)


def test_actual_source_closure_includes_new_runner_model_and_old_helper():
    cfg,_=train.load_config()
    hashes=train.source_hashes(cfg)
    for relative in ('tools/detector_b1_light/train.py','tools/detector_b1_light/queue_runner.py',
                     'tools/detector_b1_light/model.py','tools/detector_v0_2/train.py',
                     'configs/detector_b1_light_training.yaml','configs/detector_b1_light_model.yaml',
                     'configs/search_space_sonar_v0_1.yaml'):
        assert hashes[relative]==train.sha(train.ROOT/relative)
    assert 'tools/detector_b1_light/remote.py' not in hashes
    assert train.PROTOTYPES==('L1','L0') and train.SEEDS==(42,43,44)


def test_imported_core_packages_are_present_in_isolated_source_archive():
    # The local checkout can hide a missing transitive package in the remote
    # archive. A fresh process avoids unrelated imports from other test files.
    code = '''from pathlib import Path
import sys
from tools.detector_b1_light import train
cfg, _ = train.load_config()
hashes = train.source_hashes(cfg)
for name, module in tuple(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if not path or not name.startswith('hwnas_fpga'):
        continue
    relative = Path(path).resolve().relative_to(train.ROOT).as_posix()
    assert relative in hashes, f'missing imported training dependency: {name}'
'''
    subprocess.run([sys.executable, '-c', code], cwd=train.ROOT,
                   capture_output=True, text=True, check=True)


def test_manifest_drift_and_test_seal_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(train,'load_config',lambda: ({},{}))
    monkeypatch.setattr(train,'source_hashes',lambda cfg:{'source':'expected'})
    payload=dict(config_sha256=train.sha(train.CONFIG),source_sha256={'source':'changed'},test_evaluated=False)
    write(tmp_path/'execution_manifest.json',payload)
    with pytest.raises(ValueError,match='Source/config drift'):
        train.load_manifest(tmp_path,False)
    payload['source_sha256']={'source':'expected'}; payload['test_evaluated']=True
    write(tmp_path/'execution_manifest.json',payload)
    with pytest.raises(ValueError,match='Test seal violated'):
        train.load_manifest(tmp_path,False)


def test_queue_child_failure_stops_without_next_launch(gates_root,monkeypatch):
    monkeypatch.setattr(train,'load_manifest',lambda *args: ({},{}))
    calls=[]
    class Child:
        pid=12345
        def wait(self): return 7
    def spawn(command,**kwargs): calls.append(command); return Child()
    monkeypatch.setattr(queue.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError,match='exited 7'):
        queue.run(gates_root,'cpu')
    assert len(calls)==1 and 'L1' in calls[0] and 'L0' not in calls[0]
    status=json.loads((gates_root/'queue_status.json').read_text())
    assert status['status']=='FAILED_NO_AUTORETRY'
    assert (gates_root/'queue_initial_identity.json').is_file()


def test_make_run_resets_training_rng_after_different_model_construction(tmp_path,monkeypatch):
    # A tiny synthetic model exercises the runner RNG policy only. It is not
    # evidence of B1 geometry or real train/valid execution.
    class Tiny(torch.nn.Module):
        levels=('P3','P4','P5'); feature_strides=(8,16,32)
        def __init__(self,prototype):
            super().__init__(); self.weight=torch.nn.Parameter(torch.rand(1))
            torch.rand(2 if prototype=='L0' else 200); np.random.rand(3); random.random()
            self.candidate_architecture={'prototype_id':prototype}
        def initialization_identity(self): return {'backbone_sha256':'a'*64}
        def architecture_contract(self): return {'backbone':'B1'}
        def loss_config(self,baseline): return None
    baseline={'dataset':{'data_dir':'fake'}}
    monkeypatch.setattr(train,'load_manifest',lambda *args: ({},baseline))
    monkeypatch.setattr(train,'load_config',lambda: ({},baseline))
    monkeypatch.setattr(train,'build_model',lambda prototype,seed:Tiny(prototype))
    monkeypatch.setattr(train.frozen,'get_model_info',lambda *args:dict(estimated_macs=1,total_params=1))
    monkeypatch.setattr(train.frozen,'make_loader',lambda *args,**kwargs:[])
    monkeypatch.setattr(train,'DenseAnchorDetectionLoss',lambda _:torch.nn.Identity())
    witnesses=[]
    draws=[]
    for prototype in train.PROTOTYPES:
        train.make_run(tmp_path,prototype,42,'cpu',False)
        witnesses.append(json.loads((train.result_dir(tmp_path,prototype,42,False)/'training_rng_identity.json').read_text()))
        draws.append((torch.rand(8),np.random.rand(8),random.random()))
    assert witnesses[0]==witnesses[1]
    assert torch.equal(draws[0][0],draws[1][0])
    assert np.array_equal(draws[0][1],draws[1][1]) and draws[0][2]==draws[1][2]
