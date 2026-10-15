"""Single-path sharing correctness, no subnet ranking."""
from copy import deepcopy
from pathlib import Path
import random
import sys

import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/sonar_spec'))
from candidate_sanity import architecture,protection
from supernet_smoke import errors,step
import baseline as frozen
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate,validate_sonar_architecture,SPEC_PATH
from hwnas_fpga.models.sonar_supernet import (build_sonar_supernet,enumerate_architectures,architecture_id,
    serialize_architecture,deserialize_architecture,BalancedArchitectureSampler,calibrate_subnet_bn)
from hwnas_fpga.models.sonar_supernet.checkpoint import save_checkpoint,load_checkpoint

BASE=ROOT/'configs/baseline_sonar_frozen.yaml'
CONFIG=yaml.safe_load(BASE.read_text(encoding='utf-8'))
torch.set_num_threads(2)


def test_coverage_identity_and_validator():
    model=build_sonar_supernet();arches=list(enumerate_architectures())
    assert len(arches)==len({architecture_id(a) for a in arches})==144
    for a in arches:
        model.set_active_subnet(a)
        assert deserialize_architecture(serialize_architecture(a))==model.get_active_subnet()
        reverse=deepcopy(a); reverse['kernels']=dict(reversed(list(a['kernels'].items())))
        assert architecture_id(reverse)==architecture_id(a)
    bad=architecture('P2_P3_P4');bad['kernels']['stage7']=3
    for fn in [lambda:validate_sonar_architecture(bad,SPEC_PATH),lambda:model.set_active_subnet(bad)]:
        with pytest.raises(ValueError): fn()


def test_balanced_sampling_rng_restore():
    sampler=BalancedArchitectureSampler(42)
    for _ in range(12000): sampler.sample()
    s=sampler.statistics
    for name in ['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5']: assert abs(s[name]/12000-1/3)<.02
    for i in (1,2,4,5,6,7):
        assert abs(s[f'stage{i}_k3']/(s[f'stage{i}_k3']+s[f'stage{i}_k5'])-.5)<.03
    saved=sampler.state_dict();expected=[sampler.sample() for _ in range(20)]
    restored=BalancedArchitectureSampler(3);restored.load_state_dict(saved)
    assert [restored.sample() for _ in range(20)]==expected


CASES=[architecture(n,k) for n in ['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'] for k in ['k3','k5','mixed']]
CASES+=random.Random(73).sample([a for a in enumerate_architectures() if a not in CASES],10)


@pytest.mark.parametrize('arch',CASES)
def test_candidate_and_export_equivalence(arch,tmp_path):
    torch.manual_seed(42);model=build_sonar_supernet().eval();model.set_active_subnet(arch)
    exported=model.export_active_subnet().eval()
    candidate=build_sonar_candidate(SPEC_PATH,arch).eval()
    candidate.load_state_dict(exported.state_dict(),strict=True)
    torch.save(exported.state_dict(),tmp_path/'subnet.pt')
    candidate.load_state_dict(torch.load(tmp_path/'subnet.pt',weights_only=True),strict=True)
    with torch.no_grad():
        x=torch.randn(1,1,640,640);active=model(x)
        assert errors(active,candidate(x))['max_abs_error']==0
        assert errors(active,exported(x))['max_abs_error']==0
    assert len(exported.backbone.stages)==(6 if arch['output_combination']=='P2_P3_P4' else 8)
    assert not any('branches' in n for n in exported.state_dict())
    assert all(p.data_ptr()!=dict(candidate.named_parameters())[n].data_ptr() for n,p in exported.named_parameters())


@pytest.mark.parametrize('name',['P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'])
def test_single_path_gradient_and_module_identity(name):
    frozen.seed_all(42);model=build_sonar_supernet()
    identities={p:id(m) for p,m in model.heads.items()}
    sampler=BalancedArchitectureSampler()
    sampler.sample=lambda:architecture(name,'mixed')
    loader=frozen.make_loader(CONFIG,frozen.resolve(CONFIG['dataset']['data_dir']),'train',42,0,train=True)
    images,targets=next(iter(loader))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    witness=step(model,opt,sampler,images,targets,CONFIG)
    assert witness['isolation']
    assert identities=={p:id(m) for p,m in model.heads.items()}
    if name=='P2_P3_P4':
        called=[]
        hooks=[m.register_forward_hook(lambda m,i,o:called.append(True)) for m in
            [model.backbone.stages[6][0],model.backbone.stages[7][0],model.heads['P5'],model.outputs['P5'],model.laterals['P5']]]
        with torch.no_grad(): model(images[:1])
        for h in hooks:h.remove()
        assert not called


def test_bn_copy_calibration_and_train_only():
    model=build_sonar_supernet().train();a=architecture('P2_P3_P4','mixed')
    original={k:v.clone() for k,v in model.state_dict().items()};active=model.get_active_subnet()
    candidate=model.export_active_subnet(a)
    loader=frozen.make_loader(CONFIG,frozen.resolve(CONFIG['dataset']['data_dir']),'train',42,0)
    rng=torch.get_rng_state().clone()
    calibrated=calibrate_subnet_bn(model,a,loader,1)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert model.training and model.get_active_subnet()==active
    assert all(torch.equal(p,dict(calibrated.named_parameters())[n]) for n,p in candidate.named_parameters())
    assert all(int(m.num_batches_tracked)==1 for m in calibrated.modules() if isinstance(m,torch.nn.BatchNorm2d))
    assert not calibrated.training
    wrong=frozen.make_loader(CONFIG,frozen.resolve(CONFIG['dataset']['data_dir']),'valid',42,0)
    with pytest.raises(ValueError,match='train split'): calibrate_subnet_bn(model,a,wrong,1)


def test_checkpoint_mismatch_and_exact_resume(tmp_path):
    frozen.seed_all(42);model=build_sonar_supernet();sampler=BalancedArchitectureSampler()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loader=frozen.make_loader(CONFIG,frozen.resolve(CONFIG['dataset']['data_dir']),'train',42,0,train=True)
    images,targets=next(iter(loader))
    step(model,opt,sampler,images,targets,CONFIG)
    ckpt=tmp_path/'checkpoint.pt'
    save_checkpoint(ckpt,model,opt,sampler,BASE,epoch=0,global_step=1)
    expected=step(model,opt,sampler,images,targets,CONFIG)
    other=build_sonar_supernet();optimizer=torch.optim.AdamW(other.parameters(),lr=.3);sampler2=BalancedArchitectureSampler(9)
    state=load_checkpoint(ckpt,other,optimizer,sampler2,BASE)
    assert state['global_step']==1
    assert step(other,optimizer,sampler2,images,targets,CONFIG)==expected
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    state['search_space_version']='different';torch.save(state,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='search_space_version mismatch'):
        load_checkpoint(tmp_path/'bad.pt',other,optimizer,sampler2,BASE)
    protection(BASE)
