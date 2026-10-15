"""Phase2B bounded correctness witness. No ranking or formal training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/sonar_analysis'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import torch
import yaml
import baseline as frozen
from candidate_sanity import protection,architecture
from candidate_diagnostics import assignment_diagnostics,topk_diagnostics
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import (build_sonar_supernet,enumerate_architectures,architecture_id,
    serialize_architecture,deserialize_architecture,BalancedArchitectureSampler,calibrate_subnet_bn)
from hwnas_fpga.models.sonar_supernet.checkpoint import save_checkpoint,load_checkpoint
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss


def errors(a,b):
    diffs=[(u-v).abs() for u,v in zip(a['predictions'],b['predictions'])]
    return dict(max_abs_error=max(float(x.max()) for x in diffs),
                mean_abs_error=sum(float(x.sum()) for x in diffs)/sum(x.numel() for x in diffs))


def step(model,opt,sampler,images,targets,config):
    a=sampler.sample();model.set_active_subnet(a);model.train();opt.zero_grad(set_to_none=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    out=model(images);frozen.finite_outputs(out)
    criterion=DenseAnchorDetectionLoss(model.loss_config(config))
    loss,parts=criterion(out,targets)
    assert torch.isfinite(loss) and all(torch.isfinite(torch.tensor(v)) for v in parts.values())
    loss.backward()
    _,mapping=model.candidate_state_mapping(a)
    active=set(mapping.values())
    inactive=[]
    for n,p in model.named_parameters():
        if n in active:
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
        else:
            assert p.grad is None,n
            inactive.append(n)
    opt.step()
    for n,p in model.named_parameters():
        if n in inactive: assert torch.equal(p,before[n]),n
    return dict(architecture=a,id=architecture_id(a),losses=parts,inactive_parameter_tensors=len(inactive),isolation=True)


def run(output,iterations=12,threads=4):
    torch.set_num_threads(threads);frozen.seed_all(42)
    baseline_path=ROOT/'configs/baseline_sonar_frozen.yaml'
    config=yaml.safe_load(baseline_path.read_text(encoding='utf-8'))
    protect=protection(baseline_path)
    model=build_sonar_supernet()
    all_arch=list(enumerate_architectures());ids=set()
    for a in all_arch:
        model.set_active_subnet(a);assert model.get_active_subnet()==a
        assert deserialize_architecture(serialize_architecture(a))==a
        ids.add(architecture_id(a))
    assert len(ids)==len(all_arch)==144
    sampler=BalancedArchitectureSampler(42)
    for _ in range(12000): sampler.sample()
    stats=dict(sampler.statistics)
    proportions={k:v/12000 for k,v in stats.items() if k.startswith('P')}
    assert all(abs(v-1/3)<.02 for v in proportions.values())
    kernel_proportions={f'stage{s}':{f'k{k}':stats[f'stage{s}_k{k}']/sum(stats[f'stage{s}_k{j}'] for j in (3,5)) for k in (3,5)} for s in (1,2,4,5,6,7)}
    assert all(abs(v-.5)<.03 for d in kernel_proportions.values() for v in d.values())
    chosen=[architecture(n,k) for n in proportions for k in ('k3','k5','mixed')]
    pool=[a for a in all_arch if a not in chosen]
    chosen+=random.Random(42).sample(pool,10)
    fixed=torch.randn(1,1,640,640);equivalence=[]
    model.eval()
    for a in chosen:
        model.set_active_subnet(a)
        exported=model.export_active_subnet().eval()
        candidate=build_sonar_candidate(model.spec,a).eval()
        candidate.load_state_dict(exported.state_dict(),strict=True)
        with torch.no_grad():
            active=model(fixed);e=errors(active,candidate(fixed));ex=errors(active,exported(fixed))
        assert e['max_abs_error']==ex['max_abs_error']==0
        assert len(exported.backbone.stages)==(6 if a['output_combination']=='P2_P3_P4' else 8)
        assert not any('branches' in n for n in exported.state_dict())
        equivalence.append(dict(id=architecture_id(a),candidate=e,export=ex))
    print('PASS 144 activations; 12000 samples; 19 candidate/export numerical checks',flush=True)
    train=frozen.make_loader(config,frozen.resolve(config['dataset']['data_dir']),'train',42,0,train=True)
    images,targets=next(iter(train))
    sampler=BalancedArchitectureSampler(42)
    opt=torch.optim.AdamW(model.parameters(),lr=config['training']['lr'],weight_decay=config['training']['weight_decay'])
    steps=[]
    for i in range(iterations):
        steps.append(step(model,opt,sampler,images,targets,config))
        print(f'PASS smoke step {i+1}/{iterations} {steps[-1]["id"]}',flush=True)
    assert {s['architecture']['output_combination'] for s in steps}==set(proportions)
    assert all(sampler.statistics[f'stage{s}_k{k}']>0 for s in (1,2,4,5,6,7) for k in (3,5))
    # Explicit execution witness for tail/P5 and inactive kernel modules.
    model.set_active_subnet(architecture('P2_P3_P4'));model.eval();called=[]
    hooks=[m.register_forward_hook(lambda m,i,o,n=n:called.append(n)) for n,m in model.named_modules() if n in
        ['backbone.stages.6.0','backbone.stages.7.0','heads.P5','laterals.P5','outputs.P5']]
    with torch.no_grad(): model(fixed)
    for h in hooks:h.remove()
    assert not called
    output.mkdir(parents=True,exist_ok=True)
    ckpt=output/'smoke_checkpoint.pt'
    save_checkpoint(ckpt,model,opt,sampler,baseline_path,epoch=0,global_step=iterations,
        data_state=dict(mode='fixed_materialized_train_batch_replay',images=images,targets=targets,
                        loader_sampler_rng=train.sampler.generator.get_state()))
    expected=step(model,opt,sampler,images,targets,config)
    resumed=build_sonar_supernet();resume_opt=torch.optim.AdamW(resumed.parameters(),lr=.001);resume_sampler=BalancedArchitectureSampler()
    state=load_checkpoint(ckpt,resumed,resume_opt,resume_sampler,baseline_path)
    actual=step(resumed,resume_opt,resume_sampler,state['data_state']['images'],state['data_state']['targets'],config)
    assert expected==actual
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in model.state_dict().items())
    save_checkpoint(output/'resumed_checkpoint.pt',resumed,resume_opt,resume_sampler,baseline_path,epoch=0,
                    global_step=iterations+1,data_state=state['data_state'])
    calibration_loader=frozen.make_loader(config,frozen.resolve(config['dataset']['data_dir']),'train',42,0)
    original={k:v.clone() for k,v in model.state_dict().items()};original_arch=model.get_active_subnet();original_mode=model.training
    a=architecture('P2_P3_P4_P5','mixed')
    uncalibrated=model.export_active_subnet(a)
    calibrated=calibrate_subnet_bn(model,a,calibration_loader,num_batches=2)
    assert all(torch.equal(v,dict(calibrated.named_parameters())[k]) for k,v in uncalibrated.named_parameters())
    assert any(not torch.equal(v,calibrated.state_dict()[k]) for k,v in uncalibrated.state_dict().items() if 'running_mean' in k)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert original_arch==model.get_active_subnet() and original_mode==model.training
    valid=frozen.make_loader(config,frozen.resolve(config['dataset']['data_dir']),'valid',42,0)
    validation,_=frozen.evaluate(calibrated,valid,DenseAnchorDetectionLoss(calibrated.loss_config(config)),config,torch.device('cpu'),max_batches=1)
    with torch.no_grad():
        pred=calibrated(images)['predictions']
        assignment=assignment_diagnostics(pred,targets,calibrated.loss_config(config),calibrated.levels)
        topk=topk_diagnostics(pred,calibrated.loss_config(config),calibrated.levels,config['evaluation_protocol'])
    bn_inventory=[dict(name=n,features=m.num_features) for n,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)]
    report=dict(status='PASS',coverage=144,sampling=dict(samples=12000,counts=stats,output_proportions=proportions,kernel_proportions=kernel_proportions),
        equivalence=equivalence,smoke=dict(iterations=iterations,continued_iterations=1,steps=steps,sampling_statistics=dict(sampler.statistics),
        data_policy='same real augmented batch4 replay; not an epoch or performance estimate'),resume=dict(exact_state_equal=True,next_step_equal=True),
        bn=dict(inventory=bn_inventory,calibration_batches=2,weights_unchanged=True,supernet_state_unchanged=True,validation=validation),
        assignment=assignment,topk=topk,tail_execution=[],baseline_protection=protect,baseline_protection_after=protection(baseline_path),test_evaluated=False)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),*sorted((ROOT/'src/hwnas_fpga/models/sonar_supernet').glob('*.py'))]}
    frozen.dump(output/'results.json',report)
    print('PASS BN calibration, exact resume, baseline protection',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/phase2b_supernet')
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if not 12<=args.iterations<=20:p.error('smoke iterations must be 12..20; formal training is not supported')
    run(args.output,args.iterations,args.threads)
