"""Versioned Phase2C runner. Formal ranking/training fail closed on independent gate."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from formal_protocol import (ROOT,load_protocol,verify_baseline,require_gate,calibration_manifest,
    select_witness,load_witness,initialize_frozen_baseline,source_binding,sha,dump,frozen)
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import build_sonar_supernet,BalancedArchitectureSampler,enumerate_architectures,architecture_id,calibrate_subnet_bn
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss,DetectionLossConfig


def atomic_save(path,state):
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def rng_state(loader):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampler_generator=loader.sampler.generator.get_state() if getattr(loader.sampler,'generator',None) is not None else None,
        loader_generator=loader.generator.get_state() if loader.generator is not None else None)


def restore_rng(state,loader):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
    if state['sampler_generator'] is not None:loader.sampler.generator.set_state(state['sampler_generator'])
    if state['loader_generator'] is not None:loader.generator.set_state(state['loader_generator'])


def train_epoch(model,opt,loader,b,device,sampler=None):
    model.train();sums=Counter();combos=Counter();kernels=Counter();count=0;start=time.perf_counter()
    for images,targets in loader:
        if sampler is not None:
            a=sampler.sample();model.set_active_subnet(a);combos[a['output_combination']]+=1
            for stage,k in a['kernels'].items():kernels[f'{stage}_k{k}']+=1
        criterion=DenseAnchorDetectionLoss(model.loss_config(b) if hasattr(model,'loss_config') else DetectionLossConfig.from_config(b)).to(device)
        opt.zero_grad(set_to_none=True);out=model(images.to(device));frozen.finite_outputs(out)
        loss,parts=criterion(out,targets)
        if not torch.isfinite(loss) or not all(math.isfinite(v) for v in parts.values()):raise ValueError('Nonfinite training loss')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite/missing active gradients')
        opt.step();count+=1
        for key in ('loss','box_loss','object_loss','class_loss'):sums[key]+=parts[key]
    if not count:raise ValueError('Empty train loader')
    return dict(losses={k:v/count for k,v in sums.items()},batches=count,combination_counts=dict(combos),kernel_counts=dict(kernels),
        gradient_finite=True,lr=opt.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start)


def run_training(args,p,b):
    mode=args.mode
    if mode!='baseline':require_gate(p,args.protocol)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not args.resume:raise FileExistsError('Use --resume for an existing training run')
    frozen.seed_all(42)
    witness_sha=None
    if mode=='train-supernet':model=build_sonar_supernet();sampler=BalancedArchitectureSampler(42)
    elif mode=='baseline':
        model,_=frozen.make_model(b);sampler=None
        initialize_frozen_baseline(model,p,b)
        manifest=json.loads((ROOT/b['baseline_freeze']['manifest']).read_text())
        if frozen.state_hash(model)!=manifest['initial_state_sha256']:raise ValueError('Baseline seeded initialization differs from freeze')
    else:
        witness=load_witness(args.witness,args.protocol);witness_sha=sha(args.witness)
        if not 0<=args.index<12:raise ValueError('Witness index must be0..11')
        model=build_sonar_candidate(ROOT/p['space'],witness['witnesses'][args.index]['architecture']);sampler=None
    model.to(args.device)
    if mode!='train-supernet':
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device=='cuda' else []):
            info=frozen.get_model_info(model,(1,1,640,640))
        info['FLOPs_2x_MACs']=2*info['estimated_macs'];dump(out/'model_info.json',info)
    train=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0,train=True)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    binding=dict(protocol_sha256=sha(args.protocol),baseline_protocol_hash=p['baseline_sha256'],search_space_sha256=p['space_sha256'],
        search_space_version='sonar_architecture_v0_1',source_sha256=source_binding(args.protocol),mode=mode,witness_sha256=witness_sha,
        witness_index=args.index if mode=='standalone' else None)
    epoch0=0;step=0;best=-1.;history=[]
    if args.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if state['binding']!=binding:raise ValueError('Resume source/protocol/architecture mismatch')
        model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state_dict'])
        if sampler:sampler.load_state_dict(state['architecture_sampler_rng'])
        restore_rng(state['rng'],train);epoch0=state['epoch'];step=state['global_step'];best=state['best_valid'];history=state['history']
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        provenance=ROOT/'deployment_provenance.json'
        commit=json.loads(provenance.read_text())['git_commit'] if provenance.exists() else 'unavailable'
    dump(out/'run_binding.json',dict(**binding,device=args.device,workers=0,seed=42,git_commit=commit,torch=torch.__version__,resume_policy='epoch_boundary'))
    started=time.perf_counter()
    for epoch in range(epoch0,80):
        record=train_epoch(model,opt,train,b,args.device,sampler);step+=record['batches'];record['epoch']=epoch+1
        metrics=None
        if mode!='train-supernet':
            criterion=DenseAnchorDetectionLoss(model.loss_config(b) if hasattr(model,'loss_config') else DetectionLossConfig.from_config(b)).to(args.device)
            metrics,_=frozen.evaluate(model,valid,criterion,b,args.device)
            record['valid']=metrics
        improved=metrics is not None and metrics['mAP50_95']>best
        if improved:best=metrics['mAP50_95']
        history.append(record)
        state=dict(binding=binding,model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),epoch=epoch+1,
            global_step=step,seed=42,git_commit=commit,search_space_version=binding['search_space_version'],
            search_space_sha256=p['space_sha256'],baseline_protocol_hash=p['baseline_sha256'],sampling_policy=p['training']['architecture_sampling'] if sampler else 'fixed_candidate',
            architecture_sampling_statistics=dict(sampler.statistics) if sampler else {},architecture_sampler_rng=sampler.state_dict() if sampler else None,
            rng=rng_state(train),best_valid=best,history=history,active_architecture=model.get_active_subnet() if sampler else getattr(model,'candidate_architecture',None))
        if improved:
            atomic_save(out/'best.pt',state);dump(out/'best_valid_metrics.json',metrics)
        # Commit the epoch only after its best checkpoint/metrics are durable.
        # A crash before latest commit resumes/replays this epoch from the previous boundary.
        atomic_save(out/'latest.pt',state)
        dump(out/'epoch_log.json',history)
        print(f'{mode} epoch={epoch+1}/80 loss={record["losses"]["loss"]:.6f} seconds={record["elapsed_seconds"]:.1f}',flush=True)
    dump(out/'status.json',dict(status='COMPLETE',mode=mode,epochs=80,global_step=step,wall_seconds_this_invocation=time.perf_counter()-started,binding=binding,
        initialization='from_scratch_seed42',selection='final_epoch80' if sampler else 'best_valid_earliest_tie',test_evaluated=False))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def rank(args,p,b):
    require_gate(p,args.protocol)
    if (args.output/'ranking_witness_manifest.json').exists() or (args.output/'subnet_inherited_ranking.csv').exists():
        raise FileExistsError('Formal ranking/witness already exists; use a new output directory, never overwrite a witness')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['epoch']!=80 or state['binding']['mode']!='train-supernet' or state['binding']['protocol_sha256']!=sha(args.protocol):
        raise ValueError('Ranking requires completed epoch80 formal supernet checkpoint')
    if state['binding']['source_sha256']!=source_binding(args.protocol):raise ValueError('Supernet source drift')
    model=build_sonar_supernet().to(args.device);model.load_state_dict(state['model_state_dict']);model.eval()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration,manifest=calibration_manifest(b,args.output)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    rows=[]
    for a in enumerate_architectures():
        start=time.perf_counter();candidate=calibrate_subnet_bn(model,a,calibration,16)
        metrics,_=frozen.evaluate(candidate,valid,DenseAnchorDetectionLoss(candidate.loss_config(b)).to(args.device),b,args.device)
        info=frozen.get_model_info(candidate,(1,1,640,640))
        row=dict(architecture_id=architecture_id(a),output_combination=a['output_combination'],**{f'stage{s}':a['kernels'].get(f'stage{s}') for s in (1,2,4,5,6,7)},
            **{k:metrics[k] for k in ('AP50','AP75','mAP50_95','Precision','Recall')},Params=info['total_params'],MACs=info['estimated_macs'],FLOPs=2*info['estimated_macs'],
            calibration_status='PASS',calibration_sha256=manifest['samples_sha256'],evaluation_seconds=time.perf_counter()-start)
        rows.append(row);print('evaluated',len(rows),row['architecture_id'],flush=True)
        write_csv(args.output/'subnet_inherited_partial.csv',rows)
    rows.sort(key=lambda r:(-r['mAP50_95'],r['architecture_id']))
    path=args.output/'subnet_inherited_ranking.csv';write_csv(path,rows)
    witness=dict(protocol_sha256=sha(args.protocol),ranking_sha256=sha(path),supernet_checkpoint_sha256=sha(args.checkpoint),
        calibration_sha256=manifest['samples_sha256'],witnesses=select_witness(rows))
    witness_path=args.output/'ranking_witness_manifest.json'
    if witness_path.exists() and json.loads(witness_path.read_text())!=witness:raise ValueError('Cannot replace frozen witness selection')
    dump(witness_path,witness)
    shared=[]
    for item in witness['witnesses']:
        candidate=model.export_active_subnet(item['architecture']).eval()
        metrics,_=frozen.evaluate(candidate,valid,DenseAnchorDetectionLoss(candidate.loss_config(b)).to(args.device),b,args.device)
        shared.append(dict(architecture_id=item['architecture_id'],**{k:metrics[k] for k in ('AP50','AP75','mAP50_95','Precision','Recall')}))
    write_csv(args.output/'witness_shared_bn.csv',shared)
    dump(args.output/'ranking_provenance.json',dict(status='COMPLETE',protocol_sha256=sha(args.protocol),
        supernet_checkpoint_sha256=sha(args.checkpoint),calibration_manifest_sha256=sha(args.output/'calibration_manifest.json'),
        ranking_sha256=sha(path),witness_manifest_sha256=sha(witness_path),shared_bn_sha256=sha(args.output/'witness_shared_bn.csv'),
        source_sha256=source_binding(args.protocol)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','train-supernet','baseline','standalone','rank'])
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/supernet_sonar_v0_1_formal.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/phase2c_supernet')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--witness',type=Path)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(2)
    p,b=load_protocol(args.protocol);protection=verify_baseline(b)
    if args.mode=='prepare':
        args.output.mkdir(parents=True,exist_ok=True);_,manifest=calibration_manifest(b,args.output)
        try:gate=require_gate(p,args.protocol);status='PASS'
        except RuntimeError as e:gate=str(e);status='BLOCKED_INDEPENDENT_AUDIT'
        dump(args.output/'preflight.json',dict(status=status,gate=gate,protection=protection,calibration_sha256=manifest['samples_sha256'],protocol_sha256=sha(args.protocol),source_sha256=source_binding(args.protocol)))
        print(status,gate)
    elif args.mode=='rank':
        if args.checkpoint is None:parser.error('--checkpoint required')
        rank(args,p,b)
    else:
        if args.mode=='standalone' and args.witness is None:parser.error('--witness required')
        run_training(args,p,b)


if __name__=='__main__':main()
