"""Independent A3 training for the frozen Phase2G SonarOp prototype."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from formal_protocol import ROOT, dump, sha, verify_baseline, frozen
from formal_runner import atomic_save, restore_rng, rng_state, train_epoch
from phase2e_runner import state_hash
from hwnas_fpga.models.sonar_op import A3_ID, build_sonarop_candidate
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

REPORT=ROOT/'reports/phase2g_sonarop'
IMPLEMENTATION=REPORT/'implementation_manifest.json'
BASELINE_PATH=ROOT/'configs/baseline_sonar_frozen.yaml'
METRICS=('mAP50_95','AP50','AP75','Precision','Recall')


def now(): return datetime.now(timezone.utc).isoformat()


def load():
    if not IMPLEMENTATION.exists(): raise FileNotFoundError('Phase2G smoke gate has not produced implementation_manifest.json')
    manifest=json.loads(IMPLEMENTATION.read_text(encoding='utf-8'))
    if manifest['status']!='SMOKE_PASS' or manifest['sonarop_version']!='v0.2-p0' or manifest['test_evaluated'] is not False:
        raise RuntimeError('Phase2G smoke gate is not admitted')
    for relative,expected in manifest['source_sha256'].items():
        if sha(ROOT/relative)!=expected: raise ValueError(f'Phase2G source drift: {relative}')
    if manifest['frozen_sha256']!={'search_space':'ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4',
        'baseline':'23989d09cae29139be7bd74a92bf9c53f75605047b7d8b2d2e2601b98d2eb56e',
        'phase2f':'d3d31ab74490da0abbc6c18cf354a090cbe6f73c0c795b0c0e3ea41405af54a1'}:
        raise ValueError('Frozen evidence binding mismatch')
    baseline=yaml.safe_load(BASELINE_PATH.read_text(encoding='utf-8'))
    verify_baseline(baseline)
    return manifest,baseline


def binding(manifest,seed):
    return dict(implementation_manifest_sha256=sha(IMPLEMENTATION),sonarop_version='v0.2-p0',
        architecture_id=A3_ID,seed=seed,baseline_protocol_hash=manifest['frozen_sha256']['baseline'],
        search_space_sha256=manifest['frozen_sha256']['search_space'],epochs=80,test_evaluated=False)


def folder(seed): return ROOT/'results/phase2g_sonarop'/A3_ID/f'seed_{seed}'


def summarize(seed):
    manifest,_=load(); path=folder(seed); state=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
    if state['binding']!=binding(manifest,seed) or state['epoch']!=80: raise ValueError('A3 completion identity mismatch')
    history=state['history']
    if [r['epoch'] for r in history]!=list(range(1,81)): raise ValueError('Incomplete A3 history')
    best=max(history,key=lambda r:r['valid']['mAP50_95']); info=json.loads((path/'model_info.json').read_text())
    status=json.loads((path/'status.json').read_text())
    row=dict(architecture='A3',architecture_id=A3_ID,seed=seed,best_epoch=best['epoch'],
        **{k:best['valid'][k] for k in METRICS},endpoint_mAP50_95=history[-1]['valid']['mAP50_95'],
        peak_endpoint_gap=best['valid']['mAP50_95']-history[-1]['valid']['mAP50_95'],
        Params=info['total_params'],MACs=info['estimated_macs'],training_time_seconds=status['elapsed_training_seconds'],
        best_checkpoint=str(path/'best.pt'),latest_checkpoint=str(path/'latest.pt'),
        best_checkpoint_sha256=sha(path/'best.pt'),latest_checkpoint_sha256=sha(path/'latest.pt'),
        history_sha256=sha(path/'epoch_log.json'),source_sha256=manifest['source_sha256'],
        config_sha256=sha(IMPLEMENTATION),initialization_seed=seed,test_evaluated=False)
    dump(path/'result.json',row); return row


def train(seed,device):
    if seed not in (42,43,44): raise ValueError('Phase2G admits only seeds 42,43,44')
    manifest,baseline=load(); path=folder(seed); path.mkdir(parents=True,exist_ok=True)
    frozen.seed_all(seed); model=build_sonarop_candidate().to(device)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
        info=frozen.get_model_info(model,(1,1,640,640))
    info['FLOPs_2x_MACs']=2*info['estimated_macs']; dump(path/'model_info.json',info)
    train_loader=frozen.make_loader(baseline,ROOT/baseline['dataset']['data_dir'],'train',seed,0,train=True)
    valid_loader=frozen.make_loader(baseline,ROOT/baseline['dataset']['data_dir'],'valid',seed,0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    run_binding=binding(manifest,seed); history=[]; start_epoch=0; parents=[]; initial_hash=frozen.state_hash(model)
    if (path/'latest.pt').exists():
        state=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        if state['binding']!=run_binding: raise ValueError('A3 resume binding mismatch')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        restore_rng(state['rng'],train_loader);history=state['history'];start_epoch=state['epoch']
        parents=state['parents'];initial_hash=state['initial_state_sha256']
        if start_epoch<80: parents=parents+[dict(checkpoint=str(path/'latest.pt'),sha256=sha(path/'latest.pt'),epoch=start_epoch)]
    if start_epoch>=80:
        if not (path/'status.json').exists(): raise ValueError('Epoch80 checkpoint lacks completion status')
        return summarize(seed)
    criterion=DenseAnchorDetectionLoss(model.loss_config(baseline)).to(device)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit='unavailable'
    gpu=torch.cuda.get_device_name(torch.cuda.current_device()) if device=='cuda' else 'cpu'; started=now()
    dump(path/'run_binding.json',dict(**run_binding,initialization='independent_random_from_scratch',
        initialization_seed=seed,git_commit=commit,gpu=gpu,torch=torch.__version__,resume_policy='epoch_boundary',start=started))
    for epoch in range(start_epoch,80):
        tick=time.perf_counter();old_best=max((r['valid']['mAP50_95'] for r in history),default=-1.)
        row=train_epoch(model,optimizer,train_loader,baseline,device,None)
        metrics,_=frozen.evaluate(model,valid_loader,criterion,baseline,device)
        row.update(epoch=epoch+1,valid=metrics,wall_seconds=time.perf_counter()-tick);history.append(row)
        state=dict(binding=run_binding,architecture_id=A3_ID,sonarop_version='v0.2-p0',seed=seed,epoch=epoch+1,
            global_step=sum(r['batches'] for r in history),model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),optimizer_state_sha256=state_hash(optimizer.state_dict()),
            rng=rng_state(train_loader),history=history,parents=parents,initial_state_sha256=initial_hash,
            initialization='independent_random_from_scratch',git_commit=commit,invocation_start=started,
            updated_at=now(),test_evaluated=False)
        if metrics['mAP50_95']>old_best: atomic_save(path/'best.pt',state)
        atomic_save(path/'latest.pt',state);dump(path/'epoch_log.json',history)
        print(A3_ID,f'seed={seed}',f'epoch={epoch+1}/80',f'loss={row["losses"]["loss"]:.6f}',flush=True)
    elapsed=sum(float(r['wall_seconds']) for r in history)
    dump(path/'status.json',dict(status='COMPLETE',architecture_id=A3_ID,seed=seed,epochs=80,
        binding=run_binding,elapsed_training_seconds=elapsed,best_checkpoint='best.pt',latest_checkpoint='latest.pt',
        start=started,end=now(),test_evaluated=False))
    return summarize(seed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','train','summarize'])
    p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    torch.set_num_threads(2)
    if args.mode=='preflight': manifest,_=load();print('Phase2G training gate PASS',sha(IMPLEMENTATION));return
    if args.seed is None:p.error('--seed is required')
    (train if args.mode=='train' else summarize)(args.seed,args.device) if args.mode=='train' else summarize(args.seed)


if __name__=='__main__':main()
