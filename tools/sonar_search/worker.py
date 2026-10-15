"""Real independent-training/PTQ workers for the bound joint-search campaign."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'tools/sonar_spec'),
    str(ROOT/'tools/sonar_analysis'),str(ROOT/'scripts'),str(ROOT/'tools/fpga_h4')]
import torch
from candidate_graphs import architectures
from quant_contracts import PROFILES
from formal_runner import train_epoch, rng_state, restore_rng, atomic_save
import baseline as frozen
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate, SPEC_PATH
from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
from phase5a_quantization.calibrate import build_calibration, calibration_tensor_batches

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def assert_binding(out):
    freeze=read(out/'freeze.json')
    for rel,expected in freeze['files'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Frozen source/cost drift: '+rel)
    b=frozen.load_config(str(ROOT/'configs/baseline_sonar_frozen.yaml'))
    data=ROOT/b['dataset']['data_dir']
    for entries in freeze['dataset'].values():
        for row in entries:
            if sha(data/row['image'])!=row['image_sha256'] or sha(data/row['annotation'])!=row['annotation_sha256']:
                raise ValueError('Frozen data drift')
    return freeze,sha(out/'freeze.json'),b

def train_candidate(out,aid,seed,budget,device,freeze,freeze_sha,b):
    cfg=freeze['protocol']
    if aid not in architectures() or seed not in cfg['confirmation_seeds'] or budget not in cfg['budgets']:
        raise ValueError('Task outside frozen budget')
    if seed!=42 and budget!=80:raise ValueError('Confirmation only at full budget')
    folder=out/'training'/aid/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    frozen.seed_all(seed)
    model=build_sonar_candidate(SPEC_PATH,architectures()[aid]).to(device)
    train=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',seed,0,train=True)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',seed,0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    criterion=DenseAnchorDetectionLoss(model.loss_config(b)).to(device)
    latest=folder/'latest.pt';history=[];best=None;best_epoch=None;best_score=-1
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['freeze_sha256']!=freeze_sha or state['architecture_id']!=aid or state['seed']!=seed:raise ValueError('Training checkpoint binding mismatch')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        restore_rng(state['rng'],train);history=state['history'];best=state['best_state'];best_epoch=state['best_epoch'];best_score=state['best_score']
        if len(history)>budget:raise ValueError('Cannot recreate earlier budget from later checkpoint')
    for epoch in range(len(history)+1,budget+1):
        tick=time.monotonic();row=train_epoch(model,optimizer,train,b,device,None)
        metrics,_=frozen.evaluate(model,valid,criterion,b,device)
        row.update(epoch=epoch,valid=metrics,wall_seconds=time.monotonic()-tick);history.append(row)
        if metrics['mAP50_95']>best_score:
            best_score=metrics['mAP50_95'];best_epoch=epoch
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state=dict(freeze_sha256=freeze_sha,architecture_id=aid,architecture=architectures()[aid],seed=seed,
            epoch=epoch,history=history,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            rng=rng_state(train),best_state=best,best_score=best_score,best_epoch=best_epoch,test_evaluated=False)
        # Single atomic commit contains optimizer, all RNGs AND best weights.
        # A crash cannot leave a future best.pt paired with an older latest.pt.
        atomic_save(latest,state)
        print(aid,seed,epoch,best_score,flush=True)
    if best is None:raise ValueError('No completed training')
    checkpoint=folder/f'best_{budget}.pt'
    atomic_save(checkpoint,dict(freeze_sha256=freeze_sha,architecture_id=aid,architecture=architectures()[aid],seed=seed,
        epoch=best_epoch,budget=budget,model_state_dict=best,valid=history[best_epoch-1]['valid'],test_evaluated=False))
    result=dict(architecture_id=aid,seed=seed,budget=budget,best_epoch=best_epoch,mAP50_95=best_score,
        checkpoint=checkpoint.relative_to(out).as_posix(),checkpoint_sha256=sha(checkpoint),test_evaluated=False)
    return result,[checkpoint]

def quant_candidate(out,aid,seed,budget,device,freeze,freeze_sha,b):
    if budget!=80:raise ValueError('Quantization only after full training budget')
    checkpoint=out/'training'/aid/f'seed_{seed}'/'best_80.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['freeze_sha256']!=freeze_sha or state['architecture_id']!=aid or state['seed']!=seed or state['budget']!=80:
        raise ValueError('Quantized checkpoint identity mismatch')
    folder=out/'quantization'/aid/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    calibration_dir=out/'calibration';path=calibration_dir/'calibration_manifest.json'
    if path.exists():manifest=read(path)
    else:manifest=build_calibration(calibration_dir)
    if manifest['split']!='train' or manifest['test_evaluated'] or manifest['valid_used_for_selection']:raise ValueError('Calibration split violation')
    data=ROOT/b['dataset']['data_dir']
    for entry in manifest['entries']:
        if sha(data/entry['image'])!=entry['image_sha256'] or sha(data/entry['annotation'])!=entry['annotation_sha256']:
            raise ValueError('Calibration contents changed')
    batches=list(calibration_tensor_batches(manifest,device))
    model=build_sonar_candidate(SPEC_PATH,architectures()[aid]).to(device)
    model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
    valid=frozen.make_loader(b,data,'valid',seed,0)
    criterion=DenseAnchorDetectionLoss(model.loss_config(b)).to(device)
    def evaluate(m):
        metrics,_=frozen.evaluate(m,valid,criterion,b,device)
        return {k:float(metrics[k]) for k in ('mAP50_95','AP50','AP75','Precision','Recall')}
    result=dict(architecture_id=aid,seed=seed,checkpoint_sha256=sha(checkpoint),freeze_sha256=freeze_sha,
        calibration_sha256=sha(path),split='valid',test_evaluated=False,fp32=evaluate(model),profiles={})
    artifacts=[checkpoint,path]
    for q in freeze['protocol']['quant_profiles']:
        wb,ab=PROFILES[q]
        _,quantized,_,profile=prepare_h2_ptq(model,batches,device=device,weight_granularity='per_tensor',
            activation_granularity='per_tensor',scheme='symmetric',
            weight_bits_by_prefix={'head':wb},activation_bits_by_prefix={'head':ab},quantize_neck_adds=True)
        profile_path=folder/(q+'_profile.json');save(profile_path,profile);artifacts.append(profile_path)
        result['profiles'][q]=dict(evaluate(quantized),profile_sha256=sha(profile_path))
        del quantized
    result_path=folder/'result.json';save(result_path,result);artifacts.append(result_path)
    return result,artifacts

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','quant']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--architecture',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--budget',type=int,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();torch.set_num_threads(2)
    out=a.output.resolve();freeze,freeze_sha,b=assert_binding(out)
    if a.architecture not in architectures() or a.seed not in freeze['protocol']['confirmation_seeds']:raise ValueError('Unknown candidate/seed')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    fn=train_candidate if a.mode=='train' else quant_candidate
    result,outputs=fn(out,a.architecture,a.seed,a.budget,a.device,freeze,freeze_sha,b)
    receipt=out/'tasks'/f'{a.mode}_{a.architecture}_{a.seed}_{a.budget}.json'
    save(receipt,dict(freeze_sha256=freeze_sha,result=result,outputs={p.relative_to(out).as_posix():sha(p) for p in outputs}))

if __name__=='__main__':main()
