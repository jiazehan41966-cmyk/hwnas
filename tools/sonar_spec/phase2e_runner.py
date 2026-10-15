"""Independent budget training; frozen-prefix promotion, own-state resume, no test."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime,timezone
import torch
import yaml
from formal_protocol import ROOT,load_protocol,verify_baseline,source_binding,sha,dump,frozen
from formal_runner import train_epoch,rng_state,restore_rng,atomic_save
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import enumerate_architectures,architecture_id
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

REPORT=ROOT/'reports/phase2e_multifidelity'
PROTOCOL=REPORT/'MULTIFIDELITY_PROTOCOL.yaml'


def now():return datetime.now(timezone.utc).isoformat()


def sources():
    result=source_binding(ROOT/'configs/supernet_sonar_v0_1_formal.yaml')
    for name in ('phase2e_runner.py','phase2e_pipeline.py','phase2e_results.py','phase2e_calibration.py','phase2d_analysis.py'):
        p=ROOT/'tools/sonar_spec'/name;result[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    return result


def binding():
    return dict(protocol_sha256=sha(PROTOCOL),source_sha256=sources(),calibration_inputs_sha256=sha(REPORT/'calibration_inputs.json'))


def load(require_admission=True):
    cfg=yaml.safe_load(PROTOCOL.read_text());_,b=load_protocol(ROOT/cfg['phase2c_protocol'])
    assert sha(ROOT/cfg['baseline'])==cfg['baseline_sha256']
    assert sha(ROOT/cfg['search_space'])==cfg['search_space_sha256']
    assert cfg['training']==dict(seed=42,input=640,batch_size=4,optimizer='AdamW',lr=.001,weight_decay=.0001,
        scheduler='none',amp=False,workers=0,initialization='independent_random_from_scratch',inherited_weights='forbidden',
        loss_augmentation_assignment_anchors_evaluator='frozen_baseline',resume='own_epoch_boundary_checkpoint_with_optimizer_and_all_RNG')
    assert cfg['budgets']==[40,60,80] and cfg['population_sizes']==[144,108,72]
    assert cfg['promotion_fractions']==[[3,4],[2,3]] and cfg['promotion_metric']=='best_valid_mAP50_95_up_to_budget'
    assert cfg['test_policy']=='sealed_no_inference_metrics_selection_or_promotion'
    frozen_binding=json.loads((REPORT/'execution_binding.json').read_text())
    if frozen_binding!=binding():raise ValueError('Phase2E source/protocol/calibration binding drift')
    inputs=json.loads((REPORT/'calibration_inputs.json').read_text())
    if sha(ROOT/'reports/phase2c_supernet/ranking_witness_manifest.json')!=inputs['witness_sha256']:
        raise ValueError('Original witness manifest drift')
    for path,expected in inputs['histories'].items():
        if sha(REPORT/path)!=expected:raise ValueError('Calibration history drift')
    if require_admission:
        decision=json.loads((REPORT/'INDEPENDENT_RUNNER_REVIEW.json').read_text())
        if decision['status']!='PASS' or decision.get('open_critical_major')!=[] or decision['execution_binding_sha256']!=sha(REPORT/'execution_binding.json'):
            raise RuntimeError('Phase2E independent execution review not passed for current sources')
    return cfg,b


def arch_map():return {architecture_id(a):a for a in enumerate_architectures()}


def prefix_record(history,budget,aid):
    prefix=[r for r in history if r['epoch']<=budget]
    if [r['epoch'] for r in prefix]!=list(range(1,budget+1)):raise ValueError('Incomplete budget history')
    best=max(prefix,key=lambda r:r['valid']['mAP50_95'])
    return dict(architecture_id=aid,budget=budget,best_epoch=best['epoch'],
        **{k:best['valid'][k] for k in ('mAP50_95','AP50','AP75','Precision','Recall')},
        endpoint_mAP50_95=prefix[-1]['valid']['mAP50_95'],train_epoch_seconds=sum(r['elapsed_seconds'] for r in prefix))


def promote(rows,numerator,denominator):
    ids=[r['architecture_id'] for r in rows]
    if len(set(ids))!=len(ids) or any(not math.isfinite(r['mAP50_95']) for r in rows):raise ValueError('Invalid promotion rows')
    return sorted(rows,key=lambda r:(-r['mAP50_95'],r['architecture_id']))[:math.ceil(len(rows)*numerator/denominator)]


def rejected_audit(rows,promoted,stage,existing):
    removed=[r for r in rows if r['architecture_id'] not in {x['architecture_id'] for x in promoted} and r['architecture_id'] not in existing]
    result=[]
    for combo in ('P3_P4_P5','P2_P3_P4','P2_P3_P4_P5'):
        group=[r for r in removed if r['output_combination']==combo]
        if group:
            result.append(min(group,key=lambda r:hashlib.sha256(f'42:{stage}:{r["architecture_id"]}'.encode()).hexdigest())['architecture_id'])
    return result


def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def existing_map():
    m=json.loads((ROOT/'reports/phase2c_supernet/ranking_witness_manifest.json').read_text())
    return {v['architecture_id']:i for i,v in enumerate(m['witnesses'])}


def record(aid,budget):
    existing=existing_map();a=arch_map()[aid]
    if aid in existing:
        i=existing[aid];folder=ROOT/'results'/f'witness_{i:02d}'
        # Historical data is independently bound during calibration and never consumed beyond budget for promotion.
        hist=REPORT/'input_snapshot/results'/f'witness_{i:02d}'/'epoch_log.json'
        expected=json.loads((REPORT/'calibration_inputs.json').read_text())['histories'][str(hist.relative_to(REPORT)).replace('\\','/')]
        if sha(hist)!=expected:raise ValueError('Reused witness history drift')
        history=json.loads(hist.read_text());info=json.loads((folder/'model_info.json').read_text())
        checkpoint=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['active_architecture']!=a or checkpoint['epoch']!=80 or checkpoint['binding']['source_sha256']!=source_binding(ROOT/'configs/supernet_sonar_v0_1_formal.yaml'):
            raise ValueError('Original standalone checkpoint identity/source drift')
        result=prefix_record(history,budget,aid)
        result.update(history_reuse=True,checkpoint=str(folder/'latest.pt'),checkpoint_epoch=80,checkpoint_matches_budget=budget==80,
                      source_history_sha256=sha(hist),elapsed_training_seconds=None)
    else:
        folder=ROOT/'results/phase2e_independent'/aid
        checkpoint_path=folder/f'budget_{budget}.pt'
        if not checkpoint_path.exists():raise FileNotFoundError('Completed budget checkpoint missing')
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if state['architecture_id']!=aid or state['binding']!=binding() or state['epoch']<budget:raise ValueError('Candidate checkpoint binding/budget mismatch')
        info=json.loads((folder/'model_info.json').read_text());result=prefix_record(state['history'],budget,aid)
        result.update(history_reuse=False,checkpoint=str(checkpoint_path),checkpoint_epoch=state['epoch'],
            checkpoint_matches_budget=state['epoch']==budget,source_history_sha256=sha(folder/'epoch_log.json'),
            elapsed_training_seconds=sum(r['wall_seconds'] for r in state['history'][:budget]))
    result.update(output_combination=a['output_combination'],kernels=json.dumps(a['kernels'],sort_keys=True),seed=42,
        Params=info['total_params'],MACs=info['estimated_macs'],checkpoint_sha256=sha(Path(result['checkpoint'])),
        baseline_protocol_hash=yaml.safe_load(PROTOCOL.read_text())['baseline_sha256'])
    return result


def train(aid,budget,device,smoke=False):
    cfg,b=load(require_admission=not smoke);verify_baseline(b)
    architectures=arch_map()
    if aid not in architectures:raise ValueError('Unknown architecture ID')
    if not smoke and budget not in cfg['budgets']:raise ValueError('Only frozen budgets permitted')
    if not smoke and aid in existing_map():return record(aid,budget)
    folder=ROOT/('results/phase2e_smoke' if smoke else 'results/phase2e_independent')/aid;folder.mkdir(parents=True,exist_ok=True)
    if smoke and budget not in (1,2):raise ValueError('Smoke only supports epochs1/2')
    frozen.seed_all(42);model=build_sonar_candidate(ROOT/cfg['search_space'],architectures[aid]).to(device)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):info=frozen.get_model_info(model,(1,1,640,640))
    dump(folder/'model_info.json',info)
    train_loader=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0,train=True)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];start_epoch=0
    current_binding=binding();initial_hash=frozen.state_hash(model);parents=[]
    if (folder/'latest.pt').exists():
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if state['binding']!=current_binding or state['architecture_id']!=aid or state['smoke']!=smoke:raise ValueError('Resume binding mismatch')
        model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state_dict']);restore_rng(state['rng'],train_loader)
        history=state['history'];start_epoch=state['epoch'];initial_hash=state['initial_state_sha256'];parents=state['parents']
        if start_epoch<budget:parents=parents+[dict(checkpoint=str(folder/'latest.pt'),sha256=sha(folder/'latest.pt'),epoch=start_epoch)]
    if start_epoch>=budget:
        # latest is the epoch commit; regenerate sidecar history after an interrupted write.
        dump(folder/'epoch_log.json',history)
        if start_epoch==budget and not (folder/f'budget_{budget}.pt').exists():os.link(folder/'latest.pt',folder/f'budget_{budget}.pt')
        if not (folder/f'budget_{budget}_status.json').exists():
            committed=torch.load(folder/f'budget_{budget}.pt',map_location='cpu',weights_only=False)
            prefix=committed['history']
            dump(folder/f'budget_{budget}_status.json',dict(status='COMPLETE',architecture_id=aid,budget=budget,seed=42,smoke=smoke,
                binding=current_binding,parent_checkpoint=committed['parents'][-1] if committed['parents'] else None,
                checkpoint_sha256=sha(folder/f'budget_{budget}.pt'),optimizer_state_sha256=committed['optimizer_state_sha256'],
                git_commit=committed['git_commit'],start=committed['invocation_start'],end=committed['updated_at'],
                best_valid_mAP50_95=max(r['valid']['mAP50_95'] for r in prefix),final_valid_mAP50_95=prefix[-1]['valid']['mAP50_95'],
                derived_metadata_recovered=True,test_evaluated=False))
        return None if smoke else record(aid,budget)
    criterion=DenseAnchorDetectionLoss(model.loss_config(b)).to(device)
    try:
        import subprocess
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:git_commit=json.loads((ROOT/'deployment_provenance.json').read_text()).get('git_commit','unavailable') if (ROOT/'deployment_provenance.json').exists() else 'unavailable'
    invocation_start=now()
    for epoch in range(start_epoch,budget):
        tick=time.perf_counter();old_best=max((r['valid']['mAP50_95'] for r in history),default=-1)
        batches=[next(iter(train_loader))] if smoke else train_loader
        row=train_epoch(model,opt,batches,b,device,None)
        metrics,_=frozen.evaluate(model,valid,criterion,b,device,max_batches=1 if smoke else None)
        row.update(epoch=epoch+1,valid=metrics,wall_seconds=time.perf_counter()-tick);history.append(row)
        state=dict(binding=current_binding,architecture_id=aid,architecture=architectures[aid],seed=42,epoch=epoch+1,
            global_step=sum(r['batches'] for r in history),model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
            optimizer_state_sha256=state_hash(opt.state_dict()),rng=rng_state(train_loader),history=history,parents=parents,
            initial_state_sha256=initial_hash,smoke=smoke,git_commit=git_commit,invocation_start=invocation_start,updated_at=now(),
            initialization='independent_from_scratch',test_evaluated=False)
        if metrics['mAP50_95']>old_best:atomic_save(folder/'best.pt',state)
        atomic_save(folder/'latest.pt',state);dump(folder/'epoch_log.json',history)
        print(aid,f'epoch={epoch+1}/{budget}',f'loss={row["losses"]["loss"]:.6f}',flush=True)
    budget_checkpoint=folder/f'budget_{budget}.pt'
    if not budget_checkpoint.exists():os.link(folder/'latest.pt',budget_checkpoint)
    dump(folder/f'budget_{budget}_status.json',dict(status='COMPLETE',architecture_id=aid,budget=budget,seed=42,smoke=smoke,
        binding=current_binding,parent_checkpoint=parents[-1] if parents else None,checkpoint_sha256=sha(folder/'latest.pt'),
        optimizer_state_sha256=state['optimizer_state_sha256'],git_commit=git_commit,start=invocation_start,end=now(),
        best_valid_mAP50_95=max(r['valid']['mAP50_95'] for r in history),final_valid_mAP50_95=history[-1]['valid']['mAP50_95'],test_evaluated=False))
    return None if smoke else record(aid,budget)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train','smoke'])
    p.add_argument('--architecture-id');p.add_argument('--budget',type=int);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if args.mode=='prepare':
        cfg,b=load(False);print(json.dumps(verify_baseline(b)));print('Phase2E binding PASS');return
    if not args.architecture_id or not args.budget:p.error('--architecture-id and --budget required')
    train(args.architecture_id,args.budget,args.device,args.mode=='smoke')


if __name__=='__main__':main()
