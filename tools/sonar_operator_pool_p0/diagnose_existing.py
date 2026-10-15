"""Frozen D checkpoint diagnostics: inference only on unaugmented train/valid."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import torch
import run as p0

REPORT = p0.ROOT / 'reports/sonar_operator_pool_p0_diagnosis'
AUDIT = p0.ROOT / 'reports/sonar_operator_pool_p0_confirmation/BASELINE_COMPARABILITY.json'


def immutable_json(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError(f'Frozen diagnostic binding differs: {path}')
    else:
        p0.dump_json(path,payload)


def run(output, device):
    cfg=p0.load_frozen_config()
    audit=json.loads(AUDIT.read_text(encoding='utf-8'))
    if audit['status']!='PASS' or p0.sha256(p0.BASELINE_CONFIG)!=audit['baseline_sha256']:
        raise ValueError('Baseline admission mismatch')
    for rel, expected in audit['source_sha256'].items():
        if p0.sha256(p0.ROOT/rel)!=expected:
            raise ValueError(f'Frozen source drift: {rel}')
    if p0.sha256(p0.ROOT/'configs/search_space_sonar_v0_1.yaml')!=audit['search_space_sha256']:
        raise ValueError('Frozen space drift')
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='D_existing_checkpoint_diagnosis_v1',seeds=[42,43,44],
        checkpoints=['best','epoch80'],splits=['train','valid'],epochs_trained=0,test_accessed=False,
        input=640,batch=4,workers=0,augmentation=False,weighted_sampler=False,shuffle=False,
        bn='eval-only; no recalibration; state hash verified before/after',
        evaluation_protocol=cfg['evaluation_protocol'],
        diagnostic_score_threshold=0.5,auxiliary_retained_score_threshold=0.01,
        match_iou=[0.5,0.75],background_iou_boundary=0.1,
        size_bins_input_sqrt_area=[0,16,32,'infinity'],size_bins_note='diagnostic input-pixel bins, not COCO',
        background_proxy=dict(intensity='0..1 grayscale bilinear640',gradient='mean absolute x/y adjacent differences',
            edge_density='fraction adjacent absolute differences >0.1',controls='one same-image same-size GT-free crop per background FP; 32 fixed-seed attempts',
            semantic_inference=False),
        history_windows=[[41,60],[61,80]],bootstrap=dict(metric='paired recall endpoint-minus-best',samples=2000,seed=42,unit='image; same checkpoint pair; per split per seed'),
        baseline_audit_sha256=p0.sha256(AUDIT),baseline_config_sha256=p0.sha256(p0.BASELINE_CONFIG),
        inference_source_sha256=p0.sha256(Path(__file__)))
    immutable_json(output/'DIAGNOSTIC_PROTOCOL.json',protocol)
    loaders={}
    for split in ('train','valid'):
        loaders[split]=p0.make_loader(cfg,p0.resolve(cfg['dataset']['data_dir']),split,42,0,train=False)
        b=p0.bind_dataset(loaders[split])
        if b!=audit['datasets'][split]:raise ValueError(f'Dataset content drift: {split}')
    controls=sorted(audit['controls'],key=lambda c:c['seed'])
    # Train records are produced first; all valid definitions above are already locked.
    artifacts=[]
    for split in ('train','valid'):
        for c in controls:
            seed=c['seed'];folder=Path(c['best_checkpoint']).parent
            hp=folder/'epoch_log.json'
            if p0.sha256(hp)!=c['history_sha256']:raise ValueError('History drift')
            history=json.loads(hp.read_text())
            if [h['epoch'] for h in history]!=list(range(1,81)):raise ValueError('Non-contiguous history')
            immutable_json(output/'histories'/f'D_seed{seed}.json',history)
            for kind in ('best','epoch80'):
                path=folder/('best.pt' if kind=='best' else 'latest.pt')
                expected=c['best_sha256'] if kind=='best' else c['latest_sha256']
                if p0.sha256(path)!=expected:raise ValueError(f'Checkpoint changed: {path}')
                tag=f'{split}_D_seed{seed}_{kind}'
                metric_path=output/'metrics'/f'{tag}.json'
                records_path=output/'predictions'/f'{tag}.jsonl.gz'
                if metric_path.exists():
                    cached=json.loads(metric_path.read_text())
                    if cached['checkpoint_sha256']!=expected or p0.sha256(records_path)!=cached['predictions_sha256']:
                        raise ValueError('Cached diagnostic artifact mismatch')
                    artifacts.append(cached);continue
                state=torch.load(path,map_location='cpu',weights_only=False)
                epoch=c['best_epoch'] if kind=='best' else 80
                if state['epoch']!=epoch or state.get('seed',seed)!=seed or state.get('test_evaluated') is not False:
                    raise ValueError('Checkpoint identity mismatch')
                p0.set_run_seed(seed)
                model=p0.build_d_baseline().to(device).eval()
                model.load_state_dict(state['model_state_dict'],strict=True)
                before=p0.state_hash(model)
                start=time.perf_counter()
                metrics,records=p0.evaluate(model,loaders[split],p0.loss_for(cfg,device),cfg,device)
                after=p0.state_hash(model)
                if before!=after:raise ValueError('Inference changed parameters or BN state')
                if split=='valid':
                    old=c['best_metrics'] if kind=='best' else c['endpoint_metrics']
                    for k in ('mAP50_95','AP50','AP75','Precision','Recall'):
                        if abs(metrics[k]-old[k])>1e-7:raise ValueError(f'Historical valid metric mismatch: {tag} {k}')
                records_path.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(records_path,'wt',encoding='utf-8') as f:
                    for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
                result=dict(tag=tag,split=split,seed=seed,checkpoint=kind,epoch=epoch,checkpoint_path=str(path),
                    checkpoint_sha256=expected,history_sha256=c['history_sha256'],metrics=metrics,
                    predictions_sha256=p0.sha256(records_path),model_state_unchanged=True,
                    inference_seconds=time.perf_counter()-start,test_accessed=False)
                p0.dump_json(metric_path,result);artifacts.append(result)
                print(tag,'AP',metrics['mAP50_95'],'loss',metrics['validation_loss'],flush=True)
                del state,model
    p0.dump_json(output/'INFERENCE_MANIFEST.json',dict(status='COMPLETE',evaluations=len(artifacts),artifacts=artifacts,
        protocol_sha256=p0.sha256(output/'DIAGNOSTIC_PROTOCOL.json'),torch=torch.__version__,
        device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,new_training_runs=0,test_accessed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPORT)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.output,args.device)
