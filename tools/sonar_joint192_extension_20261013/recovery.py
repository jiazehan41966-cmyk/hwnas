"""Finalize two already-trained reference runs in a new namespace, no training."""
from __future__ import annotations
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_joint192_extension_20261013 import train
import torch

ORIGIN=Path('/home/jiajia/experiments/sonar_joint192_extension_20261013_02')
ORIGIN_RESULT=ORIGIN/'results/sonar_joint192_extension_20261013_01'
EXPECTED_ORIGIN_MANIFEST='ffbc246b30eaaadef046b7e2660d7456545b917e44e0ebee4aaff66f505d5772'


def training_loop_ast(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    return ast.dump(loop,include_attributes=False)


def validate_completed_checkpoints(origin_folder, origin_manifest_sha, arm, seed):
    folder=Path(origin_folder)
    binding=json.loads((folder/'run_binding.json').read_text(encoding='utf-8'))
    if binding['execution_manifest_sha256']!=origin_manifest_sha or binding['arm']!=arm or binding['seed']!=seed or binding['training_epochs']!=80:
        raise ValueError('Original checkpoint binding mismatch')
    latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    history=json.loads((folder/'epoch_log.json').read_text(encoding='utf-8'))
    if latest['binding']!=binding or best['binding']!=binding or latest['epoch']!=80 or latest['global_step']!=6640:
        raise ValueError('Original checkpoint is not complete epoch80')
    if latest['history']!=history or [r['epoch'] for r in history]!=list(range(1,81)) or any(r['batches']!=83 or not r['gradient_finite'] for r in history):
        raise ValueError('Original training coverage/history differs')
    selected=max(history,key=lambda r:r['valid']['mAP50_95'])
    if best['epoch']!=selected['epoch'] or best['history']!=history[:selected['epoch']]:
        raise ValueError('Original earliest-best checkpoint differs')
    init=json.loads((folder/'initialization.json').read_text(encoding='utf-8'))
    if latest['initialization']!=init or best['initialization']!=init:
        raise ValueError('Original initialization identity differs')
    return binding,latest,best,history,selected


def verify_source_compatibility(old,new):
    if old['config_sha256']!=new['config_sha256'] or old['input_identity']!=new['input_identity'] or old['baseline']!=new['baseline']:
        raise ValueError('Recovery protocol/data changed')
    allowed={'tools/sonar_joint192_extension_20261013/train.py',
             'tools/sonar_joint192_extension_20261013/queue_runner.py',
             'tools/sonar_joint192_extension_20261013/remote.py'}
    changed=[]
    for rel,sha in old['source_sha256'].items():
        if train.sha(ORIGIN/rel)!=sha:raise ValueError('Original frozen source was modified:'+rel)
        if new['source_sha256'].get(rel)!=sha:
            if rel not in allowed:raise ValueError('Training dependency changed:'+rel)
            changed.append(rel)
    for name in ('make_components','domain_summary','latency'):
        def node(path):
            t=ast.parse(Path(path).read_text(encoding='utf-8'))
            return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        if node(ORIGIN/'tools/sonar_joint192_extension_20261013/train.py')!=node(ROOT/'tools/sonar_joint192_extension_20261013/train.py'):
            raise ValueError('Training/evaluation construction changed:'+name)
    if training_loop_ast(ORIGIN/'tools/sonar_joint192_extension_20261013/train.py')!=training_loop_ast(ROOT/'tools/sonar_joint192_extension_20261013/train.py'):
        raise ValueError('Optimizer/training loop changed')
    return dict(status='PASS',changed_original_files=changed,
                model_and_training_dependencies_unchanged=True,training_loop_ast_equal=True)


def recover_reference(seed):
    if type(seed) is not int or seed not in (42,43):
        raise ValueError('Only the two completed reference seeds 42/43 may be recovered')
    m=train.manifest();old=json.loads((ORIGIN_RESULT/'execution_manifest.json').read_text())
    old_sha=train.sha(ORIGIN_RESULT/'execution_manifest.json')
    if old_sha!=EXPECTED_ORIGIN_MANIFEST:raise ValueError('Unexpected original freeze')
    source_check=verify_source_compatibility(old,m)
    historical,baseline=train.inherited.load_config()
    if train.inherited.input_identity(train.inherited.verify_inputs(historical,baseline))!=m['input_identity']:
        raise ValueError('Frozen train/valid data identity differs during recovery')
    arm=m['config']['reference'];src=ORIGIN_RESULT/'runs'/arm/f'seed_{seed}'
    dst=train.RESULT/'runs'/arm/f'seed_{seed}'
    if dst.exists():raise FileExistsError('Recovery destination already exists')
    binding,latest,best,history,selected=validate_completed_checkpoints(src,old_sha,arm,seed)
    expected=dict(arm=arm,seed=seed,config_sha256=train.sha(train.CONFIG),
                  execution_manifest_sha256=train.sha(train.RESULT/'execution_manifest.json'),
                  architecture_sha256=train.architecture_hash(m['config']['arms'][arm]['architecture']),
                  training_epochs=80,fresh_initialization=True,test_evaluated=False)
    old_expected=deepcopy(expected);old_expected['execution_manifest_sha256']=old_sha
    if binding!=old_expected:raise ValueError('Original training differs from current fixed design')
    dst.mkdir(parents=True)
    hashes={}
    for p in sorted(src.iterdir()):
        if not p.is_file():continue
        target=dst/('original_status.json' if p.name=='status.json' else p.name)
        shutil.copyfile(p,target)
        hashes[p.name]=train.sha(p)
        if train.sha(target)!=hashes[p.name]:raise ValueError('Checkpoint/artifact copy hash differs')
    started=time.monotonic()
    train.environment()
    train.frozen.seed_all(seed)
    model=train.build_model(m['config']['arms'][arm]['architecture'],seed).to('cuda')
    root=ROOT/m['baseline']['dataset']['data_dir']
    loader=train.frozen.make_loader(m['baseline'],root,'valid',seed,0)
    criterion=train.DenseAnchorDetectionLoss(model.loss_config(m['baseline'])).to('cuda')
    evaluated={};domains={}
    for name,state in (('best',best),('endpoint',latest)):
        model.load_state_dict(state['model_state_dict'],strict=True)
        metrics,records=train.frozen.evaluate(model,loader,criterion,m['baseline'],'cuda')
        expected_metrics=selected['valid'] if name=='best' else history[-1]['valid']
        if any(abs(metrics[k]-expected_metrics[k])>1e-10 for k in train.METRICS):
            raise ValueError('Recovery actual checkpoint replay differs:'+name)
        saved=json.loads((src/('best_valid_records.json' if name=='best' else 'latest_valid_records.json')).read_text())
        domain=train.evaluate_domain_metrics(records,m['baseline'])
        evaluated[name]=metrics;domains[name]=train.domain_summary(domain)
        train.dump(dst/f'recovery_{name}_records.json',records)
        train.dump(dst/f'recovery_{name}_domain.json',domain)
        if train.inherited.canonical_sha([r['target'] for r in records])!=train.inherited.canonical_sha([r['target'] for r in saved]):
            raise ValueError('Recovery target replay differs')
    model.load_state_dict(best['model_state_dict'],strict=True)
    benchmark=train.latency(model,loader,m['baseline']);benchmark['measured_during_recovery']=True
    origin=dict(status='VERIFIED',origin_root=str(ORIGIN),origin_run=str(src),
                origin_execution_manifest_sha256=old_sha,origin_run_binding_sha256=hashes['run_binding.json'],
                artifact_sha256=hashes,source_compatibility=source_check,
                input_identity_reverified=True,
                optimizer_steps_during_recovery=0,training_epochs_repeated=0,
                recovery_seconds=time.monotonic()-started,
                training_elapsed_basis='sum_of_logged_epoch_wall_seconds_excludes_setup_and_finalization')
    result=train.pack_completed_result(binding,best_epoch=selected['epoch'],best_metrics=evaluated['best'],
          endpoint_metrics=evaluated['endpoint'],best_domain=domains['best'],endpoint_domain=domains['endpoint'],
          benchmark=benchmark,info=json.loads((src/'model_info.json').read_text()),
          best_checkpoint_sha256=hashes['best.pt'],latest_checkpoint_sha256=hashes['latest.pt'],
          history_sha256=hashes['epoch_log.json'],wall_seconds=sum(r['wall_seconds'] for r in history),
          finalization_binding=expected,recovery_origin=origin)
    train.dump(dst/'recovery_receipt.json',dict(status='PASS',origin=origin,finalization_binding=expected))
    train.commit_completed_result(dst,result)
    print(json.dumps(dict(status='RECOVERED_WITHOUT_TRAINING',seed=seed,best=result['best']['mAP50_95'],endpoint=result['endpoint']['mAP50_95'])),flush=True)
    return result


if __name__=='__main__':
    torch.set_num_threads(2)
    for seed in (42,43):recover_reference(seed)
    train.report()
