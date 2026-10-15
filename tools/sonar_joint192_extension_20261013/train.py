"""Isolated full-80-epoch screening of extensions to fixed joint192 B1.

Validation-selected candidate effects only; not a complete architecture search.
Uses the immutable historical train330/valid99 recipe, with fresh paired seeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.sonar_joint192_validation import train as inherited
from tools.sonar_joint192_validation.domain_metrics import evaluate_domain_metrics
from tools.sonar_joint192_extension_20261013.model import (
    build_model, validate_architecture, architecture_hash, profile_model, DEFAULT_CHANGES,
)
from tools.sonar_analysis import baseline as frozen
from tools.sonar_spec.formal_runner import train_epoch, atomic_save, rng_state
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, decode_detection_predictions
import numpy as np
import torch

CONFIG = ROOT / 'configs/sonar_joint192_extension_20261013.json'
RESULT = ROOT / 'results/sonar_joint192_extension_20261013_01'
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall')
sha = inherited.sha
dump = inherited.dump
now = inherited.now


def config():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if cfg['training'] != dict(epochs=80, input=640, batch_size=4, optimizer='AdamW',
                              lr=.001, weight_decay=.0001, scheduler='none', amp=False, workers=0):
        raise ValueError('Full frozen training recipe changed')
    if len(cfg['arms']) != 16 or cfg['reference'] != 'A00_reference':
        raise ValueError('Registered 16-arm design changed')
    historical, _ = inherited.load_config()
    base = historical['arms']['B1']
    if cfg['arms'][cfg['reference']]['architecture'] != dict(base_architecture=base,changes=DEFAULT_CHANGES):
        raise ValueError('Fixed B1 reference must have no extension changes')
    hashes=set()
    for name,arm in cfg['arms'].items():
        arch=validate_architecture(arm['architecture'])
        if name=='A01_strided':
            if arch != dict(base_architecture=historical['arms']['B0'],changes=DEFAULT_CHANGES):
                raise ValueError('Strided control must differ only in stage1 sampling')
        else:
            if arch['base_architecture']!=base:
                raise ValueError('All extension probes must use fixed B1')
            count=sum(arch['changes'][k]!=DEFAULT_CHANGES[k] for k in DEFAULT_CHANGES)
            if count!=(0 if name==cfg['reference'] else 1):
                raise ValueError('Each extension must change exactly one registered factor')
        identity=architecture_hash(arch)
        if identity in hashes:raise ValueError('Duplicate architecture encoding')
        hashes.add(identity)
    if cfg['maximum_runs'] != 22 or cfg['confirmation']['maximum_candidates'] != 2:
        raise ValueError('Bounded campaign budget changed')
    return cfg


def source_hashes():
    old, _ = inherited.load_config()
    paths = set(inherited.source_hashes(old))
    paths.update((
        'tools/sonar_joint192_validation/domain_metrics.py',
        'tools/sonar_joint192_extension_20261013/model.py',
        'tools/sonar_joint192_extension_20261013/train.py',
        'tools/sonar_joint192_extension_20261013/queue_runner.py',
        'tools/sonar_joint192_extension_20261013/remote.py',
        'tools/sonar_joint192_extension_20261013/recovery.py',
        'tests/test_sonar_joint192_extension_model.py',
        'tests/test_sonar_joint192_extension_completion.py',
        'configs/sonar_joint192_extension_20261013.json',
        'src/hwnas_fpga/models/sonar_operator_pool_p0.py',
        'src/hwnas_fpga/models/sonar_op/sonar_dw_reparam.py',
        'src/hwnas_fpga/models/sonar_op/__init__.py',
        'src/hwnas_fpga/models/sonar_op/candidate.py',
    ))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/hwnas_fpga').rglob('*.py'))
    return {rel: sha(ROOT/rel) for rel in sorted(paths)}


def verify_test_absent(baseline):
    data=ROOT/baseline['dataset']['data_dir']
    path=data/'test'
    if path.exists() or path.is_symlink():
        raise ValueError('This isolated namespace must not mount the test split')
    return False


def prepare():
    cfg = config()
    old, baseline = inherited.load_config()
    verify_test_absent(baseline)
    inputs = inherited.verify_inputs(old, baseline)
    payload = dict(config=cfg, config_sha256=sha(CONFIG), source_sha256=source_hashes(),
                   baseline=baseline, input_identity=inherited.input_identity(inputs),
                   source_space='sonar_backbone_neck192_v1',
                   experiment_scope='fixed_background_single_factor_full80_candidate_screen_not_full_space_search',
                   test_evaluated=False, test_mounted=False)
    path = RESULT/'execution_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('Existing source/config/data freeze differs')
    else:
        dump(path, payload)
    return payload


def manifest():
    m = json.loads((RESULT/'execution_manifest.json').read_text(encoding='utf-8'))
    if m['config_sha256'] != sha(CONFIG) or m['source_sha256'] != source_hashes():
        raise ValueError('Source/config drift against isolated freeze')
    if m['test_evaluated'] is not False or m['test_mounted'] is not False:
        raise ValueError('Test boundary changed')
    verify_test_absent(m['baseline'])
    return m


def environment():
    if not torch.cuda.is_available():
        raise RuntimeError('Existing server CUDA required; no CPU training fallback')
    actual = inherited.environment('cuda')
    expected = dict(python='3.13.11', torch='2.7.1+cu128', numpy='2.4.4', cuda_runtime='12.8')
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError('Pinned reused environment differs')
    return actual


def make_components(arm, seed, folder):
    m = manifest()
    environment()
    old, baseline = inherited.load_config()
    if inherited.input_identity(inherited.verify_inputs(old, baseline)) != m['input_identity']:
        raise ValueError('Frozen train/valid data identity differs')
    architecture = m['config']['arms'][arm]['architecture']
    frozen.seed_all(seed)
    model = build_model(architecture, seed)
    initial_state = frozen.state_hash(model)
    with torch.random.fork_rng(devices=[]):
        info = profile_model(model)
        deployed_info = profile_model(model, deploy=True)
    for p in (info,deployed_info):
        p.update(estimated_macs=p['macs_total'],total_params=p['params_total'])
    if frozen.state_hash(model) != initial_state:
        raise ValueError('Profiling mutated model state')
    model = model.to('cuda')
    initial = model.initialization_identity()
    dump(folder/'initialization.json', initial)
    dump(folder/'model_info.json', dict(training_form=info, deployed_form=deployed_info,
                                       architecture_contract=model.architecture_contract()))
    frozen.seed_all(seed)
    train_loader = frozen.make_loader(baseline, ROOT/baseline['dataset']['data_dir'], 'train', seed, 0, train=True)
    valid_loader = frozen.make_loader(baseline, ROOT/baseline['dataset']['data_dir'], 'valid', seed, 0)
    if len(train_loader.dataset) != 330 or len(valid_loader.dataset) != 99 or len(train_loader) != 83:
        raise ValueError('Expected historical train330/valid99 coverage changed')
    if len(train_loader.dataset) % 4 == 1:
        raise ValueError('DMDC regional BN would receive a singleton last batch')
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    return model, train_loader, valid_loader, criterion, opt, baseline, initial


def smoke():
    m = manifest()
    reports = []
    env = environment()
    for arm in m['config']['arms']:
        folder = RESULT/'smoke'/arm
        if folder.exists():
            raise FileExistsError('Preserve existing smoke; use new namespace for repairs')
        folder.mkdir(parents=True)
        model, tl, vl, criterion, opt, b, initial = make_components(arm, 42, folder)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        rows=[]
        for i, (images, targets) in enumerate(tl):
            if i == 2: break
            opt.zero_grad(set_to_none=True)
            output = model(images.to('cuda')); frozen.finite_outputs(output)
            loss, parts = criterion(output, targets)
            if not torch.isfinite(loss): raise ValueError('Nonfinite smoke loss')
            loss.backward()
            gradients = frozen.gradient_summary(model)
            opt.step()
            rows.append(dict(images_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
                             targets_sha256=inherited.canonical_sha(inherited.tensor_json(targets)),
                             loss=parts, gradients=gradients))
        metrics, _ = frozen.evaluate(model, vl, criterion, b, 'cuda', max_batches=1)
        report = dict(status='PASS', arm=arm, seed=42, batches=rows, validation=metrics,
                      initialization=initial, environment=env,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), test_evaluated=False)
        if len(rows)!=2 or metrics['num_images']!=4:
            raise ValueError('Smoke real-data coverage differs')
        dump(folder/'smoke_report.json', report); reports.append(report)
        print(json.dumps(dict(status='SMOKE_PASS',arm=arm,peak_reserved_bytes=report['peak_reserved_bytes'])),flush=True)
        del model, tl, vl, criterion, opt
        gc.collect();torch.cuda.empty_cache()
    for key in ('images_sha256','targets_sha256'):
        for i in (0,1):
            if len({r['batches'][i][key] for r in reports}) != 1:
                raise ValueError('Unpaired training/augmentation stream')
    for key in ('head_sha256','lateral_sha256'):
        if len({r['initialization'][key] for r in reports}) != 1:
            raise ValueError('Common initial tensors differ: '+key)
    gate = dict(status='PASS', arms=len(reports), real_train_batches_per_arm=2,
                valid_images_per_arm=4, environment=env,
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in reports),
                execution_manifest_sha256=sha(RESULT/'execution_manifest.json'), test_evaluated=False)
    dump(RESULT/'smoke_gate.json',gate)
    print(json.dumps(gate),flush=True)
    return gate


def run(arm, seed):
    m = manifest()
    if arm not in m['config']['arms'] or seed not in (42,43,44):
        raise ValueError('Unregistered run')
    if arm != m['config']['reference'] and seed != 42:
        selection = validated_selection()
        if arm not in selection['selected']:
            raise ValueError('Unselected confirmation refused')
    folder=RESULT/'runs'/arm/f'seed_{seed}'
    if folder.exists(): raise FileExistsError('Existing training invocation refused; no overwrite/autoretry')
    folder.mkdir(parents=True)
    binding=dict(arm=arm,seed=seed,config_sha256=sha(CONFIG),
                 execution_manifest_sha256=sha(RESULT/'execution_manifest.json'),
                 architecture_sha256=architecture_hash(m['config']['arms'][arm]['architecture']),
                 training_epochs=80,fresh_initialization=True,test_evaluated=False)
    dump(folder/'run_binding.json',binding)
    dump(folder/'status.json',dict(status='RUNNING',epoch=0,binding=binding,updated_at=now()))
    started=time.monotonic()
    try:
        model, tl, vl, criterion, opt, b, initial=make_components(arm,seed,folder)
        history=[];best=-1.
        for epoch in range(1,81):
            tick=time.monotonic()
            record=train_epoch(model,opt,tl,b,'cuda')
            metrics,records=frozen.evaluate(model,vl,criterion,b,'cuda')
            if record['batches']!=83 or metrics['num_images']!=99:
                raise ValueError('Frozen epoch coverage differs')
            record.update(epoch=epoch,valid=metrics,wall_seconds=time.monotonic()-tick)
            history.append(record)
            state=dict(binding=binding,architecture=m['config']['arms'][arm]['architecture'],
                       model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                       rng=rng_state(tl),epoch=epoch,global_step=83*epoch,history=history,
                       initialization=initial,test_evaluated=False)
            if metrics['mAP50_95']>best:
                best=metrics['mAP50_95']
                atomic_save(folder/'best.pt',state)
                dump(folder/'best_valid_records.json',records)
            atomic_save(folder/'latest.pt',state)
            dump(folder/'latest_valid_records.json',records)
            dump(folder/'epoch_log.json',history)
            dump(folder/'status.json',dict(status='RUNNING',epoch=epoch,binding=binding,
                 best_mAP50_95=best,latest_valid={k:metrics[k] for k in METRICS},updated_at=now()))
            print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,best_mAP50_95=best,
                                  latest_mAP50_95=metrics['mAP50_95'],epoch_seconds=record['wall_seconds'])),flush=True)
        best_state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_state['model_state_dict'],strict=True)
        replay,replayed=frozen.evaluate(model,vl,criterion,b,'cuda')
        selected=max(history,key=lambda r:r['valid']['mAP50_95'])
        for key in METRICS:
            if not math.isclose(replay[key],selected['valid'][key],abs_tol=1e-10,rel_tol=0):
                raise ValueError('Checkpoint replay differs: '+key)
        dump(folder/'replayed_best_valid_records.json',replayed)
        domain=evaluate_domain_metrics(replayed,b)
        endpoint_records=json.loads((folder/'latest_valid_records.json').read_text(encoding='utf-8'))
        endpoint_domain=evaluate_domain_metrics(endpoint_records,b)
        dump(folder/'best_domain.json',domain);dump(folder/'endpoint_domain.json',endpoint_domain)
        benchmark=latency(model,vl,b)
        info=json.loads((folder/'model_info.json').read_text(encoding='utf-8'))
        result=pack_completed_result(binding,best_epoch=selected['epoch'],best_metrics=replay,
                    endpoint_metrics=history[-1]['valid'],best_domain=domain_summary(domain),
                    endpoint_domain=domain_summary(endpoint_domain),benchmark=benchmark,info=info,
                    best_checkpoint_sha256=sha(folder/'best.pt'),latest_checkpoint_sha256=sha(folder/'latest.pt'),
                    history_sha256=sha(folder/'epoch_log.json'),wall_seconds=time.monotonic()-started)
        commit_completed_result(folder,result)
        return result
    except Exception as error:
        dump(folder/'failure.json',dict(status='FAILED_NO_AUTORETRY',binding=binding,error=str(error),updated_at=now()))
        dump(folder/'status.json',dict(status='FAILED_NO_AUTORETRY',binding=binding,error=str(error),updated_at=now()))
        raise


def pack_completed_result(binding, *, best_epoch, best_metrics, endpoint_metrics,
                          best_domain, endpoint_domain, benchmark, info,
                          best_checkpoint_sha256, latest_checkpoint_sha256,
                          history_sha256, wall_seconds,
                          finalization_binding=None, recovery_origin=None):
    """One shared completion path, with explicit original training provenance."""
    required={'arm','seed','config_sha256','execution_manifest_sha256',
              'architecture_sha256','training_epochs','fresh_initialization','test_evaluated'}
    if set(binding)!=required or binding['training_epochs']!=80 or binding['test_evaluated'] is not False:
        raise ValueError('Invalid completion binding')
    if type(best_epoch) is not int or not 1<=best_epoch<=80:
        raise ValueError('Invalid completed best epoch')
    if any(not math.isfinite(metrics[k]) for metrics in (best_metrics,endpoint_metrics) for k in METRICS):
        raise ValueError('Nonfinite completed metrics')
    result=deepcopy(binding)
    result.update(status='COMPLETE',best_epoch=best_epoch,
                  best={k:best_metrics[k] for k in METRICS},endpoint={k:endpoint_metrics[k] for k in METRICS},
                  best_domain=best_domain,endpoint_domain=endpoint_domain,latency=benchmark,model_info=info,
                  checkpoint_replay='PASS',best_checkpoint_sha256=best_checkpoint_sha256,
                  latest_checkpoint_sha256=latest_checkpoint_sha256,history_sha256=history_sha256,
                  wall_seconds=wall_seconds,full_epochs=80,global_step=6640,
                  selection_bias='valid-selected checkpoint and seed42-selected candidate; not untouched test')
    if (finalization_binding is None)!=(recovery_origin is None):
        raise ValueError('Recovered result requires origin and finalization bindings together')
    if recovery_origin is not None:
        if recovery_origin.get('status')!='VERIFIED' or finalization_binding.get('test_evaluated') is not False:
            raise ValueError('Unverified recovery origin')
        result.update(origin_run_binding=deepcopy(binding),
                      finalization_binding=deepcopy(finalization_binding),recovery_origin=recovery_origin)
    return result


def commit_completed_result(folder,result):
    """Write result first, COMPLETE status last; preserve all failure evidence."""
    folder=Path(folder)
    if (folder/'result.json').exists():raise FileExistsError('Completed result already exists')
    binding={k:result[k] for k in ('arm','seed','config_sha256','execution_manifest_sha256',
              'architecture_sha256','training_epochs','fresh_initialization','test_evaluated')}
    status=dict(status='COMPLETE',epoch=80,binding=binding,updated_at=now())
    if 'finalization_binding' in result:status['finalization_binding']=result['finalization_binding']
    dump(folder/'result.json',result)
    dump(folder/'status.json',status)


def domain_summary(d):
    return dict(lowcontrast_recall=d['groups']['contrast_low']['Recall'],
                extremely_small_recall=d['groups']['very_small_min_side']['Recall'],
                FP_per_image=d['overall']['FP_per_image'],
                background_FP_per_image=d['overall']['background_FP_per_image'],
                per_class_recall={k:v['all']['Recall'] for k,v in d['per_class'].items()},
                GT_groups={k:v['GT'] for k,v in d['groups'].items()})


def latency(model,loader,b):
    model.eval();images,_=next(iter(loader));image=images[:1].to('cuda')
    measurements={}
    # Concurrent training can contaminate timings; retained as descriptive only.
    with torch.inference_mode():
        for mode in ('model_forward','model_plus_decode_NMS'):
            values=[]
            for i in range(30):
                torch.cuda.synchronize();tick=time.monotonic();out=model(image)
                if mode=='model_plus_decode_NMS':
                    p=b['evaluation_protocol']
                    decode_detection_predictions(out,config=model.loss_config(b),
                        score_threshold=p['ap_decode_score_threshold'],nms_iou_threshold=p['nms_iou_threshold'],
                        max_detections=p['post_nms_max_detections_per_image'],pre_nms_topk=p['pre_nms_topk'])
                torch.cuda.synchronize()
                if i>=10:values.append((time.monotonic()-tick)*1000)
            measurements[mode]=dict(median_ms=statistics.median(values),repeats=20,raw_ms=values)
    return dict(measurements=measurements,device=torch.cuda.get_device_name(),precision='FP32',batch=1,
                concurrent_training_possible=True,claimable_efficiency_comparison=False,
                FPGA_latency_measured=False)


def collect_results(verify_artifacts=False):
    m=manifest();cfg=m['config'];rows={}
    for arm in cfg['arms']:
        for seed in (42,43,44):
            p=RESULT/'runs'/arm/f'seed_{seed}'/'result.json'
            if p.exists():
                status=json.loads((p.parent/'status.json').read_text(encoding='utf-8'))
                # COMPLETE is the final commit marker. A result can be durable
                # just before its status commit, so running rows are excluded.
                if status.get('status')!='COMPLETE':continue
                r=json.loads(p.read_text(encoding='utf-8'))
                expected=dict(arm=arm,seed=seed,config_sha256=sha(CONFIG),
                     execution_manifest_sha256=sha(RESULT/'execution_manifest.json'),
                     architecture_sha256=architecture_hash(cfg['arms'][arm]['architecture']),
                     training_epochs=80,fresh_initialization=True,test_evaluated=False)
                training_expected=deepcopy(expected)
                if 'recovery_origin' in r:
                    origin=r['recovery_origin']
                    if r.get('finalization_binding')!=expected or origin.get('status')!='VERIFIED':
                        raise ValueError('Recovered result missing verified finalization binding')
                    training_expected['execution_manifest_sha256']=origin['origin_execution_manifest_sha256']
                    if r.get('origin_run_binding')!=training_expected or status.get('finalization_binding')!=expected:
                        raise ValueError('Original training binding not preserved in recovered result')
                    receipt_path=p.parent/'recovery_receipt.json'
                    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                    if receipt.get('status')!='PASS' or receipt.get('finalization_binding')!=expected or receipt.get('origin')!=origin:
                        raise ValueError('Recovery receipt identity differs')
                    verify_artifacts=True
                if any(r.get(k)!=v for k,v in training_expected.items()) or r.get('status')!='COMPLETE' or r.get('full_epochs')!=80 or r.get('global_step')!=6640:
                    raise ValueError('Result identity or completion differs:'+str(p))
                binding=json.loads((p.parent/'run_binding.json').read_text(encoding='utf-8'))
                if binding!=training_expected or status.get('status')!='COMPLETE' or status.get('epoch')!=80 or status.get('binding')!=training_expected:
                    raise ValueError('Result lacks matching completion status/binding')
                for metric in METRICS:
                    if any(not math.isfinite(r[phase][metric]) for phase in ('best','endpoint')):
                        raise ValueError('Nonfinite completed metric')
                if verify_artifacts:
                    for field,file in (('best_checkpoint_sha256','best.pt'),('latest_checkpoint_sha256','latest.pt'),('history_sha256','epoch_log.json')):
                        if sha(p.parent/file)!=r[field]:raise ValueError('Result artifact hash drift')
                rows[(arm,seed)]=r
    return cfg,rows


def select():
    cfg,rows=collect_results(True);ref=rows[(cfg['reference'],42)]
    if any((arm,42) not in rows for arm in cfg['arms']):
        raise ValueError('All 16 seed42 runs must finish before selection')
    ranking=[]
    for arm in cfg['arms']:
        if arm==cfg['reference']:continue
        r=rows[(arm,42)]
        delta=r['best']['mAP50_95']-ref['best']['mAP50_95']
        end=r['endpoint']['mAP50_95']-ref['endpoint']['mAP50_95']
        ranking.append(dict(arm=arm,delta=delta,endpoint_delta=end,
                            estimated_macs=r['model_info']['training_form']['estimated_macs']))
    ranking.sort(key=lambda r:(-r['delta'],-r['endpoint_delta'],r['estimated_macs'],r['arm']))
    value=dict(status='FROZEN_SELECTION',selected=[r['arm'] for r in ranking if r['delta']>0][:2],
               ranking=ranking,rule=cfg['confirmation']['selection_rule'],
               execution_manifest_sha256=sha(RESULT/'execution_manifest.json'),
               seed42_used_for_selection=True,test_evaluated=False)
    p=RESULT/'selection.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=value:
        raise ValueError('Cannot replace frozen selection')
    dump(p,value)
    return value


def validated_selection():
    m=manifest();selection=json.loads((RESULT/'selection.json').read_text(encoding='utf-8'))
    selected=selection.get('selected',[])
    if (selection.get('status')!='FROZEN_SELECTION' or
        selection.get('execution_manifest_sha256')!=sha(RESULT/'execution_manifest.json') or
        selection.get('rule')!=m['config']['confirmation']['selection_rule'] or
        not isinstance(selected,list) or len(selected)>2 or len(set(selected))!=len(selected) or
        any(a not in m['config']['arms'] or a==m['config']['reference'] for a in selected)):
        raise ValueError('Invalid frozen selection binding or bounded candidates')
    cfg,rows=collect_results(True)
    ref=rows[(cfg['reference'],42)]
    ranked=[]
    for arm in cfg['arms']:
        if arm==cfg['reference']:continue
        r=rows[(arm,42)]
        ranked.append(dict(arm=arm,delta=r['best']['mAP50_95']-ref['best']['mAP50_95'],
                           endpoint_delta=r['endpoint']['mAP50_95']-ref['endpoint']['mAP50_95'],
                           estimated_macs=r['model_info']['training_form']['estimated_macs']))
    ranked.sort(key=lambda r:(-r['delta'],-r['endpoint_delta'],r['estimated_macs'],r['arm']))
    if selection.get('ranking')!=ranked or selected!=[r['arm'] for r in ranked if r['delta']>0][:2]:
        raise ValueError('Stored selection differs from preregistered result ranking')
    return selection


def report():
    cfg,rows=collect_results();ref_id=cfg['reference'];comparisons=[]
    for arm in cfg['arms']:
        if (arm,42) not in rows:continue
        r=rows[(arm,42)];ref=rows.get((ref_id,42))
        seed_deltas={}
        for seed in (42,43,44):
            if (arm,seed) in rows and (ref_id,seed) in rows:
                a,c=rows[(arm,seed)],rows[(ref_id,seed)]
                seed_deltas[str(seed)]=dict(best=a['best']['mAP50_95']-c['best']['mAP50_95'],
                                            endpoint=a['endpoint']['mAP50_95']-c['endpoint']['mAP50_95'])
        confirmed=len(seed_deltas)==3
        benefit=None
        if confirmed:
            bs=[d['best'] for d in seed_deltas.values()];es=[d['endpoint'] for d in seed_deltas.values()]
            held=[seed_deltas[str(s)] for s in (43,44)]
            benefit=(statistics.mean(bs)>0 and statistics.mean(es)>0 and
                     sum(v>0 for v in bs)>=2 and sum(v>0 for v in es)>=2 and
                     all(d['best']>0 for d in held) and statistics.mean(d['endpoint'] for d in held)>0)
        comparisons.append(dict(arm=arm,label=cfg['arms'][arm]['label'],seed42_best=r['best'],
            seed42_endpoint=r['endpoint'],seed42_best_domain=r['best_domain'],
            Params=r['model_info']['training_form']['total_params'],
            estimated_macs=r['model_info']['training_form']['estimated_macs'],
            reference_macs=ref['model_info']['training_form']['estimated_macs'] if ref else None,
            seed_deltas=seed_deltas,three_seed_confirmed=confirmed,
            stable_valid_accuracy_signal=benefit,test_evaluated=False))
    queue_path=RESULT/'queue_status.json'
    queue=json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else {'status':'NOT_LAUNCHED'}
    value=dict(status=queue['status'],completed_runs=len(rows),maximum_runs=cfg['maximum_runs'],
               comparisons=comparisons,updated_at=now(),test_evaluated=False,
               efficiency_scope='Conv and fixed-functional MAC estimates; timings under shared GPU are descriptive',
               software_vs_hardware='No integer/HLS/route/board/power validation in this campaign')
    dump(RESULT/'comparison.json',value)
    lines=['# 192背景下的候选操作与P2融合验证','',f'状态：{value["status"]}；完成 {len(rows)} 跑，最多22跑。',
           '','冻结B1低通参考；train330/valid99；从零80轮，早期最佳与第80轮均报告。',
           '新增操作、连接和融合属于扩展实验，不是原192空间的全量搜索。','',
           '|实验臂|seed42 best mAP %|seed42 endpoint %|Δbest pp|Δendpoint pp|参数|G MACs|确认|',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        d=r['seed_deltas'].get('42',{})
        signal='待确认' if not r['three_seed_confirmed'] else ('通过' if r['stable_valid_accuracy_signal'] else '未通过')
        lines.append(f'|{r["arm"]} {r["label"]}|{r["seed42_best"]["mAP50_95"]*100:.4f}|{r["seed42_endpoint"]["mAP50_95"]*100:.4f}|{d.get("best",0)*100:+.4f}|{d.get("endpoint",0)*100:+.4f}|{r["Params"]}|{r["estimated_macs"]/1e9:.4f}|{signal}|')
    lines += ['','选择最多前两项seed42正增量；seed43/44的确认增量单独保留在comparison.json。',
              '三种子均值包含筛选种子，仍有选择偏差；同99图valid不代表独立测试泛化。',
              '低对比分组为框/环统计代理；纯背景样本缺失与标注完整性问题保持原有边界。',
              'MAC不是FPGA资源或板端延迟，GPU并行训练下的时间不能用作可靠效率比较。','']
    (RESULT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','smoke','run','select','report'])
    parser.add_argument('--arm');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.mode=='run':run(args.arm,args.seed)
    else:
        value=globals()[args.mode]()
        if args.mode!='smoke':print(json.dumps({k:value[k] for k in ('status','completed_runs','selected') if k in value}),flush=True)
