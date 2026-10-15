"""Bounded remote-only random search of the two registered detector spaces.

Existing models, data, loss, augmentation and COCO evaluator are reused.
Fresh initialization overrides the two space-specific new-kernel seed salts.
No test access, weights inheritance, hardware search or silent job retry.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tools/sonar_spec'),
                str(ROOT / 'tools/sonar_analysis'), str(ROOT / 'scripts')]
import numpy as np
import torch
from torch import nn
import yaml
from tools.sonar_joint192_validation import train as inherited
from tools.sonar_joint192_validation import model as joint
from tools.modnas_elastic import model as elastic
from tools.detector_elastic_pilot.model import build_model as build_reference
from tools.sonar_joint192_validation.domain_metrics import evaluate_domain_metrics
from formal_runner import atomic_save, rng_state, train_epoch
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, decode_detection_predictions

CONFIG = ROOT / 'configs/sonar_space_comparison_20261010.yaml'
RESULT = ROOT / 'results/sonar_space_comparison_20261010_01/remote'
SPACES = (elastic.SEARCH_SPACE_VERSION, joint.SEARCH_SPACE_VERSION)
MODULES = dict(zip(SPACES, (elastic, joint)))
POLICY = 'common_E1_reference_matching_tensors_semantic_new_shapes_v1'
dump, sha, digest, now = inherited.dump, inherited.sha, inherited.canonical_sha, inherited.now
frozen = inherited.frozen


def config():
    value = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if tuple(value['spaces']) != SPACES or value['training']['epochs'] != 80:
        raise ValueError('Unregistered space or epoch protocol')
    return value


def require_remote_cuda():
    if sys.platform != 'linux' or not torch.cuda.is_available():
        raise RuntimeError('Experiment execution requires the existing Linux CUDA server')
    actual = inherited.environment('cuda')
    spec = yaml.safe_load((ROOT / config()['environment_spec']).read_text())
    for key, value in spec['expected'].items():
        if actual.get(key) != str(value):
            raise ValueError('Pinned runtime drift: ' + key)
    if actual['gpu'] != spec['gpu']:
        raise ValueError('Pinned GPU model drift')
    torch.set_num_threads(2)
    return actual


def build(space, architecture, seed):
    """All common tensors copy the same scratch E1; new Conv weights seed by name/shape."""
    if space not in SPACES or seed not in (42, 43, 44):
        raise ValueError('Unregistered model or seed')
    with torch.random.fork_rng(devices=[]):
        model = (elastic.StaticElasticNeckDetector(architecture, seed) if space == SPACES[0]
                 else joint.build_model(architecture, seed))
        reference = build_reference('E1', seed).state_dict()
        state = model.state_dict()
        for name, tensor in state.items():
            if name in reference and reference[name].shape == tensor.shape:
                state[name] = reference[name].clone()
            elif name.endswith('weight') and tensor.ndim == 4:
                semantic = digest(dict(policy=POLICY, seed=seed, name=name, shape=list(tensor.shape)))
                torch.random.default_generator.manual_seed(int(semantic[:16], 16) % (2**63 - 1))
                nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        model.load_state_dict(state, strict=True)
    identity = dict(policy=POLICY, seed=seed, model_sha256=frozen.state_hash(model),
        head_sha256=inherited.canonical_sha(inherited.tensor_json(model.head.state_dict())),
        lateral_sha256=inherited.canonical_sha(inherited.tensor_json(model.neck.lateral_convs.state_dict())))
    model._comparison_initialization = identity
    return model


def measure_cost(model):
    state = frozen.state_hash(model)
    info = frozen.get_model_info(model, (1, 1, 640, 640))
    if frozen.state_hash(model) != state:
        raise ValueError('Cost measurement changed initial model state')
    blur = isinstance(model.backbone.stages[1][0], joint.BlurDownMBConv)
    fixed = 48 * 160 * 160 * 9 if blur else 0
    return dict(Params=info['total_params'], MACs=info['estimated_macs'] + fixed,
                conv_module_MACs=info['estimated_macs'], fixed_blur_MACs=fixed)


def verify_source(manifest):
    for rel, expected in manifest['source_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Frozen campaign source drift: ' + rel)
    if sha(CONFIG) != manifest['config_sha256']:
        raise ValueError('Frozen campaign configuration drift')
    if (ROOT / manifest['baseline']['dataset']['data_dir'] / 'test').exists():
        raise ValueError('Test mounted')


def manifest():
    value = json.loads((RESULT / 'execution_manifest.json').read_text())
    verify_source(value)
    return value


def prepare():
    require_remote_cuda()
    if (RESULT / 'execution_manifest.json').exists():
        raise FileExistsError('Campaign already prepared; refuse to replace manifest')
    started = time.monotonic()
    cfg = config()
    old, baseline = inherited.load_config()
    for key in ('epochs', 'batch_size', 'lr', 'weight_decay', 'scheduler', 'amp'):
        if cfg['training'][key] != baseline['training'][key]:
            raise ValueError('Frozen training protocol mismatch: ' + key)
    inputs = inherited.verify_inputs(old, baseline)
    staging = json.loads((ROOT / 'staging_manifest.json').read_text())
    inventory = {}
    for space in SPACES:
        rows = []
        for arch in MODULES[space].enumerate_architectures():
            candidate = build(space, arch, 42).to('cuda')
            rows.append(dict(space=space, architecture=arch,
                architecture_id=MODULES[space].architecture_id(arch),
                architecture_sha256=MODULES[space].architecture_hash(arch), **measure_cost(candidate)))
            del candidate
        if len(rows) != MODULES[space].ARCHITECTURE_COUNT:
            raise ValueError('Space enumeration count differs')
        inventory[space] = rows
        print('INVENTORY', space, len(rows), flush=True)
    upper = min(max(r['MACs'] for r in rows) for rows in inventory.values())
    ref = next(r for r in inventory[SPACES[0]] if r['architecture'] ==
               dict(pan=False, td_kernels=[3, 3, 3], pan_kernels=[]))['MACs']
    # A +/-2% common band bounds any cross-space pair's MAC ratio below 1.05.
    band = [ref * .98, ref * 1.02]
    eligible = {space: [r for r in rows if r['MACs'] <= upper] for space, rows in inventory.items()}
    if any(not any(band[0] <= r['MACs'] <= band[1] for r in rows) for rows in eligible.values()):
        raise ValueError('No common matched-cost band')
    orders = {}
    for space in SPACES:
        for order_seed in cfg['search_order_seeds']:
            rows = deepcopy(eligible[space])
            random.Random(order_seed).shuffle(rows)
            orders[f'{space}/search_{order_seed}'] = rows
    payload = dict(version=cfg['version'], created_at=now(), config=cfg,
        config_sha256=sha(CONFIG), source_sha256=staging['source_sha256'],
        environment=require_remote_cuda(), baseline=baseline,
        input_verification=inputs, inventory=inventory, eligible_counts={s:len(v) for s,v in eligible.items()},
        MAC_upper_bound=upper, matched_MAC_band=band, random_orders=orders,
        cost_policy='Conv2d Linear MACs plus functional fixed blur; excludes norm activation pooling resize add',
        search_training_seed=cfg['candidate_training_seed'], confirmation_seeds=cfg['confirmation_seeds'],
        initialization_policy=POLICY, test_evaluated=False,
        common_preparation_seconds=time.monotonic()-started)
    verify_source(payload)
    dump(RESULT / 'execution_manifest.json', payload)
    dump(RESULT / 'inventory.json', dict(inventory=inventory, eligible_counts=payload['eligible_counts'],
        MAC_upper_bound=upper, matched_MAC_band=band))
    print(json.dumps(dict(status='PREPARED', eligible_counts=payload['eligible_counts'],
        MAC_upper_bound=upper, matched_MAC_band=band,
        common_preparation_seconds=payload['common_preparation_seconds'])), flush=True)


class BudgetExpired(RuntimeError):
    pass


class DeadlineLoader:
    def __init__(self, loader, deadline):
        self.loader, self.deadline = loader, deadline
    def __iter__(self):
        for batch in self.loader:
            if time.monotonic() >= self.deadline:
                raise BudgetExpired('Search budget exhausted at batch boundary')
            yield batch
    def __getattr__(self, key):
        return getattr(self.loader, key)


def run_components(request, folder, baseline):
    seed = request['seed']
    frozen.seed_all(seed)
    model = build(request['space'], request['architecture'], seed).to('cuda')
    initial = deepcopy(model._comparison_initialization)
    frozen.seed_all(seed)  # Construction/profiling must not advance augmentation/sampler RNG.
    train_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'train', seed, 0, train=True)
    valid_loader = frozen.make_loader(baseline, ROOT / baseline['dataset']['data_dir'], 'valid', seed, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    dump(folder / 'initialization.json', initial)
    dump(folder / 'rng_identity.json', dict(seed=seed,
        torch_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        numpy_sha256=digest(np.random.get_state()[1].tolist()), python_sha256=digest(random.getstate())))
    return model, train_loader, valid_loader, criterion, optimizer, initial


def smoke():
    m = manifest()
    started = time.monotonic()
    requests = []
    # References certify space parity, k5 certifies semantic initialization;
    # maximal-cost branches certify PAN, identity and BlurDown dispatch.
    for space in SPACES:
        rows = m['inventory'][space]
        if space == SPACES[0]:
            picks = [next(r for r in rows if r['architecture'] == dict(pan=False, td_kernels=[k]*3, pan_kernels=[])) for k in (3,5)]
            picks += [max(rows, key=lambda r:r['MACs'])]
        else:
            picks = [next(r for r in rows if r['architecture'] == dict(
                backbone=dict(stage1=dict(kernel=3,downsample='strided_dw'), stage2=dict(kernel=5)),
                neck=dict(td_ops=dict(P2=f'mbconv_k{k}',P3=f'mbconv_k{k}',P4=f'mbconv_k{k}'),pan=False))) for k in (3,5)]
            picks += [max(rows, key=lambda r:r['MACs']), next(r for r in rows if
                r['architecture']['neck']['td_ops']['P2']=='identity' and
                r['architecture']['backbone']['stage1']['downsample']=='blur3_after_dense_dw')]
        requests += [dict(**r, seed=42) for r in picks]
    reports = []
    states = {}
    for idx, request in enumerate(requests):
        folder = RESULT / 'smoke' / str(idx)
        if folder.exists():
            raise FileExistsError('Existing smoke invocation')
        folder.mkdir(parents=True)
        model, train_loader, valid_loader, criterion, optimizer, initial = run_components(request, folder, m['baseline'])
        if idx in (0,1,3,4):
            states[idx] = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        batches = []
        model.train()
        torch.cuda.reset_peak_memory_stats()
        for i, (images, targets) in enumerate(train_loader):
            if i == 2: break
            optimizer.zero_grad(set_to_none=True)
            output = model(images.to('cuda')); frozen.finite_outputs(output)
            loss, parts = criterion(output, targets)
            if not torch.isfinite(loss): raise ValueError('Nonfinite smoke loss')
            loss.backward(); gradients = frozen.gradient_summary(model); optimizer.step()
            batches.append(dict(images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                targets_sha256=digest(inherited.tensor_json(targets)), parts=parts, gradients=gradients))
        metrics, _ = frozen.evaluate(model, valid_loader, criterion, m['baseline'], 'cuda', max_batches=1)
        report = dict(status='PASS', request=request, initial=initial, batches=batches,
            validation=metrics, environment=require_remote_cuda(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), formal_eligible=False, test_evaluated=False)
        if len(batches)!=2 or metrics['num_images']!=4:
            raise ValueError('Smoke batch/validation count differs')
        dump(folder / 'report.json', report); reports.append(report)
        print('SMOKE_PASS', idx, request['space'], flush=True)
        del model, optimizer, criterion, train_loader, valid_loader
        gc.collect(); torch.cuda.empty_cache()
    for left,right in ((0,3),(1,4)):
        if set(states[left]) != set(states[right]) or any(not torch.equal(v,states[right][k]) for k,v in states[left].items()):
            raise ValueError('Cross-space common architecture initialized differently')
    for idx in (3,4):
        for key in ('mAP50_95','AP50','AP75','Precision','Recall'):
            if not math.isclose(reports[idx]['validation'][key],reports[idx-3]['validation'][key],abs_tol=1e-12,rel_tol=0):
                raise ValueError('Common architecture smoke numerical parity failed')
    if len({digest(r['batches'][i]['images_sha256']) for r in reports for i in [0]})!=1:
        raise ValueError('Unpaired augmentation stream')
    for i in (0,1):
        for key in ('images_sha256','targets_sha256'):
            if len({r['batches'][i][key] for r in reports})!=1:
                raise ValueError('Paired data witness failed')
    result = dict(status='PASS', cases=len(reports), parity_cases=2, common_seconds=time.monotonic()-started,
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in reports),
        execution_manifest_sha256=sha(RESULT/'execution_manifest.json'), test_evaluated=False)
    dump(RESULT / 'smoke_gate.json', result)
    print(json.dumps(result), flush=True)


def latency(model, loader, baseline):
    model.eval(); images, _ = next(iter(loader)); image = images[:1].to('cuda')
    proto = baseline['evaluation_protocol']
    def forward(): return model(image)
    def complete():
        out = forward()
        return decode_detection_predictions(out, config=model.loss_config(baseline),
            score_threshold=proto['ap_decode_score_threshold'], nms_iou_threshold=proto['nms_iou_threshold'],
            max_detections=proto['post_nms_max_detections_per_image'],
            pre_nms_topk=proto['pre_nms_topk'])
    results = {}
    with torch.inference_mode():
        for name, fn in (('model_forward',forward),('model_plus_decode_NMS',complete)):
            for _ in range(config()['latency']['warmup']): fn()
            torch.cuda.synchronize(); values=[]
            for _ in range(config()['latency']['repeats']):
                start=time.perf_counter(); fn(); torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000)
            results[name]=dict(median_ms=statistics.median(values), p95_ms=float(np.percentile(values,95)),
                               mean_ms=statistics.mean(values), repetitions=len(values), raw_ms=values)
    return dict(device=require_remote_cuda(), batch=1, precision='FP32', measurements=results,
                input_transfer_included=False, preprocessing_included=False, FPGA_latency_measured=False)


def candidate(request_path):
    require_remote_cuda(); m=manifest()
    request=json.loads(Path(request_path).read_text()); folder=Path(request['folder'])
    if (folder/'run_binding.json').exists(): raise FileExistsError('Existing candidate refused; no overwrite/retry')
    if request['execution_manifest_sha256']!=sha(RESULT/'execution_manifest.json'):
        raise ValueError('Request manifest identity differs')
    module=MODULES[request['space']]
    if module.architecture_hash(request['architecture'])!=request['architecture_sha256']:
        raise ValueError('Request architecture identity differs')
    folder.mkdir(parents=True,exist_ok=True)
    binding=dict(request_sha256=sha(request_path),execution_manifest_sha256=request['execution_manifest_sha256'],
        space=request['space'],architecture_sha256=request['architecture_sha256'],seed=request['seed'],
        phase=request['phase'],fresh_initialization=True,test_evaluated=False)
    dump(folder/'run_binding.json',binding)
    started=time.monotonic(); deadline=request.get('deadline_monotonic',float('inf'))
    history=[]; best=-1.; initial=None
    dump(folder/'status.json',dict(status='RUNNING',epoch=0,binding=binding,updated_at=now()))
    try:
        model,train_loader,valid_loader,criterion,opt,initial=run_components(request,folder,m['baseline'])
        train_bounded=DeadlineLoader(train_loader,deadline); valid_bounded=DeadlineLoader(valid_loader,deadline)
        best_metrics=None
        for epoch in range(80):
            tick=time.monotonic()
            record=train_epoch(model,opt,train_bounded,m['baseline'],'cuda')
            metrics,records=frozen.evaluate(model,valid_bounded,criterion,m['baseline'],'cuda')
            if record['batches']!=83 or metrics['num_images']!=99: raise ValueError('Frozen data coverage differs')
            record.update(epoch=epoch+1,valid=metrics,wall_seconds=time.monotonic()-tick); history.append(record)
            state=dict(binding=binding,architecture=request['architecture'],initialization=initial,
                model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),rng=rng_state(train_loader),
                epoch=epoch+1,global_step=83*(epoch+1),history=history,test_evaluated=False)
            if metrics['mAP50_95']>best:
                best=metrics['mAP50_95']; best_metrics=metrics
                atomic_save(folder/'best.pt',state); dump(folder/'best_valid_records.json',records)
            atomic_save(folder/'latest.pt',state); dump(folder/'epoch_log.json',history)
            dump(folder/'status.json',dict(status='RUNNING',epoch=epoch+1,binding=binding,
                latest_valid={k:metrics[k] for k in inherited.METRICS},updated_at=now()))
            print(request['space'],request['phase'],request['seed'],f'epoch={epoch+1}/80',f'mAP50_95={metrics["mAP50_95"]:.8f}',flush=True)
        if time.monotonic()>=deadline: raise BudgetExpired('Completion exceeded search budget')
        selected=max(history,key=lambda r:r['valid']['mAP50_95'])
        result=dict(status='COMPLETE',**binding,architecture=request['architecture'],
            architecture_id=request['architecture_id'],MACs=request['MACs'],Params=request['Params'],
            best_epoch=selected['epoch'],**{k:best_metrics[k] for k in inherited.METRICS},
            endpoint_mAP50_95=history[-1]['valid']['mAP50_95'],
            best_checkpoint_sha256=sha(folder/'best.pt'),history_sha256=sha(folder/'epoch_log.json'),
            full_training_epochs=80,global_step=6640,initialization=initial,folder=str(folder))
        if request['phase']=='confirmation':
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            if state['binding']!=binding or state['epoch']!=selected['epoch']: raise ValueError('Best checkpoint mismatch')
            model.load_state_dict(state['model_state_dict'],strict=True)
            replay,replayed_records=frozen.evaluate(model,valid_loader,criterion,m['baseline'],'cuda')
            for key in inherited.METRICS:
                if not math.isclose(replay[key],result[key],rel_tol=0,abs_tol=1e-12):
                    raise ValueError('Saved checkpoint inference replay differs: '+key)
            result['checkpoint_replay']='PASS'
            domain=evaluate_domain_metrics(replayed_records,m['baseline'])
            dump(folder/'domain_metrics.json',domain)
            result['domain']=dict(extremely_small_recall=domain['groups']['very_small_min_side']['Recall'],
                lowcontrast_recall=domain['groups']['contrast_low']['Recall'],
                FP_per_image=domain['overall']['FP_per_image'],groups_GT={k:v['GT'] for k,v in domain['groups'].items()})
            result['latency']=latency(model,valid_loader,m['baseline'])
        result['total_seconds']=time.monotonic()-started
        dump(folder/'result.json',result); dump(folder/'status.json',dict(status='COMPLETE',epoch=80,binding=binding,updated_at=now()))
        return result
    except Exception as error:
        dump(folder/'status.json',dict(status='BUDGET_EXHAUSTED' if isinstance(error,BudgetExpired) else 'FAILED_NO_AUTORETRY',
            completed_epochs=len(history),error_type=type(error).__name__,error=str(error),binding=binding,updated_at=now()))
        raise


def invoke(request, timeout):
    folder=Path(request['folder']); folder.mkdir(parents=True,exist_ok=True)
    path=folder/'request.json'
    if path.exists(): raise FileExistsError('Existing candidate request')
    dump(path,request)
    with (folder/'stdout.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__)),'candidate','--request',str(path)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(folder/'child_process.json',dict(pid=proc.pid,created_at=now(),owned_by_campaign=True))
        try:
            code=proc.wait(timeout=max(.01,timeout))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            dump(folder/'watchdog.json',dict(status='BUDGET_TIMEOUT_OWNED_CHILD_TERMINATED',pid=proc.pid,updated_at=now()))
            return None, 'TIMEOUT'
    if code:
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if status.get('status')=='BUDGET_EXHAUSTED': return None,'TIMEOUT'
        raise RuntimeError(f'Candidate failed; no auto retry: {folder} exit={code}')
    result=json.loads((folder/'result.json').read_text())
    if result['status']!='COMPLETE' or result['full_training_epochs']!=80 or result['global_step']!=6640:
        raise ValueError('Incomplete candidate must not enter ranking')
    return result,'COMPLETE'


def select(rows,m):
    def winner(values):
        return min(values,key=lambda r:(-r['mAP50_95'],r['MACs'],r['architecture_id'])) if values else None
    lo,hi=m['matched_MAC_band']
    return dict(overall=winner(rows),matched=winner([r for r in rows if lo<=r['MACs']<=hi]))


def search(space,order_seed,gpu):
    m=manifest(); cfg=m['config']; order=m['random_orders'][f'{space}/search_{order_seed}']
    folder=RESULT/'search'/space/f'search_{order_seed}'
    if (folder/'search_status.json').exists(): raise FileExistsError('Search already invoked')
    start=time.monotonic(); deadline=start+cfg['gpu_hours_per_space_per_search']*3600
    status=dict(status='RUNNING',space=space,order_seed=order_seed,gpu=gpu,started_at=now(),
        completed=[],attempts=[],active=None,gpu_hours=0.,eligible_count=len(order),test_evaluated=False)
    dump(folder/'search_status.json',status)
    for index,row in enumerate(order):
        if time.monotonic()>=deadline: break
        request=dict(**row,seed=cfg['candidate_training_seed'],phase='search',
            folder=str(folder/f'candidate_{index:03d}'),deadline_monotonic=deadline,
            execution_manifest_sha256=sha(RESULT/'execution_manifest.json'))
        status.update(active=dict(index=index,architecture_id=row['architecture_id']),updated_at=now())
        dump(folder/'search_status.json',status)
        before=time.monotonic()
        result, outcome=invoke(request,deadline-time.monotonic())
        elapsed=(time.monotonic()-start)/3600
        status['attempts'].append(dict(index=index,architecture_id=row['architecture_id'],outcome=outcome,
            charged_seconds=time.monotonic()-before,cumulative_GPU_hours=elapsed))
        if result is not None and time.monotonic()<=deadline:
            result.update(completion_GPU_hours=elapsed)
            status['completed'].append(result)
        elif result is not None:
            status['attempts'][-1]['outcome']='FINISHED_AFTER_BUDGET_EXCLUDED'
        status.update(active=None,gpu_hours=elapsed,updated_at=now())
        dump(folder/'search_status.json',status)
        if outcome=='TIMEOUT': break
    status.update(status='COMPLETE',gpu_hours=(time.monotonic()-start)/3600,
        selected=select(status['completed'],m),ended_at=now())
    best=-1.; curve=[]
    for row in status['completed']:
        best=max(best,row['mAP50_95']);curve.append(dict(GPU_hours=row['completion_GPU_hours'],best_valid_mAP50_95=best))
    status['best_so_far_curve']=curve
    dump(folder/'search_status.json',status)
    if not status['completed']: raise RuntimeError('No complete candidate within search budget')
    return status


def confirm(search_status,gpu):
    m=manifest(); selected=search_status['selected']; unique={}
    for role,row in selected.items():
        if row is not None:
            key=row['architecture_sha256'];unique.setdefault(key,dict(row=row,roles=[]))['roles'].append(role)
    search_folder=RESULT/'confirmation'/search_status['space']/f'search_{search_status["order_seed"]}'
    slots=[]
    for item in unique.values():
        row=item['row']; runs=[]
        for seed in m['config']['confirmation_seeds']:
            request={key:row[key] for key in ('space','architecture','architecture_id','architecture_sha256','MACs','Params')}
            request.update(seed=seed,phase='confirmation',
                folder=str(search_folder/row['architecture_sha256'][:16]/f'seed_{seed}'),
                execution_manifest_sha256=sha(RESULT/'execution_manifest.json'))
            result,outcome=invoke(request,m['config']['confirmation']['maximum_hours_per_training_run']*3600)
            if outcome!='COMPLETE': raise RuntimeError('Confirmation exceeded frozen per-run limit')
            runs.append(result)
        values=[r['mAP50_95'] for r in runs]
        slots.append(dict(roles=item['roles'],architecture_id=row['architecture_id'],architecture_sha256=row['architecture_sha256'],
            MACs=row['MACs'],Params=row['Params'],runs=runs,
            mean_mAP50_95=statistics.mean(values),std_mAP50_95=statistics.stdev(values)))
        dump(search_folder/'confirmation_summary.json',dict(status='RUNNING',slots=slots,gpu=gpu,test_evaluated=False))
    summary=dict(status='COMPLETE',space=search_status['space'],order_seed=search_status['order_seed'],slots=slots,
        missing_matched=selected['matched'] is None,gpu=gpu,test_evaluated=False)
    dump(search_folder/'confirmation_summary.json',summary)
    return summary


def worker(gpu):
    require_remote_cuda();m=manifest()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu) or gpu not in m['config']['gpu_indices']:
        raise ValueError('GPU assignment mismatch')
    gate=json.loads((RESULT/'smoke_gate.json').read_text())
    if gate['status']!='PASS' or gate['execution_manifest_sha256']!=sha(RESULT/'execution_manifest.json'):
        raise ValueError('Bound real CUDA smoke required')
    tasks = [(SPACES[(index+(0 if gpu==m['config']['gpu_indices'][0] else 1))%2],seed)
             for index,seed in enumerate(m['config']['search_order_seeds'])]
    folder=RESULT/f'worker_gpu{gpu}';folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (folder/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=folder/'queue_status.json'
        if path.exists(): raise FileExistsError('Worker already invoked')
        status=dict(status='RUNNING',pid=os.getpid(),gpu=gpu,tasks=tasks,completed=[],active=None,started_at=now(),test_evaluated=False)
        dump(path,status)
        try:
            searches=[]
            # All search jobs on this worker finish before its confirmation queue.
            for space,seed in tasks:
                status.update(active=dict(phase='search',space=space,order_seed=seed),updated_at=now());dump(path,status)
                searches.append(search(space,seed,gpu));status['completed'].append(dict(phase='search',space=space,order_seed=seed));dump(path,status)
            for s in searches:
                status.update(active=dict(phase='confirmation',space=s['space'],order_seed=s['order_seed']),updated_at=now());dump(path,status)
                confirm(s,gpu);status['completed'].append(dict(phase='confirmation',space=s['space'],order_seed=s['order_seed']));dump(path,status)
            status.update(status='COMPLETE',active=None,ended_at=now());dump(path,status)
            report()
        except Exception as error:
            status.update(status='FAILED_NO_AUTORETRY',error=str(error),updated_at=now());dump(path,status)
            raise


def report():
    m=manifest();summary={};complete=True
    for space in SPACES:
        searches=[];confirmations=[]
        for seed in m['config']['search_order_seeds']:
            sp=RESULT/'search'/space/f'search_{seed}'/'search_status.json'
            cp=RESULT/'confirmation'/space/f'search_{seed}'/'confirmation_summary.json'
            if sp.exists(): searches.append(json.loads(sp.read_text()))
            if cp.exists(): confirmations.append(json.loads(cp.read_text()))
        complete &= len(searches)==3 and len(confirmations)==3 and all(x['status']=='COMPLETE' for x in confirmations+searches)
        aggregate={}
        for role in ('overall','matched'):
            slots=[slot for c in confirmations if c['status']=='COMPLETE' for slot in c['slots'] if role in slot['roles']]
            values=[s['mean_mAP50_95'] for s in slots]
            aggregate[role]=dict(n_independent_searches=len(values),search_winner_seed_means=values,
                mean=statistics.mean(values) if values else None,
                between_search_std=statistics.stdev(values) if len(values)>1 else None,
                mean_within_architecture_std=statistics.mean(s['std_mAP50_95'] for s in slots) if slots else None,
                mean_MACs=statistics.mean(s['MACs'] for s in slots) if slots else None,
                mean_Params=statistics.mean(s['Params'] for s in slots) if slots else None)
            runs=[r for slot in slots for r in slot['runs']]
            for key in ('extremely_small_recall','lowcontrast_recall','FP_per_image'):
                metric=[r['domain'][key] for r in runs if r['domain'][key] is not None]
                aggregate[role]['mean_'+key]=statistics.mean(metric) if metric else None
            for key in ('model_forward','model_plus_decode_NMS'):
                metric=[r['latency']['measurements'][key]['median_ms'] for r in runs]
                aggregate[role]['mean_'+key+'_median_ms']=statistics.mean(metric) if metric else None
        summary[space]=dict(search_statuses=searches,confirmations=confirmations,aggregates=aggregate)
    value=dict(status='COMPLETE' if complete else 'IN_PROGRESS',spaces=summary,updated_at=now(),
        selection_bias='valid-selected checkpoints and architectures; not untouched-test generalization',
        test_evaluated=False,execution_manifest_sha256=sha(RESULT/'execution_manifest.json'))
    dump(RESULT/'comparison_report.json',value)
    lines=['# 72 与 192 空间随机搜索比较', '', '状态：'+value['status'], '',
        '冻结 train/valid 划分；80 轮独立训练；测试集未挂载。检查点和架构均由 valid 选择，结果属于开发集比较。', '',
        '| 空间 | 选择方式 | 独立搜索数 | mAP50–95 均值 | 搜索间标准差 | 平均训练种子标准差 | 平均 G MACs | 极小目标召回 | 低对比召回 | FP/图 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return 'PENDING' if x is None else f'{x:.6f}'
    for space in SPACES:
        for role in ('overall','matched'):
            a=summary[space]['aggregates'][role]
            cells=[space,role,str(a['n_independent_searches']),fmt(a['mean']),fmt(a['between_search_std']),
                fmt(a['mean_within_architecture_std']),fmt(a['mean_MACs']/1e9 if a['mean_MACs'] else None),
                fmt(a['mean_extremely_small_recall']),fmt(a['mean_lowcontrast_recall']),fmt(a['mean_FP_per_image'])]
            lines.append('| '+' | '.join(cells)+' |')
    lines += ['',f'共同 MACs 上限：{m["MAC_upper_bound"]:,}；匹配区间：{m["matched_MAC_band"]}。',
        '匹配区间为共同 E0 计算量 ±2%，任意两候选的 MACs 比值小于 1.05。',
        '搜索曲线仅接纳预算内完成 80 轮的候选；失败/超时尝试消耗预算但不进入排名。',
        '延迟为服务器 FP32、batch=1；模型前向与含解码/NMS分开报告，未测 FPGA 延迟。',
        '共同准备/冒烟成本与搜索/复训成本分别保留。每个空间只有三次独立搜索，不能将九个复训值视作九次独立搜索。',
        '低对比和极小目标分组重叠；置信度 0.5、匹配 IoU 0.5；阈值源于训练集，未按结果调整。']
    path=RESULT/'COMPARISON_REPORT.md';tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');tmp.replace(path)
    curves=[]
    for space in SPACES:
        for search_status in summary[space]['search_statuses']:
            for point in search_status.get('best_so_far_curve',[]):
                curves.append(dict(space=space,search_order_seed=search_status['order_seed'],**point))
    if curves:
        tmp=RESULT/'best_so_far_curves.csv.tmp'
        with tmp.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
        tmp.replace(RESULT/'best_so_far_curves.csv')
    print(json.dumps(dict(status=value['status'],spaces={s:summary[s]['aggregates'] for s in SPACES})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','smoke','candidate','worker','report'])
    parser.add_argument('--request',type=Path)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.mode=='candidate': candidate(args.request)
    elif args.mode=='worker': worker(args.gpu)
    else: globals()[args.mode]()
