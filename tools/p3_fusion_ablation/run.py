"""One-factor A1 P3 fusion comparison using the original formal train/eval loop."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/p3_fusion_ablation_20261003_02'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/sonar_spec')]
import torch
import yaml
from formal_protocol import frozen
from formal_runner import train_epoch, atomic_save, rng_state, restore_rng
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from hwnas_fpga.models.sonar_p3_fusion import (
    A1_ARCHITECTURE_ID, A1_ARCHITECTURE, build_p3_fusion_candidate, p3_fusion_parameter_groups,
    register_p3_alpha_projection)
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

VARIANTS = ('equal_add', 'weighted')
SEEDS = (42, 43, 44)
METRICS = ('mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall')
COMMIT_SCHEMA = 'p3_fusion_epoch_commit_v1'
VALID_IMAGES = 99


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, obj): frozen.dump(path, obj)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def folder(variant, seed): return OUT/'runs'/variant/f'seed_{seed}'


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def shared_digest(model):
    return digest({k:v for k,v in model.state_dict().items() if k!='neck.alpha'})


def baseline():
    return yaml.safe_load((ROOT/'configs/baseline_sonar_frozen.yaml').read_text(encoding='utf-8'))


def verify_baseline(b):
    """Verify every originally frozen file, allowing unrelated later additions.

    Legacy verification compared whole directory inventories; three later H2
    deployment files cause an inventory mismatch despite all 42 original hashes
    matching.  Do not delete those files or weaken any frozen-byte/data check.
    """
    manifest=read(ROOT/b['baseline_freeze']['manifest'])
    normalize=lambda d:{k.replace('\\','/'):v for k,v in d.items()}
    expected=normalize(manifest['source_sha256']);actual=normalize(frozen.source_hashes())
    for relative,h in expected.items():
        if sha(ROOT/relative)!=h:raise ValueError('Original frozen source changed: '+relative)
    extra=sorted(set(actual)-set(expected))
    for split in ('train','valid'):
        loader=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],split,42,0)
        entries=frozen.bind_dataset(loader)['entries'];original=manifest['datasets'][split]['entries']
        clean=lambda rs:[{k:(v.replace('\\','/') if k in ('image','annotation') else v) for k,v in r.items()} for r in rs]
        if clean(entries)!=clean(original):raise ValueError('Frozen dataset changed: '+split)
    return dict(status='PASS',original_source_files=len(expected),all_original_source_hashes_verified=True,
                additional_unrelated_files=extra,manifest_sha256=sha(ROOT/b['baseline_freeze']['manifest']),
                data_verified=['train','valid'],test_accessed=False)


def source_files():
    paths=list(frozen.source_hashes())
    paths += ['configs/baseline_sonar_frozen.yaml','configs/search_space_sonar_v0_1.yaml',
              'reports/baseline_sonar_frozen/freeze_manifest.json',
              'tools/sonar_spec/formal_protocol.py','tools/sonar_spec/formal_runner.py',
              'tools/p3_fusion_ablation/run.py','tools/p3_fusion_ablation/pipeline.py',
              'src/hwnas_fpga/models/sonar_candidate/__init__.py',
              'src/hwnas_fpga/models/sonar_p3_fusion/__init__.py']
    return {str(Path(n)).replace('\\','/'):sha(ROOT/n) for n in sorted(set(paths))}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);b=baseline();protection=verify_baseline(b)
    base_manifest=read(ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json')
    p=dict(version='A1_P3_scalar_fusion_v2',created_date='2026-10-03',
        architecture_id=A1_ARCHITECTURE_ID,architecture=A1_ARCHITECTURE,
        variants=list(VARIANTS),seeds=list(SEEDS),epochs=80,
        question='Does one nonnegative learned deep-branch scalar at P3 improve fixed-A1 detection?',
        scope='A1 backbone architecture, operator positions, P3/P4/P5 outputs, neck96 and head unchanged; ordinary weights train from scratch',
        formula=dict(equal_add='M3=L3+nearest(M4)',weighted='M3=L3+alpha*nearest(M4)'),
        alpha=dict(initial=1.,constraint='project_after_every_optimizer_step_to_[0,infinity)',
                   parameters=1,lr=.001,weight_decay=0.,reason_no_decay='avoid an extra regularization incentive to suppress the deep branch'),
        training=dict(input=640,batch_size=4,optimizer='AdamW',lr=.001,weight_decay=.0001,
                      betas=[.9,.999],eps=1e-8,amp=False,scheduler='none',workers=0,threads=2,
                      gradient_clipping=False,initialization='fresh_per_seed_shared_weights_exactly_equal_between_variants',
                      checkpoint_selection='earliest_best_valid_mAP50_95_within80',endpoint=80,
                      loop='unchanged formal_runner.train_epoch',evaluator='unchanged baseline.evaluate',
                      deterministic='original seed_all; deterministic_algorithms warn_only=True; no bitwise training guarantee'),
        dataset=dict(train=base_manifest['datasets']['train']['images'],valid=base_manifest['datasets']['valid']['images'],
                     original_manifest_sha256=sha(ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json'),
                     augmentation=b['training']['detection_augmentation'],sampler=b['training']['sampler_protocol']),
        evaluation_protocol=b['evaluation_protocol'],
        gate=dict(supported='all3_best_deltas_positive AND mean_best_delta>sample_std_best_delta AND mean_epoch80_delta>=0',
                  uncertain='mean_best_delta>0 but full direction/dispersion/endpoint gate fails',
                  unsupported='mean_best_delta<=0',
                  rule_kind='preregistered descriptive stability rule, not a statistical significance test'),
        resources=dict(gpus=[2,3],workers_per_gpu=1,allocator_fraction=.20,minimum_free_bytes=16*1024**3,
                       thread_env=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                       note='shared existing GPUs; GPU0/1 not used; no hardware latency or power claim'),
        checkpoint_transaction=dict(schema=COMMIT_SCHEMA,commit_point='atomic_latest.pt_replace',
            resume='latest_only_with_optimizer_and_RNG',best='inference_only_snapshot_inside_latest',
            exports='repair_all_derived_sidecars_from_committed_latest_before_resume_or_summary'),
        source_sha256=source_files(),test_accessed=False,board_accessed=False,
        historical_A1='context only; fresh equal_add is the paired baseline')
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('New experiment protocol drift; use a new run version')
    dump(path,p);dump(OUT/'data_protection.json',protection)
    print('PROTOCOL_FROZEN',sha(path),flush=True);return p,b


def load(require_gate=True):
    p=read(OUT/'protocol.json')
    for name,expected in p['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source drift: '+name)
    b=baseline();verify_baseline(b)
    if require_gate:
        gate=read(OUT/'execution_review.json')
        if gate['status']!='PASS' or gate['protocol_sha256']!=sha(OUT/'protocol.json'):
            raise RuntimeError('Source-bound independent execution review not PASS')
    return p,b


def optimizer_for(model):
    opt=torch.optim.AdamW(p3_fusion_parameter_groups(model,weight_decay=.0001),lr=.001)
    register_p3_alpha_projection(opt,model)
    return opt


def gpu_setup(p):
    if not torch.cuda.is_available():raise RuntimeError('This campaign requires existing CUDA; no silent CPU substitution')
    free,total=torch.cuda.mem_get_info()
    if free<p['resources']['minimum_free_bytes']:raise RuntimeError('GPU below preregistered free-memory headroom')
    torch.cuda.set_per_process_memory_fraction(p['resources']['allocator_fraction'])
    return dict(device=torch.cuda.get_device_name(),free_before=free,total_bytes=total,
                allocator_fraction=p['resources']['allocator_fraction'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def smoke():
    p,b=load(False);admission=gpu_setup(p);checks=[]
    # Fresh initial weights and RNG must match in the actual frozen remote stack.
    for seed in SEEDS:
        frozen.seed_all(seed);original=build_sonar_candidate(SPEC_PATH,A1_ARCHITECTURE).eval()
        expected=shared_digest(original);expected_rng=torch.get_rng_state().clone()
        for variant in VARIANTS:
            frozen.seed_all(seed);m=build_p3_fusion_candidate(variant).eval()
            assert shared_digest(m)==expected and torch.equal(torch.get_rng_state(),expected_rng)
            checks.append(dict(seed=seed,variant=variant,shared_initial_state_sha256=expected,RNG_equal=True))
        del original,m
    frozen.seed_all(42)
    train=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0,train=True)
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    images,targets=next(iter(train));images=images.to('cuda')
    frozen.seed_all(42);original=build_sonar_candidate(SPEC_PATH,A1_ARCHITECTURE).to('cuda').eval()
    with torch.no_grad():reference=original(images)
    del original
    runtime=[]
    for variant in VARIANTS:
        frozen.seed_all(42);m=build_p3_fusion_candidate(variant).to('cuda').eval()
        with torch.no_grad():output=m(images)
        for key in reference:
            assert all(torch.equal(a,z) for a,z in zip(reference[key],output[key])),(variant,key)
        del output
        opt=optimizer_for(m);m.train();criterion=DenseAnchorDetectionLoss(m.loss_config(b)).to('cuda')
        opt.zero_grad(set_to_none=True);output=m(images);loss,parts=criterion(output,targets)
        loss.backward();frozen.gradient_summary(m)
        alpha_grad=float(m.neck.alpha.grad) if variant=='weighted' else None
        if variant=='weighted':assert math.isfinite(alpha_grad) and alpha_grad!=0
        opt.step();assert float(m.neck.alpha)>=0
        metrics,_=frozen.evaluate(m,valid,criterion,b,'cuda',max_batches=1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            profile=frozen.get_model_info(m,(1,1,640,640))
        canonical_profile=json.loads(json.dumps(profile,allow_nan=False))
        profile_path=OUT/f'smoke_model_info_{variant}.json'
        dump(profile_path,canonical_profile)
        assert read(profile_path)==canonical_profile
        runtime.append(dict(variant=variant,loss=float(loss.detach()),alpha=float(m.neck.alpha.detach()),
                            alpha_gradient=alpha_grad,valid_smoke=metrics,finite=True,
                            actual_model_profile_json_roundtrip=True,profile_sha256=sha(profile_path)))
        del m,opt,criterion,output,loss
    witness=dict(status='PASS',protocol_sha256=sha(OUT/'protocol.json'),date=now(),
                 initial_checks=checks,forward_exact=True,real_batch_size=4,runtime=runtime,
                 admission=admission,torch=torch.__version__,numpy=frozen.np.__version__,
                 peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                 scope='smoke only; discarded models; formal jobs fresh initialization',test_accessed=False)
    dump(OUT/'smoke_witness.json',witness);print('SMOKE_PASS',json.dumps({k:witness[k] for k in ('forward_exact','real_batch_size','peak_cuda_allocated')}),flush=True)


def binding(variant,seed):
    return dict(protocol_sha256=sha(OUT/'protocol.json'),architecture_id=A1_ARCHITECTURE_ID,
                variant=variant,seed=seed,epochs=80,test_accessed=False)


def _cpu_clone(value):
    """Copy checkpoint values without drawing randomness or changing live tensors."""
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_clone(v) for v in value)
    return deepcopy(value)


def expected_validation_images():
    manifest=read(ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json')
    images=[r['image'].replace('\\','/') for r in manifest['datasets']['valid']['entries']]
    if len(images)!=VALID_IMAGES or len(set(images))!=VALID_IMAGES:
        raise ValueError('Frozen validation manifest must contain exactly 99 unique images')
    return images


def _require(condition,message):
    if not condition:raise ValueError(message)


def _validate_metrics(metrics):
    _require(metrics['num_images']==VALID_IMAGES,'Validation metrics must cover 99 images')
    for key in METRICS:
        value=metrics[key]
        _require(isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=1,
                 'Invalid committed validation metric: '+key)
    # Include the detailed class AP/protocol/loss payload in the strict JSON check.
    json.dumps(metrics,allow_nan=False)


def _validate_predictions(records):
    _require(isinstance(records,list) and len(records)==VALID_IMAGES,'Predictions must cover 99 images')
    _require([r['image'].replace('\\','/') for r in records]==expected_validation_images(),
             'Prediction image order/identity differs from the frozen validation manifest')
    json.dumps(records,allow_nan=False)
    for record in records:
        for name in ('prediction','target'):
            item=record[name]
            _require(len(item['boxes'])==len(item['labels']),'Prediction/target box-label count mismatch')
            _require(all(len(box)==4 and all(math.isfinite(float(v)) for v in box) for box in item['boxes']),
                     'Invalid validation boxes')
            _require(all(type(label) is int and label in (0,1) for label in item['labels']),
                     'Invalid two-class validation label')
        pred=record['prediction']
        _require(len(pred['scores'])==len(pred['boxes']) and
                 all(math.isfinite(float(v)) and 0<=v<=1 for v in pred['scores']),
                 'Invalid validation prediction scores')


def make_epoch_commit(*,run_binding,history,parents,initial_shared,start_rng,
                      model_state_dict,optimizer_state_dict,rng,records,best_snapshot,model_info):
    """Build the next epoch transaction; only saving latest commits this value.

    Best is the earliest strict improvement and contains inference weights only.
    The full latest state, including optimizer and RNG, is the sole resume source.
    """
    history=deepcopy(history);row=history[-1];weights=_cpu_clone(model_state_dict)
    predictions=deepcopy(records);stamp=now()
    if best_snapshot is None or row['valid']['mAP50_95']>best_snapshot['valid']['mAP50_95']:
        best_snapshot=dict(snapshot_schema=COMMIT_SCHEMA,purpose='inference_only_not_resume',
            binding=deepcopy(run_binding),epoch=row['epoch'],valid=deepcopy(row['valid']),alpha=row['alpha'],
            initial_shared_sha256=initial_shared,train_start_rng_sha256=start_rng,
            model_state_dict=weights,model_state_sha256=digest(weights),
            validation_predictions=predictions,validation_predictions_sha256=digest(predictions),
            updated_at=stamp,test_accessed=False)
    state=dict(commit_schema=COMMIT_SCHEMA,purpose='training_resume',binding=deepcopy(run_binding),
        epoch=row['epoch'],history=history,history_sha256=digest(history),parents=deepcopy(parents),
        initial_shared_sha256=initial_shared,train_start_rng_sha256=start_rng,
        model_state_dict=weights,model_state_sha256=digest(weights),
        optimizer_state_dict=_cpu_clone(optimizer_state_dict),rng=_cpu_clone(rng),
        validation_predictions=predictions,validation_predictions_sha256=digest(predictions),
        best_snapshot=best_snapshot,model_info=json.loads(json.dumps(model_info,allow_nan=False)),
        updated_at=stamp,test_accessed=False)
    state['optimizer_state_sha256']=digest(state['optimizer_state_dict'])
    state['commit_sha256']=digest(state)
    validate_committed(state,run_binding,initial_shared,start_rng)
    return state


def validate_committed(state,expected_binding=None,expected_initial=None,expected_rng=None):
    """Fail closed on malformed commits; derived files are deliberately ignored."""
    _require(state.get('commit_schema')==COMMIT_SCHEMA and state.get('purpose')=='training_resume',
             'Only the current latest training commit can be resumed')
    b=state['binding']
    _require(b['variant'] in VARIANTS and b['seed'] in SEEDS,'Invalid run variant/seed')
    _require(b==binding(b['variant'],b['seed']) and (expected_binding is None or b==expected_binding),
             'Committed protocol/architecture/run binding mismatch')
    _require(state['test_accessed'] is False,'Committed state unexpectedly accessed test')
    _require(expected_initial is None or state['initial_shared_sha256']==expected_initial,
             'Committed shared initialization mismatch')
    _require(expected_rng is None or state['train_start_rng_sha256']==expected_rng,
             'Committed starting RNG mismatch')
    epoch=state['epoch'];history=state['history']
    _require(type(epoch) is int and 1<=epoch<=80 and [r['epoch'] for r in history]==list(range(1,epoch+1)),
             'Committed history must contain exactly epochs 1 through latest')
    _require(state['history_sha256']==digest(history),'Committed history hash mismatch')
    for row in history:
        _validate_metrics(row['valid'])
        _require(math.isfinite(row['alpha']) and row['alpha']>=0 and
                 (b['variant']=='weighted' or row['alpha']==1.),'Committed alpha is invalid')
        _require(math.isfinite(row['wall_seconds']) and row['wall_seconds']>=0,'Invalid epoch duration')
    _validate_predictions(state['validation_predictions'])
    _require(state['validation_predictions_sha256']==digest(state['validation_predictions']),
             'Committed endpoint prediction hash mismatch')
    _require(state['model_state_sha256']==digest(state['model_state_dict']),
             'Committed latest model hash mismatch')
    _require(state['optimizer_state_sha256']==digest(state['optimizer_state_dict']),
             'Committed optimizer hash mismatch')
    _require(float(state['model_state_dict']['neck.alpha'])==history[-1]['alpha'],
             'Latest model alpha differs from committed history')
    best=max(history,key=lambda r:r['valid']['mAP50_95']);snapshot=state['best_snapshot']
    _require(snapshot['snapshot_schema']==COMMIT_SCHEMA and snapshot['purpose']=='inference_only_not_resume',
             'Best snapshot must be inference only')
    _require(not any(k in snapshot for k in ('optimizer_state_dict','rng','history')),
             'Best inference checkpoint cannot carry resume state')
    _require(snapshot['binding']==b and snapshot['test_accessed'] is False and
             snapshot['initial_shared_sha256']==state['initial_shared_sha256'] and
             snapshot['train_start_rng_sha256']==state['train_start_rng_sha256'],
             'Best snapshot binding/initialization mismatch')
    _require(snapshot['epoch']==best['epoch'] and snapshot['valid']==best['valid'] and snapshot['alpha']==best['alpha'],
             'Best snapshot differs from earliest best committed validation epoch')
    _validate_predictions(snapshot['validation_predictions'])
    _require(snapshot['validation_predictions_sha256']==digest(snapshot['validation_predictions']),
             'Committed best prediction hash mismatch')
    _require(snapshot['model_state_sha256']==digest(snapshot['model_state_dict']) and
             float(snapshot['model_state_dict']['neck.alpha'])==snapshot['alpha'],
             'Best model weights/alpha mismatch')
    _require(state['commit_sha256']==digest({k:v for k,v in state.items() if k!='commit_sha256'}),
             'Latest transaction hash mismatch')
    return state


def _export_json(path,value):
    try:
        if read(path)==value:return
    except (OSError,ValueError):pass
    temp=path.with_suffix(path.suffix+'.tmp');dump(temp,value);temp.replace(path)


def materialize_committed(state,d):
    """Repair every export from latest, never from an independent best sidecar."""
    d=Path(d);validate_committed(state)
    b=state['binding']
    _require(d.resolve()==folder(b['variant'],b['seed']).resolve(),'Commit export run directory mismatch')
    on_disk=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
    _require(on_disk.get('commit_sha256')==state['commit_sha256'] and digest(on_disk)==digest(state),
             'Exports require the state already committed as latest.pt')
    snapshot=state['best_snapshot']
    try:
        existing=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
        same=digest(existing)==digest(snapshot)
    except (OSError,ValueError,RuntimeError,EOFError):same=False
    if not same:atomic_save(d/'best.pt',snapshot)
    _export_json(d/'best_valid_metrics.json',snapshot['valid'])
    _export_json(d/'best_valid_predictions.json',snapshot['validation_predictions'])
    _export_json(d/'epoch_log.json',state['history'])
    _export_json(d/'model_info.json',state['model_info'])
    if state['epoch']==80:
        _export_json(d/'endpoint_valid_predictions.json',state['validation_predictions'])
    else:
        # An interrupted future export is not evidence for an uncommitted epoch.
        for name in ('endpoint_valid_predictions.json','result.json'):
            path=d/name
            if path.exists():path.unlink()
    progress=dict(status='TRAINING' if state['epoch']<80 else 'COMMITTED_COMPLETE_REQUIRES_SUMMARY',
        binding=b,epoch=state['epoch'],updated_at=state['updated_at'],alpha=state['history'][-1]['alpha'],
        last_mAP50_95=state['history'][-1]['valid']['mAP50_95'],best_mAP50_95=snapshot['valid']['mAP50_95'],
        latest_checkpoint_sha256=sha(d/'latest.pt'),commit_sha256=state['commit_sha256'],test_accessed=False)
    _export_json(d/'progress.json',progress)
    _export_json(d/'status.json',dict(status=progress['status'],binding=b,epochs=state['epoch'],
        latest_checkpoint_sha256=progress['latest_checkpoint_sha256'],commit_sha256=state['commit_sha256'],test_accessed=False))
    # Assert the actual repaired files, not merely the values intended for export.
    _require(digest(torch.load(d/'best.pt',map_location='cpu',weights_only=False))==digest(snapshot),
             'Materialized best checkpoint mismatch')
    for name,value in (('best_valid_metrics.json',snapshot['valid']),('best_valid_predictions.json',snapshot['validation_predictions']),
                       ('epoch_log.json',state['history']),('model_info.json',state['model_info'])):
        _require(read(d/name)==value,'Materialized sidecar mismatch: '+name)
    if state['epoch']==80:
        _require(read(d/'endpoint_valid_predictions.json')==state['validation_predictions'],
                 'Materialized endpoint prediction mismatch')
    return snapshot


def _verify_prediction_metrics(records,metrics,b):
    recomputed=frozen.metrics_from_records(records,b)
    for key in METRICS:
        _require(math.isclose(recomputed[key],metrics[key],rel_tol=0.,abs_tol=1e-12),
                 'Committed prediction/metric mismatch: '+key)
    for key in ('num_images','coco_details','protocol'):
        _require(recomputed[key]==metrics[key],'Committed detailed prediction/metric mismatch: '+key)


def summarize_one(variant,seed):
    d=folder(variant,seed);s=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
    initial=torch.load(OUT/'initializations'/f'seed_{seed}.pt',map_location='cpu',weights_only=False)
    _require(initial['seed']==seed and initial['shared_sha256']==digest(
        {k:v for k,v in initial['model_state_dict'].items() if k!='neck.alpha'}),
        'Shared initialization reference hash mismatch')
    validate_committed(s,binding(variant,seed),initial['shared_sha256'])
    best_state=materialize_committed(s,d)
    _require(s['epoch']==80,'Only 80 committed epochs can be summarized as complete')
    h=s['history'];best=h[best_state['epoch']-1];info=s['model_info']
    b=baseline()
    _verify_prediction_metrics(best_state['validation_predictions'],best['valid'],b)
    _verify_prediction_metrics(s['validation_predictions'],h[-1]['valid'],b)
    row=dict(variant=variant,seed=seed,best_epoch=best['epoch'],
             best_valid={k:best['valid'][k] for k in METRICS},endpoint_valid={k:h[-1]['valid'][k] for k in METRICS},
             best_class_AP={name:statistics.mean(scores.values()) for name,scores in best['valid']['coco_details']['per_class_ap']['all'].items()},
             endpoint_class_AP={name:statistics.mean(scores.values()) for name,scores in h[-1]['valid']['coco_details']['per_class_ap']['all'].items()},
             best_alpha=best['alpha'],endpoint_alpha=h[-1]['alpha'],initial_shared_sha256=s['initial_shared_sha256'],
             train_start_rng_sha256=s['train_start_rng_sha256'],params=info['total_params'],conv_linear_MACs=info['estimated_macs'],
             added_elementwise_multiplications=614400 if variant=='weighted' else 0,
             best_checkpoint_sha256=sha(d/'best.pt'),latest_checkpoint_sha256=sha(d/'latest.pt'),
             history_sha256=sha(d/'epoch_log.json'),best_prediction_sha256=sha(d/'best_valid_predictions.json'),
             endpoint_prediction_sha256=sha(d/'endpoint_valid_predictions.json'),
             training_seconds=sum(r['wall_seconds'] for r in h),test_accessed=False)
    _export_json(d/'result.json',row)
    _export_json(d/'status.json',dict(status='COMPLETE',binding=s['binding'],epochs=80,
        commit_sha256=s['commit_sha256'],latest_checkpoint_sha256=sha(d/'latest.pt'),
        result_sha256=sha(d/'result.json'),test_accessed=False))
    _export_json(d/'progress.json',dict(status='COMPLETE',binding=s['binding'],epoch=80,
        updated_at=s['updated_at'],alpha=h[-1]['alpha'],last_mAP50_95=h[-1]['valid']['mAP50_95'],
        best_mAP50_95=best['valid']['mAP50_95'],latest_checkpoint_sha256=sha(d/'latest.pt'),
        commit_sha256=s['commit_sha256'],test_accessed=False))
    return row


def train(variant,seed):
    p,b=load();admission=gpu_setup(p);d=folder(variant,seed);d.mkdir(parents=True,exist_ok=True)
    frozen.seed_all(seed);model=build_p3_fusion_candidate(variant).to('cuda')
    initial_shared=shared_digest(model)
    initial_path=OUT/'initializations'/f'seed_{seed}.pt'
    initial_path.parent.mkdir(exist_ok=True)
    if initial_path.exists():
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)
        assert initial['seed']==seed and initial['shared_sha256']==initial_shared
        model.load_state_dict(initial['model_state_dict'],strict=True)
        assert shared_digest(model)==initial_shared
        assert torch.equal(initial['torch_rng_after_build'],torch.get_rng_state())
        del initial
    else:
        atomic_save(initial_path,dict(seed=seed,shared_sha256=initial_shared,
                    model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    torch_rng_after_build=torch.get_rng_state()))
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):info=frozen.get_model_info(model,(1,1,640,640))
    dump(d/'model_info.json',info)
    tl=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',seed,0,train=True)
    vl=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',seed,0)
    opt=optimizer_for(model);criterion=DenseAnchorDetectionLoss(model.loss_config(b)).to('cuda')
    start_rng=digest(rng_state(tl));run_binding=binding(variant,seed);history=[];parents=[];start=0;best_snapshot=None
    if (d/'latest.pt').exists():
        old=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
        validate_committed(old,run_binding,initial_shared,start_rng)
        best_snapshot=materialize_committed(old,d)
        model.load_state_dict(old['model_state_dict'],strict=True);opt.load_state_dict(old['optimizer_state_dict'])
        restore_rng(old['rng'],tl);history=old['history'];start=old['epoch'];parents=old['parents']
        assert old['train_start_rng_sha256']==start_rng
        if start<80:parents=parents+[dict(epoch=start,sha256=sha(d/'latest.pt'))]
        del old
    if start==80:return summarize_one(variant,seed)
    dump(d/'run_binding.json',dict(**run_binding,admission=admission,python=sys.version,torch=torch.__version__,
                                initial_shared_sha256=initial_shared,train_start_rng_sha256=start_rng,start=now()))
    dump(d/'progress.json',dict(status='TRAINING',epoch=start,updated_at=now(),alpha=float(model.neck.alpha.detach())))
    for epoch in range(start,80):
        tick=time.perf_counter()
        row=train_epoch(model,opt,tl,b,'cuda',None)
        metrics,records=frozen.evaluate(model,vl,criterion,b,'cuda')
        alpha=float(model.neck.alpha.detach());assert math.isfinite(alpha) and alpha>=0
        row.update(epoch=epoch+1,valid=metrics,alpha=alpha,wall_seconds=time.perf_counter()-tick)
        history.append(row)
        state=make_epoch_commit(run_binding=run_binding,history=history,parents=parents,
            initial_shared=initial_shared,start_rng=start_rng,model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),rng=rng_state(tl),records=records,
            best_snapshot=best_snapshot,model_info=info)
        # The only commit point. Any interruption before replace leaves the old
        # epoch authoritative; any interruption afterward is repaired from here.
        atomic_save(d/'latest.pt',state)
        best_snapshot=materialize_committed(state,d)
        del state
        print(f'{variant} seed={seed} epoch={epoch+1}/80 alpha={alpha:.6f} mAP={metrics["mAP50_95"]:.6f} seconds={row["wall_seconds"]:.1f}',flush=True)
    return summarize_one(variant,seed)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['prepare','smoke','train','summarize'])
    ap.add_argument('--variant',choices=VARIANTS);ap.add_argument('--seed',type=int,choices=SEEDS)
    args=ap.parse_args();torch.set_num_threads(2)
    if args.mode=='prepare':prepare()
    elif args.mode=='smoke':smoke()
    else:
        if args.variant is None or args.seed is None:ap.error('--variant and --seed required')
        print(json.dumps(train(args.variant,args.seed) if args.mode=='train' else summarize_one(args.variant,args.seed)))


if __name__=='__main__':main()
