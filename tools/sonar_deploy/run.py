"""Real fixed-model deployment evidence; never trains or accesses test."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools'),
                str(ROOT/'tools/sonar_analysis'), str(ROOT/'tools/sonar_spec'), str(ROOT/'scripts')]
RUN = ROOT/'results/sonar_real_int8_deployment_v1/cpu_decode_v1_route02'
REPORT = ROOT/'reports/sonar_real_int8_deployment_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def utc():return datetime.now(timezone.utc).isoformat()


def modules():
    import torch
    import baseline as frozen
    from phase5a_quantization.common import baseline,build_network
    from phase5a_quantization.run import load_weights
    return torch,frozen,baseline,build_network,load_weights


def prepare(root):
    torch,frozen,baseline,_,_=modules()
    config=baseline()
    old_root=ROOT/'reports/phase5a_quantization'
    ptq=read(old_root/'PTQ_MANIFEST.json')
    calibration=read(old_root/'CALIBRATION_MANIFEST.json')
    checkpoints=read(old_root/'CHECKPOINT_SELECTION_MANIFEST.json')
    if checkpoints['status']!='READY_FOR_PTQ' or checkpoints['frozen_checkpoint_policy']!='ALL_BEST_PT':
        raise ValueError('Original checkpoint admission failed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_ENVIRONMENT_NOT_READY')
    assert calibration['split']=='train' and calibration['test_evaluated'] is False
    data=frozen.resolve(config['dataset']['data_dir'])
    for entry in calibration['entries']:
        for key in ['image','annotation']:
            assert Path(entry[key]).parts[0]=='train'
            assert sha(data/entry[key])==entry[key+'_sha256']
    loader=frozen.make_loader(config,data,'valid',42,0)
    valid=frozen.bind_dataset(loader)
    freeze=read(ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json')
    frozen_valid=freeze['datasets']['valid']
    # The original manifest was serialized on Windows. Compare every ordered
    # path and content hash, allowing only path-separator normalization.
    normalized=[dict(e,image=e['image'].replace('\\','/'),annotation=e['annotation'].replace('\\','/'))
                for e in frozen_valid['entries']]
    assert valid['images']==99 and valid['entries']==normalized
    valid['frozen_windows_manifest_sha256']=frozen_valid['sha256']
    valid['cross_platform_path_normalization_only']=True
    pinned={}
    for name,digest in freeze['source_sha256'].items():
        path=ROOT/name.replace('\\','/')
        assert sha(path)==digest, name
        pinned[name]=digest
    evaluator_sha=hashlib.sha256(json.dumps(dict(baseline_sha256=sha(ROOT/'configs/baseline_sonar_frozen.yaml'),
        protocol=config['evaluation_protocol'],frozen_sources=pinned),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    sources={name:{'path':str(path),'sha256':sha(path)} for name,path in {
        'checkpoint_manifest':old_root/'CHECKPOINT_SELECTION_MANIFEST.json',
        'ptq_manifest':old_root/'PTQ_MANIFEST.json',
        'calibration_manifest':old_root/'CALIBRATION_MANIFEST.json',
        'baseline':ROOT/'configs/baseline_sonar_frozen.yaml',
        'baseline_freeze':ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json',
        'search_space':ROOT/'configs/search_space_sonar_v0_1.yaml',
        'candidate_set':ROOT/'reports/phase2f_multiseed/network_candidate_set.json',
        'numeric_contract':ROOT/'reports/fpga_h2_contract/numeric_contract_v1.json',
    }.items()}
    expected={(r['network'],int(r['seed'])):r for r in checkpoints['runs']}
    runs=[]
    for r in ptq['runs']:
        n,s=r['network'],int(r['seed']);original=expected[(n,s)]
        assert original['verification_status']=='VERIFIED_ORIGINAL'
        assert original['best_pt']['sha256']==r['checkpoint_sha256']
        path=Path(r['checkpoint_path']);assert sha(path)==r['checkpoint_sha256']
        profile=ROOT/f'artifacts/phase5a_quantization/{n}/seed_{s}/h2_ptq_profile.json'
        assert read(profile)['numeric_contract_hash']==ptq['numeric_contract_hash']
        runs.append(dict(network=n,seed=s,architecture_id=r['architecture_id'],checkpoint_path=str(path),
            checkpoint_sha256=r['checkpoint_sha256'],profile_path=str(profile),profile_sha256=sha(profile),
            calibration_sha256=sources['calibration_manifest']['sha256'],numeric_contract_hash=ptq['numeric_contract_hash'],
            split_manifest_sha256=frozen_valid['sha256'],evaluator_sha256=evaluator_sha,
            historical_reference_metrics={'AP50':r['int8_mAP50'],'mAP50_95':r['int8_mAP50_95'],
                'Precision':r['int8_Precision'],'Recall':r['int8_Recall']}))
    binding=dict(schema_version='real-int8-deployment-v1',status='FROZEN',split='valid',test_evaluated=False,
        source_files=sources,runs=runs,protocol=dict(precision='W8A8',seeds=[42,43,44],
            board_seed=42,order=['D','C','B1'],historical_ap_tolerance=1e-6,calibration_images=64,
            recalibration=False,train=False,board_train_prefix=8,valid_images=99,
            board_warmup=1,board_repetitions=10,criterion='D same integer implementation three-seed mean mAP AND Recall'))
    binding['protocol']['parameter_preparation_device']='cuda'
    binding['protocol']['parameter_preparation_reason']='Replay original Phase5A CUDA BN folding before INT8/INT32 rounding; CPU folding can change half-tie decisions'
    binding['protocol']['historical_postprocessing_device']='cuda'
    binding['protocol']['integer_postprocessing_device']='cpu'
    binding['protocol']['integer_postprocessing_reason']='Match frozen CPU decode/NMS used after FPGA; same-head CPU/CUDA top-k and score numerics can differ'
    path=root/'DEPLOYMENT_BINDING.json'
    if path.exists() and read(path)!=binding:raise ValueError('Frozen deployment binding changed')
    write(path,binding)
    write(root/'VALID_MANIFEST.json',valid)
    write(REPORT/'ENVIRONMENT.json',dict(time=utc(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),
        source_root=str(ROOT),test_evaluated=False))
    write(REPORT/'DEPLOYMENT_BINDING.json',binding)
    return dict(status='PREPARED',checkpoints=9,calibration_images_verified=64,valid_images=99,
        binding_sha256=sha(path),test_evaluated=False)


def evaluate(model,device,postprocess_device=None):
    torch,frozen,baseline,_,_=modules()
    from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
    config=baseline();loader=frozen.make_loader(config,frozen.resolve(config['dataset']['data_dir']),'valid',42,0)
    criterion=DenseAnchorDetectionLoss(model.loss_config(config)).to(postprocess_device or device)
    model.to(device).eval()
    if postprocess_device=='cpu':
        class CPUPostprocess(torch.nn.Module):
            def __init__(self,inner):
                super().__init__();self.inner=inner
            def forward(self,images):
                return {key:tuple(x.cpu() for x in tensors) for key,tensors in self.inner(images).items()}
        model=CPUPostprocess(model)
    metrics,records=frozen.evaluate(model,loader,criterion,config,device)
    assert metrics['num_images']==99
    return metrics,records


def export_evaluate(root,network,seed,device):
    from tools.sonar_deploy.ptq_export import restore_archived_ptq,export_ptq_package,IntegerGraphModel
    from tools.sonar_deploy.admission import record_result_metrics,load_result,load_binding
    torch,frozen,baseline,build_network,load_weights=modules()
    if not device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('CUDA_ENVIRONMENT_NOT_READY: original parameter preparation requires CUDA folding')
    frozen.seed_all(seed)
    torch.set_num_threads(2)
    binding=read(root/'DEPLOYMENT_BINDING.json')
    load_binding(root)
    row=next(r for r in binding['runs'] if r['network']==network and r['seed']==seed)
    for item in binding['source_files'].values():assert sha(item['path'])==item['sha256']
    assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
    assert sha(row['profile_path'])==row['profile_sha256']
    out=root/network/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        previous=read(out/'result.json')
        if previous['status']=='COMPLETE':
            load_result(root,row)
            return dict(status='ALREADY_COMPLETE',network=network,seed=seed)
    begin=time.perf_counter();write(out/'progress.json',dict(stage='HISTORICAL_REPRODUCTION',time=utc(),test_evaluated=False))
    model,aid=build_network(network);load_weights(model,Path(row['checkpoint_path']))
    assert aid==row['architecture_id']
    profile=read(row['profile_path'])
    folded,quantized=restore_archived_ptq(model.to(device),profile)
    historical,records=evaluate(quantized,device)
    write(out/'historical_metrics.json',historical);write(out/'historical_predictions.json',records)
    failures=[]
    for key,value in row['historical_reference_metrics'].items():
        tolerance=1e-6 if key in ('AP50','mAP50_95') else 0.
        if abs(historical[key]-value)>tolerance:failures.append(dict(metric=key,actual=historical[key],expected=value))
    write(out/'historical_reproduction.json',dict(status='FAIL' if failures else 'PASS',differences=failures))
    if failures:raise RuntimeError('BASELINE_REPRODUCTION_MISMATCH: '+str(failures))
    quantized.cpu();folded.cpu();torch.cuda.empty_cache()
    provenance={k:row[k] for k in ['network','seed','architecture_id','checkpoint_sha256','profile_sha256',
        'calibration_sha256','numeric_contract_hash','split_manifest_sha256','evaluator_sha256']}
    provenance.update(test_evaluated=False,precision='W8A8',parameter_preparation_device='cuda',
        parameter_preparation_torch_version=torch.__version__,parameter_preparation_cuda_version=torch.version.cuda)
    def obtain_package(stem):
        target=out/stem
        # Preserve incomplete attempts. A complete matching package is reusable;
        # a partial one gets a new sibling, never an in-place destructive repair.
        attempt=1
        while target.exists() and not (target/'export_manifest.json').exists():
            attempt+=1;target=out/(stem+'_attempt_'+str(attempt))
        if target.exists():
            existing=read(target/'export_manifest.json')
            if existing['provenance']!=provenance:raise ValueError('Existing package identity changed')
            if sha(target/'graph.json')!=existing['graph_sha256']:raise ValueError('Existing graph corrupt')
            IntegerGraphModel(target,model)  # verifies every binary digest
            return target,read(target/'graph.json')
        return target,export_ptq_package(model,profile,target,provenance)
    package,graph=obtain_package('package')
    repeat,repeated=obtain_package('determinism_check')
    model.cpu()
    for name in ['graph.json','weights.bin','biases.bin','layers.bin']:
        assert sha(package/name)==sha(repeat/name), 'Nondeterministic export: '+name
    package_manifest=dict(**provenance,files={name:sha(package/name) for name in
        ['graph.json','weights.bin','biases.bin','layers.bin']},deterministic_export='PASS')
    write(package/'manifest.json',package_manifest)
    write(out/'progress.json',dict(stage='INTEGER_VALIDATION',time=utc(),test_evaluated=False))
    integer=IntegerGraphModel(package,model)
    metrics,predictions=evaluate(integer,device,postprocess_device='cpu')
    write(out/'integer_metrics.json',metrics);write(out/'integer_predictions.json',predictions)
    macs=sum(int(__import__('math').prod(n['weight_shape'])) * int(graph['edges'][n['id']]['shape'][1]) *
             int(graph['edges'][n['id']]['shape'][2]) for n in graph['nodes'] if n['op']=='conv')
    complexity=dict(params=sum(p.numel() for p in model.parameters()),macs=macs,flops=2*macs,
        scope='Conv MACs, batch1 input640; excludes activation/requant/add/resize/decode/NMS',
        deployment_weight_elements=sum(__import__('math').prod(n['weight_shape']) for n in graph['nodes'] if n['op']=='conv'),
        deployment_bias_elements=sum(n['bias_count'] for n in graph['nodes'] if n['op']=='conv'))
    result=dict(**provenance,status='COMPLETE',historical_reproduction='PASS',split='valid',complexity=complexity,
        integer_execution=dict(arithmetic='exact integer with bounded FP64 dot products',arithmetic_device=device,postprocessing_device='cpu'),
        historical_metrics=historical,integer_metrics=metrics,
        package_manifest_path=str(package/'manifest.json'),package_manifest_sha256=sha(package/'manifest.json'),
        binding_sha256=sha(root/'DEPLOYMENT_BINDING.json'),elapsed_seconds=time.perf_counter()-begin,
        execution_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/sonar_deploy').glob('*.py')})
    result['metric_event_ids']=record_result_metrics(root,result)
    write(out/'result.json',result)
    write(out/'progress.json',dict(stage='COMPLETE',time=utc(),test_evaluated=False))
    return {k:result[k] for k in ('network','seed','status','elapsed_seconds','integer_metrics')}


def integer_parity(root,network,device):
    from tools.sonar_deploy import admission
    from tools.sonar_deploy.parity import prepare_train_witnesses,verify_numpy_first_image
    decision=admission.action_admission(root,network,'integer_parity',42)
    if not decision['allowed']:raise RuntimeError(json.dumps(decision))
    binding=admission.load_binding(root)
    row=next(r for r in binding['runs'] if r['network']==network and r['seed']==42)
    result=admission.load_result(root,row)
    out=root/network/'seed_42';trace=out/'parity/train_traces';oracle=out/'parity/numpy'
    if (out/'integer_parity.json').exists():return admission.stage_evidence(root,network,'integer_parity',result)
    package=Path(result['package_manifest_path']).parent
    torch,frozen,_,build_network,load_weights=modules()
    torch.set_num_threads(2);frozen.seed_all(42)
    if not (trace/'TRACE_MANIFEST.json').exists():
        model,aid=build_network(network);load_weights(model,Path(row['checkpoint_path']))
        if aid!=row['architecture_id']:raise ValueError('Architecture identity changed')
        prepare_train_witnesses(package,model,Path(row['profile_path']),
            Path(binding['source_files']['calibration_manifest']['path']),trace,device=device)
    manifest=read(trace/'TRACE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        if sha(trace/name)!=digest:raise ValueError('Train witness changed: '+name)
    parity=read(oracle/'NUMPY_PARITY.json') if (oracle/'NUMPY_PARITY.json').exists() else verify_numpy_first_image(package,trace,oracle)
    if parity['graph_sha256']!=sha(package/'graph.json') or parity['trace_manifest_sha256']!=sha(trace/'TRACE_MANIFEST.json'):
        raise ValueError('Independent oracle binding mismatch')
    if parity['status']!='PASS' or parity['mismatch_count']!=0:raise ValueError('INTEGER_PARITY_FAILED')
    evidence={k:row[k] for k in admission.IDENTITY}
    artifacts={label:{'path':str(path),'sha256':sha(path)} for label,path in {
        'trace_manifest':trace/'TRACE_MANIFEST.json','numpy_parity':oracle/'NUMPY_PARITY.json',
        'integer_source':ROOT/'tools/sonar_deploy/ptq_export.py',
        'parity_source':ROOT/'tools/sonar_deploy/parity.py','canonical_arithmetic':ROOT/'tools/fpga_h2/numeric.py'}.items()}
    evidence.update(stage='integer_parity',status='PASS',scope='full_detector',mismatch_count=0,
        oracle_scope='first fixed train image every graph node vs independent canonical NumPy INT64; fixed8 trace witnesses',
        train_sample_count=8,package_manifest_sha256=result['package_manifest_sha256'],
        deployment_binding_sha256=sha(root/'DEPLOYMENT_BINDING.json'),artifacts=artifacts,
        test_evaluated=False,metric_values={})
    evidence['metric_event_id']=admission.record_stage_metrics(root,evidence)
    write(out/'integer_parity.json',evidence)
    return admission.stage_evidence(root,network,'integer_parity',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','record-complexity','export-evaluate','integer_parity','hls','route','board'])
    parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--network',choices=['D','C','B1'])
    parser.add_argument('--seed',type=int,choices=[42,43,44])
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.root.resolve())
    elif args.action=='record-complexity':
        from tools.sonar_deploy.admission import load_binding,record_complexity_metrics,load_result
        binding=load_binding(args.root)
        for row in binding['runs']:
            r=read(args.root/row['network']/f"seed_{row['seed']}"/'result.json')
            record_complexity_metrics(args.root,r);load_result(args.root,row)
        result=dict(status='COMPLEXITY_RECORDED',runs=9,test_evaluated=False)
    else:
        if args.network is None or args.seed is None:parser.error('--network and --seed required')
        try:
            if args.action=='export-evaluate':result=export_evaluate(args.root.resolve(),args.network,args.seed,args.device)
            else:
                if args.seed!=42:raise ValueError('First hardware qualification is fixed to seed42')
                if args.action=='integer_parity':result=integer_parity(args.root.resolve(),args.network,args.device)
                else:
                    from tools.sonar_deploy.hardware import execute_stage
                    result=execute_stage(args.root.resolve(),args.network,args.action)
        except Exception as exc:
            write(args.root/args.network/f'seed_{args.seed}'/'failure.json',dict(error=str(exc),time=utc(),test_evaluated=False))
            raise
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
