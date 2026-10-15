"""Read-only evidence import plus fail-closed detector deployment dispatch.

Legacy frozen experiment scripts are not modified. Future downstream commands
must go through this entry point; direct legacy invocations are not intercepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from hwnas_fpga.deploy.detection_gate import relative_accuracy_gate, action_decision, guarded_execute


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_policy(cfg):
    exact={'task':'detection','evaluation_split':'valid','test_evaluated':False,
        'reference_network':'D','reference_precision':'same_as_candidate','aggregation':'mean_over_seeds_42_43_44',
        'require_both_metrics':True,'checkpoint_policy':'ALL_BEST_PT','training_budget':80}
    for key,value in exact.items():
        if cfg.get(key)!=value:raise ValueError(f'Unsupported/different gate semantics: {key}')
    if cfg.get('metrics')!=['mAP50_95','Recall']:raise ValueError('mAP and Recall AND gate required')
    if cfg.get('absolute_application_threshold') is not None:raise ValueError('Absolute gate needs a separately frozen protocol')
    if cfg['search_control']['max_new_architecture_evaluations']!=0:raise ValueError('No new search authorized by this gate version')


def load_evidence(cfg,root=ROOT):
    validate_policy(cfg)
    paths={k:root/v for k,v in cfg['source_paths'].items()}
    digest={k:sha(p) for k,p in paths.items()}
    ck=read(paths['checkpoint_manifest']);ptq=read(paths['ptq_manifest'])
    cal=read(paths['calibration_manifest']);qs=yaml.safe_load(paths['quant_config'].read_text())
    baseline=yaml.safe_load(paths['baseline'].read_text())
    freeze=read(paths['baseline_manifest'])
    if freeze['config_sha256']!=digest['baseline']:raise ValueError('baseline freeze config mismatch')
    frozen_sources={}
    for name,expected_sha in freeze['source_sha256'].items():
        actual=sha(root/Path(name.replace('\\','/')))
        if actual!=expected_sha:raise ValueError(f'frozen source changed: {name}')
        frozen_sources[name]=actual
    split_sha=freeze['datasets']['valid']['sha256']
    if ck.get('status')!='READY_FOR_PTQ':raise ValueError('checkpoint audit status mismatch')
    if ck['frozen_checkpoint_policy']!='ALL_BEST_PT':raise ValueError('checkpoint policy mismatch')
    if ptq.get('checkpoint_policy')!='ALL_BEST_PT':raise ValueError('PTQ checkpoint policy mismatch')
    if ck['common_provenance']['search_space_sha256']!=digest['search_space']:raise ValueError('search space hash mismatch')
    if ck['common_provenance']['phase2f_baseline_protocol_hash']!=digest['baseline']:raise ValueError('baseline hash mismatch')
    if ptq['calibration_manifest_sha256']!=digest['calibration_manifest']:raise ValueError('calibration hash mismatch')
    if ptq['quant_config_sha256']!=digest['quant_config']:raise ValueError('quantization config mismatch')
    if ptq['numeric_contract_hash']!=qs['official_policy']['numeric_contract_hash']:raise ValueError('numeric contract mismatch')
    if cal.get('split')!='train' or cal.get('test_evaluated') is not False:raise ValueError('invalid calibration split')
    if ptq.get('test_evaluated') is not False:raise ValueError('test is sealed')
    networks={r['label']:r['architecture_id'] for r in read(paths['candidate_set'])['architectures']}
    if set(networks)!={'B1','C','D'}:raise ValueError('frozen candidate set mismatch')
    bound={}
    for r in ck['runs']:
        key=(r['network'],int(r['seed']))
        if key in bound or r['verification_status']!='VERIFIED_ORIGINAL':raise ValueError('duplicate/unverified checkpoint')
        if r['latest_pt'].get('epoch')!=80 or r['latest_pt'].get('history_len')!=80:
            raise ValueError('completed 80-epoch history/checkpoint missing')
        bound[key]=r
    raw=ptq['runs'];expected={(n,s) for n in networks for s in (42,43,44)}
    if len(raw)!=9 or {(r['network'],int(r['seed'])) for r in raw}!=expected:raise ValueError('missing/duplicate PTQ runs')
    evaluator_hash=json_hash(dict(baseline_sha256=digest['baseline'],protocol=baseline['evaluation_protocol'],frozen_sources=frozen_sources))
    rows=[]
    for r in raw:
        key=(r['network'],int(r['seed']))
        if r['architecture_id']!=networks[r['network']] or r['architecture_id']!=bound[key]['architecture_id']:raise ValueError('architecture mismatch')
        if r['checkpoint_sha256']!=bound[key]['best_pt']['sha256']:raise ValueError('checkpoint hash mismatch')
        if r.get('checkpoint_policy')!='ALL_BEST_PT':raise ValueError('run checkpoint policy mismatch')
        if r['status']!='MEASURED' or r.get('test_evaluated') is not False or r['split']!='valid':raise ValueError('non-valid or unmeasured PTQ run')
        for precision,prefix in [('FP32','fp32'),('W8A8','int8')]:
            metrics={m:float(r[f'{prefix}_{m}']) for m in ['mAP50','mAP50_95','Precision','Recall']}
            if any(not math.isfinite(v) or not 0<=v<=1 for v in metrics.values()):raise ValueError('nonfinite/out-of-range metric')
            rows.append(dict(network=key[0],seed=key[1],architecture_id=r['architecture_id'],precision=precision,
                checkpoint_sha256=r['checkpoint_sha256'],checkpoint_path=r['checkpoint_path'],
                evaluator_sha256=evaluator_hash,calibration_sha256=digest['calibration_manifest'],
                split_manifest_sha256=split_sha,
                checkpoint_policy='ALL_BEST_PT',training_budget=80,split='valid',test_evaluated=False,**metrics))
    return rows,paths,digest,networks


def journal(path,event):
    payload=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def evidence_layers(paths,artifact_aliases=None,root=ROOT):
    hw=read(paths['hardware_summary']);board=read(paths['board_manifest']);cases=read(paths['board_results'])
    verified=[];missing=[];mismatched=[];resolved={};used_aliases={}
    for name,expected in board['files'].items():
        path=paths['board_manifest'].parent/name
        if not path.is_file() and name in (artifact_aliases or {}):
            path=root/artifact_aliases[name]
            used_aliases[name]=str(path)
        if not path.is_file():missing.append(name)
        elif sha(path)!=expected:mismatched.append(name)
        else:
            verified.append(name)
            resolved[name]=dict(path=str(path),sha256=expected)
    return dict(hardware_cost=dict(total=hw['total_candidates'],covered=hw['hardware_measured'],
        architectures=hw['architectures'],scope=hw['measurement_scope'],
        board_measured_latency=False,full_detector_route_proven=False),
        board_witness=dict(scope=board['scope'],full_detector=board['full_detector'],
                          verified_file_count=len(verified),missing_files=missing,mismatched_files=mismatched,
                          resolved_files=resolved,artifact_aliases_used=used_aliases,
                          artifact_verification='COMPLETE' if not missing and not mismatched else 'INCOMPLETE_LOCAL_BUNDLE',cases=cases),
        full_detector=dict(trained_parameter_package=False,trained_integer_parity=False,
                          full_detector_hls=False,full_detector_route=False),
        readiness_basis='No B1/C/D trained-checkpoint-bound package/parity/route evidence admitted in these sources',
        reuse_prototype='h2_graph.cpp/pynq_graph_runner.py exist; graph_attempt02 is synthetic H2 graph_A HLS, not trained B1/C/D validation')


def record_metrics(out,rows,paths,digests):
    from hwnas_fpga.metrics_ledger import build_metric_record,append_metric_record,require_measured_metrics,METRIC_SCHEMA
    records=[]
    split_sha=read(paths['baseline_manifest'])['datasets']['valid']['sha256']
    complexity={r['label']:r for r in read(paths['candidate_set'])['architectures']}
    numeric=read(paths['ptq_manifest'])['numeric_contract']
    names={'mAP50':'map50','mAP50_95':'map50_95','Precision':'precision','Recall':'recall'}
    for r in rows:
        source=dict(source=str(paths['ptq_manifest']),source_sha256=digests['ptq_manifest'],
                    split='valid',precision=r['precision'],checkpoint_sha256=r['checkpoint_sha256'],
                    evaluation_protocol_hash=r['evaluator_sha256'],split_manifest_sha256=split_sha,
                    numeric_contract=numeric,calibration_manifest_sha256=digests['calibration_manifest'])
        measured={names[k]:dict(value=r[k],provenance=source) for k in names}
        for precision,prefix in [('FP32','fp32'),('W8A8','int8')]:
            paired=next(x for x in rows if x['network']==r['network'] and x['seed']==r['seed'] and x['precision']==precision)
            for k in names:measured[f'{prefix}_{names[k]}']=dict(value=paired[k],provenance=dict(source,precision=precision))
        for k in names:
            delta=measured['fp32_'+names[k]]['value']-measured['int8_'+names[k]]['value']
            measured['int8_'+names[k]+'_drop']=dict(value=delta,provenance=dict(source,definition='fp32_minus_int8'))
        info=complexity[r['network']]
        comp_source=dict(source=str(paths['candidate_set']),source_sha256=digests['candidate_set'],
                         input_shape=[1,1,640,640],convention='FLOPs=2*MACs estimate; not hardware latency')
        measured.update(params=dict(value=int(info['Params']),provenance=comp_source),
            macs=dict(value=int(info['MACs']),provenance=comp_source),
            flops=dict(value=2*int(info['MACs']),provenance=comp_source))
        record=build_metric_record(task_type='detection',dataset='Cylinder2',network=r['network'],seed=r['seed'],
            stage='frozen_accuracy_import',identity=dict(run_id=f"{r['network']}_seed{r['seed']}_{r['precision']}",
            precision=r['precision'],checkpoint_sha256=r['checkpoint_sha256'],protocol_sha256=digests['baseline'],
            evaluator_sha256=r['evaluator_sha256'],calibration_sha256=r['calibration_sha256'],
            source_sha256=digests['ptq_manifest'],test_evaluated=False,clean_reference_available=False),measured=measured)
        require_measured_metrics(record,['map50','map50_95','precision','recall','params','macs','flops'])
        append_metric_record(out/'METRICS.jsonl',record);records.append(record)
    write(out/'METRIC_SCHEMA.json',METRIC_SCHEMA)
    coverage=[]
    for metric in METRIC_SCHEMA:
        coverage.append(dict(metric=metric,**{s:sum(r['metrics'][metric]['status']==s for r in records)
            for s in ['MEASURED','PENDING','NOT_APPLICABLE']}))
    with (out/'METRIC_COVERAGE.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    return records


def _run_audit(config,out):
    cfg=yaml.safe_load(config.read_text(encoding='utf-8'))
    rows,paths,digests,networks=load_evidence(cfg)
    layers=evidence_layers(paths,cfg.get('board_artifact_aliases'))
    before={str(p):digests[k] for k,p in paths.items()}
    before.update({r['path']:r['sha256'] for r in layers['board_witness']['resolved_files'].values()})
    gates={};table=[]
    for precision in ['FP32','W8A8']:
        ref=[r for r in rows if r['network']=='D' and r['precision']==precision]
        for network in cfg['execution_priority']:
            c=[r for r in rows if r['network']==network and r['precision']==precision]
            decision=relative_accuracy_gate(c,ref,precision=precision)
            gates[(network,precision)]=decision
            table.append(dict(network=network,**decision))
    actions=[]
    for network in cfg['execution_priority']:
        for action in ['search','promotion','ptq','parameter_export','integer_parity','hls','route','board']:
            actions.append(dict(network=network,precision='W8A8',**action_decision(action,gates[(network,'FP32')],gates[(network,'W8A8')],
                hardware=layers['full_detector'],new_evaluation_budget=cfg['search_control']['max_new_architecture_evaluations'])))
    write(out/'ACCURACY_GATES.json',table);write(out/'ACTION_QUEUE.json',actions)
    write(out/'HARDWARE_EVIDENCE.json',layers)
    mixed=[]
    source_bound={(r['network'],r['seed']):r for r in rows if r['precision']=='W8A8'}
    q=read(paths['frozen_q'])['Q']
    with paths['mixed_results'].open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            profile={(8,16):'HEAD_W8A16',(16,16):'HEAD_W16A16'}.get((int(r['head_weight_bits']),int(r['head_activation_bits'])))
            if profile and profile in q.get(r['network'],[]):
                ref=source_bound.get((r['network'],int(r['seed'])))
                valid=bool(ref and r.get('status')=='MEASURED' and r.get('split')=='valid'
                    and str(r.get('test_evaluated')).lower()=='false' and r.get('checkpoint_sha256')==ref['checkpoint_sha256']
                    and r.get('architecture_id')==ref['architecture_id'] and r.get('calibration_manifest_sha256')==digests['calibration_manifest']
                    and r.get('numeric_contract_sha256')==read(paths['ptq_manifest'])['numeric_contract_hash'])
                if not valid:raise ValueError('Mixed evidence identity/split/contract mismatch')
                mixed.append(dict(network=r['network'],seed=int(r['seed']),precision=profile,
                    gate='PENDING_SAME_PRECISION_CONTROL_OR_BACKEND',mAP50_95=float(r['mixed_mAP50_95']),
                    Recall=float(r['mixed_Recall']),source=str(paths['mixed_results']),test_evaluated=False))
    write(out/'FROZEN_MIXED_SCOPE.json',mixed)
    records=record_metrics(out,rows,paths,digests)
    flat=[]
    for d in table:
        row=dict(network=d['network'],precision=d['precision'],status=d['status'])
        for k,v in d['metrics'].items():
            row[k]=v['candidate_mean'];row['D_'+k]=v['reference_mean'];row['delta_'+k]=v['mean_delta']
        flat.append(row)
    with (out/'ACCURACY_GATE_SUMMARY.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for r in rows:
        if r['precision']=='W8A8' and r['seed']==cfg['initial_board_seed']:
            write(out/f"{r['network']}_DEPLOYMENT_BINDING.json",dict(**r,
                numeric_contract=read(paths['ptq_manifest'])['numeric_contract'],
                numeric_contract_hash=read(paths['ptq_manifest'])['numeric_contract_hash'],
                frozen_config_sha256=digests['baseline'],search_space_sha256=digests['search_space'],
                board_claim_allowed=False,real_parameter_export='PENDING',precision_gate=gates[(r['network'],'W8A8')]['status']))
    status=dict(status='ACCURACY_PASS_HARDWARE_PENDING' if all(g['passed'] for g in gates.values()) else 'CANDIDATE_STOP_APPLIED',
        relative_reference='D_same_precision_mean_mAP50_95_AND_Recall',
        application_accuracy_sufficient='NOT_ESTABLISHED_RELATIVE_ONLY',
        existing_results_only=True,new_training_started=False,new_synthesis_started=False,test_evaluated=False,
        metrics_records=len(records),hardware_scope=layers['hardware_cost']['scope'],
        board_ready=False,execution_priority=cfg['execution_priority'])
    report=['# Detection accuracy stop gate and board-readiness audit',
        '\n用户已选择：D作为相对精度底线。FP32对FP32，W8A8对W8A8；三seed平均mAP50:95与Recall均不得下降。',
        'D相等时为CONTROL_PASS。单seed差异原样保留，不要求每个seed均胜出。此底线不证明实际应用精度够用。',
        '\n| Network | Precision | mAP50:95 | Recall | Gate |','|---|---|---:|---:|---|']
    for r in flat:report.append(f"| {r['network']} | {r['precision']} | {r['mAP50_95']:.6f} | {r['Recall']:.6f} | {r['status']} |")
    report += ['\n## Stop and recording semantics',
        '- 任一主指标低于D：STOP_ACCURACY_BELOW_D，候选后续量化/硬件任务不得启动。缺证据：BLOCKED_EVIDENCE，不冒充精度失败。',
        '- 新架构评价预算=0；surrogate停止。已有冻结80epoch协议不改，不根据40epoch弱代理临时截断正式运行。',
        '- METRICS.jsonl明确记录MEASURED/PENDING/NOT_APPLICABLE与来源。历史没有记录的指标不能补造。',
        '- 检测主指标不是分类Top1；Recall固定score0.5、IoU0.5、micro；AP IoU0.50:0.05:0.95。',
        '- 保存gate事件成功后才派发命令；指标写盘失败则停止派发。旧入口未改，后续任务必须经本wrapper，不声称能拦截手工绕过。',
        '\n## Current hardware evidence',
        f"- 当前成本表覆盖{layers['hardware_cost']['covered']}/{layers['hardware_cost']['total']}，范围OP_LEVEL_HLS，不能写成整网板测延迟。",
        '- PW/FPN两个DMA witness报告零误差；不是B1/C/D整网。其单事务时间不进入网络latency字段。',
        f"- 本地板端包校验：{layers['board_witness']['artifact_verification']}；缺失{layers['board_witness']['missing_files']}；不匹配{layers['board_witness']['mismatched_files']}。缺件只阻断对应硬件证据，不否认已绑定的软件精度。",
        f"- 显式路径别名恢复{len(layers['board_witness']['artifact_aliases_used'])}个文件，逐项对原始manifest校验SHA；详见HARDWARE_EVIDENCE.json。不改原始manifest或数据。",
        '- h2_graph.cpp/pynq_graph_runner.py已有串行DDR整图功能核心；graph_attempt02是合成graph_A的HLS证据，不是B1/C/D训练权重整网验证。',
        '- 下一缺口：取回并验真实best.pt → 保存的PTQ profile转H2参数图 → 逐层整数parity → 复用整图核心完成候选绑定的HLS/route/板测。',
        '- 不能直接复用合成PackageBuilder的硬编码scale。生产者output_exp与消费者input_exp可能不同，必须显式复现重定标及饱和；否则会改变已测PTQ语义。',
        '- 固定工程顺序D→C→B1，初始seed42，uniformINT8。D最简单作为工程control，非最终精度选模。',
        '- 已有混精Q(x)保留；不以不同profile之间的比较冒充同精度D控制，混精后端需独立验证。',
        '\n## Reproduce',
        '`python tools/sonar_spec/deployment_pipeline.py audit`',
        '`python tools/sonar_spec/deployment_pipeline.py check --network D --action board` (目前应拒绝)',
        '`python tools/sonar_spec/deployment_pipeline.py check --network D --action parameter_export` (精度允许准备参数包)',
        'dispatch目前没有注册可用的真实权重导出/硬件handler，明确返回BLOCKED_ACTION_HANDLER；不会接受任意命令冒充经过绑定的部署任务。',
        '\nNo images read. No test evaluation. No training or FPGA programming was launched by this audit.']
    (out/'DEPLOYMENT_READINESS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    if any(sha(Path(p))!=h for p,h in before.items()):raise RuntimeError('Input changed during audit')
    write(out/'MANIFEST.json',dict(config_sha256=sha(config),input_sha256=before,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'src/hwnas_fpga/deploy/detection_gate.py':sha(ROOT/'src/hwnas_fpga/deploy/detection_gate.py'),
            'src/hwnas_fpga/metrics_ledger.py':sha(ROOT/'src/hwnas_fpga/metrics_ledger.py')},
        test_evaluated=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file()
            and p.name not in ['MANIFEST.json','ADMISSION.json','dispatch.jsonl']
            and not p.name.endswith(('.lock','.tmp'))}))
    return status


def run_audit(config,out):
    """Serialize audits; the atomic final admission commits a complete snapshot.

    Readers must require a successful ADMISSION and its matching manifest hash.
    A terminated process leaves IN_PROGRESS, never a stale successful admission.
    """
    from hwnas_fpga.metrics_ledger import _ledger_lock
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with _ledger_lock(out/'audit.lock',timeout=120):
        write(out/'ADMISSION.json',dict(status='IN_PROGRESS',allowed=False,test_evaluated=False))
        write(out/'ACTION_QUEUE.json',[])
        try:
            status=_run_audit(Path(config),out)
            status['manifest_sha256']=sha(out/'MANIFEST.json')
            write(out/'ADMISSION.json',status)
            return status
        except Exception as exc:
            status=dict(status='BLOCKED_EVIDENCE',allowed=False,
                reason=f'{type(exc).__name__}: {exc}',test_evaluated=False)
            # Write the authoritative failure first, even if subsequent I/O fails.
            write(out/'ADMISSION.json',status)
            write(out/'ACTION_QUEUE.json',[])
            write(out/'MANIFEST.json',status)
            return status


def real_dispatch_command(root, network, action, seed):
    """Fixed command construction, never an arbitrary argv or shell bridge."""
    actions = {'parameter_export': 'export-evaluate', 'integer_parity': 'integer_parity',
               'hls': 'hls', 'route': 'route', 'board': 'board'}
    if action not in actions or network not in ('D', 'C', 'B1') or seed not in (42, 43, 44):
        raise ValueError('Unregistered real-deployment command')
    return [sys.executable, str(ROOT/'tools/sonar_deploy/run.py'), actions[action],
            '--root', str(Path(root).resolve()), '--network', network, '--seed', str(seed)]


def real_pipeline(mode, root, out, network=None, action=None, seed=42, runner=None):
    """New checkpoint-bound path; no dependency on residual search admissions."""
    sys.path.insert(0, str(ROOT))
    from tools.sonar_deploy import admission
    from hwnas_fpga.metrics_ledger import _ledger_lock
    root = Path(root).resolve(); out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if mode == 'audit':
        value = admission.audit(root)
        admission.atomic_json(out/'REAL_DEPLOYMENT_ADMISSION.json', value)
        return value, 2 if value['status'] == 'BLOCKED_EVIDENCE' else 0
    if not network or not action:
        raise ValueError('--network and --action are required')
    decision = admission.action_admission(root, network, action, seed)
    if mode == 'check':
        return decision, 0 if decision['allowed'] else 3
    if mode != 'dispatch':
        raise ValueError('Unknown pipeline mode')
    def durable_event(event):
        with _ledger_lock(out/'dispatch-journal.lock', timeout=30):
            journal(out/'dispatch.jsonl', event)
    # Serialize the same network/seed while permitting different evaluations in
    # parallel.  Check eligibility again after obtaining the execution lock.
    with _ledger_lock(root/f'{network}_seed{seed}.dispatch.lock', timeout=1):
        decision = admission.action_admission(root, network, action, seed)
        if not decision['allowed']:
            durable_event(dict(event='ACTION_GATE', decision=decision))
            return decision, 3
        command = real_dispatch_command(root, network, action, seed)
        binding_sha = admission.sha256(root/'DEPLOYMENT_BINDING.json')
        decision['deployment_binding_sha256'] = binding_sha
        completed = guarded_execute(command, decision, journal=durable_event, runner=runner)
        if completed.returncode:
            return dict(decision, allowed=False, status='ACTION_FAILED', returncode=completed.returncode), completed.returncode
        try:
            if admission.sha256(root/'DEPLOYMENT_BINDING.json') != binding_sha:
                raise ValueError('Deployment binding changed during action')
            binding = admission.load_binding(root)
            expected = next(r for r in binding['runs'] if (r['network'], r['seed']) == (network, seed))
            result = admission.load_result(root, expected)
            if action != 'parameter_export':
                admission.stage_evidence(root, network, action, result)
            final = dict(decision, status='COMPLETE_VERIFIED')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            final = dict(decision, allowed=False, status='BLOCKED_RESULT_EVIDENCE', reasons=[str(exc)])
        durable_event(dict(event='ACTION_VERIFIED', decision=final))
        return final, 0 if final['allowed'] else 4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['audit','check','dispatch'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/sonar_deployment_gate_v1.yaml')
    p.add_argument('--output',type=Path)
    p.add_argument('--real-deployment-root',type=Path,
        help='Use checkpoint-bound real INT8 evidence; independent of legacy search admissions')
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--network',choices=['B1','C','D']);p.add_argument('--action',choices=['search','promotion','ptq','parameter_export','integer_parity','hls','route','board'])
    args=p.parse_args()
    if args.real_deployment_root:
        if args.mode != 'audit' and (not args.network or not args.action):
            p.error('--network and --action required')
        out = args.output or ROOT/'reports/sonar_real_int8_deployment_v1'
        result, code = real_pipeline(args.mode, args.real_deployment_root, out,
            args.network, args.action, args.seed)
        print(json.dumps(result,indent=2));return code
    out=(args.output or ROOT/'reports/sonar_deployment_admission_v1').resolve()
    result=run_audit(args.config.resolve(),out)
    if args.mode=='audit':print(json.dumps(result,indent=2));return 2 if result['status']=='BLOCKED_EVIDENCE' else 0
    if not args.network or not args.action:p.error('--network and --action required')
    if result['status']=='BLOCKED_EVIDENCE':print(json.dumps(result));return 2
    decision=next(r for r in read(out/'ACTION_QUEUE.json') if r['network']==args.network and r['action']==args.action)
    print(json.dumps(decision,indent=2))
    if args.mode=='check':return 0 if decision['allowed'] else 3
    # No arbitrary argv bridge: each future handler must construct the exact
    # action from the validated network/checkpoint/precision binding.
    if decision['allowed']:
        decision=dict(decision,allowed=False,status='BLOCKED_ACTION_HANDLER',reasons=['No trained-detector action handler registered yet'])
    journal(out/'dispatch.jsonl',dict(event='ACTION_GATE',decision=decision))
    print(json.dumps(decision,indent=2));return 3


if __name__=='__main__':raise SystemExit(main())
