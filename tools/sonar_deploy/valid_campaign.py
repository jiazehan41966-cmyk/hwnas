"""Isolated train8/valid99 board queue; reuse frozen driver and CPU evaluator.

No training, calibration, test access, or automatic retry. Historical performance
evidence stays separate; this queue does not manufacture full board admission.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.sonar_deploy import admission as a, board_campaign as c


def validate_resume_protocol(protocol, snapshot, original_snapshot, performance):
    """Resume collection, never change the frozen experiment or retry a board call."""
    fixed = dict(order=list(c.NETWORKS), seed=42, train_frames=8, valid_frames=99,
                 historical_driver_benchmark=dict(warmup=1, repetitions=10),
                 deployment_binding_sha256=snapshot['deployment_binding_sha256'],
                 test_evaluated=False,
                 performance_reports={n:a.sha256(performance/n/'PERFORMANCE_REPORT.json') for n in c.NETWORKS})
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError('Frozen valid99 resume protocol changed')
    if original_snapshot.get('test_evaluated') is not False or original_snapshot.get('deployment_binding_sha256') != snapshot['deployment_binding_sha256']:
        raise ValueError('Original deployment binding/seal changed')
    keys = ('network', 'architecture_id', 'checkpoint_sha256', 'package_manifest_sha256',
            'graph_sha256', 'build', 'regression')
    for network in c.NETWORKS:
        before = [r for r in original_snapshot['runs'] if r['network'] == network]
        current = [r for r in snapshot['runs'] if r['network'] == network]
        if len(before) != 1 or len(current) != 1 or any(before[0].get(k) != current[0].get(k) for k in keys):
            raise ValueError('Resume candidate identity changed: '+network)


def validate_recovered_build(folder, row, binding_sha):
    """A recovered transaction must retain its original frozen build identity."""
    bound = a.read_json(folder/'manifest.json')
    if not bound.get('real_weights_bound') or bound['graph_sha256'] != row['graph_sha256']:
        raise ValueError('Recovered build graph identity changed')
    for name in ('h2_graph.cpp', 'numeric_v1.h', 'pynq_graph_runner.py', 'graph.json'):
        if bound['files'].get(name) != a.sha256(folder/name):
            raise ValueError('Recovered build source changed: '+name)
    launch = a.read_json(folder/'remote_launch.json')
    if launch != dict(remote=row['build'], network=row['network'],
                      deployment_binding_sha256=binding_sha, test_evaluated=False):
        raise ValueError('Recovered remote execution identity changed')


def upload_or_verify_evidence(tx, row, folder):
    """A disconnected post-upload transaction may reuse only byte-identical files."""
    evidence = folder/'board_evidence'
    files = {p.relative_to(evidence).as_posix():a.sha256(p) for p in evidence.rglob('*') if p.is_file()}
    destination = str(PurePosixPath(row['build']).parent/'board_evidence')
    state = c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
from tools.sonar_deploy import admission as a
e=Path({destination!r});expected={files!r}
if e.exists():
 for name,digest in expected.items():
  if not (e/name).is_file() or a.sha256(e/name)!=digest:raise ValueError('Existing returned evidence differs: '+name)
 print(json.dumps(dict(status='IDENTICAL_EXISTING_EVIDENCE')))
else:print(json.dumps(dict(status='MISSING')))
''')
    if state['status'] == 'MISSING':
        c.upload_evidence(dict(remote_project=tx.REMOTE), row, evidence)


def run(output, performance, resume=False):
    output = output.resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError('Resume requires the original valid99 campaign')
    if a.read_json(performance / 'STATUS.json')['status'] != 'FORMAL_PERFORMANCE_COMPLETE':
        raise ValueError('Completed formal performance campaign required')
    for n in c.NETWORKS:
        if a.read_json(performance/n/'PERFORMANCE_REPORT.json')['status'] != 'PERFORMANCE_PASS':
            raise ValueError('Missing performance gate: '+n)
    tx = c.transport()
    root = tx.REMOTE+'/results/sonar_real_int8_deployment_v1/cpu_decode_v1_route02'
    remote = tx.REMOTE+'/results/sonar_real_int8_deployment_v1/'+output.name
    snapshot = c.snapshot(root, tx.REMOTE)
    protocol = dict(order=list(c.NETWORKS), seed=42, train_frames=8, valid_frames=99,
                    historical_driver_benchmark=dict(warmup=1,repetitions=10),
                    deployment_binding_sha256=snapshot['deployment_binding_sha256'],
                    source_sha256=a.sha256(__file__), test_evaluated=False,
                    performance_reports={n:a.sha256(performance/n/'PERFORMANCE_REPORT.json') for n in c.NETWORKS})
    if resume:
        original_protocol = a.read_json(output/'PROTOCOL.json')
        validate_resume_protocol(original_protocol, snapshot, a.read_json(output/'UPSTREAM_SNAPSHOT.json'), performance)
        claim = dict(original_protocol_sha256=a.sha256(output/'PROTOCOL.json'),
                     original_source_sha256=original_protocol['source_sha256'],
                     resume_source_sha256=a.sha256(__file__), started=time.time(),
                     original_status=a.read_json(output/'STATUS.json'),
                     policy='Recover completed returns; never rerun an existing board transaction',
                     test_evaluated=False)
        with (output/'RESUME_CLAIM.json').open('x', encoding='utf-8') as handle:
            json.dump(claim, handle, indent=2)
    else:
        a.atomic_json(output/'UPSTREAM_SNAPSHOT.json', snapshot)
        a.atomic_json(output/'PROTOCOL.json', protocol)
        c.remote_json(c.remote_prefix(tx.REMOTE)+f"Path({remote!r}).mkdir(exist_ok=False);print(json.dumps(dict(status='CREATED')))")
    for network in c.NETWORKS:
        row = next(r for r in snapshot['runs'] if r['network']==network)
        if not row['action']['allowed']:
            raise ValueError('Board admission rejected: '+str(row['action']))
        new_build=remote+'/'+network+'/graph_build'
        folder=output/network/'graph_build'
        recovering = resume and folder.exists()
        if recovering:
            recovered_row = dict(row, build=new_build)
            validate_recovered_build(folder, recovered_row, snapshot['deployment_binding_sha256'])
            # Missing completion means a partial/unknown transaction; never launch again.
            from tools.sonar_deploy.board_result import validate_actual_frames
            validate_actual_frames(folder/'board_evidence',a.read_json(folder/'graph.json'),
                                   a.read_json(folder/'frames.json'),a.read_json(folder/'manifest.json'))
            if (output/network/'VALID99_REPORT.json').is_file():
                report = a.read_json(output/network/'VALID99_REPORT.json')
                if (report.get('status') != 'VALID99_PASS' or report.get('network') != network or
                    report.get('graph_sha256') != row['graph_sha256'] or
                    report.get('deployment_binding_sha256') != snapshot['deployment_binding_sha256'] or
                    report.get('test_evaluated') is not False):
                    raise ValueError('Previously completed valid99 report changed')
                continue
        else:
            a.atomic_json(output/'STATUS.json',dict(network=network,status='PREPARING_BOUND_FRAMES',test_evaluated=False))
            c._remote_command(tx.REMOTE,[tx.PYTHON,'tools/sonar_deploy/frames.py','--root',root,
                              '--network',network,'--device','cuda:0','--install'],timeout=1800)
            # Copy files; link only immutable synthesis directories and bound inputs.
            # All runtime artifacts are created in the new isolated build.
            c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
import shutil
from tools.sonar_deploy import admission as a
src=Path({row['build']!r});dst=Path({new_build!r});dst.mkdir(parents=True,exist_ok=False)
for item in src.iterdir():
 if item.is_file() and (item.suffix in {{'.json','.bin','.cpp','.h','.py','.tcl','.rpt','.xsa'}} or item.name in {{'ROUTE_TIMING_PASS','hls.log'}}):
  if item.name not in {{'board_return.tar.gz','board_execution_binding.json'}}:shutil.copy2(item,dst/item.name)
(dst/'hls_project').symlink_to(src/'hls_project',target_is_directory=True)
shutil.copytree(src/'frames',dst/'frames')
assert a.sha256(dst/'graph.json')=={row['graph_sha256']!r}
print(json.dumps(dict(status='ISOLATED_BUILD_READY')))
''')
        row=dict(row, build=new_build)
        if not recovering:
            c.fetch_small_build(dict(deployment_binding_sha256=snapshot['deployment_binding_sha256']),row,folder)
            a.atomic_json(output/'STATUS.json',dict(network=network,status='BOARD_TRAIN8_VALID99_RUNNING',test_evaluated=False))
            c._local_command([sys.executable,str(ROOT/'tools/detector_preflight/run_graph_board.py'),
                              str(folder),row['regression'],'--node-timeout','900'],folder/'board_queue.log')
        from tools.sonar_deploy.board_result import validate_actual_frames
        validate_actual_frames(folder/'board_evidence',a.read_json(folder/'graph.json'),
                               a.read_json(folder/'frames.json'),a.read_json(folder/'manifest.json'))
        upload_or_verify_evidence(tx, row, folder)
        a.atomic_json(output/'STATUS.json',dict(network=network,status='FROZEN_CPU_DECODE_EVALUATION',test_evaluated=False))
        c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
from tools.sonar_deploy import admission as a
from tools.sonar_deploy.board_result import validate_actual_frames,evaluate_board_cpu
from hwnas_fpga.metrics_ledger import build_metric_record,append_metric_record
r=Path({root!r});build=Path({new_build!r});e=build.parent/'board_evidence'
binding=a.load_binding(r)
assert a.sha256(r/'DEPLOYMENT_BINDING.json')=={snapshot['deployment_binding_sha256']!r}
identity=next(v for v in binding['runs'] if v['network']=={network!r} and v['seed']==42)
result=a.load_result(r,identity);graph=a.read_json(build/'graph.json');frames=a.read_json(build/'frames.json')
assert a.sha256(build/'graph.json')=={row['graph_sha256']!r}
actual=validate_actual_frames(e,graph,frames,a.read_json(build/'manifest.json'))
metrics,records=evaluate_board_cpu(r,result,graph,actual,frames,e)
a.atomic_json(e/'board_valid_metrics.json',metrics);a.atomic_json(e/'board_valid_predictions.json',records)
diff={{k:metrics[k]-result['integer_metrics'][k] for k in a.METRICS if metrics[k]!=result['integer_metrics'][k]}}
if diff:raise ValueError('Exact integer software/board metrics mismatch: '+str(diff))
prov=dict(source=str(e/'board_valid_metrics.json'),artifact_sha256=a.sha256(e/'board_valid_metrics.json'),
 evidence_layer='board_measurement',target_device='ZCU104',precision='INT8',batch_size=1,input_shape=[1,1,640,640],
 measurement_method='actual valid99 FPGA raw heads and frozen CPU decode/NMS',
 checkpoint_sha256=result['checkpoint_sha256'],evaluation_protocol_hash=result['evaluator_sha256'],split_manifest_sha256=result['split_manifest_sha256'])
event=build_metric_record(task_type='detection',dataset='Cylinder2',network={network!r},seed=42,
 stage='board_valid99',identity=dict(run_id={output.name!r}+'_'+{network!r},checkpoint_sha256=result['checkpoint_sha256'],test_evaluated=False,clean_reference_available=False),
 measured={{target:dict(value=metrics[source],provenance=prov) for source,target in a.METRICS.items()}})
append_metric_record(e/'METRICS.jsonl',event)
a.atomic_json(e/'VALID99_REPORT.json',dict(status='VALID99_PASS',network={network!r},seed=42,metrics=metrics,
 integer_metric_differences=diff,train_frames=8,valid_frames=99,mismatch_count=0,test_evaluated=False,
 deployment_binding_sha256={snapshot['deployment_binding_sha256']!r},graph_sha256={row['graph_sha256']!r},
 metric_event_id=event['event_id'],returned_archive_sha256=a.sha256(e/'board_return.tar.gz'),
 full_deployment_acceptance='SEPARATE_FINAL_REVIEW_REQUIRED'))
print(json.dumps(dict(status='VALID99_PASS',metrics=metrics)))
''',timeout=1800)
        c._scp([tx.HOST+':'+remote+'/'+network+'/board_evidence/'+n for n in
                ['VALID99_REPORT.json','METRICS.jsonl','board_valid_metrics.json','board_valid_predictions.json']],output/network)
        a.atomic_json(output/network/'STATUS.json',dict(status='VALID99_PASS',test_evaluated=False))
    a.atomic_json(output/'STATUS.json',dict(status='ALL_VALID99_COMPLETE',networks=list(c.NETWORKS),test_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Recover a completed board return and continue unstarted networks; never retry a partial transaction')
    args=p.parse_args()
    from hwnas_fpga.metrics_ledger import _ledger_lock
    with _ledger_lock(ROOT/'results/sonar_real_int8_deployment_v1/zcu104_board_campaign.lock',timeout=1):
        try:run(args.output,args.performance,args.resume)
        except BaseException as exc:
            failure = args.output/('RESUME_FAILURE.json' if args.resume else 'FAILURE.json')
            if not failure.exists():
                a.atomic_json(failure,dict(status='FAILED_NO_AUTORETRY',error=str(exc),test_evaluated=False))
            raise
