"""One bounded native+E2E performance queue after three verified smoke runs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a, board_campaign as c
from tools.sonar_deploy.power import verify_iio_evidence
from hwnas_fpga.metrics_ledger import build_metric_record, append_metric_record


def board_source(old, new, sha):
    return c.with_rpc_heartbeat(f'''import urllib.request,hashlib,tarfile,io,importlib.util,traceback,json,os
from pathlib import Path
url='http://192.168.31.1:18775'
raw=urllib.request.urlopen(url+'/board_payload.tar.gz',timeout=60).read()
assert hashlib.sha256(raw).hexdigest()=={sha!r}
root=Path({new!r});root.mkdir(exist_ok=False);adapter=root/'adapter';adapter.mkdir()
old=Path({old!r});bound=json.loads((old/'board_manifest.json').read_text())
for name,digest in bound['files'].items():
 assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest
 target=root/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,target)
os.link(old/'board_manifest.json',root/'board_manifest.json')
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
 for m in tf.getmembers():
  target=(adapter/m.name).resolve()
  if not m.isfile() or not target.is_relative_to(adapter.resolve()):raise ValueError('unsafe adapter')
  target.write_bytes(tf.extractfile(m).read())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v
error=None
native=load('retained_native',adapter/'performance_target.py')
rpc=load('retained_rpc',adapter/'board_e2e.py')
try:
 native.run(root,adapter)
 rpc.serve_board(root,adapter/'benchmark_manifest.json',port=18774,node_timeout=900,idle_timeout=600,max_session_seconds=14400)
except BaseException as exc:error=exc;traceback.print_exc()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tf:
 for prefix in ['native_results','results']:
  for p in (root/prefix).rglob('*'):
   if p.is_file():tf.add(p,arcname=p.relative_to(root).as_posix())
req=urllib.request.Request(url+'/board_return.tar.gz',data=stream.getvalue(),method='PUT')
print('PERFORMANCE_RETURN',urllib.request.urlopen(req,timeout=60).status,flush=True)
if error is not None:raise error
print('PERFORMANCE_COMPLETE',flush=True)
''')


def extract(archive, output):
    with tarfile.open(archive) as tf:
        rows=tf.getmembers();names=[m.name for m in rows]
        if len(names)!=len(set(names)) or sum(m.size for m in rows)>32*1024*1024:raise ValueError('Invalid result archive')
        for m in rows:
            target=(output/m.name).resolve()
            if not m.isfile() or not target.is_relative_to(output.resolve()) or target.exists():raise ValueError('Unsafe result')
        for m in rows:
            p=output/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(m).read())


def verify(folder, network):
    manifest=a.read_json(folder/'benchmark_manifest.json')
    native=a.read_json(folder/'native_results/completion.json')
    rpc=a.read_json(folder/'results/e2e_rpc_session.json');e2e=a.read_json(folder/'e2e.json')
    if native['status']!='PASS' or rpc['status']!='PASS' or e2e['status']!='PASS':raise ValueError('Incomplete performance session')
    binding={k:manifest[k] for k in ('deployment_binding_sha256','package_manifest_sha256','board_manifest_sha256','graph_sha256')}
    binding['benchmark_manifest_sha256']=a.sha256(folder/'benchmark_manifest.json')
    powers={}
    for label,rows in [('native_results',native['rows']),('results',rpc['rows'])]:
        powers[label]=verify_iio_evidence(folder/label/'power',binding,manifest['power_monitor']['source_sha256'],rows)
    measured={}
    provenance=dict(source=str(folder/'e2e.json'),artifact_sha256=a.sha256(folder/'e2e.json'),
        evidence_layer='board_measurement',target_device='ZCU104',precision='INT8',batch_size=1,
        input_shape=[1,1,640,640],warmup=1,repetitions=10,
        measurement_method='real CPU preprocess + RPC FPGA + CPU decode/NMS; actual completed-window FPS')
    for name,value in [('board_latency_ms',e2e['timing']['end_to_end_median_ms']),
                       ('throughput_fps',e2e['completed_window_fps']),('actual_clock_mhz',e2e['actual_clock_mhz'])]:
        measured[name]=dict(value=value,provenance=provenance)
    p=powers['results']
    if p['status']=='MEASURED':
        prov=dict(provenance,source=str(folder/'results/power/power.json'),
            artifact_sha256=a.sha256(folder/'results/power/power.json'),
            power_scope='whole_board_12v_infer_call_windows_only; host energy excluded',idle_subtracted=False)
        for name in ['power_w','energy_per_inference_mj']:
            measured[name]=dict(value=p['summary'][name],provenance=prov)
    event=build_metric_record(task_type='detection',dataset='Cylinder2',network=network,seed=42,
        stage='formal_board_performance',identity=dict(run_id=folder.parent.name+'_'+network,
            checkpoint_sha256=manifest['checkpoint_sha256'],test_evaluated=False,clean_reference_available=False),measured=measured)
    append_metric_record(folder/'METRICS.jsonl',event)
    report=dict(status='PERFORMANCE_PASS',network=network,native=native,e2e=e2e,power=powers,
        full_board_acceptance=False,valid99='NOT_RUN',test_evaluated=False,metric_event_id=event['event_id'])
    a.atomic_json(folder/'PERFORMANCE_REPORT.json',report)
    return report


def run(smoke, output, resume=False):
    if not resume: output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir(): raise ValueError("Missing original campaign")
    tx=c.transport();project=tx.REMOTE
    remote=project+'/results/sonar_real_int8_deployment_v1/'+output.name
    root=project+'/results/sonar_real_int8_deployment_v1/cpu_decode_v1_route02'
    previous=a.read_json(smoke/'SMOKE_MANIFEST.json')
    if a.read_json(smoke/'STATUS.json')['status']!='THREE_SINGLE_IMAGE_SMOKES_COMPLETE':raise ValueError('All smoke runs must pass first')
    for spec in previous['candidates']:
        receipt=a.read_json(smoke/spec['network']/'evidence/VERIFIED.json')
        for name,digest in receipt['files'].items():
            if a.sha256(smoke/spec['network']/'evidence'/name)!=digest:raise ValueError('Smoke changed')
    if not resume:
        a.atomic_json(output/'PROTOCOL.json',dict(order=['D','C','B1'],seed=42,batch=1,input=640,
            groups=['native_graph','host_end_to_end'],warmup_per_group=1,repetitions_per_group=10,
            layer_trace=False,weights_resident=True,idle_seconds=5,power_poll_hz=10,
            test_evaluated=False,valid_evaluated=False,smoke_manifest_sha256=a.sha256(smoke/'SMOKE_MANIFEST.json')))
    else:
        protocol=a.read_json(output/'PROTOCOL.json')
        if protocol['smoke_manifest_sha256']!=a.sha256(smoke/'SMOKE_MANIFEST.json'):
            raise ValueError('Resume smoke protocol changed')
        if (output/'RESUME_VERIFICATION.json').exists():raise ValueError('Resume already claimed')
        a.atomic_json(output/'RESUME_VERIFICATION.json',dict(
            original_status=a.read_json(output/'STATUS.json'),protocol_sha256=a.sha256(output/'PROTOCOL.json'),
            verifier_sha256=a.sha256(Path(__file__).with_name('power.py')),
            policy='derived power/energy floats <=8 binary64 ULP; all other fields exact',test_evaluated=False))

    try:
        a.atomic_json(output/'STATUS.json',dict(status='PREPARING',test_evaluated=False))
        if not resume:
            c.remote_json(c.remote_prefix(project)+f"Path({remote!r}).mkdir(exist_ok=False);print(json.dumps({{'status':'CREATED'}}))")
            files=['board_e2e.py','power.py','performance_target.py','performance_admission.py']
            c._scp([str(Path(__file__).with_name(n)) for n in files],tx.HOST+':'+project+'/tools/sonar_deploy/')
            archive=output/'smoke_evidence.tar.gz'
            with tarfile.open(archive,'w:gz') as tf:
                for network in ['D','C','B1']:
                    for p in (smoke/network/'evidence').rglob('*'):
                        if p.is_file():tf.add(p,arcname=p.relative_to(smoke).as_posix())
            c._scp([str(archive)],tx.HOST+':'+remote+'/smoke_evidence.tar.gz')
            c.remote_json(c.remote_prefix(project)+f'''
    import tarfile
    r=Path({remote!r})
    with tarfile.open(r/'smoke_evidence.tar.gz') as tf:
     for m in tf.getmembers():
      target=(r/'smoke'/m.name).resolve()
      if not m.isfile() or not target.is_relative_to((r/'smoke').resolve()):raise ValueError('unsafe smoke')
      target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(m).read())
    source=(p/'tools/detector_preflight/http_transfer.py').read_text()
    assert source.count("('192.168.31.1',18773)")==1
    (r/'transfer.py').write_text(source.replace("('192.168.31.1',18773)","('192.168.31.1',18775)"))
    print(json.dumps(dict(status='PREPARED')))
    ''')
        summaries=[]
        for network in ['D','C','B1']:
            folder=output/network
            if resume and folder.exists():
                if (folder/'PERFORMANCE_REPORT.json').exists():
                    report=a.read_json(folder/'PERFORMANCE_REPORT.json')
                    if report['status']!='PERFORMANCE_PASS':raise ValueError('Previous result failed')
                    summaries.append(report)
                else:
                    summaries.append(verify(folder,network))
                continue
            folder.mkdir();rd=remote+'/'+network
            old='/home/xilinx/'+smoke.name+'_'+network
            board='/home/xilinx/'+output.name+'_'+network
            a.atomic_json(output/'STATUS.json',dict(status='VERIFYING_CANDIDATE',network=network,test_evaluated=False))
            prepared=c.remote_json(c.remote_prefix(project)+f'''
import tarfile,hashlib,subprocess
from tools.sonar_deploy.board_e2e import prepare_manifest
r=Path({rd!r});r.mkdir()
board=Path({previous['remote']!r})/{network!r}/'payload/board_manifest.json'
m=prepare_manifest(Path({root!r}),{network!r},board,r/'benchmark_manifest.json',power_monitor=True,
 smoke_evidence=Path({remote!r})/'smoke'/{network!r}/'evidence')
with tarfile.open(r/'board_payload.tar.gz','w:gz') as tf:
 tf.add(r/'benchmark_manifest.json',arcname='benchmark_manifest.json')
 for name in ['board_e2e.py','power.py','performance_target.py']:tf.add(p/'tools/sonar_deploy'/name,arcname=name)
proc=subprocess.Popen(['python3',{remote!r}+'/transfer.py',str(r),'--max-return-bytes','33554432','--max-seconds','18000'],stdout=(r/'http.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,sha=hashlib.sha256((r/'board_payload.tar.gz').read_bytes()).hexdigest())))
''',timeout=600)
            script=folder/'execute.py';script.write_text(board_source(old,board,prepared['sha']),encoding='utf-8')
            with (folder/'bridge.log').open('x',encoding='utf-8') as log:
                bridge=subprocess.Popen([sys.executable,str(c.BRIDGE),str(script),str(folder/'board.log'),'--keep-kernel'],stdout=log,stderr=subprocess.STDOUT)
                a.atomic_json(output/'STATUS.json',dict(status='NATIVE_BENCHMARK_RUNNING',network=network,bridge_pid=bridge.pid,test_evaluated=False))
                started=time.monotonic()
                while 'E2E_RPC_READY' not in ((folder/'board.log').read_text(errors='replace') if (folder/'board.log').exists() else ''):
                    if bridge.poll() is not None:raise RuntimeError('Board exited before RPC readiness; inspect logs')
                    if time.monotonic()-started>9000:raise TimeoutError('Native phase/readiness timeout; retain board kernel')
                    time.sleep(2)
                a.atomic_json(output/'STATUS.json',dict(status='E2E_BENCHMARK_RUNNING',network=network,test_evaluated=False))
                c._remote_command(project,[tx.PYTHON,'tools/sonar_deploy/board_e2e.py','host','--root',root,
                    '--network',network,'--manifest',rd+'/benchmark_manifest.json','--url','http://192.168.31.217:18774',
                    '--output',rd+'/e2e.json','--smoke-evidence',remote+'/smoke/'+network+'/evidence'],timeout=14700)
                if bridge.wait(timeout=240):raise RuntimeError('Board session failed')
            c._scp([tx.HOST+':'+rd+'/'+n for n in ['board_return.tar.gz','benchmark_manifest.json','e2e.json']],folder)
            extract(folder/'board_return.tar.gz',folder)
            summaries.append(verify(folder,network))
            c.remote_json(c.remote_prefix(project)+f'''
import signal
proc=Path('/proc')/str({prepared['pid']})/'cmdline'
if proc.exists() and {rd!r}.encode() in proc.read_bytes().split(bytes([0])):os.kill({prepared['pid']},signal.SIGTERM)
print(json.dumps(dict(status='TRANSFER_STOPPED')))
''')
        a.atomic_json(output/'STATUS.json',dict(status='FORMAL_PERFORMANCE_COMPLETE',networks=['D','C','B1'],test_evaluated=False,valid99='NOT_RUN'))
        a.atomic_json(output/'PERFORMANCE_RESULTS.json',summaries)
        table=[]
        for summary in summaries:
            native,e2e=summary['native'],summary['e2e']
            power=summary['power']['results'].get('summary',{})
            table.append(dict(network=summary['network'],seed=42,
                graph_median_ms=native['graph_latency']['median_ms'],graph_p95_ms=native['graph_latency']['p95_ms'],
                native_completed_window_fps=native['throughput_fps'],
                e2e_median_ms=e2e['timing']['end_to_end_median_ms'],e2e_p95_ms=e2e['timing']['end_to_end_p95_ms'],
                e2e_completed_window_fps=e2e['completed_window_fps'],clock_mhz=e2e['actual_clock_mhz'],
                board_power_w=power.get('power_w','NOT_MEASURED'),
                board_energy_mj_image=power.get('energy_per_inference_mj','NOT_MEASURED'),
                board_dynamic_energy_mj_image=power.get('dynamic_energy_per_inference_mj','NOT_MEASURED'),
                power_scope='board infer-call windows; excludes host energy',valid99='NOT_RUN',test_evaluated=False))
        with (output/'PERFORMANCE_SUMMARY.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        (output/'PERFORMANCE_SUMMARY.md').write_text('# Formal performance results\n\n'
            'Batch1, input640, seed42, resident weights, warmup1 + measured10 per native/E2E group. '
            'No intermediate-layer trace. Board VCC12 energy excludes host energy. '
            'FPS uses completed images divided by the measured wall window. '
            'Full valid99 and complete board accuracy acceptance remain pending; test_evaluated=false.\n\n'
            '| Network | Graph median ms | E2E median ms | E2E FPS | Board W | Board mJ/image |\n'
            '|---|---:|---:|---:|---:|---:|\n'+''.join(
                f"| {r['network']} | {r['graph_median_ms']:.3f} | {r['e2e_median_ms']:.3f} | {r['e2e_completed_window_fps']:.6f} | {r['board_power_w']} | {r['board_energy_mj_image']} |\n" for r in table),encoding='utf-8')
    except BaseException as exc:
        a.atomic_json(output/'STATUS.json',dict(status='STOP_FAILED_NO_AUTORETRY',error=str(exc),test_evaluated=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.smoke.resolve(),args.output.resolve(),args.resume)
