"""One-attempt Windows orchestration of admitted D/C/B1 ZCU104 validation.

No hardware evidence is synthesized here. Each stage waits for real upstream
admission. Failed board attempts are retained and are never automatically rerun.
Authentication is exclusively delegated to the existing local-only bridges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import textwrap
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a

NETWORKS=('D','C','B1')
DEFAULT_RUN='results/sonar_real_int8_deployment_v1/cpu_decode_v1'
BRIDGE=Path('C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/jupyter_reuse_bridge.py')


def transport():
    # Import lazily: --help and tests need no credential/helper environment.
    from tools.sonar_deploy import remote
    return remote


def validate_remote_run(path, project):
    root=PurePosixPath(path);base=PurePosixPath(project)/'results/sonar_real_int8_deployment_v1'
    if not root.is_absolute() or not root.is_relative_to(base) or '..' in root.parts:
        raise ValueError('Campaign requires an isolated real INT8 deployment root')
    return str(root)


def remote_json(code, *, timeout=120):
    value=transport().execute(code,timeout=timeout)
    # Torch/tool logs can precede a final compact JSON status line.
    return json.loads(value.rstrip().splitlines()[-1])


def remote_prefix(project):
    return f"from pathlib import Path\nimport os,sys,json\np=Path({project!r});os.chdir(p)\nsys.path[:0]=[str(p),str(p/'src'),str(p/'tools'),str(p/'tools/sonar_analysis'),str(p/'scripts')]\n"


def snapshot(remote_run, project):
    validate_remote_run(remote_run,project)
    return remote_json(remote_prefix(project)+f'''
from tools.sonar_deploy import admission as a
r=Path({remote_run!r});binding=a.load_binding(r);quality=a.accuracy_admission(r,binding)
if not quality['allowed']:raise RuntimeError('Nine verified software results required: '+json.dumps(quality))
rows=[]
for network in a.NETWORKS:
 expected=next(v for v in binding['runs'] if v['network']==network and v['seed']==42)
 result=a.load_result(r,expected);build=r/network/'seed_42/hardware/graph_build'
 package=Path(result['package_manifest_path']);package=package if package.is_absolute() else r/package
 row={{'network':network,'architecture_id':result['architecture_id'],'checkpoint_sha256':result['checkpoint_sha256'],
      'package_manifest_sha256':result['package_manifest_sha256'],'graph_sha256':a.sha256(package.parent/'graph.json'),
      'accuracy':quality['gates'][network]['status'],'action':a.action_admission(r,network,'board',42),
      'build':str(build),'regression':str(build.parent/'numeric_regression'),
      'graph_session_available':(build/'pynq_graph_runner.py').exists() and 'class GraphSession:' in (build/'pynq_graph_runner.py').read_text(),
      'board_complete':False,'attempt_exists':(build.parent/'BOARD_CAMPAIGN_ATTEMPT.json').exists()}}
 if (r/network/'seed_42/board.json').exists():
  a.stage_evidence(r,network,'board',result);row['board_complete']=True
 rows.append(row)
execution_path=r/'HARDWARE_EXECUTION.json'
execution=a.read_json(execution_path) if execution_path.is_file() else None
print(json.dumps({{'deployment_binding_sha256':a.sha256(r/'DEPLOYMENT_BINDING.json'),'runs':rows,
 'hardware_execution':execution,'hardware_execution_sha256':a.sha256(execution_path) if execution else None,
 'test_evaluated':False}}))
''')


def prepare(output, remote_run, networks, *, board_ip='192.168.31.217', node_timeout=900, power_monitor=True):
    import ipaddress
    if not set(networks)<=set(NETWORKS):raise ValueError('Unknown network')
    if not ipaddress.ip_address(board_ip) in ipaddress.ip_network('192.168.31.0/24'):
        raise ValueError('Board must be on the existing private board LAN')
    if not 0<node_timeout<=3600:raise ValueError('Node watchdog must be bounded')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    project=transport().REMOTE
    state=snapshot(remote_run,project)
    rows=[row for row in state['runs'] if row['network'] in networks]
    manifest=dict(schema_version='real-int8-board-campaign-v1',remote_project=project,
        remote_run=remote_run,board_ip=board_ip,node_timeout_seconds=node_timeout,
        node_timeout_role='functional watchdog, not FPS/latency acceptance threshold',
        networks=[n for n in NETWORKS if n in networks],board_seed=42,
        deployment_binding_sha256=state['deployment_binding_sha256'],one_attempt_per_network=True,
        power_monitor_enabled=bool(power_monitor),
        candidates=[{k:r[k] for k in ('network','architecture_id','checkpoint_sha256','package_manifest_sha256','graph_sha256')} for r in rows],
        test_evaluated=False)
    path=output/'CAMPAIGN_MANIFEST.json'
    if path.exists() and a.read_json(path)!=manifest:
        raise ValueError('Frozen campaign differs; never overwrite an admitted experiment')
    a.atomic_json(path,manifest);a.atomic_json(output/'LATEST_SNAPSHOT.json',state)
    return manifest,state


def _verify_snapshot(manifest,state,network):
    if state.get('test_evaluated') is not False or state['deployment_binding_sha256']!=manifest['deployment_binding_sha256']:
        raise ValueError('Deployment identity changed')
    row=next(r for r in state['runs'] if r['network']==network)
    expected=next(r for r in manifest['candidates'] if r['network']==network)
    if any(row[key]!=value for key,value in expected.items()):raise ValueError('Candidate identity changed')
    return row


def _local_command(argv,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as handle:
        process=subprocess.Popen(list(argv),stdout=handle,stderr=subprocess.STDOUT,shell=False)
        a.atomic_json(log.with_suffix(log.suffix+'.process.json'),dict(pid=process.pid,command=list(argv),started=time.time()))
        code=process.wait()
    if code:raise RuntimeError(f'Command failed ({code}); retained log {log}')


def _remote_command(project,argv,timeout=1200):
    return remote_json(remote_prefix(project)+f'''
import subprocess
command={list(argv)!r}
process=subprocess.run(command,cwd=p,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
print(process.stdout,flush=True)
if process.returncode:raise RuntimeError('Command failed: '+str(process.returncode))
print(json.dumps({{'status':'COMMAND_EXIT_0','command':command}}))
''',timeout=timeout)


def _scp(sources,destination):
    tx=transport()
    subprocess.run(tx.SCP+list(sources)+[str(destination)],env=tx.ENV,check=True,shell=False)


def fetch_small_build(manifest,row,folder):
    folder.mkdir(parents=True,exist_ok=True)
    names=('manifest.json','h2_graph.cpp','numeric_v1.h','pynq_graph_runner.py','graph.json','frames.json')
    tx=transport()
    _scp([tx.HOST+':'+row['build']+'/'+name for name in names],folder)
    bound=a.read_json(folder/'manifest.json')
    if bound['graph_sha256']!=row['graph_sha256'] or not bound.get('real_weights_bound'):
        raise ValueError('Fetched build graph identity mismatch')
    for name in ('h2_graph.cpp','numeric_v1.h','pynq_graph_runner.py','graph.json'):
        if bound['files'].get(name)!=a.sha256(folder/name):raise ValueError('Fetched frozen build source differs: '+name)
    if 'class GraphSession:' not in (folder/'pynq_graph_runner.py').read_text():
        raise ValueError('Frozen build lacks GraphSession; explicit bound runtime adapter required')
    a.atomic_json(folder/'remote_launch.json',dict(remote=row['build'],network=row['network'],
        deployment_binding_sha256=manifest['deployment_binding_sha256'],test_evaluated=False))


def claim_attempt(manifest,row,output):
    payload=dict(schema_version='real-int8-board-attempt-v1',network=row['network'],seed=42,
        started=time.time(),deployment_binding_sha256=manifest['deployment_binding_sha256'],
        graph_sha256=row['graph_sha256'],package_manifest_sha256=row['package_manifest_sha256'],
        node_timeout_seconds=manifest['node_timeout_seconds'],test_evaluated=False)
    remote_json(remote_prefix(manifest['remote_project'])+f'''
from tools.sonar_deploy import admission as a
r=Path({manifest['remote_run']!r});network={row['network']!r}
decision=a.action_admission(r,network,'board',42)
if not decision['allowed']:raise RuntimeError('Board gate changed: '+json.dumps(decision))
if a.sha256(r/'DEPLOYMENT_BINDING.json')!={manifest['deployment_binding_sha256']!r}:raise RuntimeError('Binding changed')
path=r/network/'seed_42/hardware/BOARD_CAMPAIGN_ATTEMPT.json'
with path.open('x') as f:
 json.dump({payload!r},f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({{'status':'ATTEMPT_CLAIMED','network':network}}))
''')
    a.atomic_json(Path(output)/row['network']/'ATTEMPT.json',payload)
    return payload


def upload_evidence(manifest,row,local_evidence):
    tx=transport();archive=local_evidence.parent/'returned_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for path in local_evidence.rglob('*'):
            if path.is_file():tf.add(path,arcname=path.relative_to(local_evidence).as_posix())
    remote_archive=row['build']+'/campaign_returned_evidence.tar.gz'
    _scp([str(archive)],tx.HOST+':'+remote_archive)
    expected=a.sha256(archive)
    return remote_json(remote_prefix(manifest['remote_project'])+f'''
import hashlib,tarfile
archive=Path({remote_archive!r});destination=Path({row['build']!r}).parent/'board_evidence'
if hashlib.sha256(archive.read_bytes()).hexdigest()!={expected!r}:raise ValueError('Evidence transport corrupt')
destination.mkdir(exist_ok=False)
with tarfile.open(archive) as tf:
 for member in tf.getmembers():
  target=(destination/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(destination):raise ValueError('Unsafe evidence archive')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(member).read())
print(json.dumps({{'status':'BOARD_RETURN_IMPORTED','path':str(destination)}}))
''')


def _start_transfer(project,directory,max_seconds):
    return remote_json(remote_prefix(project)+f'''
import subprocess
folder=Path({directory!r})
log=(folder/'http.log').open('x')
process=subprocess.Popen(['python3',str(p/'tools/detector_preflight/http_transfer.py'),str(folder),
 '--max-return-bytes','33554432','--max-seconds',str({max_seconds})],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split(') ',1)[1].split()[19]
print(json.dumps({{'pid':process.pid,'start_ticks':start_ticks,'directory':str(folder)}}))
''')


def _stop_transfer(project,owner):
    return remote_json(remote_prefix(project)+f'''
import signal
proc=Path('/proc')/str({owner['pid']})
if proc.exists():
 args=(proc/'cmdline').read_bytes().split(bytes([0]))
 ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]
 if ticks!={owner['start_ticks']!r} or {owner['directory']!r}.encode() not in args or not any(v.endswith(b'http_transfer.py') for v in args):raise RuntimeError('Transfer ownership changed')
 os.kill({owner['pid']},signal.SIGTERM)
print(json.dumps({{'status':'OWNED_TRANSFER_STOPPED'}}))
''')


def extract_rpc_return(archive,destination):
    """Import only explicitly named RPC/power artifacts, preserving failure rows."""
    destination=Path(destination)
    allowed={'e2e_rpc_session.json','power/power.json','power/power_samples.csv'}
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or 'e2e_rpc_session.json' not in names or not set(names)<=allowed:
            raise ValueError('RPC return contains unregistered/duplicate/missing artifacts')
        if any(not m.isfile() or m.size>32*1024*1024 for m in members):
            raise ValueError('RPC evidence member is invalid or unbounded')
        if any((destination/m.name).exists() for m in members):
            raise ValueError('Do not replace existing RPC/power evidence')
        files={}
        for member in members:
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tf.extractfile(member).read());files[member.name]=a.sha256(target)
    value=a.read_json(destination/'e2e_rpc_session.json')
    return dict(status=value.get('status'),successful_calls=value.get('successful_calls'),
        archive_sha256=a.sha256(archive),files=files)


def collect_rpc_evidence(project,rpc_dir,remote_evidence,local_evidence):
    """Preserve board-close status even when the retained cell reports failure."""
    tx=transport()
    result=remote_json(remote_prefix(project)+f'''
from tools.sonar_deploy.board_campaign import extract_rpc_return
source=Path({rpc_dir!r})/'board_return.tar.gz'
value=extract_rpc_return(source,Path({remote_evidence!r}))
print(json.dumps(value))
''')
    for name,digest in result['files'].items():
        target=local_evidence/name;target.parent.mkdir(parents=True,exist_ok=True)
        _scp([tx.HOST+':'+remote_evidence+'/'+name],target)
        if a.sha256(target)!=digest:raise ValueError('RPC evidence transport hash mismatch')
    a.atomic_json(local_evidence/'e2e_rpc_session_binding.json',result)
    return result


def validate_rpc_completion(folder,bridge_code):
    from tools.sonar_deploy.board_result import _validate_rpc_session
    folder=Path(folder)
    _validate_rpc_session(folder/'e2e_rpc_session.json',a.read_json(folder/'e2e.json'))
    if bridge_code:raise RuntimeError('Board RPC cell failed; preserved board session evidence')


def with_rpc_heartbeat(code):
    """Keep the existing 60s Jupyter transport alive; all FPGA work stays main-thread."""
    return '''import threading
_e2e_heartbeat_stop=threading.Event()
def _e2e_heartbeat():
 while not _e2e_heartbeat_stop.wait(10):print('E2E_RUNNING',flush=True)
_e2e_heartbeat_worker=threading.Thread(target=_e2e_heartbeat,daemon=True,name='e2e_transport_heartbeat')
_e2e_heartbeat_worker.start()
try:
'''+textwrap.indent(code,'    ')+'''
finally:
 _e2e_heartbeat_stop.set()
 _e2e_heartbeat_worker.join()
'''


def e2e_transaction(manifest,row,folder):
    tx=transport();project=manifest['remote_project'];remote_evidence=str(PurePosixPath(row['build']).parent/'board_evidence')
    remote_manifest=remote_evidence+'/benchmark_manifest.json'
    prepare_argv=[tx.PYTHON,'tools/sonar_deploy/board_e2e.py','prepare','--root',manifest['remote_run'],
        '--network',row['network'],'--board-manifest',remote_evidence+'/board_manifest.json','--manifest',remote_manifest]
    if manifest.get('power_monitor_enabled'):prepare_argv.append('--power-monitor')
    _remote_command(project,prepare_argv)
    local_evidence=folder/'board_evidence'
    _scp([tx.HOST+':'+remote_manifest],local_evidence/'benchmark_manifest.json')
    remote_adapter=project+'/tools/sonar_deploy/board_e2e.py'
    if a.read_json(local_evidence/'benchmark_manifest.json')['rpc_source_sha256']!=a.sha256(ROOT/'tools/sonar_deploy/board_e2e.py'):
        raise ValueError('Host/board RPC adapter source version mismatch')
    rpc_dir=str(PurePosixPath(row['build']).parent/'e2e_transfer')
    payload=remote_json(remote_prefix(project)+f'''
import tarfile,hashlib
folder=Path({rpc_dir!r});folder.mkdir(exist_ok=False)
archive=folder/'board_payload.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 tf.add({remote_adapter!r},arcname='board_e2e.py');tf.add({remote_manifest!r},arcname='benchmark_manifest.json')
 protocol=json.loads(Path({remote_manifest!r}).read_text()).get('power_monitor')
 if protocol:
  power=Path({remote_adapter!r}).with_name('power.py')
  if hashlib.sha256(power.read_bytes()).hexdigest()!=protocol['source_sha256']:raise ValueError('Power adapter changed')
  tf.add(power,arcname='power.py')
print(json.dumps({{'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))
''')
    board_folder=a.read_json(folder/'board_execution_binding.json')['board_directory']
    code=f'''import urllib.request,tarfile,io,hashlib,importlib.util,traceback,json
from pathlib import Path
url='http://192.168.31.1:18773'
raw=urllib.request.urlopen(url+'/board_payload.tar.gz',timeout=30).read()
assert hashlib.sha256(raw).hexdigest()=={payload['sha256']!r}
folder=Path({board_folder!r});adapter=folder/'rpc_adapter';adapter.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
 for member in tf.getmembers():
  target=(adapter/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(adapter):raise ValueError('Unsafe RPC payload')
  target.write_bytes(tf.extractfile(member).read())
spec=importlib.util.spec_from_file_location('bounded_real_int8_rpc',adapter/'board_e2e.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
error=None
try:module.serve_board(folder,adapter/'benchmark_manifest.json',port=18774,node_timeout={manifest['node_timeout_seconds']!r},idle_timeout=180,max_session_seconds=14400)
except BaseException as exc:error=exc;traceback.print_exc()
path=folder/'results/e2e_rpc_session.json'
if not path.exists():path.write_text(json.dumps({{'status':'FAILED_NO_SESSION_EVIDENCE'}}))
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tf:
 for name in ('e2e_rpc_session.json','power/power.json','power/power_samples.csv'):
  item=folder/'results'/name
  if item.is_file():tf.add(item,arcname=name)
data=stream.getvalue()
req=urllib.request.Request(url+'/board_return.tar.gz',data=data,method='PUT')
print('E2E_SESSION_UPLOAD',urllib.request.urlopen(req,timeout=30).status,flush=True)
if error is not None:raise error
print('E2E_TRANSACTION_COMPLETE',flush=True)
'''
    script=folder/'e2e_board_execute.py';script.write_text(with_rpc_heartbeat(code),encoding='utf-8')
    owner=_start_transfer(project,rpc_dir,14700)
    logfile=folder/'e2e_bridge.log';cell_log=folder/'e2e_cell.log'
    with logfile.open('x',encoding='utf-8') as handle:
        bridge=subprocess.Popen([sys.executable,str(BRIDGE),str(script),str(cell_log),'--keep-kernel'],
                                stdout=handle,stderr=subprocess.STDOUT,shell=False)
        a.atomic_json(folder/'E2E_PROCESS.json',dict(pid=bridge.pid,transfer=owner,test_evaluated=False))
        try:
            start=time.monotonic()
            while 'E2E_RPC_READY' not in (cell_log.read_text(errors='replace') if cell_log.exists() else ''):
                if bridge.poll() is not None:raise RuntimeError('Board RPC exited before READY; inspect retained logs')
                if time.monotonic()-start>600:raise TimeoutError('Board RPC readiness timeout; retain kernel/process for inspection')
                time.sleep(.5)
            host_error=None
            try:
                _remote_command(project,[tx.PYTHON,'tools/sonar_deploy/board_e2e.py','host','--root',manifest['remote_run'],
                    '--network',row['network'],'--manifest',remote_manifest,'--url','http://'+manifest['board_ip']+':18774',
                    '--output',remote_evidence+'/e2e.json','--timeout','3600'],timeout=14700)
            except BaseException as exc:host_error=exc
            # Host receives response 11 before the board closes its buffers.
            # Wait for the retained cell, and preserve QUARANTINED evidence too.
            bridge_code=bridge.wait(timeout=240)
            completion=collect_rpc_evidence(project,rpc_dir,remote_evidence,local_evidence)
            if host_error is not None:raise host_error
            _scp([tx.HOST+':'+remote_evidence+'/'+name for name in ('e2e.json','benchmark_manifest.json')],local_evidence)
            validate_rpc_completion(local_evidence,bridge_code)
            if completion['status']!='PASS':raise RuntimeError('Board RPC did not release buffers safely')
        finally:
            if bridge.poll() is not None:
                _stop_transfer(project,owner)
            else:
                # No SIGKILL and no kernel cleanup while device ownership may
                # remain. Both RPC and transfer have bounded own lifetimes.
                a.atomic_json(folder/'E2E_QUARANTINE.json',dict(status='RETAINED_KERNEL_AND_PROCESS',
                    bridge_pid=bridge.pid,transfer=owner,test_evaluated=False))


def run_network(output,manifest,network,*,gate_wait_seconds=0,poll_seconds=30):
    output=Path(output);folder=output/network/'graph_build';status_file=output/network/'STATUS.json'
    start=time.monotonic()
    while True:
        state=snapshot(manifest['remote_run'],manifest['remote_project'])
        a.atomic_json(output/'LATEST_SNAPSHOT.json',state)
        row=_verify_snapshot(manifest,state,network)
        if (state.get('hardware_execution') or {}).get('status')=='FAILED':
            result=dict(status='STOP_HARDWARE_FAILURE',network=network,
                hardware_execution=state['hardware_execution'],
                hardware_execution_sha256=state.get('hardware_execution_sha256'),test_evaluated=False)
            a.atomic_json(status_file,result);return result
        if row['board_complete']:
            result=dict(status='ALREADY_VERIFIED',network=network,test_evaluated=False)
            a.atomic_json(status_file,result);return result
        if row['attempt_exists'] or (output/network/'ATTEMPT.json').exists():
            result=dict(status='STOP_EXISTING_ATTEMPT_REVIEW_REQUIRED',network=network,test_evaluated=False)
            a.atomic_json(status_file,result);return result
        if row['accuracy']=='STOP_ACCURACY_BELOW_D':
            result=dict(status='STOP_ACCURACY_BELOW_D',network=network,test_evaluated=False)
            a.atomic_json(status_file,result);return result
        if row['action']['allowed']:
            if not row['graph_session_available']:raise ValueError('Frozen graph driver lacks required GraphSession interface')
            break
        result=dict(status='WAITING_STAGE_EVIDENCE',network=network,decision=row['action'],test_evaluated=False)
        a.atomic_json(status_file,result)
        if time.monotonic()-start>=gate_wait_seconds:return result
        time.sleep(min(poll_seconds,max(.01,gate_wait_seconds-(time.monotonic()-start))))
    tx=transport()
    # Integer reference frame preparation is not a board attempt. No programming
    # occurs until this deterministic train8/valid99 bundle is complete.
    _remote_command(manifest['remote_project'],[tx.PYTHON,'tools/sonar_deploy/frames.py','--root',manifest['remote_run'],
                    '--network',network,'--device','cuda:0','--install'],timeout=1800)
    fetch_small_build(manifest,row,folder)
    claim_attempt(manifest,row,output)
    try:
        a.atomic_json(status_file,dict(status='BOARD_TRAIN8_VALID99_RUNNING',network=network,test_evaluated=False))
        _local_command([sys.executable,str(ROOT/'tools/detector_preflight/run_graph_board.py'),str(folder),row['regression'],
                        '--node-timeout',str(manifest['node_timeout_seconds'])],folder/'full_board.log')
        from tools.sonar_deploy.board_result import validate_actual_frames
        validate_actual_frames(folder/'board_evidence',a.read_json(folder/'graph.json'),
                               a.read_json(folder/'frames.json'),a.read_json(folder/'manifest.json'))
        upload_evidence(manifest,row,folder/'board_evidence')
        a.atomic_json(status_file,dict(status='END_TO_END_BENCHMARK_RUNNING',network=network,test_evaluated=False))
        e2e_transaction(manifest,row,folder)
        _remote_command(manifest['remote_project'],[tx.PYTHON,'tools/sonar_deploy/run.py','board','--root',manifest['remote_run'],
                        '--network',network,'--seed','42'],timeout=900)
        final=_verify_snapshot(manifest,snapshot(manifest['remote_run'],manifest['remote_project']),network)
        if not final['board_complete']:raise RuntimeError('Returned evidence did not pass final board admission')
        result=dict(status='COMPLETE_VERIFIED',network=network,test_evaluated=False)
        a.atomic_json(status_file,result);return result
    except BaseException as exc:
        a.atomic_json(status_file,dict(status='FAILED_RETAINED_NO_AUTORETRY',network=network,error=type(exc).__name__+': '+str(exc),test_evaluated=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    parser.add_argument('--network',choices=[*NETWORKS,'all'],default='D')
    parser.add_argument('--remote-run')
    parser.add_argument('--output',type=Path,default=ROOT/'results/sonar_real_int8_deployment_v1/board_campaign_cpu_decode_v1')
    parser.add_argument('--board-ip',default='192.168.31.217')
    parser.add_argument('--node-timeout',type=float,default=900)
    parser.add_argument('--gate-wait-seconds',type=float,default=0)
    parser.add_argument('--no-power-monitor',action='store_true',help='Freeze timing-only mode; power remains NOT_MEASURED')
    args=parser.parse_args()
    networks=NETWORKS if args.network=='all' else (args.network,)
    path=args.output/'CAMPAIGN_MANIFEST.json'
    if args.action=='prepare':
        remote_run=args.remote_run or transport().REMOTE+'/'+DEFAULT_RUN
        manifest,state=prepare(args.output,remote_run,networks,board_ip=args.board_ip,node_timeout=args.node_timeout,
                               power_monitor=not args.no_power_monitor)
        print(json.dumps(dict(status='PREPARED_NOT_PROGRAMMED',manifest=manifest,snapshot=state),indent=2));return
    if not path.is_file():raise RuntimeError('Prepare the immutable campaign manifest first')
    manifest=a.read_json(path)
    if args.remote_run is not None and args.remote_run!=manifest['remote_run']:
        raise ValueError('Requested run differs from frozen campaign; prepare an independent output directory')
    if args.action=='status':
        print(json.dumps(snapshot(manifest['remote_run'],manifest['remote_project']),indent=2));return
    if any(n not in manifest['networks'] for n in networks):raise ValueError('Requested network was not registered in campaign')
    from hwnas_fpga.metrics_ledger import _ledger_lock
    with _ledger_lock(ROOT/'results/sonar_real_int8_deployment_v1/zcu104_board_campaign.lock',timeout=1):
        for network in networks:
            result=run_network(args.output,manifest,network,gate_wait_seconds=args.gate_wait_seconds)
            print(json.dumps(result),flush=True)
            if result['status'] not in ('COMPLETE_VERIFIED','ALREADY_VERIFIED','STOP_ACCURACY_BELOW_D'):
                return 2


if __name__=='__main__':raise SystemExit(main())
