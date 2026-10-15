"""Bounded D -> C -> B1 single-image queue, separate from full acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from tools.sonar_deploy import board_campaign as c
from tools.sonar_deploy import admission as a
from hwnas_fpga.metrics_ledger import build_metric_record, append_metric_record


def extract_smoke(archive, destination):
    destination=Path(destination).resolve()
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or sum(m.size for m in members)>32*1024*1024:
            raise ValueError('Duplicate or oversized smoke return')
        if not {'completion.json','layers.json','power.json'}<=set(names):
            raise ValueError('Incomplete smoke return')
        for m in members:
            target=(destination/m.name).resolve()
            if not m.isfile() or not target.is_relative_to(destination) or target.exists():
                raise ValueError('Unsafe or existing smoke result')
        for m in members:
            target=destination/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tf.extractfile(m).read())


def prepare(output):
    tx = c.transport()
    root = tx.REMOTE+'/results/sonar_real_int8_deployment_v1/cpu_decode_v1_route02'
    state = c.snapshot(root, tx.REMOTE)
    if state['hardware_execution']['status'] != 'HARDWARE_IMAGE_READY_BOARD_PENDING':
        raise ValueError('Hardware readiness changed')
    a.atomic_json(output/'UPSTREAM_SNAPSHOT.json', state)
    remote = tx.REMOTE+'/results/sonar_real_int8_deployment_v1/'+output.name
    c.remote_json(c.remote_prefix(tx.REMOTE)+f"r=Path({remote!r});r.mkdir(exist_ok=False);print(json.dumps({{'status':'CREATED'}}))")
    c._scp([str(Path(__file__).with_name('board_smoke_target.py')), str(Path(__file__).with_name('power.py'))], tx.HOST+':'+remote+'/')
    specs = c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
import hashlib,re,shutil,tarfile,zipfile
from tools.sonar_deploy import admission as a
r=Path({root!r});destination=Path({remote!r});binding=a.load_binding(r)
quality=a.accuracy_admission(r,binding)
assert quality['allowed']
assert a.sha256(r/'DEPLOYMENT_BINDING.json')=={state['deployment_binding_sha256']!r}
rows=[]
for network in ['D','C','B1']:
 assert quality['gates'][network]['status'] in ('CONTROL_PASS','PASS')
 identity=next(v for v in binding['runs'] if v['network']==network and v['seed']==42)
 result=a.load_result(r,identity)
 for stage in ['integer_parity','hls','route']:a.stage_evidence(r,network,stage,result)
 build=r/network/'seed_42/hardware/graph_build';source=a.read_json(build/'manifest.json')
 for name,digest in source['files'].items():assert a.sha256(build/name)==digest,name
 assert source['test_evaluated'] is False and source['real_weights_bound']
 assert 'All user specified timing constraints are met.' in (build/'timing.rpt').read_text()
 target=destination/network;target.mkdir();payload=target/'payload';payload.mkdir()
 with zipfile.ZipFile(build/'h2_graph.xsa') as z:
  for src,dst in [('h2_graph.bit','h2_graph.bit'),('h2_graph_bd.hwh','h2_graph.hwh')]:
   assert z.namelist().count(src)==1
   (payload/dst).write_bytes(z.read(src))
 header=next((build/'hls_project/solution1/impl/ip/drivers').rglob('*_hw.h'))
 macros=dict(re.findall(r'#define\\s+XH2_GRAPH_CONTROL_ADDR_(\\w+)\\s+(0x[0-9a-fA-F]+)',header.read_text()))
 abi={{name:int(macros[key],16) for name,key in {{'arena':'ARENA_DATA','weights':'WEIGHTS_DATA','biases':'BIASES_DATA','descriptor':'DESCRIPTOR_DATA','return':'AP_RETURN'}}.items()}}
 for name in ['arena.bin','descriptors.bin','expected_layers.bin','pynq_graph_runner.py','graph.json','export_manifest.json']:
  shutil.copy2(build/name,payload/name)
 shutil.copy2(build/'manifest.json',payload/'source_manifest.json')
 clock=re.search(r'^clk_pl_0\\s+\\{{[^}}]+\\}}\\s+([\\d.]+)\\s+([\\d.]+)',(build/'timing.rpt').read_text(),re.M)
 assert clock
 board=dict(files={{v.name:a.sha256(v) for v in payload.iterdir()}},abi=abi,
  implemented_clock_mhz=float(clock[2]),executor=source['implementation'],
  graph_sha256=source['graph_sha256'],test_evaluated=False,real_weights_bound=True)
 a.atomic_json(payload/'board_manifest.json',board)
 shutil.copy2(destination/'board_smoke_target.py',payload/'board_smoke_target.py')
 shutil.copy2(destination/'power.py',payload/'smoke_power.py')
 smoke=dict(scope='single_fixed_train_image',network=network,seed=42,
  deployment_binding_sha256={state['deployment_binding_sha256']!r},
  package_manifest_sha256=result['package_manifest_sha256'],checkpoint_sha256=result['checkpoint_sha256'],
  graph_sha256=source['graph_sha256'],input_sha256=source['input_sha256'],
  files={{v.name:a.sha256(v) for v in payload.iterdir()}},test_evaluated=False,
  full_board_acceptance=False,valid_evaluated=False,previous_scope='train8_valid99_paused')
 a.atomic_json(payload/'SMOKE_BINDING.json',smoke)
 archive=target/'board_payload.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for v in payload.iterdir():tf.add(v,arcname=v.name)
 rows.append(dict(network=network,remote=str(target),package_sha256=a.sha256(archive),smoke=smoke,source=source))
http=(p/'tools/detector_preflight/http_transfer.py').read_text()
assert http.count("('192.168.31.1',18773)")==1
(destination/'http_transfer_smoke.py').write_text(http.replace("('192.168.31.1',18773)","('192.168.31.1',18774)"))
print(json.dumps(rows))
''', timeout=120)
    manifest=dict(scope='single_fixed_train_image', order=['D','C','B1'], candidates=specs,
                  remote=remote, test_evaluated=False, valid_evaluated=False)
    a.atomic_json(output/'SMOKE_MANIFEST.json',manifest)
    return manifest


def board_code(spec, board_folder):
    return f'''import urllib.request,hashlib,tarfile,io,importlib.util,threading,traceback,json
from pathlib import Path
url='http://192.168.31.1:18774'
raw=urllib.request.urlopen(url+'/board_payload.tar.gz',timeout=60).read()
assert hashlib.sha256(raw).hexdigest()=={spec['package_sha256']!r}
folder=Path({board_folder!r});folder.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=(folder/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(folder.resolve()):raise ValueError('unsafe payload')
  target.write_bytes(archive.extractfile(member).read())
spec=importlib.util.spec_from_file_location('retained_smoke_target',folder/'board_smoke_target.py')
smoke_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke_module)
stop=threading.Event()
def heartbeat():
 while not stop.wait(10):print('SINGLE_IMAGE_RUNNING',{spec['network']!r},flush=True)
worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
errors=[]
try:smoke_module.run(folder)
except BaseException as exc:errors.append(str(exc));traceback.print_exc()
finally:stop.set();worker.join()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
 for path in (folder/'smoke_results').rglob('*'):
  if path.is_file():archive.add(path,arcname=path.relative_to(folder/'smoke_results').as_posix())
req=urllib.request.Request(url+'/board_return.tar.gz',data=stream.getvalue(),method='PUT')
print('SMOKE_UPLOADED',urllib.request.urlopen(req,timeout=45).status,flush=True)
if errors:raise RuntimeError(errors[0])
'''


def verify_and_record(folder, spec):
    result=a.read_json(folder/'completion.json')
    if result['status']!='SINGLE_TRAIN_SMOKE_PASS' or result['binding']!=spec['smoke']:
        raise ValueError('Smoke identity/status mismatch')
    rows=a.read_json(folder/'layers.json')
    if len(rows)!=len(spec['source']['layers']):raise ValueError('Missing layer')
    for row,expected in zip(rows,spec['source']['layers']):
        if row['node']!=expected['node'] or row['mismatch_count'] or row['output_sha256']!=expected['output_sha256']:
            raise ValueError('Layer parity failed')
    if [h['level'] for h in result['heads']]!=[h['level'] for h in spec['source']['plan']['outputs']]:
        raise ValueError('Missing or reordered heads')
    for head in result['heads']:
        if a.sha256(folder/(head['level']+'.bin'))!=head['sha256']:raise ValueError('Head transfer differs')
        planned=next(h for h in spec['source']['plan']['outputs'] if h['level']==head['level'])
        expected=next(v for v in spec['source']['layers'] if v['node']==planned['edge'])
        if head['sha256']!=expected['output_sha256'] or head['bytes']!=planned['real_bytes']:
            raise ValueError('Head differs from integer reference')
    provenance=dict(evidence_layer='board_measurement',target_device='ZCU104',
        source=str(folder/'completion.json'),
        artifact_sha256=a.sha256(folder/'completion.json'),measurement_method='one fixed train image, layer trace enabled',
        precision='INT8',batch_size=1,input_shape=[1,3,640,640],warmup=0,repetitions=1,
        power_scope='whole_board_12v; graph including layer-check overhead',idle_subtracted=False,
        formal_benchmark=False,test_evaluated=False)
    measured={'actual_clock_mhz':dict(value=result['board_clock_mhz'],provenance=provenance),
              'board_latency_ms':dict(value=1000*result['graph_with_layer_checks_seconds'],provenance=provenance)}
    if result['power']['status']=='MEASURED_SINGLE_SMOKE':
        power=result['power']['measurements']
        import csv
        from tools.sonar_deploy.power import paired_energy
        with (folder/'power/power_samples.csv').open(newline='') as f:
            samples=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        recomputed=paired_energy(samples,samples,idle_window=result['power']['idle_window'],
            active_window=result['power']['active_window'],completed_inferences=1)
        if recomputed!=power:raise ValueError('Power integral differs from raw samples')
        measured['power_w']=dict(value=power['active_power_w'],provenance=provenance)
        measured['energy_per_inference_mj']=dict(value=power['total_energy_mj_per_image'],provenance=provenance)
    event=build_metric_record(task_type='detection',dataset='Cylinder2',network=spec['network'],seed=42,
        stage='single_train_board_smoke',identity=dict(run_id=folder.parent.parent.name+'_'+spec['network'],
            checkpoint_sha256=spec['smoke']['checkpoint_sha256'],test_evaluated=False,
            clean_reference_available=False),measured=measured)
    append_metric_record(folder/'METRICS.jsonl',event)
    a.atomic_json(folder/'VERIFIED.json',dict(status='SINGLE_TRAIN_SMOKE_PASS',metric_event_id=event['event_id'],
        files={p.relative_to(folder).as_posix():a.sha256(p) for p in folder.rglob('*') if p.is_file()},
        full_board_acceptance=False,test_evaluated=False))


def run(output):
    output.mkdir(exist_ok=False,parents=True)
    tx=c.transport()
    try:
        a.atomic_json(output/'STATUS.json',dict(status='PREPARING',test_evaluated=False))
        manifest=prepare(output)
        for spec in manifest['candidates']:
            folder=output/spec['network'];folder.mkdir()
            owner=c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
import subprocess
r=Path({spec['remote']!r});log=(r/'http.log').open('x')
v=subprocess.Popen(['python3',{manifest['remote']!r}+'/http_transfer_smoke.py',str(r),'--max-return-bytes','33554432','--max-seconds','3600'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=v.pid)))
''')
            script=folder/'execute.py'
            script.write_text(board_code(spec,'/home/xilinx/'+output.name+'_'+spec['network']),encoding='utf-8')
            a.atomic_json(output/'STATUS.json',dict(status='SINGLE_IMAGE_RUNNING',network=spec['network'],
                http_pid=owner['pid'],test_evaluated=False))
            try:
                c._local_command([sys.executable,str(c.BRIDGE),str(script),str(folder/'board.log'),'--keep-kernel'],folder/'transport.log')
                archive=folder/'board_return.tar.gz'
                c._scp([tx.HOST+':'+spec['remote']+'/board_return.tar.gz'],archive)
                evidence=folder/'evidence';evidence.mkdir()
                extract_smoke(archive,evidence)
                verify_and_record(evidence,spec)
            finally:
                c.remote_json(c.remote_prefix(tx.REMOTE)+f'''
import signal
pid={owner['pid']};proc=Path('/proc')/str(pid)/'cmdline'
if proc.exists():
 args=proc.read_bytes().split(bytes([0]))
 if {spec['remote']!r}.encode() in args and any(v.endswith(b'http_transfer_smoke.py') for v in args):os.kill(pid,signal.SIGTERM)
print(json.dumps(dict(status='OWNED_TRANSFER_STOPPED')))
''')
        a.atomic_json(output/'STATUS.json',dict(status='THREE_SINGLE_IMAGE_SMOKES_COMPLETE',
            networks=['D','C','B1'],full_board_acceptance=False,test_evaluated=False))
    except BaseException as exc:
        a.atomic_json(output/'STATUS.json',dict(status='STOP_FAILED_NO_AUTORETRY',error=str(exc),test_evaluated=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve())
