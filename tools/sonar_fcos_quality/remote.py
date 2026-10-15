"""Isolated transport for the authorized six-run FCOS quality side experiment."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from tools.sonar_fcos_pilot.common import ROOT, read_json, sha256_file, utc_now, write_json

REMOTE = '/home/jiajia/experiments/sonar_fcos_quality_20261012_01'
PARENT = '/home/jiajia/experiments/sonar_fcos_pilot_20261008'
BASELINE = PARENT + '_recovery_v1/configs/sonar_fcos_pilot_recovery_v1.json'
PYTHON = '/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python'
CONFIG = 'configs/sonar_fcos_quality_20261012.json'
RESULT = 'results/sonar_fcos_quality_20261012_01'
LOCAL = ROOT / RESULT / 'transport'
EXPECTED_BASELINE_SHA = 'c1bbeb42ac23ae8be243a1974194acac746504776a6fa19e869759df243aeb8b'


def transport():
    # Existing Credential Manager askpass transport; credentials never enter artifacts.
    from tools.detector_v0_2.remote import execute, SCP, ENV, HOST
    return execute, SCP, ENV, HOST


def execute_json(code, timeout=60):
    return json.loads(transport()[0](code, timeout))


def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r}); result=root/{RESULT!r}; config=root/{CONFIG!r}\n'
            f'python={PYTHON!r}; parent=Path({PARENT!r}); overlay=parent/"vendor"\n'
            'def digest(path):\n h=hashlib.sha256()\n'
            ' with Path(path).open("rb") as f:\n'
            '  for b in iter(lambda:f.read(1048576),b""):h.update(b)\n'
            ' return h.hexdigest()\n')


def environment_spec():
    return {'schema_version': 1, 'provider': 'existing_user_owned_server',
            'python_executable': PYTHON, 'python_version': '3.13.11',
            'overlay': PARENT + '/vendor', 'shared_environment_modified': False,
            'packages': {'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128',
                         'numpy': '2.4.4', 'Pillow': '11.3.0', 'pycocotools': '2.0.11'},
            'cuda_runtime': '12.8', 'gpu': 'NVIDIA GeForce RTX 5090',
            'gpu_indices': [0], 'reserved_mainline_gpus': [1], 'torch_threads': 2,
            'baseline_config_sha256': EXPECTED_BASELINE_SHA,
            'weight_cache': PARENT + '/weights/torch',
            'smoke': 'real train image forward/backward/update and validation inference for AAA and QFL_DFL',
            'scope': 'finite six-run queue, 30 epochs each, development validation only'}


def source_files():
    paths = [ROOT / CONFIG]
    for name in ('sonar_fcos_pilot', 'sonar_fcos_quality'):
        paths += [p for p in (ROOT / 'tools' / name).iterdir() if p.is_file() and p.suffix in ('.py', '.md')]
    paths += list((ROOT / 'tests').glob('test_sonar_fcos_quality*.py'))
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing or symlinked source: ' + str(path))
    return sorted(set(paths))


def stage():
    paths = source_files()
    LOCAL.mkdir(parents=True, exist_ok=True)
    spec = environment_spec()
    write_json(LOCAL / 'environment_spec.json', spec)
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    canonical = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    marker = {'source_sha256': hashes, 'source_set_sha256': canonical(hashes),
              'environment_spec_sha256': canonical(spec), 'training_started': False,
              'existing_data_read_only': True, 'legacy_results_modified': False}
    write_json(LOCAL / 'source_freeze.json', marker)
    archive = LOCAL / ('source_' + marker['source_set_sha256'][:16] + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        for path in paths:
            stream.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        for name in ('source_freeze.json', 'environment_spec.json'):
            stream.add(LOCAL / name, arcname=name, recursive=False)
    execute_json(header() + '''
if root.exists():
 if not (root/'source_freeze.json').exists():raise FileExistsError('Existing unowned namespace')
else:root.mkdir()
(root/'staging').mkdir(exist_ok=True)
print(json.dumps({'status':'DIRECTORY_READY'}))
''')
    _, scp, env, host = transport()
    subprocess.run(scp + [str(archive), host + ':' + REMOTE + '/staging/' + archive.name],
                   env=env, check=True, capture_output=True, timeout=120)
    code = header() + f'archive=root/"staging"/{archive.name!r}\nexpected={sha256_file(archive)!r}\n' + '''
import tarfile,shutil
if shutil.disk_usage(root).free<30*1024**3:raise RuntimeError('Less than 30 GiB free; do not start six runs')
if digest(archive)!=expected:raise ValueError('Transport archive hash mismatch')
payload={}
with tarfile.open(archive,'r:gz') as stream:
 for member in stream.getmembers():
  p=Path(member.name); destination=(root/p).resolve()
  if not member.isfile() or p.is_absolute() or '..' in p.parts or not destination.is_relative_to(root.resolve()):raise ValueError('Unsafe archive member')
  if member.name in payload:raise ValueError('Duplicate archive member')
  payload[member.name]=stream.extractfile(member).read()
marker=json.loads(payload['source_freeze.json']); spec=json.loads(payload['environment_spec.json'])
canonical=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if canonical(marker['source_sha256'])!=marker['source_set_sha256'] or canonical(spec)!=marker['environment_spec_sha256']:raise ValueError('Source/spec freeze mismatch')
if set(payload)!=set(marker['source_sha256'])|{'source_freeze.json','environment_spec.json'}:raise ValueError('Archive member list mismatch')
for rel,sha in marker['source_sha256'].items():
 if hashlib.sha256(payload[rel]).hexdigest()!=sha:raise ValueError('Source payload mismatch')
parity={}
recovery=Path(str(parent)+'_recovery_v1')
for name in ('model.py','data.py','metrics.py','numerics.py','train.py','common.py'):
 rel='tools/sonar_fcos_pilot/'+name
 observed=digest(recovery/rel)
 if observed!=marker['source_sha256'][rel]:raise ValueError('Baseline helper differs from frozen recovery: '+name)
 parity[name]={'sha256':observed,'recovery_path':str(recovery/rel),'equal':True}
if (root/'source_freeze.json').exists():
 if json.loads((root/'source_freeze.json').read_text())!=marker:raise RuntimeError('Staged source differs; preserve existing evidence and use reviewed restaging')
 for rel,sha in marker['source_sha256'].items():
  if digest(root/rel)!=sha:raise RuntimeError('Live staged source changed')
else:
 for rel,data in payload.items():
  path=root/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
if not overlay.is_dir():raise FileNotFoundError('Original FCOS overlay missing')
if not Path(spec['python_executable']).is_file():raise FileNotFoundError('Frozen interpreter absent')
cfg=json.loads(config.read_text())
if digest(Path(cfg['baseline_config']))!=spec['baseline_config_sha256']:raise ValueError('Original baseline config changed')
result.mkdir(parents=True,exist_ok=True)
receipt={'status':'STAGED','source_set_sha256':marker['source_set_sha256'],'source_freeze_sha256':digest(root/'source_freeze.json'),'archive_sha256':expected,'existing_data_read_only':True,'shared_environment_modified':False,'baseline_source_parity':parity,'disk_free_bytes':shutil.disk_usage(root).free}
(result/'stage.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    value = execute_json(code)
    write_json(LOCAL / 'stage.json', value)
    return value


def probe():
    code = header() + f'baseline=Path({BASELINE!r})\n' + '''
cfg=json.loads(baseline.read_text());manifest=Path(cfg['manifest_path'])
print(json.dumps({'root_exists':root.exists(),'baseline_config_sha256':digest(baseline),'manifest_path':str(manifest),'manifest_sha256':digest(manifest),'overlay_exists':overlay.is_dir(),'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))
'''
    value = execute_json(code)
    write_json(LOCAL / 'probe.json', value)
    return value


def preflight():
    code = header() + '''
sys.path.insert(0,str(root))
from tools.sonar_fcos_quality.scheduler import gpu_snapshot,select_idle_gpu
marker=json.loads((root/'source_freeze.json').read_text())
for rel,sha in marker['source_sha256'].items():
 if digest(root/rel)!=sha:raise ValueError('Source changed: '+rel)
if (result/'preflight_launch.json').exists():raise FileExistsError('Preflight was already invoked')
card=select_idle_gpu(gpu_snapshot(),{'gpu_indices':[0]})
if card is None:raise RuntimeError('GPU0 is occupied; mainline GPU1 is reserved')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(overlay),str(root)]),CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
commands=[[python,'-m','pytest','-q',*[str(p.relative_to(root)) for p in sorted((root/'tests').glob('test_sonar_fcos_quality*.py'))]],
 [python,'-m','tools.sonar_fcos_quality.train','prepare','--config',str(config)],
 [python,'-m','tools.sonar_fcos_quality.train','smoke','--config',str(config)]]
supervisor=r"""
import importlib.metadata,json,pathlib,platform,subprocess,hashlib
import torch,pycocotools
root=pathlib.Path(ROOT); result=root/RESULT
spec=json.loads((root/'environment_spec.json').read_text())
packages={name:importlib.metadata.version(name) for name in spec['packages']}
assert packages==spec['packages'],('Environment packages changed',packages)
assert platform.python_version()==spec['python_version']
assert torch.version.cuda==spec['cuda_runtime']
assert torch.cuda.get_device_name(0)==spec['gpu']
assert pathlib.Path(pycocotools.__file__).resolve().is_relative_to(pathlib.Path(spec['overlay']).resolve())
lock={'status':'PASS','packages':packages,'python':platform.python_version(),'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'physical_gpu':0,'pycocotools_path':pycocotools.__file__,'shared_environment_modified':False}
(result/'environment_lock.json').write_text(json.dumps(lock,indent=2)+'\\n')
for command in COMMANDS:subprocess.run(command,check=True,timeout=1800)
manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
value={'status':'PASS','execution_manifest_sha256':manifest_sha,'commands':COMMANDS,'environment_lock_sha256':hashlib.sha256((result/'environment_lock.json').read_bytes()).hexdigest()}
(result/'preflight_complete.json').write_text(json.dumps(value,indent=2)+'\\n')
""".replace('ROOT',repr(str(root))).replace('RESULT',repr(str(result.relative_to(root)))).replace('COMMANDS',repr(commands))
with (result/'preflight_stdout.log').open('x') as log:
 child=subprocess.Popen([python,'-c',supervisor],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'status':'PREFLIGHT_LAUNCHED','pid':child.pid,'physical_gpu':0,'commands':commands}
(result/'preflight_launch.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
    value = execute_json(code)
    write_json(LOCAL / 'preflight_launch.json', value)
    return value


def upload_review(path=None):
    path = Path(path) if path else ROOT / RESULT / 'independent_review.json'
    payload = path.read_bytes()
    if json.loads(payload).get('status') != 'PASS':
        raise ValueError('Independent review is not PASS')
    code = header() + f'payload={base64.b64encode(payload).decode()!r}\n' + '''
import base64
raw=base64.b64decode(payload); review=json.loads(raw)
if review['status']!='PASS' or review['execution_manifest_sha256']!=digest(result/'execution_manifest.json'):raise ValueError('Independent review binding differs')
path=result/'independent_review.json'
if path.exists() and path.read_bytes()!=raw:raise FileExistsError('Existing independent review differs')
path.write_bytes(raw)
print(json.dumps({'status':'REVIEW_UPLOADED','sha256':digest(path)}))
'''
    return execute_json(code)


def launch(resume=False):
    code = header() + f'resume={resume!r}\n' + '''
import shutil
if shutil.disk_usage(root).free<30*1024**3:raise RuntimeError('Less than 30 GiB free; launch refused')
sys.path.insert(0,str(root))
from tools.sonar_fcos_quality.scheduler import singleton
marker=json.loads((root/'source_freeze.json').read_text())
for rel,sha in marker['source_sha256'].items():
 if digest(root/rel)!=sha:raise ValueError('Frozen source changed: '+rel)
manifest_sha=digest(result/'execution_manifest.json')
for name in ('preflight_complete.json','smoke_gate.json','independent_review.json'):
 gate=json.loads((result/name).read_text())
 if gate.get('status')!='PASS' or gate.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Gate/binding not passed: '+name)
if (result/'launch.json').exists() and not resume:raise FileExistsError('Already launched; explicit --resume required after a stopped queue')
if resume and not (result/'launch.json').exists():raise ValueError('Nothing to resume')
if (result/'queue_status.json').exists() and json.loads((result/'queue_status.json').read_text()).get('status')=='COMPLETE':raise ValueError('The finite six-run queue is already complete')
# Check both the advisory lock and matching current command. The child locks itself.
with singleton(result/'scheduler'/'queue.lock'):
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
  try:
   argv=(entry/'cmdline').read_bytes().split(bytes([0]))
   if (entry/'cwd').resolve()==root and b'tools.sonar_fcos_quality.scheduler' in argv:raise RuntimeError('Existing side queue process')
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(overlay),str(root)]),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[python,'-m','tools.sonar_fcos_quality.scheduler','--config',str(config)]
stamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (result/('queue_'+stamp+'.log')).open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'status':'LAUNCHED','pid':child.pid,'command':command,'execution_manifest_sha256':manifest_sha,'finite_runs':6,'reserved_mainline_gpu':1,'eligible_gpu_indices':[0],'resume':resume}
path=result/('resume_'+stamp+'.json' if resume else 'launch.json')
path.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
    value = execute_json(code)
    write_json(LOCAL / ('resume_' + utc_now().replace(':', '') + '.json' if resume else 'launch.json'), value)
    return value


def status():
    code = header() + '''
value={'root_exists':root.exists()}
for name in ('stage.json','preflight_launch.json','preflight_complete.json','environment_lock.json','smoke_gate.json','execution_manifest.json','independent_review.json','launch.json','queue_status.json','branch_results.json'):
 path=result/name
 if path.is_file():value[name]=json.loads(path.read_text())
for path in (result/'runs').glob('*/status.json'):
 value[path.relative_to(result).as_posix()]=json.loads(path.read_text())
for key in ('preflight_launch.json','launch.json','queue_status.json'):
 if key in value and 'pid' in value[key]:value[key]['pid_exists']=Path('/proc',str(value[key]['pid'])).exists()
if (result/'preflight_stdout.log').is_file():value['preflight_log_tail']=(result/'preflight_stdout.log').read_text()[-5000:]
value['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(value))
'''
    value = execute_json(code)
    write_json(LOCAL / 'status.json', value)
    return value


def collect():
    code = header() + '''
import base64
files=[p for p in result.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.log') and p.stat().st_size<16000000]
if sum(p.stat().st_size for p in files)>150000000:raise RuntimeError('Evidence unexpectedly large; inspect before collection')
print(json.dumps({p.relative_to(result).as_posix():{'base64':base64.b64encode(p.read_bytes()).decode(),'sha256':digest(p)} for p in files}))
'''
    files = execute_json(code, timeout=120)
    destination = ROOT / RESULT / 'remote_snapshot'
    for name, item in files.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError('Evidence path escapes campaign')
        content = base64.b64decode(item['base64'])
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Evidence content hash mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {'status': 'COLLECTED', 'files': len(files), 'local': str(destination)}


def fetch_weights():
    value = execute_json(header() + '''
path=result/'pretrained'/'pretrained_manifest.json'; manifest=json.loads(path.read_text())
weight=Path(manifest['checkpoint']).resolve()
if not weight.is_relative_to((result/'pretrained').resolve()) or digest(weight)!=manifest['sha256']:raise ValueError('Exported checkpoint binding invalid')
print(json.dumps(manifest))
''')
    destination = Path('F:/sonar_fcos_quality/pretrained/20261012_01')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'uatd_fcos_best.pt'
    if path.exists():
        if sha256_file(path) != value['sha256']:
            raise FileExistsError('Local exported weight differs')
    else:
        partial = destination / 'uatd_fcos_best.pt.download'
        _, scp, env, host = transport()
        subprocess.run(scp + [host + ':' + value['checkpoint'], str(partial)], env=env,
                       check=True, capture_output=True, timeout=600)
        if sha256_file(partial) != value['sha256']:
            raise ValueError('Downloaded checkpoint hash differs')
        partial.replace(path)
    write_json(destination / 'pretrained_manifest.json', {**value, 'local_checkpoint': str(path)})
    return {'status': 'WEIGHT_FETCHED', 'path': str(path), 'sha256': value['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'stage', 'preflight', 'upload-review', 'launch', 'status', 'collect', 'fetch-weights'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--review')
    args = parser.parse_args()
    if args.mode == 'launch':
        value = launch(args.resume)
    elif args.mode == 'upload-review':
        value = upload_review(args.review)
    else:
        value = globals()[args.mode.replace('-', '_')]()
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
