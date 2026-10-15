"""Isolated B1-neck/head seed42 campaign using the existing owned server."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
# The imported transport retrieves credentials from local Credential Manager.
from tools.detector_v0_2.remote import execute, SSH, SCP, ENV, HOST, PYTHON, SOURCE

REMOTE = '/home/jiajia/experiments/detector_b1_light_20261003'
RESULT = 'results/detector_b1_light_20261003/remote'
LOCAL = ROOT / 'results/detector_b1_light_20261003/transport'
PROTOTYPES = ('L1', 'L0')


def expand(code):
    return (code.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')')
        .replace('root/RESULT', 'root/' + repr(RESULT))
        .replace('Path(SOURCE)', 'Path(' + repr(SOURCE) + ')')
        .replace('[PYTHON,', '[' + repr(PYTHON) + ',')
        .replace("'--result-root',RESULT", "'--result-root'," + repr(RESULT)))


def preflight():
    code = '''from pathlib import Path
import json,subprocess
root=Path(REMOTE)
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if 'tools/detector_b1_light/' in cmd or 'tools/detector_v0_2/queue_runner.py' in cmd:
   processes.append({'pid':int(p.name),'command':cmd,'cwd':str((p/'cwd').resolve())})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({'root_exists':root.exists(),'root':str(root),'gpu_csv':gpus,'processes':processes}))
'''
    return json.loads(execute(expand(code), 40))


def stage():
    from tools.detector_b1_light import train
    cfg, baseline = train.load_config()
    paths = {ROOT / name for name in train.source_hashes(cfg)}
    paths.add(Path(__file__))
    paths.add(ROOT / 'tools/detector_v0_2/remote.py')
    paths.add(ROOT / 'tools/fpga_v0_1/remote_bridge.py')
    paths.add(ROOT / baseline['dataset']['data_dir'] / 'data.yaml')
    LOCAL.mkdir(parents=True, exist_ok=True)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    marker = dict(version='B1_LIGHT_SOURCE_V1', source_sha256=hashes,
                  test_mounted=False, legacy_trained_weights_copied=False)
    (LOCAL / 'staging_manifest.json').write_text(json.dumps(marker, indent=2), encoding='utf-8')
    archive = LOCAL / 'training_source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(paths):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json')
    execute('from pathlib import Path;Path(' + repr(REMOTE) + ').mkdir(parents=True,exist_ok=True)')
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/training_source.tar.gz'], env=ENV, check=True)
    code = '''from pathlib import Path
import tarfile,json,hashlib,yaml
root=Path(REMOTE)
if (root/RESULT/'execution_manifest.json').exists():raise RuntimeError('Frozen campaign exists; source staging refused')
if (root/RESULT/'queue_status.json').exists():raise RuntimeError('Invoked campaign exists; source staging refused')
previous=root/'staging_manifest.json'
archived=root/'results/detector_b1_light_20261003/preflight_failures/transport01/staging_manifest.json'
preserved_failure=previous.is_file() and archived.is_file() and previous.read_bytes()==archived.read_bytes()
with tarfile.open(root/'training_source.tar.gz') as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()):raise ValueError('Unsafe archive member')
  data=tar.extractfile(member).read()
  if previous.exists() and target.exists() and target.read_bytes()!=data and not preserved_failure:
   raise ValueError('Existing staged source differs; preserve preflight attempt: '+member.name)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
marker=json.loads((root/'staging_manifest.json').read_bytes())
for name,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source transport SHA mismatch: '+name)
b=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/b['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ['train','valid']:
 link=data/split;target=Path(SOURCE)/b['dataset']['data_dir']/split
 if not target.is_dir():raise ValueError('Required original split absent')
 if link.exists() or link.is_symlink():
  if not link.is_symlink() or link.resolve()!=target.resolve():raise ValueError('Split identity changed')
 else:link.symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test must not be mounted')
print(json.dumps({'status':'STAGED','source_files':len(marker['source_sha256']),'train_valid_only':True,'legacy_weights_copied':False}))
'''
    return json.loads(execute(expand(code), 60))


def invoke(mode, prototype='L1', timeout=180):
    command = [PYTHON, 'tools/detector_b1_light/train.py', mode, '--result-root', RESULT,
               '--prototype', prototype, '--seed', '42', '--device', 'cuda']
    code = ('import subprocess,os\nos.environ["CUDA_VISIBLE_DEVICES"]="2"\n'
            'subprocess.run(' + repr(command) + ',cwd=' + repr(REMOTE) + ',check=True)\n')
    return execute(code, timeout)


def launch():
    code = '''from pathlib import Path
import subprocess,os,json,hashlib
root=Path(REMOTE);result=root/RESULT
if (result/'queue_status.json').exists() or (result/'launch.json').exists():raise FileExistsError('Existing queue launch refused')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().decode(errors='replace')
  if (p/'cwd').resolve()==root and ('queue_runner.py' in cmd or 'tools/detector_b1_light/train.py' in cmd):
   raise RuntimeError('Existing campaign process')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
smokes=[];identities=[];training_rng=[]
for proto in ['L1','L0']:
 p=result/'smoke'/proto/'seed_42'
 report=json.loads((p/'smoke_report.json').read_bytes())
 if report['status']!='PASS' or report['binding']['execution_manifest_sha256']!=manifest_sha or report['formal_eligible'] is not False:
  raise ValueError('Actual two-batch smoke and source binding required')
 smokes.append(report)
 identities.append(json.loads((p/'initialization_identity.json').read_bytes()))
 training_rng.append(json.loads((p/'training_rng_identity.json').read_bytes()))
for key in ['backbone_sha256','neck_sha256','prediction_sha256']:
 if not identities[0].get(key) or identities[0][key]!=identities[1].get(key):raise ValueError('Shared initialization differs: '+key)
if training_rng[0]!=training_rng[1]:raise ValueError('Paired training RNG differs')
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpu=next([int(x.strip()) for x in line.split(',')] for line in lines.splitlines() if int(line.split(',')[0])==2)
required=max(r['peak_cuda_reserved_bytes'] for r in smokes)/1048576+1024
if gpu[2]-gpu[1]<required or gpu[3]>10:raise RuntimeError('GPU2 availability changed')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',PYTHONUNBUFFERED='1')
with (result/'queue_stdout.log').open('a') as log:
 child=subprocess.Popen([PYTHON,'tools/detector_b1_light/queue_runner.py','--device','cuda','--result-root',RESULT],
   cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
payload={'status':'LAUNCHED','pid':child.pid,'gpu':2,'order':['L1','L0'],'seed':42,'epochs_each':80,
 'execution_manifest_sha256':manifest_sha,'shared_initialization':{k:identities[0][k] for k in ['backbone_sha256','neck_sha256','prediction_sha256']},
 'shared_training_rng':training_rng[0],
 'gpu_prelaunch':{'used_mib':gpu[1],'total_mib':gpu[2],'utilization_percent':gpu[3],'required_free_mib':required},
 'test_evaluated':False,'automatic_hardware_or_nas_promotion':False}
(result/'launch.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload))
'''
    return json.loads(execute(expand(code), 40))


def status():
    code = '''from pathlib import Path
import json
root=Path(REMOTE);result=root/RESULT
payload={}
paths=['launch.json','queue_status.json']+[f'formal/{p}/seed_42/status.json' for p in ['L1','L0']]
for name in paths:
 p=result/name
 if p.exists():payload[name]=json.loads(p.read_bytes())
for name in ['queue_stdout.log']+[f'formal/{p}/seed_42/stdout.log' for p in ['L1','L0']]:
 p=result/name
 if p.exists():payload[name]=p.read_text()[-1200:]
payload['processes']=[]
queue=payload.get('queue_status.json',{})
for pid in [queue.get('pid'),queue.get('child_pid')]:
 if not pid:continue
 p=Path('/proc')/str(pid)
 try:payload['processes'].append({'pid':pid,'alive':p.exists(),'cwd':str((p/'cwd').resolve()),'command':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')})
 except (FileNotFoundError,PermissionError,ProcessLookupError):payload['processes'].append({'pid':pid,'alive':False})
print(json.dumps(payload))
'''
    value = json.loads(execute(expand(code), 40))
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    return value


def evidence():
    code = '''from pathlib import Path
import json,base64,hashlib
root=Path(REMOTE);result=root/RESULT
paths=[result/name for name in ['execution_manifest.json','launch.json','queue_status.json']]
for proto in ['L1','L0']:
 for mode in ['smoke','formal']:
  p=result/mode/proto/'seed_42'
  paths.extend(p/name for name in ['smoke_report.json','initialization_identity.json','training_rng_identity.json','model_info.json','run_binding.json','status.json','epoch_log.json','result.json'])
out={}
for p in paths:
 if p.is_file():
  data=p.read_bytes();out[str(p.relative_to(result))]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps(out))
'''
    records = json.loads(execute(expand(code), 60))
    destination = ROOT / 'results/detector_b1_light_20261003/remote_snapshot/raw_bytes'
    for name, record in records.items():
        raw = base64.b64decode(record['base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Transport byte SHA mismatch')
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError('Unexpected remote evidence path')
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'execution_manifest.json' and target.exists() and target.read_bytes() != raw:
            raise ValueError('Frozen manifest raw bytes changed')
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_bytes(raw); temporary.replace(target)
    return dict(status='FETCHED', files=len(records), destination=str(destination),
                raw_sha256={name: row['sha256'] for name, row in records.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight','stage','prepare','verify','smoke','launch','status','evidence'])
    p.add_argument('--prototype', choices=PROTOTYPES, default='L1')
    args = p.parse_args()
    if args.mode in ('prepare','verify','smoke'):
        print(invoke('preflight' if args.mode == 'verify' else args.mode, args.prototype))
    else:
        print(json.dumps(globals()[args.mode](), indent=2))
