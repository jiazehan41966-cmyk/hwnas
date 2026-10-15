"""Existing Credential Manager transport; isolated train/valid-only staging."""
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
REMOTE = '/home/jiajia/experiments/detector_v0_2_execution_20261002'
PYTHON = '/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python'
SOURCE = '/home/jiajia/experiments/sonar_phase2c_20260914_v1'
RESULT = 'results/detector_v0_2_execution_20261002/remote'
LOCAL = ROOT/'results/detector_v0_2_execution_20261002/transport'
sys.path.insert(0, str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import SSH, SCP, ENV, HOST


def execute(code: str, timeout: int = 60):
    process = subprocess.run(SSH+[PYTHON, '-'], input=code.encode(), env=ENV,
        capture_output=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f'SSH exited {process.returncode}: '+process.stderr.decode(errors='replace')[-6000:])
    return process.stdout.decode()


def expand(code):
    return (code.replace('Path(REMOTE)', 'Path('+repr(REMOTE)+')')
        .replace('root/RESULT', 'root/'+repr(RESULT))
        .replace('Path(SOURCE)', 'Path('+repr(SOURCE)+')')
        .replace('[PYTHON,', '['+repr(PYTHON)+',')
        .replace("'--result-root',RESULT", "'--result-root',"+repr(RESULT)))


def preflight():
    code = r'''
import json, subprocess
from pathlib import Path
root=Path(REMOTE)
print(json.dumps({'root_exists':root.exists(),'entries':[x.name for x in root.iterdir()] if root.exists() else []}))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True))
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'tools/detector_v0_2/queue_runner.py' in command or 'tools/detector_v0_2/train.py' in command:processes.append({'pid':int(p.name),'command':command})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({'detector_processes':processes}))
'''
    return execute(expand(code), 40)


def stage():
    # Never copy legacy weights, results, data or .git.
    frozen=json.loads((ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json').read_text(encoding='utf-8'))
    paths={ROOT/k.replace('\\','/') for k in frozen['source_sha256']}
    paths.update((ROOT/'src/hwnas_fpga').rglob('*.py'))
    for directory in ('tools/sonar_spec','tools/sonar_analysis'):
        paths.update((ROOT/directory).glob('*.py'))
    paths.update(ROOT/p for p in ('tools/detector_v0_2/train.py','tools/detector_v0_2/queue_runner.py',
        'configs/detector_v0_2_training.yaml','configs/sonar_detector_v0_2.yaml',
        'configs/baseline_sonar_frozen.yaml','configs/search_space_sonar_v0_1.yaml',
        'configs/supernet_sonar_v0_1_formal.yaml','reports/baseline_sonar_frozen/freeze_manifest.json',
        'reports/phase2f_multiseed/multiseed_results.csv'))
    baseline = __import__('yaml').safe_load((ROOT/'configs/baseline_sonar_frozen.yaml').read_text(encoding='utf-8'))
    paths.add(ROOT/baseline['dataset']['data_dir']/'data.yaml')
    LOCAL.mkdir(parents=True,exist_ok=True)
    archive=LOCAL/'training_source.tar.gz'
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    marker={'version':'V02_TRAINING_SOURCE_V1','source_sha256':hashes,
        'test_mounted':False,'legacy_weights_copied':False}
    (LOCAL/'staging_manifest.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
        tar.add(LOCAL/'staging_manifest.json',arcname='staging_manifest.json')
    execute('from pathlib import Path;Path('+repr(REMOTE)+').mkdir(parents=True,exist_ok=True)')
    subprocess.run(SCP+[str(archive),HOST+':'+REMOTE+'/training_source.tar.gz'],env=ENV,check=True)
    code=r'''
from pathlib import Path
import tarfile,json,hashlib,yaml
root=Path(REMOTE)
for p in [root/RESULT/'execution_manifest.json',root/RESULT/'queue_status.json']:
 if p.exists():raise RuntimeError('Frozen campaign already exists; staging refused')
with tarfile.open(root/'training_source.tar.gz') as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root):raise ValueError('Unsafe source archive member')
  data=tar.extractfile(member).read()
  if target.exists() and target.read_bytes()!=data and (root/'staging_manifest.json').exists():raise ValueError('Existing staged source differs: '+member.name)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Transport hash mismatch: '+rel)
baseline=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/baseline['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
source=Path(SOURCE)/baseline['dataset']['data_dir']
for split in ['train','valid']:
 expected=(source/split).resolve(strict=True);link=data/split
 if link.exists():
  if link.resolve()!=expected:raise ValueError('Existing data link differs')
 else:link.symlink_to(expected,target_is_directory=True)
if (data/'test').exists():raise ValueError('Test mount detected')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False,'source_data':str(source)}))
'''
    return execute(expand(code),60)


def invoke(mode: str, prototype: str = 'R1', timeout: int = 60):
    if mode not in ('prepare','preflight','smoke'):raise ValueError('Unsupported mode')
    command=[PYTHON,'tools/detector_v0_2/train.py',mode,'--result-root',RESULT,'--device','cuda','--prototype',prototype,'--seed','42']
    return execute('import subprocess,os\nos.environ["CUDA_VISIBLE_DEVICES"]="2"\nsubprocess.run('+repr(command)+',cwd='+repr(REMOTE)+',check=True)\n',timeout)


def launch():
    code=r'''
from pathlib import Path
import subprocess,os,json,hashlib
root=Path(REMOTE);result=root/RESULT
if (result/'queue_status.json').exists():raise FileExistsError('Queue already invoked; launch refused')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  command=(p/'cmdline').read_bytes().decode(errors='replace')
  cwd=(p/'cwd').resolve()
  if cwd==root and ('queue_runner.py' in command or 'tools/detector_v0_2/train.py' in command):raise RuntimeError('Existing detector process; launch refused')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
expected_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
smokes=[]
for prototype in ['R1','R0','R8','R7']:
 p=result/'smoke'/prototype/'seed_42'/'smoke_report.json'
 if not p.exists():raise RuntimeError('All four smoke passes required')
 report=json.loads(p.read_text())
 if report['status']!='PASS' or report['binding']['execution_manifest_sha256']!=expected_sha or report['formal_eligible'] is not False:raise RuntimeError('Smoke status/identity invalid')
 smokes.append(report)
gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpu=next([int(x.strip()) for x in line.split(',')] for line in gpu_lines.splitlines() if int(line.split(',')[0])==2)
required_mib=max(r['peak_cuda_reserved_bytes'] for r in smokes)/1048576+1024
if gpu[2]-gpu[1]<required_mib or gpu[3]>10:raise RuntimeError('GPU 2 changed after smoke; launch refused')
result.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',PYTHONUNBUFFERED='1')
with (result/'queue_stdout.log').open('a') as log:
 process=subprocess.Popen([PYTHON,'tools/detector_v0_2/queue_runner.py','--device','cuda','--result-root',RESULT],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
payload={'status':'LAUNCHED','pid':process.pid,'gpu':2,'order':['R1','R0','R8','R7'],'seed':42,'test_evaluated':False,'gpu_prelaunch':{'used_mib':gpu[1],'total_mib':gpu[2],'utilization_percent':gpu[3],'required_free_mib':required_mib},'execution_manifest_sha256':expected_sha}
(result/'launch.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload))
'''
    return execute(expand(code),40)


def fetch_status():
    code=r'''
from pathlib import Path
import json
root=Path(REMOTE);result=root/RESULT
payload={}
for rel in ['launch.json','queue_status.json','formal/R1/seed_42/status.json','formal/R0/seed_42/status.json','formal/R8/seed_42/status.json','formal/R7/seed_42/status.json']:
 p=result/rel
 if p.exists():payload[rel]=json.loads(p.read_text())
for rel in ['queue_stdout.log','formal/R1/seed_42/stdout.log','formal/R0/seed_42/stdout.log','formal/R8/seed_42/stdout.log','formal/R7/seed_42/stdout.log']:
 p=result/rel
 if p.exists():payload[rel]=p.read_text()[-4000:]
payload['processes']=[]
for name in ['queue_status.json']:
 p=result/name
 if not p.exists():continue
 status=json.loads(p.read_text())
 for key in ['pid','child_pid']:
  pid=status.get(key)
  if not pid:continue
  proc=Path('/proc')/str(pid)
  try:
   payload['processes'].append({'pid':pid,'alive':proc.exists(),'cwd':str((proc/'cwd').resolve()),'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')})
  except (FileNotFoundError,PermissionError,ProcessLookupError):payload['processes'].append({'pid':pid,'alive':False})
print(json.dumps(payload))
'''
    value=json.loads(execute(expand(code),40))
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'remote_status.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value


def fetch_evidence():
    code=r'''
from pathlib import Path
import json,base64,hashlib
root=Path(REMOTE);result=root/RESULT
paths=[result/'execution_manifest.json',result/'launch.json',result/'queue_status.json']
for prototype in ['R1','R0','R8','R7']:
 paths.extend([result/'smoke'/prototype/'seed_42'/'smoke_report.json',
  result/'smoke'/prototype/'seed_42'/'model_info.json',
  result/'formal'/prototype/'seed_42'/'run_binding.json',
  result/'formal'/prototype/'seed_42'/'model_info.json',
  result/'formal'/prototype/'seed_42'/'status.json',
  result/'formal'/prototype/'seed_42'/'epoch_log.json',
  result/'formal'/prototype/'seed_42'/'result.json'])
values={}
for p in paths:
 if p.exists():
  data=p.read_bytes()
  values[str(p.relative_to(result))]={'raw_base64':base64.b64encode(data).decode(),
   'sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps(values))
'''
    values=json.loads(execute(expand(code),40))
    destination=ROOT/'results/detector_v0_2_execution_20261002/remote_snapshot'
    for rel,record in values.items():
        raw=base64.b64decode(record['raw_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Remote evidence byte hash mismatch: '+rel)
        value=json.loads(raw)
        target=(destination/rel).resolve()
        if not target.is_relative_to(destination.resolve()):raise ValueError('Unexpected remote evidence path')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value,indent=2),encoding='utf-8')
        exact=(destination/'raw_bytes'/rel).resolve()
        if not exact.is_relative_to((destination/'raw_bytes').resolve()):
            raise ValueError('Unexpected exact evidence path')
        exact.parent.mkdir(parents=True,exist_ok=True)
        if rel=='execution_manifest.json' and exact.exists() and exact.read_bytes()!=raw:
            raise ValueError('Frozen execution manifest bytes changed')
        tmp=exact.with_suffix(exact.suffix+'.tmp')
        tmp.write_bytes(raw);tmp.replace(exact)
    return dict(status='FETCHED',files=len(values),destination=str(destination),
        exact_bytes_destination=str(destination/'raw_bytes'),
        raw_sha256={rel:record['sha256'] for rel,record in values.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','stage','prepare','verify','smoke','launch','status','evidence'])
    parser.add_argument('--prototype',choices=['R1','R0','R8','R7'],default='R1')
    args=parser.parse_args()
    if args.mode=='preflight':print(preflight())
    elif args.mode=='stage':print(stage())
    elif args.mode=='prepare':print(invoke('prepare'))
    elif args.mode=='verify':print(invoke('preflight'))
    elif args.mode=='smoke':print(invoke('smoke',args.prototype,timeout=180))
    elif args.mode=='launch':print(launch())
    elif args.mode=='evidence':print(json.dumps(fetch_evidence()))
    else:print(json.dumps(fetch_status(),indent=2))


if __name__=='__main__':main()
