"""Transport via existing local Credential Manager helper; isolated owned server."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.detector_v0_2.remote import execute,SSH,SCP,ENV,HOST,PYTHON,SOURCE

REMOTE='/home/jiajia/experiments/sonar_joint192_extension_20261013_03'
RESULT='results/sonar_joint192_extension_20261013_01'
LOCAL=ROOT/RESULT/'transport_attempt03'


def stage():
    from tools.sonar_joint192_extension_20261013 import train
    cfg=train.config();paths={ROOT/p for p in train.source_hashes()}
    baseline=train.inherited.load_config()[1]
    paths.add(ROOT/baseline['dataset']['data_dir']/'data.yaml')
    LOCAL.mkdir(parents=True,exist_ok=True)
    marker=dict(source_sha256={p.relative_to(ROOT).as_posix():train.sha(p) for p in sorted(paths)},
                test_mounted=False,trained_weights_copied=False,maximum_runs=cfg['maximum_runs'])
    mp=LOCAL/'staging_manifest.json'
    if mp.exists():raise FileExistsError('Transport evidence already exists')
    train.dump(mp,marker)
    archive=LOCAL/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
        tar.add(mp,arcname='staging_manifest.json')
    execute('from pathlib import Path\np=Path('+repr(REMOTE)+')\nif p.exists():raise FileExistsError("Isolated namespace already exists")\np.mkdir()\n')
    subprocess.run(SCP+[str(archive),HOST+':'+REMOTE+'/source.tar.gz'],env=ENV,check=True)
    code='''from pathlib import Path
import tarfile,json,hashlib,yaml
root=Path(REMOTE)
with tarfile.open(root/'source.tar.gz') as tar:
 for m in tar.getmembers():
  p=(root/m.name).resolve()
  if not m.isfile() or not p.is_relative_to(root.resolve()):raise ValueError('Unsafe archive member')
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,sha in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=sha:raise ValueError('Transport hash differs:'+rel)
b=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/b['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ('train','valid'):
 target=(Path(SOURCE)/b['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
print(json.dumps(dict(status='STAGED',source_files=len(marker['source_sha256']),test_mounted=False)))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('Path(SOURCE)','Path('+repr(SOURCE)+')')
    result=json.loads(execute(code,60));train.dump(LOCAL/'stage_receipt.json',result)
    return result


def invoke(mode,timeout=180):
    if mode not in ('prepare','smoke','report','tests','recover'):
        raise ValueError('Only registered non-training preflight modes')
    cmd=([PYTHON,'-B','-m','pytest','tests/test_sonar_joint192_extension_model.py','tests/test_sonar_joint192_extension_completion.py','-q','--junitxml='+RESULT+'/model_tests.xml']
         if mode=='tests' else [PYTHON,'-B','tools/sonar_joint192_extension_20261013/recovery.py']
         if mode=='recover' else [PYTHON,'-B','tools/sonar_joint192_extension_20261013/train.py',mode])
    code='import subprocess,os\nos.environ.update(CUDA_VISIBLE_DEVICES="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1",PYTHONPATH='+repr(REMOTE+':'+REMOTE+'/src')+')\np=subprocess.run('+repr(cmd)+',cwd='+repr(REMOTE)+',capture_output=True,text=True)\nprint(p.stdout)\nprint(p.stderr)\nif p.returncode:raise RuntimeError("Registered invocation failed:"+str(p.returncode))\n'
    return execute(code,timeout)


def launch():
    code='''from pathlib import Path
import json,os,subprocess,sys,hashlib
root=Path(REMOTE);result=root/RESULT
sys.path[:0]=[str(root),str(root/'src')]
from tools.sonar_joint192_extension_20261013 import train
m=train.manifest();cfg=m['config'];gate=json.loads((result/'smoke_gate.json').read_text())
if gate['status']!='PASS' or gate['execution_manifest_sha256']!=train.sha(result/'execution_manifest.json'):raise ValueError('No bound smoke')
if (result/'launch.json').exists() or (result/'queue_status.json').exists():raise FileExistsError('Already launched')
output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[int(x) for x in v[1:]] for line in output.splitlines() if (v:=line.split(','))}
used,total,util=gpus[cfg['gpu']]
per_worker=gate['peak_reserved_bytes']/1048576+2048
actual=min(cfg['concurrency'],int((total-used)//per_worker))
if actual<1 or util>5 or used>512:raise RuntimeError('Requested GPU no longer independently free')
launch=dict(status='LAUNCHED',gpu=cfg['gpu'],actual_concurrency=actual,maximum_concurrency=cfg['concurrency'],
            smoke_gate_sha256=train.sha(result/'smoke_gate.json'),execution_manifest_sha256=train.sha(result/'execution_manifest.json'),
            gpu_snapshot=gpus,required_mib_per_worker=per_worker,maximum_runs=cfg['maximum_runs'],test_evaluated=False,updated_at=train.now())
train.dump(result/'launch.json',launch)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(cfg['gpu']),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (result/'queue_stdout.log').open('xb') as log:
 proc=subprocess.Popen([PYTHON,'-B','tools/sonar_joint192_extension_20261013/queue_runner.py','--gpu',str(cfg['gpu'])],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch['manager_pid']=proc.pid;train.dump(result/'launch.json',launch)
print(json.dumps(launch))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT)).replace('[PYTHON,','['+repr(PYTHON)+',')
    value=json.loads(execute(code,50))
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'launch_receipt.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value


def status():
    code='''from pathlib import Path
import json
root=Path(REMOTE);result=root/RESULT
output={}
for name in ('launch.json','queue_status.json','comparison.json','selection.json'):
 p=result/name
 if p.exists():output[name]=json.loads(p.read_text())
active=output.get('queue_status.json',{}).get('active',[])
output['active_runs']=[]
for item in active:
 p=result/'runs'/item['arm']/('seed_'+str(item['seed']))/'status.json'
 output['active_runs'].append(dict(task=item,status=json.loads(p.read_text()) if p.exists() else None))
print(json.dumps(output))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT))
    data=json.loads(execute(code,45));LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'live_status.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return data


def collect():
    code='''from pathlib import Path
import tarfile,json
root=Path(REMOTE);result=root/RESULT;archive=root/'evidence_snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(result.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.md','.log') and not p.name.endswith('.tmp'):
   tar.add(p,arcname=p.relative_to(result).as_posix())
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size)))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT))
    receipt=json.loads(execute(code,90));LOCAL.mkdir(parents=True,exist_ok=True)
    archive=LOCAL/'evidence_snapshot.tar.gz'
    subprocess.run(SCP+[HOST+':'+receipt['archive'],str(archive)],env=ENV,check=True)
    dest=ROOT/RESULT/'remote_snapshot_attempt03';dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            target=(dest/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe snapshot member')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
    return dict(local=str(dest),**receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['stage','prepare','tests','smoke','recover','launch','status','collect'])
    args=parser.parse_args()
    if args.mode in ('prepare','tests','smoke','recover'):print(invoke(args.mode,600 if args.mode in ('smoke','recover') else 180))
    else:print(json.dumps(globals()[args.mode](),ensure_ascii=False,indent=2))
