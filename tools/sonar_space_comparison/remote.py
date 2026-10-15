"""Local source transport and bounded status; all model execution is remote."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.detector_v0_2.remote import execute, SCP, ENV, HOST, PYTHON, SOURCE

REMOTE='/home/jiajia/experiments/sonar_space_comparison_20261010_01'
RESULT='results/sonar_space_comparison_20261010_01/remote'
LOCAL=ROOT/'results/sonar_space_comparison_20261010_01/transport'


def stage():
    # Loading only configuration and source hashes; no local model or data execution.
    from tools.sonar_joint192_validation import train
    cfg,_=train.load_config()
    paths={ROOT/p for p in train.source_hashes(cfg)}
    paths.update(ROOT/p for p in (
        'tools/modnas_elastic/model.py','tools/sonar_joint192_validation/domain_metrics.py',
        'tools/sonar_space_comparison/campaign.py','tools/sonar_space_comparison/remote.py',
        'configs/sonar_space_comparison_20261010.yaml','tests/test_sonar_space_comparison.py'))
    baseline=__import__('yaml').safe_load((ROOT/'configs/baseline_sonar_frozen.yaml').read_text(encoding='utf-8'))
    paths.add(ROOT/baseline['dataset']['data_dir']/'data.yaml')
    LOCAL.mkdir(parents=True,exist_ok=True)
    marker=dict(source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        test_mounted=False,trained_weights_copied=False)
    (LOCAL/'staging_manifest.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
    archive=LOCAL/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
        tar.add(LOCAL/'staging_manifest.json',arcname='staging_manifest.json')
    execute('from pathlib import Path\np=Path('+repr(REMOTE)+')\nif p.exists(): raise FileExistsError("Namespace exists; preserve it")\np.mkdir()\n')
    subprocess.run(SCP+[str(archive),HOST+':'+REMOTE+'/source.tar.gz'],env=ENV,check=True)
    code='''from pathlib import Path
import tarfile,json,hashlib,yaml
root=Path(REMOTE)
with tarfile.open(root/'source.tar.gz') as tar:
 for m in tar.getmembers():
  p=(root/m.name).resolve()
  if not m.isfile() or not p.is_relative_to(root.resolve()): raise ValueError('Unsafe archive')
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,expected in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Transport hash drift')
b=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/b['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ('train','valid'):
 target=(Path(SOURCE)/b['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False}))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('Path(SOURCE)','Path('+repr(SOURCE)+')')
    return json.loads(execute(code,50))


def preflight_launch():
    code='''from pathlib import Path
import json,subprocess,os
root=Path(REMOTE);r=root/RESULT;r.mkdir(parents=True,exist_ok=True)
if (r/'preflight_launch.json').exists():raise FileExistsError('Preflight already invoked')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[int(x) for x in v[1:]] for line in rows.splitlines() if (v:=line.split(','))}
if any(gpus[g][0]>500 or gpus[g][1]>10 for g in (0,1)):raise RuntimeError('GPU availability changed')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[PYTHON,'-c',"import subprocess,sys; commands=[[sys.executable,'tools/sonar_space_comparison/campaign.py','prepare'],[sys.executable,'tests/test_sonar_space_comparison.py'],[sys.executable,'tools/sonar_space_comparison/campaign.py','smoke']]; [subprocess.run(c,check=True,timeout=900) for c in commands]"]
with (r/'preflight_stdout.log').open('a') as log:
 p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
info={'status':'PREFLIGHT_LAUNCHED','pid':p.pid,'gpu':0};(r/'preflight_launch.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT)).replace('[PYTHON,','['+repr(PYTHON)+',')
    return json.loads(execute(code,45))


def sync_before_prepare():
    """Update only our staged source before its first execution manifest exists."""
    import base64
    paths=['configs/sonar_space_comparison_20261010.yaml',
           'tools/sonar_space_comparison/campaign.py','tools/sonar_space_comparison/remote.py',
           'tests/test_sonar_space_comparison.py']
    files={rel:base64.b64encode((ROOT/rel).read_bytes()).decode() for rel in paths}
    code='''from pathlib import Path
import base64,json,hashlib
root=Path(REMOTE);r=root/RESULT
if (r/'execution_manifest.json').exists() or (r/'preflight_launch.json').exists():raise RuntimeError('Frozen/preflight-started campaign; refuse source update')
marker=json.loads((root/'staging_manifest.json').read_text())
files=FILES
for rel,encoded in files.items():
 data=base64.b64decode(encoded);(root/rel).write_bytes(data)
 marker['source_sha256'][rel]=hashlib.sha256(data).hexdigest()
(root/'staging_manifest.json').write_text(json.dumps(marker,indent=2))
print(json.dumps({'status':'PRE_PREPARE_SOURCE_SYNCED','source_sha256':marker['source_sha256']}))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT)).replace('FILES',repr(files))
    value=json.loads(execute(code,45));marker=dict(source_sha256=value['source_sha256'],test_mounted=False,trained_weights_copied=False)
    (LOCAL/'staging_manifest.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
    return dict(status=value['status'],updated_files=len(files))


def launch():
    code='''from pathlib import Path
import subprocess,os,json,sys
root=Path(REMOTE);r=root/RESULT
sys.path[:0]=[str(root),str(root/'src')]
from tools.sonar_space_comparison.campaign import manifest
m=manifest()
gate=json.loads((r/'smoke_gate.json').read_text());tests=json.loads((r/'protocol_tests.json').read_text())
manifest_sha=__import__('hashlib').sha256((r/'execution_manifest.json').read_bytes()).hexdigest()
if (gate['status']!='PASS' or tests['status']!='PASS' or gate['execution_manifest_sha256']!=manifest_sha or tests['execution_manifest_sha256']!=manifest_sha):raise RuntimeError('Bound preflight gates not passed')
if (r/'launch.json').exists():raise FileExistsError('Campaign already launched')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[v[1].strip(),int(v[2]),int(v[3])] for line in rows.splitlines() if (v:=line.split(','))}
for gpu in m['config']['gpu_indices']:
 if gpus[gpu][0]!='NVIDIA GeForce RTX 5090' or gpus[gpu][1]>500 or gpus[gpu][2]>10:raise RuntimeError('GPU availability changed')
info={'status':'LAUNCHED','workers':[],'gpu_snapshot':gpus,'budget_GPU_hours_per_search':m['config']['gpu_hours_per_space_per_search'],'test_evaluated':False}
for gpu in m['config']['gpu_indices']:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (r/f'worker_gpu{gpu}_stdout.log').open('a') as log:
  p=subprocess.Popen([PYTHON,'tools/sonar_space_comparison/campaign.py','worker','--gpu',str(gpu)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info['workers'].append({'gpu':gpu,'pid':p.pid})
(r/'launch.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')').replace('root/RESULT','root/'+repr(RESULT)).replace('[PYTHON,','['+repr(PYTHON)+',')
    info=json.loads(execute(code,45))
    (LOCAL/'launch.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def status():
    code='''from pathlib import Path
import json,os
r=Path(REMOTE)/RESULT;out={}
for name in ('preflight_launch.json','launch.json','smoke_gate.json','protocol_tests.json','inventory.json'):
 p=r/name
 if p.exists():
  value=json.loads(p.read_text())
  if name=='inventory.json':value={k:v for k,v in value.items() if k!='inventory'}
  out[name]=value
  if name in ('preflight_launch.json','launch.json'):
   processes=[value] if 'pid' in value else value['workers']
   for proc in processes:
    try:os.kill(proc['pid'],0);proc['alive']=True
    except ProcessLookupError:proc['alive']=False
for p in sorted(r.glob('worker_gpu*/queue_status.json')):
 out[str(p.relative_to(r))]=json.loads(p.read_text())
for p in sorted(r.glob('search/*/search_*/search_status.json')):
 value=json.loads(p.read_text());completed=value.pop('completed',[])
 value['n_completed']=len(completed);value['best_valid_mAP50_95']=max((v['mAP50_95'] for v in completed),default=None)
 value.pop('selected',None);value.pop('attempts',None);out[str(p.relative_to(r))]=value
 for q in sorted(p.parent.glob('candidate_*/status.json')):
  v=json.loads(q.read_text())
  if v['status']=='RUNNING':out[str(q.relative_to(r))]=v
for p in [r/'preflight_stdout.log',*r.glob('worker_gpu*_stdout.log')]:
 if p.exists():out[p.name]=p.read_text()[-3000:]
print(json.dumps(out))
'''.replace('Path(REMOTE)/RESULT','Path('+repr(REMOTE)+')/'+repr(RESULT))
    value=json.loads(execute(code,45));LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'status.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value


def collect():
    # Read-only partial snapshot: keep all model weights on server.
    code='''from pathlib import Path
import json,hashlib,tarfile
r=Path(REMOTE)/RESULT
files=[p for p in r.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.md','.csv')]
snapshots={str(p.relative_to(r)):p.read_bytes() for p in files}
with tarfile.open(r.parent/'evidence.tar.gz','w:gz') as tar:
 import io
 for name,data in snapshots.items():
  member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
 marker=json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in snapshots.items()},indent=2).encode()
 member=tarfile.TarInfo('collection_manifest.json');member.size=len(marker);tar.addfile(member,io.BytesIO(marker))
print(json.dumps({'files':len(files)}))
'''.replace('Path(REMOTE)/RESULT','Path('+repr(REMOTE)+')/'+repr(RESULT))
    info=json.loads(execute(code,50));archive=LOCAL/'evidence.tar.gz'
    subprocess.run(SCP+[HOST+':'+REMOTE+'/'+str(Path(RESULT).parent).replace('\\','/')+'/evidence.tar.gz',str(archive)],env=ENV,check=True)
    dest=ROOT/'results/sonar_space_comparison_20261010_01/remote_snapshot';dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target=(dest/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe evidence')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
    marker=json.loads((dest/'collection_manifest.json').read_text())
    for rel,expected in marker.items():
        if hashlib.sha256((dest/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Evidence transfer mismatch')
    return dict(**info,transport_sha256_verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['stage','sync-before-prepare','preflight-launch','launch','status','collect'])
    args=parser.parse_args();print(json.dumps(globals()[args.mode.replace('-','_')](),ensure_ascii=False))
