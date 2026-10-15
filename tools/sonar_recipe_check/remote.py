"""Isolated transport using the existing user-owned C01 server and environment."""
from __future__ import annotations
import argparse,base64,hashlib,json,subprocess,tarfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_head_factorial import remote as prior
execute,SCP,ENV,HOST,PYTHON,SOURCE=prior.execute,prior.SCP,prior.ENV,prior.HOST,prior.PYTHON,prior.SOURCE
CAMPAIGN='sonar_recipe_check_20261013_01'
REMOTE='/home/jiajia/experiments/'+CAMPAIGN
RESULT='results/'+CAMPAIGN+'/remote'
LOCAL=ROOT/'results'/CAMPAIGN/'transport'
OVERLAY=prior.OVERLAY
CONFIG='configs/sonar_recipe_check_20261013.yaml'
def dump(path,value):prior.dump(path,value)
def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r});result=root/{RESULT!r};python={PYTHON!r};overlay={OVERLAY!r}\n')
def stage():
    import yaml
    cfg=yaml.safe_load((ROOT/CONFIG).read_text(encoding='utf-8'))
    gate=json.loads((ROOT/cfg['r0_reuse_gate']).read_text(encoding='utf-8'))
    paths={p.relative_to(ROOT).as_posix():p for p in prior.source_files()}
    for rel in gate['source_sha256']:
        paths[rel]=ROOT/gate.get('staging_overrides',{}).get(rel,rel)
    for path in list((ROOT/'tools/sonar_recipe_check').glob('*.py'))+list((ROOT/'tests').glob('test_sonar_recipe_check*.py')):
        paths[path.relative_to(ROOT).as_posix()]=path
    for rel in (CONFIG,cfg['preregistration'],cfg['r0_reuse_gate']):paths[rel]=ROOT/rel
    for rel,digest in gate['source_sha256'].items():
        if hashlib.sha256(paths[rel].read_bytes()).hexdigest()!=digest:raise ValueError('R0 source staging drift: '+rel)
    marker=dict(source_sha256={rel:hashlib.sha256(path.read_bytes()).hexdigest() for rel,path in paths.items()},
                staging_overrides=gate.get('staging_overrides',{}),test_evaluated=False)
    LOCAL.mkdir(parents=True,exist_ok=True);dump(LOCAL/'staging_manifest.json',marker)
    archive=LOCAL/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for rel,path in paths.items():tar.add(path,arcname=rel)
        tar.add(LOCAL/'staging_manifest.json',arcname='staging_manifest.json')
    execute(header()+"if root.exists():raise FileExistsError('Preserve existing campaign namespace')\nroot.mkdir()\n",30)
    subprocess.run(SCP+[str(archive),HOST+':'+REMOTE+'/source.tar.gz'],env=ENV,check=True,timeout=60)
    code=header()+f'source=Path({SOURCE!r})\n'+'''
import tarfile,yaml
with tarfile.open(root/'source.tar.gz') as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()):raise ValueError('Unsafe archive')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,expected in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Transport drift: '+rel)
baseline=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/baseline['dataset']['data_dir']
for split in ('train','valid'):(data/split).symlink_to((source/baseline['dataset']['data_dir']/split).resolve(strict=True),target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False}))
'''
    value=json.loads(execute(code,50));dump(LOCAL/'stage.json',value);return value
def probe():
    return json.loads(execute(header()+'''
import shutil
print(json.dumps({'root_exists':root.exists(),'overlay_exists':Path(overlay).is_dir(),
 'free_bytes':shutil.disk_usage(root.parent).free,
 'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))
''',30))
def launch_mode(mode):
    code=header()+f'mode={mode!r};config={CONFIG!r}\n'+'''
import yaml
from datetime import datetime,timezone
cfg=yaml.safe_load((root/config).read_text());gpu=cfg['gpu_indices'][0]
result.mkdir(parents=True,exist_ok=True)
marker=result/('preflight_launch.json' if mode=='preflight' else 'launch.json')
if marker.exists():raise FileExistsError('Already launched; preserve evidence')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[v[1].strip(),int(v[2]),int(v[3])] for line in rows.splitlines() if (v:=line.split(','))}
if gpus[gpu][0]!='NVIDIA GeForce RTX 5090' or gpus[gpu][1]>500 or gpus[gpu][2]>10:raise RuntimeError('GPU occupied')
if mode=='campaign':
 manifest=json.loads((result/'execution_manifest.json').read_text())
 manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
 for rel,digest in manifest['source_sha256'].items():
  if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source drift: '+rel)
 for name in ('preflight_complete.json','smoke_gate.json','independent_review.json'):
  item=json.loads((result/name).read_text())
  if item['status']!='PASS':raise ValueError('Required gate not PASS: '+name)
  if name!='preflight_complete.json' and item['execution_manifest_sha256']!=manifest_sha:raise ValueError('Gate binding drift')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([overlay,str(root),str(root/'src')]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLOv5_AUTOINSTALL='false')
command=[python,'tools/sonar_recipe_check/supervisor.py',mode,'--output',str(result)]
with (result/(mode+'_stdout.log')).open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'status':mode.upper()+'_LAUNCHED','pid':child.pid,'gpu':gpu,'command':command,'launched_at_utc':datetime.now(timezone.utc).isoformat(),'test_evaluated':False}
if mode=='campaign':value['execution_manifest_sha256']=manifest_sha
marker.write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
    value=json.loads(execute(code,40));dump(LOCAL/('preflight_launch.json' if mode=='preflight' else 'launch.json'),value);return value
def preflight():return launch_mode('preflight')
def launch():return launch_mode('campaign')
def upload_review():
    payload=(ROOT/'results'/CAMPAIGN/'independent_review.json').read_bytes()
    if json.loads(payload)['status']!='PASS':raise ValueError('Independent review not PASS')
    code=header()+f'payload={base64.b64encode(payload).decode()!r}\n'+'''
import base64
payload=base64.b64decode(payload);review=json.loads(payload)
if review['execution_manifest_sha256']!=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest():raise ValueError('Review binding drift')
path=result/'independent_review.json'
if path.exists() and path.read_bytes()!=payload:raise FileExistsError('Different review exists')
path.write_bytes(payload);print(json.dumps({'status':'REVIEW_UPLOADED'}))
'''
    return json.loads(execute(code,30))
def status():
    code=header()+'''
out={}
for path in [result/n for n in ('preflight_launch.json','preflight_complete.json','preflight_supervisor_status.json','smoke_gate.json','launch.json','campaign_supervisor_status.json','queue_status.json','finalization_status.json')]+list(result.glob('formal/**/status.json')):
 if path.exists():
  value=json.loads(path.read_text())
  for key in ('pid','child_pid'):
   if value.get(key):value[key+'_alive']=Path('/proc',str(value[key])).exists()
  out[path.relative_to(result).as_posix()]=value
for path in [result/'preflight_stdout.log',result/'campaign_stdout.log']+list(result.glob('formal/**/stdout.log')):
 if path.exists():out[path.relative_to(result).as_posix()]=path.read_text()[-1800:]
print(json.dumps(out))
'''
    value=json.loads(execute(code,40));dump(LOCAL/'status.json',value);return value
def collect():
    code=header()+'''
import base64
files=[p for p in result.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.log','.npz','.xlsx') and p.stat().st_size<12000000 and 'epoch_predictions' not in p.parts]
print(json.dumps({p.relative_to(result).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}))
'''
    files=json.loads(execute(code,55));dest=ROOT/'results'/CAMPAIGN/'remote_snapshot'
    for name,content in files.items():
        path=(dest/name).resolve()
        if not path.is_relative_to(dest.resolve()):raise ValueError('Evidence path escape')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(content))
    return dict(status='COLLECTED',files=len(files),local=str(dest))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['probe','stage','preflight','upload-review','launch','status','collect'])
    args=parser.parse_args();print(json.dumps(globals()[args.mode.replace('-','_')](),ensure_ascii=False))
