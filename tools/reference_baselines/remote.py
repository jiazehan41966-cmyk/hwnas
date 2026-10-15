"""Credential Manager transport for the isolated 72-run reference campaign."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.detector_v0_2.remote import execute, PYTHON

CAMPAIGN = 'reference_baselines_20261015_01'
REMOTE = '/home/jiajia/experiments/' + CAMPAIGN
LOCAL = ROOT / 'results' / CAMPAIGN
OLD_OVERLAY = '/home/jiajia/experiments/sonar_yolov9_baseline_20261005_remote_coco/results/sonar_yolov9_baseline_20261005/env_overlay'


def save(name, value):
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r});python={PYTHON!r};old_overlay={OLD_OVERLAY!r}\n')


def upload(rel, source=None):
    path = Path(source) if source else ROOT / rel
    contents = path.read_bytes()
    code = header() + f'rel={rel!r};payload={base64.b64encode(contents).decode()!r}\n' + '''
import base64
path=root/rel
if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Path escape')
if (root/'launch.json').exists():raise RuntimeError('Campaign frozen after launch')
path.parent.mkdir(parents=True,exist_ok=True)
data=base64.b64decode(payload)
path.write_bytes(data)
print(json.dumps({'file':rel,'sha256':hashlib.sha256(data).hexdigest()}))
'''
    return json.loads(execute(code, 45))


def prepare():
    spec = {
        'base_interpreter': PYTHON,
        'base_pins': {'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128', 'numpy': '2.4.4'},
        'readonly_overlay': OLD_OVERLAY,
        'private_overlay': REMOTE + '/vendor',
        'pip_phases': [['ultralytics==8.3.221', 'ultralytics-thop==2.0.18',
                        'polars==1.35.2', 'polars-runtime-32==1.35.2']],
        'installation': '--target private_overlay --no-deps; shared base and old overlay unchanged',
        'device_allocation': [2, 3], 'threads_per_worker': 2,
        'user_authorized_shared_residents': {'2': 5406, '3': 5300},
        'test_evaluated': False,
    }
    save('environment_spec.json', spec)
    upload('environment_spec.json', LOCAL / 'environment_spec.json')
    code = header() + '''
from datetime import datetime,timezone
import importlib.metadata as md
spec=json.loads((root/'environment_spec.json').read_text())
for name,version in spec['base_pins'].items():
 if md.version(name)!=version:raise RuntimeError('Base environment drift: '+name)
if (root/'environment_install_launch.json').exists():raise FileExistsError('Installation already launched')
cmd=[python,'-m','pip','install','--disable-pip-version-check','--no-deps','--only-binary=:all:','--target',str(root/'vendor'),*spec['pip_phases'][0]]
with (root/'environment_install.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out={'pid':child.pid,'command':cmd,'created_utc':datetime.now(timezone.utc).isoformat()}
(root/'environment_install_launch.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
'''
    result = json.loads(execute(code, 40))
    save('environment_install_launch.json', result)
    return result


def stage():
    files = ['scripts/run_reference_baselines.py', 'tools/reference_baselines/campaign.py']
    receipt = {p: upload(p)['sha256'] for p in files}
    upload('data_audit.json', LOCAL / 'data_audit.json')
    upload('data_files.json', LOCAL / 'data_files.json')
    save('source_manifest.json', receipt)
    upload('source_manifest.json', LOCAL / 'source_manifest.json')
    return invoke('stage-data')


def invoke(mode):
    code = header() + f'mode={mode!r}\n' + '''
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'vendor'),old_overlay,str(root)]),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLO_CONFIG_DIR=str(root/'yolo_settings'),YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
cmd=[python,str(root/'tools/reference_baselines/campaign.py'),mode]
process=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=240)
print(json.dumps({'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr[-6000:]}))
'''
    return json.loads(execute(code, 250))


def background(mode):
    code = header() + f'mode={mode!r}\n' + '''
from datetime import datetime,timezone
marker=root/(mode+'_launch.json')
if marker.exists():raise FileExistsError('Already launched: '+mode)
if mode=='campaign':
 for name in ('smoke_gate.json','independent_review.json'):
  gate=json.loads((root/name).read_text())
  if gate['status']!='PASS':raise RuntimeError('Gate not passed: '+name)
  if gate['source_manifest_sha256']!=hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest():raise RuntimeError('Gate source binding mismatch')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'vendor'),old_overlay,str(root)]),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLO_CONFIG_DIR=str(root/'yolo_settings'),YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
cmd=[python,str(root/'tools/reference_baselines/campaign.py'),mode]
with (root/(mode+'.log')).open('x') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out={'pid':child.pid,'command':cmd,'created_utc':datetime.now(timezone.utc).isoformat(),'test_evaluated':False}
marker.write_text(json.dumps(out,indent=2))
if mode=='campaign':(root/'launch.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
'''
    out = json.loads(execute(code, 40))
    save(mode + '_launch.json', out)
    return out


def status():
    code = header() + '''
out={}
for p in [root/n for n in ('environment_install_launch.json','data_stage.json','environment_lock.json','smoke_launch.json','smoke_status.json','smoke_gate.json','independent_review.json','campaign_launch.json','campaign_status.json')]:
 if p.exists():
  val=json.loads(p.read_text())
  if 'pid' in val:val['pid_exists']=Path('/proc',str(val['pid'])).exists()
  out[p.name]=val
for name in ('environment_install.log','smoke.log','campaign.log'):
 p=root/name
 if p.exists():out[name]=p.read_text(errors='replace')[-3000:]
for p in (root/'logs').glob('*.log') if (root/'logs').exists() else []:
 out[str(p.relative_to(root))]=p.read_text(errors='replace')[-1100:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
out['completed_formal_results']=[str(p.relative_to(root)) for p in (root/'results/formal').glob('**/result.json')]
print(json.dumps(out))
'''
    out = json.loads(execute(code, 45))
    save('remote_status.json', out)
    return out


def collect():
    code = header() + '''
import base64
paths=list(root.glob('*.json'))+list((root/'results').glob('**/result.json'))+list((root/'results').glob('**/summary.csv'))
print(json.dumps({str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in paths if p.stat().st_size<12000000}))
'''
    files = json.loads(execute(code, 50))
    target = LOCAL / 'remote_snapshot'
    for rel, body in files.items():
        path = target / rel
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError('Remote path escape')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(body))
    return {'files': len(files), 'local': str(target)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['prepare', 'stage', 'environment', 'smoke', 'campaign', 'status', 'collect', 'verify'])
    a = ap.parse_args()
    if a.mode in ('smoke', 'campaign'):
        result = background(a.mode)
    elif a.mode in ('environment', 'verify'):
        result = invoke(a.mode)
    else:
        result = globals()[a.mode]()
    print(json.dumps(result, indent=2, ensure_ascii=False))
