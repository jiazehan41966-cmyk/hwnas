"""Existing-credential transport and guarded launch of the isolated C01b run.

Only the explicit CLI operation accesses the server.  Import and source_files
are local filesystem operations; dependencies are never installed.  Mutable
baseline clarification is distinct from the frozen candidate source closure.
"""
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
from tools.sonar_head_factorial import remote as oldr

# Reuse the existing SSH/askpass transport rather than reading any credentials.
execute, SCP, ENV, HOST, PYTHON, SOURCE = (
    oldr.execute, oldr.SCP, oldr.ENV, oldr.HOST, oldr.PYTHON, oldr.SOURCE)
OVERLAY = oldr.OVERLAY
CAMPAIGN = 'sonar_head_c01b_20261013_01'
FAILED_PREFLIGHT_NAMESPACE = '/home/jiajia/experiments/' + CAMPAIGN
REMOTE = FAILED_PREFLIGHT_NAMESPACE + '_revision2'
RESULT = 'results/' + CAMPAIGN + '/remote'
CONFIG = 'configs/sonar_head_c01b_20261013.yaml'
CAMPAIGN_LOCAL = ROOT / 'results' / CAMPAIGN
LOCAL = CAMPAIGN_LOCAL / 'transport_revision2'
RECEIPT = ROOT / 'results/sonar_head_factorial_20261012_02_stable/transport/author_source_receipt.json'
REVIEW_SCOPE = 'C01b_66_to_6_readiness'
RESOLVED_CHOICE = 'use_completed_C01_regmax16_66_channels'
EVIDENCE_SUFFIXES = ('.json', '.csv', '.md', '.log', '.npz')
MAX_EVIDENCE_BYTES = 12_000_000


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r}); result=root/{RESULT!r}\n'
            f'python={PYTHON!r}; overlay={OVERLAY!r}; config_relative={CONFIG!r}\n')


def _config():
    import yaml
    cfg = yaml.safe_load((ROOT / CONFIG).read_text(encoding='utf-8'))
    if (cfg.get('version') != CAMPAIGN or cfg.get('result_root') != RESULT
            or cfg.get('gpu_index') != 1):
        raise ValueError('Unregistered C01b namespace or GPU')
    return cfg


def source_files():
    """Exact old staging closure plus C01b sources/tests/config/preregistration."""
    paths = set(oldr.source_files())
    paths.update((ROOT / 'tools/sonar_head_c01b').glob('*.py'))
    paths.update((ROOT / 'tests').glob('test_sonar_head_c01b_*.py'))
    paths.update((ROOT / CONFIG, ROOT / 'tools/sonar_head_c01b/README.md',
                  CAMPAIGN_LOCAL / 'PREREGISTRATION.md'))
    required = (ROOT / 'tools/sonar_head_c01b/train.py',
                ROOT / 'tools/sonar_head_c01b/model.py',
                ROOT / 'tools/sonar_head_c01b/analysis.py',
                ROOT / 'tests/test_sonar_head_c01b_model.py',
                ROOT / 'tests/test_sonar_head_c01b_analysis.py')
    paths.update(required)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Staged source must stay inside the repository')
        if path.suffix in ('.pt', '.pth', '.ckpt'):
            raise ValueError('Model weights cannot be staged')
    return sorted(paths)


def stage():
    """Create one new namespace; preserve every existing namespace on failure."""
    _config()
    paths = source_files()
    receipt_bytes = RECEIPT.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_arc = 'reference/yolov9/source_receipt.json'
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    if receipt_arc in hashes and hashes[receipt_arc] != receipt_sha:
        raise ValueError('Old source receipt bytes differ from the staged closure')
    hashes[receipt_arc] = receipt_sha
    resolution = CAMPAIGN_LOCAL / 'baseline_resolution.json'
    control_bytes = resolution.read_bytes() if resolution.is_file() else None
    if control_bytes is not None:
        control = json.loads(control_bytes)
        if control.get('status') not in ('UNRESOLVED', 'RESOLVED'):
            raise ValueError('Invalid initial baseline clarification status')
    else:
        control = None
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Never regenerate the old author receipt JSON; its whitespace is evidence.
    (LOCAL / 'author_source_receipt.json').write_bytes(receipt_bytes)
    marker = dict(schema_version=1, campaign=CAMPAIGN, source_sha256=hashes,
        remote_namespace=REMOTE, source_revision=2,
        prior_failed_preflight_namespace_preserved=FAILED_PREFLIGHT_NAMESPACE,
        author_source_receipt_sha256=receipt_sha,
        author_source_receipt_source=RECEIPT.relative_to(ROOT).as_posix(),
        mutable_baseline_resolution_in_frozen_source=False,
        initial_baseline_resolution_sha256=(hashlib.sha256(control_bytes).hexdigest()
                                          if control_bytes is not None else None),
        initial_baseline_status=control.get('status') if control else 'UNRESOLVED',
        candidate_preflight_provisional=control is None or control.get('status') != 'RESOLVED',
        test_mounted=False, trained_weights_copied=False, dependency_installation=False)
    dump(LOCAL / 'staging_manifest.json', marker)
    archive = LOCAL / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            arcname = path.relative_to(ROOT).as_posix()
            if arcname != receipt_arc:
                tar.add(path, arcname=arcname, recursive=False)
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json', recursive=False)
        tar.add(LOCAL / 'author_source_receipt.json', arcname=receipt_arc, recursive=False)
        # Result-side prereg is byte-identical to its frozen repository source.
        tar.add(CAMPAIGN_LOCAL / 'PREREGISTRATION.md',
                arcname=RESULT + '/PREREGISTRATION.md', recursive=False)
        if control_bytes is not None:
            (LOCAL / 'initial_baseline_resolution.json').write_bytes(control_bytes)
            tar.add(LOCAL / 'initial_baseline_resolution.json',
                    arcname=RESULT + '/baseline_resolution.json', recursive=False)
    execute(header() + "if root.exists() or root.is_symlink(): raise FileExistsError('Namespace exists; preserve it')\nroot.mkdir()\n", 30)
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/source.tar.gz'],
                   env=ENV, check=True, timeout=60)
    code = header() + f'source=Path({SOURCE!r})\n' + '''
import tarfile,yaml
with tarfile.open(root/'source.tar.gz') as tar:
 seen=set()
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()) or member.name in seen:raise ValueError('Unsafe or duplicate archive member')
  seen.add(member.name)
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:out.write(tar.extractfile(member).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Transport mismatch: '+rel)
receipt=root/'reference/yolov9/source_receipt.json'
if hashlib.sha256(receipt.read_bytes()).hexdigest()!=marker['author_source_receipt_sha256']:raise ValueError('Old author receipt byte identity differs')
prereg=root/('results/'+marker['campaign']+'/PREREGISTRATION.md')
if (result/'PREREGISTRATION.md').read_bytes()!=prereg.read_bytes():raise ValueError('Preregistration alias byte mismatch')
baseline=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/baseline['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ('train','valid'):
 target=(source/baseline['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
if not Path(python).is_file() or not Path(overlay).is_dir():raise FileNotFoundError('Existing Python or YOLO overlay absent')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False,'weights_copied':False,'dependency_installation':False,'baseline_status':marker['initial_baseline_status'],'candidate_preflight_provisional':marker['candidate_preflight_provisional'],'author_source_receipt_sha256':marker['author_source_receipt_sha256']}))
'''
    value = json.loads(execute(code, 55))
    dump(LOCAL / 'stage_result.json', value)
    return value


def probe():
    code = header() + '''
import shutil
print(json.dumps({'root_exists':root.exists(),'overlay_exists':Path(overlay).is_dir(),'python_exists':Path(python).is_file(),
 'result_volume_free_bytes':shutil.disk_usage(root.parent).free,
 'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))
'''
    return json.loads(execute(code, 30))


def preflight():
    """One controlled GPU1 candidate preflight; no automatic retry or install."""
    code = header() + '''
import yaml
cfg=yaml.safe_load((root/config_relative).read_text())
gpu=cfg['gpu_index']
if gpu!=1:raise ValueError('C01b preflight requires GPU1')
if not Path(python).is_file() or not Path(overlay).is_dir():raise FileNotFoundError('Existing runtime absent')
result.mkdir(parents=True,exist_ok=True)
if (result/'preflight_launch.json').exists():raise FileExistsError('Preflight already invoked; no retry')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[v[1].strip(),int(v[2]),int(v[3])] for line in rows.splitlines() if (v:=line.split(','))}
if gpu not in gpus or gpus[gpu][0]!='NVIDIA GeForce RTX 5090' or gpus[gpu][1]>500 or gpus[gpu][2]>10:raise RuntimeError('Preflight GPU1 occupied or differs')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([overlay,str(root),str(root/'src')]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLOv5_AUTOINSTALL='false')
tests=['tests/test_sonar_head_factorial_protocol.py','tests/test_sonar_head_factorial_model.py','tests/test_sonar_head_factorial_stable_analysis.py']
tests+=sorted(p.relative_to(root).as_posix() for p in (root/'tests').glob('test_sonar_head_c01b_*.py'))
commands=[
 [python,'tools/sonar_head_c01b/train.py','prepare','--config',config_relative,'--output',str(result),'--device','cuda'],
 [python,'-m','pytest','-q',*tests],
 [python,'tools/sonar_head_c01b/train.py','smoke','--config',config_relative,'--output',str(result),'--device','cuda']]
supervisor="""import subprocess,json,pathlib,hashlib,datetime,traceback
commands=COMMANDS
result=pathlib.Path(RESULTPATH)
try:
 for command in commands:subprocess.run(command,check=True,timeout=1200)
 manifest=result/'execution_manifest.json'
 resolution=json.loads((result/'baseline_resolution.json').read_text())
 value={'status':'PASS','execution_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'commands':commands,'baseline_status':resolution.get('status'),'candidate_preflight_provisional':resolution.get('status')!='RESOLVED','formal_eligible':False,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
except Exception as error:
 traceback.print_exc()
 value={'status':'FAILED_NO_AUTORETRY','error_type':type(error).__name__,'error':str(error),'formal_eligible':False,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(result/'preflight_complete.json').write_text(json.dumps(value,indent=2))
""".replace('COMMANDS',repr(commands)).replace('RESULTPATH',repr(str(result)))
resolution_path=result/'baseline_resolution.json'
baseline_status=json.loads(resolution_path.read_text()).get('status') if resolution_path.is_file() else 'UNRESOLVED'
with (result/'preflight_launch.json').open('x') as record:
 with (result/'preflight_stdout.log').open('x') as log:
  p=subprocess.Popen([python,'-c',supervisor],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value={'status':'PREFLIGHT_LAUNCHED','pid':p.pid,'gpu':gpu,'commands':commands,'baseline_status':baseline_status,'candidate_preflight_provisional':baseline_status!='RESOLVED','formal_training_started':False}
 record.write(json.dumps(value,indent=2))
print(json.dumps(value))
'''
    value = json.loads(execute(code, 40))
    dump(LOCAL / 'preflight_launch.json', value)
    return value


def _validate_resolution(value, cfg):
    if (value.get('status') != 'RESOLVED' or value.get('choice') != RESOLVED_CHOICE
            or not isinstance(value.get('user_authorization'), str)
            or not value['user_authorization'].strip()
            or value.get('C01_execution_manifest_sha256') != cfg['C01_execution_manifest_sha256']
            or value.get('C01_reg_max') != cfg['C01_reg_max']
            or value.get('C01_raw_output_channels_per_scale') != cfg['C01_raw_output_channels_per_scale']):
        raise ValueError('Resolved explicit user authorization for the actual frozen C01 is required')


def upload_resolution(path=None):
    """Upload the user's clarification before formal authorization is frozen."""
    path = Path(path) if path is not None else CAMPAIGN_LOCAL / 'baseline_resolution.json'
    content = path.read_bytes()
    _validate_resolution(json.loads(content), _config())
    code = header() + f'encoded={base64.b64encode(content).decode()!r}\n' + f'choice={RESOLVED_CHOICE!r}\n' + '''
import base64,yaml
payload=base64.b64decode(encoded); value=json.loads(payload)
cfg=yaml.safe_load((root/config_relative).read_text())
if value.get('status')!='RESOLVED' or value.get('choice')!=choice or not isinstance(value.get('user_authorization'),str) or not value['user_authorization'].strip():raise ValueError('User baseline authorization missing')
for field in ('C01_execution_manifest_sha256','C01_reg_max','C01_raw_output_channels_per_scale'):
 if value.get(field)!=cfg[field]:raise ValueError('Resolved C01 identity differs: '+field)
path=result/'baseline_resolution.json'
if (result/'launch.json').exists() or (result/'formal_authorization.json').exists():
 if not path.is_file() or path.read_bytes()!=payload:raise FileExistsError('Formal authorization frozen; clarification cannot change')
else:
 if path.is_file() and path.read_bytes()!=payload:
  initial=result/'baseline_resolution_initial.json'
  if not initial.exists():initial.write_bytes(path.read_bytes())
 path.write_bytes(payload)
print(json.dumps({'status':'RESOLUTION_UPLOADED','baseline_resolution_sha256':hashlib.sha256(payload).hexdigest(),'execution_manifest_sha256':hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest(),'baseline_status':value['status']}))
'''
    value = json.loads(execute(code, 30))
    dump(LOCAL / 'resolution_upload.json', value)
    return value


def upload_review(path=None):
    path = Path(path) if path is not None else CAMPAIGN_LOCAL / 'independent_review.json'
    content = path.read_bytes()
    review = json.loads(content)
    if (review.get('status') != 'PASS' or review.get('independent') is not True
            or review.get('scope') != REVIEW_SCOPE
            or any(not isinstance(review.get(key), str)
                   or len(review[key]) != 64
                   or any(char not in '0123456789abcdef' for char in review[key])
                   for key in ('execution_manifest_sha256', 'smoke_gate_sha256',
                               'preflight_complete_sha256'))):
        raise ValueError('Actual independent C01b readiness review PASS is required')
    code = header() + f'encoded={base64.b64encode(content).decode()!r}\n' + f'scope={REVIEW_SCOPE!r}\n' + '''
import base64
payload=base64.b64decode(encoded); review=json.loads(payload)
manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
if review.get('status')!='PASS' or review.get('independent') is not True or review.get('scope')!=scope or review.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Independent review status/scope/manifest binding differs')
complete=json.loads((result/'preflight_complete.json').read_text())
gate_path=result/'smoke_gate.json'; gate=json.loads(gate_path.read_text())
if complete.get('status')!='PASS' or complete.get('execution_manifest_sha256')!=manifest_sha or gate.get('status')!='PASS' or gate.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Actual preflight/smoke evidence not ready')
if review.get('smoke_gate_sha256')!=hashlib.sha256(gate_path.read_bytes()).hexdigest():raise ValueError('Reviewed smoke gate digest differs')
if review.get('preflight_complete_sha256')!=hashlib.sha256((result/'preflight_complete.json').read_bytes()).hexdigest():raise ValueError('Reviewed preflight completion digest differs')
path=result/'independent_review.json'
if path.exists() and path.read_bytes()!=payload:raise FileExistsError('Different final review already present; preserve it')
path.write_bytes(payload)
print(json.dumps({'status':'REVIEW_UPLOADED','independent_review_sha256':hashlib.sha256(payload).hexdigest(),'execution_manifest_sha256':manifest_sha}))
'''
    value = json.loads(execute(code, 30))
    dump(LOCAL / 'review_upload.json', value)
    return value


def launch():
    """Start one GPU1 queue only after all explicit formal guards pass."""
    code = header() + f'choice={RESOLVED_CHOICE!r}; scope={REVIEW_SCOPE!r}\n' + '''
import yaml
cfg=yaml.safe_load((root/config_relative).read_text())
if (result/'launch.json').exists() or (result/'queue_status.json').exists():raise FileExistsError('Campaign already launched; no automatic retry')
manifest_path=result/'execution_manifest.json'; manifest=json.loads(manifest_path.read_text())
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
for rel,digest in manifest['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+rel)
resolution_path=result/'baseline_resolution.json'; resolution=json.loads(resolution_path.read_text())
if resolution.get('status')!='RESOLVED' or resolution.get('choice')!=choice or not isinstance(resolution.get('user_authorization'),str) or not resolution['user_authorization'].strip():raise ValueError('Explicit resolved C01 user authorization required')
for field in ('C01_execution_manifest_sha256','C01_reg_max','C01_raw_output_channels_per_scale'):
 if resolution.get(field)!=cfg[field]:raise ValueError('Resolved frozen C01 identity differs: '+field)
complete=json.loads((result/'preflight_complete.json').read_text())
gate_path=result/'smoke_gate.json'; gate=json.loads(gate_path.read_text())
review_path=result/'independent_review.json'; review=json.loads(review_path.read_text())
if complete.get('status')!='PASS' or complete.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Actual preflight incomplete or binding differs')
if gate.get('status')!='PASS' or gate.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Actual smoke gate incomplete or binding differs')
if review.get('status')!='PASS' or review.get('independent') is not True or review.get('scope')!=scope or review.get('execution_manifest_sha256')!=manifest_sha:raise ValueError('Actual independent readiness review is missing or differs')
if review.get('smoke_gate_sha256')!=hashlib.sha256(gate_path.read_bytes()).hexdigest():raise ValueError('Reviewed smoke digest differs')
if review.get('preflight_complete_sha256')!=hashlib.sha256((result/'preflight_complete.json').read_bytes()).hexdigest():raise ValueError('Reviewed preflight completion digest differs')
gpu=cfg['gpu_index']
if gpu!=1:raise ValueError('C01b formal queue requires GPU1')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[v[1].strip(),int(v[2]),int(v[3])] for line in rows.splitlines() if (v:=line.split(','))}
if gpu not in gpus or gpus[gpu][0]!='NVIDIA GeForce RTX 5090' or gpus[gpu][1]>500 or gpus[gpu][2]>10:raise RuntimeError('GPU1 availability changed')
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
 try:
  argv=[v.decode(errors='replace') for v in (entry/'cmdline').read_bytes().split(bytes([0])) if v]
  if (entry/'cwd').resolve()==root and ('tools/sonar_head_c01b/train.py' in argv or any('tools/sonar_head_c01b/train.py' in a for a in argv)):raise RuntimeError('Existing campaign process')
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:
  if entry.stat().st_uid==os.getuid():raise
env=dict(os.environ,PYTHONPATH=os.pathsep.join([overlay,str(root),str(root/'src')]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLOv5_AUTOINSTALL='false')
command=[python,'tools/sonar_head_c01b/train.py','queue','--config',config_relative,'--output',str(result),'--device','cuda']
info={'status':'LAUNCHED','execution_manifest_sha256':manifest_sha,'baseline_resolution_sha256':hashlib.sha256(resolution_path.read_bytes()).hexdigest(),'independent_review_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),'smoke_gate_sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest(),'preflight_complete_sha256':hashlib.sha256((result/'preflight_complete.json').read_bytes()).hexdigest(),'gpu':gpu,'gpu_snapshot':gpus[gpu],'workers':[],'test_evaluated':False,'maximum_hours_per_seed':cfg['queue']['maximum_hours_per_run'],'retry':cfg['queue']['retry']}
with (result/'launch.json').open('x') as record:
 with (result/'worker_gpu1_stdout.log').open('x') as log:
  p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info['workers'].append({'gpu':gpu,'pid':p.pid,'command':command})
 record.write(json.dumps(info,indent=2))
print(json.dumps(info))
'''
    value = json.loads(execute(code, 45))
    dump(LOCAL / 'launch.json', value)
    return value


def status(*, preflight_only=False):
    """Compact files and liveness; no model import or checkpoint loading."""
    code = header() + f'preflight_only={preflight_only!r}\n' + '''
out={'root_exists':root.exists(),'result_exists':result.exists()}
names=['preflight_launch.json','preflight_complete.json','baseline_resolution.json','smoke_gate.json']
if not preflight_only:names+=['launch.json','queue_status.json','formal_authorization.json','c01b_summary.json','D17_HEAD_CONTRACT.json']
for name in names:
 path=result/name
 if not path.is_file():continue
 try:value=json.loads(path.read_text())
 except (ValueError,UnicodeError):out[name]={'status':'UNREADABLE_OR_INCOMPLETE_JSON'};continue
 if name=='c01b_summary.json':value={'status':value.get('status'),'n_runs':value.get('n_runs'),'decision':value.get('decision')}
 if name=='smoke_gate.json':value={k:v for k,v in value.items() if k not in ('reports','arms','history','checks')}
 for proc in value.get('workers',[value]):
  if 'pid' in proc:proc['alive']=Path('/proc',str(proc['pid'])).exists()
 out[name]=value
manifest_path=result/'execution_manifest.json'
if manifest_path.is_file():
 import yaml
 manifest=json.loads(manifest_path.read_text());cfg=yaml.safe_load((root/config_relative).read_text())
 baseline=yaml.safe_load((root/cfg['baseline']).read_text());test=root/baseline['dataset']['data_dir']/'test'
 mismatches=[rel for rel,digest in manifest['source_sha256'].items() if not (root/rel).is_file() or hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest]
 out['readonly_verification']={'execution_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'source_hashes_match':not mismatches,'source_mismatches':mismatches[:20],'test_path_exists':test.exists(),'test_path_is_symlink':test.is_symlink()}
if not preflight_only:
 for path in result.glob('formal/**/status.json'):
  try:value=json.loads(path.read_text())
  except ValueError:value={'status':'UNREADABLE_OR_INCOMPLETE_JSON'}
  for key in ('pid','child_pid'):
   if value.get(key):value[key+'_alive']=Path('/proc',str(value[key])).exists()
  out[path.relative_to(result).as_posix()]=value
logs=[result/'preflight_stdout.log']
if not preflight_only:logs+=[result/'worker_gpu1_stdout.log']
for path in logs:
 if path.is_file():
  with path.open('rb') as log:
   log.seek(max(0,path.stat().st_size-2500));out[path.name]=log.read().decode(errors='replace')
print(json.dumps(out))
'''
    value = json.loads(execute(code, 45))
    dump(LOCAL / ('preflight_status.json' if preflight_only else 'status.json'), value)
    return value


def preflight_status():
    return status(preflight_only=True)


def collect():
    """Download bounded evidence with SHA checks; leave all checkpoints remote."""
    code = header() + f'allowed={EVIDENCE_SUFFIXES!r}; maximum={MAX_EVIDENCE_BYTES!r}\n' + '''
files=[]; skipped=0
for path in sorted(result.rglob('*')):
 if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(result.resolve()):continue
 if path.suffix not in allowed:continue
 size=path.stat().st_size
 if size>=maximum:skipped+=1;continue
 files.append({'path':path.relative_to(result).as_posix(),'size_bytes':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps({'files':files,'skipped_oversize':skipped}))
'''
    inventory = json.loads(execute(code, 50))
    dest = CAMPAIGN_LOCAL / 'remote_snapshot_revision2'
    downloaded, reused = 0, 0
    for item in inventory['files']:
        name = item['path']
        path = (dest / name).resolve()
        if (not path.is_relative_to(dest.resolve()) or Path(name).is_absolute()
                or '\\' in name or any(c in name for c in ('\n', '\r', ':'))):
            raise ValueError('Evidence path escapes campaign or is invalid')
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            reused += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.collecting')
        subprocess.run(SCP + [HOST + ':' + REMOTE + '/' + RESULT + '/' + name, str(temporary)],
                       env=ENV, check=True, timeout=60)
        if (temporary.stat().st_size != item['size_bytes']
                or hashlib.sha256(temporary.read_bytes()).hexdigest() != item['sha256']):
            raise ValueError('Evidence changed during collection: ' + name)
        temporary.replace(path)
        downloaded += 1
    dump(LOCAL / 'collection_inventory.json', inventory)
    return dict(status='COLLECTED', files=len(inventory['files']), downloaded=downloaded,
                reused=reused, skipped_oversize=inventory['skipped_oversize'], local=str(dest),
                checkpoints_downloaded=False)


def exec_readonly(query='runtime', *, relative_path=None, tail_bytes=4000):
    """Bounded built-in diagnostics, rather than arbitrary remote Python execution."""
    if query not in ('runtime', 'files', 'log-tail', 'source-parity'):
        raise ValueError('Unsupported read-only query')
    if not isinstance(tail_bytes, int) or not 1 <= tail_bytes <= 16000:
        raise ValueError('Log tail must be between1 and16000 bytes')
    code = header() + f'query={query!r}; relative={relative_path!r}; tail_bytes={tail_bytes!r}\n' + '''
if query=='runtime':
 value={'root_exists':root.exists(),'python_exists':Path(python).is_file(),'overlay_exists':Path(overlay).is_dir(),'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
elif query=='files':
 items=sorted(p.relative_to(result).as_posix() for p in result.rglob('*') if p.is_file())
 value={'total_files':len(items),'files':items[:150],'truncated':len(items)>150}
elif query=='log-tail':
 if not isinstance(relative,str) or not relative:raise ValueError('A campaign-relative log path is required')
 path=(result/relative).resolve()
 if not path.is_relative_to(result.resolve()) or path.suffix!='.log':raise ValueError('Log path escapes result namespace')
 with path.open('rb') as log:
  log.seek(max(0,path.stat().st_size-tail_bytes));content=log.read().decode(errors='replace')
 value={'path':relative,'tail':content}
else:
 path=result/'execution_manifest.json';manifest=json.loads(path.read_text())
 missing=[rel for rel,digest in manifest['source_sha256'].items() if not (root/rel).is_file() or hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest]
 value={'execution_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_hashes_match':not missing,'mismatch_count':len(missing),'mismatches':missing[:30]}
print(json.dumps(value))
'''
    return json.loads(execute(code, 45))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'stage', 'preflight', 'preflight-status',
        'upload-resolution', 'upload-review', 'launch', 'status', 'collect', 'exec-readonly'))
    parser.add_argument('--file', type=Path, help='Local resolution/review JSON to upload')
    parser.add_argument('--query', choices=('runtime', 'files', 'log-tail', 'source-parity'), default='runtime')
    parser.add_argument('--path', help='Result-relative log path for log-tail')
    parser.add_argument('--tail-bytes', type=int, default=4000)
    args = parser.parse_args()
    if args.mode == 'exec-readonly':
        value = exec_readonly(args.query, relative_path=args.path, tail_bytes=args.tail_bytes)
    elif args.mode in ('upload-resolution', 'upload-review'):
        value = globals()[args.mode.replace('-', '_')](args.file)
    else:
        value = globals()[args.mode.replace('-', '_')]()
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
