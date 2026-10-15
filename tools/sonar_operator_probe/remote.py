"""Bounded isolated transport through the existing credential-safe SSH bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from tools.sonar_head_factorial import remote as prior

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = 'sonar_operator_probe_20261013_01'
REMOTE = '/home/jiajia/experiments/' + CAMPAIGN + '_revision2'
RESULT = 'results/' + CAMPAIGN + '/remote'
CONFIG = 'configs/sonar_operator_probe_20261013.yaml'
LOCAL = ROOT / 'results' / CAMPAIGN / 'transport_revision2'
RECEIPT = ROOT / 'results/sonar_head_factorial_20261012_02_stable/transport/author_source_receipt.json'
execute, SCP, ENV, HOST, PYTHON = prior.execute, prior.SCP, prior.ENV, prior.HOST, prior.PYTHON
SOURCE, OVERLAY = prior.SOURCE, prior.OVERLAY


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r});result=root/{RESULT!r};python={PYTHON!r};overlay={OVERLAY!r}\n')


def source_files():
    paths = set(prior.source_files())
    from tools.sonar_operator_probe import train
    cfg, _ = train.load_config(ROOT / CONFIG)
    paths.update(train.source_paths(cfg, ROOT / CONFIG))
    for directory in ('tools/sonar_head_c01b', 'tools/sonar_operator_probe'):
        paths.update((ROOT / directory).glob('*.py'))
    paths.update((ROOT / 'tests').glob('test_sonar_operator_probe_*.py'))
    paths.update(ROOT / rel for rel in (
        CONFIG, 'configs/sonar_head_c01b_20261013.yaml',
        'tools/sonar_joint192_validation/model.py',
        'tools/sonar_joint192_extension_20261013/model.py',
        'tools/sonar_operator_probe/README.md',
        'results/' + CAMPAIGN + '/PREREGISTRATION.md'))
    for path in paths:
        if (not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve())
                or path.suffix in ('.pt', '.pth', '.ckpt')):
            raise ValueError('Invalid bounded source member: ' + str(path))
    return sorted(paths)


def stage():
    """Create a new namespace once; no overwrite of an existing campaign."""
    paths = source_files()
    receipt_bytes = RECEIPT.read_bytes()
    receipt_arc = 'reference/yolov9/source_receipt.json'
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.relative_to(ROOT).as_posix() != receipt_arc}
    hashes[receipt_arc] = hashlib.sha256(receipt_bytes).hexdigest()
    marker = dict(campaign=CAMPAIGN, source_sha256=hashes, remote_namespace=REMOTE,
                  test_mounted=False, trained_weights_copied=False,
                  dependency_installation=False, formal_eligible=False)
    LOCAL.mkdir(parents=True, exist_ok=True)
    archive = LOCAL / 'source.tar.gz'
    if archive.exists():
        raise FileExistsError('Preserve the existing staging archive')
    dump(LOCAL / 'staging_manifest.json', marker)
    (LOCAL / 'author_source_receipt.json').write_bytes(receipt_bytes)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if relative != receipt_arc:
                tar.add(path, arcname=relative, recursive=False)
        tar.add(LOCAL / 'author_source_receipt.json', arcname=receipt_arc, recursive=False)
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json', recursive=False)
    execute(header() + "if root.exists() or root.is_symlink():raise FileExistsError('Preserve existing namespace')\nroot.mkdir()\n", 30)
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/source.tar.gz'],
                   env=ENV, check=True, timeout=60)
    code = header() + f'source=Path({SOURCE!r})\n' + '''
import tarfile,yaml
with tarfile.open(root/'source.tar.gz') as tar:
 seen=set()
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()) or member.name in seen:
   raise ValueError('Unsafe or duplicate source archive member')
  seen.add(member.name);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(tar.extractfile(member).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Source transport mismatch: '+rel)
baseline=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/baseline['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ('train','valid'):
 target=(source/baseline['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test must not be mounted')
if not Path(python).is_file() or not Path(overlay).is_dir():raise FileNotFoundError('Existing runtime missing')
print(json.dumps({'status':'STAGED','source_files':len(marker['source_sha256']),'test_mounted':False,
 'trained_weights_copied':False,'dependency_installation':False,'formal_eligible':False}))
'''
    value = json.loads(execute(code, 55))
    dump(LOCAL / 'stage_result.json', value)
    return value


def probe():
    code = header() + '''
from datetime import datetime,timezone
print(json.dumps({'observed_at_utc':datetime.now(timezone.utc).isoformat(),
 'root_exists':root.exists(),'python_exists':Path(python).is_file(),'overlay_exists':Path(overlay).is_dir(),
 'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 'owners':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)}))
'''
    return json.loads(execute(code, 40))


def dispatch(phase):
    if phase not in ('preflight', 'queue', 'campaign'):
        raise ValueError('Explicit preflight, queue or bounded campaign required')
    code = header() + f'phase={phase!r};config_relative={CONFIG!r}\n' + '''
import yaml
from datetime import datetime,timezone
cfg=yaml.safe_load((root/config_relative).read_text())
if cfg['queue']['allowed_gpus']!=[1]:raise ValueError('Only authorized GPU1')
result.mkdir(parents=True,exist_ok=True)
path=result/(phase+'_launch.json')
if path.exists():raise FileExistsError('No duplicate dispatch or automatic retry')
lines=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
uuid,memory=lines[0].strip(),float(lines[1])
owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
other=[line for line in owners.splitlines() if line.split(',')[0].strip()==uuid]
if phase!='campaign' and (other or memory>cfg['queue']['gpu_memory_admission_mib']):raise RuntimeError('GPU1 occupied; do not displace existing jobs')
if phase=='queue':
 review=json.loads((result/'readiness_review.json').read_text())
 if review.get('status')!='PASS':raise ValueError('New campaign readiness review missing')
 for name,key in [('execution_manifest.json','execution_manifest_sha256'),('smoke_gate.json','smoke_gate_sha256')]:
  if hashlib.sha256((result/name).read_bytes()).hexdigest()!=review.get(key):raise ValueError('Review binding drift: '+name)
 preflight=json.loads((result/'preflight_supervisor_status.json').read_text())
 if preflight['status']!='COMPLETE':raise ValueError('Actual preflight not complete')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 PYTHONPATH=os.pathsep.join((overlay,str(root),str(root/'src'),str(root/'scripts'))),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
 YOLOv5_AUTOINSTALL='false',YOLOM='10',YOLOA='0.5',YOLOB='6.0')
cache=result/'runtime_cache';cache.mkdir(parents=True,exist_ok=True)
env.update(MPLCONFIGDIR=str(cache/'mpl'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(cache/'tmp'))
for relative in ('mpl','torch','xdg','tmp'):(cache/relative).mkdir(exist_ok=True)
command=[python,'-B','tools/sonar_operator_probe/supervisor.py',phase,'--config',config_relative,'--output',str(result)]
with (result/(phase+'_stdout.log')).open('xb') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'status':phase.upper()+'_LAUNCHED','pid':child.pid,'physical_gpu':1,'command':command,
 'launched_at_utc':datetime.now(timezone.utc).isoformat(),'formal_eligible':False,'test_evaluated':False}
path.write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
    value = json.loads(execute(code, 45))
    dump(LOCAL / (phase + '_launch.json'), value)
    return value


def status():
    code = header() + '''
from datetime import datetime,timezone
out={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'root_exists':root.exists()}
names=['campaign_launch.json','campaign_supervisor_status.json','resource_wait_status.json',
 'preflight_launch.json','preflight_supervisor_status.json','smoke_gate.json','queue_launch.json',
 'queue_supervisor_status.json','queue_status.json','campaign_summary.json']
for name in names:
 path=result/name
 if not path.is_file():continue
 value=json.loads(path.read_text())
 if name=='smoke_gate.json':value={key:value.get(key) for key in ('status','variants','execution_manifest_sha256','environment_fingerprint','test_evaluated')}
 if value.get('pid'):value['pid_alive']=Path('/proc',str(value['pid'])).exists()
 out[name]=value
for path in result.glob('runs/**/status.json'):
 value=json.loads(path.read_text());out[path.relative_to(result).as_posix()]=value
for phase in ('preflight','queue','campaign'):
 path=result/(phase+'_stdout.log')
 if path.is_file():
  with path.open('rb') as stream:
   stream.seek(max(0,path.stat().st_size-5000));out[path.name]=stream.read().decode(errors='replace')
out['gpu1']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
'''
    value = json.loads(execute(code, 45))
    dump(LOCAL / 'status.json', value)
    return value


def collect():
    """Collect small receipts only; weights and full histories remain remote."""
    names = ('execution_manifest.json','environment_lock.json','prepare_admission.json','campaign_launch.json',
             'campaign_supervisor_status.json','resource_wait_status.json',
             'preflight_supervisor_status.json','smoke_gate.json','readiness_review.json',
             'queue_status.json','preflight_launch.json','queue_launch.json','preflight_stdout.log','queue_supervisor_status.json')
    code = header() + f'names={names!r}\n' + '''
files=[name for name in names if (result/name).is_file() and (result/name).stat().st_size<2000000]
print(json.dumps(files))
'''
    files = json.loads(execute(code, 30))
    destination = ROOT / 'results' / CAMPAIGN / 'remote_snapshot_revision2'
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        subprocess.run(SCP + [HOST + ':' + REMOTE + '/' + RESULT + '/' + name, str(destination / name)],
                       env=ENV, check=True, timeout=40)
    dump(LOCAL / 'collection_receipt.json', dict(files=files, local=str(destination)))
    return dict(files=files, local=str(destination))


def upload_review(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('status') != 'PASS':
        raise ValueError('Explicit new readiness review must pass')
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    code = header() + f'payload={payload!r}\n' + '''
value=json.loads(payload)
for name,key in [('execution_manifest.json','execution_manifest_sha256'),('smoke_gate.json','smoke_gate_sha256')]:
 if hashlib.sha256((result/name).read_bytes()).hexdigest()!=value.get(key):raise ValueError('Readiness review binding drift')
path=result/'readiness_review.json'
with path.open('x') as stream:stream.write(payload)
print(json.dumps({'status':'REVIEW_UPLOADED','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
'''
    return json.loads(execute(code, 40))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe','stage','preflight','status','collect','launch','campaign','upload-review'))
    parser.add_argument('--review')
    args = parser.parse_args()
    if args.mode == 'preflight':
        value = dispatch('preflight')
    elif args.mode == 'launch':
        value = dispatch('queue')
    elif args.mode == 'campaign':
        value = dispatch('campaign')
    elif args.mode == 'upload-review':
        value = upload_review(args.review)
    else:
        value = globals()[args.mode]()
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
