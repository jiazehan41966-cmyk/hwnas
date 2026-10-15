"""Transport and launch the isolated sixteen-run stable-metric factorial."""
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
from tools.detector_v0_2.remote import execute, SCP, ENV, HOST, PYTHON, SOURCE

CAMPAIGN = 'sonar_head_factorial_20261012_02_stable'
REMOTE = '/home/jiajia/experiments/' + CAMPAIGN
RESULT = 'results/' + CAMPAIGN + '/remote'
LOCAL = ROOT / 'results' / CAMPAIGN / 'transport'
OVERLAY = '/home/jiajia/experiments/sonar_yolov9_baseline_20261005_remote_coco/results/sonar_yolov9_baseline_20261005/env_overlay'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return ('from pathlib import Path\nimport os,sys,json,subprocess,hashlib\n'
            f'root=Path({REMOTE!r}); result=root/{RESULT!r}\n'
            f'python={PYTHON!r}; overlay={OVERLAY!r}\n')


def source_files():
    """Pure filesystem closure: importing the transport executes no models."""
    frozen = json.loads((ROOT / 'reports/baseline_sonar_frozen/freeze_manifest.json').read_text(encoding='utf-8'))
    paths = {ROOT / p.replace('\\', '/') for p in frozen['source_sha256']}
    paths.update((ROOT / 'src/hwnas_fpga').rglob('*.py'))
    for name in ('sonar_spec', 'sonar_analysis', 'detector_b1_light', 'detector_elastic_pilot',
                 'sonar_yolov9_baseline', 'sonar_head_factorial'):
        paths.update((ROOT / 'tools' / name).glob('*.py'))
    author = ROOT / 'reference/yolov9'
    for pattern in ('*.py', '*.yaml', '*.yml'):
        paths.update(author.rglob(pattern))
    paths.update(ROOT / p for p in (
        'tools/detector_v0_2/train.py',
        'configs/sonar_head_factorial_20261012.yaml',
        'configs/sonar_head_factorial_environment.yaml',
        'configs/baseline_sonar_frozen.yaml',
        'configs/detector_elastic_pilot_model.yaml',
        'configs/detector_elastic_pilot_training.yaml',
        'configs/detector_elastic_pilot_environment.yaml',
        'configs/detector_b1_light_model.yaml',
        'configs/search_space_sonar_v0_1.yaml',
        'configs/supernet_sonar_v0_1_formal.yaml',
        'reports/baseline_sonar_frozen/freeze_manifest.json',
        'reports/phase2f_multiseed/multiseed_results.csv',
        'reference/yolov9/LICENSE.md',
        'tests/test_sonar_head_factorial_model.py',
        'tests/test_sonar_head_factorial_protocol.py',
        'tests/test_sonar_head_factorial_stable_analysis.py',
        'results/experiment0_similarity_supplement_20261012_01/subset_contract.json',
        'results/experiment0_eval_protocol_v1_20261012/data_audit/train_valid_pairs_36.csv',
        'results/experiment0_eval_protocol_v1_20261012/image_order.json',
        'results/experiment0_eval_protocol_v1_20261012/data_audit/image_label_counts_429.csv'))
    import yaml
    baseline = yaml.safe_load((ROOT / 'configs/baseline_sonar_frozen.yaml').read_text(encoding='utf-8'))
    paths.add(ROOT / baseline['dataset']['data_dir'] / 'data.yaml')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def stage():
    paths = source_files()
    LOCAL.mkdir(parents=True, exist_ok=True)
    marker = dict(source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  test_mounted=False, trained_weights_copied=False)
    dump(LOCAL / 'staging_manifest.json', marker)
    receipt = LOCAL / 'author_source_receipt.json'
    dump(receipt, dict(commit='5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff',
                       source='https://github.com/WongKinYiu/yolov9', source_files_bound_by_staging_manifest=True))
    archive = LOCAL / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json')
        tar.add(receipt, arcname='reference/yolov9/source_receipt.json')
    execute(header() + "if root.exists(): raise FileExistsError('Namespace exists; preserve it')\nroot.mkdir()\n", 30)
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/source.tar.gz'], env=ENV, check=True, timeout=60)
    code = header() + f'source=Path({SOURCE!r})\n' + '''
import tarfile,yaml
with tarfile.open(root/'source.tar.gz') as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()):raise ValueError('Unsafe source archive')
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(tar.extractfile(member).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,expected in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Transport mismatch: '+rel)
baseline=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/baseline['dataset']['data_dir']
for split in ('train','valid'):
 target=(source/baseline['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
if not Path(overlay).is_dir():raise FileNotFoundError('Existing YOLO overlay absent')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False,'weights_copied':False}))
'''
    return json.loads(execute(code, 50))


def probe():
    return json.loads(execute(header() + '''
import shutil
print(json.dumps({'root_exists':root.exists(),'overlay_exists':Path(overlay).is_dir(),
 'result_volume_free_bytes':shutil.disk_usage(root.parent).free,
 'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))
''', 30))


def preflight():
    code = header() + '''
import yaml
cfg=yaml.safe_load((root/'configs/sonar_head_factorial_20261012.yaml').read_text())
gpu=cfg['gpu_indices'][0]
result.mkdir(parents=True,exist_ok=True)
if (result/'preflight_launch.json').exists():raise FileExistsError('Preflight already invoked')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[int(x) for x in v[1:]] for line in rows.splitlines() if (v:=line.split(','))}
if gpus[gpu][0]>500 or gpus[gpu][1]>10:raise RuntimeError('Preflight GPU occupied')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([overlay,str(root),str(root/'src')]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLOv5_AUTOINSTALL='false')
commands=[
 [python,'tools/sonar_head_factorial/train.py','prepare','--output',str(result),'--device','cuda'],
 [python,'tests/test_sonar_head_factorial_protocol.py'],
 [python,'-m','pytest','-q','tests/test_sonar_head_factorial_model.py','tests/test_sonar_head_factorial_stable_analysis.py'],
 [python,'tools/sonar_head_factorial/train.py','smoke','--output',str(result),'--device','cuda']]
supervisor="import subprocess,json,pathlib; commands="+repr(commands)+"; result=pathlib.Path("+repr(str(result))+"); [subprocess.run(c,check=True,timeout=1200) for c in commands]; (result/'preflight_complete.json').write_text(json.dumps({'status':'PASS','commands':commands}))"
with (result/'preflight_stdout.log').open('x') as log:
 p=subprocess.Popen([python,'-c',supervisor],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'status':'PREFLIGHT_LAUNCHED','pid':p.pid,'gpu':gpu,'commands':commands}
(result/'preflight_launch.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
'''
    value = json.loads(execute(code, 40))
    dump(LOCAL / 'preflight_launch.json', value)
    return value


def upload_review():
    review = ROOT / 'results' / CAMPAIGN / 'independent_review.json'
    content = review.read_bytes()
    value = json.loads(content)
    if value.get('status') != 'PASS':
        raise ValueError('Independent review is not PASS')
    code = header() + f'content={base64.b64encode(content).decode()!r}\n' + '''
import base64
payload=base64.b64decode(content); review=json.loads(payload)
manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
if review['execution_manifest_sha256']!=manifest_sha:raise ValueError('Review manifest binding differs')
path=result/'independent_review.json'
if path.exists() and path.read_bytes()!=payload:raise FileExistsError('Review already present')
path.write_bytes(payload)
print(json.dumps({'status':'REVIEW_UPLOADED','sha256':hashlib.sha256(payload).hexdigest()}))
'''
    return json.loads(execute(code, 30))


def launch():
    code = header() + '''
sys.path[:0]=[str(root),overlay]
import yaml
cfg=yaml.safe_load((root/'configs/sonar_head_factorial_20261012.yaml').read_text())
if (result/'launch.json').exists():raise FileExistsError('Campaign already launched')
manifest=json.loads((result/'execution_manifest.json').read_text())
manifest_sha=hashlib.sha256((result/'execution_manifest.json').read_bytes()).hexdigest()
for rel,digest in manifest['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
complete=json.loads((result/'preflight_complete.json').read_text())
gate=json.loads((result/'smoke_gate.json').read_text())
review=json.loads((result/'independent_review.json').read_text())
if complete['status']!='PASS' or gate['status']!='PASS' or review['status']!='PASS':raise ValueError('Preflight/review not passed')
if gate['execution_manifest_sha256']!=manifest_sha or review['execution_manifest_sha256']!=manifest_sha:raise ValueError('Preflight/review binding differs')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[v[1].strip(),int(v[2]),int(v[3])] for line in rows.splitlines() if (v:=line.split(','))}
for gpu in cfg['gpu_indices']:
 if gpus[gpu][0]!='NVIDIA GeForce RTX 5090' or gpus[gpu][1]>500 or gpus[gpu][2]>10:raise RuntimeError('GPU availability changed')
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
 try:
  argv=[v.decode(errors='replace') for v in (entry/'cmdline').read_bytes().split(bytes([0])) if v]
  if (entry/'cwd').resolve()==root and 'tools/sonar_head_factorial/train.py' in argv:raise RuntimeError('Existing campaign process')
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:
  if entry.stat().st_uid==os.getuid():raise
info={'status':'LAUNCHED','execution_manifest_sha256':manifest_sha,'workers':[],'gpu_snapshot':gpus,'test_evaluated':False}
for shard,gpu in enumerate(cfg['gpu_indices']):
 env=dict(os.environ,PYTHONPATH=os.pathsep.join([overlay,str(root),str(root/'src')]),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',YOLOv5_AUTOINSTALL='false')
 command=[python,'tools/sonar_head_factorial/train.py','queue','--output',str(result),'--device','cuda','--shard-index',str(shard),'--shard-count',str(len(cfg['gpu_indices']))]
 with (result/f'worker_gpu{gpu}_stdout.log').open('x') as log:
  p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info['workers'].append({'gpu':gpu,'pid':p.pid,'command':command})
(result/'launch.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info))
'''
    value = json.loads(execute(code, 45))
    dump(LOCAL / 'launch.json', value)
    return value


def status():
    code = header() + '''
out={}
manifest_path=result/'execution_manifest.json'
if manifest_path.is_file():
 import yaml
 manifest=json.loads(manifest_path.read_text())
 baseline=yaml.safe_load((root/manifest['config']['baseline']).read_text())
 test_path=root/baseline['dataset']['data_dir']/'test'
 out['runtime_readonly_verification']={
  'execution_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
  'test_path_exists':test_path.exists(),'test_path_is_symlink':test_path.is_symlink(),
  'source_hashes_match':all(hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest for rel,digest in manifest['source_sha256'].items())}
for name in ('preflight_launch.json','preflight_complete.json','smoke_gate.json','launch.json','factorial_summary.json','extension_decision.json'):
 path=result/name
 if path.exists():out[name]=json.loads(path.read_text())
for key in ('preflight_launch.json','launch.json'):
 if key in out:
  for proc in out[key].get('workers',[out[key]]):
   if 'pid' in proc:proc['alive']=Path('/proc',str(proc['pid'])).exists()
for path in list(result.glob('formal/**/status.json'))+list(result.glob('*queue*status*.json')):
 out[path.relative_to(result).as_posix()]=json.loads(path.read_text())
 for key in ('pid','child_pid'):
  pid=out[path.relative_to(result).as_posix()].get(key)
  if pid:out[path.relative_to(result).as_posix()][key+'_alive']=Path('/proc',str(pid)).exists()
for path in [result/'preflight_stdout.log',*result.glob('worker_gpu*_stdout.log')]:
 if path.exists():out[path.name]=path.read_text()[-4000:]
for path in result.glob('formal/**/stdout.log'):
 if path.exists():out[path.relative_to(result).as_posix()]=path.read_text()[-1800:]
print(json.dumps(out))
'''
    value = json.loads(execute(code, 40))
    dump(LOCAL / 'status.json', value)
    return value


def collect():
    # Small evidence files only, model checkpoints stay on the server.
    code = header() + '''
import base64
files=[p for p in result.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.log','.npz') and p.stat().st_size<12000000]
print(json.dumps({p.relative_to(result).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}))
'''
    files = json.loads(execute(code, 55))
    dest = ROOT / 'results' / CAMPAIGN / 'remote_snapshot'
    for name, content in files.items():
        path = (dest / name).resolve()
        if not path.is_relative_to(dest.resolve()):
            raise ValueError('Evidence path escapes campaign')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(content))
    return dict(status='COLLECTED', files=len(files), local=str(dest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe','stage','preflight','upload-review','launch','status','collect'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.mode.replace('-', '_')](), ensure_ascii=False))
