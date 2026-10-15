"""Isolated COCO fine-tune deployment via the existing credential transport."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.detector_v0_2.remote import execute, SSH, SCP, ENV, HOST, PYTHON, SOURCE

REMOTE = '/home/jiajia/experiments/sonar_yolov9_baseline_20261005_remote_coco'
BASE = 'results/sonar_yolov9_baseline_20261005'
RESULT = BASE + '/remote_coco'
LOCAL = ROOT / BASE / 'remote_coco'
GROUPS = ('L1_scratch', 'YOLOv9_T_scratch', 'YOLOv9_T_COCO')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return ('from pathlib import Path\nimport json,os,subprocess,hashlib,sys\n'
            + f'root=Path({REMOTE!r}); result=root/{RESULT!r}; base=root/{BASE!r}\n'
            + f'python={PYTHON!r}\n')


def probe():
    return execute(header() + """
import importlib.metadata
names=['torch','torchvision','numpy','Pillow','PyYAML','matplotlib','pandas','scipy']
packages={}
for name in names:
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
print(json.dumps({'root_exists':root.exists(),'interpreter':sys.executable,
 'python':sys.version,'packages':packages,
 'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)}))
""", 40)


def stage():
    local_base = ROOT / BASE
    frozen = json.loads((local_base / 'execution_manifest.json').read_text(encoding='utf-8'))
    source_hashes = frozen['source_sha256']
    paths = {ROOT / name for name in source_hashes}
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Local source freeze changed: ' + name)
    cfg = frozen['config']
    paths.add(ROOT / cfg['author']['checkpoint'])
    data_dir = cfg['baseline']
    import yaml
    baseline = yaml.safe_load((ROOT / data_dir).read_text(encoding='utf-8'))
    paths.add(ROOT / baseline['dataset']['data_dir'] / 'data.yaml')
    paths.add(ROOT / 'tools/sonar_yolov9_remote/launch_remote.py')
    paths.add(ROOT / 'tools/sonar_yolov9_remote/remote.py')
    spec = dict(version='sonar_yolov9_remote_coco_20261005_v1',
                provider='existing_user_owned_SSH_server', base_interpreter=PYTHON,
                base_pins={'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128', 'numpy': '2.4.4'},
                overlay=BASE + '/env_overlay', gpu=2, threads=2,
                pip_phases=json.loads((local_base / 'environment_spec.json').read_text(encoding='utf-8'))['pip_phases'] + [['Pillow==12.2.0']],
                installation='--target --no-deps; shared base unchanged',
                local_environment_files='frozen provenance only; actual remote environment in remote_environment_lock.json',
                comparison_scope='same_protocol_different_environment_not_paired_pretraining_effect',
                smoke_groups=list(GROUPS), test_mounted=False)
    spec_path = local_base / 'remote_environment_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf-8')) != spec:
        raise ValueError('Remote environment spec already differs')
    dump(spec_path, spec)
    paths.add(spec_path)
    LOCAL.mkdir(parents=True, exist_ok=True)
    manifest = dict(version='REMOTE_COCO_TRANSPORT_V1',
                    original_execution_manifest_sha256=sha(local_base / 'execution_manifest.json'),
                    source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)},
                    remote_root=REMOTE, result_root=RESULT, test_mounted=False,
                    train_valid_data_source=SOURCE)
    dump(LOCAL / 'staging_manifest.json', manifest)
    receipt = LOCAL / 'source_receipt.json'
    dump(receipt, {'commit': cfg['author']['commit'], 'source': cfg['author']['repository'],
                   'transport': 'verified frozen file bytes, no git metadata'})
    archive = LOCAL / 'training_source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(paths):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json')
        tar.add(receipt, arcname='reference/yolov9/source_receipt.json')
    execute(header() + 'root.mkdir(parents=True,exist_ok=True)\n', 30)
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/training_source.tar.gz'], env=ENV, check=True)
    code = header() + """
import tarfile,yaml
if (result/'execution_manifest.json').exists() or (result/'launch.json').exists():
 raise RuntimeError('Existing frozen remote campaign; restaging refused')
with tarfile.open(root/'training_source.tar.gz') as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root.resolve()):raise ValueError('Unsafe archive entry')
  contents=tar.extractfile(member).read()
  if target.exists() and target.read_bytes()!=contents:raise ValueError('Existing remote file differs: '+member.name)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(contents)
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Transport SHA mismatch: '+rel)
b=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/b['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
""" + f'source=Path({SOURCE!r})/b["dataset"]["data_dir"]\n' + """
for split in ('train','valid'):
 link=data/split;target=source/split
 if not target.is_dir():raise ValueError('Required train/valid data absent')
 if link.exists() or link.is_symlink():
  if not link.is_symlink() or link.resolve()!=target.resolve():raise ValueError('Existing split differs')
 else:link.symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test must remain unmounted')
print(json.dumps({'status':'STAGED','files':len(marker['source_sha256']),'test_mounted':False,'remote_root':str(root)}))
"""
    return execute(code, 90)


def environment():
    code = header() + """
import importlib.metadata
spec=json.loads((base/'remote_environment_spec.json').read_text())
for name,version in spec['base_pins'].items():
 if importlib.metadata.version(name)!=version:raise ValueError('Remote base pin mismatch: '+name)
overlay=root/spec['overlay'];overlay.mkdir(parents=True,exist_ok=True)
if (base/'remote_environment_lock.json').exists():raise ValueError('Environment already locked; no silent reinstall')
base_versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
with (result/'environment_install.log').open('x') as log:
 for phase in spec['pip_phases']:
  subprocess.run([python,'-m','pip','install','--disable-pip-version-check','--no-deps','--only-binary=:all:','--target',str(overlay),*phase],stdout=log,stderr=subprocess.STDOUT,check=True)
for name,version in spec['base_pins'].items():
 if importlib.metadata.version(name)!=version:raise ValueError('Shared base drifted')
os.chdir(root);sys.path.insert(0,str(root));os.environ['CUDA_VISIBLE_DEVICES']=str(spec['gpu'])
from tools.sonar_yolov9_baseline import train
from tools.sonar_yolov9_baseline import protocol as p
lock=dict(remote_environment_spec_canonical_sha256=p.canonical_sha(spec),
 base_interpreter=python,base_packages=base_versions,
 overlay_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(overlay)])},
 environment=train.environment('cuda'),test_evaluated=False,local_frozen_spec_is_provenance_only=True)
p.dump(base/'remote_environment_lock.json',lock)
print(json.dumps({'status':'ENV_INSTALLED_NOT_YET_READY','environment':lock['environment'],
 'remote_environment_spec_canonical_sha256':lock['remote_environment_spec_canonical_sha256']}))
"""
    code = code.replace("with (result/'environment_install.log')", "result.mkdir(parents=True,exist_ok=True)\nwith (result/'environment_install.log')")
    return execute(code, 600)


def invoke(mode, group='YOLOv9_T_COCO'):
    if mode not in ('prepare', 'verify', 'smoke') or group not in GROUPS:
        raise ValueError('Unsupported preflight')
    command = [PYTHON, 'tools/sonar_yolov9_baseline/train.py', mode, '--result-root', RESULT,
               '--group', group, '--seed', '42', '--device', 'cuda']
    code = header() + f'command={command!r}\n' + """
spec=json.loads((base/'remote_environment_spec.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(spec['gpu']),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
result.mkdir(parents=True,exist_ok=True)
""" + f'logfile=result/{(mode + "_" + group + ".log")!r}\n' + """
with logfile.open('x') as log:
 proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
if proc.returncode:raise RuntimeError('Preflight failed: '+logfile.read_text()[-7000:])
"""
    if mode == 'smoke':
        code += f"r=json.loads((result/'smoke'/{group!r}/'seed_42/smoke_report.json').read_text())\n" + """
print(json.dumps({'status':r['status'],'group':r['binding']['group'],'manifest_sha256':r['binding']['execution_manifest_sha256'],
 'seconds':r['seconds'],'peak_reserved_bytes':r['peak_cuda_reserved_bytes'],'images':r['training']['images'],
 'image_order_sha256':r['training']['image_order_sha256'],'main_graph':'verified_by_make_run','test_evaluated':r['test_evaluated']}))
"""
    else:
        code += "print(logfile.read_text()[-2500:])\n"
    return execute(code, 600)


def launch():
    command = [PYTHON, 'tools/sonar_yolov9_remote/launch_remote.py', '--result-root', RESULT, '--gpu', '2']
    return execute(header() + f'subprocess.run({command!r},cwd=root,check=True)\n', 60)


def status():
    return execute(header() + """
value={}
for relative in ('launch.json','formal/YOLOv9_T_COCO/seed_42/status.json','formal/YOLOv9_T_COCO/seed_42/result.json'):
 path=result/relative
 if path.exists():value[relative]=json.loads(path.read_text())
value['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
if 'launch.json' in value:
 pid=value['launch.json']['pid'];value['process_alive']=Path('/proc')/str(pid)
 value['process_alive']=value['process_alive'].exists()
log=result/'formal/YOLOv9_T_COCO/seed_42/stdout.log'
if log.exists():value['stdout_tail']=log.read_text()[-2000:]
print(json.dumps(value))
""", 40)


def fetch():
    """Copy evidence and logs, without syncing checkpoints or data."""
    payload = execute(header() + """
import base64
paths=[base/'remote_environment_lock.json',result/'execution_manifest.json',result/'launch.json',result/'independent_review.json',result/'independent_review.md']
paths += list((result/'smoke').rglob('*.json'))
paths += list((result/'formal').rglob('*.json')) + list((result/'formal').rglob('*.jsonl')) + list((result/'formal').rglob('stdout.log'))
print(json.dumps({p.relative_to(root).as_posix():base64.b64encode(p.read_bytes()).decode() for p in paths if p.is_file()}))
""", 60)
    import base64
    files = json.loads(payload)
    for relative, content in files.items():
        target = (ROOT / relative).resolve()
        if not target.is_relative_to((ROOT / BASE).resolve()):
            raise ValueError('Fetch path outside campaign')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(content))
    return json.dumps({'status': 'FETCHED_EVIDENCE', 'files': len(files), 'local': str(LOCAL)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe','stage','environment','prepare','verify','smoke','launch','status','fetch'))
    parser.add_argument('--group', choices=GROUPS, default='YOLOv9_T_COCO')
    args = parser.parse_args()
    if args.mode in ('prepare','verify','smoke'):
        value = invoke(args.mode, args.group)
    else:
        value = globals()[args.mode]()
    print(value)


if __name__ == '__main__':
    main()
