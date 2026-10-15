"""Additive transport for an explicitly authorized parallel resource amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from tools.sonar_operator_probe import remote as base

ROOT = base.ROOT
CAMPAIGN = 'sonar_operator_probe_parallel_20261014_01'
REMOTE = '/home/jiajia/experiments/' + CAMPAIGN + '_revision2'
OUTPUT = base.REMOTE + '/results/' + CAMPAIGN + '/remote_revision2'
LOCAL = ROOT / 'results' / CAMPAIGN
TOOL = 'tools/sonar_operator_probe_parallel_20261014'
AMENDMENT = 'results/' + CAMPAIGN + '/RESOURCE_AMENDMENT_REVISION2.json'
REVIEW = 'review/STATIC_REVIEW_REVISION2.json'
TRANSPORT = 'transport_revision2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def header():
    return base.header() + (f'parallel_root=Path({REMOTE!r});output=Path({OUTPUT!r})\n'
        f'runner=parallel_root/{(TOOL + "/runner.py")!r};amendment=parallel_root/{AMENDMENT!r}\n')


def stage():
    paths = [ROOT / TOOL / name for name in ('runner.py', 'remote.py', 'README.md')]
    paths += [ROOT / AMENDMENT, LOCAL / 'LEGACY_HANDOFF.json', LOCAL / REVIEW]
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Missing or invalid new transport member: ' + str(path))
    review = json.loads((LOCAL / REVIEW).read_text(encoding='utf-8'))
    if review['status'] != 'PASS' or review['wrapper_source_sha256'] != sha(ROOT / TOOL / 'runner.py'):
        raise ValueError('Independent parallel-wrapper review missing or stale')
    for path in paths:
        if path == LOCAL / REVIEW:
            continue
        if review['reviewed_source_sha256'].get(path.relative_to(ROOT).as_posix()) != sha(path):
            raise ValueError('Reviewed parallel transport member changed: ' + str(path))
    marker = dict(scope='parallel_training_resource_amendment', remote_namespace=REMOTE,
        frozen_base_namespace=base.REMOTE, output_namespace=OUTPUT,
        source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
        original_sources_modified=False, test_evaluated=False, formal_eligible=False)
    archive = LOCAL / TRANSPORT / 'source.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError('Preserve existing parallel staging archive')
    dump(archive.parent / 'staging_manifest.json', marker)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        tar.add(archive.parent / 'staging_manifest.json', arcname='staging_manifest.json', recursive=False)
    base.execute(header() + "if parallel_root.exists() or parallel_root.is_symlink():raise FileExistsError('Preserve existing namespace')\nparallel_root.mkdir()\n", 30)
    subprocess.run(base.SCP + [str(archive), base.HOST + ':' + REMOTE + '/source.tar.gz'],
                   env=base.ENV, check=True, timeout=60)
    value = json.loads(base.execute(header() + '''
import tarfile
with tarfile.open(parallel_root/'source.tar.gz') as tar:
 seen=set()
 for member in tar.getmembers():
  target=(parallel_root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(parallel_root.resolve()) or member.name in seen:
   raise ValueError('Unsafe or duplicate transport member')
  seen.add(member.name);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(tar.extractfile(member).read())
marker=json.loads((parallel_root/'staging_manifest.json').read_text())
for name,digest in marker['source_sha256'].items():
 if hashlib.sha256((parallel_root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Parallel source transport mismatch')
print(json.dumps({'status':'STAGED','source_files':len(marker['source_sha256']),'original_sources_modified':False}))
''', 40))
    dump(LOCAL / TRANSPORT / 'stage_result.json', value)
    return value


def invocation(mode, gpu=None):
    args = [base.PYTHON, '-B', REMOTE + '/' + TOOL + '/runner.py', mode,
        '--base-root', base.REMOTE, '--base-output', base.REMOTE + '/' + base.RESULT,
        '--output', OUTPUT, '--config', base.REMOTE + '/' + base.CONFIG]
    if mode == 'prepare':
        args += ['--amendment', REMOTE + '/' + AMENDMENT]
    if gpu is not None:
        args += ['--gpu', str(gpu)]
    return args


def environment_code(gpu):
    return f'physical_gpu={gpu!r}\n' + '''
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 PYTHONPATH=os.pathsep.join((overlay,str(root),str(root/'src'),str(root/'scripts'))),
 PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',YOLOv5_AUTOINSTALL='false',YOLOM='10',YOLOA='0.5',YOLOB='6.0')
if physical_gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(physical_gpu)
cache=output/'runtime_cache';cache.mkdir(parents=True,exist_ok=True)
for name in ('mpl','torch','xdg','tmp'):(cache/name).mkdir(exist_ok=True)
env.update(MPLCONFIGDIR=str(cache/'mpl'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(cache/'tmp'))
'''


def invoke(mode, gpu=None):
    code = header() + environment_code(gpu) + f'command={invocation(mode, gpu)!r}\n' + '''
completed=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
if completed.returncode:raise RuntimeError(completed.stdout[-10000:])
print(completed.stdout)
'''
    text = base.execute(code, 55)
    value = json.loads(text.strip().splitlines()[-1])
    dump(LOCAL / TRANSPORT / f'{mode}_gpu{gpu if gpu is not None else "cpu"}.json', value)
    return value


def dispatch(mode, gpu=None):
    if mode not in ('slot-smoke', 'scheduler') or (mode == 'slot-smoke' and gpu not in (2, 3)):
        raise ValueError('Only new GPU2/3 slot smoke or bounded scheduler may be dispatched')
    key = f'{mode}_gpu{gpu}' if gpu is not None else mode
    code = header() + environment_code(gpu) + f'command={invocation(mode, gpu)!r};key={key!r};mode={mode!r}\n' + '''
from datetime import datetime,timezone
marker=json.loads((parallel_root/'staging_manifest.json').read_text())
for name,digest in marker['source_sha256'].items():
 if hashlib.sha256((parallel_root/name).read_bytes()).hexdigest()!=digest:raise ValueError('New source changed after review')
receipt=output/(key+'_launch.json')
if receipt.exists():raise FileExistsError('No duplicate dispatch or automatic retry')
with (output/(key+'_stdout.log')).open('xb') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='LAUNCHED',mode=mode,pid=child.pid,physical_gpu=physical_gpu,command=command,
 launched_at_utc=datetime.now(timezone.utc).isoformat(),formal_eligible=False,test_evaluated=False)
with receipt.open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(value))
'''
    value = json.loads(base.execute(code, 40))
    dump(LOCAL / TRANSPORT / (key + '_launch.json'), value)
    return value


def status():
    code = header() + '''
from datetime import datetime,timezone
out=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),output=str(output))
for name in ('parallel_status.json','parallel_execution_manifest.json','slot_smoke_gate.json','slot-smoke_gpu2_launch.json','slot-smoke_gpu3_launch.json','scheduler_launch.json'):
 path=output/name
 if path.is_file():out[name]=json.loads(path.read_text())
for path in output.glob('resource_smoke/gpu_*/slot_smoke_gate.json'):
 out[path.relative_to(output).as_posix()]=json.loads(path.read_text())
for path in output.glob('resource_smoke/**/status.json'):
 out[path.relative_to(output).as_posix()]=json.loads(path.read_text())
for path in output.glob('runs/**/status.json'):
 out[path.relative_to(output).as_posix()]=json.loads(path.read_text())
for path in output.glob('*_stdout.log'):
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-4000));out[path.name]=stream.read().decode(errors='replace')
out['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
out['owners']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
'''
    value = json.loads(base.execute(code, 40))
    dump(LOCAL / 'LAUNCH_STATUS_REVISION2.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('stage','prepare','slot-smoke','scheduler','status','admission'))
    parser.add_argument('--gpu', type=int, choices=(1,2,3))
    args = parser.parse_args()
    if args.mode == 'stage':value=stage()
    elif args.mode == 'status':value=status()
    elif args.mode in ('slot-smoke','scheduler'):value=dispatch(args.mode,args.gpu)
    else:value=invoke(args.mode,args.gpu)
    print(json.dumps(value,ensure_ascii=False))


if __name__ == '__main__':
    main()
