"""Isolated campaign transport on the existing owned server; train/valid only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.detector_v0_2.remote import execute, SSH, SCP, ENV, HOST, PYTHON, SOURCE
from tools.sonar_joint192_validation import train

REMOTE = '/home/jiajia/experiments/sonar_joint192_validation_20261008_01'
RESULT = 'results/sonar_joint192_validation_20261008_01/remote'
LOCAL = ROOT / 'results/sonar_joint192_validation_20261008_01/transport'


def stage():
    cfg, baseline = train.load_config()
    paths = {ROOT / p for p in train.source_hashes(cfg)}
    paths.add(Path(__file__))
    paths.add(ROOT / baseline['dataset']['data_dir'] / 'data.yaml')
    LOCAL.mkdir(parents=True, exist_ok=True)
    marker = dict(source_sha256={p.relative_to(ROOT).as_posix(): train.sha(p) for p in sorted(paths)},
                  test_mounted=False, trained_weights_copied=False)
    train.dump(LOCAL / 'staging_manifest.json', marker)
    archive = LOCAL / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(paths):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        tar.add(LOCAL / 'staging_manifest.json', arcname='staging_manifest.json')
    execute('from pathlib import Path\np=Path(' + repr(REMOTE) + ')\n'
            'if p.exists(): raise FileExistsError("Isolated namespace already exists")\np.mkdir()\n')
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/source.tar.gz'], env=ENV, check=True)
    code = '''from pathlib import Path
import tarfile,json,hashlib,yaml
root=Path(REMOTE)
with tarfile.open(root/'source.tar.gz') as tar:
 for m in tar.getmembers():
  p=(root/m.name).resolve()
  if not m.isfile() or not p.is_relative_to(root.resolve()):raise ValueError('Unsafe member')
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
marker=json.loads((root/'staging_manifest.json').read_text())
for rel,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Transport hash mismatch:'+rel)
b=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=root/b['dataset']['data_dir'];data.mkdir(parents=True,exist_ok=True)
for split in ('train','valid'):
 target=(Path(SOURCE)/b['dataset']['data_dir']/split).resolve(strict=True)
 (data/split).symlink_to(target,target_is_directory=True)
if (data/'test').exists() or (data/'test').is_symlink():raise ValueError('Test mounted')
print(json.dumps({'status':'STAGED','source_files':len(marker['source_sha256']),'test_mounted':False}))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')').replace('Path(SOURCE)', 'Path(' + repr(SOURCE) + ')')
    return json.loads(execute(code, 60))


def invoke(mode, arm='B0', gpu=2, timeout=240):
    if mode not in ('prepare', 'preflight', 'smoke'):
        raise ValueError('Only documented preflight modes')
    command = [PYTHON, 'tools/sonar_joint192_validation/train.py', mode,
               '--result-root', RESULT, '--prototype', arm, '--seed', '42', '--device', 'cuda']
    return execute('import subprocess,os\nos.environ["CUDA_VISIBLE_DEVICES"]=' + repr(str(gpu)) +
                   '\nsubprocess.run(' + repr(command) + ',cwd=' + repr(REMOTE) + ',check=True)\n', timeout)


def launch():
    code = '''from pathlib import Path
import json,subprocess,os,sys
root=Path(REMOTE);result=root/RESULT
sys.path[:0]=[str(root),str(root/'src')]
from tools.sonar_joint192_validation.queue_runner import smoke_gate
gate=smoke_gate(result)
if (result/'launch.json').exists():raise FileExistsError('Already launched')
cfg=json.loads(json.dumps(__import__('yaml').safe_load((root/'configs/sonar_joint192_validation_20261008.yaml').read_text())))
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus={int(v[0]):[int(x) for x in v[1:]] for line in rows.splitlines() if (v:=line.split(','))}
reports=[json.loads((result/f'smoke/{a}/seed_42/smoke_report.json').read_text()) for a in ('B0','B1','B2','B3')]
required=max(r['peak_cuda_reserved_bytes'] for r in reports)/1048576+2048
for gpu in (2,3):
 used,total,util=gpus[gpu]
 if total-used<required or util>10:raise RuntimeError('GPU availability changed:'+str(gpu))
launch={'status':'LAUNCHED','workers':[],'gate':gate,'gpu_snapshot':gpus,'required_free_mib':required,'test_evaluated':False}
for gpu in (2,3):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (result/f'worker_gpu{gpu}_stdout.log').open('a') as log:
  p=subprocess.Popen([PYTHON,'tools/sonar_joint192_validation/queue_runner.py','--result-root',RESULT,'--gpu',str(gpu)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch['workers'].append({'gpu':gpu,'pid':p.pid})
(result/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')').replace('root/RESULT', 'root/' + repr(RESULT)).replace('[PYTHON,', '[' + repr(PYTHON) + ',').replace("'--result-root',RESULT", "'--result-root'," + repr(RESULT))
    return json.loads(execute(code, 90))


def status():
    code = '''from pathlib import Path
import json
r=Path(REMOTE)/RESULT
out={}
for p in sorted(r.rglob('status.json'))+sorted(r.rglob('queue_status.json')):
 out[p.relative_to(r).as_posix()]=json.loads(p.read_text())
for p in sorted(r.glob('worker_gpu*_stdout.log')):
 out[p.name]=p.read_text()[-1500:]
for p in sorted(r.glob('formal/*/seed_*/stdout.log')):
 out[p.relative_to(r).as_posix()]=p.read_text()[-400:]
print(json.dumps(out))
'''.replace('Path(REMOTE)/RESULT', 'Path(' + repr(REMOTE) + ')/' + repr(RESULT))
    value = json.loads(execute(code, 60))
    train.dump(LOCAL / 'status.json', value)
    return value


def init_replay():
    code = '''from pathlib import Path
import json,sys,torch
root=Path(REMOTE);result=root/RESULT
sys.path[:0]=[str(root),str(root/'src')]
from tools.sonar_joint192_validation import train
torch.set_num_threads(2)
train.load_manifest(result,False)
cfg,_=train.load_config();identities={}
for arm in train.PROTOTYPES:
 for seed in train.SEEDS:
  m=train.build_model(cfg['arms'][arm],seed)
  identities[f'{arm}/seed_{seed}']=m.initialization_identity()
report={'execution_manifest_sha256':train.sha(result/'execution_manifest.json'),'environment':train.environment('cpu'),'identities':identities,'purpose':'reconstruct_cpu_initialization_in_exact_training_runtime_before_device_transfer'}
for arm in train.PROTOTYPES:
 actual=json.loads((result/f'smoke/{arm}/seed_42/initialization_identity.json').read_text())
 if actual!=identities[f'{arm}/seed_42']:raise ValueError('Pinned-runtime initialization mismatch:'+arm)
p=result/'initialization_replay.json'
if p.exists() and json.loads(p.read_text())!=report:raise ValueError('Existing initialization witness differs')
if not p.exists():train.dump(p,report)
print(json.dumps({'status':'PINNED_INITIALIZATION_REPLAY_PASS','models':len(identities),'seed42_smoke_all_match':True}))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')').replace('root/RESULT', 'root/' + repr(RESULT))
    return json.loads(execute(code, 120))


def checkpoint_replay():
    """Transfer only the independently bound inference helper after completion."""
    import base64
    helper = ROOT / 'tools/sonar_joint192_validation/checkpoint_replay.py'
    encoded = base64.b64encode(helper.read_bytes()).decode('ascii')
    code = '''from pathlib import Path
import base64,hashlib,json,os,subprocess
root=Path(REMOTE);result=root/RESULT
for gpu in (2,3):
 worker=json.loads((result/f'worker_gpu{gpu}/queue_status.json').read_text())
 if worker['status']!='COMPLETE' or len(worker['completed'])!=6:raise RuntimeError('Inference replay requires all twelve complete runs')
helper=root/'tools/sonar_joint192_validation/checkpoint_replay.py'
data=base64.b64decode(ENCODED)
if hashlib.sha256(data).hexdigest()!=DIGEST:raise ValueError('Inference helper transport SHA mismatch')
if helper.exists() and helper.read_bytes()!=data:raise ValueError('Existing inference helper differs; preserve it')
if not helper.exists():helper.write_bytes(data)
if hashlib.sha256(helper.read_bytes()).hexdigest()!=DIGEST:raise ValueError('Uploaded inference helper SHA mismatch')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
output=result/'checkpoint_replay_final'
command=[PYTHON,str(helper),'--result-root',str(result),'--output-root',str(output),'--physical-gpu','2']
process=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True)
if process.returncode:
 print(process.stdout,flush=True)
 raise RuntimeError('Final checkpoint inference replay failed; independent receipt preserved: '+process.stderr[-4000:])
receipt=json.loads((output/'replay_receipt.json').read_text())
if receipt['status']!='PASS' or not receipt['all_twelve_replayed'] or receipt['replay_script_sha256']!=DIGEST:raise ValueError('Checkpoint replay completion identity mismatch')
print(json.dumps({'status':'CHECKPOINT_INFERENCE_REPLAY_PASS','completed_runs':len(receipt['runs']),
 'all_strict_comparisons_pass':receipt['all_strict_comparisons_pass'],
 'receipt_sha256':hashlib.sha256((output/'replay_receipt.json').read_bytes()).hexdigest(),
 'replay_script_sha256':DIGEST,'physical_gpu':2,'test_evaluated':False}))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')').replace('root/RESULT', 'root/' + repr(RESULT))
    code = code.replace('ENCODED', repr(encoded)).replace('DIGEST', repr(train.sha(helper))).replace('[PYTHON,', '[' + repr(PYTHON) + ',')
    return json.loads(execute(code, 600))


def collect():
    code = '''from pathlib import Path
import json,tarfile,hashlib
root=Path(REMOTE);r=root/RESULT
for gpu in (2,3):
 s=json.loads((r/f'worker_gpu{gpu}/queue_status.json').read_text())
 if s['status']!='COMPLETE' or len(s['completed'])!=6:raise RuntimeError('Final collection requires two complete workers')
files=[p for p in r.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp') and p.name!='queue.lock']
manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
checkpoints={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*.pt')}
(root/'collection_manifest.json').write_text(json.dumps({'artifacts':manifest,'remote_checkpoints':checkpoints},indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as tar:
 for p in files:tar.add(p,arcname=p.relative_to(r).as_posix())
 tar.add(root/'collection_manifest.json',arcname='collection_manifest.json')
print(json.dumps({'files':len(files),'checkpoint_files_retained_remotely':len(checkpoints)}))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')').replace('root/RESULT', 'root/' + repr(RESULT))
    info = json.loads(execute(code, 120))
    LOCAL.mkdir(parents=True, exist_ok=True)
    archive = LOCAL / 'evidence.tar.gz'
    subprocess.run(SCP + [HOST + ':' + REMOTE + '/evidence.tar.gz', str(archive)], env=ENV, check=True)
    dest = train.DEFAULT_ROOT / 'remote_snapshot'
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            p = (dest / m.name).resolve()
            if not m.isfile() or not p.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe result archive')
            data = tar.extractfile(m).read()
            if p.exists() and p.read_bytes() != data:
                raise ValueError('Existing result snapshot differs; preserve it')
            p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data)
    manifest = json.loads((dest / 'collection_manifest.json').read_text())
    for rel, digest in manifest['artifacts'].items():
        if train.sha(dest / rel) != digest:
            raise ValueError('Result transport hash mismatch')
    return dict(**info, local_snapshot=str(dest), transport_sha_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage','prepare','preflight','smoke','launch','status','collect','init-replay','checkpoint-replay'])
    parser.add_argument('--arm', choices=train.PROTOTYPES, default='B0')
    parser.add_argument('--gpu', type=int, choices=[2,3], default=2)
    args = parser.parse_args()
    if args.mode in ('prepare','preflight','smoke'):
        print(invoke(args.mode, args.arm, args.gpu))
    else:
        print(json.dumps(globals()[args.mode.replace('-','_')](),ensure_ascii=False))
