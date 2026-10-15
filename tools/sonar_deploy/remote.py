"""Bounded deployment transport; credentials stay in the existing local bridge."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/jiajia/experiments/sonar_real_int8_deployment_20260928_v1'
PYTHON = '/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python'
sys.path.insert(0, str(ROOT / 'tools/fpga_v0_1'))
from remote_bridge import SSH, SCP, ENV, HOST


def execute(code, timeout=60):
    p = subprocess.run(SSH + [PYTHON, '-'], input=code.encode(), env=ENV,
                       capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-5000:])
    return p.stdout.decode()


def setup():
    code = r'''
from pathlib import Path
import shutil,json,hashlib,yaml
source=Path('/home/jiajia/experiments/sonar_phase5a_20260920_v1')
root=Path(REMOTE)
marker=root/'ISOLATED_SOURCE.json'
if root.exists():
 if marker.exists():
  print(json.dumps({'status':'ALREADY_PREPARED','root':str(root)}));raise SystemExit(0)
 if not (root/'SETUP_IN_PROGRESS.json').exists():raise RuntimeError('Existing directory has no matching setup marker')
else:
 root.mkdir()
 (root/'SETUP_IN_PROGRESS.json').write_text(json.dumps({'source_root':str(source)}))
for name in ['src','tools','scripts','configs']:
 for p in (source/name).rglob('*'):
  if p.is_file() and p.suffix in {'.py','.h','.cpp','.sv','.v','.tcl','.yaml','.yml','.json','.md','.sh'} and '__pycache__' not in p.parts:
   target=root/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for name in ['phase5a_quantization','phase2f_multiseed','baseline_sonar_frozen','fpga_h2_contract']:
 report_source=source/'reports'/name
 if not report_source.exists():report_source=Path('/home/jiajia/experiments/sonar_phase2c_20260914_v2/reports')/name
 for p in report_source.glob('*'):
  if p.is_file() and p.suffix in {'.json','.yaml','.csv','.md'}:
   target=root/'reports'/name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for p in (source/'artifacts/phase5a_quantization').rglob('*.json'):
 target=root/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
cfg=yaml.safe_load((root/'configs/baseline_sonar_frozen.yaml').read_text())
data=Path(cfg['dataset']['data_dir']);(root/data).mkdir(parents=True,exist_ok=True)
for split in ['train','valid']:
 link=root/data/split;expected=(source/data/split).resolve()
 if link.exists():assert link.resolve()==expected
 else:link.symlink_to(expected,target_is_directory=True)
shutil.copy2(source/data/'data.yaml',root/data/'data.yaml')
assert not (root/data/'test').exists()
frozen=json.loads((root/'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
mismatch=[]
for p,h in frozen['source_sha256'].items():
 f=root/p.replace('\\','/')
 if hashlib.sha256(f.read_bytes()).hexdigest()!=h:mismatch.append(p)
if mismatch:raise RuntimeError('Frozen baseline sources differ: '+str(mismatch))
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tools','scripts','configs'] for p in (root/d).rglob('*') if p.is_file()}
payload={'root':str(root),'source_root':str(source),'source_sha256':sources,'frozen_baseline_sources':'PASS','test_evaluated':False,'test_mounted':False}
marker.write_text(json.dumps(payload,indent=2))
print(json.dumps({'root':str(root),'copied_source_files':len(sources),'frozen_baseline_sources':'PASS','test_mounted':False}))
'''.replace('Path(REMOTE)', 'Path(' + repr(REMOTE) + ')')
    return execute(code)


def sync():
    # Only the new adapter, ledger, and canonical H2 dependencies. Never replace
    # the archived model, PTQ simulator, evaluator, dataset or baseline sources.
    paths = list((ROOT / 'tools/sonar_deploy').glob('*.py'))
    paths += [ROOT / 'src/hwnas_fpga/metrics_ledger.py']
    paths += [ROOT / 'src/hwnas_fpga/deploy/detection_gate.py']
    paths += [ROOT/'tools/sonar_spec/deployment_pipeline.py']
    paths += list((ROOT / 'tools/fpga_h2').glob('*.py'))
    paths += list((ROOT / 'tools/fpga_h2').glob('*.h'))
    paths += [p for p in (ROOT/'tools/detector_preflight').glob('*') if p.suffix in {'.py','.cpp','.h','.tcl'}]
    paths += list((ROOT / 'tests').glob('test_real*.py'))
    paths += [ROOT/'tests/test_phase5a_quantization.py']
    out = ROOT/'results/sonar_real_int8_deployment_v1/transport'
    out.mkdir(parents=True, exist_ok=True)
    archive = out/'adapter.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    subprocess.run(SCP+[str(archive), HOST+':'+REMOTE+'/adapter.tar.gz'], env=ENV, check=True)
    return execute(r'''
from pathlib import Path
import tarfile,hashlib,json
root=Path(REMOTE)
with tarfile.open(root/'adapter.tar.gz') as tar:
 for m in tar.getmembers():
  dst=(root/m.name).resolve()
  if not m.isfile() or not dst.is_relative_to(root):raise ValueError('Unsafe archive member')
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(tar.extractfile(m).read())
print('ADAPTER_SYNCED')
'''.replace('Path(REMOTE)', 'Path('+repr(REMOTE)+')'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['setup','sync','launch','status','fetch'])
    p.add_argument('--network',choices=['D','C','B1'])
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    args=p.parse_args()
    if args.action in ('setup','sync'):
        print(setup() if args.action=='setup' else sync());return
    if args.action=='launch':
        if args.network is None or args.seed is None:p.error('network and seed required')
        code='''
import subprocess,pathlib,json,os,time
r=pathlib.Path(REMOTE);out=r/'results/sonar_real_int8_deployment_v1';out.mkdir(parents=True,exist_ok=True)
key=NETWORK+'_seed'+str(SEED);pidfile=out/(key+'.pid')
if pidfile.exists():
 try:
  pid=int(pidfile.read_text());os.kill(pid,0)
  if '/sonar_deploy/run.py' in pathlib.Path('/proc/'+str(pid)+'/cmdline').read_text():raise RuntimeError('Same run already active')
 except (ProcessLookupError,FileNotFoundError):pass
log=out/(key+'_'+time.strftime('%Y%m%dT%H%M%S')+'.log')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(GPU);env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
p=subprocess.Popen([PYTHON,'-u','tools/sonar_deploy/run.py','export-evaluate','--network',NETWORK,'--seed',str(SEED),'--device','cuda:0'],cwd=r,env=env,stdout=log.open('w'),stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));print(json.dumps({'pid':p.pid,'log':str(log),'network':NETWORK,'seed':SEED,'gpu':GPU}))
'''
        for key,val in {'REMOTE':REMOTE,'PYTHON':PYTHON,'NETWORK':args.network,'SEED':args.seed,'GPU':args.gpu}.items():
            code=code.replace(key,repr(val))
        print(execute(code));return
    if args.action=='status':
        print(execute('''
from pathlib import Path
import json,subprocess
r=Path(REMOTE)/'results/sonar_real_int8_deployment_v1'
for p in sorted(r.glob('*/seed_*/progress.json')):
 print(str(p.relative_to(r)),p.read_text())
for p in sorted(r.glob('*.log'))[-12:]:
 print(p.name,p.read_text()[-1800:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],text=True))
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')')));return
    if args.action=='fetch':
        print(execute('''
from pathlib import Path
import tarfile
r=Path(REMOTE)
with tarfile.open(r/'deployment_reports.tar.gz','w:gz') as t:
 for name in ['reports/sonar_real_int8_deployment_v1','results/sonar_real_int8_deployment_v1']:
  for p in (r/name).rglob('*'):
   report=p.suffix in {'.json','.jsonl','.csv','.md','.log'}
   route_report=(p.suffix=='.rpt' and p.parent.name=='graph_build')
   execution_recipe=(p.suffix=='.py' and p.parent.name.startswith('cpu_decode_v1'))
   package=(p.suffix=='.bin' and p.parent.name.startswith('package') and any(v in p.parts for v in ('cuda_fold_v1','cpu_decode_v1')))
   if p.is_file() and (report or route_report or execution_recipe or package):t.add(p,arcname=p.relative_to(r))
print('REPORTS_PACKED')
'''.replace('Path(REMOTE)','Path('+repr(REMOTE)+')')))
        out=ROOT/'results/sonar_real_int8_deployment_v1/transport';out.mkdir(parents=True,exist_ok=True)
        archive=out/'remote_reports.tar.gz'
        subprocess.run(SCP+[HOST+':'+REMOTE+'/deployment_reports.tar.gz',str(archive)],env=ENV,check=True)
        # Keep original remote paths inside evidence; never rewrite a signed manifest.
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                dst=(ROOT/m.name).resolve()
                if not m.isfile() or not dst.is_relative_to(ROOT):raise ValueError('Unsafe report member')
                dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(tar.extractfile(m).read())
        print('REPORTS_FETCHED')


if __name__=='__main__':main()
