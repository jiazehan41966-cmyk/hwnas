"""Use existing Credential Manager transport, with no environment installation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.detector_v0_2.remote import execute, SCP, ENV, HOST

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "results/sonar_background_pilot_20261010"
REMOTE = "/home/jiajia/experiments/sonar_background_pilot_20261010"
PYTHON = "/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python"


def save(name, value):
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def probe():
    code = '''
import json,sys,subprocess,importlib.metadata
from pathlib import Path
candidates=[Path('/home/jiajia/experiments/sonar_phase2c_20260914_v1/data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8'),Path('/home/jiajia/experiments/detection_fixed_normalizer_20260912/data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8'),Path('/home/jiajia/experiments/zcu104_full_campaign_v1/data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8')]
found=[]
for p in candidates:
 if (p/'train/images').is_dir() and (p/'valid/images').is_dir():
  found.append({'path':str(p.resolve()),'train_images':len(list((p/'train/images').iterdir())),'valid_images':len(list((p/'valid/images').iterdir()))})
print(json.dumps({'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','Pillow','PyYAML']},'gpu_status':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines(),'data_candidates':found,'test_accessed':False}))
'''
    value = json.loads(execute(code, 45))
    save("server_probe.json", value)
    return value


def stage():
    archive = LOCAL / "source_bundle.tar.gz"
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    execute("from pathlib import Path\np=Path(" + repr(REMOTE) + ")\nassert p.parent==Path('/home/jiajia/experiments')\np.mkdir(exist_ok=False)\n")
    subprocess.run(SCP + [str(archive), HOST + ":" + REMOTE + "/source_bundle.tar.gz"], env=ENV,
                   check=True, capture_output=True, timeout=120)
    code = "from pathlib import Path\nimport tarfile,hashlib,json\nroot=Path(" + repr(REMOTE) + ")\n" + """
archive=root/'source_bundle.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  target=(root/member.name).resolve()
  if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():raise RuntimeError('Unsafe archive member')
 tar.extractall(root,filter='data')
print(json.dumps({'status':'STAGED','archive_sha256':EXPECTED}))
""".replace("EXPECTED", repr(expected))
    value = json.loads(execute(code))
    save("stage_receipt.json", value)
    return value


def invoke(mode, data_dir=None):
    command = [PYTHON, "-m", "tools.sonar_background_pilot.campaign", mode, "--campaign", REMOTE]
    if data_dir:
        command += ["--data-dir", data_dir, "--device", "cuda"]
    code = "import subprocess,os,json\ncommand=" + repr(command) + "\n" + """
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
result=subprocess.run(command,cwd=SNAPSHOT,env=env,capture_output=True,text=True,timeout=240)
print(result.stdout)
if result.returncode:raise RuntimeError(result.stderr[-10000:])
""".replace("SNAPSHOT", repr(REMOTE + "/snapshot"))
    value = json.loads(execute(code, 260))
    save("preflight_cuda.json" if mode == "preflight" else "remote_verify.json", value)
    return value


def launch(data_dir):
    command = [PYTHON, "-m", "tools.sonar_background_pilot.campaign", "queue", "--campaign", REMOTE,
               "--data-dir", data_dir, "--gpus", "0", "1", "--parallel-per-gpu", "2", "--epochs", "30"]
    code = "from pathlib import Path\nimport subprocess,json\nroot=Path(" + repr(REMOTE) + ")\ncommand=" + repr(command) + "\n" + """
assert json.loads((root/'preflight_cuda.json').read_text())['status']=='PASS'
if (root/'launch_receipt.json').exists():raise RuntimeError('Already launched')
with (root/'queue.log').open('x') as out:
 proc=subprocess.Popen(command,cwd=root/'snapshot',stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':proc.pid,'command':command,'cwd':str(root/'snapshot'),'budget':{'jobs':12,'epochs_each':30},'test_accessed':False}
(root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
"""
    value = json.loads(execute(code))
    save("launch_receipt.json", value)
    return value


def status():
    code = "from pathlib import Path\nimport json\nroot=Path(" + repr(REMOTE) + ")\n" + """
state=json.loads((root/'queue_status.json').read_text()) if (root/'queue_status.json').exists() else {'status':'NO_QUEUE_STATUS'}
if (root/'scheduling_status.json').exists():
 state['scheduling']=json.loads((root/'scheduling_status.json').read_text())
 if state['status']=='NO_QUEUE_STATUS':state['status']=state['scheduling']['status']
for job in state.get('jobs',[]):
 run=root/'runs'/f"{job['variant']}_seed{job['seed']}"
 path=run/'epoch_metrics.jsonl'
 if path.exists():
  lines=path.read_text().splitlines()
  job['completed_epochs']=len(lines)
  job['latest_epoch']=json.loads(lines[-1]) if lines else None
 if job['status']=='FAILED':
  job['log_tail']=(root/'logs'/f"{job['variant']}_seed{job['seed']}.log").read_text()[-5000:]
if state['status']=='FAILED':state['queue_log_tail']=(root/'queue.log').read_text()[-5000:]
print(json.dumps(state))
"""
    value = json.loads(execute(code, 40))
    save("queue_status.json", value)
    return value


def enqueue_wait(data_dir):
    controller = (ROOT / "tools/sonar_background_pilot/wait_idle.py").read_text(encoding="utf-8")
    analysis = (ROOT / "tools/sonar_background_pilot/analyze.py").read_text(encoding="utf-8")
    independent = json.loads((LOCAL / "independent_doc_check.json").read_text(encoding="utf-8"))
    binding = json.loads((LOCAL / "preflight_cpu640.json").read_text(encoding="utf-8"))["data_bindings"]
    payloads = {"wait_idle.py": controller, "analysis.py": analysis,
                "independent_doc_check.json": json.dumps(independent, indent=2),
                "cpu640_data_bindings.json": json.dumps(binding, indent=2)}
    code = "from pathlib import Path\nimport json,subprocess,hashlib\nroot=Path(" + repr(REMOTE) + ")\npayloads=" + repr(payloads) + "\n" + """
if (root/'waiting_launch_receipt.json').exists() or (root/'launch_receipt.json').exists():raise RuntimeError('Already scheduled or launched')
hashes={}
for name,content in payloads.items():
 path=root/name
 with path.open('x') as handle:handle.write(content)
 hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
command=[PYTHON,str(root/'wait_idle.py'),'--campaign',str(root),'--data-dir',DATA]
with (root/'wait_controller.log').open('x') as handle:
 proc=subprocess.Popen(command,cwd=root/'snapshot',stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':proc.pid,'command':command,'auxiliary_source_sha256':hashes,'policy':'wait for idle GPU0/1; do not share or stop other tasks','maximum_wait_hours':None,'test_accessed':False}
(root/'waiting_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
""".replace("PYTHON", repr(PYTHON)).replace("DATA", repr(data_dir))
    value = json.loads(execute(code))
    save("waiting_launch_receipt.json", value)
    return value


def refresh_waiting(data_dir):
    """Preserve the first controller receipt when removing its wait timeout."""
    code = "from pathlib import Path\nimport json,os,signal,time\nroot=Path(" + repr(REMOTE) + ")\n" + """
state=json.loads((root/'scheduling_status.json').read_text())
if state['status']!='WAITING_FOR_IDLE':raise RuntimeError('Cannot revise an active training controller')
receipt=json.loads((root/'waiting_launch_receipt.json').read_text())
pid=receipt['pid']
actual=(Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))
if [part.decode() for part in actual]!=receipt['command']:raise RuntimeError('Controller identity mismatch')
if str(root/'wait_idle.py') not in receipt['command']:raise RuntimeError('Outside requested scope')
os.kill(pid,signal.SIGTERM)
time.sleep(1)
archive=root/'controller_revision_1'
archive.mkdir(exist_ok=False)
for name in ['wait_idle.py','analysis.py','independent_doc_check.json','cpu640_data_bindings.json','waiting_launch_receipt.json','scheduling_status.json','wait_controller.log']:
 (root/name).rename(archive/name)
print(json.dumps({'status':'ONLY_WAITING_CONTROLLER_REVISED','prior_pid':pid,'other_jobs_modified':False}))
"""
    value = json.loads(execute(code))
    save("controller_revision_receipt.json", value)
    return enqueue_wait(data_dir)


def fetch():
    code = "from pathlib import Path\nimport tarfile,json\nroot=Path(" + repr(REMOTE) + ")\n" + """
assert json.loads((root/'queue_status.json').read_text())['status']=='COMPLETE'
paths=[p for name in ['runs','logs'] for p in (root/name).rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.log'}]
paths += [root/name for name in ['queue_status.json','comparison.json','preflight_cuda.json','queue.log','acceptance.json','scheduling_status.json','cuda_preflight.log','acceptance.log'] if (root/name).is_file()]
with tarfile.open(root/'result_bundle.tar.gz','w:gz') as tar:
 for p in paths:tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps({'files':len(paths)}))
"""
    value = json.loads(execute(code, 90))
    subprocess.run(SCP + [HOST + ":" + REMOTE + "/result_bundle.tar.gz", str(LOCAL / "result_bundle.tar.gz")],
                   env=ENV, check=True, capture_output=True, timeout=120)
    import tarfile
    with tarfile.open(LOCAL / "result_bundle.tar.gz") as tar:
        tar.extractall(LOCAL / "remote_results", filter="data")
    save("fetch_receipt.json", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "stage", "verify", "preflight", "launch", "enqueue_wait", "refresh_waiting", "status", "fetch"])
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    if args.mode in {"verify", "preflight"}:
        result = invoke(args.mode, args.data_dir)
    elif args.mode == "launch":
        result = launch(args.data_dir)
    elif args.mode == "enqueue_wait":
        result = enqueue_wait(args.data_dir)
    elif args.mode == "refresh_waiting":
        result = refresh_waiting(args.data_dir)
    else:
        result = globals()[args.mode]()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
