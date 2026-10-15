"""Isolated V0.2 operator HLS transport; reuses Credential Manager bridge.

No board access, package installation, or existing-process termination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/fpga_v0_1'))
from remote_bridge import SSH, SCP, ENV, HOST

OUT = ROOT / 'results/detector_v0_2_execution_20261002/hardware'
REMOTE_BASE = '/home/jiajia/experiments/detector_v0_2_hardware_20261002'


def ssh_python(code: str, timeout: int = 35) -> dict:
    p = subprocess.run(SSH + ['python3', '-'], input=code, env=ENV,
                       capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(p.stdout)


def preflight() -> dict:
    return ssh_python('''import json,os,shutil,subprocess,time
from pathlib import Path
paths=['/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls','/opt/Xilinx/Vivado/2022.2/bin/vivado']
p=subprocess.run(['ps','-eo','pid,ppid,etimes,pcpu,pmem,comm'],capture_output=True,text=True,check=True)
rows=[s for s in p.stdout.splitlines()[1:] if any(n in s.lower() for n in ['vitis','vivado','v++','csynth','xsim','xvlog','clang','g++'])]
u=shutil.disk_usage('/home/jiajia/experiments')
print(json.dumps(dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  tools={s:Path(s).is_file() for s in paths},gxx=shutil.which('g++'),
  cpu_count=os.cpu_count(),loadavg=os.getloadavg(),disk_free_bytes=u.free,
  relevant_processes=rows)))''')


def launch(attempt: str) -> dict:
    if not attempt.replace('_', '').isalnum():
        raise ValueError('attempt must contain only letters, digits and underscore')
    remote = REMOTE_BASE + '_' + attempt
    folder = OUT / attempt
    if not folder.is_dir():
        raise ValueError('prepare local independent case folder first')
    status = preflight()
    (folder / 'remote_preflight.json').write_text(json.dumps(status, indent=2) + '\n')
    if not all(status['tools'].values()):
        raise RuntimeError('Required remote Xilinx tool missing')
    if status['relevant_processes']:
        raise RuntimeError('Existing tool jobs found: do not overlap without resource review')
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob('*') if p.is_file()}
    (folder / 'source_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    archive = OUT / (attempt + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for p in folder.rglob('*'):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(folder).as_posix())
    subprocess.run(SSH + ['mkdir ' + remote], env=ENV, check=True)
    subprocess.run(SCP + [str(archive), HOST + ':' + remote + '/input.tar.gz'],
                   env=ENV, check=True)
    result = ssh_python(f'''import json,subprocess,tarfile
from pathlib import Path
r=Path({remote!r})
with tarfile.open(r/'input.tar.gz') as t:
 for member in t.getmembers():
  target=(r/member.name).resolve()
  if not target.is_relative_to(r.resolve()) or not member.isfile():
   raise ValueError('unsafe source archive member')
 t.extractall(r)
p=subprocess.Popen(['python3','remote_run.py'],cwd=r,
 stdout=(r/'campaign.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
d=dict(remote_root=str(r),pid=p.pid,scope='standalone V02 C simulation and HLS synthesis; no route or board')
(r/'launch.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d))''')
    (folder / 'remote_launch.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def status(attempt: str) -> dict:
    if not attempt.replace('_', '').isalnum():
        raise ValueError('invalid attempt')
    remote = REMOTE_BASE + '_' + attempt
    return ssh_python(f'''import json,os
from pathlib import Path
r=Path({remote!r})
d={{'remote_root':str(r)}}
for n in ['launch.json','status.json']:
 p=r/n
 d[n]=json.loads(p.read_text()) if p.is_file() else None
p=r/'campaign.log'
d['log_tail']=p.read_text(errors='replace').splitlines()[-24:] if p.is_file() else []
launch=d.get('launch.json') or {{}}
pid=launch.get('pid')
try:
 os.kill(pid,0) if pid else None
 alive=bool(pid)
except ProcessLookupError:alive=False
d['runner_alive']=alive
active=(d.get('status.json') or {{}}).get('active')
if active:
 p=r/active/'hls.log'
 d['active_hls_log_tail']=p.read_text(errors='replace').splitlines()[-32:] if p.is_file() else []
print(json.dumps(d))''')


def fetch(attempt: str) -> dict:
    if not attempt.replace('_', '').isalnum():
        raise ValueError('invalid attempt')
    remote = REMOTE_BASE + '_' + attempt
    folder = OUT / attempt
    result = ssh_python(f'''import json,tarfile
from pathlib import Path
r=Path({remote!r})
with tarfile.open(r/'evidence.tar.gz','w:gz') as t:
 for p in r.rglob('*'):
  if p.is_file() and p.suffix in ['.rpt','.xml','.log','.json','.v']:
   t.add(p,arcname=p.relative_to(r).as_posix())
print(json.dumps({{'archive':str(r/'evidence.tar.gz')}}))''', timeout=60)
    dest = folder / 'remote_evidence.tar.gz'
    subprocess.run(SCP + [HOST + ':' + remote + '/evidence.tar.gz', str(dest)],
                   env=ENV, check=True)
    with tarfile.open(dest) as tar:
        for member in tar.getmembers():
            target = (folder / 'remote_evidence' / member.name).resolve()
            if not target.is_relative_to((folder / 'remote_evidence').resolve()) or not member.isfile():
                raise ValueError('unsafe evidence member')
        tar.extractall(folder / 'remote_evidence', filter='data')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'launch', 'status', 'fetch'])
    parser.add_argument('--attempt', default='operator_v1')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = preflight() if args.action == 'preflight' else globals()[args.action](args.attempt)
    if args.action == 'preflight':
        (OUT / 'remote_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
