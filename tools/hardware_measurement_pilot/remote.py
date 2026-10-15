"""Credential-Manager transport for the finite, isolated FPGA pilot queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'results/hardware_measurement_pilot_20261012'
sys.path.insert(0, str(ROOT / 'tools/fpga_v0_1'))
from remote_bridge import SSH, SCP, ENV, HOST
from tools.hardware_measurement_pilot.remote_worker import digest, verify_inputs, atomic_json


def remote_python(code, timeout=60):
    result = subprocess.run(SSH + ['python3', '-'], input=code, env=ENV,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'remote command failed')
    return json.loads(result.stdout)


def remote_path(attempt):
    if not re.fullmatch(r'attempt[0-9]+', attempt):
        raise ValueError('expected attemptNN')
    return '/home/jiajia/experiments/hardware_measurement_pilot_20261012_' + attempt


def package(folder):
    manifest = verify_inputs(folder)
    paths = {name: folder / name for name in manifest['files']}
    paths.update({'source_manifest.json': folder / 'source_manifest.json',
                  'native_summary.json': folder / 'native_summary.json',
                  'remote_worker.py': Path(__file__).with_name('remote_worker.py')})
    transport = folder / 'transport'
    transport.mkdir(exist_ok=True)
    transfer = {'schema': 'pilot_transport_v1', 'files': {name: digest(path) for name, path in paths.items()},
                'source_manifest_sha256': digest(folder / 'source_manifest.json')}
    frozen = transport / 'transport_manifest.json'
    if frozen.exists() and json.loads(frozen.read_text()) != transfer:
        raise ValueError('transport inputs changed; preserve this attempt and prepare a new attempt')
    atomic_json(frozen, transfer)
    archive = transport / 'input.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as target:
            for name, path in sorted(paths.items()):
                target.add(path, arcname=name, recursive=False)
            target.add(frozen, arcname='transport_manifest.json', recursive=False)
    return archive, transfer


def launch(args, folder, remote):
    manifest = verify_inputs(folder)
    if args.case_id and args.case_id not in {row['case_id'] for row in manifest['cases']}:
        raise ValueError('requested case is absent from manifest')
    archive, transfer = package(folder)
    archive_sha = digest(archive)
    state = remote_python(f'''
import json
from pathlib import Path
r=Path({remote!r})
if r.exists():
 p=r/'transport_manifest.json'
 if not p.is_file():raise RuntimeError('existing root is incomplete; will not overwrite')
 if json.loads(p.read_text())!={transfer!r}:raise RuntimeError('existing remote manifest differs')
 print(json.dumps({{'created':False}}))
else:
 r.mkdir(exist_ok=False)
 print(json.dumps({{'created':True}}))
''')
    if state['created']:
        subprocess.run(SCP + [str(archive), HOST + ':' + remote + '/input.tar.gz'], env=ENV,
                       check=True, timeout=600)
        remote_python(f'''
import hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
r=Path({remote!r});a=r/'input.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(a)!={archive_sha!r}:raise RuntimeError('transport archive SHA mismatch')
with tarfile.open(a,'r:gz') as t:
 members=t.getmembers();names=[m.name for m in members]
 if len(names)!=len(set(names)):raise ValueError('duplicate tar member')
 data=json.load(t.extractfile('transport_manifest.json'))
 if data!={transfer!r}:raise ValueError('transport manifest mismatch')
 if set(names)!=set(data['files'])|{{'transport_manifest.json'}}:raise ValueError('unexpected members')
 for m in members:
  p=PurePosixPath(m.name)
  if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('unsafe tar member')
  target=r.joinpath(*p.parts)
  if not target.resolve().is_relative_to(r.resolve()) or target.exists():raise ValueError('unsafe overwrite')
  if m.name!='transport_manifest.json':
   h=hashlib.sha256()
   f=t.extractfile(m)
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
   if h.hexdigest()!=data['files'][m.name]:raise ValueError('member hash mismatch')
 for m in members:
  target=r/m.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:
   source=t.extractfile(m)
   for b in iter(lambda:source.read(1048576),b''):f.write(b)
print(json.dumps({{'extraction':'VERIFIED','files':len(members)}}))
''', timeout=180)
    command = ['python3', '-u', remote + '/remote_worker.py', '--root', remote,
               '--stage', args.stage, '--limit', str(args.limit),
               '--hls-timeout', str(args.hls_timeout), '--route-timeout', str(args.route_timeout)]
    if args.case_id:
        command += ['--case', args.case_id]
    if args.retry_failed:
        command += ['--retry-failed']
    launched = remote_python(f'''
import hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path({remote!r})
if (r/'worker.lock').exists():raise RuntimeError('worker lock exists; refusing concurrent execution')
transport=json.loads((r/'transport_manifest.json').read_text())
for name in ['remote_worker.py','native_summary.json','source_manifest.json']:
 if hashlib.sha256((r/name).read_bytes()).hexdigest()!=transport['files'][name]:raise RuntimeError('transport binding changed')
sys.path.insert(0,str(r));from remote_worker import verify_inputs
verify_inputs(r)
(r/'results').mkdir(exist_ok=True)
name='worker_'+str(time.time_ns())+'.log'
with (r/'results'/name).open('x') as log:
 p=subprocess.Popen({command!r},cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={{'remote':str(r),'pid':p.pid,'worker_log':name,'stage':{args.stage!r},'limit':{args.limit!r},'case':{args.case_id!r},'source_manifest_sha256':{transfer['source_manifest_sha256']!r}}}
print(json.dumps(value))
''')
    atomic_json(folder / 'remote_launch.json', launched)
    return launched


def status(remote):
    return remote_python(f'''
import json,os
from pathlib import Path
r=Path({remote!r});rows=[]
for p in sorted((r/'results').glob('*/*/latest.json')):
 s=json.loads(p.read_text());rows.append({{k:s.get(k) for k in ['case_id','case_sha256','stage','status','wall_seconds','returncode','error','evidence_directory']}})
queue=r/'results/queue_status.json';lock=r/'worker.lock'
print(json.dumps({{'remote':str(r),'queue':json.loads(queue.read_text()) if queue.exists() else None,'lock':json.loads(lock.read_text()) if lock.exists() else None,'stages':rows}}))
''')


def fetch(folder, remote, include_bit):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive_name = 'evidence_' + timestamp + '.tar.gz'
    info = remote_python(f'''
import hashlib,io,json,tarfile
from pathlib import Path
r=Path({remote!r});selected={{}}
for p in (r/'results').rglob('*'):
 if p.is_file() and not p.is_symlink():selected[p.relative_to(r).as_posix()]=p
for pattern in ['cases/*/hls_stages.jsonl','cases/*/**/shell_stages.jsonl','cases/*/**/stage_elapsed.jsonl','cases/*/**/shell_metadata.json','cases/*/**/ROUTE_TIMING*_PASS']:
 for p in r.glob(pattern):
  if p.is_file() and not p.is_symlink():selected[p.relative_to(r).as_posix()]=p
if {include_bit!r}:
 for pattern in ['cases/*/**/*.bit','cases/*/**/*.hwh']:
  for p in r.glob(pattern):
   if p.is_file() and not p.is_symlink():selected[p.relative_to(r).as_posix()]=p
snapshot={{name:p.read_bytes() for name,p in selected.items()}}
files={{name:hashlib.sha256(data).hexdigest() for name,data in snapshot.items()}}
snapshot['fetch_manifest.json']=json.dumps({{'files':files}}).encode()
with tarfile.open(r/{archive_name!r},'w:gz') as t:
 for name,data in sorted(snapshot.items()):
  member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
p=r/{archive_name!r};print(json.dumps({{'archive':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':len(files),'bytes':p.stat().st_size}}))
''', timeout=180)
    folder = folder / 'remote_evidence' / timestamp
    folder.mkdir(parents=True, exist_ok=False)
    archive = folder / archive_name
    subprocess.run(SCP + [HOST + ':' + info['archive'], str(archive)], env=ENV, check=True, timeout=600)
    if digest(archive) != info['sha256']:
        raise ValueError('fetched archive SHA mismatch')
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        manifest = json.load(source.extractfile('fetch_manifest.json'))
        if len(members) != len(manifest['files']) + 1:
            raise ValueError('unexpected fetched archive members')
        for member in members:
            path = folder / member.name
            if not member.isfile() or not path.resolve().is_relative_to(folder.resolve()) or path.exists():
                raise ValueError('unsafe fetched archive member')
            data = source.extractfile(member).read()
            if member.name != 'fetch_manifest.json' and hashlib.sha256(data).hexdigest() != manifest['files'][member.name]:
                raise ValueError('fetched evidence file SHA mismatch')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    atomic_json(folder.parent / 'latest_fetch.json', {'directory': str(folder), **info})
    return {'directory': str(folder), **info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'fetch'))
    parser.add_argument('--attempt', default='attempt01')
    parser.add_argument('--stage', choices=('hls', 'route', 'all'), default='hls')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--case', dest='case_id')
    parser.add_argument('--hls-timeout', type=int, default=7200)
    parser.add_argument('--route-timeout', type=int, default=7200)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--include-bit', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.hls_timeout <= 7200 or not 1 <= args.route_timeout <= 7200:
        parser.error('limit must be positive and stage timeouts must be in 1..7200 seconds')
    folder, remote = PILOT / args.attempt, remote_path(args.attempt)
    result = launch(args, folder, remote) if args.action == 'launch' else status(remote) if args.action == 'status' else fetch(folder, remote, args.include_bit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
