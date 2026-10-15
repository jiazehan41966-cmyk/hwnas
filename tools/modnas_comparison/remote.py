"""Transport a source-bound finite HLS probe to a fresh owned-server directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.detector_v0_2.hardware.remote_hls import preflight, ssh_python, SSH, SCP, ENV, HOST

DEFAULT = ROOT / 'results/modnas_vs_ours_20261007/mapping_probe'
REMOTE_BASE = '/home/jiajia/experiments/modnas_vs_ours_20261007_'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def location(attempt):
    if not attempt or not attempt.replace('_', '').isalnum():
        raise ValueError('Attempt must contain only letters, digits, or underscore')
    return REMOTE_BASE + attempt


def stage(folder, attempt):
    folder = folder.resolve()
    if not folder.is_relative_to((ROOT / 'results/modnas_vs_ours_20261007').resolve()):
        raise ValueError('Probe folder must stay inside the declared new experiment root')
    if (folder / 'remote_stage.json').exists():
        raise FileExistsError('Already staged: preserve the previous attempt')
    preparation = json.loads((folder / 'preparation_manifest.json').read_text(encoding='utf-8'))
    if not 1 <= len(preparation['cases']) <= 3:
        raise ValueError('Only a finite batch of one to three registered stem probes may be staged')
    for row in preparation['cases']:
        case = folder / 'cases' / row['name']
        key = json.loads((case / 'measurement_key.json').read_text(encoding='utf-8'))
        for field in ('source_files', 'witness_files'):
            if row[field] != key[field]:
                raise ValueError('Case manifest mismatch')
            for name, expected in key[field].items():
                target = (case / name).resolve()
                if not target.is_relative_to(case.resolve()) or sha(target) != expected:
                    raise ValueError('Prepared case source drift')
    worker = Path(__file__).with_name('remote_worker.py')
    (folder / 'remote_worker.py').write_bytes(worker.read_bytes())
    files = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()
             and p.name not in ('staging_manifest.json', 'input.tar.gz', 'remote_stage.json', 'remote_launch.json')}
    record = dict(schema_version=1, files=files, remote_root=location(attempt), concurrency=1,
                  max_cases=3, timeout_per_case_seconds=900, scope='STEM_MAPPING_HLS_PROBE_ONLY',
                  training_started=False, route_started=False, board_started=False)
    (folder / 'staging_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    archive = folder / 'input.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for name in [*files, 'staging_manifest.json']:
            handle.add(folder / name, arcname=name)
    remote = location(attempt)
    ssh_python(f"from pathlib import Path; import json; p=Path({remote!r}); p.mkdir(exist_ok=False); print(json.dumps({{'created':str(p)}}))")
    subprocess.run(SCP + [str(archive), HOST + ':' + remote + '/input.tar.gz'], env=ENV, check=True)
    result = ssh_python(f'''from pathlib import Path
import json,tarfile,hashlib
r=Path({remote!r})
with tarfile.open(r/'input.tar.gz') as t:
 for member in t.getmembers():
  target=(r/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(r.resolve()):raise ValueError('Unsafe archive member')
  if target.exists():raise FileExistsError('Existing staged file')
  target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(t.extractfile(member).read())
manifest=json.loads((r/'staging_manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((r/name).read_bytes()).hexdigest()!=expected:raise ValueError('Transport hash mismatch')
print(json.dumps(dict(status='STAGED_SOURCE_VERIFIED',remote_root=str(r),source_files=len(manifest['files']))))''')
    (folder / 'remote_stage.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def launch(folder, attempt):
    folder = folder.resolve()
    if (folder / 'remote_launch.json').exists():
        raise FileExistsError('Existing launch receipt; do not start another queue')
    check = preflight()
    (folder / 'launch_preflight.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8')
    if not all(check['tools'].values()) or check['relevant_processes']:
        raise RuntimeError('Required tools unavailable or another hardware tool queue is active')
    remote = location(attempt)
    result = ssh_python(f'''from pathlib import Path
import json,subprocess,hashlib
r=Path({remote!r})
if (r/'launch.json').exists() or (r/'status.json').exists():raise FileExistsError('Existing experiment invocation')
manifest=json.loads((r/'staging_manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((r/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source changed after staging')
p=subprocess.Popen(['python3','remote_worker.py'],cwd=r,stdout=(r/'campaign.log').open('w'),
 stderr=subprocess.STDOUT,start_new_session=True)
d=dict(status='LAUNCHED',pid=p.pid,remote_root=str(r),scope='finite sequential standalone stem CSIM/CSYNTH cases',
 concurrency=1,timeout_per_case_seconds=900,training_started=False,route_started=False,board_started=False)
(r/'launch.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d))''')
    (folder / 'remote_launch.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result



def status(attempt):
    remote = location(attempt)
    return ssh_python(f'''from pathlib import Path
import json,os
r=Path({remote!r}); d=dict(remote_root=str(r))
for name in ['launch.json','status.json']:
 p=r/name; d[name]=json.loads(p.read_text()) if p.is_file() else None
pid=(d.get('launch.json') or {{}}).get('pid')
try: os.kill(pid,0) if pid else None; d['runner_alive']=bool(pid)
except ProcessLookupError:d['runner_alive']=False
p=r/'campaign.log';d['log_tail']=p.read_text(errors='replace').splitlines()[-12:] if p.is_file() else []
active=(d.get('status.json') or {{}}).get('active')
if active:
 p=r/'cases'/active/'hls.log'
 d['active_log_tail']=p.read_text(errors='replace').splitlines()[-16:] if p.is_file() else []
print(json.dumps(d))''')


def fetch(folder, attempt):
    remote = location(attempt)
    ssh_python(f'''from pathlib import Path
import json,tarfile
r=Path({remote!r})
with tarfile.open(r/'evidence.tar.gz','w:gz') as t:
 for p in r.rglob('*'):
  if p.is_file() and p.name not in ['input.tar.gz','evidence.tar.gz'] and ('project' not in p.parts or p.suffix in ['.xml','.rpt','.log'] or p.name=='hardware_output.bin'):
   t.add(p,arcname=p.relative_to(r).as_posix())
print(json.dumps(dict(status='EVIDENCE_ARCHIVED')))''', timeout=60)
    target = folder / 'evidence.tar.gz'
    subprocess.run(SCP + [HOST + ':' + remote + '/evidence.tar.gz', str(target)], env=ENV, check=True)
    evidence = folder / 'remote_evidence'
    evidence.mkdir(exist_ok=False)
    with tarfile.open(target) as handle:
        for member in handle.getmembers():
            path = (evidence / member.name).resolve()
            if not member.isfile() or not path.is_relative_to(evidence.resolve()):
                raise ValueError('Unsafe evidence archive')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(handle.extractfile(member).read())
    return dict(status='EVIDENCE_FETCHED',path=str(evidence),archive_sha256=sha(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','stage','launch','status','fetch'])
    parser.add_argument('--local',type=Path,default=DEFAULT)
    parser.add_argument('--attempt',default='stem_parallel_v1')
    args = parser.parse_args()
    if args.mode == 'preflight': result = preflight()
    elif args.mode == 'stage': result = stage(args.local,args.attempt)
    elif args.mode == 'launch': result = launch(args.local,args.attempt)
    elif args.mode == 'status': result = status(args.attempt)
    else: result = fetch(args.local,args.attempt)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
