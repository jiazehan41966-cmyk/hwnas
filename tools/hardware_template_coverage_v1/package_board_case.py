"""Fetch one accepted immutable case, with all evidence needed before programming."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

from .remote import PILOT, remote_path, remote_python, SCP, HOST, ENV
from .board_runner import SHELL_FILES, prepare_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_case(attempt, case_id):
    if not re.fullmatch(r'[a-z0-9_]+', case_id):
        raise ValueError('invalid case ID')
    remote = remote_path(attempt)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive_name = 'board_package_' + case_id + '_' + stamp + '.tar.gz'
    info = remote_python(f'''
import hashlib,io,json,tarfile
from pathlib import Path
r=Path({remote!r});c=r/'cases'/{case_id!r}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
case=json.loads((c/'case.json').read_text())
source_sha=sha(r/'source_manifest.json')
stages={{}}
for stage in ['hls','route']:
 p=r/'results'/{case_id!r}/stage/'latest.json';s=json.loads(p.read_text())
 if s.get('status')!='PASS' or s.get('case_sha256')!=case['case_sha256'] or s.get('source_manifest_sha256')!=source_sha:
  raise ValueError('case needs matching accepted HLS and route: '+stage)
 stages[stage]=s
selected={{name:c/name for name in case['source_files']}}
selected['case.json']=c/'case.json'
for fixture in case['fixtures']:
 selected.update({{name:c/name for name in fixture['files']}})
for name in {list(SHELL_FILES)!r}:
 selected['vivado_shell/'+name]=c/'vivado_shell'/name
for pattern in ['hls_project/solution1/impl/ip/component.xml','hls_project/solution1/syn/report/*.xml',
                'hls_project/solution1/syn/report/*.rpt','hls_project/solution1/syn/verilog/*.v']:
 for p in c.glob(pattern):selected[p.relative_to(c).as_posix()]=p
selected['evidence/source_manifest.json']=r/'source_manifest.json'
for stage,s in stages.items():
 selected['evidence/'+stage+'_status.json']=r/'results'/{case_id!r}/stage/'latest.json'
 for p in (r/s['evidence_directory']).iterdir():
  if p.is_file():selected['evidence/'+stage+'/'+p.name]=p
phase=r/'results'/{case_id!r}/'phase_structural_gate.json'
if phase.is_file():selected['evidence/phase_structural_gate.json']=phase
for name,digest in case['source_files'].items():
 if sha(c/name)!=digest:raise ValueError('source hash mismatch: '+name)
for fixture in case['fixtures']:
 for name,digest in fixture['files'].items():
  if sha(c/name)!=digest:raise ValueError('fixture hash mismatch: '+name)
bit_manifest=json.loads((r/stages['route']['evidence_directory']/'bitstream_manifest.json').read_text())
for row in bit_manifest['files']:
 name=row['path'];p=c/name
 if sha(p)!=row['sha256']:raise ValueError('routed bit/HWH changed: '+name)
data={{name:p.read_bytes() for name,p in selected.items()}}
manifest={{'schema':'pilot_board_package_transport_v1','case_id':case['case_id'],
 'case_sha256':case['case_sha256'],'source_manifest_sha256':source_sha,
 'files':{{name:hashlib.sha256(value).hexdigest() for name,value in data.items()}}}}
data['package_transport_manifest.json']=json.dumps(manifest,indent=2).encode()
with tarfile.open(r/{archive_name!r},'w:gz') as target:
 for name,value in sorted(data.items()):
  member=tarfile.TarInfo(name);member.size=len(value);target.addfile(member,io.BytesIO(value))
p=r/{archive_name!r}
print(json.dumps({{'archive':str(p),'sha256':sha(p),'files':len(data),'bytes':p.stat().st_size,'manifest':manifest}}))
''', timeout=180)
    parent = PILOT / attempt / 'board_packages' / case_id / stamp
    parent.mkdir(parents=True, exist_ok=False)
    archive = parent / archive_name
    subprocess.run(SCP + [HOST + ':' + info['archive'], str(archive)], env=ENV, check=True, timeout=600)
    if sha(archive) != info['sha256']:
        raise ValueError('downloaded archive hash mismatch')
    root = parent / 'case'
    root.mkdir()
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        names = [item.name for item in members]
        expected = info['manifest']['files']
        if len(names) != len(set(names)) or set(names) != set(expected) | {'package_transport_manifest.json'}:
            raise ValueError('archive membership mismatch')
        for item in members:
            relative = PurePosixPath(item.name)
            if not item.isfile() or relative.is_absolute() or '..' in relative.parts or '\\' in item.name:
                raise ValueError('unsafe archive path')
            value = source.extractfile(item).read()
            if item.name == 'package_transport_manifest.json':
                if json.loads(value) != info['manifest']:
                    raise ValueError('embedded transport manifest mismatch')
            elif hashlib.sha256(value).hexdigest() != expected[item.name]:
                raise ValueError('archive member hash mismatch')
            path = root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(value)
    # Numeric, source, route, and register-ABI gates run offline here.
    prepare_bundle(root, mode='probe')
    result = dict(case_dir=str(root), transport_sha256=info['sha256'],
                  files=info['files'], probe_bundle_sha256=sha(root/'board_probe_bundle.json'))
    (parent/'package_result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--case', required=True)
    arguments = parser.parse_args()
    print(json.dumps(fetch_case(arguments.attempt, arguments.case), indent=2))
