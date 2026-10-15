"""Read-only source provenance and board connectivity audit via local askpass."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import SSH, ENV

REMOTE_CODE = r'''
import json,hashlib,socket,subprocess,zipfile
from pathlib import Path
root=Path('/home/jiajia/experiments/sonar_joint_v1_20260922')
out=root/'results/sonar_joint_v1'
f=json.loads((out/'freeze.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={name:sha(root/name) if (root/name).is_file() else None for name in f['files']}
snap=json.loads((out/'snapshot.json').read_text())
with zipfile.ZipFile(out/'source_snapshot.zip') as z:
    archived={name:hashlib.sha256(z.read(name)).hexdigest() for name in f['files']}
boards={}
for host in ['192.168.1.100','192.168.31.217']:
    try:
        with socket.create_connection((host,22),timeout=3) as s:
            s.settimeout(2);boards[host]={'tcp22':True,'banner':s.recv(128).decode(errors='replace')}
    except OSError as e:boards[host]={'tcp22':False,'error':type(e).__name__}
tools={str(p):p.is_file() for p in [Path('/opt/Xilinx/Vivado/2022.2/bin/vivado'),Path('/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'),Path('/opt/Xilinx/Vitis/2022.2/bin/xsdb')]}
r=dict(remote_root=str(root),freeze_sha256=sha(out/'freeze.json'),files=files,
    remote_frozen_mismatches=[n for n in files if files[n]!=f['files'][n]],
    archive_mismatches=[n for n in archived if archived[n]!=f['files'][n]],
    archive_sha256=sha(out/'source_snapshot.zip'),archive_expected=snap['sha256'],
    state=json.loads((out/'state.json').read_text()),boards=boards,tools=tools)
print(json.dumps(r))
'''


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    proc = subprocess.run(SSH+['python3', '-'], input=REMOTE_CODE, env=ENV,
        capture_output=True, text=True, timeout=40, check=True)
    remote = json.loads(proc.stdout)
    local = json.loads((ROOT/'results/sonar_joint_v1/freeze.json').read_text())
    rows = []
    with zipfile.ZipFile(ROOT/'results/sonar_joint_v1/source_snapshot.zip') as archive:
        for name, frozen in local['files'].items():
            data = (ROOT/name).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual == frozen:
                continue
            old = archive.read(name)
            rows.append(dict(path=name, frozen=frozen, local=actual,
                archived=hashlib.sha256(old).hexdigest(), remote=remote['files'].get(name)))
            diff = ''.join(difflib.unified_diff(old.decode().splitlines(True),data.decode().splitlines(True),
                fromfile='frozen/'+name,tofile='local/'+name))
            (output/(Path(name).stem+'.diff')).write_text(diff,encoding='utf-8')
    remote['local_changes'] = rows
    remote['local_freeze_sha256'] = hashlib.sha256((ROOT/'results/sonar_joint_v1/freeze.json').read_bytes()).hexdigest()
    (output/'audit.json').write_text(json.dumps(remote,indent=2)+'\n')
    print(json.dumps({k:remote[k] for k in ('remote_frozen_mismatches','archive_mismatches','state','boards','tools')},indent=2))
    print('local changed files',len(rows))


if __name__ == '__main__':
    main()
