"""Launch route once, only after full-graph C simulation and HLS export succeed."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fpga_v0_1'))
from remote_bridge import SSH,ENV

folder=Path(sys.argv[1]).resolve()
launch=json.loads((folder/'remote_launch.json').read_text())
remote=launch['remote']
code=r'''from pathlib import Path
import json,subprocess,hashlib,re
r=Path(ROOT)
log=(r/'hls.log').read_text()
manifest=json.loads((r/'manifest.json').read_text())
for name,digest in manifest['files'].items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest, 'bound build artifact changed: '+name
rows=re.findall(r'node=(\d+) bytes=(\d+) mismatch=(\d+) status=(\d+)',log)
assert len(rows)==len(manifest['layers']) and rows, 'full graph C simulation incomplete'
assert [int(row[0]) for row in rows]==list(range(len(rows))), 'node sequence mismatch'
assert all(int(row[2])==0 and int(row[3])==0 for row in rows), 'C simulation failed'
assert 'Finished Command export_design' in log, 'IP export incomplete'
assert 'ERROR:' not in log, 'HLS errors'
assert not (r/'route_launch.json').exists(), 'route already launched'
p=subprocess.Popen(['/opt/Xilinx/Vivado/2022.2/bin/vivado','-mode','batch','-source','build_graph_bd.tcl'],cwd=r,stdout=(r/'route.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'route_launch.json').write_text(json.dumps(dict(pid=p.pid)))
print(p.pid)
'''
p=subprocess.run(SSH+['python3','-'],input=code.replace('ROOT',repr(remote)),env=ENV,capture_output=True,text=True,timeout=30,check=True)
(folder/'route_launch.json').write_text(json.dumps(dict(remote=remote,pid=int(p.stdout.strip())),indent=2))
print(p.stdout)
