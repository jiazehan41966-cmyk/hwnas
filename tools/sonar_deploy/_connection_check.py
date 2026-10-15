from tools.sonar_deploy import board_campaign as c
from pathlib import Path
from tools.sonar_deploy.remote import REMOTE
rd=REMOTE+'/results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build'
r=c.remote_json(c.remote_prefix(REMOTE)+f'''
import subprocess
r=Path({rd!r})
assert not (r/'board_return.tar.gz').exists()
proc=subprocess.Popen(['python3',str(p/'tools/detector_preflight/http_transfer.py'),str(r),'--max-return-bytes','54883072','--max-seconds','600'],stdout=(r/'http_recovery_20261002.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,start_ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().split(') ',1)[1].split()[19],directory=str(r))))
''')
Path('reports/sonar_real_int8_deployment_v1/D_RECOVERY_TRANSPORT_20261002.json').write_text(__import__('json').dumps(r,indent=2))
print(r)
