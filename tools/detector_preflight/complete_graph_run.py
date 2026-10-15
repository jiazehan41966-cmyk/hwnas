"""One-shot continuation: HLS -> matching RTL check -> route -> board parity.

No search restart, candidate selection, recurring job, or automatic retry.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from run_graph_board import remote_python

folder=Path(sys.argv[1]).resolve()
regression=sys.argv[2]
remote=json.loads((folder/'remote_launch.json').read_text())['remote']
status=folder/'continuation_status.json'


def save(stage, **extra):
    status.write_text(json.dumps(dict(stage=stage,remote=remote,regression=regression,**extra),indent=2))
    print(stage,flush=True)


def wait_for(logname, marker, seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        code=f'''from pathlib import Path
import json
p=Path({remote!r})/{logname!r}
t=p.read_text(errors='replace') if p.exists() else ''
print(json.dumps(dict(done={marker!r} in t,errors=[l for l in t.splitlines() if l.startswith('ERROR:')][-3:])))
'''
        progress=json.loads(remote_python(code))
        if progress['errors']:raise RuntimeError(str(progress['errors']))
        if progress['done']:return
        time.sleep(10)
    raise TimeoutError('bounded continuation deadline for '+logname)


try:
    save('WAIT_HLS')
    wait_for('hls.log','Finished Command export_design',300)
    code=Path(__file__).with_name('rtl_equivalence.py').read_text()+f'''
from pathlib import Path
import hashlib
r=Path({remote!r});test=Path({regression!r})
assert 'C/RTL co-simulation finished: PASS' in (test/'hls.log').read_text()
assert (r/'h2_graph.cpp').read_bytes()==(test/'h2_graph.cpp').read_bytes()
import json
report=compare_rtl(r/'hls_project/solution1/syn/verilog',test/'hls_project/solution1/syn/verilog')
(r/'rtl_equivalence.json').write_text(json.dumps(report,indent=2))
print(report['status'])
'''
    print(remote_python(code),flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('launch_graph_route.py')),str(folder)],check=True)
    save('ROUTE_RUNNING')
    wait_for('route.log','Exiting Vivado',2700)
    # Child Vivado runs also print exit markers; the route marker is the real gate.
    deadline=time.monotonic()+2700
    while time.monotonic()<deadline:
        code=f'''from pathlib import Path
import json
r=Path({remote!r});t=(r/'route.log').read_text(errors='replace')
print(json.dumps(dict(done=(r/'ROUTE_TIMING_PASS').is_file(),errors=[l for l in t.splitlines() if l.startswith('ERROR:')][-3:])))
'''
        progress=json.loads(remote_python(code))
        if progress['errors']:raise RuntimeError(str(progress['errors']))
        if progress['done']:break
        time.sleep(10)
    else:raise TimeoutError('route gate did not complete')
    save('BOARD_RUNNING')
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_graph_board.py')),str(folder),regression],check=True)
    save('BOARD_AND_LOCAL_PARITY_PASS')
except BaseException as error:
    save('FAILED_NO_AUTOMATIC_RETRY',error=str(error))
    raise
