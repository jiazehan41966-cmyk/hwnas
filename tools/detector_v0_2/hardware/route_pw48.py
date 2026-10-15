"""Isolated PW48 out-of-context route; never writes a bitstream or accesses a board."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from remote_hls import OUT, HOST, ENV, SSH, SCP, preflight, ssh_python

ATTEMPT = 'route_pw48_v1'
REMOTE = '/home/jiajia/experiments/detector_v0_2_hardware_20261002_' + ATTEMPT
SOURCE = OUT / 'operator_v2/remote_evidence/pw48/project/solution1/syn/verilog'
HOLD_UNCERTAINTY_NS = 0.050


def launch() -> dict:
    folder = OUT / ATTEMPT
    folder.mkdir(exist_ok=False)
    live = preflight()
    (folder / 'remote_preflight.json').write_text(json.dumps(live, indent=2) + '\n')
    if live['relevant_processes']:
        raise RuntimeError('Existing FPGA tool process; route was not launched')
    rtl = folder / 'rtl'; rtl.mkdir()
    for path in SOURCE.glob('*.v'):
        shutil.copy2(path, rtl / path.name)
    if not (rtl / 'v02_pw48.v').is_file():
        raise RuntimeError('Missing exact completed PW48 RTL')
    # The 1.44 ns setup uncertainty is the actual HLS report value, not a
    # zero-uncertainty relaxation. Ports outside ap_clk have no IO delays:
    # this only verifies the isolated core's intra-clock timing.
    (folder / 'route.tcl').write_text('''set_param general.maxThreads 8
read_verilog [glob rtl/*.v]
synth_design -top v02_pw48 -part xczu7ev-ffvc1156-2-e -mode out_of_context
create_clock -period 5.333 -name ap_clk [get_ports ap_clk]
set_clock_uncertainty -setup 1.440 [get_clocks ap_clk]
set_clock_uncertainty -hold HOLD_UNCERTAINTY_NS_PLACEHOLDER [get_clocks ap_clk]
report_utilization -file post_synth_utilization.rpt
opt_design
place_design
phys_opt_design
route_design
report_route_status -file route_status.rpt
report_timing_summary -delay_type max -max_paths 10 -report_unconstrained -file route_timing_summary.rpt
report_utilization -file route_utilization.rpt
report_utilization -hierarchical -file route_utilization_hier.rpt
report_clocks -file route_clocks.rpt
write_checkpoint -force routed.dcp
set paths [get_timing_paths -quiet -delay_type max -max_paths 1]
set slack "NO_CONSTRAINED_PATH"
if {[llength $paths]} {set slack [get_property SLACK [lindex $paths 0]]}
set f [open route_metrics.txt w]
puts $f "period_ns=5.333"
puts $f "setup_uncertainty_ns=1.440"
puts $f "hold_uncertainty_ns=HOLD_UNCERTAINTY_NS_PLACEHOLDER"
puts $f "worst_constrained_setup_slack_ns=$slack"
close $f
exit
'''.replace('HOLD_UNCERTAINTY_NS_PLACEHOLDER', f'{HOLD_UNCERTAINTY_NS:.3f}'))
    (folder / 'route_constraints.json').write_text(json.dumps(dict(
        clock_period_ns=5.333, setup_uncertainty_ns=1.440,
        hold_uncertainty_ns=HOLD_UNCERTAINTY_NS,
        setup_source='Actual PW48 HLS csynth report ClockUncertainty=1.44ns',
        hold_source='Explicit OOC assumption; not a board clock measurement',
        external_io_delays='NOT_CONSTRAINED',
        source_hls_attempt='operator_v2'), indent=2) + '\n')
    (folder / 'route_runner.py').write_text('''import hashlib,json,os,signal,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
manifest=json.loads((B/'route_source_manifest.json').read_text())
assert all(hashlib.sha256((B/p).read_bytes()).hexdigest()==s for p,s in manifest.items())
state=dict(scope='PW48 isolated OOC route; no full graph, no bitstream, no board',
 source_manifest_sha256=hashlib.sha256((B/'route_source_manifest.json').read_bytes()).hexdigest(),
 start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='RUNNING')
def save():(B/'route_state.json').write_text(json.dumps(state,indent=2)+'\\n')
save();t=time.monotonic()
with (B/'route.log').open('w') as f:
 p=subprocess.Popen(['/opt/Xilinx/Vivado/2022.2/bin/vivado','-mode','batch','-source','route.tcl'],
  cwd=B,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state['tool_pid']=p.pid;save()
 try:rc=p.wait(3600)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  rc='TIMEOUT_3600s'
state.update(returncode=rc,wall_seconds=time.monotonic()-t,
 status='ROUTED' if rc==0 and (B/'route_metrics.txt').is_file() else 'FAILED')
save()
''')
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob('*') if p.is_file()}
    (folder / 'route_source_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    archive = OUT / (ATTEMPT + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for p in folder.rglob('*'):
            if p.is_file():tar.add(p, arcname=p.relative_to(folder).as_posix())
    subprocess.run(SSH + ['mkdir ' + REMOTE], env=ENV, check=True)
    subprocess.run(SCP + [str(archive), HOST + ':' + REMOTE + '/input.tar.gz'], env=ENV, check=True)
    result = ssh_python(f'''import json,tarfile,subprocess
from pathlib import Path
r=Path({REMOTE!r})
with tarfile.open(r/'input.tar.gz') as t:
 for m in t.getmembers():
  if not (r/m.name).resolve().is_relative_to(r.resolve()) or not m.isfile():raise ValueError('unsafe archive')
 t.extractall(r)
p=subprocess.Popen(['python3','route_runner.py'],cwd=r,
 stdout=(r/'runner.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
d=dict(remote_root=str(r),pid=p.pid,scope='isolated PW48 OOC route, no board')
(r/'launch.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d))''')
    (folder / 'remote_launch.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def status() -> dict:
    return ssh_python(f'''import json,os
from pathlib import Path
r=Path({REMOTE!r});p=r/'route_state.json'
d=json.loads(p.read_text()) if p.is_file() else {{'status':'NOT_STARTED'}}
p=r/'route.log';d['log_tail']=p.read_text(errors='replace').splitlines()[-28:] if p.is_file() else []
p=r/'runner.log';d['runner_log_tail']=p.read_text(errors='replace').splitlines()[-8:] if p.is_file() else []
print(json.dumps(d))''')


def fetch() -> dict:
    folder = OUT / ATTEMPT
    result = ssh_python(f'''import json,tarfile
from pathlib import Path
r=Path({REMOTE!r})
with tarfile.open(r/'evidence.tar.gz','w:gz') as t:
 for p in r.rglob('*'):
  if p.is_file() and p.suffix in ['.rpt','.log','.json','.txt','.tcl']:
   t.add(p,arcname=p.relative_to(r).as_posix())
print(json.dumps({{'archive':str(r/'evidence.tar.gz')}}))''')
    dest = folder / 'evidence.tar.gz'
    subprocess.run(SCP + [HOST + ':' + REMOTE + '/evidence.tar.gz', str(dest)], env=ENV, check=True)
    with tarfile.open(dest) as tar:
        for member in tar.getmembers():
            target=(folder/'remote_evidence'/member.name).resolve()
            if not target.is_relative_to((folder/'remote_evidence').resolve()) or not member.isfile():
                raise ValueError('unsafe route evidence')
        tar.extractall(folder/'remote_evidence',filter='data')
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','fetch'])
    p.add_argument('--attempt',default='route_pw48_v1')
    p.add_argument('--hold-uncertainty-ns',type=float,default=0.050)
    a=p.parse_args()
    if not re.match(r'^route_pw48_v[1-9][0-9]*$',a.attempt):raise ValueError('invalid independent attempt')
    if not 0.0<=a.hold_uncertainty_ns<=2.0:raise ValueError('invalid hold uncertainty')
    ATTEMPT=a.attempt
    REMOTE='/home/jiajia/experiments/detector_v0_2_hardware_20261002_'+ATTEMPT
    HOLD_UNCERTAINTY_NS=a.hold_uncertainty_ns
    print(json.dumps(globals()[a.action](),indent=2))
