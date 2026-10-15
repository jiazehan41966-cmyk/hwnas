"""Bounded sequential V0.2 HLS runner; never accesses or reprograms a board."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

B=Path(__file__).resolve().parent
TOOL='/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'
TIMEOUT_SECONDS=1800
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=json.loads((B/'source_manifest.json').read_text())
    assert all(sha(B/p)==s for p,s in manifest.items()), 'uploaded source manifest mismatch'
    state=dict(scope='standalone synthetic operator CSIM/CSYNTH',start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               source_manifest_sha256=sha(B/'source_manifest.json'),statuses=[],active=None)
    def save(): (B/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    with (B/'tool_version.log').open('w') as f:subprocess.run([TOOL,'-version'],stdout=f,stderr=subprocess.STDOUT)
    allowed=['numeric_boundary','pw48','prediction21','dw48','dw48_stride2','stem32']
    prepared=json.loads((B/'prepared_manifest.json').read_text())
    names=[case['name'] for case in prepared['cases']]
    assert names and names[0]=='numeric_boundary' and len(set(names))==len(names)
    assert all(name in allowed for name in names), 'unexpected case in frozen preparation manifest'
    missing=[name for name in names if not (B/name/'run.tcl').is_file()]
    if missing:
        state['failed_stop']='MISSING_REQUIRED_CASE'
        state['missing_cases']=missing
        save();return 1
    for name in names:
        d=B/name
        t=time.monotonic();state['active']=name;save()
        with (d/'hls.log').open('w') as f:
            p=subprocess.Popen([TOOL,'-f','run.tcl'],cwd=d,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=p.wait(TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                # Only this runner's independently spawned HLS process group.
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                rc='TIMEOUT_1800s'
        build=d/'project/solution1/csim/build'
        parity=[]
        for ref in d.glob('*reference.bin'):
            actual=build/ref.name.replace('reference.bin','actual.bin')
            parity.append(dict(test=ref.stem,actual_exists=actual.is_file(),
                               binary_exact=actual.is_file() and sha(ref)==sha(actual),
                               reference_sha256=sha(ref),actual_sha256=sha(actual) if actual.is_file() else None))
        for p in build.glob('*_csim.json'):shutil.copy2(p,d/p.name)
        case=json.loads((d/'case.json').read_text())
        expected_report=d/'project/solution1/syn/report'/(case.get('top','')+'_csynth.xml')
        status=dict(name=name,returncode=rc,wall_seconds=time.monotonic()-t,parity=parity,
                    csim_pass=bool(parity) and all(r['binary_exact'] for r in parity),
                    csynth_report_present=expected_report.is_file())
        state['statuses'].append(status);state['active']=None;save();print(json.dumps(status),flush=True)
        if rc!=0 or not status['csim_pass']:
            state['failed_stop']=name;save();return 1
        if name!='numeric_boundary' and not status['csynth_report_present']:
            state['failed_stop']=name
            state['failure_reason']='FAILED_SYNTHESIS_EVIDENCE'
            save();return 1
    state['complete']=True;save();return 0


if __name__=='__main__':raise SystemExit(main())
