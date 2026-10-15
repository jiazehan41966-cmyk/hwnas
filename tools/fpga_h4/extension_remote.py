"""Credential-store transport for the finite joint-cost measurement extension."""
import argparse
import shlex
import subprocess
import sys
import tarfile
from common import ROOT,sha
from measure_extension import OUT,save
sys.path.insert(0,str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import SSH,SCP,HOST,ENV

REMOTE='/home/jiajia/experiments/sonar_joint_cost_20260922_v1'
def run(args):
    subprocess.run(args,env=ENV,check=True,timeout=120)

def launch():
    # Keep the tested two-worker H3 execution and 900-second per-case budget.
    text=(ROOT/'tools/fpga_h3/remote_run.py').read_text()
    text=text.replace("assert all(digest(d/p)==h for p,h in key['source_files'].items())",
        "assert all(digest(d/p)==h for p,h in {**key['source_files'],**key['witness_files']}.items())")
    text=text.replace("t=time.monotonic();cmd=[TOOL,'-f','run.tcl']",
        "prior=R/name/'status.json'\n    if prior.exists():\n        old=json.loads(prior.read_text())\n        if old['source_hash']!=key['source_hash']:raise ValueError('Resume source drift')\n        return old\n    t=time.monotonic();cmd=[TOOL,'-f','run.tcl']")
    runner=OUT/'remote_run.py';runner.write_text(text)
    archive=OUT/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(OUT/'cases',arcname='cases');tf.add(runner,arcname='remote_run.py')
    run(SSH+[f'mkdir {shlex.quote(REMOTE)}'])
    run(SCP+[str(archive),HOST+':'+REMOTE+'/bundle.tar.gz'])
    launcher="import subprocess; from pathlib import Path; p=Path("+repr(REMOTE)+"); f=(p/'queue.log').open('w'); subprocess.Popen(['python3','-u','remote_run.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)"
    run(SSH+[f'cd {shlex.quote(REMOTE)} && tar -xzf bundle.tar.gz && python3 -c {shlex.quote(launcher)}'])
    save(OUT/'launch.json',dict(remote=REMOTE,archive_sha256=sha(archive),workers=2,case_timeout_seconds=900,
        command='python3 -u remote_run.py',deployment='NOT_CLAIMED'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','status','fetch']);args=p.parse_args()
    if args.mode=='launch':launch()
    elif args.mode=='status':run(SSH+[f'cd {shlex.quote(REMOTE)} && tail -n 3 queue.log && if test -f results/DONE; then echo COMPLETE; else echo RUNNING; fi'])
    else:
        (OUT/'results').mkdir(exist_ok=True)
        run(SCP+['-r',HOST+':'+REMOTE+'/results/.',str(OUT/'results')])
