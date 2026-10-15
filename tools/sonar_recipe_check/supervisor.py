"""Finite one-shot preflight or nine-run campaign, followed by D14 analysis."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','campaign'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=args.output
    result.mkdir(parents=True,exist_ok=True)
    train=[sys.executable,'tools/sonar_recipe_check/train.py']
    if args.mode=='preflight':
        commands=[train+['prepare','--output',str(result)],
            [sys.executable,'-m','unittest','tests.test_sonar_recipe_check_training','tests.test_sonar_recipe_check_analysis','-v'],
            train+['smoke','--output',str(result),'--device','cuda'],
            train+['verify','--output',str(result)]]
        timeouts=[180,300,600,180]
    else:
        commands=[train+['queue','--output',str(result),'--device','cuda'],
            [sys.executable,'tools/sonar_recipe_check/finalize.py','--output',str(result)]]
        timeouts=[19*3600,2*3600]
    status=dict(status='RUNNING',mode=args.mode,pid=os.getpid(),started_at=now(),commands=commands,completed=[])
    path=result/(args.mode+'_supervisor_status.json'); write(path,status)
    child=None
    try:
        for command,timeout in zip(commands,timeouts):
            child=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
            status.update(active_command=command,child_pid=child.pid,updated_at=now());write(path,status)
            try: code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Dedicated process group ensures a timeout cannot leave a train child running.
                raise RuntimeError('Bounded supervisor timed out; no automatic retry')
            if code: raise RuntimeError('Command failed with exit '+str(code))
            status['completed'].append(command)
        status.update(status='PASS' if args.mode=='preflight' else 'COMPLETE',child_pid=None,active_command=None,updated_at=now())
        write(path,status)
        if args.mode=='preflight':write(result/'preflight_complete.json',status)
    except Exception as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        status.update(status='FAILED_NO_AUTORETRY',error=str(error),updated_at=now());write(path,status)
        raise

if __name__=='__main__':main()
