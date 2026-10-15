"""Bounded two-process runner on one otherwise free owned GPU; no auto retry."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_joint192_extension_20261013 import train


def main(gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
        raise ValueError('Explicit visible GPU assignment required')
    m=train.manifest();cfg=m['config'];result=train.RESULT
    if gpu!=cfg['gpu']:
        raise ValueError('GPU assignment differs from frozen config')
    gate=json.loads((result/'smoke_gate.json').read_text())
    if gate['status']!='PASS' or gate['execution_manifest_sha256']!=train.sha(result/'execution_manifest.json'):
        raise ValueError('Bound real CUDA smoke required')
    path=result/'queue_status.json'
    if path.exists():raise FileExistsError('Queue already invoked; no overwrite/retry')
    launch=json.loads((result/'launch.json').read_text())
    concurrency=launch['actual_concurrency']
    if type(concurrency) is not int or not 1<=concurrency<=cfg['concurrency']:
        raise ValueError('Runtime concurrency exceeds frozen upper bound')
    import fcntl
    lock=(result/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[(cfg['reference'],42),(cfg['reference'],43),(cfg['reference'],44)]
    jobs += [(arm,42) for arm in cfg['arms'] if arm!=cfg['reference']]
    _,existing=train.collect_results(True)
    if (set(existing)!={(cfg['reference'],42),(cfg['reference'],43)} or
            any('recovery_origin' not in r for r in existing.values())):
        raise ValueError('Both verified recovered reference runs are required before continuation')
    jobs=[job for job in jobs if job not in existing]
    inherited_rows=[dict(arm=arm,seed=seed,status='COMPLETE',recovered_without_training=True,
                        result_sha256=train.sha(result/'runs'/arm/f'seed_{seed}'/'result.json'))
                    for arm,seed in existing]
    status=dict(status='RUNNING',gpu=gpu,pid=os.getpid(),concurrency=concurrency,
                phase='screening',completed=inherited_rows,active=[],selection=None,started_at=train.now(),test_evaluated=False)
    train.dump(path,status)
    running={};failure=False;selection_done=False;counter=len(existing)
    try:
        while True:
            for pid,entry in list(running.items()):
                rc=entry['process'].poll()
                if rc is None:continue
                entry['log'].close();del running[pid]
                task={k:entry[k] for k in ('arm','seed','pid','log_path')}
                if rc:
                    failure=True;task.update(returncode=rc,status='FAILED_NO_AUTORETRY')
                else:
                    rp=result/'runs'/entry['arm']/f'seed_{entry["seed"]}'/'result.json'
                    r=json.loads(rp.read_text())
                    if r['status']!='COMPLETE' or r['full_epochs']!=80:raise ValueError('Child missing full completion')
                    task.update(status='COMPLETE',result_sha256=train.sha(rp))
                status['completed'].append(task)
            while jobs and len(running)<concurrency and not failure:
                arm,seed=jobs.pop(0);counter+=1
                if counter>cfg['maximum_runs']:raise RuntimeError('Run cap exceeded')
                log_path=result/'logs'/f'{arm}_seed{seed}.log';log_path.parent.mkdir(exist_ok=True)
                if (result/'runs'/arm/f'seed_{seed}').exists() or log_path.exists():
                    raise FileExistsError('Existing run/log refused before child launch')
                log=log_path.open('xb')
                env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                proc=subprocess.Popen([sys.executable,'-B','tools/sonar_joint192_extension_20261013/train.py','run','--arm',arm,'--seed',str(seed)],
                                      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[proc.pid]=dict(process=proc,log=log,arm=arm,seed=seed,pid=proc.pid,log_path=str(log_path))
            if failure and not running:
                status.update(status='FAILED_NO_AUTORETRY',pending_unstarted=[dict(arm=a,seed=s) for a,s in jobs],ended_at=train.now());break
            if not jobs and not running:
                if not selection_done:
                    selection=train.select();selection_done=True
                    status.update(phase='confirmation',selection=selection)
                    jobs=[(arm,seed) for arm in selection['selected'] for seed in (43,44)]
                    if jobs:continue
                status.update(status='COMPLETE',ended_at=train.now());break
            status.update(active=[{k:e[k] for k in ('arm','seed','pid','log_path')} for e in running.values()],
                          pending_count=len(jobs),updated_at=train.now())
            train.dump(path,status);train.report()
            time.sleep(15)
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY',error=str(error),updated_at=train.now())
        train.dump(path,status)
        raise
    finally:
        # Existing children finish their independent run; this manager never
        # kills unrelated jobs or retries a failed frozen invocation.
        status.update(active=[{k:e[k] for k in ('arm','seed','pid','log_path')} for e in running.values()])
        train.dump(path,status);train.report();lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True,type=int)
    main(parser.parse_args().gpu)
