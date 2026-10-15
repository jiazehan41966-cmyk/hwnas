"""Bounded six-run, one-process-per-GPU paired training queue."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import threading
import time

from run import ROOT, OUT, SEEDS, VARIANTS, binding, dump, load, now, sha, summarize_one


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',nargs=2,type=int,default=[2,3]);ap.add_argument('--resume',action='store_true',required=True)
    args=ap.parse_args();p,_=load()
    if args.gpus!=p['resources']['gpus']:raise ValueError('GPU selection differs from frozen new protocol')
    import fcntl
    lock=(OUT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'logs').mkdir(exist_ok=True);active={};complete=[];errors=[];mutex=threading.Lock();stop=threading.Event()
    def status(phase,**extra):
        dump(OUT/'campaign_status.json',dict(phase=phase,pid=os.getpid(),updated_at=now(),
             protocol_sha256=sha(OUT/'protocol.json'),gpus=args.gpus,total_jobs=6,
             completed_jobs=len(complete),completed=complete,active=dict(active),errors=list(errors),test_accessed=False,**extra))
    # Both variants of each seed run on the same physical GPU.  One process per
    # GPU retains headroom for existing services; no existing process is killed.
    assignments={2:[42,44],3:[43]}
    def worker(gpu):
        for seed in assignments[gpu]:
            for variant in VARIANTS:
                if stop.is_set():return
                key=f'{variant}_seed{seed}'
                try:
                    with mutex:active[key]=dict(variant=variant,seed=seed,gpu=gpu);status('TRAINING')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    cmd=[sys.executable,'-B','-u','tools/p3_fusion_ablation/run.py','train','--variant',variant,'--seed',str(seed)]
                    with (OUT/'logs'/f'{key}.log').open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        with mutex:active[key]['pid']=child.pid;status('TRAINING')
                        code=child.wait()
                    if code:raise RuntimeError(f'{key} subprocess exit {code}; inspect its dedicated log')
                    row=summarize_one(variant,seed)
                    with mutex:complete.append(row);active.pop(key);status('TRAINING')
                except Exception as e:
                    stop.set()
                    with mutex:errors.append(dict(job=key,gpu=gpu,error=str(e)));active.pop(key,None);status('FAILED_RESUMABLE')
                    raise
    status('TRAINING')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,gpu) for gpu in args.gpus]
            for future in futures:future.result()
        assert len(complete)==6
        for seed in SEEDS:
            pair=[r for r in complete if r['seed']==seed]
            assert len({r['initial_shared_sha256'] for r in pair})==1
            assert len({r['train_start_rng_sha256'] for r in pair})==1
        dump(OUT/'all_results.json',dict(status='COMPLETE',runs=complete,test_accessed=False))
        status('COMPLETE_REQUIRES_ANALYSIS')
    except Exception:
        status('FAILED_RESUMABLE');raise


if __name__=='__main__':main()
