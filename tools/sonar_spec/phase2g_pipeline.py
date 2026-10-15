"""Bounded three-run Phase2G launcher; no test and no control reruns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from formal_protocol import ROOT,dump,sha
from phase2g_runner import REPORT,IMPLEMENTATION,folder,load,now,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true',required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--wait-for-free',action='store_true');args=p.parse_args();load()
    if args.preflight: print('Phase2G pipeline preflight PASS',sha(IMPLEMENTATION));return
    import fcntl
    lock=(REPORT/'phase2g_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[(42,0),(43,0),(44,1)];mutex=threading.Lock();active={};completed=[]
    for seed,_ in jobs:
        if (folder(seed)/'status.json').exists(): completed.append(summarize(seed))
    pending=[j for j in jobs if not (folder(j[0])/'status.json').exists()]
    logs=ROOT/'logs/phase2g_sonarop';logs.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):dump(REPORT/'campaign_status.json',dict(phase=phase,updated_at=now(),pid=os.getpid(),
        total_jobs=3,completed_jobs=len(completed),active_jobs=dict(active),implementation_manifest_sha256=sha(IMPLEMENTATION),
        test_evaluated=False,**extra))
    def gpu_has_compute_process(gpu):
        result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],
            text=True,capture_output=True,check=True)
        return bool(result.stdout.strip())
    def run(job):
        seed,gpu=job
        with mutex: active[str(seed)]=dict(seed=seed,gpu=gpu);status('TRAINING')
        if shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Disk below 10 GiB reserve')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        with (logs/f'A3_seed{seed}.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','tools/sonar_spec/phase2g_runner.py','train','--seed',str(seed),'--device','cuda'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=summarize(seed)
        with mutex: completed.append(row);active.pop(str(seed));status('TRAINING',last_completed_seed=seed)
    try:
        if args.wait_for_free:
            while any(gpu_has_compute_process(gpu) for gpu in (0,1)):
                status('WAITING_FOR_GPU',waiting_for_gpus=[0,1])
                time.sleep(60)
        status('TRAINING')
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run,j) for j in pending]
            for future in futures:future.result()
        status('ANALYSIS');subprocess.run([sys.executable,'tools/sonar_spec/phase2g_analysis.py'],cwd=ROOT,check=True)
        status('COMPLETE_REQUIRES_REVIEW', sonarop_frozen=False)
    except Exception as error:status('FAILED_RESUMABLE',error=str(error));raise


if __name__=='__main__':main()
