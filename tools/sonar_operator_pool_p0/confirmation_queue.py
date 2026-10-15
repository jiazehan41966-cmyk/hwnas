"""Finite eight-job confirmation queue; uses one explicitly assigned GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PAIRS=(('DW5',7),('DCN',1),('FrequencyLite',1),('DirectionalDW',5))


def worker(seed):
    logdir=ROOT/'logs/sonar_operator_pool_p0_confirmation'
    logdir.mkdir(parents=True,exist_ok=True)
    for operator,stage in PAIRS:
        args=[sys.executable,'-u','tools/sonar_operator_pool_p0/confirm.py','train',
              '--operator',operator,'--stage',str(stage),'--seed',str(seed)]
        logfile=logdir/f'{operator}_stage{stage}_seed{seed}.log'
        with logfile.open('a') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return seed


if __name__=='__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['OMP_NUM_THREADS']='2'
    report=ROOT/'reports/sonar_operator_pool_p0_confirmation'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,s) for s in (43,44)]
        for future in concurrent.futures.as_completed(futures):
            print('SEED LANE COMPLETE',future.result(),flush=True)
    subprocess.run([sys.executable,'tools/sonar_operator_pool_p0/confirm.py','aggregate'],cwd=ROOT,check=True)
    (report/'QUEUE_COMPLETE.json').write_text(json.dumps(dict(new_runs=8,status='COMPLETE',test_evaluated=False),indent=2))
