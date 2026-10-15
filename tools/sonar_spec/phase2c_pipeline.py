"""One bounded Phase2C job chain. No final architecture selection or test access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from formal_protocol import ROOT,load_protocol,require_gate,dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-status',type=Path,required=True)
    args=p.parse_args()
    protocol=ROOT/'configs/supernet_sonar_v0_1_formal.yaml'
    config,b=load_protocol(protocol);gate=require_gate(config,protocol)
    report=ROOT/'reports/phase2c_supernet';logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    status_path=report/'pipeline_status.json'
    if status_path.exists():raise FileExistsError('Pipeline already initialized; inspect existing run instead of duplicating it')
    def status(phase,**kwargs):
        dump(status_path,dict(phase=phase,time_utc=datetime.now(timezone.utc).isoformat(),gate_sha256=gate,protocol_sha256=sha(protocol),test_evaluated=False,**kwargs))
    def command(label,arguments,gpu):
        import os
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        with (logs/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    runner='tools/sonar_spec/formal_runner.py'
    try:
        status('FORMAL_SUPERNET_TRAINING')
        command('supernet_seed42',[runner,'train-supernet','--device','cuda','--output','results/supernet_seed42'],0)
        history=json.loads((ROOT/'results/supernet_seed42/epoch_log.json').read_text())
        (report/'FORMAL_SUPERNET_TRAINING.md').write_text('# Formal Supernet Training\n\n80 epochs completed; seed42, batch4, AdamW0.001/0.0001, scheduler none, AMP false. '
            'Final epoch80 is used without validation-driven selection.\n\nEpoch training logs: results/supernet_seed42/epoch_log.json\n\n'
            f'Total iterations: {sum(r["batches"] for r in history)}. All recorded gradients finite: {all(r["gradient_finite"] for r in history)}. '
            'Training loss is an aggregate of sampled subnet paths, not an architecture performance score.\n',encoding='utf-8')
        status('INHERITED_EVALUATION_144')
        command('ranking',[runner,'rank','--device','cuda','--checkpoint','results/supernet_seed42/latest.pt','--output','reports/phase2c_supernet'],0)
        status('STANDALONE_WITNESSES_12')
        def worker(gpu,indices):
            if gpu==1:
                # Respect the independently running baseline; never preempt it.
                deadline=time.monotonic()+12*3600
                while not args.baseline_status.exists():
                    if time.monotonic()>deadline:raise RuntimeError('Baseline status still unavailable after12h; GPU1 witness queue stopped')
                    time.sleep(30)
                if json.loads(args.baseline_status.read_text())['status']!='COMPLETE':raise RuntimeError('Baseline did not complete successfully')
            for index in indices:
                command(f'witness_{index:02d}',[runner,'standalone','--device','cuda','--witness','reports/phase2c_supernet/ranking_witness_manifest.json',
                    '--index',str(index),'--output',f'results/witness_{index:02d}'],gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(worker,0,range(0,12,2)),pool.submit(worker,1,range(1,12,2))]
            for job in jobs:job.result()
        status('RANKING_RELIABILITY_ANALYSIS')
        command('ranking_analysis',['tools/sonar_spec/ranking_analysis.py','--directory','reports/phase2c_supernet','--standalone-root','results'],0)
        status('COMPLETE_REQUIRES_RELIABILITY_REVIEW',final_selection_executed=False)
    except Exception as error:
        status('FAILED',error=str(error));raise


if __name__=='__main__':main()
