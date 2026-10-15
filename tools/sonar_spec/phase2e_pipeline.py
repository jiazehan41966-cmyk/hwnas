"""One bounded independent-training campaign; no hardware or inherited search."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from phase2e_runner import ROOT,REPORT,load,binding,arch_map,existing_map,record,promote,rejected_audit,now
from formal_protocol import dump,sha


def csv_write(path,rows):
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


def manifest(path,payload):
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ValueError('Immutable promotion manifest changed')
    else:dump(path,payload)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,nargs='+',default=[1]);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if len(set(args.gpus))!=len(args.gpus):p.error('Duplicate GPU IDs')
    cfg,b=load();statepath=REPORT/'campaign_status.json'
    if statepath.exists() and not args.resume:raise FileExistsError('Campaign exists; inspect status or explicitly --resume')
    logs=ROOT/'logs/phase2e_multifidelity';logs.mkdir(parents=True,exist_ok=True)
    ids=sorted(arch_map());existing=existing_map()
    def status(phase,**kw):dump(statepath,dict(phase=phase,updated_at=now(),pid=os.getpid(),gpus=args.gpus,binding=binding(),test_evaluated=False,**kw))
    def runone(aid,budget,gpu):
        if aid not in existing:
            if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Disk below3GiB reserve; safe epoch-boundary resume available')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
            with (logs/f'{aid}.log').open('a') as log:
                subprocess.run([sys.executable,'-u','tools/sonar_spec/phase2e_runner.py','train','--architecture-id',aid,
                    '--budget',str(budget),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return record(aid,budget)
    def stage(name,selected,budget):
        path=REPORT/f'{name}_results.csv';rows=[]
        if path.exists():
            with path.open(newline='') as f:rows=list(csv.DictReader(f))
            if set(r['architecture_id'] for r in rows)!=set(selected):raise ValueError('Completed stage ID mismatch')
            for r in rows:
                current=record(r['architecture_id'],budget)
                if abs(float(r['mAP50_95'])-current['mAP50_95'])>1e-14:raise ValueError('Completed stage score drift')
            return [record(r['architecture_id'],budget) for r in rows]
        mutex=threading.Lock();failed=threading.Event();active={};ordered=sorted(selected)
        def worker(gpu,assigned):
            try:
                for aid in assigned:
                    if failed.is_set():return
                    with mutex:
                        active[str(gpu)]=aid
                        status(name.upper(),budget=budget,completed=len(rows),total=len(selected),active_architectures=dict(active))
                    row=runone(aid,budget,gpu)
                    with mutex:
                        rows.append(row);active.pop(str(gpu),None)
                        csv_write(REPORT/f'{name}_partial.csv',sorted(rows,key=lambda r:r['architecture_id']))
                        status(name.upper(),budget=budget,completed=len(rows),total=len(selected),active_architectures=dict(active))
            except Exception:failed.set();raise
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            jobs=[pool.submit(worker,gpu,ordered[i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
            for job in jobs:job.result()
        rows.sort(key=lambda r:r['architecture_id'])
        csv_write(path,rows);return rows
    try:
        first=stage('stage1',ids,40);second_ids=promote(first,*cfg['promotion_fractions'][0]);audit1=rejected_audit(first,second_ids,1,existing)
        manifest(REPORT/'promotion_stage1.json',dict(binding=binding(),input_sha256=sha(REPORT/'stage1_results.csv'),
            promoted=[r['architecture_id'] for r in second_ids],audit=audit1,metric=cfg['promotion_metric']))
        second=stage('stage2',[r['architecture_id'] for r in second_ids],60);final_ids=promote(second,*cfg['promotion_fractions'][1]);audit2=rejected_audit(second,final_ids,2,existing)
        manifest(REPORT/'promotion_stage2.json',dict(binding=binding(),input_sha256=sha(REPORT/'stage2_results.csv'),
            promoted=[r['architecture_id'] for r in final_ids],audit=audit2,metric=cfg['promotion_metric']))
        stage('final',[r['architecture_id'] for r in final_ids],80)
        if audit1+audit2:stage('rejection_audit',audit1+audit2,80)
        status('ANALYSIS')
        subprocess.run([sys.executable,'tools/sonar_spec/phase2e_results.py'],cwd=ROOT,check=True)
        outcome=json.loads((REPORT/'promotion_reliability.json').read_text())
        status('COMPLETE_REQUIRES_REVIEW' if not outcome['audit_top5_violation'] else 'STOP_PROMOTION_RELIABILITY_FAILED',
            final_selection_executed=False,sonarop_implemented=False)
    except Exception as error:status('FAILED_RESUMABLE',error=str(error));raise


if __name__=='__main__':main()
