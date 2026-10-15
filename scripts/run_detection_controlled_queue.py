"""Finite six-run queue on one explicitly selected, initially idle GPU."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--parallel",type=int,default=2)
    args=p.parse_args()
    manifest_path=Path(args.manifest).resolve()
    spec=json.loads(manifest_path.read_text())
    root=manifest_path.parent
    lock=(Path.home()/f".hwnas_detection_controlled_gpu{args.gpu}.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={"status":"WAITING_FOR_IDLE_GPU","started_at":datetime.now().astimezone().isoformat(),
           "gpu":args.gpu,"jobs":[],"test_accessed":False}
    def save():
        state["updated_at"]=datetime.now().astimezone().isoformat()
        temp=root/"queue_status.tmp"
        temp.write_text(json.dumps(state,indent=2))
        temp.replace(root/"queue_status.json")
    save()
    deadline=time.monotonic()+3600
    while True:
        mem=int(subprocess.check_output(["nvidia-smi","-i",str(args.gpu),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        if mem < 500:
            break
        if time.monotonic()>deadline:
            state["status"]="WAIT_TIMEOUT_NO_JOB_STARTED"
            save()
            return
        time.sleep(15)
    state["status"]="RUNNING"
    pending=list(spec["jobs"])
    active=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",
             PYTHONHASHSEED="0",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    failed=False
    while pending or active:
        while pending and len(active)<args.parallel and not failed:
            job=pending.pop(0)
            dest=root/"runs"/job["name"]
            log=root/"logs"/(job["name"]+".log")
            log.parent.mkdir(parents=True,exist_ok=True)
            out=log.open("x")
            cmd=[sys.executable,str(root/"snapshot/scripts/detection_controlled_study.py"),"train",
                 "--config",str(root/job["config"]),"--data-dir",spec["data_dir"],
                 "--output-dir",str(dest),"--seed",str(job["seed"]),"--epochs",str(spec["epochs"]),
                 "--workers","2","--device","cuda"]
            proc=subprocess.Popen(cmd,cwd=root/"snapshot",env=env,stdout=out,stderr=subprocess.STDOUT)
            item={**job,"pid":proc.pid,"command":cmd,"status":"RUNNING","log":str(log),
                  "started_at":datetime.now().astimezone().isoformat()}
            state["jobs"].append(item)
            active.append((proc,out,item,dest))
            print("START",job["name"],proc.pid,flush=True)
            save()
        for entry in list(active):
            proc,out,item,dest=entry
            rc=proc.poll()
            if rc is None:
                continue
            out.close()
            item["returncode"]=rc
            item["status"]="COMPLETE" if rc==0 and (dest/"status.json").is_file() else "FAILED"
            item["completed_at"]=datetime.now().astimezone().isoformat()
            failed=failed or item["status"]=="FAILED"
            active.remove(entry)
            print(item["status"],item["name"],flush=True)
            save()
        if failed and not active:
            break
        time.sleep(10)
    state["status"]="FAILED" if failed else "COMPLETE"
    state["unstarted_jobs"]=[j["name"] for j in pending]
    save()


if __name__=="__main__":
    main()
