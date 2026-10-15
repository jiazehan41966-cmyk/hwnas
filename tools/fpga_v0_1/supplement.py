"""Bounded full-shape P2 tower C-sim, without synthesis or P&R."""
import subprocess
import sys
import tarfile
from remote_bridge import ROOT,ART,REMOTE,HOST,SSH,SCP,ENV

subprocess.run([sys.executable,str(ROOT/'tools/fpga_v0_1/build_cases.py'),'C_P2_tower'],check=True)
path=ART/'tower_supplement.tar.gz'
with tarfile.open(path,'w:gz') as tf:
    tf.add(ROOT/'reports/fpga_v0_1_closure/representative_cases/C_P2_tower',arcname='C_P2_tower')
subprocess.run(SCP+[str(path),HOST+':'+REMOTE+'/tower_supplement.tar.gz'],env=ENV,check=True)
program='''from pathlib import Path
import os,signal,subprocess,tarfile,time,json,shutil
b=Path("/home/jiajia/experiments/fpga_v0_1_closure_20260914_v1")
d=b/"supplement";d.mkdir()
with tarfile.open(b/"tower_supplement.tar.gz") as tf: tf.extractall(d,filter="data")
case=d/"C_P2_tower";start=time.monotonic()
cmd=["/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls","-f","run.tcl"]
with (b/"results/C_P2_tower_hls.log").open("w") as log:
 p=subprocess.Popen(cmd,cwd=case,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try: rc=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM);p.wait();rc="TIMEOUT_300_SECONDS"
out=case/"project/solution1/csim/build/hardware_output.bin"
if out.exists(): shutil.copyfile(out,b/"results/C_P2_tower_hardware_output.bin")
state=dict(case="C_P2_tower",command=cmd,returncode=rc,wall_seconds=time.monotonic()-start)
(b/"results/supplement_status.json").write_text(json.dumps(state,indent=2))
print(json.dumps(state))
'''
subprocess.run(SSH+['python3 -'],input=program,text=True,env=ENV,check=True)
