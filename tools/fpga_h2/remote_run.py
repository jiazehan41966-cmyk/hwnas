"""Finite H2 C-sim/RTL verification only; no NAS, synthesis campaign or P&R."""
import json,os,signal,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent;R=B/'results';R.mkdir(exist_ok=True)
def execute(cmd,cwd,log,timeout=300):
    start=time.monotonic()
    with (R/log).open('w') as f:
        p=subprocess.Popen(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();rc='TIMEOUT'
    return dict(command=cmd,returncode=rc,wall_seconds=time.monotonic()-start,log=log)
statuses=[]
for tool,cmd in [('hls',['/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls','-version']),('vivado',['/opt/Xilinx/Vivado/2022.2/bin/vivado','-version'])]:
    execute(cmd,B,tool+'_version.log')
for d in sorted((B/'cases').iterdir()):
    if not (d/'run.tcl').exists():continue
    s=execute(['/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls','-f','run.tcl'],d,d.name+'.log')
    s['case']=d.name;statuses.append(s)
    p=d/'project/solution1/csim/build/hardware_output.bin'
    if p.exists():(R/(d.name+'_output.bin')).write_bytes(p.read_bytes())
    (R/'status.json').write_text(json.dumps(statuses,indent=2));print(d.name,s['returncode'],flush=True)
rtl=B/'rtl';bin='/opt/Xilinx/Vivado/2022.2/bin/'
for name,cmd in [('xvlog',[bin+'xvlog','--sv','axis_repack_v1.sv','tb.sv']),('xelab',[bin+'xelab','tb','-s','packet_sim','-mt','4']),('xsim',[bin+'xsim','packet_sim','-runall'])]:
    s=execute(cmd,rtl,name+'.log');statuses.append(dict(case=name,**s));print(name,s['returncode'],flush=True)
    if s['returncode']!=0:break
(R/'status.json').write_text(json.dumps(statuses,indent=2))
(R/'DONE').write_text('finite H2 witnesses finished\n')
