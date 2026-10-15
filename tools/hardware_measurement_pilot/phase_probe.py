"""Bounded miniature HLS experiment for observable load/compute/store markers."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/fpga_v0_1'))
from remote_bridge import SSH, SCP, ENV, HOST

VARIANTS = ('vld_only', 'protocol_fixed', 'full_fence')


def source(variant):
    def mark(value):
        if variant == 'protocol_fixed':
            return f'''phase{value}: {{
#pragma HLS protocol fixed
  phase={value};
  ap_wait();
}}'''
        if variant == 'full_fence':
            return f'hls::fence(phase,a,y,x,z); phase={value}; hls::fence(phase,a,y,x,z);'
        return f'phase={value};'
    return '''#include <cstdint>
#include <ap_utils.h>
#include <hls_fence.h>
extern "C" void phase_probe(const int a[64], int y[64], volatile uint8_t& phase) {
#pragma HLS INTERFACE ap_memory port=a
#pragma HLS INTERFACE ap_memory port=y
#pragma HLS INTERFACE ap_vld port=phase
#pragma HLS INTERFACE ap_ctrl_hs port=return
 int x[64],z[64];
#pragma HLS BIND_STORAGE variable=x type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=z type=ram_2p impl=bram
''' + mark(1) + '''
 load_loop: for(int i=0;i<64;i++) {
#pragma HLS PIPELINE II=1
 x[i]=a[i];
 }
''' + mark(2) + '''
 compute_loop: for(int i=0;i<64;i++) {
#pragma HLS PIPELINE II=1
 z[i]=x[i]*3+7;
 }
''' + mark(3) + '''
 store_loop: for(int i=0;i<64;i++) {
#pragma HLS PIPELINE II=1
 y[i]=z[i];
 }
''' + mark(0) + '\n}\n'


TCL = '''open_project project
set_top phase_probe
add_files probe.cpp -cflags {-std=c++14}
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5
csynth_design
exit
'''

WORKER = r'''
import json,os,signal,subprocess,time,shutil
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for name in ['vld_only','protocol_fixed','full_fence']:
 d=r/name;t=time.monotonic()
 with (d/'hls.log').open('w') as f:
  p=subprocess.Popen(['/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls','-f','run.tcl'],cwd=d,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc='TIMEOUT'
 row={'variant':name,'returncode':rc,'wall_seconds':time.monotonic()-t}
 rtl=d/'project/solution1/syn/verilog/phase_probe.v'
 if rtl.is_file():
  shutil.copy2(rtl,d/'phase_probe.v')
  row['ports']=[s.strip() for s in rtl.read_text().splitlines() if s.strip().startswith(('input ','output '))]
 rows.append(row);(r/'status.json').write_text(json.dumps(rows,indent=2))
(r/'DONE').write_text('finite probe finished\n')
'''

SIM_TB = r'''`timescale 1ns/1ps
module tb;
 reg clk=0,rst=1,start=0;
 wire done,idle,ready,a_ce,y_ce,y_we,phase_valid;
 wire [5:0] a_address,y_address;
 reg [31:0] a_q=0;
 wire [31:0] y_data;
 wire [7:0] phase;
 integer cycle=0,current=0,event_count=0,numeric_bad=0,load_bad=0,compute_bad=0,store_bad=0;
 integer loads=0,computes=0,stores=0,event1=0,event2=0,event3=0,event0=0,sequence_bad=0;
 phase_probe dut(.ap_clk(clk),.ap_rst(rst),.ap_start(start),.ap_done(done),.ap_idle(idle),.ap_ready(ready),
 .a_address0(a_address),.a_ce0(a_ce),.a_q0(a_q),.y_address0(y_address),.y_ce0(y_ce),.y_we0(y_we),.y_d0(y_data),
 .phase(phase),.phase_ap_vld(phase_valid));
 always #2.5 clk=~clk;
 always @(posedge clk) begin
  if(a_ce) a_q <= a_address+11;
  if(!rst) begin
   cycle=cycle+1;
   if(phase_valid) begin
    $display("PHASE_EVENT cycle=%0d phase=%0d",cycle,phase);
    event_count=event_count+1;
    if((event_count==1 && phase!=1)||(event_count==2 && phase!=2)||(event_count==3 && phase!=3)||(event_count==4 && phase!=0)) sequence_bad=sequence_bad+1;
    current=phase;
    case(phase) 1:event1=cycle;2:event2=cycle;3:event3=cycle;0:event0=cycle;endcase
   end
   if(dut.x_ce1 && dut.x_we1) begin loads=loads+1;if(current!=1)load_bad=load_bad+1;end
   if(dut.z_ce1 && dut.z_we1) begin computes=computes+1;if(current!=2)compute_bad=compute_bad+1;end
   if(y_ce && y_we) begin
    stores=stores+1;if(current!=3)store_bad=store_bad+1;
    if(y_data !== (y_address+11)*3+7) numeric_bad=numeric_bad+1;
   end
   if(done) begin
    $display("PHASE_PROBE_RESULT events=%0d loads=%0d computes=%0d stores=%0d numeric_bad=%0d load_wrong_phase=%0d compute_wrong_phase=%0d store_wrong_phase=%0d sequence_bad=%0d load_cycles=%0d compute_cycles=%0d store_cycles=%0d",event_count,loads,computes,stores,numeric_bad,load_bad,compute_bad,store_bad,sequence_bad,event2-event1,event3-event2,event0-event3);
    $finish;
   end
   if(cycle>2000) $fatal(1,"TIMEOUT");
  end
 end
 initial begin repeat(6) @(negedge clk);rst=0;@(negedge clk);start=1;@(negedge clk);start=0;end
endmodule
'''


def remote(code, timeout=60):
    p = subprocess.run(SSH+['python3','-'],input=code,env=ENV,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'fetch', 'simulate'))
    parser.add_argument('--id', default='barrier01')
    args = parser.parse_args()
    if not args.id.isalnum():
        raise ValueError('alphanumeric probe ID required')
    local = ROOT / 'results/hardware_measurement_pilot_20261012/phase_probe' / args.id
    directory = '/home/jiajia/experiments/hardware_measurement_pilot_phase_probe_20261012_' + args.id
    if args.action == 'launch':
        local.mkdir(parents=True, exist_ok=False)
        files = {'worker.py': WORKER}
        for name in VARIANTS:
            files[name+'/probe.cpp']=source(name);files[name+'/run.tcl']=TCL
        for name,data in files.items():
            p=local/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
        value=remote(f'''
from pathlib import Path
import json,subprocess
r=Path({directory!r});r.mkdir(exist_ok=False)
for name,data in {files!r}.items():
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
with (r/'worker.log').open('w') as f:
 p=subprocess.Popen(['python3','-u',str(r/'worker.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'remote':str(r),'pid':p.pid}}))
''')
        (local/'launch.json').write_text(json.dumps(value,indent=2))
    elif args.action == 'status':
        value=remote(f'''
from pathlib import Path
import json
r=Path({directory!r});p=r/'status.json'
print(json.dumps({{'complete':(r/'DONE').exists(),'rows':json.loads(p.read_text()) if p.exists() else []}}))
''')
    elif args.action == 'simulate':
        value=remote(rf'''
import json,re,subprocess,time
from pathlib import Path
r=Path({directory!r});tools=Path('/opt/Xilinx/Vivado/2022.2/bin');rows=[]
for name in {VARIANTS!r}:
 d=r/name;(d/'tb.sv').write_text({SIM_TB!r});rtls=list((d/'project/solution1/syn/verilog').glob('*.v'))
 commands=[[str(tools/'xvlog'),'--sv','tb.sv']+[str(p) for p in rtls],[str(tools/'xelab'),'tb','-s','probe_sim'],[str(tools/'xsim'),'probe_sim','-runall']]
 row={{'variant':name}};start=time.monotonic()
 for i,cmd in enumerate(commands):
  with (d/f'rtl_stage{{i}}.log').open('w') as f:
   p=subprocess.run(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=90)
  row[f'stage{{i}}_returncode']=p.returncode
  if p.returncode:break
 text=(d/'rtl_stage2.log').read_text(errors='replace') if (d/'rtl_stage2.log').exists() else ''
 matches=re.findall(r'PHASE_PROBE_RESULT (.*)',text)
 if matches:row['result']={{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',matches[-1])}}
 row['wall_seconds']=time.monotonic()-start;rows.append(row)
(r/'rtl_results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
''',timeout=300)
        (local/'rtl_results.json').write_text(json.dumps(value,indent=2))
    else:
        local.mkdir(parents=True,exist_ok=True)
        for name in VARIANTS:
            (local/name).mkdir(exist_ok=True)
            subprocess.run(SCP+[HOST+':'+directory+'/'+name+'/hls.log',HOST+':'+directory+'/'+name+'/phase_probe.v',str(local/name)],env=ENV,check=False,timeout=30)
            subprocess.run(SCP+[HOST+':'+directory+'/'+name+'/rtl_stage*.log',str(local/name)],env=ENV,check=False,timeout=30)
        subprocess.run(SCP+[HOST+':'+directory+'/status.json',str(local)],env=ENV,check=True,timeout=30)
        value={'local':str(local)}
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
