"""Launch one isolated H2 witness build. Does not touch the search or board."""
from pathlib import Path
import json
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import SSH,SCP,ENV,HOST

folder=Path(sys.argv[1]).resolve()
tcl=sys.argv[2] if len(sys.argv)>2 else 'run_dma.tcl'
if tcl not in ('run_dma.tcl','run_graph.tcl'):raise ValueError('unsupported build entry')
if not folder.is_relative_to(ROOT/'results'):
    raise ValueError('input must be an isolated results directory')
remote='/home/jiajia/experiments/detector_execution_20260922_'+folder.name
archive=folder.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as tf:
    for p in folder.rglob('*'):
        if p.is_file():tf.add(p,arcname=p.relative_to(folder).as_posix())
subprocess.run(SSH+['mkdir '+remote],env=ENV,check=True)
subprocess.run(SCP+[str(archive),HOST+':'+remote+'/input.tar.gz'],env=ENV,check=True)
code=f'''import subprocess,tarfile,json
from pathlib import Path
root=Path({remote!r})
with tarfile.open(root/'input.tar.gz') as t:t.extractall(root)
log=(root/'hls.log').open('w')
p=subprocess.Popen(['/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls','-f',{tcl!r}],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launch.json').write_text(json.dumps({{'pid':p.pid,'scope':'H2 DMA CSIM CSYNTH IP export'}}))
print(p.pid)
'''
p=subprocess.run(SSH+['python3','-'],input=code,env=ENV,capture_output=True,text=True,timeout=20,check=True)
(folder/'remote_launch.json').write_text(json.dumps(dict(remote=remote,pid=int(p.stdout.strip())),indent=2))
print(remote,p.stdout)
