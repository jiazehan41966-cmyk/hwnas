"""Two-worker, bounded H3 C-sim/CSYNTH queue; retain all failures and reports."""
import concurrent.futures,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
B=Path(__file__).resolve().parent;R=B/'results';R.mkdir(exist_ok=True)
TOOL='/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(d):
    key=json.loads((d/'measurement_key.json').read_text());name=d.name
    if key['LEGAL_MAPPING_CHECK']!='PASS':return dict(point_id=name,status='REJECTED')
    assert all(digest(d/p)==h for p,h in key['source_files'].items())
    t=time.monotonic();cmd=[TOOL,'-f','run.tcl']
    with (R/(name+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(900)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            rc='TIMEOUT_900s'
    target=R/name;target.mkdir(exist_ok=True)
    for p in (d/'project/solution1/syn/report').glob('*'):
        if p.suffix in ['.xml','.rpt']:shutil.copy2(p,target/p.name)
    out=d/'project/solution1/csim/build/hardware_output.bin'
    parity=out.exists() and digest(out)==digest(d/'reference.bin')
    if out.exists():shutil.copy2(out,target/'hardware_output.bin')
    result=dict(point_id=name,returncode=rc,wall_seconds=time.monotonic()-t,csim_binary_exact=parity,command=cmd,remote_path=str(d),source_hash=key['source_hash'])
    (target/'status.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    with (R/'tool_version.log').open('w') as f:subprocess.run([TOOL,'-version'],stdout=f,stderr=subprocess.STDOUT)
    statuses=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,d) for d in sorted((B/'cases').iterdir()) if (d/'run.tcl').exists()]
        for f in concurrent.futures.as_completed(fs):
            s=f.result();statuses.append(s);(R/'status.json').write_text(json.dumps(statuses,indent=2));print(s,flush=True)
    (R/'DONE').write_text('finite batch complete\n')
