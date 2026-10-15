"""One-shot, gated graph board run using the existing local Jupyter bridge.

This is not a recurring job. It validates a completed route and a matching RTL
regression, packages the image, runs the board, and checks returned raw heads.
"""
import hashlib
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import SSH,SCP,ENV,HOST


def remote_python(code):
    p=subprocess.run(SSH+['python3','-'],input=code,env=ENV,capture_output=True,
                     text=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout


def board_directory(remote, retry=''):
    identity=hashlib.sha256(remote.encode('utf-8')).hexdigest()[:12]
    return '/home/xilinx/'+PurePosixPath(remote).name+'_'+identity+('_'+retry if retry else '')


def result_upload_limit(manifest):
    # Both real and padded returned heads for train8 + valid99, plus first-image
    # heads and bounded JSON metadata. Compression is not assumed to save space.
    outputs=manifest['plan']['outputs']
    heads=sum(int(row['real_bytes'])+int(row['tail_zero_bytes']) for row in outputs)
    limit=2*108*heads+16*1024*1024
    if not 0<limit<=512*1024*1024:raise ValueError('Graph return size outside bounded transport capacity')
    return limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('regression')
    parser.add_argument('retry',nargs='?',default='')
    parser.add_argument('--node-timeout',type=float,default=180)
    args=parser.parse_args()
    folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'results'):raise ValueError('isolated results required')
    regression=args.regression
    retry=args.retry
    if not 0<args.node_timeout<=3600:raise ValueError('node timeout must be positive and bounded')
    if retry and not re.fullmatch(r'retry[0-9]+',retry):raise ValueError('expected retryN label')
    regression_path=PurePosixPath(regression)
    if not regression_path.is_relative_to('/home/jiajia/experiments') or '..' in regression_path.parts:
        raise ValueError('unexpected regression directory')
    remote=json.loads((folder/'remote_launch.json').read_text())['remote']
    remote_path=PurePosixPath(remote)
    if not remote_path.is_relative_to('/home/jiajia/experiments') or '..' in remote_path.parts:
        raise ValueError('unexpected graph build directory')
    source_sha=hashlib.sha256((folder/'h2_graph.cpp').read_bytes()).hexdigest()
    code=Path(__file__).with_name('rtl_equivalence.py').read_text()+f'''
from pathlib import Path
import hashlib
r=Path({remote!r});test=Path({regression!r})
assert (r/'ROUTE_TIMING_PASS').is_file()
assert 'All user specified timing constraints are met.' in (r/'timing.rpt').read_text()
assert 'C/RTL co-simulation finished: PASS' in (test/'hls.log').read_text()
assert hashlib.sha256((test/'h2_graph.cpp').read_bytes()).hexdigest()=={source_sha!r}
assert hashlib.sha256((r/'h2_graph.cpp').read_bytes()).hexdigest()=={source_sha!r}
assert (r/'numeric_v1.h').read_bytes()==(test/'numeric_v1.h').read_bytes()
def config(path):
 return sorted(line.strip() for line in path.read_text().splitlines()
               if line.startswith(('set_part ','create_clock ','config_interface ','config_schedule ','config_op ')))
assert config(r/'run_graph.tcl')==config(test/'run.tcl'), 'regression synthesis configuration mismatch'
compare_rtl(r/'hls_project/solution1/syn/verilog',test/'hls_project/solution1/syn/verilog')
import json,re
manifest=json.loads((r/'manifest.json').read_text())
rows=re.findall(r'node=(\\d+) bytes=(\\d+) mismatch=(\\d+) status=(\\d+)',(r/'hls.log').read_text())
assert len(rows)==len(manifest['layers']) and rows
assert [int(row[0]) for row in rows]==list(range(len(rows)))
assert all(int(row[2])==0 and int(row[3])==0 for row in rows)
print('ROUTE_AND_MATCHING_RTL_GATE_PASS')
'''
    print(remote_python(code),flush=True)
    if retry:
        package=json.loads((folder/'board_package.json').read_text())
        transfer=remote+'/board_'+retry
        setup=f'''from pathlib import Path
import hashlib
r=Path({remote!r});destination=Path({transfer!r})
assert hashlib.sha256((r/'board_payload.tar.gz').read_bytes()).hexdigest()=={package['sha256']!r}
destination.mkdir(exist_ok=False)
(destination/'board_payload.tar.gz').symlink_to(r/'board_payload.tar.gz')
'''
        remote_python(setup)
    else:
        files=['package_graph_board.py','http_transfer.py']
        frozen_source=json.loads((folder/'manifest.json').read_text())
        # New builds already bind their tested driver. Do not replace it with
        # a later local version while reusing the synthesized graph manifest.
        if 'pynq_graph_runner.py' not in frozen_source.get('files',{}):
            files.append('pynq_graph_runner.py')
        subprocess.run(SCP+[str(Path(__file__).parent/n) for n in files]+[HOST+':'+remote+'/'],env=ENV,check=True)
        p=subprocess.run(SSH+['python3',remote+'/package_graph_board.py',remote],env=ENV,
                         capture_output=True,text=True,timeout=60,check=True)
        package=json.loads(p.stdout)
        (folder/'board_package.json').write_text(json.dumps(package,indent=2))
        transfer=remote
    suffix='_'+retry if retry else ''
    evidence=folder/('board_evidence'+suffix);evidence.mkdir(exist_ok=False)
    board_folder=board_directory(remote,retry)
    upload_limit=result_upload_limit(json.loads((folder/'manifest.json').read_text()))
    (folder/('board_execution_binding'+suffix+'.json')).write_text(json.dumps(dict(
        remote_directory=remote,board_directory=board_folder,package_sha256=package['sha256'],
        node_timeout_seconds=args.node_timeout,return_limit_bytes=upload_limit,
        graph_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        timeout_interpretation='functional watchdog, not an FPS/latency acceptance criterion',
        test_evaluated=False),indent=2))
    code=f'''import urllib.request,hashlib,tarfile,io,importlib.util,threading,traceback
from pathlib import Path
url='http://192.168.31.1:18773'
raw=urllib.request.urlopen(url+'/board_payload.tar.gz',timeout=30).read()
assert hashlib.sha256(raw).hexdigest()=={package['sha256']!r}
folder=Path({board_folder!r});folder.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=(folder/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(folder.resolve()):raise ValueError('unsafe member')
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(archive.extractfile(member).read())
print('GRAPH_PAYLOAD_VERIFIED',flush=True)
spec=importlib.util.spec_from_file_location('graph_driver',folder/'pynq_graph_runner.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
errors=[];stop=threading.Event()
def heartbeat():
 while not stop.wait(10):print('GRAPH_RUNNING',flush=True)
worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
try:driver.run(folder,timeout={args.node_timeout!r})
except BaseException as error:
 errors.append(str(error));traceback.print_exc()
finally:stop.set();worker.join()
output=io.BytesIO()
with tarfile.open(fileobj=output,mode='w:gz') as archive:
 for path in (folder/'results').rglob('*'):
  if path.is_file():archive.add(path,arcname=path.relative_to(folder/'results').as_posix())
data=output.getvalue()
req=urllib.request.Request(url+'/board_return.tar.gz',data=data,method='PUT')
print('GRAPH_RESULT_UPLOAD',urllib.request.urlopen(req,timeout=30).status,flush=True)
if errors:raise RuntimeError(errors[0])
print('BOARD_GRAPH_COMPLETE',flush=True)
'''
    board_code=folder/('board_execute'+suffix+'.py');board_code.write_text(code)
    start=f'''from pathlib import Path
import subprocess
r=Path({transfer!r})
p=subprocess.Popen(['python3',{remote!r}+'/http_transfer.py',str(r),'--max-return-bytes',str({upload_limit})],stdout=(r/'http.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'http_pid').write_text(str(p.pid));print(p.pid)
'''
    pid=int(remote_python(start).strip())
    try:
        bridge=Path('C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/jupyter_reuse_bridge.py')
        result=subprocess.run([sys.executable,str(bridge),str(board_code),str(folder/('board_execute'+suffix+'.log')),'--keep-kernel'])
        names=['timing.rpt','utilization_hier.rpt','route_status.rpt','rtl_equivalence.json','hls.log',
               'board_payload/board_manifest.json','board_payload/source_manifest.json',
               'board_payload/h2_graph.bit','board_payload/h2_graph.hwh']
        if json.loads((folder/'manifest.json').read_text()).get('real_weights_bound'):
            names += ['board_payload/graph.json','board_payload/export_manifest.json','utilization.rpt']
        subprocess.run(SCP+[HOST+':'+transfer+'/board_return.tar.gz']+[HOST+':'+remote+'/'+n for n in names]+[str(evidence)],env=ENV,check=True)
        with tarfile.open(evidence/'board_return.tar.gz') as archive:
            for member in archive.getmembers():
                target=(evidence/member.name).resolve()
                if not member.isfile() or not target.is_relative_to(evidence):raise ValueError('unsafe result member')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        if result.returncode:raise RuntimeError('Board failed; retained kernel and result evidence')
        source=json.loads((folder/'manifest.json').read_text())
        layers=json.loads((evidence/'layers.json').read_text())
        assert len(layers)==len(source['layers'])
        for row,expected in zip(layers,source['layers']):
            assert row['node']==expected['node'] and row['mismatch_count']==0
            assert row['output_sha256']==expected['output_sha256']
        heads=[]
        for head in source['plan']['outputs']:
            raw=(evidence/(head['level']+'.bin')).read_bytes()
            sha=hashlib.sha256(raw).hexdigest()
            expected=next(x for x in source['layers'] if x['node']==head['edge'])
            assert len(raw)==head['real_bytes'] and sha==expected['output_sha256']
            heads.append(dict(level=head['level'],bytes=len(raw),sha256=sha))
        completion=json.loads((evidence/'completion.json').read_text())
        assert completion['status']=='FPGA_GRAPH_PARITY_PASS'
        (evidence/'local_verification.json').write_text(json.dumps(dict(
            status='PASS',nodes=len(layers),heads=heads,source_sha256=source_sha,
            board_archive_sha256=hashlib.sha256((evidence/'board_return.tar.gz').read_bytes()).hexdigest(),
            scope=source['scope'],real_weights_bound=source.get('real_weights_bound',False),test_evaluated=False),indent=2))
        print('LOCAL_GRAPH_AND_RAW_HEAD_VERIFICATION_PASS',flush=True)
    finally:
        stop=f'''import os,signal
from pathlib import Path
pid={pid};p=Path('/proc')/str(pid)/'cmdline'
if p.exists():
 c=p.read_bytes()
 assert b'http_transfer.py' in c and {transfer!r}.encode() in c
 os.kill(pid,signal.SIGTERM)
'''
        remote_python(stop)


if __name__=='__main__':main()
