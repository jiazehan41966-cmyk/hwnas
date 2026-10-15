"""Isolated H3 transport; credentials come only from existing local ASKPASS."""
import sys,subprocess,tarfile
from common import *
sys.path.insert(0,str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import ENV,SSH,SCP,HOST
REMOTE='/home/jiajia/experiments/fpga_h3_20260918_batch01'
def run(cmd,**kw):return subprocess.run(cmd,env=ENV,check=True,**kw)
if __name__=='__main__':
    action=sys.argv[1]
    extra=len(sys.argv)>2 and sys.argv[2]=='extra'
    aux=len(sys.argv)>2 and sys.argv[2]=='aux'
    retry=len(sys.argv)>2 and sys.argv[2]=='chain_retry'
    auto=len(sys.argv)>2 and sys.argv[2]=='chain_auto'
    stream=len(sys.argv)>2 and sys.argv[2]=='chain_stream'
    if extra:REMOTE=REMOTE.replace('batch01','batch02')
    if aux:REMOTE=REMOTE.replace('batch01','aux01')
    if retry:REMOTE=REMOTE.replace('batch01','aux02')
    if auto:REMOTE=REMOTE.replace('batch01','aux03')
    if stream:REMOTE=REMOTE.replace('batch01','aux04')
    if action=='launch':
        if aux:
            run(SSH+['test -f /home/jiajia/experiments/fpga_h3_20260918_batch02/results/DONE'])
        if retry:
            run(SSH+['test -f /home/jiajia/experiments/fpga_h3_20260918_aux01/results/DONE'])
        if auto:
            run(SSH+['test -f /home/jiajia/experiments/fpga_h3_20260918_aux02/results/DONE'])
        if stream:
            run(SSH+['test -f /home/jiajia/experiments/fpga_h3_20260918_aux03/results/DONE'])
        hashes=contracts()
        case_root=ART/('aux_stream_cases' if stream else 'aux_auto_cases' if auto else 'aux_retry_cases' if retry else 'aux_cases' if aux else 'extra_cases' if extra else 'cases')
        for case in case_root.iterdir():
            if not case.is_dir():continue
            key=json.loads((case/'measurement_key.json').read_text())
            assert not legality(key,hashes),(case.name,legality(key,hashes))
            assert all(sha(case/p)==h for p,h in key['source_files'].items()),case.name
            assert hashobj(key['source_files'])==key['source_hash'],case.name
        bundle=ART/('aux_stream_bundle.tar.gz' if stream else 'aux_auto_bundle.tar.gz' if auto else 'aux_retry_bundle.tar.gz' if retry else 'aux_bundle.tar.gz' if aux else 'extra_bundle.tar.gz' if extra else 'bundle.tar.gz')
        with tarfile.open(bundle,'w:gz') as tf:
            tf.add(case_root,arcname='cases');tf.add(Path(__file__).with_name('remote_run.py'),arcname='remote_run.py')
        run(SSH+[f'mkdir {REMOTE}']);run(SCP+[str(bundle),HOST+':'+REMOTE+'/bundle.tar.gz'])
        run(SSH+[f'cd {REMOTE} && tar -xzf bundle.tar.gz && python3 remote_run.py'])
    elif action=='fetch':
        dest=ART/('aux_stream_results' if stream else 'aux_auto_results' if auto else 'aux_retry_results' if retry else 'aux_results' if aux else 'extra_results' if extra else 'remote_results');dest.mkdir(exist_ok=True)
        run(SCP+['-r',HOST+':'+REMOTE+'/results/.',str(dest)])
    elif action=='status':
        run(SSH+[f'cat {REMOTE}/results/status.json'])
    else:raise ValueError(action)
