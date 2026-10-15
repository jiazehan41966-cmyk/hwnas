"""Use existing local-only Credential Manager transport in an isolated H2 directory."""
import sys,tarfile,subprocess,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_v0_1'))
from remote_bridge import ENV,SSH,SCP,HOST
ART=ROOT/'artifacts/fpga_h2/20260916_v1';OUT=ROOT/'reports/fpga_h2_contract'
REMOTE='/home/jiajia/experiments/fpga_h2_20260916_v1'
def main():
    action=sys.argv[1]
    attempt=sys.argv[2] if len(sys.argv)>2 else ''
    remote=REMOTE+('_'+attempt if attempt else '')
    def run(c,**k):return subprocess.run(c,env=ENV,check=True,**k)
    if action=='launch':
        # Bind the entire arithmetic implementation, not only the top source.
        for d in (OUT/'representative_cases').iterdir():
            if not d.is_dir():continue
            p=d/'case.json';c=json.loads(p.read_text())
            sources={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in d.iterdir() if x.suffix in ['.cpp','.h','.tcl']}
            c['source_files']=sources;c['source_hash']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();p.write_text(json.dumps(c,indent=2))
        bundle=ART/('h2_bundle'+('_'+attempt if attempt else '')+'.tar.gz')
        with tarfile.open(bundle,'w:gz') as tf:
            if len(sys.argv)>3:tf.add(OUT/'representative_cases'/sys.argv[3],arcname='cases/'+sys.argv[3])
            else:tf.add(OUT/'representative_cases',arcname='cases')
            tf.add(ART/'rtl',arcname='rtl');tf.add(Path(__file__).parent/'remote_run.py',arcname='remote_run.py')
        run(SSH+[f'mkdir {remote}']);run(SCP+[str(bundle),HOST+':'+remote+'/h2_bundle.tar.gz'])
        run(SSH+[f'cd {remote} && tar -xzf h2_bundle.tar.gz && python3 remote_run.py'])
    elif action=='fetch':
        dest=ART/('remote_results'+('_'+attempt if attempt else ''));dest.mkdir(exist_ok=True)
        run(SCP+[HOST+':'+remote+'/results/*',str(dest)])
    else:raise ValueError(action)
if __name__=='__main__':main()
