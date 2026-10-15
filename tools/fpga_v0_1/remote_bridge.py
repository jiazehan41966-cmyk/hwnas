"""Credential-Manager-only SSH transport for this bounded closure run."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/home/jiajia/experiments/fpga_v0_1_closure_20260914_v1'
HOST='jiajia@10.70.23.121'
ART=ROOT/'artifacts/fpga_v0_1_closure/20260914_v1'
ENV=os.environ.copy()
ENV.update(SSH_ASKPASS='C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/ssh_askpass.cmd',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex:0')
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','7889','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','7889','-o','StrictHostKeyChecking=yes']


def main():
    action=sys.argv[1]
    def run(cmd,**kwargs):return subprocess.run(cmd,env=ENV,check=True,**kwargs)
    if action=='launch':
        archive=ART/'input_bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            tf.add(ROOT/'reports/fpga_v0_1_closure/representative_cases',arcname='cases')
            tf.add(ART/'export',arcname='export')
            tf.add(ROOT/'tools/fpga_v0_1/remote_witness.py',arcname='remote_witness.py')
        run(SSH+[f'mkdir {REMOTE}'])
        run(SCP+[str(archive),HOST+':'+REMOTE+'/input_bundle.tar.gz'])
        run(SSH+[f'cd {REMOTE} && tar -xzf input_bundle.tar.gz && /home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python remote_witness.py'])
    elif action=='fetch':
        (ART/'remote_results').mkdir(exist_ok=True)
        run(SCP+[HOST+':'+REMOTE+'/results/*',str(ART/'remote_results')])
    elif action=='status':
        run(SSH+[f'cat {REMOTE}/results/case_status.json'])
    else:raise ValueError(action)


if __name__=='__main__':main()
