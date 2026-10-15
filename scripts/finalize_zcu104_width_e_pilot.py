"""Finish this single authorized hardware queue: collect, calibrate, preflight.

Finite local companion process, not a recurring monitor. Never starts NAS.
SSH authentication uses the existing local Credential Manager ASKPASS helper.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/jiajia/experiments/zcu104_width_e_extension_20260913_v1'
HOST='jiajia@10.70.23.121'
SSH='C:/Windows/System32/OpenSSH/ssh.exe'
SCP='C:/Windows/System32/OpenSSH/scp.exe'
READ_STATUS="""from pathlib import Path
import json
b=Path('/home/jiajia/experiments/zcu104_width_e_extension_20260913_v1')
result={}
for key,rel in [('operators','output/queue_status.json'),('integration','integration/status.json')]:
 p=b/rel
 result[key]=json.loads(p.read_text()) if p.exists() else {'status':'NOT_CREATED'}
print(json.dumps(result))
"""

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extension-root',type=Path,default=ROOT/'results/zcu104_width_e_extension_20260913_v1')
    parser.add_argument('--timeout-hours',type=float,default=12)
    args=parser.parse_args()
    out=args.extension_root.resolve()
    status=out/'finalizer_status.json'
    if status.exists(): raise FileExistsError('Preserve previous finalizer status before an explicit resume')
    state=dict(status='WAITING_HARDWARE',pid=os.getpid(),nas_started=False,
               started_at=datetime.now(timezone.utc).isoformat(),commands=[],errors=[])
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=status.with_suffix('.json.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status)
    env=os.environ.copy()
    env.update(SSH_ASKPASS='C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/ssh_askpass.cmd',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex:0')
    scripts=['build_zcu104_width_e_search_inputs.py','calibrate_zcu104_width_e_storage.py']
    frozen={str(ROOT/'scripts'/name):hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in scripts}
    state['source_sha256']=frozen
    def execute(command,*,input_text=None,record=True,timeout=300):
        if record: state['commands'].append(command);save()
        result=subprocess.run(command,input=input_text,text=True,capture_output=True,env=env,cwd=ROOT,timeout=timeout)
        if record:
            with (out/'finalizer_commands.log').open('a') as log: log.write(result.stdout+result.stderr+'\n')
        if result.returncode: raise RuntimeError(f'Command exited {result.returncode}: {result.stderr[-1000:]}')
        return result.stdout
    def copy_remote(paths,destination):
        destination.mkdir(parents=True,exist_ok=True)
        execute([SCP,'-P','7889','-o','StrictHostKeyChecking=yes','-r',*[HOST+':'+REMOTE+'/'+p for p in paths],str(destination)],timeout=900)
    def local_script(name,extra=()):
        path=ROOT/'scripts'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=frozen[str(path)]: raise ValueError('Analysis script changed after finalizer launch: '+name)
        return execute([sys.executable,str(path),*extra],timeout=900)
    save()
    deadline=time.monotonic()+args.timeout_hours*3600
    operator_done=False
    network_done=False
    try:
        while not (operator_done and network_done):
            if time.monotonic()>deadline: raise TimeoutError('Finite finalizer deadline reached; remote evidence preserved')
            try:
                snapshot=json.loads(execute([SSH,'-p','7889','-o','ConnectTimeout=12','-o','NumberOfPasswordPrompts=1',
                    '-o','StrictHostKeyChecking=yes',HOST,'python3 -'],input_text=READ_STATUS,record=False,timeout=40))
            except (subprocess.TimeoutExpired,RuntimeError) as error:
                state['errors'].append(dict(time=datetime.now(timezone.utc).isoformat(),error=str(error)))
                save();time.sleep(30);continue
            state['remote']=snapshot;save()
            op_status=snapshot['operators']['status']
            if not operator_done and op_status in ['OPERATOR_PILOT_COMPLETE','OPERATOR_PILOT_INCOMPLETE']:
                copy_remote(['output/operators','output/evidence','output/gap_witness','output/queue_status.json'],out/'remote_output')
                local_script('build_zcu104_width_e_search_inputs.py')
                operator_done=True;state['operator_inputs_collected']=True;save()
            integration_status=snapshot['integration']['status']
            if integration_status in ['ERROR','SEED_REJECTED_CALIBRATION_QUEUE_NOT_RUN','CALIBRATION_POINTS_FINISHED_PENDING_MODEL_AUDIT']:
                network_done=True
            if not (operator_done and network_done): time.sleep(30)
        copy_remote(['integration','integration_launch.json','integration_launch.log','launch.json','status.json'],out/'remote_completion')
        integration=out/'remote_completion/integration'
        if (snapshot['integration']['status']=='CALIBRATION_POINTS_FINISHED_PENDING_MODEL_AUDIT'
            and all(r.get('actual') is not None for r in snapshot['integration']['runs'])):
            local_script('calibrate_zcu104_width_e_storage.py',['--integration',str(integration)])
            local_script('build_zcu104_width_e_search_inputs.py')
        preflight=json.loads((ROOT/'artifacts/zcu104_width_e_sonar_serial_gap_v1/space_preflight.json').read_text())
        state.update(status='PREFLIGHT_READY_NO_NAS_STARTED' if preflight['nas_resume_allowed'] else 'FINISHED_WITH_ADMISSION_BLOCKERS',
                     nas_resume_allowed=preflight['nas_resume_allowed'],core_feasible_seed=snapshot['integration'].get('core_feasible_seed'),
                     operator_rows=preflight['ready_operator_rows'],required_operator_rows=preflight['required_operator_rows'],
                     legal_architectures=preflight['legal_architectures'],operator_sum_proxy_feasible=preflight['operator_sum_proxy_feasible'],
                     calibrated_proxy_feasible=preflight.get('calibrated_proxy_feasible'))
    except Exception as error:
        state.update(status='ERROR',error=str(error));raise
    finally: save()

if __name__=='__main__': main()
