"""Finite collector/calibrator for the four-network sonar/front experiment."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
VERSION='zcu104_width_e_sonar_serial_gap_v2'
REMOTE='/home/jiajia/experiments/zcu104_sonar_front_integration_20260913_v1'
HOST='jiajia@10.70.23.121'

def main():
    out=ROOT/'results/zcu104_sonar_front_integration_20260913_v1';out.mkdir(exist_ok=True)
    status=out/'finalizer_status.json'
    if status.exists(): raise FileExistsError('Preserve the previous finite finalizer status')
    scripts=['calibrate_zcu104_sonar_front_storage.py','build_zcu104_sonar_front_space.py']
    hashes={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in scripts}
    state=dict(status='WAITING_WHOLE_NETWORKS',pid=os.getpid(),source_sha256=hashes,
               started_at=datetime.now(timezone.utc).isoformat(),nas_started=False,version=VERSION,commands=[],errors=[])
    env=os.environ.copy();env.update(SSH_ASKPASS='C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/ssh_askpass.cmd',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex:0')
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=status.with_suffix('.json.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status)
    def execute(cmd,input_text=None,record=True,timeout=900):
        if record: state['commands'].append(cmd);save()
        p=subprocess.run(cmd,input=input_text,capture_output=True,text=True,env=env,cwd=ROOT,timeout=timeout)
        if record:
            with (out/'commands.log').open('a') as f: f.write(p.stdout+p.stderr+'\n')
        if p.returncode: raise RuntimeError(f'Command exited {p.returncode}: {p.stderr[-1200:]}')
        return p.stdout
    def run_local(name,args=()):
        if hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()!=hashes[name]: raise ValueError('Analysis code changed: '+name)
        execute([sys.executable,str(ROOT/'scripts'/name),*args])
    save();deadline=time.monotonic()+12*3600
    try:
        while True:
            if time.monotonic()>deadline: raise TimeoutError('Finite 12-hour finalization deadline reached')
            code="from pathlib import Path\np=Path('"+REMOTE+"/integration/status.json')\nprint(p.read_text())\n"
            try:
                remote=json.loads(execute(['C:/Windows/System32/OpenSSH/ssh.exe','-p','7889','-o','ConnectTimeout=12','-o','NumberOfPasswordPrompts=1','-o','StrictHostKeyChecking=yes',HOST,'python3 -'],input_text=code,record=False,timeout=40))
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                state['errors'].append(str(error));save();time.sleep(30);continue
            state['remote']=remote;save()
            if remote['status'] in ['ERROR','SEED_REJECTED_CALIBRATION_QUEUE_NOT_RUN','CALIBRATION_POINTS_FINISHED_PENDING_MODEL_AUDIT']: break
            time.sleep(30)
        destination=out/'remote_completion';destination.mkdir(exist_ok=True)
        execute(['C:/Windows/System32/OpenSSH/scp.exe','-P','7889','-o','StrictHostKeyChecking=yes','-r',
                 *[HOST+':'+REMOTE+'/'+p for p in ['integration','preparation.json','manifest.json','launch.json','launch.log']],str(destination)])
        if (remote['status']=='CALIBRATION_POINTS_FINISHED_PENDING_MODEL_AUDIT'
            and len(remote['runs'])==4 and all(r.get('actual') for r in remote['runs'])):
            run_local(scripts[0],['--integration',str(destination/'integration')])
        run_local(scripts[1])
        result=json.loads((ROOT/'artifacts'/VERSION/'space_preflight.json').read_text())
        state.update(status='PREFLIGHT_READY_NO_NAS_STARTED' if result['nas_resume_allowed'] else 'FINISHED_WITH_ADMISSION_BLOCKERS',
                     core_feasible_seed=remote.get('core_feasible_seed'),nas_resume_allowed=result['nas_resume_allowed'],
                     legal_architectures=result['legal_architectures'],operator_sum_proxy_feasible=result['operator_sum_proxy_feasible'],
                     calibrated_proxy_feasible=result['calibrated_proxy_feasible'])
        lines=['# 声呐前段扩展与整网校准结果','',f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",'',
            f"搜索空间：{result['legal_architectures']} 个合法架构；算子求和可行 {result['operator_sum_proxy_feasible']} 个；校准后可行数：{result['calibrated_proxy_feasible']}。",
            f"NAS 准入：{result['nas_resume_allowed']}；本流程未启动 NAS。",'',
            '| 架构 | 整网资源及时序 | LUT | LUTRAM | BRAM36 | DSP | WNS ns |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in remote['runs']:
            actual=row.get('actual',{});timing=row.get('timing',{})
            lines.append(f"| {row['pilot_id']} | {row['status']} | {actual.get('lut','待定')} | {actual.get('lut_as_memory','待定')} | {actual.get('bram36','待定')} | {actual.get('dsp','待定')} | {timing.get('wns_ns','待定')} |")
        lines+=['','证据范围：独立算子 HLS/布局布线、确定性参数整网核心资源及时序。当前旧测试壳没有板级 I/O 约束；不代表板端分类、实测延迟或实测功耗。',
                '校准采用三个训练架构，完整留出 C/DMDC 架构验证；若预测上界检查、基准时序或可行域检查未通过，准入保持关闭。']
        (out/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except Exception as error:
        state.update(status='ERROR',error=str(error));raise
    finally: save()

if __name__=='__main__': main()
