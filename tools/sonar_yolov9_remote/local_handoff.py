"""One-time local COCO handoff: suspend only the queue, retain scratch child."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/sonar_yolov9_baseline_20261005'
sys.path.insert(0, str(RESULT / 'env_overlay'))
import psutil

CONTROL = RESULT / 'local_handoff_20261005_01'
QUEUE = RESULT / 'queue_status.json'
MANIFEST_SHA = 'ad898d3713055aeda890793304afb6062645f16c3efd69ae3ee6674d62e39f0d'
EXPECTED = {
    38752: dict(parent=43316, created='2026-10-05T11:43:07.545408+08:00', kind='queue'),
    38308: dict(parent=38752, created='2026-10-05T11:43:07.575253+08:00', kind='queue'),
    42512: dict(parent=38308, created='2026-10-05T12:39:18.667822+08:00', kind='scratch'),
    42992: dict(parent=42512, created='2026-10-05T12:39:18.680382+08:00', kind='scratch'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def cim_snapshot():
    command = """Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(38752,38308,42512,42992) } | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId; parent=$_.ParentProcessId; created=$_.CreationDate.ToUniversalTime().ToString('o'); command=$_.CommandLine} } | ConvertTo-Json -Depth 3"""
    completed = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                               capture_output=True, text=True, check=True)
    if not completed.stdout.strip():
        return {}
    payload = json.loads(completed.stdout)
    return {int(row['pid']): row for row in (payload if isinstance(payload, list) else [payload])}


def require_identity(snapshot, pid):
    if pid not in snapshot:
        raise psutil.NoSuchProcess(pid)
    row, expected = snapshot[pid], EXPECTED[pid]
    delta = abs(datetime.fromisoformat(row['created']).timestamp() -
                datetime.fromisoformat(expected['created']).timestamp())
    if delta > .000002 or row['parent'] != expected['parent']:
        raise ValueError(f'Process PID/creation/parent binding drift: {pid}')
    normalized = row['command'].replace('\\', '/').lower()
    script = 'queue_runner.py' if expected['kind'] == 'queue' else 'train.py'
    if ('tools/sonar_yolov9_baseline/' + script not in normalized or
            RESULT.as_posix().lower() not in normalized):
        raise ValueError(f'Unexpected process command: {pid}')
    if expected['kind'] == 'scratch' and '--group yolov9_t_scratch' not in normalized:
        raise ValueError(f'Unexpected training group: {pid}')
    return row


def same_process(pid, creation):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.create_time() == creation
    except psutil.NoSuchProcess:
        return False


def suspend_queue_only(pid):
    if pid != 38308:
        raise ValueError('Only the verified actual queue PID is authorized for suspension')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    native = ctypes.WinDLL('ntdll')
    native.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    native.NtSuspendProcess.restype = wintypes.LONG
    handle = kernel.OpenProcess(0x0800 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status = native.NtSuspendProcess(handle)
        if status != 0:
            raise RuntimeError(f'NtSuspendProcess returned NTSTATUS {status:#x}')
    finally:
        kernel.CloseHandle(handle)
    return status


def apply():
    if CONTROL.exists():
        raise FileExistsError('One-time handoff path already exists; preserve its evidence')
    if sha(RESULT / 'execution_manifest.json') != MANIFEST_SHA:
        raise ValueError('Frozen execution manifest drift')
    status = json.loads(QUEUE.read_text(encoding='utf-8'))
    if (status['pid'] != 38308 or status['child_pid'] != 42512 or
            status['active'] != 'YOLOv9_T_scratch' or status['status'] != 'RUNNING' or
            status['completed'] != ['L1_scratch'] or
            status['pending'] != ['YOLOv9_T_scratch', 'YOLOv9_T_COCO'] or
            status['execution_manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Queue no longer matches the authorized handoff state')
    snapshot = cim_snapshot()
    identities = {str(pid): require_identity(snapshot, pid) for pid in EXPECTED}
    creation = {str(pid): psutil.Process(pid).create_time() for pid in EXPECTED}
    CONTROL.mkdir(parents=True)
    original = QUEUE.read_bytes()
    (CONTROL / 'original_queue_status.json').write_bytes(original)
    receipt = dict(status='PREPARED_NOT_YET_SUSPENDED', execution_manifest_sha256=MANIFEST_SHA,
                   original_queue_status_sha256=hashlib.sha256(original).hexdigest(),
                   processes=identities, process_creation_epoch=creation, created_at=now(),
                   scope='suspend_only_38308_keep_42512_42992_then_cleanup_queue_after_scratch_completion',
                   local_COCO_cancelled_before_launch=True, remote_COCO_status='PENDING_REMOTE_LAUNCH',
                   frozen_source_config_manifest_changed=False, test_evaluated=False)
    dump(CONTROL / 'handoff_receipt.json', receipt)
    native_status = suspend_queue_only(38308)
    receipt.update(status='QUEUE_SUSPENDED_SCRATCH_CONTINUES', native_suspend_status=native_status,
                   suspended_at=now(), queue_psutil_status=psutil.Process(38308).status())
    if not all(same_process(pid, creation[str(pid)]) for pid in (42512, 42992)):
        receipt.update(status='QUEUE_SUSPENDED_CHILD_IDENTITY_CHANGED_STOP_REVIEW_REQUIRED')
        dump(CONTROL / 'handoff_receipt.json', receipt)
        raise RuntimeError('Scratch process identity changed during suspension; parent remains suspended')
    status.update(status='STOPPED_AFTER_HANDOFF_CURRENT_CHILD_CONTINUES',
                  scheduler_suspended=True, scheduler_suspended_pid=38308,
                  pending=['YOLOv9_T_scratch'], remote_group='YOLOv9_T_COCO',
                  remote_COCO_status='PENDING_REMOTE_LAUNCH', local_COCO_start_disabled=True,
                  handoff_receipt=str((CONTROL / 'handoff_receipt.json').relative_to(ROOT)),
                  updated_at=now())
    dump(QUEUE, status)
    dump(CONTROL / 'handoff_receipt.json', receipt)
    with (CONTROL / 'observer_stdout.log').open('a', encoding='utf-8') as log:
        observer = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'observe'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    receipt.update(observer_wrapper_pid=observer.pid, observer_started_at=now(), observer_hidden=True)
    dump(CONTROL / 'handoff_receipt.json', receipt)
    print(json.dumps(dict(status=receipt['status'], suspended_pid=38308,
                          retained_child_pids=[42512, 42992], observer_wrapper_pid=observer.pid,
                          receipt=str(CONTROL / 'handoff_receipt.json'))), flush=True)


def validate_scratch_completion():
    folder = RESULT / 'formal/YOLOv9_T_scratch/seed_42'
    row = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    final = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    history = json.loads((folder / 'epoch_log.json').read_text(encoding='utf-8'))
    expected = dict(execution_manifest_sha256=MANIFEST_SHA, group='YOLOv9_T_scratch',
                    seed=42, mode='formal', test_evaluated=False)
    if (row['epochs'] != 80 or row['binding'] != expected or row['formal_eligible'] is not True or
            row['test_evaluated'] is not False or final['status'] != 'COMPLETE' or
            final['epochs'] != 80 or final['binding'] != expected or
            [item['epoch'] for item in history] != list(range(1, 81)) or
            any(item['valid']['num_images'] != 99 for item in history)):
        raise ValueError('Scratch child exited without complete bound 80-epoch result')
    for name, key in (('latest.pt', 'latest_checkpoint_sha256'), ('best.pt', 'best_checkpoint_sha256'),
                      ('epoch_log.json', 'epoch_log_sha256')):
        if sha(folder / name) != row[key]:
            raise ValueError('Scratch completed artifact SHA drift: ' + name)
    return row


def observe():
    receipt = json.loads((CONTROL / 'handoff_receipt.json').read_text(encoding='utf-8'))
    creation = receipt['process_creation_epoch']
    state = dict(status='WAITING_FOR_SCRATCH_ONLY', pid=os.getpid(),
                 execution_manifest_sha256=MANIFEST_SHA, started_at=now(), test_evaluated=False)
    dump(CONTROL / 'observer_status.json', state)
    try:
        while any(same_process(pid, creation[str(pid)]) for pid in (42512, 42992)):
            time.sleep(10)
        row = validate_scratch_completion()
        if sha(RESULT / 'execution_manifest.json') != MANIFEST_SHA:
            raise ValueError('Execution manifest changed before queue cleanup')
        # No training process is terminated. Only original bound queue processes
        # are eligible, after both original scratch processes have exited.
        cleaned = []
        for pid in (38308, 38752):
            if same_process(pid, creation[str(pid)]):
                try:
                    snapshot = cim_snapshot()
                    require_identity(snapshot, pid)
                    psutil.Process(pid).terminate()
                    cleaned.append(pid)
                except psutil.NoSuchProcess:
                    # The venv wrapper may exit immediately when its bound
                    # Python queue child is terminated. This is normal cleanup.
                    continue
        status = json.loads(QUEUE.read_text(encoding='utf-8'))
        if status['execution_manifest_sha256'] != MANIFEST_SHA or not status['local_COCO_start_disabled']:
            raise ValueError('Queue administrative handoff binding drift')
        status.update(status='LOCAL_SCRATCH_COMPLETE_REMOTE_COCO_HANDOFF',
                      completed=['L1_scratch', 'YOLOv9_T_scratch'], pending=[],
                      active=None, child_pid=None, scheduler_cleanup_pids=cleaned,
                      scheduler_suspended=False, updated_at=now())
        dump(QUEUE, status)
        state.update(status='COMPLETE_SCRATCH_VALIDATED_QUEUE_CLEANED', cleaned_queue_pids=cleaned,
                     scratch_result_sha256=sha(RESULT / 'formal/YOLOv9_T_scratch/seed_42/result.json'),
                     scratch_epochs=row['epochs'], completed_at=now(), training_children_terminated=False)
        dump(CONTROL / 'observer_status.json', state)
        print(json.dumps(state), flush=True)
    except Exception as error:
        state.update(status='FAILED_REVIEW_REQUIRED_NO_TRAINING_CHILD_TERMINATED',
                     error_type=type(error).__name__, error=str(error), failed_at=now())
        dump(CONTROL / 'observer_status.json', state)
        raise


def recover_completed_handoff():
    """Explicit recovery of this failed observer, without rerunning apply."""
    recovery_path = CONTROL / 'manual_recovery_20261005_01.json'
    if recovery_path.exists():
        raise FileExistsError('Manual recovery receipt exists; preserve the completed operation')
    receipt = json.loads((CONTROL / 'handoff_receipt.json').read_text(encoding='utf-8'))
    creation = receipt['process_creation_epoch']
    failure_path = CONTROL / 'observer_status.json'
    failure = json.loads(failure_path.read_text(encoding='utf-8'))
    if (failure['status'] != 'FAILED_REVIEW_REQUIRED_NO_TRAINING_CHILD_TERMINATED' or
            failure['error_type'] != 'JSONDecodeError' or
            receipt['execution_manifest_sha256'] != MANIFEST_SHA or
            sha(RESULT / 'execution_manifest.json') != MANIFEST_SHA):
        raise ValueError('Manual recovery is limited to this bound observer failure')
    if any(psutil.pid_exists(pid) for pid in (42512, 42992)):
        raise RuntimeError('Both original training PID values must be absent before recovery')
    scratch = validate_scratch_completion()
    local_coco = RESULT / 'formal/YOLOv9_T_COCO/seed_42'
    if any((local_coco / name).exists() for name in ('run_binding.json', 'latest.pt', 'result.json')):
        raise RuntimeError('Local COCO formal artifacts exist; independent duplicate audit required')
    remote_snapshot_path = RESULT / 'remote_coco/startup_verification.json'
    remote_snapshot = json.loads(remote_snapshot_path.read_text(encoding='utf-8'))
    if (remote_snapshot['pid'] != 1204207 or remote_snapshot['gpu'] != 2 or
            remote_snapshot['process_alive'] is not True or
            remote_snapshot['status.json']['status'] != 'RUNNING' or
            remote_snapshot['status.json']['binding']['group'] != 'YOLOv9_T_COCO'):
        raise ValueError('Remote startup receipt differs from the authorized handoff destination')
    queue_before = QUEUE.read_bytes()
    status = json.loads(queue_before)
    if status['execution_manifest_sha256'] != MANIFEST_SHA or status['local_COCO_start_disabled'] is not True:
        raise ValueError('Queue handoff administrative binding drift')
    for original, preserved in ((failure_path, CONTROL / 'observer_status.failed_original.json'),
                                (CONTROL / 'observer_stdout.log', CONTROL / 'observer_stdout.failed_original.log')):
        if preserved.exists():
            raise FileExistsError('Preserved observer failure already exists')
        preserved.write_bytes(original.read_bytes())
    (CONTROL / 'queue_status.before_manual_recovery.json').write_bytes(queue_before)
    operations = []
    for pid in (38308, 38752):
        if same_process(pid, creation[str(pid)]):
            try:
                require_identity(cim_snapshot(), pid)
                psutil.Process(pid).terminate()
                operations.append(dict(pid=pid, operation='terminated_verified_queue_only'))
            except psutil.NoSuchProcess:
                operations.append(dict(pid=pid, operation='already_exited_during_cleanup'))
        else:
            operations.append(dict(pid=pid, operation='original_process_already_exited_no_action'))
    for pid in (38308, 38752):
        if same_process(pid, creation[str(pid)]):
            psutil.Process(pid).wait(timeout=5)
    if any(same_process(pid, creation[str(pid)]) for pid in (38308, 38752)):
        raise RuntimeError('Original scheduler process still exists after verified cleanup')
    status.update(status='LOCAL_SCRATCH_COMPLETE_REMOTE_COCO_RUNNING',
                  completed=['L1_scratch', 'YOLOv9_T_scratch'], pending=[], active=None, child_pid=None,
                  scheduler_suspended=False, scheduler_processes_exited=True,
                  remote_COCO_status='RUNNING', remote_COCO_pid=1204207, remote_COCO_gpu=2,
                  remote_COCO_root='/home/jiajia/experiments/sonar_yolov9_baseline_20261005_remote_coco',
                  remote_status_evidence='remote_coco/startup_verification.json',
                  remote_status_evidence_updated_at=remote_snapshot['status.json']['updated_at'],
                  local_COCO_start_disabled=True, manual_recovery_receipt=str(recovery_path.relative_to(ROOT)),
                  updated_at=now())
    dump(QUEUE, status)
    recovered = dict(status='PASS', failure_reason='venv_queue_wrapper_exited_between_live_check_and_CIM_snapshot',
                     original_failure_preserved=True, scheduler_operations=operations,
                     training_PID_values_absent=[42512, 42992], training_children_terminated=False,
                     scratch_epochs=scratch['epochs'], scratch_result_sha256=sha(RESULT / 'formal/YOLOv9_T_scratch/seed_42/result.json'),
                     scratch_artifact_hashes_validated=True, local_COCO_formal_not_started=True,
                     remote_COCO_status='RUNNING', remote_COCO_pid=1204207,
                     remote_COCO_root=status['remote_COCO_root'],
                     remote_startup_verification_sha256=sha(remote_snapshot_path),
                     execution_manifest_sha256=MANIFEST_SHA, frozen_source_config_manifest_changed=False,
                     new_training_started=False, test_evaluated=False, recovered_at=now())
    dump(recovery_path, recovered)
    dump(CONTROL / 'observer_recovery_status.json', recovered)
    print(json.dumps(recovered, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('apply', 'observe', 'recover'))
    arguments = parser.parse_args()
    {'apply': apply, 'observe': observe, 'recover': recover_completed_handoff}[arguments.mode]()
