"""Single GPU, locked, fail-closed V0.2 initial seed42 queue.

This filename intentionally avoids shadowing Python's standard-library queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import train


def run(result_root: Path, device: str, resume: bool = False):
    train.load_manifest(result_root, True)
    result_root.mkdir(parents=True, exist_ok=True)
    handle = (result_root/'queue.lock').open('a+')
    # Kernel-managed locking releases on a crashed process; no stale-PID guess.
    if os.name == 'nt':
        import msvcrt
        handle.seek(0)
        if (result_root/'queue.lock').stat().st_size == 0:
            handle.write('0'); handle.flush(); handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = dict(execution_manifest_sha256=train.sha(result_root/'execution_manifest.json'),
        order=list(train.PROTOTYPES), seeds=[42], device=device,
        concurrency=1, test_evaluated=False, automatic_hardware_or_nas_promotion=False)
    state_path = result_root/'queue_status.json'
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding='utf-8'))
        if old['binding'] != binding: raise ValueError('Existing queue identity differs')
        if old['status'] == 'COMPLETE':
            print('QUEUE ALREADY COMPLETE', flush=True)
            return old
        if not resume:
            raise FileExistsError('Existing queue requires explicit --resume; no duplicate launch')
    state = dict(status='RUNNING', pid=os.getpid(), binding=binding,
        started_at=train.now(), completed=[], active=None)
    train.dump(state_path, state)
    try:
        for prototype in train.PROTOTYPES:
            folder = train.result_dir(result_root, prototype, 42)
            status = folder/'status.json'
            if status.exists() and json.loads(status.read_text())['status'] == 'COMPLETE':
                train.summarize(result_root, prototype, 42)
                state['completed'].append(dict(prototype=prototype, result_sha256=train.sha(folder/'result.json')))
                continue
            checkpoint = folder/'latest.pt'
            if (checkpoint.exists() or (folder/'run_binding.json').exists()) and not resume:
                raise FileExistsError(f'{prototype} existing formal invocation requires explicit --resume')
            command = [sys.executable, str(Path(__file__).with_name('train.py')), 'train',
                '--prototype', prototype, '--seed', '42', '--device', device,
                '--result-root', str(result_root)]
            if checkpoint.exists(): command.append('--resume')
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'stdout.log').open('a', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=train.ROOT)
                state.update(active=prototype, child_pid=child.pid, updated_at=train.now())
                train.dump(state_path, state)
                print(f'QUEUE START {prototype} seed=42 pid={child.pid}', flush=True)
                code = child.wait()
            if code:
                raise RuntimeError(f'{prototype} exited {code}; queue stopped without autoretry')
            train.summarize(result_root, prototype, 42)
            state['completed'].append(dict(prototype=prototype, result_sha256=train.sha(folder/'result.json')))
            state.update(active=None, child_pid=None, updated_at=train.now())
            train.dump(state_path, state)
        state.update(status='COMPLETE', active=None, child_pid=None, updated_at=train.now())
        train.dump(state_path, state)
        return state
    except Exception as error:
        state.update(status='FAILED_NO_AUTORETRY', error_type=type(error).__name__,
            error=str(error), updated_at=train.now())
        train.dump(state_path, state)
        raise
    finally:
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=train.DEFAULT_ROOT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.result_root.resolve(), args.device, args.resume)


if __name__ == '__main__': main()
