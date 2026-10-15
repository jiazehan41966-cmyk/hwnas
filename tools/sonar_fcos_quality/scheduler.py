"""Finite, single-GPU, epoch-resumable side queue; never interrupts other jobs."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.sonar_fcos_pilot.common import ROOT, read_json, resolve_path, utc_now, write_json


def gpu_snapshot():
    """Read resource ownership without recording process arguments or environment."""
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits']
    lines = subprocess.check_output(command, text=True, timeout=15)
    cards = []
    for line in lines.splitlines():
        index, uuid, name, used, total, utilization = [v.strip() for v in line.split(',')]
        cards.append({'index': int(index), 'uuid': uuid, 'name': name,
                      'memory_used_mib': int(used), 'memory_total_mib': int(total),
                      'utilization_percent': int(utilization), 'compute_pids': []})
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    by_uuid = {row['uuid']: row for row in cards}
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid in by_uuid:
            by_uuid[uuid]['compute_pids'].append(int(pid))
    return cards


def select_idle_gpu(cards, scheduling=None):
    scheduling = scheduling or {}
    # GPU 1 is permanently reserved for the mainline; all paired runs use GPU 0.
    allowed = scheduling.get('gpu_indices', [0])
    reserved = set(scheduling.get('reserved_gpu_indices', [1])) | {1}
    for index in allowed:
        card = next((r for r in cards if r['index'] == index), None)
        if (card is not None and index not in reserved
                and card['name'] == 'NVIDIA GeForce RTX 5090'
                and not card['compute_pids']
                and card['memory_used_mib'] <= scheduling.get('max_memory_used_mib', 500)
                and card['utilization_percent'] <= scheduling.get('max_utilization_percent', 5)
                and card['memory_total_mib'] - card['memory_used_mib'] >= 12000):
            return card
    return None


def process_family(pid):
    """Own subprocess and descendants only; inaccessible ownership fails closed."""
    parents = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            # comm may contain spaces or parentheses, so split only after its last ')'.
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(directory.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            continue  # Unrecognized PID is foreign, hence cannot be ignored below.
    family = {pid}
    while True:
        expanded = family | {child for child, parent in parents.items() if parent in family}
        if expanded == family:
            return family
        family = expanded


def foreign_pids(cards, gpu_index, own_pids):
    card = next((r for r in cards if r['index'] == gpu_index), None)
    if card is None:
        raise RuntimeError('Selected GPU disappeared')
    return sorted(set(card['compute_pids']) - set(own_pids))


def run_order(config):
    if config['variants'] != ['AAA', 'QFL_DFL'] or config['seeds'] != [42, 43, 44]:
        raise ValueError('The finite queue is exactly AAA/QFL_DFL x seeds 42-44')
    if config['epochs'] != 30 or config['max_formal_runs'] != 6:
        raise ValueError('The queue cannot exceed six 30-epoch runs')
    return [(variant, seed) for seed in config['seeds'] for variant in config['variants']]


def run_progress(output, variant, seed, checkpoint_root=None, expected_source=None):
    path = Path(output) / 'runs' / f'{variant}_seed{seed}' / 'status.json'
    result = read_json(path) if path.exists() else {'status': 'PENDING', 'completed_epochs': 0}
    epochs = int(result.get('completed_epochs', result.get('trained_epochs', 0)))
    if checkpoint_root is not None:
        checkpoint = Path(checkpoint_root) / f'{variant}_seed{seed}' / 'last.pt'
        if checkpoint.exists():
            import torch
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            binding = saved['binding']
            if binding['variant'] != variant or binding['seed'] != seed:
                raise ValueError('Checkpoint belongs to a different run')
            if expected_source is not None and binding['source_binding'] != expected_source:
                raise ValueError('Committed checkpoint source binding differs')
            epochs = int(saved['epoch']) + 1
            if len(saved['history']) != epochs:
                raise ValueError('Committed checkpoint history length differs')
        elif epochs:
            raise ValueError('Status claims epochs without a committed checkpoint')
    if epochs < 0 or epochs > 30:
        raise ValueError('Run epoch count is outside the authorized budget')
    if result['status'] == 'COMPLETE' and epochs != 30:
        raise ValueError('A COMPLETE run must contain exactly thirty committed epochs')
    return {**result, 'completed_epochs': epochs}


@contextlib.contextmanager
def singleton(path):
    """Kernel-held lock releases on crash; a stale file is not a stale lock."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A side queue already owns this campaign') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def queue(config_path):
    from .campaign import load_config, source_binding, summarize
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output = resolve_path(config['output_dir'])
    checkpoint_root = resolve_path(config['checkpoint_root'])
    source = source_binding(config)
    scheduling = config.get('scheduling', {})
    interval = min(60., max(2., float(scheduling.get('poll_seconds', 20))))
    primary = resolve_path(scheduling.get('primary_request_file', str(output / 'PRIMARY_GPU_REQUEST')))
    stop = output / 'STOP_QUEUE'
    pause = output / 'scheduler' / 'PAUSE_ACTIVE_EPOCH'
    status_path = output / 'queue_status.json'
    (output / 'scheduler').mkdir(parents=True, exist_ok=True)
    manifest = read_json(output / 'execution_manifest.json')
    if manifest['source_binding'] != source:
        raise RuntimeError('Execution source/config binding changed')
    order = run_order(config)

    def status(state, **extra):
        value = {'status': state, 'updated_at': utc_now(), 'pid': os.getpid(),
                 'finite_runs': 6, 'max_epochs_per_run': 30, 'mainline_gpu_reserved': [1],
                 'run_order': [{'variant': v, 'seed': s} for v, s in order], **extra}
        write_json(status_path, value)
        return value

    with singleton(output / 'scheduler' / 'queue.lock'):
        try:
            for variant, seed in order:
                while True:
                    before = run_progress(output, variant, seed, checkpoint_root, source)
                    if before['status'] == 'COMPLETE':
                        summarize(config)
                        break
                    if stop.exists():
                        return status('PAUSED_BY_REQUEST', active_run=f'{variant}_seed{seed}')
                    if primary.exists():
                        status('WAITING_MAINLINE', active_run=f'{variant}_seed{seed}')
                        time.sleep(interval)
                        continue
                    cards = gpu_snapshot()
                    selected = select_idle_gpu(cards, scheduling)
                    if selected is None:
                        status('WAITING_IDLE_GPU', gpu_snapshot=cards, active_run=f'{variant}_seed{seed}')
                        time.sleep(interval)
                        continue
                    if pause.exists():
                        pause.unlink()
                    # Narrow the allocation race by observing idle ownership again immediately.
                    selected2 = select_idle_gpu(gpu_snapshot(), scheduling)
                    if selected2 is None or selected2['index'] != selected['index'] or primary.exists() or stop.exists():
                        continue
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(selected['index']),
                               SONAR_FCOS_QUALITY_PAUSE_FILE=str(pause), PYTHONUNBUFFERED='1')
                    command = [sys.executable, '-m', 'tools.sonar_fcos_quality.train', 'run',
                               '--config', str(config_path), '--variant', variant, '--seed', str(seed),
                               '--epoch-budget', '1', '--pause-file', str(pause)]
                    log_path = output / 'scheduler' / f'{variant}_seed{seed}.log'
                    with log_path.open('a', encoding='utf-8') as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        status('RUNNING', active_run=f'{variant}_seed{seed}', child_pid=child.pid,
                               physical_gpu=selected['index'], completed_epochs=before['completed_epochs'], command=command)
                        pause_reason = None
                        while child.poll() is None:
                            time.sleep(interval)
                            if child.poll() is not None:
                                break
                            try:
                                foreign = foreign_pids(gpu_snapshot(), selected['index'], process_family(child.pid))
                                pause_reason = pause_reason or ('stop_queue' if stop.exists() else 'primary_request'
                                                if primary.exists() else 'foreign_gpu_process' if foreign else None)
                            except Exception as exc:
                                pause_reason = pause_reason or 'resource_observation_failed:' + type(exc).__name__
                            if pause_reason:
                                pause.write_text(pause_reason + '\n', encoding='utf-8')
                                status('YIELDING_GPU', active_run=f'{variant}_seed{seed}', child_pid=child.pid,
                                       physical_gpu=selected['index'], reason=pause_reason)
                        rc = child.wait()
                    after = run_progress(output, variant, seed, checkpoint_root, source)
                    if rc == 75:
                        if not pause_reason:
                            raise RuntimeError('Unexpected pause exit without a scheduler request')
                        status('YIELDED_GPU', active_run=f'{variant}_seed{seed}', reason=pause_reason,
                               completed_epochs=after['completed_epochs'])
                        continue
                    if rc != 0:
                        raise RuntimeError(f'{variant}_seed{seed} exited {rc}; inspect {log_path}')
                    delta = after['completed_epochs'] - before['completed_epochs']
                    if delta != 1 and not (delta == 0 and after['status'] == 'COMPLETE' and before['completed_epochs'] == 30):
                        raise RuntimeError('A one-epoch child made unexpected committed progress')
            completed = subprocess.run([sys.executable, '-m', 'tools.sonar_fcos_quality.train', 'summarize',
                                        '--config', str(config_path)], cwd=ROOT, check=True)
            return status('COMPLETE', completed_runs=6, summarize_returncode=completed.returncode)
        except BaseException as exc:
            # Cooperative stop affects only our child. Do not signal another user's process.
            if 'child' in locals() and child.poll() is None:
                pause.write_text('scheduler_exit\n', encoding='utf-8')
            status('ERROR', error_type=type(exc).__name__, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(queue(args.config), ensure_ascii=False))


if __name__ == '__main__':
    main()
