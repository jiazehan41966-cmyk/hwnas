"""Finite GPU workers with persistent seed-pair affinity and mainline priority."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.sonar_fcos_pilot.common import ROOT, read_json, resolve_path, sha256_file, utc_now, write_json
from tools.sonar_fcos_quality.scheduler import (gpu_snapshot, select_idle_gpu, foreign_pids,
    process_family, singleton, run_progress as parent_run_progress, run_order)

ALLOWED_GPUS = (0, 2, 3)


def run_progress(output, variant, seed, checkpoint_root, expected_source):
    result = parent_run_progress(output, variant, seed, checkpoint_root, expected_source)
    summary_path = Path(output) / 'runs' / f'{variant}_seed{seed}' / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        binding = summary['binding']
        if (summary.get('status') != 'COMPLETE' or summary.get('trained_epochs') != 30
                or result['completed_epochs'] != 30 or binding['variant'] != variant
                or binding['seed'] != seed or binding['source_binding'] != expected_source
                or summary['source_binding'] != expected_source):
            raise RuntimeError('Committed summary and checkpoint completion differ')
        result['status'] = 'COMPLETE'
    return result


def child_environment(card, pause):
    if card['index'] not in ALLOWED_GPUS or card['compute_pids']:
        raise ValueError('Only an idle approved physical GPU can be assigned')
    # NVIDIA device UUIDs avoid any CUDA-ordinal/NVML-index ordering ambiguity.
    return dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                SONAR_FCOS_QUALITY_PAUSE_FILE=str(pause), PYTHONUNBUFFERED='1')


def archive_stop_marker(output, *, resume, processes_clear):
    output = Path(output).resolve()
    stop = output / 'STOP_QUEUE'
    if not stop.exists():
        return None
    if not resume or not processes_clear:
        raise RuntimeError('Clearing a queue stop requires explicit resume and idle process/lock checks')
    destination = output / ('STOP_QUEUE.resumed_' + utc_now().replace(':', '').replace('+', '_'))
    if destination.exists():
        raise FileExistsError('A stop-marker archive already exists for this timestamp')
    stop.rename(destination)
    return str(destination)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextlib.contextmanager
def blocking_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initialize_claims(output, source, affinity=None):
    output = Path(output)
    path = output / 'scheduler' / 'seed_pair_claims.json'
    affinity = affinity or {}
    if any(int(seed) not in (42, 43, 44) or gpu not in ALLOWED_GPUS for seed, gpu in affinity.items()):
        raise ValueError('Invalid inherited seed/GPU affinity')
    with blocking_lock(output / 'scheduler' / 'claims.lock'):
        if path.exists():
            claims = read_json(path)
            if claims['source_binding_sha256'] != canonical_sha(source):
                raise ValueError('Persistent seed claims have a different source binding')
            for seed, gpu in affinity.items():
                if claims['pairs'][str(seed)]['gpu'] != gpu:
                    raise ValueError('Inherited paired GPU changed')
        else:
            claims = {'source_binding_sha256': canonical_sha(source), 'created_at': utc_now(),
                      'pairs': {str(seed): {'gpu': affinity.get(str(seed), affinity.get(seed)), 'complete': False}
                                for seed in (42, 43, 44)}}
            write_json(path, claims)
    return claims


def claim_pair(output, gpu, source, completed_seeds=()):
    """Called only after an idle check; one locked assignment owns both variants."""
    if gpu not in ALLOWED_GPUS:
        raise ValueError('GPU 1 and GPUs outside the approved set are never eligible')
    output = Path(output)
    with blocking_lock(output / 'scheduler' / 'claims.lock'):
        path = output / 'scheduler' / 'seed_pair_claims.json'
        claims = read_json(path)
        if claims['source_binding_sha256'] != canonical_sha(source):
            raise ValueError('Claim source binding changed')
        for seed in completed_seeds:
            claims['pairs'][str(seed)]['complete'] = True
        pairs = claims['pairs']
        owned = [int(s) for s, row in pairs.items() if row['gpu'] == gpu and not row['complete']]
        unassigned = [int(s) for s, row in pairs.items() if row['gpu'] is None and not row['complete']]
        selected = min(owned) if owned else min(unassigned) if unassigned else None
        if selected is not None and pairs[str(selected)]['gpu'] is None:
            pairs[str(selected)].update(gpu=gpu, assigned_at=utc_now())
        write_json(path, claims)
        return selected


def complete_seeds(output, source):
    """Inspect completion/bindings only; metric values are never exposed here."""
    complete = []
    for seed in (42, 43, 44):
        valid = True
        for variant in ('AAA', 'QFL_DFL'):
            path = Path(output) / 'runs' / f'{variant}_seed{seed}' / 'summary.json'
            if not path.exists():
                valid = False
                break
            item = read_json(path)
            if item.get('status') != 'COMPLETE' or item.get('trained_epochs') != 30 or item.get('source_binding') != source:
                raise ValueError('Invalid completed run summary: ' + str(path))
        if valid:
            complete.append(seed)
    return complete


def request_reason(output, gpu, foreign=()):
    output = Path(output)
    if (output / 'STOP_QUEUE').exists():
        return 'stop_queue'
    if (output / 'PRIMARY_GPU_REQUEST').exists() or (output / f'PRIMARY_GPU_REQUEST_GPU{gpu}').exists():
        return 'mainline_request'
    if foreign:
        return 'foreign_gpu_process'
    return None


def summarize_locked(config):
    from .campaign import summarize
    import torch
    torch.set_num_threads(2)
    output = resolve_path(config['output_dir'])
    with blocking_lock(output / 'scheduler' / 'summary.lock'):
        return summarize(config)


def completed_run_count(output, source):
    count = 0
    for seed in (42, 43, 44):
        for variant in ('AAA', 'QFL_DFL'):
            path = Path(output) / 'runs' / f'{variant}_seed{seed}' / 'summary.json'
            if not path.exists():
                continue
            item = read_json(path)
            if item.get('status') != 'COMPLETE' or item.get('trained_epochs') != 30 or item.get('source_binding') != source:
                raise ValueError('Invalid completed run summary')
            count += 1
    return count


def worker(config_path, gpu):
    from .campaign import load_config, source_binding
    if gpu not in ALLOWED_GPUS:
        raise ValueError('Unapproved physical GPU')
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    run_order(config)  # The inherited fixed budget validator.
    output = resolve_path(config['output_dir'])
    checkpoint_root = resolve_path(config['checkpoint_root'])
    source = source_binding(config)
    if read_json(output / 'execution_manifest.json')['source_binding'] != source:
        raise ValueError('Execution source changed')
    interval = min(60., max(2., float(config.get('scheduling', {}).get('poll_seconds', 20))))
    pause = output / 'scheduler' / f'PAUSE_GPU{gpu}'
    status_path = output / 'scheduler' / f'worker_gpu{gpu}.json'
    child = None

    def status(state, **extra):
        value = {'status': state, 'updated_at': utc_now(), 'pid': os.getpid(),
                 'physical_gpu': gpu, 'mainline_reserved_gpu': 1, **extra}
        write_json(status_path, value)
        return value

    with singleton(output / 'scheduler' / f'worker_gpu{gpu}.lock'):
        try:
            while True:
                complete = complete_seeds(output, source)
                if len(complete) == 3:
                    return status('COMPLETE', completed_runs=6)
                reason = request_reason(output, gpu)
                if reason == 'stop_queue':
                    return status('PAUSED_BY_REQUEST')
                if reason:
                    status('WAITING_MAINLINE', reason=reason)
                    time.sleep(interval)
                    continue
                cards = gpu_snapshot()
                selected = select_idle_gpu(cards, {'gpu_indices': [gpu]})
                if selected is None:
                    status('WAITING_IDLE_GPU')
                    time.sleep(interval)
                    continue
                selected = select_idle_gpu(gpu_snapshot(), {'gpu_indices': [gpu]})
                if selected is None or request_reason(output, gpu):
                    continue
                seed = claim_pair(output, gpu, source, complete)
                if seed is None:
                    status('WAITING_OTHER_ASSIGNED_PAIRS')
                    time.sleep(interval)
                    continue
                for variant in ('AAA', 'QFL_DFL'):
                    progress = run_progress(output, variant, seed, checkpoint_root, source)
                    if progress['status'] != 'COMPLETE':
                        break
                else:
                    continue
                # Ownership can change while reading the committed checkpoint.
                selected = select_idle_gpu(gpu_snapshot(), {'gpu_indices': [gpu]})
                if selected is None or request_reason(output, gpu):
                    continue
                if pause.exists():
                    pause.unlink()
                env = child_environment(selected, pause)
                command = [sys.executable, '-m', 'tools.sonar_fcos_quality_v2.train', 'run',
                           '--config', str(config_path), '--variant', variant, '--seed', str(seed),
                           '--epoch-budget', '1', '--pause-file', str(pause)]
                log_path = output / 'scheduler' / f'{variant}_seed{seed}_gpu{gpu}.log'
                with log_path.open('a', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    status('RUNNING', variant=variant, seed=seed, child_pid=child.pid,
                           completed_epochs=progress['completed_epochs'])
                    pause_reason = None
                    while child.poll() is None:
                        time.sleep(interval)
                        if child.poll() is not None:
                            break
                        try:
                            foreign = foreign_pids(gpu_snapshot(), gpu, process_family(child.pid))
                            pause_reason = pause_reason or request_reason(output, gpu, foreign)
                        except Exception as exc:
                            pause_reason = pause_reason or 'resource_observation_failed:' + type(exc).__name__
                        if pause_reason:
                            pause.write_text(pause_reason + '\n', encoding='utf-8')
                            status('YIELDING_GPU', variant=variant, seed=seed, child_pid=child.pid, reason=pause_reason)
                    rc = child.wait()
                after = run_progress(output, variant, seed, checkpoint_root, source)
                if rc == 75:
                    if pause_reason is None:
                        raise RuntimeError('Unexpected cooperative pause without scheduler request')
                    status('YIELDED_GPU', variant=variant, seed=seed, reason=pause_reason,
                           completed_epochs=after['completed_epochs'])
                    continue
                if rc:
                    raise RuntimeError(f'{variant}_seed{seed} exited {rc}; no automatic retry')
                delta = after['completed_epochs'] - progress['completed_epochs']
                if delta != 1 and not (delta == 0 and progress['completed_epochs'] == 30 and after['status'] == 'COMPLETE'):
                    raise RuntimeError('One-epoch worker exceeded or missed committed progress')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                pause.write_text('worker_exit\n', encoding='utf-8')
            status('ERROR', error_type=type(exc).__name__, error=str(exc))
            raise


def supervise(config_path):
    from .campaign import load_config, source_binding
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    run_order(config)
    output = resolve_path(config['output_dir'])
    source = source_binding(config)
    migration = read_json(output / 'migration.json')
    if migration.get('status') != 'MIGRATION_PASS' or migration['new_source_binding'] != source:
        raise ValueError('Verified parent migration is required')
    affinity = {str(r['seed']): r['physical_gpu'] for r in migration['migrated_runs']}
    workers = []
    with singleton(output / 'scheduler' / 'supervisor.lock'):
        initialize_claims(output, source, affinity)
        try:
            for gpu in ALLOWED_GPUS:
                command = [sys.executable, '-m', 'tools.sonar_fcos_quality_v2.scheduler', 'worker',
                           '--config', str(config_path), '--gpu', str(gpu)]
                with (output / 'scheduler' / f'worker_gpu{gpu}_supervisor.log').open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                workers.append((gpu, child))
            value = {'status': 'RUNNING', 'pid': os.getpid(), 'updated_at': utc_now(),
                     'workers': [{'physical_gpu': gpu, 'pid': p.pid} for gpu, p in workers],
                     'finite_runs': 6, 'reserved_mainline_gpu': 1,
                     'migration_sha256': sha256_file(output / 'migration.json')}
            write_json(output / 'queue_status.json', value)
            failed = []
            last_summarized_count = -1
            while any(p.poll() is None for _, p in workers):
                for gpu, child in workers:
                    if child.poll() not in (None, 0) and gpu not in failed:
                        failed.append(gpu)
                        (output / 'STOP_QUEUE').write_text('Worker failure; preserve all committed state for review.\n')
                count = completed_run_count(output, source)
                if count != last_summarized_count and count < 6:
                    summarize_locked(config)
                    last_summarized_count = count
                time.sleep(5)
            if failed or any(p.returncode != 0 for _, p in workers):
                raise RuntimeError('One or more GPU workers failed; no automatic retry')
            complete = complete_seeds(output, source)
            value.update(status='COMPLETE' if len(complete) == 3 else 'PAUSED_BY_REQUEST',
                         updated_at=utc_now(), completed_runs=completed_run_count(output, source))
            if len(complete) == 3:
                write_json(output / 'queue_status.json', {**value, 'status': 'FINAL_ANALYSIS', 'completed_runs': 6})
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                summarize_locked(config)
            write_json(output / 'queue_status.json', value)
            return value
        except BaseException as exc:
            (output / 'STOP_QUEUE').write_text('Supervisor exit; cooperatively release owned GPU work.\n')
            write_json(output / 'queue_status.json', {'status': 'ERROR', 'pid': os.getpid(),
                'updated_at': utc_now(), 'error_type': type(exc).__name__, 'error': str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise', 'worker'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, choices=ALLOWED_GPUS)
    args = parser.parse_args()
    value = worker(args.config, args.gpu) if args.mode == 'worker' else supervise(args.config)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
