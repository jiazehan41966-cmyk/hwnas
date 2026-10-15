"""Resource-only Phase2E scheduler; frozen training and promotion code is reused.

Use after the old supervisor and its children have drained. Does not modify the
execution binding, checkpoint format, experimental protocol, or bound sources.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import threading
import time


def run_queue(selected, gpus, workers_per_gpu, execute, started, finished):
    """Dispatch each unique ID once, with a hard per-GPU concurrency limit."""
    if workers_per_gpu not in (1, 2):
        raise ValueError('Only one or two workers per GPU authorized')
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPU IDs must be nonempty and unique')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate architecture IDs')
    queue = Queue()
    for aid in selected:
        queue.put(aid)
    failed = threading.Event()
    mutex = threading.Lock()

    def worker(gpu, slot):
        key = f'{gpu}:{slot}'
        while not failed.is_set():
            try:
                aid = queue.get_nowait()
            except Empty:
                return
            try:
                with mutex:
                    if failed.is_set():
                        return
                    started(key, gpu, aid)
                row = execute(aid, gpu)
                with mutex:
                    finished(key, gpu, aid, row)
            except Exception:
                failed.set()
                raise

    with ThreadPoolExecutor(max_workers=len(gpus) * workers_per_gpu) as pool:
        jobs = [pool.submit(worker, gpu, slot)
                for gpu in gpus for slot in range(workers_per_gpu)]
        for job in jobs:
            job.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--workers-per-gpu', type=int, choices=[1, 2], default=2)
    p.add_argument('--resume', action='store_true', required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or set(args.gpus) - {0, 1}:
        p.error('Only unique GPU IDs 0 and 1 are authorized')

    from phase2e_runner import (ROOT, REPORT, load, binding, arch_map,
                               existing_map, record, promote, rejected_audit, now)
    from phase2e_pipeline import csv_write, manifest
    from formal_protocol import dump, sha
    cfg, _ = load()  # Unmodified source hashes and independent admission gate.
    current_binding = binding()
    print('Frozen protocol/source binding PASS', flush=True)
    if args.preflight:
        return

    # record() repeatedly enumerates the same immutable 144 architectures.
    # Cache that pure lookup only in this supervisor process. Subprocesses still
    # import the original runner; binding/checkpoint checks remain uncached.
    import phase2e_runner as runner
    runner.arch_map = cache(runner.arch_map)

    import fcntl
    lock = (REPORT / 'resource_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse coexistence with another supervisor or any in-flight writer.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if (entry / 'cwd').resolve() != ROOT:
                continue
            cmd = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if any(name in cmd for name in ('phase2e_pipeline.py',
                                           'phase2e_resource_queue.py',
                                           'phase2e_runner.py train')):
                raise RuntimeError(f'Concurrent Phase2E writer {entry.name}: {cmd}')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue

    logs = ROOT / 'logs/phase2e_multifidelity'
    logs.mkdir(parents=True, exist_ok=True)
    existing = existing_map()
    scheduler = dict(path=str(Path(__file__).relative_to(ROOT)),
                     sha256=sha(Path(__file__)), gpus=args.gpus,
                     workers_per_gpu=args.workers_per_gpu,
                     max_concurrent_jobs=len(args.gpus) * args.workers_per_gpu,
                     started_at=now(), pid=os.getpid(), binding=current_binding)
    dump(REPORT / f'resource_queue_launch_{os.getpid()}.json', scheduler)
    events = REPORT / 'resource_queue_events.jsonl'

    def event(kind, **kw):
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(event=kind, timestamp=now(), **kw)) + '\n')

    def status(phase, **kw):
        dump(REPORT / 'campaign_status.json', dict(
            phase=phase, updated_at=now(), pid=os.getpid(), gpus=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            binding=current_binding, scheduler_sha256=scheduler['sha256'],
            test_evaluated=False, **kw))

    def stage(name, selected, budget):
        import csv
        if len(selected) != len(set(selected)):
            raise ValueError('Duplicate stage architecture IDs')
        path = REPORT / f'{name}_results.csv'
        if path.exists():
            with path.open(newline='') as f:
                saved = list(csv.DictReader(f))
            if len(saved) != len(selected) or {r['architecture_id'] for r in saved} != set(selected):
                raise ValueError('Completed stage ID mismatch')
            rows = [record(r['architecture_id'], budget) for r in saved]
            if any(abs(float(old['mAP50_95']) - row['mAP50_95']) > 1e-14
                   for old, row in zip(saved, rows)):
                raise ValueError('Completed stage score drift')
            return rows

        rows, pending, active, ticks = [], [], {}, {}
        for aid in sorted(selected):
            checkpoint = ROOT / 'results/phase2e_independent' / aid / f'budget_{budget}.pt'
            if aid in existing or checkpoint.exists():
                rows.append(record(aid, budget))
            else:
                pending.append(aid)

        def save_status():
            status(name.upper(), budget=budget, completed=len(rows), total=len(selected),
                   active_architectures=dict(active))

        def execute(aid, gpu):
            if shutil.disk_usage(ROOT).free < 3 * 1024**3:
                raise RuntimeError('Disk below 3 GiB reserve')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
            with (logs / f'{aid}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'tools/sonar_spec/phase2e_runner.py',
                                'train', '--architecture-id', aid, '--budget', str(budget),
                                '--device', 'cuda'], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            return record(aid, budget)

        def started(key, gpu, aid):
            active[key] = aid
            ticks[key] = time.monotonic()
            event('start', slot=key, gpu=gpu, architecture_id=aid, budget=budget)
            save_status()

        def finished(key, gpu, aid, row):
            rows.append(row)
            active.pop(key)
            event('complete', slot=key, gpu=gpu, architecture_id=aid, budget=budget,
                  wall_seconds=time.monotonic() - ticks.pop(key))
            csv_write(REPORT / f'{name}_partial.csv', sorted(rows, key=lambda r: r['architecture_id']))
            save_status()

        save_status()
        run_queue(pending, args.gpus, args.workers_per_gpu, execute, started, finished)
        rows.sort(key=lambda r: r['architecture_id'])
        csv_write(path, rows)
        return rows

    try:
        first = stage('stage1', sorted(arch_map()), 40)
        second_ids = promote(first, *cfg['promotion_fractions'][0])
        audit1 = rejected_audit(first, second_ids, 1, existing)
        manifest(REPORT / 'promotion_stage1.json', dict(binding=binding(),
            input_sha256=sha(REPORT / 'stage1_results.csv'),
            promoted=[r['architecture_id'] for r in second_ids], audit=audit1,
            metric=cfg['promotion_metric']))
        second = stage('stage2', [r['architecture_id'] for r in second_ids], 60)
        final_ids = promote(second, *cfg['promotion_fractions'][1])
        audit2 = rejected_audit(second, final_ids, 2, existing)
        manifest(REPORT / 'promotion_stage2.json', dict(binding=binding(),
            input_sha256=sha(REPORT / 'stage2_results.csv'),
            promoted=[r['architecture_id'] for r in final_ids], audit=audit2,
            metric=cfg['promotion_metric']))
        stage('final', [r['architecture_id'] for r in final_ids], 80)
        if audit1 + audit2:
            stage('rejection_audit', audit1 + audit2, 80)
        status('ANALYSIS')
        subprocess.run([sys.executable, 'tools/sonar_spec/phase2e_results.py'], cwd=ROOT, check=True)
        outcome = json.loads((REPORT / 'promotion_reliability.json').read_text())
        status('COMPLETE_REQUIRES_REVIEW' if not outcome['audit_top5_violation']
               else 'STOP_PROMOTION_RELIABILITY_FAILED',
               final_selection_executed=False, sonarop_implemented=False)
    except Exception as error:
        status('FAILED_RESUMABLE', error=str(error))
        raise


if __name__ == '__main__':
    main()
