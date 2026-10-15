"""Run the frozen 12-job Phase2F campaign with bounded per-GPU concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import threading
import time

from formal_protocol import ROOT, dump, sha
from phase2f_runner import ARCHITECTURES, MANIFEST, REPORT, load_manifest, now, result_dir, summarize


def run_queue(jobs, gpus, workers_per_gpu, execute, started, finished):
    if workers_per_gpu not in (1, 2):
        raise ValueError('Only one or two workers per GPU are supported')
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('GPU IDs must be nonempty and unique')
    if len(jobs) != len(set(jobs)):
        raise ValueError('Duplicate architecture/seed jobs')
    queue = Queue()
    for job in jobs:
        queue.put(job)
    failed = threading.Event()
    mutex = threading.Lock()

    def worker(gpu, slot):
        key = f'{gpu}:{slot}'
        while not failed.is_set():
            try:
                aid, seed = queue.get_nowait()
            except Empty:
                return
            try:
                with mutex:
                    started(key, gpu, aid, seed)
                row = execute(aid, seed, gpu)
                with mutex:
                    finished(key, gpu, aid, seed, row)
            except Exception:
                failed.set()
                raise

    with ThreadPoolExecutor(max_workers=len(gpus) * workers_per_gpu) as pool:
        futures = [pool.submit(worker, gpu, slot) for gpu in gpus for slot in range(workers_per_gpu)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--workers-per-gpu', type=int, choices=[1, 2], default=2)
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error('Duplicate GPU IDs')
    manifest, _ = load_manifest(True)
    if args.preflight:
        print('Phase2F preflight PASS', sha(MANIFEST))
        return

    import fcntl
    lock = (REPORT / 'phase2f_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ancestors = set()
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        try:
            parent = int((Path('/proc') / str(parent) / 'stat').read_text().split()[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            break
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid() or int(entry.name) in ancestors:
            continue
        try:
            cmd = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'phase2f_pipeline.py' in cmd:
                raise RuntimeError(f'Concurrent Phase2F supervisor {entry.name}')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass

    jobs = [(aid, seed) for _, aid, _ in ARCHITECTURES for seed in (43, 44)]
    logs = ROOT / 'logs/phase2f_multiseed'
    logs.mkdir(parents=True, exist_ok=True)
    active, completed, ticks = {}, [], {}
    for aid, seed in jobs:
        if (result_dir(aid, seed) / 'status.json').exists():
            completed.append(summarize(aid, seed))
    pending = [(aid, seed) for aid, seed in jobs if not (result_dir(aid, seed) / 'status.json').exists()]

    def status(phase, **extra):
        dump(REPORT / 'campaign_status.json', dict(phase=phase, updated_at=now(), pid=os.getpid(),
            gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, total_jobs=12,
            completed_jobs=len(completed), active_jobs=dict(active), manifest_sha256=sha(MANIFEST),
            test_evaluated=False, **extra))

    def execute(aid, seed, gpu):
        if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
            raise RuntimeError('Disk below 10 GiB reserve')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
        log_path = logs / f'{aid}_seed{seed}.log'
        with log_path.open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', 'tools/sonar_spec/phase2f_runner.py', 'train',
                '--architecture-id', aid, '--seed', str(seed), '--device', 'cuda'], cwd=ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return summarize(aid, seed)

    def started(key, gpu, aid, seed):
        active[key] = dict(architecture_id=aid, seed=seed, gpu=gpu)
        ticks[key] = time.monotonic()
        status('TRAINING')

    def finished(key, gpu, aid, seed, row):
        completed.append(row)
        active.pop(key)
        status('TRAINING', last_completed=dict(architecture_id=aid, seed=seed,
            wall_seconds=time.monotonic() - ticks.pop(key)))

    try:
        status('TRAINING')
        run_queue(pending, args.gpus, args.workers_per_gpu, execute, started, finished)
        status('ANALYSIS')
        subprocess.run([sys.executable, 'tools/sonar_spec/phase2f_analysis.py'], cwd=ROOT, check=True)
        status('COMPLETE_REQUIRES_REVIEW', sonarop_implemented=False,
            network_candidate_selected=False)
    except Exception as error:
        status('FAILED_RESUMABLE', error=str(error))
        raise


if __name__ == '__main__':
    main()
