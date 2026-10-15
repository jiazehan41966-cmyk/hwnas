"""Launch only after all three real GPU smokes and independent review pass."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_yolov9_baseline import protocol as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=ROOT/'results/sonar_yolov9_baseline_20261005')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    result = args.result_root.resolve()
    if (result/'launch.json').exists() or (result/'queue_status.json').exists():
        raise FileExistsError('Campaign queue already launched')
    from tools.sonar_yolov9_baseline.train import verify_manifest
    verify_manifest(result)
    manifest_sha = p.sha(result/'execution_manifest.json')
    review = json.loads((result/'independent_review.json').read_text())
    if review['status'] != 'PASS' or review['execution_manifest_sha256'] != manifest_sha:
        raise ValueError('Independent review must pass against this exact source freeze')
    smokes = []
    for group in ('L1_scratch','YOLOv9_T_scratch','YOLOv9_T_COCO'):
        report = json.loads((result/'smoke'/group/'seed_42/smoke_report.json').read_text())
        if (report['status'] != 'PASS' or report['formal_eligible'] is not False
                or report['environment']['device'] != 'cuda'
                or report['binding']['execution_manifest_sha256'] != manifest_sha):
            raise ValueError('Three actual CUDA smokes bound to this freeze are required')
        smokes.append(report)
    if len({report['training']['image_order_sha256'] for report in smokes}) != 1:
        raise ValueError('Paired smoke data order differs')
    rng = [json.loads((result/'smoke'/group/'seed_42/training_rng_identity.json').read_text())
           for group in ('L1_scratch','YOLOv9_T_scratch','YOLOv9_T_COCO')]
    if any(value != rng[0] for value in rng[1:]):
        raise ValueError('Training RNG identity differs across groups')
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
    gpu = next([int(value.strip()) for value in line.split(',')] for line in gpu_text.splitlines()
               if int(line.split(',')[0]) == args.gpu)
    required = max(report['peak_cuda_reserved_bytes'] for report in smokes)/1048576
    # WDDM's utilization includes the live desktop. The existing environment
    # contract defines an unused GPU by memory.used < 500 MiB; real batch4
    # witnesses additionally establish the free-memory requirement.
    busy = gpu[1] >= 500 if os.name == 'nt' else gpu[3] > 10
    if gpu[2]-gpu[1] < required or busy:
        raise RuntimeError('GPU availability changed since real batch4 smoke')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable, str(ROOT/'tools/sonar_yolov9_baseline/queue_runner.py'),
               '--result-root', str(result), '--device', 'cuda']
    with (result/'queue_stdout.log').open('a', encoding='utf-8') as log:
        kwargs = dict(cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        process = subprocess.Popen(command, **kwargs)
    p.dump(result/'launch.json', dict(status='LAUNCHED', pid=process.pid, gpu=args.gpu,
           order=['L1_scratch','YOLOv9_T_scratch','YOLOv9_T_COCO'], epochs_each=80, seed=42,
           execution_manifest_sha256=manifest_sha, command=command,
           gpu_prelaunch=dict(used_mib=gpu[1], total_mib=gpu[2], utilization_percent=gpu[3],
                              idle_policy='WDDM_memory_under500MiB' if os.name == 'nt' else 'utilization_at_most10'),
           launched_at=p.now(), test_evaluated=False))
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError('Queue process exited immediately; inspect retained log')
    print(json.dumps(dict(status='LAUNCHED', pid=process.pid, gpu=args.gpu,
                         result_root=str(result), execution_manifest_sha256=manifest_sha)))


if __name__ == '__main__':
    main()
