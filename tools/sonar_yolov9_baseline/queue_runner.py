"""One-GPU, stop-on-failure queue for the three matched seed42 controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_yolov9_baseline.protocol import dump, now, sha

GROUPS = ('L1_scratch', 'YOLOv9_T_scratch', 'YOLOv9_T_COCO')


def execute_queue(result, device, status):
    manifest_sha = status['execution_manifest_sha256']
    for group in GROUPS:
        smoke = json.loads((result/'smoke'/group/'seed_42/smoke_report.json').read_text())
        if (smoke['status'] != 'PASS' or smoke['formal_eligible'] is not False
                or smoke['binding']['execution_manifest_sha256'] != manifest_sha):
            raise ValueError('Every queued group requires its SHA-bound successful smoke')
        folder = result/'formal'/group/'seed_42'
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT/'tools/sonar_yolov9_baseline/train.py'), 'train',
                   '--result-root', str(result), '--group', group, '--seed', '42', '--device', device]
        with (folder/'stdout.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status.update(active=group, child_pid=process.pid, command=command, updated_at=now())
            dump(result/'queue_status.json', status)
            returncode = process.wait()
        if returncode:
            status.update(status='FAILED_NO_AUTORETRY', failed_group=group,
                          returncode=returncode, updated_at=now())
            dump(result/'queue_status.json', status)
            return
        row = json.loads((folder/'result.json').read_text())
        if not row['formal_eligible'] or row['epochs'] != 80:
            raise ValueError('Child exited without complete formal result')
        status['completed'].append(group)
        status['pending'].remove(group)
        dump(result/'queue_status.json', status)
    status.update(status='COMPLETE', active=None, child_pid=None, updated_at=now())
    dump(result/'queue_status.json', status)
    records = [json.loads((result/'formal'/group/'seed_42/result.json').read_text()) for group in GROUPS]
    dump(result/'comparison.json', dict(status='COMPLETE', groups=records,
         interpretation='single_seed_budget_matched_partial_COCO_transfer', test_evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args()
    result = args.result_root.resolve()
    if (result / 'queue_status.json').exists():
        raise FileExistsError('Queue already invoked; no silent retry')
    status = dict(status='RUNNING', pid=os.getpid(), order=list(GROUPS), seed=42,
                  epochs_each=80, completed=[], pending=list(GROUPS),
                  execution_manifest_sha256=sha(result/'execution_manifest.json'),
                  started_at=now(), test_evaluated=False)
    dump(result / 'queue_status.json', status)
    try:
        execute_queue(result, args.device, status)
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY', error=str(error),
                      error_type=type(error).__name__, updated_at=now())
        dump(result/'queue_status.json', status)
        raise


if __name__ == '__main__':
    main()
