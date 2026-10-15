"""Bounded read-only monitor; never retries or restarts training."""
import json
from datetime import datetime, timezone
import time
import subprocess
from tools.sonar_joint192_validation.remote import status


if __name__ == '__main__':
    started = time.monotonic()
    failures = 0
    # Let a persistently refusing endpoint cool down; remote training is untouched.
    for _ in range(7):
        print('READ_ONLY_MONITOR_COOLDOWN: original remote workers preserved', flush=True)
        time.sleep(45)
    while time.monotonic() - started < 8 * 3600:
        try:
            value = status()
            failures = 0
            workers = {k:v for k,v in value.items() if k.endswith('queue_status.json')}
            runs = {k:dict(status=v.get('status'), epoch=v.get('epoch',v.get('epochs')),
                           latest_mAP50_95=v.get('latest_valid',{}).get('mAP50_95'))
                    for k,v in value.items() if k.endswith('/status.json')}
            payload = dict(time=datetime.now(timezone.utc).isoformat(), runs=runs,
                           workers={k:dict(status=v['status'], active=v.get('active'), completed=len(v['completed']))
                                    for k,v in workers.items()})
            print(json.dumps(payload), flush=True)
            if any(v['status'].startswith('FAILED') for v in workers.values()):
                raise SystemExit('WORKER_FAILED: preserved evidence, no automatic restart')
            if len(workers) == 2 and all(v['status'] == 'COMPLETE' for v in workers.values()):
                print('ALL_12_RUNS_COMPLETE', flush=True)
                break
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                  monitoring_error=str(error), action='read_only_retry_next_poll_no_training_restart')), flush=True)
            for _ in range(min(12, 2 ** min(failures, 4))):
                time.sleep(45)
        time.sleep(45)
    else:
        raise SystemExit('MONITOR_TIME_LIMIT: inspect live status before continuing')
