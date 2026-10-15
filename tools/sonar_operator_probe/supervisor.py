"""Finite resource-aware preflight, evidence check and training dispatcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    from tools.sonar_operator_probe.train import wait_for_gpu
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'queue', 'campaign'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    if cfg['queue']['allowed_gpus'] != [1]:
        raise ValueError('This dispatcher is limited to GPU1')
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, '-B', 'tools/sonar_operator_probe/train.py']
    common = ['--config', args.config, '--output', args.output, '--device', 'cuda']
    waiting_seconds = 0.
    name = 'campaign_supervisor_status.json' if args.phase == 'campaign' else args.phase + '_dispatcher_status.json'
    status_path = output / name
    status = dict(status='RUNNING', phase=args.phase, pid=os.getpid(), started_at=stamp(),
                  completed_phases=[], formal_eligible=False, test_evaluated=False)
    dump(status_path, status)

    def wait(context):
        nonlocal waiting_seconds
        remaining = max(0., float(cfg['queue']['resource_wait_maximum_hours']) * 3600 - waiting_seconds)

        def record(receipt):
            value = dict(receipt, context=context, resource_wait_seconds=waiting_seconds + receipt['waiting_seconds'])
            dump(output / 'resource_wait_status.json', value)
            status.update(status='WAITING_RESOURCE', waiting_for=context,
                          resource_wait_seconds=value['resource_wait_seconds'], updated_at=stamp())
            dump(status_path, status)

        receipt = wait_for_gpu(cfg, 'cuda', on_wait=record, maximum_wait_seconds=remaining)
        waiting_seconds += receipt['waiting_seconds']
        status.update(status='RUNNING', resource_wait_seconds=waiting_seconds, updated_at=stamp())
        dump(status_path, status)
        dump(output / 'resource_wait_status.json', dict(receipt, status='RESOURCE_AVAILABLE', context=context))

    def run_phase(phase, commands):
        path = output / (phase + '_supervisor_status.json')
        value = dict(status='RUNNING', phase=phase, pid=os.getpid(), started_at=stamp(),
                     commands=commands, completed=[], formal_eligible=False, test_evaluated=False)
        dump(path, value)
        for command in commands:
            if 'tools/sonar_operator_probe/train.py' in command and any(
                    mode in command for mode in ('prepare', 'smoke', 'queue')):
                mode = next(mode for mode in ('prepare', 'smoke', 'queue') if mode in command)
                wait(phase + ':' + mode)
            value.update(command=command, updated_at=stamp())
            dump(path, value)
            print('EXECUTE ' + json.dumps(command), flush=True)
            try:
                timeout = 1800 if phase != 'queue' else (48 + 6) * 3600 + 600
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), timeout=timeout)
                if result.returncode:
                    raise RuntimeError(f'Owned command exited {result.returncode}')
            except Exception as error:
                value.update(status='FAILED_NO_AUTORETRY', error_type=type(error).__name__,
                             error=str(error), updated_at=stamp())
                dump(path, value)
                raise
            value['completed'].append(command)
            dump(path, value)
        value.update(status='COMPLETE', completed_at=stamp())
        dump(path, value)
        status['completed_phases'].append(phase)
        dump(status_path, status)

    try:
        if args.phase in ('preflight', 'campaign'):
            run_phase('preflight', [prefix + ['prepare'] + common, prefix + ['verify'] + common,
                [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 'tests/test_sonar_operator_probe_model.py', 'tests/test_sonar_operator_probe_train.py'],
                prefix + ['smoke'] + common])
        if args.phase == 'campaign':
            run_phase('runtime_readiness', [[sys.executable, '-B', 'tools/sonar_operator_probe/validate_ready.py',
                                           '--config', args.config, '--output', args.output]])
        if args.phase in ('queue', 'campaign'):
            review = json.loads((output / 'readiness_review.json').read_text(encoding='utf-8'))
            for filename, key in (('execution_manifest.json','execution_manifest_sha256'),
                                  ('smoke_gate.json','smoke_gate_sha256')):
                if hashlib.sha256((output / filename).read_bytes()).hexdigest() != review.get(key):
                    raise ValueError('Readiness binding drift: ' + filename)
            if review.get('status') != 'PASS':
                raise ValueError('Actual runtime evidence review did not pass')
            run_phase('queue', [prefix + ['verify'] + common, prefix + ['queue'] + common])
        status.update(status='COMPLETE', completed_at=stamp())
        dump(status_path, status)
    except Exception as error:
        status.update(status='STOPPED', error_type=type(error).__name__, error=str(error), updated_at=stamp())
        dump(status_path, status)
        raise


if __name__ == '__main__':
    main()
