"""At most three isolated FPGA case builders sharing immutable pilot inputs.

Each case runs HLS then route serially. The driver uses the existing run_stage
implementation; separate child processes permit bounded cancellation without
touching other experiments. This file is deployed under its content hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def valid_root(path):
    root = Path(path).resolve()
    if root.parent != Path('/home/jiajia/experiments') or not re.fullmatch(
            r'hardware_measurement_pilot_20261012_attempt\d+', root.name):
        raise ValueError('unexpected isolated remote root')
    return root


def imported_worker(root):
    sys.path.insert(0, str(root))
    import remote_worker
    transport = json.loads((root / 'transport_manifest.json').read_text())
    for name in ('remote_worker.py', 'source_manifest.json', 'native_summary.json'):
        if digest(root / name) != transport['files'][name]:
            raise ValueError(f'immutable transport input changed: {name}')
    return remote_worker


def matching_pass(old, row, source_sha):
    return (old.get('status') == 'PASS' and old.get('case_sha256') == row['case_sha256']
            and old.get('source_manifest_sha256') == source_sha)


def phase_structure(root, row):
    """Strict, limited RTL scheduling check; does not claim dynamic timing proof."""
    rtls = list((root / 'cases' / row['case_id']).glob('**/syn/verilog/pilot_kernel.v'))
    if len(rtls) != 1:
        raise ValueError('one synthesized pilot_kernel.v required for phase inspection')
    rtl = rtls[0].read_text(errors='replace')
    if not re.search(r'output\s+phase_ap_vld\s*;', rtl):
        raise ValueError('phase valid output is absent')
    branches = re.findall(r'(?:if|else if)\s*\(([^\n]+)\)\s*begin\s*\n\s*phase\s*=\s*8\x27d([0-3]);', rtl)
    states, conditions = {}, {}
    for condition, value in branches:
        if value in states:
            raise ValueError('phase has multiple scheduling conditions')
        found = re.findall(r'ap_CS_fsm_state(\d+)', condition)
        other_names = re.findall(r'\b[A-Za-z_]\w*\b', re.sub(r'ap_CS_fsm_state\d+|\d+\x27b[01]', '', condition))
        if len(found) != 1 or any(name != 'ap_start' or value != '1' for name in other_names):
            raise ValueError('phase marker is tied to another operation or non-boundary condition')
        states[value], conditions[value] = int(found[0]), condition.strip()
    if set(states) != {'0', '1', '2', '3'}:
        raise ValueError('four distinct phase events are required')
    if not states['1'] < states['2'] < states['3'] < states['0']:
        raise ValueError('phase boundary FSM order is invalid')
    if any(states[end] - states[start] < 2 for start, end in [('1', '2'), ('2', '3'), ('3', '0')]):
        raise ValueError('phase interval lacks a separate work state')
    # A protocol-fixed marker has its own state; ap_vld qualifies its otherwise
    # undefined data output. Dynamic correctness remains a separate evidence gate.
    value = {'status': 'STRUCTURAL_PASS', 'case_id': row['case_id'],
             'case_sha256': row['case_sha256'], 'rtl_sha256': digest(rtls[0]),
             'phase_states': states, 'phase_conditions': conditions,
             'scope': 'static dedicated-state and ap_vld check; not cycle simulation or board timing'}
    atomic(root / 'results' / row['case_id'] / 'phase_structural_gate.json', value)
    return value


def case_worker(args):
    root = valid_root(args.root)
    worker = imported_worker(root)
    manifest = worker.verify_inputs(root)
    row = next(row for row in manifest['cases'] if row['case_id'] == args.case_id)
    source_sha = digest(root / 'source_manifest.json')
    # run_stage writes a single shared event journal. An OS lock, rather than a
    # thread lock, protects it because case workers are separate processes.
    import fcntl
    original_event = worker.event
    def locked_event(path, value):
        with (root / 'results' / 'events.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            original_event(path, value)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    worker.event = locked_event
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'owned case worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status = {'case_id': row['case_id'], 'case_sha256': row['case_sha256'],
              'source_manifest_sha256': source_sha, 'status': 'RUNNING',
              'pid': os.getpid(), 'started_utc': utc(), 'stages': []}
    target = root / 'results' / row['case_id'] / 'parallel_case_status.json'
    atomic(target, status)
    try:
        for stage in (('hls', 'route') if args.stage == 'all' else (args.stage,)):
            old = worker.latest(root, row['case_id'], stage)
            if matching_pass(old, row, source_sha):
                status['stages'].append({'stage': stage, 'status': 'PASS', 'reused': True})
                if stage == 'hls':
                    status['phase_structure'] = phase_structure(root, row)
                continue
            if old.get('status') in ('FAILED', 'INTERRUPTED') and not args.retry_failed:
                raise RuntimeError(f'{stage} previously failed; explicit retry required')
            if stage == 'route':
                if not matching_pass(worker.latest(root, row['case_id'], 'hls'), row, source_sha):
                    raise RuntimeError('route requires matching HLS PASS')
                status['phase_structure'] = phase_structure(root, row)
            worker.verify_inputs(root)
            result = worker.run_stage(root, row, stage, 7200, source_sha)
            status['stages'].append({key: result[key] for key in ('stage', 'status', 'wall_seconds')})
            atomic(target, status)
            if result['status'] != 'PASS':
                raise RuntimeError(f'{stage} failed; independent cases may continue')
            if stage == 'hls':
                status['phase_structure'] = phase_structure(root, row)
        status['status'] = 'PASS'
    except BaseException as error:
        status.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                      error=f'{type(error).__name__}: {error}')
    finally:
        status['finished_utc'] = utc()
        atomic(target, status)
    return 0 if status['status'] == 'PASS' else 1


def proc_identity(pid):
    try:
        base = Path('/proc') / str(pid)
        # Field 22 is the Linux process creation tick. Splitting after the last
        # ')' also handles process names containing spaces or parentheses.
        ticks = (base / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return {'pid': pid, 'start_ticks': ticks, 'cwd': str((base / 'cwd').resolve()),
                'argv': [x.decode(errors='replace') for x in (base / 'cmdline').read_bytes().split(b'\0') if x],
                'process_group': os.getpgid(pid)}
    except (OSError, IndexError):
        return None


def same_process(identity):
    current = proc_identity(identity['pid'])
    return current if current and current['start_ticks'] == identity['start_ticks'] else None


def drive(args):
    root = valid_root(args.root)
    worker = imported_worker(root)
    manifest = worker.verify_inputs(root)
    source_sha = digest(root / 'source_manifest.json')
    results = root / 'results'; results.mkdir(exist_ok=True)
    lock_path = root / 'worker.lock'
    lock = {'pid': os.getpid(), 'started_utc': utc(), 'driver': str(Path(__file__).resolve()),
            'driver_sha256': digest(__file__), 'concurrency': args.jobs}
    with lock_path.open('x') as stream:
        json.dump(lock, stream)
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'stage': args.stage,
             'concurrency': args.jobs, 'limit': args.limit, 'started_utc': utc(),
             'driver_sha256': digest(__file__), 'source_manifest_sha256': source_sha,
             'timing_context': 'up to three simultaneous case builds; wall times include shared-server contention and must not be labeled isolated serial estimates',
             'cases': [], 'pending_case_ids': [], 'active_workers': []}
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    active, pending = {}, []
    try:
        final_stage = 'hls' if args.stage == 'hls' else 'route'
        for row in manifest['cases']:
            if args.case_id and row['case_id'] != args.case_id:
                continue
            old = worker.latest(root, row['case_id'], final_stage)
            if matching_pass(old, row, source_sha):
                continue
            if old.get('status') in ('FAILED', 'INTERRUPTED') and not args.retry_failed:
                state['cases'].append({'case_id': row['case_id'], 'status': 'SKIPPED_PRIOR_FAILURE'})
                continue
            if args.stage == 'route' and not matching_pass(worker.latest(root, row['case_id'], 'hls'), row, source_sha):
                continue
            pending.append(row)
        if args.limit:
            pending = pending[:args.limit]
        while pending or active:
            while pending and len(active) < args.jobs and not stop['requested']:
                row = pending.pop(0)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--case-worker',
                           '--root', str(root), '--stage', args.stage, '--case', row['case_id']]
                if args.retry_failed:
                    command.append('--retry-failed')
                folder = results / row['case_id']; folder.mkdir(exist_ok=True)
                logfile = folder / f'parallel_child_{time.time_ns()}.log'
                with logfile.open('x') as log:
                    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                identity = proc_identity(process.pid)
                if not identity:
                    raise RuntimeError('cannot record child process identity')
                active[row['case_id']] = {'process': process, 'identity': identity, 'row': row,
                                          'started_utc': utc(), 'tool_identity': None}
            for name, item in list(active.items()):
                process, row = item['process'], item['row']
                if process.poll() is not None:
                    child = results / name / 'parallel_case_status.json'
                    value = json.loads(child.read_text()) if child.exists() else {
                        'case_id': name, 'status': 'FAILED', 'error': 'case worker exited without status'}
                    value['child_returncode'] = process.returncode
                    state['cases'].append(value); del active[name]
                    continue
                for stage in ('hls', 'route'):
                    latest = worker.latest(root, name, stage)
                    if (latest.get('status') == 'RUNNING' and latest.get('pid')
                            and latest.get('case_sha256') == row['case_sha256']
                            and latest.get('source_manifest_sha256') == source_sha):
                        identity = proc_identity(latest['pid'])
                        if identity and identity['cwd'] == str(root / 'cases' / name) and identity['process_group'] == identity['pid']:
                            item['tool_identity'] = identity
                            item['active_stage'] = stage
                if stop['requested'] and 'termination_sent' not in item:
                    current = same_process(item['identity'])
                    if current and str(Path(__file__).resolve()) in current['argv'] and str(root) in current['argv']:
                        process.send_signal(signal.SIGTERM)
                    item['termination_sent'] = time.monotonic()
                if stop['requested'] and time.monotonic() - item.get('termination_sent', time.monotonic()) > 40:
                    tool = item.get('tool_identity')
                    if tool and same_process(tool) and tool['process_group'] == tool['pid']:
                        os.killpg(tool['pid'], signal.SIGKILL)
                    if same_process(item['identity']):
                        process.kill()
            state['pending_case_ids'] = [row['case_id'] for row in pending]
            state['active_workers'] = [{key: value for key, value in item.items() if key not in ('process', 'row')}
                                       | {'case_id': name} for name, item in active.items()]
            state['stop_requested'] = stop['requested']
            atomic(results / 'parallel_registry.json', state['active_workers'])
            atomic(results / 'queue_status.json', state)
            if stop['requested'] and not active:
                break
            if active:
                time.sleep(1)
        state['status'] = ('INTERRUPTED' if stop['requested'] else 'COMPLETE_WITH_FAILURES'
                           if any(row.get('status') != 'PASS' for row in state['cases']) else 'COMPLETE')
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        # Exceptions still notify only the exact children created by this driver.
        for item in active.values():
            if same_process(item['identity']):
                item['process'].send_signal(signal.SIGTERM)
        for item in active.values():
            try:
                item['process'].wait(timeout=45)
            except subprocess.TimeoutExpired:
                tool = item.get('tool_identity')
                if tool and same_process(tool) and tool['process_group'] == tool['pid']:
                    os.killpg(tool['pid'], signal.SIGKILL)
                if same_process(item['identity']):
                    item['process'].kill(); item['process'].wait(timeout=10)
        raise
    finally:
        state['finished_utc'] = utc()
        atomic(results / 'queue_status.json', state)
        if lock_path.exists() and json.loads(lock_path.read_text()).get('pid') == os.getpid():
            lock_path.unlink()
    print(json.dumps(state), flush=True)
    return 0 if state['status'] == 'COMPLETE' else 1


def client(args):
    if not re.fullmatch(r'attempt\d+', args.attempt):
        raise ValueError('expected attemptNN')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'tools/fpga_v0_1'))
    from remote_bridge import SSH, ENV
    remote = '/home/jiajia/experiments/hardware_measurement_pilot_20261012_' + args.attempt
    source = Path(__file__).read_text(encoding='utf-8')
    sha = hashlib.sha256(source.encode()).hexdigest()
    driver = remote + '/scripts/parallel_build_' + sha[:16] + '.py'
    command = ['python3', '-u', driver, '--driver', '--root', remote, '--jobs', str(args.jobs),
               '--stage', args.stage, '--limit', str(args.limit)]
    if args.case_id:
        command += ['--case', args.case_id]
    if args.retry_failed:
        command.append('--retry-failed')
    code = rf'''
import hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path({remote!r})
if not r.is_dir():raise RuntimeError('initial immutable transport must exist first')
if (r/'worker.lock').exists():raise RuntimeError('another worker holds the root lock')
sys.path.insert(0,str(r));from remote_worker import verify_inputs
verify_inputs(r)
scripts=r/'scripts';scripts.mkdir(exist_ok=True);p=Path({driver!r});data={source!r}.encode()
if hashlib.sha256(data).hexdigest()!={sha!r}:raise RuntimeError('driver source hash mismatch')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!={sha!r}:raise RuntimeError('deployed driver differs')
else:
 with p.open('xb') as f:f.write(data)
 (scripts/(p.name+'.sha256')).write_text({sha!r}+'\n')
name='parallel_driver_'+str(time.time_ns())+'.log'
with (r/'results'/name).open('x') as log:
 proc=subprocess.Popen({command!r},cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'pid':proc.pid,'remote':str(r),'driver':str(p),'driver_sha256':{sha!r},'jobs':{args.jobs},'stage':{args.stage!r},'log':name}}))
'''
    result = subprocess.run(SSH + ['python3', '-'], input=code, env=ENV,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    value = json.loads(result.stdout)
    output = root / 'results/hardware_measurement_pilot_20261012' / args.attempt
    atomic(output / 'parallel_launch.json', value)
    print(json.dumps(value, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('launch',), default='launch')
    parser.add_argument('--attempt', default='attempt03')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--stage', choices=('hls', 'route', 'all'), default='all')
    parser.add_argument('--limit', type=int, default=0, help='0 selects all remaining cases')
    parser.add_argument('--case', dest='case_id')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--root')
    parser.add_argument('--driver', action='store_true')
    parser.add_argument('--case-worker', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 3 or args.limit < 0:
        parser.error('jobs must be 1..3 and limit must be nonnegative')
    if args.case_worker:
        return case_worker(args)
    if args.driver:
        return drive(args)
    return client(args)


if __name__ == '__main__':
    raise SystemExit(main())
