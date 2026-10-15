"""Finite serial invocation of the already reviewed per-case board driver.

Stops at unresolved nonzero exits and never retries a measurement.
Optional recovery downloads an existing PASS snapshot after a client timeout.
Exclusions require an existing successful selected run and verified import.
No direct board access; the existing driver owns all admission and DMA gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def terminal_fetch_recovery_candidate(attempt, row):
    """Recognize only an empty fetch stopped after a recorded timing PASS."""
    attempt = Path(attempt).resolve()
    job = row['job_prefix'] + '_timing'
    execution = attempt / 'board_execution' / row['job_prefix']
    events_path = execution / 'events.jsonl'
    if not events_path.is_file():
        return False
    try:
        events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
    except (ValueError, OSError):
        return False
    if not events or any(event.get('case_id') != row['case_id'] for event in events):
        return False
    stop = events[-1]
    states = [event.get('state') for event in events if event.get('stage') == 'timing_state']
    if (stop.get('stage') != 'case_stopped_preserving_evidence' or
            stop.get('error_type') != 'RuntimeError' or
            stop.get('reason') != 'bridge response timed out; inspect status before any launch retry' or
            not states or states[-1] != 'PASS' or any(state not in ('RUNNING', 'PASS') for state in states)):
        return False
    original = attempt / 'board_snapshots' / job
    recovered = attempt / 'board_snapshots' / (job + '_recovered')
    pointer = attempt / 'cases' / row['case_id'] / 'board/selected_run.json'
    return (original.is_dir() and not original.is_symlink() and not any(original.iterdir()) and
            not recovered.exists() and not pointer.exists())


def recover_terminal_fetch_once(attempt, row, helper_sources):
    """Use the existing fetch/import/incident guards, without relaunching hardware."""
    attempt = Path(attempt).resolve()
    if not terminal_fetch_recovery_candidate(attempt, row):
        raise ValueError('not an eligible terminal snapshot timeout')
    job = row['job_prefix'] + '_timing'
    snapshot = attempt / 'board_snapshots' / (job + '_recovered')
    package = row['argv'][row['argv'].index('--case-dir') + 1]
    base = [sys.executable, '-X', 'utf8', '-m', 'tools.hardware_template_coverage_v1.']
    commands = [
        base[:-1] + [base[-1] + 'fetch_existing_snapshot', '--transport-dir',
            str(attempt / 'board_transport' / job), '--job-id', job, '--output', str(snapshot)],
        base[:-1] + [base[-1] + 'import_recovered_snapshot', '--snapshot', str(snapshot),
            '--package', package, '--attempt', str(attempt), '--run-name', job],
        base[:-1] + [base[-1] + 'archive_incomplete_fetch', '--attempt', str(attempt),
            '--job-id', job, '--recovered', str(snapshot)],
    ]
    for command in commands:
        for name, expected in helper_sources.items():
            if sha(ROOT / name) != expected:
                raise ValueError('recovery helper changed')
        if subprocess.run(command, cwd=ROOT).returncode:
            raise RuntimeError('single existing-snapshot recovery failed; preserve evidence and stop')
    audit_path = attempt / 'cases' / row['case_id'] / 'board' / job / 'verified_snapshot_import.json'
    checked = load(audit_path)
    pointer = load(audit_path.parent.parent / 'selected_run.json')
    if (checked.get('case_id') != row['case_id'] or checked.get('status') != 'PASS_IMPORT_AUDIT' or
            checked.get('numerical_status') != 'PASS' or pointer.get('run_directory') != job or
            pointer.get('evidence_kind') != 'BOARD_MEASUREMENT'):
        raise ValueError('recovered case is not a verified successful selected run')
    source = ROOT / 'tools/hardware_template_coverage_v1/fetch_existing_snapshot.py'
    with (snapshot / 'fetcher_source_executed.py').open('xb') as stream:
        stream.write(source.read_bytes())
    result = dict(status='PASS_SINGLE_TERMINAL_SNAPSHOT_RECOVERY', measurement_restarted=False,
        commands=commands, helper_sources=helper_sources, selected_audit_sha256=sha(audit_path),
        hardware_measurement_eligible=checked.get('hardware_measurement_eligible'),
        protocol_conditions_met=checked.get('protocol_conditions_met'))
    save_new(snapshot / 'queue_recovery.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--ownership-config', type=Path, required=True)
    parser.add_argument('--exclude-completed', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-tag', required=True)
    parser.add_argument('--recover-terminal-snapshot', action='store_true')
    args = parser.parse_args()
    from .board_transfer import valid_job
    valid_job(args.job_tag)
    index, ownership = load(args.index), load(args.ownership_config)
    if index['status'] != 'PASS_13_OFFLINE_REVIEWED_PROBE_PACKAGES' or len(index['cases']) != 13:
        raise ValueError('reviewed 13-case package index required')
    expected_index, expected_ownership = sha(args.index), sha(args.ownership_config)
    helper_sources = {('tools/hardware_template_coverage_v1/' + name):
                      sha(ROOT / 'tools/hardware_template_coverage_v1' / name)
                      for name in ('fetch_existing_snapshot.py', 'import_recovered_snapshot.py',
                                   'archive_incomplete_fetch.py')}
    attempt = ROOT / 'results/hardware_template_coverage_20261013/attempt01'
    excluded = set(args.exclude_completed)
    if not excluded <= {row['case_id'] for row in index['cases']}:
        raise ValueError('unknown excluded case')
    exclusions = []
    for case_id in sorted(excluded):
        parent = attempt / 'cases' / case_id / 'board'
        pointer = parent / 'selected_run.json'
        pointer_record = load(pointer)
        name = pointer_record['run_directory']
        if Path(name).name != name or pointer_record.get('evidence_kind') != 'BOARD_MEASUREMENT':
            raise ValueError('excluded case pointer is not an exact board run')
        audit = parent / name / 'verified_snapshot_import.json'
        result = load(audit)
        if (result.get('case_id') != case_id or result.get('numerical_status') != 'PASS' or
                result.get('status') != 'PASS_IMPORT_AUDIT'):
            raise ValueError('excluded case lacks verified complete timing: ' + case_id)
        exclusions.append(dict(case_id=case_id, pointer_sha256=sha(pointer), audit_sha256=sha(audit),
            protocol_conditions_met=result.get('protocol_conditions_met'),
            hardware_measurement_eligible=result.get('hardware_measurement_eligible'),
            condition_deviations=result.get('condition_deviations', [])))
    rows = [row for row in index['cases'] if row['case_id'] not in excluded]
    commands = []
    for row in rows:
        prefix = valid_job('t6_r2_' + row['case_id'][3:-6] + '_' + args.job_tag)
        valid_job(prefix + '_timing')
        commands.append(dict(case_id=row['case_id'], job_prefix=prefix, argv=[sys.executable, '-X', 'utf8', '-m',
            'tools.hardware_template_coverage_v1.run_board_case', '--attempt', 'attempt01',
            '--case-dir', row['case_dir'], '--structure', row['structure_path'], '--review', row['review_path'],
            '--job-prefix', prefix, '--board-id', 'machine-id:' + ownership['expected_machine_id'],
            '--ownership-config', str(args.ownership_config.resolve()), '--wait-seconds', '3600']))
    args.output.mkdir(parents=True, exist_ok=False)
    save_new(args.output / 'inputs.json', dict(schema='t6_finite_serial_campaign_v1',
        index_sha256=expected_index, ownership_config_sha256=expected_ownership, exclusions=exclusions,
        driver_sources=index['sources'], wrapper_sha256=sha(__file__), commands=commands,
        automatic_retry=False, stop_on_nonzero=True,
        single_terminal_snapshot_recovery=args.recover_terminal_snapshot,
        recovery_helper_sources=helper_sources if args.recover_terminal_snapshot else {}))
    completed = []
    for row in commands:
        if sha(args.index) != expected_index or sha(args.ownership_config) != expected_ownership:
            raise ValueError('campaign input changed')
        for name, expected in index['sources'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('reviewed runtime changed before next case')
        print(json.dumps(dict(stage='serial_case_begin', case_id=row['case_id'], completed=len(completed),
                              utc=datetime.now(timezone.utc).isoformat())), flush=True)
        code = subprocess.run(row['argv'], cwd=ROOT).returncode
        driver_code, recovery = code, None
        if code and args.recover_terminal_snapshot and terminal_fetch_recovery_candidate(attempt, row):
            try:
                recovery = recover_terminal_fetch_once(attempt, row, helper_sources)
                code = 0
            except (OSError, ValueError, RuntimeError) as error:
                recovery = dict(status='STOP_FAILED_SNAPSHOT_RECOVERY', error_type=type(error).__name__,
                                reason=str(error), measurement_restarted=False)
        result = dict(case_id=row['case_id'], job_prefix=row['job_prefix'], returncode=code,
                      driver_returncode=driver_code, terminal_snapshot_recovery=recovery,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save_new(args.output / (row['case_id'] + '.json'), result)
        print(json.dumps(dict(stage='serial_case_end', **result)), flush=True)
        if code:
            save_new(args.output / 'stopped.json', dict(status='STOPPED_NONZERO_NO_RETRY', failed=result, completed=completed))
            return code
        completed.append(result)
    save_new(args.output / 'completed.json', dict(status='COMPLETE', completed=completed, exclusions=exclusions))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
