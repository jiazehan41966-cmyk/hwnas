"""One finite, reviewed pilot case: probe, phase binding, three-batch timing.

Requires a pre-existing approval of the exact static phase report. It never
creates approval records, resets hardware, kills a process, retries a failed
run, or removes a board lock. Terminal snapshots are preserved before import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

from tools.hardware_template_coverage_v1 import board_runner as board
from tools.hardware_template_coverage_v1 import board_transfer as transfer
from tools.hardware_template_coverage_v1 import phase_audit as audit
from tools.hardware_template_coverage_v1.remote_worker import verify_inputs
from tools.hardware_template_coverage_v1.timing_validation import validate_timing_snapshot

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'results/hardware_template_coverage_20261013'


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def import_timing(snapshot, package, attempt, run_name):
    """Import faithful PASS or FAIL statistics; preserve environmental deviations."""
    checked = validate_timing_snapshot(snapshot, package)
    case_id = checked['case_id']
    target_case = board._case_identity(attempt / 'cases' / case_id)
    if target_case['case_sha256'] != checked['case_sha256']:
        raise ValueError('report destination case identity differs')
    parent = attempt / 'cases' / case_id / 'board'
    pointer = parent / 'selected_run.json'
    destination = parent / run_name
    if pointer.exists() or destination.exists():
        raise FileExistsError('a board result is already selected; preserve it for review')
    shutil.copytree(Path(snapshot) / 'evidence/output', destination)
    save_new(destination / 'verified_snapshot_import.json', checked)
    save_new(pointer, dict(run_directory=run_name, evidence_kind='BOARD_MEASUREMENT'))
    return checked


def run(args):
    attempt = (PILOT / args.attempt).resolve()
    if not __import__('re').fullmatch(r'attempt[0-9]+', args.attempt):
        raise ValueError('expected an immutable attemptNN')
    package = Path(args.case_dir).resolve()
    if not package.is_relative_to(attempt / 'board_packages'):
        raise ValueError('use an immutable board package within this attempt')
    case = board._case_identity(package)
    manifest = verify_inputs(attempt)
    inventory = board._load(attempt/'inventory.json')
    if (manifest['files'].get('inventory.json') != board._sha(attempt/'inventory.json') or
            case['case_id'] not in inventory.get('representative_case_ids', [])):
        raise ValueError('board execution is restricted to the source-bound representative whitelist')
    expected = next((row for row in manifest['cases'] if row['case_id'] == case['case_id']), None)
    if not expected or expected['case_sha256'] != case['case_sha256']:
        raise ValueError('case is not in the authorized source-bound 65-case attempt')
    board._route_identity(package)
    structure_path, review_path = Path(args.structure).resolve(), Path(args.review).resolve()
    structure, review = board._load(structure_path), board._load(review_path)
    if (structure.get('status') != 'STRUCTURE_PASS_REVIEW_REQUIRED' or
        structure.get('case_sha256') != case['case_sha256'] or
        structure.get('auditor_sha256') != board._sha(audit.__file__) or
        review.get('status') != 'APPROVED_RTL_PHASE_STRUCTURE' or not review.get('reviewer') or
        review.get('case_sha256') != case['case_sha256'] or
        review.get('structure_sha256') != board._sha(structure_path)):
        raise ValueError('exact phase structure has not been reviewed')
    prefix = transfer.valid_job(args.job_prefix)
    ownership_path = Path(args.ownership_config).resolve()
    ownership_config = board._load(ownership_path)
    for suffix in ('_probe', '_timing'):
        transfer.valid_job(prefix + suffix)
    execution = attempt / 'board_execution' / prefix
    execution.mkdir(parents=True, exist_ok=False)
    save_new(execution / 'inputs.json', dict(case_id=case['case_id'], case_sha256=case['case_sha256'],
        package=str(package), source_manifest_sha256=board._sha(attempt/'source_manifest.json'),
        structure_sha256=board._sha(structure_path), review_sha256=board._sha(review_path),
        command=[sys.executable, '-m', 'tools.hardware_template_coverage_v1.run_board_case', *sys.argv[1:]],
        cwd=str(Path.cwd().resolve()), structure_source=str(structure_path), review_source=str(review_path),
        wait_seconds=args.wait_seconds,
        driver_sha256=board._sha(__file__), board_id=args.board_id, formal_cost_eligible=False))
    save_new(execution / 'ownership_config.json', dict(config=ownership_config, source=str(ownership_path),
        source_sha256=board._sha(ownership_path)))

    def emit(stage, **fields):
        row = dict(stage=stage, case_id=case['case_id'], utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (execution/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
        print(json.dumps(row), flush=True)

    def stage(mode):
        job_id = prefix + '_' + mode
        folder = attempt / 'board_transport' / job_id
        snapshot = attempt / 'board_snapshots' / job_id
        transfer.prepare(package, folder, mode=mode,
            bridge=ROOT/'tools/hardware_template_coverage_v1/board_bridge_stdin.py', chunk_bytes=4194304,
            ownership_config=ownership_config)
        emit(mode+'_upload_begin')
        transfer.upload(folder)
        local_folder, config = transfer.config_for(folder)
        transfer.call(local_folder, config, 'launch', job_id=job_id,
                      board_id=args.board_id, timeout_seconds=30)
        emit(mode+'_dispatched', job_id=job_id)
        deadline = time.monotonic()+args.wait_seconds
        last = None
        while True:
            state = transfer.call(local_folder, config, 'status', job_id=job_id)
            if state['state'] != last:
                emit(mode+'_state', state=state['state'])
                last = state['state']
            if state['state'] in ('PASS', 'FAIL', 'DEVICE_STATE_UNKNOWN',
                                  'WORKER_FAILED_LOCK_RETAINED', 'DMA_QUARANTINED'):
                break
            if time.monotonic() >= deadline:
                emit(mode+'_wait_expired_no_kill_or_retry', last_state=state['state'])
                raise TimeoutError('finite board wait expired; retain job/lock and inspect it')
            time.sleep(10)
        fetched = transfer.fetch(folder, job_id, snapshot)
        emit(mode+'_snapshot_verified', observed_state=fetched['observed_state'], snapshot=str(snapshot))
        if state['state'] not in ('PASS', 'FAIL'):
            raise RuntimeError('unsafe or unknown device state; snapshot retained without import or another launch')
        if mode == 'probe' and state['state'] != 'PASS':
            raise RuntimeError('probe failed; snapshot retained, timing not launched')
        return snapshot, state

    try:
        probe, _ = stage('probe')
        audit.finalize(package, structure_path, review_path, probe/'evidence/output')
        emit('integrated_phase_binding_pass')
        board.prepare_bundle(package, mode='timing')
        timing, state = stage('timing')
        checked = import_timing(timing, package, attempt, prefix+'_timing')
        save_new(execution/'result.json', dict(**checked, worker_state=state['state']))
        emit('timing_imported', numerical_status=checked['numerical_status'],
             hardware_measurement_eligible=checked['hardware_measurement_eligible'],
             protocol_conditions_met=checked['protocol_conditions_met'])
        return 0 if checked['numerical_status'] == 'PASS' else 2
    except BaseException as error:
        emit('case_stopped_preserving_evidence', error_type=type(error).__name__, reason=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',default='attempt01')
    parser.add_argument('--case-dir',required=True)
    parser.add_argument('--structure',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--job-prefix',required=True)
    parser.add_argument('--board-id',required=True)
    parser.add_argument('--ownership-config',required=True)
    parser.add_argument('--wait-seconds',type=int,default=1800)
    args=parser.parse_args()
    if not 1 <= args.wait_seconds <= 3600:
        parser.error('wait-seconds must be between 1 and 3600')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
