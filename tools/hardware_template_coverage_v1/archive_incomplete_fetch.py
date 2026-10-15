"""Preserve an empty failed-download directory as a bound transport incident.

Requires the same job's fully imported successful recovery. No remote actions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .board_transfer import sha, save, valid_job


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--recovered', type=Path, required=True)
    args = parser.parse_args()
    attempt, recovered = args.attempt.resolve(), args.recovered.resolve()
    job_id = valid_job(args.job_id)
    original = (attempt / 'board_snapshots' / job_id).resolve()
    target = (attempt / 'board_incidents' / ('fetch_timeout_' + job_id)).resolve()
    assert original.parent == attempt / 'board_snapshots' and target.parent == attempt / 'board_incidents'
    assert recovered.parent == attempt / 'board_snapshots' and original != recovered
    assert original.is_dir() and not list(original.iterdir()), 'only empty unfinished fetch directory is supported'
    assert not any(p.is_symlink() for p in (original, recovered, target, *target.parents))
    assert not target.exists()
    imported = load(recovered / 'import_recovery.json')['result']
    manifest = load(recovered / 'evidence/snapshot_manifest.json')
    status = load(recovered / 'evidence/status.json')
    request = load(recovered / 'evidence/request.json')
    assert manifest['job_id'] == request['job_id'] == status['job_id'] == job_id
    assert status['state'] == 'PASS' and status['returncode'] == 0 and status['board_lock_retained'] is False
    assert imported['status'] == 'PASS_IMPORT_AUDIT' and imported['numerical_status'] == 'PASS'
    case_id = imported['case_id']
    pointer = attempt / 'cases' / case_id / 'board/selected_run.json'
    selected = load(pointer)['run_directory']
    selected_audit = attempt / 'cases' / case_id / 'board' / selected / 'verified_snapshot_import.json'
    assert load(selected_audit) == imported and selected == job_id
    prefix = job_id.removesuffix('_timing')
    execution = attempt / 'board_execution' / prefix
    events = [json.loads(line) for line in (execution / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    stop = events[-1]
    assert stop['stage'] == 'case_stopped_preserving_evidence' and 'bridge response timed out' in stop['reason']
    assert any(row['stage'] == 'timing_state' and row['state'] == 'PASS' for row in events)
    target.mkdir(parents=True, exist_ok=False)
    destination = target / 'empty_unfinished_snapshot_directory'
    assert destination.parent == target and not destination.exists()
    original.rename(destination)
    record = dict(schema='t6_transport_recovery_incident_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        case_id=case_id, job_id=job_id, run_valid=True,
        conclusion='CLIENT_SNAPSHOT_TIMEOUT_AFTER_BOARD_PASS; EXISTING_ARCHIVE_RECOVERED_AND_REPLAYED; CPU_CONDITION_DEVIATION_RETAINED',
        timeline=[row for row in events if row['stage'] in ('timing_dispatched','timing_state','case_stopped_preserving_evidence')],
        protective_action=dict(queue_stopped=True, measurement_restarted=False, remote_files_changed=False),
        original_empty_directory=str(original), archived_empty_directory=str(destination),
        recovered_snapshot=str(recovered), selected_audit=str(selected_audit),
        hardware_measurement_eligible=imported['hardware_measurement_eligible'], formal_cost_eligible=False)
    save(target / 'incident.json', record, exclusive=True)
    paths = [target / 'incident.json', execution / 'inputs.json', execution / 'events.jsonl',
        recovered / 'download.json', recovered / 'recovery.json', recovered / 'import_recovery.json',
        recovered / 'evidence/snapshot_manifest.json', recovered / 'evidence/status.json',
        recovered / 'evidence/output/summary.json', recovered / 'evidence/output/timing_raw.csv',
        pointer, selected_audit]
    save(target / 'evidence_manifest.json', dict(schema='t6_local_incident_evidence_v1',
        files={path.relative_to(attempt).as_posix():sha(path) for path in paths}), exclusive=True)
    print(json.dumps(dict(status='PRESERVED_BOUND_RECOVERED_TRANSPORT_INCIDENT', path=str(target))))


if __name__ == '__main__':
    main()
