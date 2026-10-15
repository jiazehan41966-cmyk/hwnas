"""Repackage exact reviewed hardware bytes with a separately reviewed runtime.

Offline only. Preserve old packages and approvals. Recompute the existing
static RTL/HWH proof against each new package before returning its old exact
review references. Hardware sources, bitstreams and fixtures cannot change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import board_runner as board
from . import phase_audit

ROOT = Path(__file__).resolve().parents[2]


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(attempt, previous_index, phase_index, runtime_review):
    attempt = Path(attempt).resolve()
    previous_index, phase_index, runtime_review = map(Path, (previous_index, phase_index, runtime_review))
    previous, phases, review = map(board._load, (previous_index, phase_index, runtime_review))
    if review.get('status') != 'PASS_SCOPED_OFFLINE_OWNERSHIP_INTEGRATION_REVIEW':
        raise ValueError('successful independent current runtime review required')
    sources = review['sources']
    for name in ('board_runner.py', 'board_transfer.py', 'board_ownership.py', 'run_board_case.py'):
        relative = 'tools/hardware_template_coverage_v1/' + name
        if board._sha(ROOT / relative) != sources[relative]:
            raise ValueError('runtime changed after review: ' + name)
    inventory = board._load(attempt / 'inventory.json')
    if set(row['case_id'] for row in previous['cases']) != set(inventory['representative_case_ids']):
        raise ValueError('exact representative whitelist required')
    phase_map = {row['case_id']: row for row in phases['cases']}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rows = []
    for row in previous['cases']:
        case_id = row['case_id']
        old = Path(row['case_dir'])
        if board._sha(old / 'board_probe_bundle.json') != row['probe_bundle_sha256']:
            raise ValueError('old probe bundle changed')
        old_bundle = board._load(old / 'board_probe_bundle.json')
        board._checked_files(old, old_bundle['files'])
        raw_path = old / 'package_transport_manifest.json'
        if board._sha(raw_path) != row['raw_transport_manifest_sha256']:
            raise ValueError('raw hardware manifest changed')
        raw = board._load(raw_path)
        board._checked_files(old, raw['files'])
        destination = attempt / 'board_packages' / case_id / stamp / 'case'
        destination.mkdir(parents=True, exist_ok=False)
        names = set(raw['files']) | {'package_transport_manifest.json', 'case.json'}
        if names & {'board_runner.py', 'board_ownership.py', 'board_probe_bundle.json', 'phase_validation.json'}:
            raise ValueError('raw hardware unexpectedly contains runtime or board evidence')
        for name in sorted(names):
            source = board._child(old, name)
            target = board._child(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if board._sha(target) != board._sha(source):
                raise ValueError('copied hardware bytes differ')
        bundle = board.prepare_bundle(destination, mode='probe')
        board.validate_bundle(destination, mode='probe')
        phase = phase_map[case_id]
        for key in ('structure', 'review'):
            if board._sha(phase[key + '_path']) != phase[key + '_sha256']:
                raise ValueError('old exact static approval changed')
        structure = board._load(phase['structure_path'])
        fresh = phase_audit.inspect_case(destination, timer_validation=structure['timer_validation']['validation_path'])
        for key in ('case_sha256', 'auditor_sha256', 'source_files', 'rtl', 'wiring', 'timer_validation'):
            if json.loads(json.dumps(fresh[key])) != structure[key]:
                raise ValueError('new package static evidence differs: ' + key)
        result = dict(case_id=case_id, case_sha256=row['case_sha256'], case_dir=str(destination),
            previous_case_dir=str(old), status='PASS_OFFLINE_CURRENT_RUNTIME_AND_UNCHANGED_STATIC_REVIEW',
            runtime_review_path=str(runtime_review.resolve()), runtime_review_sha256=board._sha(runtime_review),
            probe_bundle_sha256=board._sha(destination / 'board_probe_bundle.json'),
            bitstream_sha256=bundle['files']['vivado_shell/top.bit'],
            raw_transport_manifest_sha256=board._sha(raw_path),
            structure_path=phase['structure_path'], structure_sha256=phase['structure_sha256'],
            review_path=phase['review_path'], review_sha256=phase['review_sha256'],
            board_actions_performed=False, formal_cost_eligible=False)
        save_new(destination.parent / 'package_result.json', result)
        rows.append(result)
    result = dict(schema='t6_shared_ownership_runtime_packages_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), status='PASS_13_OFFLINE_REVIEWED_PROBE_PACKAGES',
        previous_index_sha256=board._sha(previous_index), static_phase_index_sha256=board._sha(phase_index),
        runtime_review_sha256=board._sha(runtime_review), sources=sources,
        board_actions_performed=False, formal_cost_eligible=False, cases=rows)
    output = attempt / 'runtime_packages' / stamp / 'index.json'
    save_new(output, result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'previous-index', 'phase-index', 'runtime-review'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(run(args.attempt, args.previous_index, args.phase_index, args.runtime_review))


if __name__ == '__main__':
    main()
