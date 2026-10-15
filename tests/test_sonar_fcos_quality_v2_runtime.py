"""V2 resource affinity, numerical migration and generated transport contracts."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sonar_fcos_pilot.common import read_json, sha256_file, write_json
from tools.sonar_fcos_quality_v2 import migration, remote, scheduler


@pytest.fixture
def portable_claims(monkeypatch):
    try:
        import fcntl  # noqa: F401
    except ImportError:
        monkeypatch.setattr(scheduler, 'blocking_lock', lambda path: nullcontext())


def test_gpu0_can_finish_all_three_pairs_when_other_cards_busy(tmp_path, portable_claims):
    source = {'freeze': 'test'}
    scheduler.initialize_claims(tmp_path, source, {'42': 0})
    assert scheduler.claim_pair(tmp_path, 0, source) == 42
    assert scheduler.claim_pair(tmp_path, 0, source, [42]) == 43
    assert scheduler.claim_pair(tmp_path, 0, source, [42, 43]) == 44
    assert scheduler.claim_pair(tmp_path, 0, source, [42, 43, 44]) is None


def test_pair_affinity_is_persistent_and_mainline_never_claimed(tmp_path, portable_claims):
    source = {'freeze': 'test'}
    scheduler.initialize_claims(tmp_path, source, {'42': 0})
    assert scheduler.claim_pair(tmp_path, 2, source) == 43
    assert scheduler.claim_pair(tmp_path, 3, source) == 44
    assert scheduler.claim_pair(tmp_path, 0, source, [42]) is None
    assert scheduler.claim_pair(tmp_path, 2, source) == 43
    with pytest.raises(ValueError, match='GPU 1'):
        scheduler.claim_pair(tmp_path, 1, source)
    with pytest.raises(ValueError, match='different source'):
        scheduler.initialize_claims(tmp_path, {'freeze': 'changed'})


def test_parallel_worker_claims_are_distinct_on_linux(tmp_path):
    pytest.importorskip('fcntl')
    source = {'freeze': 'test'}
    scheduler.initialize_claims(tmp_path, source)
    with ThreadPoolExecutor(max_workers=3) as pool:
        seeds = list(pool.map(lambda gpu: scheduler.claim_pair(tmp_path, gpu, source), (0, 2, 3)))
    assert sorted(seeds) == [42, 43, 44]
    claims = read_json(tmp_path / 'scheduler/seed_pair_claims.json')
    assert {row['gpu'] for row in claims['pairs'].values()} == {0, 2, 3}


def test_priority_request_only_affects_own_or_all_workers(tmp_path):
    assert scheduler.request_reason(tmp_path, 2, [900]) == 'foreign_gpu_process'
    (tmp_path / 'PRIMARY_GPU_REQUEST_GPU2').touch()
    assert scheduler.request_reason(tmp_path, 2) == 'mainline_request'
    assert scheduler.request_reason(tmp_path, 0) is None
    (tmp_path / 'STOP_QUEUE').touch()
    assert scheduler.request_reason(tmp_path, 0) == 'stop_queue'


def test_resume_archives_only_own_stop_after_explicit_idle_check(tmp_path):
    stop = tmp_path / 'STOP_QUEUE'
    stop.write_text('paused by supervisor')
    primary = tmp_path / 'PRIMARY_GPU_REQUEST'
    primary.write_text('mainline priority')
    with pytest.raises(RuntimeError):
        scheduler.archive_stop_marker(tmp_path, resume=False, processes_clear=True)
    with pytest.raises(RuntimeError):
        scheduler.archive_stop_marker(tmp_path, resume=True, processes_clear=False)
    archive = scheduler.archive_stop_marker(tmp_path, resume=True, processes_clear=True)
    assert not stop.exists() and Path(archive).read_text() == 'paused by supervisor'
    assert primary.read_text() == 'mainline priority'


def test_cuda_visibility_uses_physical_uuid_instead_of_ordinal(tmp_path):
    env = scheduler.child_environment({'index': 2, 'uuid': 'GPU-physical-two', 'compute_pids': []}, tmp_path / 'pause')
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-physical-two'
    with pytest.raises(ValueError):
        scheduler.child_environment({'index': 1, 'uuid': 'GPU-mainline', 'compute_pids': []}, tmp_path / 'pause')


def test_atomic_summary_recovers_completion_when_status_lags(tmp_path):
    source = {'freeze': 'test'}
    binding = {'source_binding': source, 'variant': 'AAA', 'seed': 42}
    run, cp = tmp_path / 'runs/AAA_seed42', tmp_path / 'checkpoints/AAA_seed42'
    cp.mkdir(parents=True)
    write_json(run / 'status.json', {'status': 'DIAGNOSING_SELECTED_CHECKPOINT', 'completed_epochs': 30})
    write_json(run / 'summary.json', {'status': 'COMPLETE', 'trained_epochs': 30,
                                     'source_binding': source, 'binding': binding})
    torch.save({'epoch': 29, 'history': [{}] * 30, 'binding': binding}, cp / 'last.pt')
    assert scheduler.run_progress(tmp_path, 'AAA', 42, tmp_path / 'checkpoints', source)['status'] == 'COMPLETE'
    assert scheduler.completed_run_count(tmp_path, source) == 1
    assert scheduler.complete_seeds(tmp_path, source) == []


def state(binding):
    return {'model': {'weight': torch.arange(6).reshape(2, 3).float()},
            'optimizer': {'state': {0: {'momentum_buffer': torch.tensor([.1, .2])}}},
            'scaler': {'scale': 1024.0}, 'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()},
            'history': [{'epoch': 1, 'seconds': 1.0, 'validation': {'ap75': 12.5}},
                        {'epoch': 2, 'seconds': 1.1, 'validation': {'ap75': 13.5}}],
            'best_map': 9.75, 'epoch': 1, 'best_epoch': 2, 'binding': binding,
            'best_artifacts': {'placeholder': {'path': 'old'}}}


def test_rebinding_preserves_every_numerical_state_without_aliasing():
    before = state({'source': 'old'})
    after = migration.rebind_checkpoint(before, {'source': 'new'}, {'checkpoint': {'path': 'new'}})
    assert migration.numerical_digest(before) == migration.numerical_digest(after)
    assert after['binding'] != before['binding']
    after['model']['weight'][0, 0] = 99
    assert before['model']['weight'][0, 0] == 0
    assert migration.numerical_digest(before) != migration.numerical_digest(after)


def test_late_retention_gap_is_explicit():
    assert migration.retention_check(3)['late_retention_fully_prospective']
    assert not migration.retention_check(25)['late_retention_gap']
    assert migration.retention_check(26)['late_retention_gap']
    with pytest.raises(ValueError):
        migration.retention_check(0)


def test_checked_copy_never_replaces_different_or_external_target(tmp_path):
    staging = tmp_path / 'source.bin'
    staging.write_bytes(b'preserved')
    root = tmp_path / 'new'
    target = root / 'sub/file.bin'
    migration.checked_copy(staging, target, sha256_file(staging), root)
    assert target.read_bytes() == b'preserved'
    target.write_bytes(b'different')
    with pytest.raises(FileExistsError):
        migration.checked_copy(staging, target, sha256_file(staging), root)
    with pytest.raises(ValueError, match='escapes'):
        migration.checked_copy(staging, tmp_path / 'outside.bin', sha256_file(staging), root)


def test_full_migration_copies_rebinds_and_replays_idempotently(tmp_path, monkeypatch):
    from tools.sonar_fcos_quality_v2 import campaign
    parent, new = tmp_path / 'parent', tmp_path / 'new'
    output = new / 'results/amendment'
    checkpoint_root = new / 'checkpoints/amendment'
    old_output = parent / migration.PARENT_RESULT
    relative = 'tools/sonar_fcos_quality/model.py'
    for root in (parent, new):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes(b'unchanged implementation')
    old_source = {'sources': {relative: sha256_file(parent / relative)}, 'config_sha256': 'old'}
    new_source = {'sources': {relative: sha256_file(new / relative)}, 'config_sha256': 'new'}
    environment = {'runtime': 'test'}
    old_binding = {'source_binding': old_source, 'environment': environment, 'variant': 'AAA', 'seed': 42}
    config = {'quality': {}, 'variants': ['AAA', 'QFL_DFL'], 'seeds': [42, 43, 44], 'epochs': 30,
              'max_formal_runs': 6, 'expected_manifest_sha256': 'data', 'output_dir': str(output),
              'checkpoint_root': str(checkpoint_root)}
    old_config = {**config, 'checkpoint_root': 'checkpoints/parent'}
    write_json(parent / migration.PARENT_CONFIG, old_config)
    write_json(old_output / 'execution_manifest.json', {'source_binding': old_source, 'environment': environment})
    write_json(output / 'execution_manifest.json', {'source_binding': new_source, 'environment': environment})
    write_json(old_output / 'launch.json', {'eligible_gpu_indices': [0]})
    old_run = old_output / 'runs/AAA_seed42'
    old_cp = parent / 'checkpoints/parent/AAA_seed42'
    old_cp.mkdir(parents=True)
    old_receipt = {**old_binding, 'binding': old_binding, 'model': {'kind': 'unchanged'},
                   'training_images': 2, 'validation_images': 2}
    write_json(old_run / 'run_manifest.json', old_receipt)
    best_path = old_cp / 'best_epoch_002.pt'
    saved = state(old_binding)
    torch.save({'model': saved['model'], 'epoch': 1, 'binding': old_binding}, best_path)
    prediction_path, metric_path = old_run / 'validation/predictions.json', old_run / 'validation/metrics.json'
    write_json(prediction_path, [{'image_id': 1, 'boxes': []}])
    write_json(metric_path, {'ap75': 13.5})
    saved['best_artifacts'] = {key: {'path': str(path), 'sha256': sha256_file(path)}
        for key, path in [('checkpoint', best_path), ('predictions', prediction_path), ('metrics', metric_path)]}
    torch.save(saved, old_cp / 'last.pt')
    write_json(old_run / 'history.json', saved['history'])
    parent_hashes = {str(p): sha256_file(p) for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(migration, 'ROOT', new)
    monkeypatch.setattr(migration, 'PARENT_ROOT', parent)
    monkeypatch.setattr(migration, 'PARENT_EXECUTION_SHA', sha256_file(old_output / 'execution_manifest.json'))
    monkeypatch.setattr(migration, 'ensure_parent_stopped', lambda p: {'status': 'PARENT_PAUSED'})
    monkeypatch.setattr(campaign, 'load_config', lambda path: config)
    monkeypatch.setattr(campaign, 'source_binding', lambda cfg: new_source)
    receipt = migration.migrate('synthetic_config.json')
    migrated = torch.load(checkpoint_root / 'AAA_seed42/last.pt', map_location='cpu', weights_only=False)
    assert receipt['status'] == 'MIGRATION_PASS'
    assert receipt['remaining_epoch_budget'] == 178
    assert migrated['binding']['source_binding'] == new_source
    assert migration.numerical_digest(migrated) == migration.numerical_digest(saved)
    assert migrated['retained_epochs'] == {}
    assert Path(migrated['best_artifacts']['predictions']['path']).read_bytes() == prediction_path.read_bytes()
    assert {str(p): sha256_file(p) for p in parent.rglob('*') if p.is_file()} == parent_hashes
    assert migration.migrate('synthetic_config.json') == receipt


def capture_remote(monkeypatch, tmp_path, method, *args):
    captured = []
    monkeypatch.setattr(remote, 'LOCAL', tmp_path)
    monkeypatch.setattr(remote, 'execute_json', lambda code, timeout=60: captured.append(code) or {'status': 'MOCK'})
    method(*args)
    for code in captured:
        compile(code, '<generated-v2-remote>', 'exec')
    return captured


def test_preflight_supervisor_uses_v2_entry_and_compiles(monkeypatch, tmp_path):
    script = capture_remote(monkeypatch, tmp_path, remote.preflight)[0]
    tree = ast.parse(script)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'supervisor' for t in n.targets))
    scope = {'root': Path('/isolated'), 'result': Path('/isolated/results/v2'), 'commands': [['python', 'run']]}
    supervisor = eval(compile(ast.fix_missing_locations(ast.Expression(assignment.value)), '<builder>', 'eval'), scope)
    compile(supervisor, '<supervisor>', 'exec')
    assert 'tools.sonar_fcos_quality_v2.train' in script
    assert "{'gpu_indices':[0]}" in script


def test_launch_and_migration_scripts_have_new_gates(monkeypatch, tmp_path):
    launch = capture_remote(monkeypatch, tmp_path, remote.launch)[0]
    assert 'MIGRATION_PASS' in launch and 'migration_sha256' in launch and 'preregistration_sha256' in launch
    assert "'supervise'" in launch and "'eligible_gpu_indices':[0,2,3]" in launch
    code = capture_remote(monkeypatch, tmp_path, remote.migrate)[0]
    assert 'preflight_complete.json' in code and 'tools.sonar_fcos_quality_v2.migration' in code


@pytest.mark.parametrize('path', ['confidence_intervals/best.json', 'branch_results.json',
    'runs/AAA_seed42/summary.json', 'runs/AAA_seed42/retained/epoch_026/evaluation.json',
    'runs/AAA_seed42/retained/epoch_026/metrics.json'])
def test_compact_collection_keeps_reports_and_complete_ci_draws(path):
    assert remote.collection_policy(path, 100000)[0]


@pytest.mark.parametrize('path', ['runs/AAA_seed42/retained/epoch_026/predictions.json',
    'runs/AAA_seed42/validation/epoch_001_predictions.json',
    'runs/AAA_seed42/evaluations/best/stage_rows.json', 'runs/AAA_seed42/history.json',
    'migration_staging/prepared/run/validation/metrics.json', 'pretrained/uatd_fcos_best.pt'])
def test_compact_collection_preserves_large_raw_caches_on_server(path):
    include, reason = remote.collection_policy(path, 100000)
    assert not include and 'remote' in reason


def test_compact_status_does_not_expose_validation_values(monkeypatch, tmp_path):
    root = tmp_path / 'remote'
    result = root / 'results/v2'
    write_json(result / 'smoke_gate.json', {'status': 'PASS', 'validation': {'ap75': 99.123}})
    write_json(result / 'runs/AAA_seed42/status.json', {'status': 'TRAINING', 'completed_epochs': 3,
                                                     'loss': .111, 'best_map': 99.123})
    monkeypatch.setattr(remote, 'REMOTE', str(root))
    monkeypatch.setattr(remote, 'RESULT', 'results/v2')
    script = capture_remote(monkeypatch, tmp_path / 'local', remote.status)[0]
    monkeypatch.setattr(remote.subprocess, 'check_output', lambda *a, **k: '0, RTX 5090, 29, 0\n')
    stream = io.StringIO()
    with redirect_stdout(stream):
        exec(compile(script, '<status>', 'exec'), {})
    value = json.loads(stream.getvalue())
    assert value['validation_metric_values_examined'] is False
    assert '99.123' not in stream.getvalue() and '0.111' not in stream.getvalue()
    assert value['runs/AAA_seed42/status.json']['completed_epochs'] == 3
