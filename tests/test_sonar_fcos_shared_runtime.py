"""Behavioral tests for the resource-only frozen-v2 sharing adapter."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools.sonar_fcos_pilot.common import read_json, write_json
from tools.sonar_fcos_quality_shared_runtime import resource, scheduler
from tools.sonar_fcos_quality_v2 import scheduler as original


@pytest.fixture
def policy():
    return {'schema_version': 1, 'allowed_gpu_indices': [0, 2, 3], 'reserved_gpu_indices': [1],
            'boot_id': 'boot-fixed', 'shared_gpu_indices': [2, 3],
            'pair_gpu_mapping': {'42': 0, '43': 2, '44': 3},
            'gpu_uuids': {'2': 'GPU-two', '3': 'GPU-three'},
            'resident_allowlist': [
                {'gpu_index': gpu, 'gpu_uuid': uuid, 'pid': pid, 'uid': 1000,
                 'start_ticks': start, 'comm': 'python', 'process_path': '/registered/env/bin/python',
                 'approved_memory_mib': 1690}
                for gpu, uuid, pid, start in [(2, 'GPU-two', 5406, 2307), (3, 'GPU-three', 5300, 2156)]],
            'guard': {'startup_max_utilization_percent': 5, 'min_free_memory_mib': 12288,
                      'max_resident_memory_mib': 2048, 'max_resident_memory_growth_mib': 64,
                      'cpu_busy_core_fraction': .02, 'cpu_min_sample_seconds': 5,
                      'shared_worker_nice': 10, 'per_pid_sm_supported': False}}


@pytest.fixture
def observed(policy):
    cards = [{'index': gpu, 'uuid': uuid, 'name': 'NVIDIA GeForce RTX 5090',
              'memory_used_mib': 1690, 'memory_total_mib': 32768,
              'utilization_percent': 0, 'compute_pids': [pid]}
             for gpu, uuid, pid in [(2, 'GPU-two', 5406), (3, 'GPU-three', 5300)]]
    identities = {row['pid']: {key: row[key] for key in ('pid', 'uid', 'comm', 'start_ticks')}
                  | {'cpu_ticks': 100, 'boot_id': policy['boot_id']}
                  for row in policy['resident_allowlist']}
    apps = {(card['uuid'], card['compute_pids'][0]):
            {'process_path': '/registered/env/bin/python', 'memory_mib': 1690} for card in cards}
    now = [0.0]
    guard = resource.SharedResources(policy, identity_reader=lambda pid: identities[pid],
        apps_reader=lambda: apps, monotonic=lambda: now[0], clock_ticks=100)
    return SimpleNamespace(cards=cards, identities=identities, apps=apps, now=now, guard=guard)


def ready(obs, gpu=2):
    assert not obs.guard.observe(obs.cards, gpu, startup=True)
    obs.now[0] += 5
    assert obs.guard.observe(obs.cards, gpu, startup=True)


def test_exact_resident_requires_two_samples_and_waits_minimum(observed):
    obs = observed
    assert not obs.guard.observe(obs.cards, 2, startup=True)
    assert obs.guard.last[2]['reason'] == 'cpu_second_sample_required'
    obs.now[0] = 4.99
    assert not obs.guard.observe(obs.cards, 2, startup=True)
    obs.now[0] = 5
    assert obs.guard.observe(obs.cards, 2, startup=True)
    assert obs.guard.last[2]['gpu_service_idle_verified'] is False


@pytest.mark.parametrize('key,new', [('pid', 900), ('uid', 999), ('start_ticks', 2308),
                                   ('comm', 'other'), ('boot_id', 'reboot')])
def test_identity_reuse_or_reboot_is_not_allowed(observed, key, new):
    ready(observed)
    observed.identities[5406][key] = new
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == 'resident_identity_changed'


@pytest.mark.parametrize('change,reason', [('uuid', 'gpu_identity_changed'),
    ('path', 'resident_process_path_changed'), ('pid', 'unknown_compute_pid'),
    ('missing', 'registered_resident_missing'), ('memory', 'resident_memory_increased'),
    ('memory_growth', 'resident_memory_increased'), ('low_free', 'insufficient_free_gpu_memory')])
def test_resource_changes_fail_closed(observed, change, reason):
    ready(observed)
    if change == 'uuid': observed.cards[0]['uuid'] = 'replaced'
    elif change == 'path': observed.apps[('GPU-two', 5406)]['process_path'] = '/other/python'
    elif change == 'pid': observed.cards[0]['compute_pids'].append(999)
    elif change == 'missing': observed.cards[0]['compute_pids'] = []
    elif change == 'memory': observed.apps[('GPU-two', 5406)]['memory_mib'] = 2049
    elif change == 'memory_growth': observed.apps[('GPU-two', 5406)]['memory_mib'] += 65
    elif change == 'low_free': observed.cards[0]['memory_used_mib'] = 22000
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == reason


def test_observation_read_failure_yields_without_ignoring_resident(observed):
    ready(observed)
    def unreadable(pid): raise PermissionError('unreadable')
    observed.guard.identity_reader = unreadable
    with pytest.raises(resource.ResourceBlockedError, match='resource_observation_failed'):
        observed.guard.foreign_pids(observed.cards, 2, set(), lambda *args: [])


def test_cpu_busy_boundary_and_short_launch_recheck(observed):
    obs = observed
    ready(obs)
    obs.now[0] += 5
    obs.identities[5406]['cpu_ticks'] += 10
    assert not obs.guard.observe(obs.cards, 2)
    assert obs.guard.last[2]['reason'] == 'resident_cpu_activity'
    obs.now[0] += 5
    assert obs.guard.observe(obs.cards, 2)
    obs.now[0] += .1
    obs.identities[5406]['cpu_ticks'] += 1
    assert not obs.guard.observe(obs.cards, 2)


@pytest.mark.parametrize('field', ['ticks', 'clock'])
def test_counter_or_clock_rollback_does_not_use_previous_quiet_sample(observed, field):
    ready(observed)
    if field == 'ticks': observed.identities[5406]['cpu_ticks'] -= 1
    else: observed.now[0] -= 1
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == 'cpu_counter_or_clock_reset'


def test_stale_sample_requires_new_pair_instead_of_diluting_activity(observed):
    ready(observed)
    observed.now[0] += 1000
    observed.identities[5406]['cpu_ticks'] += 100
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == 'cpu_sample_stale'
    observed.now[0] += 5
    assert observed.guard.observe(observed.cards, 2)


def test_memory_growth_guard_uses_registered_baseline_even_on_first_observation(observed):
    observed.apps[('GPU-two', 5406)]['memory_mib'] = 1800
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == 'resident_memory_increased'


def test_running_aggregate_utilization_does_not_claim_resident_busy(observed):
    obs = observed
    ready(obs)
    obs.cards[0]['utilization_percent'] = 100
    obs.cards[0]['compute_pids'].append(900)
    obs.apps[('GPU-two', 900)] = {'process_path': '/own/train/python', 'memory_mib': 1800}
    assert obs.guard.foreign_pids(obs.cards, 2, {900}, lambda *args: None) == []
    assert not obs.guard.observe(obs.cards, 2, {900}, startup=True)
    assert obs.guard.last[2]['reason'] == 'startup_gpu_busy'


def test_second_nvidia_query_unknown_pid_is_not_ignored(observed):
    ready(observed)
    observed.apps[('GPU-two', 901)] = {'process_path': '/unknown', 'memory_mib': 100}
    assert not observed.guard.observe(observed.cards, 2)
    assert observed.guard.last[2]['reason'] == 'inconsistent_compute_observation'


def test_gpu0_preserves_original_checks_and_gpu1_never_selected(observed):
    calls = []
    def select(cards, scheduling):
        calls.append(scheduling)
        return {'index': 0}
    assert observed.guard.select_gpu(observed.cards, {'gpu_indices': [1]}, select) is None
    assert observed.guard.select_gpu(observed.cards, {'gpu_indices': [0]}, select) == {'index': 0}
    assert calls == [{'gpu_indices': [0]}]
    assert observed.guard.foreign_pids(observed.cards, 0, {1}, lambda *args: [2]) == [2]


def test_child_cuda_visibility_uses_registered_uuid_and_rechecks(observed, tmp_path):
    ready(observed)
    env = observed.guard.child_environment(observed.cards[0], tmp_path / 'pause', lambda *args: None)
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-two'
    observed.cards[0]['compute_pids'].append(999)
    with pytest.raises(resource.ResourceBlockedError):
        observed.guard.child_environment(observed.cards[0], tmp_path / 'pause', lambda *args: None)


def test_policy_cannot_implicitly_allow_gpu1_or_unreviewed_relaxation(policy):
    for field, new in [('shared_gpu_indices', [1, 2]), ('pair_gpu_mapping', {'42': 0, '43': 1, '44': 3})]:
        changed = copy.deepcopy(policy)
        changed[field] = new
        with pytest.raises(ValueError): resource.validate_policy(changed)
    changed = copy.deepcopy(policy)
    changed['guard']['max_resident_memory_mib'] = 10000
    with pytest.raises(ValueError): resource.validate_policy(changed)


def test_read_process_identity_handles_spaces_and_checks_all_uids(tmp_path):
    root = tmp_path / 'proc'
    (root / '5406').mkdir(parents=True)
    (root / 'sys/kernel/random').mkdir(parents=True)
    fields = ['S'] + ['0'] * 21
    fields[11], fields[12], fields[19] = '9', '3', '2307'
    (root / '5406/stat').write_text('5406 (python (worker)) ' + ' '.join(fields))
    (root / '5406/status').write_text('Uid:\t1000\t1000\t1000\t1000\n')
    (root / 'sys/kernel/random/boot_id').write_text('boot-fixed\n')
    identity = resource.read_process_identity(5406, root)
    assert identity['comm'] == 'python (worker)' and identity['start_ticks'] == 2307
    assert identity['cpu_ticks'] == 12 and identity['boot_id'] == 'boot-fixed'
    (root / '5406/status').write_text('Uid:\t1000\t0\t1000\t1000\n')
    with pytest.raises(ValueError, match='UIDs differ'): resource.read_process_identity(5406, root)


def test_popen_rewrites_worker_only_and_train_command_is_unchanged(tmp_path):
    commands = []
    base = SimpleNamespace(Popen=lambda cmd, *args, **kwargs: commands.append(cmd) or cmd)
    adapter = scheduler.SubprocessAdapter(tmp_path / 'policy', tmp_path / 'freeze', tmp_path / 'review', base)
    worker = [sys.executable, '-m', 'tools.sonar_fcos_quality_v2.scheduler', 'worker', '--config', 'fixed', '--gpu', '2']
    adapter.Popen(worker)
    assert commands[0][2] == scheduler.MODULE and commands[0][3:8] == worker[3:8]
    assert '--policy' in commands[0] and '--freeze-manifest' in commands[0] and '--review-gate' in commands[0]
    train = [sys.executable, '-m', 'tools.sonar_fcos_quality_v2.train', 'run', '--epoch-budget', '1']
    adapter.Popen(train)
    assert commands[1] == train and worker[2] == 'tools.sonar_fcos_quality_v2.scheduler'


def test_claims_fill_null_only_under_existing_lock_and_preserve_completed(tmp_path, policy, monkeypatch):
    # Windows lacks fcntl; the real lock is separately exercised on the server.
    try: import fcntl  # noqa: F401
    except ImportError: monkeypatch.setattr(original, 'blocking_lock', lambda path: nullcontext())
    source = {'frozen': 'v2'}
    scheduler._ORIGINAL_INITIALIZE(tmp_path, source, {'42': 0})
    path = tmp_path / 'scheduler/seed_pair_claims.json'
    claims = read_json(path)
    claims['pairs']['42']['complete'] = True
    write_json(path, claims)
    result = scheduler.prepare_claims(tmp_path, source, policy, {'42': 0})
    assert {seed: row['gpu'] for seed, row in result['pairs'].items()} == policy['pair_gpu_mapping']
    assert result['pairs']['42']['complete'] is True
    assert original.claim_pair(tmp_path, 0, source, [42]) is None
    assert original.claim_pair(tmp_path, 2, source, [42]) == 43
    assert original.claim_pair(tmp_path, 3, source, [42]) == 44
    result['pairs']['43']['gpu'] = 0
    write_json(path, result)
    with pytest.raises(ValueError, match='cannot be moved'):
        scheduler.prepare_claims(tmp_path, source, policy, {'42': 0})


def test_launch_veto_waits_and_retries_after_worker_lock_release(tmp_path, monkeypatch):
    calls = []
    def worker(*args):
        calls.append(args)
        if len(calls) == 1:
            raise resource.LaunchResourceBlockedError('late resident activity')
        return {'status': 'COMPLETE'}
    sleeps = []
    monkeypatch.setattr(original, 'worker', worker)
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: sleeps.append(seconds))
    assert scheduler.run_worker('fixed_config', 2, tmp_path)['status'] == 'COMPLETE'
    assert len(calls) == 2 and sleeps == [5]
    assert read_json(tmp_path / 'scheduler/worker_gpu2.json')['status'] == 'WAITING_SHARED_RESOURCE'


@pytest.mark.parametrize('exception', [RuntimeError('real training fault'),
                                      resource.ResourceBlockedError('non-launch fault')])
def test_worker_does_not_retry_training_fault_or_nonlaunch_resource_fault(tmp_path, monkeypatch, exception):
    def worker(*args): raise exception
    monkeypatch.setattr(original, 'worker', worker)
    with pytest.raises(type(exception)):
        scheduler.run_worker('fixed_config', 2, tmp_path)


@pytest.mark.parametrize('tamper', ['policy', 'runtime_source', 'review', 'config'])
def test_freeze_rejects_tampering_before_any_worker_start(tmp_path, policy, monkeypatch, tamper):
    from tools.sonar_fcos_quality_v2 import campaign
    from tools.sonar_fcos_pilot.common import sha256_file
    root = tmp_path / 'root'
    root.mkdir()
    output = root / 'results'
    config_path, policy_path = root / 'config.json', root / 'policy.json'
    freeze_path, review_path = root / 'freeze.json', root / 'review.json'
    source = {'original': 'fixed-v2'}
    config = {'output_dir': str(output), 'variants': ['AAA', 'QFL_DFL'], 'seeds': [42, 43, 44],
              'epochs': 30, 'max_formal_runs': 6}
    write_json(config_path, config)
    write_json(policy_path, policy)
    write_json(output / 'execution_manifest.json', {'source_binding': source})
    sources = {}
    for relative in scheduler.RUNTIME_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen synthetic runtime\n')
        sources[relative] = sha256_file(path)
    manifest = {'schema_version': 1, 'policy_sha256': sha256_file(policy_path),
                'config_sha256': sha256_file(config_path), 'source_binding_sha256': original.canonical_sha(source),
                'runtime_sources': sources}
    write_json(freeze_path, manifest)
    review = {'status': 'PASS', 'policy_sha256': manifest['policy_sha256'],
              'runtime_source_set_sha256': original.canonical_sha(sources),
              'freeze_manifest_sha256': sha256_file(freeze_path)}
    write_json(review_path, review)
    monkeypatch.setattr(scheduler, 'ROOT', root)
    monkeypatch.setattr(campaign, 'load_config', lambda path: config)
    monkeypatch.setattr(campaign, 'source_binding', lambda cfg: source)
    assert scheduler.verify_freeze(config_path, policy_path, freeze_path, review_path)[2] == source
    if tamper == 'policy':
        changed = copy.deepcopy(policy)
        changed['registered_at'] = 'changed after review'
        write_json(policy_path, changed)
    elif tamper == 'runtime_source':
        (root / scheduler.RUNTIME_FILES[0]).write_text('# modified runtime\n')
    elif tamper == 'review':
        review['freeze_manifest_sha256'] = 'wrong'
        write_json(review_path, review)
    else:
        write_json(config_path, {**config, 'epochs': 31})
    with pytest.raises(ValueError): scheduler.verify_freeze(config_path, policy_path, freeze_path, review_path)
