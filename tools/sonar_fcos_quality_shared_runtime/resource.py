"""Exact resident allowlist and conservative shared-GPU resource observations.

Per-PID SM counters are unavailable on the deployed GPUs. Resident CPU changes
are an activity proxy, not proof that a resident has no asynchronous GPU work.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import time


class ResourceBlockedError(RuntimeError):
    """The frozen v2 worker turns observation errors into cooperative pauses."""


class LaunchResourceBlockedError(ResourceBlockedError):
    """A fresh resource veto before Popen has created any train child."""


def validate_policy(policy):
    if policy.get('schema_version') != 1:
        raise ValueError('Unsupported shared resource policy version')
    if policy.get('allowed_gpu_indices') != [0, 2, 3] or policy.get('reserved_gpu_indices') != [1]:
        raise ValueError('GPU 1 is reserved and the allowed GPU set is fixed')
    if policy.get('shared_gpu_indices') != [2, 3]:
        raise ValueError('Only GPUs 2 and 3 may share with registered residents')
    if policy.get('pair_gpu_mapping') != {'42': 0, '43': 2, '44': 3}:
        raise ValueError('The seed-pair GPU mapping is fixed')
    if not isinstance(policy.get('boot_id'), str) or not policy['boot_id']:
        raise ValueError('A boot identity is required')
    residents = policy.get('resident_allowlist', [])
    if len(residents) != 2 or sorted(row['gpu_index'] for row in residents) != [2, 3]:
        raise ValueError('Exactly one registered resident per shared card is required')
    for row in residents:
        if row['gpu_uuid'] != policy['gpu_uuids'][str(row['gpu_index'])]:
            raise ValueError('Resident GPU UUID differs from the registered physical card')
        if any(type(row[key]) is not int or row[key] < 0 for key in ('pid', 'uid', 'start_ticks')):
            raise ValueError('Invalid resident process identity')
        if row['pid'] == 0 or not row['comm'] or not row['process_path'].startswith('/'):
            raise ValueError('Complete resident process identity is required')
        if type(row.get('approved_memory_mib')) is not int or not 0 < row['approved_memory_mib'] <= 2048:
            raise ValueError('A fixed approved per-process GPU memory baseline is required')
    guard = policy['guard']
    limits = {'startup_max_utilization_percent': (0, 5), 'min_free_memory_mib': (12288, None),
              'max_resident_memory_mib': (1, 2048), 'max_resident_memory_growth_mib': (0, 64),
              'cpu_busy_core_fraction': (0.000001, 0.02), 'cpu_min_sample_seconds': (5, 60)}
    for key, (minimum, maximum) in limits.items():
        value = guard[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid resource guard: ' + key)
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError('Resource guard exceeds reviewed limits: ' + key)
    if guard['shared_worker_nice'] != 10 or guard['per_pid_sm_supported'] is not False:
        raise ValueError('Shared workers require nice 10 and explicit unavailable SM counters')
    return policy


def read_process_identity(pid, proc_root=Path('/proc')):
    """Read only identity and CPU counters, never command arguments/environment."""
    proc_root = Path(proc_root)
    raw = (proc_root / str(pid) / 'stat').read_text()
    left, right = raw.find('('), raw.rfind(')')
    if left < 0 or right <= left or int(raw[:left].strip()) != pid:
        raise ValueError('Malformed process stat identity')
    fields = raw[right + 1:].split()
    uid_lines = [line.split()[1:] for line in (proc_root / str(pid) / 'status').read_text().splitlines()
                 if line.startswith('Uid:')]
    if len(uid_lines) != 1 or len(uid_lines[0]) != 4:
        raise ValueError('Missing process UID identity')
    uids = [int(value) for value in uid_lines[0]]
    if len(set(uids)) != 1:
        raise ValueError('Resident real/effective/saved/filesystem UIDs differ')
    return {'pid': pid, 'uid': uids[0], 'comm': raw[left + 1:right],
            'start_ticks': int(fields[19]), 'cpu_ticks': int(fields[11]) + int(fields[12]),
            'boot_id': (proc_root / 'sys/kernel/random/boot_id').read_text().strip()}


def compute_apps():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_path, memory = [value.strip() for value in line.split(',')]
        key = (uuid, int(pid))
        if key in rows:
            raise ValueError('Duplicate GPU process observation')
        rows[key] = {'process_path': process_path, 'memory_mib': int(memory)}
    return rows


class SharedResources:
    def __init__(self, policy, *, identity_reader=None, apps_reader=None, monotonic=None,
                 clock_ticks=None, record=None):
        self.policy = validate_policy(policy)
        self.identity_reader = identity_reader or read_process_identity
        self.apps_reader = apps_reader or compute_apps
        self.monotonic = monotonic or time.monotonic
        self.clock_ticks = clock_ticks if clock_ticks is not None else os.sysconf('SC_CLK_TCK')
        if self.clock_ticks <= 0:
            raise ValueError('CPU clock frequency must be positive')
        self.record = record or (lambda value: None)
        self.cpu_samples = {}
        self.last = {}

    def _publish(self, gpu, ready, reason, **extra):
        value = {'physical_gpu': gpu, 'ready': ready, 'reason': reason,
                 'activity_observation': 'resident_cpu_proxy', 'per_pid_sm_supported': False,
                 'gpu_service_idle_verified': False, **extra}
        self.last[gpu] = value
        self.record(value)
        return ready

    def observe(self, cards, gpu, own_pids=(), *, startup=False):
        """True only after complete fresh identity/memory and a CPU sample pair."""
        if gpu not in (2, 3):
            raise ValueError('Shared observations are restricted to GPUs 2 and 3')
        try:
            card = next((row for row in cards if row['index'] == gpu), None)
            if card is None:
                return self._publish(gpu, False, 'gpu_disappeared')
            resident = next(row for row in self.policy['resident_allowlist'] if row['gpu_index'] == gpu)
            if card['uuid'] != resident['gpu_uuid'] or card['name'] != 'NVIDIA GeForce RTX 5090':
                return self._publish(gpu, False, 'gpu_identity_changed')
            registered = {resident['pid']}
            unknown = set(card['compute_pids']) - set(own_pids) - registered
            if unknown:
                return self._publish(gpu, False, 'unknown_compute_pid', unknown_pids=sorted(unknown))
            if not registered <= set(card['compute_pids']):
                return self._publish(gpu, False, 'registered_resident_missing')
            apps = self.apps_reader()
            identity = self.identity_reader(resident['pid'])
            if identity['boot_id'] != self.policy['boot_id'] or any(
                    identity[key] != resident[key] for key in ('pid', 'uid', 'comm', 'start_ticks')):
                return self._publish(gpu, False, 'resident_identity_changed')
            app = apps[(card['uuid'], resident['pid'])]
            if app['process_path'] != resident['process_path']:
                return self._publish(gpu, False, 'resident_process_path_changed')
            # Both NVIDIA queries must describe the same set for this GPU.
            app_pids = {pid for (uuid, pid) in apps if uuid == card['uuid']}
            if app_pids != set(card['compute_pids']):
                return self._publish(gpu, False, 'inconsistent_compute_observation')
            memory = app['memory_mib']
            if type(memory) is not int or memory < 0:
                raise ValueError('Invalid resident memory observation')
            baseline = resident['approved_memory_mib']
            guard = self.policy['guard']
            if memory > guard['max_resident_memory_mib'] or memory - baseline > guard['max_resident_memory_growth_mib']:
                return self._publish(gpu, False, 'resident_memory_increased', resident_memory_mib=memory,
                                     resident_memory_baseline_mib=baseline)
            if card['memory_total_mib'] - card['memory_used_mib'] < guard['min_free_memory_mib']:
                return self._publish(gpu, False, 'insufficient_free_gpu_memory')
            if startup and card['utilization_percent'] > guard['startup_max_utilization_percent']:
                return self._publish(gpu, False, 'startup_gpu_busy')
            now = self.monotonic()
            ticks = identity['cpu_ticks']
            previous = self.cpu_samples.get(gpu)
            if previous is None:
                self.cpu_samples[gpu] = (now, ticks, None)
                return self._publish(gpu, False, 'cpu_second_sample_required')
            before_time, before_ticks, previous_fraction = previous
            elapsed = now - before_time
            if elapsed < 0 or ticks < before_ticks:
                return self._publish(gpu, False, 'cpu_counter_or_clock_reset')
            if elapsed > 60:
                self.cpu_samples[gpu] = (now, ticks, None)
                return self._publish(gpu, False, 'cpu_sample_stale')
            if elapsed < guard['cpu_min_sample_seconds']:
                if previous_fraction is None:
                    return self._publish(gpu, False, 'cpu_sample_interval_short')
                # Rechecks before Popen still read fresh CPU counters. A new
                # activity burst may veto admission before the next full window.
                recent_fraction = ((ticks - before_ticks) / self.clock_ticks / elapsed
                                   if elapsed > 0 else 0.0 if ticks == before_ticks else float('inf'))
                fraction = max(previous_fraction, recent_fraction)
            else:
                fraction = (ticks - before_ticks) / self.clock_ticks / elapsed
                self.cpu_samples[gpu] = (now, ticks, fraction)
            ready = fraction < guard['cpu_busy_core_fraction']
            return self._publish(gpu, ready, 'shared_gpu_ready' if ready else 'resident_cpu_activity',
                                 cpu_core_fraction=fraction, resident_memory_mib=memory,
                                 resident_memory_baseline_mib=baseline)
        except Exception as exc:
            return self._publish(gpu, False, 'resource_observation_failed', error_type=type(exc).__name__)

    def select_gpu(self, cards, scheduling, original_select):
        for gpu in (scheduling or {}).get('gpu_indices', [0]):
            if gpu == 0:
                selected = original_select(cards, {'gpu_indices': [0]})
                if selected is not None:
                    return selected
            elif gpu in (2, 3) and self.observe(cards, gpu, startup=True):
                return next(card for card in cards if card['index'] == gpu)
        return None

    def foreign_pids(self, cards, gpu, own_pids, original_foreign):
        if gpu == 0:
            return original_foreign(cards, gpu, own_pids)
        if gpu not in (2, 3):
            raise ResourceBlockedError('Unapproved physical GPU')
        if not self.observe(cards, gpu, own_pids, startup=False):
            raise ResourceBlockedError(self.last[gpu]['reason'])
        return []

    def child_environment(self, card, pause, original_environment):
        if card['index'] == 0:
            return original_environment(card, pause)
        if card['index'] not in (2, 3) or not self.observe([card], card['index'], startup=True):
            raise LaunchResourceBlockedError('Shared launch resource check failed')
        return dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    SONAR_FCOS_QUALITY_PAUSE_FILE=str(pause), PYTHONUNBUFFERED='1')
