"""One scoped reconciliation after the recorded cpu_freeze03 timer-substate failure.

This never starts/stops services, writes IRQ/CPU controls, resets hardware, signals
processes, retries restoration, edits original evidence, or releases the global
reservation. Only the exact owned system lock may be archived after all original
conditions are observed again. Import is inert.
"""
from __future__ import annotations
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

BOOT = 'da641785-1120-499a-8b22-0d89bfc73687'
MACHINE = 'bd081b0be1fd457093e07237b0dee576'
IDENTITY = dict(machine_id=MACHINE, boot_id=BOOT)
ROOT = '/home/xilinx/d09_cpu_freeze_20261013'
EVIDENCE = ROOT + '/system_conditions/cpu_freeze03'
OUTPUT = ROOT + '/system_conditions_reconciliation/cpu_freeze03_timer01'
SYSTEM_LOCK = '/home/xilinx/hardware_measurement_pilot_20261012/system_conditions.lock'
SYSTEM_SHA = '4ffd5a92cd08ea0e6867041c245b1a165ee8956918f17abc4fe71274a257169d'
LEASE = '/home/xilinx/.hwnas_board_ownership/' + MACHINE + '.json'
LEASE_SHA = '26616beb235529c5157e158da430919cde34ffca3e038006427e1749eb7c581e'
SYSTEM_SOURCE = '2c9980998b90b19bb834d5f6a5509a99933899a0bbaa59327161d8451581b8a9'
GUARD_SOURCE = '78a181010ed09c70189a49a841e5c53bd252b2c96df68a5ef09437464807a01d'
OBSERVATION_SHA = '8b08cfeaceebee43e572a60b884a4a23c5126696f2df94a1e53fb66a96a497ad'
PINS = {
    'state.json': '36491a38357eed049e4e3823791dfa34b0fa49bfc22054705e217b62512b8a5a',
    'latest_check.json': 'add3e4dc5b19a0ebb6ea52b7abd88847a37ffb5043153dd81c5d715f63794c66',
    'restore.json': '98ff66b3af7c206cd8f264f6db780432c9240de2281697f5bbbb422ddf1295bb',
    'usage_before.json': '33fb950d3616751fa723ec8289d9364c60a95368352816074f9b62ad431cca9e',
    'events.jsonl': '3ad290c5a154b15ed0a0e950294a79be378b8702521784421c848ddfaf353141',
    'original.json': '6bd2f34e17c04766495cf99b16d657d65ff7da4ddc3876a94eded4ae8efb9536',
    'config.json': 'a902953b45321cec6deb1b00579b9cfe9c122df9441fa1ac303338b74fea7f8f',
}
OWNERSHIP = dict(schema='d09_board_ownership_config_v1', campaign_id='d09_cpu_freeze_20261013_quiet_v3',
                 owner_id='d09_quiet_v3_da641785', expected_machine_id=MACHINE, expected_boot_id=BOOT,
                 reservation_root='/home/xilinx/.hwnas_board_ownership',
                 known_board_roots=['/home/xilinx/hardware_measurement_pilot_20261012', '/home/xilinx/hardware_template_coverage_20261013'],
                 reservation=dict(path=LEASE, sha256=LEASE_SHA),
                 system_conditions_lock=dict(path=SYSTEM_LOCK, sha256=SYSTEM_SHA, condition_id='d09_cpu_freeze03'))
FAILURE = dict(operation=dict(id='apt-daily-upgrade.timer', kind='service'),
               error_type='ValueError', error='SERVICE_NOT_EXACTLY_RESTORED')

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

class ReconciliationError(RuntimeError):
    pass

def require(condition, reason):
    if not condition:
        raise ReconciliationError(reason)

def regular(path):
    path = Path(path)
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'SYMLINK_IN_EVIDENCE_PATH')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), 'NONREGULAR_EVIDENCE')
    raw = path.read_bytes()
    after = path.stat()
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    require(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'EVIDENCE_CHANGED_DURING_READ')
    return raw, identity

class Backend:
    def __init__(self):
        namespaces = []
        for expected in [SYSTEM_SOURCE, GUARD_SOURCE]:
            file = Path(ROOT) / 'helpers' / (expected + '.py')
            raw, _ = regular(file)
            require(digest(raw) == expected, 'BOUND_READ_HELPER_SOURCE_CHANGED')
            namespace = {'__name__': 'd09_readonly_reconciliation_dependency', '__file__': str(file)}
            exec(compile(raw, str(file), 'exec'), namespace)
            namespaces.append(namespace)
        self.system, self.guard = namespaces
        self.linux = self.system['LinuxBackend']()
        self.controller = self.system['SystemConditions'](backend=self.linux)
    def path(self, logical):
        return Path(logical)
    def identity(self):
        return self.linux.identity()
    def conditions(self):
        return self.controller._snapshot()
    def ownership(self, system_present):
        config = dict(OWNERSHIP)
        if not system_present:
            config.pop('system_conditions_lock')
        return self.guard['check'](config, require_system_conditions=system_present)
    def apt_idle(self):
        services = {name: self.linux.service(name)['ActiveState'] for name in ['apt-daily.service', 'apt-daily-upgrade.service']}
        processes = []
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():
                continue
            try:
                name = (folder / 'comm').read_text().strip()
            except FileNotFoundError:
                continue
            if name in ('apt', 'apt-get', 'aptitude', 'dpkg', 'dpkg-deb', 'dpkg-preconfigu'):
                processes.append(dict(pid=int(folder.name), name=name))
        return dict(services=services, processes=processes)
    def sync_dir(self, path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    def rename_no_replace(self, source, target):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

def save(path, value, backend):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    backend.sync_dir(path.parent)

def reconcile(config, *, backend=None):
    require(config.get('schema') == 'd09_timer_restore_reconciliation_config_v1' and
            config.get('preserve_prior_failure') is True and config.get('authorized_scope') == 'archive_exact_owned_system_lock_after_natural_timer_completion',
            'INVALID_OR_UNAUTHORIZED_RECONCILIATION_CONFIG')
    require(config.get('pre_reconciliation_observation_sha256') == OBSERVATION_SHA, 'PRE_RECONCILIATION_OBSERVATION_BINDING_CHANGED')
    require(digest(Path(__file__).read_bytes()) == config.get('expected_helper_sha256'), 'RECONCILIATION_HELPER_SOURCE_CHANGED')
    backend = backend or Backend()
    require(backend.identity() == IDENTITY, 'BOOT_OR_MACHINE_CHANGED')
    frozen = {}
    for name, expected in PINS.items():
        logical = EVIDENCE + '/' + name
        raw, identity = regular(backend.path(logical))
        require(digest(raw) == expected, 'PRIOR_FAILURE_EVIDENCE_CHANGED:' + name)
        frozen[logical] = (raw, identity)
    for logical, expected in [(SYSTEM_LOCK, SYSTEM_SHA), (LEASE, LEASE_SHA)]:
        raw, identity = regular(backend.path(logical))
        require(digest(raw) == expected, 'OWNED_LOCK_OR_RESERVATION_CHANGED')
        frozen[logical] = (raw, identity)
    def old(name):
        return json.loads(frozen[EVIDENCE + '/' + name][0])
    original, state, prior, condition = old('original.json'), old('state.json'), old('restore.json'), old('config.json')
    require(condition == dict(condition_id='d09_cpu_freeze03', expected_boot_id=BOOT,
                              expected_machine_id=MACHINE, schema='d09_system_conditions_config_v1'), 'CONDITION_IDENTITY_CHANGED')
    require(state['status'] == 'FAIL_RESTORE' and state['restore_attempted'] is True and
            state['lock_binding'] == json.loads(frozen[SYSTEM_LOCK][0]) and
            state['original_sha256'] == PINS['original.json'] and state['config_sha256'] == PINS['config.json'], 'FAILED_STATE_BINDING_INVALID')
    require(prior['status'] == 'FAIL_RESTORE' and prior['lock_retained'] is True and prior['boot_id'] == BOOT and
            prior['condition_id'] == 'd09_cpu_freeze03' and prior['original'] == original and
            prior['failures'] == [FAILURE, dict(error='FINAL_STATE_DIFFERS_FROM_ORIGINAL')] and
            prior['enabled_state_changed'] is False and prior['automatic_retry_permitted'] is False, 'PRIOR_FAILURE_NOT_EXACT_TIMER_CASE')
    transient = json.loads(json.dumps(original))
    require(transient['services']['apt-daily-upgrade.timer']['SubState'] == 'waiting', 'ORIGINAL_TIMER_STATE_INVALID')
    transient['services']['apt-daily-upgrade.timer']['SubState'] = 'running'
    require(prior['restored'] == transient, 'PRIOR_FAILURE_HAS_OTHER_CONDITION_DIFFERENCE')
    events = [json.loads(line) for line in frozen[EVIDENCE + '/events.jsonl'][0].decode().splitlines() if line]
    restoration = [event for event in events if event['event'].startswith('RESTORE_')]
    expected = []
    require(len(state['attempted_operations']) == 12, 'RESTORE_OPERATION_COUNT_CHANGED')
    for operation in reversed(state['attempted_operations']):
        expected.append(('RESTORE_INTENT', operation))
        expected.append(('RESTORE_STEP_FAILED' if operation == FAILURE['operation'] else 'RESTORE_VERIFIED', operation))
    actual = [(event['event'], event.get('operation', event.get('failure', {}).get('operation'))) for event in restoration]
    require(actual == expected and [event['failure'] for event in restoration if event['event'] == 'RESTORE_STEP_FAILED'] == [FAILURE],
            'RESTORE_EVENT_SEQUENCE_CHANGED')
    observations = []
    def current(system_present):
        require(backend.identity() == IDENTITY, 'BOOT_OR_MACHINE_CHANGED')
        owned = backend.ownership(system_present)
        require(owned.get('ok') is True and owned.get('status') == 'PASS_OWNED_NO_KNOWN_CONFLICT' and
                owned.get('identity') == IDENTITY and owned.get('reservation') == OWNERSHIP['reservation'] and
                owned.get('occupancy', {}).get('conflicts') == [] and owned.get('occupancy', {}).get('active_processes') == [],
                'OWNERSHIP_OR_KNOWN_WORKER_CONFLICT')
        snapshot = backend.conditions()
        require(snapshot == original, 'CURRENT_SERVICE_OR_IRQ_NOT_EXACTLY_ORIGINAL')
        apt = backend.apt_idle()
        require(set(apt['services']) == {'apt-daily.service', 'apt-daily-upgrade.service'} and
                all(value in ('inactive', 'failed') for value in apt['services'].values()) and apt['processes'] == [],
                'APT_WORK_NOT_FINISHED_NO_INTERVENTION_PERMITTED')
        require(backend.identity() == IDENTITY, 'BOOT_OR_MACHINE_CHANGED')
        item = dict(observed_utc=utc(), conditions=snapshot, ownership=owned, apt=apt)
        observations.append(item)
        return item
    def unchanged(include_system=True):
        for logical, expected in frozen.items():
            if logical == SYSTEM_LOCK and not include_system:
                continue
            require(regular(backend.path(logical)) == expected, 'FROZEN_EVIDENCE_OR_LOCK_CHANGED_DURING_RECONCILIATION')
    current(True)
    unchanged()
    output = backend.path(OUTPUT)
    require(not output.exists() and not output.is_symlink(), 'RECONCILIATION_OUTPUT_ALREADY_EXISTS')
    require(not any(path.is_symlink() for path in output.parents), 'RECONCILIATION_OUTPUT_PARENT_SYMLINK')
    output.mkdir(parents=True, exist_ok=False)
    backend.sync_dir(output.parent)
    archive = output / 'archived_system_conditions_lock.json'
    moved = False
    base = dict(schema='d09_timer_restore_reconciliation_receipt_v1', created_utc=utc(), identity=IDENTITY,
                helper_sha256=config['expected_helper_sha256'], input_hashes=PINS,
                pre_reconciliation_observation_sha256=OBSERVATION_SHA,
                prior_restore_status='FAIL_RESTORE', original=original, original_failure_preserved=True,
                archived_system_lock=str(archive), archived_system_lock_sha256=SYSTEM_SHA,
                global_reservation=OWNERSHIP['reservation'], global_reservation_retained=True,
                services_changed=False, irqs_changed=False, cpu_controls_changed=False,
                restoration_retried=False, board_ready_for_measurement=False,
                scope='Known-root cooperative ownership; not exclusion of unknown legacy programs')
    try:
        save(output / 'prepared.json', dict(base, status='PREPARED_NOT_COMMITTED', observations=observations), backend)
        current(True)
        unchanged()
        require(backend.identity() == IDENTITY, 'BOOT_OR_MACHINE_CHANGED')
        backend.rename_no_replace(backend.path(SYSTEM_LOCK), archive)
        moved = True
        backend.sync_dir(archive.parent)
        backend.sync_dir(backend.path(SYSTEM_LOCK).parent)
        require(regular(archive) == frozen[SYSTEM_LOCK] and not backend.path(SYSTEM_LOCK).exists(), 'ARCHIVED_SYSTEM_LOCK_POSTCONDITION_FAILED')
        unchanged(False)
        final = current(False)
        unchanged(False)
        result = dict(base, ok=True, status='PASS_CURRENT_ORIGINAL_CONDITIONS_RECONCILED_WITH_PRIOR_FAILURE',
                      finished_utc=utc(), system_lock_archived=True, system_lock_retained=False,
                      current=final['conditions'], observations=observations)
        save(output / 'reconciliation.json', result, backend)
        return result
    except BaseException as error:
        result = dict(base, ok=False, status=('FAIL_RECONCILIATION_SYSTEM_LOCK_ARCHIVED_RESERVATION_RETAINED' if moved else
                                            'FAIL_RECONCILIATION_SYSTEM_LOCK_RETAINED'), finished_utc=utc(),
                      error_type=type(error).__name__, error=str(error), system_lock_archived=moved,
                      system_lock_retained=not moved, observations=observations)
        try:
            save(output / 'failure.json', result, backend)
        except BaseException:
            pass
        return result
