"""External resource amendment for the frozen operator probe training runner.

Only the old runner's physical_gpu/gpu_admission functions are replaced. Model,
data, loss, optimizer, augmentation, checkpoint and evaluation code stay in the
original frozen tree. GPU0 and other users' processes are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import time
import types
import uuid

RESOURCE_FUNCTIONS = {'physical_gpu', 'gpu_admission'}
MANIFEST_FILES = ('execution_manifest.json', 'environment_lock.json', 'smoke_gate.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def process_identity(pid, proc_root=Path('/proc')):
    """Read public identity only; never save command arguments or credentials."""
    directory = Path(proc_root) / str(int(pid))
    try:
        tail = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(tail[19])
        status = (directory / 'status').read_text()
        uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('Uid:')))
        cmdline = (directory / 'cmdline').read_bytes()
        btime = float(next(line.split()[1] for line in (Path(proc_root) / 'stat').read_text().splitlines()
                           if line.startswith('btime ')))
        return dict(pid=int(pid), start_ticks=ticks, uid=uid,
            cmdline_sha256=hashlib.sha256(cmdline).hexdigest(),
            executable=Path(cmdline.split(b'\0')[0].decode(errors='replace')).name,
            started_at_unix=btime + ticks / os.sysconf('SC_CLK_TCK'), state=tail[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected, actual):
    return actual is not None and actual.get('state') != 'Z' and all(
        actual.get(key) == expected.get(key) for key in ('pid', 'start_ticks', 'uid', 'cmdline_sha256'))


def gpu_rules(amendment):
    supplied = amendment['gpus']
    values = [dict(value, index=int(index)) for index, value in supplied.items()] if isinstance(supplied, dict) else supplied
    rules = {}
    for entry in values:
        index = int(entry['index'])
        rule = dict(entry, index=index,
            min_free_mib=float(entry.get('min_free_mib', 12288)),
            max_start_util=float(entry.get('max_start_util', 15)),
            residents=entry.get('residents', []))
        if (index not in (1, 2, 3) or index in rules or not rule.get('uuid')
                or rule['mode'] != ('exclusive' if index == 1 else 'shared')
                or not math.isfinite(rule['min_free_mib']) or rule['min_free_mib'] < 12288
                or not 0 <= rule['max_start_util'] <= 15
                or (index == 1 and rule['residents'])
                or (index in (2, 3) and not rule['residents'])):
            raise ValueError('GPU amendment must explicitly bind exclusive1/shared2/shared3 with 12GiB/15% limits')
        for resident in rule['residents']:
            if (resident.get('pid') not in (5300, 5406)
                    or any(key not in resident for key in ('pid', 'start_ticks', 'uid', 'cmdline_sha256'))):
                raise ValueError('Shared GPUs require the exact original resident speech process identities')
        rules[index] = rule
    if set(rules) != {1, 2, 3}:
        raise ValueError('This amendment authorizes exactly GPU1/2/3; GPU0 remains unavailable')
    return rules


def gpu_snapshot(index):
    output = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    fields = [value.strip() for value in output.split(',')]
    if len(fields) != 5:
        raise ValueError('Malformed nvidia-smi GPU snapshot')
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    pids = []
    for line in owners.splitlines():
        values = [value.strip() for value in line.split(',')]
        if len(values) == 2 and values[0] == fields[0]:
            pids.append(int(values[1]))
    return dict(index=index, uuid=fields[0], name=fields[1], total_mib=float(fields[2]),
                free_mib=float(fields[3]), utilization=float(fields[4]), compute_pids=sorted(set(pids)))


def admission(amendment, index, resource_error=RuntimeError, *, own_pid=None):
    rules = gpu_rules(amendment)
    if index not in rules:
        raise ValueError('Physical GPU is not authorized by the explicit resource amendment')
    rule = rules[index]
    snapshot = gpu_snapshot(index)
    if snapshot['uuid'] != rule['uuid']:
        raise ValueError('Physical GPU UUID drift')
    own = os.getpid() if own_pid is None else int(own_pid)
    residents = {entry['pid'] for entry in rule['residents']}
    actual_residents = []
    for expected in rule['residents']:
        actual = process_identity(expected['pid'])
        if not same_process(expected, actual) or expected['pid'] not in snapshot['compute_pids']:
            raise ValueError('The original resident speech service disappeared or changed identity')
        actual_residents.append(actual)
    other = set(snapshot['compute_pids']) - residents - {own}
    if (other or snapshot['free_mib'] < rule['min_free_mib']
            or snapshot['utilization'] > rule['max_start_util']):
        raise resource_error(f'GPU{index} unavailable: free={snapshot["free_mib"]}MiB '
            f'util={snapshot["utilization"]}% other_compute_pids={sorted(other)}')
    return dict(snapshot, mode=rule['mode'], residents=actual_residents, admitted=True,
                other_compute_pids=[], physical_gpu=index, recorded_at=now(),
                performance_claim_scope='training_resource_only_not_hardware_latency_or_power')


def _constant_identity(value):
    """Canonical compiler constants, independent of interning/object references."""
    if isinstance(value, types.CodeType):
        return dict(type='code', value=_code_identity(value))
    if value is None or value is Ellipsis:
        return dict(type='none' if value is None else 'ellipsis')
    if type(value) in (bool, int, str):
        return dict(type=type(value).__name__, value=value)
    if type(value) is bytes:
        return dict(type='bytes', value=value.hex())
    if type(value) is float:
        return dict(type='float64', value=struct.pack('!d', value).hex())
    if type(value) is complex:
        return dict(type='complex128', real=struct.pack('!d', value.real).hex(),
                    imaginary=struct.pack('!d', value.imag).hex())
    if type(value) is tuple:
        return dict(type='tuple', value=[_constant_identity(item) for item in value])
    if type(value) is frozenset:
        items = [_constant_identity(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True, separators=(',', ':')))
        return dict(type='frozenset', value=items)
    raise TypeError(f'Unsupported compiler constant in immutable code identity: {type(value).__name__}')


def _code_identity(code):
    # marshal encodes interning/reference flags, whose bytes may change when an
    # unrelated function releases shared constant references (observed in3.13).
    # Structural fields preserve executable identity without that object graph.
    return dict(schema='python_structural_code_v1', name=code.co_name,
        qualname=getattr(code, 'co_qualname', code.co_name), filename=code.co_filename,
        firstlineno=code.co_firstlineno, argcount=code.co_argcount,
        posonlyargcount=code.co_posonlyargcount, kwonlyargcount=code.co_kwonlyargcount,
        nlocals=code.co_nlocals, stacksize=code.co_stacksize, flags=code.co_flags,
        bytecode=code.co_code.hex(), constants=[_constant_identity(value) for value in code.co_consts],
        names=list(code.co_names), varnames=list(code.co_varnames), freevars=list(code.co_freevars),
        cellvars=list(code.co_cellvars), linetable=(code.co_linetable if hasattr(code, 'co_linetable') else code.co_lnotab).hex(),
        exceptiontable=getattr(code, 'co_exceptiontable', b'').hex())


def function_hashes(module, excluded=()):
    return {name: hashlib.sha256(json.dumps(_code_identity(value.__code__), sort_keys=True,
            ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest()
            for name, value in module.__dict__.items()
            if isinstance(value, types.FunctionType) and value.__module__ == module.__name__ and name not in excluded}


def load_base(base_root):
    base_root = Path(base_root).resolve()
    sys.path[:0] = [str(base_root), str(base_root / 'src'), str(base_root / 'scripts')]
    old = importlib.import_module('tools.sonar_operator_probe.train')
    model = importlib.import_module('tools.sonar_operator_probe.model')
    if (Path(old.__file__).resolve() != base_root / 'tools/sonar_operator_probe/train.py'
            or Path(model.__file__).resolve() != base_root / 'tools/sonar_operator_probe/model.py'
            or old.ROOT.resolve() != base_root):
        raise ValueError('Training/model imports did not resolve to the specified frozen base tree')
    return old, model


def load_amendment(path, old, base_output, config):
    value = read(path)
    manifest = read(Path(base_output) / 'execution_manifest.json')
    digest = value.get('base_manifest_sha256', value.get('base_execution_manifest_sha256'))
    authorization = value.get('authorization', value.get('user_authorization'))
    authorization = authorization.get('request') if isinstance(authorization, dict) else authorization
    if (digest != sha(Path(base_output) / 'execution_manifest.json')
            or not isinstance(authorization, str) or not authorization.strip()
            or value.get('maximum_gpu_hours', 48) != 48
            or value.get('formal_eligible', False) is not False
            or value.get('retry', 'none') != 'none'
            or value.get('training_contract_unchanged', True) is not True):
        raise ValueError('Resource amendment requires explicit authorization and the exact original scientific manifest')
    if value.get('wrapper_source_sha256') not in (None, sha(__file__)):
        raise ValueError('External resource wrapper source SHA256 drift')
    gpu_rules(value)
    cfg, baseline = old.verify_manifest(Path(base_output), Path(config))
    if sha(old.__file__) != manifest['source_sha256']['tools/sonar_operator_probe/train.py']:
        raise ValueError('Original training file bytes differ from the frozen execution manifest')
    prior = float(value['prior_gpu_seconds'])
    smoke = float(read(Path(base_output) / 'smoke_gate.json')['smoke_gpu_seconds'])
    if not math.isfinite(prior) or prior < smoke or value.get('prior_gpu_seconds_excludes_legacy_active', True) is not True:
        raise ValueError('GPU accounting must carry original smoke/completed workers and count active workers separately')
    for entry in value.get('legacy_active', []):
        if (entry['variant'] not in cfg['variants'] or entry['seed'] != 42
                or any(key not in entry for key in ('pid', 'start_ticks', 'uid', 'cmdline_sha256', 'started_at_unix'))):
            raise ValueError('Legacy active worker must retain its exact registered identity and original seed42 job')
    return value, cfg, baseline


def _copy_exact(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f'Existing imported freeze differs: {target.name}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(Path(root).rglob('*')) if path.is_file()}


def prepare(old, model, base_output, output, config, amendment_path):
    base_output, output = Path(base_output).resolve(), Path(output).resolve()
    amendment, cfg, _ = load_amendment(amendment_path, old, base_output, config)
    if output == base_output or not output.is_relative_to(old.ROOT / 'results'):
        raise ValueError('Parallel output must be a new isolated namespace inside original base/results')
    output.mkdir(parents=True, exist_ok=True)
    for name in MANIFEST_FILES:
        _copy_exact(base_output / name, output / name)
    source_smoke, destination = base_output / 'smoke', output / 'smoke'
    if destination.exists():
        if tree_hashes(source_smoke) != tree_hashes(destination):
            raise ValueError('Imported full smoke tree differs')
    else:
        shutil.copytree(source_smoke, destination)
    # These are original-GPU1 preflight receipts, not a new shared-GPU approval.
    for name in ('preflight_supervisor_status.json',):
        if (base_output / name).is_file():
            _copy_exact(base_output / name, output / name)
    _copy_exact(amendment_path, output / 'RESOURCE_AMENDMENT.json')
    receipt = dict(schema_version=1, scope='external_training_resource_amendment',
        base_source_root=str(old.ROOT), base_output_root=str(base_output), base_config=str(Path(config).resolve()),
        base_manifest_sha256=sha(base_output / 'execution_manifest.json'),
        base_config_sha256=sha(config), base_source_sha256=read(base_output / 'execution_manifest.json')['source_sha256'],
        imported_freeze_sha256={name: sha(output / name) for name in MANIFEST_FILES},
        imported_smoke_tree_sha256=canonical_sha(tree_hashes(destination)),
        amendment_sha256=sha(output / 'RESOURCE_AMENDMENT.json'), wrapper_source_sha256=sha(__file__),
        original_resource_function_code_sha256={name: function_hashes(old)[name] for name in RESOURCE_FUNCTIONS},
        original_training_function_code_sha256=function_hashes(old, RESOURCE_FUNCTIONS),
        original_model_function_code_sha256=function_hashes(model),
        original_factorial_function_code_sha256=function_hashes(old.t),
        original_protocol_function_code_sha256=function_hashes(old.p),
        original_C01b_function_code_sha256=function_hashes(old.c01b),
        scientific_config_changed=False, original_smoke_gpu=1,
        original_smoke_reused_as_model_gate_only=True,
        shared_gpu_admission_is_separate=True, formal_eligible=False, test_evaluated=False,
        automatic_search=False, automatic_hardware=False)
    path = output / 'parallel_execution_manifest.json'
    if path.exists() and read(path) != receipt:
        raise ValueError('Existing external wrapper/source amendment freeze differs')
    if not path.exists():
        dump(path, receipt)
    old.verify_manifest(output, config)
    old.require_smoke(output, cfg)
    return receipt


def verify_prepared(old, model, base_output, output, config):
    output = Path(output)
    receipt = read(output / 'parallel_execution_manifest.json')
    amendment, cfg, baseline = load_amendment(output / 'RESOURCE_AMENDMENT.json', old, base_output, config)
    if (receipt['wrapper_source_sha256'] != sha(__file__)
            or receipt['amendment_sha256'] != sha(output / 'RESOURCE_AMENDMENT.json')
            or receipt['original_training_function_code_sha256'] != function_hashes(old, RESOURCE_FUNCTIONS)
            or receipt['original_model_function_code_sha256'] != function_hashes(model)
            or receipt['original_factorial_function_code_sha256'] != function_hashes(old.t)
            or receipt['original_protocol_function_code_sha256'] != function_hashes(old.p)
            or receipt['original_C01b_function_code_sha256'] != function_hashes(old.c01b)
            or any(sha(output / name) != receipt['imported_freeze_sha256'][name] for name in MANIFEST_FILES)
            or canonical_sha(tree_hashes(output / 'smoke')) != receipt['imported_smoke_tree_sha256']):
        raise ValueError('Wrapper or frozen training/model/evaluation code identity drift')
    old.verify_manifest(output, config)
    old.require_smoke(output, cfg)
    return amendment, cfg, baseline, receipt


def install_resource_shim(old, amendment, physical_index):
    before = function_hashes(old, RESOURCE_FUNCTIONS)
    rules = gpu_rules(amendment)
    if physical_index not in rules:
        raise ValueError('Worker GPU is outside amended resource authorization')

    def physical_gpu(cfg, device):
        if device == 'cpu':
            return None
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible != str(physical_index):
            raise ValueError('Worker visibility must match its explicitly amended physical GPU')
        return physical_index

    def gpu_admission(cfg, device):
        index = physical_gpu(cfg, device)
        if index is None:
            raise ValueError('Amended experiment workers require a registered CUDA GPU')
        return admission(amendment, index, old.ResourceUnavailable)

    old.physical_gpu, old.gpu_admission = physical_gpu, gpu_admission
    if function_hashes(old, RESOURCE_FUNCTIONS) != before:
        raise ValueError('External resource amendment changed a training function')
    return dict(replaced_functions=sorted(RESOURCE_FUNCTIONS), training_functions_unchanged=True,
                installed_resource_code_sha256=function_hashes(sys.modules[__name__]))


def slot_smoke(old, model, base_output, output, config, physical_index):
    """Execute the unchanged control's two training steps on a newly allowed GPU."""
    if physical_index not in (2, 3):
        raise ValueError('New shared-slot smoke is required only for GPU2/3')
    amendment, cfg, _, receipt = verify_prepared(old, model, base_output, output, config)
    root = Path(output) / 'resource_smoke' / f'gpu_{physical_index}'
    if root.exists():
        raise FileExistsError('Existing resource smoke is preserved; no automatic retry')
    shim = install_resource_shim(old, amendment, physical_index)
    admission_receipt = old.gpu_admission(cfg, 'cuda')
    root.mkdir(parents=True)
    for name in ('execution_manifest.json', 'environment_lock.json'):
        _copy_exact(Path(output) / name, root / name)
    tick = time.monotonic()
    report = old.smoke_one(root, 'control', 42, 'cuda', config)
    reference = read(Path(base_output) / 'smoke' / 'control' / 'seed_42' / 'smoke_report.json')
    old.verify_pairing([reference, dict(report, binding=dict(report['binding'], variant='slot_control'))], smoke=True)
    if (report['initial_state_sha256'] != reference['initial_state_sha256']
            or function_hashes(old, RESOURCE_FUNCTIONS) != receipt['original_training_function_code_sha256']):
        raise ValueError('Shared-slot control initialization or frozen training functions drifted')
    gate = dict(status='PASS', physical_gpu=physical_index, gpu_uuid=admission_receipt['uuid'],
        admission=admission_receipt, amendment_sha256=receipt['amendment_sha256'],
        wrapper_source_sha256=receipt['wrapper_source_sha256'], base_manifest_sha256=receipt['base_manifest_sha256'],
        original_training_function_code_sha256=receipt['original_training_function_code_sha256'],
        original_environment_lock_sha256=sha(Path(output) / 'environment_lock.json'),
        smoke_report_sha256=sha(root / 'smoke' / 'control' / 'seed_42' / 'smoke_report.json'),
        original_gpu1_control_smoke_sha256=sha(Path(base_output) / 'smoke' / 'control' / 'seed_42' / 'smoke_report.json'),
        actual_two_optimizer_steps=True, actual_one_validation_batch=True,
        actual_paired_augmented_data=True, identical_control_initialization=True,
        environment_fingerprint=report['binding']['environment_fingerprint'],
        final_state_sha256=report['final_state_sha256'], smoke_gpu_seconds=time.monotonic() - tick,
        shim=shim, formal_eligible=False, test_evaluated=False,
        scope='shared_training_slot_runtime_smoke_not_new_accuracy_or_hardware_evidence')
    dump(root / 'slot_smoke_gate.json', gate)
    return gate


def require_slot_smoke(old, output, physical_index, receipt):
    if physical_index == 1:
        # Original complete smoke remains the unchanged exclusive-GPU1 gate.
        return dict(physical_gpu=1, source='original_full21_gpu1_smoke', smoke_gpu_seconds=0.)
    root = Path(output) / 'resource_smoke' / f'gpu_{physical_index}'
    gate = read(root / 'slot_smoke_gate.json')
    report_path = root / 'smoke' / 'control' / 'seed_42' / 'smoke_report.json'
    expected_uuid = gpu_rules(read(Path(output) / 'RESOURCE_AMENDMENT.json'))[physical_index]['uuid']
    if (gate.get('status') != 'PASS' or gate.get('physical_gpu') != physical_index
            or gate.get('gpu_uuid') != expected_uuid
            or gate.get('amendment_sha256') != receipt['amendment_sha256']
            or gate.get('wrapper_source_sha256') != receipt['wrapper_source_sha256']
            or gate.get('base_manifest_sha256') != receipt['base_manifest_sha256']
            or gate.get('original_training_function_code_sha256') != receipt['original_training_function_code_sha256']
            or gate.get('original_environment_lock_sha256') != sha(Path(output) / 'environment_lock.json')
            or gate.get('smoke_report_sha256') != sha(report_path)
            or not all(gate.get(key) is True for key in ('actual_two_optimizer_steps',
                'actual_one_validation_batch', 'actual_paired_augmented_data', 'identical_control_initialization'))):
        raise ValueError('Actual shared-slot smoke gate is absent or belongs to another amendment/source/UUID')
    report = read(report_path)
    if (report.get('status') != 'PASS' or report['training']['batches'] != 2
            or report['training']['images'] != 8 or report['valid']['num_images'] != 4
            or report['training'].get('gradient_finite') is not True
            or report['binding']['execution_manifest_sha256'] != receipt['base_manifest_sha256']
            or report['binding']['environment_fingerprint'] != read(Path(output) / 'environment_lock.json')['fingerprint']
            or report['initial_state_sha256'] == report['final_state_sha256']
            or gate.get('environment_fingerprint') != report['binding']['environment_fingerprint']):
        raise ValueError('Shared-slot smoke lacks actual finite optimizer/validation execution')
    return gate


def job_key(variant, seed):
    if not variant.replace('_', '').isalnum() or seed not in (42, 43, 44):
        raise ValueError('Invalid job identity')
    return f'{variant}__seed_{seed}'


def claim_job(output, variant, seed, owner):
    path = Path(output) / 'claims' / f'{job_key(variant, seed)}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    value = dict(variant=variant, seed=seed, token=token, owner=owner,
                 status='CLAIMED_NO_AUTORETRY', created_at=now())
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    return value


@contextmanager
def coordinator_lock(output):
    path = Path(output) / 'coordinator.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def import_completed(old, base_output, output, variant, seed, config):
    source = old.result_dir(Path(base_output), variant, seed)
    if read(source / 'status.json')['status'] != 'COMPLETE':
        raise ValueError('Active/incomplete legacy runs cannot be imported')
    original = old.run_result(Path(base_output), variant, seed, config, write=False)
    hashes = tree_hashes(source)
    destination = old.result_dir(Path(output), variant, seed)
    temporary = destination.with_name(destination.name + '.importing')
    if destination.exists() or temporary.exists():
        raise FileExistsError('Preserve existing imported evidence; no automatic overwrite/retry')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    if tree_hashes(temporary) != hashes or tree_hashes(source) != hashes:
        raise ValueError('Legacy completed artifacts changed while importing')
    temporary.rename(destination)
    derived = old.run_result(Path(output), variant, seed, config, write=False)
    if derived['binding'] != original['binding'] or derived['history'] != original['history']:
        raise ValueError('Imported run lost its original scientific binding/history')
    receipt = dict(status='IMPORTED_COMPLETE_READ_ONLY', variant=variant, seed=seed,
        original_folder=str(source), original_result_sha256=sha(source / 'result.json'),
        original_manifest_sha256=sha(Path(base_output) / 'execution_manifest.json'),
        original_config_sha256=sha(config), original_source_sha256=read(Path(base_output) / 'execution_manifest.json')['source_sha256'],
        imported_artifact_sha256=hashes, imported_result_json_byte_identical=True,
        result_paths_consumed_via_run_result_write_false=True,
        amendment_sha256=sha(Path(output) / 'RESOURCE_AMENDMENT.json'),
        original_physical_gpu=1, formal_eligible=False, test_evaluated=False)
    dump(Path(output) / 'imports' / f'{job_key(variant, seed)}.json', receipt)
    dump(destination / 'resource_binding.json', receipt)
    return derived


def worker(old, model, base_output, output, config, variant, seed, physical_index, token):
    amendment, cfg, _, receipt = verify_prepared(old, model, base_output, output, config)
    slot_gate = require_slot_smoke(old, output, physical_index, receipt)
    claim = read(Path(output) / 'claims' / f'{job_key(variant, seed)}.json')
    if claim['token'] != token or claim['owner'].get('physical_gpu') != physical_index:
        raise ValueError('Worker does not own the unique amended job claim')
    folder = old.result_dir(Path(output), variant, seed)
    if (folder / 'run_binding.json').exists():
        raise FileExistsError('Existing worker cannot be automatically retried')
    shim = install_resource_shim(old, amendment, physical_index)
    admission_receipt = old.gpu_admission(cfg, 'cuda')
    resource = dict(schema_version=1, variant=variant, seed=seed,
        physical_gpu=physical_index, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        process=process_identity(os.getpid()), admission=admission_receipt,
        amendment_sha256=receipt['amendment_sha256'], wrapper_source_sha256=receipt['wrapper_source_sha256'],
        base_manifest_sha256=receipt['base_manifest_sha256'],
        base_source_sha256=receipt['base_source_sha256'], shim=shim,
        original_training_function_code_sha256=receipt['original_training_function_code_sha256'],
        actual_slot_smoke_gate_sha256=sha(Path(output) / 'resource_smoke' / f'gpu_{physical_index}' / 'slot_smoke_gate.json')
            if physical_index in (2, 3) else sha(Path(output) / 'smoke_gate.json'),
        formal_eligible=False, test_evaluated=False,
        evidence_scope='same frozen numerical software protocol with amended training GPU resources')
    path = folder / 'resource_binding.json'
    if path.exists():
        raise FileExistsError('Existing resource binding cannot be overwritten')
    dump(path, resource)
    try:
        # This is the complete original function; no training/evaluation logic is
        # copied into this external scheduler or replaced by a new implementation.
        result = old.run_training(Path(output), variant, seed, 'cuda', resume=False, config=Path(config))
        if function_hashes(old, RESOURCE_FUNCTIONS) != receipt['original_training_function_code_sha256']:
            raise ValueError('Original numerical training functions changed during execution')
        dump(folder / 'resource_completion.json', dict(status='COMPLETE', binding=result['binding'],
            resource_binding_sha256=sha(path), original_training_functions_unchanged=True,
            amendment_sha256=receipt['amendment_sha256'], formal_eligible=False, test_evaluated=False))
        return dict(status='COMPLETE', variant=variant, seed=seed, physical_gpu=physical_index)
    except Exception as error:
        failure = folder / f'external_failure_{time.time_ns()}.json'
        dump(failure, dict(status='RESOURCE_UNAVAILABLE_NO_AUTORETRY'
            if isinstance(error, old.ResourceUnavailable) else 'FAILED_NO_AUTORETRY',
            error_type=type(error).__name__, error=str(error), resource_binding_sha256=sha(path),
            formal_eligible=False, test_evaluated=False))
        raise


def ledger_seconds(prior, legacy, running, completed_seconds, *, monotonic=None, unix=None):
    mono = time.monotonic() if monotonic is None else monotonic
    wall = time.time() if unix is None else unix
    legacy_seconds = sum(max(0., item.get('finished_at_unix', wall) - item['started_at_unix']) for item in legacy)
    live_seconds = sum(max(0., mono - item['started_monotonic']) for item in running.values())
    return float(prior) + legacy_seconds + float(completed_seconds) + live_seconds


def scheduler(old, model, base_output, output, config, poll_seconds=5):
    output, base_output = Path(output), Path(base_output)
    amendment, cfg, _, receipt = verify_prepared(old, model, base_output, output, config)
    slot_gates = [require_slot_smoke(old, output, index, receipt) for index in (2, 3)]
    if not 0 < poll_seconds <= 30:
        raise ValueError('Coordinator poll interval must be 1-30 seconds')
    for expected in amendment.get('legacy_controllers', []):
        if same_process(expected, process_identity(expected['pid'])):
            raise ValueError('The original CPU queue/controller is still active; prevent duplicate jobs before handoff')
    status_path = output / 'parallel_status.json'
    if status_path.exists():
        raise FileExistsError('Coordinator already invoked; no automatic resume/retry')
    running, completed, failed = {}, {}, []
    legacy = [dict(entry) for entry in amendment.get('legacy_active', [])]
    legacy_jobs = {(entry['variant'], entry['seed']) for entry in legacy}
    for entry in legacy:
        if not same_process(entry, process_identity(entry['pid'])):
            # It may have completed during preparation; completion still needs
            # the full original artifact gate before an import is accepted.
            folder = old.result_dir(base_output, entry['variant'], entry['seed'])
            if not (folder / 'status.json').exists() or read(folder / 'status.json')['status'] != 'COMPLETE':
                raise ValueError('Registered active legacy worker disappeared without complete evidence')
    resource_smoke_seconds = sum(float(gate['smoke_gpu_seconds']) for gate in slot_gates)
    prior = float(amendment['prior_gpu_seconds']) + resource_smoke_seconds
    completed_seconds = 0.
    waiting_seconds = 0.
    waiting_started = None
    maximum = 48 * 3600
    stage = 1
    selection = None
    owner = dict(coordinator_pid=os.getpid(), amendment_sha256=receipt['amendment_sha256'])
    pending = [(variant, 42) for variant in cfg['variants']]
    state = dict(status='RUNNING', stage=1, coordinator_pid=os.getpid(), started_at=now(),
        amendment_sha256=receipt['amendment_sha256'], base_manifest_sha256=receipt['base_manifest_sha256'],
        completed=[], active=[], legacy_active=legacy, failed=[], pending=pending,
        formal_eligible=False, test_evaluated=False, automatic_search=False, automatic_hardware=False)

    def publish(status='RUNNING'):
        seconds = ledger_seconds(prior, legacy, running, completed_seconds)
        state.update(status=status, stage=stage, completed=[list(job) for job in completed],
            active=[dict(variant=slot['job'][0], seed=slot['job'][1], physical_gpu=gpu,
                         pid=slot['child'].pid) for gpu, slot in running.items()],
            legacy_active=legacy, failed=failed, pending=pending,
            aggregate_gpu_seconds=seconds, aggregate_gpu_hours=seconds / 3600,
            prior_gpu_seconds=prior, completed_new_worker_seconds=completed_seconds,
            resource_smoke_gpu_seconds=resource_smoke_seconds,
            waiting_resource_seconds=waiting_seconds + (time.monotonic() - waiting_started if waiting_started is not None else 0.),
            slot_smoke_gate_sha256={str(index): sha(output / 'resource_smoke' / f'gpu_{index}' / 'slot_smoke_gate.json') for index in (2, 3)},
            updated_at=now())
        dump(status_path, state)

    with coordinator_lock(output):
        for variant, seed in list(pending):
            folder = old.result_dir(base_output, variant, seed)
            if (variant, seed) in legacy_jobs:
                claim_job(output, variant, seed, dict(owner, purpose='adopt_existing_active_worker', physical_gpu=1))
                pending.remove((variant, seed))
            elif (folder / 'status.json').exists():
                if read(folder / 'status.json')['status'] != 'COMPLETE':
                    raise ValueError('Unregistered existing legacy invocation cannot be duplicated or retried')
                claim_job(output, variant, seed, dict(owner, purpose='import_completed_legacy', physical_gpu=1))
                completed[variant, seed] = import_completed(old, base_output, output, variant, seed, config)
                pending.remove((variant, seed))
        publish()
        while True:
            for item in legacy:
                if item.get('imported') or item.get('failure_recorded'):
                    continue
                actual = process_identity(item['pid'])
                if same_process(item, actual):
                    continue
                item['finished_at_unix'] = time.time()
                job = (item['variant'], item['seed'])
                try:
                    completed[job] = import_completed(old, base_output, output, *job, config)
                    item['imported'] = True
                except Exception as error:
                    failed.append(dict(job=list(job), origin='legacy_worker', error=str(error)))
                    item['imported'] = False
                    item['failure_recorded'] = True
            for gpu, slot in list(running.items()):
                code = slot['child'].poll()
                timeout = time.monotonic() - slot['started_monotonic'] >= 2 * 3600
                if code is None and not timeout:
                    continue
                if timeout and code is None:
                    # Only this scheduler's own Popen instance is terminated.
                    slot['child'].terminate()
                    try:
                        slot['child'].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        slot['child'].kill()
                        slot['child'].wait(timeout=10)
                    code = slot['child'].returncode
                duration = time.monotonic() - slot['started_monotonic']
                completed_seconds += duration
                slot['log'].close()
                job = slot['job']
                if code == 0 and not timeout:
                    try:
                        completed[job] = old.run_result(output, *job, config, write=False)
                        old.verify_pairing(list(completed.values()))
                    except Exception as error:
                        failed.append(dict(job=list(job), physical_gpu=gpu, error=str(error)))
                else:
                    failed.append(dict(job=list(job), physical_gpu=gpu, exit_code=code,
                                       timeout=timeout, status='FAILED_NO_AUTORETRY'))
                del running[gpu]
            elapsed = ledger_seconds(prior, legacy, running, completed_seconds)
            budget_exhausted = elapsed >= maximum
            if budget_exhausted:
                for gpu, slot in list(running.items()):
                    # Budget limits only stop children created by this scheduler;
                    # the adopted legacy worker and resident services are untouched.
                    slot['child'].terminate()
                    try:
                        slot['child'].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        slot['child'].kill()
                        slot['child'].wait(timeout=10)
                    completed_seconds += time.monotonic() - slot['started_monotonic']
                    slot['log'].close()
                    failed.append(dict(job=list(slot['job']), status='GPU_BUDGET_STOP_NO_AUTORETRY', physical_gpu=gpu))
                    del running[gpu]
                publish('GPU_BUDGET_EXHAUSTED_NO_CONTINUATION')
                return state
            if failed:
                publish('DRAINING_FAILURE_NO_AUTORETRY' if running else 'FAILED_NO_AUTORETRY')
                if not running:
                    return state
                time.sleep(poll_seconds)
                continue
            if stage == 1 and not pending and not running and all(item.get('imported') for item in legacy):
                rows = [completed[variant, 42] for variant in cfg['variants']]
                selection = old.select_confirmation(rows, cfg)
                selection['execution_manifest_sha256'] = sha(output / 'execution_manifest.json')
                selection['stage1_result_sha256'] = {variant: sha(old.result_dir(output, variant, 42) / 'result.json')
                                                     for variant in cfg['variants']}
                dump(output / 'stage1_selection.json', selection)
                pending = [tuple(job) for job in selection['jobs']]
                stage = 2
                state['stage1_selection'] = selection
            if stage == 2 and not pending and not running:
                summary = old.summarize(output, config)
                state['confirmation_decisions'] = summary['confirmation_decisions']
                publish('COMPLETE')
                return state
            blocked = []
            for gpu in (2, 3, 1):
                if not pending or gpu in running or (gpu == 1 and any(not item.get('imported') and not item.get('failure_recorded') for item in legacy)):
                    continue
                try:
                    resource = admission(amendment, gpu, old.ResourceUnavailable)
                except old.ResourceUnavailable as error:
                    blocked.append(dict(physical_gpu=gpu, reason=str(error)))
                    continue
                job = pending.pop(0)
                folder = old.result_dir(output, *job)
                if folder.exists():
                    raise FileExistsError('Existing parallel job cannot be automatically retried')
                claim = claim_job(output, *job, dict(owner, physical_gpu=gpu, purpose='new_amended_worker'))
                folder.mkdir(parents=True)
                log = (folder / 'parallel_stdout.log').open('x', encoding='utf-8')
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--base-root', str(old.ROOT),
                    '--base-output', str(base_output), '--output', str(output), '--config', str(config),
                    '--variant', job[0], '--seed', str(job[1]), '--gpu', str(gpu), '--claim-token', claim['token']]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                tick = time.monotonic()
                child = subprocess.Popen(command, cwd=old.ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = dict(job=job, child=child, log=log, started_monotonic=tick, admission=resource)
                dump(folder / 'parallel_launch.json', dict(variant=job[0], seed=job[1], physical_gpu=gpu,
                    pid=child.pid, claim_token=claim['token'], admission=resource, started_at=now(),
                    amendment_sha256=receipt['amendment_sha256'], formal_eligible=False))
                # Publish every owned child immediately, even if admission for a
                # later GPU fails, so a supervisor can preserve/observe this run.
                publish()
            state['blocked_resources'] = blocked
            progress_active = bool(running) or any(not item.get('imported') and not item.get('failure_recorded') for item in legacy)
            if progress_active and waiting_started is not None:
                waiting_seconds += time.monotonic() - waiting_started
                waiting_started = None
            elif pending and not progress_active and waiting_started is None:
                waiting_started = time.monotonic()
            total_wait = waiting_seconds + (time.monotonic() - waiting_started if waiting_started is not None else 0.)
            if total_wait >= 6 * 3600:
                publish('RESOURCE_WAIT_TIMEOUT_NO_AUTORETRY')
                return state
            publish('RUNNING' if progress_active else 'WAITING_RESOURCE')
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'slot-smoke', 'scheduler', 'worker', 'status', 'admission'))
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--base-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--amendment', type=Path)
    parser.add_argument('--variant', default='control')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--claim-token')
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    old, model = load_base(args.base_root)
    old.torch.set_num_threads(2)
    if args.mode == 'prepare':
        if args.amendment is None:
            raise ValueError('prepare requires an explicit new RESOURCE_AMENDMENT')
        result = prepare(old, model, args.base_output, args.output, args.config, args.amendment)
    elif args.mode == 'worker':
        result = worker(old, model, args.base_output, args.output, args.config,
                        args.variant, args.seed, args.gpu, args.claim_token)
    elif args.mode == 'slot-smoke':
        result = slot_smoke(old, model, args.base_output, args.output, args.config, args.gpu)
    elif args.mode == 'scheduler':
        try:
            result = scheduler(old, model, args.base_output, args.output, args.config, args.poll_seconds)
        except Exception as error:
            path = args.output / 'parallel_status.json'
            if path.exists():
                value = read(path)
                value.update(status='COORDINATOR_STOPPED_EXISTING_WORKERS_PRESERVED_NO_AUTORETRY',
                    error_type=type(error).__name__, error=str(error), updated_at=now())
                dump(path, value)
            raise
    elif args.mode == 'admission':
        amendment, _, _, _ = verify_prepared(old, model, args.base_output, args.output, args.config)
        result = admission(amendment, args.gpu, old.ResourceUnavailable)
    else:
        result = read(args.output / 'parallel_status.json')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
