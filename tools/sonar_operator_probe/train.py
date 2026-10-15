"""Isolated R0/C01b software operator screening, without NAS or hardware actions.

All comparisons use fresh same-seed controls and the frozen train/valid loaders.
Smoke artifacts never count as experimental results or formal search evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
from tools.sonar_head_factorial import train as t
from tools.sonar_head_c01b import model as c01b

import torch
import yaml

p = t.p
read_json = t.read_json
VERSION = 'sonar_operator_probe_20261013_01'
CONFIG = ROOT / 'configs/sonar_operator_probe_20261013.yaml'
STABLE_EPOCHS = t.STABLE_EPOCHS
SEEDS = (42, 43, 44)
WITNESSES = ('augmented_batches_sha256', 'image_order_sha256', 'valid_batches_sha256')


class ResourceUnavailable(RuntimeError):
    """Authorized GPU is occupied; this is not a model/training failure."""


class ResourceWaitTimeout(TimeoutError):
    """The bounded resource waiting allowance is exhausted without a retry."""


def _inside(relative):
    return t._inside_root(relative)


def _preregistration(cfg):
    entry = cfg['preregistration']
    return _inside(entry['path'] if isinstance(entry, dict) else entry)


def load_config(config=CONFIG):
    from tools.sonar_operator_probe import model as adapter
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    old, baseline = t.load_config()
    expected = dict(epochs=80, batch_size=4, input=640, optimizer='AdamW',
                    lr=.001, weight_decay=.0001, scheduler='none', amp=False,
                    workers=0, ema=False, accumulation=1,
                    save_prediction_epochs=list(STABLE_EPOCHS))
    if (cfg.get('version') != VERSION or cfg.get('scope') != 'exploratory_operator_screen'
            or cfg.get('formal_eligible') is not False
            or cfg.get('baseline') != old['baseline']
            or cfg.get('environment_spec') != old['environment_spec']
            or cfg.get('author') != old['author']
            or cfg.get('subset_contract') != old['subset_contract']
            or cfg.get('subset_contract_sha256') != old['subset_contract_sha256']
            or cfg.get('evaluation') != old['evaluation']
            or cfg.get('seeds') != list(SEEDS) or cfg.get('screening_seed') != 42
            or cfg.get('test_policy') != 'sealed_not_mounted'
            or any(cfg['training'].get(key) != value for key, value in expected.items())):
        raise ValueError('Operator screen must retain isolated R0/C01b train/valid protocol')
    authorization = cfg.get('authorization', cfg.get('user_authorization'))
    if isinstance(authorization, dict):
        authorization = authorization.get('request')
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError('New operator campaign requires its own explicit user authorization')
    variants = cfg.get('variants', [])
    if (not isinstance(variants, list) or not variants or len(set(variants)) != len(variants)
            or any(name not in adapter.VARIANTS for name in variants) or 'control' not in variants):
        raise ValueError('Variants must be unique registered names with a fresh control')
    comparisons = cfg.get('comparisons', {})
    if (not isinstance(comparisons, dict)
            or any(name not in variants or not isinstance(parents, list) or not parents
                   or len(set(parents)) != len(parents)
                   or any(parent not in variants or parent == name for parent in parents)
                   for name, parents in comparisons.items())):
        raise ValueError('Every screened candidate needs its frozen matching controls')
    eligible = cfg.get('confirmation_eligible_variants', list(comparisons))
    if (not isinstance(eligible, list) or len(set(eligible)) != len(eligible)
            or any(name not in comparisons for name in eligible)):
        raise ValueError('Confirmation eligibility must be a frozen subset of candidate comparisons')
    queue_cfg = cfg.get('queue', {})
    if (queue_cfg.get('allowed_gpus') != [1]
            or queue_cfg.get('maximum_hours_per_run') != 2
            or queue_cfg.get('maximum_gpu_hours') != 48
            or queue_cfg.get('maximum_confirm_variants') != 3
            or queue_cfg.get('adaptive_seed_extension') is not True
            or queue_cfg.get('retry', 'none') != 'none'
            or queue_cfg.get('resource_wait_maximum_hours', 6) != 6
            or queue_cfg.get('resource_poll_seconds', 30) != 30):
        raise ValueError('Unregistered GPU, retry, run deadline or total campaign budget')
    decision = cfg['decision']
    if (decision.get('metric') != 'mAP50_95'
            or decision.get('primary_checkpoint') != 'mean_epochs_76_80'
            or decision.get('primary_epochs') != list(STABLE_EPOCHS)
            or decision.get('automatic_search') is not False
            or decision.get('automatic_hardware') is not False):
        raise ValueError('Screening may only recommend subsequent software review')
    preregistration = _preregistration(cfg)
    if not preregistration.is_file():
        raise FileNotFoundError('The new campaign preregistration is absent')
    if isinstance(cfg['preregistration'], dict):
        digest = cfg['preregistration'].get('sha256')
        if digest is not None and digest != p.sha(preregistration):
            raise ValueError('Preregistration SHA256 drift')
    t.load_subset_contract(cfg)
    return cfg, baseline


def source_paths(cfg, config=CONFIG):
    # Source closure is inherited, but none of the old campaign's execution or
    # mutable baseline-resolution/paused-work gates are imported.
    old, _ = t.load_config()
    paths = set(t.source_paths(old, t.CONFIG))
    for directory in ('tools/sonar_head_c01b', 'tools/sonar_operator_probe',
                      'tools/sonar_joint192_validation', 'tools/sonar_joint192_extension_20261013'):
        paths.update((ROOT / directory).glob('*.py'))
    paths.update((Path(config).resolve(), _preregistration(cfg)))
    for name in ('test_sonar_operator_probe_train.py', 'test_sonar_operator_probe_model.py'):
        path = ROOT / 'tests' / name
        if path.is_file():
            paths.add(path)
    for name in cfg.get('extra_source_paths', []):
        paths.add(_inside(name))
    return sorted(paths)


def source_hashes(cfg, config=CONFIG):
    return {path.relative_to(ROOT).as_posix(): p.sha(path)
            for path in source_paths(cfg, config)}


def physical_gpu(cfg, device):
    if device == 'cpu':
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not visible or len(visible.split(',')) != 1 or not visible.isdigit()
            or int(visible) not in cfg['queue']['allowed_gpus']):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to the single authorized physical GPU1')
    return int(visible)


def gpu_admission(cfg, device):
    """Refuse any other compute owner; never kill or reschedule another job."""
    index = physical_gpu(cfg, device)
    if index is None:
        return dict(device='cpu', admitted=True)
    command = ['nvidia-smi', '-i', str(index),
               '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits']
    gpu = subprocess.check_output(command, text=True).strip().split(',')
    if len(gpu) != 2:
        raise ValueError('Cannot determine authorized GPU UUID and memory occupancy')
    uuid, memory = gpu[0].strip(), float(gpu[1].strip())
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True)
    other = []
    for line in owners.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) == 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
            other.append(int(fields[1]))
    threshold = float(cfg['queue'].get('gpu_memory_admission_mib', 1024))
    if other or memory > threshold:
        raise ResourceUnavailable(f'GPU{index} occupied: memory={memory} MiB, other_compute_pids={other}')
    return dict(device='cuda', physical_gpu=index, gpu_uuid=uuid, memory_used_mib=memory,
                other_compute_pids=other, admitted=True, recorded_at=p.now())


def wait_for_gpu(cfg, device, *, on_wait=None, maximum_wait_seconds=None, poll_seconds=None):
    """Wait for GPU1 without touching other jobs; only resource errors are retried.

    on_wait receives a compact receipt for a WAITING_RESOURCE status. A successful
    receipt reports waiting_seconds, which callers must keep out of GPU budgets.
    A malformed nvidia-smi response or any nonresource exception fails immediately.
    """
    maximum = (float(cfg['queue'].get('resource_wait_maximum_hours', 6)) * 3600
               if maximum_wait_seconds is None else float(maximum_wait_seconds))
    interval = (float(cfg['queue'].get('resource_poll_seconds', 30))
                if poll_seconds is None else float(poll_seconds))
    if (not math.isfinite(maximum) or not 0 <= maximum <= 6 * 3600
            or not math.isfinite(interval) or not 0 < interval <= 60):
        raise ValueError('GPU waiting must be bounded to six hours and polls of at most one minute')
    tick = time.monotonic()
    while True:
        try:
            admission = gpu_admission(cfg, device)
            return dict(admission, waiting_seconds=time.monotonic() - tick)
        except ResourceUnavailable as error:
            elapsed = time.monotonic() - tick
            remaining = max(0., maximum - elapsed)
            receipt = dict(status='WAITING_RESOURCE', error=str(error),
                error_type=type(error).__name__, waiting_seconds=elapsed,
                remaining_wait_seconds=remaining, updated_at=p.now(),
                formal_eligible=False, test_evaluated=False)
            if on_wait is not None:
                on_wait(receipt)
            if remaining <= 0:
                raise ResourceWaitTimeout('Authorized GPU1 remained occupied for the bounded resource waiting allowance') from error
            time.sleep(min(interval, remaining))


def prepare(root, config=CONFIG, device='cuda'):
    root = Path(root)
    cfg, baseline = load_config(config)
    author = ROOT / cfg['author']['source_root']
    if (author / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        commit = read_json(author / 'source_receipt.json')['commit']
    if commit != cfg['author']['commit']:
        raise ValueError('Pinned author source commit drift')
    inputs = p.verify_inputs(baseline)
    subset = t.load_subset_contract(cfg, inputs)
    targets = t.frozen_validation_targets(baseline)
    if [row['image'] for row in targets] != subset['full_images']:
        raise ValueError('Frozen validation target IDs differ from the original full99')
    admission = gpu_admission(cfg, device)
    manifest = dict(version=VERSION, scope=cfg['scope'], config=cfg,
        config_sha256=p.sha(config), source_sha256=source_hashes(cfg, config),
        author_commit=commit, inputs=inputs, validation_targets=targets,
        validation_target_sha256=p.canonical_sha(targets),
        subset_contract_sha256=cfg['subset_contract_sha256'], subset_id=subset['subset_id'],
        preregistration_sha256=p.sha(_preregistration(cfg)),
        formal_eligible=False, test_evaluated=False, automatic_search=False,
        automatic_hardware=False)
    path = root / 'execution_manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Existing campaign freeze differs; preserve it and use a new root')
    root.mkdir(parents=True, exist_ok=True)
    # The same deterministic flags must precede every environment fingerprint.
    # make_run uses seed_all(seed); seeding here avoids a prepare/smoke lock drift.
    p.seed_all(42)
    t.bind_environment(root, device, cfg)
    if not path.exists():
        p.dump(path, manifest)
        p.dump(root / 'prepare_admission.json', admission)
    return manifest


def verify_manifest(root, config=CONFIG):
    cfg, baseline = load_config(config)
    manifest = read_json(Path(root) / 'execution_manifest.json')
    if (manifest['config'] != cfg or manifest['config_sha256'] != p.sha(config)
            or manifest['source_sha256'] != source_hashes(cfg, config)
            or manifest['inputs'] != p.verify_inputs(baseline)
            or manifest['preregistration_sha256'] != p.sha(_preregistration(cfg))
            or manifest['validation_target_sha256'] != p.canonical_sha(manifest['validation_targets'])
            or manifest.get('formal_eligible') is not False
            or manifest.get('test_evaluated') is not False):
        raise ValueError('New campaign source/config/train-valid identity drift')
    subset = t.load_subset_contract(cfg, manifest['inputs'])
    if [row['image'] for row in manifest['validation_targets']] != subset['full_images']:
        raise ValueError('Frozen validation target membership drift')
    return cfg, baseline


def result_dir(root, variant, seed, mode='runs'):
    return Path(root) / mode / variant / f'seed_{seed}'


def binding(root, variant, seed, mode='runs'):
    manifest = read_json(Path(root) / 'execution_manifest.json')
    return dict(execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
        environment_fingerprint=read_json(Path(root) / 'environment_lock.json')['fingerprint'],
        subset_contract_sha256=manifest['subset_contract_sha256'],
        validation_target_sha256=manifest['validation_target_sha256'],
        variant=variant, seed=seed, mode=mode, scope='exploratory_operator_screen',
        formal_eligible=False, test_evaluated=False)


def make_run(root, variant, seed, device, mode, config=CONFIG):
    from tools.sonar_operator_probe import model as adapter
    cfg, baseline = verify_manifest(root, config)
    if variant not in cfg['variants'] or seed not in SEEDS:
        raise ValueError('Unregistered operator/seed')
    physical_gpu(cfg, device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    p.seed_all(seed)
    lock = t.bind_environment(root, device, cfg)
    model = adapter.build_model(variant, seed, source_root=ROOT / cfg['author']['source_root']).to(device)
    criterion = c01b.build_criterion(model, baseline)
    identity, metadata = model.initialization_identity(), model.metadata()
    metadata['criterion'] = c01b.criterion_metadata(model, criterion)
    metadata['variant_spec'] = adapter.variant_spec(variant)
    metadata['head_contract'] = dict(reg_max=model.head.reg_max, raw_channels=model.head.no,
                                    measured_channels=[r[-1].out_channels + c[-1].out_channels
                                        for r, c in zip(model.head.cv2, model.head.cv3)])
    if metadata['head_contract'] != dict(reg_max=1, raw_channels=6, measured_channels=[6] * 4):
        raise ValueError('Every screen arm must retain the selected C01b head')
    initial = p.state_hash(model)
    p.seed_all(seed)
    rng_identity = p.training_rng_identity(seed)
    data = ROOT / baseline['dataset']['data_dir']
    train = p.frozen.make_loader(baseline, data, 'train', seed, 0, train=True)
    valid = p.frozen.make_loader(baseline, data, 'valid', seed, 0)
    groups = adapter.parameter_groups(model, weight_decay=.0001)
    optimizer = torch.optim.AdamW(groups, lr=.001, weight_decay=.0001)
    optimizer.register_step_post_hook(lambda opt, args, kwargs: adapter.project_nonnegative_(model))
    metadata['optimizer_groups'] = [dict(parameters=sum(parameter.numel() for parameter in group['params']),
                                            weight_decay=group.get('weight_decay', .0001)) for group in groups]
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    folder = result_dir(root, variant, seed, mode)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (('initialization_identity.json', identity), ('model_info.json', metadata),
                        ('training_rng_identity.json', rng_identity)):
        target = folder / name
        if target.exists() and read_json(target) != value:
            raise ValueError(f'Existing run construction identity drift: {name}')
        if not target.exists():
            p.dump(target, value)
    return folder, cfg, baseline, model, criterion, optimizer, train, valid, initial, identity, lock


def _shared_hashes(identity):
    value = identity.get('shared_tensor_sha256', identity.get('shared_state_sha256'))
    if not isinstance(value, dict) or not value:
        raise ValueError('Model must expose the unchanged initialization tensor fingerprints')
    return value


def verify_pairing(rows, smoke=False):
    """Check actual augmentation/order and every mutually unchanged initial tensor."""
    for seed in sorted({row['binding']['seed'] for row in rows}):
        group = [row for row in rows if row['binding']['seed'] == seed]
        if len({row['binding']['variant'] for row in group}) != len(group):
            raise ValueError('Duplicated comparison arm')
        for key in ('execution_manifest_sha256', 'environment_fingerprint',
                    'subset_contract_sha256', 'validation_target_sha256'):
            if len({row['binding'][key] for row in group}) != 1:
                raise ValueError(f'Paired rows differ in {key}')
        fingerprints = [_shared_hashes(row['initialization']) for row in group]
        for first in range(len(fingerprints)):
            for second in range(first + 1, len(fingerprints)):
                shared = set(fingerprints[first]) & set(fingerprints[second])
                if not shared or any(fingerprints[first][name] != fingerprints[second][name] for name in shared):
                    raise ValueError('Common unchanged initialization tensors differ across operator arms')
        if smoke:
            witnesses = [p.canonical_sha([row['training'][key] for key in WITNESSES[:2]]
                                         + [row['valid_batches_sha256']]) for row in group]
        else:
            witnesses = [p.canonical_sha([[epoch[key] for key in WITNESSES]
                                          for epoch in row['history']]) for row in group]
        if len(set(witnesses)) != 1:
            raise ValueError('Actual training pixels/labels/order or validation batches differ')


def smoke_one(root, variant, seed, device, config=CONFIG):
    folder = result_dir(root, variant, seed, 'smoke')
    if folder.exists():
        raise FileExistsError('Preserve existing smoke; no automatic overwrite/retry')
    run = make_run(root, variant, seed, device, 'smoke', config)
    folder, cfg, baseline, model, criterion, optimizer, train, valid, initial, identity, lock = run
    tick = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        trained = t.train_epoch(model, criterion, optimizer, train, device, 2)
        metrics, records, valid_sha = t.evaluate(model, criterion, valid, baseline, device, 1)
        final = p.state_hash(model)
        if (trained['batches'] != 2 or trained['images'] != 8
                or metrics['num_images'] != 4 or initial == final):
            raise ValueError('Smoke must contain two optimizer steps and one validation batch')
        report = dict(status='PASS', binding=binding(root, variant, seed, 'smoke'),
            training=trained, valid=metrics, valid_batches_sha256=valid_sha,
            initialization=identity, initial_state_sha256=initial, final_state_sha256=final,
            seconds=time.perf_counter() - tick,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            formal_eligible=False, test_evaluated=False)
        p.dump(folder / 'valid_records.json', records)
        p.dump(folder / 'smoke_report.json', report)
        print(f'SMOKE_PASS variant={variant} seconds={report["seconds"]:.2f}', flush=True)
        return report
    except Exception as error:
        p.dump(folder / 'smoke_failure.json', dict(status='FAILED_NO_AUTORETRY',
               error=str(error), error_type=type(error).__name__, formal_eligible=False))
        raise
    finally:
        del run, model, criterion, optimizer, train, valid
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def smoke(root, device, config=CONFIG):
    cfg, _ = verify_manifest(root, config)
    admission = gpu_admission(cfg, device)
    tick = time.perf_counter()
    rows = [smoke_one(root, variant, 42, device, config) for variant in cfg['variants']]
    verify_pairing(rows, smoke=True)
    gate = dict(status='PASS', execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
        environment_fingerprint=read_json(Path(root) / 'environment_lock.json')['fingerprint'],
        variants=cfg['variants'], admission=admission, actual_two_train_batches_each=True,
        actual_one_valid_batch_each=True, identical_augmented_batches=True,
        paired_unchanged_initial_tensors=True, formal_eligible=False, test_evaluated=False)
    gate['smoke_gpu_seconds'] = time.perf_counter() - tick if device == 'cuda' else 0
    p.dump(Path(root) / 'smoke_gate.json', gate)
    return gate


def require_smoke(root, cfg):
    gate = read_json(Path(root) / 'smoke_gate.json')
    if (gate.get('status') != 'PASS' or gate.get('variants') != cfg['variants']
            or gate['execution_manifest_sha256'] != p.sha(Path(root) / 'execution_manifest.json')
            or gate['environment_fingerprint'] != read_json(Path(root) / 'environment_lock.json')['fingerprint']):
        raise ValueError('Complete source-bound all-variant smoke gate required')
    rows = []
    for variant in cfg['variants']:
        report = read_json(result_dir(root, variant, 42, 'smoke') / 'smoke_report.json')
        if report.get('status') != 'PASS' or report['binding'] != binding(root, variant, 42, 'smoke'):
            raise ValueError('Smoke report identity mismatch')
        rows.append(report)
    verify_pairing(rows, smoke=True)


def _verify_late_weights(folder, history, expected, initial, identity):
    hashes = {}
    for row in history:
        if row['epoch'] not in STABLE_EPOCHS:
            continue
        archive = row.get('weight_archive', {})
        relative = f'epoch_weights/epoch_{row["epoch"]:03d}.pt'
        path = folder / relative
        if (archive.get('path') != relative or not path.is_file()
                or p.sha(path) != archive.get('sha256')):
            raise ValueError('Late-epoch weight archive missing or hash drift')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['binding'] != expected or state['epoch'] != row['epoch']
                or state['initial_state_sha256'] != initial or state['initialization'] != identity):
            raise ValueError('Late-epoch weights belong to another run/source/initialization')
        hashes[str(row['epoch'])] = archive['sha256']
    return hashes


def run_result(root, variant, seed, config=CONFIG, write=False):
    cfg, baseline = verify_manifest(root, config)
    folder = result_dir(root, variant, seed)
    expected = binding(root, variant, seed)
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = latest['history']
    t.validate_history(history)
    best = max(history, key=lambda row: row['valid']['mAP50_95'])
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    identity = read_json(folder / 'initialization_identity.json')
    if (latest['binding'] != expected or latest['epoch'] != 80
            or latest['initialization'] != identity or checkpoint['binding'] != expected
            or checkpoint['epoch'] != best['epoch'] or checkpoint['history'] != history[:best['epoch']]
            or read_json(folder / 'epoch_log.json') != history
            or read_json(folder / 'best_valid_metrics.json') != best['valid']):
        raise ValueError('Completed run checkpoint/metrics/initialization binding drift')
    subset = t.load_subset_contract(cfg)
    terminal = read_json(folder / 'terminal_valid_records.json')
    terminal_metrics = t.enrich_metrics(p.metrics_from_records(terminal, baseline))
    if (terminal_metrics != history[-1]['valid']
            or terminal_metrics != read_json(folder / 'terminal_valid_metrics.json')
            or t.enrich_metrics(p.metrics_from_records(read_json(folder / 'best_valid_records.json'), baseline)) != best['valid']):
        raise ValueError('Endpoint or best records do not reproduce logged metrics')
    predictions = t.verify_stable_predictions(folder, history, expected, baseline, subset['full_images'])
    weights = _verify_late_weights(folder, history, expected, latest['initial_state_sha256'], identity)
    if set(weights) != {str(epoch) for epoch in STABLE_EPOCHS}:
        raise ValueError('Five late-epoch weight archives required for completion')
    stable = t.stable_metrics_from_history(history)
    report = dict(status='COMPLETE', binding=expected, epochs=80, best_epoch=best['epoch'],
        primary_checkpoint='mean_epochs_76_80', stable_metrics=stable, metrics=stable,
        secondary_terminal_metrics=terminal_metrics, secondary_best_metrics=best['valid'],
        initialization=identity, initial_state_sha256=latest['initial_state_sha256'],
        model_info=read_json(folder / 'model_info.json'), history=history,
        stable_prediction_paths=predictions['paths'], stable_prediction_sha256=predictions['sha256'],
        stable_weight_sha256=weights,
        hashes={name: p.sha(folder / name) for name in ('latest.pt', 'best.pt', 'epoch_log.json',
                'best_valid_records.json', 'terminal_valid_records.json')},
        formal_eligible=False, test_evaluated=False)
    if write:
        p.dump(folder / 'result.json', report)
    return report


def run_training(root, variant, seed, device, resume=False, config=CONFIG):
    cfg, _ = verify_manifest(root, config)
    require_smoke(root, cfg)
    if seed != 42:
        selection = read_json(Path(root) / 'stage1_selection.json')
        if (selection.get('execution_manifest_sha256') != p.sha(Path(root) / 'execution_manifest.json')
                or [variant, seed] not in selection['jobs']):
            raise ValueError('Seed43/44 workers require the frozen positive-screen confirmation selection')
    folder = result_dir(root, variant, seed)
    if (folder / 'run_binding.json').exists() and not resume:
        raise FileExistsError('Existing run requires explicit epoch-boundary --resume')
    if resume and not (folder / 'latest.pt').is_file():
        raise FileNotFoundError('Explicit resume requires a committed epoch checkpoint')
    try:
        # Recheck in the child because a different owner may arrive after the
        # parent admission. A race never causes automatic rerunning of a worker.
        gpu_admission(cfg, device)
    except ResourceUnavailable as error:
        receipt = dict(status='RESOURCE_UNAVAILABLE_NO_AUTORETRY',
            variant=variant, seed=seed, error=str(error), error_type=type(error).__name__,
            training_started=False, updated_at=p.now(), formal_eligible=False, test_evaluated=False)
        p.dump(folder / f'resource_admission_failure_{time.time_ns()}.json', receipt)
        if not (folder / 'status.json').exists():
            p.dump(folder / 'status.json', receipt)
        raise
    run = make_run(root, variant, seed, device, 'runs', config)
    folder, cfg, baseline, model, criterion, optimizer, train, valid, initial, identity, lock = run
    expected, expected_images = binding(root, variant, seed), t.load_subset_contract(cfg)['full_images']
    start, history = 0, []
    tick_run = time.perf_counter()
    if resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        start, history = t.validate_resume_state(folder, state, expected, initial, identity,
                                                 baseline, expected_images)
        _verify_late_weights(folder, history, expected, initial, identity)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        p.restore_rng(state['rng'], train)
        if valid.generator is not None:
            valid.generator.set_state(state['valid_loader_generator'])
    else:
        p.dump(folder / 'run_binding.json', dict(binding=expected, initialization=identity,
            initial_state_sha256=initial, started_at=p.now(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        for epoch in range(start, 80):
            if time.perf_counter() - tick_run >= cfg['queue']['maximum_hours_per_run'] * 3600:
                raise TimeoutError('Run deadline reached at committed epoch boundary; no automatic retry')
            tick = time.perf_counter()
            previous_best = max((row['valid']['mAP50_95'] for row in history), default=-1.)
            record = t.train_epoch(model, criterion, optimizer, train, device)
            metrics, records, valid_sha = t.evaluate(model, criterion, valid, baseline, device)
            record.update(epoch=epoch + 1, valid=metrics, valid_batches_sha256=valid_sha,
                          wall_seconds=time.perf_counter() - tick)
            if epoch + 1 in STABLE_EPOCHS:
                record['prediction_archive'] = t.save_stable_predictions(
                    folder, epoch + 1, records, expected, expected_images)
            history.append(record)
            t.validate_history(history, epoch + 1)
            state = dict(binding=expected, initial_state_sha256=initial, initialization=identity,
                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch + 1, rng=p.rng_state(train), history=history,
                valid_loader_generator=valid.generator.get_state() if valid.generator is not None else None,
                formal_eligible=False, test_evaluated=False)
            if epoch + 1 in STABLE_EPOCHS:
                relative = f'epoch_weights/epoch_{epoch + 1:03d}.pt'
                if (folder / relative).exists():
                    raise FileExistsError('Uncommitted late-epoch weights preserved; no silent overwrite')
                (folder / 'epoch_weights').mkdir(parents=True, exist_ok=True)
                p.atomic_save(folder / relative, dict(binding=expected, epoch=epoch + 1,
                    initial_state_sha256=initial, initialization=identity,
                    model_state_dict=model.state_dict(), formal_eligible=False, test_evaluated=False))
                record['weight_archive'] = dict(path=relative, sha256=p.sha(folder / relative))
            if metrics['mAP50_95'] > previous_best:
                p.atomic_save(folder / 'best.pt', state)
                p.dump(folder / 'best_valid_metrics.json', metrics)
                p.dump(folder / 'best_valid_records.json', records)
            if epoch + 1 == 80:
                p.dump(folder / 'terminal_valid_metrics.json', metrics)
                p.dump(folder / 'terminal_valid_records.json', records)
            p.atomic_save(folder / 'latest.pt', state)
            p.dump(folder / 'epoch_log.json', history)
            p.dump(folder / 'status.json', dict(status='RUNNING', epoch=epoch + 1,
                binding=expected, latest_valid={key: metrics[key] for key in t.METRICS}, updated_at=p.now()))
            print(f'{variant} seed={seed} epoch={epoch + 1}/80 '
                  f'valid_mAP50_95={metrics["mAP50_95"]:.8f} seconds={record["wall_seconds"]:.2f}', flush=True)
        report = run_result(root, variant, seed, config, write=True)
        p.dump(folder / 'status.json', dict(status='COMPLETE', epochs=80, binding=expected,
            stable_mAP50_95=report['stable_metrics']['mAP50_95'], updated_at=p.now(),
            formal_eligible=False, test_evaluated=False))
        return report
    except Exception as error:
        p.dump(folder / 'status.json', dict(status='FAILED_NO_AUTORETRY', completed_epochs=len(history),
            error=str(error), error_type=type(error).__name__, binding=expected, updated_at=p.now()))
        raise
    finally:
        del run, model, criterion, optimizer, train, valid
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def candidate_deltas(rows, cfg, seeds):
    lookup = {(row['binding']['variant'], row['binding']['seed']): row for row in rows}
    result = {}
    for name, parents in cfg['comparisons'].items():
        if not all((variant, seed) in lookup for variant in [name] + parents for seed in seeds):
            continue
        pairs = {}
        for parent in parents:
            pairs[parent] = [dict(seed=seed,
                stable_delta=lookup[name, seed]['stable_metrics']['mAP50_95']
                    - lookup[parent, seed]['stable_metrics']['mAP50_95'],
                endpoint_delta=lookup[name, seed]['secondary_terminal_metrics']['mAP50_95']
                    - lookup[parent, seed]['secondary_terminal_metrics']['mAP50_95']) for seed in seeds]
        result[name] = dict(parents=pairs, ranking_delta=min(statistics.mean(
            row['stable_delta'] for row in pair) for pair in pairs.values()))
    return result


def select_confirmation(rows, cfg):
    verify_pairing(rows)
    expected = {(variant, 42) for variant in cfg['variants']}
    if {(row['binding']['variant'], row['binding']['seed']) for row in rows} != expected:
        raise ValueError('Adaptive confirmation requires all registered seed42 runs first')
    candidates = candidate_deltas(rows, cfg, [42])
    eligible = set(cfg.get('confirmation_eligible_variants', cfg['comparisons']))
    passing = [name for name, value in candidates.items()
               if name in eligible and all(pair[0]['stable_delta'] > 0 and pair[0]['endpoint_delta'] >= 0
                      for pair in value['parents'].values())]
    passing.sort(key=lambda name: (-candidates[name]['ranking_delta'], name))
    selected = passing[:cfg['queue']['maximum_confirm_variants']]
    controls = {parent for name in selected for parent in cfg['comparisons'][name]}
    names = [name for name in cfg['variants'] if name in set(selected) | controls]
    jobs = [(name, seed) for seed in SEEDS[1:] for name in names]
    return dict(selected_variants=selected, selected_controls=sorted(controls), jobs=jobs,
                stage1_deltas=candidates, status='CONFIRM' if selected else 'NO_STAGE1_POSITIVE',
                formal_eligible=False, test_evaluated=False)


def confirmation_decision(rows, cfg, selection):
    verify_pairing(rows)
    candidates = candidate_deltas(rows, cfg, list(SEEDS))
    decisions = {}
    for name in selection['selected_variants']:
        if name not in candidates:
            decisions[name] = dict(status='INCOMPLETE_NO_CLAIM')
            continue
        parents = {}
        for parent, pairs in candidates[name]['parents'].items():
            stable = statistics.mean(pair['stable_delta'] for pair in pairs)
            endpoint = statistics.mean(pair['endpoint_delta'] for pair in pairs)
            positives = sum(pair['stable_delta'] > 0 for pair in pairs)
            endpoint_positives = sum(pair['endpoint_delta'] > 0 for pair in pairs)
            parents[parent] = dict(paired_seed_deltas=pairs, mean_stable_delta=stable,
                                   mean_endpoint_delta=endpoint, positive_stable_seeds=positives,
                                   positive_endpoint_seeds=endpoint_positives,
                                   pass_gate=stable > 0 and endpoint > 0
                                             and positives >= 2 and endpoint_positives >= 2)
        passed = all(parent['pass_gate'] for parent in parents.values())
        decisions[name] = dict(status='REVIEW_SOFTWARE_ONLY' if passed else 'NO_STABLE_POSITIVE',
                               parents=parents, formal_eligible=False)
    return decisions


def summarize(root, config=CONFIG):
    cfg, _ = verify_manifest(root, config)
    require_smoke(root, cfg)
    rows, incomplete = [], []
    for variant in cfg['variants']:
        for seed in SEEDS:
            folder = result_dir(root, variant, seed)
            if (folder / 'result.json').is_file():
                rows.append(run_result(root, variant, seed, config))
            elif folder.exists():
                incomplete.append(dict(variant=variant, seed=seed,
                    status=read_json(folder / 'status.json') if (folder / 'status.json').exists() else 'NOT_COMMITTED'))
    verify_pairing(rows)
    stage1 = [row for row in rows if row['binding']['seed'] == 42]
    selection = select_confirmation(stage1, cfg) if len(stage1) == len(cfg['variants']) else None
    decisions = confirmation_decision(rows, cfg, selection) if selection else {}
    result = dict(status='SUMMARY', scope=cfg['scope'], complete_runs=len(rows),
        completed=[dict(variant=row['binding']['variant'], seed=row['binding']['seed'],
            stable_mAP50_95=row['stable_metrics']['mAP50_95'],
            endpoint_mAP50_95=row['secondary_terminal_metrics']['mAP50_95'],
            best_mAP50_95=row['secondary_best_metrics']['mAP50_95']) for row in rows],
        incomplete=incomplete, stage1_selection=selection, confirmation_decisions=decisions,
        source_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
        formal_eligible=False, test_evaluated=False, automatic_search=False, automatic_hardware=False)
    p.dump(Path(root) / 'campaign_summary.json', result)
    return result


def status(root, config=CONFIG):
    cfg, _ = load_config(config)
    root = Path(root)
    jobs = []
    for variant in cfg['variants']:
        for seed in SEEDS:
            path = result_dir(root, variant, seed) / 'status.json'
            if path.exists():
                jobs.append(dict(variant=variant, seed=seed, **read_json(path)))
    queue_path = root / 'queue_status.json'
    return dict(queue=read_json(queue_path) if queue_path.exists() else None, runs=jobs,
                formal_eligible=False, test_evaluated=False)


def queue(root, device, config=CONFIG):
    root = Path(root)
    cfg, _ = verify_manifest(root, config)
    require_smoke(root, cfg)
    status_path, lock_path = root / 'queue_status.json', root / 'queue_invocation.lock'
    if status_path.exists() or lock_path.exists():
        raise FileExistsError('Queue already invoked; preserve status and use explicit worker resume only')
    with lock_path.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), started_at=p.now()), handle)
    stage1 = [(variant, 42) for variant in cfg['variants']]
    smoke_seconds = float(read_json(root / 'smoke_gate.json')['smoke_gpu_seconds'])
    if not math.isfinite(smoke_seconds) or smoke_seconds < 0:
        raise ValueError('Invalid smoke GPU budget receipt')
    value = dict(status='RUNNING', pid=os.getpid(), stage=1, jobs=stage1, completed=[],
        admission=None, started_at=p.now(), maximum_gpu_hours=48,
        smoke_gpu_seconds=smoke_seconds,
        worker_gpu_seconds=0., waiting_total_seconds=0., gpu_hours_elapsed=smoke_seconds / 3600,
        execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
        formal_eligible=False, test_evaluated=False)
    p.dump(status_path, value)

    def run_jobs(jobs):
        for variant, seed in jobs:
            elapsed = (smoke_seconds + value['worker_gpu_seconds']) / 3600
            remaining = cfg['queue']['maximum_gpu_hours'] - elapsed
            if remaining <= 0:
                raise TimeoutError('Total 48 GPU-hour budget reached; no automatic continuation')
            waited_before = value['waiting_total_seconds']

            def report_wait(receipt):
                value.update(status='WAITING_RESOURCE', waiting_for=[variant, seed],
                    active=None, child_pid=None, resource_receipt=receipt,
                    waiting_total_seconds=waited_before + receipt['waiting_seconds'],
                    gpu_hours_elapsed=elapsed, updated_at=p.now())
                p.dump(status_path, value)

            admission = wait_for_gpu(cfg, device, on_wait=report_wait,
                maximum_wait_seconds=max(0., float(cfg['queue'].get('resource_wait_maximum_hours', 6))
                                            * 3600 - waited_before))
            value.update(status='RUNNING', admission=admission, waiting_for=None,
                         waiting_total_seconds=waited_before + admission['waiting_seconds'],
                         updated_at=p.now())
            p.dump(status_path, value)
            folder = result_dir(root, variant, seed)
            if folder.exists():
                raise FileExistsError(f'Existing invocation cannot be automatically retried: {folder}')
            folder.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--config', str(config),
                       '--output', str(root), '--variant', variant, '--seed', str(seed), '--device', device]
            with (folder / 'stdout.log').open('x', encoding='utf-8') as handle:
                tick_child = time.perf_counter()
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                value.update(active=[variant, seed], child_pid=child.pid, updated_at=p.now(),
                             gpu_hours_elapsed=elapsed)
                p.dump(status_path, value)
                try:
                    code = child.wait(timeout=min(float(cfg['queue']['maximum_hours_per_run']), remaining) * 3600)
                except subprocess.TimeoutExpired:
                    # This PID was created here; no other user's process is touched.
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    p.dump(folder / 'status.json', dict(status='TIMEOUT_NO_AUTORETRY',
                        variant=variant, seed=seed, updated_at=p.now(), formal_eligible=False))
                    raise RuntimeError('Owned child run exceeded registered deadline/budget')
                finally:
                    if device == 'cuda':
                        value['worker_gpu_seconds'] += time.perf_counter() - tick_child
                    value['gpu_hours_elapsed'] = (smoke_seconds + value['worker_gpu_seconds']) / 3600
                    p.dump(status_path, value)
            if code:
                if not (folder / 'status.json').exists():
                    p.dump(folder / 'status.json', dict(status='FAILED_NO_AUTORETRY',
                        variant=variant, seed=seed, exit_code=code, updated_at=p.now(), formal_eligible=False))
                if read_json(folder / 'status.json')['status'] == 'RESOURCE_UNAVAILABLE_NO_AUTORETRY':
                    raise ResourceUnavailable('Another GPU1 owner arrived after admission; worker evidence preserved, no automatic retry')
                raise RuntimeError(f'Child exited {code}; campaign stopped with no automatic retry')
            run_result(root, variant, seed, config)
            value['completed'].append([variant, seed])
            value.update(active=None, child_pid=None, updated_at=p.now())
            p.dump(status_path, value)

    try:
        run_jobs(stage1)
        rows = [run_result(root, variant, seed, config) for variant, seed in stage1]
        selection = select_confirmation(rows, cfg)
        selection['execution_manifest_sha256'] = p.sha(root / 'execution_manifest.json')
        selection['stage1_result_sha256'] = {variant: p.sha(result_dir(root, variant, 42) / 'result.json')
                                             for variant in cfg['variants']}
        p.dump(root / 'stage1_selection.json', selection)
        value.update(stage=2, stage1_selection=selection, jobs=stage1 + selection['jobs'])
        p.dump(status_path, value)
        run_jobs(selection['jobs'])
        summary = summarize(root, config)
        value.update(status='COMPLETE', active=None, child_pid=None,
            confirmation_decisions=summary['confirmation_decisions'], updated_at=p.now(),
            gpu_hours_elapsed=(smoke_seconds + value['worker_gpu_seconds']) / 3600)
        p.dump(status_path, value)
        return value
    except Exception as error:
        state = ('RESOURCE_WAIT_TIMEOUT_NO_AUTORETRY' if isinstance(error, ResourceWaitTimeout)
                 else 'RESOURCE_UNAVAILABLE_NO_AUTORETRY' if isinstance(error, ResourceUnavailable)
                 else 'GPU_BUDGET_EXHAUSTED_NO_CONTINUATION' if isinstance(error, TimeoutError)
                 else 'FAILED_NO_AUTORETRY')
        value.update(status=state, error=str(error), error_type=type(error).__name__,
            active=None, child_pid=None, updated_at=p.now(),
            gpu_hours_elapsed=(smoke_seconds + value['worker_gpu_seconds']) / 3600)
        p.dump(status_path, value)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'smoke', 'worker', 'queue', 'status', 'summarize'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--variant', default='control')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    root = (args.output or ROOT / cfg['result_root']).resolve()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        prepare(root, args.config, args.device)
        result = dict(status='PREPARED', formal_eligible=False)
    elif args.mode == 'verify':
        verify_manifest(root, args.config)
        result = dict(status='SOURCE_DATA_PREFLIGHT_PASS', formal_eligible=False)
    elif args.mode == 'smoke':
        result = smoke(root, args.device, args.config)
    elif args.mode == 'worker':
        report = run_training(root, args.variant, args.seed, args.device, args.resume, args.config)
        result = dict(status='COMPLETE', binding=report['binding'], metrics=report['metrics'])
    elif args.mode == 'queue':
        result = queue(root, args.device, args.config)
    elif args.mode == 'status':
        result = status(root, args.config)
    else:
        result = summarize(root, args.config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
