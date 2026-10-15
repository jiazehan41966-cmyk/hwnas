"""Frozen three-seed C01b runner; candidate smoke is separate from authorization.

The completed C01 archive is verified read-only using its own frozen runner.
New results belong only to this campaign. No NAS, HLS, board or test run occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'results/sonar_yolov9_baseline_20261005/env_overlay'
if OVERLAY.is_dir():
    sys.path.insert(0, str(OVERLAY))
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.update(YOLOv5_AUTOINSTALL='false', YOLOM='10', YOLOA='0.5', YOLOB='6.0')
import torch
import yaml

from tools.sonar_head_factorial import train as t
from tools.sonar_head_c01b import analysis, model as adapter

p = t.p
read_json = t.read_json
VERSION = 'sonar_head_c01b_20261013_01'
CONFIG = ROOT / 'configs/sonar_head_c01b_20261013.yaml'
PREREGISTRATION = ROOT / 'results' / VERSION / 'PREREGISTRATION.md'
SEEDS = (42, 43, 44)
STABLE_EPOCHS = t.STABLE_EPOCHS
CHOICE = 'use_completed_C01_regmax16_66_channels'
WITNESSES = ('augmented_batches_sha256', 'image_order_sha256', 'valid_batches_sha256')


def load_config(config=CONFIG):
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    old, baseline = t.load_config()
    expected_train = {key: old['training'][key] for key in (
        'epochs', 'batch_size', 'input', 'optimizer', 'lr', 'weight_decay',
        'scheduler', 'amp', 'workers', 'ema', 'accumulation', 'save_prediction_epochs')}
    expected_variant = dict(backbone='E0', head='quality_aware', reg_max=1,
        use_dfl=False, raw_output_channels_per_scale=[6] * 4, box_loss='GIoU',
        distance_mapping='ReLU_nonnegative_same_in_training_and_decode',
        preserve_regression_hidden_width=64, preserve_classification_hidden_width=48,
        preserve_shared_head_initialization=True)
    if (cfg.get('version') != VERSION or cfg.get('seeds') != list(SEEDS)
            or cfg.get('baseline') != old['baseline']
            or cfg.get('environment_spec') != old['environment_spec']
            or cfg.get('training') != expected_train or cfg.get('C01b') != expected_variant
            or cfg.get('C01_variant') != 'E0__quality_aware'
            or cfg.get('C01_reg_max') != 16
            or cfg.get('C01_raw_output_channels_per_scale') != [66] * 4
            or cfg.get('C01_box_loss') != 'CIoU'
            or cfg.get('subset_contract') != old['subset_contract']
            or cfg.get('subset_contract_sha256') != old['subset_contract_sha256']
            or cfg.get('gpu_index') != 1
            or cfg.get('queue') != {'maximum_hours_per_run': 2.0, 'retry': 'none'}
            or cfg.get('test_policy') != 'sealed_not_mounted'
            or cfg.get('result_root') != f'results/{VERSION}/remote'):
        raise ValueError('Unregistered C01b model, data, training or queue contract')
    decision = cfg['decision']
    analysis.validate_decision(decision)
    if (decision.get('automatic_search') is not False
            or decision.get('on_pass') != 'select_C01b_direct_distance_6_channels'
            or decision.get('on_fail') != 'retain_actual_frozen_C01_DFL_channels'):
        raise ValueError('C01b decision must preserve the frozen channel gate and no automatic search')
    if not isinstance(cfg.get('C01_execution_manifest_sha256'), str) or len(cfg['C01_execution_manifest_sha256']) != 64:
        raise ValueError('Exact frozen C01 manifest SHA256 required')
    t.load_subset_contract(cfg)
    return cfg, baseline


def source_paths(cfg, config=CONFIG):
    old, _ = t.load_config()
    paths = set(t.source_paths(old, t.CONFIG))
    paths.update((ROOT / 'tools/sonar_head_c01b').glob('*.py'))
    paths.update((Path(config).resolve(), PREREGISTRATION))
    readme = ROOT / 'tools/sonar_head_c01b/README.md'
    if readme.is_file():
        paths.add(readme)
    for name in ('test_sonar_head_c01b_model.py', 'test_sonar_head_c01b_train.py',
                 'test_sonar_head_c01b_analysis.py'):
        path = ROOT / 'tests' / name
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def source_hashes(cfg, config=CONFIG):
    return {path.relative_to(ROOT).as_posix(): p.sha(path) for path in source_paths(cfg, config)}


def result_dir(root, seed, mode='formal'):
    return t.result_dir(Path(root), 'E0', 'quality_aware', seed, mode)


def _resolution(root, cfg, *, resolved=False):
    value = read_json(Path(root) / 'baseline_resolution.json')
    if (value.get('C01_execution_manifest_sha256') != cfg['C01_execution_manifest_sha256']
            or value.get('C01_reg_max') != 16
            or value.get('C01_raw_output_channels_per_scale') != [66] * 4):
        raise ValueError('Baseline resolution points to a different C01 candidate')
    if resolved and (value.get('status') != 'RESOLVED' or value.get('choice') != CHOICE
                     or not isinstance(value.get('user_authorization'), str)
                     or not value['user_authorization'].strip()):
        raise ValueError('Formal C01b requires the user-resolved actual 66-channel C01 baseline')
    if value.get('status') not in ('UNRESOLVED', 'RESOLVED'):
        raise ValueError('Unknown baseline resolution state')
    return value


def measure_head_contract(model, expected_channels):
    measured = [reg[-1].out_channels + cls[-1].out_channels
                for reg, cls in zip(model.head.cv2, model.head.cv3)]
    if (model.levels != ('P2', 'P3', 'P4', 'P5') or measured != [expected_channels] * 4
            or model.head.no != expected_channels
            or [tower[0].conv.out_channels for tower in model.head.cv2] != [64] * 4
            or [tower[0].conv.out_channels for tower in model.head.cv3] != [48] * 4):
        raise ValueError('Measured model head channels or hidden widths differ from the contract')
    final_regression = [dict(input_channels=conv.in_channels, output_channels=conv.out_channels,
        groups=conv.groups, kernel_size=list(conv.kernel_size), weight_shape=list(conv.weight.shape),
        parameters=sum(parameter.numel() for parameter in conv.parameters()))
        for conv in (tower[-1] for tower in model.head.cv2)]
    final_classification = [dict(input_channels=conv.in_channels, output_channels=conv.out_channels,
        groups=conv.groups, kernel_size=list(conv.kernel_size), weight_shape=list(conv.weight.shape),
        parameters=sum(parameter.numel() for parameter in conv.parameters()))
        for conv in (tower[-1] for tower in model.head.cv3)]
    return dict(scales=list(model.levels), raw_output_channels_per_scale=measured,
                reg_max=model.head.reg_max, regression_hidden_channels=[64] * 4,
                classification_hidden_channels=[48] * 4,
                final_regression_convolutions=final_regression,
                final_classification_convolutions=final_classification,
                measurement='actual initialized module final-convolution out_channels and head.no')


def validate_head_info(info, expected_channels):
    head = info.get('head_contract', {})
    if (head.get('raw_output_channels_per_scale') != [expected_channels] * 4
            or head.get('reg_max') != (16 if expected_channels == 66 else 1)
            or head.get('scales') != ['P2', 'P3', 'P4', 'P5']
            or head.get('regression_hidden_channels') != [64] * 4
            or head.get('classification_hidden_channels') != [48] * 4
            or not head.get('measurement')):
        raise ValueError('Actual measured raw channel metadata is required for every comparison row')


def _reference_evidence(folder, row):
    names = set(row['hashes']) | {'model_info.json', 'initialization_identity.json',
                                'training_rng_identity.json', 'run_binding.json'}
    for epoch in STABLE_EPOCHS:
        names.update((f'epoch_predictions/epoch_{epoch:03d}_valid_records.json',
                      f'epoch_predictions/epoch_{epoch:03d}_receipt.json'))
    return {name: p.sha(folder / name) for name in sorted(names)}


def _verify_c01(root, cfg):
    if p.sha(root / 'execution_manifest.json') != cfg['C01_execution_manifest_sha256']:
        raise ValueError('Original C01 execution manifest hash differs')
    # This preserves the old config/source receipt closure and uses its own gate.
    old, _ = t.verify_manifest(root, t.CONFIG)
    return old, read_json(root / 'execution_manifest.json')


def collect_c01_rows(root, cfg, comparison_sha):
    from tools.sonar_head_factorial import model as reference_adapter
    old, _ = _verify_c01(root, cfg)
    rows = []
    for seed in SEEDS:
        row = deepcopy(t.run_result(root, 'E0', 'quality_aware', seed, write=False, config=t.CONFIG))
        reference = reference_adapter.build_model('E0', 'quality_aware', seed,
                         source_root=ROOT / old['author']['source_root'])
        if reference.initialization_identity() != row['initialization']:
            raise ValueError('Archived C01 initialization differs from its frozen reconstruction')
        row['model_info']['head_contract'] = measure_head_contract(reference, 66)
        validate_head_info(row['model_info'], 66)
        folder = result_dir(root, seed)
        row['comparison_contract_sha256'] = comparison_sha
        row['C01_reference_evidence_sha256'] = _reference_evidence(folder, row)
        row['training_rng_identity'] = read_json(folder / 'training_rng_identity.json')
        rows.append(row)
        del reference
    return rows


def prepare(root, config=CONFIG, C01_result_root=None):
    root = Path(root)
    cfg, baseline = load_config(config)
    original_root = Path(C01_result_root or cfg['C01_result_root']).resolve()
    if root.resolve() == original_root:
        raise ValueError('C01b output must be isolated from the frozen C01 root')
    old, original = _verify_c01(original_root, cfg)
    subset = t.load_subset_contract(cfg, original['inputs'])
    comparison = dict(schema_version=1, C01_execution_manifest_sha256=cfg['C01_execution_manifest_sha256'],
        input_identity_sha256=p.canonical_sha(original['inputs']),
        validation_target_sha256=original['validation_target_sha256'],
        subset_contract_sha256=cfg['subset_contract_sha256'], seeds=list(SEEDS),
        training=cfg['training'], evaluation=old['evaluation'],
        stable_epochs=list(STABLE_EPOCHS), C01_reg_max=16, C01_raw_output_channels_per_scale=[66] * 4,
        C01b=cfg['C01b'], decision=cfg['decision'], test_evaluated=False)
    comparison_sha = p.canonical_sha(comparison)
    rows = collect_c01_rows(original_root, cfg, comparison_sha)
    manifest = dict(version=VERSION, config=cfg, config_sha256=p.sha(config),
        source_sha256=source_hashes(cfg, config), inputs=original['inputs'],
        validation_targets=original['validation_targets'],
        validation_target_sha256=original['validation_target_sha256'],
        subset_contract_sha256=cfg['subset_contract_sha256'],
        C01_result_root=str(original_root), C01_execution_manifest_sha256=p.sha(original_root / 'execution_manifest.json'),
        C01_environment_lock_sha256=p.sha(original_root / 'environment_lock.json'),
        environment_fingerprint=read_json(original_root / 'environment_lock.json')['fingerprint'],
        C01_reference_sha256={str(row['binding']['seed']): p.canonical_sha(row) for row in rows},
        comparison_contract=comparison, comparison_contract_sha256=comparison_sha,
        preregistration_sha256=p.sha(PREREGISTRATION), baseline_resolution_frozen_separately_for_formal=True,
        authorization_policy='mutable resolution outside source freeze; resolved digest frozen separately before formal',
        test_evaluated=False, automatic_search=False)
    path = root / 'execution_manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Existing C01b freeze differs; preserve it and use a new result root')
    # Check all inputs first; then write only the new campaign's files.
    root.mkdir(parents=True, exist_ok=True)
    for source, destination in ((PREREGISTRATION, root / 'PREREGISTRATION.md'),
                               (original_root / 'environment_lock.json', root / 'environment_lock.json')):
        if destination.exists() and p.sha(destination) != p.sha(source):
            raise ValueError(f'Existing candidate file differs: {destination.name}')
        if not destination.exists():
            shutil.copyfile(source, destination)
    resolution = root / 'baseline_resolution.json'
    if not resolution.exists():
        template = PREREGISTRATION.parent / 'baseline_resolution.json'
        if template.is_file() and template.resolve() != resolution.resolve():
            shutil.copyfile(template, resolution)
        else:
            p.dump(resolution, dict(status='UNRESOLVED', choice=None, user_authorization=None,
                candidate=CHOICE, C01_execution_manifest_sha256=cfg['C01_execution_manifest_sha256'],
                C01_reg_max=16, C01_raw_output_channels_per_scale=[66] * 4))
    _resolution(root, cfg)
    for row in rows:
        target = root / 'C01_reference' / f"seed_{row['binding']['seed']}.json"
        if target.exists() and read_json(target) != row:
            raise ValueError('Existing C01 reference copy differs')
        if not target.exists():
            p.dump(target, row)
    if not path.exists():
        p.dump(path, manifest)
    return manifest


def verify_manifest(root, config=CONFIG, C01_result_root=None):
    root = Path(root)
    cfg, baseline = load_config(config)
    manifest = read_json(root / 'execution_manifest.json')
    original_root = Path(C01_result_root or manifest['C01_result_root']).resolve()
    _, original = _verify_c01(original_root, cfg)
    if (str(original_root) != manifest['C01_result_root']
            or manifest['config'] != cfg or manifest['config_sha256'] != p.sha(config)
            or manifest['source_sha256'] != source_hashes(cfg, config)
            or manifest['inputs'] != original['inputs']
            or manifest['validation_targets'] != original['validation_targets']
            or manifest['validation_target_sha256'] != p.canonical_sha(manifest['validation_targets'])
            or manifest['comparison_contract_sha256'] != p.canonical_sha(manifest['comparison_contract'])
            or manifest['preregistration_sha256'] != p.sha(PREREGISTRATION)
            or manifest['preregistration_sha256'] != p.sha(root / 'PREREGISTRATION.md')
            or manifest['C01_environment_lock_sha256'] != p.sha(original_root / 'environment_lock.json')
            or manifest['C01_environment_lock_sha256'] != p.sha(root / 'environment_lock.json')
            or manifest['environment_fingerprint'] != read_json(root / 'environment_lock.json')['fingerprint']):
        raise ValueError('Frozen C01b source/config/data/reference/environment/preregistration drift')
    t.load_subset_contract(cfg, manifest['inputs'])
    _resolution(root, cfg)
    for seed in SEEDS:
        row = read_json(root / 'C01_reference' / f'seed_{seed}.json')
        if (p.canonical_sha(row) != manifest['C01_reference_sha256'][str(seed)]
                or row['comparison_contract_sha256'] != manifest['comparison_contract_sha256']):
            raise ValueError('Frozen C01 reference-copy hash drift')
        validate_head_info(row['model_info'], 66)
        for name, expected in row['C01_reference_evidence_sha256'].items():
            if p.sha(result_dir(original_root, seed) / name) != expected:
                raise ValueError('Frozen C01 checkpoint/prediction/reference evidence changed')
    return cfg, baseline


def reference_row(root, seed):
    return read_json(Path(root) / 'C01_reference' / f'seed_{seed}.json')


def verify_paired_epoch(row, reference, epoch):
    if row.get('epoch') != epoch or reference.get('epoch') != epoch:
        raise ValueError('Paired epoch index differs')
    for key in WITNESSES + ('batches', 'images'):
        if not row.get(key) or row.get(key) != reference.get(key):
            raise ValueError(f'Paired C01 training/validation data witness differs: epoch{epoch}/{key}')
    if row.get('batch_sha256') != reference.get('batch_sha256') or not row.get('batch_sha256'):
        raise ValueError(f'Paired augmented batch hashes differ: epoch{epoch}')
    if row.get('losses', {}).get('dfl_loss') != 0.0:
        raise ValueError('C01b must keep DFL loss exactly zero')


def verify_pairing_history(history, reference):
    t.validate_history(history, len(history))
    for index, row in enumerate(history):
        verify_paired_epoch(row, reference['history'][index], index + 1)


def binding(root, seed, mode):
    manifest = read_json(Path(root) / 'execution_manifest.json')
    result = dict(execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
        environment_fingerprint=manifest['environment_fingerprint'],
        subset_contract_sha256=manifest['subset_contract_sha256'],
        validation_target_sha256=manifest['validation_target_sha256'],
        comparison_contract_sha256=manifest['comparison_contract_sha256'],
        C01_execution_manifest_sha256=manifest['C01_execution_manifest_sha256'],
        backbone='E0', head='quality_aware', variant='C01b', seed=seed, mode=mode,
        test_evaluated=False)
    if mode == 'formal':
        authorization = verify_formal_authorization(root)
        result.update(baseline_resolution_sha256=authorization['baseline_resolution_sha256'],
                      independent_review_sha256=authorization['independent_review_sha256'],
                      smoke_gate_sha256=authorization['smoke_gate_sha256'],
                      preflight_complete_sha256=authorization['preflight_complete_sha256'],
                      formal_authorization_sha256=p.sha(Path(root) / 'formal_authorization.json'))
    return result


def make_run(root, seed, device, mode, config=CONFIG):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered C01b seed')
    cfg, baseline = verify_manifest(root, config)
    if device == 'cuda' and (not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') != '1'):
        raise RuntimeError('C01b requires CUDA_VISIBLE_DEVICES=1 and available CUDA; no fallback')
    if mode == 'formal' and device != 'cuda':
        raise ValueError('Formal C01b must use the original GPU1 CUDA environment')
    p.seed_all(seed)
    lock = t.bind_environment(Path(root), device, cfg)
    original_cfg, _ = t.load_config()
    detector = adapter.build_model(seed, source_root=ROOT / original_cfg['author']['source_root']).to(device)
    criterion = adapter.build_criterion(detector, baseline)
    identity = detector.initialization_identity()
    reference = reference_row(root, seed)
    if (identity['c01_reference'] != reference['initialization']
            or identity['shared_head_sha256'] != identity['c01_shared_head_sha256']
            or lock['fingerprint'] != reference['binding']['environment_fingerprint']):
        raise ValueError('C01b initialization or environment is not paired with archived C01')
    metadata = detector.metadata()
    metadata['criterion'] = adapter.criterion_metadata(detector, criterion)
    metadata['head_contract'] = measure_head_contract(detector, 6)
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    initial = p.state_hash(detector)
    p.seed_all(seed)
    rng_identity = p.training_rng_identity(seed)
    if rng_identity != reference['training_rng_identity']:
        raise ValueError('Post-construction training RNG differs from the paired C01 seed')
    train, valid = p.make_loaders(baseline, seed)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=.001, weight_decay=.0001)
    folder = result_dir(root, seed, mode)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, content in (('initialization_identity.json', identity), ('model_info.json', metadata),
                              ('training_rng_identity.json', rng_identity)):
        path = folder / filename
        if path.exists() and read_json(path) != content:
            raise ValueError(f'Existing C01b run identity drift: {filename}')
        if not path.exists():
            p.dump(path, content)
    return folder, cfg, baseline, detector, criterion, optimizer, train, valid, initial, identity


def smoke_one(root, seed, device, config=CONFIG):
    cfg, _ = verify_manifest(root, config)
    folder = result_dir(root, seed, 'smoke')
    if folder.exists():
        raise FileExistsError('Preserve the existing smoke; no overwrite or automatic retry')
    run = make_run(root, seed, device, 'smoke', config)
    folder, _, baseline, model, criterion, optimizer, train, valid, initial, identity = run
    try:
        trained = t.train_epoch(model, criterion, optimizer, train, device, 2)
        metrics, records, valid_sha = t.evaluate(model, criterion, valid, baseline, device, 1)
        reference = reference_row(root, seed)
        if (trained['batch_sha256'] != reference['history'][0]['batch_sha256'][:2]
                or trained['batches'] != 2 or trained['images'] != 8
                or trained['losses']['dfl_loss'] != 0.0 or metrics['num_images'] != 4
                or p.state_hash(model) == initial):
            raise ValueError('Candidate smoke lacks paired augmented batches, gradients or valid4')
        original_root = Path(read_json(Path(root) / 'execution_manifest.json')['C01_result_root'])
        original_smoke = read_json(result_dir(original_root, 42, 'smoke') / 'smoke_report.json')
        if valid_sha != original_smoke['valid_batches_sha256']:
            raise ValueError('Candidate validation smoke differs from original C01 valid4 pixels/GT')
        report = dict(status='PASS', binding=binding(root, seed, 'smoke'), training=trained,
            valid=metrics, valid_batches_sha256=valid_sha, initialization=identity,
            initial_state_sha256=initial, final_state_sha256=p.state_hash(model),
            provisional_candidate=True, baseline_resolution_status=_resolution(root, cfg)['status'],
            model_info_sha256=p.sha(folder / 'model_info.json'), formal_eligible=False, test_evaluated=False)
        p.dump(folder / 'valid_records.json', records)
        p.dump(folder / 'smoke_report.json', report)
        return report
    except Exception as error:
        p.dump(folder / 'smoke_failure.json', dict(status='FAILED', error=str(error),
               error_type=type(error).__name__, formal_eligible=False))
        raise
    finally:
        del run, model, criterion, optimizer, train, valid
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def smoke(root, device, config=CONFIG):
    root = Path(root)
    verify_manifest(root, config)
    if (root / 'smoke_gate.json').exists():
        raise FileExistsError('Candidate smoke gate already exists; no automatic retry')
    reports = [smoke_one(root, seed, device, config) for seed in SEEDS]
    if len({report['valid_batches_sha256'] for report in reports}) != 1:
        raise ValueError('Three candidate seeds must use identical valid4 batches')
    gate = dict(status='PASS', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
        environment_fingerprint=read_json(root / 'environment_lock.json')['fingerprint'],
        smoke_report_sha256={str(seed): p.sha(result_dir(root, seed, 'smoke') / 'smoke_report.json') for seed in SEEDS},
        provisional_candidate=True, candidate='actual_C01_66_to_C01b_6', seeds=list(SEEDS),
        actual_train_batches_per_seed=2, actual_validation_images_per_seed=4,
        paired_augmented_batches=True, paired_initial_backbone_neck_shared_head=True,
        formal_eligible=False, test_evaluated=False)
    p.dump(root / 'smoke_gate.json', gate)
    return gate


def require_smoke(root):
    root = Path(root)
    gate = read_json(root / 'smoke_gate.json')
    if (gate.get('status') != 'PASS'
            or gate.get('execution_manifest_sha256') != p.sha(root / 'execution_manifest.json')
            or gate.get('environment_fingerprint') != read_json(root / 'environment_lock.json')['fingerprint']
            or gate.get('seeds') != list(SEEDS)
            or set(gate.get('smoke_report_sha256', {})) != set(map(str, SEEDS))):
        raise ValueError('Candidate three-seed smoke gate is missing, changed or unpaired')
    valid_witnesses = []
    for seed in SEEDS:
        folder = result_dir(root, seed, 'smoke')
        report = read_json(folder / 'smoke_report.json')
        if (gate['smoke_report_sha256'][str(seed)] != p.sha(folder / 'smoke_report.json')
                or report.get('status') != 'PASS' or report['binding'] != binding(root, seed, 'smoke')
                or report.get('model_info_sha256') != p.sha(folder / 'model_info.json')
                or report['training']['batch_sha256'] != reference_row(root, seed)['history'][0]['batch_sha256'][:2]
                or report['training']['batches'] != 2 or report['training']['images'] != 8
                or report['training']['losses']['dfl_loss'] != 0.0 or report['valid']['num_images'] != 4
                or report['initialization']['c01_reference'] != reference_row(root, seed)['initialization']
                or report['initialization']['shared_head_sha256'] != report['initialization']['c01_shared_head_sha256']
                or report['initial_state_sha256'] == report['final_state_sha256']):
            raise ValueError(f'Candidate smoke artifact is changed or unpaired: seed{seed}')
        validate_head_info(read_json(folder / 'model_info.json'), 6)
        valid_witnesses.append(report['valid_batches_sha256'])
    if len(set(valid_witnesses)) != 1:
        raise ValueError('Candidate smoke valid4 pixels/GT differ across seeds')
    return gate


def verify_formal_authorization(root, *, freeze=False):
    root = Path(root)
    manifest = read_json(root / 'execution_manifest.json')
    _resolution(root, manifest['config'], resolved=True)
    require_smoke(root)
    preflight = read_json(root / 'preflight_complete.json')
    if (preflight.get('status') != 'PASS'
            or preflight.get('execution_manifest_sha256') != p.sha(root / 'execution_manifest.json')):
        raise ValueError('Actual completed preflight PASS for this exact C01b manifest is required')
    review = read_json(root / 'independent_review.json')
    if (review.get('status') != 'PASS' or review.get('independent') is not True
            or review.get('scope') != 'C01b_66_to_6_readiness'
            or review.get('execution_manifest_sha256') != p.sha(root / 'execution_manifest.json')):
        raise ValueError('Independent review PASS for this exact frozen C01b campaign is required')
    if (review.get('smoke_gate_sha256') != p.sha(root / 'smoke_gate.json')
            or review.get('preflight_complete_sha256') != p.sha(root / 'preflight_complete.json')):
        raise ValueError('Independent review actual smoke/preflight digest differs or is missing')
    expected = dict(status='AUTHORIZED', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
        baseline_resolution_sha256=p.sha(root / 'baseline_resolution.json'),
        independent_review_sha256=p.sha(root / 'independent_review.json'),
        smoke_gate_sha256=p.sha(root / 'smoke_gate.json'),
        preflight_complete_sha256=p.sha(root / 'preflight_complete.json'), test_evaluated=False, automatic_search=False)
    path = root / 'formal_authorization.json'
    if path.exists():
        if read_json(path) != expected:
            raise ValueError('Frozen formal authorization resolution/review/smoke digest changed')
    elif freeze:
        p.dump(path, expected)
    else:
        raise ValueError('Formal authorization has not been frozen')
    return expected


def run_result(root, seed, *, write=False, config=CONFIG):
    root = Path(root)
    cfg, baseline = verify_manifest(root, config)
    verify_formal_authorization(root)
    if seed not in SEEDS:
        raise ValueError('Unregistered C01b result seed')
    folder = result_dir(root, seed)
    expected = binding(root, seed, 'formal')
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = latest['history']
    t.validate_history(history)
    verify_pairing_history(history, reference_row(root, seed))
    best = max(history, key=lambda row: row['valid']['mAP50_95'])
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    info = read_json(folder / 'model_info.json')
    identity = read_json(folder / 'initialization_identity.json')
    if (latest['binding'] != expected or latest['epoch'] != 80
            or checkpoint['binding'] != expected or checkpoint['epoch'] != best['epoch']
            or checkpoint['history'] != history[:best['epoch']]
            or read_json(folder / 'epoch_log.json') != history
            or read_json(folder / 'best_valid_metrics.json') != best['valid']
            or latest['initialization'] != identity
            or identity['c01_reference'] != reference_row(root, seed)['initialization']
            or identity['shared_head_sha256'] != identity['c01_shared_head_sha256']
            or latest['model_info_sha256'] != p.sha(folder / 'model_info.json')
            or checkpoint['model_info_sha256'] != p.sha(folder / 'model_info.json')
            or info.get('quality_package', {}).get('box') != 'GIoU'
            or info.get('quality_package', {}).get('distribution') != 'disabled'):
        raise ValueError('C01b checkpoint/binding/history/model/initialization evidence differs')
    validate_head_info(info, 6)
    contract = t.load_subset_contract(cfg, read_json(root / 'execution_manifest.json')['inputs'])
    for name in ('best_valid_records.json', 'terminal_valid_records.json'):
        t.validate_prediction_records(read_json(folder / name), contract['full_images'], expected['validation_target_sha256'])
    best_records = read_json(folder / 'best_valid_records.json')
    if t.enrich_metrics(p.metrics_from_records(best_records, baseline)) != best['valid']:
        raise ValueError('C01b best prediction metrics do not replay')
    terminal = t.enrich_metrics(p.metrics_from_records(read_json(folder / 'terminal_valid_records.json'), baseline))
    if terminal != history[-1]['valid'] or terminal != read_json(folder / 'terminal_valid_metrics.json'):
        raise ValueError('C01b terminal prediction metrics do not replay')
    stable = t.verify_stable_predictions(folder, history, expected, baseline, contract['full_images'])
    if set(stable['paths']) != {str(epoch) for epoch in STABLE_EPOCHS}:
        raise ValueError('Formal completion requires all five full99 prediction archives')
    stable_metrics = t.stable_metrics_from_history(history)
    report = dict(binding=expected, epochs=80, best_epoch=best['epoch'],
        primary_checkpoint='mean_epochs_76_80', primary_metric='mAP50_95',
        metrics=stable_metrics, stable_metrics=stable_metrics,
        stable_prediction_paths=stable['paths'], stable_prediction_sha256=stable['sha256'],
        stable_epoch_metrics=stable['metrics'], secondary_terminal_metrics=terminal,
        secondary_best_metrics=best['valid'], initialization=identity,
        initial_state_sha256=latest['initial_state_sha256'], model_info=info, history=history,
        comparison_contract_sha256=read_json(root / 'execution_manifest.json')['comparison_contract_sha256'],
        hashes={name: p.sha(folder / name) for name in ('latest.pt', 'best.pt', 'epoch_log.json',
            'model_info.json', 'best_valid_records.json', 'terminal_valid_records.json', 'terminal_valid_metrics.json')},
        formal_eligible=True, test_evaluated=False)
    if write:
        p.dump(folder / 'result.json', report)
    return report


def run_training(root, seed, device, resume=False, config=CONFIG):
    root = Path(root)
    verify_manifest(root, config)
    verify_formal_authorization(root, freeze=True)
    folder = result_dir(root, seed)
    if (folder / 'run_binding.json').exists() and not resume:
        raise FileExistsError('Existing formal C01b invocation requires explicit epoch-boundary resume')
    run = make_run(root, seed, device, 'formal', config)
    folder, cfg, baseline, model, criterion, optimizer, train, valid, initial, identity = run
    expected = binding(root, seed, 'formal')
    expected_images = t.load_subset_contract(cfg)['full_images']
    reference = reference_row(root, seed)
    history, start = [], 0
    if resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        start, history = t.validate_resume_state(folder, state, expected, initial, identity, baseline, expected_images)
        verify_pairing_history(history, reference)
        if state['model_info_sha256'] != p.sha(folder / 'model_info.json'):
            raise ValueError('Resume model contract changed')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        p.restore_rng(state['rng'], train)
        if valid.generator is not None:
            valid.generator.set_state(state['valid_loader_generator'])
    else:
        p.dump(folder / 'run_binding.json', dict(binding=expected, initialization=identity,
            initial_state_sha256=initial, started_at=p.now(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        for epoch in range(start, 80):
            tick = time.perf_counter()
            previous_best = max((row['valid']['mAP50_95'] for row in history), default=-1.)
            record = t.train_epoch(model, criterion, optimizer, train, device)
            metrics, records, valid_sha = t.evaluate(model, criterion, valid, baseline, device)
            record.update(epoch=epoch + 1, valid=metrics, valid_batches_sha256=valid_sha,
                          wall_seconds=time.perf_counter() - tick)
            # Abort before committing any epoch whose actual data differs from C01.
            verify_paired_epoch(record, reference['history'][epoch], epoch + 1)
            if epoch + 1 in STABLE_EPOCHS:
                record['prediction_archive'] = t.save_stable_predictions(folder, epoch + 1, records, expected, expected_images)
            history.append(record)
            t.validate_history(history, epoch + 1)
            state = dict(binding=expected, initial_state_sha256=initial, initialization=identity,
                model_info_sha256=p.sha(folder / 'model_info.json'), model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), epoch=epoch + 1, rng=p.rng_state(train),
                history=history, valid_loader_generator=valid.generator.get_state() if valid.generator is not None else None,
                test_evaluated=False)
            if metrics['mAP50_95'] > previous_best:
                p.atomic_save(folder / 'best.pt', state)
                p.dump(folder / 'best_valid_metrics.json', metrics)
                p.dump(folder / 'best_valid_records.json', records)
            if epoch + 1 == 80:
                p.dump(folder / 'terminal_valid_metrics.json', metrics)
                p.dump(folder / 'terminal_valid_records.json', records)
            p.atomic_save(folder / 'latest.pt', state)
            p.dump(folder / 'epoch_log.json', history)
            p.dump(folder / 'status.json', dict(status='RUNNING', epoch=epoch + 1, binding=expected,
                latest_valid={key: metrics[key] for key in t.METRICS}, paired_data_pass=True, updated_at=p.now()))
            print(f'C01b seed={seed} epoch={epoch + 1}/80 mAP50_95={metrics["mAP50_95"]:.8f} '
                  f'AP75={metrics["AP75"]:.8f} paired_data=PASS seconds={record["wall_seconds"]:.2f}', flush=True)
        report = run_result(root, seed, write=True, config=config)
        p.dump(folder / 'status.json', dict(status='COMPLETE', epochs=80, binding=expected,
            stable_metrics=report['stable_metrics'], stable_prediction_epochs=list(STABLE_EPOCHS), updated_at=p.now()))
        return report
    except Exception as error:
        p.dump(folder / 'status.json', dict(status='FAILED_NO_AUTORETRY', completed_epochs=len(history),
            error_type=type(error).__name__, error=str(error), binding=expected, updated_at=p.now()))
        raise


def summarize(root, config=CONFIG):
    root = Path(root)
    cfg, _ = verify_manifest(root, config)
    verify_formal_authorization(root)
    manifest = read_json(root / 'execution_manifest.json')
    original = Path(manifest['C01_result_root'])
    c01 = collect_c01_rows(original, cfg, manifest['comparison_contract_sha256'])
    for row in c01:
        seed = row['binding']['seed']
        if p.canonical_sha(row) != manifest['C01_reference_sha256'][str(seed)]:
            raise ValueError('Final C01 read-only replay differs from its prepared reference')
    c01b = [run_result(root, seed, config=config) for seed in SEEDS]
    for old, new in zip(c01, c01b):
        validate_head_info(old['model_info'], 66)
        validate_head_info(new['model_info'], 6)
        verify_pairing_history(new['history'], old)
    summary = analysis.c01b_summary(dict(C01=c01, C01b=c01b), cfg['decision'],
        subset_contract=t.load_subset_contract(cfg), channels_C01=[66] * 4, output_dir=root / 'paired_analysis')
    if not summary['paired_training_witnesses']['all_expected_present']:
        raise ValueError('All six comparison rows must carry every epoch data witness')
    summary.update(execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                   formal_authorization_sha256=p.sha(root / 'formal_authorization.json'))
    p.dump(root / 'c01b_summary.json', summary)
    verdict = summary['decision']
    contract = dict(schema_version=1, status='FROZEN', supplement_to='historical D17 quality-head adoption gate',
        historical_D17_changed=False, decision=verdict, actual_C01_raw_output_channels_per_scale=[66] * 4,
        C01b_raw_output_channels_per_scale=[6] * 4, selected_variant=verdict['selected_variant'],
        selected_raw_output_channels_per_scale=verdict['selected_raw_output_channels_per_scale'],
        selected_reg_max=1 if verdict['gate_pass'] else 16,
        selected_box_loss='GIoU' if verdict['gate_pass'] else 'CIoU',
        distance_mapping='ReLU_nonnegative' if verdict['gate_pass'] else 'DFL_softmax_expectation_16bins',
        regression_hidden_width=64, classification_hidden_width=48, final_regression_groups=4,
        C01_final_convolutions=c01[0]['model_info']['head_contract'],
        C01b_final_convolutions=c01b[0]['model_info']['head_contract'],
        raw_output_channel_ratio_C01_to_C01b=11.0,
        raw_channel_ratio_is_not_head_or_board_speedup=True,
        scope='combined direct distances/ReLU/noDFL/CIoU-to-GIoU; not single-component attribution',
        comparison_contract_sha256=manifest['comparison_contract_sha256'],
        execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
        formal_authorization_sha256=p.sha(root / 'formal_authorization.json'),
        c01b_summary_sha256=p.sha(root / 'c01b_summary.json'), automatic_search_executed=False,
        HLS_route_board_power_status='NOT_MEASURED', test_evaluated=False)
    p.dump(root / 'D17_HEAD_CONTRACT.json', contract)
    text = f'# D17头合同补充\n\n选择：{verdict["selected_variant"]}，P2/P3/P4/P5每尺度原始输出{verdict["selected_raw_output_channels"]}通道。\n\n'
    text += '真实历史C01为reg_max=16、66通道；34通道不是本次历史基线。C01b为直接距离、6通道、ReLU、GIoU，无DFL。隐藏层保持回归64和分类48通道。\n\n'
    for metric in analysis.PRIMARY_METRICS:
        effect = summary['paired_difference_C01b_minus_C01'][metric]
        arms = summary['subsets']['full99']['arms']
        interval = effect['hierarchical_ci']
        text += f'- {metric}：C01同3seed稳定均值 {arms["C01"][metric]["mean"] * 100:.6f}%，'
        text += f'C01b {arms["C01b"][metric]["mean"] * 100:.6f}%；配对差 {effect["mean"] * 100:.6f} 个百分点，'
        text += f'95%区间 [{interval["low"] * 100:.6f}, {interval["high"] * 100:.6f}] 个百分点；门槛 ≥ −0.5 个百分点。\n'
    text += '\n每尺度末端回归1×1卷积：C01为64→64、groups=4、权重[64,16,1,1]、1088参数；C01b为64→4、groups=4、权重[4,16,1,1]、68参数。分类末层均为48→2、groups=1、98参数。原始输出66→6为11倍通道比，不能据此宣称整个head或上板加速。\n'
    text += '\n区间及每seed结果见c01b_summary.json；区间不追加选择门槛，点估计通过不等于统计证明非劣效。此结果仅为验证开发集的组合适配比较。历史D17记录保持不变，不自动启动NAS/HLS/上板。\n'
    (root / 'D17_HEAD_CONTRACT.md').write_text(text, encoding='utf-8')
    return summary


def queue(root, device, config=CONFIG):
    root = Path(root)
    cfg, _ = verify_manifest(root, config)
    verify_formal_authorization(root, freeze=True)
    if device != 'cuda' or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('C01b queue must use the original GPU1 CUDA device')
    path = root / 'queue_status.json'
    if path.exists():
        raise FileExistsError('C01b queue already invoked; no automatic retry or overwrite')
    status = dict(status='RUNNING', pid=os.getpid(), jobs=list(SEEDS), completed=[],
        execution_manifest_sha256=p.sha(root / 'execution_manifest.json'), started_at=p.now(), test_evaluated=False)
    p.dump(path, status)
    try:
        for seed in SEEDS:
            folder = result_dir(root, seed)
            if folder.exists():
                raise FileExistsError(f'Existing formal folder cannot be queued: {folder}')
            folder.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--config', str(config),
                       '--output', str(root), '--seed', str(seed), '--device', device]
            with (folder / 'stdout.log').open('x', encoding='utf-8') as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                status.update(active_seed=seed, child_pid=child.pid, updated_at=p.now())
                p.dump(path, status)
                try:
                    code = child.wait(timeout=cfg['queue']['maximum_hours_per_run'] * 3600)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    raise RuntimeError('Bounded C01b worker timed out; no automatic retry')
            if code:
                raise RuntimeError(f'C01b worker exited {code}; no automatic retry')
            run_result(root, seed, config=config)
            status['completed'].append(seed)
            p.dump(path, status)
        summary_command = [sys.executable, str(Path(__file__).resolve()), 'summarize',
                           '--config', str(config), '--output', str(root), '--device', 'cpu']
        with (root / 'summary_stdout.log').open('x', encoding='utf-8') as handle:
            child = subprocess.Popen(summary_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            status.update(status='FINALIZING', active_seed=None, child_pid=child.pid, updated_at=p.now())
            p.dump(path, status)
            try:
                code = child.wait(timeout=2 * 3600)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                raise RuntimeError('C01b CPU finalization exceeded its independent two-hour bound')
        if code:
            raise RuntimeError(f'C01b finalization exited {code}; completed training remains preserved')
        summary = read_json(root / 'c01b_summary.json')
        status.update(status='COMPLETE', active_seed=None, child_pid=None,
                      decision=summary['decision'], updated_at=p.now())
        p.dump(path, status)
        return status
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY', error=str(error), error_type=type(error).__name__, updated_at=p.now())
        p.dump(path, status)
        raise


def status(root):
    root = Path(root)
    runs = []
    for seed in SEEDS:
        path = result_dir(root, seed) / 'status.json'
        runs.append(dict(seed=seed, **read_json(path)) if path.exists() else dict(seed=seed, status='NOT_STARTED'))
    return dict(status='OBSERVED_ARTIFACT_STATUS', runs=runs,
        completed=sum(row['status'] == 'COMPLETE' for row in runs), total=3,
        baseline_resolution=read_json(root / 'baseline_resolution.json') if (root / 'baseline_resolution.json').is_file() else None,
        queue=read_json(root / 'queue_status.json') if (root / 'queue_status.json').is_file() else None,
        conclusion_verified=False, D17_contract_present=(root / 'D17_HEAD_CONTRACT.json').is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'smoke', 'worker', 'queue', 'summarize', 'status'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', '--result-root', type=Path)
    parser.add_argument('--C01-result-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    root = (args.output or ROOT / cfg['result_root']).resolve()
    torch.set_num_threads(2)
    if args.mode == 'prepare':
        prepare(root, args.config, args.C01_result_root)
        result = dict(status='CANDIDATE_PREPARED', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'),
                      baseline_resolution_status=_resolution(root, cfg)['status'], formal_training_started=False)
    elif args.mode == 'status':
        result = status(root)
    elif args.mode == 'verify':
        verify_manifest(root, args.config, args.C01_result_root)
        result = dict(status='CANDIDATE_PREFLIGHT_PASS', formal_training_started=False)
    elif args.mode == 'smoke':
        result = smoke(root, args.device, args.config)
    elif args.mode == 'worker':
        result = run_training(root, args.seed, args.device, args.resume, args.config)
        result = dict(status='COMPLETE', binding=result['binding'], metrics=result['metrics'])
    elif args.mode == 'queue':
        result = queue(root, args.device, args.config)
    else:
        result = summarize(root, args.config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
