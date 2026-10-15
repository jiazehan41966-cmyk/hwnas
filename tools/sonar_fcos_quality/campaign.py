"""Frozen paired six-run protocol and honest partial/final reporting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import statistics

from tools.sonar_fcos_pilot.common import (ROOT, environment, read_json, require_manifest,
    resolve_path, sha256_file, utc_now, write_json)

DEFAULT_CONFIG = ROOT / 'configs/sonar_fcos_quality_20261012.json'
TRAINING_KEYS = ('image_size', 'epochs', 'batch_size', 'gradient_accumulation',
    'learning_rate', 'momentum', 'weight_decay', 'learning_rate_drop_epochs',
    'warmup_epochs', 'warmup_start_factor', 'amp', 'amp_initial_scale', 'workers',
    'torch_threads', 'pretrained', 'diagnostic_score_threshold', 'device')


def load_config(path=DEFAULT_CONFIG):
    experiment = read_json(path)
    base_path = Path(experiment['baseline_config'])
    baseline = read_json(base_path)
    if sha256_file(base_path) != 'c1bbeb42ac23ae8be243a1974194acac746504776a6fa19e869759df243aeb8b':
        raise ValueError('Historical AAA baseline config changed')
    from .model import QUALITY_PARAMETERS
    if experiment['quality'] != QUALITY_PARAMETERS:
        raise ValueError('Quality model hyperparameters differ from frozen config')
    if experiment['variants'] != ['AAA', 'QFL_DFL'] or experiment['seeds'] != [42, 43, 44]:
        raise ValueError('Only the authorized two variants and three seeds are allowed')
    if experiment['epochs'] != 30 or experiment['max_formal_runs'] != 6:
        raise ValueError('The authorized experiment is exactly six 30-epoch runs')
    if experiment['checkpoint_selection'] != 'earliest_best_validation_coco_map50_95':
        raise ValueError('Checkpoint selection is frozen')
    if experiment['acceptance'] != {'mean_paired_ap75_gain_pp_min': 2.0, 'mean_nms_first_loss_relative_reduction_min': .25}:
        raise ValueError('User acceptance criteria changed')
    config = {**baseline, **experiment, 'config_path': str(Path(path).resolve()),
              'config_sha256': sha256_file(path), 'baseline_config_sha256': sha256_file(base_path)}
    config.pop('recovery_parent', None)
    for key in TRAINING_KEYS:
        if config[key] != baseline[key]:
            raise ValueError('Baseline training setting changed: ' + key)
    manifest_path, manifest = require_manifest(config)
    if sha256_file(manifest_path) != config['expected_manifest_sha256']:
        raise ValueError('Historical UATD manifest differs')
    validation = [r for r in manifest['samples'] if r['split'] == 'val']
    if len(validation) != 1517 or sum(len(r['boxes']) for r in validation) != 2692:
        raise ValueError('Not the same 1517-image, 2692-object validation split')
    return config


def source_binding(config):
    paths = list((ROOT / 'tools/sonar_fcos_pilot').glob('*.py'))
    paths += list((ROOT / 'tools/sonar_fcos_quality').glob('*.py'))
    return {'config_sha256': config['config_sha256'],
            'baseline_config_sha256': config['baseline_config_sha256'],
            'data_manifest_sha256': config['expected_manifest_sha256'],
            'pretrained_checkpoint_sha256': sha256_file(resolve_path(config['weight_cache']) / 'checkpoints' /
                                                       'fcos_resnet50_fpn_coco-99b0c9b7.pth'),
            'sources': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}}


def verify_development_inputs(manifest):
    for row in manifest['samples']:
        if row['split'] in ('train', 'val'):
            for field, hash_field in [('image_path', 'sha256'), ('annotation_path', 'annotation_sha256')]:
                if hash_field not in row or sha256_file(row[field]) != row[hash_field]:
                    raise RuntimeError('Development input changed: ' + str(row.get(field)))


def prepare(config):
    manifest_path, manifest = require_manifest(config)
    verify_development_inputs(manifest)
    output = resolve_path(config['output_dir'])
    receipt = {'status': 'PASS', 'source_binding': source_binding(config),
               'environment': environment(), 'training_settings': {k: config[k] for k in TRAINING_KEYS},
               'validation_images': 1517, 'validation_objects': 2692,
               'test_evaluated': False, 'protocol_id': manifest.get('protocol_id'),
               'limitations': ['Development validation; no verified acquisition-sequence separation',
                              'FCOS weights require explicit architecture-aware mapping for a different target network'],
               'run_order': [{'variant': v, 'seed': s} for s in config['seeds'] for v in config['variants']]}
    path = output / 'execution_manifest.json'
    if path.exists() and read_json(path) != receipt:
        raise RuntimeError('Campaign already prepared with different source/config/environment')
    write_json(path, receipt)
    summarize(config)
    return receipt


def paired_decision(rows):
    expected = {(v, s) for s in (42, 43, 44) for v in ('AAA', 'QFL_DFL')}
    keys = [(r['variant'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate runs')
    if set(keys) != expected or any(r['trained_epochs'] != 30 for r in rows):
        return {'status': 'PENDING', 'completed_runs': len(rows), 'required_runs': 6}
    by = {(r['variant'], r['seed']): r for r in rows}
    differences = [by['QFL_DFL', s]['ap75'] - by['AAA', s]['ap75'] for s in (42, 43, 44)]
    old_nms = statistics.mean(by['AAA', s]['nms_first_loss'] for s in (42, 43, 44))
    new_nms = statistics.mean(by['QFL_DFL', s]['nms_first_loss'] for s in (42, 43, 44))
    reduction = (old_nms - new_nms) / old_nms if old_nms > 0 else None
    gate_ap = statistics.mean(differences) >= 2.
    gate_nms = reduction is not None and reduction >= .25
    stats = {}
    for variant in ('AAA', 'QFL_DFL'):
        stats[variant] = {metric: {'mean': statistics.mean(by[variant, s][metric] for s in (42, 43, 44)),
                                 'sample_sd': statistics.stdev(by[variant, s][metric] for s in (42, 43, 44))}
                         for metric in ('ap75', 'ap85', 'map', 'nms_first_loss', 'raw_no_good_box', 'wrong_class_fp')}
    return {'status': 'PASS' if gate_ap and gate_nms else 'FAIL',
            'paired_ap75_gains_pp': differences, 'mean_ap75_gain_pp': statistics.mean(differences),
            'mean_nms_first_loss_relative_reduction': reduction, 'ap75_gate': gate_ap, 'nms_gate': gate_nms,
            'group_statistics': stats, 'scope': 'fixed development validation only; not a sealed test result'}


def summarize(config):
    output = resolve_path(config['output_dir'])
    binding = source_binding(config)
    rows = []
    summaries = []
    for seed in config['seeds']:
        for variant in config['variants']:
            path = output / 'runs' / f'{variant}_seed{seed}' / 'summary.json'
            if not path.exists():
                continue
            item = read_json(path)
            if item['source_binding'] != binding or item['status'] != 'COMPLETE' or item['trained_epochs'] != 30:
                raise RuntimeError('Invalid completed run receipt')
            for artifact in item['best_artifacts'].values():
                if sha256_file(artifact['path']) != artifact['sha256']:
                    raise RuntimeError('Completed artifact changed')
            metrics, diagnostics = item['validation'], item['diagnostics']
            rows.append({'variant': variant, 'seed': seed, 'trained_epochs': item['trained_epochs'],
                         'selected_epoch': item['best_epoch'],
                         **{k: metrics[k] for k in ('ap75', 'ap85', 'map')},
                         **{k: diagnostics[k] for k in ('nms_first_loss', 'raw_no_good_box', 'wrong_class_fp')}})
            summaries.append(item)
    decision = paired_decision(rows)
    result = {'updated_at': utc_now(), 'rows': rows, 'decision': decision,
              'historical_reference_only': config['historical_reference_only'],
              'test_evaluated': False, 'source_binding': binding}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'branch_results.json', result)
    columns = ('variant', 'seed', 'trained_epochs', 'selected_epoch', 'ap75', 'ap85', 'map',
               'nms_first_loss', 'raw_no_good_box', 'wrong_class_fp')
    with (output / 'branch_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 实验2 FCOS / UATD 支线', '', f"状态：{decision['status']}；已完成 {len(rows)}/6 次训练。", '',
             '固定1517张开发验证图；每次30轮，按验证mAP50–95选择最早最佳权重。AP单位为百分点。', '',
             '|方案|种子|选中轮次|AP75|AP85|mAP50–95|NMS首次丢好框|原始无好框|错类误检|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('|{variant}|{seed}|{selected_epoch}|{ap75:.4f}|{ap85:.4f}|{map:.4f}|{nms_first_loss}|{raw_no_good_box}|{wrong_class_fp}|'.format(**row))
    if decision['status'] != 'PENDING':
        lines += ['', f"平均AP75增量：{decision['mean_ap75_gain_pp']:.4f}个百分点。",
                  f"NMS首次丢好框相对下降：{decision['mean_nms_first_loss_relative_reduction']!r}。",
                  '完整逐种子差值和均值/样本标准差见 branch_results.json。']
        selected = max(summaries, key=lambda s: s['validation']['map'])
        package = output / 'pretrained'
        package.mkdir(exist_ok=True)
        src = Path(selected['best_artifacts']['checkpoint']['path'])
        destination = package / 'uatd_fcos_best.pt'
        if destination.exists() and sha256_file(destination) != sha256_file(src):
            raise RuntimeError('Do not replace an exported pretrained checkpoint')
        if not destination.exists():
            shutil.copy2(src, destination)
        write_json(package / 'pretrained_manifest.json', {
            'checkpoint': str(destination), 'sha256': sha256_file(destination),
            'variant': selected['variant'], 'seed': selected['seed'], 'epoch': selected['best_epoch'],
            'selection': 'highest best-validation COCO mAP50:95 across all six runs; stable run-order ties',
            'model_contract': selected['model'], 'source_binding': binding,
            'validation': selected['validation'], 'decision': decision['status'],
            'transfer_status': 'AVAILABLE_FCOS_STATE_DICT; different backbone/head requires explicit mapping',
            'test_evaluated': False})
        lines += ['', f"预训练权重：`{destination}`。", '权重属于FCOS ResNet50-FPN；双类轻量网络需另行核对层名与形状，不能宣称可直接整网加载。']
    lines += ['', '历史333/448/577仅作诊断参照，不代替本次三个种子的对照。',
              '固定开发验证集结果不等同于独立测试集或新采集场景泛化证据。']
    (output / '支线结果表.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
