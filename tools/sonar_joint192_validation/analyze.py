"""Replay all complete runs before reporting paired sonar candidate effects."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.sonar_joint192_validation import train
from tools.sonar_joint192_validation.domain_metrics import evaluate_domain_metrics
from tools.sonar_joint192_validation.checkpoint_replay import POLICY as CHECKPOINT_REPLAY_POLICY


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), values=values)


def analyze(snapshot, output):
    cfg, baseline = train.load_config()
    manifest = read(snapshot / 'execution_manifest.json')
    collection = read(snapshot / 'collection_manifest.json')
    staged = read(train.DEFAULT_ROOT / 'transport/staging_manifest.json')
    replay = read(snapshot / 'initialization_replay.json')
    for rel in ('execution_manifest.json','initialization_replay.json',
                'worker_gpu2/queue_status.json','worker_gpu3/queue_status.json'):
        if rel not in collection['artifacts']:
            raise ValueError('Required campaign identity is not transport-bound: ' + rel)
    for rel, sha in collection['artifacts'].items():
        if train.sha(snapshot / rel) != sha:
            raise ValueError('Collected artifact SHA mismatch: ' + rel)
    for rel, sha in manifest['source_sha256'].items():
        if staged['source_sha256'].get(rel) != sha:
            raise ValueError('Execution source differs from staged source: ' + rel)
        if train.sha(ROOT / rel) != sha:
            raise ValueError('Local analysis training dependency differs from executed bytes: ' + rel)
    if manifest['config_sha256'] != train.sha(train.CONFIG) or manifest['test_evaluated'] is not False:
        raise ValueError('Config identity or test seal mismatch')
    actual = train.input_identity(train.verify_inputs(cfg, baseline))
    if actual != train.input_identity(manifest['input_verification']):
        raise ValueError('Current train/valid differs from executed dataset')
    for gpu in (2, 3):
        worker = read(snapshot / f'worker_gpu{gpu}/queue_status.json')
        if worker['status'] != 'COMPLETE' or len(worker['completed']) != 6:
            raise ValueError('Twelve complete runs required')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    domain_profile = None
    manifest_sha = train.sha(snapshot / 'execution_manifest.json')
    inference_root = snapshot / 'checkpoint_replay_final'
    inference = read(inference_root / 'replay_receipt.json')
    if 'checkpoint_replay_final/replay_receipt.json' not in collection['artifacts']:
        raise ValueError('Checkpoint inference witness is not transport-bound')
    if (inference['status'] != 'PASS' or not inference['all_twelve_replayed']
            or inference['test_evaluated'] is not False or inference['training_performed'] is not False
            or inference['original_results_modified'] is not False
            or inference['binding']['execution_manifest_sha256'] != manifest_sha
            or inference['binding']['physical_gpu'] != 2
            or inference['comparison_policy'] != dict(CHECKPOINT_REPLAY_POLICY,require_exact=False)
            or inference['binding']['comparison_policy'] != inference['comparison_policy']
            or inference['binding']['replay_script_sha256'] != inference['replay_script_sha256']
            or inference['replay_script_sha256'] != train.sha(ROOT / 'tools/sonar_joint192_validation/checkpoint_replay.py')):
        raise ValueError('Checkpoint inference witness binding/completion mismatch')
    inference_runs = {(r['arm'],r['seed']):r for r in inference['runs']}
    if (len(inference['runs']) != 12 or set(inference_runs) != {(a,s) for a in train.PROTOTYPES for s in train.SEEDS}
            or any(r['status'] != 'PASS' or r['valid_images'] != 99 or not r['tolerance_comparison_pass']
                   for r in inference_runs.values())):
        raise ValueError('All twelve original checkpoints must reproduce all valid predictions')
    for rel, digest in inference['binding']['original_artifact_sha256'].items():
        identities = collection['remote_checkpoints'] if rel.endswith('.pt') else collection['artifacts']
        if identities.get(rel) != digest:
            raise ValueError('Inference did not use the collected original artifact: ' + rel)
    for rel, digest in inference['output_artifact_sha256'].items():
        if (collection['artifacts'].get('checkpoint_replay_final/' + rel) != digest
                or train.sha(inference_root / rel) != digest):
            raise ValueError('Checkpoint inference output SHA mismatch: ' + rel)
    if replay['execution_manifest_sha256'] != manifest_sha:
        raise ValueError('Pinned-runtime initialization replay binding mismatch')
    env = yaml.safe_load((ROOT / cfg['environment_spec']).read_text(encoding='utf-8'))
    for key, value in env['expected'].items():
        if inference['environment'][key] != str(value):
            raise ValueError('Checkpoint inference used a different training runtime')
        if key == 'cuda_runtime':
            continue  # CPU reconstruction uses the same CUDA torch package without dispatch.
        if replay['environment'][key] != str(value):
            raise ValueError('Initialization replay used a different training runtime')
    if inference['environment']['gpu'] != env['gpu']:
        raise ValueError('Checkpoint inference used a different GPU model')
    for arm in train.PROTOTYPES:
        for seed in train.SEEDS:
            folder = snapshot / 'formal' / arm / f'seed_{seed}'
            required = ['result.json','status.json','epoch_log.json','run_binding.json',
                        'best_valid_records.json','initialization_identity.json','model_info.json']
            for name in required:
                if (folder / name).relative_to(snapshot).as_posix() not in collection['artifacts']:
                    raise ValueError('Required analysis input is not transport-bound: ' + name)
            result, status = read(folder / 'result.json'), read(folder / 'status.json')
            history, binding = read(folder / 'epoch_log.json'), read(folder / 'run_binding.json')
            expected = dict(execution_manifest_sha256=manifest_sha, prototype=arm, seed=seed,
                            epochs=80, formal_eligible=True, namespace='sonar_joint192_validation', test_evaluated=False)
            if (status['status'] != 'COMPLETE' or status['epochs'] != 80 or status['binding'] != expected
                    or binding['binding'] != expected or result['execution_manifest_sha256'] != manifest_sha
                    or result['formal_eligible'] is not True or result['test_evaluated'] is not False):
                raise ValueError('Completion identity mismatch')
            if ([x['epoch'] for x in history] != list(range(1, 81))
                    or sum(x['batches'] for x in history) != 6640
                    or any(x['batches'] != 83 or x['valid']['num_images'] != 99 for x in history)):
                raise ValueError('Incomplete training/validation history')
            best = max(history, key=lambda x: x['valid']['mAP50_95'])
            if result['best_epoch'] != best['epoch'] or result['history_sha256'] != train.sha(folder / 'epoch_log.json'):
                raise ValueError('Earliest best or history identity mismatch')
            for name in ('best', 'latest'):
                rel = (folder / (name + '.pt')).relative_to(snapshot).as_posix()
                if collection['remote_checkpoints'][rel] != result[name + '_checkpoint_sha256']:
                    raise ValueError('Remote retained checkpoint SHA mismatch')
            if inference_runs[arm,seed]['best_checkpoint_sha256'] != result['best_checkpoint_sha256']:
                raise ValueError('Inference loaded a different best checkpoint')
            model = train.build_model(cfg['arms'][arm], seed)
            info = read(folder / 'model_info.json')
            if result['Params'] != sum(p.numel() for p in model.parameters()) or result['MACs'] != info['estimated_macs']:
                raise ValueError('Model parameter/MAC metadata mismatch')
            contract = json.loads(json.dumps(model.architecture_contract()))
            identity = replay['identities'][f'{arm}/seed_{seed}']
            if (binding['architecture_contract'] != contract or binding['initialization_identity'] != identity
                    or read(folder / 'initialization_identity.json') != identity
                    or info['initialization_identity'] != identity):
                raise ValueError('Model architecture/initialization identity mismatch')
            records = read(folder / 'best_valid_records.json')
            metrics = train.frozen.metrics_from_records(records, baseline)
            for key in train.METRICS:
                for value in (result[key], best['valid'][key]):
                    if not math.isclose(metrics[key], value, rel_tol=0., abs_tol=1e-12):
                        raise ValueError(f'Raw prediction metric replay mismatch: {arm}/{seed}/{key}')
            domain = evaluate_domain_metrics(records, baseline, records_path=folder / 'best_valid_records.json')
            profile = dict(
                num_images=domain['num_images'],
                images_without_GT=sum(x['GT'] == 0 for x in domain['per_image']),
                images_with_one_GT=sum(x['GT'] == 1 for x in domain['per_image']),
                group_GT={k:v['GT'] for k,v in domain['groups'].items()},
                lowcontrast_very_small_overlap=sum(
                    'contrast_low' in t['groups'] and 'very_small_min_side' in t['groups']
                    for x in domain['per_image'] for t in x['targets']),
                valid_entries_sha256=domain['provenance']['valid_entries_sha256'],
                train_contrast_reference_sha256=domain['train_contrast_reference']['contrast_values_sha256'])
            if domain_profile is not None and profile != domain_profile:
                raise ValueError('Domain GT/groups changed across paired runs')
            domain_profile = profile
            for key in ('Precision', 'Recall'):
                if not math.isclose(domain['overall'][key], metrics[key], rel_tol=0., abs_tol=1e-12):
                    raise ValueError('Domain matching differs from frozen P/R: ' + key)
            train.dump(output / 'domain' / arm / f'seed_{seed}.json', domain)
            row = dict(arm=arm, seed=seed, best_epoch=best['epoch'],
                       **{k: metrics[k] for k in train.METRICS},
                       endpoint_mAP50_95=history[-1]['valid']['mAP50_95'],
                       lowcontrast_recall=domain['groups']['contrast_low']['Recall'],
                       lowcontrast_FN=domain['groups']['contrast_low']['FN'],
                       very_small_recall=domain['groups']['very_small_min_side']['Recall'],
                       very_small_FN=domain['groups']['very_small_min_side']['FN'],
                       FP_per_image=domain['overall']['FP_per_image'],
                       background_FP_per_image=domain['overall']['background_FP_per_image'],
                       Params=result['Params'], MACs=result['MACs'],
                       training_seconds=status['elapsed_training_seconds'])
            rows.append(row)
    aggregates, paired = {}, {}
    keys = list(train.METRICS) + ['endpoint_mAP50_95','lowcontrast_recall','lowcontrast_FN',
                                  'very_small_recall','very_small_FN','FP_per_image','background_FP_per_image']
    lookup = {(r['arm'], r['seed']): r for r in rows}
    for arm in train.PROTOTYPES:
        aggregates[arm] = {k: stats([lookup[arm, s][k] for s in train.SEEDS]) for k in keys}
        if arm == 'B0':
            continue
        paired[arm] = {k: stats([lookup[arm,s][k]-lookup['B0',s][k] for s in train.SEEDS]) for k in keys}
        main = paired[arm]['mAP50_95']['values']
        paired[arm]['positive_main_seeds'] = sum(v > 0 for v in main)
        paired[arm]['main_direction_consistent'] = all(v > 0 for v in main)
        paired[arm]['lowcontrast_positive_seeds'] = sum(v > 0 for v in paired[arm]['lowcontrast_recall']['values'])
    interaction = {k: stats([lookup['B3',s][k]-lookup['B1',s][k]-lookup['B2',s][k]+lookup['B0',s][k]
                            for s in train.SEEDS]) for k in keys}
    summary = dict(status='COMPLETE_REPLAY_VERIFIED', rows=rows, aggregates=aggregates, paired_deltas=paired,
                   joint_interaction=interaction, seeds=list(train.SEEDS),
                   domain_group_GT={k:v['GT'] for k,v in domain['groups'].items()},
                   domain_profile=domain_profile,
                   contrast_cutoffs={k:domain['train_contrast_reference'][k] for k in ('q25','q75')},
                   acceptance=dict(full_runs=12, full_epoch_histories=True, full_unique_valid_GT_verified=True,
                                   raw_prediction_AP_PR_replay=True, artifact_transport_sha_verified=True,
                                   source_dataset_model_identity_verified=True, remote_checkpoint_sha_verified=True,
                                   remote_checkpoint_inference_replayed=True,
                                   checkpoint_inference_strict_equal=inference['all_strict_comparisons_pass'],
                                   checkpoint_bytes_downloaded_and_rerun=False, test_evaluated=False),
                   limitations=['valid selected checkpoints; no independent test', '99 images, lowcontrast18, verysmall31',
                                'three seeds describe training randomness on the same valid images, not generalization confidence',
                                'no target-free valid images; background FP cannot estimate pure-background false-alarm rate',
                                'lowcontrast/very_small groups overlap; FN cannot be added',
                                'group diagnostics use best checkpoints; endpoint diagnostics only include mAP',
                                'GPU2 B0/B2; GPU3 B1/B3, same model/runtime',
                                'deterministic warn_only, not guaranteed CUDA bitwise',
                                'candidate ablation only, no 192-space search or hardware benefit'],
                   provenance=dict(execution_manifest_sha256=manifest_sha, collection_manifest_sha256=train.sha(snapshot / 'collection_manifest.json'),
                                   pinned_runtime_initialization_replay_sha256=train.sha(snapshot / 'initialization_replay.json'),
                                   checkpoint_inference_receipt_sha256=train.sha(inference_root / 'replay_receipt.json'),
                                   checkpoint_inference_source_sha256=inference['replay_script_sha256'],
                                   analysis_source_sha256=train.sha(Path(__file__)),
                                   domain_source_sha256=train.sha(ROOT / 'tools/sonar_joint192_validation/domain_metrics.py')))
    train.dump(output / 'analysis.json', summary)
    def meanstd(arm, key, factor=100):
        x = aggregates[arm][key]
        return f"{factor*x['mean']:.3f} ± {factor*x['sample_std']:.3f}"
    lines = ['# 声呐联合候选操作受控验证', '',
             '四组×三种子均完成80轮。下表为同一99张valid、最早最佳checkpoint的均值±样本标准差；AP和Recall以百分数表示。原始预测重算、原训练环境最佳权重重新推理、数据/架构/初始化/SHA核验全部通过。', '',
             '| 候选 | 最佳mAP50-95 (%) | 第80轮mAP50-95 (%) | AP50 (%) | Recall (%) | 低对比Recall (%) | 更小目标Recall (%) | FP/图 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = dict(B0='B0 参考', B1='B1 低通', B2='B2 P2直连', B3='B3 联合')
    for arm in train.PROTOTYPES:
        vals = [meanstd(arm, key, 1 if key == 'FP_per_image' else 100)
                for key in ('mAP50_95','endpoint_mAP50_95','AP50','Recall','lowcontrast_recall','very_small_recall','FP_per_image')]
        lines.append('| ' + labels[arm] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '| 候选 | 参数量 | MAC/图 (G) |', '|---|---:|---:|']
    for arm in train.PROTOTYPES:
        row = lookup[arm, 42]
        lines.append(f"| {labels[arm]} | {row['Params']} | {row['MACs']/1e9:.6f} |")
    lines += ['', 'MAC为软件计算量估计，已计入固定低通；不是实测延迟、FPGA资源或功耗。']
    lines += ['', '| 相对参考 | seed42 ΔmAP | seed43 ΔmAP | seed44 ΔmAP | 均值ΔmAP | 低对比FN均值变化 | 更小目标FN均值变化 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('B1','B2','B3'):
        d = paired[arm]
        vals = [f'{100*x:+.3f}' for x in d['mAP50_95']['values']]
        vals += [f"{100*d['mAP50_95']['mean']:+.3f}", f"{d['lowcontrast_FN']['mean']:+.3f}", f"{d['very_small_FN']['mean']:+.3f}"]
        lines.append('| ' + labels[arm] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', 'ΔmAP单位为百分点；FN变化为负代表漏检减少。分组先按全图GT进行匹配，再按尺寸/对比度归组；分组数量固定为低对比18、更小目标31。', '',
              '## 结论边界', '',
              '主指标、领域分组和误检必须一起解读。三种子方向一致可支持本valid协议下的候选收益；分组Recall未改善时不能声称解决低对比或极小目标漏检。联合交互项为 B3−B1−B2+B0，不能把两项独立增益直接相加。', '',
              f"联合mAP交互均值：{100*interaction['mAP50_95']['mean']:+.3f}个百分点。完整逐种子指标、终点指标、分组漏检及背景FP见analysis.json与domain/。", '',
              '这些checkpoint由valid选择，因此结果不是独立测试泛化证据。三种子标准差只描述相同99图上的训练随机性，不是数据泛化置信区间。分组漏检来自各自最佳checkpoint，第80轮仅比较mAP。全部99个GT面积均小于32²，缺少非小目标对照。低对比分组来自图像框环统计，并非声呐物理信噪比标签。', '',
              f"验证集纯背景图为{domain_profile['images_without_GT']}张、单目标图为{domain_profile['images_with_one_GT']}张；背景型FP只描述含目标图中的额外错误框，不能据此声称纯背景场景虚警率改善。它采用maxIoU<0.1诊断定义，并非人工确认的声呐杂波。低对比与更小目标组重叠{domain_profile['lowcontrast_very_small_overlap']}例，不能相加漏检数。两GPU同型号/环境，但低通因素固定GPU3；CUDA确定性采用warn_only。", '',
              '本实验估计完整候选操作的效果：dense DW、BN统计分辨率、有效感受野和固定滤波一起改变，尚未通过dense-decimate控制分离固定低通本身的贡献。P2直连同时改变参数量和计算量；不把检测改善直接解释为物理去噪。未验证完整NAS、量化、FPGA延迟/资源/功耗。', '',
              f"训练checkpoint原件和哈希保留在本轮远端独立目录；未下载权重到Windows。原训练解释器/GPU2逐项加载12份best.pt，对99图重新推理并核对保存预测、GT及AP/P-R。所有逐位比较通过：{inference['all_strict_comparisons_pass']}。容差在执行前固定为框坐标1e-4输入像素、score 1e-6、指标1e-8；实际差异与原始重算预测见checkpoint_replay_final/。"]
    (output / 'VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], paired={a:paired[a]['mAP50_95'] for a in paired}, output=str(output))))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=train.DEFAULT_ROOT / 'remote_snapshot')
    parser.add_argument('--output', type=Path, default=train.DEFAULT_ROOT / 'analysis')
    args = parser.parse_args()
    analyze(args.snapshot, args.output)
