"""Frozen R0--R3 validation recipe selection; CPU prediction replay only.

Input rows are C01 ``run_result`` dictionaries, with ``recipe`` added without
changing the source file. All twelve cells use YOLOv9_T/quality_aware and seeds
42--44. ``load_rows`` binds on-disk result files, manifests, environment locks
and archived predictions; callers may instead pass already verified runner
rows directly. The primary endpoint is full99 mean epochs76--80, never best.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.sonar_head_factorial import stable_analysis as stable

RECIPES = ('R0', 'R1', 'R2', 'R3')
CANDIDATES = RECIPES[1:]
SEEDS = (42, 43, 44)
EPOCHS = stable.EPOCHS
METRICS = stable.METRICS
REPLICATES = 2000
RANDOM_SEED = 20261013
DECISION = dict(metric='mAP50_95', primary_checkpoint='mean_epochs_76_80',
                threshold_absolute=.01, simplest_within_absolute=.005,
                bootstrap=dict(replicates=REPLICATES, random_seed=RANDOM_SEED, confidence=.95))
MULTIPLICITY_NOTE = ('Three prespecified candidates; pointwise 95% intervals are not '
                     'adjusted for multiple comparisons. This is the registered '
                     'validation-development selection rule, not family-wise inference.')


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_decision(decision):
    for field in ('metric', 'primary_checkpoint', 'threshold_absolute', 'simplest_within_absolute'):
        if decision.get(field) != DECISION[field]:
            raise ValueError(f'Unregistered recipe decision: {field}')
    boot = decision.get('bootstrap', {})
    for field, expected in DECISION['bootstrap'].items():
        if boot.get(field) != expected:
            raise ValueError(f'Unregistered recipe bootstrap: {field}')


def select_recipe(arms, effects):
    """Apply both effectiveness gates, then simplest within .005 of best effective.

    ``arms`` maps R0--R3 to stable means; ``effects`` maps R1--R3 to dictionaries
    containing ``mean`` (paired delta) and ``hierarchical_ci``. No auxiliary
    endpoint is consumed here. Exact zero is not a positive CI lower bound.
    """
    if set(arms) != set(RECIPES) or set(effects) != set(CANDIDATES):
        raise ValueError('Selection requires R0--R3 and exactly three candidate effects')
    if not all(np.isfinite(float(value)) for value in arms.values()):
        raise ValueError('All recipe means must be finite')
    gates = {}
    for recipe in CANDIDATES:
        effect = effects[recipe]
        delta = float(effect['mean'])
        if not np.isfinite(delta) or abs(delta - (arms[recipe] - arms['R0'])) > 1e-12:
            raise ValueError('Candidate delta must equal its paired stable mean minus R0')
        ci = effect['hierarchical_ci']
        low, high = ci.get('low'), ci.get('high')
        valid_ci = (low is not None and high is not None
                    and np.isfinite(low) and np.isfinite(high) and low <= high)
        threshold_met = bool(delta >= .01 - 1e-12)
        positive = bool(valid_ci and low > 0.)
        gates[recipe] = dict(mean_delta=delta, delta_percentage_points=100 * delta,
            threshold_met=threshold_met, hierarchical_ci=ci,
            ci_lower_strictly_positive=positive, effective=bool(threshold_met and positive))
    effective = [recipe for recipe in CANDIDATES if gates[recipe]['effective']]
    best = max(effective, key=lambda name: arms[name]) if effective else None
    eligible = [name for name in effective if arms[best] - arms[name] <= .005 + 1e-12]
    selected = eligible[0] if eligible else 'R0'
    return dict(selected_recipe=selected, effective_candidates=effective,
        best_effective_recipe=best, simplest_within_best=eligible,
        stable_mean_by_recipe={k: float(v) for k, v in arms.items()}, candidate_gates=gates,
        criterion='full99 mean_epochs_76_80 delta >= 0.01 AND paired hierarchical CI95 lower > 0',
        simplicity_rule='Among effective R1<R2<R3, select simplest within 0.005 of highest effective stable mean; otherwise R0',
        threshold_absolute=.01, simplest_within_absolute=.005,
        primary_subset='full99', primary_checkpoint='mean_epochs_76_80',
        multiple_comparisons_adjusted=False, multiplicity_note=MULTIPLICITY_NOTE,
        auxiliary89_used_for_selection=False, test_evaluated=False,
        decision_status='SELECTED_EFFECTIVE_RECIPE' if effective else 'RETAIN_R0')


def _flatten_rows(rows):
    if isinstance(rows, dict):
        flattened = []
        for recipe, group in rows.items():
            for row in group:
                if row.get('recipe', recipe) != recipe:
                    raise ValueError('Conflicting recipe wrapper and row identity')
                flattened.append(dict(row, recipe=recipe))
        return flattened
    return list(rows)


def _structure(model_info):
    # Metadata contains seed-specific initialization but no learned weights.
    # Absolute checkout locations have no structural meaning. Only these
    # explicit location keys are normalized; arbitrary string differences fail.
    def normalize(value):
        if isinstance(value, dict):
            return {key: '<checkout-location>' if key in ('source_root', 'author_source_root')
                    else normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return normalize({key: value for key, value in model_info.items()
                      if key not in ('initialization', 'legacy_e0_initialization')})


def validate_rows(rows):
    """Reject incomplete grids, changed structures, initialization or environment."""
    rows = _flatten_rows(rows)
    expected = {(r, s) for r in RECIPES for s in SEEDS}
    result = {}
    for row in rows:
        binding = row.get('binding', {})
        recipe = row.get('recipe', binding.get('recipe'))
        key = (recipe, binding.get('seed'))
        if key in result or key not in expected:
            raise ValueError('Exactly twelve registered recipe/seed cells required')
        if (binding.get('backbone') != 'YOLOv9_T' or binding.get('head') != 'quality_aware'
                or binding.get('mode') != 'formal' or binding.get('test_evaluated') is not False
                or row.get('test_evaluated') is not False):
            raise ValueError('Recipe comparison requires formal YOLOv9_T/quality_aware validation only')
        if binding.get('recipe', recipe) != recipe:
            raise ValueError('Conflicting recipe binding')
        if (row.get('epochs') != 80 or row.get('formal_eligible') is not True
                or row.get('primary_checkpoint') != 'mean_epochs_76_80'):
            raise ValueError('All twelve runs require complete eighty-epoch stable formal evidence')
        for field in ('stable_prediction_paths', 'stable_prediction_sha256'):
            if set(map(str, row.get(field, {}))) != set(map(str, EPOCHS)):
                raise ValueError(f'{field}: require exactly epochs76--80')
        history = row.get('history', [])
        if ([item.get('epoch') for item in history] != list(range(1, 81))
                or any(not np.isfinite(float(item.get('valid', {}).get('mAP50_95', np.nan))) for item in history)):
            raise ValueError('Full finite history epochs1--80 required')
        stable_mean = float(np.mean([history[e - 1]['valid']['mAP50_95'] for e in EPOCHS]))
        for field in ('metrics', 'stable_metrics'):
            value = float(row.get(field, {}).get('mAP50_95', np.nan))
            if not np.isfinite(value) or abs(value - stable_mean) > 1e-12:
                raise ValueError('Stable metric differs from arithmetic mean epochs76--80')
        if not row.get('model_info') or not row.get('initial_state_sha256') or not row.get('initialization'):
            raise ValueError('Structure and exact seed initialization evidence required')
        result[key] = row
    if set(result) != expected:
        raise ValueError('Exactly twelve registered recipe/seed cells required')
    for field in ('environment_fingerprint', 'subset_contract_sha256', 'validation_target_sha256'):
        identities = {row['binding'].get(field) for row in rows}
        if None in identities or '' in identities or len(identities) != 1:
            raise ValueError(f'Common environment/data contract missing or changed: {field}')
    for group in (['R0'], list(CANDIDATES)):
        manifests = {result[r, s]['binding'].get('execution_manifest_sha256') for r in group for s in SEEDS}
        if None in manifests or '' in manifests or len(manifests) != 1:
            raise ValueError('Each campaign requires one common frozen execution manifest')
    structures = {canonical_sha(_structure(row['model_info'])) for row in rows}
    if len(structures) != 1:
        raise ValueError('Model structure/criterion metadata changed across recipes or seeds')
    for seed in SEEDS:
        for field in ('initial_state_sha256', 'initialization'):
            reference = result['R0', seed][field]
            if any(result[r, seed][field] != reference for r in CANDIDATES):
                raise ValueError(f'Same-seed model initialization differs: {seed}/{field}')
    return result


def load_rows(r0_root, new_run_root, *, candidate_layout='formal/{recipe}/seed_{seed}'):
    """Read immutable result files and relocate prediction paths within their run.

    R0 layout is C01's ``formal/YOLOv9_T__quality_aware/seed_N/result.json``.
    The configurable candidate layout is relative to ``new_run_root``. It must
    stay inside that root. Reading does not rewrite any source artifact.
    """
    rows = []
    for recipe in RECIPES:
        root = Path(r0_root if recipe == 'R0' else new_run_root).resolve()
        manifest_path, environment_path = root / 'execution_manifest.json', root / 'environment_lock.json'
        manifest, environment = stable.read_json(manifest_path), stable.read_json(environment_path)
        if canonical_sha(environment['environment']) != environment['fingerprint']:
            raise ValueError('Environment lock canonical fingerprint differs')
        for seed in SEEDS:
            relative = (f'formal/YOLOv9_T__quality_aware/seed_{seed}' if recipe == 'R0'
                        else candidate_layout.format(recipe=recipe, seed=seed))
            folder = (root / relative).resolve()
            if not folder.is_relative_to(root):
                raise ValueError('Run layout escaped specified source root')
            source_path = folder / 'result.json'
            row = deepcopy(stable.read_json(source_path))
            binding = row['binding']
            if (binding['execution_manifest_sha256'] != stable.sha256(manifest_path)
                    or binding['environment_fingerprint'] != environment['fingerprint']):
                raise ValueError('Result source manifest/environment binding differs')
            for field in ('subset_contract_sha256', 'validation_target_sha256'):
                if binding[field] != manifest[field]:
                    raise ValueError(f'Result data contract differs from source manifest: {field}')
            source_paths = dict(row['stable_prediction_paths'])
            for epoch in EPOCHS:
                path = folder / 'epoch_predictions' / f'epoch_{epoch:03d}_valid_records.json'
                if stable.sha256(path) != row['stable_prediction_sha256'][str(epoch)]:
                    raise ValueError('Relocated stable prediction source hash drift')
                row['stable_prediction_paths'][str(epoch)] = str(path.resolve())
            row['recipe'] = recipe
            row['provenance'] = dict(source_result_path=str(source_path),
                source_result_sha256=stable.sha256(source_path), source_root=str(root),
                original_prediction_paths=source_paths, reused_C01=recipe == 'R0',
                manifest_path=str(manifest_path), manifest_sha256=stable.sha256(manifest_path),
                environment_lock_path=str(environment_path), environment_lock_sha256=stable.sha256(environment_path))
            rows.append(row)
    validate_rows(rows)
    return rows


def _means(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(mean=float(array.mean()), sd=float(array.std(ddof=1)),
                n_seeds=len(SEEDS), seeds=list(SEEDS), per_seed=array.tolist())


def summarize_values(values, seed_draws):
    """Reduce one subset's shared-draw values; no additional random sampling.

    ``values[(recipe, seed)]`` has shape (1 + replicates, 5, 9), where index0
    is the unresampled point estimate. Contrast subtraction precedes seed
    resampling and uses the identical image draw for every cell and epoch.
    """
    per_epoch, per_seed, arms, effects = [], [], {}, {}
    for recipe in RECIPES:
        point = np.stack([values[recipe, seed][0] for seed in SEEDS])
        stable_point = point.mean(axis=1)
        arms[recipe] = {name: _means(stable_point[:, j]) for j, name in enumerate(METRICS)}
        for i, seed in enumerate(SEEDS):
            per_seed.append(dict(recipe=recipe, seed=seed, **dict(zip(METRICS, stable_point[i].tolist()))))
            for q, epoch in enumerate(EPOCHS):
                per_epoch.append(dict(recipe=recipe, seed=seed, epoch=epoch,
                                      **dict(zip(METRICS, point[i, q].tolist()))))
        if recipe != 'R0':
            delta = np.stack([values[recipe, seed] - values['R0', seed] for seed in SEEDS], axis=1)
            hierarchical, image_only = stable.reduce_paired_bootstrap(delta[1:], seed_draws)
            stable_delta = delta[0].mean(axis=1)
            effects[recipe] = {name: dict(**_means(stable_delta[:, j]),
                hierarchical_ci=stable.interval(hierarchical[:, j]),
                image_only_ci=stable.interval(image_only[:, j])) for j, name in enumerate(METRICS)}
    return dict(arms=arms, candidate_effects_vs_R0=effects, per_epoch=per_epoch, per_seed=per_seed)


def _compatibility(cells, reports):
    rows = []
    for subset, report in reports.items():
        for point in report['per_seed']:
            recipe, seed = point['recipe'], point['seed']
            source = cells[recipe, seed]
            stable_metrics = {name: point[name] for name in METRICS}
            stable_metrics['map'] = (point['AP50'] + point['AP75']) / 2.
            last = next(row for row in report['per_epoch'] if row['recipe'] == recipe
                        and row['seed'] == seed and row['epoch'] == 80)
            entry = dict(recipe=recipe, seed=seed, subset=subset, n_images=report['n_images'],
                         primary_checkpoint='mean_epochs_76_80',
                         **{f'stable_{name}': value for name, value in stable_metrics.items()},
                         **{f'terminal_{name}': last[name] for name in METRICS})
            entry['terminal_map'] = (last['AP50'] + last['AP75']) / 2.
            entry['best_full99_epoch'] = source.get('best_epoch')
            entry['best_selection_scope'] = 'full99 mAP50_95; secondary only'
            # There is no best89 checkpoint search; best columns belong only to full99.
            if subset == 'full99':
                for scope, field in (('best', 'secondary_best_metrics'), ('terminal_logged', 'secondary_terminal_metrics')):
                    for name, value in source.get(field, {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            entry[f'{scope}_{name}'] = value
            rows.append(entry)
    return rows


def recipe_summary(rows, decision=None, *, subset_contract, output_dir=None):
    """Validate twelve rows, replay sixty archives, and apply frozen D14 selection."""
    decision = deepcopy(DECISION) if decision is None else decision
    validate_decision(decision)
    cells = validate_rows(rows)
    names = stable.validate_contract(subset_contract)
    positions = {name: i for i, name in enumerate(names['full_images'])}
    populations = dict(full99=np.arange(99), without_similar89=np.asarray(
        [positions[name] for name in names['kept_images']]))
    plans = {subset: stable.bootstrap_plan(population, SEEDS, replicates=REPLICATES,
             random_seed=RANDOM_SEED + 100 * i) for i, (subset, population) in enumerate(populations.items())}
    values = {subset: {} for subset in populations}
    sources, parity, reference_targets = [], [], None
    source_files = {}
    for key in sorted(cells):
        row = cells[key]
        provenance = row.get('provenance', {})
        for pathkey, hashkey in (('source_result_path', 'source_result_sha256'),
                                 ('manifest_path', 'manifest_sha256'),
                                 ('environment_lock_path', 'environment_lock_sha256')):
            if pathkey in provenance:
                path, digest = provenance[pathkey], provenance[hashkey]
                if stable.sha256(path) != digest:
                    raise ValueError('Run provenance file changed before analysis')
                source_files[path] = digest
        run_values = {subset: [] for subset in populations}
        for epoch in EPOCHS:
            path = Path(row['stable_prediction_paths'][str(epoch)])
            digest = stable.sha256(path)
            if digest != row['stable_prediction_sha256'][str(epoch)]:
                raise ValueError(f'Stable prediction source hash drift: {key}/{epoch}')
            records = stable.load_records(path, names['full_images'], reference_targets)
            if any(len(item['prediction']['scores']) > 100 or
                   any(score < 0 or score > 1 for score in item['prediction']['scores']) for item in records):
                raise ValueError('Prediction scores must be probabilities with global max_det100')
            target_rows = [dict(image=item['image'], target=item['target']) for item in records]
            if canonical_sha(target_rows) != row['binding']['validation_target_sha256']:
                raise ValueError('Stable prediction GT differs from frozen source-label reference')
            if reference_targets is None:
                reference_targets = {item['image']: item['target'] for item in records}
                counts = Counter(label for item in records for label in item['target']['labels'])
                if counts != {0: 70, 1: 29} or any(len(item['target']['labels']) != 1 for item in records):
                    raise ValueError('Frozen full99 requires one GT per image and class counts70/29')
            prepared = stable.PreparedAP(records)
            checks = prepared.validate(dict(full99=populations['full99'],
                without_similar89=populations['without_similar89'],
                duplicate_bootstrap=plans['full99']['image_draws'][0],
                one_class_repeated=np.repeat(0, 99)))
            parity.append(dict(recipe=key[0], seed=key[1], epoch=epoch, checks=checks,
                               has_score_ties=any(item['has_score_ties'] for item in prepared.classes)))
            for subset, population in populations.items():
                point = prepared.evaluate_one(population)
                if subset == 'full99':
                    logged = row['history'][epoch - 1]['valid']
                    for index, metric in enumerate(('mAP50_95', 'AP50', 'AP75')):
                        if metric in logged and abs(point[index] - float(logged[metric])) > stable.PARITY_TOLERANCE:
                            raise ValueError(f'Logged stable AP differs from epoch predictions: {key}/{epoch}/{metric}')
                samples = prepared.evaluate(plans[subset]['image_draws'])
                run_values[subset].append(np.vstack((point, samples)))
            sources.append(dict(recipe=key[0], seed=key[1], epoch=epoch,
                                path=str(path.resolve()), sha256=digest))
            source_files[str(path.resolve())] = digest
        for subset in populations:
            values[subset][key] = np.stack(run_values[subset], axis=1)
        print(f'recipe analysis {key}: five full99 archives verified', flush=True)
    reports = {subset: dict(n_images=len(population),
        images=[names['full_images'][int(i)] for i in population],
        **summarize_values(values[subset], plans[subset]['seed_draws']))
        for subset, population in populations.items()}
    primary = reports['full99']
    verdict = select_recipe({r: primary['arms'][r]['mAP50_95']['mean'] for r in RECIPES},
                            {r: primary['candidate_effects_vs_R0'][r]['mAP50_95'] for r in CANDIDATES})
    direction = {r: dict(full99_delta=primary['candidate_effects_vs_R0'][r]['mAP50_95']['mean'],
        without_similar89_delta=reports['without_similar89']['candidate_effects_vs_R0'][r]['mAP50_95']['mean'],
        used_for_selection=False) for r in CANDIDATES}
    report = dict(schema_version=1, status='COMPLETE', n_runs=12, newly_trained_runs=9,
        reused_C01_runs=3, recipes=list(RECIPES), seeds=list(SEEDS), epochs=list(EPOCHS),
        metric='all-area COCO-style AP101 mAP50-95', primary_checkpoint='mean_epochs_76_80',
        test_evaluated=False, training_or_inference_performed=False,
        bootstrap=dict(method='hierarchical_paired_seed_and_image_percentile',
            replicates=REPLICATES, random_seed=RANDOM_SEED, auxiliary89_random_seed=RANDOM_SEED + 100,
            confidence=.95, quantile_method='linear',
            image_sampling='One image draw shared across all recipes, seeds, and five epochs within each subset',
            seed_sampling='One paired seed-index draw shared across all four recipes and five epochs',
            epoch_reduction='Recompute sampled AP each epoch; mean five epochs per seed; mean sampled seed IDs',
            image_only='Auxiliary: same image draws, observed seeds fixed',
            multiple_comparisons_adjusted=False, multiplicity_note=MULTIPLICITY_NOTE,
            limitation='Validation-development evidence from three observed seeds; no independent test claim'),
        subsets=reports, decision=verdict, direction_consistency_auxiliary=direction,
        compatibility_rows=_compatibility(cells, reports),
        evaluator_parity=dict(tolerance=stable.PARITY_TOLERANCE, all_passed=True, checks=parity),
        source_records=sources, run_provenance=[dict(recipe=r, seed=s,
            binding=cells[r, s]['binding'], initial_state_sha256=cells[r, s]['initial_state_sha256'],
            provenance=cells[r, s].get('provenance', {})) for r in RECIPES for s in SEEDS],
        model_structure_metadata_sha256=canonical_sha(_structure(cells['R0', 42]['model_info'])),
        subset_contract_canonical_sha256=canonical_sha(subset_contract),
        analysis_sha256=stable.sha256(__file__), stable_analysis_sha256=stable.sha256(stable.__file__))
    if any(stable.sha256(path) != digest for path, digest in source_files.items()):
        raise ValueError('Source file changed during recipe analysis')
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for subset, plan in plans.items():
            draws_path = out / f'{subset}_bootstrap_draws.npz'
            np.savez_compressed(draws_path, **plan, image_ids=np.asarray(names['full_images']),
                                population=populations[subset])
            keys = sorted(values[subset])
            value_path = out / f'{subset}_epoch_bootstrap_values.npz'
            np.savez_compressed(value_path, values=np.stack([values[subset][key] for key in keys]),
                keys=np.asarray([f'{r}/{s}' for r, s in keys]), metrics=np.asarray(METRICS), epochs=np.asarray(EPOCHS))
            for path in (draws_path, value_path):
                artifacts.append(dict(path=str(path.resolve()), sha256=stable.sha256(path)))
        report['bootstrap_artifacts'] = artifacts
        stable.write_json(out / 'evaluator_parity.json', report['evaluator_parity'])
        stable.write_json(out / 'source_verification.json', dict(all_passed=True, files=source_files,
            source_records=sources, run_provenance=report['run_provenance']))
        fields = list(dict.fromkeys(key for row in report['compatibility_rows'] for key in row))
        with (out / 'compatibility_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report['compatibility_rows'])
        stable.write_json(out / 'recipe_analysis.json', report)
        write_d14(report, out)
    return stable._json_safe(report)


def write_d14(report, output_dir):
    """Write the completed selection record without mutating project ledgers."""
    if report.get('status') != 'COMPLETE' or report.get('n_runs') != 12:
        raise ValueError('D14 decision requires all twelve complete recipe cells')
    out = Path(output_dir)
    verdict = report['decision']
    payload = dict(decision_id='D14', status='COMPLETE', **verdict,
        evidence='recipe_analysis.json', metric='mAP50_95',
        analysis_sha256=report['analysis_sha256'],
        subset_contract_canonical_sha256=report['subset_contract_canonical_sha256'])
    stable.write_json(out / 'D14.json', payload)
    lines = ['# D14 训练配方选择', '', f"最终配方：**{verdict['selected_recipe']}**。", '',
        '依据完整 99 图、第 76–80 轮 mAP50–95 均值；R0 复用 C01，R1–R3 各 3 个配对 seed（42–44）。', '',
        '| 配方 | 稳定均值 | 对 R0 提升（百分点） | 配对 95% 区间（百分点） | 有效 |',
        '|---|---:|---:|---|---|']
    for recipe in RECIPES:
        mean = verdict['stable_mean_by_recipe'][recipe]
        if recipe == 'R0':
            lines.append(f'| R0 | {100 * mean:.4f}% | — | — | 基线 |')
        else:
            gate = verdict['candidate_gates'][recipe]
            ci = gate['hierarchical_ci']
            ci_text = (f"[{100 * ci['low']:.4f}, {100 * ci['high']:.4f}]"
                       if ci['low'] is not None and ci['high'] is not None else '未定义')
            lines.append(f"| {recipe} | {100 * mean:.4f}% | {gate['delta_percentage_points']:.4f} | {ci_text} | {'是' if gate['effective'] else '否'} |")
    lines += ['', '判定同时要求提升至少 1 个百分点且区间下限严格大于 0。多个有效配方中，选择距最高有效稳定均值不超过 0.5 个百分点的最简单配方；全部无效则沿用 R0。', '',
        '区间来自 2,000 次配对 seed/图像分层自助抽样，随机种子 20261013；同一图像及 seed 抽样在全部配方和五轮之间共享。三个预先登记候选使用点态 95% 区间，未做多重比较校正。', '',
        '89 图子集、best 与第 80 轮指标仅作辅助，不参与选择。结论属于验证集开发证据；没有独立测试集或硬件实测结论。', '',
        '完整数值、逐 seed/逐轮/逐类结果、原始指标及来源校验见 [recipe_analysis.json](recipe_analysis.json) 与 [compatibility_metrics.csv](compatibility_metrics.csv)。', '']
    (out / 'D14.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r0-root', type=Path, required=True)
    parser.add_argument('--new-run-root', type=Path, required=True)
    parser.add_argument('--candidate-layout', default='formal/{recipe}/seed_{seed}')
    parser.add_argument('--subset-contract', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    stable.torch.set_num_threads(2)
    rows = load_rows(args.r0_root, args.new_run_root, candidate_layout=args.candidate_layout)
    if any(row['binding']['subset_contract_sha256'] != stable.sha256(args.subset_contract) for row in rows):
        raise ValueError('CLI subset-contract file differs from the frozen campaign binding')
    report = recipe_summary(rows, subset_contract=stable.read_json(args.subset_contract), output_dir=args.output_dir)
    print(json.dumps(report['decision'], ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
