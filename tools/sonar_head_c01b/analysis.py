"""CPU-only, paired C01b versus C01 analysis for the frozen late epochs.

Each epoch is evaluated separately.  Five epoch AP values are averaged before
the three paired training seeds; predictions are never pooled or ensembled.
The channel choice uses only the two registered point-estimate conditions.
Bootstrap intervals describe uncertainty and do not introduce another gate.
No training, inference, remote access, or edits to D17 are performed here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.sonar_head_factorial import stable_analysis as stable

EPOCHS = stable.EPOCHS
METRICS = stable.METRICS
VARIANTS = ('C01', 'C01b')
SEEDS = (42, 43, 44)
SCALES = ('P2', 'P3', 'P4', 'P5')
PRIMARY_METRICS = ('mAP50_95', 'AP75')
MINIMUM_DELTA = -.005
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_SEED = 20261013
THRESHOLD_COMPARE_TOLERANCE = 1e-12
PARITY_TOLERANCE = stable.PARITY_TOLERANCE


def _channels(value):
    """Normalize a measured uniform E0 raw channel contract, without assuming 34."""
    if isinstance(value, Mapping):
        if set(value) != set(SCALES):
            raise ValueError('C01 channels must cover exactly P2, P3, P4, and P5')
        values = [value[name] for name in SCALES]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    else:
        values = [value] * len(SCALES)
    if (len(values) != len(SCALES)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or int(v) <= 0 for v in values)
            or len(set(map(int, values))) != 1):
        raise ValueError('Measured C01 channels must be four identical positive integers')
    return [int(v) for v in values]


def channel_decision(mean_deltas, *, channels_C01):
    """Inclusive two-metric point-estimate gate; CI never changes this choice."""
    baseline = _channels(channels_C01)
    conditions = {}
    for name in PRIMARY_METRICS:
        value = float(mean_deltas.get(name, float('nan')))
        if not np.isfinite(value):
            raise ValueError(f'Finite paired stable delta required: {name}')
        conditions[name] = dict(
            delta=value, delta_percentage_points=value * 100.,
            minimum_delta_absolute=MINIMUM_DELTA,
            minimum_delta_percentage_points=MINIMUM_DELTA * 100.,
            threshold_met=bool(value >= MINIMUM_DELTA - THRESHOLD_COMPARE_TOLERANCE))
    passed = all(c['threshold_met'] for c in conditions.values())
    chosen = [6] * len(SCALES) if passed else baseline
    return dict(
        primary_subset='full99', primary_checkpoint='mean_epochs_76_80',
        direction='C01b_minus_C01', paired_seeds=list(SEEDS),
        epoch_reduction='mean per-epoch AP over epochs76-80, then mean three paired seeds',
        conditions=conditions, gate_pass=passed,
        criterion='mean(C01b-C01) mAP50_95 >= -0.005 AND AP75 >= -0.005',
        threshold_compare_tolerance=THRESHOLD_COMPARE_TOLERANCE,
        ci_used_in_gate=False, statistical_noninferiority_established=False,
        selected_variant='C01b' if passed else 'C01',
        selected_distribution='direct_distance_no_DFL' if passed else 'retain_C01_DFL',
        decision_status='SELECT_NO_DFL' if passed else 'KEEP_DFL',
        baseline_raw_output_channels_per_scale=baseline,
        no_DFL_raw_output_channels_per_scale=[6] * len(SCALES),
        selected_raw_output_channels_per_scale=chosen,
        selected_raw_output_channels=chosen[0], scales=list(SCALES),
        automatic_D17_edit_executed=False, automatic_search_executed=False)


def validate_decision(decision):
    """Reject changes to the prespecified gate or bootstrap after observing data."""
    if not isinstance(decision, dict):
        raise ValueError('Frozen C01b decision must be a mapping')
    required = dict(primary_checkpoint='mean_epochs_76_80', primary_subset='full99',
                    direction='C01b_minus_C01', ci_used_in_gate=False)
    for name, expected in required.items():
        if decision.get(name, expected) != expected:
            raise ValueError(f'Unregistered C01b decision field: {name}')
    thresholds = decision.get('minimum_delta_absolute',
                              {name: MINIMUM_DELTA for name in PRIMARY_METRICS})
    if thresholds != {name: MINIMUM_DELTA for name in PRIMARY_METRICS}:
        raise ValueError('C01b requires both mAP50_95 and AP75 inclusive -0.005 thresholds')
    boot = decision.get('bootstrap', {})
    if (not isinstance(boot, dict)
            or boot.get('replicates', BOOTSTRAP_REPLICATES) != BOOTSTRAP_REPLICATES
            or boot.get('random_seed', BOOTSTRAP_RANDOM_SEED) != BOOTSTRAP_RANDOM_SEED
            or boot.get('confidence', .95) != .95):
        raise ValueError('Frozen bootstrap requires 2000 replicates, seed20261013, CI95')
    return dict(**required, minimum_delta_absolute=thresholds,
                bootstrap=dict(replicates=BOOTSTRAP_REPLICATES,
                               random_seed=BOOTSTRAP_RANDOM_SEED, confidence=.95))


def _identity(rows, field, *, binding=True):
    values = [(r.get('binding', {}) if binding else r).get(field) for r in rows]
    if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != 1:
        raise ValueError(f'Run comparison identity differs or is missing: {field}')
    return values[0]


def validate_rows(rows_by_variant, *, channels_C01):
    """Check six complete rows and the shared paired comparison identity."""
    if not isinstance(rows_by_variant, dict) or set(rows_by_variant) != set(VARIANTS):
        raise ValueError('Exactly C01 and C01b variant groups required')
    channels = dict(C01=_channels(channels_C01), C01b=[6] * len(SCALES))
    cells, all_rows, channel_checks = {}, [], []
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        if not isinstance(rows, (list, tuple)) or len(rows) != len(SEEDS):
            raise ValueError('Exactly six rows required: C01 and C01b seeds42-44')
        _identity(rows, 'execution_manifest_sha256')
        for row in rows:
            binding = row.get('binding', {})
            seed = binding.get('seed')
            key = (variant, seed)
            if (binding.get('backbone') != 'E0' or binding.get('head') != 'quality_aware'
                    or type(seed) is not int or seed not in SEEDS or key in cells):
                raise ValueError('Exactly six E0 quality_aware rows required; unexpected or duplicate seed')
            if (row.get('epochs') != 80 or row.get('formal_eligible') is not True
                    or row.get('primary_checkpoint') != 'mean_epochs_76_80'):
                raise ValueError('C01b comparison requires complete80 formal runs and mean_epochs_76_80')
            if row.get('test_evaluated', False) is not False or binding.get('test_evaluated', False) is not False:
                raise ValueError('Test-set evaluation is outside the C01b validation protocol')
            for field in ('stable_prediction_paths', 'stable_prediction_sha256'):
                if set(row.get(field, {})) != set(map(str, EPOCHS)):
                    raise ValueError(f'{field}: require exactly stable epochs76-80')
            history = row.get('history', [])
            if [h.get('epoch') for h in history] != list(range(1, 81)):
                raise ValueError('Full history epochs1-80 required, with no missing epoch')
            for item in history:
                for name in PRIMARY_METRICS:
                    value = float(item.get('valid', {}).get(name, float('nan')))
                    if not np.isfinite(value) or not 0. <= value <= 1.:
                        raise ValueError(f'Full finite AP history required: {name}')
            for name in PRIMARY_METRICS:
                value = float(row.get('metrics', {}).get(name, float('nan')))
                observed = np.mean([history[e-1]['valid'][name] for e in EPOCHS])
                if not np.isfinite(value) or abs(value - observed) > PARITY_TOLERANCE:
                    raise ValueError(f'Row stable metrics differ from mean epochs76-80: {name}')
            measured = row.get('model_info', {}).get('head_contract', {}).get('raw_output_channels_per_scale')
            if measured is not None and _channels(measured) != channels[variant]:
                raise ValueError(f'Measured model_info output channels disagree: {variant}/{seed}')
            channel_checks.append(dict(variant=variant, seed=seed,
                measured_model_info_present=measured is not None,
                raw_output_channels_per_scale=channels[variant],
                checked_against_model_info=measured is not None))
            cells[key] = row
            all_rows.append(row)
    if set(cells) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Exactly six paired variant/seed cells required')
    common = dict(environment_fingerprint=_identity(all_rows, 'environment_fingerprint'),
                  comparison_contract_sha256=_identity(all_rows, 'comparison_contract_sha256', binding=False))
    for field in ('subset_contract_sha256', 'validation_target_sha256'):
        if any(field in r.get('binding', {}) for r in all_rows):
            common[field] = _identity(all_rows, field)
    return cells, common, channel_checks


def _pairing_witnesses(cells):
    """Check archived data-order/augmentation witnesses when supplied by training."""
    checks = []
    fields = ('augmented_batches_sha256', 'image_order_sha256', 'valid_batches_sha256')
    for seed in SEEDS:
        left, right = (cells[v, seed]['history'] for v in VARIANTS)
        for epoch, (a, b) in enumerate(zip(left, right), 1):
            for field in fields:
                if field in a or field in b:
                    if not a.get(field) or a.get(field) != b.get(field):
                        raise ValueError(f'Paired training data witness differs: seed{seed}/{epoch}/{field}')
                    checks.append(dict(seed=seed, epoch=epoch, field=field, sha256=a[field], passed=True))
    return dict(checks=checks, n_checks=len(checks),
                all_expected_present=len(checks) == len(SEEDS) * 80 * len(fields),
                explanation='Missing witnesses are not invented; source-run verification remains upstream')


def _effect_report(delta, seed_draws):
    # [point + repeats, paired seed, five epochs, metrics]
    return stable._effect_report(delta[0], delta[1:], seed_draws, SEEDS)


def c01b_summary(rows_by_variant, decision, *, subset_contract, channels_C01,
                 output_dir=None):
    """Recompute exact AP, paired bootstrap intervals, and the raw-channel choice.

    ``channels_C01`` comes from the actual C01 model; 34 is never presumed.
    Each row follows frozen ``run_result`` and additionally carries the same
    ``comparison_contract_sha256``.  Execution manifests may differ by variant,
    while all six rows must share the environment and comparison identity.
    """
    registered = validate_decision(decision)
    cells, common, channel_checks = validate_rows(rows_by_variant, channels_C01=channels_C01)
    witnesses = _pairing_witnesses(cells)
    names = stable.validate_contract(subset_contract)
    positions = {name: i for i, name in enumerate(names['full_images'])}
    populations = dict(full99=np.arange(99), without_similar89=np.asarray(
        [positions[name] for name in names['kept_images']], dtype=np.int64))
    plans = {subset: stable.bootstrap_plan(population, SEEDS,
        replicates=BOOTSTRAP_REPLICATES, random_seed=BOOTSTRAP_RANDOM_SEED + 100 * k)
        for k, (subset, population) in enumerate(populations.items())}
    values = {subset: {} for subset in populations}
    sources, parity, reference_targets = [], [], None
    for key in sorted(cells):
        variant, seed = key
        row = cells[key]
        run_values = {subset: [] for subset in populations}
        for epoch in EPOCHS:
            path = Path(row['stable_prediction_paths'][str(epoch)])
            source_hash = stable.sha256(path)
            if source_hash != row['stable_prediction_sha256'][str(epoch)]:
                raise ValueError(f'Stable prediction source hash drift: {variant}/{seed}/{epoch}')
            records = stable.load_records(path, names['full_images'], reference_targets)
            if reference_targets is None:
                reference_targets = {r['image']: r['target'] for r in records}
                counts = Counter(c for r in records for c in r['target']['labels'])
                if counts != {0: 70, 1: 29} or any(len(r['target']['labels']) != 1 for r in records):
                    raise ValueError('Frozen valid99 requires one GT/image and class counts70/29')
            prepared = stable.PreparedAP(records)
            oracle_samples = dict(full99=populations['full99'],
                without_similar89=populations['without_similar89'],
                duplicate_bootstrap=plans['full99']['image_draws'][0],
                one_class_repeated=np.repeat(0, 99))
            checks = prepared.validate(oracle_samples)
            parity.append(dict(variant=variant, seed=seed, epoch=epoch,
                has_score_ties=any(c['has_score_ties'] for c in prepared.classes), checks=checks))
            for subset, population in populations.items():
                point = prepared.evaluate_one(population)
                if subset == 'full99':
                    for name in PRIMARY_METRICS:
                        j = METRICS.index(name)
                        logged = float(row['history'][epoch-1]['valid'][name])
                        if abs(point[j] - logged) > PARITY_TOLERANCE:
                            raise ValueError(f'Logged stable {name} differs from epoch predictions: {variant}/{seed}/{epoch}')
                samples = prepared.evaluate(plans[subset]['image_draws'])
                run_values[subset].append(np.vstack((point, samples)))
            sources.append(dict(variant=variant, seed=seed, epoch=epoch,
                path=str(path.resolve()), sha256=source_hash,
                execution_manifest_sha256=row['binding']['execution_manifest_sha256']))
        for subset in populations:
            values[subset][key] = np.stack(run_values[subset], axis=1)
        print(f'C01b paired analysis {variant} seed{seed}: five epochs, both subsets checked', flush=True)
    reports = {}
    for subset, population in populations.items():
        subset_values = values[subset]
        arms, per_epoch, per_seed = {}, [], []
        for variant in VARIANTS:
            points = np.stack([subset_values[variant, seed][0] for seed in SEEDS])
            stable_points = points.mean(axis=1)
            arms[variant] = {name: dict(**stable._mean_sd(stable_points[:, j]), seeds=list(SEEDS))
                             for j, name in enumerate(METRICS)}
            for i, seed in enumerate(SEEDS):
                per_seed.append(dict(variant=variant, seed=seed,
                                     **dict(zip(METRICS, stable_points[i].tolist()))))
                for q, epoch in enumerate(EPOCHS):
                    per_epoch.append(dict(variant=variant, seed=seed, epoch=epoch,
                                         **dict(zip(METRICS, points[i, q].tolist()))))
        delta = np.stack([subset_values['C01b', seed] - subset_values['C01', seed]
                          for seed in SEEDS], axis=1)
        reports[subset] = dict(n_images=len(population),
            images=[names['full_images'][i] for i in population],
            per_epoch=per_epoch, per_seed=per_seed, arms=arms,
            paired_difference_C01b_minus_C01=_effect_report(delta, plans[subset]['seed_draws']),
            used_in_channel_gate=subset == 'full99')
    primary = reports['full99']['paired_difference_C01b_minus_C01']
    verdict = channel_decision({name: primary[name]['mean'] for name in PRIMARY_METRICS},
                               channels_C01=channels_C01)
    report = dict(schema_version=1, status='COMPLETE', n_runs=6, variants=list(VARIANTS),
        backbone='E0', head='quality_aware', paired_seeds=list(SEEDS), epochs=list(EPOCHS),
        primary_checkpoint='mean_epochs_76_80', primary_subset='full99',
        metric='all-area COCO-style AP101 mAP50-95 and AP75',
        comparison_scope='C01b direct-distance/no-DFL with GIoU versus frozen C01 quality-head package; not isolated causal attribution to DFL',
        registered_decision=registered, comparison_identity=common,
        source_bindings=[dict(variant=v, binding=cells[v, s]['binding']) for v in VARIANTS for s in SEEDS],
        channel_contract_checks=channel_checks, paired_training_witnesses=witnesses,
        test_evaluated=False, training_or_inference_performed=False,
        bootstrap=dict(method='hierarchical_paired_seed_and_image_percentile',
            replicates=BOOTSTRAP_REPLICATES, random_seed=BOOTSTRAP_RANDOM_SEED,
            confidence=.95, quantile_method='linear', auxiliary_subset_random_seed=BOOTSTRAP_RANDOM_SEED + 100,
            image_sampling='One image draw shared across both variants, all three seeds, and all five epochs',
            seed_sampling='Three paired seed IDs sampled with replacement; same indices for both variants',
            epoch_reduction='Recompute AP per sampled epoch, mean5 per seed, then mean sampled paired seeds',
            prediction_ensembling=False, image_only='Same image draws, fixed observed paired training seeds; auxiliary CI',
            ci_used_in_gate=False, limitation='Validation-development comparison with only three observed training seeds; intervals are not a formal noninferiority claim'),
        subsets=reports, arms=reports['full99']['arms'],
        paired_difference_C01b_minus_C01=primary, decision=verdict,
        evaluator_parity=dict(tolerance=PARITY_TOLERANCE, checks=parity, all_passed=True),
        source_records=sources, subset_contract_canonical_sha256=hashlib.sha256(
            json.dumps(subset_contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        analysis_sha256=stable.sha256(__file__),
        stable_analysis_sha256=stable.sha256(stable.__file__))
    if any(stable.sha256(item['path']) != item['sha256'] for item in sources):
        raise ValueError('Prediction source changed during C01b analysis')
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for subset, plan in plans.items():
            np.savez_compressed(output_dir / f'{subset}_bootstrap_draws.npz',
                                image_draws=plan['image_draws'], seed_draws=plan['seed_draws'],
                                seed_ids=np.asarray(SEEDS), random_seed=plan['random_seed'])
            keys = sorted(values[subset])
            np.savez_compressed(output_dir / f'{subset}_epoch_bootstrap_values.npz',
                values=np.stack([values[subset][key] for key in keys]),
                keys=np.asarray(['/'.join(map(str, key)) for key in keys]),
                metrics=np.asarray(METRICS), epochs=np.asarray(EPOCHS))
        stable.write_json(output_dir / 'c01b_analysis.json', report)
        stable.write_json(output_dir / 'evaluator_parity.json', report['evaluator_parity'])
    return stable._json_safe(report)
