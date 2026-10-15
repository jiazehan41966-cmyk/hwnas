"""Quantized multi-objective screening; operator costs are never board costs."""
from __future__ import annotations

from copy import deepcopy
import math

METRICS = ('Latency', 'LUT', 'DSP', 'BRAM')
PROFILES = ('W8A8', 'HEAD_W8A16', 'HEAD_W16A16')
POLICIES = ('reference_low', 'reference_high', 'reference_min_latency',
            'reference_min_dsp', 'reference_min_lut', 'reference_min_bram')


def finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def score(value):
    if not finite(value) or value > 1:
        raise ValueError('Invalid mAP50_95')
    return value


def validate_budget(budget, required=True):
    if budget.get('scope') != 'operator_peak_and_serial_compute':
        raise ValueError('Only operator screening budgets are supported; board costs are separate')
    limits = budget.get('limits', {})
    if set(limits) != set(METRICS):
        raise ValueError('Budget requires Latency/LUT/DSP/BRAM limits')
    missing = [k for k in METRICS if limits[k] is None]
    if missing and required:
        raise ValueError('Hardware budget required: ' + ','.join(missing))
    if any(v is not None and not finite(v, positive=True) for v in limits.values()):
        raise ValueError('Budget limits must be finite and positive')
    return not missing


def validate_config(cfg, require_budget=False):
    if cfg.get('version') != 'sonar_joint_v2':
        raise ValueError('Use a separate v2 configuration')
    if cfg.get('test_policy') != 'sealed' or cfg.get('evaluation_split') != 'valid' or cfg.get('deployment_claim') is not False:
        raise ValueError('Validation-only selection and sealed test required')
    if cfg.get('budgets') != [40, 60, 80] or cfg.get('populations') != [144, 108, 72]:
        raise ValueError('This runner preserves the 40/60/80 training budgets')
    if cfg.get('search_seed') != 42 or cfg.get('confirmation_seeds') != [42, 43, 44]:
        raise ValueError('Frozen paired seeds required')
    if (cfg.get('quant_profiles') != list(PROFILES) or len(cfg.get('mapping_policies', [])) != len(POLICIES)
            or set(cfg.get('mapping_policies', [])) != set(POLICIES)):
        raise ValueError('Measured profiles and mappings required')
    if cfg.get('calibration') != dict(split='train', images=64, augmentation=False):
        raise ValueError('Fixed train-only calibration required')
    if cfg.get('checkpoint_selection') != 'best_valid_mAP50_95_earliest_tie':
        raise ValueError('Checkpoint selection must match the reused training worker')
    expected = dict(optimizer='AdamW', lr=.001, weight_decay=.0001, scheduler='none',
                    amp=False, workers=0, initialization='independent_from_scratch')
    if cfg.get('training') != expected:
        raise ValueError('Training settings must match the reused training worker')
    if cfg.get('shortlist') != 12:
        raise ValueError('Shortlist must retain its paired confirmation budget')
    promotion = cfg['promotion']
    if promotion.get('metric') != 'budget_feasible_quantized_mAP50_95_pareto':
        raise ValueError('Quantized Pareto promotion required')
    if not finite(promotion.get('fraction')) or promotion['fraction'] > .5:
        raise ValueError('Invalid boundary fraction')
    if not finite(promotion.get('accuracy_epsilon')) or promotion['accuracy_epsilon'] > 1:
        raise ValueError('Invalid boundary accuracy epsilon')
    if not finite(promotion.get('resource_fraction'), True) or promotion['resource_fraction'] > 1:
        raise ValueError('Invalid boundary resource fraction')
    if cfg['hardware_budget'].get('latency_unit') != 'us' or cfg['hardware_budget'].get('bram_unit') != 'BRAM18':
        raise ValueError('Cost API uses microseconds and BRAM18')
    validate_budget(cfg['hardware_budget'], require_budget)
    gate = cfg['admission']
    if not finite(gate['absolute_drop']) or not finite(gate['relative_drop']) or gate['relative_drop'] > 1 or gate['improved_seeds'] != 2:
        raise ValueError('Invalid paired quantization admission')
    return cfg


def filter_costs(rows, budget):
    if not validate_budget(budget, required=False):
        return dict(status='BUDGET_REQUIRED', rows=[], rejected=deepcopy(rows))
    accepted, rejected, seen = [], [], set()
    for original in rows:
        row = deepcopy(original)
        key = (row['architecture_id'], row['profile'], row['mapping_policy'])
        if key in seen:
            raise ValueError('Duplicate hardware triple')
        seen.add(key)
        reasons = []
        if row.get('status') != 'LOOKED_UP' or row.get('measurement_valid') is not True or row.get('timing_eligible') is not True:
            reasons.append('NEEDS_VALID_TIMING_ELIGIBLE_MEASUREMENT')
        for metric in METRICS:
            if not finite(row.get(metric)):
                reasons.append('INVALID_' + metric)
            elif row[metric] > budget['limits'][metric]:
                reasons.append('OVER_BUDGET_' + metric)
        row['screening_reasons'] = reasons
        row['deployment'] = 'NOT_CLAIMED'
        row['screening_scope'] = budget['scope']
        if reasons:
            rejected.append(row)
        else:
            row['cost_pressure'] = max(row[k] / budget['limits'][k] for k in METRICS)
            accepted.append(row)
    return dict(status='READY' if accepted else 'NO_BUDGET_FEASIBLE_TRIPLES', rows=accepted, rejected=rejected)


def joint_points(evaluations, costs, budget):
    filtered = filter_costs(costs, budget)
    by_key = {}
    for result in evaluations:
        aid = result['architecture_id']
        if aid in by_key:
            raise ValueError('Duplicate architecture evaluation')
        if result.get('split') != 'valid' or result.get('test_evaluated') is not False:
            raise ValueError('Validation-only quantization required')
        score(result['fp32']['mAP50_95'])
        for p in result['profiles'].values():
            score(p['mAP50_95'])
        by_key[aid] = result
    points = []
    for cost in filtered['rows']:
        result = by_key.get(cost['architecture_id'])
        if result is None:
            continue
        profile = result['profiles'].get(cost['profile'])
        if profile is None:
            raise ValueError('Missing quantized profile evaluation')
        point = deepcopy(cost)
        point.update(Accuracy=profile['mAP50_95'], fp32_mAP50_95=result['fp32']['mAP50_95'],
                     checkpoint_sha256=result['checkpoint_sha256'],
                     calibration_sha256=result['calibration_sha256'],
                     quant_profile_sha256=profile['profile_sha256'], seed=result['seed'], budget=result['budget'],
                     accuracy_split='valid', test_evaluated=False, n_seeds=1,
                     accuracy_admission='PROVISIONAL_SINGLE_SEED', deployment='NOT_CLAIMED')
        points.append(point)
    return points


def dominates(a, b):
    return (a['Accuracy'] >= b['Accuracy'] and all(a[k] <= b[k] for k in METRICS)
            and (a['Accuracy'] > b['Accuracy'] or any(a[k] < b[k] for k in METRICS)))


def fronts(points):
    remaining = list(points)
    result = []
    while remaining:
        front = [p for p in remaining if not any(dominates(other, p) for other in remaining)]
        result.append(front)
        removed = {id(p) for p in front}
        remaining = [p for p in remaining if id(p) not in removed]
    return result


def point_key(point):
    return (-point['Accuracy'], point['cost_pressure'], point['architecture_id'],
            point['profile'], point['mapping_policy'])


def promote(points, count, promotion):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Positive promotion count required')
    # Remove within-architecture dominated points before the global ranking.
    pools = {}
    for point in points:
        score(point['Accuracy'])
        if any(not finite(point.get(k)) for k in METRICS) or not finite(point.get('cost_pressure')):
            raise ValueError('Invalid screening point')
        pools.setdefault(point['architecture_id'], []).append(point)
    reduced = [p for pool in pools.values() for p in fronts(pool)[0]]
    representatives = {}
    for rank, front in enumerate(fronts(reduced)):
        for point in sorted(front, key=point_key):
            representatives.setdefault(point['architecture_id'], dict(
                architecture_id=point['architecture_id'], front_rank=rank,
                quantized_mAP50_95=point['Accuracy'], cost_pressure=point['cost_pressure'],
                profile=point['profile'], mapping_policy=point['mapping_policy'], point=deepcopy(point)))
    ranked = sorted(representatives.values(), key=lambda r: (r['front_rank'],
                    -r['quantized_mAP50_95'], r['cost_pressure'], r['architecture_id']))
    n = min(count, len(ranked))
    quota = min(math.ceil(n * promotion['fraction']), max(0, n - 1))
    main = ranked[:n - quota]
    rest = ranked[n - quota:]
    cutoff = main[-1]['quantized_mAP50_95'] if main else 0
    boundary_pool = [r for r in rest if abs(r['quantized_mAP50_95'] - cutoff) <= promotion['accuracy_epsilon']
                     or r['cost_pressure'] >= promotion['resource_fraction']]
    boundary_pool.sort(key=lambda r: (abs(r['quantized_mAP50_95'] - cutoff), r['front_rank'],
                                     -r['quantized_mAP50_95'], r['architecture_id']))
    boundary = boundary_pool[:quota]
    chosen = main + boundary
    chosen_ids = {r['architecture_id'] for r in chosen}
    for row in ranked:
        if len(chosen) == n:
            break
        if row['architecture_id'] not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row['architecture_id'])
    return dict(selected=[r['architecture_id'] for r in chosen], ranked=ranked,
                boundary=[r['architecture_id'] for r in boundary], points=deepcopy(points),
                scope='provisional_quantized_accuracy_with_operator_costs', test_evaluated=False,
                deployment='NOT_CLAIMED')
