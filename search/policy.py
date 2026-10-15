"""Retained numeric Pareto and promotion policy, independent of runtime."""
from copy import deepcopy
import math

METRICS = ("Latency", "LUT", "DSP", "BRAM")

def finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))

def score(value):
    if not finite(value) or value > 1:
        raise ValueError('Invalid mAP50_95')
    return value

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
