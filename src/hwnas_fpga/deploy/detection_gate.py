"""Detection-only accuracy admission; classification gates remain unchanged."""
from __future__ import annotations

import math
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence


METRICS = ('mAP50_95', 'Recall')
SEEDS = (42, 43, 44)


def relative_accuracy_gate(candidate: Sequence[Mapping], reference: Sequence[Mapping], *,
                           precision: str, reference_name: str = 'D') -> dict:
    """Compare same-precision three-seed means; missing evidence is not low accuracy.

    Required row identity: network, architecture_id, seed, precision,
    checkpoint_sha256, evaluator_sha256, calibration_sha256, split,
    checkpoint_policy, training_budget, test_evaluated; two explicit metrics.
    """
    reason = []
    def validate(rows, role):
        if len(rows) != 3 or {r.get('seed') for r in rows} != set(SEEDS):
            reason.append(f'{role}: expected exactly seeds 42/43/44')
        if len({r.get('architecture_id') for r in rows}) != 1 or len({r.get('network') for r in rows}) != 1:
            reason.append(f'{role}: inconsistent architecture/network')
        for r in rows:
            if r.get('precision') != precision:
                reason.append(f'{role}: precision mismatch')
            if r.get('split') != 'valid' or r.get('test_evaluated') is not False:
                reason.append(f'{role}: only sealed-test valid evidence allowed')
            if r.get('checkpoint_policy') != 'ALL_BEST_PT' or r.get('training_budget') != 80:
                reason.append(f'{role}: incomplete or incompatible training/checkpoint protocol')
            for key in ('checkpoint_sha256', 'evaluator_sha256', 'calibration_sha256', 'split_manifest_sha256'):
                value = r.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    reason.append(f'{role}: missing/invalid {key}')
            for key in METRICS:
                v = r.get(key)
                if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1:
                    reason.append(f'{role}: invalid {key}')
    validate(candidate, 'candidate'); validate(reference, 'reference')
    if any(r.get('network') != reference_name for r in reference):
        reason.append('wrong reference network')
    for key in ('evaluator_sha256', 'calibration_sha256', 'split_manifest_sha256'):
        if len({r.get(key) for r in list(candidate)+list(reference)}) != 1:
            reason.append(f'non-matched {key}')
    base = dict(precision=precision, reference_network=reference_name, reference_precision=precision,
                criterion='mean(candidate) >= mean(D) for both mAP50_95 and Recall',
                application_accuracy_sufficient='NOT_ESTABLISHED_RELATIVE_ONLY', test_evaluated=False)
    if reason:
        return dict(base, status='BLOCKED_EVIDENCE', passed=False, reasons=sorted(set(reason)), metrics={})
    cand = {r['seed']: r for r in candidate}; ref = {r['seed']: r for r in reference}
    stats = {}
    for key in METRICS:
        a = [float(cand[s][key]) for s in SEEDS]; b = [float(ref[s][key]) for s in SEEDS]
        d = [x-y for x,y in zip(a,b)]
        stats[key] = dict(candidate_mean=statistics.mean(a), reference_mean=statistics.mean(b),
            candidate_std=statistics.stdev(a), reference_std=statistics.stdev(b),
            paired_deltas={str(s):v for s,v in zip(SEEDS,d)}, mean_delta=statistics.mean(d),
            std_delta=statistics.stdev(d), min_delta=min(d), max_delta=max(d),
            positive_seed_count=sum(v>0 for v in d), passes=statistics.mean(a) >= statistics.mean(b)-1e-12)
    passed = all(r['passes'] for r in stats.values())
    control = all(r['network'] == reference_name for r in candidate)
    return dict(base,status=('CONTROL_PASS' if control else 'PASS') if passed else 'STOP_ACCURACY_BELOW_D',
                passed=passed, metrics=stats, reasons=[] if passed else [k+' below D' for k,v in stats.items() if not v['passes']])


def action_decision(action: str, fp32: Mapping, int8: Mapping, *,
                    hardware: Mapping | None = None, new_evaluation_budget: int = 0) -> dict:
    """Downstream transitions fail closed; never stop an existing formal run mid-epoch."""
    stages = ('search', 'promotion', 'ptq', 'parameter_export', 'integer_parity', 'hls', 'route', 'board')
    if action not in stages:
        raise ValueError(f'unknown action {action}')
    quality = [fp32] if action in ('search','promotion','ptq') else [fp32,int8]
    if any(not q.get('passed',False) for q in quality):
        status = 'STOP_ACCURACY_BELOW_D' if any(q.get('status')=='STOP_ACCURACY_BELOW_D' for q in quality) else 'BLOCKED_EVIDENCE'
        return dict(action=action,allowed=False,status=status,reasons=[q.get('status','MISSING_QUALITY') for q in quality if not q.get('passed')])
    if action == 'search' and new_evaluation_budget <= 0:
        return dict(action=action,allowed=False,status='STOP_SEARCH_BUDGET',reasons=['No new architecture evaluation budget; reuse frozen candidates'])
    required = {'integer_parity':['trained_parameter_package'],
                'hls':['trained_parameter_package','trained_integer_parity'],
                'route':['trained_parameter_package','trained_integer_parity','full_detector_hls'],
                'board':['trained_parameter_package','trained_integer_parity','full_detector_hls','full_detector_route']} .get(action,[])
    hw = hardware or {}
    missing=[k for k in required if hw.get(k) is not True]
    return dict(action=action,allowed=not missing,status='BLOCKED_HARDWARE_EVIDENCE' if missing else 'READY',reasons=missing)


def guarded_execute(command: Sequence[str], decision: Mapping, *, journal: Callable, runner=None):
    """Journal the decision durably BEFORE dispatch; never shell-expand a command."""
    if not command or isinstance(command,str):
        raise ValueError('command must be a nonempty argv sequence')
    journal(dict(event='ACTION_GATE',decision=dict(decision),command=list(command)))
    if decision.get('allowed') is not True:
        raise RuntimeError(f"Action stopped: {decision.get('status')}: {decision.get('reasons')}")
    executor=runner or subprocess.run
    result=executor(list(command),check=False,shell=False)
    journal(dict(event='ACTION_EXIT',returncode=result.returncode,action=decision['action']))
    return result
