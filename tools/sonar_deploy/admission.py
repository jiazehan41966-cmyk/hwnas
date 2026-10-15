"""Checkpoint-bound admission for real INT8 detectors, independent of old searches.

No stage is inferred from a synthetic operator witness or a claimed boolean.
Every successful transition is backed by immutable identities, local artifact
digests and a durable 77-field metric ledger.  The legacy deployment CLI remains
available; this module is used only with its explicit real-deployment root.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hwnas_fpga.deploy.detection_gate import relative_accuracy_gate
from hwnas_fpga.metrics_ledger import (
    append_metric_record, build_metric_record, read_metric_records,
    require_measured_metrics,
)

NETWORKS = ('D', 'C', 'B1')
ARCHITECTURES = {
    'D': 'P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3',
    'C': 'P3_P4_P5_s1k3_s2k5_s4k5_s5k3_s6k5_s7k3',
    'B1': 'P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5',
}
SEEDS = (42, 43, 44)
SCHEMA = 'real-int8-deployment-v1'
IDENTITY = ('network', 'seed', 'architecture_id', 'checkpoint_sha256',
            'profile_sha256', 'calibration_sha256', 'numeric_contract_hash',
            'split_manifest_sha256', 'evaluator_sha256')
METRICS = {'AP50': 'map50', 'AP75': 'map75', 'mAP50_95': 'map50_95',
           'Precision': 'precision', 'Recall': 'recall'}
STAGES = ('integer_parity', 'hls', 'route', 'board')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label}: invalid SHA256')


def _path(base, value):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def _file(base, path, expected, label):
    _digest(expected, label)
    p = _path(base, path)
    if not p.is_file() or sha256(p) != expected:
        raise ValueError(f'{label}: missing file or SHA256 mismatch: {p}')
    return p


def _files(base, files, label):
    if not isinstance(files, dict) or not files:
        raise ValueError(f'{label}: nonempty artifact bindings required')
    for name, item in files.items():
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError(f'{label}/{name}: expected path and sha256')
        _file(base, item['path'], item['sha256'], f'{label}/{name}')


def _sealed(obj):
    if obj.get('test_evaluated') is not False:
        raise ValueError('test must remain sealed: test_evaluated=false required')
    if obj.get('split', 'valid') != 'valid':
        raise ValueError('Only clean valid evaluation is admitted')


def _metrics(values, label, keys=None):
    for name in keys or METRICS:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{label}/{name}: finite fraction required')


def load_binding(root):
    root = Path(root)
    binding = read_json(root / 'DEPLOYMENT_BINDING.json')
    _sealed(binding)
    if binding.get('split') != 'valid':
        raise ValueError('Binding requires an explicit valid split')
    if binding.get('schema_version') != SCHEMA or binding.get('status') != 'FROZEN':
        raise ValueError('Deployment binding must be FROZEN real-int8-deployment-v1')
    _files(root, binding.get('source_files'), 'source_files')
    rows = binding.get('runs', [])
    expected = {(n, s) for n in NETWORKS for s in SEEDS}
    if len(rows) != 9 or {(r.get('network'), r.get('seed')) for r in rows} != expected:
        raise ValueError('Binding requires exactly nine unique B1/C/D seed42/43/44 runs')
    for row in rows:
        if row.get('architecture_id') != ARCHITECTURES[row['network']]:
            raise ValueError('Only the frozen B1/C/D architectures are admitted')
        for key in IDENTITY:
            if key.endswith(('_sha256', '_hash')):
                _digest(row.get(key), key)
            elif not row.get(key):
                raise ValueError(f'Missing identity: {key}')
        for key in ('checkpoint', 'profile'):
            _file(root, row[f'{key}_path'], row[f'{key}_sha256'], key)
        _metrics(row.get('historical_reference_metrics', {}), 'historical_reference_metrics',
                 ('AP50', 'mAP50_95', 'Precision', 'Recall'))
    for key in ('calibration_sha256', 'numeric_contract_hash', 'split_manifest_sha256', 'evaluator_sha256'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f'Unmatched deployment protocol: {key}')
    for network in NETWORKS:
        if len({r['architecture_id'] for r in rows if r['network'] == network}) != 1:
            raise ValueError(f'{network}: architecture changes across seeds')
    return binding


def record_result_metrics(root, result):
    """Persist both measured evaluations before the caller marks a run COMPLETE."""
    root = Path(root)
    _sealed(result)
    events = []
    for prefix in ('historical', 'integer'):
        values = result[f'{prefix}_metrics']
        _metrics(values, prefix)
        provenance = dict(source=str(root / result['network'] / f"seed_{result['seed']}" / 'result.json'),
                          checkpoint_sha256=result['checkpoint_sha256'],
                          profile_sha256=result['profile_sha256'],
                          evaluation_protocol_hash=result['evaluator_sha256'],
                          split_manifest_sha256=result['split_manifest_sha256'],
                          calibration_manifest_sha256=result['calibration_sha256'],
                          numeric_contract=result['numeric_contract_hash'], split='valid')
        record = build_metric_record(task_type='detection', dataset='Cylinder2',
            network=result['network'], seed=result['seed'], stage=f'real_int8_{prefix}_valid',
            identity={**{key: result[key] for key in IDENTITY},
                      'run_id': f"real_int8_{result['network']}_seed{result['seed']}_{prefix}",
                      'precision': 'W8A8_INTEGER' if prefix == 'integer' else 'W8A8_HISTORICAL_SIMULATION',
                      'package_manifest_sha256': result['package_manifest_sha256'],
                      'test_evaluated': False, 'clean_reference_available': False},
            measured={target: dict(value=values[source], provenance=provenance)
                      for source, target in METRICS.items()})
        require_measured_metrics(record, METRICS.values())
        append_metric_record(root / 'METRICS.jsonl', record)
        events.append(record['event_id'])
    if result.get('complexity'):
        record_complexity_metrics(root,result)
    return events


def record_complexity_metrics(root, result):
    """Append analytical complexity without altering immutable accuracy records."""
    values=result['complexity'];root=Path(root)
    if not all(isinstance(values.get(k),int) and values[k]>0 for k in ('params','macs','flops')):
        raise ValueError('Complexity counts must be positive integers')
    if values['flops']!=2*values['macs']:raise ValueError('FLOPs estimate must equal 2*MACs')
    identity={**{k:result[k] for k in IDENTITY},
        'run_id':f"real_int8_{result['network']}_seed{result['seed']}_complexity",
        'package_manifest_sha256':result['package_manifest_sha256'],'test_evaluated':False}
    provenance=dict(source=str(root/result['network']/f"seed_{result['seed']}"/'result.json'),
        package_manifest_sha256=result['package_manifest_sha256'],checkpoint_sha256=result['checkpoint_sha256'],
        scope=values['scope'],hardware_latency='NOT_INFERRED')
    event=build_metric_record(task_type='detection',dataset='Cylinder2',network=result['network'],seed=result['seed'],
        stage='real_int8_complexity',identity=identity,
        measured={k:dict(value=values[k],provenance=provenance) for k in ('params','macs','flops')})
    require_measured_metrics(event,['params','macs','flops'])
    append_metric_record(root/'METRICS.jsonl',event)
    atomic_json(root/result['network']/f"seed_{result['seed']}"/'complexity_record.json',
        dict(event_id=event['event_id'],package_manifest_sha256=result['package_manifest_sha256'],test_evaluated=False))
    return event['event_id']


def _check_recorded_metrics(root, result):
    ids = result.get('metric_event_ids')
    if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('Two durable metric_event_ids required before completion')
    ledger = {r['event_id']: r for r in read_metric_records(Path(root) / 'METRICS.jsonl')}
    for prefix, event_id in zip(('historical', 'integer'), ids):
        if event_id not in ledger:
            raise ValueError('Run cannot complete before required metric events are durable')
        record = ledger[event_id]
        if record['network'] != result['network'] or record['seed'] != result['seed'] or record['stage'] != f'real_int8_{prefix}_valid':
            raise ValueError('Metric event belongs to a different run/stage')
        for key in IDENTITY:
            if record['identity'].get(key) != result[key]:
                raise ValueError(f'Metric identity mismatch: {key}')
        if record['identity'].get('package_manifest_sha256') != result['package_manifest_sha256']:
            raise ValueError('Metric package binding mismatch')
        require_measured_metrics(record, METRICS.values())
        for source, target in METRICS.items():
            if record['metrics'][target]['value'] != result[f'{prefix}_metrics'][source]:
                raise ValueError(f'Metric ledger/result mismatch: {source}')
    if result.get('complexity'):
        sidecar=read_json(Path(root)/result['network']/f"seed_{result['seed']}"/'complexity_record.json')
        event=ledger.get(sidecar.get('event_id'))
        if not event or event['stage']!='real_int8_complexity':raise ValueError('Required complexity metrics not durable')
        if event['identity'].get('package_manifest_sha256')!=result['package_manifest_sha256']:
            raise ValueError('Complexity package binding differs')
        for k in IDENTITY:
            if event['identity'].get(k)!=result[k]:raise ValueError('Complexity identity differs: '+k)
        require_measured_metrics(event,['params','macs','flops'])
        if any(event['metrics'][k]['value']!=result['complexity'][k] for k in ('params','macs','flops')):
            raise ValueError('Complexity record differs from graph counts')


def load_result(root, expected):
    root = Path(root)
    path = root / expected['network'] / f"seed_{expected['seed']}" / 'result.json'
    result = read_json(path)
    _sealed(result)
    if result.get('split') != 'valid':
        raise ValueError('Result requires an explicit valid split')
    if result.get('status') != 'COMPLETE' or result.get('historical_reproduction') != 'PASS':
        raise ValueError('Result incomplete or historical PTQ reproduction failed')
    if result.get('binding_sha256') != sha256(root / 'DEPLOYMENT_BINDING.json'):
        raise ValueError('Result deployment binding has changed')
    for key in IDENTITY:
        if result.get(key) != expected[key]:
            raise ValueError(f'Result identity mismatch: {key}')
    for prefix in ('historical', 'integer'):
        _metrics(result.get(f'{prefix}_metrics', {}), prefix)
    for key, ref in expected['historical_reference_metrics'].items():
        tolerance = 1e-6 if key in ('AP50', 'AP75', 'mAP50_95') else 0
        if abs(result['historical_metrics'][key] - ref) > tolerance:
            raise ValueError(f'Historical PTQ reproduction mismatch: {key}')
    package_path = _file(root, result['package_manifest_path'], result['package_manifest_sha256'], 'package_manifest')
    package = read_json(package_path)
    _sealed(package)
    if package.get('deterministic_export') != 'PASS':
        raise ValueError('Repeat export determinism not verified')
    for key in ('network', 'seed', 'architecture_id', 'checkpoint_sha256', 'profile_sha256', 'calibration_sha256', 'numeric_contract_hash'):
        if package.get(key) != expected[key]:
            raise ValueError(f'Package identity mismatch: {key}')
    files = package.get('files', {})
    if not {'graph.json', 'weights.bin', 'biases.bin', 'layers.bin'} <= set(files):
        raise ValueError('Package must bind graph.json, weights.bin, biases.bin, layers.bin')
    for name, digest in files.items():
        _file(package_path.parent, name, digest, f'package/{name}')
    _check_recorded_metrics(root, result)
    return result


def accuracy_admission(root, binding=None):
    binding = binding or load_binding(root)
    results, errors = [], []
    for expected in binding['runs']:
        try:
            results.append(load_result(root, expected))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{expected['network']}/seed_{expected['seed']}: {exc}")
    if errors:
        return dict(status='BLOCKED_EVIDENCE', allowed=False, reasons=errors,
                    complete_runs=len(results), required_runs=9, test_evaluated=False, gates={})
    rows = [{**{key: r[key] for key in IDENTITY}, **r['integer_metrics'],
             'precision': 'W8A8_INTEGER', 'checkpoint_policy': 'ALL_BEST_PT',
             'training_budget': 80, 'split': 'valid', 'test_evaluated': False} for r in results]
    ref = [r for r in rows if r['network'] == 'D']
    gates = {n: relative_accuracy_gate([r for r in rows if r['network'] == n], ref,
             precision='W8A8_INTEGER') for n in NETWORKS}
    return dict(status='INTEGER_ACCURACY_REVIEW_COMPLETE', allowed=True,
                complete_runs=9, required_runs=9, gates=gates, test_evaluated=False)


def record_stage_metrics(root, evidence):
    """Durable hardware-stage event; measured metrics require their own provenance.

    The caller supplies metric_values as the normal ledger measured mapping.
    Parity-only stages legitimately leave performance fields PENDING.  Route and
    board completion enforce their mandatory measurements in stage_evidence.
    """
    stage = evidence['stage']
    if stage not in STAGES:
        raise ValueError('Unknown hardware stage')
    record = build_metric_record(task_type='detection', dataset='Cylinder2',
        network=evidence['network'], seed=evidence['seed'], stage=f'real_int8_{stage}',
        identity={**{key: evidence[key] for key in IDENTITY},
                  'run_id': f"real_int8_{evidence['network']}_seed{evidence['seed']}_{stage}",
                  'package_manifest_sha256': evidence['package_manifest_sha256'],
                  'deployment_binding_sha256': evidence['deployment_binding_sha256'],
                  'artifacts': evidence['artifacts'], 'test_evaluated': False,
                  'clean_reference_available': False},
        measured=evidence.get('metric_values', {}))
    append_metric_record(Path(root) / 'METRICS.jsonl', record)
    return record['event_id']


def _check_stage_metrics(root, evidence, stage):
    events = {r['event_id']: r for r in read_metric_records(Path(root) / 'METRICS.jsonl')}
    event = events.get(evidence.get('metric_event_id'))
    if event is None or event['stage'] != f'real_int8_{stage}':
        raise ValueError(f'{stage}: durable stage metrics missing')
    if event['network'] != evidence['network'] or event['seed'] != evidence['seed']:
        raise ValueError(f'{stage}: metric record belongs to another run')
    for key in (*IDENTITY, 'package_manifest_sha256', 'deployment_binding_sha256', 'artifacts'):
        if event['identity'].get(key) != evidence.get(key):
            raise ValueError(f'{stage}: ledger identity mismatch: {key}')
    required = {
        'route': ['wns_ns', 'tns_ns', 'timing_met'] +
                 [f'{r}_{q}' for r in ('lut', 'ff', 'dsp', 'bram') for q in ('used', 'utilization_pct')],
        'board': ['board_latency_ms', 'throughput_fps', 'actual_clock_mhz', *METRICS.values()],
    }.get(stage, [])
    require_measured_metrics(event, required)
    for name, measured in evidence.get('metric_values', {}).items():
        if event['metrics'][name]['value'] != measured['value']:
            raise ValueError(f'{stage}: recorded value differs: {name}')
    for name in ('wns_ns', 'tns_ns', 'timing_met') if stage == 'route' else ():
        if event['metrics'][name]['value'] != evidence[name]:
            raise ValueError(f'{stage}: timing metric differs: {name}')
    if stage == 'board':
        for source, target in METRICS.items():
            if event['metrics'][target]['value'] != evidence['valid_metrics'][source]:
                raise ValueError(f'Board evaluation ledger mismatch: {source}')
        latency = evidence['timing']['end_to_end_median_ms']
        if not math.isclose(event['metrics']['board_latency_ms']['value'], latency, rel_tol=1e-12):
            raise ValueError('Board latency must be measured end-to-end median')
        if not math.isclose(event['metrics']['throughput_fps']['value'], 1000 / latency, rel_tol=1e-12):
            raise ValueError('Board throughput does not match batch-one end-to-end latency')


def _timing_samples(evidence):
    timing = evidence.get('timing', {})
    for scope in ('graph', 'end_to_end'):
        values = timing.get(f'{scope}_seconds')
        if not isinstance(values, list) or len(values) != 10:
            raise ValueError(f'Board {scope} timing requires ten measured samples')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Board timing samples must be positive finite seconds')
        ordered = sorted(values)
        expected = {'median': statistics.median(values) * 1000,
                    'p95': (ordered[8] * .45 + ordered[9] * .55) * 1000}
        for name, value in expected.items():
            actual = timing.get(f'{scope}_{name}_ms')
            if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-12):
                raise ValueError(f'{scope}/{name} does not match raw timing samples')
    if any(end < graph for end, graph in zip(timing['end_to_end_seconds'], timing['graph_seconds'])):
        raise ValueError('End-to-end measurement must include graph execution')


def stage_evidence(root, network, stage, result):
    if stage not in STAGES:
        raise ValueError(f'Unsupported evidence stage: {stage}')
    root = Path(root)
    path = root / network / 'seed_42' / f'{stage}.json'
    evidence = read_json(path)
    _sealed(evidence)
    if evidence.get('status') != 'PASS' or evidence.get('scope') != 'full_detector' or evidence.get('stage') != stage:
        raise ValueError(f'{stage}: full trained-detector PASS required')
    for key in IDENTITY:
        if evidence.get(key) != result[key]:
            raise ValueError(f'{stage}: identity mismatch: {key}')
    if evidence.get('package_manifest_sha256') != result['package_manifest_sha256']:
        raise ValueError(f'{stage}: package binding mismatch')
    if evidence.get('deployment_binding_sha256') != sha256(root / 'DEPLOYMENT_BINDING.json'):
        raise ValueError(f'{stage}: deployment binding changed')
    _files(path.parent, evidence.get('artifacts'), f'{stage}/artifacts')
    index = STAGES.index(stage)
    if index:
        previous = STAGES[index - 1]
        stage_evidence(root, network, previous, result)
        expected = sha256(path.with_name(previous + '.json'))
        if evidence.get('dependencies', {}).get(previous) != expected:
            raise ValueError(f'{stage}: predecessor evidence is unbound')
    if stage in ('integer_parity', 'hls', 'board') and (type(evidence.get('mismatch_count')) is not int or evidence['mismatch_count'] != 0):
        raise ValueError(f'{stage}: exact integer parity requires zero mismatches')
    if stage == 'hls' and (evidence.get('csim') != 'PASS' or evidence.get('cosim') != 'PASS'):
        raise ValueError('C and RTL simulation must both pass')
    if stage == 'route':
        if evidence.get('timing_met') is not True:
            raise ValueError('Route timing did not pass')
        for name in ('wns_ns', 'tns_ns', 'actual_clock_mhz'):
            value = evidence.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid routed timing: {name}')
        if evidence['actual_clock_mhz'] == 0:
            raise ValueError('Clock must be positive')
    if stage == 'board':
        if evidence.get('train_sample_count') != 8 or evidence.get('valid_sample_count') != 99:
            raise ValueError('Board parity requires fixed train8 and complete valid99')
        if evidence.get('warmup') != 1 or evidence.get('repetitions') != 10 or evidence.get('batch_size') != 1:
            raise ValueError('Board timing protocol mismatch')
        _timing_samples(evidence)
        _metrics(evidence.get('valid_metrics', {}), 'board valid')
        if any(evidence['valid_metrics'][key] != result['integer_metrics'][key] for key in METRICS):
            raise ValueError('Board valid metrics differ from integer software evaluation')
    _check_stage_metrics(root, evidence, stage)
    return evidence


def action_admission(root, network, action, seed=42):
    """Read-only action eligibility. Missing artifacts never become a PASS."""
    base = dict(network=network, seed=seed, action=action, allowed=False, test_evaluated=False)
    if network not in NETWORKS or seed not in SEEDS:
        return dict(base, status='BLOCKED_IDENTITY', reasons=['Only fixed B1/C/D and seeds42/43/44'])
    if action in ('search', 'promotion', 'ptq'):
        return dict(base, status='STOP_SCOPE', reasons=['No search, retraining, or recalibration in this deployment'])
    if action not in ('parameter_export', *STAGES):
        return dict(base, status='BLOCKED_ACTION', reasons=['Unregistered action'])
    try:
        binding = load_binding(root)
        expected = next(r for r in binding['runs'] if (r['network'], r['seed']) == (network, seed))
        if action == 'parameter_export':
            path = Path(root) / network / f'seed_{seed}' / 'result.json'
            if path.exists():
                load_result(root, expected)
                return dict(base, status='ALREADY_COMPLETE', reasons=['Verified export/evaluation is immutable'])
            return dict(base, allowed=True, status='READY', reasons=[])
        if seed != 42:
            return dict(base, status='STOP_SCOPE', reasons=['Board qualification fixed to seed42'])
        accuracy = accuracy_admission(root, binding)
        if not accuracy['allowed']:
            return dict(base, status=accuracy['status'], reasons=accuracy['reasons'])
        if not accuracy['gates'][network]['passed']:
            return dict(base, status=accuracy['gates'][network]['status'], reasons=['Integer mean mAP and Recall must both meet D'])
        result = load_result(root, expected)
        for stage in STAGES[:STAGES.index(action)]:
            stage_evidence(root, network, stage, result)
        if action == 'board':
            for prior in NETWORKS[:NETWORKS.index(network)]:
                if accuracy['gates'][prior]['passed']:
                    prior_expected = next(r for r in binding['runs'] if (r['network'], r['seed']) == (prior, 42))
                    stage_evidence(root, prior, 'board', load_result(root, prior_expected))
        return dict(base, allowed=True, status='READY', reasons=[])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(base, status='BLOCKED_EVIDENCE', reasons=[f'{type(exc).__name__}: {exc}'])


def audit(root):
    root = Path(root)
    try:
        binding = load_binding(root)
        quality = accuracy_admission(root, binding)
        actions = [action_admission(root, n, action) for n in NETWORKS
                   for action in ('parameter_export', *STAGES)]
        return dict(schema_version=SCHEMA, deployment_binding_sha256=sha256(root/'DEPLOYMENT_BINDING.json'),
                    status=quality['status'], accuracy=quality, actions=actions, test_evaluated=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(status='BLOCKED_EVIDENCE', reasons=[f'{type(exc).__name__}: {exc}'],
                    actions=[], test_evaluated=False)
