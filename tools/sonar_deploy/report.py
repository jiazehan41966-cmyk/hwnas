"""Summarize verified real INT8 evidence without images, calibration or test access."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from tools.sonar_deploy import admission as a
from hwnas_fpga.metrics_ledger import METRIC_SCHEMA, read_metric_records


def _csv(path, rows, columns):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _proof_value(proof, name, default='PENDING'):
    return proof.get(name, proof.get('metric_values', {}).get(name, {}).get('value', default))


def _power_summary(board, folder):
    """Optional measured power must include admitted ledger values and raw CSV."""
    fields = ('power_scope', 'idle_power_mean_w', 'idle_power_min_w', 'idle_power_max_w',
              'active_power_mean_w', 'active_power_min_w', 'active_power_max_w',
              'energy_per_inference_mj', 'dynamic_energy_per_inference_mj', 'power_samples_sha256')
    values = dict(power_status='NOT_MEASURED', power_reason='No admitted board sensor capture',
                  **{key: 'NOT_MEASURED' for key in fields})
    evidence = board.get('power_evidence', {})
    if board.get('power_status') != 'MEASURED' or evidence.get('status') != 'MEASURED':
        values['power_reason'] = evidence.get('reason', values['power_reason'])
        return values
    try:
        summary = evidence['summary']
        for key in ('power_w', 'energy_per_inference_mj'):
            if board['metric_values'][key]['value'] != summary[key]:
                raise ValueError('Optional power metric lacks matching admitted ledger value')
        candidates = [item for item in board['artifacts'].values()
                      if item['path'].replace('\\', '/').endswith('/power/power_samples.csv')]
        if len(candidates) != 1 or candidates[0]['sha256'] != evidence['samples_sha256']:
            raise ValueError('Unique bound raw power CSV unavailable')
        item = candidates[0]
        path = a._file(folder, item['path'], item['sha256'], 'admitted raw power samples')
        with path.open(encoding='utf-8', newline='') as stream:
            samples = [(float(row['monotonic_s']), float(row['power_w'])) for row in csv.DictReader(stream)]
        if not samples or any(not math.isfinite(t) or not math.isfinite(p) or p < 0 for t,p in samples):
            raise ValueError('Invalid raw power samples')
        idle_window = evidence['idle_window']
        idle = [p for t,p in samples if idle_window[0] <= t <= idle_window[1]]
        measured_calls = [call for call in summary['calls'] if call['warmup'] is False]
        if len(measured_calls) != 10:
            raise ValueError('Power summary requires ten measured inference windows')
        active = [p for t,p in samples if any(call['start_monotonic_s'] <= t <= call['end_monotonic_s'] for call in measured_calls)]
        if not idle or not active: raise ValueError('Raw idle/active window samples unavailable')
        values.update(power_status='MEASURED', power_reason='SHA-bound board sensor samples and ledger values',
            power_scope=evidence['scope'], idle_power_mean_w=summary['idle_power_w'],
            idle_power_min_w=min(idle), idle_power_max_w=max(idle), active_power_mean_w=summary['power_w'],
            active_power_min_w=min(active), active_power_max_w=max(active),
            energy_per_inference_mj=summary['energy_per_inference_mj'],
            dynamic_energy_per_inference_mj=summary['dynamic_energy_per_inference_mj'],
            power_samples_sha256=item['sha256'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        values['power_reason'] = 'INCOMPLETE_POWER_REPORT_EVIDENCE: ' + str(exc)
    return values


def _comparison_rows(root, verified, admitted, hardware, quality):
    """Render only already admitted proofs; unfinished stages never gain values."""
    rows = []
    by_key = {(r['network'], r['seed']): r for r in verified}
    statuses = {(r['network'], r['stage']): r['status'] for r in hardware}
    for network in a.NETWORKS:
        results = [by_key[(network, s)] for s in a.SEEDS if (network, s) in by_key]
        full = len(results) == 3
        seed42 = by_key.get((network, 42), {})
        route = admitted.get((network, 'route'), {}); board = admitted.get((network, 'board'), {})
        row = dict(network=network, architecture_id=a.ARCHITECTURES[network],
            software_status='VERIFIED_3_SEEDS' if full else 'PENDING', verified_seed_count=len(results),
            accuracy_gate=quality.get('gates', {}).get(network, {}).get('status', 'PENDING'),
            hardware_seed=42, route_status=statuses.get((network, 'route'), 'PENDING'),
            board_status=statuses.get((network, 'board'), 'PENDING'), test_evaluated=False)
        for metric in a.METRICS:
            row['integer_3seed_mean_' + metric] = statistics.mean(r['integer_metrics'][metric] for r in results) if full else 'PENDING'
            row['integer_3seed_sample_std_' + metric] = statistics.stdev(r['integer_metrics'][metric] for r in results) if full else 'PENDING'
            row['seed42_software_' + metric] = seed42.get('integer_metrics', {}).get(metric, 'PENDING')
            row['seed42_board_' + metric] = board.get('valid_metrics', {}).get(metric, 'PENDING')
        row['seed42_exact_metric_parity'] = ('PASS' if all(board['valid_metrics'][m] == seed42['integer_metrics'][m] for m in a.METRICS) else 'FAIL') if board else 'PENDING'
        row.update({key: seed42.get('complexity', {}).get(key, 'PENDING') for key in ('params','macs','flops')})
        for resource in ('lut','ff','dsp','bram'):
            for quantity in ('used','capacity','utilization_pct'):
                row[resource+'_'+quantity] = _proof_value(route, resource+'_'+quantity)
        for key in ('wns_ns','tns_ns','timing_met'):
            row[key] = _proof_value(route, key)
        row['route_clock_mhz'] = _proof_value(route, 'actual_clock_mhz')
        row['board_clock_mhz'] = _proof_value(board, 'actual_clock_mhz')
        for scope in ('graph','end_to_end'):
            for quantile in ('median','p95'):
                milliseconds = board.get('timing', {}).get(f'{scope}_{quantile}_ms', 'PENDING')
                row[f'{scope}_{quantile}_ms'] = milliseconds
                row[f'{scope}_fps_from_{quantile}_latency'] = 1000/milliseconds if isinstance(milliseconds, (int,float)) and milliseconds > 0 else 'PENDING'
        row.update(_power_summary(board, Path(root)/network/'seed_42'))
        row['checkpoint_sha256_seed42'] = seed42.get('checkpoint_sha256', 'PENDING')
        row['package_manifest_sha256_seed42'] = seed42.get('package_manifest_sha256', 'PENDING')
        for stage in ('route','board'):
            row[stage+'_evidence_sha256'] = a.sha256(Path(root)/network/'seed_42'/f'{stage}.json') if (network,stage) in admitted else 'PENDING'
        rows.append(row)
    return rows


def _candidate_markdown(root, row):
    n = row['network']
    lines = [f'# {n} real INT8 board evidence', '', f"Active evidence root: `{root}`.",
        f"Architecture: `{row['architecture_id']}`.",
        f"Software: `{row['software_status']}`; route: `{row['route_status']}`; board: `{row['board_status']}`.",
        f"Same-integer D accuracy admission: `{row['accuracy_gate']}` (relative floor only; application adequacy not established).", '',
        'Software means require all three seeds42/43/44. Hardware uses fixed seed42; no best-seed selection.', '',
        '| Metric | Integer three-seed mean | Sample std | Seed42 software | Seed42 board |',
        '| --- | --- | --- | --- | --- |']
    for metric in a.METRICS:
        lines.append(f"| {metric} | {row['integer_3seed_mean_'+metric]} | {row['integer_3seed_sample_std_'+metric]} | {row['seed42_software_'+metric]} | {row['seed42_board_'+metric]} |")
    lines += ['', f"Five-metric exact parity: `{row['seed42_exact_metric_parity']}`.", '',
        f"Params / MACs / FLOPs: `{row['params']}` / `{row['macs']}` / `{row['flops']}`. MACs are not latency.", '',
        '| Routed resource | Used | Capacity | Utilization (%) |', '| --- | --- | --- | --- |']
    for key in ('lut','ff','dsp','bram'):
        lines.append(f"| {key.upper()} | {row[key+'_used']} | {row[key+'_capacity']} | {row[key+'_utilization_pct']} |")
    lines += ['', f"WNS / TNS (ns): `{row['wns_ns']}` / `{row['tns_ns']}`; Timing Met: `{row['timing_met']}`.",
        f"Routed / actual board clock (MHz): `{row['route_clock_mhz']}` / `{row['board_clock_mhz']}`.", '',
        '| Measured scope | Median ms/image | P95 ms/image | FPS from median latency | FPS from P95 latency |',
        '| --- | --- | --- | --- | --- |']
    for scope in ('graph','end_to_end'):
        lines.append(f"| {scope} | {row[scope+'_median_ms']} | {row[scope+'_p95_ms']} | {row[scope+'_fps_from_median_latency']} | {row[scope+'_fps_from_p95_latency']} |")
    lines += ['', 'Batch1/input640, resident weights, warmup1 and10 measured calls. FPS columns are reciprocal latency diagnostics, not empirical FPS percentiles.',
        'End-to-end includes host preprocessing, host/board transfer, graph execution, raw-head return and host CPU decode/NMS; it is not an ARM-only deployment time.', '',
        f"Power: `{row['power_status']}`; scope: `{row['power_scope']}`. {row['power_reason']}", '',
        '| Power window | Time-weighted mean W | Observed raw-sample min W | Observed raw-sample max W |',
        '| --- | --- | --- | --- |',
        f"| idle | {row['idle_power_mean_w']} | {row['idle_power_min_w']} | {row['idle_power_max_w']} |",
        f"| active (10 measured calls) | {row['active_power_mean_w']} | {row['active_power_min_w']} | {row['active_power_max_w']} |", '',
        f"Total / idle-subtracted diagnostic energy (mJ/image): `{row['energy_per_inference_mj']}` / `{row['dynamic_energy_per_inference_mj']}`.",
        'Energy integrates only the board inference windows, excludes warmup and does not include host/network/decode energy. Negative dynamic values remain visible; ten calls do not establish multi-session repeatability.', '',
        f"Raw power CSV SHA256: `{row['power_samples_sha256']}`.",
        f"Checkpoint SHA256: `{row['checkpoint_sha256_seed42']}`.",
        f"Parameter package manifest SHA256: `{row['package_manifest_sha256_seed42']}`.",
        f"Admitted route / board evidence SHA256: `{row['route_evidence_sha256']}` / `{row['board_evidence_sha256']}`.",
        '`test_evaluated=false`. Missing values are PENDING or NOT_MEASURED; no pending stage is a measurement.', '']
    return '\n'.join(lines)


def _generate_report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = a.load_binding(root)
    verified, problems, accuracy_rows, reproduction_rows, deltas, packages = [], [], [], [], [], []
    inputs = {'DEPLOYMENT_BINDING.json': a.sha256(root/'DEPLOYMENT_BINDING.json')}
    attempt = root/'HARDWARE_ATTEMPT_BINDING.json'
    if attempt.is_file():
        inputs[attempt.name] = a.sha256(attempt)
    for expected in sorted(binding['runs'], key=lambda r: (a.NETWORKS.index(r['network']), r['seed'])):
        network, seed = expected['network'], expected['seed']
        coordinates = dict(network=network, seed=seed, architecture_id=expected['architecture_id'])
        row = dict(**coordinates, status='PENDING', checkpoint_sha256=expected['checkpoint_sha256'],
                   profile_sha256=expected['profile_sha256'], package_manifest_sha256='', test_evaluated=False)
        for prefix in ('historical', 'integer'):
            row.update({f'{prefix}_{metric}': '' for metric in a.METRICS})
        path = root/network/f'seed_{seed}'/'result.json'
        if path.is_file():
            inputs[str(path.relative_to(root))] = a.sha256(path)
        try:
            result = a.load_result(root, expected)
            package_manifest = Path(result['package_manifest_path'])
            if not package_manifest.is_absolute():
                package_manifest = root/package_manifest
            metadata = a.read_json(package_manifest)
            for name, digest in metadata['files'].items():
                file = package_manifest.parent/name
                packages.append(dict(**coordinates, file=name, path=str(file), sha256=digest,
                    bytes=file.stat().st_size, package_manifest_sha256=result['package_manifest_sha256'],
                    deterministic_export=metadata['deterministic_export'], test_evaluated=False))
            verified.append(result)
            row.update(status='VERIFIED', package_manifest_sha256=result['package_manifest_sha256'])
            for prefix in ('historical', 'integer'):
                row.update({f'{prefix}_{metric}': result[f'{prefix}_metrics'][metric] for metric in a.METRICS})
            for metric in a.METRICS:
                old = expected['historical_reference_metrics'].get(metric)
                current, integer = result['historical_metrics'][metric], result['integer_metrics'][metric]
                reproduction_rows.append(dict(**coordinates, metric=metric,
                    historical_reference=old if old is not None else '', reproduced=current,
                    delta=current-old if old is not None else '',
                    tolerance=(1e-6 if metric in ('AP50', 'AP75', 'mAP50_95') else 0) if old is not None else '',
                    status='PASS' if old is not None else 'NEW_MEASUREMENT_NO_HISTORICAL_REFERENCE'))
                deltas.append(dict(**coordinates, metric=metric, historical_ptq=current,
                                   pure_integer=integer, delta_integer_minus_historical=integer-current,
                                   test_evaluated=False))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['status'] = 'BLOCKED_EVIDENCE' if path.exists() else 'PENDING'
            problems.append(dict(**coordinates, status=row['status'], reason=str(exc)))
        accuracy_rows.append(row)
    _csv(output/'INTEGER_ACCURACY_RESULTS.csv', accuracy_rows, list(accuracy_rows[0]))
    _csv(output/'PARAMETER_PACKAGES.csv', packages,
         ['network', 'seed', 'architecture_id', 'file', 'path', 'sha256', 'bytes',
          'package_manifest_sha256', 'deterministic_export', 'test_evaluated'])
    _csv(output/'HISTORICAL_PTQ_REPRODUCTION.csv', reproduction_rows,
         ['network', 'seed', 'architecture_id', 'metric', 'historical_reference', 'reproduced', 'delta', 'tolerance', 'status'])
    _csv(output/'INTEGER_VS_HISTORICAL.csv', deltas,
         ['network', 'seed', 'architecture_id', 'metric', 'historical_ptq', 'pure_integer', 'delta_integer_minus_historical', 'test_evaluated'])
    quality = a.accuracy_admission(root, binding)
    a.atomic_json(output/'INTEGER_ACCURACY_GATE.json', quality)
    ledger = root/'METRICS.jsonl'
    records = read_metric_records(ledger) if ledger.is_file() else []
    if ledger.is_file():
        inputs['METRICS.jsonl'] = a.sha256(ledger)
    coverage = [dict(metric=name, **{status: sum(r['metrics'][name]['status'] == status for r in records)
                 for status in ('MEASURED', 'PENDING', 'NOT_APPLICABLE')}) for name in METRIC_SCHEMA]
    _csv(output/'METRIC_COVERAGE.csv', coverage, ['metric', 'MEASURED', 'PENDING', 'NOT_APPLICABLE'])
    hardware, admitted = [], {}
    results_by_key = {(r['network'], r['seed']): r for r in verified}
    for network in a.NETWORKS:
        gate = quality.get('gates', {}).get(network)
        for stage in a.STAGES:
            proof = root/network/'seed_42'/f'{stage}.json'
            row = dict(network=network, stage=stage, status='PENDING', reason='Verified software evaluation and stage evidence are required')
            if gate and not gate['passed']:
                row.update(status='STOP_ACCURACY_BELOW_D', reason='Integer mean mAP50:95 and Recall must both meet same-implementation D')
            elif quality.get('allowed') and (network, 42) in results_by_key and proof.exists():
                try:
                    admitted[(network, stage)] = a.stage_evidence(root, network, stage, results_by_key[(network, 42)])
                    row.update(status='PASS', reason='SHA-bound trained-detector evidence and required ledger metrics verified')
                    inputs[str(proof.relative_to(root))] = a.sha256(proof)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status='BLOCKED_EVIDENCE', reason=str(exc))
            hardware.append(row)
    a.atomic_json(output/'HARDWARE_STAGE_STATUS.json', hardware)
    comparison = _comparison_rows(root, verified, admitted, hardware, quality)
    _csv(output/'FINAL_DEPLOYMENT_COMPARISON.csv', comparison, list(comparison[0]))
    for row in comparison:
        (output/f"{row['network']}_BOARD_REPORT.md").write_text(_candidate_markdown(root, row), encoding='utf-8')
    eligible = {n for n, g in quality.get('gates', {}).items() if g['passed']}
    board_complete = {r['network'] for r in hardware if r['stage'] == 'board' and r['status'] == 'PASS'}
    status = ('PARTIAL_EVIDENCE' if len(verified) != 9 else
              'FULL_DEPLOYMENT_PASS' if eligible <= board_complete else
              'SOFTWARE_EVALUATION_COMPLETE_HARDWARE_PENDING')
    progress = dict(status=status, verified_runs=len(verified), required_runs=9,
                    metrics_events=len(records), metric_fields=len(METRIC_SCHEMA),
                    integer_accuracy_gates={n: g['status'] for n, g in quality.get('gates', {}).items()},
                    board_completed=sorted(board_complete), problems=problems,
                    final_comparison='FINAL_DEPLOYMENT_COMPARISON.csv',
                    candidate_board_reports=[f'{n}_BOARD_REPORT.md' for n in a.NETWORKS],
                    test_evaluated=False, application_accuracy_sufficient='NOT_ESTABLISHED_RELATIVE_ONLY')
    a.atomic_json(output/'DEPLOYMENT_PROGRESS.json', progress)
    text = ['# Real INT8 deployment evidence review', '', f"Status: `{status}`.",
            f"Active evidence root: `{root}`.",
            f"Verified software evaluations: {len(verified)}/9; durable ledger events: {len(records)}.",
            'Test remains sealed: `test_evaluated=false`.', '',
            '| Network | Seed | Historical PTQ mAP50:95 | Pure integer mAP50:95 | Pure integer Recall | Status |',
            '|---|---:|---:|---:|---:|---|']
    for row in accuracy_rows:
        text.append(f"| {row['network']} | {row['seed']} | {row['historical_mAP50_95']} | {row['integer_mAP50_95']} | {row['integer_Recall']} | {row['status']} |")
    text += ['', '## Same-implementation D admission', '',
             'All nine verified results are required. B1/C each require three-seed mean mAP50:95 and Recall to meet the pure-integer D control. This is not an absolute application-quality guarantee.']
    for network, gate in quality.get('gates', {}).items():
        text.append(f"- {network}: `{gate['status']}`.")
    text += ['', '## Hardware evidence', '',
             'Software export/evaluation does not prove C/RTL parity, routed timing, or board execution. A synthetic graph/operator result is not accepted as a trained detector.']
    for row in hardware:
        text.append(f"- {row['network']} {row['stage']}: `{row['status']}` — {row['reason']}")
    text += ['', '## Candidate comparison outputs', '',
        '- [Final comparison table](FINAL_DEPLOYMENT_COMPARISON.csv): three-seed integer accuracy, fixed seed42 software/board parity, routed resources/timing, measured latency and optional raw-bound power.',
        *[f'- [{n} board report]({n}_BOARD_REPORT.md).' for n in a.NETWORKS],
        '- FINAL in the filename denotes the delivery comparison schema; incomplete board rows remain explicitly PENDING.',
        '- FPS is derived from batch-one median/P95 latency; power energy integrates board call windows and is not host end-to-end energy.']
    text += ['', '## Scope', '',
             '- No new training, architecture search, recalibration, QAT, or test evaluation.',
             '- New measured AP75 has no historical reference when the old report did not preserve it.',
             '- Classification Top-1 is not applicable; unmeasured hardware or power fields remain PENDING.',
             '- Integer-versus-historical PTQ differences are reported independently; historical PTQ accuracy is not reused as integer accuracy.']
    (output/'REAL_INT8_DEPLOYMENT_REVIEW.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    a.atomic_json(output/'REPORT_MANIFEST.json', dict(input_sha256=inputs, test_evaluated=False,
        source_sha256={str(p.relative_to(ROOT)): a.sha256(p) for p in (Path(__file__), Path(a.__file__))},
        output_sha256={p.name: a.sha256(p) for p in output.iterdir() if p.is_file() and p.name in {
            'INTEGER_ACCURACY_RESULTS.csv', 'HISTORICAL_PTQ_REPRODUCTION.csv', 'INTEGER_VS_HISTORICAL.csv',
            'PARAMETER_PACKAGES.csv',
            'INTEGER_ACCURACY_GATE.json', 'METRIC_COVERAGE.csv', 'HARDWARE_STAGE_STATUS.json',
            'FINAL_DEPLOYMENT_COMPARISON.csv', 'D_BOARD_REPORT.md', 'C_BOARD_REPORT.md', 'B1_BOARD_REPORT.md',
            'DEPLOYMENT_PROGRESS.json', 'REAL_INT8_DEPLOYMENT_REVIEW.md'}}))
    return progress


def generate_report(root, output):
    """A failed refresh invalidates old success markers, never preserves a PASS."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    a.atomic_json(output/'DEPLOYMENT_PROGRESS.json', dict(status='IN_PROGRESS', test_evaluated=False))
    try:
        return _generate_report(root, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failed = dict(status='BLOCKED_EVIDENCE', verified_runs=0, required_runs=9,
                      reasons=[f'{type(exc).__name__}: {exc}'], test_evaluated=False)
        a.atomic_json(output/'DEPLOYMENT_PROGRESS.json', failed)
        a.atomic_json(output/'REPORT_MANIFEST.json', failed)
        (output/'REAL_INT8_DEPLOYMENT_REVIEW.md').write_text(
            '# Real INT8 deployment evidence review\n\nCurrent refresh blocked: ' + str(exc) +
            '\n\nEarlier CSV files are not an admitted current snapshot. `test_evaluated=false`.\n', encoding='utf-8')
        stale = _comparison_rows(root, [], {}, [], {})
        for row in stale:
            row.update(software_status='BLOCKED_EVIDENCE', route_status='BLOCKED_EVIDENCE', board_status='BLOCKED_EVIDENCE')
            (output/f"{row['network']}_BOARD_REPORT.md").write_text(
                _candidate_markdown(root, row) + '\nCurrent refresh blocked: ' + str(exc) + '\n', encoding='utf-8')
        _csv(output/'FINAL_DEPLOYMENT_COMPARISON.csv', stale, list(stale[0]))
        return failed


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/sonar_real_int8_deployment_v1/cpu_decode_v1_route02')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/sonar_real_int8_deployment_v1')
    args = parser.parse_args()
    result = generate_report(args.root, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result['verified_runs'] == 9 else 2


if __name__ == '__main__':
    raise SystemExit(main())
