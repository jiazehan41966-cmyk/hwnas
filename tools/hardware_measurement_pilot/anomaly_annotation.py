"""Preserve raw timing rows and add descriptive latency labels in new artifacts.

These annotations never change sample selection, official statistics, protocol
thresholds, or acceptance. MAD-zero distributions do not yield robust outliers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from tools.hardware_measurement_pilot.measurement import CSV_FIELDS, HASH_FIELDS


SCOPES = ('compute', 'transfer', 'call')
ANNOTATION_FIELDS = ['ann_source_row', 'ann_input_field_count', 'ann_extra_fields_json', 'ann_invalid', 'ann_flags_json'] + [
    f'ann_{scope}_{field}' for scope in SCOPES for field in
    ('status', 'high_latency', 'above_median', 'observation_id', 'observation_weight', 'median_ns', 'mad_ns', 'threshold_ns')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('finite positive number required')
    return result


def _integer(value, minimum=0):
    number = float(value)
    if not math.isfinite(number) or number < minimum or not number.is_integer():
        raise ValueError('bounded integer required')
    return int(number)


def annotate_rows(rows, *, input_fields=None, manifest=None):
    """Return a one-to-one annotated copy plus descriptive rules/counts.

    Finite positive PL timings remain in their descriptive baseline even if a
    separate identity/parity condition flags the row. Invalid numeric values or
    malformed host groups cannot enter a numeric MAD calculation; all their raw
    rows still appear in the output, explicitly flagged. Warmup is separate.
    """
    input_fields = list(CSV_FIELDS if input_fields is None else input_fields)
    if len(set(input_fields)) != len(input_fields) or any(name.startswith(('ann_', '__anomaly_')) for name in input_fields):
        raise ValueError('duplicate or reserved annotation input column')
    missing = set(CSV_FIELDS) - set(input_fields)
    if missing:
        raise ValueError('input schema lacks timing fields: ' + ','.join(sorted(missing)))
    expected = {item['case_id']: item for item in (manifest or {}).get('cases', [])}
    batch_expected = {(item['case_id'], item['batch_id']): item for item in (manifest or {}).get('batches', [])}
    row_flags, row_invalid, results = [], [], []
    strata = defaultdict(list)
    groups, duplicate_iterations = defaultdict(list), defaultdict(list)
    parsed = []
    for index, source in enumerate(rows):
        flags, item = [], {name: source.get(name, '') for name in input_fields}
        field_count = source.get('__anomaly_field_count', len(input_fields))
        extras = source.get('__anomaly_extra_fields', [])
        item.update(ann_source_row=index + 1, ann_input_field_count=field_count,
                    ann_extra_fields_json=json.dumps(extras, ensure_ascii=False), ann_invalid=False, ann_flags_json='[]')
        for scope in SCOPES:
            item.update({f'ann_{scope}_{name}': value for name, value in dict(status='NO_VALID_OBSERVATION',
                high_latency='', above_median='', observation_id='', observation_weight=0,
                median_ns='', mad_ns='', threshold_ns='').items()})
        if field_count != len(input_fields):
            flags.append('INVALID_RAW_FIELD_COUNT')
        case, batch, phase = source.get('case_id', ''), source.get('batch_id', ''), source.get('phase', '')
        key_valid = bool(case) and bool(batch) and phase in ('warmup', 'measured')
        if not key_valid:
            flags.append('INVALID_CASE_BATCH_PHASE_KEY')
        try:
            iteration = _integer(source.get('iteration'))
            duplicate_iterations[(case, batch, phase, iteration)].append(index)
        except (ValueError, TypeError, OverflowError):
            iteration = None
            flags.append('INVALID_ITERATION')
        values = {}
        for scope in SCOPES:
            try:
                values[scope] = _positive(source.get(scope + '_ns'))
            except (ValueError, TypeError, OverflowError):
                values[scope] = None
                flags.append('INVALID_' + scope.upper() + '_NS')
                item[f'ann_{scope}_status'] = 'INVALID_NUMERIC_VALUE'
        try:
            frequency = _positive(source.get('clock_hz'))
        except (ValueError, TypeError, OverflowError):
            frequency = None
            flags.append('INVALID_CLOCK_HZ')
        for field in (*HASH_FIELDS, 'observed_output_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', str(source.get(field, ''))):
                flags.append('INVALID_HASH_' + field.upper())
        if not source.get('board_id'):
            flags.append('INVALID_BOARD_ID')
        try:
            compared = _integer(source.get('compared_elements'), 1)
            mismatches = _integer(source.get('mismatched_elements'))
            if mismatches != 0 or source.get('observed_output_sha256') != source.get('expected_output_sha256'):
                flags.append('INVALID_OUTPUT_PARITY')
        except (ValueError, TypeError, OverflowError):
            compared = None
            flags.append('INVALID_OUTPUT_COMPARISON_COUNT')
        if manifest is not None:
            spec, batch_spec = expected.get(case), batch_expected.get((case, batch))
            if spec is None or batch_spec is None:
                flags.append('INVALID_MANIFEST_CASE_OR_BATCH')
            else:
                for field in (*HASH_FIELDS, 'board_id'):
                    if source.get(field) != spec.get(field):
                        flags.append('INVALID_MANIFEST_BINDING_' + field.upper())
                if compared != spec.get('expected_elements'):
                    flags.append('INVALID_MANIFEST_OUTPUT_ELEMENT_COUNT')
                if frequency != batch_spec.get('clock_hz'):
                    flags.append('INVALID_MANIFEST_CLOCK')
        group_id = str(source.get('group_id', ''))
        try:
            size = _integer(source.get('group_size'), 1)
            total = _positive(source.get('group_call_total_ns'))
        except (ValueError, TypeError, OverflowError):
            size, total = None, None
            flags.append('INVALID_HOST_GROUP_SIZE_OR_TOTAL')
        if not group_id:
            flags.append('INVALID_HOST_GROUP_ID')
        record = dict(index=index, iteration=iteration, values=values, group_size=size, group_total=total,
                      key_valid=key_valid, key=(case, batch, phase))
        parsed.append(record)
        row_flags.append(flags)
        results.append(item)
        if key_valid:
            for scope in ('compute', 'transfer'):
                if values[scope] is not None:
                    strata[(case, batch, phase, scope)].append(dict(value=values[scope], rows=[index],
                        observation_id='row:' + str(index + 1)))
            if group_id:
                groups[(case, batch, phase, group_id)].append(record)
        if not key_valid or not group_id:
            item['ann_call_status'] = 'INVALID_GROUP_KEY'
    for indexes in duplicate_iterations.values():
        if len(indexes) > 1:
            for index in indexes:
                row_flags[index].append('INVALID_DUPLICATE_PHASE_ITERATION')
    invalid_groups, group_size_counts = [], defaultdict(Counter)
    for key, members in groups.items():
        sizes = {item['group_size'] for item in members}
        totals = {item['group_total'] for item in members}
        reasons = []
        if None in sizes or None in totals or len(sizes) != 1 or len(totals) != 1:
            reasons.append('INVALID_HOST_GROUP_INCONSISTENT_SIZE_OR_TOTAL')
        else:
            size, total = next(iter(sizes)), next(iter(totals))
            if len(members) != size:
                reasons.append('INVALID_HOST_GROUP_MEMBER_COUNT')
            if any(item['iteration'] is None for item in members) or len({item['iteration'] for item in members}) != len(members):
                reasons.append('INVALID_HOST_GROUP_ITERATIONS')
            if any(item['values']['call'] is None or not math.isclose(item['values']['call'], total/size,
                    rel_tol=1e-12, abs_tol=1e-6) for item in members):
                reasons.append('INVALID_HOST_GROUP_CALL_MEAN')
            if (all(item['values']['compute'] is not None and item['values']['transfer'] is not None for item in members)
                    and total < sum(item['values']['compute'] + item['values']['transfer'] for item in members)):
                reasons.append('INVALID_HOST_GROUP_PL_CONTAINMENT')
        if reasons:
            for item in members:
                row_flags[item['index']].extend(reasons)
                results[item['index']]['ann_call_status'] = 'INVALID_HOST_GROUP'
            invalid_groups.append(dict(case_id=key[0], batch_id=key[1], phase=key[2], group_id=key[3],
                                       source_rows=[item['index']+1 for item in members], flags=reasons))
            continue
        strata[(*key[:3], 'call')].append(dict(value=total/size, rows=[item['index'] for item in members],
            observation_id='group:' + json.dumps(key, separators=(',', ':')), group_size=size))
        group_size_counts[key[:3]][size] += 1
    baseline_rows = []
    for key, observations in sorted(strata.items()):
        values = [item['value'] for item in observations]
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        threshold = median + 6 * 1.4826 * mad
        status = ('NONFINITE_BASELINE_NO_DECISION' if not all(math.isfinite(x) for x in (median, mad, threshold))
                  else 'MAD_ZERO_NO_ROBUST_DECISION' if mad == 0 else 'DESCRIPTIVE_THRESHOLD_AVAILABLE')
        if status != 'DESCRIPTIVE_THRESHOLD_AVAILABLE':
            threshold = None
        flags_count = above_count = flagged_member_rows = 0
        for observation in observations:
            high = observation['value'] > threshold if threshold is not None else None
            above = observation['value'] > median if math.isfinite(median) else None
            flags_count += int(high is True)
            above_count += int(above is True)
            flagged_member_rows += len(observation['rows']) if high else 0
            for position, index in enumerate(observation['rows']):
                item = results[index]
                scope = key[3]
                item.update({f'ann_{scope}_{name}': value for name, value in dict(status=status,
                    high_latency=high if high is not None else '', above_median=above if above is not None else '',
                    observation_id=observation['observation_id'], observation_weight=1 if position == 0 else 0,
                    median_ns=median if math.isfinite(median) else '', mad_ns=mad if math.isfinite(mad) else '',
                    threshold_ns=threshold if threshold is not None else '').items()})
                if high:
                    row_flags[index].append(scope.upper() + '_DESCRIPTIVE_HIGH_LATENCY')
                elif status == 'MAD_ZERO_NO_ROBUST_DECISION' and above:
                    row_flags[index].append(scope.upper() + '_ABOVE_MEDIAN_MAD_ZERO_NO_ROBUST_DECISION')
                elif status == 'NONFINITE_BASELINE_NO_DECISION':
                    row_flags[index].append(scope.upper() + '_NONFINITE_BASELINE_NO_DECISION')
        baseline_rows.append(dict(case_id=key[0], batch_id=key[1], phase=key[2], scope=key[3],
            observation_count=len(observations), mapped_row_count=sum(len(item['rows']) for item in observations),
            median_ns=median if math.isfinite(median) else None, mad_ns=mad if math.isfinite(mad) else None,
            threshold_ns=threshold, status=status, high_latency_observations=flags_count,
            high_latency_mapped_rows=flagged_member_rows, above_median_observations=above_count,
            host_group_size_distribution=dict(group_size_counts[key[:3]]) if key[3] == 'call' else None,
            mixed_host_group_sizes=len(group_size_counts[key[:3]]) > 1 if key[3] == 'call' else None))
    for item, flags in zip(results, row_flags):
        unique = sorted(set(flags))
        item['ann_flags_json'] = json.dumps(unique, ensure_ascii=False)
        item['ann_invalid'] = any(flag.startswith('INVALID_') for flag in unique)
        row_invalid.append(item['ann_invalid'])
    rules = dict(schema='pilot_descriptive_latency_annotations_v1', scope='DERIVED_ANNOTATION_ONLY_NO_SELECTION_OR_ACCEPTANCE_CHANGE',
        formula='high iff value > median + 6 * 1.4826 * MAD; MAD = median(abs(value - median))',
        grouping=['case_id', 'batch_id', 'phase', 'scope'], warmup_policy='Separate strata; never merged into measured',
        host_policy='One complete unique (case_id,batch_id,phase,group_id) total/N observation, then map its label to all member rows; ann_call_observation_weight is 1 only on the first member',
        mixed_host_group_sizes_policy='Retain every unique group equally; disclose N distribution and mixed-N descriptive comparability limitation',
        mad_zero_policy='No robust high-latency decision, no threshold; above-median values may receive a separate non-robust descriptive label',
        invalid_policy='Preserve every raw row, including malformed rows and invalid values/groups; flag invalid. Finite positive PL values remain descriptive observations even with separate identity/parity errors; invalid numeric values and invalid host groups have no numeric MAD observation. No official statistic is recomputed or replaced.',
        group_call_consistency=dict(relative_tolerance=1e-12, absolute_tolerance_ns=1e-6),
        statistical_or_acceptance_gate=False, samples_deleted=0, raw_row_order_preserved=True,
        input_row_count=len(rows), output_row_count=len(results), invalid_rows=sum(row_invalid),
        rows_with_any_flag=sum(bool(flags) for flags in row_flags),
        flag_row_counts=dict(sorted(Counter(flag for flags in row_flags for flag in set(flags)).items())),
        phase_row_counts=dict(sorted(Counter(str(row.get('phase', '')) for row in rows).items())),
        strata=baseline_rows, invalid_host_groups=invalid_groups)
    return results, rules


def annotate_csv(input_csv, output_csv, rules_json, *, manifest_path=None):
    input_csv, output_csv, rules_json = map(lambda path: Path(path).resolve(), (input_csv, output_csv, rules_json))
    protected = {input_csv}
    if manifest_path is not None:
        manifest_path = Path(manifest_path).resolve()
        protected.add(manifest_path)
    if output_csv == rules_json or output_csv in protected or rules_json in protected:
        raise ValueError('annotation outputs must be distinct from each other and all inputs')
    if output_csv.exists() or rules_json.exists():
        raise FileExistsError('preserve existing annotation outputs; choose a new output name')
    before = {path.as_posix(): sha(path) for path in protected}
    with input_csv.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        fields = next(reader)
        rows = []
        for values in reader:
            item = {name: values[index] if index < len(values) else '' for index, name in enumerate(fields)}
            item['__anomaly_field_count'] = len(values)
            item['__anomaly_extra_fields'] = values[len(fields):]
            rows.append(item)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path else None
    annotated, rules = annotate_rows(rows, input_fields=fields, manifest=manifest)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rules_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(annotated)
    if any(sha(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError('input changed during annotation; derived output is not accepted')
    rules.update(input_csv=input_csv.as_posix(), input_sha256=before[input_csv.as_posix()],
                 input_manifest=manifest_path.as_posix() if manifest_path else None,
                 input_manifest_sha256=before.get(manifest_path.as_posix()) if manifest_path else None,
                 output_csv=output_csv.as_posix(), output_sha256=sha(output_csv),
                 annotator_source_sha256=sha(__file__), input_hashes_unchanged=True)
    with rules_json.open('x', encoding='utf-8') as stream:
        json.dump(rules, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-csv', required=True, type=Path)
    parser.add_argument('--rules-json', required=True, type=Path)
    args = parser.parse_args()
    rules = annotate_csv(args.csv, args.output_csv, args.rules_json, manifest_path=args.manifest)
    print(json.dumps({key: rules[key] for key in ('input_row_count', 'output_row_count', 'invalid_rows', 'rows_with_any_flag', 'output_sha256')}))


if __name__ == '__main__':
    main()
