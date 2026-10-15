"""Verify fetched raw numerical/source/report evidence before summarizing HLS."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.modnas_comparison.remote_worker import parse_report, verify_case, sha


def collect(folder):
    folder = Path(folder).resolve()
    evidence = folder / 'remote_evidence'
    target = folder / 'verified_results.json'
    if target.exists():
        raise FileExistsError('Preserve existing verified report')
    preparation = json.loads((folder / 'preparation_manifest.json').read_text(encoding='utf-8'))
    staging = json.loads((evidence / 'staging_manifest.json').read_text(encoding='utf-8'))
    for name, expected in staging['files'].items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()) or sha(path) != expected:
            raise ValueError('Fetched source closure drift: ' + name)
    status = json.loads((evidence / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'FINITE_PROBE_COMPLETE':
        raise ValueError('Probe is incomplete; retain evidence without a completed summary')
    results = []
    baseline = preparation.get('baseline_exact_measurement')
    if baseline is None:
        parent = Path(preparation['parent_probe']).resolve()
        if not parent.is_relative_to((ROOT / 'results/modnas_vs_ours_20261007').resolve()):
            raise ValueError('Parent probe is outside the declared experiment')
        if sha(folder / 'causal_diagnosis.json') != preparation['causal_diagnosis_sha256']:
            raise ValueError('Causal follow-up provenance mismatch')
        baseline = json.loads((parent / 'preparation_manifest.json').read_text(encoding='utf-8'))['baseline_exact_measurement']
    for row in preparation['cases']:
        case = evidence / 'cases' / row['name']
        key = verify_case(case)
        if key['source_hash'] != row['source_hash']:
            raise ValueError('Source hash differs from preparation')
        result = json.loads((case / 'result.json').read_text(encoding='utf-8'))
        if result['source_hash'] != key['source_hash'] or result['PE'] != key['PE']:
            raise ValueError('Result provenance differs from the registered case')
        verified = dict(result, numerical_report_verification='FAILED_OR_INCOMPLETE')
        if result['status'] == 'PASS_CSIM_AND_CSYNTH':
            output = case / 'project/solution1/csim/build/hardware_output.bin'
            report = case / 'project/solution1/syn/report/closure_kernel_csynth.xml'
            if not output.is_file() or sha(output) != sha(case / 'reference.bin'):
                raise ValueError('Fetched integer byte parity failed')
            if sha(report) != result['source_report_sha256']:
                raise ValueError('Fetched report hash mismatch')
            metrics = parse_report(report)
            if metrics != result['metrics']:
                raise ValueError('Stored metrics differ from independently parsed raw report')
            verified.update(numerical_report_verification='PASS_RAW_SOURCE_INTEGER_AND_XML',
                            measured_stem_speedup_over_original=baseline['latency_us'] / metrics['latency_us'],
                            stem_meets_120ms_serial_proxy=metrics['latency_us'] <= 120000,
                            report_path=str(report.relative_to(ROOT)), output_sha256=sha(output),
                            report_sha256=sha(report))
        results.append(verified)
    valid = [r for r in results if r['numerical_report_verification'] == 'PASS_RAW_SOURCE_INTEGER_AND_XML'
             and r['metrics']['timing_eligible'] and r['metrics']['resource_screen_eligible']]
    best = min(valid, key=lambda r: r['metrics']['latency_us']) if valid else None
    lower_path = ROOT / 'results/modnas_vs_ours_20261007/hardware_lower_bound.json'
    lower = json.loads(lower_path.read_text(encoding='utf-8'))
    known = lower['candidates'][0]['known_latency_lower_bound_decimal_us']
    replaced = (Decimal(known) - Decimal(str(baseline['latency_us']))
                + Decimal(str(best['metrics']['latency_us']))) if best else None
    report = dict(status='PASS_EVIDENCE_COLLECTION', measurement_scope='STEM_OP_LEVEL_HLS_CSYNTH_ONLY',
                  baseline_exact_measurement=baseline, results=results,
                  best_timing_resource_eligible_case=best['case_name'] if best else None,
                  known_backbone_partial_serial_us_with_best_stem=float(replaced) if replaced is not None else None,
                  fixed_other_backbone_mapping_costs_preserved=True,
                  inherited_proxy_budget_us=120000, full_network_latency_us=None,
                  full_network_hardware_feasibility='UNKNOWN', board_measured=False,
                  network_search_started=False, test_evaluated=False,
                  preparation_manifest_sha256=sha(folder / 'preparation_manifest.json'),
                  staging_manifest_sha256=sha(evidence / 'staging_manifest.json'),
                  result_status_sha256=sha(evidence / 'status.json'), lower_bound_sha256=sha(lower_path))
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=ROOT / 'results/modnas_vs_ours_20261007/mapping_probe')
    args = parser.parse_args()
    result = collect(args.local)
    print(json.dumps({key: result[key] for key in ['status', 'best_timing_resource_eligible_case',
                                                  'known_backbone_partial_serial_us_with_best_stem']}), flush=True)


if __name__ == '__main__':
    main()
