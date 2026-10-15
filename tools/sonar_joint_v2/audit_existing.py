"""Read retained measurements and assess engineering budgets; never launch work."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_joint_v2.deployment_budget import assess
from tools.sonar_search_finalize.run import sha, read, bounded

PREREG = ROOT / 'artifacts/zcu104_full_campaign_v1/control/hardware_gate_preregistration_20260810.json'
PERFORMANCE = ROOT / 'results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114'
D_VALID = ROOT / 'results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D'


def audit(performance=PERFORMANCE, d_valid=D_VALID):
    sources = {}

    def record(path):
        sources[Path(path).relative_to(ROOT).as_posix()] = sha(path)
        return Path(path)

    prereg = read(record(PREREG))
    board = d_valid / 'graph_build/board_evidence'
    manifest = read(record(board / 'board_manifest.json'))
    for rel in ('timing.rpt', 'utilization_hier.rpt', 'h2_graph.bit'):
        if sha(record(bounded(board, rel))) != manifest['files'][rel]:
            raise ValueError('Route artifact differs from board manifest: ' + rel)
    timing = (board / 'timing.rpt').read_text(encoding='utf-8')
    match = re.search(r'^\s*WNS\(ns\).*\n[^\n]*\n\s*([-\d.]+)\s+([-\d.]+)', timing, re.M)
    clock = re.search(r'^clk_pl_0\s+\{[^}]+\}\s+([\d.]+)\s+([\d.]+)', timing, re.M)
    utilization = (board / 'utilization_hier.rpt').read_text(encoding='utf-8')
    top = next((line for line in utilization.splitlines() if '| h2_graph_bd_wrapper ' in line), None)
    if not match or not clock or not top or 'Design State : Routed' not in utilization:
        raise ValueError('Retained routed design fields missing')
    cells = [v.strip() for v in top.split('|')[1:-1]]
    route = dict(LUT=int(cells[2]), DSP=int(cells[10]), BRAM18=2*int(cells[7])+int(cells[8]),
                 URAM=int(cells[9]), wns_ns=float(match[1]), tns_ns=float(match[2]), implemented_period_ns=float(clock[1]))
    rows = []
    for network in ('D', 'C', 'B1'):
        folder = performance / network
        report = read(record(folder / 'PERFORMANCE_REPORT.json'))
        if report['network'] != network or report['status'] != 'PERFORMANCE_PASS' or report['test_evaluated'] is not False:
            raise ValueError('Wrong/unfinished performance report')
        power = report['power']['results']; summary = power['summary']
        csv_path = record(folder / 'results/power/power_samples.csv')
        if sha(csv_path) != power['samples_sha256']:
            raise ValueError('Power raw CSV hash mismatch')
        with csv_path.open(encoding='utf-8', newline='') as stream:
            samples = [(float(r['monotonic_s']), float(r['power_w'])) for r in csv.DictReader(stream)]
        idle_count = sum(power['idle_window'][0] <= t <= power['idle_window'][1] for t, _ in samples)
        calls = [c for c in summary['calls'] if c['warmup'] is False]
        latencies = report['e2e']['timing']['end_to_end_seconds']
        # Only D has a locally byte-verified full valid99 return. C/B1 values
        # are not inferred from the D executor or from their single-image smoke.
        valid = None
        if network == 'D':
            valid_report = read(record(d_valid / 'VALID99_REPORT.json'))
            valid = (valid_report.get('status') == 'VALID99_PASS'
                     and valid_report.get('network') == 'D' and valid_report.get('seed') == 42
                     and valid_report.get('test_evaluated') is False
                     and valid_report.get('train_frames') == 8 and valid_report.get('valid_frames') == 99
                     and valid_report.get('mismatch_count') == 0
                     and valid_report.get('integer_metric_differences') == {}
                     and valid_report.get('deployment_binding_sha256') == report['e2e']['deployment_binding_sha256']
                     and valid_report.get('graph_sha256') == report['e2e']['graph_sha256'])
        metrics = dict(board_e2e_ms=report['e2e']['timing']['end_to_end_median_ms'],
             dynamic_power_w=summary['power_w']-summary['idle_power_w'],
             dynamic_energy_mj=summary['dynamic_energy_per_inference_mj'],
             idle_active_pair_repetitions=1,
             capture_duration_s=min(summary['idle']['duration_s'], summary['active_duration_s']),
             idle_sample_count=idle_count, active_sample_count=sum(c['sample_count_inside'] for c in calls),
             active_completed_ops=summary['measured_inferences'], latency_timed_ops=len(latencies),
             coefficient_of_variation=statistics.stdev(latencies)/statistics.mean(latencies),
             same_integer_valid99=valid, matched_candidate_quant_mapping_bitstream=None,
             power_above_noise_floor=None)
        if network == 'D':
            metrics.update(route)
        result = assess(metrics, prereg)
        result.update(network=network, precision='W8A8', hardware_seed=42,
             reported_full_board_acceptance=report['full_board_acceptance'],
             note='Numerical check of retained measurements; optimized H4 mapping identity and final artifact admission not established')
        rows.append(result)
    return dict(status='EXISTING_DEPLOYMENT_BUDGETS_NOT_MET', candidates=rows,
                sources=sources, preregistration=prereg, test_evaluated=False,
                full_deployment_acceptance=False, hardware_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new budget-assessment file')
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({r['network']: r['status'] for r in result['candidates']}))


if __name__ == '__main__':
    main()
