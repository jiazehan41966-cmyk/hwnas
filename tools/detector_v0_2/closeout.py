"""Read completed pilot artifacts; write a bounded, reproducible evidence summary."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.detector_v0_2 import remote, train
from tools.detector_v0_2.provenance import verify_execution_manifest

BASE = ROOT / 'results/detector_v0_2_execution_20261002'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_bytes())


def main():
    snapshot = BASE / 'remote_snapshot/raw_bytes'
    queue = read(snapshot / 'queue_status.json')
    if queue['status'] != 'COMPLETE' or len(queue['completed']) != 4:
        raise ValueError('all four seed42 pilot runs must complete before closeout')
    completed = {row['prototype']: row['result_sha256'] for row in queue['completed']}
    manifest = snapshot / 'execution_manifest.json'
    verify_execution_manifest(manifest, queue['binding'])
    baseline = next(row for row in csv.DictReader((ROOT / 'reports/phase2f_multiseed/multiseed_results.csv').open())
                    if row['label'] == 'B1' and row['seed'] == '42')
    rows = []
    for prototype in train.PROTOTYPES:
        folder = snapshot / 'formal' / prototype / 'seed_42'
        result = read(folder / 'result.json')
        status = read(folder / 'status.json')
        history = read(folder / 'epoch_log.json')
        best = max(history, key=lambda row: row['valid']['mAP50_95'])
        assert status['status'] == 'COMPLETE' and status['epochs'] == 80
        assert status['binding']['execution_manifest_sha256'] == sha(manifest)
        assert completed[prototype] == sha(folder / 'result.json')
        assert result['history_sha256'] == sha(folder / 'epoch_log.json')
        assert result['best_epoch'] == best['epoch']
        assert [row['epoch'] for row in history] == list(range(1, 81))
        assert all(row['batches'] == 83 and row['gradient_finite'] and row['valid']['num_images'] == 99 for row in history)
        assert result['test_evaluated'] is False and status['test_evaluated'] is False
        for metric in train.METRICS:
            assert result[metric] == best['valid'][metric]
        rows.append(dict(prototype=prototype, seed=42, best_epoch=result['best_epoch'],
            **{metric: result[metric] for metric in train.METRICS},
            AP_delta_vs_B1_seed42=result['mAP50_95'] - float(baseline['mAP50_95']),
            Recall_delta_vs_B1_seed42=result['Recall'] - float(baseline['Recall']),
            Params=result['Params'], MACs=result['MACs'],
            per_class_precision_recall=result['supplemental']['per_class_precision_recall'],
            result_sha256=sha(folder / 'result.json')))
    # Read the actual remote checkpoint bytes rather than trusting result fields.
    code = '''from pathlib import Path
import json,hashlib
b=Path(REMOTE_FOLDER)
out={}
for proto in ['R1','R0','R8','R7']:
 p=b/'formal'/proto/'seed_42'
 r=json.loads((p/'result.json').read_bytes())
 h=lambda name:hashlib.sha256((p/name).read_bytes()).hexdigest()
 out[proto]={'best':h('best.pt'),'latest':h('latest.pt'),'history':h('epoch_log.json')}
 assert out[proto]['best']==r['best_checkpoint_sha256']
 assert out[proto]['latest']==r['latest_checkpoint_sha256']
 assert out[proto]['history']==r['history_sha256']
print(json.dumps(out))
'''.replace('REMOTE_FOLDER', repr(remote.REMOTE + '/' + remote.RESULT))
    checkpoint_bytes = json.loads(remote.execute(code))

    evidence = BASE / 'hardware/route_pw48_v2/remote_evidence'
    proof = read(evidence / 'timing_audit_v1/proof.json')
    route_root = '/home/jiajia/experiments/detector_v0_2_hardware_20261002_route_pw48_v2'
    route_code = '''from pathlib import Path
import json,hashlib
b=Path(ROUTE_ROOT)
names=['routed.dcp','timing_audit_v1/audit.tcl','timing_audit_v1/timing_min_max.rpt']
print(json.dumps({n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in names}))
'''.replace('ROUTE_ROOT', repr(route_root))
    route_remote = json.loads(remote.execute(route_code))
    assert route_remote['routed.dcp'] == proof['routed_checkpoint_sha256']
    assert route_remote['timing_audit_v1/audit.tcl'] == proof['tcl_sha256']
    assert route_remote['timing_audit_v1/timing_min_max.rpt'] == sha(evidence / 'timing_audit_v1/timing_min_max.rpt')
    slacks = dict(line.split('=') for line in (evidence / 'timing_audit_v1/slack.txt').read_text().splitlines())
    timing = (evidence / 'timing_audit_v1/timing_min_max.rpt').read_text()
    assert 'All user specified timing constraints are met.' in timing
    state = read(evidence / 'route_state.json')
    assert state['status'] == 'ROUTED' and state['returncode'] == 0
    utilization = (evidence / 'route_utilization.rpt').read_text()
    resources = {key: int(re.search(r'\|\s*' + re.escape(label) + r'\s*\|\s*(\d+)', utilization).group(1))
                 for key, label in [('LUT', 'CLB LUTs'), ('DSP', 'DSPs'), ('BRAM36', 'Block RAM Tile')]}
    route = dict(status='PASS_INTERNAL_OOC_CONSTRAINTS_ONLY', period_ns=5.333,
        setup_slack_ns=float(slacks['max']), hold_slack_ns=float(slacks['min']), resources=resources,
        remote_byte_hashes=route_remote, setup_uncertainty_ns=1.440, hold_uncertainty_ns=0.050,
        external_IO_constraints='NOT_SET', clock_source_skew='HD.CLK_SRC_NOT_SET',
        full_graph_route='NOT_RUN', board_latency='NOT_MEASURED')
    assert route['setup_slack_ns'] >= 0 and route['hold_slack_ns'] >= 0

    junit = ET.parse(BASE / 'final_validation/pytest.xml').getroot()
    suites = junit.findall('.//testsuite')
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == dict(tests=95, failures=0, errors=0, skipped=0)
    negative_dir = BASE / 'final_validation/provenance_negative_replay'
    negative_dir.mkdir(exist_ok=False)
    review_path = ROOT / 'reports/detector_v0_2_execution_20261002/independent_check.py'
    spec = importlib.util.spec_from_file_location('initial_independent_review', review_path)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    review.OUT = negative_dir
    negative = review.provenance_negative_review()
    assert negative['status'] == 'PASS_REJECTED' and negative['accepted'] is False
    witness_path = BASE / 'trained_contract/R1_seed42_v2/WITNESS.json'
    witness = read(witness_path)
    assert witness['integer_vs_canonical_equal'] and witness['real_weights_bound'] and witness['formal_result_complete']
    assert witness['execution_manifest_sha256'] == sha(manifest)
    sources = [ROOT / 'tools/detector_v0_2' / name for name in
        ('closeout.py', 'provenance.py', 'ptq_export.py', 'remote.py', 'trained_contract.py')]
    sources += [ROOT / 'src/hwnas_fpga/deploy' / name for name in ('sonar_v0_2_integer.py', 'sonar_v0_2_ptq.py')]
    payload = dict(scope='first seed42 FP32 pilot and bounded software/operator engineering evidence',
        execution_manifest_sha256=sha(manifest), baseline_B1_seed42=baseline, pilots=rows,
        actual_remote_checkpoint_hashes=checkpoint_bytes, route=route, tests=counts,
        trained_witness=witness, original_independent_negative_replay=negative,
        negative_replay_operator='root; original independently designed fixture',
        fresh_independent_review_of_repair='NOT_RUN', three_seed_precision_gate='NOT_EVALUATED',
        seeds43_44_started=False, NAS_started=False, full_graph_board_latency='NOT_MEASURED',
        test_evaluated=False, source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
        evidence_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in
            [review_path, witness_path, BASE / 'final_validation/pytest.xml', manifest,
             evidence / 'timing_audit_v1/timing_min_max.rpt', evidence / 'route_utilization.rpt']})
    output = BASE / 'final_validation/CLOSEOUT.json'
    if output.exists():
        raise ValueError('preserve existing closeout evidence')
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status='CLOSEOUT_WRITTEN', pilots=rows, tests=counts,
        negative_replay=negative, route=route), ensure_ascii=False))


if __name__ == '__main__':
    main()
