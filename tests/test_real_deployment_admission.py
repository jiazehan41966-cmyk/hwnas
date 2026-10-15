from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools/sonar_spec')]
from tools.sonar_deploy import admission as a
import deployment_pipeline as p


def save(path, value):
    a.atomic_json(path, value)


def artifact(path, text='artifact'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {'path': str(path), 'sha256': a.sha256(path)}


def metrics(value=.02, recall=.4):
    return {'AP50': value * 2, 'AP75': value / 2, 'mAP50_95': value,
            'Precision': .3, 'Recall': recall}


@pytest.fixture
def deployment(tmp_path):
    source = artifact(tmp_path/'frozen_source.json')
    rows = []
    for network in a.NETWORKS:
        for seed in a.SEEDS:
            checkpoint = artifact(tmp_path/'inputs'/f'{network}{seed}.pt', f'{network}{seed}')
            profile = artifact(tmp_path/'inputs'/f'{network}{seed}.json', f'profile{network}{seed}')
            row = dict(network=network, seed=seed, architecture_id=a.ARCHITECTURES[network],
                checkpoint_path=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
                profile_path=profile['path'], profile_sha256=profile['sha256'],
                calibration_sha256='c'*64, numeric_contract_hash='d'*64,
                split_manifest_sha256='e'*64, evaluator_sha256='f'*64,
                historical_reference_metrics={k: v for k, v in metrics().items() if k != 'AP75'})
            rows.append(row)
    binding = dict(schema_version=a.SCHEMA, status='FROZEN', split='valid', test_evaluated=False,
                   source_files={'frozen': source}, runs=rows)
    save(tmp_path/'DEPLOYMENT_BINDING.json', binding)
    return tmp_path


def write_result(root, network, seed, value=.03, recall=.5):
    expected = next(r for r in a.read_json(root/'DEPLOYMENT_BINDING.json')['runs']
                    if (r['network'], r['seed']) == (network, seed))
    identity = {k: expected[k] for k in a.IDENTITY}
    package = root/network/f'seed_{seed}'/'package'
    files = {name: artifact(package/name)['sha256'] for name in ('graph.json', 'weights.bin', 'biases.bin', 'layers.bin')}
    save(package/'manifest.json', {**identity, 'files': files, 'test_evaluated': False, 'deterministic_export': 'PASS'})
    result = dict(**identity, status='COMPLETE', historical_reproduction='PASS', split='valid', test_evaluated=False,
                  historical_metrics=metrics(), integer_metrics=metrics(value, recall),
                  binding_sha256=a.sha256(root/'DEPLOYMENT_BINDING.json'),
                  package_manifest_path=str(package/'manifest.json'), package_manifest_sha256=a.sha256(package/'manifest.json'))
    result['metric_event_ids'] = a.record_result_metrics(root, result)
    save(package.parent/'result.json', result)
    return result


def all_results(root, c_value=.03, c_recall=.5):
    for network in a.NETWORKS:
        for seed in a.SEEDS:
            write_result(root, network, seed, c_value if network == 'C' else .03,
                         c_recall if network == 'C' else .5)


def stage(root, network, kind):
    result = a.read_json(root/network/'seed_42/result.json')
    directory = root/network/'seed_42'
    item = artifact(directory/f'{kind}.artifact', kind)
    evidence = dict(**{k: result[k] for k in a.IDENTITY}, stage=kind, status='PASS', scope='full_detector',
        test_evaluated=False, package_manifest_sha256=result['package_manifest_sha256'],
        deployment_binding_sha256=a.sha256(root/'DEPLOYMENT_BINDING.json'), artifacts={'proof': item},
        metric_values={})
    previous = a.STAGES.index(kind)
    if previous:
        previous_kind = a.STAGES[previous - 1]
        evidence['dependencies'] = {previous_kind: a.sha256(directory/f'{previous_kind}.json')}
    provenance = dict(source=item['path'], evidence_layer='place_and_route', target_device='ZCU104',
                      artifact_sha256=item['sha256'], tool_version='2022.2')
    if kind in ('integer_parity', 'hls', 'board'):
        evidence['mismatch_count'] = 0
    if kind == 'hls':
        evidence.update(csim='PASS', cosim='PASS')
    if kind == 'route':
        evidence.update(timing_met=True, wns_ns=.1, tns_ns=0., actual_clock_mhz=100.)
        values = {'timing_met': True, 'wns_ns': .1, 'tns_ns': 0.,
                  **{f'{r}_{q}': 1 for r in ('lut', 'ff', 'dsp', 'bram') for q in ('used', 'utilization_pct')}}
        evidence['metric_values'] = {k: {'value': v, 'provenance': provenance} for k, v in values.items()}
    if kind == 'board':
        evidence.update(train_sample_count=8, valid_sample_count=99, warmup=1, repetitions=10, batch_size=1,
                        valid_metrics=result['integer_metrics'], timing={
                            'graph_seconds': [.8]*10, 'end_to_end_seconds': [1.]*10,
                            'graph_median_ms': 800., 'graph_p95_ms': 800.,
                            'end_to_end_median_ms': 1000., 'end_to_end_p95_ms': 1000.})
        provenance.update(evidence_layer='board_measurement', measurement_method='board_monotonic_timer',
                          precision='W8A8', batch_size=1, input_shape=[1,1,640,640], warmup=1, repetitions=10)
        values = {'board_latency_ms': 1000., 'throughput_fps': 1., 'actual_clock_mhz': 100.,
                  **{target: result['integer_metrics'][source] for source, target in a.METRICS.items()}}
        evidence['metric_values'] = {k: {'value': v, 'provenance': provenance} for k, v in values.items()}
    evidence['metric_event_id'] = a.record_stage_metrics(root, evidence)
    save(directory/f'{kind}.json', evidence)
    return evidence


def test_binding_admits_exports_without_previous_search_or_accuracy(deployment):
    assert a.action_admission(deployment, 'D', 'parameter_export', 44)['allowed']
    assert not a.action_admission(deployment, 'D', 'hls')['allowed']
    assert a.audit(deployment)['accuracy']['complete_runs'] == 0


def test_complexity_must_be_durably_bound_without_replacing_accuracy(deployment):
    result=write_result(deployment,'D',42)
    result['complexity']=dict(params=100,macs=200,flops=400,scope='batch1 Conv2d/Linear estimate only')
    save(deployment/'D/seed_42/result.json',result)
    expected=a.read_json(deployment/'DEPLOYMENT_BINDING.json')['runs'][0]
    with pytest.raises(FileNotFoundError):a.load_result(deployment,expected)
    old_ids=result['metric_event_ids'][:]
    first=a.record_complexity_metrics(deployment,result)
    assert a.record_complexity_metrics(deployment,result)==first
    assert a.load_result(deployment,expected)['metric_event_ids']==old_ids
    result['complexity']['params']=101
    save(deployment/'D/seed_42/result.json',result)
    with pytest.raises(ValueError,match='Complexity record differs'):a.load_result(deployment,expected)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'architecture', 'test', 'split', 'source', 'checkpoint', 'profile', 'calibration'])
def test_invalid_binding_fails_closed(deployment, mutation):
    path = deployment/'DEPLOYMENT_BINDING.json'
    binding = a.read_json(path)
    if mutation == 'duplicate': binding['runs'][-1] = deepcopy(binding['runs'][0])
    elif mutation == 'missing': binding['runs'].pop()
    elif mutation == 'architecture': binding['runs'][0]['architecture_id'] = 'different'
    elif mutation == 'test': binding['test_evaluated'] = True
    elif mutation == 'split': binding['split'] = 'test'
    elif mutation == 'source': Path(binding['source_files']['frozen']['path']).write_text('changed')
    elif mutation in ('checkpoint', 'profile'): Path(binding['runs'][0][mutation+'_path']).write_text('changed')
    else: binding['runs'][0]['calibration_sha256'] = '1'*64
    save(path, binding)
    assert not a.action_admission(deployment, 'D', 'parameter_export')['allowed']


def test_exact_nine_integer_results_and_same_implementation_D_gate(deployment):
    all_results(deployment)
    report = a.accuracy_admission(deployment)
    assert report['complete_runs'] == 9
    assert report['gates']['D']['status'] == 'CONTROL_PASS'
    assert report['gates']['B1']['passed']
    assert a.action_admission(deployment, 'D', 'integer_parity')['allowed']
    assert not a.action_admission(deployment, 'D', 'hls')['allowed']


@pytest.mark.parametrize('metric', ['mAP50_95', 'Recall'])
def test_either_accuracy_metric_below_D_stops_hardware_only_for_candidate(deployment, metric):
    all_results(deployment, c_value=.029 if metric == 'mAP50_95' else .03,
                c_recall=.49 if metric == 'Recall' else .5)
    gate = a.action_admission(deployment, 'C', 'integer_parity')
    assert not gate['allowed'] and gate['status'] == 'STOP_ACCURACY_BELOW_D'
    assert a.action_admission(deployment, 'D', 'integer_parity')['allowed']


@pytest.mark.parametrize('mutation', ['status', 'test', 'metric_missing', 'nan', 'historical_ap', 'historical_recall',
                                    'package', 'ledger_missing', 'ledger_wrong_value', 'identity'])
def test_unverified_result_never_completes(deployment, mutation):
    all_results(deployment)
    path = deployment/'D/seed_42/result.json'
    result = a.read_json(path)
    if mutation == 'status': result['status'] = 'IN_PROGRESS'
    elif mutation == 'test': result['test_evaluated'] = True
    elif mutation == 'metric_missing': del result['integer_metrics']['AP75']
    elif mutation == 'nan': result['integer_metrics']['AP50'] = float('nan')
    elif mutation == 'historical_ap': result['historical_metrics']['AP50'] += 1.1e-6
    elif mutation == 'historical_recall': result['historical_metrics']['Recall'] += 1e-12
    elif mutation == 'package': (path.parent/'package/weights.bin').write_text('changed')
    elif mutation == 'ledger_missing': result['metric_event_ids'] = ['a'*64, 'b'*64]
    elif mutation == 'ledger_wrong_value': result['integer_metrics']['AP50'] += .001
    else: result['profile_sha256'] = 'a'*64
    path.write_text(json.dumps(result))
    assert a.accuracy_admission(deployment)['status'] == 'BLOCKED_EVIDENCE'


def test_all77_fields_durable_and_repeat_import_idempotent(deployment):
    result = write_result(deployment, 'D', 42)
    assert a.record_result_metrics(deployment, result) == result['metric_event_ids']
    records = a.read_metric_records(deployment/'METRICS.jsonl')
    assert len(records) == 2
    assert all(len(record['metrics']) == 77 for record in records)
    assert records[0]['metrics']['top1_accuracy']['status'] == 'NOT_APPLICABLE'
    assert records[0]['metrics']['board_latency_ms']['status'] == 'PENDING'


def test_actual_bound_stage_evidence_unblocks_successors(deployment):
    all_results(deployment)
    for kind in a.STAGES:
        assert a.action_admission(deployment, 'D', kind)['allowed']
        stage(deployment, 'D', kind)
    result = a.read_json(deployment/'D/seed_42/result.json')
    assert a.stage_evidence(deployment, 'D', 'board', result)['status'] == 'PASS'


@pytest.mark.parametrize('mutation', ['synthetic', 'mismatch', 'artifact', 'ledger', 'package', 'test'])
def test_synthetic_or_unbound_integer_proof_rejected(deployment, mutation):
    all_results(deployment)
    evidence = stage(deployment, 'D', 'integer_parity')
    path = deployment/'D/seed_42/integer_parity.json'
    if mutation == 'synthetic': evidence['scope'] = 'graph_A'
    elif mutation == 'mismatch': evidence['mismatch_count'] = 1
    elif mutation == 'artifact': Path(evidence['artifacts']['proof']['path']).write_text('changed')
    elif mutation == 'ledger': evidence['metric_event_id'] = 'a'*64
    elif mutation == 'package': evidence['package_manifest_sha256'] = 'a'*64
    else: evidence['test_evaluated'] = True
    save(path, evidence)
    assert not a.action_admission(deployment, 'D', 'hls')['allowed']


def test_no_new_search_calibration_or_other_board_seed(deployment):
    for action in ('search', 'promotion', 'ptq'):
        assert a.action_admission(deployment, 'D', action)['status'] == 'STOP_SCOPE'
    assert a.action_admission(deployment, 'D', 'board', 43)['status'] == 'STOP_SCOPE'


def test_board_order_requires_D_then_C_before_B1(deployment):
    all_results(deployment)
    for network in a.NETWORKS:
        for kind in a.STAGES[:-1]:
            stage(deployment, network, kind)
    assert a.action_admission(deployment, 'D', 'board')['allowed']
    assert not a.action_admission(deployment, 'C', 'board')['allowed']
    stage(deployment, 'D', 'board')
    assert a.action_admission(deployment, 'C', 'board')['allowed']
    assert not a.action_admission(deployment, 'B1', 'board')['allowed']
    stage(deployment, 'C', 'board')
    assert a.action_admission(deployment, 'B1', 'board')['allowed']


def test_failed_C_is_skipped_in_fixed_board_order(deployment):
    all_results(deployment, c_value=.01)
    for network in ('D', 'B1'):
        for kind in a.STAGES[:-1]:
            stage(deployment, network, kind)
    stage(deployment, 'D', 'board')
    assert a.action_admission(deployment, 'B1', 'board')['allowed']


def test_dispatch_is_fixed_and_rejects_arbitrary_command(deployment):
    command = p.real_dispatch_command(deployment, 'D', 'parameter_export', 42)
    assert command[2:] == ['export-evaluate', '--root', str(deployment), '--network', 'D', '--seed', '42']
    with pytest.raises(ValueError): p.real_dispatch_command(deployment, 'D;rm', 'board', 42)


def test_dispatch_durably_journals_before_fixed_handler(deployment, tmp_path):
    report = tmp_path/'report'
    def runner(command, **kw):
        assert kw == {'check': False, 'shell': False}
        assert 'ACTION_GATE' in (report/'dispatch.jsonl').read_text()
        write_result(deployment, 'D', 42)
        return SimpleNamespace(returncode=0)
    result, code = p.real_pipeline('dispatch', deployment, report, 'D', 'parameter_export', runner=runner)
    assert code == 0 and result['status'] == 'COMPLETE_VERIFIED'
    result, code = p.real_pipeline('dispatch', deployment, report, 'D', 'parameter_export', runner=lambda *a, **k: pytest.fail('duplicate execution'))
    assert code == 3 and result['status'] == 'ALREADY_COMPLETE'


def test_handler_zero_exit_without_metrics_is_not_completion(deployment, tmp_path):
    result, code = p.real_pipeline('dispatch', deployment, tmp_path/'report', 'D', 'parameter_export',
                                  runner=lambda *args, **kw: SimpleNamespace(returncode=0))
    assert code == 4 and result['status'] == 'BLOCKED_RESULT_EVIDENCE'


def test_failed_handler_not_marked_success(deployment, tmp_path):
    result, code = p.real_pipeline('dispatch', deployment, tmp_path/'report', 'D', 'parameter_export',
                                  runner=lambda *args, **kw: SimpleNamespace(returncode=17))
    assert code == 17 and result['status'] == 'ACTION_FAILED'


def test_partial_report_does_not_invent_metrics_or_hardware(deployment, tmp_path):
    import csv
    from tools.sonar_deploy.report import generate_report
    write_result(deployment, 'D', 42)
    output = tmp_path/'report'
    report = generate_report(deployment, output)
    assert report['status'] == 'PARTIAL_EVIDENCE' and report['verified_runs'] == 1
    rows = list(csv.DictReader((output/'INTEGER_ACCURACY_RESULTS.csv').open()))
    assert len(rows) == 9 and sum(r['status'] == 'VERIFIED' for r in rows) == 1
    assert sum(r['integer_mAP50_95'] == '' for r in rows) == 8
    assert report['board_completed'] == [] and not report['test_evaluated']


def test_nine_software_results_not_full_deployment_pass(deployment, tmp_path):
    from tools.sonar_deploy.report import generate_report
    all_results(deployment)
    report = generate_report(deployment, tmp_path/'report')
    assert report['verified_runs'] == 9 and report['metrics_events'] == 18
    assert report['status'] == 'SOFTWARE_EVALUATION_COMPLETE_HARDWARE_PENDING'
    assert report['board_completed'] == []


def test_raw_board_timing_is_required(deployment):
    all_results(deployment)
    for kind in a.STAGES:
        stage(deployment, 'D', kind)
    path = deployment/'D/seed_42/board.json'
    proof = a.read_json(path)
    proof['timing']['end_to_end_seconds'].pop()
    save(path, proof)
    with pytest.raises(ValueError, match='ten measured'):
        a.stage_evidence(deployment, 'D', 'board', a.read_json(deployment/'D/seed_42/result.json'))


def test_report_refresh_failure_invalidates_previous_admission(deployment, tmp_path):
    from tools.sonar_deploy.report import generate_report
    all_results(deployment)
    output = tmp_path/'report'
    assert generate_report(deployment, output)['verified_runs'] == 9
    (deployment/'frozen_source.json').write_text('changed')
    report = generate_report(deployment, output)
    assert report['status'] == 'BLOCKED_EVIDENCE'
    assert a.read_json(output/'REPORT_MANIFEST.json')['status'] == 'BLOCKED_EVIDENCE'
    assert 'Earlier CSV files are not an admitted current snapshot' in (output/'REAL_INT8_DEPLOYMENT_REVIEW.md').read_text()
