"""Reporting fixtures only; no synthetic values enter actual deployment reports."""
import csv
from pathlib import Path

from test_real_deployment_admission import deployment, all_results, stage, write_result, save
from tools.sonar_deploy import admission as a
from tools.sonar_deploy.report import generate_report


def comparison(output):
    with (output/'FINAL_DEPLOYMENT_COMPARISON.csv').open(encoding='utf-8', newline='') as f:
        return {r['network']:r for r in csv.DictReader(f)}


def test_incomplete_three_seed_mean_and_hardware_stay_pending(deployment,tmp_path):
    write_result(deployment,'D',42)
    output=tmp_path/'reports';generate_report(deployment,output)
    row=comparison(output)['D']
    assert row['verified_seed_count']=='1' and row['integer_3seed_mean_mAP50_95']=='PENDING'
    assert float(row['seed42_software_mAP50_95'])==.03
    assert row['seed42_board_mAP50_95']=='PENDING' and row['graph_median_ms']=='PENDING'
    assert row['power_status']=='NOT_MEASURED' and row['energy_per_inference_mj']=='NOT_MEASURED'
    outputs=a.read_json(output/'REPORT_MANIFEST.json')['output_sha256']
    assert {'FINAL_DEPLOYMENT_COMPARISON.csv','D_BOARD_REPORT.md','C_BOARD_REPORT.md','B1_BOARD_REPORT.md'}<=set(outputs)
    assert all(a.sha256(output/name)==outputs[name] for name in outputs)


def test_admitted_board_values_and_reciprocal_latency_fps(deployment,tmp_path):
    all_results(deployment)
    for kind in a.STAGES:stage(deployment,'D',kind)
    output=tmp_path/'reports';result=generate_report(deployment,output);rows=comparison(output)
    assert result['board_completed']==['D']
    row=rows['D']
    assert row['board_status']=='PASS' and row['seed42_exact_metric_parity']=='PASS'
    assert float(row['integer_3seed_mean_mAP50_95'])==.03 and float(row['integer_3seed_sample_std_mAP50_95'])==0
    assert float(row['graph_median_ms'])==800 and float(row['graph_fps_from_median_latency'])==1.25
    assert float(row['end_to_end_p95_ms'])==1000 and float(row['end_to_end_fps_from_p95_latency'])==1
    assert float(row['lut_used'])==1 and row['lut_capacity']=='PENDING'
    assert float(row['wns_ns'])==.1 and float(row['board_clock_mhz'])==100
    assert rows['C']['board_status']=='PENDING' and rows['C']['lut_used']=='PENDING'
    text=(output/'D_BOARD_REPORT.md').read_text()
    assert 'not empirical FPS percentiles' in text and 'test_evaluated=false' in text


def test_unadmitted_or_damaged_proof_never_supplies_hardware_values(deployment,tmp_path):
    all_results(deployment,c_value=.02)
    for network in ('D','C'):
        for kind in a.STAGES:stage(deployment,network,kind)
    (deployment/'D/seed_42/route.artifact').write_text('damaged')
    output=tmp_path/'reports';generate_report(deployment,output);rows=comparison(output)
    assert rows['D']['route_status']=='BLOCKED_EVIDENCE' and rows['D']['board_status']=='BLOCKED_EVIDENCE'
    assert rows['D']['wns_ns']=='PENDING' and rows['D']['seed42_board_AP50']=='PENDING'
    assert rows['C']['board_status']=='STOP_ACCURACY_BELOW_D' and rows['C']['graph_median_ms']=='PENDING'


def test_power_min_max_requires_bound_raw_samples_and_ledger(deployment,tmp_path):
    all_results(deployment)
    for kind in a.STAGES:proof=stage(deployment,'D',kind)
    folder=deployment/'D/seed_42';raw=folder/'hardware/board_evidence/power/power_samples.csv'
    raw.parent.mkdir(parents=True)
    with raw.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['monotonic_s','power_w']);writer.writeheader()
        writer.writerows(dict(monotonic_s=i,power_w=2 if i<2 else 4) for i in range(13))
    calls=[dict(warmup=False,start_monotonic_s=2+i,end_monotonic_s=2.8+i) for i in range(10)]
    summary=dict(power_w=4.,energy_per_inference_mj=3200.,idle_power_w=2.,dynamic_energy_per_inference_mj=1600.,calls=calls)
    proof.update(power_status='MEASURED',power_evidence=dict(status='MEASURED',scope='whole_board_12v_input',
        samples_sha256=a.sha256(raw),idle_window=[0,1],summary=summary))
    proof['artifacts']['power']={'path':str(raw),'sha256':a.sha256(raw)}
    provenance=dict(proof['metric_values']['board_latency_ms']['provenance'],power_scope='whole_board_12v_input',idle_subtracted=False)
    for key in ('power_w','energy_per_inference_mj'):
        proof['metric_values'][key]={'value':summary[key],'provenance':provenance}
    proof['metric_event_id']=a.record_stage_metrics(deployment,proof);save(folder/'board.json',proof)
    output=tmp_path/'reports';generate_report(deployment,output);row=comparison(output)['D']
    assert row['power_status']=='MEASURED' and float(row['active_power_min_w'])==4
    assert float(row['active_power_max_w'])==4 and float(row['idle_power_min_w'])==2
    assert float(row['energy_per_inference_mj'])==3200 and row['power_samples_sha256']==a.sha256(raw)
    raw.write_text('changed')
    generate_report(deployment,output);row=comparison(output)['D']
    assert row['board_status']=='BLOCKED_EVIDENCE' and row['power_status']=='NOT_MEASURED'


def test_refresh_failure_removes_stale_comparison_claims(deployment,tmp_path):
    all_results(deployment)
    for kind in a.STAGES:stage(deployment,'D',kind)
    output=tmp_path/'reports';generate_report(deployment,output)
    (deployment/'frozen_source.json').write_text('changed')
    report=generate_report(deployment,output);row=comparison(output)['D']
    assert report['status']=='BLOCKED_EVIDENCE'
    assert row['software_status']=='BLOCKED_EVIDENCE' and row['graph_median_ms']=='PENDING'
    assert 'Current refresh blocked' in (output/'D_BOARD_REPORT.md').read_text()
