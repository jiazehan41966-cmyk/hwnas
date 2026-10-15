import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hwnas_fpga.hardware.width_e_operator_cost import (
    ZCU104WidthEOperatorCostEstimator,VERSION,MAPPING,architecture_operator_keys,
)
from hwnas_fpga.hardware.mobilenetA_operator_cost import canonical_sha256
from hwnas_fpga.interfaces import SearchCandidate
from hwnas_fpga.runtime import build_constraints,build_hardware_spec,build_search_space
from hwnas_fpga.search.pareto import compute_pareto_front,resolve_pareto_objectives
from build_zcu104_width_e_search_inputs import enumerate_space,write_hashed
from audit_zcu104_storage_feedback import timing_metrics


def coupled_calibration(inputs,table,**overrides):
    calibration=dict(inputs['calibration'],operator_cost_payload_sha256=table['payload_sha256'],
        validated=True,routed_seed_accepted=True,heldout_architecture_check_passed=True,
        model_kind='coupled_bram_lutram_v1',bram_request_model=dict(slope=1,intercept=200),
        hls_bram18_domain=[0,2000],max_calibrated_excess_bram18=1000,
        overflow_residual_models={k:dict(slope=0,intercept=0)
                                  for k in ['lut_as_logic','lut_as_memory','ff','dsp']})
    calibration.update(overrides)
    return write_hashed(inputs['kwargs']['calibration_path'],calibration)

@pytest.fixture
def inputs(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/search/zcu104_width_e_sonar_serial_gap_v1.yaml').read_text())
    constraints=build_constraints(cfg)
    space=build_search_space(cfg,image_size=224,input_channels=1,num_classes=8,constraints=constraints)
    architectures=list(enumerate_space(space))
    keys={key for a in architectures for key in architecture_operator_keys(a,space)}
    rows=[dict(key=key,case_name=str(key),resources=dict(lut=10,lut_as_logic=8,lut_as_memory=2,ff=10,bram36=1,dsp=1),
               latency_cycles=100,power_w=1,timing_met=True) for key in keys]
    table=write_hashed(tmp_path/'table.json',dict(version=VERSION,mapping_id=MAPPING,rows=rows,
                         coverage_complete=True,search_space=cfg['search_space']))
    calibration=dict(version=VERSION,mapping_id=MAPPING,operator_cost_payload_sha256=table['payload_sha256'],
                     validated=False,routed_seed_accepted=False,heldout_architecture_check_passed=False)
    write_hashed(tmp_path/'calibration.json',calibration)
    return dict(cfg=cfg,space=space,architectures=architectures,table=table,calibration=calibration,
                kwargs=dict(hardware_spec=build_hardware_spec(cfg),constraints=constraints,
                            operator_cost_path=tmp_path/'table.json',calibration_path=tmp_path/'calibration.json'))

def test_formal_runtime_rejects_complete_lut_without_whole_network_calibration(inputs):
    with pytest.raises(ValueError,match='whole_network_storage_calibration_unvalidated'):
        ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'])

def test_full_cartesian_width_space_and_head_binding(inputs):
    architectures=inputs['architectures'];space=inputs['space']
    assert len(architectures)==360
    assert len({canonical_sha256(a.to_dict()) for a in architectures})==360
    keys={key for a in architectures for key in architecture_operator_keys(a,space)}
    assert len(keys)==39
    assert {key[1] for key in keys if key[0]=='pw_conv' and key[2]==1280}=={192,256}
    assert any(key[0]=='mbconv' and key[1:3]==(112,256) for key in keys)
    assert {'dmdc_conv_sonar','cfem_sonar'} <= {key[0] for key in keys}
    assert 'skip' not in {key[0] for key in keys}

def test_unknown_transition_is_not_zero_cost_or_nearest_neighbor(inputs):
    table=copy.deepcopy(inputs['table'])
    removed=next(row for row in table['rows'] if row['key'][0]=='pw_conv' and row['key'][1:3]==(192,1280))
    table['rows'].remove(removed)
    table=write_hashed(inputs['kwargs']['operator_cost_path'],table)
    calibration={**inputs['calibration'],'operator_cost_payload_sha256':table['payload_sha256']}
    write_hashed(inputs['kwargs']['calibration_path'],calibration)
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'],diagnostic_only=True)
    architecture=next(a for a in inputs['architectures'] if a.stages[-1].channels==192)
    with pytest.raises(ValueError,match='Unknown exact hardware costs'):
        estimator.estimate(architecture,inputs['space'])

def test_calibration_mapping_cannot_be_reused_after_fifo_or_gap_change(inputs):
    calibration={**inputs['calibration'],'mapping_id':'old_bram_fifo_parallel_gap'}
    write_hashed(inputs['kwargs']['calibration_path'],calibration)
    with pytest.raises(ValueError,match='mapping mismatch'):
        ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'],diagnostic_only=True)

def test_resource_budget_applies_to_calibrated_memory_not_total_lut_scale(inputs):
    calibration={**inputs['calibration'],'validated':True,'routed_seed_accepted':True,'heldout_architecture_check_passed':True,
                 'resource_upper_residuals':dict(lut=200000,lut_as_logic=0,lut_as_memory=102000,ff=0,bram36=0,dsp=0)}
    write_hashed(inputs['kwargs']['calibration_path'],calibration)
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'])
    cost=estimator.estimate(inputs['architectures'][0],inputs['space'])
    raw=estimator.estimate(inputs['architectures'][0],inputs['space'],apply_calibration=False)
    assert not raw.violations
    assert 'lut_as_memory>101760' in cost.violations
    assert cost.physical_metrics['zcu104_predicted_lut_as_memory']>102000

def test_bitstream_success_does_not_override_negative_setup_slack():
    report=ROOT/'results/zcu104_width_expansion_pilot_20260912_v1/remote_fifo_storage_control/evidence/harness_project/reports/timing_summary.rpt'
    if not report.exists(): pytest.skip('Local recorded hardware regression report is unavailable')
    timing=timing_metrics(report)
    assert timing['wns_ns']==-0.895
    assert timing['whs_ns']>0
    assert timing['timing_met'] is False

def test_configured_physical_axes_stay_populated_and_gated(inputs):
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'],diagnostic_only=True)
    cost=estimator.estimate(inputs['architectures'][0],inputs['space'])
    objectives,directions=resolve_pareto_objectives(
        inputs['cfg']['search']['pareto'],inputs['cfg']['search'].get('objective_weights'),
        inputs['kwargs']['constraints'],selection_metric='macro_f1',
    )
    assert {'physical_risk','early_expand_pressure'} <= set(objectives)
    metrics=cost.to_candidate_metrics()
    metrics.macro_f1=0.5
    assert [axis for axis in objectives if getattr(metrics,axis,None) is None]==[]
    # A missing physical axis silently empties the front instead of failing.
    front=compute_pareto_front([SearchCandidate('a',{},metrics)],objectives,directions)
    assert len(front)==1


def test_declared_bram18_pressure_budget_is_enforced(inputs):
    table=copy.deepcopy(inputs['table'])
    for row in table['rows']: row['resources']['bram36']=40
    table=write_hashed(inputs['kwargs']['operator_cost_path'],table)
    write_hashed(inputs['kwargs']['calibration_path'],
                 {**inputs['calibration'],'operator_cost_payload_sha256':table['payload_sha256']})
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'],diagnostic_only=True)
    cost=estimator.estimate(inputs['architectures'][0],inputs['space'])
    assert cost.physical_metrics['zcu104_predicted_bram18_pressure']>1.0
    assert any('zcu104_predicted_bram18_pressure' in violation for violation in cost.violations)


def test_lut_stats_keep_cost_semantics_and_fallback_policy(inputs):
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'],diagnostic_only=True)
    estimator.estimate(inputs['architectures'][0],inputs['space'])
    stats=estimator.get_lut_stats()
    assert stats['fallback_rate']==0.0 and stats['operator_cost_queries']>0
    assert stats['operator_cost_hit_rate']==1.0
    for field in ['latency_ms_semantics','power_w_semantics','energy_mj_semantics','fallback_policy']:
        assert stats[field]


def test_coupled_calibration_without_bram18_hls_is_rejected_at_load(inputs):
    table=write_hashed(inputs['kwargs']['operator_cost_path'],copy.deepcopy(inputs['table']))
    coupled_calibration(inputs,table)
    with pytest.raises(ValueError,match='requires a valid bram18_hls'):
        ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'])


def test_bram_overflow_before_fallback_reaches_candidate_metrics(inputs):
    table=copy.deepcopy(inputs['table'])
    for row in table['rows']: row['bram18_hls']=100
    table=write_hashed(inputs['kwargs']['operator_cost_path'],table)
    coupled_calibration(inputs,table)
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'])
    cost=estimator.estimate(inputs['architectures'][0],inputs['space'])
    metrics=cost.to_candidate_metrics()
    # ``bram`` is clamped to the device tile count, so the pre-fallback request
    # is the only field that still shows a saturated, overflowing design.
    assert metrics.bram==312
    assert metrics.zcu104_requested_bram18_before_fallback>624
    assert metrics.zcu104_predicted_bram18_overflow>0
    assert metrics.whole_network_storage_calibration_validated==1.0
    assert metrics.full_network_route_established==0.0


def test_coupled_bram_allocation_still_enforces_lutram_limit(inputs):
    table=copy.deepcopy(inputs['table'])
    for row in table['rows']: row['bram18_hls']=100
    table=write_hashed(inputs['kwargs']['operator_cost_path'],table)
    calibration={**inputs['calibration'],'operator_cost_payload_sha256':table['payload_sha256'],
        'validated':True,'routed_seed_accepted':True,'heldout_architecture_check_passed':True,
        'model_kind':'coupled_bram_lutram_v1','bram_request_model':dict(slope=1,intercept=200),
        'hls_bram18_domain':[0,2000],'max_calibrated_excess_bram18':1000,
        'overflow_residual_models':{k:dict(slope=200 if k=='lut_as_memory' else 0,intercept=0)
                                   for k in ['lut_as_logic','lut_as_memory','ff','dsp']}}
    write_hashed(inputs['kwargs']['calibration_path'],calibration)
    estimator=ZCU104WidthEOperatorCostEstimator(**inputs['kwargs'])
    cost=estimator.estimate(inputs['architectures'][0],inputs['space'])
    assert cost.total_bram==312
    assert 'lut_as_memory>101760' in cost.violations
