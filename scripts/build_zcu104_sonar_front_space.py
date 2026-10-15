"""Build a separate 720-point successor; preserve the measured 360-point v1."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hwnas_fpga.hardware.width_e_operator_cost import MAPPING,ZCU104WidthEOperatorCostEstimator,architecture_operator_keys,operator_key
from hwnas_fpga.hardware.mobilenetA_operator_cost import load_hashed_json,canonical_sha256
from hwnas_fpga.runtime import build_constraints,build_search_space,build_hardware_spec
from build_zcu104_width_e_search_inputs import enumerate_space,write_hashed

VERSION='zcu104_width_e_sonar_serial_gap_v2'

def main():
    front=ROOT/'results/zcu104_sonar_front_pilot_20260913_v1'
    artifact=ROOT/'artifacts'/VERSION
    _,old=load_hashed_json(ROOT/'artifacts/zcu104_width_e_sonar_serial_gap_v1/operator_cost.json')
    if not old['coverage_complete']: raise ValueError('Finish the 360-point table before extending it')
    manifest=json.loads((front/'manifest.json').read_text())
    record=manifest['operators'][0]
    result_path=front/'remote_output/operators'/f"{record['case_name']}.json"
    result=json.loads(result_path.read_text())
    if result.get('status')!='PASS_OPERATOR_REPORTS' or result['source_sha256']!=record['source_sha256_lf']:
        raise ValueError('Front operator evidence incomplete or stale')
    for rel,expected in result['evidence_sha256'].items():
        path=front/'remote_output/evidence'/record['case_name']/rel
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('Changed front operator report '+str(path))
    r=result['post_route']
    clean=(r['setup_wns_ns']>=0 and r['setup_tns_ns']==0 and r['hold_whs_ns']>=0 and r['hold_ths_ns']==0)
    cfg=yaml.safe_load((ROOT/'configs/search/zcu104_width_e_sonar_serial_gap_v1.yaml').read_text())
    cfg['project'].update(name=VERSION,output_dir='results/'+VERSION)
    cfg['search_space']['search_space_version']=VERSION
    cfg['search_space']['stage_block_choices'][2].insert(0,dict(op='mbconv',kernel_size=3,expand_ratio=3,stride=2))
    for field,name in [('operator_cost_lut_path','operator_cost.json'),('operator_cost_lut_preflight_path','space_preflight.json'),('storage_calibration_path','storage_calibration.json')]:
        cfg['hardware'][field]=str((artifact/name).relative_to(ROOT)).replace('\\','/')
    cfg['admission']['note']='Separate successor adds S3 k3/e3 to leave latency room for sonar slots. Requires new calibration; v1 remains frozen.'
    config_path=ROOT/'configs/search'/f'{VERSION}.yaml'
    config_text=yaml.safe_dump(cfg,sort_keys=False)
    if config_path.exists() and config_path.read_text()!=config_text:
        backup=front/'config_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/config_path.name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(config_path,backup)
    config_path.write_text(config_text)
    resources={k:r[k] for k in ['lut','lut_as_logic','lut_as_memory','ff','dsp']};resources['bram36']=r['block_ram_tile']
    rows=copy.deepcopy(old['rows'])+[dict(key=operator_key('mbconv',24,32,56,2,3,3),case_name=record['case_name'],resources=resources,
        latency_cycles=result['hls']['latency_cycles'],power_w=r['power_w'],timing_met=clean,source_sha256=result['source_sha256'],
        evidence_sha256=result['evidence_sha256'],report_record=str(result_path.relative_to(ROOT)).replace('\\','/'))]
    for row in rows:
        source=json.loads((ROOT/row['report_record']).read_text())
        row['bram18_hls']=source['hls']['bram_18k']
    table=write_hashed(artifact/'operator_cost.json',dict(version=VERSION,mapping_id=MAPPING,search_space=cfg['search_space'],rows=rows,
        required_rows=40,coverage_complete=True,parent_payload_sha256=old['payload_sha256'],
        verified_files=old['verified_files']+len(result['evidence_sha256']),evidence_boundary=old['evidence_boundary']))
    calibration_path=artifact/'storage_calibration.json'
    if calibration_path.exists():
        _,calibration=load_hashed_json(calibration_path)
        if calibration.get('operator_cost_payload_sha256')!=table['payload_sha256']:
            if calibration.get('validated') is True:
                raise ValueError('New operator table needs a new calibration artifact')
            backup=front/'calibration_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/calibration_path.name
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(calibration_path,backup)
            calibration['operator_cost_payload_sha256']=table['payload_sha256']
            calibration=write_hashed(calibration_path,calibration)
    else:
        calibration=write_hashed(calibration_path,dict(version=VERSION,mapping_id=MAPPING,operator_cost_payload_sha256=table['payload_sha256'],
            validated=False,routed_seed_accepted=False,heldout_architecture_check_passed=False,
            status='AWAITING_FRONT_E3_SONAR_WHOLE_NETWORK_FEEDBACK',resource_upper_residuals=None))
    constraints=build_constraints(cfg)
    space=build_search_space(cfg,image_size=224,input_channels=1,num_classes=8,constraints=constraints)
    estimator=ZCU104WidthEOperatorCostEstimator(build_hardware_spec(cfg),constraints,operator_cost_path=artifact/'operator_cost.json',calibration_path=calibration_path,diagnostic_only=True)
    architectures=list(enumerate_space(space));assert len(architectures)==720
    required={k for a in architectures for k in architecture_operator_keys(a,space)}
    assert required=={tuple(r['key']) for r in rows} and len(required)==40
    outcomes=[];sonar={}
    for a in architectures:
        cost=estimator.estimate(a,space,apply_calibration=False)
        b=a.stages[3].blocks[0];label=f'{b.op}_k{b.kernel_size}_e{b.expand_ratio}'
        group=sonar.setdefault(label,dict(count=0,operator_sum_feasible=0,min_latency_ms=float('inf')))
        group['count']+=1;group['operator_sum_feasible']+=not cost.violations;group['min_latency_ms']=min(group['min_latency_ms'],cost.latency_ms)
        row=dict(architecture=a.to_dict(),architecture_sha256=canonical_sha256(a.to_dict()),operator_sum_proxy_feasible=not cost.violations,
                 latency_ms=cost.latency_ms,bram36=cost.total_bram,lut=cost.total_lut,dsp=cost.total_dsp,violations=cost.violations)
        if calibration['validated']:
            corrected=estimator.estimate(a,space)
            row.update(calibrated_proxy_feasible=not corrected.violations,calibrated_violations=corrected.violations,
                       calibration_domain_unknown='storage_overflow_extrapolation_unknown' in corrected.violations,
                       calibrated_bram36=corrected.total_bram,calibrated_lut=corrected.total_lut,
                       calibrated_lut_as_memory=corrected.physical_metrics['zcu104_predicted_lut_as_memory'])
            group['calibrated_feasible']=group.get('calibrated_feasible',0)+int(not corrected.violations)
        outcomes.append(row)
    rng=random.Random(42);sample=[rng.choice(outcomes) for _ in range(1000)]
    report=dict(version=VERSION,mapping_id=MAPPING,legal_architectures=720,ready_operator_rows=40,required_operator_rows=40,
        operator_cost_payload_sha256=table['payload_sha256'],calibration_payload_sha256=calibration['payload_sha256'],unknown_architectures=0,
        operator_sum_proxy_feasible=sum(x['operator_sum_proxy_feasible'] for x in outcomes),sonar_breakdown=sonar,
        calibrated_proxy_feasible=sum(x.get('calibrated_proxy_feasible') is True for x in outcomes) if calibration['validated'] else None,
        calibrated_domain_unknown=sum(x.get('calibration_domain_unknown') is True for x in outcomes) if calibration['validated'] else None,
        sampling=dict(seed=42,draws=1000,with_replacement=True,operator_sum_feasible=sum(x['operator_sum_proxy_feasible'] for x in sample),
                      calibrated_feasible=sum(x.get('calibrated_proxy_feasible') is True for x in sample) if calibration['validated'] else None),
        calibration_validated=calibration['validated'],nas_resume_allowed=estimator.search_ready and any(x.get('calibrated_proxy_feasible') is True for x in outcomes),outcomes=outcomes)
    write_hashed(artifact/'space_preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='outcomes'},indent=2))

if __name__=='__main__': main()
