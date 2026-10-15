"""Bind the 360-point space to exact reports and audit feasibility without NAS."""
import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hwnas_fpga.hardware.width_e_operator_cost import (
    VERSION,MAPPING,ZCU104WidthEOperatorCostEstimator,operator_key,architecture_operator_keys,
)
from hwnas_fpga.hardware.mobilenetA_operator_cost import canonical_sha256
from hwnas_fpga.runtime import build_constraints,build_hardware_spec,build_search_space
from hwnas_fpga.search_space import StageSpec

def write_hashed(path,payload):
    payload={k:v for k,v in payload.items() if k!='payload_sha256'}
    payload['payload_sha256']=canonical_sha256(payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+'\n')
    return payload

def enumerate_space(space):
    base=space.baseline_architecture()
    for widths in itertools.product(*space.config.stage_channel_choices):
        options=[]
        for i,width in enumerate(widths):
            options.append(space.concrete_block_choices_for_stage(stage_index=i,
                in_channels=32 if i==0 else widths[i-1],out_channels=width,stride=space.config.stage_strides[i]))
        for blocks in itertools.product(*options):
            yield replace(base,stages=tuple(StageSpec(channels=c,depth=1,stride=s,blocks=(b,))
                                            for c,s,b in zip(widths,space.config.stage_strides,blocks)))

def make_config(root,extension,artifact):
    cfg=yaml.safe_load((ROOT/'configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml').read_text())
    manifest=json.loads((root/'pilot_manifest.json').read_text())
    seed=next(p for p in manifest['pilots'] if p['pilot_id']=='pilot_c_tail_e1')['architecture']
    ss=cfg['search_space']
    ss.update(search_space_version=VERSION,stage_channel_choices=[[16],[24],[32],[32],[64],[80,96],[112,128],[192,256]],expand_choices=[1,2,3,6])
    sonar=ss['stage_block_choices'][3]
    ss['stage_block_choices']=[copy.deepcopy(stage['blocks']) for stage in seed['stages']]
    ss['stage_block_choices'][3]=sonar
    for i in [6,7]:
        ss['stage_block_choices'][i]=[dict(op='mbconv',kernel_size=3,expand_ratio=e,stride=ss['stage_strides'][i]) for e in [1,2,3]]
    cfg['project'].update(name=VERSION,output_dir='results/'+VERSION)
    # The existing nonempty feasible-pool shortcut samples uniformly and bypasses
    # the controller. Keep the controller sampling path for this new profile.
    cfg['search']['reward_cfg']['hardware_feasible_pool_max_architectures']=0
    hw=cfg['hardware']
    for key in ['operator_cost_lut_coverage_path','operator_cost_lut_preflight_path']: hw.pop(key,None)
    hw.update(estimator='zcu104_width_e_operator_cost_v1',operator_cost_lut_path=str((artifact/'operator_cost.json').relative_to(ROOT)).replace('\\','/'),
              operator_cost_lut_preflight_path=str((artifact/'space_preflight.json').relative_to(ROOT)).replace('\\','/'),
              storage_calibration_path=str((artifact/'storage_calibration.json').relative_to(ROOT)).replace('\\','/'),mapping_id=MAPPING)
    physical=cfg['constraints']['physical']
    for key in list(physical):
        if key.startswith('zcu104_') or key=='route_risk_note': physical.pop(key)
    physical['route_risk_note']='Use exact LUT-as-Logic/LUT-as-Memory and mapping-bound whole-network calibration; 2.01 total-LUT heuristic retired for this profile.'
    cfg['admission']=dict(required=['exact_shape_coverage','whole_network_resource_timing_seed','storage_calibration','architecture_heldout_check'],
                          note='Runtime gates are checked from current hashed artifacts; this YAML does not authorize a NAS run by itself.')
    path=ROOT/'configs/search/zcu104_width_e_sonar_serial_gap_v1.yaml'
    text=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
    if path.exists() and path.read_text()!=text:
        backup=extension/'config_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/path.name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
    path.write_text(text)
    return cfg,path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',type=Path,default=ROOT/'results/zcu104_width_expansion_pilot_20260912_v1')
    parser.add_argument('--extension-root',type=Path,default=ROOT/'results/zcu104_width_e_extension_20260913_v1')
    args=parser.parse_args()
    root=args.pilot_root.resolve();extension=args.extension_root.resolve()
    artifact=ROOT/'artifacts/zcu104_width_e_sonar_serial_gap_v1'
    cfg,config_path=make_config(root,extension,artifact)
    constraints=build_constraints(cfg)
    space=build_search_space(cfg,image_size=224,input_channels=1,num_classes=8,constraints=constraints)
    architectures=list(enumerate_space(space))
    assert len(architectures)==360 and all(not space.validate(a) for a in architectures)
    required={key for a in architectures for key in architecture_operator_keys(a,space)}
    rows={};pending=[];verified_files=0
    for manifest_path,out,is_extension in [(root/'pilot_manifest.json',root/'remote_output',False),(extension/'manifest.json',extension/'remote_output',True)]:
        manifest=json.loads(manifest_path.read_text())
        for record in manifest['operators']:
            p=record['parameters'];kind=p['operator_name'].split('_h')[0]
            if kind=='skip' or (kind=='global_avg_pool' and not is_extension): continue
            key=operator_key(kind,p['in_channels'],p['out_channels'],p['feature_h'],p['stride'],p['kernel_size'],p['expand_ratio'],p.get('apply_relu') if kind in ['stem','pw_conv'] else None)
            if key not in required: continue
            path=out/'operators'/f"{record['case_name']}.json"
            if not path.exists(): pending.append(record['case_name']);continue
            result=json.loads(path.read_text())
            if result.get('status')!='PASS_OPERATOR_REPORTS': pending.append(record['case_name']);continue
            if result['source_sha256']!=record['source_sha256_lf']: raise ValueError('Source identity mismatch '+str(path))
            for rel,expected in result['evidence_sha256'].items():
                evidence=out/'evidence'/record['case_name']/rel
                if hashlib.sha256(evidence.read_bytes()).hexdigest()!=expected: raise ValueError('Evidence hash mismatch '+str(evidence))
                verified_files+=1
            r=result['post_route'];resources={k:r[k] for k in ['lut','lut_as_logic','lut_as_memory','ff','dsp']}
            resources['bram36']=r['block_ram_tile']
            clean=r.get('setup_wns_ns',-1)>=0 and r.get('setup_tns_ns')==0 and r.get('hold_whs_ns',-1)>=0 and r.get('hold_ths_ns')==0
            rows[key]=dict(key=key,case_name=record['case_name'],resources=resources,
                           latency_cycles=result['hls']['latency_cycles'],power_w=r['power_w'],timing_met=clean,
                           source_sha256=result['source_sha256'],evidence_sha256=result['evidence_sha256'],
                           report_record=str(path.relative_to(ROOT)).replace('\\','/'))
    table=write_hashed(artifact/'operator_cost.json',dict(version=VERSION,mapping_id=MAPPING,search_space=cfg['search_space'],
        rows=list(rows.values()),required_rows=len(required),coverage_complete=set(rows)==required,
        missing_keys=[list(k) for k in sorted(required-set(rows),key=str)],verified_files=verified_files,
        evidence_boundary='Exact isolated HLS/OOC route reports, with structural Skip as wire; no board latency/power.'))
    calibration_path=artifact/'storage_calibration.json'
    if calibration_path.exists():
        old=json.loads(calibration_path.read_text())
        if old.get('validated') is True:
            if old.get('operator_cost_payload_sha256')!=table['payload_sha256']:
                raise ValueError('Table changed under validated calibration; create a new calibration version')
            calibration=old
        else: calibration=None
    else: calibration=None
    if calibration is None:
        calibration=write_hashed(calibration_path,dict(version=VERSION,mapping_id=MAPPING,
            operator_cost_payload_sha256=table['payload_sha256'],validated=False,routed_seed_accepted=False,
            heldout_architecture_check_passed=False,resource_upper_residuals=None,
            observed_other_mapping_feedback=str((root/'storage_calibration_feedback.json').relative_to(ROOT)).replace('\\','/'),
            status='AWAITING_SERIAL_GAP_WHOLE_NETWORK_FEEDBACK',
            note='The earlier BRAM/LUTRAM observations use parallel GAP. They explain coupling but cannot calibrate the new GAP mapping.'))
    estimator=ZCU104WidthEOperatorCostEstimator(build_hardware_spec(cfg),constraints,
        operator_cost_path=artifact/'operator_cost.json',calibration_path=calibration_path,diagnostic_only=True)
    outcomes=[]
    for a in architectures:
        unknown=[k for k in architecture_operator_keys(a,space) if k not in rows]
        row=dict(architecture=a.to_dict(),architecture_sha256=canonical_sha256(a.to_dict()),status='UNKNOWN_COST' if unknown else 'COSTED')
        if not unknown:
            cost=estimator.estimate(a,space,apply_calibration=False)
            row.update(operator_sum_proxy_feasible=not cost.violations,violations=cost.violations,
                       lut=cost.total_lut,bram36=cost.total_bram,dsp=cost.total_dsp,latency_ms=cost.latency_ms,
                       lut_as_memory=cost.physical_metrics['zcu104_predicted_lut_as_memory'])
            if calibration['validated']:
                calibrated=estimator.estimate(a,space)
                row.update(calibrated_proxy_feasible=not calibrated.violations,calibrated_violations=calibrated.violations,
                           calibrated_lut=calibrated.total_lut,calibrated_bram36=calibrated.total_bram,
                           calibrated_lut_as_memory=calibrated.physical_metrics['zcu104_predicted_lut_as_memory'])
        outcomes.append(row)
    rng=random.Random(42);sample=[rng.choice(outcomes) for _ in range(1000)]
    report=dict(version=VERSION,mapping_id=MAPPING,config=str(config_path.relative_to(ROOT)),
        operator_cost_payload_sha256=table['payload_sha256'],calibration_payload_sha256=calibration['payload_sha256'],
        legal_architectures=len(outcomes),required_operator_rows=len(required),ready_operator_rows=len(rows),
        unknown_architectures=sum(r['status']=='UNKNOWN_COST' for r in outcomes),
        operator_sum_proxy_feasible=sum(r.get('operator_sum_proxy_feasible') is True for r in outcomes),
        calibrated_proxy_feasible=sum(r.get('calibrated_proxy_feasible') is True for r in outcomes) if calibration['validated'] else None,
        sampling=dict(seed=42,draws=1000,with_replacement=True,operator_sum_feasible=sum(r.get('operator_sum_proxy_feasible') is True for r in sample),
                      calibrated_feasible=sum(r.get('calibrated_proxy_feasible') is True for r in sample) if calibration['validated'] else None,unknown=sum(r['status']=='UNKNOWN_COST' for r in sample)),
        nas_resume_allowed=estimator.search_ready and any(r.get('calibrated_proxy_feasible') is True for r in outcomes),calibration_validated=calibration['validated'],
        note='Unknown is separate from infeasible. Operator-sum feasibility is diagnostic until whole-network calibration and timing seed pass.',outcomes=outcomes)
    write_hashed(artifact/'space_preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='outcomes'},indent=2))

if __name__=='__main__': main()
