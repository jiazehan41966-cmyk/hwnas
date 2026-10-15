"""Fit a conservative resource-residual envelope on architecture-disjoint points.

This small pilot calibrates resource aggregation, not a learned timing predictor.
All residuals include the net effect of Vivado's BRAM-to-LUTRAM conversion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hwnas_fpga.hardware.width_e_operator_cost import VERSION,MAPPING,RESOURCE_FIELDS,architecture_operator_keys
from hwnas_fpga.hardware.mobilenetA_operator_cost import load_hashed_json,canonical_sha256
from hwnas_fpga.runtime import build_constraints,build_search_space
from hwnas_fpga.search_space import ArchitectureSpec
from build_zcu104_width_e_search_inputs import write_hashed
from audit_zcu104_storage_feedback import timing_metrics,route_metrics
from run_zcu104_fifo_storage_control import resource_metrics,resource_violations

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--integration',type=Path,required=True)
    parser.add_argument('--artifact-root',type=Path,default=ROOT/'artifacts/zcu104_width_e_sonar_serial_gap_v1')
    args=parser.parse_args()
    artifact=args.artifact_root.resolve()
    _,table=load_hashed_json(artifact/'operator_cost.json')
    if table.get('coverage_complete') is not True: raise ValueError('Complete exact costs before fitting calibration')
    cfg=yaml.safe_load((ROOT/'configs/search/zcu104_width_e_sonar_serial_gap_v1.yaml').read_text())
    space=build_search_space(cfg,image_size=224,input_channels=1,num_classes=8,constraints=build_constraints(cfg))
    by_key={tuple(r['key']):r for r in table['rows']}
    samples=[]
    for pilot in ['pilot_c_tail_e1','pilot_c_tail_e2','pilot_b_tail_e1']:
        directory=args.integration/pilot
        acceptance=json.loads((directory/'acceptance.json').read_text())
        contract=json.loads((directory/'mapping_contract.json').read_text())
        candidate_path=directory/f'{pilot}_candidate.json'
        if hashlib.sha256(candidate_path.read_bytes()).hexdigest()!=contract['candidate_sha256']:
            raise ValueError('Candidate identity mismatch '+pilot)
        if contract['mapping_id']!=MAPPING: raise ValueError('Cannot pool different GAP/FIFO mappings')
        for rel,expected in acceptance['evidence_sha256'].items():
            path=directory/'evidence'/rel
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('Changed whole-network evidence '+str(path))
        candidate=json.loads(candidate_path.read_text())
        architecture=ArchitectureSpec.from_dict(candidate['encoding'])
        selected=[by_key[k] for k in architecture_operator_keys(architecture,space)]
        gap=next(r for r in selected if r['key'][0]=='global_avg_pool')
        if contract['gap_source_sha256']!=gap['source_sha256']: raise ValueError('GAP cost and integrated source differ')
        # Parse raw reports again; never trust a process-success status as a label.
        actual=resource_metrics(directory/'evidence/reports/utilization.rpt')
        timing=timing_metrics(directory/'evidence/reports/timing_summary.rpt')
        route=route_metrics(directory/'evidence/reports/route_status.rpt')
        raw={k:sum(r['resources'][k] for r in selected) for k in RESOURCE_FIELDS}
        accepted=(not resource_violations(actual) and timing['timing_met'] and route['complete']
                  and timing['no_clock']==0 and timing['unconstrained_internal_endpoints']==0)
        samples.append(dict(pilot_id=pilot,architecture_sha256=canonical_sha256(candidate['encoding']),operator_sum=raw,
            actual=actual,residual={k:actual[k]-raw[k] for k in RESOURCE_FIELDS},timing=timing,route=route,
            core_resource_timing_accepted=accepted,bram_to_lutram=acceptance['bram_to_lutram'],
            acceptance_sha256=hashlib.sha256((directory/'acceptance.json').read_bytes()).hexdigest()))
    if len({r['architecture_sha256'] for r in samples})!=3: raise ValueError('Holdout leakage: duplicate architecture')
    train=samples[:2];heldout=samples[2]
    corrections={k:max(0,max(r['residual'][k] for r in train)) for k in RESOURCE_FIELDS}
    corrections['lut']=corrections['lut_as_logic']+corrections['lut_as_memory']
    predicted={k:heldout['operator_sum'][k]+corrections[k] for k in RESOURCE_FIELDS}
    predicted['lut']=predicted['lut_as_logic']+predicted['lut_as_memory']
    errors={k:predicted[k]-heldout['actual'][k] for k in RESOURCE_FIELDS}
    heldout_pass=all(v>=0 for v in errors.values())
    seed_pass=samples[0]['core_resource_timing_accepted']
    calibrated_seed={k:samples[0]['operator_sum'][k]+corrections[k] for k in RESOURCE_FIELDS}
    calibrated_seed['lut']=calibrated_seed['lut_as_logic']+calibrated_seed['lut_as_memory']
    seed_admitted=not resource_violations(calibrated_seed)
    result=dict(version=VERSION,mapping_id=MAPPING,operator_cost_payload_sha256=table['payload_sha256'],
        validated=heldout_pass and seed_pass and seed_admitted,routed_seed_accepted=seed_pass,
        heldout_architecture_check_passed=heldout_pass,calibrated_seed_admitted=seed_admitted,
        resource_upper_residuals=corrections,training_architectures=[r['architecture_sha256'] for r in train],
        heldout_architectures=[heldout['architecture_sha256']],heldout_prediction=predicted,heldout_error=errors,
        samples=samples,model='nonnegative per-resource maximum integration residual; LUT = logic + memory',
        conversion_model='Coupled storage effects are observed in per-resource residuals, including pre-fallback BRAM demand and post-route LUTRAM. No fixed total-LUT-to-LUTRAM multiplier.',
        scope='Two training architectures and one heldout architecture at one mapping. Empirical conservative proxy, not a physical feasibility certificate for all candidates.',
        timing_scope='Routed seed plus per-operator timing gates; no trained whole-network WNS regressor.',
        board_io_accepted=False,board_runtime_measured=False,updated_at=datetime.now(timezone.utc).isoformat())
    path=artifact/'storage_calibration.json'
    if path.exists():
        backup=args.integration/'calibration_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/path.name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
    write_hashed(path,result)
    print(json.dumps({k:result[k] for k in ['validated','routed_seed_accepted','heldout_architecture_check_passed','calibrated_seed_admitted','resource_upper_residuals','heldout_error']},indent=2))

if __name__=='__main__': main()
