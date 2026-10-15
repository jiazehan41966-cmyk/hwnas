"""Calibrate storage overflow on three networks, hold out C/DMDC entirely."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hwnas_fpga.hardware.width_e_operator_cost import MAPPING,RESOURCE_FIELDS,architecture_operator_keys
from hwnas_fpga.hardware.storage_calibration import fit_coupled_storage,predict_coupled_storage
from hwnas_fpga.hardware.mobilenetA_operator_cost import load_hashed_json,canonical_sha256
from hwnas_fpga.runtime import build_constraints,build_search_space
from hwnas_fpga.search_space import ArchitectureSpec
from build_zcu104_width_e_search_inputs import write_hashed
from audit_zcu104_storage_feedback import timing_metrics,route_metrics
from run_zcu104_fifo_storage_control import resource_metrics,resource_violations

VERSION='zcu104_width_e_sonar_serial_gap_v2'
PILOTS=['pilot_c_front_e3_skip','pilot_c_front_e3_dmdc','pilot_c_front_e3_cfem','pilot_b_front_e3_dmdc']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--integration',type=Path,required=True)
    args=parser.parse_args()
    artifact=ROOT/'artifacts'/VERSION
    _,table=load_hashed_json(artifact/'operator_cost.json')
    if table.get('version')!=VERSION or not table.get('coverage_complete'): raise ValueError('The complete v2 table is required')
    cfg=yaml.safe_load((ROOT/'configs/search'/f'{VERSION}.yaml').read_text())
    space=build_search_space(cfg,image_size=224,input_channels=1,num_classes=8,constraints=build_constraints(cfg))
    by_key={tuple(r['key']):r for r in table['rows']}
    samples=[]
    for pilot in PILOTS:
        directory=args.integration/pilot
        acceptance_path=directory/'acceptance.json';acceptance=json.loads(acceptance_path.read_text())
        contract=json.loads((directory/'mapping_contract.json').read_text())
        candidate_path=directory/f'{pilot}_candidate.json'
        if hashlib.sha256(candidate_path.read_bytes()).hexdigest()!=contract['candidate_sha256']: raise ValueError('Changed candidate '+pilot)
        if contract['mapping_id']!=MAPPING or contract.get('postroute_physopt') is not True: raise ValueError('Different mapping/implementation flow')
        for rel,expected in acceptance['evidence_sha256'].items():
            p=directory/'evidence'/rel
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError('Changed whole-network report '+str(p))
        encoding=json.loads(candidate_path.read_text())['encoding']
        selected=[by_key[key] for key in architecture_operator_keys(ArchitectureSpec.from_dict(encoding),space)]
        if contract['gap_source_sha256']!=next(r['source_sha256'] for r in selected if r['key'][0]=='global_avg_pool'):
            raise ValueError('Wrong GAP implementation')
        actual=resource_metrics(directory/'evidence/reports/utilization.rpt')
        synthesis=resource_metrics(directory/'evidence/reports/synth_utilization.rpt')
        timing=timing_metrics(directory/'evidence/reports/timing_summary.rpt');route=route_metrics(directory/'evidence/reports/route_status.rpt')
        conversion=acceptance['bram_to_lutram']
        requested=conversion['requested_bram18'] if conversion['observed'] else synthesis['bram36']*2
        raw={k:sum(r['resources'][k] for r in selected) for k in RESOURCE_FIELDS}
        clean=not resource_violations(actual) and timing['timing_met'] and route['complete'] and timing['no_clock']==0 and timing['unconstrained_internal_endpoints']==0
        samples.append(dict(pilot_id=pilot,architecture_sha256=canonical_sha256(encoding),operator_sum=raw,actual=actual,
            bram18_hls_sum=sum(r['bram18_hls'] for r in selected),requested_bram18=requested,
            requested_bram18_source='explicit_synth_fallback_warning' if conversion['observed'] else 'synthesis_allocation_without_fallback_warning',
            bram_to_lutram=conversion,timing=timing,route=route,core_resource_timing_accepted=clean,
            acceptance_sha256=hashlib.sha256(acceptance_path.read_bytes()).hexdigest()))
    if len({r['architecture_sha256'] for r in samples})!=4: raise ValueError('Architecture split leakage')
    # Fixed before observing the whole-network results: hold out C/DMDC.
    training=[samples[i] for i in [0,2,3]];heldout=samples[1]
    model=fit_coupled_storage(training)
    prediction,domain=predict_coupled_storage(heldout['operator_sum'],heldout['bram18_hls_sum'],model)
    errors={k:prediction[k]-heldout['actual'][k] for k in RESOURCE_FIELDS}
    heldout_pass=not domain['domain_unknown'] and all(v>=-1e-8 for v in errors.values()) and domain['requested_bram18']>=heldout['requested_bram18']-1e-8
    seed_prediction,seed_domain=predict_coupled_storage(samples[0]['operator_sum'],samples[0]['bram18_hls_sum'],model)
    seed_admitted=not seed_domain['domain_unknown'] and not resource_violations(seed_prediction)
    seed_pass=samples[0]['core_resource_timing_accepted']
    # The routed seed is training[0], so seed admission is a fit check, not an
    # independent one. Only the heldout architecture tests generalization.
    seed_is_training_architecture=samples[0]['architecture_sha256'] in {r['architecture_sha256'] for r in training}
    result=dict(version=VERSION,mapping_id=MAPPING,operator_cost_payload_sha256=table['payload_sha256'],**model,
        validated=heldout_pass and seed_pass and seed_admitted,routed_seed_accepted=seed_pass,
        heldout_architecture_check_passed=heldout_pass,calibrated_seed_admitted=seed_admitted,
        seed_is_training_architecture=seed_is_training_architecture,
        independent_architecture_checks=1,training_architecture_count=len(training),
        training_architectures=[r['architecture_sha256'] for r in training],heldout_architectures=[heldout['architecture_sha256']],
        heldout_prediction=prediction,heldout_error=errors,heldout_storage=domain,samples=samples,
        model_description='HLS BRAM18 sum -> upper estimate of pre-fallback demand -> coupled BRAM allocation and LUTRAM/logic/FF/DSP overflow residuals.',
        scope='Three training architectures and one heldout architecture; the routed seed is a training point, so only one architecture is independent. Reject unseen HLS-memory/overflow ranges; empirical proxy, not a per-candidate route certificate.',
        implementation_flow='AltSpreadLogic_high + MoreGlobalIterations + post-route AggressiveExplore + MoreGlobalIterations',
        timing_scope='Per-operator timing and a routed core seed; no whole-network WNS regression.',
        board_io_accepted=False,board_runtime_measured=False,updated_at=datetime.now(timezone.utc).isoformat())
    path=artifact/'storage_calibration.json'
    if path.exists():
        backup=args.integration/'calibration_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/path.name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
    write_hashed(path,result)
    print(json.dumps({k:result[k] for k in ['validated','routed_seed_accepted','heldout_architecture_check_passed','calibrated_seed_admitted','bram_request_model','overflow_residual_models','heldout_error']},indent=2))

if __name__=='__main__': main()
