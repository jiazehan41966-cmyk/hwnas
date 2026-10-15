"""Archive coupled BRAM/LUTRAM feedback without equating bitgen with acceptance."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from run_zcu104_fifo_storage_control import resource_metrics, resource_violations

def timing_metrics(path):
    text=path.read_text(errors='replace')
    summary=text.split('| Design Timing Summary',1)[1]
    match=re.search(r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$',summary,re.M)
    if not match: raise ValueError('Missing complete setup/hold/pulse timing summary')
    names=['wns_ns','tns_ns','setup_failing_endpoints','setup_total_endpoints','whs_ns','ths_ns',
           'hold_failing_endpoints','hold_total_endpoints','wpws_ns','tpws_ns','pulse_failing_endpoints','pulse_total_endpoints']
    metrics=dict(zip(names,map(float,match.groups())))
    metrics['timing_met']=all(metrics[k]>=0 for k in ['wns_ns','whs_ns','wpws_ns']) and all(metrics[k]==0 for k in ['tns_ns','ths_ns','tpws_ns','setup_failing_endpoints','hold_failing_endpoints','pulse_failing_endpoints'])
    for name in ['no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay']:
        found=re.search(r'checking '+name+r' \((\d+)\)',text)
        if not found: raise ValueError('Missing check_timing '+name)
        metrics[name]=int(found[1])
    return metrics

def route_metrics(path):
    text=path.read_text(errors='replace')
    values={}
    for key,label in [('routable','routable nets'),('fully_routed','fully routed nets'),('errors','nets with routing errors')]:
        match=re.search(r'# of '+label+r'\.+\s*:\s*(\d+)',text)
        if not match: raise ValueError('Missing route status '+label)
        values[key]=int(match[1])
    values['complete']=values['routable']>0 and values['routable']==values['fully_routed'] and values['errors']==0
    return values

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'pilot_manifest.json').read_text())
    pilot=next(p for p in manifest['pilots'] if p['pilot_id']=='pilot_c_tail_e1')
    records=[json.loads((root/'remote_output/operators'/f'{n}.json').read_text()) for n in pilot['case_names'] if not n.startswith('skip_')]
    sums={k:sum(r['post_route'][k] for r in records) for k in ['lut','lut_as_logic','lut_as_memory','ff','dsp','block_ram_tile']}
    sums['bram36']=sums.pop('block_ram_tile')
    original=root/'remote_integration_output_v2/pilot_c_tail_e1'
    control=root/'remote_fifo_storage_control'
    rows=[]
    for mapping,util,timing,synth,route in [
        ('independent_blocks_gap_parallel_fifo_bram_d64',original/'vivado_reports/utilization.rpt',original/'vivado_reports/timing_summary.rpt',original/'vivado_reports/synth_runme.log',original/'vivado_reports/harness_top_route_status_routed.rpt'),
        ('independent_blocks_gap_parallel_fifo_sync_distributed_d64',control/'evidence/harness_project/reports/utilization.rpt',control/'evidence/harness_project/reports/timing_summary.rpt',control/'synth_vivado.log',control/'routed_diagnostics/route_status.rpt')]:
        actual=resource_metrics(util)
        tm=timing_metrics(timing)
        rm=route_metrics(route) if route.exists() else {'complete':None,'status':'MISSING_ROUTE_STATUS'}
        spill=re.search(r'Resources of type BRAM have been overutilized\. Used = (\d+), Available = (\d+)',synth.read_text(errors='replace'))
        violations=resource_violations(actual)
        if not tm['timing_met']: violations.append('timing_not_met')
        if rm['complete'] is not True: violations.append('route_completion_unverified')
        rows.append(dict(mapping_id=mapping,architecture=pilot['architecture'],architecture_id=pilot['pilot_id'],
            clock_mhz=200,evidence_stage='whole_network_routed',operator_sum=sums,actual=actual,
            residual={k:actual[k]-v for k,v in sums.items()},timing=tm,route=rm,
            bram_to_lutram=dict(observed=spill is not None,
                requested_bram18_before_fallback=int(spill[1]) if spill else None,
                available_bram18=int(spill[2]) if spill else None,
                requested_excess_bram18=int(spill[1])-int(spill[2]) if spill else None,
                note='Requested excess is not the count of primitives ultimately converted.'),
            resource_timing_accepted=not violations,violations=violations,
            board_io_accepted=False,board_runtime_measured=False,
            source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [util,timing,synth,route] if p.exists()}))
    result=dict(schema_version=1,samples=rows,independent_architectures=1,mapping_controls=2,
        calibration_status='OBSERVED_COUPLED_STORAGE_FEEDBACK_NOT_GENERALIZED',
        training_split_policy='Group by architecture before holdout; mappings of one architecture are not independent validation samples.',
        predictor_inputs=['operator_sum_lut_logic','operator_sum_lut_memory','operator_sum_bram36','operator_sum_ff','operator_sum_dsp','per_layer_width','expansion','parameter_storage','fifo_width_depth_style','gap_implementation','clock_mhz'],
        predictor_targets=['lut_as_logic','lut_as_memory','bram36','ff','dsp','wns_ns','timing_met'],
        calibration_validated=False,feasible_baselines=sum(r['resource_timing_accepted'] for r in rows),nas_resume_allowed=False,
        io_boundary='The legacy harness has intentionally unconstrained board IO; these are core integration experiments, not board-deployable platform acceptance.')
    dest=root/'storage_calibration_feedback.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(dest),'samples':[{k:r[k] for k in ['mapping_id','actual','timing','bram_to_lutram','violations']} for r in rows]},indent=2))

if __name__=='__main__': main()
