"""One controlled post-route physical optimization, preserving RTL and clock."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from run_zcu104_serial_gap_integration import run,write,sha
from run_zcu104_fifo_storage_control import resource_metrics,resource_violations
from audit_zcu104_storage_feedback import timing_metrics,route_metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-result',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--source-status',default='acceptance.json')
    parser.add_argument('--mode',choices=['aggressive','fifo_read_replication'],default='aggressive')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True);args.work.mkdir(parents=True,exist_ok=True)
    state=dict(status='WAITING_SOURCE_ACCEPTANCE',started_at=datetime.now(timezone.utc).isoformat(),nas_resume_allowed=False,
               rtl_changed=False,clock_mhz=200,board_io_accepted=False,board_runtime_measured=False,
               implementation_flow='frozen source route -> '+args.mode+' -> route MoreGlobalIterations')
    status=args.output/'status.json';write(status,state)
    try:
        deadline=time.monotonic()+1800
        acceptance_path=args.source_result/args.source_status
        while True:
            acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
            if acceptance.get('status') in ['CORE_RESOURCE_TIMING_PASS','CORE_RESOURCE_TIMING_REJECTED','ERROR']:
                break
            if time.monotonic()>deadline: raise TimeoutError('Source route reports did not finish')
            time.sleep(10)
        if not acceptance.get('actual') or resource_violations(acceptance['actual']):
            raise ValueError('Timing-only repair requires source routed resources inside budget')
        if acceptance['accepted']:
            state['status']='SOURCE_ALREADY_ACCEPTED_NO_REPAIR_NEEDED';return
        if not acceptance['route']['complete']: raise ValueError('Source route is incomplete')
        state.update(source_checkpoint_sha256=sha(args.checkpoint),source_acceptance_sha256=sha(acceptance_path),
                     candidate_sha256=acceptance['candidate_sha256'],before=acceptance['timing'],status='RUNNING_POSTROUTE_PHYSOPT')
        tcl=args.output/'repair.tcl'
        def report_lines():
            return '\n'.join([f'report_timing_summary -file {{{args.output}/timing_summary.rpt}}',
                f'report_utilization -file {{{args.output}/utilization.rpt}}',
                f'report_route_status -file {{{args.output}/route_status.rpt}}',
                f'report_timing -max_paths 30 -file {{{args.output}/setup_paths.rpt}}'])
        physical_opt='phys_opt_design -directive AggressiveExplore\n'
        if args.mode=='fifo_read_replication':
            physical_opt='''set read_pins [get_pins -hierarchical -filter {NAME =~ u_e2e_fifo_*/rd_ptr_reg*/Q}]
set selected_nets {}
foreach net [get_nets -of_objects $read_pins] {
    set fanout [llength [get_pins -of_objects $net -filter {DIRECTION == IN}]]
    if {$fanout > 512} {
        puts "FIFO_READ_REPLICATION $net fanout=$fanout"
        lappend selected_nets $net
    }
}
if {[llength $selected_nets] == 0} {error "No expected high-fanout FIFO read-address nets found"}
phys_opt_design -force_replication_on_nets $selected_nets
'''
        tcl.write_text(f'set_param general.maxThreads 8\nopen_checkpoint {{{args.checkpoint}}}\n'
            f'report_timing -max_paths 30 -file {{{args.output}/before_setup_paths.rpt}}\n'
            +physical_opt+'route_design -directive MoreGlobalIterations\n'+report_lines()+
            f'\nwrite_checkpoint -force {{{args.work}/after_postroute_physopt.dcp}}\nexit\n')
        state['command']=['/opt/Xilinx/Vivado/2022.2/bin/vivado','-mode','batch','-source',str(tcl),'-log',str(args.output/'vivado.log'),'-journal',str(args.output/'vivado.jou')]
        state['tcl_sha256']=sha(tcl);write(status,state)
        run(state['command'],args.output/'driver.log',timeout=7200,cwd=args.work)
        actual=resource_metrics(args.output/'utilization.rpt');timing=timing_metrics(args.output/'timing_summary.rpt');route=route_metrics(args.output/'route_status.rpt')
        violations=resource_violations(actual)
        if not timing['timing_met']: violations.append('timing_not_met')
        if not route['complete']: violations.append('route_incomplete')
        for key in ['no_clock','unconstrained_internal_endpoints']:
            if timing[key]: violations.append(key)
        state.update(status='CORE_RESOURCE_TIMING_PASS' if not violations else 'CORE_RESOURCE_TIMING_REJECTED',
                     accepted=not violations,actual=actual,timing=timing,route=route,violations=violations,
                     result_checkpoint_sha256=sha(args.work/'after_postroute_physopt.dcp'))
    except Exception as error:
        state.update(status='ERROR',error=str(error));raise
    finally:
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),evidence_sha256={p.name:sha(p) for p in args.output.glob('*.rpt')})
        write(status,state)

if __name__=='__main__': main()
