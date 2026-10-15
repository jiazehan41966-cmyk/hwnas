"""Finite, evidence-gated whole-network queue for the serial GAP mapping.

Uses three frozen pilot architectures; no NAS, training, programming, or board IO.
The first architecture must pass core route/timing before calibration points run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from run_zcu104_fifo_storage_control import resource_metrics, resource_violations
from audit_zcu104_storage_feedback import timing_metrics, route_metrics

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def run(command,log,timeout=14400,cwd=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True,cwd=cwd)
        try: code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    if code: raise RuntimeError(f'Command exited {code}: {log}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-base',type=Path,required=True)
    parser.add_argument('--reference-bundle',type=Path,required=True)
    parser.add_argument('--extension',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--pilots',nargs='+',default=['pilot_c_tail_e1','pilot_c_tail_e2','pilot_b_tail_e1'])
    parser.add_argument('--postroute-opt',action='store_true')
    args=parser.parse_args()
    out=args.extension/'integration'
    if (out/'status.json').exists(): raise FileExistsError('Use a new integration output to preserve the previous run')
    out.mkdir(parents=True,exist_ok=True)
    state=dict(status='WAITING_SERIAL_GAP_OPERATOR',started_at=datetime.now(timezone.utc).isoformat(),
               pid=os.getpid(),nas_resume_allowed=False,core_feasible_seed=None,
               mapping_id='independent_blocks_pi4_po4_u1_int8_serial_gap_sync_distributed_fifo_d64_200mhz',
               postroute_physopt=args.postroute_opt,
               board_io_accepted=False,board_runtime_measured=False,runs=[])
    write(out/'status.json',state)
    try:
        em=json.loads((args.extension/'manifest.json').read_text())
        gap=next(r for r in em['operators'] if r['operator_name'].startswith('global_avg_pool_'))
        record_path=args.extension/'output/operators'/f"{gap['case_name']}.json"
        deadline=time.monotonic()+7200
        while not record_path.exists():
            if time.monotonic()>deadline: raise TimeoutError('Serial GAP operator did not finish within 120 minutes')
            time.sleep(10)
        record=json.loads(record_path.read_text())
        if record.get('status')!='PASS_OPERATOR_REPORTS' or not record.get('timing_met'):
            raise ValueError('Serial GAP operator failed report/timing gate')
        witness=json.loads((args.extension/'output/gap_witness/status.json').read_text())
        template=args.extension/'bundle/hls_lut_builder/templates/global_avg_pool_serial.cpp.tmpl'
        if witness['status']!='PASS' or witness['template_sha256']!=sha(template):
            raise ValueError('Serial GAP numerical witness stale')
        bundle=args.work/'bundle'
        shutil.copytree(args.reference_bundle,bundle,ignore=shutil.ignore_patterns('_vivado_builds','.Xil','__pycache__','*.log','*.jou'))
        # Only measured GAP implementation changes. All other operator projects
        # remain the exact same immutable cached sources used by the pilot.
        manifest=json.loads((bundle/'pilot_manifest.json').read_text())
        manifest['operators']=[gap if r['operator_name'].startswith('global_avg_pool_') else r for r in manifest['operators']]
        write(bundle/'pilot_manifest.json',manifest)
        combined=args.work/'combined_output/operators';combined.mkdir(parents=True)
        for path in (args.reference_base/'output/operators').glob('*.json'):
            source=record_path if path.name==record_path.name else path
            (combined/path.name).symlink_to(source)
        original_script=Path(__file__).with_name('run_zcu104_pilot_integration.py')
        for pilot_id in args.pilots:
            dest=out/pilot_id;dest.mkdir()
            work=args.work/pilot_id;work.mkdir()
            row=dict(pilot_id=pilot_id,status='GENERATING',board_io_accepted=False,board_runtime_measured=False)
            state['runs'].append(row);state['status']='RUNNING_'+pilot_id;write(out/'status.json',state)
            generate=[sys.executable,str(original_script),'--bundle',str(bundle),'--operator-output',str(combined.parent),
                      '--output',str(dest),'--work',str(work),'--pilot',pilot_id,'--generate-only']
            row['generate_command']=generate
            run(generate,dest/'generate_driver.log',timeout=600,cwd=bundle)
            project=work/'integration'/pilot_id/'harness_project'
            fifo=project/'rtl/axis_simple_fifo.v'
            text=fifo.read_text()
            if text.count('ram_style = "block"')!=1: raise ValueError('Unexpected synchronous FIFO implementation')
            original_sha=sha(fifo)
            fifo.write_text(text.replace('ram_style = "block"','ram_style = "distributed"'))
            contract=dict(fifo_original_sha256=original_sha,fifo_sha256=sha(fifo),fifo_depth=64,
                fifo_style='sync_distributed_same_control_as_bram',gap_source_sha256=record['source_sha256'],
                gap_template_sha256=sha(template),candidate_sha256=sha(dest/f'{pilot_id}_candidate.json'),
                mapping_id=state['mapping_id'],rtl_sha256={str(p.relative_to(project)):sha(p) for directory in ['rtl','export_rtl'] for p in (project/directory).glob('*.v')})
            contract['postroute_physopt']=args.postroute_opt
            write(dest/'mapping_contract.json',contract)
            tcl=(project/'scripts/build_bitstream.tcl').read_text()
            marker='launch_runs impl_1 -to_step route_design -jobs 8'
            if tcl.count(marker)!=1: raise ValueError('Unexpected implementation TCL')
            prefix,tail=tcl.split(marker)
            synth=project/'scripts/serial_gap_synth.tcl'
            synth.write_text(prefix+'\nopen_run synth_1\nfile mkdir $reports_output_dir\nreport_utilization -file ${reports_output_dir}/synth_utilization.rpt\nreport_utilization -hierarchical -hierarchical_depth 2 -file ${reports_output_dir}/synth_hier.rpt\nexit\n')
            row['status']='SYNTHESIS';write(out/'status.json',state)
            vivado='/opt/Xilinx/Vivado/2022.2/bin/vivado'
            run([vivado,'-mode','batch','-source',str(synth),'-log',str(dest/'synth_vivado.log'),'-journal',str(dest/'synth_vivado.jou')],dest/'synth_driver.log',cwd=bundle)
            row['synthesis_resources']=resource_metrics(project/'reports/synth_utilization.rpt')
            row['violations']=resource_violations(row['synthesis_resources'])
            row['status']='SYNTHESIS_RESOURCE_REJECTED' if row['violations'] else 'ROUTING'
            write(out/'status.json',state)
            if not row['violations']:
                definitions=tcl.split('create_project',1)[0]
                # This is a core integration acceptance run; do not generate a
                # board bitstream with intentionally unconstrained legacy IO.
                tail=tail.split('write_bitstream',1)[0]
                if args.postroute_opt:
                    tail=tail.replace('report_timing_summary -file',
                        'phys_opt_design -directive AggressiveExplore\nroute_design -directive MoreGlobalIterations\nreport_timing_summary -file',1)
                route=project/'scripts/serial_gap_route.tcl'
                route.write_text(definitions+'\nopen_project ${vivado_project_dir}/${project_name}.xpr\n'+marker+tail+
                    '\nreport_route_status -file ${reports_output_dir}/route_status.rpt\nreport_utilization -hierarchical -hierarchical_depth 2 -file ${reports_output_dir}/route_hier.rpt\nreport_timing -max_paths 30 -file ${reports_output_dir}/setup_paths.rpt\nwrite_checkpoint -force ${reports_output_dir}/accepted_candidate_routed.dcp\nexit\n')
                run([vivado,'-mode','batch','-source',str(route),'-log',str(dest/'route_vivado.log'),'-journal',str(dest/'route_vivado.jou')],dest/'route_driver.log',cwd=bundle)
                row['actual']=resource_metrics(project/'reports/utilization.rpt')
                row['timing']=timing_metrics(project/'reports/timing_summary.rpt')
                row['route']=route_metrics(project/'reports/route_status.rpt')
                row['violations']=resource_violations(row['actual'])
                if not row['timing']['timing_met']: row['violations'].append('timing_not_met')
                if not row['route']['complete']: row['violations'].append('route_incomplete')
                for key in ['no_clock','unconstrained_internal_endpoints']:
                    if row['timing'][key]!=0: row['violations'].append(key)
                row['status']='CORE_RESOURCE_TIMING_PASS' if not row['violations'] else 'CORE_RESOURCE_TIMING_REJECTED'
            row['accepted']=not row['violations']
            spill=re.search(r'Resources of type BRAM have been overutilized\. Used = (\d+), Available = (\d+)',(dest/'synth_vivado.log').read_text(errors='replace'))
            row['bram_to_lutram']=dict(observed=spill is not None,requested_bram18=int(spill[1]) if spill else None,available_bram18=int(spill[2]) if spill else None)
            row['candidate_sha256']=contract['candidate_sha256']
            row['evidence_sha256']={}
            for source in project.rglob('*'):
                if source.is_file() and source.suffix in ['.rpt','.tcl','.json','.xdc']:
                    target=dest/'evidence'/source.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target);row['evidence_sha256'][str(source.relative_to(project))]=sha(target)
            write(dest/'acceptance.json',row)
            if pilot_id==args.pilots[0]:
                if row['accepted']: state['core_feasible_seed']=pilot_id
                else:
                    state['status']='SEED_REJECTED_CALIBRATION_QUEUE_NOT_RUN'
                    break
        else: state['status']='CALIBRATION_POINTS_FINISHED_PENDING_MODEL_AUDIT'
    except Exception as error:
        state.update(status='ERROR',error=str(error));raise
    finally:
        state['updated_at']=datetime.now(timezone.utc).isoformat();write(out/'status.json',state)

if __name__=='__main__': main()
