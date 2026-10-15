"""Run a frozen operator pilot through existing HLS/route tools, without NAS.

The bundle is its own repository root. JSON outputs are resumable and fail closed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'hls_lut_builder/scripts'),str(ROOT/'scripts')]
from common import build_cases, build_tcl, load_config, render_template
from run_vivado_downstream import parse_downstream_metrics
from build_zcu104_mobilenetA_full_formal_lut_from_reports import parse_hls_xml


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized_digest(path): return hashlib.sha256(Path(path).read_text().replace('\r\n','\n').encode()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)
def now(): return datetime.now(timezone.utc).isoformat()


def parse_storage(report: Path):
    text=report.read_text(errors='replace')
    def field(label):
        match=re.search(r'\|\s*'+label+r'\*?\s*\|\s*([\d,.]+)\s*\|',text,re.I)
        return float(match.group(1).replace(',','')) if match else None
    return {'lut_as_logic':field('LUT as Logic'), 'lut_as_memory':field('LUT as Memory')}


def old_source_check(case, record):
    row=record.get('legacy_row')
    if not row: return None, {'reason':'no exact-shape historical row'}
    report=Path(row['hls_report_path'])
    old_root=report.parents[4]
    source=old_root/'src'/case.source_filename
    expected_source=render_template(case.template_path.read_text(),{'top_function':case.top_function,**case.parameters})
    expected_hash=hashlib.sha256(expected_source.replace('\r\n','\n').encode()).hexdigest()
    checks={'source':source.exists() and normalized_digest(source)==expected_hash,
            'hls_report':report.exists() and digest(report)==row.get('hls_report_sha256'),
            'tcl':(old_root/'synth.tcl').exists() and (old_root/'synth.tcl').read_text().replace('\r\n','\n')==build_tcl(case)}
    for header in (ROOT/'hls_lut_builder/include').glob('*.h'):
        other=old_root/'include'/header.name
        checks['header:'+header.name]=other.exists() and digest(header)==digest(other)
    route_dir=Path(row['route_report_dir'])
    for name in ['post_route_utilization.rpt','post_route_timing_summary.rpt']:
        checks[name]=(route_dir/name).exists()
    return (old_root if all(checks.values()) else None), checks


def collect(case,record,root:Path,mode:str,checks:dict,out:Path):
    from dataclasses import replace
    selected=replace(case,project_dir=root)
    reports=selected.vivado_downstream_reports_dir
    metrics=parse_downstream_metrics(selected)
    if not metrics: return {'case_name':case.case_name,'status':'MISSING_ROUTE_METRICS'}
    util=reports/'post_route_utilization.rpt'
    hls_report=root/'project'/case.solution_name/'syn/report'/f'{case.top_function}_csynth.xml'
    if not hls_report.exists(): return {'case_name':case.case_name,'status':'MISSING_HLS_REPORT'}
    hls=parse_hls_xml(hls_report,200)
    routed={**metrics['post_route'],**parse_storage(util)}
    required=['lut','ff','dsp','block_ram_tile','lut_as_logic','lut_as_memory','setup_wns_ns','setup_tns_ns','power_w']
    missing=[key for key in required if routed.get(key) is None]
    if hls.get('latency_cycles') is None: missing.append('hls_latency_cycles')
    evidence=out/'evidence'/case.case_name
    hashes={}
    for source in [*reports.glob('*.rpt'),hls_report,* (root/'src').glob('*.cpp'),*(root/'include').glob('*.h'),root/'synth.tcl']:
        dest=evidence/source.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        hashes[str(source.relative_to(root))]=digest(dest)
    return {'case_name':case.case_name,'status':'MISSING_FIELDS' if missing else 'PASS_OPERATOR_REPORTS',
        'missing_fields':missing,'evidence_mode':mode,'source_match_checks':checks,'source_root':str(root),
        'source_sha256':record['source_sha256_lf'],'hls':hls,'post_route':routed,'evidence_sha256':hashes,
        'timing_met':routed.get('setup_wns_ns') is not None and routed['setup_wns_ns']>=0 and routed.get('setup_tns_ns')==0,
        'full_network_route':False,'board_latency_measured':False}


def aggregate(manifest, records, out):
    by_name={r['case_name']:r for r in records}
    pilots=[]
    for p in manifest['pilots']:
        rr=[by_name.get(n) for n in p['case_names']]
        complete=all(r and r.get('status')=='PASS_OPERATOR_REPORTS' for r in rr)
        row={'pilot_id':p['pilot_id'],'channels':p['channels'],'tail_expansion':p['tail_expansion'],
             'complete':complete,'missing_cases':[n for n,r in zip(p['case_names'],rr) if not r or r.get('status')!='PASS_OPERATOR_REPORTS'],
             'proxy_feasible':None,'implementation_feasible':None}
        if complete:
            total={k:sum(r['post_route'][k] for r in rr) for k in ['lut','lut_as_logic','lut_as_memory','ff','dsp','block_ram_tile']}
            total['bram36']=math.ceil(total.pop('block_ram_tile'))
            total['latency_ms']=sum(r['hls']['latency_cycles'] for r in rr)/200000
            total['energy_mj_proxy']=sum(r['hls']['latency_cycles']/200000*r['post_route']['power_w'] for r in rr)
            total['power_w_proxy']=max(r['post_route']['power_w'] for r in rr)
            total['legacy_lutram_risk_only']=2.01*total['lut']/101760
            limits={'bram36':312,'lut':161280,'ff':460800,'dsp':1209,'lut_as_memory':101760,'latency_ms':120,'energy_mj_proxy':375,'power_w_proxy':15}
            violations=[f'{k}>{v}' for k,v in limits.items() if total[k]>v]
            if not all(r['timing_met'] for r in rr): violations.append('operator timing not met')
            row.update(operator_sum=total,violations=violations,proxy_feasible=not violations)
        pilots.append(row)
    write(out/'pilot_results.json',{'updated_at':now(),'pilots':pilots,'complete_count':sum(p['complete'] for p in pilots),
        'proxy_feasible_count':sum(p['proxy_feasible'] is True for p in pilots),
        'full_network_feasible_seed':None,'nas_resume_allowed':False,
        'boundary':'Matched operator reports only. Resource sum omits integration overhead; full-network implementation required.'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    work=args.work_root.resolve();work.mkdir(parents=True,exist_ok=True)
    config=args.config or ROOT/'hls_lut_builder/configs/width_expansion_pilot.yaml'
    cases=build_cases(load_config(config),project_root_override=work/'p')
    cmap={c.case_name:c for c in cases}
    rmap={r['case_name']:r for r in manifest['operators']}
    witness=json.loads((out/'e1_witness/status.json').read_text())
    if witness['status']!='PASS' or witness['template_sha256']!=digest(ROOT/'hls_lut_builder/templates/mbconv_e1_noexpand.cpp.tmpl'):
        raise RuntimeError('e1 numeric witness missing or stale')
    if any(c.template_path.name=='global_avg_pool_serial.cpp.tmpl' for c in cases):
        gap_witness=json.loads((out/'gap_witness/status.json').read_text())
        if gap_witness['status']!='PASS' or gap_witness['template_sha256']!=digest(ROOT/'hls_lut_builder/templates/global_avg_pool_serial.cpp.tmpl'):
            raise RuntimeError('serial GAP numeric witness missing or stale')
    state={'started_at':now(),'pid':os.getpid(),'status':'RUNNING','completed':0,'total':len(cases),'nas_resume_allowed':False}
    write(out/'queue_status.json',state)
    ordered=[]
    # First get a light candidate complete, then fill the remaining factorial points.
    priority=sorted(manifest['pilots'],key=lambda p:(p['pilot_id']!='pilot_c_tail_e1',p['tail_expansion'],p['channels'][-1]))
    for p in priority:
        for n in p['case_names']:
            if n not in ordered:ordered.append(n)
    ordered.extend(n for n in cmap if n not in ordered)

    def execute(name):
        case=cmap[name];record=rmap[name];saved=out/'operators'/f'{name}.json'
        if saved.exists():
            previous=json.loads(saved.read_text())
            if previous.get('status')=='PASS_OPERATOR_REPORTS' and previous.get('source_sha256')==record['source_sha256_lf']:
                return previous
        if shutil.disk_usage(work).free < 8*1024**3:
            result={'case_name':name,'status':'BLOCKED_WORKSPACE_SPACE'}
            write(saved,result);return result
        old,checks=old_source_check(case,record)
        if old:
            result=collect(case,record,old,'EXACT_SOURCE_REUSE',checks,out)
        else:
            command=[sys.executable,str(ROOT/'hls_lut_builder/scripts/run_synthesis.py'),'--config',str(config),
                '--project-root',str(work/'p'),'--case',name,'--downstream-check','--timeout-minutes','90',
                '--downstream-timeout-minutes','90','--summary-json',str(out/'summaries'/f'{name}.json')]
            log=out/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
            write(out/'active'/f'{name}.json',{'started_at':now(),'command':command,'status':'RUNNING'})
            with log.open('w') as f:
                process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            statusfile=case.project_dir/'synthesis_status.json'
            result=collect(case,record,case.project_dir,'FRESH_HLS_OOC_ROUTE',checks,out) if statusfile.exists() else {'case_name':name,'status':'MISSING_SYNTHESIS_STATUS'}
            result['command_returncode']=process.returncode
            write(out/'active'/f'{name}.json',{'finished_at':now(),'status':result['status']})
        write(saved,result)
        return result
    records=[]
    with ThreadPoolExecutor(max_workers=max(1,min(args.workers,2))) as executor:
        futures={executor.submit(execute,name):name for name in ordered}
        for future in as_completed(futures):
            try: result=future.result()
            except Exception as error:
                result={'case_name':futures[future],'status':'ERROR','error':str(error)}
                write(out/'operators'/f"{futures[future]}.json",result)
            records.append(result)
            state.update(completed=len(records),updated_at=now(),last_case=result['case_name'],last_status=result['status'])
            write(out/'queue_status.json',state);aggregate(manifest,records,out)
            print(json.dumps({'completed':len(records),'total':len(cases),'case':result['case_name'],'status':result['status']}),flush=True)
    state.update(status='OPERATOR_PILOT_COMPLETE' if all(r.get('status')=='PASS_OPERATOR_REPORTS' for r in records) else 'OPERATOR_PILOT_INCOMPLETE',finished_at=now())
    write(out/'queue_status.json',state)


if __name__=='__main__':main()
