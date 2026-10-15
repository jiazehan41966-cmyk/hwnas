"""Admit report-backed measurements only; keep stage and feasibility separate."""
import math,xml.etree.ElementTree as ET
from common import *

def parse_report(p):
    root=ET.parse(p).getroot()
    def val(path):
        s=root.findtext(path)
        if s is None or s.strip() in ['undef','?','-']:return None
        x=float(s);return int(x) if x.is_integer() else x
    r={k:val('.//AreaEstimates/Resources/'+tag) for k,tag in [('DSP','DSP'),('LUT','LUT'),('FF','FF'),('BRAM18','BRAM_18K'),('URAM','URAM')]}
    r.update(cycles=val('.//SummaryOfOverallLatency/Worst-caseLatency'),initiation_interval=val('.//SummaryOfOverallLatency/Interval-max'),estimated_clock_ns=val('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'))
    r['part_reported']=root.findtext('.//UserAssignments/Part');r['clock_reported']=val('.//UserAssignments/TargetClockPeriod')
    r['tool_reported']=root.findtext('.//ReportVersion/Version')
    r['latency_us']=r['cycles']*r['clock_reported']/1000 if r['cycles'] is not None else None
    r['latency_us_at_estimated_clock']=r['cycles']*r['estimated_clock_ns']/1000 if r['cycles'] is not None and r['estimated_clock_ns'] is not None else None
    r['achieved_or_estimated_fmax']=1000/r['estimated_clock_ns'] if r['estimated_clock_ns'] else None
    return r

def canonical_mapping(k):
    r=dict(k);r.pop('source_files',None)
    r['submitted_SIMD']=r['SIMD'];r['expanded_PE']=r.get('expanded_PE',r['SIMD'] if r['kind']=='mb' else 0)
    if r['kind'] in ['mb','dw','up']:
        # These templates unroll independent channels, but dot-product reduction is serial.
        r['SIMD']=1;r['input_tile']=1
    if r['kind']=='chain':
        # Composite scalar fields describe input SIMD and final output PE.
        # The complete heterogeneous mapping is carried explicitly per stage.
        r['SIMD']=2;r['input_tile']=2
        r['stage_mapping_json']=json.dumps([dict(stage='tower',PE=2,SIMD=2),dict(stage='head',PE=1,SIMD=2)],sort_keys=True)
    return r

def main():
    hashes=contracts();grid=json.loads((ART/'grid.json').read_text())
    membership={}
    with (OUT/'H3_SHAPE_COVERAGE_PLAN.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):membership.setdefault(row['shape_id'],set()).add(row['source'])
    extra=ART/'extra_grid.json'
    if extra.exists():grid+=json.loads(extra.read_text())
    aux=ART/'aux_grid.json'
    if aux.exists():grid+=json.loads(aux.read_text())
    retry=ART/'aux_retry_grid.json'
    if retry.exists():grid+=json.loads(retry.read_text())
    auto=ART/'aux_auto_grid.json'
    if auto.exists():grid+=json.loads(auto.read_text())
    stream=ART/'aux_stream_grid.json'
    if stream.exists():grid+=json.loads(stream.read_text())
    rows=[];clean=[];failed=[]
    for k in grid:
        r=canonical_mapping(k);reasons=[]
        r['witness_sources']=k['sources']
        if k['shape_id'] in membership:r['sources']='|'.join(sorted(membership[k['shape_id']]))
        d=ROOT/k['case_dir'];runroot=ART/('aux_stream_results' if k.get('aux_stream') else 'aux_auto_results' if k.get('aux_auto') else 'aux_retry_results' if k.get('aux_retry') else 'aux_results' if k.get('aux') else 'extra_results' if k.get('extra') else 'remote_results')
        sfile=runroot/k['point_id']/'status.json';report=runroot/k['point_id']/'closure_kernel_csynth.xml'
        log=runroot/(k['point_id']+'.log')
        if log.exists():
            r['tool_log_path']=str(log.relative_to(ROOT));r['tool_log_sha256']=sha(log)
            r['tool_errors']=' | '.join(s.strip() for s in log.read_text(errors='replace').splitlines() if 'ERROR:' in s)
        diagnostic=runroot/k['point_id']/'synthesis_diagnostic.log'
        if diagnostic.exists():
            r['diagnostic_path']=str(diagnostic.relative_to(ROOT));r['diagnostic_sha256']=sha(diagnostic)
        r.update(tool_stage='CSYNTH',csim_status='NOT_RUN',csynth_status='NOT_RUN',cosim_status='NOT_RUN',timing_status='UNKNOWN',latency_scope=k.get('latency_scope','operator_compute_excludes_DMA'),source_report_path='',source_report_sha256='',fmax_units='MHz_estimated',initiation_interval_scope='top_function_frame_interval')
        if k['LEGAL_MAPPING_CHECK']!='PASS':reasons.append(k['rejection_reason'])
        elif not sfile.exists():r['status']='PENDING';rows.append(r);continue
        else:
            s=json.loads(sfile.read_text());r.update(returncode=s['returncode'],wall_seconds=s['wall_seconds'])
            intact=all(sha(d/p)==h for p,h in k['source_files'].items()) and hashobj(k['source_files'])==k['source_hash'] and s['source_hash']==k['source_hash']
            if not intact:reasons.append('source_hash_mismatch')
            if any(k[h]!=v for h,v in hashes.items()):reasons.append('contract_hash_mismatch')
            binary=runroot/k['point_id']/'hardware_output.bin'
            exact=s['csim_binary_exact'] and binary.exists() and sha(binary)==sha(d/'reference.bin')
            r['csim_status']='PASS_BIT_EXACT' if exact else 'FAIL'
            r['reference_hash']=sha(d/'reference.bin');r['input_hash']=sha(d/'input.bin')
            r['binary_parameters_hash']=hashobj({p.name:sha(p) for p in d.glob('*.bin') if p.name not in ['input.bin','reference.bin']})
            if not exact:reasons.append('csim_not_bit_exact')
            r['csynth_status']='PASS' if s['returncode']==0 and report.exists() else 'FAIL'
            if r['csynth_status']!='PASS':reasons.append('csynth_failed_or_missing_report')
            if report.exists():
                try:r.update(parse_report(report))
                except (ValueError,ET.ParseError) as e:reasons.append('report_parse_'+str(e))
                r['source_report_path']=str(report.relative_to(ROOT));r['source_report_sha256']=sha(report)
                if r.get('part_reported')!=PART or r.get('clock_reported')!=5 or r.get('tool_reported')!='2022.2':reasons.append('platform_binding_mismatch')
                required=['cycles','initiation_interval','DSP','LUT','FF','BRAM18','URAM','estimated_clock_ns']
                if any(r.get(v) is None or not math.isfinite(float(r[v])) for v in required):reasons.append('nonfinite_or_missing_cost')
                else:r['timing_status']='ESTIMATED_TARGET_MET' if r['estimated_clock_ns']<=5 else 'ESTIMATED_TARGET_MISSED'
        r['status']='INVALID' if reasons else 'VALID';r['failure_reason']='|'.join(reasons)
        rows.append(r)
        if reasons:failed.append(r)
        else:clean.append(r)
    csvwrite(OUT/'H3_MEASUREMENTS_RAW.csv',rows);csvwrite(OUT/'H3_MEASUREMENTS_CLEAN.csv',clean,fields=list(dict.fromkeys(k for r in rows for k in r)))
    csvwrite(OUT/'H3_FAILED_POINTS.csv',failed,fields=list(dict.fromkeys(k for r in rows for k in r)))
    csvwrite(OUT/'H3_MAPPING_GRID.csv',[canonical_mapping(k) for k in grid])
    save(ART/'clean.json',clean)
    print(dict(total=len(rows),valid=len(clean),failed=len(failed),pending=sum(r['status']=='PENDING' for r in rows)),flush=True)
if __name__=='__main__':main()
