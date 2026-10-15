"""Exact-lookup mapping search over H3 measured points. No architecture NAS."""
import csv,itertools,json,re
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from common import *

def load_h3():
    man=json.loads(H3_MANIFEST.read_text())
    assert man['status']=='H3_DB_READY',man['status']
    hashes=contracts()
    assert man['numeric_contract_hash']==hashes['numeric_contract_hash']
    assert man['interface_contract_hash']==hashes['interface_contract_hash']
    rows=json.loads((ART/'clean.json').read_text())
    info=json.loads((ART/'network_usage.json').read_text())
    return man,hashes,rows,info

def mapping_tuple(r):
    return tuple(r[k] for k in MAPPING_KEY)

def validate_h3_row(r,hashes,bits=None):
    errors=[]
    if r.get('status')!='VALID':errors.append('status_not_valid')
    if r.get('LEGAL_MAPPING_CHECK')!='PASS':errors.append('legal_mapping_check_not_pass')
    if r.get('csim_status')!='PASS_BIT_EXACT':errors.append('csim_not_bit_exact')
    if r.get('csynth_status')!='PASS':errors.append('csynth_not_pass')
    if r.get('tool_stage')!='CSYNTH':errors.append('tool_stage_not_csynth')
    if r.get('part')!=PART or r.get('part_reported')!=PART:errors.append('part_mismatch')
    if r.get('tool_version')!='2022.2' or r.get('tool_reported')!='2022.2':errors.append('tool_version_mismatch')
    if int(r.get('target_clock_ns',-1))!=5 or str(r.get('clock_reported'))!='5':errors.append('target_clock_mismatch')
    for k,v in hashes.items():
        if r.get(k)!=v:errors.append('contract_mismatch_'+k)
    for k,v in (bits or V1_BITS).items():
        if int(r.get(k,-1))!=v:errors.append('bitwidth_mismatch_'+k)
    source_hash=r.get('source_hash') or ''
    if not re.fullmatch(r'[0-9a-f]{64}',source_hash):errors.append('source_hash_invalid')
    report_rel=r.get('source_report_path') or ''
    report_sha=r.get('source_report_sha256') or ''
    if not report_rel or relpath_hash(report_rel)!=report_sha:errors.append('source_report_sha256_mismatch')
    log_rel=r.get('tool_log_path') or ''
    log_sha=r.get('tool_log_sha256') or ''
    if log_rel and relpath_hash(log_rel)!=log_sha:errors.append('tool_log_sha256_mismatch')
    case_dir=ROOT/(r.get('case_dir') or '')
    key_path=case_dir/'measurement_key.json'
    status_path=(ROOT/(r.get('tool_log_path') or '')).with_suffix('')/'status.json' if r.get('tool_log_path') else None
    if not key_path.is_file():
        errors.append('measurement_key_missing')
    else:
        key=json.loads(key_path.read_text(encoding='utf-8'))
        files=key.get('source_files') or {}
        if key.get('source_hash')!=source_hash or hashobj(files)!=source_hash:
            errors.append('measurement_key_source_hash_mismatch')
        for name,expected in files.items():
            if not (case_dir/name).is_file() or sha(case_dir/name)!=expected:
                errors.append('source_file_sha_mismatch')
                break
        # Report and mapping identity apply to legacy H3 as well as new witnesses.
        expected_key=canonical_h3_mapping(key) if 'kind' in key else key
        for field in (*MAPPING_KEY,'shape_id','operator_type','Cin','Cout','H','W','kernel','stride','groups','activation'):
            if field in expected_key and r.get(field)!=expected_key[field]:errors.append('measurement_key_mismatch_'+field)
        report_path=ROOT/report_rel
        if report_path.is_file():
            try:
                tree=ET.parse(report_path).getroot()
                for field,tag in [('DSP','DSP'),('LUT','LUT'),('FF','FF'),('BRAM18','BRAM_18K'),('URAM','URAM')]:
                    if r.get(field)!=float(tree.findtext('.//AreaEstimates/Resources/'+tag)):errors.append('report_metric_mismatch_'+field)
                cycles=float(tree.findtext('.//SummaryOfOverallLatency/Worst-caseLatency'))
                clock=float(tree.findtext('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'))
                if r.get('cycles')!=cycles or r.get('latency_us')!=cycles*5/1000:errors.append('report_latency_mismatch')
                if r.get('estimated_clock_ns')!=clock or (r.get('timing_status')=='ESTIMATED_TARGET_MET')!=(clock<=5):errors.append('report_timing_mismatch')
            except (ET.ParseError, TypeError, ValueError):
                errors.append('report_metrics_invalid')
        if key.get('witness_files'):
            for name,expected in key['witness_files'].items():
                if not (case_dir/name).is_file() or sha(case_dir/name)!=expected:errors.append('witness_file_sha_mismatch')
            binary=(ROOT/report_rel).parent/'hardware_output.bin'
            if not binary.is_file() or sha(binary)!=key['witness_files']['reference.bin']:errors.append('retained_csim_output_mismatch')
    if status_path and status_path.is_file():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('source_hash') and status.get('source_hash')!=source_hash:
            errors.append('remote_status_source_hash_mismatch')
    measurement_valid=not errors
    timing_eligible=r.get('timing_status')=='ESTIMATED_TARGET_MET'
    out=dict(r)
    out['measurement_valid']=measurement_valid
    out['timing_eligible']=timing_eligible
    out['admission_errors']='|'.join(errors)
    return out,errors

def admit_rows(rows,hashes):
    by=defaultdict(dict)
    for r in rows:
        if r['operator_type'] in AUX_OPS:continue
        r,errors=validate_h3_row(r,hashes)
        if errors or not r['timing_eligible']:
            continue
        key=mapping_tuple(r);cur=by[r['shape_id']].get(key)
        # Prefer a successful repair over an original row with the same PE/SIMD.
        better=cur is None or bool(r.get('retry_of')) and not cur.get('retry_of') or (bool(r.get('retry_of'))==bool(cur.get('retry_of')) and r['point_id']>cur['point_id'])
        if better:by[r['shape_id']][key]=r
    return {sid:list(opts.values()) for sid,opts in by.items()}

def admit_aux_rows(rows,hashes):
    out=[]
    for r in rows:
        if r['operator_type'] not in AUX_OPS:continue
        r,errors=validate_h3_row(r,hashes)
        if not errors and r['timing_eligible']:
            out.append(r)
    return out

def lookup(shape_id,options,pe=None,simd=None):
    pool=options.get(shape_id,[])
    if pe is None:return pool
    hit=[r for r in pool if r['PE']==pe and r['SIMD']==simd]
    return hit

def needs_measurement(shape_id,options,pe,simd):
    if lookup(shape_id,options,pe,simd):return None
    return dict(shape_id=shape_id,requested_PE=pe,requested_SIMD=simd,status='NEEDS_MEASUREMENT')

def load_accuracy():
    cand={r['label']:r for r in json.loads(ACCURACY_SET.read_text())['architectures']}
    seeds=defaultdict(list)
    with ACCURACY_CSV.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seeds[row['label']].append(row)
    out={}
    for label,arch in ARCH.items():
        src='D' if label=='baseline' else label
        r=cand[src]
        assert r['architecture_id']==arch or label=='baseline'
        rows=seeds[src]
        assert len(rows)==3
        out[label]=dict(network=label,architecture_id=arch,accuracy_source_label=src,
            mean_mAP50_95=r['mean_mAP50_95'],std_mAP50_95=r['std_mAP50_95'],
            min_mAP50_95=r['min_mAP50_95'],max_mAP50_95=r['max_mAP50_95'],
            endpoint_mean_mAP50_95=r['endpoint_mean_mAP50_95'],n_seeds=3,test_evaluated=False,
            Params=r['Params'],MACs=r['MACs'],
            seed_mAP50_95=[float(x['mAP50_95']) for x in rows],
            seed_endpoint_mAP50_95=[float(x['endpoint_mAP50_95']) for x in rows],
            accuracy_split='official_valid_99_images',
            accuracy_note='Phase2F best-valid mAP50:95; test sealed; n=3' if label!='baseline' else 'Phase2F D all-k3 best-valid mAP50:95 inherited because frozen baseline has no 80-epoch detection mAP')
    return out

def family(node,r):
    if r['Cout']==21:return 'head21'
    if r['operator_type']=='DW' and r['kernel']==5:return 'DW5'
    if r['operator_type']=='DW':return 'DW3'
    if r['H']==160 and r['Cin']==96 and r['Cout']==96 and r['kernel']==3:return 'P2_tower'
    if r['operator_type']=='upsample_add' and r['H']==160:return 'P2_upsample'
    if r['operator_type']=='upsample_add':return 'upsample'
    if 'expand_conv' in node:return 'MBConv_expand'
    if 'project_conv' in node:return 'MBConv_project'
    if 'dw_conv' in node:return 'MBConv_dw'
    if node.startswith('neck.lateral'):return 'lateral'
    if node.startswith('backbone'):return 'backbone_conv'
    if node.startswith(('neck.output','head.towers')):return 'tower'
    return 'other'

def choose(pool,mode):
    if mode=='low':return min(pool,key=lambda r:(r['PE'],r['SIMD'],r['point_id']))
    if mode=='high':return max(pool,key=lambda r:(r['PE'],r['SIMD'],r['point_id']))
    if mode=='min_latency':return min(pool,key=lambda r:(r['latency_us'],r['DSP'],r['point_id']))
    if mode=='min_dsp':return min(pool,key=lambda r:(r['DSP'],r['latency_us'],r['point_id']))
    if mode=='min_lut':return min(pool,key=lambda r:(r['LUT'],r['latency_us'],r['point_id']))
    if mode=='min_bram':return min(pool,key=lambda r:(r['BRAM18'],r['latency_us'],r['point_id']))
    raise ValueError(mode)

def packing_cost(graph,rows):
    selected=[];missing=[]
    for d in graph['descriptors']:
        match=[r for r in rows if r['operator_type']=='DMARepack' and r['H']==d['height'] and r['W']==d['width'] and r['real_bytes']==d['real_bytes'] and r['padded_bytes']==d['padded_bytes']]
        if match:selected.append(match[0])
        else:missing.append(d['name'])
    return selected,missing

def operator_level_cost(selected,compute_us):
    """Name operator-level HLS proxies so they cannot be read as board measurements."""
    cycles=None
    if selected and all(r.get('cycles') is not None for r in selected):
        cycles=sum(float(r['cycles']) for r in selected)
    peak=None
    if selected:
        peak=dict(lut=max(r['LUT'] for r in selected),ff=max(r['FF'] for r in selected),
            dsp=max(r['DSP'] for r in selected),bram=max(r['BRAM18'] for r in selected),bram_unit='BRAM18')
    return dict(
        latency=dict(serial_op_estimate_us=compute_us,serial_op_estimate_cycles=cycles,board_measured_ms=None),
        resource=dict(peak_operator=peak,integrated_post_synth=None,board_routed=None),
        measurement_scope='OP_LEVEL_HLS',
        latency_metric='serial_op_estimate_us',
        resource_metric='peak_operator')

def bind_quantized_accuracy(row,acc):
    """Copy accuracy only when it was measured under this quantization contract."""
    measured=acc.get('accuracy_status')=='MEASURED' and acc.get('quantized_accuracy_bound') is True and acc.get('Accuracy') is not None
    keep=('accuracy_status','accuracy_precision','accuracy_split','accuracy_source','n_seeds','test_evaluated',
        'quantized_accuracy_bound','eval_contract_hash','quant_contract_hash','checkpoint_sha256',
        'calibration_sha256','seeds','accuracy_key','architecture_hash')
    for key in keep:
        if key in acc:row[key]=acc[key]
    if not measured:
        row.update(accuracy_status=acc.get('accuracy_status') or 'NEEDS_ACCURACY',Accuracy=None,
            mean_mAP50_95=None,std_mAP50_95=None,quantized_accuracy_bound=False)
        return row
    row.update(Accuracy=acc['Accuracy'],mean_mAP50_95=acc.get('mean_mAP50_95',acc['Accuracy']),
        std_mAP50_95=acc.get('std_mAP50_95'),accuracy_status='MEASURED',quantized_accuracy_bound=True)
    return row

def evaluate_assignment(net,usage,graph,assignment,acc,pack_rows,pack_missing,policy):
    selected=[];missing=[]
    for u in usage:
        row=assignment.get(u['node'])
        if row is None:missing.append(u)
        else:selected.append(row)
    if missing:
        return dict(network=net,mapping_policy=policy,status='NEEDS_MEASUREMENT',hardware_status='NEEDS_MEASUREMENT',
            missing_nodes=missing,measurement_scope='OP_LEVEL_HLS',
            latency=dict(serial_op_estimate_us=None,serial_op_estimate_cycles=None,board_measured_ms=None),
            resource=dict(peak_operator=None,integrated_post_synth=None,board_routed=None))
    families=defaultdict(float)
    details=[]
    for u,r in zip(usage,selected):
        fam=family(u['node'],r);families[fam]+=r['latency_us']
        details.append(dict(node=u['node'],shape_id=u['shape_id'],family=fam,point_id=r['point_id'],operator_type=r['operator_type'],
            Cin=r['Cin'],Cout=r['Cout'],H=r['H'],W=r['W'],kernel=r['kernel'],PE=r['PE'],SIMD=r['SIMD'],
            latency_us=r['latency_us'],DSP=r['DSP'],LUT=r['LUT'],FF=r['FF'],BRAM18=r['BRAM18'],
            source_report_path=r['source_report_path'],source_report_sha256=r['source_report_sha256'],
            source_hash=r['source_hash'],measurement_valid=r['measurement_valid'],timing_eligible=r['timing_eligible']))
    peak={k:max(r[k] for r in selected) for k in METRICS[1:]}
    total={k:sum(r[k] for r in selected) for k in METRICS[1:]}
    compute=sum(r['latency_us'] for r in selected)
    pack=None if pack_missing else sum(r['latency_us'] for r in pack_rows)
    row=dict(network=net,architecture_id=ARCH.get(net) or acc.get('architecture_id'),mapping_policy=policy,status='LOOKED_UP',
        hardware_status='MEASURED',compute_latency_us=compute,packing_latency_us=pack,
        combined_uncalibrated_us=None if pack is None else compute+pack,
        DSP_peak=peak['DSP'],LUT_peak=peak['LUT'],FF_peak=peak['FF'],BRAM18_peak=peak['BRAM18'],URAM_peak=peak['URAM'],
        DSP_sum=total['DSP'],LUT_sum=total['LUT'],FF_sum=total['FF'],BRAM18_sum=total['BRAM18'],URAM_sum=total['URAM'],
        lookup_nodes=len(selected),total_nodes=len(usage),missing_nodes=0,
        measurement_ids='|'.join(r['point_id'] for r in selected),
        packing_measurement_ids='|'.join(r['point_id'] for r in pack_rows),
        packing_missing_heads='|'.join(pack_missing),
        measurement_valid=all(r.get('measurement_valid') for r in selected+pack_rows),
        timing_eligible=all(r.get('timing_eligible') for r in selected+pack_rows),
        shell_reservation='UNKNOWN',latency_scope='HLS_CSYNTH_serial_operator_sum_not_board_e2e',
        resource_scope='kernel-only peak_operator; sum=static instantiation of operators, not whole-network post-route resource; exclude shell/adapters/DDR',
        device_DSP=DEVICE['DSP'],device_LUT=DEVICE['LUT'],device_BRAM18=DEVICE['BRAM18'],
        constraint_status='REPORTED_NOT_BUDGETED',
        timing_missed_points='|'.join(r['point_id'] for r in selected if r['timing_status']!='ESTIMATED_TARGET_MET'))
    row.update(operator_level_cost(selected,compute))
    bind_quantized_accuracy(row,acc or {})
    for fam,val in families.items():row['latency_'+fam]=val
    row['_details']=details;row['_families']=dict(families)
    return row

def named_assignment(usage,options,mode,boost_families=()):
    assignment={};unmeasured=[]
    for u in usage:
        pool=options.get(u['shape_id'],[])
        if not pool:
            unmeasured.append(dict(node=u['node'],shape_id=u['shape_id'],status='NEEDS_MEASUREMENT',reason='no_valid_csynth_row'))
            continue
        r=choose(pool,'min_latency' if family(u['node'],pool[0]) in boost_families else mode)
        assignment[u['node']]=r
    return assignment,unmeasured

def enumerate_pareto(usage,options,acc,graph,rows,net,pack_rows,pack_missing):
    shapes=sorted({u['shape_id'] for u in usage})
    dual=[s for s in shapes if len(options.get(s,[]))>1]
    fixed={s:options[s][0] for s in shapes if len(options.get(s,[]))==1}
    front=[]
    ncombo=1
    for s in dual:ncombo*=len(options[s])
    # Keep only the 2-objective front over latency and DSP peak; full dump is not a report.
    best=None
    for combo in itertools.product(*[options[s] for s in dual]) if dual else [()]:
        assignment=dict(fixed)
        assignment.update(zip(dual,combo))
        selected=[assignment[u['shape_id']] for u in usage]
        lat=sum(r['latency_us'] for r in selected)
        dsp=max(r['DSP'] for r in selected)
        lut=max(r['LUT'] for r in selected)
        bram=max(r['BRAM18'] for r in selected)
        ids='|'.join(r['point_id'] for r in selected)
        cand=dict(compute_latency_us=lat,DSP_peak=dsp,LUT_peak=lut,BRAM18_peak=bram,measurement_ids=ids)
        if best is None or lat<best['compute_latency_us']-1e-9 or (abs(lat-best['compute_latency_us'])<=1e-9 and dsp<best['DSP_peak']):
            best=cand
        dominated=False;keep=[]
        for p in front:
            if p['compute_latency_us']<=lat+1e-9 and p['DSP_peak']<=dsp and (p['compute_latency_us']<lat-1e-9 or p['DSP_peak']<dsp):
                dominated=True;break
            if not (lat<=p['compute_latency_us']+1e-9 and dsp<=p['DSP_peak'] and (lat<p['compute_latency_us']-1e-9 or dsp<p['DSP_peak'])):
                keep.append(p)
        if not dominated:
            keep.append(cand);front=keep
    return dict(network=net,enumerated_combinations=ncombo,dual_mapped_shapes=len(dual),
        latency_dsp_pareto=sorted(front,key=lambda r:(r['compute_latency_us'],r['DSP_peak'])),
        min_latency_among_measured=best)

def requested_high(r):
    if r['Cout']==21:return 3,4 if r['Cin']%4==0 else 2
    pe,simd=4,r['SIMD']
    if r['operator_type'] in ['DW','upsample_add']:simd=1
    elif r['Cin']==1:simd=1
    else:simd=4
    return pe,simd

def unmeasured_requests(info,options,hashes,rows):
    req=[];seen=set()
    sample={r['shape_id']:r for r in rows if r['operator_type'] not in AUX_OPS}
    for net,usage in info['usage'].items():
        for u in usage:
            sid=u['shape_id'];pool=options.get(sid,[])
            if not pool:
                key=(net,sid,'missing_any')
                if key in seen:continue
                seen.add(key)
                req.append(dict(network=net,node=u['node'],shape_id=sid,status='NEEDS_MEASUREMENT',reason='no_valid_measured_mapping',requested_PE='',requested_SIMD=''))
                continue
            low=choose(pool,'low');pe,simd=requested_high(low)
            if any(r['PE']==pe and r['SIMD']==simd for r in pool):continue
            probe=dict(low,PE=pe,SIMD=simd,input_tile=simd,output_tile=pe)
            reasons=legality(probe,hashes)
            key=(sid,pe,simd)
            if key in seen:continue
            seen.add(key)
            req.append(dict(network=net,node=u['node'],shape_id=sid,operator_type=low['operator_type'],
                Cin=low['Cin'],Cout=low['Cout'],H=low['H'],W=low['W'],kernel=low['kernel'],
                measured_PE=low['PE'],measured_SIMD=low['SIMD'],measured_point_id=low['point_id'],
                requested_PE=pe,requested_SIMD=simd,status='NEEDS_MEASUREMENT',
                legality='REJECT:'+'|'.join(reasons) if reasons else 'LEGAL_UNMEASURED',
                reason='second_measured_parallelism_absent' if not reasons else 'requested_mapping_illegal_under_h3_rules'))
    return req

def run_search():
    from accuracy_api import lookup_accuracy
    man,hashes,rows,info=load_h3()
    options=admit_rows(rows,hashes)
    pack_db=admit_aux_rows(rows,hashes)
    # Same-architecture quantized W8A8 scores only. Phase2F FP32 is not the Pareto accuracy.
    acc={label:lookup_accuracy(arch,'W8A8',hashes) for label,arch in ARCH.items()}
    named=['low','high','min_latency','min_dsp','min_lut','min_bram']
    boosts=[('boost_P2_tower',{'P2_tower','P2_upsample'}),('boost_head21',{'head21'}),('boost_DW5',{'DW5'}),
            ('boost_bottlenecks',{'P2_tower','P2_upsample','head21','DW5','MBConv_expand','MBConv_dw','MBConv_project'})]
    results=[];details={};pareto_enum=[];requests=unmeasured_requests(info,options,hashes,rows)
    for net in NETWORKS:
        usage=info['usage'][net]
        graph=json.loads((ROOT/info['packages'][net]/'graph.json').read_text())
        pack_rows,pack_missing=packing_cost(graph,pack_db)
        for mode in named:
            assignment,missing=named_assignment(usage,options,mode)
            assert not missing,(net,mode,missing)
            row=evaluate_assignment(net,usage,graph,assignment,acc[net],pack_rows,pack_missing,'reference_'+mode)
            details[(net,row['mapping_policy'])]=row.pop('_details');row.pop('_families')
            results.append(row)
        for name,fams in boosts:
            assignment,_=named_assignment(usage,options,'low',boost_families=fams)
            row=evaluate_assignment(net,usage,graph,assignment,acc[net],pack_rows,pack_missing,name)
            details[(net,row['mapping_policy'])]=row.pop('_details');row.pop('_families')
            results.append(row)
        pareto_enum.append(enumerate_pareto(usage,options,acc[net],graph,rows,net,pack_rows,pack_missing))
    return dict(h3=man,hashes=hashes,named=results,details=details,enumeration=pareto_enum,unmeasured=requests,options=options,info=info,accuracy=acc,rows=rows)
