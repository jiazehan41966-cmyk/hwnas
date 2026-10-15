"""Simple, explicitly scoped operator models and serial network estimator."""
import csv,sys
import numpy as np
from common import *
METRICS=['latency_us','DSP','LUT','FF','BRAM18','URAM']
FEATURES=['Cin','Cout','H','W','kernel','stride','groups','PE','SIMD','expanded_channels']

def family(r):
    return (r['operator_type'],r['activation'],r.get('implementation_variant','v1_original'),
            *[r.get(k) for k in ['numeric_contract_hash','interface_contract_hash','part','tool_version','target_clock_ns','input_bitwidth','weight_bitwidth','accumulator_bitwidth','output_bitwidth','buffer_profile']])
def feature(r):return np.log2([max(float(r[k]),1) for k in FEATURES])
def predict(query,fit,method):
    pool=[r for r in fit if family(r)==family(query)]
    if not pool:return None
    if method=='lookup':
        exact=[r for r in pool if all(r[k]==query[k] for k in FEATURES) and r['buffer_profile']==query['buffer_profile']]
        return {k:float(exact[0][k]) for k in METRICS} if exact else None
    if method=='nearest':
        r=min(pool,key=lambda p:float(np.linalg.norm(feature(p)-feature(query))))
        return {k:float(r[k]) for k in METRICS}
    if method=='interpolation':
        same=[r for r in pool if all(r[k]==query[k] for k in FEATURES if k not in ['H','W']) and r['H']==r['W']]
        lo=sorted([r for r in same if r['H']<=query['H']],key=lambda r:r['H']);hi=sorted([r for r in same if r['H']>=query['H']],key=lambda r:r['H'])
        if not lo or not hi:return None
        a,b=lo[-1],hi[0];t=0 if a['H']==b['H'] else (query['H']**2-a['H']**2)/(b['H']**2-a['H']**2)
        return {k:float(a[k])*(1-t)+float(b[k])*t for k in METRICS}
    if method=='linear':
        if len(pool)<4:return None
        # Log-linear regression with a small fixed ridge; normalization fit-only.
        x=np.array([feature(r) for r in pool]);mu=x.mean(0);sd=np.maximum(x.std(0),1)
        x=np.column_stack([np.ones(len(pool)),(x-mu)/sd]);q=np.r_[1,(feature(query)-mu)/sd]
        penalty=np.eye(x.shape[1])*.01;penalty[0,0]=0
        y=np.log1p([[r[k] for k in METRICS] for r in pool]);coef=np.linalg.solve(x.T@x+penalty,x.T@y)
        return dict(zip(METRICS,np.maximum(0,np.expm1(np.clip(q@coef,0,40))).tolist()))
    raise ValueError(method)

def evaluate(rows):
    fit=[r for r in rows if r['split']=='fit'];held=[r for r in rows if r['split']=='validation']
    assert not {r['shape_id'] for r in fit}&{r['shape_id'] for r in held}
    errors=[];summary=[]
    for method in ['lookup','nearest','interpolation','linear']:
        for r in held:
            p=predict(r,fit,method)
            if p is None:continue
            for k in METRICS:
                actual=float(r[k]);err=abs(p[k]-actual)
                errors.append(dict(method=method,point_id=r['point_id'],metric=k,actual=actual,prediction=p[k],absolute_error=err,percentage_error=100*err/actual if actual else None))
        for k in METRICS:
            es=[r for r in errors if r['method']==method and r['metric']==k];percent=[r['percentage_error'] for r in es if r['percentage_error'] is not None]
            summary.append(dict(method=method,metric=k,fit_points=len(fit),held_out_points=len(held),predicted_points=len(es),coverage=len(es)/len(held) if held else 0,
                MAE=float(np.mean([r['absolute_error'] for r in es])) if es else None,MAPE=float(np.mean(percent)) if percent else None,
                max_absolute_error=max((r['absolute_error'] for r in es),default=None),max_percentage_error=max(percent,default=None),zero_actual_excluded_from_MAPE=len(es)-len(percent)))
    csvwrite(OUT/'H3_MODEL_VALIDATION.csv',summary);csvwrite(OUT/'H3_MODEL_PREDICTIONS.csv',errors)
    save(OUT/'H3_MODEL_SPLIT.json',dict(fit=[r['point_id'] for r in fit],held_out=[r['point_id'] for r in held],split_unit='shape',features=FEATURES,model_policy='exact lookup for supported network estimates; others diagnostic; no held-out fitting'))
    return summary

def estimate_network_cost(x,q='NUMERIC_CONTRACT_V1',m='reference_low_parallelism',rows=None):
    if x not in ['baseline','B1','C','D'] or q!='NUMERIC_CONTRACT_V1' or m!='reference_low_parallelism':raise ValueError('Unsupported network/contract/mapping')
    hashes=contracts();rows=rows if rows is not None else json.loads((ART/'clean.json').read_text())
    assert all(all(r[k]==v for k,v in hashes.items()) for r in rows)
    assert all(r['part']==PART and r['tool_version']=='2022.2' and r['target_clock_ns']==5 for r in rows)
    info=json.loads((ART/'network_usage.json').read_text());g=json.loads((ROOT/info['packages'][x]/'graph.json').read_text())
    selected=[];missing=[];traffic=0
    byid={n['id']:n for n in g['nodes']}
    for u in info['usage'][x]:
        pool=[r for r in rows if r['shape_id']==u['shape_id'] and r['tool_stage']=='CSYNTH']
        if pool:selected.append(min(pool,key=lambda r:(r['PE'],r['SIMD'],r['point_id'])))
        else:missing.append(u)
        n=byid[u['node']]
        traffic+=sum(int(np.prod(g['edges'][i]['shape'])) for i in n['inputs'])+int(np.prod(g['edges'][n['id']]['shape']))
        traffic+=int(np.prod(n.get('weight_shape',[0])))+4*n.get('bias_count',0)
        if n['op']=='upsample_add':
            # Scratch write of low input and read once per expanded output sample.
            traffic+=int(np.prod(g['edges'][n['inputs'][1]]['shape']))+int(np.prod(g['edges'][n['id']]['shape']))
    # Final head writes use DMA padding rather than raw logical byte count.
    traffic+=sum(d['padded_bytes']-d['real_bytes'] for d in g['descriptors'])
    packing=[];packing_missing=[]
    for d in g['descriptors']:
        match=[r for r in rows if r['operator_type']=='DMARepack' and r['H']==d['height'] and r['W']==d['width'] and r['real_bytes']==d['real_bytes'] and r['padded_bytes']==d['padded_bytes']]
        if match:packing.append(match[0])
        else:packing_missing.append(d['name'])
    return dict(network=x,contract=q,mapping=m,**hashes,lookup_nodes=len(selected),total_nodes=len(info['usage'][x]),missing_nodes=missing,
        timing_missed_points=[r['point_id'] for r in selected if r['timing_status']!='ESTIMATED_TARGET_MET'],
        reference_target_clock_ns=5,latency_feasibility='target-clock projection; reject target-missed points before any mapping feasibility decision',
        estimated_compute_latency_us=sum(r['latency_us'] for r in selected) if not missing else None,
        covered_subtotal_compute_us=sum(r['latency_us'] for r in selected),
        componentwise_resource_peak={k:max((r[k] for r in selected),default=0) for k in METRICS[1:]},
        static_instantiation_resource_sum={k:sum(r[k] for r in selected) for k in METRICS[1:]},
        resource_scope='peak is a lower bound for hypothetical reuse; sum assumes all atomic engines instantiated; both are kernel-only and exclude shell/adapters/interconnect and storage backing external BRAM parameter ports',
        estimated_DDR_traffic_bytes=traffic,DDR_policy='serial materialization; cold parameter reads plus explicit low scratch write/read; no bandwidth latency assumed',
        compute_scope='HLS CSYNTH operator sum, uncalibrated scheduling; NOT board end-to-end latency',packing_latency_us=None if packing_missing else sum(r['latency_us'] for r in packing),
        packing_missing_heads=packing_missing,packing_points=[r['point_id'] for r in packing],packing_scope='HLS reference repack implementation; excludes DDR stalls and does not measure the frozen RTL implementation',
        shell_reservation='UNKNOWN',measured_points=[r['point_id'] for r in selected])

if __name__=='__main__':
    rows=json.loads((ART/'clean.json').read_text());evaluate(rows)
    estimates=[estimate_network_cost(x,rows=rows) for x in ['baseline','B1','C','D']]
    save(OUT/'H3_NETWORK_ESTIMATES.json',estimates)
    print([(r['network'],r['lookup_nodes'],r['total_nodes']) for r in estimates])
