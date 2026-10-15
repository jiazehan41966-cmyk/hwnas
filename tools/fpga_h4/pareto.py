"""Pareto fronts over measured B1/C/D (x, q, m) lookups. No new mapping sweep."""
from collections import defaultdict
import math
from common import *
from cost_api import estimate_cost

PRIMARY_POLICIES=('reference_min_latency','reference_low','reference_high','reference_min_dsp','reference_min_lut','reference_min_bram')
QUANT='NUMERIC_CONTRACT_V1'


def public_point(row):
    return dict(
        network=row.get('network'),
        architecture_id=row.get('architecture_id'),
        mapping_policy=row.get('mapping_policy'),
        quant_config=row.get('quant_config'),
        status=row.get('status'),
        Accuracy=row.get('Accuracy'),
        Latency=row.get('Latency'),
        DSP=row.get('DSP'),
        LUT=row.get('LUT'),
        BRAM=row.get('BRAM'),
        mean_mAP50_95=row.get('mean_mAP50_95'),
        std_mAP50_95=row.get('std_mAP50_95'),
        compute_latency_us=row.get('compute_latency_us'),
        packing_latency_us=row.get('packing_latency_us'),
        DSP_peak=row.get('DSP_peak'),
        LUT_peak=row.get('LUT_peak'),
        BRAM18_peak=row.get('BRAM18_peak'),
        DSP_sum=row.get('DSP_sum'),
        LUT_sum=row.get('LUT_sum'),
        BRAM18_sum=row.get('BRAM18_sum'),
        lookup_nodes=row.get('lookup_nodes'),
        total_nodes=row.get('total_nodes'),
        measurement_ids=row.get('measurement_ids'),
        measurement_valid=row.get('measurement_valid'),
        timing_eligible=row.get('timing_eligible'),
        shell_reservation=row.get('shell_reservation'),
        deployment=row.get('deployment'),
        latency_scope=row.get('latency_scope'),
        resource_scope=row.get('resource_scope'),
        constraint_status=row.get('constraint_status'),
        accuracy_split=row.get('accuracy_split'),
        n_seeds=row.get('n_seeds'),
        test_evaluated=row.get('test_evaluated'),
    )


def collect_points(networks=None,policies=None):
    networks=list(networks or PARETO_NETWORKS)
    policies=list(policies or PRIMARY_POLICIES)
    rows=[]
    for net in networks:
        for policy in policies:
            row=estimate_cost(net,QUANT,policy)
            rows.append(row)
    return rows


def eligible(row):
    """Final Pareto admission: frozen Q(x), measured hardware, and measured quantized accuracy."""
    from q_admission import final_eligible
    return final_eligible(row)


def pareto_accuracy_peak_resources(rows):
    """Maximize accuracy and minimize operator-level LUT, FF, DSP, and BRAM peaks together."""
    keys=('Accuracy','LUT','FF','DSP','BRAM')
    keep=[]
    for r in rows:
        if not _finite(r,*keys):continue
        def dominates(a,b):
            better_or_equal=a['Accuracy']>=b['Accuracy']-1e-15 and a['LUT']<=b['LUT'] and a['FF']<=b['FF'] and a['DSP']<=b['DSP'] and a['BRAM']<=b['BRAM']
            strict=a['Accuracy']>b['Accuracy']+1e-15 or a['LUT']<b['LUT'] or a['FF']<b['FF'] or a['DSP']<b['DSP'] or a['BRAM']<b['BRAM']
            return better_or_equal and strict
        if any(dominates(p,r) for p in keep):continue
        keep=[p for p in keep if not dominates(r,p)]
        keep.append(r)
    return sorted(keep,key=lambda r:(-r['Accuracy'],r['LUT'],r['FF'],r['DSP'],r['BRAM']))


def _finite(row,*keys):
    return eligible(row) and all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in keys)


def pareto_max_min(rows,max_key,min_key):
    keep=[]
    for r in rows:
        if not _finite(r,max_key,min_key):continue
        dominated=any(p[max_key]>=r[max_key]-1e-15 and p[min_key]<=r[min_key]+1e-9 and (p[max_key]>r[max_key]+1e-15 or p[min_key]<r[min_key]-1e-9) for p in keep)
        if dominated:continue
        keep=[p for p in keep if not (r[max_key]>=p[max_key]-1e-15 and r[min_key]<=p[min_key]+1e-9 and (r[max_key]>p[max_key]+1e-15 or r[min_key]<p[min_key]-1e-9))]
        keep.append(r)
    return sorted(keep,key=lambda r:(-r[max_key],r[min_key]))


def pareto_accuracy_resources(rows):
    keys=('Accuracy','DSP','LUT','BRAM')
    keep=[]
    for r in rows:
        if not _finite(r,*keys):continue
        def dominates(a,b):
            better_or_equal=a['Accuracy']>=b['Accuracy']-1e-15 and a['DSP']<=b['DSP'] and a['LUT']<=b['LUT'] and a['BRAM']<=b['BRAM']
            strict=a['Accuracy']>b['Accuracy']+1e-15 or a['DSP']<b['DSP'] or a['LUT']<b['LUT'] or a['BRAM']<b['BRAM']
            return better_or_equal and strict
        if any(dominates(p,r) for p in keep):continue
        keep=[p for p in keep if not dominates(r,p)]
        keep.append(r)
    return sorted(keep,key=lambda r:(-r['Accuracy'],r['DSP'],r['LUT'],r['BRAM']))


def representative_min_latency(rows):
    by=defaultdict(list)
    for r in rows:
        if eligible(r):by[r['network']].append(r)
    out=[]
    for net in PARETO_NETWORKS:
        pool=by.get(net) or []
        if not pool:continue
        named=[r for r in pool if r.get('mapping_policy')=='reference_min_latency']
        out.append(named[0] if named else min(pool,key=lambda r:r['compute_latency_us']))
    return out


def build_fronts(rows=None):
    rows=rows if rows is not None else collect_points()
    looked=[r for r in rows if r.get('status')=='LOOKED_UP']
    admitted=[r for r in rows if eligible(r)]
    reps=representative_min_latency(admitted)
    return dict(
        all_looked_up=looked,
        all_rows=rows,
        eligible=admitted,
        representatives=reps,
        accuracy_latency=pareto_max_min(admitted,'Accuracy','Latency'),
        accuracy_dsp=pareto_max_min(admitted,'Accuracy','DSP'),
        accuracy_lut=pareto_max_min(admitted,'Accuracy','LUT'),
        accuracy_bram=pareto_max_min(admitted,'Accuracy','BRAM'),
        accuracy_resource=pareto_accuracy_resources(admitted),
        accuracy_latency_minlat=pareto_max_min(reps,'Accuracy','Latency'),
        accuracy_resource_minlat=pareto_accuracy_resources(reps),
    )
