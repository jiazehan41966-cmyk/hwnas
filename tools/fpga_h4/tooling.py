"""Generate H4 tooling reports from exact-lookup cost API. No NAS, P&R, or board claim."""
from collections import defaultdict
from common import *
from cost_api import estimate_cost, get_database
from pareto import QUANT, build_fronts, collect_points, public_point
from search import family
from validate_tooling import checks

SUMMARY_FIELDS=[
    'network','architecture_id','quant_config','mapping_policy','status',
    'Accuracy','Latency','DSP','LUT','BRAM',
    'mean_mAP50_95','std_mAP50_95','compute_latency_us','packing_latency_us',
    'DSP_peak','LUT_peak','BRAM18_peak','DSP_sum','LUT_sum','BRAM18_sum',
    'lookup_nodes','total_nodes','measurement_ids',
    'measurement_valid','timing_eligible',
    'shell_reservation','deployment','latency_scope','resource_scope','constraint_status',
    'accuracy_split','n_seeds','test_evaluated',
]


def _fmt(value,digits=6,strip=True):
    if value is None:return 'null'
    if isinstance(value,float):
        text=f'{value:.{digits}f}'
        return text.rstrip('0').rstrip('.') if strip else text
    return str(value)


def bottleneck(label,policy='reference_min_latency'):
    db=get_database()
    row=estimate_cost(label,QUANT,policy)
    if row['status']!='LOOKED_UP':return row,[]
    usage=db.info['usage'][label]
    ids=row['measurement_ids'].split('|')
    by={r['point_id']:r for r in db.rows}
    fam=defaultdict(lambda:dict(latency_us=0.0,DSP=0,LUT=0,BRAM18=0,n=0))
    ops=[]
    for u,pid in zip(usage,ids):
        r=by[pid]
        name=family(u['node'],r)
        rec=fam[name]
        rec['latency_us']+=r['latency_us'];rec['DSP']=max(rec['DSP'],r['DSP']);rec['LUT']=max(rec['LUT'],r['LUT']);rec['BRAM18']=max(rec['BRAM18'],r['BRAM18']);rec['n']+=1
        ops.append(dict(node=u['node'],family=name,point_id=pid,operator_type=r['operator_type'],H=r['H'],kernel=r['kernel'],PE=r['PE'],SIMD=r['SIMD'],latency_us=r['latency_us'],DSP=r['DSP'],LUT=r['LUT'],BRAM18=r['BRAM18']))
    total=sum(v['latency_us'] for v in fam.values())
    families=[dict(family=k,n_ops=v['n'],latency_us=v['latency_us'],share=v['latency_us']/total if total else None,**{x:v[x] for x in ('DSP','LUT','BRAM18')}) for k,v in sorted(fam.items(),key=lambda kv:-kv[1]['latency_us'])]
    return row,families,sorted(ops,key=lambda x:-x['latency_us'])[:8]


def table(rows,columns):
    header='| '+' | '.join(c[0] for c in columns)+' |'
    sep='|'+'|'.join('---:' if c[2]=='num' else '---' for c in columns)+'|'
    lines=[header,sep]
    for r in rows:
        cells=[]
        for _,key,kind in columns:
            val=r.get(key) if isinstance(r,dict) else r
            if kind=='num':
                key_l=key.lower()
                if 'latency' in key_l or key=='Latency':cells.append(_fmt(val,3))
                elif 'accuracy' in key_l or key=='Accuracy':cells.append(_fmt(val,6,strip=False))
                else:cells.append(_fmt(val,6))
            else:cells.append('' if val is None else str(val))
        lines.append('| '+' | '.join(cells)+' |')
    return lines


def write_cost_api(db,sample):
    text='\n'.join([
        '# Cost database API','',
        'Status: **H4_API_READY_V2**. Deployment: **NOT_CLAIMED**. Shell resource: **UNKNOWN**.','',
        '`estimate_cost(network_graph, quant_config, mapping)` is an exact-lookup interface over the H3 clean CSYNTH database. It is the (x, q, m) cost handle for later network / quantization / FPGA mapping work. It does not search, fit, or deploy.','',
        '## Signature','',
        '```python',
        'from cost_api import estimate_cost',
        "row = estimate_cost(network_graph, quant_config, mapping)",
        '```','',
        '| Argument | Accepted now | Unmeasured result |',
        '|---|---|---|',
        '| `network_graph`（网络图） | frozen labels `baseline` / `B1` / `C` / `D`, their `graph.json`, or a graph dict with the same atomic shapes | `NEEDS_MEASUREMENT` |',
        '| `quant_config`（量化配置） | `NUMERIC_CONTRACT_V1`, its canonical JSON, or its hash | any other bitwidth / rounding / contract → `NEEDS_MEASUREMENT` |',
        '| `mapping`（FPGA 映射） | named measured policies, or explicit per-node / per-shape full H3 key that exists in H3 | absent key field or mismatched constraint → `NEEDS_MEASUREMENT` |','',
        'Returned primary fields: `Accuracy`, `Latency`, `DSP`, `LUT`, `BRAM`. On `LOOKED_UP`, `Latency` is HLS CSYNTH serial operator sum in μs, not board e2e latency（板级端到端延迟）. `DSP` / `LUT` / `BRAM` are kernel-only peaks. On `NEEDS_MEASUREMENT` those five fields are null.','',
        '## Lookup key','',
        'Exact match on operator `shape_id` and the full H3 mapping key: V1 numeric/interface hashes, INT8/INT8/INT32/INT8 bitwidths, `PE`, `SIMD`, channel/spatial tile, buffer profile, part `xczu7ev-ffvc1156-2-e`, Vitis HLS 2022.2, 5 ns, source hash, and CSYNTH report hash. Composite MBConv rows remain coverage alternatives and are not added on top of atomic expand/dw/project nodes.','',
        'Named policies only choose among already-admitted H3 rows: `reference_low`, `reference_high`, `reference_min_latency`, `reference_min_dsp`, `reference_min_lut`, `reference_min_bram`. DW canonical SIMD in the clean database is 1; requesting DW SIMD=2 is `NEEDS_MEASUREMENT`, not an alias.','',
        '## Forbidden','',
        '- nearest interpolation',
        '- regression extrapolation',
        '- filling a missing operator from a neighbour shape',
        '- treating a partial lookup as a network cost','',
        '## Smoke','',
        f"B1 + NUMERIC_CONTRACT_V1 + `reference_min_latency` → `{sample['status']}`, Latency `{_fmt(sample['Latency'],3)}` μs (CSYNTH), DSP `{sample['DSP']}`, LUT `{sample['LUT']}`, BRAM `{sample['BRAM']}`, shell `{sample['shell_reservation']}`, deployment `{sample['deployment']}`.",'',
        f"H3 clean points available: {len(db.rows)}. Admitted non-aux H4 options: {sum(len(v) for v in db.options.values())}. Frozen contract hashes: numeric `{db.hashes['numeric_contract_hash']}`, interface `{db.hashes['interface_contract_hash']}`.",''
    ])
    (H4_TOOLING_OUT/'COST_API.md').write_text(text,encoding='utf-8')


def write_pareto(fronts,bottlenecks):
    reps=fronts['representatives']
    lat_cols=[('Network','network','str'),('Policy','mapping_policy','str'),('Accuracy (mAP50:95)','Accuracy','num'),('Latency CSYNTH μs','Latency','num'),('DSP peak','DSP','num'),('LUT peak','LUT','num'),('BRAM18 peak','BRAM','num')]
    b1=next(r for r in reps if r['network']=='B1')
    c=next(r for r in reps if r['network']=='C')
    d=next(r for r in reps if r['network']=='D')
    fam_lines=['| Family | μs | Share | n |','|---|---:|---:|---:|']
    for rec in bottlenecks['B1'][1]:
        fam_lines.append(f"| {rec['family']} | {_fmt(rec['latency_us'],3)} | {_fmt(100*rec['share'],2)}% | {rec['n_ops']} |")
    top_lines=['| Rank | Node | μs | Mapping | ID |','|---|---|---:|---|---|']
    for i,op in enumerate(bottlenecks['B1'][2],1):
        top_lines.append(f"| {i} | `{op['node']}` | {_fmt(op['latency_us'],3)} | PE{op['PE']} SIMD{op['SIMD']} | `{op['point_id']}` |")
    text='\n'.join([
        '# Pareto generation','',
        'Input networks: **B1**, **C**, **D**. Quantization: **NUMERIC_CONTRACT_V1** only. Mapping: existing H3 measured entries, no new mapping sweep.','',
        'Accuracy（精度） is the measured quantized mAP50:95 for the same quantization contract. Phase2F FP32 scores are not substituted. Latency（延迟） is the HLS CSYNTH serial operator estimate, not board latency. DSP / LUT / BRAM are peak operator resources, not a whole-network post-route total. Shell resource remains UNKNOWN. Deployment remains NOT_CLAIMED.','',
        '## Representative points (`reference_min_latency`)','',
        *table(reps,lat_cols),'',
        f"B1 Accuracy `{_fmt(b1['Accuracy'],6,strip=False)}` (std `{_fmt(b1['std_mAP50_95'],6,strip=False)}`), Latency `{_fmt(b1['Latency'],3)}` μs, DSP `{b1['DSP']}`, LUT `{b1['LUT']}`, BRAM `{b1['BRAM']}`.",
        f"C Accuracy `{_fmt(c['Accuracy'],6,strip=False)}` (std `{_fmt(c['std_mAP50_95'],6,strip=False)}`), Latency `{_fmt(c['Latency'],3)}` μs, DSP `{c['DSP']}`, LUT `{c['LUT']}`, BRAM `{c['BRAM']}`.",
        f"D Accuracy `{_fmt(d['Accuracy'],6,strip=False)}` (std `{_fmt(d['std_mAP50_95'],6,strip=False)}`), Latency `{_fmt(d['Latency'],3)}` μs, DSP `{d['DSP']}`, LUT `{d['LUT']}`, BRAM `{d['BRAM']}`.",'',
        'Baseline is shape-equivalent to D and is excluded from this Pareto input to avoid a duplicate point.','',
        '## Accuracy-Latency Pareto','',
        'Maximize Accuracy, minimize CSYNTH Latency. Mapping does not change Accuracy, so each network contributes its min-latency measured mapping.','',
        *table(fronts['accuracy_latency_minlat'],lat_cols),'',
        'Front membership uses quantized accuracy and the serial operator latency estimate together. A hardware-only or FP32-only row is not on this front.','',
        '## Accuracy-Resource Pareto','',
        'Combined dominance: maximize Accuracy and minimize DSP, LUT, and BRAM peaks together. Pairwise Accuracy-DSP / Accuracy-LUT / Accuracy-BRAM fronts are listed after the combined front.','',
        '### Combined (Accuracy, DSP, LUT, BRAM)','',
        *table(fronts['accuracy_resource_minlat'],lat_cols),'',
        '### Accuracy-DSP','',
        *table(pareto_subset(fronts,'accuracy_dsp'),lat_cols),'',
        '### Accuracy-LUT','',
        *table(pareto_subset(fronts,'accuracy_lut'),lat_cols),'',
        '### Accuracy-BRAM','',
        *table(pareto_subset(fronts,'accuracy_bram'),lat_cols),'',
        'Combined dominance uses the min-latency representative of each network. Pairwise Accuracy-DSP / LUT / BRAM fronts are computed over all named measured policies; a lower-LUT mapping of the same network can dominate its min-latency point. C is absent from Accuracy-DSP and Accuracy-LUT because B1 matches its peaks with higher Accuracy. D remains only on Accuracy-BRAM.','',
        'On kernel-only peaks at min-latency, B1 weakly dominates C on Accuracy with equal DSP/LUT/BRAM, so C drops from the combined resource front. D remains because BRAM peak is lower (48 vs 96) despite worse Accuracy, DSP, and LUT. This is not a shell budget and not a detector-fit claim.','',
        'Named measured policies other than min-latency are recorded in `NETWORK_COST_SUMMARY.csv`. They cannot improve Accuracy. They only move Latency/DSP/LUT/BRAM inside the measured mapping set.','',
        '## Bottleneck at B1 min-latency','',
        *fam_lines,'',
        *top_lines,'',
        'P2 tower / 160² Conv-96-96-k3 is the dominant B1 term. No measured mapping removes it. Unmeasured PE4 on remaining backbone shapes is `NEEDS_MEASUREMENT`, not a hidden speedup.','',
        '## Method','',
        '1. `estimate_cost(x, NUMERIC_CONTRACT_V1, m)` for `x ∈ {B1,C,D}` and named measured `m`.',
        '2. Keep `LOOKED_UP` rows only.',
        '3. Build two-objective fronts with strict Pareto dominance.',
        '4. Build the combined Accuracy-Resource front on (Accuracy, DSP_peak, LUT_peak, BRAM18_peak).',
        '5. Never insert a `NEEDS_MEASUREMENT` row into a front.','',
        'Reproduce: `python tools/fpga_h4/tooling.py` from the repository root, or the same file from `tools/fpga_h4`.',''
    ])
    (H4_TOOLING_OUT/'PARETO_GENERATION.md').write_text(text,encoding='utf-8')


def pareto_subset(fronts,name):
    wanted={(r['network'],r['mapping_policy']) for r in fronts['representatives']}
    rows=[r for r in fronts[name] if (r['network'],r['mapping_policy']) in wanted]
    return rows or fronts[name]


def write_limitations(fronts):
    n_lookup=sum(1 for r in fronts['all_rows'] if r['status']=='LOOKED_UP')
    n_need=sum(1 for r in fronts['all_rows'] if r['status']=='NEEDS_MEASUREMENT')
    text='\n'.join([
        '# Limitations','',
        'H4 tooling finalization exposes an exact-lookup (x, q, m) cost handle. It does not make a system feasible, quantized, or deployed.','',
        '## Must remain explicit','',
        '| Item | Value |',
        '|---|---|',
        '| Latency | HLS CSYNTH estimate (serial operator sum at 5 ns). **Not** board latency. |',
        '| Shell resource | **UNKNOWN** |',
        '| Deployment | **NOT_CLAIMED** |',
        '| Database | exact lookup only |',
        '| Unknown triple | `NEEDS_MEASUREMENT`, primary metrics null |','',
        '## Out of scope for this gate','',
        '- hardware-aware NAS',
        '- new mapping sweep',
        '- place-and-route',
        '- board deployment',
        '- shell feasibility claim',
        '- INT4 / INT6 / mixed-precision cost',
        '- nearest interpolation or regression extrapolation','',
        '## Current coverage','',
        f'- Pareto input rows looked up: {n_lookup}; abstained: {n_need}.',
        '- q is a singleton: `NUMERIC_CONTRACT_V1`.',
        '- m is the H3 measured PE/SIMD/buffer set, not a generator of new mappings.',
        '- Accuracy is quantized validation mAP for the same contract, n=3, test sealed. FP32 is not a fallback.',
        '- Resources exclude shell, interconnect, DDR backing, and H2 RTL DMA.',
        '- Device LUT 230400 / DSP 1728 / BRAM18 624 are part totals, not an application budget.',
        '- Constraint status is `REPORTED_NOT_BUDGETED` because shell reservation is UNKNOWN.','',
        '## (x, q, m) after quantization','',
        'The API already takes the triple. After a future quantization campaign, new measured `q` rows can be admitted to the same lookup table. Until those rows exist, a new `quant_config` returns `NEEDS_MEASUREMENT` and cannot enter Pareto or an optimizer. This is fail-closed support for later (x, q, m) search, not a running quantization optimizer.','',
        '## Stop','',
        'Wait for human review. Do not start hardware-aware NAS, a quantization sweep, P&R, or board deployment from these reports.','',
    ])
    (H4_TOOLING_OUT/'LIMITATIONS.md').write_text(text,encoding='utf-8')


def write_figures(reps,fronts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=H4_TOOLING_OUT/'figures';figdir.mkdir(exist_ok=True)
    colors={'B1':'#d95f02','C':'#1b9e77','D':'#7570b3'}
    def scatter_front(ax,xs,ys,front,xlabel,title):
        for r in reps:
            ax.scatter(xs(r),ys(r),c=colors[r['network']],s=42,zorder=3,label=r['network'])
            ax.annotate(r['network'],(xs(r),ys(r)),textcoords='offset points',xytext=(5,4),fontsize=8)
        fx=[xs(r) for r in front];fy=[ys(r) for r in front]
        ax.plot(fx,fy,color='#666666',lw=1,zorder=2)
        ax.set_xlabel(xlabel);ax.set_ylabel('Quantized valid mAP50:95')
        ax.set_title(title);ax.grid(True,alpha=0.3)
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    scatter_front(ax,lambda r:r['Latency']/1e6,lambda r:r['Accuracy'],fronts['accuracy_latency_minlat'],
                  'HLS CSYNTH serial compute latency (s), not board e2e','Accuracy-Latency (shell UNKNOWN)')
    ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(figdir/'h4_tooling_accuracy_latency.png',dpi=140);fig.savefig(figdir/'h4_tooling_accuracy_latency.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.6),sharey=True)
    panels=[('DSP',lambda r:r['DSP'],'Accuracy-DSP peak'),('LUT',lambda r:r['LUT'],'Accuracy-LUT peak'),('BRAM18',lambda r:r['BRAM'],'Accuracy-BRAM peak')]
    fronts_pair={'DSP':fronts['accuracy_dsp'],'LUT':fronts['accuracy_lut'],'BRAM18':fronts['accuracy_bram']}
    for ax,(name,getter,title) in zip(axes,panels):
        pair=[r for r in fronts_pair[name] if r['mapping_policy']=='reference_min_latency'] or fronts_pair[name]
        scatter_front(ax,getter,lambda r:r['Accuracy'],pair,name+' peak (kernel-only)',title)
    axes[0].legend(frameon=False,loc='lower right')
    fig.suptitle('Accuracy-Resource (CSYNTH peaks; shell UNKNOWN; not board)',fontsize=10)
    fig.tight_layout();fig.savefig(figdir/'h4_tooling_accuracy_resource.png',dpi=140);fig.savefig(figdir/'h4_tooling_accuracy_resource.pdf');plt.close(fig)


def main():
    H4_TOOLING_OUT.mkdir(parents=True,exist_ok=True)
    db=get_database()
    sample=estimate_cost('B1',QUANT,'reference_min_latency')
    rows=collect_points()
    fronts=build_fronts(rows)
    bottlenecks={net:bottleneck(net) for net in PARETO_NETWORKS}
    summary=[public_point(r) for r in rows if r.get('network') in PARETO_NETWORKS]
    csvwrite(H4_TOOLING_OUT/'NETWORK_COST_SUMMARY.csv',[{k:r.get(k) for k in SUMMARY_FIELDS} for r in summary],SUMMARY_FIELDS)
    write_cost_api(db,sample)
    write_pareto(fronts,bottlenecks)
    write_limitations(fronts)
    write_figures(fronts['representatives'],fronts)
    tests=checks()
    save(H4_TOOLING_OUT/'H4_TOOLING_VALIDATION.json',dict(status='PASS',tests=tests,n_tests=len(tests),
        looked_up=sum(1 for r in rows if r['status']=='LOOKED_UP'),needs_measurement=sum(1 for r in rows if r['status']=='NEEDS_MEASUREMENT'),
        accuracy_latency_points=len(fronts['accuracy_latency_minlat']),accuracy_resource_points=len(fronts['accuracy_resource_minlat']),
        shell_reservation='UNKNOWN',deployment='NOT_CLAIMED',latency='HLS_CSYNTH_estimate_not_board'))
    reports={str(p.relative_to(ROOT)):sha(p) for p in H4_TOOLING_OUT.rglob('*') if p.is_file() and p.name!='H4_TOOLING_MANIFEST.json'}
    save(H4_TOOLING_OUT/'H4_TOOLING_MANIFEST.json',dict(version=2,status='H4_API_READY',H4_deployment_ready='NO',
        deployment='NOT_CLAIMED',shell_reservation='UNKNOWN',latency_scope='HLS_CSYNTH_serial_operator_sum_not_board_e2e',
        stop='wait_for_human_review',next_forbidden=['hardware-aware NAS','new mapping sweep','P&R','board deployment','shell feasibility claim'],
        **db.hashes,h3_status=db.manifest['status'],pareto_networks=PARETO_NETWORKS,quant_config=QUANT,
        reports=reports,source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/fpga_h4').glob('*.py')}))
    print('H4 tooling',len(summary),'rows',len(fronts['accuracy_latency_minlat']),'acc-lat',len(fronts['accuracy_resource_minlat']),'acc-res')


if __name__=='__main__':
    main()
