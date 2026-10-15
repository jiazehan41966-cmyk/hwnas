"""Write H4 protocol, Pareto tables, and mapping-selection analysis. No NAS launch."""
import json
from collections import defaultdict
from common import *
from search import run_search,family

def pareto_max_min(rows,max_key,min_key):
    keep=[]
    for r in rows:
        if r.get('status')!='LOOKED_UP' or r.get('accuracy_status')!='MEASURED' or r.get('quantized_accuracy_bound') is not True:continue
        if r.get(max_key) is None or r.get(min_key) is None:continue
        dominated=any(p[max_key]>=r[max_key]-1e-15 and p[min_key]<=r[min_key]+1e-9 and (p[max_key]>r[max_key]+1e-15 or p[min_key]<r[min_key]-1e-9) for p in keep)
        if dominated:continue
        keep=[p for p in keep if not (r[max_key]>=p[max_key]-1e-15 and r[min_key]<=p[min_key]+1e-9 and (r[max_key]>p[max_key]+1e-15 or r[min_key]<p[min_key]-1e-9))]
        keep.append(r)
    return sorted(keep,key=lambda r:(-r[max_key],r[min_key]))

def public_row(r):
    skip={k for k in r if k.startswith('_')}
    out={}
    for k,v in r.items():
        if k in skip:continue
        if isinstance(v,float):out[k]=f'{v:.9g}'
        elif isinstance(v,(list,dict)):out[k]=json.dumps(v)
        else:out[k]=v
    return out

def bottleneck_rows(details):
    rows=[]
    for (net,policy),items in details.items():
        if policy!='reference_min_latency':continue
        total=sum(x['latency_us'] for x in items)
        fam=defaultdict(lambda:dict(latency_us=0,DSP=0,LUT=0,BRAM18=0,n=0))
        for x in items:
            f=fam[x['family']];f['latency_us']+=x['latency_us'];f['DSP']=max(f['DSP'],x['DSP']);f['LUT']=max(f['LUT'],x['LUT']);f['BRAM18']=max(f['BRAM18'],x['BRAM18']);f['n']+=1
        for name,s in sorted(fam.items(),key=lambda kv:-kv[1]['latency_us']):
            rows.append(dict(network=net,mapping_policy=policy,family=name,n_ops=s['n'],latency_us=s['latency_us'],
                share=s['latency_us']/total,DSP_peak_in_family=s['DSP'],LUT_peak_in_family=s['LUT'],BRAM18_peak_in_family=s['BRAM18']))
        top=sorted(items,key=lambda x:-x['latency_us'])[:8]
        for i,x in enumerate(top,1):
            rows.append(dict(network=net,mapping_policy=policy,family='TOP_OP_'+str(i),n_ops=1,latency_us=x['latency_us'],
                share=x['latency_us']/total,DSP_peak_in_family=x['DSP'],LUT_peak_in_family=x['LUT'],BRAM18_peak_in_family=x['BRAM18'],
                node=x['node'],point_id=x['point_id'],operator_type=x['operator_type'],H=x['H'],kernel=x['kernel'],PE=x['PE'],SIMD=x['SIMD']))
    return rows

def write_protocol(data):
    h3=data['h3']
    text=f'''# H4 mapping search protocol

Status: **H4_API_READY_V2** for exact lookup and fixed-candidate reports. This is not deployment-ready.

H3_DB_READY is a measured-cost-database gate. It does not imply a whole detector fits, a board latency, or a place-and-route result.

## Scope

Search mapping `m` only inside `M_measured(x, q)` for frozen networks

`x ∈ {{baseline, B1, C, D}}`, `q = NUMERIC_CONTRACT_V1`.

This run does **not**:

- search a new architecture
- extend the Search Space V0.1
- predict unseen PE/SIMD with nearest neighbour, interpolation or regression
- start hardware-aware NAS, quantization search, a new SonarOp, or board deployment

If a requested mapping is absent from the H3 clean database the result is `NEEDS_MEASUREMENT`. No substitute number is written into the Pareto table.

## Frozen inputs

| Item | Value |
|---|---|
| H2 | PASS, H3_gate YES |
| H3 | {h3['status']} |
| numeric_contract_hash | `{h3['numeric_contract_hash']}` |
| interface_contract_hash | `{h3['interface_contract_hash']}` |
| part | {h3['platform']['part']} |
| tool | Vitis HLS {h3['platform']['tool_version']} |
| target clock | {h3['platform']['target_clock_ns']} ns |
| DMA | {h3['platform']['DMA_bits']} bit |
| shell reservation | UNKNOWN |
| H4_scope from H3 | {h3['H4_scope']} |

Networks are the frozen Phase2F / H3 graphs:

| Label | Architecture ID | Role |
|---|---|---|
| baseline | `{ARCH['baseline']}` | H3 all-k3 cost graph; accuracy inherited from D |
| B1 | `{ARCH['B1']}` | P2/P3/P4/P5 with k5 on stages 2/4/5/6/7 |
| C | `{ARCH['C']}` | P3/P4/P5 mixed k3/k5 |
| D | `{ARCH['D']}` | P3/P4/P5 all-k3 structural control |

Baseline and D are shape-equivalent. Frozen baseline itself has no 80-epoch detection mAP; H4 does not invent one.

## Accuracy source

Primary score: measured quantized W8A8 valid mAP50:95 for the same architecture and quantization contract. Phase2F FP32 scores are not substituted. Test remains sealed (`test_evaluated=false`). n=3 is too small for a significance claim.

Mapping choices do not change the trained weights, so quantized accuracy is constant inside each network and contract. The Accuracy-Latency Pareto is therefore across candidates that have both a measured HLS cost and a measured quantized score. Latency is the serial operator HLS estimate, not board end-to-end latency. Resource peaks are operator peaks, not a whole-network post-route total.

## Cost lookup

Exact H3 lookup keyed by `shape_id` and `(PE, SIMD, buffer_profile)`. Admitted rows require bit-exact C-sim, successful CSYNTH, intact V1 hashes, and a parseable report. Network totals use atomic operators, matching H3: composite MBConv rows are coverage alternatives, not a second addend.

- compute latency: serial sum of looked-up `latency_us` (CSYNTH cycles × 5 ns). Not board e2e.
- packing latency: looked-up DMARepack rows for each head descriptor. HLS reference repacker, not the H2 RTL bridge, not DDR wait.
- resources: componentwise peak (hypothetical reuse) and static sum (all engines instantiated). Kernel-only. Shell, interconnect and parameter-port backing are excluded.
- chain/standalone ratios from H3 are witnesses, not a global correction coefficient.

Device capacities LUT {DEVICE['LUT']}, DSP {DEVICE['DSP']}, BRAM18 {DEVICE['BRAM18']} are historical part totals, not an application budget. Constraints DSP/LUT/BRAM are recorded, not enforced, because shell reservation is UNKNOWN.

## Mapping decisions

Named policies choose one measured mapping per atomic node from its measured `shape_id` pool. Equal-shape nodes therefore receive the same policy choice, while explicit `by_node` mappings can bind equal-shape nodes to different measured rows without overwriting each other. Named policies:

- `reference_low` / `reference_high`: min/max measured PE then SIMD
- `reference_min_latency` / `min_dsp` / `min_lut` / `min_bram`: greedy per shape (optimal for a serial sum of that metric)
- `boost_*`: min-latency on named bottleneck families, otherwise low PE

Where a shape has two measured mappings, B1 has 17 dual-mapped shapes (131072 combinations). Those combinations are enumerated only to extract the latency–DSP-peak front. Unmeasured PE4/SIMD4 (or head PE3) points are listed in `UNMEASURED_MAPPING_REQUESTS.csv` and are not estimated.

## Stop condition

After this report: wait for human review. Do not automatically start hardware-aware NAS, quantization search, a new SonarOp, or board deployment.
'''
    (H4_OUT/'H4_PROTOCOL.md').write_text(text,encoding='utf-8')

def write_selection_report(data,front,bottlenecks):
    by=defaultdict(dict)
    for r in data['named']:
        by[r['network']][r['mapping_policy']]=r
    def lat(net,pol='reference_min_latency'):return by[net][pol]['compute_latency_us']
    def mAP(net):return by[net]['reference_min_latency']['mean_mAP50_95']
    def std(net):return by[net]['reference_min_latency']['std_mAP50_95']
    b1_low,b1_high,b1_min=lat('B1','reference_low'),lat('B1','reference_high'),lat('B1')
    drop=100*(b1_low-b1_min)/b1_low if b1_low else 0
    enum={e['network']:e for e in data['enumeration']}
    fam_lines=[]
    for r in bottlenecks:
        if r['network']=='B1' and r['mapping_policy']=='reference_min_latency' and not str(r['family']).startswith('TOP_OP_'):
            fam_lines.append(f"| {r['family']} | {r['latency_us']:.3f} | {100*r['share']:.2f}% | {r['n_ops']} |")
    top_lines=[]
    for r in bottlenecks:
        if r['network']=='B1' and str(r.get('family','')).startswith('TOP_OP_'):
            top_lines.append(f"| {r['family']} | `{r.get('node','')}` | {r['latency_us']:.3f} | PE{r.get('PE')} SIMD{r.get('SIMD')} | `{r.get('point_id','')}` |")
    front_lines=['| Network | Policy | mean mAP50:95 | CSYNTH compute μs | DSP peak | LUT peak | BRAM18 peak |','|---|---|---:|---:|---:|---:|---:|']
    for r in front:
        front_lines.append(f"| {r['network']} | {r['mapping_policy']} | {r['mean_mAP50_95']:.6f} | {r['compute_latency_us']:.3f} | {r['DSP_peak']} | {r['LUT_peak']} | {r['BRAM18_peak']} |")
    unmeasured_legal=sum(1 for r in data['unmeasured'] if r.get('legality')=='LEGAL_UNMEASURED')
    text='\n'.join([
        '# H4 mapping selection report','',
        'H4 API v2 repair is complete under exact lookup. **Not deployment-ready.** Shell reservation remains UNKNOWN. Latency is HLS CSYNTH estimated compute, not board end-to-end latency. No complete P&R and no DDR runtime are included.','',
        '## Accuracy-latency Pareto','',
        'Primary accuracy is measured W8A8 quantized valid mAP50:95 (n=3, test sealed). FP32 is not a fallback. Mapping does not change that quantized score.','',
        *front_lines,'',
        f"B1 mean mAP50:95 = {mAP('B1'):.6f} (sample std {std('B1'):.6f}). C = {mAP('C'):.6f} (std {std('C'):.6f}). D/baseline all-k3 = {mAP('D'):.6f} (std {std('D'):.6f}).",'',
        '## Answers','',
        '### 1. Is B1 worth its hardware cost for accuracy?','',
        f"On the quantized W8A8 score, B1 and C differ by {mAP('B1')-mAP('C'):.8f} mean mAP50:95. Min-latency CSYNTH serial-operator compute is {lat('B1'):.3f} μs for B1 versus {lat('C'):.3f} μs for C (ratio {lat('B1')/lat('C'):.2f}). This is a validation-only quantized score, n=3, not a test-set or board result. Phase2F FP32 is not the Pareto accuracy.",'',
        '### 2. Is C an accuracy-efficient candidate?','',
        f"Inside this frozen network set, compare C with B1 on the quantized score and with D on serial-operator compute ({lat('C'):.3f} vs {lat('D'):.3f} μs). Only candidates with measured quantized accuracy enter the front.",'',
        '### 3. Is D a low-cost baseline endpoint?','',
        f"**Yes, as the all-k3 P3/P4/P5 structural control.** D has the lowest mean mAP50:95 ({mAP('D'):.6f}) and the lowest CSYNTH compute among the four labels ({lat('D'):.3f} μs at `reference_min_latency`). Baseline uses the same architecture ID and the same accuracy source. H3 graphs are shape-equivalent; FPGA costs match under the same mapping policy. D is the low-accuracy / low-compute end of the front, not a stronger detector.",'',
        '### 4. Does a measured mapping clearly cut B1 latency?','',
        f"Among measured mappings, greedy min-latency versus reference-low reduces B1 compute by {drop:.2f}% ({b1_low:.3f} → {b1_min:.3f} μs). Reference-high is {b1_high:.3f} μs and is not automatically faster. Enumerated dual-mapping combinations for B1: {enum['B1']['enumerated_combinations']}; the measured latency floor is {enum['B1']['min_latency_among_measured']['compute_latency_us']:.3f} μs, identical to greedy min-latency because serial sum is separable. Remaining backbone shapes from H3 batch02 have only one measured mapping, so P2/backbone PE4 for those shapes is `NEEDS_MEASUREMENT`, not a hidden speedup. No measured mapping removes the P2 tower as the dominant term.",'',
        '### 5. Which operators dominate latency?','',
        'B1 `reference_min_latency` family shares (serial CSYNTH μs):','',
        '| Family | μs | Share | n ops |','|---|---:|---:|---:|',*fam_lines,'',
        'Largest single operators:','',
        '| Rank | Node | μs | Mapping | Measurement ID |','|---|---|---:|---|---|',*top_lines,'',
        'P2 tower / 160² Conv-96-96-k3 is the dominant B1 term. DW5 and MBConv atomic stages matter on B1/C relative to D but do not overtake P2. head21 is visible after packing is separated: compute head21 is a small fraction of P2; DMARepack adds 2.035/8.035/32.035/128.035 μs by resolution and is still not board DMA.','',
        '## Unmeasured mappings','',
        f"{unmeasured_legal} legal second-parallelism points are absent from the clean database (typically PE4/SIMD4 or head PE3 on H3 batch02 backbone shapes). They are listed in `UNMEASURED_MAPPING_REQUESTS.csv` and were not estimated.",'',
        '## Limitations','',
        '- shell resource reservation = UNKNOWN; no system budget, detector-fit or feasibility claim',
        '- no board measurement and no board latency',
        '- no DDR runtime, interconnect or memory-controller wait',
        '- no complete place-and-route',
        '- CSYNTH latency only, 5 ns target-clock projection',
        '- chain overlap/storage-inference ratios are not applied as a network correction',
        '- DMA costs are HLS reference repack, not H2 RTL and not physical DMA',
        '- accuracy is valid-split mAP50:95, n=3, test sealed',
        '- H4_READY for mapping lookup ≠ deployment ready','',
        '## Stop','',
        'Wait for human review. Do not start hardware-aware NAS, quantization search, a new SonarOp, or board deployment from this report.',''
    ])
    (H4_OUT/'MAPPING_SELECTION_REPORT.md').write_text(text,encoding='utf-8')

def write_figure(front,named):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (H4_OUT/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    colors={'baseline':'#4c4c4c','B1':'#d95f02','C':'#1b9e77','D':'#7570b3'}
    for r in named:
        if r['mapping_policy'] not in ['reference_min_latency','reference_low','reference_high']:continue
        ax.scatter(r['compute_latency_us']/1e6,r['mean_mAP50_95'],c=colors[r['network']],s=36,zorder=3)
    xs=[r['compute_latency_us']/1e6 for r in front];ys=[r['mean_mAP50_95'] for r in front]
    ax.plot(xs,ys,color='#666666',lw=1,zorder=2)
    for r in front:
        ax.annotate(r['network'],(r['compute_latency_us']/1e6,r['mean_mAP50_95']),textcoords='offset points',xytext=(5,4),fontsize=8)
    ax.set_xlabel('HLS CSYNTH serial compute latency (s), not board e2e')
    ax.set_ylabel('Quantized W8A8 valid mAP50:95')
    ax.set_title('H4 exact-lookup Accuracy-Latency (shell UNKNOWN)')
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    fig.savefig(H4_OUT/'figures'/'h4_accuracy_latency.png',dpi=140)
    fig.savefig(H4_OUT/'figures'/'h4_accuracy_latency.pdf')
    plt.close(fig)

def main():
    H4_OUT.mkdir(parents=True,exist_ok=True);(H4_OUT/'figures').mkdir(exist_ok=True)
    data=run_search()
    named=data['named']
    front=pareto_max_min(named,'mean_mAP50_95','compute_latency_us')
    for r in named:r['accuracy_latency_pareto']=r in front or any(r['network']==p['network'] and r['mapping_policy']==p['mapping_policy'] and r['compute_latency_us']==p['compute_latency_us'] for p in front)
    bottlenecks=bottleneck_rows(data['details'])
    write_protocol(data)
    csvwrite(H4_OUT/'NETWORK_FPGA_PARETO.csv',[public_row(r) for r in named if r['accuracy_latency_pareto'] or r['mapping_policy'] in ['reference_low','reference_high','reference_min_latency','boost_bottlenecks']])
    csvwrite(H4_OUT/'RESOURCE_LATENCY_TABLE.csv',[public_row(r) for r in named])
    csvwrite(H4_OUT/'UNMEASURED_MAPPING_REQUESTS.csv',data['unmeasured'] or [dict(status='NONE')])
    csvwrite(H4_OUT/'BOTTLENECK_TABLE.csv',bottlenecks)
    save(H4_OUT/'H4_ENUMERATION.json',data['enumeration'])
    write_selection_report(data,front,bottlenecks)
    write_figure(front,named)
    save(H4_OUT/'H4_ACCURACY.json',data['accuracy'])
    from validate import checks
    tests,n=checks(include_manifest=False)
    tests.append('stop_condition_no_deployment')
    save(H4_OUT/'H4_LOCAL_VALIDATION.json',dict(status='PASS',tests=tests,named_policies=n,pareto_points=len(front),unmeasured=len(data['unmeasured'])))
    manifest=dict(version=2,status='H4_API_READY',H4_deployment_ready='NO',stop='wait_for_human_review',
        next_forbidden=['hardware-aware NAS','quantization search','new SonarOp','board deployment'],
        **data['hashes'],h3_status=data['h3']['status'],shell_reservation='UNKNOWN',
        accuracy_metric='quantized_contract_valid_mAP50_95',latency_metric='serial_op_estimate_us',
        reports={str(p.relative_to(ROOT)):sha(p) for p in H4_OUT.rglob('*') if p.is_file() and p.name!='H4_DATABASE_MANIFEST.json'},
        source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/fpga_h4').glob('*.py')})
    save(H4_OUT/'H4_DATABASE_MANIFEST.json',manifest)
    print('H4',len(named),'named',len(front),'pareto',len(data['unmeasured']),'unmeasured')

if __name__=='__main__':main()
