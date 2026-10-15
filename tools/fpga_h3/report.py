"""Rebuild H3 status strictly from locally retained measured evidence."""
import csv
from common import *
from cost_model import evaluate,estimate_network_cost

def readcsv(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8')))
def main():
    rows=json.loads((ART/'clean.json').read_text());raw=readcsv('H3_MEASUREMENTS_RAW.csv');plan=readcsv('H3_SHAPE_COVERAGE_PLAN.csv')
    valid={r['shape_id'] for r in rows};coverage=[]
    for source in ['baseline','B1','C','D']:
        shapes={r['shape_id'] for r in plan if r['source']==source};atomic={r['shape_id'] for r in plan if r['source']==source and r['role']=='atomic'}
        coverage.append(dict(source=source,unique_shapes=len(shapes),measured_shapes=len(shapes&valid),coverage_percent=100*len(shapes&valid)/len(shapes),atomic_shapes=len(atomic),measured_atomic_shapes=len(atomic&valid)))
    csvwrite(OUT/'H3_COVERAGE_SUMMARY.csv',coverage)
    ev=evaluate([r for r in rows if r['split']!='auxiliary'])
    save(OUT/'H3_MODEL_SELECTION.json',dict(selected='exact_lookup',scope='measured shape and mapping points only',unseen_points='ABSTAIN_REQUIRE_MEASUREMENT',
        compared=['lookup','nearest','interpolation','linear'],held_out_fitting=False,
        rationale='Bounded measured grid uses exact lookup; held-out diagnostics do not authorize generalization to arbitrary shapes or mappings. Interpolation remains bracket-only.',
        timing_rule='ESTIMATED_TARGET_MISSED rows remain valid cost observations but must not be treated as feasible at 5 ns'))
    network=[estimate_network_cost(s,rows=rows) for s in ['baseline','B1','C','D']];save(OUT/'H3_NETWORK_ESTIMATES.json',network)
    chains=[]
    for chain in [r for r in rows if r['operator_type']=='ChainTowerHead']:
        tower=[r for r in rows if r['operator_type']=='Conv' and r['Cin']==96 and r['Cout']==96 and r['kernel']==3 and r['H']==chain['H'] and r['PE']==2 and r['SIMD']==2]
        head=[r for r in rows if r['operator_type']=='Conv' and r['Cout']==21 and r['H']==chain['H'] and r['PE']==1 and r['SIMD']==2]
        if tower and head:
            summed=tower[0]['latency_us']+head[0]['latency_us'];measured=chain['latency_us']
            chains.append(dict(point_id=chain['point_id'],H=chain['H'],sum_operator_us=summed,chain_csynth_us=measured,chain_over_sum=measured/summed,absolute_error_us=abs(summed-measured),percentage_error=100*abs(summed-measured)/measured,
                sum_operator_DSP=tower[0]['DSP']+head[0]['DSP'],chain_DSP=chain['DSP'],sum_operator_LUT=tower[0]['LUT']+head[0]['LUT'],chain_LUT=chain['LUT'],
                sum_operator_BRAM18=tower[0]['BRAM18']+head[0]['BRAM18'],chain_BRAM18=chain['BRAM18'],
                intermediate_DDR_bytes_avoided_static=2*96*chain['H']**2,
                scope='HLS compute dataflow chain with inferred storage vs individually bound storage; ratio includes both overlap and storage changes; excludes DMA/DDR stalls; no cosim proof of finite-FIFO runtime'))
    csvwrite(OUT/'H3_CHAIN_SANITY.csv',chains,fields=['point_id','H','sum_operator_us','chain_csynth_us','chain_over_sum','absolute_error_us','percentage_error','sum_operator_DSP','chain_DSP','sum_operator_LUT','chain_LUT','sum_operator_BRAM18','chain_BRAM18','intermediate_DDR_bytes_avoided_static','scope'])
    core_shapes={r['shape_id'] for r in plan};complete=core_shapes<=valid and all(not n['missing_nodes'] for n in network)
    gate_checks=dict(critical_shapes_covered=complete,
        chain_checks_complete={20,160}<={r['H'] for r in chains},
        interface_measurements_complete={20,40,80,160}<={r['H'] for r in rows if r['operator_type']=='DMARepack'},
        held_out_evaluation_complete=all(any(r['metric']==metric and r['method']=='nearest' and r['predicted_points']>0 for r in ev) for metric in ['latency_us','DSP','LUT','BRAM18']),
        no_unfinished_points=all(r['status']!='PENDING' for r in raw))
    status='H3_DB_READY' if all(gate_checks.values()) else 'H3_DB_PARTIAL' if rows else 'H3_DB_NOT_READY'
    counts=dict(planned=len(raw),attempted=sum(r['status']!='PENDING' for r in raw),valid=len(rows),failed=sum(r['status']=='INVALID' for r in raw),not_run=sum(r['status']=='PENDING' for r in raw))
    failures=[r for r in raw if r['status']=='INVALID'];transport=ART/'transport_status.json'
    blockers=[]
    if not complete:blockers.append('Critical shape coverage incomplete; exact lookup network totals withheld where missing.')
    if not any(r['operator_type']=='MBConv' for r in rows):blockers.append('No V1 composite MBConv CSYNTH admitted; shared numeric-array repair still needs tool verification.')
    if not any(r['operator_type']=='DW' and r['kernel']==5 for r in rows):blockers.append('No admitted DW5 measurement; serial-window scheduling repair still needs tool verification.')
    if not gate_checks['chain_checks_complete']:blockers.append('P2/P5 chain scheduling/overlap checks incomplete.')
    if not gate_checks['interface_measurements_complete']:blockers.append('Compute/interface latency split incomplete across all four head resolutions.')
    blockers.append('Shell resource reservation UNKNOWN: no system budget, detector-fit or board latency claim.')
    if transport.exists() and json.loads(transport.read_text()).get('status')=='UNAVAILABLE':blockers.append('SSH access unavailable after batch 01; pending batches not launched.')
    text=['# H3 coverage and measurement closure','',f'**{status}**; H4 readiness: **'+('YES' if status=='H3_DB_READY' else 'NO')+'**.','',f"Planned {counts['planned']}; attempted {counts['attempted']}; valid {counts['valid']}; failed {counts['failed']}; not run {counts['not_run']}.",'',
          'All admitted rows are real Vitis HLS 2022.2 CSYNTH reports bound to V1 contracts and bit-exact full-shape C-sim. None is place-and-route or board evidence.','',
          '| Source | Measured unique shapes | Coverage | Atomic coverage |','|---|---:|---:|---:|']
    for r in coverage:text.append(f"| {r['source']} | {r['measured_shapes']}/{r['unique_shapes']} | {r['coverage_percent']:.2f}% | {r['measured_atomic_shapes']}/{r['atomic_shapes']} |")
    text+=['','Coverage denominator includes atomic primitives and composite MBConv alternatives. These are not summed together by the network estimator.','',
           'Mappings: Conv PE2/4 SIMD2/4; head21 PE1/3 SIMD2/4. DW channel lanes2/4, serial per-channel reduction SIMD1. MBConv submitted SIMD field was the template EXP_TILE (expanded-output lanes), not input reduction SIMD; canonical rows correct SIMD to1 and retain submitted_SIMD and expanded_PE. Raw source hashes and failed attempts are unchanged. No spatial-tile or buffer-profile sweep was performed.','',
           'The original DW spatial pipeline can auto-unroll loops beyond requested channel lanes. PE is the source scheduling factor, not a count inferred from DSP; achieved II and resources come only from reports. Operator costs must remain source-hash bound.','',
           '## Held-out evaluation','',
           'Shape-disjoint split; reported metrics include abstentions and zero-denominator handling. Exact lookup appropriately abstains on unseen held-out shapes. No complex surrogate is fitted.','',
           '| Model | Metric | Predictions | MAE | MAPE % | Worst absolute | Worst % |','|---|---|---:|---:|---:|---:|---:|']
    for r in ev:
        if r['metric'] not in ['latency_us','DSP','LUT','BRAM18']:continue
        fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
        text.append(f"| {r['method']} | {r['metric']} | {r['predicted_points']}/{r['held_out_points']} | {fmt(r['MAE'])} | {fmt(r['MAPE'])} | {fmt(r['max_absolute_error'])} | {fmt(r['max_percentage_error'])} |")
    text+=['','## Chain and interface evidence','',
           '| Chain size | Operator sum us | Chain CSYNTH us | Sum error % | Chain/sum | Saved intermediate DDR bytes (static) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in chains:text.append(f"| {r['H']}x{r['H']} | {r['sum_operator_us']:.3f} | {r['chain_csynth_us']:.3f} | {r['percentage_error']:.3f} | {r['chain_over_sum']:.6f} | {r['intermediate_DDR_bytes_avoided_static']} |")
    text+=['','These point-specific chain ratios are not fitted as a global network correction. Successful chain cases use HLS-inferred storage where helper-function BIND_STORAGE was rejected; comparison with standalone points therefore includes both storage inference and dataflow overlap. Four-entry FIFO allocation is represented by synthesis; runtime backpressure and DDR contention have not been measured. Resource comparisons are retained in H3_CHAIN_SANITY.csv.','',
           '| Repack size | Real/padded bytes | CSYNTH cycles | Target-clock us | DSP | LUT | BRAM18 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted([r for r in rows if r['operator_type']=='DMARepack'],key=lambda r:r['H']):
        text.append(f"| {r['H']}x{r['W']} | {r['real_bytes']}/{r['padded_bytes']} | {r['cycles']} | {r['latency_us']:.3f} | {r['DSP']} | {r['LUT']} | {r['BRAM18']} |")
    text+=['','DMA costs belong to the H3 HLS reference repacker implementing the frozen byte contract. They are not costs of the separately validated H2 RTL bridge. No board or physical DMA bandwidth is inferred.','']
    text+=['','Nearest/regression generalization is not accepted as a calibrated replacement for missing measurements. Interpolation accuracy applies only to its reported bracketing coverage. No MAPE is hidden by excluding hard points; abstentions are counted separately. Exact lookup is selected for the measured grid; unseen mappings require measurement. Target-missed points are cost observations, not feasible mappings at5ns.','',
           '## Bottlenecks and remaining gates','']+['- '+s for s in blockers]
    text+=['','P2 tower compute and DDR traffic are reported separately. No full96x160x160 feature is assumed in BRAM. Head21 measurements exclude continuous-byte repack unless explicitly tagged interface-only. P5 remains 8400 real / 8416 padded bytes under the frozen contract.','',
           '## Reproduce','',
           'Run with project `.venv_cuda/Scripts/python.exe`: `tools/fpga_h3/collect.py`, `tools/fpga_h3/cost_model.py`, `tools/fpga_h3/validate.py`, `tools/fpga_h3/report.py`.','',
           'Remote batches use `tools/fpga_h3/remote.py launch extra` and `launch aux` in isolated directories; complete/fetch the extra batch before launching auxiliary jobs to retain the two-worker cap. Credentials are only supplied by existing Windows Credential Manager ASKPASS. Do not rerun a completed directory or overwrite old reports.','',
           'Baseline source manifest entries and frozen V0.1 config remain unchanged. The pre-existing additional data/evidence_audit.py changes the directory glob but is not part of operator generation.']
    (OUT/'H3_COVERAGE_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    key=['operator_type','Cin','Cout','H','W','kernel','stride','groups','activation','numeric_contract_hash','interface_contract_hash','input_bitwidth','weight_bitwidth','accumulator_bitwidth','output_bitwidth','PE','SIMD','input_tile','output_tile','spatial_tile','buffer_profile','part','tool_version','target_clock_ns','source_hash']
    value=['cycles','initiation_interval','latency_us','DSP','LUT','FF','BRAM18','URAM','estimated_clock_ns','achieved_or_estimated_fmax','tool_stage','csim_status','csynth_status','cosim_status','timing_status','source_report_path']
    save(OUT/'H3_DATABASE_SCHEMA.json',dict(version=1,required_key=key,required_value=value,tool_stages=['CSIM','CSYNTH','PLACE_ROUTE','BOARD'],null_policy='unknown values stay null; only complete CSYNTH rows admitted here',latency_scope_required=True,source_hash_includes=['top','testbench','headers','Tcl']))
    manifest=dict(version=1,status=status,H4_readiness='YES' if status=='H3_DB_READY' else 'NO',gate_checks=gate_checks,H4_scope='measured mapping points and exact lookup only; no extrapolated surrogate feasibility or system fit claim',counts=counts,coverage=coverage,**contracts(),platform=dict(board_reference='ZCU104',part=PART,tool_version='2022.2',target_clock_ns=5,DMA_bits=256,shell_reservation='UNKNOWN'),critical_unique_shapes=len(core_shapes),measured_critical_shapes=len(core_shapes&valid),remaining_blockers=blockers,
        provenance= json.loads((ART/'preflight.json').read_text()),reports={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='H3_DATABASE_MANIFEST.json'},source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/fpga_h3').glob('*.py')})
    save(OUT/'H3_DATABASE_MANIFEST.json',manifest);print(status,counts)
if __name__=='__main__':main()
