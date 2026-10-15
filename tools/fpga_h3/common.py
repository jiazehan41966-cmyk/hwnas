"""Contract-bound H3 evidence utilities. No search or deployment entry point."""
import csv, hashlib, json
import importlib.util,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/fpga_h3_cost_db'
ART=ROOT/'artifacts/fpga_h3/20260918_batch01'
PART='xczu7ev-ffvc1156-2-e'
_spec=importlib.util.spec_from_file_location('h3_canonical_hls_common',ROOT/'hls_lut_builder/scripts/common.py')
_mod=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_mod;_spec.loader.exec_module(_mod)
render_template=_mod.render_template
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashobj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def contracts():
    h2=ROOT/'reports/fpga_h2_contract';m=json.loads((h2/'closure_manifest.json').read_text())
    assert m['overall_status']=='PASS' and m['H3_gate']=='YES'
    result={}
    for n in ['numeric','interface']:
        p=h2/(n+'_contract_v1.json');key=n+'_contract_hash'
        result[key]=hashobj(json.loads(p.read_text()));assert result[key]==m[key]
    for rel,h in m['artifacts'].items():
        if rel.endswith(('numeric_v1.h','numeric.py','interface.py','CONTRACT_V1.md')):
            assert sha(ROOT/rel)==h,rel
    return result
def shape_key(n):
    return hashobj({k:n[k] for k in ['operator_type','Cin','Cout','H','W','kernel','stride','groups','activation','expanded_channels']})[:16]
def legality(r,hashes):
    reasons=[]
    if r['Cin']%r['SIMD']:reasons.append('Cin_not_divisible_by_SIMD')
    if r['Cout']%r['PE']:reasons.append('Cout_not_divisible_by_PE')
    if r['input_tile']!=r['SIMD'] or r['output_tile']!=r['PE']:reasons.append('unsupported_channel_tile')
    if r['spatial_tile']!=f"{r['H']}x{r['W']}":reasons.append('unsupported_spatial_tile')
    if r['buffer_profile'] not in ['linebuffer_fifo4','linebuffer_fifo4_ddr_low','linebuffer_fifo4_auto_storage']:reasons.append('unsupported_buffer')
    if r['linebuffer_bytes']>262144:reasons.append('local_buffer_exceeds_batch_policy_256KiB')
    if r['input_stream_bits']%8 or r['output_stream_bits']%8:reasons.append('nonbyte_axis')
    if max(r['input_stream_bits'],r['output_stream_bits'])>256 and not r['wide_adapter_contract']:reasons.append('missing_wide_adapter')
    for k,v in hashes.items():
        if r.get(k)!=v:reasons.append('contract_mismatch_'+k)
    return reasons
