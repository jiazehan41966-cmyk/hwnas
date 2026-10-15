"""Fail-closed tests for legality, report admission, split and estimator scope."""
import copy,tempfile
import sys
import numpy as np
from common import *
from collect import parse_report,canonical_mapping
from cost_model import estimate_network_cost,predict,evaluate

def main():
    hashes=contracts();grid=json.loads((ART/'grid.json').read_text());base=copy.deepcopy(grid[0]);tests=[]
    assert legality(base,hashes)==[];tests.append('legal_point')
    for key,val,expected in [('Cin',49,'Cin_not_divisible'),('Cout',49,'Cout_not_divisible'),('input_tile',99,'unsupported_channel'),('spatial_tile','2x2','unsupported_spatial'),('linebuffer_bytes',9999999,'local_buffer'),('numeric_contract_hash','bad','contract_mismatch'),('output_stream_bits',7,'nonbyte_axis')]:
        r=dict(base);r[key]=val;assert any(expected in s for s in legality(r,hashes));tests.append('reject_'+key)
    rows=json.loads((ART/'clean.json').read_text())
    for r in rows:
        p=ROOT/r['source_report_path'];assert sha(p)==r['source_report_sha256'];parsed=parse_report(p)
        assert parsed['cycles']==r['cycles'] and parsed['latency_us']==r['latency_us']
        assert r['csim_status']=='PASS_BIT_EXACT' and r['csynth_status']=='PASS'
        assert r['tool_stage']=='CSYNTH';assert r['part_reported']==PART
    tests.append('clean_rows_bound_to_reports')
    assert not {r['shape_id'] for r in rows if r['split']=='fit'}&{r['shape_id'] for r in rows if r['split']=='validation'}
    tests.append('shape_disjoint_split')
    # Independent hand-built interpolation witness, neither training nor validation data.
    a=dict(rows[0],H=20,W=20,latency_us=4);b=dict(a,H=80,W=80,latency_us=64);q=dict(a,H=40,W=40)
    assert predict(q,[a,b],'interpolation')['latency_us']==16
    assert predict(dict(q,H=160,W=160),[a,b],'interpolation') is None
    tests.extend(['area_interpolation','interpolation_abstains_outside_bracket'])
    assert predict(dict(q,numeric_contract_hash='other'),[a,b],'nearest') is None
    assert predict(dict(q,target_clock_ns=10),[a,b],'nearest') is None
    tests.extend(['model_abstains_contract_mismatch','model_abstains_platform_mismatch'])
    for x in ['baseline','B1','C','D']:
        e=estimate_network_cost(x,rows=rows)
        if e['missing_nodes']:assert e['estimated_compute_latency_us'] is None
        assert e['shell_reservation']=='UNKNOWN'
    try:estimate_network_cost('not_a_frozen_candidate')
    except ValueError:pass
    else:raise AssertionError('accepted unsupported network')
    tests.append('network_estimator_fail_closed')
    frozen=json.loads((ROOT/'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items())
    assert sha(ROOT/'configs/search_space_sonar_v0_1.yaml')=='ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4'
    tests.append('frozen_sources_unchanged')
    sys.path.insert(0,str(ROOT/'tools/fpga_h2'))
    from interface import wide_axis_to_dma256,dma256_to_wide_axis
    widths=sorted({r[k]//8 for r in grid+json.loads((ART/'extra_grid.json').read_text()) for k in ['input_stream_bits','output_stream_bits']})
    witnesses=[]
    for width in widths:
        data=np.random.default_rng(42+width).integers(0,256,width*37,dtype=np.uint8).tobytes()
        words=[data[i:i+width] for i in range(0,len(data),width)]
        beats=wide_axis_to_dma256(words,width,len(data));assert dma256_to_wide_axis(beats,width,len(data))==words
        witnesses.append(dict(word_bytes=width,packets=37,status='SOFTWARE_ROUNDTRIP_BIT_EXACT',real_bytes=len(data),padded_bytes=len(beats)*32,hardware_latency_measured=False))
    save(ART/'wide_adapter_coverage.json',witnesses);tests.append('all_measured_native_widths_software_roundtrip')
    save(OUT/'H3_LOCAL_VALIDATION.json',dict(status='PASS',tests=tests,clean_rows=len(rows),**hashes));print('PASS',len(tests),'checks')
if __name__=='__main__':main()
