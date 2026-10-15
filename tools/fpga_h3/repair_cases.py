"""Finite repair of synthesis-only issues; old failed attempts are immutable."""
import shutil
from common import *
def patch(d,k):
    src=(d/'kernel.cpp').read_text()
    if k['kind']=='mb':
        src=src.replace('#pragma HLS DATAFLOW\n', '''int expand_numeric[5], depthwise_numeric[5], project_numeric[5];
    for(int i=0;i<5;++i){expand_numeric[i]=numeric[i];depthwise_numeric[i]=numeric[i+5];project_numeric[i]=numeric[i+10];}
#pragma HLS DATAFLOW
''')
        src=src.replace('expand_bias, numeric);','expand_bias, expand_numeric);').replace('depthwise_bias, numeric+5);','depthwise_bias, depthwise_numeric);').replace('project_bias, numeric+10);','project_bias, project_numeric);')
        k['implementation_variant']='v1_separate_numeric_banks'
    if k['kind']=='dw':
        src=src.replace('for (int pw = 0; pw < PADDED_W; ++pw) {\n#pragma HLS PIPELINE II=1','for (int pw = 0; pw < PADDED_W; ++pw) {\n#pragma HLS PIPELINE off')
        src=src.replace('for (int kw = 0; kw < K; ++kw) {\n                            const int column','for (int kw = 0; kw < K; ++kw) {\n#pragma HLS PIPELINE II=1\n                            const int column')
        k['implementation_variant']='v1_serial_spatial_pipelined_taps'
    (d/'kernel.cpp').write_text(src)
    k['source_files']={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']};k['source_hash']=hashobj(k['source_files'])
    save(d/'measurement_key.json',k)
def main():
    extra=json.loads((ART/'extra_grid.json').read_text())
    for k in extra:patch(ROOT/k['case_dir'],k)
    original=json.loads((ART/'grid.json').read_text())
    for k in original:
        s=json.loads((ART/'remote_results'/k['point_id']/'status.json').read_text())
        if s['returncode']==0:continue
        r=dict(k);r['retry_of']=k['point_id'];r['point_id']+='_r1';r['extra']=True
        d=ART/'extra_cases'/r['point_id'];shutil.copytree(ROOT/k['case_dir'],d);r['case_dir']=str(d.relative_to(ROOT));patch(d,r);extra.append(r)
    save(ART/'extra_grid.json',extra);print('second bounded batch:',len(extra))
if __name__=='__main__':main()
