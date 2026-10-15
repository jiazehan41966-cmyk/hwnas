"""Let HLS infer chain-local storage; do not assume inferred memories are BRAM."""
import re,shutil
from common import *
def main():
    rows=[]
    for old in json.loads((ART/'aux_retry_grid.json').read_text()):
        r=dict(old);r['retry_of']=old['point_id'];r['point_id']=old['point_id'].replace('_r1','_r2');r['aux_auto']=True
        d=ART/'aux_auto_cases'/r['point_id'];shutil.copytree(ROOT/old['case_dir'],d)
        source=(d/'kernel.cpp').read_text()
        source=re.sub(r'^#pragma HLS BIND_STORAGE[^\n]*\n','// Storage inferred by HLS in this composed kernel.\n',source,flags=re.M)
        (d/'kernel.cpp').write_text(source);r['case_dir']=str(d.relative_to(ROOT))
        r['implementation_variant']='v1_chain_auto_storage';r['buffer_profile']='linebuffer_fifo4_auto_storage'
        r['linebuffer_bytes']=96*3*(r['W']+2)+96*r['W'];r['weight_buffer_bytes']=96*96*9+21*96
        r['source_files']={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']};r['source_hash']=hashobj(r['source_files'])
        assert not legality(r,contracts());save(d/'measurement_key.json',r);rows.append(r)
    save(ART/'aux_auto_grid.json',rows)
if __name__=='__main__':main()
