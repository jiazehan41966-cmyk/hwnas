"""Use a plain finite internal stream; AXIS packet types remain at external ports."""
import shutil
from common import *
def main():
    rows=[]
    for old in json.loads((ART/'aux_auto_grid.json').read_text()):
        r=dict(old);r['retry_of']=old['point_id'];r['point_id']=old['point_id'].replace('_r2','_r3');r['aux_stream']=True
        d=ART/'aux_stream_cases'/r['point_id'];shutil.copytree(ROOT/old['case_dir'],d)
        s=(d/'kernel.cpp').read_text()
        s=s.replace('typedef MAKE_AXIS_TYPE(tower_OUT_PACK_CH * 8) tower_output_axis_t;', 'typedef internal_packet_t tower_output_axis_t;')
        s=s.replace('typedef MAKE_AXIS_TYPE(head_IN_PACK_CH * 8) head_input_axis_t;', 'typedef internal_packet_t head_input_axis_t;')
        s=s.replace('#include "common_defs.h"','#include "common_defs.h"\nstruct internal_packet_t {ap_uint<256> data; ap_uint<32> keep; ap_uint<32> strb; ap_uint<1> last;};')
        for macro in ['DECLARE_AXIS_INTERFACE()','DECLARE_BRAM_WEIGHTS()','DECLARE_CONTROL_INTERFACE()']:s=s.replace(macro,'// Interface is owned by the external chain wrapper.')
        assert 'typedef internal_packet_t tower_output_axis_t;' in s and 'typedef internal_packet_t head_input_axis_t;' in s
        (d/'kernel.cpp').write_text(s);r['case_dir']=str(d.relative_to(ROOT));r['implementation_variant']='v1_chain_plain_internal_stream_auto_storage'
        r['source_files']={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']};r['source_hash']=hashobj(r['source_files'])
        assert not legality(r,contracts());save(d/'measurement_key.json',r);rows.append(r)
    save(ART/'aux_stream_grid.json',rows)
if __name__=='__main__':main()
