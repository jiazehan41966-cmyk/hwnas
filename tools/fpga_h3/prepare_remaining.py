"""One conservative point per remaining critical shape; no mapping sweep."""
import sys,re,shutil
from common import *
import prepare as prep
import numpy as np
import torch

def main():
    hashes=contracts();torch.set_num_threads(4)
    existing=json.loads((ART/'grid.json').read_text());done={r['shape_id'] for r in existing};rows=[]
    info=json.loads((ART/'network_usage.json').read_text());prep.h2.OUT=ART/'extra_build'
    for label,rel in info['packages'].items():
        if label=='D':continue
        folder=ROOT/rel;graph=json.loads((folder/'graph.json').read_text());prep.GRAPH=graph
        _,values=prep.forward(folder,np.load(folder/'input.npy'));byid={n['id']:n for n in graph['nodes']}
        entries=[([n],'up' if n['op']=='upsample_add' else 'dw' if n.get('groups',1)>1 else 'conv') for n in graph['nodes']]
        entries += [([byid[f'backbone.stages.{s}.0.{t}_conv'] for t in ['expand','dw','project']],'mb') for s in [1,2,4,5,6,7]]
        for ns,kind in entries:
            desc=prep.description(ns,kind);key=shape_key(desc)
            if key in done:continue
            done.add(key);pe=2;simd=2 if kind=='conv' and desc['Cin']%2==0 else 1
            name=f'{key}_p{pe}_s{simd}';c=prep.h2.build_case(name,kind,ns,folder,graph,values)
            base=prep.h2.OUT/'representative_cases'/name;d=ART/'extra_cases'/name;shutil.copytree(base,d)
            src=(d/'kernel.cpp').read_text()
            for constant,value in [('IN_TILE',simd),('OUT_TILE',pe),('CH_TILE',pe),('EXP_TILE',pe)]:
                src=re.sub(r'(static constexpr int '+constant+r' = )[^;]+;',lambda m:m[1]+str(value)+';',src)
            (d/'kernel.cpp').write_text(src);(d/'run.tcl').write_text((d/'run.tcl').read_text().replace('exit\n','csynth_design\nexit\n'))
            sf={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']}
            linebytes=(c['ec'] if kind=='mb' else c['ci'])*c['k']*(c['w']+2*(c['k']//2))
            r=dict(point_id=name,shape_id=key,**desc,**hashes,input_bitwidth=8,weight_bitwidth=8,accumulator_bitwidth=32,output_bitwidth=8,
                PE=pe,SIMD=simd,expanded_PE=pe if kind=='mb' else 0,input_tile=simd,output_tile=pe,spatial_tile=f"{c['h']}x{c['w']}",buffer_profile='linebuffer_fifo4',
                part=PART,tool_version='2022.2',target_clock_ns=5,source_hash=hashobj(sf),source_files=sf,input_stream_bits=c['input_pack']*8,output_stream_bits=c['output_pack']*8,
                wide_adapter_contract=True,linebuffer_bytes=linebytes,weight_buffer_bytes=sum(int(np.prod(n.get('weight_shape',[0]))) for n in ns),ddr_scratch_bytes=0,
                sources=label,split='validation' if int(key[:4],16)%4==0 else 'fit',case_dir=str(d.relative_to(ROOT)),kind=kind,extra=True)
            reasons=legality(r,hashes);r['LEGAL_MAPPING_CHECK']='PASS' if not reasons else 'REJECT';r['rejection_reason']='|'.join(reasons)
            save(d/'measurement_key.json',r);rows.append(r)
        print('Remaining shapes prepared',label,len(rows),flush=True)
    save(ART/'extra_grid.json',rows)
    print('Remaining finite batch:',len(rows),flush=True)
if __name__=='__main__':main()
