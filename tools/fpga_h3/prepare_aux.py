"""Two real detector chains and four V1 continuous-byte DMA repack witnesses."""
import sys,shutil,re
from common import *
import prepare as prep
import numpy as np
import torch

def finish(d,desc,kind,inputs,outputs,extra_fields=None):
    hashes=contracts();sf={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']}
    r=dict(point_id=d.name,shape_id=shape_key(desc),**desc,**hashes,input_bitwidth=8,weight_bitwidth=8,accumulator_bitwidth=32,output_bitwidth=8,
        PE=1,SIMD=1,input_tile=1,output_tile=1,spatial_tile=f"{desc['H']}x{desc['W']}",buffer_profile='linebuffer_fifo4',part=PART,tool_version='2022.2',target_clock_ns=5,
        source_hash=hashobj(sf),source_files=sf,input_stream_bits=inputs,output_stream_bits=outputs,wide_adapter_contract=True,linebuffer_bytes=0,weight_buffer_bytes=0,
        ddr_scratch_bytes=0,sources='baseline|B1|C|D',split='auxiliary',case_dir=str(d.relative_to(ROOT)),kind=kind,aux=True,LEGAL_MAPPING_CHECK='PASS',rejection_reason='',**(extra_fields or {}))
    save(d/'measurement_key.json',r);return r

def tcl(d):
    s='open_project project\nset_top closure_kernel\nadd_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\n'
    for p in d.glob('*.bin'):s+=f'add_files -tb {p.name}\n'
    s+='open_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\ncreate_clock -period 5\ncsim_design -O\ncsynth_design\nexit\n'
    (d/'run.tcl').write_text(s)

def main():
    torch.set_num_threads(4);rows=[];prep.h2.OUT=ART/'aux_build'
    for hw in [20,40,80,160]:
        d=ART/'aux_cases'/f'dma21_{hw}';d.mkdir(parents=True)
        for header in ['common_defs.h','common_types.h']:shutil.copyfile(ROOT/'hls_lut_builder/include'/header,d/header)
        data=np.random.default_rng(42+hw).integers(0,256,size=hw*hw*21,dtype=np.uint8).tobytes();padded=data+bytes((-len(data))%32)
        (d/'input.bin').write_bytes(data);(d/'reference.bin').write_bytes(padded)
        (d/'kernel.cpp').write_text('''#include "common_defs.h"
typedef MAKE_AXIS_TYPE(168) IT;
typedef MAKE_AXIS_TYPE(256) OT;
void closure_kernel(hls::stream<IT>& input_stream,hls::stream<OT>& output_stream){
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control
ap_uint<424> reservoir=0;int count=0;
for(int px=0;px<PIXELS;++px){
#pragma HLS PIPELINE II=1
IT in=input_stream.read();reservoir|=ap_uint<424>(in.data)<<(count*8);count+=21;
if(count>=32){OT out;out.data=reservoir.range(255,0);out.keep=-1;out.strb=-1;out.last=(px==PIXELS-1&&count==32);output_stream.write(out);reservoir>>=256;count-=32;}}
if(count){OT out;out.data=reservoir.range(255,0);out.keep=-1;out.strb=-1;out.last=1;output_stream.write(out);}
}
'''.replace('PIXELS',str(hw*hw)))
        (d/'tb.cpp').write_text('''#include "common_defs.h"
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
typedef MAKE_AXIS_TYPE(168) IT;typedef MAKE_AXIS_TYPE(256) OT;
void closure_kernel(hls::stream<IT>&,hls::stream<OT>&);
int main(){std::ifstream f("input.bin",std::ios::binary),g("reference.bin",std::ios::binary);std::vector<unsigned char>a{std::istreambuf_iterator<char>(f),{}},b{std::istreambuf_iterator<char>(g),{}};hls::stream<IT>i;hls::stream<OT>o;
for(int px=0;px<PIXELS;++px){IT v;v.keep=-1;v.strb=-1;v.last=px==PIXELS-1;for(int c=0;c<21;++c)v.data.range(c*8+7,c*8)=a[px*21+c];i.write(v);}closure_kernel(i,o);std::ofstream out("hardware_output.bin",std::ios::binary);int err=0;
for(int beat=0;beat<BEATS;++beat){if(o.empty())return 2;OT v=o.read();if(bool(v.last)!=(beat==BEATS-1)||v.keep!=ap_uint<32>(-1)||v.strb!=ap_uint<32>(-1))++err;for(int c=0;c<32;++c){unsigned char x=v.data.range(c*8+7,c*8).to_uint();out.put(char(x));if(x!=b[beat*32+c])++err;}}
std::cout<<"DMA errors="<<err<<std::endl;return err||!o.empty()||!i.empty();}
'''.replace('PIXELS',str(hw*hw)).replace('BEATS',str(len(padded)//32)))
        tcl(d);desc=dict(operator_type='DMARepack',Cin=21,Cout=32,H=hw,W=hw,kernel=1,stride=1,groups=1,activation='linear',expanded_channels=0)
        rows.append(finish(d,desc,'dma',168,256,dict(real_bytes=len(data),padded_bytes=len(padded),latency_scope='packing_interface_only')))
    for label,hw,j in [('baseline',20,2),('B1',160,0)]:
        folder=ART/('graph_'+label);g=json.loads((folder/'graph.json').read_text());_,values=prep.forward(folder,np.load(folder/'input.npy'));byid={n['id']:n for n in g['nodes']}
        n1=byid[f'head.towers.{j}.0'];n2=byid[f'head.towers.{j}.3']
        c1=prep.h2.build_case('chain_tower_'+str(hw),'conv',[n1],folder,g,values);c2=prep.h2.build_case('chain_head_'+str(hw),'conv',[n2],folder,g,values)
        d=ART/'aux_cases'/f'chain_tower_head_{hw}';d.mkdir(parents=True)
        a=prep.h2.OUT/'representative_cases'/c1['name'];b=prep.h2.OUT/'representative_cases'/c2['name']
        for p in ['common_defs.h','common_types.h','numeric_v1.h','weights.bin','biases.bin','layers.bin','input.bin']:shutil.copyfile(a/p,d/p)
        shutil.copyfile(b/'reference.bin',d/'reference.bin')
        srcs=[]
        for name,base,pe in [('tower',a,2),('head',b,1)]:
            s=(base/'kernel.cpp').read_text()
            for constant,value in [('IN_TILE',2),('OUT_TILE',pe)]:s=re.sub(r'(static constexpr int '+constant+r' = )[^;]+;',lambda m:m[1]+str(value)+';',s)
            # Include guards are resolved before namespaces to keep types shared.
            s=s.replace('#include "numeric_v1.h"','').replace('#include "common_defs.h"','')
            srcs.append('namespace '+name+' {\n'+s+'\n}\n')
        top='''#include "numeric_v1.h"
#include "common_defs.h"
'''+''.join(srcs)+'''
void closure_kernel(hls::stream<tower::input_axis_t>& input_stream,hls::stream<head::output_axis_t>& output_stream,const weight_t w1[96][96][3][3],const acc_t b1[96],const int n1[5],const weight_t w2[21][96][1][1],const acc_t b2[21],const int n2[5]){
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control
hls::stream<tower::output_axis_t> middle;
#pragma HLS STREAM variable=middle depth=4
#pragma HLS DATAFLOW
tower::closure_kernel(input_stream,middle,w1,b1,n1);
head::closure_kernel(middle,output_stream,w2,b2,n2);
}
'''
        (d/'kernel.cpp').write_text(top)
        tb1=prep.h2.tb(c1);tb2=prep.h2.tb(c2)
        # Reuse independently checked binary loaders from H2 with distinct array names.
        start=tb1.index('static weight_t weights');end=tb1.index('closure_kernel(input,output,weights')
        load1=tb1[start:end]
        start2=tb2.index('static weight_t weights');end2=tb2.index('closure_kernel(input,output,weights')
        load2=tb2[start2:end2]
        for token in ['weights','biases','numeric','wb','bb','mb','wi','bi']:
            load1=re.sub(r'\b'+token+r'\b',token+'1',load1);load2=re.sub(r'\b'+token+r'\b',token+'2',load2)
        # Restore filenames accidentally touched by identifier substitution.
        for token in ['weights','biases']:load1=load1.replace(token+'1.bin',token+'.bin');load2=load2.replace(token+'2.bin',token+'.bin')
        pre=tb2[:tb2.index('int main()')]
        pre=re.sub(r'void closure_kernel\([^;]+;', 'void closure_kernel(hls::stream<IT>&,hls::stream<OT>&,const weight_t[96][96][3][3],const acc_t[96],const int[5],const weight_t[21][96][1][1],const acc_t[21],const int[5]);',pre)
        tail=tb2[tb2.index('if(!input.empty())return 3;'):]
        (d/'tb.cpp').write_text(pre+f'int main(){{hls::stream<IT> input;hls::stream<OT> output;feed<IT,96,32>(input,"input.bin",{hw*hw});\n'+load1+load2+'closure_kernel(input,output,weights1,biases1,numeric1,weights2,biases2,numeric2);\n'+tail)
        tcl(d);desc=dict(operator_type='ChainTowerHead',Cin=96,Cout=21,H=hw,W=hw,kernel=3,stride=1,groups=1,activation='relu_linear',expanded_channels=96)
        rows.append(finish(d,desc,'chain',256,168,dict(chain_mapping='tower_PE2_SIMD2_head_PE1_SIMD2_FIFO4',latency_scope='chain_compute_excludes_DMA')))
    save(ART/'aux_grid.json',rows);print('Auxiliary points',len(rows))
if __name__=='__main__':main()
