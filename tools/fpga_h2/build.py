"""Build three full integer graphs and contract-bound HLS numerical cases."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/fpga_v0_1'),str(ROOT/'hls_lut_builder/scripts')]
from export_contract import fold_candidate_bn
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate,SPEC_PATH
from common import render_template
from numeric import CONTRACT as NC,quantize_real,aligned_add
from interface import CONTRACT as IC
from integer_reference import PackageBuilder,forward

OUT=ROOT/'reports/fpga_h2_contract'
ART=ROOT/'artifacts/fpga_h2/20260916_v1'
HERE=Path(__file__).resolve().parent
def hashobj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
HASHES=dict(numeric_contract_hash=hashobj(NC),interface_contract_hash=hashobj(IC))
def savejson(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    savejson(OUT/'numeric_contract_v1.json',NC);savejson(OUT/'interface_contract_v1.json',IC)
    specs=[('A','P3_P4_P5',False),('B','P2_P3_P4',False),('C','P2_P3_P4_P5',True)]
    results=[];cases=[]
    for label,macro,mixed in specs:
        torch.manual_seed(42);rng=np.random.default_rng(42)
        stages=[1,2,4,5]+([] if macro=='P2_P3_P4' else [6,7])
        arch=dict(output_combination=macro,kernels={f'stage{s}':5 if mixed and j%2==0 else 3 for j,s in enumerate(stages)})
        model=build_sonar_candidate(SPEC_PATH,arch).eval()
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.running_mean.uniform_(-.1,.1);m.running_var.uniform_(.6,1.4);m.weight.uniform_(.8,1.2);m.bias.uniform_(-.05,.05)
        folded,pairs=fold_candidate_bn(model);folder=ART/('graph_'+label)
        meta=PackageBuilder(folded,folder,HASHES).build()
        torch.save(folded.state_dict(),folder/'folded_state.pt')
        image=quantize_real(rng.normal(0,1,(1,640,640)),-5);np.save(folder/'input.npy',image)
        outputs,values=forward(folder,image)
        for p,y in outputs.items():
            np.save(folder/(p+'.npy'),y)
            results.append(dict(graph=label,combination=macro,level=p,shape=list(y.shape),dtype=str(y.dtype),status='INTEGER_FORWARD_PASS',**HASHES))
        selected=[]
        if label=='A':selected=[('PW','conv',['neck.lateral_convs.0.0']),('head21_P5','conv',['head.towers.2.3'])]
        if label=='B':selected=[('P2_lateral','conv',['neck.lateral_convs.0.0']),('P2_tower','conv',['neck.output_convs.0.0']),('head21_P2','conv',['head.towers.0.3']),
            ('DW3','dw',['backbone.stages.1.0.dw_conv']),('MBConv3','mb',[f'backbone.stages.1.0.{p}_conv' for p in ['expand','dw','project']]),('upsample_add','up',['neck.fuse.0'])]
        if label=='C':selected=[('DW5','dw',['backbone.stages.1.0.dw_conv']),('MBConv5','mb',[f'backbone.stages.1.0.{p}_conv' for p in ['expand','dw','project']])]
        byid={n['id']:n for n in meta['nodes']}
        for name,kind,ids in selected:cases.append(build_case(name,kind,[byid[i] for i in ids],folder,meta,values))
        print('Integer graph',label,'PASS',list(outputs),flush=True)
    # Identical scale-alignment helper also tested on an explicit residual, unlike inactive V0.1 residual branches.
    savejson(OUT/'integer_graph_results.json',results);savejson(OUT/'representative_cases/cases.json',cases)
    print('HLS cases prepared:',len(cases),flush=True)

def build_case(name,kind,nodes,folder,graph,values):
    d=OUT/'representative_cases'/name;d.mkdir(parents=True,exist_ok=False)
    first,last=nodes[0],nodes[-1];x=values[first['inputs'][0]];y=values[last['id']]
    ci,h,w=x.shape;co,oh,ow=y.shape;k=nodes[1]['kernel'] if kind=='mb' else first.get('kernel',1);stride=nodes[1]['stride'] if kind=='mb' else first.get('stride',1)
    ec=nodes[0]['Cout'] if kind=='mb' else ci
    template={'conv':'standard_conv_packed','dw':'dw_conv','mb':'mbconv','up':'upsample_add_packed'}[kind]
    low=values[first['inputs'][1]] if kind=='up' else None
    pars=dict(in_channels=ci,out_channels=co,expanded_channels=ec,feature_h=h,feature_w=w,kernel_size=k,stride=stride,padding=k//2,out_h=oh,out_w=ow,input_parallelism=8,output_parallelism=8,array_partition_factor=8,max_axis_bits=256,pipeline_ii=1,apply_relu=0,source_h=low.shape[1] if low is not None else 0,source_w=low.shape[2] if low is not None else 0,residual_stream_depth=4,top_function='closure_kernel')
    src=render_template((ROOT/f'hls_lut_builder/templates/{template}.cpp.tmpl').read_text(),pars)
    src='#include "numeric_v1.h"\n'+src
    if kind in ['conv','dw']:
        src=src.replace('const acc_t biases[OUT_CH]\n','const acc_t biases[OUT_CH], const int numeric[5]\n').replace('const acc_t biases[IN_CH]\n','const acc_t biases[IN_CH], const int numeric[5]\n')
        src=src.replace('APPLY_RELU ? relu6_quant(acc_tile[lane]) : saturate_to_data(acc_tile[lane])','h2::finish(acc_tile[lane], numeric)')
    elif kind=='mb':
        for p,ch in [('expand','EXP_CH'),('depthwise','EXP_CH'),('project','OUT_CH')]:
            src=src.replace(f'const acc_t {p}_bias[{ch}]\n',f'const acc_t {p}_bias[{ch}], const int numeric[5]\n')
        src=src.replace('const acc_t biases[BIAS_SIZE]\n','const acc_t biases[BIAS_SIZE], const int numeric[15]\n')
        src=src.replace('relu6_quant(acc_tile[lane])','h2::finish(acc_tile[lane],numeric)')
        # V0.1 MBConv residual=false. Never retain legacy residual arithmetic in V1.
        start='''                        if (USE_RESIDUAL) {
                            acc += static_cast<acc_t>(residual_channels[oc]);
                        }
                        output_channels[oc] = saturate_to_data(acc);'''
        assert start in src
        src=src.replace(start,'                        output_channels[oc] = h2::finish(acc,numeric);')
        src+='\nstatic_assert(!USE_RESIDUAL,"Residual is an explicit separate aligned_add V1 node");\n'
        src=src.replace('expand_weights, expand_bias);','expand_weights, expand_bias, numeric);').replace('depthwise_weights, depthwise_bias);','depthwise_weights, depthwise_bias, numeric+5);').replace('project_weights, project_bias);','project_weights, project_bias, numeric+10);')
    else:
        src=src.replace('hls::stream<add_axis_t> &output_stream\n','hls::stream<add_axis_t> &output_stream, data_t low_buffer[CHANNELS][LOW_H][LOW_W], const int numeric[6]\n')
        src=src.replace('    data_t low_buffer[CHANNELS][LOW_H][LOW_W];','    #pragma HLS INTERFACE m_axi port=low_buffer offset=slave bundle=scratch\n    #pragma HLS INTERFACE s_axilite port=low_buffer bundle=control')
        src=src.replace('#pragma HLS ARRAY_PARTITION variable=low_buffer dim=1 cyclic factor=CH_TILE','// low frame is caller-owned external DDR scratch, not a full BRAM feature')
        src=src.replace('output_channels[c] = saturate_to_data(summed);','output_channels[c] = h2::add(lateral_channels[c],low_buffer[c][lh][lw],numeric);')
        src+='\nstatic_assert(OUT_H==2*LOW_H && OUT_W==2*LOW_W,"factor2 only");\n'
    assert 'h2::' in src
    (d/'kernel.cpp').write_text(src)
    for header in ['common_defs.h','common_types.h']:shutil.copyfile(ROOT/'hls_lut_builder/include'/header,d/header)
    shutil.copyfile(HERE/'numeric_v1.h',d/'numeric_v1.h')
    for f in ['weights.bin','biases.bin','layers.bin']:shutil.copyfile(folder/f,d/f)
    x.transpose(1,2,0).copy().tofile(d/'input.bin');y.transpose(1,2,0).copy().tofile(d/'reference.bin')
    if low is not None:
        low.transpose(1,2,0).copy().tofile(d/'low.bin')
        np.asarray(first['numeric'],dtype='<i4').tofile(d/'add_params.bin')
    c=dict(name=name,kind=kind,ci=ci,co=co,h=h,w=w,oh=oh,ow=ow,k=k,stride=stride,ec=ec,low_h=low.shape[1] if low is not None else 0,
        input_pack=min(ci,32) if kind in ['conv','up'] else ci,output_pack=min(co,32) if kind in ['conv','up'] else co,
        nodes=nodes,graph=folder.name,graph_metadata_hash=hashlib.sha256((folder/'graph.json').read_bytes()).hexdigest(),**HASHES,source_hash=hashlib.sha256(src.encode()).hexdigest(),
        tool_version='2022.2',FPGA_part='xczu7ev-ffvc1156-2-e',clock_ns=5,parity_kind='NUMERIC_CONTRACT_V1_PARITY')
    savejson(d/'case.json',c);(d/'tb.cpp').write_text(tb(c))
    tcl='open_project project\nset_top closure_kernel\nadd_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\n'
    for f in ['weights.bin','biases.bin','layers.bin','input.bin','reference.bin','low.bin','add_params.bin']:
        if (d/f).exists():tcl+=f'add_files -tb {f}\n'
    tcl+='open_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\ncreate_clock -period 5\ncsim_design -O\nexit\n'
    (d/'run.tcl').write_text(tcl)
    return c

def tb(c):
    ci,co,k,ec=c['ci'],c['co'],c['k'],c['ec'];nodes=c['nodes'];kind=c['kind']
    if kind=='up':
        decl=f'void closure_kernel(hls::stream<IT>&,hls::stream<IT>&,hls::stream<OT>&,data_t [{ci}][{c["low_h"]}][{c["low_h"]}],const int[6]);'
        prep=f'hls::stream<IT> low;feed<IT,{ci},{c["input_pack"]}>(low,"low.bin",{c["low_h"]**2});static data_t scratch[{ci}][{c["low_h"]}][{c["low_h"]}];int numeric[6];auto ab=bytes("add_params.bin");for(int z=0;z<6;++z)numeric[z]=le32(ab,z*4);'
        call='closure_kernel(input,low,output,scratch,numeric);if(!low.empty())return 11;'
    else:
        wc=sum(int(np.prod(n['weight_shape'])) for n in nodes);bc=sum(n['bias_count'] for n in nodes)
        wd=f'[{wc}]' if kind=='mb' else (f'[{ci}][{k}][{k}]' if kind=='dw' else f'[{co}][{ci}][{k}][{k}]')
        decl=f'void closure_kernel(hls::stream<IT>&,hls::stream<OT>&,const weight_t {wd},const acc_t[{bc}],const int[{5*len(nodes)}]);'
        prep=f'static weight_t weights{wd};static acc_t biases[{bc}];int numeric[{5*len(nodes)}];auto wb=bytes("weights.bin"),bb=bytes("biases.bin"),mb=bytes("layers.bin");int wi=0,bi=0;\n'
        for j,n in enumerate(nodes):
            fields=[n['Cin'],n['Cout'],n['H'],n['W'],n['kernel'],n['stride'],n['padding'],n['groups']]
            prep+=f'{{int rec[32];for(int z=0;z<32;++z)rec[z]=le32(mb,({n["layer_id"]}*32+z)*4);int expected[8]={{{",".join(map(str,fields))}}};for(int z=0;z<8;++z)if(rec[z+1]!=expected[z])return 20;'
            prep+=f'for(int z=0;z<rec[18];++z)reinterpret_cast<weight_t*>(weights)[wi++]=static_cast<signed char>(wb[rec[16]+z]);for(int z=0;z<rec[19];++z)biases[bi++]=le32(bb,rec[17]+4*z);for(int z=0;z<4;++z)numeric[{5*j}+z]=rec[12+z];numeric[{5*j+4}]=rec[11];}}\n'
        call='closure_kernel(input,output,weights,biases,numeric);'
    return f'''#include "common_defs.h"
#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <cstdint>
typedef MAKE_AXIS_TYPE({c['input_pack']*8}) IT;
typedef MAKE_AXIS_TYPE({c['output_pack']*8}) OT;
{decl}
std::vector<unsigned char> bytes(const char* p){{std::ifstream f(p,std::ios::binary);if(!f)throw 1;return {{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()}};}}
int32_t le32(const std::vector<unsigned char>& b,int i){{return int32_t(uint32_t(b.at(i))|(uint32_t(b.at(i+1))<<8)|(uint32_t(b.at(i+2))<<16)|(uint32_t(b.at(i+3))<<24));}}
template<typename T,int C,int P> void feed(hls::stream<T>& s,const char* file,int pixels){{auto b=bytes(file);if(b.size()!=C*pixels)throw 2;for(int px=0;px<pixels;++px)for(int base=0;base<C;base+=P){{T p;p.data=0;p.keep=-1;p.strb=-1;p.last=px==pixels-1&&base+P>=C;for(int lane=0;lane<P;++lane)if(base+lane<C)p.data.range(lane*8+7,lane*8)=b[px*C+base+lane];s.write(p);}}}}
int main(){{hls::stream<IT> input;hls::stream<OT> output;feed<IT,{ci},{c['input_pack']}>(input,"input.bin",{c['h']*c['w']});
{prep}
{call}
if(!input.empty())return 3;auto ref=bytes("reference.bin");std::ofstream f("hardware_output.bin",std::ios::binary);int count=0,errors=0,first=-1,frame=0;
for(int px=0;px<{c['oh']*c['ow']};++px)for(int base=0;base<{co};base+={c['output_pack']}){{if(output.empty())return 4;OT p=output.read();if(bool(p.last)!=(px=={c['oh']*c['ow']-1}&&base+{c['output_pack']}>={co}))++frame;
for(int lane=0;lane<{c['output_pack']};++lane)if(base+lane<{co}){{unsigned char v=p.data.range(lane*8+7,lane*8).to_uint();f.put(char(v));if(v!=ref.at(count)){{++errors;if(first<0)first=count;}}++count;}}}}
if(!output.empty())return 5;std::cout<<"V1 samples="<<count<<" mismatches="<<errors<<" first="<<first<<" framing="<<frame<<std::endl;return errors||frame?6:0;}}
'''

if __name__=='__main__':main()
