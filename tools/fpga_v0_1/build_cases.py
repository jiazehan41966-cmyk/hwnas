"""Render exact repository templates and independent seeded integer witnesses."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'hls_lut_builder/scripts'))
from common import render_template

OUT = ROOT / 'reports/fpga_v0_1_closure/representative_cases'


def case(name, kind, ci, co, h, k=1, stride=1, relu=0, low=0, run=True):
    return dict(name=name, kind=kind, ci=ci, co=co, h=h, k=k, stride=stride,
                relu=relu, low=low, run=run, exp=ci*3 if kind=='mb' else ci)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [case('A_P3_lateral', 'conv', 32, 96, 80, relu=1),
             case('B_P2_lateral', 'conv', 24, 96, 160, relu=1),
             case('C_P2_tower', 'conv', 96, 96, 160, k=3, relu=1, run=False),
             case('D_head21_P5', 'conv', 96, 21, 20),
             case('D_head21_P2', 'conv', 96, 21, 160),
             case('F_upsample20_40', 'up', 96, 96, 40, low=20),
             case('F_upsample40_80', 'up', 96, 96, 80, low=40),
             case('F_upsample80_160', 'up', 96, 96, 160, low=80),
             case('DW3_stage1', 'dw', 48, 48, 320, k=3, stride=2, relu=1),
             case('DW5_stage1', 'dw', 48, 48, 320, k=5, stride=2, relu=1)]
    for stage,ci,co,h,s in [(1,16,24,320,2),(2,24,32,160,2),(4,32,64,80,2),(5,64,96,40,1),(6,96,160,40,2),(7,160,320,20,1)]:
        for k in [3,5]:
            cases.append(case(f'E_MB_stage{stage}_k{k}', 'mb', ci, co, h, k, s, run=stage==1))
    templates = dict(conv='standard_conv_packed', dw='dw_conv', mb='mbconv', up='upsample_add_packed')
    if len(sys.argv)>1:
        cases=[c for c in cases if c['name']==sys.argv[1]]
        if not cases: raise ValueError('Unknown case')
        for c in cases:c['run']=True
    for c in cases:
        folder=OUT/c['name']; folder.mkdir(exist_ok=True)
        ci,co,h,k,s,ec=c['ci'],c['co'],c['h'],c['k'],c['stride'],c['exp']
        oh=(h+2*(k//2)-k)//s+1
        c.update(oh=oh, seed=42, target_clock_ns=5, target_part='xczu7ev-ffvc1156-2-e',
                 template=templates[c['kind']], profile='fixed_pi8_po8_u1_axismax256', synthesis=c['name']=='D_head21_P5')
        parameters=dict(in_channels=ci,out_channels=co,expanded_channels=ec,feature_h=h,feature_w=h,
            kernel_size=k,stride=s,padding=k//2,out_h=oh,out_w=oh,input_parallelism=8,
            output_parallelism=8,array_partition_factor=8,max_axis_bits=256,pipeline_ii=1,
            apply_relu=c['relu'],source_h=c['low'],source_w=c['low'],residual_stream_depth=4,
            top_function='closure_kernel')
        source=render_template((ROOT/f"hls_lut_builder/templates/{c['template']}.cpp.tmpl").read_text(),parameters)
        if c['kind']=='up':
            source += '\nstatic_assert(LOW_H>0 && LOW_W>0 && OUT_H%LOW_H==0 && OUT_W%LOW_W==0, "integer resize required");\n'
        (folder/'kernel.cpp').write_text(source)
        for header in ['common_defs.h','common_types.h']:
            shutil.copy2(ROOT/'hls_lut_builder/include'/header,folder/header)
        # No AP_INT_MAX_W override: compile exactly the width contract of the existing templates.
        if c['run']:
            rng=np.random.default_rng(42)
            x=rng.integers(-3,4,(1,ci,h,h),dtype=np.int8)
            xt=torch.from_numpy(x).double()
            def rand(shape): return rng.integers(-2,3,shape,dtype=np.int8)
            def bias(n): return rng.integers(-10,11,n,dtype=np.int32)
            def conv(inp,w,b,stride=1,pad=0,groups=1):
                return F.conv2d(inp,torch.from_numpy(w).double(),torch.from_numpy(b).double(),stride,pad,groups=groups)
            if c['kind']=='mb':
                w1,w2,w3=rand((ec,ci,1,1)),rand((ec,1,k,k)),rand((co,ec,1,1))
                b1,b2,b3=bias(ec),bias(ec),bias(co)
                y=conv(xt,w1,b1).clamp(0,6)
                y=conv(y,w2,b2,s,k//2,ec).clamp(0,6)
                y=conv(y,w3,b3).clamp(-128,127)
                w=np.concatenate([w1.ravel(),w2.ravel(),w3.ravel()]); b=np.concatenate([b1,b2,b3])
                wdecl=f'[{len(w)}]'; wc=len(w)
            elif c['kind']=='up':
                low=rng.integers(-128,128,(1,ci,c['low'],c['low']),dtype=np.int16).astype(np.int8)
                x=rng.integers(-128,128,(1,ci,h,h),dtype=np.int16).astype(np.int8)
                xt=torch.from_numpy(x).double()
                y=(xt+F.interpolate(torch.from_numpy(low).double(),size=(h,h),mode='nearest')).clamp(-128,127)
                low.transpose(0,2,3,1).copy().tofile(folder/'low.bin')
                wc=0; b=np.empty(0,dtype=np.int32); wdecl=''
            else:
                w=rand((co,1 if c['kind']=='dw' else ci,k,k)); b=bias(co)
                y=conv(xt,w,b,s,k//2,ci if c['kind']=='dw' else 1).clamp(0 if c['relu'] else -128,6 if c['relu'] else 127)
                wc=w.size; wdecl=f'[{ci}][{k}][{k}]' if c['kind']=='dw' else f'[{co}][{ci}][{k}][{k}]'
            x.transpose(0,2,3,1).copy().tofile(folder/'input.bin')
            yn=y.numpy().astype(np.int8).transpose(0,2,3,1).copy()
            yn.tofile(folder/'reference.bin')
            if wc:
                w.tofile(folder/'weights.bin'); b.tofile(folder/'bias.bin')
            c.update(weight_count=wc,bias_count=b.size,input_pack=min(ci,32) if c['kind'] in ['conv','up'] else ci,
                     output_pack=min(co,32) if c['kind'] in ['conv','up'] else co)
            (folder/'tb.cpp').write_text(testbench(c,wdecl))
            tcl='open_project project\nset_top closure_kernel\nadd_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\n'
            for file in ['input.bin','reference.bin','weights.bin','bias.bin','low.bin']:
                if (folder/file).exists(): tcl+=f'add_files -tb {file}\n'
            tcl+='open_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\ncreate_clock -period 5\ncsim_design -O\n'
            if c['synthesis']: tcl+='csynth_design\n'
            tcl+='exit\n'
            (folder/'run.tcl').write_text(tcl)
        (folder/'case.json').write_text(json.dumps(c,indent=2))
    (OUT/('supplement_cases.json' if len(sys.argv)>1 else 'cases.json')).write_text(json.dumps(cases,indent=2))
    print('Rendered',len(cases),'cases;',sum(c['run'] for c in cases),'C-sim witnesses')


def testbench(c,wdecl):
    ci,co,h,oh,ip,op=c['ci'],c['co'],c['h'],c['oh'],c['input_pack'],c['output_pack']
    if c['kind']=='up':
        decl='void closure_kernel(hls::stream<IT>&,hls::stream<IT>&,hls::stream<OT>&);'
        prepare=f'hls::stream<IT> low; feed<IT,{ci},{ip}>(low,"low.bin",{c["low"]**2});'
        invoke='closure_kernel(input,low,output); if(!low.empty()) return 8;'
    else:
        decl=f'void closure_kernel(hls::stream<IT>&,hls::stream<OT>&,const weight_t {wdecl},const acc_t [{c["bias_count"]}]);'
        prepare=f'''static weight_t weights{wdecl}; static acc_t biases[{c['bias_count']}];
    auto wb=readbytes("weights.bin"); std::ifstream bf("bias.bin",std::ios::binary);
    for(int i=0;i<{c['weight_count']};++i) reinterpret_cast<weight_t*>(weights)[i]=static_cast<signed char>(wb[i]);
    for(int i=0;i<{c['bias_count']};++i) {{int32_t v;bf.read(reinterpret_cast<char*>(&v),4);biases[i]=v;}}'''
        invoke='closure_kernel(input,output,weights,biases);'
    return f'''#include "common_defs.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <iterator>
#include <cstdint>
typedef MAKE_AXIS_TYPE({ip*8}) IT;
typedef MAKE_AXIS_TYPE({op*8}) OT;
{decl}
std::vector<unsigned char> readbytes(const char* path) {{std::ifstream f(path,std::ios::binary); if(!f) throw 1;return {{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()}};}}
template<typename T,int C,int P> void feed(hls::stream<T>& stream,const char* path,int pixels) {{
 auto bytes=readbytes(path); if(bytes.size()!=pixels*C) throw 2;
 for(int px=0;px<pixels;++px) for(int base=0;base<C;base+=P) {{T packet;packet.data=0;packet.keep=-1;packet.strb=-1;
 for(int lane=0;lane<P;++lane) packet.data.range(lane*8+7,lane*8)=base+lane<C?bytes[px*C+base+lane]:0;
 packet.last=px==pixels-1 && base+P>=C; stream.write(packet);}}
}}
int main() {{
 hls::stream<IT> input;hls::stream<OT> output;
 feed<IT,{ci},{ip}>(input,"input.bin",{h*h});
 {prepare}
 {invoke}
 if(!input.empty()) return 3;
 auto ref=readbytes("reference.bin");std::ofstream result("hardware_output.bin",std::ios::binary);
 int errors=0, count=0, invalid=0, framing=0;
 for(int px=0;px<{oh*oh};++px) for(int base=0;base<{co};base+={op}) {{
 if(output.empty()) return 4;OT p=output.read();
 if(bool(p.last)!=(px=={oh*oh-1} && base+{op}>={co})) ++framing;
 if(p.keep!=decltype(p.keep)(-1)||p.strb!=decltype(p.strb)(-1)) ++framing;
 for(int lane=0;lane<{op};++lane) {{unsigned char byte=p.data.range(lane*8+7,lane*8).to_uint();
 if(base+lane<{co}) {{result.put(static_cast<char>(byte));if(byte!=ref[count++]) ++errors;}}
 else if(byte!=0) ++invalid;
 }} }}
 if(!output.empty()) return 5;
 std::cout<<"PARITY samples="<<count<<" mismatches="<<errors<<" invalid_nonzero="<<invalid<<" framing_errors="<<framing<<std::endl;
 return errors||invalid||framing?6:0;
}}
'''


if __name__=='__main__': main()
