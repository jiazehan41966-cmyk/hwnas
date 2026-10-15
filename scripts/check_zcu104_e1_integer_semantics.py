"""Compile actual e1 HLS C++ and compare against an independent scalar reference.

Uses Vitis arbitrary precision headers, no FPGA synthesis or training. Run on Linux.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'hls_lut_builder/scripts'))
from common import render_template

TB = r'''
#include "kernel.cpp"
#include <iostream>
#include <algorithm>
int main() {
    int mismatches=0, checked=0;
    for (int seed=0; seed<3; ++seed) {
        hls::stream<input_axis_t> ins;
        hls::stream<output_axis_t> outs;
        weight_t weights[WEIGHT_SIZE]; acc_t biases[BIAS_SIZE];
        int input[IN_H][IN_W][IN_CH];
        for (int j=0;j<WEIGHT_SIZE;++j) weights[j]=((j*13+seed*7)%7)-3;
        for (int j=0;j<BIAS_SIZE;++j) biases[j]=((j*3+seed)%9)-4;
        for(int h=0;h<IN_H;++h) for(int w=0;w<IN_W;++w) {
            data_t values[IN_CH];
            for(int c=0;c<IN_CH;++c) {
                input[h][w][c]=((h*19+w*11+c*7+seed*5)%25)-12;
                values[c]=input[h][w][c];
            }
            ins.write(make_axis_packet<input_axis_t,IN_CH>(values,h==IN_H-1 && w==IN_W-1));
        }
        test_kernel(ins,outs,weights,biases);
        for(int h=0;h<OUT_H;++h) for(int w=0;w<OUT_W;++w) {
            if(outs.empty()) return 20;
            auto packet=outs.read(); data_t got[OUT_CH];
            unpack_axis_channels<OUT_CH>(packet,got);
            if(bool(packet.last)!=(h==OUT_H-1 && w==OUT_W-1)) ++mismatches;
            int depth[IN_CH];
            for(int c=0;c<IN_CH;++c) {
                int sum=int(biases[c]);
                for(int kh=0;kh<K;++kh) for(int kw=0;kw<K;++kw) {
                    int ih=h*STRIDE+kh-PAD, iw=w*STRIDE+kw-PAD;
                    if(ih>=0 && ih<IN_H && iw>=0 && iw<IN_W)
                        sum+=input[ih][iw][c]*int(weights[c*K*K+kh*K+kw]);
                }
                depth[c]=std::min(6,std::max(0,sum));
            }
            for(int o=0;o<OUT_CH;++o) {
                int sum=int(biases[IN_CH+o]);
                for(int c=0;c<IN_CH;++c) sum+=depth[c]*int(weights[IN_CH*K*K+o*IN_CH+c]);
                if(USE_RESIDUAL) sum+=input[h][w][o];
                int expected=std::min(127,std::max(-128,sum));
                if(int(got[o])!=expected) ++mismatches;
                ++checked;
            }
        }
        if(!outs.empty() || !ins.empty()) return 21;
    }
    std::cout << "E1_WITNESS checked=" << checked << " mismatches=" << mismatches << "\n";
    return mismatches ? 1 : 0;
}
'''


def run(output: Path, vitis_include: Path):
    output.mkdir(parents=True, exist_ok=True)
    template=ROOT/'hls_lut_builder/templates/mbconv_e1_noexpand.cpp.tmpl'
    results=[]
    for cin,cout,stride,kernel in [(4,4,1,3),(4,8,2,3),(5,7,1,5)]:
        folder=output/f'i{cin}_o{cout}_s{stride}_k{kernel}'
        folder.mkdir(exist_ok=True)
        parameters=dict(top_function='test_kernel',in_channels=cin,out_channels=cout,
            expand_ratio=1,expanded_channels=cin,feature_h=7,feature_w=7,kernel_size=kernel,
            stride=stride,padding=kernel//2,out_h=(7+stride-1)//stride,out_w=(7+stride-1)//stride,
            array_partition_factor=4,output_parallelism=4,max_axis_bits=4096,pipeline_ii=1,
            residual_stream_depth=64)
        source=render_template(template.read_text(),parameters)
        (folder/'kernel.cpp').write_text(source)
        (folder/'tb.cpp').write_text(TB)
        command=['g++','-std=c++14','-O1','-Wno-unknown-pragmas','-I'+str(vitis_include),
            '-I'+str(ROOT/'hls_lut_builder/include'),str(folder/'tb.cpp'),'-o',str(folder/'witness'),'-lgmp']
        compile_result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (folder/'compile.log').write_text(compile_result.stdout+compile_result.stderr)
        if compile_result.returncode:
            results.append({'case':folder.name,'status':'COMPILE_FAILED','returncode':compile_result.returncode})
            break
        result=subprocess.run([str(folder/'witness')],capture_output=True,text=True,timeout=60)
        (folder/'witness.log').write_text(result.stdout+result.stderr)
        results.append({'case':folder.name,'status':'PASS' if result.returncode==0 else 'MISMATCH',
                        'returncode':result.returncode,'output':result.stdout.strip()})
    report={'status':'PASS' if len(results)==3 and all(x['status']=='PASS' for x in results) else 'FAIL',
        'template_sha256':hashlib.sha256(template.read_bytes()).hexdigest(),'cases':results,
        'boundary':'Integer kernel semantics and stream framing only; not PTQ accuracy, RTL cosimulation, or board execution.'}
    (output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vitis-include',type=Path,default=Path('/opt/Xilinx/Vitis_HLS/2022.2/include'))
    args=parser.parse_args()
    raise SystemExit(run(args.output.resolve(),args.vitis_include))
