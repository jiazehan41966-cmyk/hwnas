"""Prepare bounded, reproducible synthetic PW/DW HLS operator witnesses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools/fpga_h2'))
from numeric import conv_acc, requantize_int32_to_int8, activation, CONTRACT, aligned_add, ratio, q6, wrap32

OUT = ROOT / 'results/detector_v0_2_execution_20261002/hardware'
CONFIGS = {
    'pw48': dict(cin=48, cout=48, kernel=1, stride=1, padding=0, groups=1),
    'prediction21': dict(cin=48, cout=21, kernel=1, stride=1, padding=0, groups=1),
    'dw48': dict(cin=48, cout=48, kernel=3, stride=1, padding=1, groups=48),
    'dw48_stride2': dict(cin=48, cout=48, kernel=3, stride=2, padding=1, groups=48),
    'stem32': dict(cin=1, cout=32, kernel=3, stride=2, padding=1, groups=1),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def padded_nhwc(chw: np.ndarray) -> bytes:
    c, h, w = chw.shape
    a = np.zeros((h, w, (c + 15) // 16 * 16), dtype='i1')
    a[:, :, :c] = chw.transpose(1, 2, 0)
    return a.tobytes()


def packed_weights(weights: np.ndarray, groups: int) -> bytes:
    co, ci, k, _ = weights.shape
    if groups == co:
        a = np.zeros((co, 16), dtype='i1')
        a[:, :9] = weights[:, 0].reshape(co, 9)
    else:
        a = np.zeros((co, k, k, (ci + 15) // 16 * 16), dtype='i1')
        a[:, :, :, :ci] = weights.transpose(0, 2, 3, 1)
    return a.tobytes()


def create_case(base: Path, name: str, cfg: dict, index: int) -> dict:
    folder = base / name
    folder.mkdir()
    for filename in ('tiled_ops.h', 'numeric_v1.h'):
        shutil.copyfile(HERE / filename if filename == 'tiled_ops.h' else ROOT / 'tools/fpga_h2/numeric_v1.h', folder / filename)
    top = 'v02_' + name
    args = ("const v02_word* input,const v02_word* weights,const int32_t* biases,"
            "v02_word* output,uint64_t* stats,int height,int width,int multiplier,"
            "int shift,int activation,int q6")
    call = (f"v02_depthwise<{cfg['cin']},{cfg['stride']}>" if cfg['groups'] == cfg['cin'] == cfg['cout']
            else f"v02_dense<{cfg['cin']},{cfg['cout']},{cfg['kernel']},{cfg['stride']},{cfg['padding']}>")
    interfaces = '''#pragma HLS INTERFACE m_axi port=input offset=slave bundle=activations depth=1228800
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=activations depth=1228800
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=parameters depth=4096
#pragma HLS INTERFACE m_axi port=biases offset=slave bundle=parameters depth=192
#pragma HLS INTERFACE m_axi port=stats offset=slave bundle=statistics depth=12
'''
    for p in ('input', 'weights', 'biases', 'output', 'stats', 'height', 'width', 'multiplier', 'shift', 'activation', 'q6', 'return'):
        interfaces += f'#pragma HLS INTERFACE s_axilite port={p} bundle=control\n'
    (folder / 'kernel.cpp').write_text('#include "tiled_ops.h"\nint ' + top + '(' + args + ') {\n' + interfaces +
         'return ' + call + '(input,weights,biases,output,stats,height,width,multiplier,shift,activation,q6);\n}\n', encoding='utf-8')
    rows = []
    rng = np.random.default_rng(4200 + index)
    # Real P2 dimensions once, odd shape/tile tails, nonlinearity, overflow stress.
    for ti, (h, w, mode, act, mult, shift) in enumerate([
        (160, 160, 'random', 'linear', 3, 13),
        (17, 19, 'random', 'relu', 1, 10),
        (1, 1, 'random', 'relu6', 3, 8),
        (7, 9, 'overflow_positive', 'linear', 1, 24),
        (9, 7, 'overflow_negative', 'linear', 1, 24),
    ]):
        ci, co, k = cfg['cin'], cfg['cout'], cfg['kernel']
        wi = 1 if cfg['groups'] == ci == co else ci
        x = rng.integers(-128, 128, (ci, h, w), dtype=np.int16).astype(np.int8)
        weights = rng.integers(-128, 128, (co, wi, k, k), dtype=np.int16).astype(np.int8)
        b = rng.integers(-20000, 20001, co, dtype=np.int32)
        if mode == 'overflow_positive':
            x.fill(127); weights.fill(127); b.fill(2**31 - 1)
        elif mode == 'overflow_negative':
            x.fill(-128); weights.fill(127); b.fill(-2**31)
        acc = conv_acc(x, weights, b, cfg['stride'], cfg['padding'], cfg['groups'])
        y = activation(requantize_int32_to_int8(acc, mult, shift), act, -1)
        prefix = 'test' + str(ti)
        (folder / (prefix + '_input.bin')).write_bytes(padded_nhwc(x))
        (folder / (prefix + '_weights.bin')).write_bytes(packed_weights(weights, cfg['groups']))
        (folder / (prefix + '_bias.bin')).write_bytes(b.astype('<i4').tobytes())
        (folder / (prefix + '_reference.bin')).write_bytes(padded_nhwc(y))
        oh, ow = y.shape[1:]
        row = dict(id=prefix, height=h, width=w, output_height=oh, output_width=ow,
                   input_words=len(padded_nhwc(x)) // 16,
                   weight_words=len(packed_weights(weights, cfg['groups'])) // 16,
                   output_words=len(padded_nhwc(y)) // 16,
                   activation={'linear': 0, 'relu': 1, 'relu6': 2}[act],
                   multiplier=mult, shift=shift, q6=q6(-1), stress=mode,
                   logical_macs=oh * ow * ci * k * k * (co if cfg['groups'] == 1 else 1))
        rows.append(row)
    tests = ''
    for row in rows:
        tests += (f"errors+=run_test(\"{row['id']}\",{row['input_words']},{row['weight_words']},"
                  f"{row['output_words']},{cfg['cout']},{row['height']},{row['width']},"
                  f"{row['multiplier']},{row['shift']},{row['activation']},{row['q6']},"
                  f"{row['logical_macs']}ULL);\n")
    (folder / 'tb.cpp').write_text('''#include "tiled_ops.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
''' + 'int ' + top + '(' + args + ');\n' + '''
std::vector<v02_word> read_words(const std::string& path,int count) {
 std::ifstream f(path.c_str(),std::ios::binary);if(!f)throw std::runtime_error(path);
 std::vector<v02_word> values(count);
 for(int i=0;i<count;++i)for(int lane=0;lane<16;++lane){
  char c=0;f.get(c);if(!f)throw std::runtime_error("truncated "+path);
  values[i].range(lane*8+7,lane*8)=static_cast<unsigned char>(c);
 }
 return values;
}
int run_test(std::string prefix,int ni,int nw,int no,int co,int h,int w,int m,int s,int act,int q6,uint64_t macs){
 auto input=read_words(prefix+"_input.bin",ni);auto weight=read_words(prefix+"_weights.bin",nw);
 auto expected=read_words(prefix+"_reference.bin",no);std::vector<v02_word> out(no);
 std::vector<int32_t> bias(co);std::ifstream bf((prefix+"_bias.bin").c_str(),std::ios::binary);
 bf.read(reinterpret_cast<char*>(bias.data()),co*sizeof(int32_t));if(!bf)return 2;
 uint64_t stats[12]={0};
 int rc=''' + top + '''(input.data(),weight.data(),bias.data(),out.data(),stats,h,w,m,s,act,q6);
 uint64_t mismatches=0;std::ofstream raw((prefix+"_actual.bin").c_str(),std::ios::binary);
 for(int i=0;i<no;++i)for(int lane=0;lane<16;++lane){
  unsigned char a=out[i].range(lane*8+7,lane*8).to_uint();
  unsigned char e=expected[i].range(lane*8+7,lane*8).to_uint();
  raw.put(static_cast<char>(a));if(a!=e)++mismatches;
 }
 bool counters=stats[0]==macs && stats[5]==uint64_t(no) && stats[4]==uint64_t(nw);
 std::ofstream jf((prefix+"_csim.json").c_str());
 jf<<"{\\\"id\\\":\\\""<<prefix<<"\\\",\\\"returncode\\\":"<<rc<<",\\\"mismatches\\\":"<<mismatches
   <<",\\\"counters_match\\\":"<<(counters?"true":"false")<<",\\\"source_word_counters\\\":[";
 for(int i=0;i<12;++i){if(i)jf<<",";jf<<stats[i];}jf<<"]}\\n";
 std::cout<<prefix<<" mismatches="<<mismatches<<" counters="<<counters<<" rc="<<rc<<std::endl;
 return rc || mismatches || !counters;
}
int main(){int errors=0;
''' + tests + 'return errors?1:0; }\n', encoding='utf-8')
    tcl = ('open_project project\nset_top ' + top + '\n'
           'add_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\n'
           'add_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\n'
           'open_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\n'
           'create_clock -period 5.333\nconfig_interface -m_axi_addr64 -m_axi_alignment_byte_size 16\n')
    # Data files are copied by HLS into its csim working directory.
    tcl += ''.join('add_files -tb ' + p.name + '\n' for p in sorted(folder.glob('*.bin')))
    tcl += 'csim_design -O\ncsynth_design\nexit\n'
    (folder / 'run.tcl').write_text(tcl, encoding='utf-8')
    case = dict(name=name, top=top, config=cfg, pe=16, simd=8, dw_lanes=32,
                tile=[16, 16], input_layout='NHWC_C16_INT8_PACK128',
                output_layout='NHWC_C16_INT8_PACK128_zero_tail',
                tests=rows, numeric_contract=CONTRACT, clock_ns=5.333,
                part='xczu7ev-ffvc1156-2-e', scope='standalone synthetic operator, not graph or board')
    (folder / 'case.json').write_text(json.dumps(case, indent=2) + '\n', encoding='utf-8')
    return case


def numeric_boundary(base: Path) -> dict:
    folder = base / 'numeric_boundary'
    folder.mkdir()
    shutil.copyfile(ROOT / 'tools/fpga_h2/numeric_v1.h', folder / 'numeric_v1.h')
    records, refs = [], []
    for a in [-2**31, -257, -129, -128, -127, -5, -3, -2, -1, 0, 1, 2, 3, 5, 126, 127, 128, 257, 2**31-1]:
        for m in [1, 3, 2**31-1]:
            for shift in [0, 1, 2, 8, 31]:
                for act in ['linear', 'relu', 'relu6']:
                    records.append([0, a, 0, m, shift, {'linear': 0, 'relu': 1, 'relu6': 2}[act], q6(-1)] + [0]*9)
                    refs.append(activation(requantize_int32_to_int8(a, m, shift), act, -1).item())
    for a, b in [(2**31-1, 1), (-2**31, -1), (2**31-1, 2**31-1), (-2**31, -2**31)]:
        records.append([2, a, b, 1, 0, 0, 0]+[0]*9)
        refs.append(requantize_int32_to_int8(wrap32(np.int64(a)+b), 1, 0).item())
    for ea, eb, eo in [(-5,-3,-4),(-2,-5,-6),(-4,-4,-4),(-3,-1,0)]:
        common = max(ea, eb);params=list(ratio(ea,common)+ratio(eb,common)+ratio(common,eo))
        for a in [-128, -127, -5, -3, -1, 0, 1, 3, 5, 127]:
            for b in [-128, -127, -5, -3, -1, 0, 1, 3, 5, 127]:
                records.append([1,a,b]+params+[0]*7);refs.append(aligned_add(a,b,ea,eb,eo).item())
    np.asarray(records,dtype='<i4').tofile(folder/'vectors.bin')
    np.asarray(refs,dtype=np.int8).tofile(folder/'reference.bin')
    (folder/'kernel.cpp').write_text('''#include "numeric_v1.h"
void v02_numeric_boundary(const int* v,ap_int<8>* out,int n){
 for(int i=0;i<n;++i){const int* p=v+i*16;
 if(p[0]==1)out[i]=h2::add(ap_int<8>(p[1]),ap_int<8>(p[2]),p+3);
 else {ap_int<32> a=p[1];if(p[0]==2)a+=ap_int<32>(p[2]);out[i]=h2::finish(a,p+3);}
 }}
''')
    (folder/'tb.cpp').write_text('''#include "numeric_v1.h"
#include <fstream>
#include <iostream>
void v02_numeric_boundary(const int*,ap_int<8>*,int);
int main(){
''' + f'static int v[{len(records)*16}];static ap_int<8> y[{len(records)}];\n' + '''
 std::ifstream f("vectors.bin",std::ios::binary);f.read(reinterpret_cast<char*>(v),sizeof(v));if(!f)return 2;
''' + f'v02_numeric_boundary(v,y,{len(records)});\n' + '''
 std::ifstream ref("reference.bin",std::ios::binary);std::ofstream out("actual.bin",std::ios::binary);int errors=0;
''' + f'for(int i=0;i<{len(records)};++i)' + '''{char r;ref.get(r);char q=y[i].to_int();out.put(q);if(r!=q)++errors;}
 std::cout<<"boundary mismatches="<<errors<<std::endl;return errors?1:0;
}
''')
    (folder/'run.tcl').write_text('''open_project project
set_top v02_numeric_boundary
add_files kernel.cpp -cflags {-std=c++14 -O3 -I.}
add_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}
add_files -tb vectors.bin
add_files -tb reference.bin
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5.333
csim_design -O
exit
''')
    case=dict(name='numeric_boundary', samples=len(records), numeric_contract=CONTRACT, scope='arithmetic boundary csim only')
    (folder/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    return case


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',default='operator_v1')
    parser.add_argument('--cases',nargs='+',choices=list(CONFIGS),default=['pw48','prediction21','dw48','dw48_stride2','stem32'])
    args=parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('invalid attempt')
    base=OUT/args.attempt;base.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(HERE/'remote_run.py',base/'remote_run.py')
    cases=[numeric_boundary(base)]
    for i,name in enumerate(args.cases):cases.append(create_case(base,name,CONFIGS[name],i))
    (base/'prepared_manifest.json').write_text(json.dumps(dict(attempt=args.attempt,
       frozen_numeric_header_sha256=sha(ROOT/'tools/fpga_h2/numeric_v1.h'),
       kernels_header_sha256=sha(HERE/'tiled_ops.h'),cases=cases,
       files={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(output=str(base),cases=[c['name'] for c in cases])))


if __name__=='__main__':main()
