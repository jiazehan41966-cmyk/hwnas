"""Boundary vectors include ties, both signs, modulo overflow and scale-aligned residual."""
import json,shutil
from pathlib import Path
import numpy as np
from numeric import requantize_int32_to_int8 as rq,activation,wrap32,ratio,aligned_add,q6
from build import OUT,HASHES

def main():
    d=OUT/'representative_cases/numeric_boundary';d.mkdir(exist_ok=True)
    records=[];refs=[]
    for a in [-2**31,-100000,-257,-129,-128,-127,-5,-3,-2,-1,0,1,2,3,5,126,127,128,257,100000,2**31-1]:
        for m in [1,3,2**31-1]:
            for s in [0,1,2,8,31]:
                for act in ['linear','relu','relu6']:
                    records.append([0,a,0,m,s,{'linear':0,'relu':1,'relu6':2}[act],q6(-1)]+[0]*9)
                    refs.append(activation(rq(a,m,s),act,-1).item())
    for a,b in [(2**31-1,1),(-2**31,-1),(2**31-1,2**31-1),(-2**31,-2**31)]:
        records.append([2,a,b,1,0,0,0]+[0]*9);refs.append(rq(wrap32(np.int64(a)+b),1,0).item())
    for ea,eb,eo in [(-5,-3,-4),(-2,-5,-6),(-4,-4,-4),(-3,-1,0)]:
        common=max(ea,eb);params=list(ratio(ea,common)+ratio(eb,common)+ratio(common,eo))
        for a in [-128,-127,-5,-3,-1,0,1,3,5,127]:
            for b in [-128,-127,-5,-3,-1,0,1,3,5,127]:
                records.append([1,a,b]+params+[0]*7);refs.append(aligned_add(a,b,ea,eb,eo).item())
    np.asarray(records,dtype='<i4').tofile(d/'vectors.bin');np.asarray(refs,dtype=np.int8).tofile(d/'reference.bin')
    shutil.copyfile(Path(__file__).parent/'numeric_v1.h',d/'numeric_v1.h')
    (d/'kernel.cpp').write_text('''#include "numeric_v1.h"
void closure_kernel(const int *v,ap_int<8>* output,int n){
for(int i=0;i<n;++i){const int*p=v+i*16;
if(p[0]==1)output[i]=h2::add(ap_int<8>(p[1]),ap_int<8>(p[2]),p+3);
else {ap_int<32> a=p[1];if(p[0]==2)a+=ap_int<32>(p[2]);output[i]=h2::finish(a,p+3);}
}}
''')
    (d/'tb.cpp').write_text(f'''#include "numeric_v1.h"
#include <fstream>
#include <iostream>
void closure_kernel(const int*,ap_int<8>*,int);
int main(){{static int v[{len(records)*16}];static ap_int<8> y[{len(records)}];
std::ifstream f("vectors.bin",std::ios::binary);f.read(reinterpret_cast<char*>(v),sizeof(v));if(!f)return 2;
closure_kernel(v,y,{len(records)});std::ifstream ref("reference.bin",std::ios::binary);std::ofstream out("hardware_output.bin",std::ios::binary);int errors=0;
for(int i=0;i<{len(records)};++i){{char r;ref.get(r);char q=y[i].to_int();out.put(q);if(r!=q)++errors;}}
std::cout<<"BOUNDARY samples={len(records)} mismatches="<<errors<<std::endl;return errors?1:0;}}
''')
    (d/'run.tcl').write_text('open_project project\nset_top closure_kernel\nadd_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb vectors.bin\nadd_files -tb reference.bin\nopen_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\ncreate_clock -period 5\ncsim_design -O\nexit\n')
    (d/'case.json').write_text(json.dumps(dict(name='numeric_boundary',kind='boundary',samples=len(records),**HASHES,tool_version='2022.2',FPGA_part='xczu7ev-ffvc1156-2-e',clock_ns=5),indent=2))
    print('Boundary vectors',len(records))

if __name__=='__main__':main()
