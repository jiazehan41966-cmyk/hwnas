"""Bounded H3 extension: missing V0.1 DW5 and real W8A16/W16A16 head CSYNTH.

Reuses frozen H3 kernels and independent NumPy integer reference. Outputs live
outside historical H3 evidence. Synthetic inputs are numerical witnesses only.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from common import ROOT, sha, hashobj, contracts, PART
from candidate_graphs import candidate_graph, architectures
from cost_api import CostDatabase, atomic_usage
from quant_contracts import operator_contract

OUT=ROOT/'artifacts/sonar_joint_cost_v1'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def testbench(ci,co,h,k,stride,groups,ab,wb,ip,op):
    oh=(h+2*(k//2)-k)//stride+1
    shape=f'[{ci}][{k}][{k}]' if groups>1 else f'[{co}][{ci}][{k}][{k}]'
    wc=co*(1 if groups>1 else ci)*k*k
    return f'''#include "common_defs.h"
#include <fstream>
#include <vector>
#include <iterator>
#include <cstdint>
#include <iostream>
typedef MAKE_AXIS_TYPE({ip*ab}) IT;
typedef MAKE_AXIS_TYPE({op*ab}) OT;
void closure_kernel(hls::stream<IT>&,hls::stream<OT>&,const weight_t {shape},const acc_t[{co}],const int[5]);
std::vector<unsigned char> bytes(const char* p){{std::ifstream f(p,std::ios::binary);if(!f)throw 1;return {{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()}};}}
uint32_t read(const std::vector<unsigned char>& b,int i,int n){{uint32_t v=0;for(int j=0;j<n;++j)v|=uint32_t(b.at(i*n+j))<<(j*8);return v;}}
int main(){{
hls::stream<IT> input;hls::stream<OT> output;auto x=bytes("input.bin"),w=bytes("weights.bin"),b=bytes("biases.bin"),p=bytes("numeric.bin");
static weight_t weights{shape};static acc_t biases[{co}];int numeric[5];
for(int i=0;i<{wc};++i)reinterpret_cast<weight_t*>(weights)[i]=read(w,i,{wb//8});
for(int i=0;i<{co};++i)biases[i]=read(b,i,4);
for(int i=0;i<5;++i)numeric[i]=read(p,i,4);
for(int px=0;px<{h*h};++px)for(int base=0;base<{ci};base+={ip}){{IT packet;packet.data=0;packet.keep=-1;packet.strb=-1;packet.last=px=={h*h-1}&&base+{ip}>={ci};
for(int lane=0;lane<{ip};++lane)if(base+lane<{ci})packet.data.range((lane+1)*{ab}-1,lane*{ab})=read(x,px*{ci}+base+lane,{ab//8});input.write(packet);}}
closure_kernel(input,output,weights,biases,numeric);
auto ref=bytes("reference.bin");std::ofstream f("hardware_output.bin",std::ios::binary);int count=0,errors=0;
for(int px=0;px<{oh*oh};++px)for(int base=0;base<{co};base+={op}){{if(output.empty())return 4;OT packet=output.read();
if(bool(packet.last)!=(px=={oh*oh-1}&&base+{op}>={co}))++errors;
for(int lane=0;lane<{op};++lane)if(base+lane<{co}){{unsigned v=packet.data.range((lane+1)*{ab}-1,lane*{ab}).to_uint();
for(int j=0;j<{ab//8};++j){{unsigned char c=(v>>(j*8))&255;f.put(char(c));if(c!=ref.at(count++))++errors;}}}}}}
if(!input.empty()||!output.empty()||count!=ref.size())return 5;
std::cout<<"integer witness mismatches="<<errors<<std::endl;return errors?6:0;}}
'''

def prepare():
    db=CostDatabase();hashes=contracts();OUT.mkdir(parents=True,exist_ok=True)
    numeric=load_module('joint_integer_reference',ROOT/'tools/fpga_h2/numeric.py')
    requests=[]
    # One measured mapping per head shape and precision; no fabricated scaling.
    heads={}
    for net in ('B1','C','D'):
        usage,_=atomic_usage(db.graphs[net])
        for u in usage:
            if u['node'].startswith('head.'):
                heads[u['shape_id']]=u
    for wb in (8,16):
        for sid,u in sorted(heads.items()):
            base=min(db.options[sid],key=lambda r:(r['PE']*r['SIMD'],r['point_id']))
            requests.append((base,u,wb,16))
    missing={}
    for aid in architectures():
        usage,_=atomic_usage(candidate_graph(aid,hashes))
        for u in usage:
            if u['shape_id'] not in db.options:missing[u['shape_id']]=u
    for sid,u in missing.items():
        # Only the missing stage1 DW5 is a permitted geometry change.
        if not (u['operator_type']=='DW' and u['Cin']==48 and u['H']==320 and u['kernel']==5):
            raise ValueError('Unplanned missing shape; freeze a new measurement batch')
        pool=[r for rs in db.options.values() for r in rs if r['operator_type']=='DW' and r['Cin']==48 and r['H']==320 and r['kernel']==3]
        requests.append((min(pool,key=lambda r:r['PE']),u,8,8))
    grid=[]
    for base,u,wb,ab in requests:
        name=f"{u['shape_id']}_w{wb}a{ab}_p{base['PE']}_s{base['SIMD']}"
        d=OUT/'cases'/name
        if (d/'measurement_key.json').exists():
            key=json.loads((d/'measurement_key.json').read_text())
            if any(sha(d/n)!=v for n,v in key['source_files'].items()):raise ValueError('Prepared case drift')
            grid.append(key);continue
        d.mkdir(parents=True,exist_ok=True)
        source=ROOT/base['case_dir']
        for f in ('kernel.cpp','common_defs.h','common_types.h','numeric_v1.h'):
            shutil.copy2(source/f,d/f)
        # All bitwidth edits are confined to a new measured case, never H2 V1.
        for f in ('common_types.h','common_defs.h','numeric_v1.h'):
            text=(d/f).read_text()
            if ab==16:
                text=re.sub(r'\b8\b','16',text).replace('127','32767').replace('-128','-32768')
            if f=='common_defs.h':text=text.replace('typedef data_t weight_t;',f'typedef ap_int<{wb}> weight_t;')
            (d/f).write_text(text)
        text=(d/'kernel.cpp').read_text()
        if ab==16:text=re.sub(r'\b8\b','16',text)
        text=re.sub(r'(static constexpr int K = )[^;]+;',lambda m:m[1]+str(u['kernel'])+';',text)
        text=re.sub(r'(static constexpr int PAD = )[^;]+;',lambda m:m[1]+str(u['kernel']//2)+';',text)
        (d/'kernel.cpp').write_text(text)
        ci,co,h,k,s,g=(u[x] for x in ('Cin','Cout','H','kernel','stride','groups'))
        ip=ci if g>1 else min(ci,256//ab);op=co if g>1 else min(co,256//ab)
        (d/'tb.cpp').write_text(testbench(ci,co,h,k,s,g,ab,wb,ip,op))
        rng=np.random.default_rng(42)
        x=rng.integers(-(1<<(ab-1)),1<<(ab-1),(ci,h,h),dtype=np.int32)
        w=rng.integers(-(1<<(wb-1)),1<<(wb-1),(co,1 if g>1 else ci,k,k),dtype=np.int32)
        b=rng.integers(-(1<<30),1<<30,co,dtype=np.int32)
        # Independent full-range reference exercises wrap32, nearest-even and saturation.
        acc=numeric.conv_acc(x,w,b,stride=s,padding=k//2,groups=g)
        shift=16 if ab==16 else 12
        y=np.clip(numeric.round_shift(acc,shift),-(1<<(ab-1)),(1<<(ab-1))-1)
        activation={'linear':0,'relu':1,'relu6':2}[u['activation']]
        if activation:y=np.maximum(y,0)
        if activation==2:y=np.minimum(y,96)
        dtype='<i2' if ab==16 else 'i1'
        x.astype(dtype).transpose(1,2,0).copy().tofile(d/'input.bin')
        y.astype(dtype).transpose(1,2,0).copy().tofile(d/'reference.bin')
        w.astype('<i2' if wb==16 else 'i1').tofile(d/'weights.bin');b.astype('<i4').tofile(d/'biases.bin')
        np.asarray([1,shift,activation,96,-4],dtype='<i4').tofile(d/'numeric.bin')
        tcl='open_project project\nset_top closure_kernel\nadd_files kernel.cpp -cflags {-std=c++14 -O3 -I.}\nadd_files -tb tb.cpp -cflags {-std=c++14 -O3 -I.}\n'
        for f in ('input.bin','reference.bin','weights.bin','biases.bin','numeric.bin'):tcl+=f'add_files -tb {f}\n'
        tcl+='open_solution solution1 -flow_target vivado\nset_part {xczu7ev-ffvc1156-2-e}\ncreate_clock -period 5\ncsim_design -O\ncsynth_design\nexit\n'
        (d/'run.tcl').write_text(tcl)
        fields=('operator_type','Cin','Cout','H','W','kernel','stride','groups','activation','expanded_channels')
        row={k:base[k] for k in ('PE','SIMD','input_tile','output_tile','spatial_tile','buffer_profile','part','tool_version','target_clock_ns')}
        row.update({k:u[k] for k in fields})
        row.update(operator_contract(wb,ab,hashes),point_id=name,shape_id=u['shape_id'],
            input_bitwidth=ab,weight_bitwidth=wb,accumulator_bitwidth=32,output_bitwidth=ab,
            kind='dw' if g>1 else 'conv',case_dir=str(d.relative_to(ROOT)).replace('\\','/'),
            LEGAL_MAPPING_CHECK='PASS',input_stream_bits=ip*ab,output_stream_bits=op*ab,
            numerical_witness='synthetic_full_range_seed42_not_detection_accuracy')
        row['source_files']={p.name:sha(p) for p in d.iterdir() if p.suffix in ('.h','.cpp','.tcl')}
        row['source_hash']=hashobj(row['source_files'])
        row['witness_files']={p.name:sha(p) for p in d.glob('*.bin')}
        save(d/'measurement_key.json',row);grid.append(row)
        print('Prepared',name,flush=True)
    save(OUT/'grid.json',grid)
    print('Bounded measurement batch:',len(grid),flush=True)

def collect():
    parser=load_module('joint_h3_collect',ROOT/'tools/fpga_h3/collect.py')
    rows=[];pending=[];failed=[]
    for key in json.loads((OUT/'grid.json').read_text()):
        d=ROOT/key['case_dir'];target=OUT/'results'/key['point_id']
        status=target/'status.json';report=target/'closure_kernel_csynth.xml'
        if not status.exists():pending.append(key['point_id']);continue
        st=json.loads(status.read_text());errors=[]
        if st['returncode']!=0 or not report.exists():errors.append('synthesis_failed')
        binary=target/'hardware_output.bin'
        if not st.get('csim_binary_exact') or not binary.exists() or sha(binary)!=key['witness_files']['reference.bin']:errors.append('parity_failed')
        if st['source_hash']!=key['source_hash']:errors.append('source_binding_failed')
        if any(sha(d/n)!=h for n,h in {**key['source_files'],**key['witness_files']}.items()):errors.append('case_drift')
        row=dict(key);row.pop('source_files',None)
        if report.exists():
            row.update(parser.parse_report(report))
            if row['part_reported']!=PART or row['clock_reported']!=5 or row['tool_reported']!='2022.2':errors.append('platform_mismatch')
        if errors:failed.append(dict(point_id=key['point_id'],errors=errors));continue
        row.update(status='VALID',tool_stage='CSYNTH',csim_status='PASS_BIT_EXACT',csynth_status='PASS',
            timing_status='ESTIMATED_TARGET_MET' if row['estimated_clock_ns']<=5 else 'ESTIMATED_TARGET_MISSED',
            source_report_path=str(report.relative_to(ROOT)).replace('\\','/'),source_report_sha256=sha(report),
            tool_log_path=str((OUT/'results'/(key['point_id']+'.log')).relative_to(ROOT)).replace('\\','/'))
        row['tool_log_sha256']=sha(ROOT/row['tool_log_path']);rows.append(row)
    save(OUT/'clean.json',rows)
    summary=dict(valid=len(rows),timing_eligible=sum(r['timing_status']=='ESTIMATED_TARGET_MET' for r in rows),pending=pending,failed=failed,
        deployment='NOT_CLAIMED',scope='operator_CSYNTH_and_CSIM_only')
    save(OUT/'status.json',summary);print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect']);args=p.parse_args()
    (prepare if args.mode=='prepare' else collect)()
