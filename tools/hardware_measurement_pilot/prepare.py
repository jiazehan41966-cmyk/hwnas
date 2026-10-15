"""Generate reproducible integer fixtures and run the identical portable HLS C++."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from tools.hardware_measurement_pilot.inventory import build_inventory
from tools.hardware_measurement_pilot import reference as ref

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEFAULT=ROOT/'results/hardware_measurement_pilot_20261012/attempt03'
GXX=Path('D:/software/clion/CLion 2022.3.1/bin/mingw/bin/g++.exe')

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def numeric_contract(bits,wb,output_bits):
    return dict(version='experiment3_integer_v2',input_bits=bits,output_bits=output_bits,weight_bits=wb,
                accumulator_bits=32,overflow='twos_complement_modulo_2pow32',rounding='nearest_even',
                zero_point=0,requant_product_bits=64,multiplier_range=[1,2**31-1],shift_range=[0,31],
                add='saturating_input_alignment_then_int32_sum_then_output_requant',
                concat='each_input_requant_to_output_then_channel_concatenate',
                parent_sources={p:sha(ROOT/p) for p in ['tools/fpga_h2/numeric.py','tools/fpga_h4/quant_contracts.py',
                    'tools/modnas_comparison/quantization_mixed.py','tools/hardware_measurement_pilot/reference.py']})

def prepare(output):
    if output.exists():
        raise FileExistsError(f'Preserve existing attempt; choose new --output: {output}')
    start=time.perf_counter();output.mkdir(parents=True)
    inventory=build_inventory();save(output/'inventory.json',inventory)
    cases=[]
    for ci,entry in enumerate(inventory['pilot_cases']):
        begun=time.perf_counter();case=dict(entry);folder=output/'cases'/case['case_id'];folder.mkdir(parents=True)
        op=case['operator'];shape=case['shape'];bits=case['input_bits'];outbits=case['output_bits'];wb=case['weight_bits'] or bits
        cin,h,w=shape['input_shapes'][0];cout,oh,ow=shape['output_shape'];k=shape.get('kernel',1)
        if op=='pw1x1' and (k!=1 or shape['stride']!=1 or shape['padding']!=0):
            raise ValueError('Only stride1 pad0 pointwise supported')
        conv=op in ('pw1x1','dw3','dw5');dw=op.startswith('dw');wpo=k*k if dw else cin if conv else 1
        nw=cout*wpo if conv else 1
        pe=case['mapping']['PE'];simd=case['mapping']['SIMD']
        # Elementwise kernels use PE lanes; SIMD is not an independent physical dimension.
        case['mapping']['effective_SIMD']=simd if op=='pw1x1' else 1
        case['mapping']['input_banks']=pe if dw else simd if op=='pw1x1' else pe
        case['mapping']['buffer_profile']='full_frame_sequential_load_compute_store_v1'
        case['mapping']['layout']='CHW_contiguous_little_endian'
        constants=dict(OP={'pw1x1':0,'dw3':1,'dw5':1,'nearest2x':2,'requant_add':3,'requant_concat':4,'body8_to_head16':5}[op],
            CI=cin,CO=cout,H=h,W=w,OH=oh,OW=ow,K=k,STRIDE=shape.get('stride',1),PAD=shape.get('padding',0),
            PE=pe,SIMD=simd,INPUT_BANKS=case['mapping']['input_banks'],DATA_BITS=outbits,
            INPUT_ELEMENTS=cin*h*w,OUTPUT_ELEMENTS=cout*oh*ow,WEIGHT_ELEMENTS=nw,WEIGHTS_PER_OUT=wpo,
            INPUT_TYPE='int8_t' if bits==8 else 'int16_t',OUTPUT_TYPE='int8_t' if outbits==8 else 'int16_t',WEIGHT_TYPE='int8_t' if wb==8 else 'int16_t')
        (folder/'config.h').write_text('#pragma once\n'+''.join(f'#define {key} {value}\n' for key,value in constants.items()),encoding='utf-8')
        for name in ('kernel.cpp','tb.cpp','build_shell.tcl','phase_timer.v'):
            if (HERE/name).exists():shutil.copyfile(HERE/name,folder/name)
        dtype=np.dtype('i1' if bits==8 else '<i2');outdtype=np.dtype('i1' if outbits==8 else '<i2');wdtype=np.dtype('i1' if wb==8 else '<i2')
        low,high=-(1<<(bits-1)),(1<<(bits-1))-1
        wlow,whigh=-(1<<(wb-1)),(1<<(wb-1))-1
        # Pair both mappings on byte-identical fixtures.
        seed=20261012+int(hashlib.sha256(f'{op}_{bits}_{outbits}_{wb}'.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed);fixtures=[]
        for index,mode in enumerate(('random_linear','random_activation','ties_even','overflow_extremes','saturate_shift0')):
            fixture='test'+str(index)
            a=rng.integers(low,high+1,(cin,h,w),dtype=np.int32).astype(dtype)
            b=rng.integers(low,high+1,(cin,h,w),dtype=np.int32).astype(dtype)
            ws=(cout,1 if dw else cin,k,k) if conv else (1,)
            weight=rng.integers(wlow,whigh+1,ws,dtype=np.int32).astype(wdtype)
            bias=rng.integers(-20000,20001,cout,dtype=np.int32)
            p=np.array([3,11 if outbits==8 else 16,1,0,1,1,0,12],dtype='<i4')
            if not conv:p[:6]=[1,1,1,0,1,1]
            if op=='body8_to_head16':p[:2]=[16,0]
            if mode=='random_activation' and conv:p[6]=2 if shape.get('activation')=='relu6' else 0
            if mode=='ties_even':
                pattern=np.array([-7,-5,-3,-1,1,3,5,7],dtype=dtype)
                a.flat[:]=np.resize(pattern,a.size);b.flat[:]=np.resize(pattern[::-1],b.size)
                weight.fill(0);bias[:]=np.resize(pattern,cout)
                p[:6]=[1,1,1,1,1,0]
            if mode=='overflow_extremes':
                a.fill(low);b.fill(high);weight.fill(wlow)
                bias[:]=np.resize(np.array([2**31-1,-2**31],dtype='<i4'),cout)
                p[:6]=[2**31-1,31,1,0,1,0]
            if mode=='saturate_shift0':
                a.flat[:]=np.resize(np.array([low,high],dtype=dtype),a.size)
                b.flat[:]=np.resize(np.array([high,low],dtype=dtype),b.size)
                p[:6]=[2**31-1,0,2**31-1,0,1,0]
            if conv:
                acc=ref.conv_accumulator(a,weight,bias,input_bits=bits,weight_bits=wb,
                    stride=shape.get('stride',1),padding=shape.get('padding',0),depthwise=dw)
                expected=ref.requantize(acc,int(p[0]),int(p[1]),bits=outbits)
                if p[6]:expected=np.maximum(expected,0).astype(outdtype)
                if p[6]==2:expected=np.minimum(expected,p[7]).astype(outdtype)
            elif op=='nearest2x':expected=ref.nearest2x(a,bits=bits)
            elif op=='body8_to_head16':expected=ref.requantize(a.astype(np.int32),int(p[0]),int(p[1]),bits=outbits)
            elif op=='requant_add':expected=ref.staged_add(a,b,a_scale=tuple(map(int,p[:2])),b_scale=tuple(map(int,p[2:4])),output_scale=tuple(map(int,p[4:6])),bits=bits)
            else:expected=ref.requant_concat([a,b],[tuple(map(int,p[:2])),tuple(map(int,p[2:4]))],bits=bits)
            assert expected.shape==(cout,oh,ow)
            values=dict(a=a,b=b,w=weight,bias=bias.astype('<i4'),params=p,expected=expected)
            files={}
            for suffix,values_array in values.items():
                name=f'{fixture}_{suffix}.bin';(folder/name).write_bytes(values_array.tobytes(order='C'));files[name]=sha(folder/name)
            fixtures.append(dict(fixture_id=fixture,mode=mode,parameters=p.tolist(),files=files,
                expected_elements=int(expected.size),expected_output_sha256=files[f'{fixture}_expected.bin']))
        (folder/'fixtures.txt').write_text(''.join(f"{row['fixture_id']}\n" for row in fixtures),encoding='utf-8')
        (folder/'run_hls.tcl').write_text('''proc measured {stage body} {
    set t [clock milliseconds]
    set rc [catch {uplevel 1 $body} value opts]
    set elapsed [expr {([clock milliseconds]-$t)/1000.0}]
    set f [open hls_stages.jsonl a]
    puts $f [format {"stage":"%s","wall_seconds":%.3f,"returncode":%d} $stage $elapsed $rc]
    close $f
    if {$rc} {return -options $opts $value}
}
open_project hls_project
set_top pilot_kernel
add_files kernel.cpp -cflags {-std=c++14 -I.}
add_files config.h
add_files -tb tb.cpp -cflags {-std=c++14 -I.}
add_files -tb fixtures.txt
foreach f [glob test*.bin] {add_files -tb $f}
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5
config_interface -m_axi_addr64
measured csim {csim_design -O}
measured hls_synthesis {csynth_design}
measured ip_export {export_design -format ip_catalog -vendor hwnas.org -library hls -version 1.0}
exit
'''.replace('puts $f [format {"stage"','puts $f [format {{"stage"').replace('"returncode":%d} $stage','"returncode":%d}} $stage'),encoding='utf-8')
        contract=numeric_contract(bits,case['weight_bits'],outbits);save(folder/'numeric_contract.json',contract)
        case.update(seed=seed,fixtures=fixtures,numeric_contract_sha256=digest(contract),
            precision_assumption='BODY8_TO_HEAD16' if op=='body8_to_head16' else 'W8A8' if bits==8 else 'W8A16',
            platform=dict(part='xczu7ev-ffvc1156-2-e',tool_version='2022.2',target_clock_ns=5),
            input_provenance='synthetic_integer_fixtures_no_trained_weights',
            hls_scope='new_pilot_kernel_not_historical_cost_row',phase_contract='1_load_2_onchip_compute_3_store_0_idle',
            phase_interface='ap_vld_latched',phase_schedule='HLS_protocol_fixed_ap_wait_each_transition',
            phase_counter_abi='P3M2',phase_overhead='fixed protocol transition cycles included; no unverified subtraction',
            preparation_wall_seconds=time.perf_counter()-begun,
            source_files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
        case['case_sha256']=digest(case);save(folder/'case.json',case);cases.append(case)
    save(output/'preparation.json',dict(status='PREPARED',cases=len(cases),wall_seconds=time.perf_counter()-start,
        command='python -m tools.hardware_measurement_pilot.prepare prepare',formal_cost_eligible=False))
    manifest(output)
    return dict(status='PREPARED',cases=len(cases),output=str(output))

def manifest(output):
    # Can add shell files after their independent review; rebind case and rerun native gate.
    previous_manifest=json.loads((output/'source_manifest.json').read_text(encoding='utf-8')) if (output/'source_manifest.json').exists() else None
    previous_summary=json.loads((output/'native_summary.json').read_text(encoding='utf-8')) if (output/'native_summary.json').exists() else None
    previous_manifest_sha=sha(output/'source_manifest.json') if previous_manifest else None
    cases=[]
    for folder in sorted((output/'cases').iterdir()):
        case=json.loads((folder/'case.json').read_text(encoding='utf-8'))
        for name in ('build_shell.tcl','phase_timer.v'):
            if (HERE/name).exists():shutil.copyfile(HERE/name,folder/name)
        files={name:sha(folder/name) for name in case['source_files']}
        for name in ('build_shell.tcl','phase_timer.v'):
            if (folder/name).exists():files[name]=sha(folder/name)
        case['source_files']=files;case.pop('case_sha256',None);case['case_sha256']=digest(case)
        save(folder/'case.json',case);cases.append(dict(case_id=case['case_id'],case_sha256=case['case_sha256']))
    files={}
    for entry in cases:
        folder=output/'cases'/entry['case_id'];case=json.loads((folder/'case.json').read_text(encoding='utf-8'))
        for name in [*case['source_files'],'case.json']:
            p=folder/name;files[p.relative_to(output).as_posix()]=sha(p)
    save(output/'source_manifest.json',dict(schema='hardware_measurement_pilot_source_v1',files=files,cases=cases))
    if previous_summary and previous_summary.get('status')=='PASS':
        # Updating an independently validated RTL shell changes bundle identity,
        # but does not require repeating already-passed C++ executions.
        excluded={'case.json','build_shell.tcl','phase_timer.v'}
        old_cpp={k:v for k,v in previous_manifest['files'].items() if Path(k).name not in excluded}
        new_cpp={k:v for k,v in files.items() if Path(k).name not in excluded}
        old_bindings={c['case_id']:c['case_sha256'] for c in previous_manifest['cases']}
        if old_cpp==new_cpp and previous_summary.get('source_manifest_sha256')==previous_manifest_sha and all(
            r['case_sha256']==old_bindings.get(r['case_id']) for r in previous_summary['cases']):
            new_bindings={c['case_id']:c['case_sha256'] for c in cases}
            for row in previous_summary['cases']:
                row.setdefault('original_tested_case_sha256',row['case_sha256'])
                row['case_sha256']=new_bindings[row['case_id']]
            previous_summary['identity_rebinding']=dict(reason='Only shell/RTL evidence added or changed; all C++ sources, config, fixtures byte-identical',
                previous_source_manifest_sha256=previous_manifest_sha,cpp_inputs_unchanged=True,shell_validated_by_native=False)
            previous_summary['source_manifest_sha256']=sha(output/'source_manifest.json');save(output/'native_summary.json',previous_summary)

def native(output,compiler=GXX):
    started=time.perf_counter();records=[]
    manifest_data=json.loads((output/'source_manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest_data['files'].items():
        if sha(output/path)!=expected:raise ValueError(f'Source changed: {path}')
    env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
    for entry in manifest_data['cases']:
        folder=output/'cases'/entry['case_id'];t=time.perf_counter()
        command=[str(compiler),'-std=c++14','-O2','-Wno-unknown-pragmas','-Wl,--stack,16777216','kernel.cpp','tb.cpp','-o','native.exe']
        cp=subprocess.run(command,cwd=folder,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        (folder/'native_compile.log').write_text(cp.stdout,encoding='utf-8');compile_seconds=time.perf_counter()-t
        row=dict(case_id=entry['case_id'],case_sha256=entry['case_sha256'],compile_seconds=compile_seconds,compile_returncode=cp.returncode)
        if cp.returncode==0:
            t=time.perf_counter();run=subprocess.run([str(folder/'native.exe')],cwd=folder,env=env,capture_output=True,text=True,timeout=180)
            (folder/'native_run.log').write_text(run.stdout+run.stderr,encoding='utf-8')
            row.update(run_seconds=time.perf_counter()-t,returncode=run.returncode)
            if run.returncode==0:
                result=json.loads((folder/'c_result.json').read_text());case=json.loads((folder/'case.json').read_text(encoding='utf-8'))
                expected_elements=sum(f['expected_elements'] for f in case['fixtures'])
                binary_pass=all(sha(folder/f"{f['fixture_id']}_actual.bin")==f['expected_output_sha256'] for f in case['fixtures'])
                row.update(result=result,binary_sha256_parity=binary_pass)
                row['status']='PASS' if result['mismatches']==0 and result['fixtures']==len(case['fixtures']) and result['elements']==expected_elements and binary_pass else 'FAIL'
        row.setdefault('status','FAIL');records.append(row)
        print(entry['case_id'],row['status'],flush=True)
        save(output/'native_summary.json',dict(status='RUNNING',cases=records))
    summary=dict(status='PASS' if all(r['status']=='PASS' for r in records) and len(records)==len(manifest_data['cases']) and records else 'FAIL',
        evidence_layer='NATIVE_CPP_PARITY_NOT_VITIS_CSIM_NOT_BOARD',compiler=str(compiler),
        source_manifest_sha256=sha(output/'source_manifest.json'),wall_seconds=time.perf_counter()-started,cases=records)
    save(output/'native_summary.json',summary)
    return dict(status=summary['status'],cases=len(records),wall_seconds=summary['wall_seconds'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','native','manifest']);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--compiler',type=Path,default=GXX);a=p.parse_args()
    output=a.output.resolve()
    result=prepare(output) if a.action=='prepare' else native(output,a.compiler) if a.action=='native' else manifest(output)
    print(json.dumps(result,ensure_ascii=False))
    if isinstance(result,dict) and result.get('status')=='FAIL':raise SystemExit(1)
if __name__=='__main__':main()
