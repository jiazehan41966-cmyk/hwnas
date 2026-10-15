"""Freeze 65 tiled specializations and validate every full-sized integer fixture."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from tools.hardware_measurement_pilot.prepare import native as run_native
from tools.hardware_measurement_pilot.prepare import GXX

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEFAULT=ROOT/'results/hardware_template_coverage_20261013/attempt01'
OP={'pw1x1':0,'dw3':1,'dw5':1,'nearest2x':2,'requant_add':3,'body8_to_head16':5,'conv3x3':6,'maxpool':7}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def constants(case):
    s=case['shape'];ci,h,w=s['input_shapes'][0];co,oh,ow=s['output_shape']
    op=case['operator'];k=s.get('kernel',1);stride=s.get('stride',1);pad=s.get('padding',0)
    conv=op in ('pw1x1','dw3','dw5','conv3x3');dw=op in ('dw3','dw5')
    pe=case['mapping']['PE'];simd=case['mapping']['SIMD']
    th,tw=case['mapping'].get('tile_rows',2),case['mapping'].get('tile_cols',32)
    if op=='nearest2x':
        if th%2 or tw%2:raise ValueError('nearest tile origins must be even')
        ph,pw=th//2,tw//2
    else:ph,pw=(th-1)*stride+k,(tw-1)*stride+k
    wpo=(k*k if dw else ci*k*k) if conv else 1
    assert s['output_shape']==[co,oh,ow]
    if conv or op=='maxpool':
        if (oh,ow)!=((h+2*pad-k)//stride+1,(w+2*pad-k)//stride+1):raise ValueError('shape formula mismatch')
    if conv and not dw and s.get('groups',1)!=1:raise ValueError('unsupported dense groups')
    if dw and (ci!=co or s.get('groups')!=ci):raise ValueError('not depthwise')
    if conv and case['weight_bits']!=8:raise ValueError('only INT8 weights in current authorized scope')
    if op=='maxpool' and case['input_bits']!=case['output_bits']:raise ValueError('pool preserves raw integer bitwidth')
    if op in ('requant_add','body8_to_head16') and (k,stride,pad)!=(1,1,0):raise ValueError('elementwise geometry differs')
    return dict(OP=OP[op],CI=ci,CO=co,H=h,W=w,OH=oh,OW=ow,K=k,STRIDE=stride,PAD=pad,
        PE=pe,SIMD=simd,TILE_H=th,TILE_W=tw,PATCH_H=ph,PATCH_W=pw,
        INPUT_BITS=case['input_bits'],DATA_BITS=case['output_bits'],INPUT_ELEMENTS=ci*h*w,
        OUTPUT_ELEMENTS=co*oh*ow,WEIGHT_ELEMENTS=co*wpo if conv else 1,WEIGHTS_PER_OUT=wpo,
        INPUT_TYPE='int8_t' if case['input_bits']==8 else 'int16_t',
        OUTPUT_TYPE='int8_t' if case['output_bits']==8 else 'int16_t',WEIGHT_TYPE='int8_t')

HLS_TCL='''proc measured {stage body} {
    set t [clock milliseconds]
    set rc [catch {uplevel 1 $body} value opts]
    set elapsed [expr {([clock milliseconds]-$t)/1000.0}]
    set f [open hls_stages.jsonl a]
    puts $f [format {{"stage":"%s","wall_seconds":%.3f,"returncode":%d}} $stage $elapsed $rc]
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
'''

def prepare(output):
    from .inventory import build_inventory
    from .reference import make_fixture,fixture_modes
    if output.exists():raise FileExistsError(f'Use a new immutable attempt: {output}')
    started=time.perf_counter();output.mkdir(parents=True)
    inventory=build_inventory();save(output/'inventory.json',inventory)
    cases=[];files={'inventory.json':sha(output/'inventory.json')}
    for entry in inventory['pilot_cases']:
        begin=time.perf_counter();case=json.loads(json.dumps(entry));cid=case['case_id']
        folder=output/'cases'/cid;folder.mkdir(parents=True)
        conf=constants(case)
        (folder/'config.h').write_text('#pragma once\n'+''.join(f'#define {k} {v}\n' for k,v in conf.items()),encoding='utf-8')
        for name in ('kernel.cpp','build_shell.tcl','phase_timer.v'):
            shutil.copyfile(HERE/name,folder/name)
        shutil.copyfile(ROOT/'tools/hardware_measurement_pilot/tb.cpp',folder/'tb.cpp')
        (folder/'run_hls.tcl').write_text(HLS_TCL,encoding='utf-8')
        seed=20261013+int(case['primitive_id'][:8],16);fixtures=[]
        for i,mode in enumerate(fixture_modes):
            fid=f'test{i}';values=make_fixture(case,mode,seed+i)
            expected=values['expected'];assert tuple(expected.shape)==tuple(case['shape']['output_shape'])
            fixture_files={}
            for suffix,value in values.items():
                if suffix not in ('a','b','w','bias','params','expected'):raise ValueError('unexpected fixture field')
                name=f'{fid}_{suffix}.bin'
                folder.joinpath(name).write_bytes(np.ascontiguousarray(value).tobytes())
                fixture_files[name]=sha(folder/name)
            fixtures.append(dict(fixture_id=fid,mode=mode,parameters=values['params'].tolist(),files=fixture_files,
                expected_elements=int(expected.size),expected_output_sha256=fixture_files[f'{fid}_expected.bin']))
        (folder/'fixtures.txt').write_text(''.join(f'{f["fixture_id"]}\n' for f in fixtures),encoding='utf-8')
        contract=dict(version='t6_tiled_integer_v1',input_bits=case['input_bits'],output_bits=case['output_bits'],
            weight_bits=case['weight_bits'],accumulator_bits=32,overflow='twos_complement_modulo_2pow32',
            rounding='nearest_even',zero_point=0,requant_product_bits=64,multiplier_range=[1,2**31-1],shift_range=[0,31],
            add='saturating_input_alignment_then_int32_sum_then_output_requant',
            maxpool='raw_integer_max_no_requant_padding_signed_min',
            parent_sources={p:sha(ROOT/p) for p in ['tools/fpga_h2/numeric.py','tools/fpga_h4/quant_contracts.py',
                'tools/hardware_measurement_pilot/reference.py','tools/hardware_template_coverage_v1/reference.py']})
        save(folder/'numeric_contract.json',contract)
        case['mapping'].update(buffer_profile='spatial_tiles_ddr_frames_v1',layout='CHW_contiguous_little_endian',
            tile_rows=conf['TILE_H'],tile_cols=conf['TILE_W'],effective_SIMD=conf['SIMD'])
        case.update(seed=seed,fixtures=fixtures,numeric_contract_sha256=digest(contract),
            platform=dict(part='xczu7ev-ffvc1156-2-e',tool_version='2022.2',target_clock_ns=5),
            precision_assumption='BODY8_TO_HEAD16' if case['operator']=='body8_to_head16' else 'W8A8' if case['input_bits']==8 else 'W8A16',
            input_provenance='synthetic_integer_fixtures_no_trained_weights',
            hls_scope='new_tiled_implementation_not_historical_cost_row',
            phase_contract='1_load_2_onchip_compute_3_store_repeated_tiles_0_idle',
            phase_interface='ap_vld_latched',phase_schedule='HLS_protocol_fixed_ap_wait_each_transition',
            phase_counter_abi='P3T1',phase_overhead='fixed transition cycles included, no subtraction',
            preparation_wall_seconds=time.perf_counter()-begin,
            source_files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},formal_cost_eligible=False)
        case['case_sha256']=digest(case);save(folder/'case.json',case)
        cases.append(dict(case_id=cid,case_sha256=case['case_sha256']))
        for name in [*case['source_files'],'case.json']:files[f'cases/{cid}/{name}']=sha(folder/name)
        print(cid,'PREPARED',round(time.perf_counter()-begin,3),flush=True)
    save(output/'source_manifest.json',dict(schema='t6_tiled_source_v1',cases=cases,files=files))
    save(output/'preparation.json',dict(status='PREPARED',cases=len(cases),wall_seconds=time.perf_counter()-started,
        command=f'python -m tools.hardware_template_coverage_v1.prepare prepare --output {output}',formal_cost_eligible=False))
    return dict(status='PREPARED',cases=len(cases),output=str(output))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','native']);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--compiler',type=Path,default=GXX)
    a=p.parse_args();result=prepare(a.output.resolve()) if a.action=='prepare' else run_native(a.output.resolve(),a.compiler)
    print(json.dumps(result,ensure_ascii=False))
    if result['status']=='FAIL':raise SystemExit(1)
if __name__=='__main__':main()
