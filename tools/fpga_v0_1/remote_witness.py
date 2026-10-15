"""Run only isolated CPU/ONNX/HLS witnesses on the authorized existing server."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort

BASE=Path(__file__).resolve().parent
HLS='/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'


def main():
    results=BASE/'results';results.mkdir(exist_ok=True)
    versions=dict(python=platform.python_version(),onnxruntime=ort.__version__,numpy=np.__version__)
    for name,cmd in [('vitis_hls',[HLS,'-version']),('g++',['g++','--version'])]:
        p=subprocess.run(cmd,capture_output=True,text=True)
        (results/(name.replace('+','p')+'_version.log')).write_text(p.stdout+p.stderr)
        versions[name]=p.stdout+p.stderr
    (results/'environment.json').write_text(json.dumps(versions,indent=2))
    export=BASE/'export';input_x=np.load(export/'input.npy');rows=[]
    for path in sorted(export.glob('*.onnx')):
        meta=json.loads(path.with_suffix('.json').read_text())
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        names=[o.name for o in session.get_outputs()]
        assert names==[o['name'] for o in meta['outputs']]
        got=session.run(names,{'image':input_x})
        for info,actual in zip(meta['outputs'],got):
            assert list(actual.shape)==info['shape']
            reference=np.load(export/f"{path.stem}_{info['name']}_reference.npy")
            diff=np.abs(actual.astype(np.float64)-reference.astype(np.float64))
            passed=bool(np.allclose(actual,reference,rtol=1e-4,atol=2e-5))
            np.save(results/f"{path.stem}_{info['name']}_ort.npy",actual)
            rows.append(dict(case=path.stem,operator=info['name'],domain='FP32_ONNX_RUNTIME',status='NUMERICAL_PARITY_VERIFIED' if passed else 'BLOCKED',max_abs_error=float(diff.max()),mean_abs_error=float(diff.mean()),exact_match_ratio=float((actual==reference).mean())))
            assert passed,rows[-1]
        print('ORT',path.stem,'PASS',flush=True)
    with (results/'ort_parity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cases=json.loads((BASE/'cases/cases.json').read_text());states=[]
    for case in cases:
        d=BASE/'cases'/case['name'];state=dict(case=case['name'],clock_target_ns=5,status='HLS_TEMPLATE_AVAILABLE',tool='Vitis HLS 2022.2',measured={})
        states.append(state)
        cmd=['g++','-std=c++14','-O2','-Wno-unknown-pragmas','-I.','-I/opt/Xilinx/Vitis_HLS/2022.2/include','-fsyntax-only','kernel.cpp']
        p=subprocess.run(cmd,cwd=d,capture_output=True,text=True,timeout=120)
        (results/(case['name']+'_compile.log')).write_text(p.stdout+p.stderr)
        state.update(compile_command=cmd,compile_returncode=p.returncode)
        if p.returncode:
            state.update(status='BLOCKED',blocker='VENDOR_HEADER_COMPILE_FAILED')
        elif case['run']:
            cmd=[HLS,'-f','run.tcl'];start=time.monotonic()
            state['command']=cmd
            try:
                with (results/(case['name']+'_hls.log')).open('w') as log:
                    p=subprocess.run(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                state['returncode']=p.returncode
            except subprocess.TimeoutExpired:
                state['returncode']='TIMEOUT_300_SECONDS'
            state['wall_seconds']=time.monotonic()-start
            outputs=list(d.glob('project/solution1/csim/build/hardware_output.bin'))
            if outputs:
                actual=outputs[0].read_bytes();(results/(case['name']+'_hardware_output.bin')).write_bytes(actual)
                a=np.frombuffer(actual,dtype=np.int8).astype(np.int32)
                b=np.fromfile(d/'reference.bin',dtype=np.int8).astype(np.int32)
                if a.shape==b.shape:
                    diff=np.abs(a-b);state['parity']=dict(max_abs_error=int(diff.max()),mean_abs_error=float(diff.mean()),exact_match_ratio=float((a==b).mean()),samples=len(a),domain='INT8_LEGACY_RAW_ARITHMETIC')
            log=(results/(case['name']+'_hls.log')).read_text(errors='replace')
            if state.get('parity',{}).get('exact_match_ratio')==1 and 'framing_errors=0' in log and 'CSim done with 0 errors.' in log:
                state['status']='HLS_CSIM_VERIFIED'
            else:
                state.update(status='BLOCKED',blocker='CSIM_DID_NOT_PASS')
            xml=d/'project/solution1/syn/report/closure_kernel_csynth.xml'
            if xml.exists() and state['returncode']==0:
                (results/(case['name']+'_csynth.xml')).write_bytes(xml.read_bytes())
                report=d/'project/solution1/syn/report/closure_kernel_csynth.rpt'
                if report.exists(): (results/(case['name']+'_csynth.rpt')).write_bytes(report.read_bytes())
                root=ET.parse(xml).getroot()
                for key,xpath in {'latency_cycles':'.//SummaryOfOverallLatency/Worst-caseLatency','II':'.//SummaryOfOverallLatency/Interval-max','DSP':'.//AreaEstimates/Resources/DSP','LUT':'.//AreaEstimates/Resources/LUT','FF':'.//AreaEstimates/Resources/FF','BRAM_18K':'.//AreaEstimates/Resources/BRAM_18K','URAM':'.//AreaEstimates/Resources/URAM','estimated_clock_ns':'.//SummaryOfTimingAnalysis/EstimatedClockPeriod'}.items():
                    state['measured'][key]=root.findtext(xpath)
                state['synthesis_status']='HLS_SYNTH_VERIFIED'
        print(case['name'],state['status'],state.get('synthesis_status',''),flush=True)
        (results/'case_status.json').write_text(json.dumps(states,indent=2))
    (results/'DONE.json').write_text(json.dumps(dict(status='WITNESSES_FINISHED',count=len(states)),indent=2))


if __name__=='__main__':main()
