"""Prepare isolated H2 PW/FPN DMA core cases with real H2 reference vectors."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_h2'))
from numeric import conv, aligned_add


def main():
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    for name in ('h2_dma.cpp','tb_dma.cpp','run_dma.tcl','build_dma_bd.tcl'):
        shutil.copy2(Path(__file__).parent/name,out/name)
    shutil.copy2(ROOT/'tools/fpga_h2/numeric_v1.h',out/'numeric_v1.h')
    manifest={}
    for label,case_name in [('pw','PW'),('fpn','upsample_add')]:
        folder=ROOT/'reports/fpga_h2_contract/representative_cases'/case_name
        case=json.loads((folder/'case.json').read_text())
        dst=out/label
        dst.mkdir()
        shape_in=(32,80,80) if label=='pw' else (96,160,160)
        shape_out=(96,80,80) if label=='pw' else (96,160,160)
        for src,shape in [('input',shape_in),('reference',shape_out)]:
            # H2 C-sim vectors are already HWC on disk (see original feed()).
            raw=np.fromfile(folder/(src+'.bin'),dtype=np.int8)
            if raw.size != int(np.prod(shape)):raise ValueError('H2 frame size mismatch')
            raw.tofile(dst/(src+'.bin'))
        if label=='pw':
            node=case['nodes'][0]
            weights=np.fromfile(folder/'weights.bin',dtype=np.int8)
            biases=np.fromfile(folder/'biases.bin',dtype='<i4')
            weights[node['weight_offset']:node['weight_offset']+3072].tofile(dst/'weights.bin')
            begin=node['bias_offset']//4
            biases[begin:begin+96].tofile(dst/'biases.bin')
            params=[node['multiplier'],node['shift'],{'linear':0,'relu':1,'relu6':2}[node['activation']],node['q6'],node['output_exp'],0]
            np.array(params,dtype='<i4').tofile(dst/'numeric.bin')
            np.zeros(614400,dtype=np.int8).tofile(dst/'low.bin')
            x=np.fromfile(dst/'input.bin',dtype=np.int8).reshape(80,80,32).transpose(2,0,1)
            w=np.fromfile(dst/'weights.bin',dtype=np.int8).reshape(96,32,1,1)
            b=np.fromfile(dst/'biases.bin',dtype='<i4')
            if hashlib.sha256(w.tobytes()).hexdigest()!=node['weight_hash']:raise ValueError('weight identity mismatch')
            if hashlib.sha256(b.tobytes()).hexdigest()!=node['bias_hash']:raise ValueError('bias identity mismatch')
            result=conv(x,w,b,node).transpose(1,2,0).copy().tobytes()
        else:
            shutil.copy2(folder/'low.bin',dst/'low.bin')
            shutil.copy2(folder/'add_params.bin',dst/'numeric.bin')
            np.zeros(3072,dtype=np.int8).tofile(dst/'weights.bin')
            np.zeros(96,dtype='<i4').tofile(dst/'biases.bin')
            x=np.fromfile(dst/'input.bin',dtype=np.int8).reshape(160,160,96).transpose(2,0,1)
            low=np.fromfile(dst/'low.bin',dtype=np.int8).reshape(80,80,96).transpose(2,0,1)
            node=case['nodes'][0]
            result=aligned_add(x,np.repeat(np.repeat(low,2,1),2,2),node['input_exp_a'],node['input_exp_b'],node['output_exp']).transpose(1,2,0).copy().tobytes()
        if result!=(dst/'reference.bin').read_bytes():raise ValueError('H2 vector layout/numeric mismatch')
        manifest[label]=dict(case=case_name,mode=0 if label=='pw' else 1,
            numeric_contract_hash=case['numeric_contract_hash'],interface_contract_hash=case['interface_contract_hash'],
            input_bytes=int(np.prod(shape_in)),output_bytes=int(np.prod(shape_out)),
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir()})
    manifest['scope']='new functional DMA mapping; no H3 cost reuse'
    manifest['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
