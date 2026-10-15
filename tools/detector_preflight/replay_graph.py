"""Execute the host DDR schedule with H2 integer arithmetic, not FPGA emulation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/fpga_h2'))
from integer_reference import forward
from numeric import conv, aligned_add
from prepare import plan_graph


def replay(folder, report):
    folder=Path(folder)
    graph=json.loads((folder/'graph.json').read_text())
    binaries={f:(folder/f).read_bytes() for f in ('weights.bin','biases.bin','layers.bin')}
    for name,raw in binaries.items():
        if hashlib.sha256(raw).hexdigest()!=graph['binary_hashes'][name]:raise ValueError('binary mismatch')
    image=np.load(folder/'input.npy',allow_pickle=False)
    _,gold=forward(folder,image)
    plan=plan_graph(graph,{k:len(v) for k,v in binaries.items()})
    memory=bytearray(plan['arena_required_bytes'])
    for name,raw in binaries.items():
        off=plan['parameters'][name]['offset'];memory[off:off+len(raw)]=raw

    def write(name,array):
        a=plan['allocations'][name]
        raw=array.transpose(1,2,0).copy().tobytes()
        if len(raw)!=a['real_bytes']:raise ValueError('bad write size')
        memory[a['offset']:a['offset']+a['padded_bytes']]=raw+bytes(a['padded_bytes']-len(raw))

    def read(name):
        a=plan['allocations'][name];c,h,w=a['shape'];off=a['offset']
        return np.frombuffer(memory,dtype=np.int8,count=a['real_bytes'],offset=off).reshape(h,w,c).transpose(2,0,1).copy()

    write('image',image)
    rows=[]
    for node in graph['nodes']:
        inputs=[read(name) for name in node['inputs']]
        for name,array in zip(node['inputs'],inputs):
            if not np.array_equal(array,gold[name]):raise AssertionError('input lifetime corruption: '+name)
        if node['op']=='conv':
            wo=plan['parameters']['weights.bin']['offset']+node['weight_offset']
            bo=plan['parameters']['biases.bin']['offset']+node['bias_offset']
            weights=np.frombuffer(memory,dtype=np.int8,count=int(np.prod(node['weight_shape'])),offset=wo).reshape(node['weight_shape'])
            biases=np.frombuffer(memory,dtype='<i4',count=node['bias_count'],offset=bo)
            result=conv(inputs[0],weights,biases,node)
        else:
            a,b=inputs
            if node['op']=='upsample_add':b=np.repeat(np.repeat(b,2,axis=1),2,axis=2)
            result=aligned_add(a,b,node['input_exp_a'],node['input_exp_b'],node['output_exp'])
        write(node['id'],result)
        errors=int(np.count_nonzero(read(node['id'])!=gold[node['id']]))
        rows.append(dict(node=node['id'],mismatch_count=errors,output_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
        if errors:raise AssertionError('node output mismatch')
    outputs={}
    for level,name in graph['outputs'].items():
        result=read(name)
        mismatch=int(np.count_nonzero(result!=np.load(folder/(level+'.npy'),allow_pickle=False)))
        if mismatch:raise AssertionError('retained H2 output mismatch')
        outputs[level]=dict(mismatch_count=mismatch,shape=list(result.shape))
    report=Path(report)
    report.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(status='HOST_INTEGER_SCHEDULE_PASS',graph_sha256=hashlib.sha256((folder/'graph.json').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((folder/'input.npy').read_bytes()).hexdigest(),seed=42,
        layers=rows,outputs=outputs,board_execution=False,rtl_cosimulation=False,
        numeric_independence='shared H2 arithmetic; this check validates scheduling and storage, not new arithmetic parity')
    with report.open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps(dict(graph=folder.name,layers=len(rows),status=payload['status'],outputs=outputs)),flush=True)


if __name__=='__main__':replay(sys.argv[1],sys.argv[2])
