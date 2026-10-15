"""Independent package, deterministic schedule and bounded FIFO contract checks."""
from collections import deque
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from integer_reference import forward
from interface import DetectorOutputDescriptor,schedule_outputs,unpack
from numeric import aligned_add,requantize_int32_to_int8
from build import ART,OUT,ROOT,HASHES

def main():
    sys.path.insert(0,str(ROOT/'tools/sonar_spec'))
    from candidate_sanity import protection
    checks={'baseline':protection(ROOT/'configs/baseline_sonar_frozen.yaml')}
    schedules={}
    for name in ['A','B','C']:
        p=ART/('graph_'+name);meta=json.loads((p/'graph.json').read_text())
        outputs,_=forward(p,np.load(p/'input.npy'))
        for level,y in outputs.items():assert np.array_equal(y,np.load(p/(level+'.npy')))
        ds=[DetectorOutputDescriptor(**d) for d in meta['descriptors']]
        memory,events=schedule_outputs(outputs,ds,base_address=0)
        for d in ds:assert np.array_equal(outputs[d.name],unpack(memory[d.ddr_offset:d.ddr_offset+d.padded_bytes],(21,d.height,d.width)))
        schedules[name]=events
        (p/'output_ddr.bin').write_bytes(memory)
    checks['binary_metadata_integer_forward_and_DDR_roundtrip']='3 graphs PASS'
    (OUT/'multihead_descriptors.json').write_text(json.dumps(schedules,indent=2))
    # Real HWC packet values, 4-entry queues; caller-owned low DDR, not BRAM.
    d=OUT/'representative_cases/upsample_add';meta=json.loads((d/'case.json').read_text())['nodes'][0]
    low=(d/'low.bin').read_bytes();lateral=(d/'input.bin').read_bytes();reference=(d/'reference.bin').read_bytes()
    qlow=deque();qlat=deque();qout=deque();scratch=bytearray(len(low));output=bytearray()
    li=lw=ai=processed=0;cycles=0;peaks=[0,0,0];backpressure=0;cap=4
    while len(output)<len(reference):
        cycles+=1
        if cycles>2000000:raise AssertionError('bounded schedule failed to complete')
        if qout and cycles%5!=0:output.extend(qout.popleft())
        if lw==len(low) and qlat and len(qout)<cap:
            packet=qlat.popleft();pixel=processed//3;seg=processed%3;row,col=divmod(pixel,160)
            offset=((row//2)*80+col//2)*96+seg*32
            y=aligned_add(np.frombuffer(packet,dtype=np.int8),np.frombuffer(scratch[offset:offset+32],dtype=np.int8),meta['input_exp_a'],meta['input_exp_b'],meta['output_exp'])
            qout.append(y.tobytes());processed+=1
        if qlow and cycles%3!=0:
            scratch[lw:lw+32]=qlow.popleft();lw+=32
        if li<len(low) and len(qlow)<cap and cycles%7!=0:qlow.append(low[li:li+32]);li+=32
        if lw==len(low) and ai<len(lateral) and len(qlat)<cap and cycles%11!=0:qlat.append(lateral[ai:ai+32]);ai+=32
        peaks=[max(a,len(q)) for a,q in zip(peaks,[qlow,qlat,qout])]
        backpressure+=int(any(len(q)==cap for q in [qlow,qlat,qout]))
        assert all(len(q)<=cap for q in [qlow,qlat,qout])
    assert bytes(output)==reference and not qout and not qlat and not qlow and backpressure>0
    checks['bounded_upsample']=dict(status='PASS',fifo_depth=4,observed_peaks=peaks,backpressure_steps=backpressure,external_low_DDR_bytes=len(scratch),
        logical_scheduler_steps=cycles,not_hardware_cycles=True,mismatch_count=0)
    checks['P2_tower']=dict(line_buffer_bytes=96*3*162,weight_bytes=96*96*3*3,full_feature_BRAM=False,
        ordering='source kernel fixed row/channel loop; acyclic input/core/output; finite FIFO4 may stall each producer',
        validation='HLS C-sim numeric plus source-local bounded line allocation; not full RTL timing verification')
    # Invalid numeric modes cannot silently select another formal rule.
    for kwargs in [dict(rounding_mode='truncate'),dict(saturation_range=(-127,127))]:
        try:requantize_int32_to_int8(1,1,1,**kwargs)
        except ValueError:pass
        else:raise AssertionError('invalid rule accepted')
    checks['V1_alternate_rounding_range_rejected']=True
    checks.update(HASHES);checks['status']='PASS'
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
