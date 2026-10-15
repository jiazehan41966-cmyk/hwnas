"""INTERFACE_CONTRACT_V1: compact HWC, frame-end zero padding, full DMA masks."""
from dataclasses import asdict,dataclass
import numpy as np

CONTRACT=dict(version=1,software_layout='CHW_batch1',stream_layout='HWC_channel0_low_byte',dma_bits=256,
    tail='zero_pad_frame_to_32_bytes_full_TKEEP_TSTRB',ddr_alignment=64,endianness='little',
    schedule='one_DMA_transaction_per_present_level_in_increasing_stride',base_address='caller_supplied_64byte_aligned_PS_DDR',
    completion='all_descriptor_padded_bytes_handshaken_TLAST_and_DMA_done',
    bridge_capacity='IN_BYTES+OUT_BYTES',fifo_depth=4,parameter_alignment=64)

@dataclass(frozen=True)
class DetectorOutputDescriptor:
    name:str
    stride:int
    channels:int
    height:int
    width:int
    anchors:list
    real_bytes:int
    padded_bytes:int
    ddr_offset:int

def align(n,a=64):return (n+a-1)//a*a

def descriptors(levels,anchors,base_address=0):
    if base_address%64:raise ValueError('DDR base must be 64-byte aligned')
    if tuple(levels) not in [('P3','P4','P5'),('P2','P3','P4'),('P2','P3','P4','P5')]:raise ValueError('unknown macro')
    result=[];offset=0
    for p in levels:
        stride=2**int(p[1]);h=640//stride;n=21*h*h
        result.append(DetectorOutputDescriptor(p,stride,21,h,h,anchors[p],n,align(n,32),offset))
        offset=align(offset+align(n,32))
    return result

def pack(x):
    raw=np.asarray(x,dtype=np.int8).transpose(1,2,0).copy().tobytes()
    return raw+bytes(align(len(raw),32)-len(raw))

def unpack(raw,shape):
    c,h,w=shape;real=c*h*w
    if len(raw)!=align(real,32) or any(raw[real:]):raise ValueError('bad padded frame')
    return np.frombuffer(raw[:real],dtype=np.int8).reshape(h,w,c).transpose(2,0,1).copy()

def wide_axis_to_dma256(words,word_bytes,real_bytes):
    if any(len(w)!=word_bytes for w in words):raise ValueError('bad native width')
    data=b''.join(words)
    if len(data)!=real_bytes:raise ValueError('bad frame length')
    data+=bytes(align(real_bytes,32)-real_bytes)
    return [data[i:i+32] for i in range(0,len(data),32)]

def dma256_to_wide_axis(beats,word_bytes,real_bytes):
    if real_bytes%word_bytes or any(len(b)!=32 for b in beats):raise ValueError('bad shape/beat')
    data=b''.join(beats)
    if len(data)!=align(real_bytes,32) or any(data[real_bytes:]):raise ValueError('bad tail')
    return [data[i:i+word_bytes] for i in range(0,real_bytes,word_bytes)]

def schedule_outputs(outputs,descriptor_list,base_address=0):
    """Executable serial DDR reference schedule; no allocation of physical DDR."""
    if base_address%64:raise ValueError('unaligned DDR base')
    if list(outputs)!=[d.name for d in descriptor_list]:raise ValueError('semantic output order mismatch')
    total=max(d.ddr_offset+d.padded_bytes for d in descriptor_list)
    memory=bytearray(total);events=[]
    for d in descriptor_list:
        x=outputs[d.name]
        if list(x.shape)!=[d.channels,d.height,d.width]:raise ValueError('descriptor shape mismatch')
        payload=pack(x)
        if len(payload)!=d.padded_bytes:raise ValueError('descriptor length mismatch')
        memory[d.ddr_offset:d.ddr_offset+d.padded_bytes]=payload
        events.append(dict(name=d.name,address=base_address+d.ddr_offset,real_bytes=d.real_bytes,padded_bytes=d.padded_bytes,
            expected_dma_beats=d.padded_bytes//32,TLAST_beat=d.padded_bytes//32-1,completion='reference_transfer_complete'))
    return bytes(memory),events
