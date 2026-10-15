import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('driver',Path(__file__).resolve().parents[1]/'tools/detector_preflight/pynq_dma_runner.py')
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Buffer:
    def __init__(self,address,nbytes,events,name):
        self.physical_address=address;self.nbytes=nbytes;self.events=events;self.name=name
    def flush(self):self.events.append((self.name,'flush'))
    def invalidate(self):self.events.append((self.name,'invalidate'))


class MMIO:
    def __init__(self,events,name):
        self.events=events;self.name=name;self.regs={};self.running=False;self.error=False;self.stall=False
    def write(self,offset,value):
        self.events.append((self.name,offset,value));self.regs[offset]=value
        if self.name=='dma' and offset==0 and value==4:
            self.regs[0]=0;self.running=False
        if self.name=='dma' and offset==0x28:self.running=True
    def read(self,offset):
        if self.name=='dma' and offset in (4,0x34):
            return 0x10 if self.error else (0 if self.stall and self.running else (2 if self.running else 1))
        if self.name=='core' and offset==0:return 6 if self.regs.get(0)==1 else 4
        return self.regs.get(offset,0)


def setup():
    events=[]
    buffers={name:Buffer(0x100000+i*0x100000,512,events,name) for i,name in enumerate(['input','output','weights','biases','numeric','low'])}
    abi=dict(weights=0x10,biases=0x1c,numeric=0x28,low=0x34,mode=0x40,return_=0x48)
    abi['return']=abi.pop('return_')
    return events,buffers,abi,MMIO(events,'dma'),MMIO(events,'core')


def test_real_driver_order_and_complete():
    events,buffers,abi,dma,core=setup()
    assert driver.transaction(dma,core,abi,buffers,0)>0
    assert events.index(('dma',0x58,512)) < events.index(('core',0,1)) < events.index(('dma',0x28,512))
    assert events[-1]==('output','invalidate')


def test_dma_error_rejects_completion():
    events,buffers,abi,dma,core=setup();dma.error=True
    with pytest.raises(RuntimeError):driver.transaction(dma,core,abi,buffers,0)


def test_timeout_is_bounded():
    events,buffers,abi,dma,core=setup();dma.stall=True
    with pytest.raises(TimeoutError):driver.transaction(dma,core,abi,buffers,0,timeout=0.005)


def test_no_producer_probe_resets_dma_with_core_idle():
    events,buffers,abi,dma,core=setup()
    result=driver.receive_timeout_probe(dma,core,buffers['output'])
    assert result['dma_channels_halted'] and result['core_idle']
    assert events[-1]==('dma',0,4)
    assert not any(e[:2]==('core',0) for e in events)
