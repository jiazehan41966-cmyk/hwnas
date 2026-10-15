"""Real PYNQ DMA/HLS driver for the isolated h2_dma witness image.

Run inside a retained root Jupyter kernel, using run(payload_directory).
Never frees device-owned buffers after a failure; reset/recovery is explicit.
"""
import hashlib
import json
from pathlib import Path
import time

QUARANTINE = []


def poll(predicate, seconds, label):
    deadline = time.monotonic()+seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(label)
        time.sleep(0.001)


def reset_dma(dma, timeout=2):
    dma.write(0x00, 4)
    poll(lambda: not dma.read(0x00)&4, timeout, 'DMA reset timeout')
    if not dma.read(0x04)&1 or not dma.read(0x34)&1:
        raise RuntimeError('DMA reset did not halt both channels')


def pointer(mmio, offset, address):
    mmio.write(offset, address & 0xffffffff)
    mmio.write(offset+4, address >> 32)


def transaction(dma, core, abi, buffers, mode, timeout=30):
    source, target = buffers['input'], buffers['output']
    for name, buf in buffers.items():
        if int(buf.physical_address)%64:
            raise ValueError('unaligned DMA allocation')
        if name != 'output':buf.flush()
    # Output is CPU-initialized once; flush before device ownership.
    target.flush()
    target.invalidate()
    reset_dma(dma)
    if not core.read(0x00)&4:
        raise RuntimeError('HLS core is not idle')
    for name in ('weights','biases','numeric','low'):
        pointer(core, abi[name], int(buffers[name].physical_address))
    core.write(abi['mode'], mode)
    dma.write(0x04, 0x7000)
    dma.write(0x34, 0x7000)
    dma.write(0x30, 1)
    pointer(dma, 0x48, int(target.physical_address))
    dma.write(0x58, target.nbytes)
    start=time.monotonic_ns()
    core.write(0, 1)
    dma.write(0x00, 1)
    pointer(dma, 0x18, int(source.physical_address))
    dma.write(0x28, source.nbytes)
    done=False

    def complete():
        nonlocal done
        tx,rx=dma.read(0x04),dma.read(0x34)
        if (tx|rx)&0x770:
            raise RuntimeError('DMA error: '+hex(tx)+'/'+hex(rx))
        ctrl=core.read(0x00)
        done=done or bool(ctrl&2)  # AP_DONE is clear-on-read
        return bool(tx&2 and rx&2 and done and ctrl&4)

    poll(complete,timeout,'HLS/DMA transfer deadline')
    elapsed=time.monotonic_ns()-start
    if core.read(abi['return']):
        raise RuntimeError('H2 core rejected parameters or frame')
    if dma.read(0x58)!=target.nbytes:
        raise RuntimeError('S2MM actual length mismatch')
    target.invalidate()
    return elapsed


def receive_timeout_probe(dma, core, target):
    """Live bounded timeout witness with an idle core and no producer started.

    This proves DMA receive cancellation only, not recovery from a stuck AXI
    master. Device buffers remain quarantined on any failed recovery.
    """
    if not core.read(0)&4:raise RuntimeError('core must be idle for timeout probe')
    reset_dma(dma)
    target.flush();target.invalidate()
    dma.write(0x30,1)
    pointer(dma,0x48,int(target.physical_address))
    dma.write(0x58,target.nbytes)
    try:
        poll(lambda: bool(dma.read(0x34)&2),0.05,'deliberate receive timeout')
    except TimeoutError:
        reset_dma(dma)
        if not core.read(0)&4:raise RuntimeError('core state changed during idle probe')
        return dict(status='PASS',scope='S2MM_without_producer; idle_core',timeout_seconds=0.05,
            dma_channels_halted=True,core_idle=True)
    raise RuntimeError('unexpected receive completion without producer')


def run(payload_directory, timeout=30):
    import os
    import numpy as np
    from pynq import Overlay, MMIO, allocate
    if os.geteuid()!=0:
        raise PermissionError('PYNQ MMIO requires root execution channel')
    folder=Path(payload_directory)
    manifest=json.loads((folder/'board_manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('payload hash mismatch: '+filename)
    if manifest['timing_status']!='PASS':
        raise ValueError('route/timing evidence missing')
    overlay=Overlay(str(folder/'h2_dma.bit'),download=False)
    for name,kind in [('core','h2_dma'),('dma','axi_dma')]:
        if name not in overlay.ip_dict or kind not in overlay.ip_dict[name]['type']:
            raise ValueError('HWH IP identity mismatch')
    overlay.download()
    dma=MMIO(overlay.ip_dict['dma']['phys_addr'],overlay.ip_dict['dma']['addr_range'])
    core=MMIO(overlay.ip_dict['core']['phys_addr'],overlay.ip_dict['core']['addr_range'])
    results=[]
    for label,mode in [('pw',0),('fpn',1)]:
        buffers={}
        try:
            for name in ('input','weights','biases','numeric','low'):
                raw=np.fromfile(folder/label/(name+'.bin'),dtype=np.uint8)
                buf=allocate(shape=(raw.size,),dtype=np.uint8)
                buffers[name]=buf
                buf[:]=raw
            expected=np.fromfile(folder/label/'reference.bin',dtype=np.uint8)
            buffers['output']=allocate(shape=(expected.size,),dtype=np.uint8)
            buffers['output'][:]=0xa5
            elapsed=transaction(dma,core,manifest['abi'],buffers,mode,timeout)
            actual=np.asarray(buffers['output']).copy()
            mismatches=int(np.count_nonzero(actual!=expected))
            actual.tofile(folder/(label+'_board_output.bin'))
            row=dict(case=label,mode=mode,bytes=int(actual.size),mismatch_count=mismatches,
                end_to_end_submit_wait_ns=elapsed,latency_scope='single_transaction_no_cache_or_allocation',
                output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
            results.append(row)
            (folder/'board_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if mismatches:raise RuntimeError('integer parity failed')
        except BaseException:
            # DMA reset does not guarantee the HLS master is quiescent.
            # Retain all buffers until an explicit successful PL reset/reload.
            QUARANTINE.append(dict(buffers=buffers,overlay=overlay))
            try:reset_dma(dma)
            except Exception:pass
            raise
        else:
            for buf in buffers.values():buf.freebuffer()
    scratch=allocate(shape=(32,),dtype=np.uint8)
    scratch[:]=0
    try:
        timeout_result=receive_timeout_probe(dma,core,scratch)
        (folder/'timeout_probe.json').write_text(json.dumps(timeout_result,indent=2)+'\n')
        print(json.dumps(timeout_result),flush=True)
    except BaseException:
        QUARANTINE.append(dict(buffers={'scratch':scratch},overlay=overlay))
        raise
    else:scratch.freebuffer()
    return results
