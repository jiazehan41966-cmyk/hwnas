"""Host sequencing adapter for the existing fixed-length shell ABI.

No MMIO backend is supplied: physical allocation, bitstream identity verification,
parameter loading and DMA driver binding remain deployment prerequisites.
The caller must verify these before supplying a backend. Not a detector executor.
"""
import time


def transfer(backend, source, destination, input_bytes, output_bytes, timeout_s=5.0):
    """backend methods are bounded; wait_dma(timeout) must check both channels.

    prepare_receive must clean/invalidate before DMA ownership, invalidate_receive
    must invalidate after completion. barrier orders CPU writes before DMA launch.
    On error/timeout abort must quiesce both DMA channels AND the core before any
    buffer is reused. Backend buffers/parameters are exclusively owned by caller.
    """
    if timeout_s <= 0:
        raise ValueError('positive timeout required')
    for address, length in ((source, input_bytes), (destination, output_bytes)):
        if type(address) is not int or address < 0 or address % 64:
            raise ValueError('physical address must be 64-byte aligned')
        if type(length) is not int or length <= 0 or length % 32 or length >= 2**26:
            raise ValueError('invalid DMA frame length')
    if max(source, destination) < min(source+input_bytes, destination+output_bytes):
        raise ValueError('input/output buffers overlap')
    if backend.read(0x2c) != 0x444D4153 or backend.read(0x30) != 0x10000:
        raise ValueError('shell identity/version mismatch')
    if backend.read(0x38) != 256:
        raise ValueError('unsupported AXIS width')
    if backend.read(0x18)*32 != input_bytes or backend.read(0x1c)*32 != output_bytes:
        raise ValueError('fixed shell word counts incompatible with transfer')
    status = backend.read(0x04)
    if not status & 1 or status & 2:
        raise RuntimeError('shell is not idle')
    deadline = time.monotonic() + timeout_s

    def remaining():
        seconds = deadline-time.monotonic()
        if seconds <= 0:
            raise TimeoutError('shell/DMA deadline exceeded')
        return seconds

    try:
        backend.flush_input(source, input_bytes)
        backend.prepare_receive(destination, output_bytes)
        backend.barrier()
        backend.write(0x00, 6)  # clear sticky done/error before a new transfer
        backend.start_receive(destination, output_bytes)
        backend.write(0x00, 1)
        backend.start_send(source, input_bytes)
        backend.wait_dma(remaining())
        while True:
            remaining()
            status = backend.read(0x04)
            if status & 8 or backend.read(0x28):
                raise RuntimeError('shell framing/core error')
            if status & 4:
                break
            time.sleep(min(0.001, remaining()))
        if backend.read(0x20)*32 != input_bytes or backend.read(0x24)*32 != output_bytes:
            raise RuntimeError('completed transfer byte-count mismatch')
        backend.invalidate_receive(destination, output_bytes)
        return dict(status='TRANSFER_COMPLETE', input_bytes=input_bytes, output_bytes=output_bytes)
    except BaseException:
        backend.abort()
        raise
