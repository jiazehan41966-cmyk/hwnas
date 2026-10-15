import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('adapter', Path(__file__).resolve().parents[1]/'tools/detector_preflight/shell_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Backend:
    def __init__(self):
        self.events = []
        self.regs = {0x2c:0x444D4153, 0x30:0x10000, 0x38:256, 0x18:784,
                     0x1c:784, 0x04:1, 0x28:0, 0x20:784, 0x24:784}
        self.error = False

    def read(self, offset):
        return self.regs[offset]

    def write(self, offset, value):
        self.events.append(('write', value))

    def __getattr__(self, name):
        def event(*args):
            self.events.append((name, *args))
            if name == 'wait_dma':
                if self.error:
                    raise TimeoutError('DMA timeout')
                self.regs[4] = 4
        return event


def test_receive_armed_before_producer_and_cache_ownership():
    b = Backend()
    assert adapter.transfer(b, 0x100000, 0x200000, 25088, 25088)['status'] == 'TRANSFER_COMPLETE'
    assert [x[0] for x in b.events] == ['flush_input','prepare_receive','barrier','write',
        'start_receive','write','start_send','wait_dma','invalidate_receive']
    assert b.events[3] == ('write',6)
    assert b.events[5] == ('write',1)


def test_detector_frame_rejected_before_mmio_mutation():
    b = Backend()
    with pytest.raises(ValueError, match='word counts'):
        adapter.transfer(b, 0x100000, 0x200000, 409600, 8416)
    assert not b.events


def test_timeout_aborts_before_cpu_can_consume_output():
    b = Backend()
    b.error = True
    with pytest.raises(TimeoutError):
        adapter.transfer(b, 0x100000, 0x200000, 25088, 25088)
    assert b.events[-1] == ('abort',)
    assert 'invalidate_receive' not in [e[0] for e in b.events]


def test_reject_aliasing_buffers():
    with pytest.raises(ValueError, match='overlap'):
        adapter.transfer(Backend(), 0x100000, 0x100040, 25088, 25088)


def test_framing_error_aborts():
    b = Backend()
    b.regs[0x28] = 1
    with pytest.raises(RuntimeError, match='framing'):
        adapter.transfer(b, 0x100000, 0x200000, 25088, 25088)
    assert b.events[-1] == ('abort',)
