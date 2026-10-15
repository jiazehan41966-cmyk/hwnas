"""Real runner control paths with fake PL/MMIO/buffers; never access hardware."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.hardware_template_coverage_v1 import board_runner as br


class Buffer(np.ndarray):
    def __new__(cls, values, dtype="i1"):
        value = np.asarray(values, dtype=dtype).copy().view(cls)
        value.physical_address = 0x12340000
        value.freed = value.flushed = value.invalidated = 0
        return value

    def freebuffer(self):
        self.freed += 1

    def flush(self):
        self.flushed += 1

    def invalidate(self):
        self.invalidated += 1


class GPIO:
    def __init__(self, zero=0, eight=0):
        self.values = {0: zero, 8: eight}

    def read(self, offset):
        return self.values[offset]


class Core:
    def __init__(self, session=None, complete=True):
        self.session, self.complete = session, complete
        self.control, self.started = 4, False
        self.writes = []
        self.on_poll = None

    def read(self, offset):
        if self.started and self.on_poll:
            self.on_poll()
        return self.control

    def write(self, offset, value):
        self.writes.append((offset, value))
        if offset == 0 and value == 1:
            self.started = True
            self.control = 6 if self.complete else 0
            if self.complete and self.session:
                self.session.buffers["output"][:] = self.session.current[1]["output"]
                self.session.gpios["timer_sequence"].values[8] += 1
                self.session.gpios["timer_identity"].values[8] = 1


class IdentityTests(unittest.TestCase):
    def setUp(self):
        br.QUARANTINE.clear()
        self.records = []

    def tearDown(self):
        # Only fake numpy buffers appear here; no device memory exists.
        br.QUARANTINE.clear()

    def session(self, complete=True):
        value = br.BoardSession.__new__(br.BoardSession)
        value.np = np
        value.pl = SimpleNamespace(bitfile_name="/known/top.bit", timestamp="epoch-original")
        value.pl_identity_lost = value.pending = value.closed = False
        value.device_state = "ORIGINAL_PL_IDENTITY_BOUND"
        value.pl_identity_checks = value.sequence = 0
        value.identity_record = self.records.append
        value.timeout_ns = 10**9
        value.clock_hz = 100000000
        value.gpios = dict(timer_identity=GPIO(br.P3T1_TIMER_MAGIC, 0),
                          timer_sequence=GPIO(30, 0), timer_counts=GPIO(10, 20))
        value.pl_identity = value._read_pl_identity()
        value.abi = dict(control_offset=0, masks=dict(AP_IDLE=4, AP_DONE=2, AP_START=1),
                         pointers=dict(a=dict(offset=0x10, high_offset=0x14),
                                       output=dict(offset=0x20, high_offset=0x24)))
        value.required = {"a", "output"}
        value.buffers = dict(a=Buffer([1, 2]), output=Buffer([99, 99]))
        value.owned = list(value.buffers.values())
        value.current = ({"fixture_id": "f0"}, dict(poison=np.array([1, 1], dtype="i1"),
                                                   output=np.array([0, 0], dtype="i1")))
        value.core = Core(value, complete)
        return value

    def assert_permanently_retained(self, session):
        original_buffers = list(session.owned)
        # A foreign idle register and even restoration of the observed old
        # state cannot undo loss of the original ownership proof.
        session.pl.bitfile_name = session.pl_identity["bitfile_name"]
        session.pl.timestamp = session.pl_identity["timestamp"]
        session.gpios["timer_identity"].values[0] = br.P3T1_TIMER_MAGIC
        session.core.control = 4
        with self.assertRaises(br.PLIdentityLost):
            session.close()
        self.assertFalse(session.closed)
        self.assertTrue(session.pending)
        self.assertEqual(session.device_state, "PL_IDENTITY_LOST")
        self.assertIn(session, br.QUARANTINE)
        self.assertEqual([buffer.freed for buffer in original_buffers], [0, 0])

    def test_magic_bitfile_epoch_and_none_drift_block_before_any_start(self):
        mutations = (
            lambda s: s.gpios["timer_identity"].values.update({0: 0x50334D32}),
            lambda s: setattr(s.pl, "bitfile_name", "/foreign/top.bit"),
            lambda s: setattr(s.pl, "timestamp", "epoch-foreign"),
            lambda s: setattr(s.pl, "timestamp", None),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                session = self.session()
                mutation(session)
                with self.assertRaises(br.PLIdentityLost):
                    session.invoke(np.empty(2, dtype="i1"))
                self.assertEqual(session.core.writes, [])
                self.assertEqual(self.records[-1]["stage"], "invoke_before")
                self.assert_permanently_retained(session)

    def test_reload_after_completion_prevents_output_capture_and_release(self):
        session = self.session()
        session.core.on_poll = lambda: setattr(session.pl, "timestamp", "foreign-epoch")
        capture = np.array([88, 88], dtype="i1")
        with self.assertRaises(br.PLIdentityLost):
            session.invoke(capture)
        self.assertEqual(capture.tolist(), [88, 88])
        self.assertEqual(session.buffers["output"].invalidated, 0)
        self.assertEqual(self.records[-1]["stage"], "invoke_after_core_completion")
        self.assert_permanently_retained(session)

    def test_timeout_checks_identity_and_records_identity_loss(self):
        session = self.session(complete=False)
        session.timeout_ns = 0
        session.core.on_poll = lambda: session.gpios["timer_identity"].values.update({0: 0x50334D32})
        with self.assertRaises(br.PLIdentityLost):
            session.invoke(np.empty(2, dtype="i1"))
        self.assertEqual(self.records[-1]["stage"], "invoke_timeout")
        self.assert_permanently_retained(session)

    def test_ordinary_timeout_retains_until_original_core_becomes_idle(self):
        session = self.session(complete=False)
        session.timeout_ns = 0
        with self.assertRaisesRegex(br.UnsafeDeviceState, "completion timed out"):
            session.invoke(np.empty(2, dtype="i1"))
        with self.assertRaises(br.UnsafeDeviceState):
            session.close()
        self.assertFalse(session.pl_identity_lost)
        buffers = list(session.owned)
        session.core.control = 4
        session.close()
        self.assertEqual([buffer.freed for buffer in buffers], [1, 1])
        self.assertTrue(session.closed)

    def test_close_detects_foreign_idle_without_freeing(self):
        session = self.session()
        session.pl.timestamp = "foreign-epoch"
        session.core.control = 4
        with self.assertRaises(br.PLIdentityLost):
            session.close()
        self.assert_permanently_retained(session)

    def test_property_read_failure_and_mixed_snapshot_fail_closed(self):
        class UnreadablePL:
            @property
            def bitfile_name(self):
                raise OSError("unavailable")

        session = self.session()
        session.pl = UnreadablePL()
        with self.assertRaises(br.PLIdentityLost):
            session.invoke(np.empty(2, dtype="i1"))
        self.assertEqual(self.records[-1]["device_state"], "PL_IDENTITY_LOST")
        self.assertIn(session, br.QUARANTINE)
        self.assertEqual([buffer.freed for buffer in session.owned], [0, 0])

        class MixedPL:
            bitfile_name = "/known/top.bit"
            reads = 0

            @property
            def timestamp(self):
                self.reads += 1
                return "epoch-original" if self.reads == 1 else "new-epoch"

        session = self.session()
        session.pl = MixedPL()
        with self.assertRaises(br.PLIdentityLost):
            session.invoke(np.empty(2, dtype="i1"))
        self.assertFalse(self.records[-1]["observed_identity"]["stable"])

    def test_loss_during_output_copy_is_caught_by_final_guard(self):
        session = self.session()

        def copyto(destination, source):
            destination[:] = source
            session.pl.timestamp = "foreign-epoch"

        session.np = SimpleNamespace(copyto=copyto)
        with self.assertRaises(br.PLIdentityLost):
            session.invoke(np.empty(2, dtype="i1"))
        self.assertEqual(self.records[-1]["stage"], "invoke_after")
        self.assert_permanently_retained(session)

    def test_original_identity_completion_and_close_pass(self):
        session = self.session()
        capture = np.empty(2, dtype="i1")
        timing = session.invoke(capture)
        self.assertTrue(timing["pl_identity_verified"])
        self.assertEqual(timing["pl_identity_epoch"], "epoch-original")
        self.assertEqual(timing["host_timing_guard_version"], br.HOST_TIMING_GUARD_VERSION)
        self.assertEqual(timing["runtime_protocol_id"], br.RUNTIME_PROTOCOL_ID)
        self.assertIn("both identity checks and their software overhead", timing["call_ns_scope"])
        self.assertEqual(timing["sequence"], 1)
        self.assertEqual(capture.tolist(), [0, 0])
        buffers = list(session.owned)
        session.close()
        self.assertEqual([buffer.freed for buffer in buffers], [1, 1])
        self.assertNotIn(session, br.QUARANTINE)

    def test_failure_evidence_retains_explicit_identity_lost_state(self):
        session = self.session()
        session.pl.timestamp = "foreign-epoch"
        with tempfile.TemporaryDirectory(prefix="identity_failure_") as directory:
            with self.assertRaises(br.PLIdentityLost):
                br._failure(Path(directory), session, RuntimeError("initial failure"))
            failure = json.loads((Path(directory) / "failure.json").read_text())
            retained = json.loads((Path(directory) / "device_buffer_retention.json").read_text())
            self.assertEqual(failure["device_state"], "PL_IDENTITY_LOST")
            self.assertEqual(retained["device_state"], "PL_IDENTITY_LOST")
            self.assertFalse(retained["buffers_freed"])

    def test_constructor_records_actual_pl_bitfile_timestamp_and_p3t1(self):
        with tempfile.TemporaryDirectory(prefix="identity_constructor_") as directory:
            root = Path(directory)
            bitfile = root / "vivado_shell/top.bit"
            bitfile.parent.mkdir()
            bitfile.write_bytes(b"fake bitstream")
            for suffix, values, dtype in (("a", [1, 2, 3, 4], "i1"), ("b", [1], "i1"),
                ("w", [1], "i1"), ("bias", [0], "<i4"), ("params", [0] * 8, "<i4"),
                ("expected", [4], "i1")):
                (root / ("f0_" + suffix + ".bin")).write_bytes(np.asarray(values, dtype=dtype).tobytes())
            pl = SimpleNamespace(bitfile_name=None, timestamp=None)
            core = Core()
            mmios = {100: core, 200: GPIO(), 300: GPIO(), 400: GPIO(br.P3T1_TIMER_MAGIC)}

            class Overlay:
                def __init__(self, filename, download):
                    self.filename = filename
                    self.ip_dict = {name: dict(type=kind, phys_addr=base, addr_range=4096)
                        for name, base, kind in (("core", 100, "hwnas.org:hls:pilot_kernel:1.0"),
                        ("timer_counts", 200, "xilinx.com:ip:axi_gpio:2.0"),
                        ("timer_sequence", 300, "xilinx.com:ip:axi_gpio:2.0"),
                        ("timer_identity", 400, "xilinx.com:ip:axi_gpio:2.0"))}

                def download(self):
                    pl.bitfile_name, pl.timestamp = self.filename, "actual-download-epoch"

            fake_pynq = SimpleNamespace(Overlay=Overlay, PL=pl, Clocks=SimpleNamespace(fclk0_mhz=100.0),
                MMIO=lambda base, size: mmios[base],
                allocate=lambda shape, dtype: Buffer(np.zeros(shape, dtype=dtype), dtype))
            case = dict(operator="maxpool", input_bits=8, output_bits=8,
                shape=dict(input_shapes=[[1, 2, 2]], output_shape=[1, 1, 1]),
                fixtures=[dict(fixture_id="f0", expected_elements=1)])
            metadata = dict(core_control_base=100, timer_counts_base=200, timer_sequence_base=300,
                timer_identity_base=400, timer_magic=br.P3T1_TIMER_MAGIC, clock_hz=100000000)
            abi = dict(control_offset=0, masks=dict(AP_IDLE=4), pointers={})
            with patch.dict(sys.modules, pynq=fake_pynq), patch.object(br, "resolve_abi", return_value=abi):
                calls = []
                session = br.BoardSession(root, case, metadata, {}, 1, identity_record=self.records.append,
                                          ownership_guard=lambda stage: calls.append(stage))
                self.assertEqual(calls, ['session_entry', 'before_overlay_download'])
                self.assertEqual(self.records[0]["original_identity"]["bitfile_name"], str(bitfile.resolve()))
                self.assertEqual(self.records[0]["original_identity"]["timestamp"], "actual-download-epoch")
                self.assertEqual(self.records[0]["original_identity"]["timer_magic"], br.P3T1_TIMER_MAGIC)
                session.close()


if __name__ == "__main__":
    unittest.main()
