"""Offline fake-board ownership/race and guarded-launch integration tests.

No PYNQ import, ARM call, process signal, reset or physical board operation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.hardware_measurement_pilot import d09_board_ownership_v1 as own
from tools.hardware_measurement_pilot import d09_cpu_guarded_transport as transport
from tools.hardware_measurement_pilot import d09_cpu_guarded_campaign as campaign


class FakeBackend:
    def __init__(self, root):
        self.root = root
        self.proc_root = root / "proc"
        self.proc_root.mkdir()
        self.current_identity = dict(machine_id="b" * 32, boot_id="da641785-1120-499a-8b22-0d89bfc73687")
        self.rows = {}

    def path(self, logical):
        return self.root / "fs" / logical.lstrip("/")

    def identity(self):
        return dict(self.current_identity)

    def process(self, pid):
        value = self.rows[pid]
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)

    def add_process(self, pid, argv, *, state="R", start_ticks="123", cwd="/home/xilinx"):
        (self.proc_root / str(pid)).mkdir(exist_ok=True)
        self.rows[pid] = dict(pid=pid, argv=argv, state=state, start_ticks=start_ticks, cwd=cwd)


class LocalCase(unittest.TestCase):
    def setUp(self):
        root = Path("results/d09_cpu_freeze_20261013/quiet_v3/guard_offline_tests").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fake_", dir=root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class OwnershipTests(LocalCase):
    def setUp(self):
        super().setUp()
        self.backend = FakeBackend(self.root)
        self.base = dict(schema=own.SCHEMA, campaign_id="quiet_v3", owner_id="campaign_one",
                         expected_machine_id="b" * 32, expected_boot_id=self.backend.identity()["boot_id"],
                         reservation_root=own.RESERVATION_ROOT, known_board_roots=list(own.KNOWN_ROOTS))

    def put(self, logical, value):
        path = self.backend.path(logical)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def reserved(self, *, with_system=True):
        result = own.reserve(self.base, backend=self.backend)
        config = dict(self.base, reservation=result["reservation"])
        if with_system:
            lock = self.put(own.SYSTEM_LOCK, dict(condition_id="d09_cpu_freeze03", identity=self.backend.identity(),
                                                 owner_token="PRIVATE_TEST_VALUE"))
            config["system_conditions_lock"] = dict(path=own.SYSTEM_LOCK, sha256=own.digest(lock.read_bytes()),
                                                     condition_id="d09_cpu_freeze03")
        return config

    def test_reserve_is_exclusive_across_campaigns_and_never_overwrites(self):
        config = self.reserved(with_system=False)
        path = self.backend.path(config["reservation"]["path"])
        before = path.read_bytes()
        other = dict(self.base, owner_id="campaign_two", campaign_id="other_campaign")
        with self.assertRaisesRegex(own.OwnershipError, "ALREADY_EXISTS"):
            own.reserve(other, backend=self.backend)
        self.assertEqual(path.read_bytes(), before)

    def test_race_after_initial_scan_retains_reservation_and_refuses(self):
        original = own._scan
        calls = []
        def raced(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                self.put(own.KNOWN_ROOTS[1] + "/board.lock", {"job_id": "foreign_t6"})
            return original(*args, **kwargs)
        with patch.object(own, "_scan", side_effect=raced):
            with self.assertRaisesRegex(own.OwnershipError, "POST_RESERVE") as captured:
                own.reserve(self.base, backend=self.backend)
        self.assertTrue(self.backend.path(captured.exception.observation["reservation"]["path"]).is_file())

    def test_two_contenders_race_on_atomic_create(self):
        original = own._scan
        contender_done = []
        def contended(config, backend, **kwargs):
            result = original(config, backend, **kwargs)
            if config["owner_id"] == self.base["owner_id"] and not contender_done:
                contender_done.append(True)
                own.reserve(dict(self.base, owner_id="racer"), backend=self.backend)
            return result
        with patch.object(own, "_scan", side_effect=contended):
            with self.assertRaisesRegex(own.OwnershipError, "ALREADY_EXISTS"):
                own.reserve(self.base, backend=self.backend)
        value = json.loads(self.backend.path(own._reservation_path(self.base)).read_text())
        self.assertEqual(value["owner_id"], "racer")

    def test_exact_own_system_binding_required_and_foreign_hash_rejected(self):
        config = self.reserved()
        self.assertEqual(own.check(config, backend=self.backend)["status"], "PASS_OWNED_NO_KNOWN_CONFLICT")
        self.put(own.SYSTEM_LOCK, {"condition_id": "someone_else", "owner_token": "PRIVATE_FOREIGN_VALUE"})
        with self.assertRaises(own.OwnershipError) as captured:
            own.check(config, backend=self.backend)
        self.assertNotIn("PRIVATE", str(captured.exception))
        self.assertNotIn("PRIVATE", json.dumps(captured.exception.observation))

    def test_system_identity_mismatch_fails_even_with_updated_lock_hash(self):
        config = self.reserved()
        path = self.put(own.SYSTEM_LOCK, dict(condition_id="d09_cpu_freeze03", identity=dict(machine_id="f"*32, boot_id=self.base["expected_boot_id"])))
        config["system_conditions_lock"]["sha256"] = own.digest(path.read_bytes())
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)

    def test_missing_system_lock_only_allowed_before_apply(self):
        config = self.reserved(with_system=False)
        own.check(config, backend=self.backend, require_system_conditions=False)
        with self.assertRaisesRegex(own.OwnershipError, "SYSTEM_CONDITIONS_BINDING_REQUIRED"):
            own.check(config, backend=self.backend)

    def test_foreign_namespace_lock_and_malformed_lock_fail_closed(self):
        config = self.reserved()
        path = self.put(own.KNOWN_ROOTS[1] + "/board.lock", dict(job_id="other"))
        for raw in (path.read_bytes(), b"not json"):
            path.write_bytes(raw)
            with self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend)

    def test_new_boot_or_machine_rejected_before_admission(self):
        config = self.reserved()
        self.backend.current_identity["boot_id"] = "1d339ada-c002-4361-a65b-6bd7cd49552d"
        with self.assertRaisesRegex(own.OwnershipError, "BOOT_OR_MACHINE"):
            own.check(config, backend=self.backend)

    def test_direct_inline_and_embedded_known_runners_and_workers_rejected(self):
        config = self.reserved()
        pilot, t6 = own.KNOWN_ROOTS
        cases = [
            ["python", t6 + "/packages/x/case/board_runner.py"],
            ["python", "-c", "import runpy;runpy.run_path(__import__('sys').argv[1])", pilot + "/packages/x/case/d09_cpu_runner.py"],
            ["python", "-c", "import runpy;runpy.run_path('" + t6 + "/packages/x/case/board_runner.py')"],
            ["python", pilot + "/helpers/abc.py", "_worker", pilot + "/jobs/old/request.json"],
        ]
        for argv in cases:
            self.backend.add_process(22, argv)
            with self.assertRaises(own.OwnershipError) as captured:
                own.check(config, backend=self.backend)
            self.assertEqual(captured.exception.observation["active_processes"][0]["pid"], 22)

    def test_stopped_runner_is_still_occupancy_zombie_is_not(self):
        config = self.reserved()
        argv = ["python", own.KNOWN_ROOTS[1] + "/packages/x/case/board_runner.py"]
        self.backend.add_process(22, argv, state="T")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)
        self.backend.rows[22]["state"] = "Z"
        own.check(config, backend=self.backend)

    def test_only_exact_worker_pid_start_and_argv_exempted(self):
        config = self.reserved()
        helper = own.KNOWN_ROOTS[0] + "/helpers/hash.py"
        request = own.KNOWN_ROOTS[0] + "/jobs/q3/request.json"
        self.backend.add_process(22, ["python", helper, "_worker", request])
        own_worker = dict(pid=22, start_ticks="123", helper_path=helper, request_path=request)
        own.check(config, backend=self.backend, own_worker=own_worker)
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_worker=dict(own_worker, start_ticks="124"))
        self.backend.add_process(23, ["python", helper, "_worker", request])
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_worker=own_worker)

    def test_own_board_lock_requires_exact_binding_and_root(self):
        config = self.reserved()
        logical = own.KNOWN_ROOTS[0] + "/board.lock"
        self.put(logical, {"job_id": "q3"})
        own.check(config, backend=self.backend, own_board_lock=dict(path=logical, binding={"job_id": "q3"}))
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_board_lock=dict(path=logical, binding={"job_id": "q2"}))
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_board_lock=dict(path=own.KNOWN_ROOTS[1] + "/board.lock", binding={"job_id": "q3"}))

    def test_disappeared_own_board_lock_cannot_be_exempted(self):
        config = self.reserved()
        logical = own.KNOWN_ROOTS[0] + "/board.lock"
        with self.assertRaises(own.OwnershipError) as caught:
            own.check(config, backend=self.backend, own_board_lock=dict(path=logical, binding={"job_id": "q3"}))
        self.assertIn({"kind":"OWN_BOARD_LOCK_NOT_VERIFIED"}, caught.exception.observation["conflicts"])

    def test_process_read_failure_refused_disappeared_process_tolerated(self):
        config = self.reserved()
        (self.backend.proc_root / "25").mkdir()
        self.backend.rows[25] = PermissionError("private error must not leak")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)
        self.backend.rows[25] = ProcessLookupError()
        own.check(config, backend=self.backend)

    def test_backend_rechecks_pid_start_ticks_across_cmdline_read(self):
        backend = own.Backend()
        backend.proc_root = self.backend.proc_root
        def stat(start):
            fields = ["S"] + ["0"] * 18 + [str(start)] + ["0"] * 4
            return "22 (python) " + " ".join(fields)
        with patch.object(Path, "read_text", side_effect=[stat(100), stat(101)]), patch.object(Path, "read_bytes", return_value=b"python\0board_runner.py\0"), patch.object(own.os, "readlink", return_value="/home/xilinx"):
            with self.assertRaisesRegex(own.OwnershipError, "PROCESS_IDENTITY_CHANGED"):
                backend.process(22)

    def test_namespace_omission_refused(self):
        with self.assertRaisesRegex(own.OwnershipError, "NAMESPACE"):
            own.reserve(dict(self.base, known_board_roots=[own.KNOWN_ROOTS[0]]), backend=self.backend)

    def test_lease_tamper_and_cross_owner_refused(self):
        config = self.reserved()
        with self.assertRaisesRegex(own.OwnershipError, "OWNERSHIP_CHANGED"):
            own.check(dict(config, owner_id="wrong_owner"), backend=self.backend)
        self.backend.path(config["reservation"]["path"]).write_bytes(b"{}")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)

    def test_release_requires_all_known_locks_gone_and_archives_bytes(self):
        config = self.reserved()
        path = self.backend.path(config["reservation"]["path"])
        raw = path.read_bytes()
        with self.assertRaises(own.OwnershipError):
            own.release(config, backend=self.backend)
        self.backend.path(own.SYSTEM_LOCK).unlink()
        result = own.release(config, backend=self.backend)
        self.assertFalse(path.exists())
        self.assertEqual(Path(result["archive"]).read_bytes(), raw)


class TransportTests(LocalCase):
    def setup_package(self):
        board = self.root / "board"
        package = board / "packages" / ("a"*64)
        case = package / "case"
        case.mkdir(parents=True)
        config = dict(board_root=str(board), package=str(package), archive_sha256="a"*64,
                      helper_sha256="b"*64, mode="timing")
        transport.save(package / "transport.json", config)
        transport.save(package / "READY.json", {"ready": True})
        transport.save(case / "transport_files.json", {"files": {}})
        transport.save(case / "cpu_conditions.json", {"cpu_index": 3})
        return board, package, config

    def test_launch_guard_failure_has_no_launch_writes_or_popen(self):
        board, package, config = self.setup_package()
        with patch.object(transport, "ownership_check", side_effect=ValueError("foreign user")) as guard, patch.object(transport.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "foreign user"):
                transport.remote_action(dict(action="launch", config=config, job_id="q3", board_id="machine-id:"+"b"*32, timeout_seconds=30))
            popen.assert_not_called()
            guard.assert_called_once()
        self.assertFalse((board / "board.lock").exists())
        self.assertFalse((board / "jobs/q3").exists())

    def setup_worker(self):
        board, package, config = self.setup_package()
        job = board / "jobs/q3"
        job.mkdir(parents=True)
        binding = {"job_id": "q3"}
        transport.save(board / "board.lock", binding)
        request = dict(config=config, lock_binding=binding, board_id="machine-id:"+"b"*32, job_id="q3", timeout_seconds=30)
        transport.save(job / "request.json", request)
        return board, package, config, job

    def test_worker_guard_failure_retains_lock_before_popen(self):
        board, package, config, job = self.setup_worker()
        with patch.object(transport.os, "sched_setaffinity", create=True), patch.object(transport.os, "sched_getaffinity", return_value={0,1,2}, create=True), patch.object(transport, "ownership_check", side_effect=ValueError("new boot")), patch.object(transport.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "new boot"):
                transport.worker(job / "request.json")
            popen.assert_not_called()
        self.assertTrue((board / "board.lock").exists())
        self.assertEqual(transport.load(job / "status.json")["state"], "WORKER_FAILED_LOCK_RETAINED")

    def test_worker_success_checks_before_and_after_and_preserves_quiet_command(self):
        board, package, config, job = self.setup_worker()
        recorded = []
        class Process:
            pid = 99999
            returncode = 0
            def __init__(self, command, **kwargs):
                recorded.append((command, kwargs))
                (job / "output").mkdir()
                transport.save(job / "output/cpu_conditions_restore.json", {"status":"PASS_RESTORED"})
            def poll(self):
                return 0
        original = Path.is_file
        def isfile(path):
            return True if str(path).replace("\\", "/") == "/usr/bin/taskset" else original(path)
        with patch.object(transport.os, "sched_setaffinity", create=True), patch.object(transport.os, "sched_getaffinity", return_value={0,1,2}, create=True), patch.object(transport, "ownership_check", return_value={"ok": True}) as guard, patch.object(transport.subprocess, "Popen", Process), patch.object(Path, "is_file", isfile):
            transport.worker(job / "request.json")
        self.assertEqual(guard.call_count, 2)
        self.assertEqual(recorded[0][0][:3], ["/usr/bin/taskset", "-c", "3"])
        self.assertIn("time.sleep(15)", recorded[0][0][5])
        self.assertTrue(all(recorded[0][1]["env"][name] == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")))
        self.assertFalse((board / "board.lock").exists())
        self.assertTrue((job / "worker_prelaunch_ownership.json").is_file())
        self.assertTrue((job / "worker_postrun_ownership.json").is_file())

    def test_postrun_ownership_loss_keeps_local_job_lock(self):
        board, package, config, job = self.setup_worker()
        class Process:
            pid = 99999
            returncode = 0
            def __init__(self, *args, **kwargs):
                (job / "output").mkdir()
                transport.save(job / "output/cpu_conditions_restore.json", {"status":"PASS_RESTORED"})
            def poll(self):
                return 0
        original = Path.is_file
        def isfile(path):
            return True if str(path).replace("\\", "/") == "/usr/bin/taskset" else original(path)
        with patch.object(transport.os, "sched_setaffinity", create=True), patch.object(transport.os, "sched_getaffinity", return_value={0,1,2}, create=True), patch.object(transport, "ownership_check", side_effect=[{"ok":True}, ValueError("foreign appeared")]), patch.object(transport.subprocess, "Popen", Process), patch.object(Path, "is_file", isfile):
            with self.assertRaisesRegex(ValueError, "foreign appeared"):
                transport.worker(job / "request.json")
        self.assertTrue((board / "board.lock").exists())
        self.assertEqual(transport.load(job / "status.json")["state"], "WORKER_FAILED_LOCK_RETAINED")

    def test_ownership_loader_verifies_source_and_config_before_loading(self):
        board, package, config = self.setup_package()
        case = package / "case"
        source = case / transport.OWNERSHIP_SOURCE
        source.write_text("def check(config, **kwargs):\n return {'ok': True, 'kwargs': kwargs}\n")
        ownconfig = dict(expected_machine_id="b"*32, expected_boot_id="testboot")
        transport.save(case / transport.OWNERSHIP_CONFIG, ownconfig)
        config.update(expected_machine_id="b"*32, expected_boot_id="testboot", ownership_source_sha256=transport.sha(source), ownership_config_sha256=transport.sha(case / transport.OWNERSHIP_CONFIG))
        result = transport.ownership_check(package, config, "machine-id:"+"b"*32)
        self.assertTrue(result["kwargs"]["require_system_conditions"])
        source.write_text("raise RuntimeError('untrusted code must never execute')")
        with self.assertRaisesRegex(ValueError, "differs from transport"):
            transport.ownership_check(package, config, "machine-id:"+"b"*32)


class PackagingTests(LocalCase):
    def test_isolated_package_binds_guard_and_preserves_frozen_runner(self):
        prior = Path("results/d09_cpu_freeze_20261013/quiet_v2")
        for name in ("pl_baseline.json", "user_authorization.json", "external_conditions.json"):
            shutil.copyfile(prior / name, self.root / name)
        transport.save(self.root / "board_ownership.json", dict(known_board_roots=list(own.KNOWN_ROOTS), expected_machine_id="b"*32, expected_boot_id="testboot"))
        protocol = transport.load(prior / "d09_protocol.json")
        protocol["board_ownership"] = dict(path="board_ownership.json", sha256=transport.sha(self.root / "board_ownership.json"))
        for name in ("d09_cpu_runner.py", "cpu_conditions.py", "d09_board_ownership_v1.py", "d09_cpu_guarded_transport.py", "d09_cpu_guarded_campaign.py"):
            protocol["source_files"][name] = transport.sha(Path(own.__file__).with_name(name))
        transport.save(self.root / "d09_protocol.json", protocol)
        with patch.object(campaign, "RESULT", self.root):
            result = campaign.prepare("dw5_int8_reference_low", "calibration", "offline_q3")
        case = Path(result["case_dir"])
        bundle = transport.load(case / "board_bundle.json")
        config = transport.load(case / "cpu_conditions.json")
        self.assertEqual(config["board_ownership"], protocol["board_ownership"])
        for name in ("board_runner.py", "measurement.py", "d09_cpu_runner.py", "cpu_conditions.py"):
            self.assertEqual(bundle["files"][name], protocol["source_files"][name])
        prepared = transport.prepare(case, self.root / "transport", mode="timing", bridge=str(Path(own.__file__).with_name("board_bridge_stdin.py")), chunk_bytes=4194304)
        self.assertEqual(prepared["ownership_config_sha256"], protocol["board_ownership"]["sha256"])
        self.assertEqual(prepared["ownership_source_sha256"], protocol["source_files"]["d09_board_ownership_v1.py"])
        self.assertEqual(prepared["schema"], "d09_cpu_guarded_board_transport_v1")


if __name__ == "__main__":
    unittest.main()
