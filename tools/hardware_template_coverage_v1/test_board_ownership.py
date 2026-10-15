"""Offline reservation and exact-process tests using only a temporary fake board.

No real /proc scan, real reservation/release, PYNQ, process launch, or board I/O.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.hardware_template_coverage_v1 import board_ownership as own
from tools.hardware_measurement_pilot import d09_board_ownership_v1 as frozen


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
        row = self.rows[pid]
        if isinstance(row, BaseException):
            raise row
        return copy.deepcopy(row)

    def add_process(self, pid, argv, *, state="R", start_ticks="123", cwd=None):
        (self.proc_root / str(pid)).mkdir(exist_ok=True)
        self.rows[pid] = dict(pid=pid, argv=list(argv), state=state, start_ticks=start_ticks,
                             cwd=cwd or own.KNOWN_ROOTS[1] + "/jobs/q3")


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        directory = Path("results/hardware_template_coverage_20261013/ownership_offline_tests").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fake_", dir=str(directory))
        self.root = Path(self.temp.name)
        self.backend = FakeBackend(self.root)
        self.base = dict(schema=own.SCHEMA, campaign_id="t6_resumed", owner_id="t6_owner",
                         expected_machine_id="b" * 32, expected_boot_id=self.backend.identity()["boot_id"],
                         reservation_root=own.RESERVATION_ROOT, known_board_roots=list(own.KNOWN_ROOTS))

    def tearDown(self):
        self.temp.cleanup()

    def put(self, logical, value):
        path = self.backend.path(logical)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    def reserved(self):
        result = own.reserve(self.base, backend=self.backend)
        return dict(self.base, reservation=result["reservation"])

    def worker(self, pid=22):
        root = own.KNOWN_ROOTS[1]
        helper, request = root + "/helpers/abc.py", root + "/jobs/q3/request.json"
        self.put(helper, b"# immutable worker helper\n")
        request_file = self.put(request, dict(job_id="q3"))
        argv = ["/usr/bin/python3", helper, "_worker", request]
        self.backend.add_process(pid, argv)
        value = own.process_identity(pid, "worker", helper_path=helper, request_path=request,
            request_sha256=own.digest(request_file.read_bytes()), expected_argv=argv, backend=self.backend)
        return value

    def runner(self, pid=23):
        root = own.KNOWN_ROOTS[1]
        runner = root + "/packages/a/case/board_runner.py"
        context = root + "/jobs/q3/ownership_context.json"
        self.put(runner, b"# immutable board runner\n")
        context_file = self.put(context, dict(config=self.base, job_request_sha256="a" * 64))
        sha = own.digest(context_file.read_bytes())
        argv = ["/usr/bin/python3", runner, "--case-dir", root + "/packages/a/case",
                own.REQUEST_FLAG, context, own.REQUEST_SHA_FLAG, sha]
        self.backend.add_process(pid, argv)
        value = own.process_identity(pid, "runner", runner_path=runner, request_path=context,
            request_sha256=sha, expected_argv=argv, backend=self.backend)
        return value

    def lock(self):
        binding = dict(job_id="q3", package=own.KNOWN_ROOTS[1] + "/packages/a", archive_sha256="a" * 64)
        path = self.put(own.OWN_BOARD_LOCK, binding)
        return dict(path=own.OWN_BOARD_LOCK, sha256=own.digest(path.read_bytes()), binding=binding)

    def test_unbound_config_refuses_without_any_write(self):
        before = list(self.root.rglob("*"))
        with self.assertRaisesRegex(own.OwnershipError, "INVALID_RESERVATION_REFERENCE"):
            own.check(self.base, backend=self.backend)
        self.assertEqual(list(self.root.rglob("*")), before)

    def test_every_configuration_boundary_is_strict(self):
        changes = [dict(schema=frozen.SCHEMA), dict(campaign_id="bad/id"), dict(owner_id=""),
                   dict(expected_machine_id="F" * 32), dict(expected_boot_id="not-uuid"),
                   dict(expected_boot_id=None), dict(reservation_root="/tmp/lease"),
                   dict(known_board_roots=[own.KNOWN_ROOTS[1]]),
                   dict(known_board_roots=list(reversed(own.KNOWN_ROOTS))),
                   dict(system_conditions_lock={})]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(own.OwnershipError):
                    own.check(dict(self.base, **change), backend=self.backend)

    def test_shared_protocol_interoperates_with_frozen_d09_without_source_change(self):
        self.assertEqual(own.digest(Path(frozen.__file__).read_bytes()), own.SOURCE_REFERENCE_SHA256)
        receipt = frozen.reserve(dict(self.base, schema=frozen.SCHEMA), backend=self.backend)
        config = dict(self.base, reservation=receipt["reservation"])
        self.assertEqual(own.check(config, backend=self.backend)["status"], "PASS_OWNED_NO_KNOWN_CONFLICT")
        frozen.check(dict(config, schema=frozen.SCHEMA), backend=self.backend, require_system_conditions=False)
        self.assertEqual(json.loads(self.backend.path(config["reservation"]["path"]).read_bytes())["schema"], own.RESERVATION_SCHEMA)

    def test_existing_d09_lease_cannot_be_taken_over_or_overwritten(self):
        receipt = frozen.reserve(dict(self.base, schema=frozen.SCHEMA, campaign_id="d09_old", owner_id="old_owner"), backend=self.backend)
        path = self.backend.path(receipt["reservation"]["path"])
        before = path.read_bytes()
        with self.assertRaisesRegex(own.OwnershipError, "ALREADY_EXISTS"):
            own.reserve(self.base, backend=self.backend)
        with self.assertRaisesRegex(own.OwnershipError, "OWNERSHIP_CHANGED"):
            own.check(dict(self.base, reservation=receipt["reservation"]), backend=self.backend)
        self.assertEqual(path.read_bytes(), before)

    def test_post_reserve_conflict_retains_new_lease(self):
        original, calls = own._scan, []
        def raced(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                self.put(own.KNOWN_ROOTS[0] + "/board.lock", dict(job_id="new_d09"))
            return original(*args, **kwargs)
        with patch.object(own, "_scan", side_effect=raced):
            with self.assertRaisesRegex(own.OwnershipError, "POST_RESERVE") as caught:
                own.reserve(self.base, backend=self.backend)
        self.assertTrue(self.backend.path(caught.exception.observation["reservation"]["path"]).is_file())

    def test_atomic_reservation_contender_wins_without_clobber(self):
        original, done = own._scan, []
        def contender(config, backend, **kwargs):
            result = original(config, backend, **kwargs)
            if config["owner_id"] == self.base["owner_id"] and not done:
                done.append(True)
                frozen.reserve(dict(self.base, schema=frozen.SCHEMA, owner_id="racer"), backend=backend)
            return result
        with patch.object(own, "_scan", side_effect=contender):
            with self.assertRaisesRegex(own.OwnershipError, "ALREADY_EXISTS"):
                own.reserve(self.base, backend=self.backend)
        self.assertEqual(json.loads(self.backend.path(own._reservation_path(self.base)).read_bytes())["owner_id"], "racer")

    def test_machine_and_boot_changes_refuse_even_with_a_valid_lease(self):
        config = self.reserved()
        for key, value in (("machine_id", "c" * 32), ("boot_id", "1d339ada-c002-4361-a65b-6bd7cd49552d")):
            before = dict(self.backend.current_identity)
            self.backend.current_identity[key] = value
            with self.assertRaisesRegex(own.OwnershipError, "BOOT_OR_MACHINE"):
                own.check(config, backend=self.backend)
            self.backend.current_identity = before

    def test_reservation_hash_and_all_owner_fields_are_bound(self):
        config = self.reserved()
        with self.assertRaisesRegex(own.OwnershipError, "OWNERSHIP_CHANGED"):
            own.check(dict(config, owner_id="someone_else"), backend=self.backend)
        path = self.backend.path(config["reservation"]["path"])
        original = path.read_bytes()
        for field, value in (("campaign_id", "another"), ("identity", dict(machine_id="f" * 32)),
                             ("schema", "new_protocol"), ("known_board_roots", []),
                             ("release_requires_explicit_owner_check", False)):
            altered = json.loads(original)
            altered[field] = value
            path.write_bytes(json.dumps(altered).encode())
            new_config = dict(config, reservation=dict(config["reservation"], sha256=own.digest(path.read_bytes())))
            with self.subTest(field=field), self.assertRaisesRegex(own.OwnershipError, "OWNERSHIP_CHANGED"):
                own.check(new_config, backend=self.backend)
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(own.OwnershipError, "OWNERSHIP_CHANGED"):
            own.check(config, backend=self.backend)

    def test_reservation_change_during_scan_is_rejected(self):
        config = self.reserved()
        original = own._scan
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.backend.path(config["reservation"]["path"]).write_bytes(b"{}")
            return result
        with patch.object(own, "_scan", side_effect=changed):
            with self.assertRaisesRegex(own.OwnershipError, "CHANGED_DURING_CHECK"):
                own.check(config, backend=self.backend)

    def test_every_known_lock_remains_a_conflict_and_private_content_does_not_leak(self):
        config = self.reserved()
        for root in own.KNOWN_ROOTS:
            for name in ("board.lock", "system_conditions.lock"):
                path = self.put(root + "/" + name, dict(owner_token="PRIVATE_SECRET"))
                with self.subTest(path=str(path)), self.assertRaises(own.OwnershipError) as caught:
                    own.check(config, backend=self.backend)
                self.assertNotIn("PRIVATE_SECRET", str(caught.exception))
                self.assertNotIn("PRIVATE_SECRET", json.dumps(caught.exception.observation))
                path.unlink()

    def test_nonregular_or_malformed_known_lock_fails_closed(self):
        config = self.reserved()
        path = self.put(own.OWN_BOARD_LOCK, b"not json")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)
        path.unlink()
        path.mkdir()
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)

    def test_own_lock_requires_second_root_exact_bytes_and_full_json(self):
        config, lock = self.reserved(), self.lock()
        own.check(config, backend=self.backend, own_board_lock=lock)
        for change in (dict(path=own.KNOWN_ROOTS[0] + "/board.lock"), dict(sha256="0" * 64),
                       dict(binding=dict(job_id="wrong")), dict(sha256=None)):
            with self.subTest(change=change), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend, own_board_lock=dict(lock, **change))
        path = self.backend.path(own.OWN_BOARD_LOCK)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_board_lock=lock)

    def test_own_lock_missing_and_change_during_scan_are_rejected(self):
        config, lock = self.reserved(), self.lock()
        path = self.backend.path(own.OWN_BOARD_LOCK)
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_board_lock=lock)
        path.write_bytes(raw)
        original = own._scan
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path.write_bytes(raw + b" ")
            return result
        with patch.object(own, "_scan", side_effect=changed):
            with self.assertRaisesRegex(own.OwnershipError, "LOCK_CHANGED_DURING_CHECK"):
                own.check(config, backend=self.backend, own_board_lock=lock)

    def test_direct_inline_relative_and_malformed_known_processes_all_conflict(self):
        config = self.reserved()
        pilot, t6 = own.KNOWN_ROOTS
        commands = [
            ["python", t6 + "/packages/x/case/board_runner.py"],
            ["taskset", "-c", "3", "python", pilot + "/packages/x/case/d09_cpu_runner.py"],
            ["python", "-c", "import runpy;runpy.run_path('" + t6 + "/packages/x/case/board_runner.py')"],
            ["python", "-c", "import runpy;runpy.run_path(__import__('sys').argv[1])", pilot + "/packages/x/case/d09_cpu_runner.py"],
            ["python", t6 + "/helpers/abc.py", "_worker", t6 + "/jobs/old/request.json"],
            ["python", "../../packages/x/case/board_runner.py"],
            ["python", "-c", "if '" + t6 + "':("],
        ]
        for argv in commands:
            self.backend.add_process(99, argv)
            with self.subTest(argv=argv), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend)

    def test_zombies_are_ignored_but_stopped_known_process_is_a_conflict(self):
        config = self.reserved()
        self.backend.add_process(99, ["python", own.KNOWN_ROOTS[0] + "/packages/a/case/d09_cpu_runner.py"], state="T")
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend)
        self.backend.rows[99]["state"] = "Z"
        own.check(config, backend=self.backend)

    def test_exact_worker_runner_and_lock_can_coexist(self):
        config, worker, runner, lock = self.reserved(), self.worker(), self.runner(), self.lock()
        result = own.check(config, backend=self.backend, own_worker=worker, own_runner=runner, own_board_lock=lock)
        self.assertEqual({r["pid"] for r in result["occupancy"]["owned_processes"]}, {22, 23})
        self.assertEqual(result["scope"], own.SCOPE)
        self.assertFalse(result["occupancy"]["active_processes"])

    def test_allowances_do_not_exempt_a_second_identical_process_or_d09(self):
        config, worker, runner = self.reserved(), self.worker(), self.runner()
        for argv in (worker["argv"], runner["argv"], ["python", own.KNOWN_ROOTS[0] + "/helpers/x.py", "_worker", "request.json"]):
            self.backend.add_process(99, argv)
            with self.subTest(argv=argv), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend, own_worker=worker, own_runner=runner)

    def test_pid_reuse_argv_cwd_zombie_and_missing_own_process_refuse(self):
        config, worker = self.reserved(), self.worker()
        original = copy.deepcopy(self.backend.rows[22])
        changes = [dict(start_ticks="124"), dict(argv=worker["argv"] + ["extra"]),
                   dict(cwd=own.KNOWN_ROOTS[1] + "/jobs/other"), dict(state="Z")]
        for change in changes:
            self.backend.rows[22] = dict(original, **change)
            with self.subTest(change=change), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend, own_worker=worker)
        self.backend.rows[22] = ProcessLookupError()
        with self.assertRaises(own.OwnershipError):
            own.check(config, backend=self.backend, own_worker=worker)

    def test_runner_allowance_requires_exact_pid_start_argv_and_request_flags(self):
        config, runner = self.reserved(), self.runner()
        original = copy.deepcopy(self.backend.rows[23])
        for change in (dict(start_ticks="999"), dict(argv=runner["argv"][:-1] + ["f" * 64]),
                       dict(argv=runner["argv"] + [own.REQUEST_FLAG, runner["request_path"]])):
            self.backend.rows[23] = dict(original, **change)
            with self.subTest(change=change), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend, own_runner=runner)

    def test_descriptor_role_root_and_pid_types_cannot_weaken_exclusions(self):
        config, worker = self.reserved(), self.worker()
        changes = [dict(schema="old"), dict(role="runner"), dict(pid=True), dict(pid=-1),
                   dict(start_ticks=123), dict(helper_path=own.KNOWN_ROOTS[0] + "/helpers/x.py"),
                   dict(request_path=own.KNOWN_ROOTS[1] + "/jobs/../x.json"),
                   dict(argv=["python", "-c", "pass"]), dict(script_sha256="bad")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend, own_worker=dict(worker, **change))

    def test_identity_creation_requires_complete_expected_argv(self):
        worker = self.worker()
        with self.assertRaisesRegex(own.OwnershipError, "EXPECTED_ARGV_CHANGED"):
            own.process_identity(22, "worker", helper_path=worker["helper_path"], request_path=worker["request_path"],
                request_sha256=worker["request_sha256"], expected_argv=worker["argv"] + ["extra"], backend=self.backend)

    def test_runner_duplicate_or_absent_cli_binding_cannot_create_identity(self):
        runner = self.runner()
        original = copy.deepcopy(self.backend.rows[23])
        for argv in (runner["argv"][:4], runner["argv"] + [own.REQUEST_SHA_FLAG, runner["request_sha256"]],
                     runner["argv"][:-1] + ["a" * 64]):
            self.backend.rows[23] = dict(original, argv=argv)
            with self.subTest(argv=argv), self.assertRaises(own.OwnershipError):
                own.process_identity(23, "runner", runner_path=runner["runner_path"], request_path=runner["request_path"],
                                     request_sha256=runner["request_sha256"], backend=self.backend)

    def test_script_or_request_byte_changes_reject_an_otherwise_exact_process(self):
        config, worker, runner = self.reserved(), self.worker(), self.runner()
        for logical in (worker["helper_path"], worker["request_path"], runner["runner_path"], runner["request_path"]):
            path = self.backend.path(logical)
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.subTest(path=logical), self.assertRaisesRegex(own.OwnershipError, "FILE_BINDING_CHANGED"):
                own.check(config, backend=self.backend, own_worker=worker, own_runner=runner)
            path.write_bytes(raw)

    def test_request_change_after_scan_is_rejected(self):
        config, runner = self.reserved(), self.runner()
        original = own._scan
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.backend.path(runner["request_path"]).write_bytes(b"{}")
            return result
        with patch.object(own, "_scan", side_effect=changed):
            with self.assertRaisesRegex(own.OwnershipError, "FILE_BINDING_CHANGED"):
                own.check(config, backend=self.backend, own_runner=runner)

    def test_process_read_error_refuses_but_disappeared_foreign_process_is_tolerated(self):
        config = self.reserved()
        self.backend.add_process(99, ["python", "other.py"])
        self.backend.rows[99] = PermissionError("PRIVATE_ERROR")
        with self.assertRaises(own.OwnershipError) as caught:
            own.check(config, backend=self.backend)
        self.assertNotIn("PRIVATE_ERROR", json.dumps(caught.exception.observation))
        self.backend.rows[99] = ProcessLookupError()
        own.check(config, backend=self.backend)

    def test_proc_directory_unavailable_refuses(self):
        config = self.reserved()
        with patch.object(Path, "iterdir", side_effect=PermissionError("private")):
            with self.assertRaises(own.OwnershipError):
                own.check(config, backend=self.backend)

    def test_symlink_in_any_ownership_ancestor_is_rejected(self):
        config = self.reserved()
        real = Path.is_symlink
        ancestor = self.backend.path(own.RESERVATION_ROOT)
        def linked(path):
            return path == ancestor or real(path)
        with patch.object(Path, "is_symlink", linked):
            with self.assertRaisesRegex(own.OwnershipError, "SYMLINK"):
                own.check(config, backend=self.backend)

    def test_release_refuses_live_task_or_lock_and_preserves_exact_lease(self):
        config = self.reserved()
        path = self.backend.path(config["reservation"]["path"])
        raw = path.read_bytes()
        lock = self.put(own.KNOWN_ROOTS[0] + "/system_conditions.lock", dict(condition_id="d09"))
        with self.assertRaises(own.OwnershipError):
            own.release(config, backend=self.backend)
        self.assertEqual(path.read_bytes(), raw)
        lock.unlink()
        worker = self.worker()
        with self.assertRaises(own.OwnershipError):
            own.release(config, backend=self.backend)
        self.backend.rows[worker["pid"]]["state"] = "Z"
        result = own.release(config, backend=self.backend)
        self.assertFalse(path.exists())
        self.assertEqual(Path(result["archive"]).read_bytes(), raw)

    def test_backend_rejects_pid_reuse_argv_exec_and_cwd_change_during_read(self):
        backend = own.Backend()
        backend.proc_root = self.backend.proc_root
        def stat(start):
            return "22 (python (worker)) " + " ".join(["S"] + ["0"] * 18 + [str(start)] + ["0"] * 4)
        cases = [([stat(100), stat(101)], [b"python\0script.py\0"] * 2, ["/tmp"] * 2),
                 ([stat(100)] * 2, [b"python\0old.py\0", b"python\0new.py\0"], ["/tmp"] * 2),
                 ([stat(100)] * 2, [b"python\0script.py\0"] * 2, ["/tmp", "/other"])]
        for stats, commands, dirs in cases:
            with self.subTest(commands=commands), patch.object(Path, "read_text", side_effect=stats), \
                 patch.object(Path, "read_bytes", side_effect=commands), patch.object(own.os, "readlink", side_effect=dirs):
                with self.assertRaisesRegex(own.OwnershipError, "PROCESS_IDENTITY_CHANGED"):
                    backend.process(22)

    def test_module_uses_python38_syntax_and_no_project_runtime_imports(self):
        source = Path(own.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 8))
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name and (name.startswith("tools") or name.startswith("pynq")) for name in imported))
        self.assertNotIn("is_relative_to(", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":
    unittest.main()
