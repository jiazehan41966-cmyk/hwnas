"""Offline transport integrity/ownership tests; never import PYNQ or connect."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.hardware_template_coverage_v1 import board_transfer as bt


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pilot_transfer_")
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        self.proc.mkdir()
        self.legacy = self.root / "legacy"
        self.proc_patch = patch.object(bt, "PROC_ROOT", self.proc)
        self.legacy_patch = patch.object(bt, "LEGACY_BOARD_ROOT", str(self.legacy))
        self.default_root_patch = patch.object(bt, "DEFAULT_BOARD_ROOT", str(self.root / "board"))
        self.proc_patch.start()
        self.legacy_patch.start()
        self.default_root_patch.start()
        self.source = self.root / "source.tar.gz"
        self.payload = b"numeric fixture bytes" * 1200
        files = {"nested/fixture.bin": hashlib.sha256(self.payload).hexdigest()}
        with tarfile.open(self.source, "w:gz") as tar:
            bt.add_bytes(tar, "nested/fixture.bin", self.payload)
            bt.add_bytes(tar, "transport_files.json", json.dumps(dict(files=files)).encode())
        digest = bt.sha(self.source)
        self.config = dict(board_root=str(self.root / "board"), archive_sha256=digest,
            archive_bytes=self.source.stat().st_size, package=str(self.root / "board" / "packages" / digest),
            chunk_bytes=1024, mode="probe")

    def tearDown(self):
        self.proc_patch.stop()
        self.legacy_patch.stop()
        self.default_root_patch.stop()
        self.temp.cleanup()

    def action(self, action, **kwargs):
        return bt.remote_action(dict(config=self.config, action=action, **kwargs))

    def upload(self):
        self.action("upload_init")
        data = self.source.read_bytes()
        for offset in range(0, len(data), 1024):
            block = data[offset:offset + 1024]
            self.action("upload_chunk", offset=offset, data=base64.b64encode(block).decode(),
                digest=hashlib.sha256(block).hexdigest())
        return self.action("upload_commit")

    def test_upload_exact_bytes_and_ready_immutable(self):
        result = self.upload()
        self.assertTrue(result["ready"])
        package = Path(self.config["package"])
        self.assertEqual((package / "case/nested/fixture.bin").read_bytes(), self.payload)
        self.assertTrue(self.action("upload_init")["ready"])
        with self.assertRaises(ValueError):
            self.action("upload_chunk", offset=0, data="AA==", digest=hashlib.sha256(b"\0").hexdigest())

    def test_repeat_chunk_accepted_conflicting_or_gap_rejected(self):
        self.action("upload_init")
        block = self.source.read_bytes()[:100]
        args = dict(offset=0, data=base64.b64encode(block).decode(), digest=hashlib.sha256(block).hexdigest())
        self.assertEqual(self.action("upload_chunk", **args)["offset"], 100)
        self.assertTrue(self.action("upload_chunk", **args)["replay"])
        args["offset"] = 101
        with self.assertRaises(ValueError):
            self.action("upload_chunk", **args)
        args.update(offset=0, data=base64.b64encode(b"x" * 100).decode(), digest=hashlib.sha256(b"x" * 100).hexdigest())
        with self.assertRaises(ValueError):
            self.action("upload_chunk", **args)

    def test_hash_mismatch_rejected_before_extraction(self):
        self.action("upload_init")
        archive = Path(self.config["package"]) / "package.tar.gz"
        archive.write_bytes(b"x" * self.config["archive_bytes"])
        with self.assertRaises(ValueError):
            self.action("upload_commit")
        self.assertFalse((archive.parent / "case").exists())

    def test_archive_traversal_link_and_hash_rejected_before_write(self):
        for index, kind in enumerate(("traversal", "link", "hash")):
            archive = self.root / (kind + ".tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                name = "../outside.bin" if kind == "traversal" else "safe.bin"
                if kind == "link":
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.SYMTYPE
                    info.linkname = "elsewhere"
                    tar.addfile(info)
                else:
                    bt.add_bytes(tar, name, b"abc")
                bt.add_bytes(tar, "transport_files.json", json.dumps(dict(files={name:"0" * 64})).encode())
            destination = self.root / ("extracted" + str(index))
            with self.assertRaises(ValueError):
                bt.unpack(archive, destination)
            self.assertFalse(destination.exists())

    def test_board_lock_refuses_concurrent_launch(self):
        self.upload()
        lock = self.root / "board/board.lock"
        bt.save(lock, dict(job_id="existing"), exclusive=True)
        with patch.object(bt.subprocess, "Popen") as popen:
            with self.assertRaises(bt.BoardConflict):
                self.action("launch", job_id="newjob", board_id="machine-id:test", timeout_seconds=30)
            popen.assert_not_called()
        self.assertEqual(bt.load(lock), dict(job_id="existing"))

    def add_process(self, pid, script, *, worker=False, state="S"):
        directory = self.proc / str(pid)
        directory.mkdir()
        argv = ["python3", str(script)] + (["_worker", "request.json"] if worker else [])
        (directory / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
        fields = [state] + ["0"] * 18 + ["98765"] + ["0"] * 2
        (directory / "stat").write_text(str(pid) + " (python3) " + " ".join(fields))

    def test_legacy_locks_block_launch_before_writes_and_redact_owner_token(self):
        self.upload()
        self.legacy.mkdir()
        secret = "do-not-copy-owner-token"
        for name in ("board.lock", "system_conditions.lock"):
            with self.subTest(lock=name):
                lock = self.legacy / name
                bt.save(lock, dict(job_id="foreign_job", condition_id="foreign_condition", owner_token=secret))
                original = lock.read_bytes()
                with patch.object(bt.subprocess, "Popen") as popen:
                    with self.assertRaises(bt.BoardConflict) as caught:
                        self.action("launch", job_id="blocked", board_id="machine-id:test", timeout_seconds=30)
                    popen.assert_not_called()
                self.assertNotIn(secret, str(caught.exception))
                self.assertNotIn("owner_token", str(caught.exception))
                self.assertIn("foreign_job", str(caught.exception))
                self.assertEqual(lock.read_bytes(), original)
                self.assertFalse((self.root / "board/board.lock").exists())
                self.assertFalse((self.root / "board/jobs/blocked").exists())
                lock.unlink()

    def test_active_legacy_and_new_runners_block_without_locks_or_raw_argv(self):
        self.upload()
        for index, (directory, worker) in enumerate(((self.legacy, False), (self.legacy, True),
            (self.root / "board", False), (self.root / "board", True))):
            with self.subTest(root=str(directory), worker=worker):
                script = directory / ("helpers/version.py" if worker else "packages/hash/case/board_runner.py")
                self.add_process(200 + index, script, worker=worker)
                with patch.object(bt.subprocess, "Popen") as popen:
                    with self.assertRaises(bt.BoardConflict) as caught:
                        self.action("launch", job_id="blocked", board_id="machine-id:test", timeout_seconds=30)
                    popen.assert_not_called()
                observation = caught.exception.observation
                self.assertEqual(observation["active_processes"][0]["role"], "worker" if worker else "runner")
                self.assertNotIn("cmdline", json.dumps(observation))
                self.assertFalse((self.root / "board/board.lock").exists())
                for item in (self.proc / str(200 + index)).iterdir():
                    item.unlink()
                (self.proc / str(200 + index)).rmdir()

    def test_missing_process_inspection_fails_closed(self):
        self.upload()
        with patch.object(bt, "PROC_ROOT", self.root / "missing_proc"), patch.object(bt.subprocess, "Popen") as popen:
            with self.assertRaises(bt.BoardConflict) as caught:
                self.action("launch", job_id="blocked", board_id="machine-id:test", timeout_seconds=30)
            popen.assert_not_called()
        self.assertIn("PROCESS_INSPECTION_UNAVAILABLE", str(caught.exception))
        self.assertFalse((self.root / "board/board.lock").exists())

    def test_relative_runner_resolves_process_cwd_and_blocks_launch(self):
        self.upload()
        self.add_process(321, Path("packages/hash/case/board_runner.py"))
        with patch.object(bt.os, "readlink", return_value=str(self.legacy)), patch.object(bt.subprocess, "Popen") as popen:
            with self.assertRaises(bt.BoardConflict) as caught:
                self.action("launch", job_id="blocked", board_id="machine-id:test", timeout_seconds=30)
            popen.assert_not_called()
        self.assertEqual(caught.exception.observation["active_processes"][0]["workflow"], "legacy_pilot")

    def test_worker_rechecks_foreign_lock_before_starting_runner(self):
        self.upload()
        job = self.root / "board/jobs/recheck"
        job.mkdir(parents=True)
        binding = dict(job_id="recheck")
        bt.save(self.root / "board/board.lock", binding)
        request = dict(config=self.config, job_id="recheck", lock_binding=binding,
                       board_id="machine-id:test", timeout_seconds=1)
        bt.save(job / "request.json", request)
        self.legacy.mkdir()
        foreign = self.legacy / "system_conditions.lock"
        bt.save(foreign, dict(condition_id="foreign", owner_token="private"))
        original = foreign.read_bytes()
        with patch.object(bt.subprocess, "Popen") as popen:
            with self.assertRaises(bt.BoardConflict):
                bt.worker(job / "request.json")
            popen.assert_not_called()
        self.assertEqual(foreign.read_bytes(), original)
        status = bt.load(job / "status.json")
        self.assertEqual(status["state"], "WORKER_FAILED_LOCK_RETAINED")
        self.assertNotIn("owner_token", json.dumps(status))
        self.assertTrue((self.root / "board/board.lock").exists())

    def test_worker_can_observe_its_own_lock_and_process(self):
        board_root = self.root / "board"
        board_root.mkdir()
        binding = dict(job_id="own", owner_token="private")
        bt.save(board_root / "board.lock", binding)
        self.add_process(123, board_root / "helpers/current.py", worker=True)
        observation = bt.require_no_board_conflict(board_root, own_lock_binding=binding, exclude_pids=(123,))
        self.assertEqual(observation["status"], "NO_KNOWN_CONFLICT_OBSERVED")
        self.assertNotIn("owner_token", json.dumps(observation))
        self.assertEqual(observation["exclusion_scope"], "KNOWN_WORKFLOWS_ONLY_NOT_GLOBAL_MUTEX_PROOF")

    def test_snapshot_frozen_and_download_hashes_match(self):
        self.upload()
        job = self.root / "board/jobs/testjob"
        (job / "output").mkdir(parents=True)
        bt.save(job / "request.json", dict(config=self.config))
        bt.save(job / "status.json", dict(state="RUNNING"))
        raw = job / "output/timing.csv"
        raw.write_text("header\nfirst\n")
        result = self.action("snapshot", job_id="testjob")
        raw.write_text("header\nchanged after snapshot\n")
        archive = self.root / "download.tar.gz"
        collected = bytearray()
        while len(collected) < result["archive_bytes"]:
            chunk = self.action("fetch_chunk", job_id="testjob", snapshot_name=result["snapshot_name"], offset=len(collected))
            data = base64.b64decode(chunk["data"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), chunk["digest"])
            collected.extend(data)
        archive.write_bytes(collected)
        self.assertEqual(bt.sha(archive), result["archive_sha256"])
        bt.unpack(archive, self.root / "snapshot", manifest_name="snapshot_manifest.json")
        self.assertEqual((self.root / "snapshot/output/timing.csv").read_text(), "header\nfirst\n")
        self.assertEqual(result["observed_state"], "RUNNING")

    def test_raw_bridge_error_suppressed(self):
        class Result:
            returncode = 1
            stdout = "sensitive service internals"
        with patch.object(bt.subprocess, "run", return_value=Result()):
            with self.assertRaisesRegex(RuntimeError, "raw transport output suppressed") as error:
                bt.bridge_call(self.root, dict(bridge="unused"), "print('safe')", "test")
            self.assertNotIn("sensitive", str(error.exception))

    def test_worker_success_releases_only_own_lock(self):
        self.run_fake_worker(quarantined=False)

    def test_worker_quarantined_exit_preserves_lock(self):
        self.run_fake_worker(quarantined=True)

    def run_fake_worker(self, quarantined):
        # Ownership is exercised by its own module/integration tests. This
        # child writes local test JSON only and cannot access a board.
        from types import SimpleNamespace
        self.config.update(helper_sha256="a" * 64, ownership_module_sha256="b" * 64,
                           ownership_config={"evidence": "SYNTHETIC_LOCAL_WORKER_ONLY"})
        self.upload()
        case = Path(self.config["package"]) / "case"
        # This fake child has no hardware imports and performs only local writes.
        runner = case / "board_runner.py"
        runner.write_text("import json,sys\nfrom pathlib import Path\n"
            "out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()\n"
            + ("(out/'device_buffer_retention.json').write_text(json.dumps(dict(status='QUARANTINED_DO_NOT_TERMINATE_PROCESS')))\nsys.exit(1)\n"
               if quarantined else "(out/'summary.json').write_text(json.dumps(dict(status='PASS')))\n"))
        manifest = bt.load(case / "transport_files.json")
        manifest["files"]["board_runner.py"] = bt.sha(runner)
        bt.save(case / "transport_files.json", manifest)
        job = self.root / "board/jobs/worker_test"
        job.mkdir(parents=True)
        binding = dict(job_id="worker_test")
        lock = self.root / "board/board.lock"
        bt.save(lock, binding)
        request = dict(config=self.config, job_id="worker_test", lock_binding=binding,
            board_id="machine-id:offline", timeout_seconds=1)
        bt.save(job / "request.json", request)
        owner = SimpleNamespace(
            process_identity=lambda *args, **kwargs: {"evidence": "SYNTHETIC_LOCAL_WORKER_ONLY"},
            check=lambda *args, **kwargs: {"status": "SYNTHETIC_LOCAL_WORKER_ONLY"})
        with patch.object(bt, "ownership_module", return_value=owner):
            bt.worker(job / "request.json")
        status = bt.load(job / "status.json")
        self.assertEqual(status["state"], "DEVICE_STATE_UNKNOWN" if quarantined else "PASS")
        self.assertEqual(lock.exists(), quarantined)


if __name__ == "__main__":
    unittest.main()
