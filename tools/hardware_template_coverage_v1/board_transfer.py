"""Finite board-case transport over the user's existing Jupyter reuse bridge.

prepare is local only. upload transfers immutable files without programming.
launch is the only action that starts board_runner (and programs the board).
No raw bridge output is persisted or displayed; only our JSON marker is read.
There is deliberately no reset/kill/retry-running-job action.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import zlib

MARKER = "PILOT_TRANSPORT_JSON="
DEFAULT_BRIDGE = "C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/jupyter_reuse_bridge.py"
DEFAULT_BOARD_ROOT = "/home/xilinx/hardware_template_coverage_20261013"
LEGACY_BOARD_ROOT = "/home/xilinx/hardware_measurement_pilot_20261012"
PROC_ROOT = Path("/proc")
CHUNK_BYTES = 12288  # Keeps twice-base64 code below Windows' 32767-char limit.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    save(temporary, value)
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_name(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or "\\" in name or p.is_absolute() or ".." in p.parts or ":" in name:
        raise ValueError("unsafe relative artifact path")
    return p


def child(root, name):
    p = Path(root).joinpath(*safe_name(name).parts)
    if not p.resolve().is_relative_to(Path(root).resolve()) or p.is_symlink():
        raise ValueError("artifact escapes root or is a symbolic link")
    return p


def unpack(archive, destination, *, manifest_name="transport_files.json"):
    """Validate all bytes before exclusive writes; reject links and traversal."""
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError("duplicate/nonregular archive member")
        for name in names:
            safe_name(name)
        manifest = json.load(tar.extractfile(manifest_name))
        if set(names) != set(manifest["files"]) | {manifest_name}:
            raise ValueError("archive member set differs")
        for member in members:
            if member.name == manifest_name:
                continue
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest["files"][member.name]:
                raise ValueError("archive member SHA256 differs: " + member.name)
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = child(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                source = tar.extractfile(member)
                for block in iter(lambda: source.read(1048576), b""):
                    stream.write(block)
    return manifest


def add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    tar.addfile(info, io.BytesIO(data))


def prepare(case_dir, output, *, mode="probe", bridge=DEFAULT_BRIDGE, board_root=DEFAULT_BOARD_ROOT, chunk_bytes=CHUNK_BYTES,
            ownership_config=None):
    from tools.hardware_template_coverage_v1 import board_runner
    root, bundle, case, _, _ = board_runner.validate_bundle(case_dir, mode=mode)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("transport output must be outside immutable case")
    if not PurePosixPath(board_root).is_absolute() or ".." in PurePosixPath(board_root).parts:
        raise ValueError("absolute isolated board root required")
    bridge = Path(bridge).resolve()
    if not bridge.is_file():
        raise ValueError("existing authorized bridge not found")
    if not isinstance(chunk_bytes, int) or not 1024 <= chunk_bytes <= 4 * 1024 * 1024:
        raise ValueError("chunk bytes must be within [1024, 4194304]")
    if bridge.name == "jupyter_reuse_bridge.py" and chunk_bytes > CHUNK_BYTES:
        raise ValueError("legacy argv bridge requires <=12288 byte chunks")
    files = dict(bundle["files"])
    manifest_name = "board_probe_bundle.json" if mode == "probe" else "board_bundle.json"
    files[manifest_name] = sha(root / manifest_name)
    output.mkdir(parents=True, exist_ok=False)
    transfer = dict(schema="pilot_board_transfer_files_v1", files=files)
    archive = output / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, digest in sorted(files.items()):
            target = child(root, name)
            if sha(target) != digest:
                raise ValueError("case changed while packaging")
            tar.add(target, arcname=name, recursive=False)
        add_bytes(tar, "transport_files.json", json.dumps(transfer, sort_keys=True).encode())
    digest = sha(archive)
    helper_sha = sha(__file__)
    config = dict(schema="pilot_board_transport_v1", created_utc=utc(), mode=mode,
        case_id=case["case_id"], case_sha256=case["case_sha256"], archive_sha256=digest,
        archive_bytes=archive.stat().st_size, bridge=str(bridge), bridge_sha256=sha(bridge),
        helper_sha256=helper_sha, board_root=board_root, chunk_bytes=chunk_bytes,
        package=board_root + "/packages/" + digest, formal_cost_eligible=False)
    if ownership_config is not None:
        config['ownership_config'] = ownership_config
        config['ownership_module_sha256'] = bundle['files']['board_ownership.py']
    save(output / "transport.json", config, exclusive=True)
    return config


def config_for(folder):
    folder = Path(folder).resolve()
    config = load(folder / "transport.json")
    if config.get("schema") != "pilot_board_transport_v1" or config.get("formal_cost_eligible") is not False:
        raise ValueError("invalid transport configuration")
    if sha(folder / "package.tar.gz") != config["archive_sha256"] or sha(config["bridge"]) != config["bridge_sha256"]:
        raise ValueError("archive/authorized bridge changed")
    if sha(__file__) != config["helper_sha256"]:
        raise ValueError("transfer helper changed; prepare a new transport")
    return folder, config


def bridge_call(folder, config, code, label):
    # Raw transport output may contain auth-service internals. Keep it in RAM.
    code_path = folder / ("request_" + label + ".py")
    code_path.write_text(code, encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, config["bridge"], str(code_path), os.devnull],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=100)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("bridge response timed out; inspect status before any launch retry") from None
    records = []
    for line in result.stdout.splitlines():
        start = line.find(MARKER)
        if start >= 0:
            try:
                records.append(json.loads(line[start + len(MARKER):]))
            except (ValueError, TypeError):
                pass
    if result.returncode or len(records) != 1:
        raise RuntimeError("bridge returned no unique safe receipt (raw transport output suppressed)")
    receipt = records[0]
    if receipt.get("ok") is not True:
        raise RuntimeError("board transport operation failed: " + receipt.get("error", "unspecified"))
    public = {k:v for k,v in receipt.items() if k != "data"}
    save(folder / ("receipt_" + label + ".json"), public)
    return receipt


def install_helper(folder, config):
    source = Path(__file__).read_bytes()
    helper = config["board_root"] + "/helpers/" + config["helper_sha256"] + ".py"
    compressed = base64.b64encode(zlib.compress(source, 9)).decode()
    code = f'''import base64,hashlib,json,zlib
from pathlib import Path
p=Path({helper!r});data=zlib.decompress(base64.b64decode({compressed!r}))
if hashlib.sha256(data).hexdigest()!={config['helper_sha256']!r}:raise ValueError('helper digest mismatch')
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
 if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!={config['helper_sha256']!r}:raise ValueError('existing helper differs')
else:
 with p.open('xb') as f:f.write(data)
print({MARKER!r}+json.dumps(dict(ok=True,helper=str(p))))
'''
    bridge_call(folder, config, code, "helper")


def call(folder, config, action, **kwargs):
    request = dict(action=action, config=config, **kwargs)
    helper = config["board_root"] + "/helpers/" + config["helper_sha256"] + ".py"
    code = f'''import hashlib,json,runpy
from pathlib import Path
p=Path({helper!r})
if hashlib.sha256(p.read_bytes()).hexdigest()!={config['helper_sha256']!r}:raise ValueError('helper changed')
ns=runpy.run_path(str(p));request={request!r}
try:answer=ns['remote_action'](request)
except Exception as error:answer=dict(ok=False,error=type(error).__name__+': '+str(error))
print({MARKER!r}+json.dumps(answer))
'''
    label = action + ("_" + str(kwargs["offset"]) if "offset" in kwargs else "")
    return bridge_call(folder, config, code, label)


def upload(folder):
    folder, config = config_for(folder)
    install_helper(folder, config)
    receipt = call(folder, config, "upload_init")
    if receipt.get("ready"):
        return receipt
    with (folder / "package.tar.gz").open("rb") as stream:
        stream.seek(receipt["offset"])
        offset = receipt["offset"]
        while block := stream.read(config["chunk_bytes"]):
            receipt = call(folder, config, "upload_chunk", offset=offset,
                data=base64.b64encode(block).decode(), digest=hashlib.sha256(block).hexdigest())
            offset = receipt["offset"]
            print(json.dumps(dict(status="UPLOADING", bytes=offset, total=config["archive_bytes"])), flush=True)
    return call(folder, config, "upload_commit")


def valid_job(job_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", job_id):
        raise ValueError("invalid job ID")
    return job_id


def remote_paths(config):
    root = Path(config["board_root"])
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("invalid board root")
    digest = config["archive_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid archive digest")
    package = root / "packages" / digest
    if str(package) != config["package"] or root.is_symlink() or package.is_symlink():
        raise ValueError("package path binding differs")
    return root, package


def process_state(pid):
    try:
        raw = (PROC_ROOT / str(pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=pid, state=fields[0], start_ticks=fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


class BoardConflict(RuntimeError):
    """A safe, read-only observation, never a claim of global exclusion."""
    def __init__(self, observation):
        self.observation = observation
        super().__init__("BOARD_CONFLICT: " + json.dumps(observation, sort_keys=True, allow_nan=False))


def _lock_observation(path):
    """Never include a lock's owner_token, arbitrary fields, or raw content."""
    value = dict(path=str(path), present=False)
    try:
        try:
            path.lstat()
        except FileNotFoundError:
            return value, None
        value["present"] = True
        if path.is_symlink() or not path.is_file():
            value["read_status"] = "NONREGULAR_LOCK"
            return value, None
        raw = path.read_bytes()
        value.update(read_status="READ", size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        binding = json.loads(raw)
        if not isinstance(binding, dict):
            raise ValueError("lock is not a mapping")
        public = {}
        for key in ("job_id", "condition_id", "created_utc", "schema", "state", "status"):
            item = binding.get(key)
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_:./+ -]{1,160}", item):
                public[key] = item
        value["public_binding"] = public
        return value, binding
    except (OSError, ValueError, TypeError):
        value.update(present=True, read_status="LOCK_INSPECTION_UNAVAILABLE")
        return value, None


def board_conflict_snapshot(board_root, *, own_lock_binding=None, exclude_pids=()):
    """Inspect known locks/runners without writing, deleting, killing, or resetting.

    Legacy/external workflows do not cooperate with this namespace's lock. These
    observations reduce known conflicts; they are not a global mutex proof.
    """
    root = Path(board_root)
    roots = [("legacy_pilot", Path(LEGACY_BOARD_ROOT)),
             ("template_coverage", Path(DEFAULT_BOARD_ROOT)), ("template_coverage", root)]
    unique_roots = []
    for family, directory in roots:
        if all(directory.resolve() != previous.resolve() for _, previous in unique_roots):
            unique_roots.append((family, directory))
    locks, processes, conflicts = [], [], []
    for family, directory in unique_roots:
        for name in ("board.lock", "system_conditions.lock"):
            observation, binding = _lock_observation(directory / name)
            observation.update(workflow=family, lock_kind=name)
            allowed = (directory.resolve() == root.resolve() and name == "board.lock" and
                       own_lock_binding is not None and binding == own_lock_binding)
            observation["own_worker_lock"] = allowed
            locks.append(observation)
            if observation["present"] and not allowed:
                conflicts.append(dict(kind="KNOWN_LOCK_PRESENT", path=observation["path"],
                                      read_status=observation.get("read_status")))
    try:
        entries = list(PROC_ROOT.iterdir())
    except OSError:
        entries = []
        conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE"))
    excluded = set(exclude_pids)
    for directory in entries:
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        pid = int(directory.name)
        try:
            argv = (directory / "cmdline").read_bytes().split(b"\0")
            argv = [item.decode("utf-8", errors="replace") for item in argv if item]
            selected = None
            for item in argv:
                script = Path(item)
                if script.suffix != ".py":
                    continue
                role = "runner" if script.name == "board_runner.py" else "worker" if "_worker" in argv else None
                if role is None:
                    continue
                if not script.is_absolute():
                    script = Path(os.readlink(directory / "cwd")) / script
                for family, known_root in unique_roots:
                    if script.resolve().is_relative_to(known_root.resolve()):
                        selected = dict(pid=pid, workflow=family, role=role)
                        break
                if selected:
                    break
            if selected is not None:
                current = process_state(pid)
                if current is not None and current["state"] != "Z":
                    selected.update(current)
                    processes.append(selected)
                    conflicts.append(dict(kind="ACTIVE_KNOWN_RUNNER", **selected))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process may exit during this read-only inspection.
        except (OSError, ValueError, IndexError):
            conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE", pid=pid))
    return dict(schema="template_board_conflict_observation_v1", observed_utc=utc(),
        status="CONFLICT" if conflicts else "NO_KNOWN_CONFLICT_OBSERVED", locks=locks,
        active_processes=processes, conflicts=conflicts, read_only=True,
        exclusion_scope="KNOWN_WORKFLOWS_ONLY_NOT_GLOBAL_MUTEX_PROOF")


def require_no_board_conflict(board_root, *, own_lock_binding=None, exclude_pids=()):
    observation = board_conflict_snapshot(board_root, own_lock_binding=own_lock_binding, exclude_pids=exclude_pids)
    if observation["conflicts"]:
        raise BoardConflict(observation)
    return observation


def ownership_module(config, package):
    """Load only the source-bound ownership helper; absent binding fails closed."""
    if not isinstance(config.get('ownership_config'), dict):
        raise ValueError('shared machine reservation configuration is required')
    source = Path(package) / 'case' / 'board_ownership.py'
    if not re.fullmatch(r'[0-9a-f]{64}', config.get('ownership_module_sha256', '')) or sha(source) != config['ownership_module_sha256']:
        raise ValueError('source-bound ownership helper differs')
    spec = importlib.util.spec_from_file_location('t6_bound_board_ownership', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote_action(request):
    config = request["config"]
    root, package = remote_paths(config)
    action = request["action"]
    if action == "upload_init":
        package.mkdir(parents=True, exist_ok=True)
        binding = package / "transport.json"
        if binding.exists():
            if load(binding) != config:
                raise ValueError("existing package configuration differs")
        else:
            save(binding, config, exclusive=True)
        ready = package / "READY.json"
        archive = package / "package.tar.gz"
        return dict(ok=True, ready=ready.exists(), offset=archive.stat().st_size if archive.exists() else 0)
    if not (package / "transport.json").is_file() or load(package / "transport.json") != config:
        raise ValueError("upload configuration not established")
    if action == "upload_chunk":
        if (package / "READY.json").exists():
            raise ValueError("immutable package is already ready")
        data = base64.b64decode(request["data"], validate=True)
        if len(data) > config["chunk_bytes"] or hashlib.sha256(data).hexdigest() != request["digest"]:
            raise ValueError("chunk size/digest differs")
        archive = package / "package.tar.gz"
        offset = int(request["offset"])
        current = archive.stat().st_size if archive.exists() else 0
        if offset < 0 or offset + len(data) > config["archive_bytes"]:
            raise ValueError("chunk exceeds declared archive")
        if offset < current:
            with archive.open("rb") as stream:
                stream.seek(offset)
                if stream.read(len(data)) != data:
                    raise ValueError("conflicting upload replay")
            return dict(ok=True, offset=offset + len(data), replay=True)
        if offset != current:
            raise ValueError("chunk offset has a gap")
        with archive.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return dict(ok=True, offset=offset + len(data))
    if action == "upload_commit":
        archive = package / "package.tar.gz"
        if archive.stat().st_size != config["archive_bytes"] or sha(archive) != config["archive_sha256"]:
            raise ValueError("complete archive size/SHA256 differs")
        if (package / "READY.json").exists():
            return dict(ok=True, ready=True, package=str(package))
        manifest = unpack(archive, package / "case")
        ready = dict(ok=True, ready=True, package=str(package), files=len(manifest["files"]), archive_sha256=sha(archive))
        save(package / "READY.json", ready, exclusive=True)
        return ready
    job_id = valid_job(request["job_id"])
    job = root / "jobs" / job_id
    lock = root / "board.lock"
    if action == "launch":
        # This must precede every launch-side write, including the local lock.
        inspection = require_no_board_conflict(root)
        owner = ownership_module(config, package)
        ownership = owner.check(config['ownership_config'])
        if not (package / "READY.json").is_file():
            raise ValueError("package not ready")
        timeout = float(request["timeout_seconds"])
        if timeout <= 0 or timeout > 3600:
            raise ValueError("per-call timeout out of range")
        board_id = request["board_id"]
        if not re.fullmatch(r"(?:serial|machine-id):[A-Za-z0-9_-]+", board_id):
            raise ValueError("board identity must be explicit")
        if job.exists():
            raise ValueError("job already exists; inspect status, never relaunch it")
        lock.parent.mkdir(parents=True, exist_ok=True)
        binding = dict(job_id=job_id, package=str(package), archive_sha256=config["archive_sha256"], created_utc=utc())
        save(lock, binding, exclusive=True)
        job.mkdir(parents=True, exist_ok=False)
        worker_request = dict(config=config, job_id=job_id, board_id=board_id, timeout_seconds=timeout, lock_binding=binding)
        save(job / "request.json", worker_request, exclusive=True)
        save(job / "prelaunch_inspection.json", inspection, exclusive=True)
        save(job / "prelaunch_ownership.json", ownership, exclusive=True)
        save(job / "status.json", dict(state="STARTING", job_id=job_id, created_utc=utc()), exclusive=True)
        helper = root / "helpers" / (config["helper_sha256"] + ".py")
        if sha(helper) != config["helper_sha256"]:
            raise ValueError("worker source changed; lock retained")
        with (job / "worker.log").open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, str(helper), "_worker", str(job / "request.json")],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, cwd=job, start_new_session=True)
        receipt = dict(ok=True, job_id=job_id, worker_pid=process.pid, worker_process=process_state(process.pid),
            package=str(package), state="DISPATCHED", mode=config["mode"])
        save(job / "launch.json", receipt, exclusive=True)
        return receipt
    if not (job / "request.json").is_file() or load(job / "request.json")["config"] != config:
        raise ValueError("job/package binding differs")
    if action == "status":
        value = load(job / "status.json")
        for key in ("worker_pid", "runner_pid"):
            if value.get(key):
                value[key + "_current_process"] = process_state(value[key])
        value.update(ok=True, lock=load(lock) if lock.exists() else None)
        return value
    if action == "snapshot":
        snapshot_name = "snapshot_" + str(time.time_ns())
        selected = {}
        for name in ("request.json", "status.json", "launch.json", "worker.log", "runner.log",
                     "prelaunch_inspection.json", "worker_prelaunch_inspection.json",
                     "prelaunch_ownership.json", "worker_prelaunch_ownership.json", "ownership_context.json"):
            path = job / name
            if path.is_file() and not path.is_symlink():
                selected[name] = path.read_bytes()
        output = job / "output"
        for path in output.rglob("*") if output.exists() else []:
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(output.resolve()):
                selected[path.relative_to(job).as_posix()] = path.read_bytes()
        if sum(map(len, selected.values())) > 128 * 1024 * 1024:
            raise ValueError("snapshot exceeds finite evidence limit")
        manifest = dict(schema="pilot_board_snapshot_v1", job_id=job_id, created_utc=utc(),
            observed_state=load(job / "status.json")["state"],
            files={name:hashlib.sha256(data).hexdigest() for name,data in selected.items()})
        archive = job / (snapshot_name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in sorted(selected.items()):
                add_bytes(tar, name, data)
            add_bytes(tar, "snapshot_manifest.json", json.dumps(manifest, sort_keys=True).encode())
        return dict(ok=True, snapshot_name=snapshot_name, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
            observed_state=manifest["observed_state"])
    if action == "fetch_chunk":
        name = request["snapshot_name"]
        if not re.fullmatch(r"snapshot_[0-9]+", name):
            raise ValueError("invalid snapshot name")
        archive = job / (name + ".tar.gz")
        offset = int(request["offset"])
        if offset < 0 or offset >= archive.stat().st_size:
            raise ValueError("invalid download offset")
        with archive.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(config["chunk_bytes"])
        return dict(ok=True, offset=offset, bytes=len(data), digest=hashlib.sha256(data).hexdigest(), data=base64.b64encode(data).decode())
    raise ValueError("unsupported operation")


def worker(request_path):
    request = load(request_path)
    config = request["config"]
    root, package = remote_paths(config)
    job = Path(request_path).parent
    lock = root / "board.lock"
    value = dict(state="STARTING", job_id=request["job_id"], worker_pid=os.getpid(),
        worker_process=process_state(os.getpid()), started_utc=utc(), mode=config["mode"])
    try:
        if load(lock) != request["lock_binding"]:
            raise ValueError("exclusive board lock differs")
        # Recheck after dispatch and immediately before starting the runner.
        inspection = require_no_board_conflict(root, own_lock_binding=request["lock_binding"],
                                                exclude_pids=(os.getpid(),))
        save(job / "worker_prelaunch_inspection.json", inspection, exclusive=True)
        atomic(job / "status.json", value)
        files = load(package / "case" / "transport_files.json")["files"]
        if any(sha(child(package / "case", name)) != digest for name,digest in files.items()):
            raise ValueError("immutable case files changed")
        command = [sys.executable, str(package / "case" / "board_runner.py"), "--case-dir", str(package / "case"),
            "--output", str(job / "output"), "--mode", config["mode"], "--board-id", request["board_id"],
            "--timeout-seconds", str(request["timeout_seconds"])]
        owner = ownership_module(config, package)
        own_worker = owner.process_identity(os.getpid(), 'worker', request_path=str(Path(request_path).resolve()),
            request_sha256=sha(request_path), helper_path=str(root / 'helpers' / (config['helper_sha256'] + '.py')))
        own_board_lock = dict(path=str(lock), binding=request['lock_binding'], sha256=sha(lock))
        ownership = owner.check(config['ownership_config'], own_board_lock=own_board_lock, own_worker=own_worker)
        save(job / 'worker_prelaunch_ownership.json', ownership, exclusive=True)
        context = dict(schema='t6_runner_ownership_context_v1', package=str(package / 'case'),
            ownership_config=config['ownership_config'], ownership_module_sha256=config['ownership_module_sha256'],
            own_worker=own_worker, own_board_lock=own_board_lock,
            request_path=str(Path(request_path).resolve()), request_sha256=sha(request_path), runner_argv_prefix=command)
        context_path = job / 'ownership_context.json'
        save(context_path, context, exclusive=True)
        command += ['--ownership-request', str(context_path), '--ownership-request-sha256', sha(context_path)]
        with (job / "runner.log").open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                cwd=job, start_new_session=True)
            value.update(state="RUNNING", runner_pid=process.pid, runner_process=process_state(process.pid))
            while process.poll() is None:
                retained = job / "output" / "device_buffer_retention.json"
                try:
                    retained_status = load(retained).get("status", "") if retained.exists() else ""
                except (ValueError, OSError):
                    retained_status = "PENDING_RETENTION_RECORD"
                value["state"] = "DMA_QUARANTINED" if retained_status.startswith("QUARANTINED") else "RUNNING"
                value["updated_utc"] = utc()
                atomic(job / "status.json", value)
                time.sleep(1)
        code = process.returncode
        value.update(returncode=code, finished_utc=utc())
        retained = job / "output" / "device_buffer_retention.json"
        unsafe = code < 0 or (retained.exists() and load(retained).get("status", "").startswith("QUARANTINED"))
        value["state"] = "DEVICE_STATE_UNKNOWN" if unsafe else ("PASS" if code == 0 else "FAIL")
        value["board_lock_retained"] = unsafe
        if not unsafe and lock.exists() and load(lock) == request["lock_binding"]:
            lock.unlink()
        atomic(job / "status.json", value)
    except BaseException as error:
        value.update(state="WORKER_FAILED_LOCK_RETAINED", error=type(error).__name__ + ": " + str(error),
            board_lock_retained=True, finished_utc=utc())
        atomic(job / "status.json", value)
        raise


def fetch(folder, job_id, output):
    folder, config = config_for(folder)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = call(folder, config, "snapshot", job_id=valid_job(job_id))
    save(output / "download.json", snapshot, exclusive=True)
    archive = output / "snapshot.tar.gz"
    offset = 0
    with archive.open("xb") as stream:
        while offset < snapshot["archive_bytes"]:
            receipt = call(folder, config, "fetch_chunk", job_id=job_id,
                snapshot_name=snapshot["snapshot_name"], offset=offset)
            data = base64.b64decode(receipt["data"], validate=True)
            if receipt["offset"] != offset or len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["digest"]:
                raise ValueError("download chunk validation failed")
            stream.write(data)
            offset += len(data)
    if archive.stat().st_size != snapshot["archive_bytes"] or sha(archive) != snapshot["archive_sha256"]:
        raise ValueError("download archive validation failed")
    manifest = unpack(archive, output / "evidence", manifest_name="snapshot_manifest.json")
    return dict(status="VERIFIED_SNAPSHOT", output=str(output), observed_state=manifest["observed_state"], files=len(manifest["files"]))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        worker(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    local = sub.add_parser("prepare")
    local.add_argument("--case-dir", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--mode", choices=("probe", "timing"), default="probe")
    local.add_argument("--bridge", default=DEFAULT_BRIDGE)
    local.add_argument("--board-root", default=DEFAULT_BOARD_ROOT)
    local.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    local.add_argument("--ownership-config", type=Path)
    for name in ("upload", "launch", "status", "fetch"):
        command = sub.add_parser(name)
        command.add_argument("--transport-dir", required=True)
        if name != "upload":
            command.add_argument("--job-id", required=True)
        if name == "launch":
            command.add_argument("--board-id", required=True)
            command.add_argument("--timeout-seconds", type=float, default=30)
        if name == "fetch":
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.case_dir, args.output, mode=args.mode, bridge=args.bridge, board_root=args.board_root,
                         chunk_bytes=args.chunk_bytes, ownership_config=load(args.ownership_config) if args.ownership_config else None)
    elif args.action == "upload":
        result = upload(args.transport_dir)
    elif args.action == "fetch":
        result = fetch(args.transport_dir, args.job_id, args.output)
    else:
        folder, config = config_for(args.transport_dir)
        kwargs = dict(job_id=valid_job(args.job_id))
        if args.action == "launch":
            kwargs.update(board_id=args.board_id, timeout_seconds=args.timeout_seconds)
        result = call(folder, config, args.action, **kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
