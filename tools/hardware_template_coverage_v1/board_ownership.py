"""T6 cooperative machine reservation and fail-closed known-workflow checks.

Adapted from frozen tools/hardware_measurement_pilot/d09_board_ownership_v1.py
SHA256 78a181010ed09c70189a49a841e5c53bd252b2c96df68a5ef09437464807a01d.
The shared reservation path and JSON schema remain compatible with D09.
This checks two known workflows; it is not OS-wide or hardware mutual exclusion.
No reset, process signal, stale-lock deletion or automatic recovery is offered.
Call only outside measurement intervals. Python 3.8, standard library only.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

SCHEMA = "t6_board_ownership_config_v1"
RESERVATION_SCHEMA = "d09_shared_board_reservation_v1"
PROCESS_SCHEMA = "t6_known_board_process_identity_v1"
SOURCE_REFERENCE_SHA256 = "78a181010ed09c70189a49a841e5c53bd252b2c96df68a5ef09437464807a01d"
RESERVATION_ROOT = "/home/xilinx/.hwnas_board_ownership"
KNOWN_ROOTS = (
    "/home/xilinx/hardware_measurement_pilot_20261012",
    "/home/xilinx/hardware_template_coverage_20261013",
)
OWN_BOARD_LOCK = KNOWN_ROOTS[1] + "/board.lock"
RUNNERS = {"board_runner.py", "d09_cpu_runner.py"}
REQUEST_FLAG = "--ownership-request"
REQUEST_SHA_FLAG = "--ownership-request-sha256"
SCOPE = "TWO_KNOWN_WORKFLOWS_AND_COOPERATIVE_RESERVATION_NOT_GLOBAL_HARD_EXCLUSION"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class OwnershipError(RuntimeError):
    def __init__(self, reason, observation=None):
        self.reason, self.observation = reason, observation
        # Never echo arbitrary lock contents, argv or owner tokens in errors.
        super().__init__(reason)


class Backend:
    proc_root = Path("/proc")

    def path(self, logical):
        return Path(logical)

    def identity(self):
        return dict(machine_id=Path("/etc/machine-id").read_text().strip(),
                    boot_id=(self.proc_root / "sys/kernel/random/boot_id").read_text().strip())

    def process(self, pid):
        directory = self.proc_root / str(pid)
        first = (directory / "stat").read_text()
        fields = first[first.rfind(")") + 2:].split()
        cmdline = (directory / "cmdline").read_bytes()
        argv = [item.decode("utf-8", errors="strict") for item in cmdline.split(b"\0") if item]
        cwd = os.readlink(directory / "cwd") if argv else None
        after = (directory / "stat").read_text()
        final = after[after.rfind(")") + 2:].split()
        last_cmdline = (directory / "cmdline").read_bytes()
        last_cwd = os.readlink(directory / "cwd") if argv else None
        if (fields[19] != final[19] or cmdline != last_cmdline or cwd != last_cwd):
            raise OwnershipError("PROCESS_IDENTITY_CHANGED_DURING_INSPECTION")
        if not final[19].isdigit() or not final[0] or len(final[0]) != 1:
            raise OwnershipError("INVALID_PROCESS_STAT")
        return dict(pid=pid, state=final[0], start_ticks=final[19], argv=argv, cwd=cwd)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _config(config):
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise OwnershipError("INVALID_OWNERSHIP_SCHEMA")
    for key in ("campaign_id", "owner_id"):
        if not isinstance(config.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", config[key]):
            raise OwnershipError("INVALID_OWNER_ID")
    if not isinstance(config.get("expected_machine_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", config["expected_machine_id"]):
        raise OwnershipError("INVALID_MACHINE_ID")
    try:
        if str(uuid.UUID(config.get("expected_boot_id", ""))) != config["expected_boot_id"]:
            raise ValueError()
    except (ValueError, AttributeError, TypeError, KeyError):
        raise OwnershipError("INVALID_BOOT_ID") from None
    if (config.get("reservation_root") != RESERVATION_ROOT or
            config.get("known_board_roots") != list(KNOWN_ROOTS)):
        raise OwnershipError("OWNERSHIP_NAMESPACE_CHANGED")
    if "system_conditions_lock" in config:
        raise OwnershipError("T6_SYSTEM_CONDITIONS_EXEMPTION_NOT_SUPPORTED")
    return config


def _reservation_path(config):
    return RESERVATION_ROOT + "/" + config["expected_machine_id"] + ".json"


def _regular(path):
    # Reject symlinks in every ancestor, including dangling links.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise OwnershipError("SYMLINK_IN_OWNERSHIP_PATH")
    if not path.is_file():
        raise OwnershipError("OWNERSHIP_FILE_MISSING_OR_NONREGULAR")
    return path.read_bytes()


def _identity(config, backend):
    value = backend.identity()
    if value != dict(machine_id=config["expected_machine_id"], boot_id=config["expected_boot_id"]):
        raise OwnershipError("BOOT_OR_MACHINE_ID_CHANGED")
    return value


def _t6_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise OwnershipError("INVALID_OWN_PROCESS_PATH")
    path = PurePosixPath(value)
    root = PurePosixPath(KNOWN_ROOTS[1])
    if not path.is_absolute() or ".." in path.parts or str(path) != value or root not in path.parents:
        raise OwnershipError("OWN_PROCESS_OUTSIDE_T6_ROOT")
    return path


def _role(row):
    """Recognize direct/taskset/inline runners and workers in both namespaces."""
    argv = row["argv"]
    candidates = list(argv)
    if "-c" in argv:
        index = argv.index("-c")
        if index + 1 >= len(argv):
            raise OwnershipError("MALFORMED_INLINE_PROCESS")
        code = argv[index + 1]
        try:
            candidates += [node.value for node in ast.walk(ast.parse(code))
                           if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        except SyntaxError:
            if any(root in code for root in KNOWN_ROOTS):
                return "unknown_known_workflow"
    for item in candidates:
        if not isinstance(item, str) or "\0" in item:
            continue
        path = PurePosixPath(item)
        if not path.is_absolute() and row.get("cwd") and path.suffix == ".py":
            path = PurePosixPath(row["cwd"]) / path
        known = any(path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in KNOWN_ROOTS)
        if known and path.name in RUNNERS:
            return "runner"
        if known and path.suffix == ".py" and "_worker" in argv:
            return "worker"
    joined = " ".join(argv)
    if any(root in joined for root in KNOWN_ROOTS) and any(name in joined for name in RUNNERS):
        return "inline_runner"
    return None


def _process_contract(value, role, backend):
    if (not isinstance(value, dict) or value.get("schema") != PROCESS_SCHEMA or
            value.get("role") != role or type(value.get("pid")) is not int or value["pid"] <= 0 or
            not isinstance(value.get("start_ticks"), str) or not value["start_ticks"].isdigit()):
        raise OwnershipError("INVALID_OWN_PROCESS_IDENTITY")
    argv = value.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise OwnershipError("INVALID_OWN_PROCESS_ARGV")
    _t6_path(value.get("cwd"))
    request = str(_t6_path(value.get("request_path")))
    script_key = "helper_path" if role == "worker" else "runner_path"
    script = _t6_path(value.get(script_key))
    if not _hash(value.get("request_sha256")) or not _hash(value.get("script_sha256")):
        raise OwnershipError("INVALID_OWN_PROCESS_FILE_HASH")
    if role == "worker":
        if len(argv) != 4 or argv[1:] != [str(script), "_worker", request] or script.suffix != ".py" or script.name in RUNNERS:
            raise OwnershipError("OWN_WORKER_ARGV_BINDING_CHANGED")
    elif role == "runner":
        if len(argv) < 6 or argv[1] != str(script) or script.name != "board_runner.py":
            raise OwnershipError("OWN_RUNNER_ARGV_BINDING_CHANGED")
        for flag, expected in ((REQUEST_FLAG, request), (REQUEST_SHA_FLAG, value["request_sha256"])):
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != expected:
                raise OwnershipError("OWN_RUNNER_REQUEST_BINDING_CHANGED")
    else:
        raise OwnershipError("INVALID_OWN_PROCESS_ROLE")
    for logical, expected in ((request, value["request_sha256"]), (str(script), value["script_sha256"])):
        if digest(_regular(backend.path(logical))) != expected:
            raise OwnershipError("OWN_PROCESS_FILE_BINDING_CHANGED")
    return value


def process_identity(pid, role, *, request_path, request_sha256, helper_path=None,
                     runner_path=None, expected_argv=None, backend=None):
    """Snapshot an exact direct T6 process, its script, and immutable request.

    Worker argv must be [python, helper_path, '_worker', request_path]. Runner
    argv must directly invoke runner_path and bind the two REQUEST_FLAG values.
    When expected_argv is supplied, the complete /proc argv must match it.
    This performs read-only checks; it does not grant permission or reserve.
    """
    backend = backend or Backend()
    if type(pid) is not int or pid <= 0 or role not in ("worker", "runner"):
        raise OwnershipError("INVALID_OWN_PROCESS_IDENTITY")
    row = backend.process(pid)
    if row["state"] == "Z" or not row["argv"]:
        raise OwnershipError("OWN_PROCESS_NOT_ALIVE")
    if expected_argv is not None and row["argv"] != expected_argv:
        raise OwnershipError("OWN_PROCESS_EXPECTED_ARGV_CHANGED")
    path = helper_path if role == "worker" else runner_path
    if (role == "worker" and runner_path is not None) or (role == "runner" and helper_path is not None):
        raise OwnershipError("INVALID_OWN_PROCESS_ROLE_PATH")
    script = str(_t6_path(path))
    value = dict(schema=PROCESS_SCHEMA, role=role, pid=pid, start_ticks=row["start_ticks"],
                 argv=list(row["argv"]), cwd=row["cwd"], request_path=request_path,
                 request_sha256=request_sha256, script_sha256=digest(_regular(backend.path(script))))
    value["helper_path" if role == "worker" else "runner_path"] = script
    _process_contract(value, role, backend)
    if _role(row) != role:
        raise OwnershipError("OWN_PROCESS_ROLE_CHANGED")
    return value


def _owned_process(row, expected, role):
    if expected is None or row["pid"] != expected["pid"]:
        return False
    if (row["start_ticks"] != expected["start_ticks"] or row["state"] == "Z" or
            row["argv"] != expected["argv"] or row.get("cwd") != expected["cwd"] or _role(row) != role):
        raise OwnershipError("OWN_" + role.upper() + "_IDENTITY_CHANGED")
    return True


def _scan(config, backend, *, own_board_lock=None, own_worker=None, own_runner=None):
    locks, processes, owned_processes, conflicts = [], [], [], []
    own_board_seen = False
    for root in KNOWN_ROOTS:
        for name in ("board.lock", "system_conditions.lock"):
            logical = root + "/" + name
            path = backend.path(logical)
            record = dict(path=logical, present=False, owned=False)
            try:
                path.lstat()
            except FileNotFoundError:
                locks.append(record)
                continue
            except OSError as error:
                record.update(present=True, inspection="UNAVAILABLE_OR_MISMATCHED")
                conflicts.append(dict(kind="LOCK_INSPECTION_FAILED", path=logical, error_type=type(error).__name__))
                locks.append(record)
                continue
            try:
                raw = _regular(path)
                value = _json(raw)
                record.update(present=True, sha256=digest(raw))
                if own_board_lock and logical == own_board_lock["path"]:
                    if digest(raw) != own_board_lock["sha256"] or value != own_board_lock["binding"]:
                        raise OwnershipError("OWN_BOARD_LOCK_CHANGED")
                    record["owned"] = own_board_seen = True
                if not record["owned"]:
                    conflicts.append(dict(kind="KNOWN_LOCK_PRESENT", path=logical))
            except (OSError, ValueError, TypeError, OwnershipError) as error:
                record.update(present=True, inspection="UNAVAILABLE_OR_MISMATCHED")
                conflicts.append(dict(kind="LOCK_INSPECTION_FAILED", path=logical, error_type=type(error).__name__))
            locks.append(record)
    if own_board_lock and not own_board_seen:
        conflicts.append(dict(kind="OWN_BOARD_LOCK_NOT_VERIFIED"))
    try:
        entries = list(backend.proc_root.iterdir())
    except OSError:
        entries = []
        conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE"))
    seen = dict(worker=False, runner=False)
    for directory in entries:
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            row = backend.process(pid)
            if row["state"] == "Z":
                continue
            owned = False
            for role, expected in (("worker", own_worker), ("runner", own_runner)):
                if _owned_process(row, expected, role):
                    seen[role] = owned = True
                    owned_processes.append(dict(pid=pid, state=row["state"], start_ticks=row["start_ticks"], role=role))
                    break
            if owned:
                continue
            role = _role(row)
            if role:
                public = {key: row[key] for key in ("pid", "state", "start_ticks")}
                public["role"] = role
                processes.append(public)
                conflicts.append(dict(kind="ACTIVE_KNOWN_BOARD_PROCESS", **public))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, OwnershipError) as error:
            conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE", pid=pid, error_type=type(error).__name__))
    for role, expected in (("worker", own_worker), ("runner", own_runner)):
        if expected and not seen[role]:
            conflicts.append(dict(kind="OWN_" + role.upper() + "_NOT_VERIFIED"))
    result = dict(schema="t6_known_board_occupancy_v1", observed_utc=utc(), locks=locks,
                  active_processes=processes, owned_processes=owned_processes, conflicts=conflicts,
                  status="CONFLICT" if conflicts else "NO_KNOWN_CONFLICT_OBSERVED", scope=SCOPE)
    if conflicts:
        raise OwnershipError("BOARD_OCCUPANCY_CONFLICT", result)
    return result


def _own_lock(value):
    if value is not None and (not isinstance(value, dict) or value.get("path") != OWN_BOARD_LOCK or
            not _hash(value.get("sha256")) or not isinstance(value.get("binding"), dict)):
        raise OwnershipError("INVALID_OWN_T6_BOARD_LOCK")


def reserve(config, *, backend=None):
    """Explicit atomic reservation; conflicts retain evidence, no stale cleanup."""
    backend = backend or Backend()
    _config(config)
    if "reservation" in config:
        raise OwnershipError("RESERVE_REQUIRES_UNBOUND_BASE_CONFIG")
    identity = _identity(config, backend)
    before = _scan(config, backend)
    logical = _reservation_path(config)
    path = backend.path(logical)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise OwnershipError("SYMLINK_IN_OWNERSHIP_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    binding = dict(schema=RESERVATION_SCHEMA, campaign_id=config["campaign_id"], owner_id=config["owner_id"],
                   identity=identity, created_utc=utc(), known_board_roots=list(KNOWN_ROOTS),
                   release_requires_explicit_owner_check=True)
    raw = (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise OwnershipError("SHARED_BOARD_RESERVATION_ALREADY_EXISTS") from None
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = dict(path=logical, sha256=digest(raw))
    try:
        _identity(config, backend)
        after = _scan(config, backend)
    except BaseException as error:
        raise OwnershipError("POST_RESERVE_CHECK_FAILED_RESERVATION_RETAINED",
            dict(reservation=receipt, reason=getattr(error, "reason", type(error).__name__),
                 occupancy=getattr(error, "observation", None))) from None
    return dict(ok=True, status="RESERVED", reservation=receipt, binding=binding, before=before, after=after, scope=SCOPE)


def check(config, *, backend=None, own_board_lock=None, own_worker=None, own_runner=None):
    """Read-only admission check; an exact current shared reservation is required."""
    backend = backend or Backend()
    _config(config)
    identity = _identity(config, backend)
    ref = config.get("reservation", {})
    if not isinstance(ref, dict) or ref.get("path") != _reservation_path(config) or not _hash(ref.get("sha256")):
        raise OwnershipError("INVALID_RESERVATION_REFERENCE")
    path = backend.path(ref["path"])
    raw = _regular(path)
    binding = _json(raw)
    if (digest(raw) != ref["sha256"] or binding.get("schema") != RESERVATION_SCHEMA or
            binding.get("identity") != identity or binding.get("campaign_id") != config["campaign_id"] or
            binding.get("owner_id") != config["owner_id"] or binding.get("known_board_roots") != list(KNOWN_ROOTS) or
            binding.get("release_requires_explicit_owner_check") is not True):
        raise OwnershipError("SHARED_RESERVATION_OWNERSHIP_CHANGED")
    _own_lock(own_board_lock)
    for role, value in (("worker", own_worker), ("runner", own_runner)):
        if value is not None:
            _process_contract(value, role, backend)
    if own_worker and own_runner and own_worker["pid"] == own_runner["pid"]:
        raise OwnershipError("OWN_PROCESS_ROLES_SHARE_PID")
    occupancy = _scan(config, backend, own_board_lock=own_board_lock, own_worker=own_worker, own_runner=own_runner)
    # Detect lease, process input file, and boot changes during inspection.
    _identity(config, backend)
    if _regular(path) != raw:
        raise OwnershipError("SHARED_RESERVATION_CHANGED_DURING_CHECK")
    for role, value in (("worker", own_worker), ("runner", own_runner)):
        if value is not None:
            _process_contract(value, role, backend)
    if own_board_lock and digest(_regular(backend.path(OWN_BOARD_LOCK))) != own_board_lock["sha256"]:
        raise OwnershipError("OWN_BOARD_LOCK_CHANGED_DURING_CHECK")
    return dict(ok=True, status="PASS_OWNED_NO_KNOWN_CONFLICT", identity=identity,
                reservation=dict(ref), occupancy=occupancy, scope=SCOPE)


def release(config, *, backend=None):
    """Archive only this exact lease with all known job/system locks absent.

    This does not certify CPU/service/IRQ restoration. Any required restoration
    evidence must independently pass before the campaign owner invokes release.
    """
    backend = backend or Backend()
    result = check(config, backend=backend)
    path = backend.path(config["reservation"]["path"])
    raw = _regular(path)
    if digest(raw) != config["reservation"]["sha256"]:
        raise OwnershipError("SHARED_RESERVATION_CHANGED_BEFORE_RELEASE")
    archived = path.with_name(path.stem + ".released." + config["owner_id"] + "." + uuid.uuid4().hex + ".json")
    if archived.exists() or archived.is_symlink():
        raise OwnershipError("RELEASE_ARCHIVE_ALREADY_EXISTS")
    path.rename(archived)
    return dict(ok=True, status="RESERVATION_RELEASED_ARCHIVED", archive=str(archived),
                sha256=digest(raw), final_check=result, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reserve", "check", "release"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        result = globals()[args.action](_json(Path(args.config).read_bytes()))
    except (OwnershipError, OSError, ValueError, TypeError) as error:
        print(json.dumps(dict(ok=False, reason=getattr(error, "reason", type(error).__name__),
                              observation=getattr(error, "observation", None)), allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
