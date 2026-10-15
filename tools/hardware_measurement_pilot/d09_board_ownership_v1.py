"""Persistent D09 campaign ownership and fail-closed checks of known board users.

This is cooperative exclusion, not a hardware or OS-wide PL access barrier.
The two historical namespaces are inspected as well as the shared machine-id
reservation. No reset, process signal, stale-lock removal or recovery is offered.
All checks are outside measurement intervals.
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

SCHEMA = "d09_board_ownership_config_v1"
RESERVATION_ROOT = "/home/xilinx/.hwnas_board_ownership"
KNOWN_ROOTS = (
    "/home/xilinx/hardware_measurement_pilot_20261012",
    "/home/xilinx/hardware_template_coverage_20261013",
)
SYSTEM_LOCK = KNOWN_ROOTS[0] + "/system_conditions.lock"
RUNNERS = {"board_runner.py", "d09_cpu_runner.py"}


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
        # Do not echo arbitrary lock content or owner tokens.
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
        raw = (directory / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        argv = [item.decode("utf-8", errors="strict")
                for item in (directory / "cmdline").read_bytes().split(b"\0") if item]
        cwd = os.readlink(directory / "cwd") if argv else None
        after = (directory / "stat").read_text()
        final = after[after.rfind(")") + 2:].split()
        if fields[19] != final[19]:
            raise OwnershipError("PROCESS_IDENTITY_CHANGED_DURING_INSPECTION")
        return dict(pid=pid, state=final[0], start_ticks=final[19], argv=argv, cwd=cwd)


def _config(config):
    if config.get("schema") != SCHEMA:
        raise OwnershipError("INVALID_OWNERSHIP_SCHEMA")
    for key in ("campaign_id", "owner_id"):
        if not isinstance(config.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", config[key]):
            raise OwnershipError("INVALID_OWNER_ID")
    if not re.fullmatch(r"[0-9a-f]{32}", config.get("expected_machine_id", "")):
        raise OwnershipError("INVALID_MACHINE_ID")
    try:
        if str(uuid.UUID(config.get("expected_boot_id", ""))) != config["expected_boot_id"]:
            raise ValueError()
    except (ValueError, AttributeError, KeyError):
        raise OwnershipError("INVALID_BOOT_ID") from None
    if (config.get("reservation_root") != RESERVATION_ROOT or
            config.get("known_board_roots") != list(KNOWN_ROOTS)):
        raise OwnershipError("OWNERSHIP_NAMESPACE_CHANGED")
    return config


def _reservation_path(config):
    return RESERVATION_ROOT + "/" + config["expected_machine_id"] + ".json"


def _regular(path):
    # Reject symlinks in every existing ancestor, including dangling links.
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


def _role(row):
    """Recognize direct, taskset and Python inline known-root runners/workers."""
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
            # A malformed known-workflow inline command is not silently ignored.
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
    # Also catch inline runpy/import commands containing a literal known path.
    joined = " ".join(argv)
    if any(root in joined for root in KNOWN_ROOTS) and any(name in joined for name in RUNNERS):
        return "inline_runner"
    return None


def _owned_worker(row, own_worker):
    if own_worker is None or row["pid"] != own_worker.get("pid"):
        return False
    expected = [own_worker.get("helper_path"), "_worker", own_worker.get("request_path")]
    if (row["start_ticks"] != own_worker.get("start_ticks") or row["state"] == "Z" or
            len(row["argv"]) < 4 or row["argv"][-3:] != expected):
        raise OwnershipError("OWN_WORKER_IDENTITY_CHANGED")
    return True


def _scan(config, backend, *, own_board_lock=None, own_worker=None,
          allow_system_lock=False, require_system_lock=False):
    locks, processes, conflicts = [], [], []
    system = config.get("system_conditions_lock") if allow_system_lock else None
    if require_system_lock and not system:
        raise OwnershipError("OWN_SYSTEM_CONDITIONS_BINDING_REQUIRED")
    if system and (system.get("path") != SYSTEM_LOCK or
                   not re.fullmatch(r"[0-9a-f]{64}", system.get("sha256", "")) or
                   not isinstance(system.get("condition_id"), str)):
        raise OwnershipError("INVALID_OWN_SYSTEM_CONDITIONS_BINDING")
    system_seen, own_board_seen = False, False
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
            try:
                raw = _regular(path)
                value = _json(raw)
                record.update(present=True, sha256=digest(raw))
                if own_board_lock and logical == own_board_lock.get("path"):
                    if value != own_board_lock.get("binding"):
                        raise OwnershipError("OWN_BOARD_LOCK_CHANGED")
                    record["owned"] = own_board_seen = True
                elif system and logical == system["path"]:
                    if (digest(raw) != system["sha256"] or value.get("condition_id") != system["condition_id"] or
                            value.get("identity") != dict(machine_id=config["expected_machine_id"],
                                                          boot_id=config["expected_boot_id"])):
                        raise OwnershipError("OWN_SYSTEM_CONDITIONS_LOCK_CHANGED")
                    record["owned"] = system_seen = True
                if not record["owned"]:
                    conflicts.append(dict(kind="KNOWN_LOCK_PRESENT", path=logical))
            except (OSError, ValueError, TypeError, OwnershipError) as error:
                record.update(present=True, inspection="UNAVAILABLE_OR_MISMATCHED")
                conflicts.append(dict(kind="LOCK_INSPECTION_FAILED", path=logical, error_type=type(error).__name__))
            locks.append(record)
    if require_system_lock and not system_seen:
        conflicts.append(dict(kind="OWN_SYSTEM_CONDITIONS_LOCK_NOT_VERIFIED"))
    if own_board_lock and not own_board_seen:
        conflicts.append(dict(kind="OWN_BOARD_LOCK_NOT_VERIFIED"))
    try:
        entries = list(backend.proc_root.iterdir())
    except OSError:
        entries = []
        conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE"))
    worker_seen = False
    for directory in entries:
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            row = backend.process(pid)
            role = _role(row)
            if row["state"] == "Z":
                continue
            if _owned_worker(row, own_worker):
                worker_seen = True
                continue
            if role:
                public = {key: row[key] for key in ("pid", "state", "start_ticks")}
                public["role"] = role
                processes.append(public)
                conflicts.append(dict(kind="ACTIVE_KNOWN_BOARD_PROCESS", **public))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError, UnicodeError, OwnershipError) as error:
            conflicts.append(dict(kind="PROCESS_INSPECTION_UNAVAILABLE", pid=pid, error_type=type(error).__name__))
    if own_worker and not worker_seen:
        conflicts.append(dict(kind="OWN_WORKER_NOT_VERIFIED"))
    result = dict(schema="d09_known_board_occupancy_v1", observed_utc=utc(), locks=locks,
                  active_processes=processes, conflicts=conflicts,
                  status="CONFLICT" if conflicts else "NO_KNOWN_CONFLICT_OBSERVED",
                  scope="TWO_KNOWN_WORKFLOWS_AND_COOPERATIVE_RESERVATION_NOT_GLOBAL_HARD_EXCLUSION")
    if conflicts:
        raise OwnershipError("BOARD_OCCUPANCY_CONFLICT", result)
    return result


def reserve(config, *, backend=None):
    """Atomically reserve this machine across namespaces; no stale-lock cleanup."""
    backend = backend or Backend()
    _config(config)
    if "reservation" in config or "system_conditions_lock" in config:
        raise OwnershipError("RESERVE_REQUIRES_UNBOUND_BASE_CONFIG")
    identity = _identity(config, backend)
    before = _scan(config, backend)
    logical = _reservation_path(config)
    path = backend.path(logical)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise OwnershipError("SYMLINK_IN_OWNERSHIP_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    binding = dict(schema="d09_shared_board_reservation_v1", campaign_id=config["campaign_id"],
                   owner_id=config["owner_id"], identity=identity, created_utc=utc(),
                   known_board_roots=list(KNOWN_ROOTS), release_requires_explicit_owner_check=True)
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
    # A conflict arising after the first scan retains the new lease for review.
    try:
        _identity(config, backend)
        after = _scan(config, backend)
    except BaseException as error:
        raise OwnershipError("POST_RESERVE_CHECK_FAILED_RESERVATION_RETAINED",
                             dict(reservation=receipt, reason=getattr(error, "reason", type(error).__name__),
                                  occupancy=getattr(error, "observation", None))) from None
    return dict(ok=True, status="RESERVED", reservation=receipt, binding=binding,
                before=before, after=after)


def check(config, *, backend=None, own_board_lock=None, own_worker=None, require_system_conditions=True):
    backend = backend or Backend()
    _config(config)
    identity = _identity(config, backend)
    ref = config.get("reservation", {})
    if ref.get("path") != _reservation_path(config) or not re.fullmatch(r"[0-9a-f]{64}", ref.get("sha256", "")):
        raise OwnershipError("INVALID_RESERVATION_REFERENCE")
    path = backend.path(ref["path"])
    raw = _regular(path)
    binding = _json(raw)
    if (digest(raw) != ref["sha256"] or binding.get("schema") != "d09_shared_board_reservation_v1" or
            binding.get("identity") != identity or binding.get("campaign_id") != config["campaign_id"] or
            binding.get("owner_id") != config["owner_id"] or binding.get("known_board_roots") != list(KNOWN_ROOTS)):
        raise OwnershipError("SHARED_RESERVATION_OWNERSHIP_CHANGED")
    if own_board_lock and own_board_lock.get("path") != KNOWN_ROOTS[0] + "/board.lock":
        raise OwnershipError("OWN_BOARD_LOCK_OUTSIDE_D09_ROOT")
    occupancy = _scan(config, backend, own_board_lock=own_board_lock, own_worker=own_worker,
                      allow_system_lock=True, require_system_lock=require_system_conditions)
    _identity(config, backend)
    if _regular(path) != raw:
        raise OwnershipError("SHARED_RESERVATION_CHANGED_DURING_CHECK")
    return dict(ok=True, status="PASS_OWNED_NO_KNOWN_CONFLICT", identity=identity,
                reservation=ref, occupancy=occupancy)


def release(config, *, backend=None):
    """Archive this exact lease after known locks/processes are absent.

    This API does not certify CPU/service/IRQ restoration. The campaign owner
    must first independently verify every per-run CPU PASS_RESTORED receipt and
    the campaign system-conditions PASS_RESTORED receipt. Missing/failed restore
    evidence is a workflow stop even if this narrower occupancy check could pass.
    """
    backend = backend or Backend()
    clean = dict(config)
    clean.pop("system_conditions_lock", None)
    result = check(clean, backend=backend, require_system_conditions=False)
    path = backend.path(config["reservation"]["path"])
    raw = _regular(path)
    if digest(raw) != config["reservation"]["sha256"]:
        raise OwnershipError("SHARED_RESERVATION_CHANGED_BEFORE_RELEASE")
    archived = path.with_name(path.stem + ".released." + config["owner_id"] + "." + uuid.uuid4().hex + ".json")
    if archived.exists():
        raise OwnershipError("RELEASE_ARCHIVE_ALREADY_EXISTS")
    path.rename(archived)
    return dict(ok=True, status="RESERVATION_RELEASED_ARCHIVED", archive=str(archived),
                sha256=digest(raw), final_check=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reserve", "check", "release"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--before-system-apply", action="store_true")
    args = parser.parse_args()
    config = _json(Path(args.config).read_bytes())
    try:
        result = (check(config, require_system_conditions=not args.before_system_apply)
                  if args.action == "check" else globals()[args.action](config))
    except OwnershipError as error:
        print(json.dumps(dict(ok=False, reason=error.reason, observation=error.observation), allow_nan=False))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
