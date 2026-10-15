"""Read-only snapshot of an isolated sonar search and its frozen source binding.

Only the selected local output directory is written. Remote execution uses the
existing SSH askpass helper and never starts, stops, or edits research jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN_REL = "results/sonar_joint_v1"
SOURCE_DIRS = (
    "tools/sonar_search", "tools/fpga_h4", "tools/sonar_spec",
    "tools/sonar_analysis", "tools/phase5a_quantization", "src/hwnas_fpga",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, ValueError) as error:
        return None, type(error).__name__


def inspect_binding(root, run_rel=RUN_REL):
    root = Path(root).resolve()
    out = root / run_rel
    freeze, error = safe_read_json(out / "freeze.json")
    if error:
        return {"status": "UNAVAILABLE", "error": error, "full_verify_passed": None}
    files = freeze.get("files", {})
    changed, missing = [], []
    for name, expected in sorted(files.items()):
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            changed.append(name)
    # These are the recursive Python inputs used by the existing frozen runner.
    discovered = {p.relative_to(root).as_posix()
                  for name in SOURCE_DIRS for p in (root / name).rglob("*.py")}
    for name in ("reports/fpga_h2_contract", "artifacts/fpga_h2/20260916_v1"):
        discovered.update(p.relative_to(root).as_posix() for p in (root / name).glob("*.json"))
    added = sorted(discovered - set(files))
    snapshot, snapshot_error = safe_read_json(out / "snapshot.json")
    archive_ok = (not snapshot_error and (out / "source_snapshot.zip").is_file()
                  and sha(out / "source_snapshot.zip") == snapshot.get("sha256"))
    cost_ok = ((out / "cost_snapshot.json").is_file()
               and sha(out / "cost_snapshot.json") == freeze.get("cost_snapshot_sha256"))
    shim = out / "sitecustomize.py"
    return {
        "status": "PASS" if not changed and not missing and not added and archive_ok and cost_ok else "DRIFT",
        "freeze_sha256": sha(out / "freeze.json"), "frozen_files": len(files),
        "changed_files": changed, "added_files": added, "missing_files": missing,
        "source_archive_matches": bool(archive_ok), "cost_snapshot_matches": bool(cost_ok),
        "full_verify_passed": None,
        "runtime_shim": {"path": shim.relative_to(root).as_posix(),
                         "sha256": sha(shim) if shim.is_file() else None,
                         "included_in_freeze": shim.relative_to(root).as_posix() in files},
    }


def summarize_run(root, run_rel, processes):
    out = Path(root) / run_rel
    state, error = safe_read_json(out / "state.json")
    state = state or {}
    matching = [p for p in processes if p.get("role") in ("joint_controller", "joint_worker")]
    active = bool(matching)
    current_epoch = None
    log_path = None
    workers = [p for p in matching if p["role"] == "joint_worker"]
    aid = state.get("architecture_id")
    budget = state.get("budget")
    if workers:
        aid = workers[0].get("architecture_id") or aid
        budget = workers[0].get("budget") or budget
        seed = workers[0].get("seed")
        mode = workers[0].get("mode")
        if aid is not None and seed is not None and budget is not None:
            log_path = out / "logs" / f"{mode}_{aid}_{seed}_{budget}.log"
            if log_path.is_file():
                # An append-only log is safe to inspect while its worker writes it.
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 16000))
                    tail = stream.read().decode("utf-8", errors="replace")
                for line in reversed(tail.splitlines()):
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] == aid and parts[1] == str(seed) and parts[2].isdigit():
                        current_epoch = int(parts[2])
                        break
    status = "RUNNING" if active else state.get("status", "INACTIVE" if not error else "NO_STATE")
    return {
        "active": active, "status": status, "stage": state.get("stage"),
        "budget": budget, "architecture_id": aid, "completed": state.get("completed"),
        "total": state.get("total"), "current_epoch": current_epoch,
        "receipts_count": len(list((out / "tasks").glob("*.json"))),
        "state_read_error": error,
        "worker_log": log_path.relative_to(root).as_posix() if log_path else None,
        "processes": matching,
    }


def process_records(text, root, own_user):
    records = []
    for line in text.splitlines():
        columns = line.strip().split(None, 2)
        if len(columns) != 3 or not columns[0].isdigit():
            continue
        pid, owner, command = int(columns[0]), columns[1], columns[2]
        role = "other_process"
        args = {}
        # Never expose command lines for another user's jobs.
        if owner == own_user:
            if "tools/sonar_search/run.py" in command:
                role = "joint_controller"
            elif "tools/sonar_search/worker.py" in command:
                role = "joint_worker"
            elif "tools/sonar_operator_pool_p0/confirmation_queue.py" in command:
                role = "p0_confirmation_controller"
            elif "tools/sonar_operator_pool_p0/confirm.py" in command:
                role = "p0_confirmation_worker"
            # Exclude shells containing a nested launch command; only Python is a worker.
            first = command.split(None, 1)[0]
            if role != "other_process" and "python" not in Path(first).name.lower():
                role = "other_process"
            if role.startswith("joint_"):
                try:
                    cwd_matches = Path(f"/proc/{pid}/cwd").resolve() == Path(root).resolve()
                except OSError:
                    cwd_matches = False
                if not cwd_matches:
                    role = "other_process"
            if role in ("joint_worker", "p0_confirmation_worker"):
                words = shlex.split(command)
                for flag, field in (("--architecture", "architecture_id"), ("--seed", "seed"),
                                    ("--budget", "budget"), ("--operator", "operator"), ("--stage", "stage")):
                    if flag in words and words.index(flag) + 1 < len(words):
                        value = words[words.index(flag) + 1]
                        args[field] = int(value) if field in ("seed", "budget", "stage") and value.isdigit() else value
                args["mode"] = "quant" if "quant" in words else "train"
        records.append({"pid": pid, "owner": "task_user" if owner == own_user else "other_user",
                        "role": role, **args})
    return records


def command_output(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return result.stdout if result.returncode == 0 else "", result.returncode
    except (OSError, subprocess.TimeoutExpired):
        return "", None


def collect_remote(root, run_rel):
    import getpass
    root = Path(root).resolve()
    binding = inspect_binding(root, run_rel)
    # The runner verifies named hashes, train/valid hashes, source archive, and live costs.
    # Calling verify does not acquire locks, write state, or dispatch any worker.
    try:
        import contextlib
        import io
        sys.path.insert(0, str(root / "tools/sonar_search"))
        spec = importlib.util.spec_from_file_location("sonar_frozen_runtime_verify", root / "tools/sonar_search/run.py")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
            module.verify(module.CONFIG, root / run_rel)
        binding["full_verify_passed"] = True
    except Exception as error:
        binding["full_verify_passed"] = False
        binding["full_verify_error"] = type(error).__name__ + ": " + str(error)
        binding["status"] = "DRIFT"
    ps, ps_status = command_output(["ps", "-eo", "pid=,user=,args="])
    records = process_records(ps, root, getpass.getuser())
    by_pid = {p["pid"]: p for p in records}
    device_text, gpu_status = command_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    devices = []
    for row in device_text.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) == 6:
            devices.append(dict(zip(("index", "uuid", "name", "memory_used_mib", "memory_total_mib", "utilization_percent"), fields)))
    apps, apps_status = command_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    gpu_processes = []
    for row in apps.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) == 3 and fields[1].isdigit():
            pid = int(fields[1])
            gpu_processes.append({"gpu_uuid": fields[0], "pid": pid, "memory_mib": fields[2],
                                  **by_pid.get(pid, {"owner": "unknown", "role": "unknown"})})
    p0 = [p for p in records if p["role"].startswith("p0_confirmation_")]
    return {
        "collection_status": "PASS", "collected_at_utc": utc_now(), "root": str(root),
        "run": summarize_run(root, run_rel, records), "binding": binding,
        "gpu": {"devices": devices, "processes": gpu_processes,
                "query_status": {"devices": gpu_status, "processes": apps_status}},
        "parallel_runs": {"p0_confirmation": {"active": bool(p0), "process_count": len(p0), "processes": p0}},
        "process_query_status": ps_status,
    }


def remote_script(root, run_rel):
    names = (utc_now, sha, safe_read_json, inspect_binding, summarize_run, process_records, command_output, collect_remote)
    header = ("from pathlib import Path\nfrom datetime import datetime, timezone\n"
              "import hashlib, importlib.util, json, os, shlex, subprocess, sys\n"
              f"RUN_REL = {run_rel!r}\nSOURCE_DIRS = {SOURCE_DIRS!r}\n")
    return header + "\n\n".join(inspect.getsource(fn) for fn in names) + f"\nprint(json.dumps(collect_remote({root!r}, {run_rel!r})))\n"


def save_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-relative", default=RUN_REL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", action="store_true", help="Collect one read-only SSH snapshot")
    parser.add_argument("--host", default="jiajia@10.70.23.121")
    parser.add_argument("--port", default="7889")
    parser.add_argument("--remote-root", default="/home/jiajia/experiments/sonar_joint_v1_20260922")
    parser.add_argument("--remote-python", default="/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python")
    parser.add_argument("--askpass", type=Path, default=Path("C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/ssh_askpass.cmd"))
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "audit.json").exists():
        raise FileExistsError("Audit exists; choose a new output directory to retain the earlier observation")
    out.mkdir(parents=True, exist_ok=True)
    local = {"root": str(args.root.resolve()), "binding": inspect_binding(args.root, args.run_relative)}
    remote = {"collection_status": "NOT_REQUESTED"}
    if args.remote:
        env = os.environ.copy()
        if args.askpass.is_file():
            env.update(SSH_ASKPASS=str(args.askpass), SSH_ASKPASS_REQUIRE="force", DISPLAY="codex:0")
        command = ("cd " + shlex.quote(args.remote_root) + " && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="
                   + shlex.quote(args.remote_root + "/" + args.run_relative) + " "
                   + shlex.quote(args.remote_python) + " -")
        ssh = "C:/Windows/System32/OpenSSH/ssh.exe" if os.name == "nt" else "ssh"
        try:
            result = subprocess.run([ssh, "-p", args.port, "-o", "StrictHostKeyChecking=yes",
                                     "-o", "ConnectTimeout=20", args.host, command],
                                    input=remote_script(args.remote_root, args.run_relative),
                                    text=True, capture_output=True, env=env, timeout=55, check=False)
            if result.returncode != 0:
                remote = {"collection_status": "FAILED", "exit_code": result.returncode}
            else:
                remote = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            remote = {"collection_status": "FAILED", "error": type(error).__name__}
    save_json(out / "remote_snapshot.json", remote)
    audit = {
        "schema_version": 1, "collected_at_utc": utc_now(), "read_only": True,
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "local": local, "remote": remote,
        "remote_snapshot": {"path": str(out / "remote_snapshot.json"), "sha256": sha(out / "remote_snapshot.json")},
        "scope": "Live process and frozen-binding observation; not training completion or deployment evidence",
        "interpretation": {
            "remote_isolated_from_local_edits": bool(remote.get("binding", {}).get("full_verify_passed") is True
                                                     and local["binding"].get("status") == "DRIFT"),
            "do_not_redeploy_over_active_run": bool(remote.get("run", {}).get("active")),
            "new_audit_tool_outside_frozen_source_directories": True,
            "runtime_shim_outside_freeze_is_separately_hashed": True,
        },
    }
    save_json(out / "audit.json", audit)
    print(json.dumps({"audit": str(out / "audit.json"), "local_binding": local["binding"]["status"],
                      "remote_collection": remote["collection_status"], "remote_run": remote.get("run", {}).get("status")}))
    return 0 if remote["collection_status"] in ("PASS", "NOT_REQUESTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
