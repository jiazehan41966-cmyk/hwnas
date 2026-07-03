#!/usr/bin/env python3
"""Fetch remote Phase0 RL300 search status and final evidence artifacts.

This helper intentionally avoids hard-coding credentials. Provide the password
with HWNAS_REMOTE_PASSWORD, or use key-based SSH through Paramiko.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import sys
import time
from pathlib import Path
from typing import Any

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("paramiko is required: python -m pip install paramiko") from exc


DEFAULT_HOST = "connect.westc.seetacloud.com"
DEFAULT_PORT = 57244
DEFAULT_USER = "root"
DEFAULT_REMOTE_ROOT = "/root/hwnas/hwnas_codex_b052bf4"
DEFAULT_RUN_NAME = (
    "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v1"
)
DEFAULT_ROUTE_GATE_DIR = "pareto_route_gate_phase0_full_rl300_pending"
DEFAULT_STATUS_DIR = Path("results/remote_phase0_full_rl300_status")
DEFAULT_SEARCH_DIR = Path("results/remote_phase0_full_rl300_search")
DEFAULT_ROUTE_DIR = Path("hls_lut_builder/board_harness/results/pareto_route_gate_phase0_full_rl300_pending")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch remote HW-NAS Phase0 full RL300 evidence")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--route-gate-dir", default=DEFAULT_ROUTE_GATE_DIR)
    parser.add_argument("--status-dir", default=str(DEFAULT_STATUS_DIR))
    parser.add_argument("--search-dir", default=str(DEFAULT_SEARCH_DIR))
    parser.add_argument("--route-dir", default=str(DEFAULT_ROUTE_DIR))
    parser.add_argument("--password-env", default="HWNAS_REMOTE_PASSWORD")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument(
        "--allow-unreachable",
        action="store_true",
        help="Write connection-failure JSON and return 0 when the remote cannot be reached.",
    )
    parser.add_argument(
        "--fetch-candidates-jsonl",
        action="store_true",
        help="Also fetch the full candidates.jsonl file when present.",
    )
    return parser.parse_args()


def _connect_once(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get(args.password_env)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=args.timeout,
        banner_timeout=args.timeout,
        auth_timeout=args.timeout,
    )
    return client


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    attempts = max(1, int(args.retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _connect_once(args)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(max(0.0, float(args.retry_delay)))
    assert last_error is not None
    raise last_error


def exec_text(client: paramiko.SSHClient, script: str, *, timeout: int = 120) -> tuple[str, str]:
    stdin, stdout, stderr = client.exec_command("bash -s", timeout=timeout)
    stdin.write(script)
    stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return out, err


def sftp_exists(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        sftp.stat(remote_path)
        return True
    except FileNotFoundError:
        return False


def sftp_is_dir(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        attrs = sftp.stat(remote_path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(attrs.st_mode)


def fetch_file(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    local_path: Path,
    *,
    fetched: list[dict[str, Any]],
) -> None:
    if not sftp_exists(sftp, remote_path):
        fetched.append({"remote": remote_path, "local": str(local_path), "status": "missing"})
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    fetched.append(
        {
            "remote": remote_path,
            "local": str(local_path),
            "status": "fetched",
            "bytes": local_path.stat().st_size,
        }
    )


def fetch_tree(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    local_dir: Path,
    *,
    fetched: list[dict[str, Any]],
) -> None:
    if not sftp_is_dir(sftp, remote_dir):
        fetched.append({"remote": remote_dir, "local": str(local_dir), "status": "missing_dir"})
        return
    stack = [(remote_dir, local_dir)]
    while stack:
        current_remote, current_local = stack.pop()
        current_local.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(current_remote):
            remote_child = posixpath.join(current_remote, entry.filename)
            local_child = current_local / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                stack.append((remote_child, local_child))
            elif stat.S_ISREG(entry.st_mode):
                fetch_file(sftp, remote_child, local_child, fetched=fetched)


def build_remote_status_script(args: argparse.Namespace) -> str:
    run_name = args.run_name
    route_gate_dir = args.route_gate_dir
    remote_root = args.remote_root
    return f"""
cd {remote_root}
RUN_NAME={run_name}
RUN_DIR=results/${{RUN_NAME}}
ROUTE_OUT=hls_lut_builder/board_harness/results/{route_gate_dir}
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import json, subprocess, time
run_name = "{run_name}"
run = Path("results") / run_name
route = Path("hls_lut_builder/board_harness/results") / "{route_gate_dir}"
status = {{
    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "remote_cwd": "{remote_root}",
    "run_name": run_name,
    "run_dir": str(run),
    "screen_sessions": subprocess.getoutput("screen -ls || true"),
    "nvidia_smi": subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null || true"),
    "progress_log_tail": subprocess.getoutput("grep -E 'Episode [0-9]+/300|RL NAS completed|Total evaluated:|Feasible:|Infeasible:|Best candidate:|Traceback|Exception|Error|error' logs/" + run_name + ".log 2>/dev/null | tail -n 80 || true"),
    "watcher_log_tail": subprocess.getoutput("tail -n 80 logs/postsearch_route_gate_plan_watcher.log 2>/dev/null || true"),
    "summary_exists": (run / "results/summary.json").exists(),
    "pareto_exists": (run / "results/pareto_selection.json").exists(),
    "ranked_csv_exists": (run / "results/pareto_ranked_candidates.csv").exists(),
    "lut_stats_exists": (run / "results/lut_stats.json").exists(),
    "route_plan_exists": (route / "route_gate_plan.json").exists(),
    "route_summary_exists": (route / "route_gate_summary.json").exists(),
}}
state = run / "checkpoints/search_state.json"
if state.exists():
    payload = json.loads(state.read_text(encoding="utf-8"))
    status["search_state"] = {{
        key: payload.get(key)
        for key in ["episode", "total_evaluated", "feasible", "infeasible", "best_score", "best_candidate_id", "completed"]
    }}
    status["search_state_mtime_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state.stat().st_mtime))
summary = run / "results/summary.json"
if summary.exists():
    payload = json.loads(summary.read_text(encoding="utf-8"))
    extra = payload.get("extra") or {{}}
    best = payload.get("best_candidate") or {{}}
    metrics = best.get("metrics") or {{}}
    status["summary"] = {{
        "status": payload.get("status"),
        "total_evaluated": payload.get("total_evaluated"),
        "feasible": payload.get("feasible"),
        "infeasible": payload.get("infeasible"),
        "best_arch_id": best.get("arch_id"),
        "macro_f1": metrics.get("macro_f1"),
        "top1": metrics.get("top1"),
        "latency_ms": metrics.get("latency_ms"),
        "dsp": metrics.get("dsp"),
        "bram": metrics.get("bram"),
        "lut": metrics.get("lut"),
        "physical_risk": metrics.get("physical_risk"),
        "num_episodes": extra.get("num_episodes"),
        "train_epochs": extra.get("train_epochs"),
        "device": extra.get("device"),
    }}
route_summary = route / "route_gate_summary.json"
if route_summary.exists():
    payload = json.loads(route_summary.read_text(encoding="utf-8"))
    status["route_summary"] = {{
        "candidate_count": payload.get("candidate_count"),
        "quick_pass_count": payload.get("quick_pass_count"),
        "full_route_clean_count": payload.get("full_route_clean_count"),
        "claimable_real_e2e_latency_count": payload.get("claimable_real_e2e_latency_count"),
        "final_best_selection_rule": payload.get("final_best_selection_rule"),
    }}
print(json.dumps(status, ensure_ascii=False, indent=2))
PY
"""


def main() -> int:
    args = parse_args()
    status_dir = Path(args.status_dir)
    search_dir = Path(args.search_dir)
    route_dir = Path(args.route_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = connect(args)
    except Exception as exc:  # pragma: no cover - network dependent
        payload = {
            "generated_at_utc": utc_timestamp(),
            "remote": {
                "host": args.host,
                "port": args.port,
                "user": args.user,
                "remote_root": args.remote_root,
                "run_name": args.run_name,
                "route_gate_dir": args.route_gate_dir,
            },
            "status": {
                "connection_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latest_reliable_status_json": str(status_dir / "remote_status_latest.json"),
                "latest_reliable_status_exists": (status_dir / "remote_status_latest.json").exists(),
            },
            "fetched": [],
        }
        (status_dir / "fetch_error_latest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if args.allow_unreachable else 2

    fetched: list[dict[str, Any]] = []
    try:
        out, err = exec_text(client, build_remote_status_script(args), timeout=180)
        (status_dir / "remote_status_latest.json").write_text(out, encoding="utf-8")
        if err:
            (status_dir / "remote_status_latest.stderr.log").write_text(err, encoding="utf-8")
        status_payload = json.loads(out)

        sftp = client.open_sftp()
        try:
            remote_root = args.remote_root.rstrip("/")
            launch_root = posixpath.join(remote_root, "results/remote_phase0_full_launch_20260602T0240Z")
            fetch_file(
                sftp,
                posixpath.join(launch_root, "launch_manifest.json"),
                status_dir / "launch_manifest.json",
                fetched=fetched,
            )
            fetch_file(
                sftp,
                posixpath.join(launch_root, "postsearch_route_gate_plan_watcher.sh"),
                status_dir / "postsearch_route_gate_plan_watcher.sh",
                fetched=fetched,
            )
            fetch_file(
                sftp,
                posixpath.join(remote_root, "logs/postsearch_route_gate_plan_watcher.log"),
                status_dir / "postsearch_route_gate_plan_watcher.log",
                fetched=fetched,
            )

            run_results = posixpath.join(remote_root, "results", args.run_name, "results")
            result_files = [
                "summary.json",
                "pareto_selection.json",
                "pareto_ranked_candidates.csv",
                "lut_stats.json",
                "best_candidate.json",
                "baseline.json",
                "dataset_summary.json",
                "search_space_summary.json",
                "search_space_pruned_summary.json",
                "candidates.json",
                "candidates.csv",
            ]
            if args.fetch_candidates_jsonl:
                result_files.append("candidates.jsonl")
            for name in result_files:
                fetch_file(
                    sftp,
                    posixpath.join(run_results, name),
                    search_dir / "results" / name,
                    fetched=fetched,
                )

            route_remote = posixpath.join(
                remote_root,
                "hls_lut_builder/board_harness/results",
                args.route_gate_dir,
            )
            route_files = [
                "route_gate_plan.json",
                "route_gate_summary.json",
                "final_candidate_manifest.json",
                "final_route_clean_best.json",
                "final_claimable_board_best.json",
                "route_gate_plan.stdout.json",
                "route_gate_plan.stderr.log",
                "route_gate_refresh.stdout.json",
                "route_gate_refresh.stderr.log",
            ]
            for name in route_files:
                fetch_file(sftp, posixpath.join(route_remote, name), route_dir / name, fetched=fetched)
            for subdir in ("candidates", "gates", "reports"):
                fetch_tree(sftp, posixpath.join(route_remote, subdir), route_dir / subdir, fetched=fetched)
        finally:
            sftp.close()
    finally:
        client.close()

    manifest = {
        "generated_at_utc": status_payload.get("captured_at_utc"),
        "remote": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "remote_root": args.remote_root,
            "run_name": args.run_name,
            "route_gate_dir": args.route_gate_dir,
        },
        "status": status_payload,
        "fetched": fetched,
    }
    (status_dir / "fetch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (status_dir / "fetch_error_latest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": status_payload.get("captured_at_utc") or utc_timestamp(),
                "remote": manifest["remote"],
                "status": {
                    "connection_status": "success",
                    "error_type": "",
                    "error": "",
                    "latest_reliable_status_json": str(status_dir / "remote_status_latest.json"),
                    "latest_reliable_status_exists": True,
                },
                "fetched_count": sum(1 for item in fetched if item.get("status") == "fetched"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
